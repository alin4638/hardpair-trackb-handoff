"""Qwen2-VL backend: assistant-only loss and fixed, summed candidate likelihoods."""
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read_json,atomic_json
from .data import CLASSES
from .build_sft import CLASS_PROMPT
from .scoring import assistant_labels, normalize_scores

class Student:
    def __init__(self,c,checkpoint=None,trainable=False):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
        if not torch.cuda.is_available(): raise RuntimeError('GPU stage requires teammate CUDA machine; CPU --dry-run is available')
        self.torch=torch;self.c=c;self.device='cuda'
        self.dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        meta=read_json(Path(checkpoint)/'checkpoint.json') if checkpoint else None
        lock_path=Path(c['run_dir'])/'model_revision.json';locked=read_json(lock_path)
        if meta and meta['base_model']!=c['model']['name']: raise ValueError('Checkpoint base model mismatch')
        revision=meta['base_revision'] if meta else (locked['resolved'] if locked else c['model']['revision'])
        if locked and meta and locked['resolved']!=meta['base_revision']: raise ValueError('Run base revision mismatch')
        kw={'torch_dtype':self.dtype,'device_map':{'':'cuda'},'attn_implementation':'sdpa','revision':revision}
        if c['model']['load_in_4bit']:
            kw['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=self.dtype,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True)
        model=Qwen2VLForConditionalGeneration.from_pretrained(c['model']['name'],**kw)
        self.base_revision=getattr(model.config,'_commit_hash',None) or revision
        atomic_json(lock_path,{'name':c['model']['name'],'requested':c['model']['revision'],'resolved':self.base_revision})
        self.proc=AutoProcessor.from_pretrained(c['model']['name'],revision=self.base_revision,
                          min_pixels=c['model']['min_pixels'],max_pixels=c['model']['max_pixels'])
        if trainable and c['model']['load_in_4bit']:
            model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=c['model']['gradient_checkpointing'])
        for p in model.parameters(): p.requires_grad=False
        if checkpoint:
            model=PeftModel.from_pretrained(model,str(Path(checkpoint)/'adapter'),is_trainable=trainable)
        elif trainable:
            projections=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
            modules=[n for n,_ in model.named_modules() if n.endswith(projections) and 'visual' not in n]
            if not modules: raise RuntimeError('No language LoRA targets matched')
            model=get_peft_model(model,LoraConfig(r=c['model']['lora_r'],lora_alpha=c['model']['lora_alpha'],
                   target_modules=modules,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM'))
        visual={}
        for pattern in c['model']['train_visual']:
            matches=[(n,p) for n,p in model.named_parameters() if pattern in n and 'lora_' not in n]
            if not matches: raise ValueError(f'Unmatched visual parameter pattern: {pattern}')
            if 'visual' not in pattern: raise ValueError('train_visual may only match visual modules')
            for n,p in matches:
                if trainable: p.requires_grad=True
                visual[n]=p
        self.visual_names=set(visual)
        if checkpoint:
            tensors=torch.load(Path(checkpoint)/'visual.pt',map_location='cpu',weights_only=True)
            params=dict(model.named_parameters())
            for n,v in tensors.items():
                if n not in params: raise ValueError(f'Visual checkpoint mismatch: {n}')
                params[n].data.copy_(v.to(params[n].device,params[n].dtype))
        if trainable:
            # fp32 trainable tensors make fp16 GradScaler safe, including optional visual modules.
            for p in model.parameters():
                if p.requires_grad: p.data=p.data.float()
            if c['model']['gradient_checkpointing']:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                model.enable_input_require_grads()
            model.config.use_cache=False
        self.model=model
        model.train(trainable)

    def inputs(self,image,prompt,answer=None,include_end=True):
        torch=self.torch
        msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        prefix=self.proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        if isinstance(image,(str,Path)):
            with Image.open(image) as im: image=im.convert('RGB').resize((224,224))
        text=prefix if answer is None else prefix+answer+('<|im_end|>' if include_end else '')
        inp=self.proc(text=[text],images=[image],return_tensors='pt').to(self.device)
        if inp['input_ids'].shape[1]>self.c['model']['max_length']:
            raise ValueError('Sequence exceeds max_length; increase explicitly (targets are never silently truncated)')
        if answer is not None:
            pref=self.proc(text=[prefix],images=[image],return_tensors='pt')['input_ids'].to(self.device)
            labels=torch.tensor([assistant_labels(inp['input_ids'][0].tolist(),pref[0].tolist(),
                                 inp['attention_mask'][0].tolist())],device=self.device)
            inp['labels']=labels
        return inp

    def scores(self,image):
        torch=self.torch;self.model.eval();scores=[];lengths=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=self.dtype):
            for label in CLASSES:
                x=self.inputs(image,CLASS_PROMPT,label,include_end=False)
                labels=x.pop('labels')[:,1:]; mask=labels!=-100
                logits=self.model(**x).logits[:,:-1].float()
                logp=logits.log_softmax(-1).gather(-1,labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                scores.append(float(logp[mask].sum()));lengths.append(int(mask.sum()))
        return {**normalize_scores(scores),'candidate_token_lengths':lengths}

    def generate(self,image,prompt,sample=False,prediction=None):
        torch=self.torch;self.model.eval()
        fixed=f'Diagnosis: {prediction}\n' if prediction else ''
        inp=self.inputs(image,prompt,fixed,include_end=False) if fixed else self.inputs(image,prompt)
        inp.pop('labels',None)
        with torch.inference_mode(),torch.autocast('cuda',dtype=self.dtype):
            options={'do_sample':sample,'max_new_tokens':256,'use_cache':True}
            if sample: options.update(temperature=0.8,top_p=0.95)
            out=self.model.generate(**inp,**options)
        return fixed+self.proc.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True).strip()
