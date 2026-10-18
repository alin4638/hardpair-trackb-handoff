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
        self.merged=False
        if checkpoint and not trainable and not c['model']['load_in_4bit']:
            # Inference only: fold LoRA into the base weights (measured ~1.9x faster decoding on the 4080).
            model=model.merge_and_unload();self.merged=True
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

    def _image(self,image):
        if isinstance(image,(str,Path)):
            with Image.open(image) as im: return im.convert('RGB').resize((224,224))
        return image

    def _prefix(self,prompt):
        msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        return self.proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)

    def _batch(self,texts,images,side):
        tok=self.proc.tokenizer;old=tok.padding_side;tok.padding_side=side
        try: x=self.proc(text=texts,images=images,padding=True,return_tensors='pt').to(self.device)
        finally: tok.padding_side=old
        if x['input_ids'].shape[1]>self.c['model']['max_length']:
            raise ValueError('Sequence exceeds max_length; increase explicitly (targets are never silently truncated)')
        return x

    def scores(self,image):
        """Nine candidate codes in one right-padded batch; same prompt, tokens and summed log-likelihood rule."""
        torch=self.torch;self.model.eval();image=self._image(image);prefix=self._prefix(CLASS_PROMPT)
        pref=self.proc(text=[prefix],images=[image],return_tensors='pt')['input_ids'][0].tolist()
        x=self._batch([prefix+label for label in CLASSES],[image]*len(CLASSES),'right')
        labels=torch.tensor([assistant_labels(x['input_ids'][i].tolist(),pref,x['attention_mask'][i].tolist())
                             for i in range(len(CLASSES))],device=self.device)
        start=len(pref)-1  # logits at position t predict token t+1; candidates begin at len(pref)
        with torch.inference_mode(),torch.autocast('cuda',dtype=self.dtype):
            logits=self.model(**x).logits[:,start:-1].float()
            target=labels[:,start+1:];mask=target!=-100
            logp=logits.log_softmax(-1).gather(-1,target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            scores=(logp*mask).sum(1)
        return {**normalize_scores([float(s) for s in scores]),'candidate_token_lengths':[int(n) for n in mask.sum(1)]}

    def generate_many(self,images,prompts,sample=False,predictions=None,n=1):
        """Left-padded batched decoding; returns len(images)*n texts in input order."""
        torch=self.torch;self.model.eval()
        predictions=predictions or [None]*len(images)
        fixed=[f'Diagnosis: {p}\n' if p else '' for p in predictions]
        x=self._batch([self._prefix(p)+f for p,f in zip(prompts,fixed)],[self._image(i) for i in images],'left')
        with torch.inference_mode(),torch.autocast('cuda',dtype=self.dtype):
            options={'do_sample':sample,'max_new_tokens':256,'use_cache':True,'num_return_sequences':n,
                     'pad_token_id':self.proc.tokenizer.pad_token_id}
            # Qwen2-VL's generation_config ships top_k=1, which silently turns sampling into greedy decoding
            # (smoke_4080_v1 R3: 4 identical candidates per tile). top_k=0 disables it: pure nucleus sampling.
            if sample: options.update(temperature=0.8,top_p=0.95,top_k=0)
            out=self.model.generate(**x,**options)
        new=out[:,x['input_ids'].shape[1]:]
        return [fixed[i//n]+self.proc.decode(new[i],skip_special_tokens=True).strip() for i in range(len(new))]

    def generate(self,image,prompt,sample=False,prediction=None):
        return self.generate_many([image],[prompt],sample,[prediction])[0]
