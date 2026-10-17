"""Run: streamlit run app.py -- --config configs/laptop_4080.yaml"""
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image,ImageDraw
import streamlit as st
from tb.common import load_config,read_json,read_jsonl
from tb.data import load_manifest,CLASSES
from tb.feedback import current,submit
from tb.crops import regions

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/laptop_4080.yaml');args=p.parse_args()
c=load_config(args.config);root=Path(c['run_dir']);feedback_dir=Path(c['feedback_dir']);m=load_manifest(False,c)
st.title('Track B · 组织图块复核')
st.caption('研究演示；模型版本与结果来自真实保存文件。区域证据为自动教师监督。')
state=read_json(root/'run_manifest.json',{});versions=state.get('results',{})
queries=read_jsonl(feedback_dir/'queries.jsonl');records=read_jsonl(feedback_dir/'feedback_records.jsonl');active=current(records)
fb=read_json(feedback_dir/'state.json',{});trained=read_json(feedback_dir/'effective_feedback.json',{})
queued=[k for k,v in active.items() if trained.get(k)!=v]
st.metric('等待类别反馈',len({q['tile_id'] for q in queries}-set(active)))
st.metric('已提交、等待训练更新',len(queued))
st.write('实际反馈模型版本',fb.get('current',{}).get('model_version','尚未运行反馈阶段'))
tab_result,tab_feedback=st.tabs(['模型结果与区域','类别反馈'])
with tab_result:
    if not versions:st.info('运行 zeroshot/r1/r2/r3 后，这里显示真实结果。')
    else:
        stage=st.selectbox('结果阶段',list(versions));entry=versions[stage];st.code(entry['model_version'])
        folder=Path(entry['path']);preds=pd.read_csv(folder/f'preds_vlm_{stage}.csv')
        tid=st.selectbox('图块',preds.tile_id.tolist());row=m.set_index('tile_id').loc[tid];pr=preds.set_index('tile_id').loc[tid]
        st.write({'prediction':pr['pred'],'rival':pr['top2'],'margin':float(pr['margin'])})
        rr=next((r for r in read_jsonl(folder/f'vlm_rationales_{stage}.jsonl') if r['tile_id']==tid),None)
        if Path(row.local_path).exists():
            with Image.open(row.local_path) as im:im=im.convert('RGB').resize((224,224))
            draw=ImageDraw.Draw(im)
            if rr:
                for r in regions():
                    if r['region_id'] in rr['cited_regions']:draw.rectangle(r['box'],outline='red',width=3)
            st.image(im,width=448)
        else:st.warning('图像未恢复到 data_root。')
        if rr:
            st.text(rr['rationale']);st.write({'format_pass':rr['format_pass'],'grounding_pass':rr['grounding_pass'],'teacher_source':rr.get('teacher_source')})
            if not rr['format_pass']:st.warning('生成解释不符合结构或预测一致性要求，不能作为已验证解释展示。')
with tab_feedback:
    if not queries:st.info('先运行 feedback 阶段生成待复核队列。')
    else:
        ids=list(dict.fromkeys(q['tile_id'] for q in queries));tid=st.selectbox('已查询 pool 图块（也可修订）',ids)
        query=next(q for q in reversed(queries) if q['tile_id']==tid);row=m.set_index('tile_id').loc[tid]
        st.write({'query_model_version':query['query_model_version'],'original_prediction':query['original_prediction']})
        if Path(row.local_path).exists():st.image(row.local_path,width=448)
        with st.form('class_feedback'):
            label=st.selectbox('确认或纠正类别',CLASSES,index=CLASSES.index(active.get(tid,{}).get('corrected_label',query['original_prediction'])))
            if st.form_submit_button('提交类别反馈'):
                submit(feedback_dir/'feedback_records.jsonl',query,label,'expert_label');st.success('已追加反馈版本；运行 feedback --resume 执行模型更新。')
        st.caption('本界面不触发后台训练；队列会在下一次流水线更新时消费。')
