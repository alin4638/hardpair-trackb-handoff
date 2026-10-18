"""Run: streamlit run app.py -- --config configs/laptop_4080.yaml"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from PIL import Image,ImageDraw
import streamlit as st
from tb.common import load_config,read_json,read_jsonl
from tb.data import load_manifest,CLASSES
from tb.feedback import current,submit,import_labels
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
st.caption(f"自动反馈来源：{c['feedback']['source']}。数据库真标签属于模拟反馈，人工复核提交属于专家反馈。")
running=(root/'pipeline.lock').exists()
ready=bool(state.get('checkpoints')) and (c.get('grounding',{}).get('mode')!='guided' or (root/'region_guide.json').exists())
if not ready: st.info('先在 GPU 机器完成 prepare 和 evidence，再执行反馈更新。')
if st.button('执行反馈更新 / 继续队列',disabled=running or not ready):
    logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    output=logs/f"app_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with output.open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).parent/'scripts/run_pipeline.py'),
                                  '--stage','feedback','--config',str(Path(args.config).resolve()),'--resume'],
                                 cwd=Path(__file__).parent,stdout=log,stderr=subprocess.STDOUT)
    st.success(f'已启动反馈流水线 PID {process.pid}；日志：{output}')
if running: st.info('流水线正在运行。完成后刷新页面查看实际新版本。')
if st.button('刷新状态'): st.rerun()
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
            st.text(rr['rationale']);st.write({'format_pass':rr['format_pass'],'grounding_pass':rr['grounding_pass'],
                    'teacher_source':rr.get('teacher_source'),'explanation_source':rr.get('explanation_source','free_student'),
                    'student_grounding_pass':rr.get('student_grounding_pass')})
            if rr.get('evidence'): st.json(rr['evidence'],expanded=False)
            if not rr['format_pass']:st.warning('生成解释不符合结构或预测一致性要求，不能作为已验证解释展示。')
with tab_feedback:
    if not queries:st.info('先运行 feedback 阶段生成待复核队列。')
    else:
        ids=list(dict.fromkeys(q['tile_id'] for q in queries));tid=st.selectbox('已查询 pool 图块（也可修订）',ids)
        query=next(q for q in reversed(queries) if q['tile_id']==tid);row=m.set_index('tile_id').loc[tid]
        st.write({'query_model_version':query['query_model_version'],'original_prediction':query['original_prediction']})
        if Path(row.local_path).exists():
            with Image.open(row.local_path) as image: image=image.convert('RGB').resize((224,224))
            if (root/'region_guide.json').exists():
                from types import SimpleNamespace
                from tb.cache import Cache
                from tb.region_guide import RegionGuide
                tile=SimpleNamespace(tile_id=tid,**row.to_dict());cache=Cache(c,m);guide=RegionGuide.load(cache)
                rival=query.get('top2')
                if not rival:
                    z=guide.normalized('conch',cache.read('conch',tile,'regions'))
                    rival=max((cl for cl in CLASSES if cl!=query['original_prediction']),
                        key=lambda cl:float(z[:,[i for i,d in enumerate(cache.descriptors) if d['label']==cl]].max()))
                explanation=guide.explanation(tile,query['original_prediction'],rival)
                for region in regions():
                    if region['region_id'] in explanation['cited_regions']:
                        ImageDraw.Draw(image).rectangle(region['box'],outline='red',width=3)
                st.text(explanation['rationale']);st.caption('上述区域说明来源：retrieved_teacher_template')
                st.json(explanation['evidence'],expanded=False)
            st.image(image,width=448)
        with st.form('class_feedback'):
            label=st.selectbox('确认或纠正类别',CLASSES,index=CLASSES.index(active.get(tid,{}).get('corrected_label',query['original_prediction'])))
            if st.form_submit_button('提交类别反馈'):
                submit(feedback_dir/'feedback_records.jsonl',query,label,'expert_label');st.rerun()
        template=pd.DataFrame([{'tile_id':i,'corrected_label':active.get(i,{}).get('corrected_label','')} for i in ids])
        st.download_button('下载已查询图块 CSV',template.to_csv(index=False),'selected_feedback.csv','text/csv')
        uploaded=st.file_uploader('批量导入类别反馈 CSV',type=['csv'])
        source=st.selectbox('导入标签来源',['expert_label','simulated_dataset_label'])
        if st.button('导入 CSV',disabled=uploaded is None):
            try:
                import_labels(pd.read_csv(uploaded),queries,m,feedback_dir/'feedback_records.jsonl',source);st.rerun()
            except (ValueError,RuntimeError) as error:st.error(str(error))
        st.dataframe(pd.DataFrame([{'tile_id':k,'corrected_label':v['corrected_label'],
                                  'source':v['feedback_source'],'pending_update':k in queued} for k,v in active.items()]))
