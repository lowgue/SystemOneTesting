import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'benchmarks/datasets/historical';out.mkdir(exist_ok=True)
base=json.loads((root/'benchmarks/contracts/description_ollama_v1.json').read_text())
base['name']='historical-tasks-v1';base['label_policy']='Historical task taxonomy: hatchback canonicalized as hatch; pickup remains distinct from truck. New runs are not pooled with historical results.'
b=base['domains']['vehicle']['questions']['body']['criteria'];b.update(pickup='Pickup truck',bus='Bus',motorcycle='Motorcycle')
base['aliases']['body']={'pickup':['pickup','pickup truck'],'truck':['truck','lorry'],'hatch':['hatch','hatchback'],'bus':['bus'],'motorcycle':['motorcycle','motorbike']}
(root/'benchmarks/contracts/historical_tasks_v1.json').write_text(json.dumps(base,ensure_ascii=False,indent=2)+'\n')
fieldmap={'vehicle_type':'body','color':'color','helmet':'helmet','carrying':'carrying','clothing':'clothing'}
labelmap={'hatchback':'hatch'}
def save(key,name,rows,source,kind):
 items=[]
 for n,r in enumerate(rows):
  r=dict(r);domain=r.get('domain','person' if kind=='person' or r.get('task') in {'helmet','carrying','clothing'} else 'vehicle')
  field=None
  if kind=='color': labels={'color':r['color']};field='color'
  elif kind=='body':labels={'body':r['expected_type']};field='body'
  elif kind=='prompts':labels={'color':r['expected_color']};field='color'
  elif kind in {'robustness','multi'}:field=fieldmap[r['task']];labels={field:r.get('target_label',r.get('expected'))}
  elif kind=='stanford':labels={'make':r['make'],'body':r['body_type']}
  else:labels=r['expected']
  labels={k:labelmap.get(v,v) for k,v in labels.items()}
  if 'make' in labels and labels['make'] not in base['domains']['vehicle']['questions']['make']['criteria']:labels['make']='other'
  rel=r.get('rel_path') or 'eval/images/stanford_cars/'+r['filename']
  p=root/rel
  if not p.is_file():raise ValueError(rel)
  for k,v in labels.items():
   if v not in base['domains'][domain]['questions'][k]['criteria']:raise ValueError((key,k,v))
  item=dict(id=r.get('id',p.name),domain=domain,filename=p.name,rel_path=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),annotations=labels)
  if set(labels)==set(base['domains'][domain]['questions']):item['expected']=labels;item.pop('annotations')
  for k in ['split','stratum','presentation','difficult','difficulty_tags','source','source_id','scene_id','annotation']:
   if k in r:item[k]=r[k]
  item['group_id']=r.get('scene_id',r.get('source_id',item['sha256']))
  if field:item['task_field']=field
  if r.get('question_en'):item['question']=r['question_en']
  if kind=='person':item['task_fields']=['helmet','carrying','clothing']
  items.append(item)
 doc=dict(schema_version=1,name=name,label_status='exploratory_historical_unadjudicated',source_manifest=source,source_manifest_sha256=hashlib.sha256((root/source).read_bytes()).hexdigest(),required_contract='historical',note='Gabaritos históricos exploratórios. Imagens derivadas compartilham grupo; eventos não equivalem a imagens independentes. hatchback → hatch; pickup e truck preservados.',items=items)
 (out/(key+'.json')).write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
for num,kind,key,name in [(1,'color','color-all','Cores — coleção completa (inclui desenvolvimento)'),(2,'body','vehicle-types','Tipos de veículo'),(3,'person','people','Pessoas: capacete, objeto e roupa'),(4,'robustness','robustness','Robustez: iluminação, desfoque e oclusão'),(5,'multi','multi-object','Múltiplos objetos: recorte e contexto'),(6,'prompts','prompt-formats','Comparação histórica de prompts: cores'),(7,'complete','complete','Descrição completa: veículos e pessoas')]:
 p=next((root/'eval/datasets').glob(f'test_{num}_*manifest.json'));d=json.loads(p.read_text());rows=d.get('items',d.get('images'))
 if d.get('question_en'):rows=[dict(r,question_en=r.get('question_en',d['question_en'])) for r in rows]
 save(key,name,rows,str(p.relative_to(root)),kind)
 if num==1:
  for split,title in [('eval','Cores — avaliação'),('dev','Cores — desenvolvimento')]:save('color-'+split,title,[r for r in rows if r['split']==split],str(p.relative_to(root)),kind)
p=root/'eval/datasets/stanford_cars_sample.json';d=json.loads(p.read_text());save('stanford','Stanford Cars: marca e carroceria',d['items'],str(p.relative_to(root)),'stanford')
