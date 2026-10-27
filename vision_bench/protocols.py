"""Explicit, reproducible question and request variants for visual evaluations."""
from copy import deepcopy

PROFILES = {
 'blip': [
  dict(id='short_open',name='Pergunta curta aberta',group='baseline'),
  dict(id='short_options',name='Pergunta com alternativas',group='prompts'),
  dict(id='long_descriptions',name='Alternativas descritivas',group='prompts'),
  dict(id='structured_json',name='Solicitação de JSON',group='prompts'),
  dict(id='short_serial',name='Pergunta curta, uma inferência por atributo',group='requests'),
 ],
 'laya': [
  dict(id='native_short',name='Alternativas nativas curtas, request conjunto',group='baseline'),
  dict(id='native_descriptive',name='Alternativas nativas descritivas',group='prompts'),
  dict(id='native_serial',name='Alternativas curtas, um request por atributo',group='requests'),
 ],
}
MODES={'baseline','prompts','requests','all'}
DESCRIPTIONS={
 'black':'Black or deep dark tone','white':'White or off-white bright tone',
 'gray_silver':'Gray or silver metallic tone','red':'Red, crimson, or cherry tone',
 'blue':'Blue, navy, or sky blue tone','green':'Green, forest, or lime tone',
 'yellow':'Yellow or bright golden tone','orange':'Orange or amber tone',
 'brown':'Brown, bronze, or earth tone','purple':'Purple or violet tone','pink':'Pink or rose tone',
 'sedan':'Sedan with separate trunk','suv':'SUV with tall enclosed body','hatch':'Hatchback with rear hatch',
 'pickup':'Pickup with open cargo bed','bus':'Bus for many passengers','motorcycle':'Motorcycle with two wheels',
 'safety_vest':'Reflective safety vest','jacket_coat':'Jacket or outer coat','shirt_tshirt':'Shirt or T-shirt',
 'backpack':'Backpack carried on the back','bag_or_box':'Bag or box carried by hand',
}

def expand_specs(specs,mode):
 if mode not in MODES:raise ValueError('Comparação inválida.')
 out=[]
 for spec in specs:
  profiles=PROFILES.get(spec['adapter'])
  if not profiles or mode=='baseline':out.append(dict(spec));continue
  for profile in profiles:
   if profile['group']!='baseline' and mode!='all' and profile['group']!=mode:continue
   out.append(dict(spec,id=spec['id']+'--'+profile['id'],base_model=spec['id'],profile=profile['id']))
 return out

def item_contract(contract,item):
 result=deepcopy(contract);domain=result['domains'][item['domain']]
 fields=item.get('task_fields') or ([item['task_field']] if item.get('task_field') else list(domain['questions']))
 domain['questions']={k:v for k,v in domain['questions'].items() if k in fields}
 domain['prompts']={k:v for k,v in domain['prompts'].items() if k in fields}
 if item.get('question') and len(fields)==1:
  domain['prompts'][fields[0]]=item['question'];domain['questions'][fields[0]]['instructions']=item['question']
 return result

def questions(spec,contract,domain):
 qs=deepcopy(contract['domains'][domain]['questions'])
 if spec.get('profile')=='native_descriptive':
  for q in qs.values():
   q['criteria']={key:DESCRIPTIONS.get(key,f'{label}: visible {label.lower()}') for key,label in q['criteria'].items()}
 return qs

def prompts(spec,contract,domain):
 result={};profile=spec.get('profile','short_open')
 for key,prompt in contract['domains'][domain]['prompts'].items():
  choices=contract['domains'][domain]['questions'][key]['criteria']
  if profile=='short_options':prompt+=' Choose one: '+', '.join(choices.values())+'.'
  elif profile=='long_descriptions':prompt+=' Choose the visible attribute from: '+'; '.join(DESCRIPTIONS.get(k,f'{label}: visible {label.lower()}') for k,label in choices.items())+'. Return one label.'
  elif profile=='structured_json':prompt+=' Return only a JSON object with exactly one key: {"'+key+'": "<label>"}. Allowed labels: '+', '.join(choices)+'.'
  result[key]=prompt
 return result

def describe(spec,contract,domain):
 adapter=spec['adapter'];qs=questions(spec,contract,domain);profile=spec.get('profile', 'short_open' if adapter=='blip' else 'native_short')
 serial=profile in {'short_serial','native_serial'}
 return dict(adapter=adapter,base_model=spec.get('base_model',spec['id']),profile=profile,
  question_type='Texto livre VQA' if adapter=='blip' else 'choice: alternativas estruturadas',
  output_format=('JSON textual solicitado; validação estrita' if profile=='structured_json' else 'Texto gerado e normalizado') if adapter=='blip' else 'answers.<atributo>.choice (alternativa nativa)',
  request_mode='Uma chamada por atributo' if serial else ('Batch de perguntas com imagem repetida' if adapter=='blip' else 'Um request com todas as perguntas'),
  calls_per_photo=len(qs) if serial else 1,
  generation=dict(max_new_tokens=64 if profile=='structured_json' else 12,do_sample=False) if adapter=='blip' else None,
  head_max_len=spec.get('head_max_len') if adapter=='laya' else None,
  request=dict(images=['<imagem RGB, sem dados binários neste exemplo>'],text=prompts(spec,contract,domain)) if adapter=='blip' else dict(images=['<imagem RGB>'],questions=qs),
  metrics=['Acurácia por atributo','Precisão macro','Recall macro','F1 macro','Aderência às alternativas antes da normalização','Formato válido','Ambiguidade','Latência média, p50 e p95','Chamadas de inferência por foto','CPU e memória RSS'])

def output_diagnostics(spec,contract,domain,raw):
 import json,re
 result={}
 for key,q in contract['domains'][domain]['questions'].items():
  if spec['adapter']=='blip':
   value=raw.get(key,'');format_valid=isinstance(value,str) and bool(value.strip())
   if spec.get('profile')=='structured_json':
    try:
     obj=json.loads(value);format_valid=isinstance(obj,dict) and set(obj)=={key} and isinstance(obj[key],str)
     value=obj[key] if format_valid else ''
    except (ValueError,KeyError,TypeError):format_valid=False;value=''
   clean=lambda x:re.sub(r'[^a-z0-9]+',' ',str(x).lower()).strip()
   choices=q['criteria'];adherent=clean(value) in {clean(v) for v in [*choices,*choices.values()]}
   # Distinct explicit class names; nested names (truck/pickup truck) are excluded.
   hits={k for k,label in choices.items() if re.search(r'(?<!\w)'+re.escape(clean(label))+r'(?!\w)',clean(value))}
   hits={k for k in hits if not any(k!=j and clean(choices[k]) in clean(choices[j]) for j in hits)}
   ambiguous=len(hits)>1
   if spec.get('profile')=='structured_json':
    adherent=isinstance(value,str) and value in choices
    format_valid=bool(format_valid and adherent)
  else:
   value=raw.get('answers',{}).get(key,{}).get('choice');format_valid=isinstance(value,str);adherent=value in q['criteria'];ambiguous=False
  result[key]=dict(format_valid=format_valid,adherent=adherent,ambiguous=ambiguous)
 return result
