"""Stable model identity across presentation and scientific figures."""
LAYA='#3378FF'
BLIP='#F78C25'

def model_color(model):
    name=(model.get('adapter','')+' '+model.get('model',model.get('id',''))) if isinstance(model,dict) else str(model)
    if 'laya' in name.lower():return LAYA
    if 'blip' in name.lower():return BLIP
    return '#64748B'

def model_label(model):
    name=model['model'];base='Laya' if 'laya' in name.lower() else 'BLIP' if 'blip' in name.lower() else name
    variant={'short_open':'pergunta curta','short_options':'com alternativas','long_descriptions':'descrição longa','structured_json':'JSON','short_serial':'perguntas separadas','native_short':'alternativas curtas','native_descriptive':'alternativas descritivas','native_serial':'perguntas separadas'}.get(model.get('profile'))
    return base+(f' · {variant}' if variant else '')

LABELS={'presence':'Presença','make':'Marca','color':'Cor do veículo','body':'Tipo de veículo','helmet':'Capacete','carrying':'Objeto carregado','clothing':'Roupa',
 'black':'Preto','white':'Branco','gray_silver':'Cinza / prata','red':'Vermelho','blue':'Azul','green':'Verde','yellow':'Amarelo','orange':'Laranja','brown':'Marrom','purple':'Roxo','pink':'Rosa','yes':'Sim','no':'Não','unknown':'Não identificado','not_visible':'Não visível'}
def label(value):return LABELS.get(value,value.replace('_',' '))

def baseline_models(models):
    selected=[]
    for adapter in ['laya','blip']:
        candidates=[m for m in models if m.get('adapter')==adapter or adapter in m['model'].lower()]
        if candidates:selected.append(next((m for m in candidates if m.get('profile') in {'native_short','short_open','baseline'}),candidates[0]))
    return selected or models[:2]
