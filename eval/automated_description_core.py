"""Strict, versioned parsing and deterministic rendering for the automated pilot."""
import re
from run_test_7_complete_benchmark import PROMPTS, QUESTIONS, gate

PARSER_VERSION = 'strict-v1'
UNKNOWN = {'vehicle': 'unknown', 'person': 'not_visible'}
EXPLICIT_UNKNOWN = {'unknown', 'not visible', 'unclear', 'cannot tell', 'not identifiable', 'unidentifiable'}
ALIASES = {
    'color': {'gray_silver':['gray','grey','silver'], 'purple':['purple','violet']},
    'body': {'truck':['pickup truck','pickup','truck'], 'hatch':['hatchback','hatch'], 'van':['minivan','van'],
             'suv':['suv','crossover'], 'coupe':['coupe'], 'sedan':['sedan'], 'convertible':['convertible'], 'wagon':['wagon']},
    'helmet': {'yes':['yes','helmet','hard hat','hardhat'], 'no':['no','not','bareheaded','cap','hat']},
    'carrying': {'backpack':['backpack','rucksack'], 'bag_or_box':['bag','bags','box','briefcase','suitcase','purse','handbag','package'],
                'none':['nothing','none','empty handed','no']},
    'clothing': {'safety_vest':['safety vest','vest','reflective vest'],
                 'jacket_coat':['jacket','coat','suit','blazer','sweater','sweatshirt','hoodie','uniform'],
                 'shirt_tshirt':['shirt','t shirt','tshirt','top','polo','blouse']},
}

def clean(raw):
    return re.sub(r'[^a-z0-9]+', ' ', str(raw).lower()).strip()

def contains(text, phrase):
    return f' {phrase} ' in f' {text} '

def normalize(domain, key, raw):
    text = clean(raw)
    if not text: return 'parse_error'
    if text in EXPLICIT_UNKNOWN: return UNKNOWN[domain] if key != 'presence' else 'parse_error'
    if key == 'presence':
        if text in {'yes','no'}: return text
        if re.match(r'^(no|not|there is no|there are no)\b',text): return 'no'
        if re.match(r'^(yes|there is a|there are)\b',text): return 'yes'
        return 'parse_error'
    if key == 'helmet' and re.match(r'^(no|not)\b',text): return 'no'
    if key == 'carrying' and re.match(r'^(no|not carrying|nothing|none)\b',text): return 'none'
    if key == 'make':
        aliases = {k:[clean(v),clean(k)] for k,v in QUESTIONS[domain][key]['criteria'].items() if k!='unknown'}
    else:
        aliases = {k:[clean(k)] for k in QUESTIONS[domain][key]['criteria'] if k not in {'unknown','not_visible'}}
        aliases.update(ALIASES.get(key,{}))
    hits = {label for label, terms in aliases.items() if any(contains(text,term) for term in terms)}
    # 'hard hat' is a helmet, rather than the regular-hat alternative.
    if key=='helmet' and (contains(text,'hard hat') or contains(text,'hardhat')): return 'yes'
    return hits.pop() if len(hits)==1 else 'parse_error'

def validate(domain, answers):
    return set(answers)==set(PROMPTS[domain]) and all(v in QUESTIONS[domain][k]['criteria'] for k,v in answers.items())

WORDS = {
    'make': {k:v for k,v in QUESTIONS['vehicle']['make']['criteria'].items()},
    'color': dict(black='preto',white='branco',gray_silver='cinza/prata',red='vermelho',blue='azul',green='verde',yellow='amarelo',orange='laranja',brown='marrom',purple='roxo',pink='rosa'),
    'body': dict(sedan='sedã',suv='SUV',hatch='hatchback',truck='picape',van='van',coupe='cupê',convertible='conversível',wagon='perua',other='outro tipo'),
    'helmet': {'yes':'com capacete','no':'sem capacete'},
    'carrying':dict(backpack='transportando mochila',bag_or_box='transportando bolsa ou caixa',none='sem objeto transportado visível'),
    'clothing':dict(safety_vest='com colete de segurança',jacket_coat='com jaqueta ou casaco',shirt_tshirt='com camisa ou camiseta'),
}

def render(domain, answers):
    if answers.get('presence')!='yes': return {'text':None,'facts':{}}
    facts = {k:v for k,v in answers.items() if k!='presence' and v not in {'unknown','not_visible','parse_error','invalid'}}
    parts=[]
    for key,value in facts.items():
        word=WORDS[key].get(value)
        if word is not None: parts.append(f'{key}: {word}')
    # No new semantic fact can be added by a language model here.
    return {'text':('Veículo' if domain=='vehicle' else 'Pessoa')+(' — '+ '; '.join(parts) if parts else '')+'.', 'facts':facts}
