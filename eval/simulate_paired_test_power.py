"""Label-free sample-size sensitivity using exact two-sided paired binomial tests."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def critical_values(max_n, alpha):
    # Exact McNemar rejection region conditional on number of discordant pairs.
    logfact=np.array([math.lgamma(k+1) for k in range(max_n+1)])
    critical=np.full(max_n+1,-1,dtype=int)
    for n in range(1,max_n+1):
        k=np.arange(n//2+1)
        cdf=np.cumsum(np.exp(logfact[n]-logfact[k]-logfact[n-k]-n*math.log(2)))
        critical[n]=int(np.searchsorted(cdf,alpha/2,side='right')-1)
    return critical

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',required=True);args=ap.parse_args();out=Path(args.run_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    seed=20261026;trials=20000;delta=.05;sizes=[300,600,900,1200,1800,2400];rates=[.1,.2,.3,.5];rows=[]
    rng=np.random.default_rng(seed)
    for alpha in [.05,.025]:
        critical=critical_values(max(sizes),alpha)
        for q in rates:
            for n in sizes:
                draws=rng.multinomial(n,[(q+delta)/2,(q-delta)/2,1-q],size=trials)
                discordant=draws[:,0]+draws[:,1]
                rejected=np.minimum(draws[:,0],draws[:,1])<=critical[discordant]
                power=float(rejected.mean());se=math.sqrt(power*(1-power)/trials)
                rows.append(dict(n_pairs=n,discordance=q,delta=delta,alpha=alpha,power=power,monte_carlo_se=se))
    negatives=[]
    for family in [1,2,6]:
        alpha=.05/family
        negatives.append(dict(comparisons=family,alpha=alpha,n_zero_failures_for_upper_1pct=math.ceil(math.log(alpha)/math.log(.99)),upper_with_zero_of_300=1-alpha**(1/300)))
    obj=dict(seed=seed,trials_per_scenario=trials,method='Exact two-sided conditional McNemar via Binomial(discordant, 0.5); simulated multinomial pairs.',
             assumptions='Independent event pairs; fixed true difference +5 percentage points; discordance is assumed, not estimated from human-unreviewed labels.',rows=rows,negative_control_requirements=negatives)
    (out/'power.json').write_text(json.dumps(obj,indent=2)+'\n')
    lines=['# Simulação de poder — comparação pareada','',
           'Simulação sem imagens ou gabaritos: diferença real de **5 pontos percentuais**, 20.000 experimentos por cenário, teste binomial pareado exato bilateral. A discordância é a fração de eventos em que só um dos modelos acerta; aqui ela é um parâmetro hipotético.', '',
           'O teste detecta diferença contra zero. Não demonstra que o limite inferior da vantagem supere a margem prática de 5 pontos: isso requer uma diferença real maior que essa margem.', '',
           '| Pares independentes por domínio | Discordância | α por domínio | Poder estimado | Erro Monte Carlo (±1 SE) |', '|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["n_pairs"]} | {100*r["discordance"]:.0f}% | {r["alpha"]} | {100*r["power"]:.1f}% | {100*r["monte_carlo_se"]:.2f} pp |')
    lines+=['','α=0,025 por domínio corresponde a Bonferroni para duas comparações primárias dentro de uma família de 5%. Dependência entre cenas exige tratar grupos e pode reduzir o poder; a simulação não resolve a representatividade da coleta.', '',
            '## Controles negativos: limite superior unilateral de 1% com zero falhas','', '| Comparações na família (exemplo Bonferroni) | Negativos necessários | Limite superior com 0/300 |', '|---:|---:|---:|']
    for r in negatives:lines.append(f'| {r["comparisons"]} | {r["n_zero_failures_for_upper_1pct"]} | {100*r["upper_with_zero_of_300"]:.3f}% |')
    lines+=['','A fórmula usa o limite binomial exato unilateral `1 − α**(1/n)`. Zero falhas não significa taxa zero. O caso de seis comparações é um cenário de ajuste, não uma seleção definitiva da família de testes.', '',
            'Esses resultados orientam a coleta antes de abrir o teste final. Não há conclusão sobre qual modelo é melhor e não se deve ampliar a amostra até obter um resultado favorável.', '', '[Resultados e parâmetros](power.json)']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print(out/'report.md')

if __name__=='__main__':main()
