import React, {useState} from 'react';
import {labels} from '../presentation.js';
const pct=v=>v==null?'Não medido':`${(100*v).toFixed(1)}%`;
export function ModelProtocol({model}){
 return <details className="method"><summary>Perguntas, formato de saída e métricas desta variante</summary>
 <p>Modelo: <strong>{model.base_model || model.model}</strong> · Variante: <strong>{model.profile || 'padrão'}</strong> · Adaptador: {model.adapter || 'não registrado'}</p>
 <p>{model.requests} avaliações de foto · {model.inference_calls || 'Não registrado'} chamadas internas de inferência. Uma avaliação pode conter várias perguntas.</p>
 {(model.protocols || []).map((p,i)=><details key={i}><summary>{p.profile}: {p.request_mode} · {p.calls_per_photo} chamada(s)</summary>
 <p>{p.question_type}. Saída: {p.output_format}.</p>
 {p.generation && <p>Geração: {p.generation.max_new_tokens} tokens novos no máximo; do_sample = {String(p.generation.do_sample)}.</p>}
 {p.head_max_len && <p>Orçamento head_max_len: {p.head_max_len}.</p>}
 <p>Métricas registradas: {p.metrics?.join('; ')}.</p><pre>{JSON.stringify(p.request,null,2)}</pre></details>)}
 {!model.protocols?.length && <p>Este relatório anterior não registrou o request detalhado. Execute uma nova rodada para obter essa informação.</p>}
 </details>
}
export default function ProtocolComparison({data}){
 const [field,setField]=useState('all');const rows=(data.comparisons || []).filter(r=>field==='all'||`${r.domain}/${r.field}`===field);
 if(!data.comparisons?.length)return null;
 return <section><h3>Comparações pareadas de modelos e prompts</h3><p>Diferença = acurácia de B menos acurácia de A nas mesmas fotos. Intervalos de 95% usam bootstrap dos grupos de origem; não são teste de significância. Gabaritos exploratórios continuam sujeitos a revisão.</p>
 <label htmlFor="compare-field">Atributo da comparação</label><select id="compare-field" value={field} onChange={e=>setField(e.target.value)}><option value="all">Todos os atributos</option>{[...new Set(data.comparisons.map(r=>`${r.domain}/${r.field}`))].map(k=><option key={k}>{k}</option>)}</select>
 <div className="table-scroll"><table><thead><tr><th>A</th><th>B</th><th>Atributo</th><th>Eventos / grupos</th><th>Diferença B − A</th><th>IC 95%</th><th>Só A acerta / só B acerta</th></tr></thead><tbody>{rows.map((r,i)=><tr key={i}><td>{r.a}</td><td>{r.b}</td><td>{r.domain} / {labels[r.field] || r.field}</td><td>{r.n} / {r.groups}</td><td>{(100*r.delta_b_minus_a).toFixed(1)} p.p.</td><td>{r.ci95 ? `${(r.ci95[0]*100).toFixed(1)} a ${(r.ci95[1]*100).toFixed(1)} p.p.` : 'Grupos insuficientes'}</td><td>{r.a_only_correct} / {r.b_only_correct}</td></tr>)}</tbody></table></div>
 </section>
}
export function AttributeCharts({data,base}){
 const [selected,setSelected]=useState(''),[level,setLevel]=useState('simple');const allCharts=data.charts || [];
 const hasSimple=allCharts.some(c=>c.level==='simple');
 const charts=hasSimple ? allCharts.filter(c=>level==='simple' ? c.level==='simple' : c.level!=='simple') : allCharts;
 if(!charts.length)return null;
 const chart=charts.find(c=>c.file===selected) || charts[0];
 return <section className="attribute-charts"><h3>Gráficos por atributo e por classe</h3><p>Laya em azul · BLIP em laranja. Os gráficos simples usam os protocolos padrão identificados na legenda, sem misturar resultados de variantes.</p>
 {hasSimple && <><label htmlFor="chart-level">Tipo de gráfico</label><select id="chart-level" value={level} onChange={e=>{setLevel(e.target.value);setSelected('')}}><option value="simple">Simples: acurácia e tempo</option><option value="technical">Técnicos: variantes, F1 e matrizes</option></select></>}
 <label htmlFor="attribute-chart">Escolher gráfico</label><select id="attribute-chart" value={chart.file} onChange={e=>setSelected(e.target.value)}>{charts.map(c=><option key={c.file} value={c.file}>{c.title}</option>)}</select>
 <figure><img loading="lazy" src={base+chart.file+'.png'} alt={chart.title}/><figcaption>{chart.title} · <a href={base+chart.file+'.pdf'} target="_blank" rel="noreferrer">Baixar PDF</a> · <a href={base+chart.file+'.png'} target="_blank" rel="noreferrer">PNG 300 dpi</a></figcaption></figure>
 </section>
}
