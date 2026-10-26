# Interface do laboratório visual

A interface serve quem precisa verificar uma imagem, comparar respostas e acompanhar uma avaliação no próprio computador. A área de imagem é o centro do trabalho; os resultados aparecem ao lado conforme cada modelo termina.

## Direção visual

Paleta: fundo gelo `#f3f6fa`, superfície branca `#ffffff`, texto azul ardósia `#24364b`, ação azul `#175fbd`, sucesso verde `#18745a`, falha vinho `#a13045`. Tipografia: Segoe UI e fontes do sistema para funcionamento sem rede; números tabulares nas medições. Alinhamento à esquerda, navegação lateral e área de trabalho ampla.

```text
Navegação       Título da tarefa + estado do ambiente
                Configuração compartilhada
Teste de imagem Imagem enviada       Respostas por modelo
Benchmark       Opções da rodada     Progresso e registros
Resultados      Execuções locais     Gráficos e arquivos
Diagnóstico     Serviço Ollama       Versão, modelos e memória
```

O destaque visual é o espaço de inspeção da imagem. As demais áreas seguem a tarefa: formulário no benchmark, lista de arquivos nos resultados e diagnóstico textual. Isso evita transformar todas as informações em cartões iguais. Em telas menores, navegação horizontal e colunas empilhadas.

## Regras de interação

- Nenhuma inferência inicia ao abrir a página ou trocar a imagem.
- Carregamento de pesos e inferência têm estados separados.
- Comparação usa a mesma imagem, domínio e contrato, sequencialmente.
- Teste interativo não é uma avaliação estatística; latência inclui a primeira inferência, sem aquecimento.
- Benchmark chama o runner existente, com aquecimento, configuração congelada e relatórios separados por computador.
- Erros aparecem junto ao modelo, com orientação; falha de um modelo não apaga os demais resultados.
- Navegação por teclado, rótulos nos formulários, estados anunciados e CSS local sem CDN.

Orientações consultadas: [frontend-design](https://skills.sh/anthropics/skills/frontend-design) e [webapp-testing](https://skills.sh/anthropics/skills/webapp-testing), da Anthropic. Não são dependências da aplicação.
