# Skills para esta reorganização

Pesquisa feita com a skill `find-skills`, consultando o leaderboard skills.sh, a busca `npx skills find webapp-testing` e as páginas das skills. As contagens abaixo são as exibidas no momento da consulta; variam com o tempo.

| Skill | Origem | Instalações exibidas | Uso neste projeto |
|---|---|---:|---|
| [frontend-design](https://skills.sh/anthropics/skills/frontend-design) | `anthropics/skills` | 1,0 milhão | Direção visual, estados de erro, responsividade e clareza da interface |
| [webapp-testing](https://skills.sh/anthropics/skills/webapp-testing) | `anthropics/skills` | 189 mil | Validação de fluxos reais no navegador com Python e Playwright |

As páginas indicavam aproximadamente 183,7 mil estrelas no repositório oficial `anthropics/skills`. As orientações foram consultadas para o trabalho; não houve instalação global de skills nem inclusão delas nas dependências da aplicação.

Para instalá-las no seu agente, se desejar:

```bash
npx skills add anthropics/skills@frontend-design
npx skills add anthropics/skills@webapp-testing
```

O painel usa HTML/CSS/JavaScript e Python existentes. Não foi acrescentado React/Next apenas para aplicar uma skill de framework.
