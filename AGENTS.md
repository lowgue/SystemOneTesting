# Agent Guidelines

## Python code and docstrings

- Write Python docstrings with triple double quotes for every module, class, function, and method. Describe purpose and, where useful, inputs, return values, side effects, errors, and behavioral constraints. Test docstrings should state the behavior being checked.
- Add type hints to public interfaces. Use descriptive names, standard four-space indentation, one statement per line, and small modules and functions with a single clear responsibility.
- Keep one authoritative implementation and import path. Put model request logic in its model adapter module, request and prompt contracts in `benchmarks/contracts/`, and chart generation in the matching `vision_bench/reporting/charts/` module.
- Do not add minified code, semicolon-separated logic, executable one-line function bodies, large modules that mix unrelated adapters, or duplicate compatibility modules that only re-export another implementation. When editing existing code, improve these patterns instead of copying them.
- These rules apply to project-maintained Python code, including scripts and tests. Do not reformat generated files, vendored code, or external submodules as project source.

## Agent skills

### Issue tracker

GitHub issues on `lowgue/SystemOneTesting`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repository (`GLOSSARY.md` + `docs/adr/` at root). See `docs/agents/domain.md`.
