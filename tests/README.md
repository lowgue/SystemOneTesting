# Testes

Ambiente Python 3.12:

```bash
python -m pip install -r requirements/cpu.txt   # ou models.txt / http.txt
python -m unittest discover -s tests
python -m unittest discover -s eval/tests
```

A suíte atual verifica o motor (métricas de dataset). Não baixa pesos nem usa APIs pagas.
