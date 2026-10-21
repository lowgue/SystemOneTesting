"""Exemplos de uso da API /v1/systemone do Laya textual (via Ollama)."""

import json
import time
import urllib.request
import urllib.error
from typing import Any

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "laya"


def query(
    state: str,
    questions: dict[str, Any],
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_URL,
) -> dict[str, Any]:
    """Envia um estado e perguntas tipadas ao Laya. Retorna as respostas com latência."""
    payload = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/systemone",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Não foi possível conectar ao Ollama em {base_url}. "
            "Certifique-se de que 'ollama serve' está em execução."
        ) from exc
    latency_ms = (time.perf_counter() - t0) * 1000
    result["_latency_ms"] = round(latency_ms, 2)
    return result


# ---------------------------------------------------------------------------
# Casos de exemplo – domínios de decisão reais
# ---------------------------------------------------------------------------

EXAMPLES = [
    # 1. Concessão de crédito
    {
        "name": "credito_alto_risco",
        "domain": "crédito",
        "state": (
            "Cliente pessoa física. Renda mensal: R$ 1.800. "
            "Score de crédito: 420. Histórico: 3 parcelas em atraso nos últimos 12 meses. "
            "Valor solicitado: R$ 5.000. Sem garantias."
        ),
        "questions": {
            "aprovar": {
                "type": "noul",
                "instructions": "Deve aprovar a concessão de crédito a este cliente?",
            },
            "risco": {
                "type": "choice",
                "instructions": "Qual é o nível de risco de inadimplência deste cliente?",
                "criteria": {
                    "baixo": "Baixo – probabilidade de inadimplência inferior a 10%",
                    "medio": "Médio – probabilidade entre 10% e 30%",
                    "alto": "Alto – probabilidade superior a 30%",
                    "muito_alto": "Muito alto – inadimplência praticamente certa",
                },
            },
        },
        "expected": {"aprovar": False, "risco": "muito_alto"},
    },
    # 2. Triagem médica de urgência
    {
        "name": "triagem_urgencia",
        "domain": "saúde",
        "state": (
            "Paciente masculino, 58 anos. Queixa: dor no peito intensa há 20 minutos, "
            "irradiando para o braço esquerdo. Sudorese fria. PA: 150/95 mmHg. "
            "FC: 105 bpm. Sem histórico registrado."
        ),
        "questions": {
            "emergencia": {
                "type": "noul",
                "instructions": "O paciente deve ser atendido como emergência imediata (prioridade máxima)?",
            },
            "prioridade": {
                "type": "choice",
                "instructions": "Qual classificação de triagem Manchester deve ser atribuída?",
                "criteria": {
                    "vermelho": "Vermelho – emergência, atendimento imediato",
                    "laranja": "Laranja – muito urgente, até 10 minutos",
                    "amarelo": "Amarelo – urgente, até 60 minutos",
                    "verde": "Verde – pouco urgente, até 120 minutos",
                    "azul": "Azul – não urgente",
                },
            },
        },
        "expected": {"emergencia": True, "prioridade": "vermelho"},
    },
    # 3. Moderação de conteúdo
    {
        "name": "moderacao_conteudo",
        "domain": "moderação",
        "state": (
            "Publicação em rede social: "
            "'Você é idiota e merece apanhar por defender essa posição. "
            "Pessoas como você deveriam desaparecer.' "
            "Contexto: comentário em debate político."
        ),
        "questions": {
            "remover": {
                "type": "noul",
                "instructions": "Esta publicação deve ser removida por violar regras de conduta (ameaça, discurso de ódio ou incitação)?",
            },
            "categoria": {
                "type": "choice",
                "instructions": "Qual é a principal categoria de violação nesta publicação?",
                "criteria": {
                    "ameaca": "Ameaça ou incitação à violência",
                    "assedio": "Assédio ou bullying",
                    "odio": "Discurso de ódio",
                    "spam": "Spam ou desinformação",
                    "nenhuma": "Nenhuma violação identificada",
                },
            },
        },
        "expected": {"remover": True, "categoria": "ameaca"},
    },
    # 4. Roteamento de suporte ao cliente
    {
        "name": "suporte_roteamento",
        "domain": "suporte",
        "state": (
            "Ticket #4821. Cliente: 'Meu cartão foi cobrado duas vezes no valor de R$ 349,90. "
            "Já tentei cancelar pelo app mas o botão não funciona. "
            "Preciso de estorno urgente.' Plano: Premium. LTV: R$ 12.000."
        ),
        "questions": {
            "escalar": {
                "type": "noul",
                "instructions": "Este ticket deve ser escalado para um agente humano especializado (em vez de ser resolvido por bot)?",
            },
            "departamento": {
                "type": "choice",
                "instructions": "Para qual departamento este ticket deve ser roteado?",
                "criteria": {
                    "financeiro": "Financeiro / estornos e cobranças",
                    "tecnico": "Técnico / bugs e falhas no produto",
                    "retencao": "Retenção / clientes em risco de churn",
                    "geral": "Suporte geral",
                },
            },
            "urgencia": {
                "type": "score",
                "instructions": "Qual é o grau de urgência deste ticket?",
                "criteria": {
                    "1": "Baixíssima urgência",
                    "2": "Baixa urgência",
                    "3": "Urgência moderada",
                    "4": "Alta urgência",
                    "5": "Urgência crítica – risco imediato de perda do cliente",
                },
            },
        },
        "expected": {"escalar": True, "departamento": "financeiro", "urgencia": "5"},
    },
    # 5. Qualificação de lead de vendas
    {
        "name": "lead_qualificacao",
        "domain": "vendas",
        "state": (
            "Lead: empresa de logística, 85 funcionários, setor de transporte. "
            "Formulário preenchido: interesse em 'otimização de rotas'. "
            "Cargo do contato: analista de TI júnior. "
            "Orçamento declarado: 'ainda não definido'. "
            "Prazo: 'talvez no próximo ano'."
        ),
        "questions": {
            "qualificado": {
                "type": "noul",
                "instructions": "Este lead está qualificado para uma abordagem de vendas ativa agora?",
            },
            "estagio": {
                "type": "choice",
                "instructions": "Em qual estágio do funil de vendas este lead se encontra?",
                "criteria": {
                    "conscientizacao": "Consciência – conhece o problema, não a solução",
                    "consideracao": "Consideração – está avaliando soluções",
                    "decisao": "Decisão – pronto para comprar",
                    "nurturing": "Nutrição – potencial futuro, não está pronto agora",
                },
            },
        },
        "expected": {"qualificado": False, "estagio": "nurturing"},
    },
]


def run_examples() -> None:
    print(f"{'='*60}")
    print("  Laya Textual – Exemplos de Decisão")
    print(f"{'='*60}\n")

    for ex in EXAMPLES:
        print(f"── {ex['name']} [{ex['domain']}] ──")
        print(f"Estado: {ex['state'][:120]}...")
        try:
            result = query(ex["state"], ex["questions"])
        except RuntimeError as exc:
            print(f"  ERRO: {exc}\n")
            continue

        answers = result.get("answers", {})
        latency = result.get("_latency_ms", "?")
        print(f"Latência: {latency} ms")
        print("Respostas:")
        for key, ans in answers.items():
            expected = ex["expected"].get(key, "?")
            # Determina o valor predito para comparação
            atype = ans.get("type")
            if atype == "noul":
                prob = ans["noul"]
                predicted = prob >= 0.5
                print(f"  {key}: prob={prob:.3f} → {'Sim' if predicted else 'Não'}  (esperado: {'Sim' if expected else 'Não'})")
            elif atype == "choice":
                predicted = max(ans["choice"], key=ans["choice"].get)
                print(f"  {key}: {predicted}  (esperado: {expected})")
                probs = {k: f"{v:.3f}" for k, v in ans["choice"].items()}
                print(f"    probabilidades: {probs}")
            elif atype == "score":
                predicted = max(ans["score"], key=ans["score"].get)
                print(f"  {key}: {predicted}  (esperado: {expected})")
        print()


if __name__ == "__main__":
    run_examples()
