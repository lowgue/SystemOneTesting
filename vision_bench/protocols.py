"""Prompt profiles and request diagnostics for visual model evaluations."""

from copy import deepcopy
import json
import re


PROFILES = {
    "blip": [
        {"id": "short_open", "name": "Pergunta curta aberta", "group": "baseline"},
        {"id": "short_options", "name": "Pergunta com alternativas", "group": "prompts"},
        {"id": "long_descriptions", "name": "Alternativas descritivas", "group": "prompts"},
        {"id": "structured_json", "name": "Solicitação de JSON", "group": "prompts"},
        {
            "id": "short_serial",
            "name": "Pergunta curta, uma inferência por atributo",
            "group": "requests",
        },
    ],
    "laya": [
        {
            "id": "native_short",
            "name": "Alternativas nativas curtas, request conjunto",
            "group": "baseline",
        },
        {
            "id": "native_descriptive",
            "name": "Alternativas nativas descritivas",
            "group": "prompts",
        },
        {
            "id": "native_serial",
            "name": "Alternativas curtas, um request por atributo",
            "group": "requests",
        },
    ],
    "systemone_http": [
        {"id": "native_short", "name": "Alternativas nativas curtas", "group": "baseline"},
        {"id": "native_descriptive", "name": "Alternativas nativas descritivas", "group": "prompts"},
    ],
    "d1_omni": [
        {"id": "native_short", "name": "Alternativas nativas curtas", "group": "baseline"},
        {"id": "native_descriptive", "name": "Alternativas nativas descritivas", "group": "prompts"},
    ],
}

MODES = {"baseline", "prompts", "requests", "all"}

DESCRIPTIONS = {
    "black": "Black or deep dark tone",
    "white": "White or off-white bright tone",
    "gray_silver": "Gray or silver metallic tone",
    "red": "Red, crimson, or cherry tone",
    "blue": "Blue, navy, or sky blue tone",
    "green": "Green, forest, or lime tone",
    "yellow": "Yellow or bright golden tone",
    "orange": "Orange or amber tone",
    "brown": "Brown, bronze, or earth tone",
    "purple": "Purple or violet tone",
    "pink": "Pink or rose tone",
    "sedan": "Sedan with separate trunk",
    "suv": "SUV with tall enclosed body",
    "hatch": "Hatchback with rear hatch",
    "pickup": "Pickup with open cargo bed",
    "bus": "Bus for many passengers",
    "motorcycle": "Motorcycle with two wheels",
    "safety_vest": "Reflective safety vest",
    "jacket_coat": "Jacket or outer coat",
    "shirt_tshirt": "Shirt or T-shirt",
    "backpack": "Backpack carried on the back",
    "bag_or_box": "Bag or box carried by hand",
}


def expand_specs(specs, mode):
    """Expand each model into the prompt profiles selected by ``mode``."""
    if mode not in MODES:
        raise ValueError("Comparação inválida.")

    expanded = []
    for spec in specs:
        profiles = PROFILES.get(spec["adapter"])
        if not profiles or mode == "baseline":
            expanded.append(dict(spec))
            continue

        for profile in profiles:
            if profile["group"] != "baseline" and mode != "all" and profile["group"] != mode:
                continue
            expanded.append(
                dict(
                    spec,
                    id=f"{spec['id']}--{profile['id']}",
                    base_model=spec["id"],
                    profile=profile["id"],
                )
            )
    return expanded


def item_contract(contract, item):
    """Return a contract narrowed to the fields required by one manifest item."""
    result = deepcopy(contract)
    domain = result["domains"][item["domain"]]
    fields = item.get("task_fields") or (
        [item["task_field"]] if item.get("task_field") else list(domain["questions"])
    )
    domain["questions"] = {
        key: value for key, value in domain["questions"].items() if key in fields
    }
    domain["prompts"] = {
        key: value for key, value in domain["prompts"].items() if key in fields
    }

    if item.get("question") and len(fields) == 1:
        field = fields[0]
        domain["prompts"][field] = item["question"]
        domain["questions"][field]["instructions"] = item["question"]
    return result


def questions(spec, contract, domain):
    """Build the model-specific question choices for a domain."""
    result = deepcopy(contract["domains"][domain]["questions"])
    if spec.get("profile") == "native_descriptive":
        for question in result.values():
            question["criteria"] = {
                key: DESCRIPTIONS.get(key, f"{label}: visible {label.lower()}")
                for key, label in question["criteria"].items()
            }
    return result


def prompts(spec, contract, domain):
    """Build the free text prompt variants requested by a model profile."""
    result = {}
    profile = spec.get("profile", "short_open")
    domain_contract = contract["domains"][domain]

    for key, prompt in domain_contract["prompts"].items():
        choices = domain_contract["questions"][key]["criteria"]
        if profile == "short_options":
            prompt += " Choose one: " + ", ".join(choices.values()) + "."
        elif profile == "long_descriptions":
            descriptions = "; ".join(
                DESCRIPTIONS.get(choice, f"{label}: visible {label.lower()}")
                for choice, label in choices.items()
            )
            prompt += " Choose the visible attribute from: " + descriptions + ". Return one label."
        elif profile == "structured_json":
            prompt += (
                ' Return only a JSON object with exactly one key: {"'
                + key
                + '": "<label>"}. Allowed labels: '
                + ", ".join(choices)
                + "."
            )
        result[key] = prompt
    return result


def describe(spec, contract, domain):
    """Describe one model protocol and its observable evaluation metrics."""
    adapter = spec["adapter"]
    model_questions = questions(spec, contract, domain)
    default_profile = "short_open" if adapter == "blip" else "native_short"
    profile = spec.get("profile", default_profile)
    serial = profile in {"short_serial", "native_serial"}

    if adapter == "blip":
        output_format = (
            "JSON textual solicitado; validação estrita"
            if profile == "structured_json"
            else "Texto gerado e normalizado"
        )
        request = {
            "images": ["<imagem RGB, sem dados binários neste exemplo>"],
            "text": prompts(spec, contract, domain),
        }
    else:
        output_format = "answers.<atributo>.choice (alternativa nativa)"
        request = {"images": ["<imagem RGB>"], "questions": model_questions}

    return {
        "adapter": adapter,
        "base_model": spec.get("base_model", spec["id"]),
        "profile": profile,
        "question_type": "Texto livre VQA" if adapter == "blip" else "choice: alternativas estruturadas",
        "output_format": output_format,
        "request_mode": (
            "Uma chamada por atributo"
            if serial
            else "Batch de perguntas com imagem repetida"
            if adapter == "blip"
            else "Um request com todas as perguntas"
        ),
        "calls_per_photo": len(model_questions) if serial else 1,
        "generation": (
            {
                "max_new_tokens": 64 if profile == "structured_json" else 12,
                "do_sample": False,
            }
            if adapter == "blip"
            else None
        ),
        "head_max_len": spec.get("head_max_len") if adapter == "laya" else None,
        "request": request,
        "metrics": [
            "Acurácia por atributo",
            "Precisão macro",
            "Recall macro",
            "F1 macro",
            "Aderência às alternativas antes da normalização",
            "Formato válido",
            "Ambiguidade",
            "Latência média, p50 e p95",
            "Chamadas de inferência por foto",
            "CPU e memória RSS",
        ],
    }


def _normalize_answer(value):
    """Normalize answer text for adherence and ambiguity checks."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _text_diagnostics(value, choices):
    """Measure whether free text matches allowed labels and contains competing labels."""
    normalized = _normalize_answer(value)
    adherent = normalized in {_normalize_answer(item) for item in [*choices, *choices.values()]}

    # Exclude nested class names such as "truck" inside "pickup truck".
    hits = {
        key
        for key, label in choices.items()
        if re.search(r"(?<!\w)" + re.escape(_normalize_answer(label)) + r"(?!\w)", normalized)
    }
    hits = {
        key
        for key in hits
        if not any(
            key != other and _normalize_answer(choices[key]) in _normalize_answer(choices[other])
            for other in hits
        )
    }
    return adherent, len(hits) > 1


def output_diagnostics(spec, contract, domain, raw):
    """Measure formatting, answer adherence and ambiguity per question."""
    result = {}
    for key, question in contract["domains"][domain]["questions"].items():
        if spec["adapter"] == "blip":
            value = raw.get(key, "")
            format_valid = isinstance(value, str) and bool(value.strip())

            if spec.get("profile") == "structured_json":
                try:
                    parsed = json.loads(value)
                    format_valid = (
                        isinstance(parsed, dict)
                        and set(parsed) == {key}
                        and isinstance(parsed[key], str)
                    )
                    value = parsed[key] if format_valid else ""
                except (ValueError, KeyError, TypeError):
                    format_valid = False
                    value = ""

            choices = question["criteria"]
            adherent, ambiguous = _text_diagnostics(value, choices)
            if spec.get("profile") == "structured_json":
                adherent = isinstance(value, str) and value in choices
                format_valid = bool(format_valid and adherent)
        else:
            value = raw.get("answers", {}).get(key, {}).get("choice")
            format_valid = isinstance(value, str)
            adherent = value in question["criteria"]
            ambiguous = False

        result[key] = {
            "format_valid": format_valid,
            "adherent": adherent,
            "ambiguous": ambiguous,
        }
    return result
