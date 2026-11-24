"""Stable model identity across presentation and scientific figures."""

LAYA = "#3378FF"
BLIP = "#F78C25"
D1_OMNI = "#8B5CF6"
CLEF_FLASH = "#10B981"


def model_color(model: dict | str) -> str:
    """Select a stable display color from a model adapter or identifier."""
    name = (
        (model.get("adapter", "") + " " + model.get("model", model.get("id", "")))
        if isinstance(model, dict)
        else str(model)
    ).lower()
    if "laya" in name:
        return LAYA
    if "blip" in name:
        return BLIP
    if "d1" in name or "omni" in name:
        return D1_OMNI
    if "clef" in name:
        return CLEF_FLASH
    return "#64748B"


def model_label(model: dict | str) -> str:
    """Build a readable model and prompt-profile label for chart legends."""
    name = model["model"] if isinstance(model, dict) else str(model)
    name_lower = name.lower()
    if "laya" in name_lower:
        base = "Laya"
    elif "blip" in name_lower:
        base = "BLIP"
    elif "d1" in name_lower or "omni" in name_lower:
        base = "D1-Omni"
    elif "clef" in name_lower:
        base = "Clef-Flash"
    else:
        base = name

    variant = {
        "short_open": "pergunta curta",
        "short_options": "com alternativas",
        "long_descriptions": "descrição longa",
        "structured_json": "JSON",
        "short_serial": "perguntas separadas",
        "native_short": "alternativas curtas",
        "native_descriptive": "alternativas descritivas",
        "native_serial": "perguntas separadas",
    }.get(model.get("profile") if isinstance(model, dict) else None)
    return base + (f" · {variant}" if variant else "")


LABELS = {
    "presence": "Presença",
    "make": "Marca",
    "color": "Cor do veículo",
    "body": "Tipo de veículo",
    "helmet": "Capacete",
    "carrying": "Objeto carregado",
    "clothing": "Roupa",
    "black": "Preto",
    "white": "Branco",
    "gray_silver": "Cinza / prata",
    "red": "Vermelho",
    "blue": "Azul",
    "green": "Verde",
    "yellow": "Amarelo",
    "orange": "Laranja",
    "brown": "Marrom",
    "purple": "Roxo",
    "pink": "Rosa",
    "yes": "Sim",
    "no": "Não",
    "unknown": "Não identificado",
    "not_visible": "Não visível",
}


def label(value: str) -> str:
    """Return the display label for a field or taxonomy value."""
    return LABELS.get(value, value.replace("_", " "))


def baseline_models(models: list[dict]) -> list[dict]:
    """Select one baseline profile per available reference adapter."""
    selected = []
    adapters = ["blip", "laya", "d1", "clef"]
    for key in adapters:
        candidates = [
            m
            for m in models
            if key in m.get("adapter", "").lower() or key in m.get("model", "").lower()
        ]
        if candidates:
            selected.append(
                next(
                    (
                        m
                        for m in candidates
                        if m.get("profile")
                        in {"native_short", "short_open", "baseline"}
                    ),
                    candidates[0],
                )
            )
    return selected if selected else models
