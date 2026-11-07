"""Dataset evaluation: quality on first repetition, performance on measured calls."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from itertools import combinations

import numpy as np

from ..core import load_json, write_json


def ratio(correct, total):
    """Return a ratio, or None when its denominator is zero."""
    return correct / total if total else None


def field_scores(pairs):
    """Invalid predictions remain errors. Macro F1 uses classes with truth support."""
    classes = sorted({truth for truth, _ in pairs})
    matrix = {
        truth: dict(Counter(pred for expected, pred in pairs if expected == truth))
        for truth in classes
    }
    per_class = []
    for label in classes:
        tp = sum(a == label and b == label for a, b in pairs)
        fp = sum(a != label and b == label for a, b in pairs)
        fn = sum(a == label and b != label for a, b in pairs)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class.append(
            dict(
                label=label, support=tp + fn, precision=precision, recall=recall, f1=f1
            )
        )
    correct = sum(a == b for a, b in pairs)
    return dict(
        n=len(pairs),
        correct=correct,
        accuracy=ratio(correct, len(pairs)),
        macro_precision=(
            statistics.mean(c["precision"] for c in per_class) if per_class else None
        ),
        macro_recall=(
            statistics.mean(c["recall"] for c in per_class) if per_class else None
        ),
        macro_f1=statistics.mean(c["f1"] for c in per_class) if per_class else None,
        classes=per_class,
        confusion=matrix,
    )


def evaluate(frozen, rows, loads=None):
    """Compute quality, latency, resource, adherence, and paired-comparison metrics for a completed run."""
    ids = frozen["item_ids"]
    lookup = {
        it["id"]: it for it in frozen["manifest"]["items"] if it["id"] in set(ids)
    }
    if len({r["request_key"] for r in rows}) != len(rows):
        raise ValueError("Duplicate predictions cannot be used for evaluation")
    summary = dict(
        schema_version=1,
        dataset_name=frozen["manifest"].get("name", "Dataset de exemplo"),
        label_status=frozen["manifest"].get(
            "label_status", "exploratory_unadjudicated"
        ),
        simulated=any(s["adapter"] == "mock" for s in frozen["models"]),
        photos=len(ids),
        unique_images=len({it["sha256"] for it in lookup.values()}),
        repetitions=frozen["repetitions"],
        models=[],
        methodology=dict(
            quality="Primeira repetição por foto; falhas contam como erros. Campos sem gabarito são excluídos.",
            exact="Todos os atributos corretos na mesma foto, somente em fotos com gabarito completo.",
            latency="Chamadas medidas com sucesso, incluindo pré/pós-processamento; carga e aquecimento excluídos.",
            macro_f1="Média por classe com suporte no gabarito; divisão por zero = 0.",
            resources="HTTP sem PID mede o cliente, não o consumo do servidor Ollama.",
        ),
    )
    for spec in frozen["models"]:
        model = spec["id"]
        rs = [r for r in rows if r["model"] == model]
        first = {r["id"]: r for r in rs if r["block"] == 0}
        times = [r["elapsed_ms"] for r in rs if not r.get("error")]
        photos = []
        fields = {}
        exact_n = exact_correct = 0
        for item_id in ids:
            item = lookup[item_id]
            row = first.get(item_id, {})
            expected = item.get("expected") or item.get("annotations") or {}
            predicted = row.get("predicted", {})
            error = row.get("error") or (None if row else "Previsão ausente")
            correct = {
                key: bool(not error and predicted.get(key) == value)
                for key, value in expected.items()
            }
            for key, value in expected.items():
                fields.setdefault((item["domain"], key), []).append(
                    (value, "__error__" if error else predicted.get(key, "parse_error"))
                )
            complete = set(expected) == set(
                frozen["contract"]["domains"][item["domain"]]["questions"]
            )
            exact = None
            if complete:
                exact_n += 1
                exact = all(correct.values())
                exact_correct += exact
            photos.append(
                dict(
                    id=item_id,
                    filename=item.get("filename", Path(item["rel_path"]).name),
                    domain=item["domain"],
                    expected=expected,
                    predicted=predicted,
                    correct=correct,
                    exact=exact,
                    error=error,
                    elapsed_ms=row.get("elapsed_ms"),
                    valid=bool(not error and row.get("valid")),
                )
            )
        field_results = [
            dict(domain=domain, field=key, **field_scores(pairs))
            for (domain, key), pairs in fields.items()
        ]
        for field in field_results:
            key = field["field"]
            domain = field["domain"]
            measured = [
                r for r in rs if r.get("domain", lookup[r["id"]]["domain"]) == domain
            ]
            times_field = [
                r["protocol_metrics"]["attribute_ms"][key]
                for r in measured
                if not r.get("error")
                and key in r.get("protocol_metrics", {}).get("attribute_ms", {})
            ]
            diagnostics = [
                r["output_diagnostics"][key]
                for r in first.values()
                if r.get("domain", lookup[r["id"]]["domain"]) == domain
                and key in r.get("output_diagnostics", {})
            ]
            field.update(
                attribute_latency=dict(
                    n=len(times_field),
                    mean_ms=statistics.mean(times_field) if times_field else None,
                    p50_ms=float(np.median(times_field)) if times_field else None,
                    p95_ms=(
                        float(np.quantile(times_field, 0.95)) if times_field else None
                    ),
                ),
                raw_adherence=ratio(
                    sum(d["adherent"] for d in diagnostics), len(diagnostics)
                ),
                output_format_valid=ratio(
                    sum(d["format_valid"] for d in diagnostics), len(diagnostics)
                ),
                ambiguity_rate=ratio(
                    sum(d["ambiguous"] for d in diagnostics), len(diagnostics)
                ),
            )
        field_n = sum(f["n"] for f in field_results)
        field_correct = sum(f["correct"] for f in field_results)
        total_ms = sum(r["elapsed_ms"] for r in rs)
        successful = sum(not bool(r.get("error")) for r in rs)
        resources = sorted({r.get("resource_scope", "unknown") for r in rs})
        model_loads = [row for row in (loads or []) if row["model"] == model]
        protocols = []
        for row in rs:
            if row.get("request") and row["request"] not in protocols:
                protocols.append(row["request"])
        result = dict(
            model=model,
            base_model=spec.get("base_model", model),
            adapter=spec["adapter"],
            profile=spec.get(
                "profile",
                (
                    "short_open"
                    if spec["adapter"] == "blip"
                    else "native_short" if spec["adapter"] == "laya" else "baseline"
                ),
            ),
            protocols=protocols,
            inference_calls=sum(
                r.get("protocol_metrics", {}).get("inference_calls", 0) for r in rs
            ),
            simulated=spec["adapter"] == "mock",
            photos=len(ids),
            requests=len(rs),
            annotated_photos=sum(bool(p["expected"]) for p in photos),
            complete_labeled_photos=exact_n,
            exact_correct=exact_correct,
            exact_accuracy=ratio(exact_correct, exact_n),
            field_correct=field_correct,
            field_n=field_n,
            field_accuracy=ratio(field_correct, field_n),
            fields=field_results,
            photo_results=photos,
            errors=len(rs) - successful,
            success_rate=ratio(successful, len(rs)),
            schema_valid_rate=ratio(
                sum(bool(not r.get("error") and r.get("valid")) for r in rs), len(rs)
            ),
            latency=dict(
                mean_ms=statistics.mean(times) if times else None,
                p50_ms=float(np.median(times)) if times else None,
                p95_ms=float(np.quantile(times, 0.95)) if times else None,
                min_ms=min(times) if times else None,
                max_ms=max(times) if times else None,
                std_ms=(
                    statistics.stdev(times)
                    if len(times) > 1
                    else (0 if times else None)
                ),
            ),
            measured_s=total_ms / 1000,
            successful_calls_s=successful / (total_ms / 1000) if total_ms else None,
            mean_cpu_s=statistics.mean(r.get("cpu_s", 0) for r in rs) if rs else None,
            peak_rss_mib=max((r.get("peak_rss_bytes", 0) for r in rs), default=0)
            / 2**20,
            resource_scope=resources,
            load_s=sum(r.get("load_or_client_init_s", 0) for r in model_loads),
            warmup_s=sum(r.get("warmup_s", 0) for r in model_loads),
            warmup_failures=sum(
                len(r.get("warmup_failed_domains", [])) for r in model_loads
            ),
        )
        summary["models"].append(result)
    summary["comparisons"] = paired_comparisons(summary, lookup)
    summary["methodology"].update(
        protocol="BLIP gera texto; Laya escolhe alternativas. Aderência bruta é medida antes da normalização e do filtro de ausência. JSON exige objeto exato e classe válida.",
        attribute_latency="Tempo por atributo é medido somente quando há chamada isolada. O tempo de um batch ou request conjunto não é dividido entre atributos.",
        comparisons="Diferenças pareadas usam as mesmas fotos. IC 95% por bootstrap de grupos de origem (2.000 reamostragens, seed fixa); não equivale a adjudicação dos gabaritos. Não há teste de significância nem seleção automática de vencedor.",
        cost="Custo monetário e energia não foram medidos. CPU, RAM e tempo local não equivalem a preço de API.",
    )
    return summary


def paired_comparisons(summary, lookup):
    """Compare models on shared examples and bootstrap confidence intervals by source group."""
    result = []
    for a, b in combinations(summary["models"], 2):
        left = {p["id"]: p for p in a["photo_results"]}
        right = {p["id"]: p for p in b["photo_results"]}
        fields = sorted(
            {(p["domain"], key) for p in left.values() for key in p["correct"]}
        )
        for domain, key in fields:
            pairs = [
                (pid, p["correct"][key], right[pid]["correct"][key])
                for pid, p in left.items()
                if p["domain"] == domain
                and key in p["correct"]
                and pid in right
                and key in right[pid]["correct"]
            ]
            if not pairs:
                continue
            groups = {}
            for pid, x, y in pairs:
                item = lookup[pid]
                g = item.get("group_id", item["sha256"])
                bucket = groups.setdefault(g, [0, 0])
                bucket[0] += int(y) - int(x)
                bucket[1] += 1
            values = np.asarray(list(groups.values()), dtype=float)
            interval = None
            if len(values) > 1:
                rng = np.random.default_rng(20261027)
                deltas = []
                batch = max(1, min(200, 1_000_000 // len(values)))
                for start in range(0, 2000, batch):
                    draws = rng.integers(
                        0, len(values), (min(batch, 2000 - start), len(values))
                    )
                    samples = values[draws].sum(axis=1)
                    deltas.extend(samples[:, 0] / samples[:, 1])
                interval = [float(v) for v in np.quantile(deltas, [0.025, 0.975])]
            result.append(
                dict(
                    a=a["model"],
                    b=b["model"],
                    domain=domain,
                    field=key,
                    n=len(pairs),
                    groups=len(groups),
                    delta_b_minus_a=sum(int(y) - int(x) for _, x, y in pairs)
                    / len(pairs),
                    ci95=interval,
                    b_only_correct=sum(not x and y for _, x, y in pairs),
                    a_only_correct=sum(x and not y for _, x, y in pairs),
                )
            )
    return result


def generate(run, output):
    """Write machine-readable evaluation tables, charts, and the summary document for a completed run."""
    run = Path(run)
    output = Path(output)
    frozen = load_json(run / "frozen_config.json")
    rows = [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text().splitlines()
    ]
    status = load_json(run / "status.json")
    if status["status"] != "complete" or len(rows) != status["planned"]:
        raise ValueError("Only complete runs can produce a final dataset evaluation")
    loads = (
        [json.loads(line) for line in (run / "loads.jsonl").read_text().splitlines()]
        if (run / "loads.jsonl").exists()
        else []
    )
    summary = evaluate(frozen, rows, loads)
    metadata = load_json(run / "run.json")
    summary.update(
        machine=metadata["machine_id"],
        contract_hash=metadata["contract_hash"],
        dataset_hash=metadata["dataset_hash"],
    )
    write_json(
        output / "protocols.json",
        [dict(model=m["model"], protocols=m["protocols"]) for m in summary["models"]],
    )
    columns = [
        "model",
        "base_model",
        "adapter",
        "profile",
        "inference_calls",
        "simulated",
        "photos",
        "annotated_photos",
        "complete_labeled_photos",
        "requests",
        "repetitions",
        "exact_correct",
        "exact_accuracy",
        "field_correct",
        "field_n",
        "field_accuracy",
        "errors",
        "success_rate",
        "schema_valid_rate",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "successful_calls_s",
        "measured_s",
        "load_s",
        "warmup_s",
        "warmup_failures",
        "mean_cpu_s",
        "peak_rss_mib",
        "resource_scope",
    ]
    with (output / "models.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for model in summary["models"]:
            row = {key: model["latency"].get(key, model.get(key)) for key in columns}
            row.update(
                repetitions=summary["repetitions"],
                resource_scope=",".join(model["resource_scope"]),
            )
            writer.writerow(row)
    with (output / "photos.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "filename",
                "domain",
                "elapsed_ms",
                "valid",
                "exact",
                "error",
                "expected",
                "predicted",
            ],
        )
        writer.writeheader()
        for model in summary["models"]:
            for photo in model["photo_results"]:
                row = {key: photo.get(key) for key in writer.fieldnames}
                row["model"] = model["model"]
                for key in ["expected", "predicted"]:
                    row[key] = json.dumps(row[key], ensure_ascii=False)
                writer.writerow(row)
    with (output / "attributes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "domain",
                "field",
                "n",
                "correct",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "raw_adherence",
                "output_format_valid",
                "ambiguity_rate",
                "isolated_calls",
                "attribute_p50_ms",
                "attribute_p95_ms",
            ],
        )
        writer.writeheader()
        for model in summary["models"]:
            for field in model["fields"]:
                row = {key: field.get(key) for key in writer.fieldnames}
                row.update(
                    model=model["model"],
                    isolated_calls=field["attribute_latency"]["n"],
                    attribute_p50_ms=field["attribute_latency"]["p50_ms"],
                    attribute_p95_ms=field["attribute_latency"]["p95_ms"],
                )
                writer.writerow(row)
    from .charts.quality import generate_quality_charts
    from .charts.summary import generate_summary_charts

    generate_quality_charts(summary, output)
    summary["charts"] = generate_summary_charts(summary, output) + summary["charts"]
    write_json(output / "evaluation.json", summary)
    with (output / "comparisons.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        columns = [
            "a",
            "b",
            "domain",
            "field",
            "n",
            "groups",
            "delta_b_minus_a",
            "ci95",
            "b_only_correct",
            "a_only_correct",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary["comparisons"])
    return summary
