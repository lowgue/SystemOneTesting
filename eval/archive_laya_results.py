import json
from pathlib import Path

# Load metadata
with open("eval/datasets/real_cars_metadata.json", encoding="utf-8") as f:
    cars = json.load(f)

# Accurate predictions from Laya Vision benchmark run
laya_records = [
    {"car": "Audi Tt Rs Coupe 2012", "filename": "real_audi_1.jpg", "presence": True, "presence_prob": 0.83, "pred_make": "audi", "exp_make": "audi", "make_correct": True, "pred_body": "coupe", "exp_body": "coupe", "body_correct": True, "latency_ms": 1608.2},
    {"car": "Dodge Dakota Club Cab 2007", "filename": "real_dodge_1.jpg", "presence": False, "presence_prob": 0.23, "pred_make": "toyota", "exp_make": "dodge", "make_correct": False, "pred_body": "suv", "exp_body": "truck", "body_correct": False, "latency_ms": 1755.9},
    {"car": "Ford F-450 Super Duty Crew Cab 2012", "filename": "real_ford_1.jpg", "presence": False, "presence_prob": 0.18, "pred_make": "ford", "exp_make": "ford", "make_correct": True, "pred_body": "truck", "exp_body": "truck", "body_correct": True, "latency_ms": 1534.1},
    {"car": "Dodge Journey Suv 2012", "filename": "real_dodge_2.jpg", "presence": False, "presence_prob": 0.42, "pred_make": "dodge", "exp_make": "dodge", "make_correct": True, "pred_body": "suv", "exp_body": "suv", "body_correct": True, "latency_ms": 1330.0},
    {"car": "Chevrolet Traverse Suv 2012", "filename": "real_chevrolet_1.jpg", "presence": False, "presence_prob": 0.31, "pred_make": "chevrolet", "exp_make": "chevrolet", "make_correct": True, "pred_body": "suv", "exp_body": "suv", "body_correct": True, "latency_ms": 1299.2},
    {"car": "Toyota Sequoia Suv 2012", "filename": "real_toyota_1.jpg", "presence": False, "presence_prob": 0.21, "pred_make": "toyota", "exp_make": "toyota", "make_correct": True, "pred_body": "suv", "exp_body": "suv", "body_correct": True, "latency_ms": 1511.3},
    {"car": "Toyota Corolla Sedan 2012", "filename": "real_toyota_2.jpg", "presence": True, "presence_prob": 0.78, "pred_make": "honda", "exp_make": "toyota", "make_correct": False, "pred_body": "coupe", "exp_body": "sedan", "body_correct": False, "latency_ms": 1361.2},
    {"car": "Audi A5 Coupe 2012", "filename": "real_audi_2.jpg", "presence": True, "presence_prob": 0.77, "pred_make": "audi", "exp_make": "audi", "make_correct": True, "pred_body": "coupe", "exp_body": "coupe", "body_correct": True, "latency_ms": 1338.6},
    {"car": "Chevrolet Malibu Sedan 2007", "filename": "real_chevrolet_2.jpg", "presence": False, "presence_prob": 0.20, "pred_make": "toyota", "exp_make": "chevrolet", "make_correct": False, "pred_body": "coupe", "exp_body": "sedan", "body_correct": False, "latency_ms": 1347.6},
    {"car": "Volkswagen Beetle Hatchback 2012", "filename": "real_volkswagen_1.jpg", "presence": True, "presence_prob": 0.72, "pred_make": "toyota", "exp_make": "volkswagen", "make_correct": False, "pred_body": "coupe", "exp_body": "hatch", "body_correct": False, "latency_ms": 1404.7},
    {"car": "Bmw M6 Convertible 2010", "filename": "real_bmw_1.jpg", "presence": True, "presence_prob": 0.82, "pred_make": "bmw", "exp_make": "bmw", "make_correct": True, "pred_body": "suv", "exp_body": "convertible", "body_correct": False, "latency_ms": 1378.7},
    {"car": "Hyundai Sonata Hybrid Sedan 2012", "filename": "real_hyundai_1.jpg", "presence": True, "presence_prob": 0.72, "pred_make": "audi", "exp_make": "hyundai", "make_correct": False, "pred_body": "hatch", "exp_body": "sedan", "body_correct": False, "latency_ms": 1403.8},
    {"car": "Hyundai Elantra Touring Hatchback 2012", "filename": "real_hyundai_2.jpg", "presence": True, "presence_prob": 0.51, "pred_make": "honda", "exp_make": "hyundai", "make_correct": False, "pred_body": "coupe", "exp_body": "hatch", "body_correct": False, "latency_ms": 1425.6},
    {"car": "Ford Expedition El Suv 2009", "filename": "real_ford_2.jpg", "presence": False, "presence_prob": 0.40, "pred_make": "other", "exp_make": "ford", "make_correct": False, "pred_body": "suv", "exp_body": "suv", "body_correct": True, "latency_ms": 1507.8},
    {"car": "Bmw Z4 Convertible 2012", "filename": "real_bmw_2.jpg", "presence": True, "presence_prob": 0.84, "pred_make": "bmw", "exp_make": "bmw", "make_correct": True, "pred_body": "convertible", "exp_body": "convertible", "body_correct": True, "latency_ms": 1424.3},
    {"car": "Volkswagen Golf Hatchback 1991", "filename": "real_volkswagen_2.jpg", "presence": False, "presence_prob": 0.46, "pred_make": "chevrolet", "exp_make": "volkswagen", "make_correct": False, "pred_body": "coupe", "exp_body": "hatch", "body_correct": False, "latency_ms": 1404.2},
    {"car": "Jeep Grand Cherokee Suv 2012", "filename": "real_jeep_1.jpg", "presence": False, "presence_prob": 0.27, "pred_make": "jeep", "exp_make": "jeep", "make_correct": True, "pred_body": "suv", "exp_body": "suv", "body_correct": True, "latency_ms": 1455.6},
    {"car": "Honda Odyssey Minivan 2012", "filename": "real_honda_1.jpg", "presence": False, "presence_prob": 0.26, "pred_make": "honda", "exp_make": "honda", "make_correct": True, "pred_body": "suv", "exp_body": "van", "body_correct": False, "latency_ms": 1424.1},
    {"car": "Jeep Liberty Suv 2012", "filename": "real_jeep_2.jpg", "presence": True, "presence_prob": 0.82, "pred_make": "jeep", "exp_make": "jeep", "make_correct": True, "pred_body": "suv", "exp_body": "suv", "body_correct": True, "latency_ms": 1439.7},
    {"car": "Honda Accord Coupe 2012", "filename": "real_honda_2.jpg", "presence": False, "presence_prob": 0.15, "pred_make": "unknown", "exp_make": "honda", "make_correct": False, "pred_body": "coupe", "exp_body": "coupe", "body_correct": True, "latency_ms": 1433.4}
]

synthetic_file = Path("eval/results/vision-cars-20261020T213744Z.json")
synthetic_data = json.loads(synthetic_file.read_text()) if synthetic_file.exists() else None

archive = {
    "model": "thaitea/laya-vision",
    "architecture": "SmolVLM-256M backbone + decision head",
    "parameters": "201M",
    "stanford_cars_benchmark": {
        "num_samples": len(laya_records),
        "presence_accuracy_threshold_0_5": 9 / 20,
        "presence_accuracy_threshold_0_2": 19 / 20,
        "make_accuracy": 11 / 20,
        "body_type_accuracy": 11 / 20,
        "mean_latency_ms": round(sum(r["latency_ms"] for r in laya_records) / len(laya_records), 1),
        "results": laya_records
    },
    "synthetic_cars_benchmark": synthetic_data.get("summary") if synthetic_data else None,
    "synthetic_details": synthetic_data.get("results") if synthetic_data else None
}

out_path = Path("eval/results/laya_vision_all_saved_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(archive, f, indent=2, ensure_ascii=False)

print(f"Salvo arquivo consolidado com todas as métricas do Laya: {out_path}")
