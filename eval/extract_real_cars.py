"""Extract a diverse, high-quality set of real car photos from Stanford Cars dataset."""

import io
import json
import os
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

TARGET_DIR = Path("eval/images/cars")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

fpath = hf_hub_download(
    repo_id="Multimodal-Fatima/StanfordCars_train",
    filename="data/train-00000-of-00003-0a0d1552d15aa359.parquet",
    repo_type="dataset"
)

table = pq.read_table(fpath)
num_rows = table.num_rows
print(f"Lendo {num_rows} exemplos de carros reais do Stanford Cars...")

TARGET_MAKES = ["toyota", "ford", "honda", "chevrolet", "volkswagen", "bmw", "audi", "dodge", "jeep", "hyundai"]
found_per_make = {m: [] for m in TARGET_MAKES}

# Scan rows to find good representations
batch_size = 500
extracted = []

for start in range(0, min(num_rows, 2000), batch_size):
    sub = table.slice(start, batch_size).to_pylist()
    for row in sub:
        full_name = row["clip_tags_ViT_L_14_simple_specific"].strip()
        parts = full_name.split()
        if not parts:
            continue
        make = parts[0].lower()
        if make in found_per_make and len(found_per_make[make]) < 2:
            caption = row.get("blip_caption_beam_5", "")
            img_bytes = row["image"]["bytes"]
            
            # Determine body style
            body_type = "other"
            if "sedan" in full_name:
                body_type = "sedan"
            elif "suv" in full_name:
                body_type = "suv"
            elif "hatchback" in full_name or "hatch" in full_name:
                body_type = "hatch"
            elif "cab" in full_name or "truck" in full_name or "pickup" in full_name:
                body_type = "truck"
            elif "van" in full_name:
                body_type = "van"
            elif "coupe" in full_name:
                body_type = "coupe"
            elif "convertible" in full_name:
                body_type = "convertible"

            filename = f"real_{make}_{len(found_per_make[make])+1}.jpg"
            save_path = TARGET_DIR / filename
            
            # Open & save image
            im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            im.save(save_path, "JPEG", quality=92)
            
            info = {
                "filename": filename,
                "full_name": full_name.title(),
                "make": make,
                "body_type": body_type,
                "caption": caption,
                "width": im.width,
                "height": im.height
            }
            found_per_make[make].append(info)
            extracted.append(info)
            print(f"Salvo: {filename} -> {full_name} ({im.size})")

print(f"\nTotal de fotos reais extraídas: {len(extracted)}")
with open("eval/datasets/real_cars_metadata.json", "w", encoding="utf-8") as f:
    json.dump(extracted, f, indent=2, ensure_ascii=False)
