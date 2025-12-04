from datasets import load_dataset
from pathlib import Path
import json

data_dir = "/home/yhuangji/.cache/huggingface/hub/datasets--riddhimanrana--coco-fastvlm-2k-val2017/snapshots/9b3023eb83ca1ea614bee18635e2606240c51617/data"
ds = load_dataset("parquet", data_dir=data_dir)
train_ds = ds["train"]

output_dir = Path("./extracted")
images_dir = output_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

captions = {}

for i, example in enumerate(train_ds):
    stem = f"{i:05d}"

    image = example["image"]  # PIL Image
    image.save(images_dir / f"{stem}.jpg")

    gpt_text = ""
    for msg in example["conversations"]:
        if msg["from"] == "gpt":
            gpt_text = msg["value"].strip()
            break

    captions[stem] = gpt_text

    if (i + 1) % 100 == 0:
        print(f"✅ Processed {i + 1} / {len(train_ds)}")

with open(output_dir / "captions.json", "w", encoding="utf-8") as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)

print(f"\n🎉 Done! Saved {len(captions)} images and their captions.")
print(f"   Images path: {images_dir.absolute()}/")
print(f"   Captions file: {output_dir.absolute()}/captions.json")