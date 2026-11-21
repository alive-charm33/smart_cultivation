"""Command-line test of the full pipeline on one leaf photo (NO training):
classification with best_tomato_model.h5 -> Grad-CAM image -> RAG advice.

    python predict_image.py path/to/tomato-leaf.jpg [--stage flowering_fruit_set] [--days 40] [--no-llm]

The Grad-CAM overlay is written next to the input as <same file name>_gradcam.png,
so the output always carries the exact name of the image you gave.
"""
import argparse
from pathlib import Path

from app.services import disease, knowledge

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--stage", choices=knowledge.STAGES)
ap.add_argument("--days", type=int, help="days after transplanting")
ap.add_argument("--question")
ap.add_argument("--no-llm", action="store_true")
a = ap.parse_args()

img_path = Path(a.image)
top, _, idx, img, x = disease.predict(img_path)
print(f"Image: {img_path.name}")
for p in top:
    print(f"  {p['label']:50s} {p['confidence']:.1%}")
cam = disease.gradcam(x, idx)
over, _ = disease.overlay(img, cam)
out = img_path.with_name(f"{img_path.stem}_gradcam.png")
over.save(out)
print("Grad-CAM saved:", out, disease.describe_cam(cam))
res = knowledge.advise_disease(top, a.question, a.stage, a.days, use_llm=not a.no_llm)
print(f"\nStages: {res['stages']} | {res['days_info']} | mode: {res['mode']}")
if res.get("warning"):
    print("Note:", res["warning"])
print("\n" + res["answer"])
