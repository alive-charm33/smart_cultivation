"""One-time check of your saved leaf model (NO training).

    python check_model.py path/to/tomato-leaf.jpg

Loads models/best_tomato_model.h5, prints the input shape and the top-4
predictions with BOTH preprocessing options (raw 0-255 and /255).
The option that gives a sensible, confident answer is the one you trained with;
the backend picks it automatically ("auto"), or set DISEASE_PREPROCESS=raw /
DISEASE_PREPROCESS=divide255 in backend/.env to force it.
"""
import sys
from pathlib import Path

import numpy as np

from app import config
from app.services import disease

if len(sys.argv) < 2:
    sys.exit(__doc__)
img_path = Path(sys.argv[1])
if not img_path.exists():
    sys.exit(f"Image not found: {img_path}")

model = disease.get_model()
names = disease.class_names()
print("Model file      :", config.DISEASE_MODEL_PATH)
print("Input shape     :", model.input_shape)
print("Auto preprocess :", disease.status()["preprocess"])
print("Image file      :", img_path.name)
_, arr = disease.load_image(img_path, disease._input_size(model))
x = np.expand_dims(arr, 0)
for tag, xx in [("raw 0-255", x), ("divided by 255", x / 255.0)]:
    p = model.predict(xx, verbose=0)[0]
    print(f"\n{tag}")
    for i in np.argsort(p)[::-1][:4]:
        print(f"  {names[i]:50s} {p[i]:.2f}")
