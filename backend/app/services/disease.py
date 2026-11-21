"""Leaf disease classification with the ALREADY-TRAINED model best_tomato_model.h5,
plus a Grad-CAM heat-map that shows which leaf region drove the prediction.

Compatible with both Keras 2 (TF 2.x bundled Keras) and Keras 3 (standalone).
The model is loaded ONCE (first request or at startup) and kept in memory.
"""
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config

_lock = threading.Lock()
_state = {"model": None, "error": None, "preprocess": None}


def class_names():
    names = json.loads((config.RAG_DIR / "class_names.json").read_text())
    return names


def label_map():
    return json.loads((config.RAG_DIR / "label_map.json").read_text())


# ------------------------------------------------------------------ loading
def get_model():
    """Load best_tomato_model.h5 once. Thread-safe."""
    if _state["model"] is not None:
        return _state["model"]
    with _lock:
        if _state["model"] is not None:
            return _state["model"]
        path = Path(config.DISEASE_MODEL_PATH)
        if not path.exists():
            _state["error"] = (f"Model file not found: {path}. Copy your trained "
                               f"'{config.DISEASE_MODEL_FILENAME}' into backend/models/ "
                               f"(keep the same file name).")
            raise RuntimeError(_state["error"])

        import tensorflow as tf

        # Keras 3 stores weights differently — load_model works for both Keras 2 and 3
        try:
            model = tf.keras.models.load_model(str(path), compile=False)
        except Exception as e:
            # Fallback: try with safe_mode disabled (Keras 3)
            try:
                model = tf.keras.models.load_model(str(path), compile=False, safe_mode=False)
            except TypeError:
                raise RuntimeError(f"Could not load model: {e}")

        n_out = model.output_shape[-1]
        if n_out != len(class_names()):
            raise RuntimeError(f"Model has {n_out} outputs but class_names.json has "
                               f"{len(class_names())} labels.")
        _state["model"] = model
        _state["preprocess"] = _decide_preprocess(model)
        _state["error"] = None
        return model


def status():
    path = Path(config.DISEASE_MODEL_PATH)
    return {"file": str(path), "exists": path.exists(), "loaded": _state["model"] is not None,
            "preprocess": _state["preprocess"] or config.DISEASE_PREPROCESS, "error": _state["error"]}


def _all_layers(model):
    for layer in model.layers:
        yield layer
        if hasattr(layer, "layers"):
            yield from _all_layers(layer)


def _decide_preprocess(model):
    if config.DISEASE_PREPROCESS in ("raw", "divide255"):
        return config.DISEASE_PREPROCESS
    for layer in _all_layers(model):
        if type(layer).__name__ in ("Rescaling", "Normalization"):
            return "raw"  # model scales pixels itself (EfficientNetB0 in Keras)
    return "divide255"


def _input_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    h, w = shape[1], shape[2]
    return (int(h or 224), int(w or 224))


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    resized = img.resize((size[1], size[0]), Image.BILINEAR)
    return img, np.asarray(resized, dtype="float32")


def _prep(x):
    return x / 255.0 if _state["preprocess"] == "divide255" else x


# ------------------------------------------------------------------ prediction
def predict(image_path, top_k=3):
    model = get_model()
    size = _input_size(model)
    original, arr = load_image(image_path, size)
    x = _prep(np.expand_dims(arr, 0))
    probs = model.predict(x, verbose=0)[0]
    names, lm = class_names(), label_map()
    order = np.argsort(probs)[::-1]
    top = [{"disease_id": lm[names[i]], "label": names[i], "confidence": float(probs[i])}
           for i in order[:top_k]]
    all_probs = {lm[names[i]]: float(probs[i]) for i in order}
    return top, all_probs, int(order[0]), original, x


# ------------------------------------------------------------------ Grad-CAM
def _find_last_conv_layer_name(model):
    """Return the name of the last Conv2D-like layer in the model (including sub-models)."""
    last_name = None
    last_model = model  # which model object owns the layer

    def _walk(m):
        nonlocal last_name, last_model
        for layer in m.layers:
            if hasattr(layer, "layers"):  # nested sub-model
                _walk(layer)
            else:
                try:
                    if len(layer.output.shape) == 4:
                        last_name = layer.name
                        last_model = m
                except Exception:
                    pass

    _walk(model)
    if last_name is None:
        raise ValueError("No 4-D convolutional layer found for Grad-CAM")
    return last_model, last_name


def gradcam(x, class_index):
    """Grad-CAM compatible with Keras 2 and Keras 3.

    Keras 3 doesn't allow building a Model whose outputs come from layers
    inside a nested sub-model. We work around this by:
      1. Finding the innermost model that contains the last conv layer.
      2. Building the grad-model from that inner model's inputs.
      3. Re-routing gradients manually.
    """
    import tensorflow as tf

    model = get_model()
    inner_model, conv_name = _find_last_conv_layer_name(model)
    conv_layer = inner_model.get_layer(conv_name)

    # If the conv layer belongs to the top-level model, build the grad-model directly.
    if inner_model is model:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )
        x_tensor = tf.cast(tf.convert_to_tensor(x), tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x_tensor, training=False)
            tape.watch(conv_out)
            loss = preds[:, class_index]
        grads = tape.gradient(loss, conv_out)

    else:
        # inner_model is a nested sub-model (e.g. EfficientNetB0 inside the saved model).
        # Build a grad-model from the inner model, then feed it manually.
        inner_grad_model = tf.keras.Model(
            inputs=inner_model.inputs,
            outputs=[conv_layer.output, inner_model.output]
        )
        x_tensor = tf.cast(tf.convert_to_tensor(x), tf.float32)

        # Find layers before the inner sub-model and pass x through them first
        inner_idx = next(i for i, l in enumerate(model.layers) if l is inner_model)
        pre_layers = [l for l in model.layers[:inner_idx]
                      if type(l).__name__ != "InputLayer"]
        h = x_tensor
        for layer in pre_layers:
            h = layer(h, training=False)

        with tf.GradientTape() as tape:
            conv_out, inner_out = inner_grad_model(h, training=False)
            tape.watch(conv_out)
            # Continue through layers after the sub-model to reach final output
            post_h = inner_out
            for layer in model.layers[inner_idx + 1:]:
                post_h = layer(post_h, training=False)
            loss = post_h[:, class_index]
        grads = tape.gradient(loss, conv_out)

    # Global average pool the gradients, then weighted sum of conv feature maps
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.nn.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay(original: Image.Image, cam, alpha=0.45):
    """Colour the heat-map (jet palette) and blend over the original leaf photo."""
    import matplotlib
    cmap = matplotlib.colormaps["jet"]
    w, h = original.size
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
    heat = cmap(np.asarray(cam_img) / 255.0)[..., :3]
    heat = Image.fromarray(np.uint8(heat * 255))
    return Image.blend(original.convert("RGB"), heat, alpha), heat


def describe_cam(cam, threshold=0.5):
    """Plain-language summary: how much of the leaf is highlighted and where."""
    mask = cam >= threshold
    pct = float(mask.mean() * 100)
    if not mask.any():
        return {"highlighted_pct": 0.0, "region": "none"}
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean() / cam.shape[0], xs.mean() / cam.shape[1]
    vert = "upper" if cy < 0.4 else "lower" if cy > 0.6 else "middle"
    horiz = "left" if cx < 0.4 else "right" if cx > 0.6 else "centre"
    return {"highlighted_pct": round(pct, 1),
            "region": f"{vert} {horiz}".replace("middle centre", "centre")}
