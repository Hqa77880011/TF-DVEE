"""Closed-vocabulary RGB/position encoder and image-derived captions; not a VLM."""

import numpy as np


def measurements(image):
    pixels = np.asarray(image.convert("RGB"), dtype=float) / 255
    blue = (pixels[:, :, 2] > pixels[:, :, 0] + 0.2) & (pixels[:, :, 2] > pixels[:, :, 1] + 0.2)
    middle = len(pixels) // 2
    return float(blue[middle:].mean()), float(blue[:middle].mean()), float(pixels.mean())


class FiniteVisualEncoder:
    """Common four-dimensional numerical image/text space for a fixed vocabulary."""

    def images(self, images):
        values = []
        for image in images:
            bottom, top, brightness = measurements(image)
            values.append([1, 4 * bottom, 4 * top, brightness])
        values = np.asarray(values, np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    def text(self, texts):
        values = []
        for text in texts:
            text = text.casefold()
            if "panel" in text:
                values.append([0.2, 0, 1, 0.1])
            elif "liquid" in text or "leak" in text:
                values.append([0.2, 1, 0.35, 0.1])
            elif "dry" in text or "normal" in text:
                values.append([1, 0, 0, 0.3])
            else:
                raise ValueError("finite_rgb_v1 supports dry housing, blue liquid and blue panel concepts")
        values = np.asarray(values, np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


class FiniteCaptioner:
    """Describe measured visible color/position; the requested state is never an answer."""

    def describe(self, images, asset, state):
        evidence = np.array([measurements(image) for image in images])
        if evidence[:, 0].max() > 0.015:
            return f"{asset} with blue liquid below the housing"
        if evidence[:, 1].max() > 0.015:
            return f"{asset} with a blue panel above the housing"
        return f"{asset} with a dry gray housing"
