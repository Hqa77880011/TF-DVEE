from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch


def visual_model(name, device):
    if name == "finite_rgb_v1":
        from .finite_visual import FiniteVisualEncoder
        return FiniteVisualEncoder()
    return FrozenClip(name, device)


def caption_model(name, device):
    if name == "finite_rgb_rules_v1":
        from .finite_visual import FiniteCaptioner
        return FiniteCaptioner()
    return FrozenCaptioner(name, device)


class FrozenClip:
    def __init__(self, name, device):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.model.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(name)
        self.device = device

    @torch.no_grad()
    def images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(features, dim=-1).cpu().numpy()

    @torch.no_grad()
    def text(self, texts):
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        features = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features, dim=-1).cpu().numpy()


class FrozenCaptioner:
    def __init__(self, name, device):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        self.dtype = torch.bfloat16 if str(device).startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(name, torch_dtype=self.dtype).to(device).eval()
        self.model.requires_grad_(False)
        self.processor = Blip2Processor.from_pretrained(name)
        self.device = device

    @torch.no_grad()
    def describe(self, images, asset, state):
        montage = Image.new("RGB", (384 * len(images), 384), "white")
        for index, image in enumerate(images):
            current = image.copy()
            current.thumbnail((384, 384))
            montage.paste(current, (384 * index, 0))
        question = f"Question: What visible evidence in these frames supports or contradicts the state {state} of {asset}? Answer:"
        inputs = self.processor(images=montage, text=question, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        generated = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def prepare_video(path, output, clip_name, device, fps=1, crop=None, batch_size=32):
    if fps <= 0 or batch_size < 1:
        raise ValueError("Positive sampling rate and batch size required")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    frames = destination / "frames"
    frames.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    native_fps = capture.get(cv2.CAP_PROP_FPS)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / native_fps if native_fps > 0 else 0
    if not capture.isOpened() or duration <= 0:
        capture.release()
        raise ValueError("Unable to read video duration or frame rate")
    model = visual_model(clip_name, device)
    features, times, paths, pending = [], [], [], []
    for index, timestamp in enumerate(np.arange(0, duration, 1 / fps)):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
        success, frame = capture.read()
        if not success:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if crop:
            left, top, right, bottom = crop
            if not 0 <= left < right <= image.width or not 0 <= top < bottom <= image.height:
                raise ValueError("Crop lies outside video frame")
            image = image.crop(tuple(crop))
        filename = frames / f"{index:08d}.jpg"
        image.save(filename, quality=95)
        pending.append(image)
        times.append(float(timestamp))
        paths.append(f"frames/{filename.name}")
        if len(pending) >= batch_size:
            features.append(model.images(pending))
            pending = []
    capture.release()
    if pending:
        features.append(model.images(pending))
    if not features:
        raise ValueError("No decoded video frames")
    np.savez_compressed(destination / "features.npz", image_features=np.concatenate(features),
                        timestamps=np.array(times), paths=np.array(paths),
                        fps=fps, duration=duration, clip_model=clip_name)


def load_features(path):
    with np.load(path, allow_pickle=False) as archive:
        features = archive["image_features"]
        timestamps = archive["timestamps"]
        paths = archive["paths"]
        fps, duration = float(archive["fps"]), float(archive["duration"])
        clip_name = str(archive["clip_model"])
    if features.ndim != 2 or len(features) != len(timestamps) or len(paths) != len(features):
        raise ValueError("Feature arrays and frame metadata disagree")
    if not len(features) or not np.isfinite(features).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Expected finite features and increasing frame timestamps")
    features = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-8)
    return dict(features=features, timestamps=timestamps, paths=paths, fps=fps,
                duration=duration, clip_model=clip_name, root=Path(path).resolve().parent)
