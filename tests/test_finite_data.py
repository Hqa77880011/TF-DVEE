import unittest
import numpy as np
from PIL import Image
from finite_data import render
from tfdvee.finite_visual import FiniteCaptioner, FiniteVisualEncoder, measurements
from tfdvee.temporal import contrast, multiscale, segment
from tfdvee.metrics import match


class FiniteInputTests(unittest.TestCase):
    def test_caption_uses_pixels_and_distractor_is_not_truth(self):
        normal = Image.fromarray(render(0, np.random.default_rng(7)))
        fluid = Image.fromarray(render(12, np.random.default_rng(7)))
        panel = Image.fromarray(render(36, np.random.default_rng(7)))
        captioner, encoder = FiniteCaptioner(), FiniteVisualEncoder()
        self.assertIn("dry gray", captioner.describe([normal], "pump", "visible leakage"))
        self.assertIn("blue liquid", captioner.describe([fluid], "pump", "operating normally"))
        self.assertIn("blue panel", captioner.describe([panel], "pump", "visible leakage"))
        self.assertEqual(measurements(panel)[0], 0)
        prompts = encoder.text(["dry gray housing", "blue liquid below housing"])
        features = encoder.images([normal, fluid, panel])
        signal = contrast(features, prompts, features[:1])
        self.assertGreater(signal[1, 0], signal[0, 0])
        self.assertGreater(signal[2, 0], signal[0, 0])
        # Cross-sensitive signage creates a genuine negative event for correctness calibration.
        self.assertNotEqual(captioner.describe([fluid], "pump", ""), captioner.describe([panel], "pump", ""))

    def test_finite_temporal_candidates_supply_both_calibration_classes(self):
        encoder = FiniteVisualEncoder()
        images = [Image.fromarray(render(t, np.random.default_rng(7))) for t in (0, 12, 36)]
        features = encoder.images(images)
        observed = features[np.array([1 if 10 <= t < 22 else 2 if 34 <= t < 44 else 0 for t in range(60)])]
        prompts = encoder.text(["dry gray housing", "blue liquid below housing"])
        baseline = contrast(features[:1], prompts, features[:1])
        filtered, _ = multiscale(contrast(observed, prompts, features[:1]), baseline, [1, 3, 7], 1)
        candidates = segment(filtered, 3, 1.5, 2, 2, 1)
        predictions = [dict(id=str(i), video_id="dev", state="leak", start=c["start"], end=c["end"], confidence=.5)
                       for i, c in enumerate(candidates)]
        pairs, _ = match(predictions, [dict(video_id="dev", state="leak", start=10, end=22)], .5)
        self.assertEqual({target is not None for _, target in pairs}, {True, False})
