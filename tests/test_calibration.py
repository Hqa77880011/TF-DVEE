import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tfdvee.calibration import features, fit, load, predict


def event(name, start, strength):
    return dict(id=name, state="leak", start=start, end=start+1, confidence=0.5,
                strength=strength, evidence_consistency=0.5, disagreement=0.1, ambiguity=0.2)


class CalibrationTests(unittest.TestCase):
    def test_monotone_features_and_application(self):
        sample = event("a", 0, 1)
        artifact = dict(coefficients=[1, 1, 1, 1], intercept=0)
        baseline = predict(sample, artifact)
        self.assertGreater(predict(dict(sample, evidence_consistency=0.8), artifact), baseline)
        self.assertLess(predict(dict(sample, disagreement=0.5), artifact), baseline)
        self.assertEqual(features(sample).tolist(), [1, 0.5, -0.1, -0.2])

    def test_device_disjoint_artifact_binding(self):
        artifact = dict(protocol="tfdvee.confidence.v1", context={"model": "fixed"},
                        video_ids=["dev"], device_groups=["pump-a"])
        with patch.object(Path, "read_text", return_value=json.dumps(artifact)):
            self.assertEqual(load("memory", {"model": "fixed"}, "test", "pump-b"), artifact)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                load("memory", {"model": "fixed"}, "test", "pump-a")
            with self.assertRaisesRegex(ValueError, "configuration"):
                load("memory", {"model": "changed"}, "test", "pump-b")

    def test_tiny_analytic_fit_and_one_to_one_labels(self):
        video = dict(video_id="dev", device_group="pump-a", calibration_context={"fixed": True},
                     events=[event("correct", 0, 2), event("incorrect", 4, -2)])
        truth = {"videos": {"dev": dict(device_group="pump-a", events=[dict(state="leak", start=0, end=1)])}}
        sources = {"predictions": json.dumps(video), "truth": json.dumps(truth)}
        written = {}
        with patch.object(Path, "read_text", lambda path, **_: sources[str(path)]), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "write_text", lambda path, value, **_: written.update(artifact=json.loads(value))):
            fit(["predictions"], "truth", "confidence.json")
        artifact = written["artifact"]
        self.assertGreater(predict(video["events"][0], artifact), predict(video["events"][1], artifact))
        self.assertEqual(artifact["development_events"], 2)
        self.assertEqual(artifact["device_groups"], ["pump-a"])
        self.assertTrue(all(value >= 0 for value in artifact["coefficients"]))


if __name__ == "__main__":
    unittest.main()
