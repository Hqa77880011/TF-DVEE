# TF-DVEE

Implementation of **TF-DVEE: Training-Free Multiscale Temporal Signal Processing for Evidence-Grounded Video Anomaly Detection and Explanation**. The pipeline converts frozen vision-language observations into intervals using normal-reference contrast, causal median filtering, hysteresis, local transition checks, visual evidence verification and explicit review decisions.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Generate rendered pump videos, vocabulary and independent development/test annotations with seed 19. This is rendered RGB data; `finite_rgb_v1` is a numerical RGB/position encoder and `finite_rgb_rules_v1` is a rule captioner. For acquired video, obtain [IPAD](https://doi.org/10.1109/TCSVT.2024.3465517) from its provider, retain separate normal commissioning/event videos and edit the vocabulary. Native CLIP/BLIP-2 frontends download their Hugging Face weights on first use.

```bash
python finite_data.py --output data/finite-videos --seed 19
```

## Training and inference

```bash
python -m tfdvee prepare --video data/finite-videos/normal-test-C.avi --output data/encoded-normal --clip-model finite_rgb_v1 --fps 1
python -m tfdvee prepare --video data/finite-videos/event-test-C.avi --output data/encoded-event --clip-model finite_rgb_v1 --fps 1
python -m tfdvee run --features data/encoded-event/features.npz --normal data/encoded-normal/features.npz --vocabulary data/finite-videos/states.json --caption-model finite_rgb_rules_v1 --video-id test-C --device-group test-C --output outputs/tfdvee
```

The complete generated-device route fits confidence on two development devices and applies it to a held-out device: inspect `python reproduce.py --plan`, then run `python reproduce.py --output outputs/finite-setting` in a fresh output directory. Seed 19 and frontend/temporal settings are fixed there. Optional human observations use `evaluate --ratings ... --reviews ...`.

## Evaluation

```bash
python -m tfdvee evaluate --predictions outputs/tfdvee/events.json --truth data/finite-videos/test_truth.json --output outputs/tfdvee/metrics.json
```

## Ablations

Supported controls: `frame_retrieval`, `no_temporal`, `no_normal_contrast`, `no_transition`, `no_evidence`, `no_calibration`, `generic_prompts`, `no_referral`. Keep the input groups, seed and budget fixed.

```bash
python -m tfdvee run --features data/encoded-event/features.npz --normal data/encoded-normal/features.npz --vocabulary data/finite-videos/states.json --caption-model finite_rgb_rules_v1 --video-id test-C --device-group test-C --output outputs/no_temporal --variant no_temporal
```

## Outputs

Prepared folders contain sampled frames and `features.npz`. Runs write `events.json`, `signals.npz` and `evidence.html`; evaluation writes the requested metrics JSON. The complete confidence workflow also writes `confidence.json`.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
