# TF-DVEE

Implementation of **TF-DVEE: Training-Free Multiscale Temporal Signal Processing for Evidence-Grounded Video Anomaly Detection and Explanation**. The pipeline converts frozen vision-language observations into intervals using normal-reference contrast, causal median filtering, hysteresis, local transition checks, visual evidence verification and explicit review decisions.

The front end uses frozen CLIP embeddings and a frozen BLIP-2 captioner. All temporal operations are explicit parameters. An optional device-disjoint logistic confidence postprocessor fits annotated development events without training either vision-language model.

## Install

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment first (`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1` in PowerShell). Install matching PyTorch/torchvision builds for the target device. For CPU-only installation, install both with `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` before the requirements. Hugging Face downloads the selected pretrained checkpoints on first use; they remain in its model cache and are not part of the repository.

The default captioner is `Salesforce/blip2-flan-t5-xl`. CUDA devices supporting bfloat16 use it; other devices use float32. This is a large frozen model, so choose a device with enough memory. `--caption-model` accepts a compatible BLIP-2 checkpoint. The embedding model defaults to `openai/clip-vit-base-patch32`.

## 1. Prepare videos and the state vocabulary

The paper uses [IPAD](https://doi.org/10.1109/TCSVT.2024.3465517) for temporal evaluation. Obtain videos and their device identities from the provider. [Real-IAD](https://realiad4ad.github.io/Real-IAD/) and [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) can support visible-state vocabulary and image-condition studies; static images do not supply event boundaries or review times.

Use a normal-only commissioning video for the asset/device and a separate event video. Fix the state vocabulary and universal thresholds on device-disjoint development data before evaluating held-out devices. If an asset occupies only part of a frame, pass `--crop left top right bottom` in pixel coordinates, using the same crop definition for its commissioning/event clips.

Create `data/states.json`:

```json
{
  "asset": "pump",
  "normal": {
    "state": "operating normally",
    "cue": "dry housing without visible liquid",
    "concepts": ["pump", "dry", "housing"]
  },
  "deviations": [
    {
      "id": "visible_leak",
      "state": "visible leakage",
      "cue": "liquid dripping from the housing",
      "concepts": ["pump", "liquid", "dripping"],
      "safety": false
    }
  ]
}
```

Concept phrases are literal observable words/phrases to search in captions. Choose them before evaluation. Each deviation needs a unique stable `id`, a state, a visible cue and a concept list. `safety: true` retains a candidate for mandatory review even when its transition statistic is weak.

## 2. Sample and encode

```bash
python -m tfdvee prepare --video data/raw/normal.mp4 --output data/normal --fps 1 --device cuda
python -m tfdvee prepare --video data/raw/process.mp4 --output data/process --fps 1 --device cuda
```

Preparation samples by requested video timestamps, saves the decoded JPEG evidence frames and writes normalized CLIP embeddings to `features.npz`. It records timestamps, duration, model name and relative frame paths. Keep a fixed sampling rate and embedding model across methods; commissioning and event embeddings must use the same model. `--batch-size` controls embedding batches. Sampling precision depends on the source video's timestamp/seek support.

## 3. Run event processing

```bash
python -m tfdvee run --features data/process/features.npz --normal data/normal/features.npz --vocabulary data/states.json --video-id process01 --device-group pump_A --output outputs/tfdvee --device cuda
```

Prompts encode asset, state and visible cue. Framewise state/normal semantic score differences are augmented by deviation from the mean normalized commissioning image prototype. The normal-only signal supplies a median location and MAD scale. Causal windows cover `[t-w,t]`; incomplete initial windows use only the available prefix. Maximum standardized responses across scales enter high/low hysteresis. Same-state events merge across short gaps. A local inside-minus-pre-event statistic rejects persistent ambiguity; events lacking pre-context are retained for review.

For each retained interval, the highest-response, largest-change and pre-event frames form a three-panel evidence image for BLIP-2. The caption is compared with every selected frame through CLIP. Evidence consistency combines minimum frame/caption cosine similarity and controlled-concept F1. Confidence uses temporal strength, consistency, scale disagreement and a decreasing function of the state margin. These are concrete conventions for details left unspecified in the paper.

| Argument | Default | Meaning |
|---|---:|---|
| `--temperature` | 0.07 | Frozen semantic score temperature |
| `--normal-coefficient` | 1 | Normal-prototype contrast weight |
| `--windows` | 1 3 7 | Causal median window durations in seconds |
| `--high`, `--low` | 3, 1.5 | Standardized hysteresis thresholds |
| `--persistence` | 2 | Continuous below-low duration before closure |
| `--merge-gap` | 2 | Maximum same-state event gap to merge |
| `--pre-context` | 10 | Pre-onset context duration in seconds |
| `--transition-threshold` | 0.5 | Minimum local response increase |
| `--evidence-alpha` | 0.5 | Minimum-cosine versus concept-overlap mixing |
| `--confidence-coefficients` | 0.5 2 0.5 1 | Strength, evidence, disagreement and ambiguity coefficients |
| `--confidence-threshold` | 0.7 | Minimum automatic-action confidence |
| `--evidence-threshold` | 0.3 | Minimum evidence consistency |
| `--condition` | nominal | Label identifying the supplied visual disturbance |

Outputs are `events.json`, `signals.npz` and `evidence.html`. Open the HTML locally to inspect captions and the three source frames. Event records retain boundaries, state, confidence, review/safety flags, transition statistic and evidence provenance. Review decisions do not erase detected intervals.

## 4. Baseline and ablations

```bash
python -m tfdvee run --features data/process/features.npz --normal data/normal/features.npz --vocabulary data/states.json --video-id process01 --device-group pump_A --variant frame_retrieval --output outputs/frame_retrieval --device cuda
python -m tfdvee run --features data/process/features.npz --normal data/normal/features.npz --vocabulary data/states.json --video-id process01 --device-group pump_A --variant no_temporal --output outputs/no_temporal --device cuda
```

Variants are `tfdvee`, `frame_retrieval`, `no_temporal`, `no_normal_contrast`, `no_transition`, `no_evidence`, `no_calibration`, `generic_prompts` and `no_referral`.

`frame_retrieval` uses standardized raw state scores with threshold-connected events, no temporal median/transition test and strength-only confidence. `no_temporal` preserves the rest of TF-DVEE with a zero-duration median window. `no_evidence` retains generated captions but removes semantic verification from confidence/referral. `no_calibration` uses a sigmoid of temporal strength. Generic prompts remove the asset and cue from embedding prompts. `no_referral` retains every event without sending low-confidence cases to review.

The implemented comparison is frozen frame retrieval plus mechanism ablations. Published LAVAD, Open-VAD, Vid2Seq and FreeDVC outputs can use the same evaluation schema; those distinct algorithms are not replaced with a renamed local heuristic.

For disturbance-specific comparisons, encode the actual low-light, camera-vibration, occlusion or compressed videos with the same source event identities and sample rate, then run with the corresponding `--condition`. For transition stress, annotate the actual repeated-motion, slow-onset or ambiguous videos. Freeze temporal/frontend settings on development groups before test processing.

## Development confidence fitting

First generate uncalibrated development event files with the same frozen temporal/frontend configuration, and independently annotate those videos. Fit the event-confidence postprocessor:

```bash
python -m tfdvee calibrate --predictions outputs/dev_A/events.json outputs/dev_B/events.json --truth data/development_truth.json --output data/confidence.json
python -m tfdvee run --features data/process/features.npz --normal data/normal/features.npz --vocabulary data/states.json --video-id process01 --device-group heldout_pump --confidence-calibration data/confidence.json --output outputs/calibrated --device cuda
```

The fit uses `[strength, evidence_consistency, -disagreement, -ambiguity]`, a learned intercept and nonnegative coefficients, with L2 regularization 0.001 and equal total weight per development device. Targets are same-state one-to-one temporal matches at IoU 0.50 under the frozen development detector ordering; both correct and incorrect events are required. The artifact binds the complete vocabulary, CLIP/captioner and temporal configuration. Reusing any development video/device at test or changing that configuration is rejected. This is empirical event-confidence fitting, without a distribution-free risk guarantee. Preset coefficients remain available as explicit uncalibrated scores; `no_calibration` remains the strength-only ablation.

## 5. Supply ground truth and evaluate

Create a ground-truth JSON with device identity and annotated intervals:

```json
{
  "videos": {
    "process01": {
      "device_group": "pump_A",
      "events": [
        {"state": "visible_leak", "start": 60.0, "end": 90.0,
         "concepts": ["pump", "liquid", "dripping"]}
      ]
    }
  }
}
```

Use real event annotations; example boundaries above illustrate the schema. Predictions from another method need the same outer `video_id`, `device_group`, `duration`, `variant`, `condition` and `events` structure. Each event requires unique `id`, state, start/end seconds, confidence in `[0,1]`, review flag and extracted concepts. Evaluate one method and one disturbance condition at a time:

```bash
python -m tfdvee evaluate --predictions outputs/tfdvee/events.json --truth data/truth.json --output outputs/tfdvee/metrics.json
```

The evaluator reports class-macro event AP at temporal IoU 0.50/0.75, concept micro-F1, 15-bin event-correctness ECE, AURC, false alarms/hour, automated coverage and automated risk. Matching is confidence-ordered, same-state, same-video and one-to-one. False detections include duplicate event matches. AP uses the all-points interpolated precision envelope. ECE/AURC event correctness uses IoU 0.50. AURC integrates the stepwise prefix error after sorting confidence. Concept F1 counts concepts against matched ground truth and includes unmatched prediction/ground-truth concepts as false positives/negatives. These metric definitions are explicit because the paper does not give implementation-level matching details.

Localization metrics include retained review events; automated coverage/risk report the separate accepted subset. A zero-prediction run reports zero false alarms and null confidence metrics. Videos with no positive ground-truth class have null mAP rather than a fabricated score.

## 6. Human evidence, agreement and review-time metrics

Evidence sufficiency and unsupported concepts use actual blinded ratings. Supply a CSV:

```csv
event_id,evidence_sufficient,unsupported_concepts,asset_r1,asset_r2,state_r1,state_r2,evidence_r1,evidence_r2,safety_r1,safety_r2,adjudicated
```

Each completed row identifies the exact generated event, a binary sufficiency judgment and a nonnegative unsupported-concept count. Optional r1/r2 columns contain the two independent annotator labels before adjudication. `adjudicated` is 1 for an adjudicated row, otherwise 0. The evaluator computes agreement, Cohen's kappa and adjudicated-item counts for the available paired fields. Human judgments are separate from the model's evidence-consistency score.

Actual review-time logs use:

```csv
video_id,reviewer,review_seconds
```

Add the measurements to evaluation:

```bash
python -m tfdvee evaluate --predictions outputs/tfdvee/events.json --truth data/truth.json --ratings data/evidence_ratings.csv --reviews data/review_times.csv --output outputs/tfdvee/metrics.json
```

Review time is normalized to a 12-minute source video, then summarized by its median over supplied reviews. Evidence metrics state how many events were rated. Without human inputs, the corresponding fields remain null; language-model scores are not substituted for these observations.

## 7. Paired statistics and figures

Pass multiple prediction files to `evaluate` to aggregate device-disjoint videos. Device-level results support paired comparisons:

```bash
python -m tfdvee compare --first outputs/tfdvee/metrics.json --second outputs/frame_retrieval/metrics.json --output outputs/paired.json
python -m tfdvee plot --signals outputs/tfdvee/signals.npz --metrics outputs/tfdvee/metrics.json outputs/frame_retrieval/metrics.json --output outputs/figures
```

Paired analysis requires at least two matching device groups. It computes method-minus-baseline mean differences, 10,000 device bootstrap intervals, Wilcoxon tests and Holm-adjusted p-values over endpoints available for both methods. Missing human endpoints do not acquire synthetic values. Figures include temporal score traces, risk–coverage curves and metric comparisons.

GitHub Actions installs compatible CPU dependencies and checks compilation, module imports and frozen-model API imports. It does not download model weights or process video.

## Fixed reproduction setting

The default setting needs no pretrained weights. It renders actual 60-second MJPG commissioning/event videos for two development devices and one held-out test device. Physical blue liquid below a pump appears at 10–22 seconds; unrelated blue signage above the pump at 34–44 seconds is a negative distractor. Device tone and recording noise differ, and normal recordings are independent.

```bash
python reproduce.py --plan
python reproduce.py --output outputs/finite-setting
```

`finite_data.py --output outputs/finite-setting/generated --seed 19` creates videos, closed state vocabulary and separate development/test annotations. The entry point decodes each video at 1 fps with `--clip-model finite_rgb_v1`: the retained CLI/NPZ field name identifies a four-component numerical RGB/position encoder, not CLIP. `--caption-model finite_rgb_rules_v1` describes measured blue location/dry housing from actual evidence frames and ignores the requested state. It is a finite parser, not BLIP/VLM. Native CLIP/BLIP paths remain available separately.

The same normal-reference contrast, 1/3/7-second causal filtering, 3/1.5 hysteresis, transition/evidence checks and matching metrics consume these observations. Uncalibrated development event predictions are matched against physical interval truth; their correct/incorrect outcomes fit the monotone confidence model with standardized development features. The frozen artifact is applied only to the disjoint test device. Neither annotation files nor device IDs are frontend features. Human sufficiency ratings and review-time measurements remain separate optional observational endpoints.
