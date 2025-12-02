# MITRE ATLAS Mapping for milliAmpere2 YOLO Perception

## Purpose
To formalize the adversarial threat landscape, we adopt the MITRE ATLAS framework as a structured taxonomy of AI-focused tactics and attack vectors. This mapping keeps threat modeling consistent across the perception stack while aligning with a recognized AI-security standard for maritime autonomy.

## ATLAS tactic alignment for YOLO-based detection
- **EXP-001: Adversarial Perturbations** – Small input manipulations that cause misclassification of maritime objects (e.g., docks, buoys, vessels), stressing robustness of onboard YOLO models.
- **EXP-002: Physical Patch Attacks** – Printed or physical overlays applied to markers or hull surfaces that divert bounding boxes or suppress detections during docking and obstacle avoidance.
- **MEA-001: Targeted Model Evasion** – Intentional trajectories or visual profiles crafted to slip past class- or confidence-thresholds, enabling covert approach or denial of obstacle detection.
- **COL-001: Confidence Manipulation** – Feeding inputs or timing cues that inflate or deflate detection confidence, potentially triggering false positives/negatives in collision-avoidance logic.

## Defence strategies aligned to ATLAS mitigations
- **Adversarial training and data augmentation** to harden YOLO weights against EXP-001 and EXP-002 patterns.
- **Patch and anomaly detection** using spatial consistency checks or secondary classifiers to flag physical overlays and unexpected feature saliency.
- **Cross-sensor consistency** (e.g., radar/vision fusion) to reduce single-sensor spoofing and curb MEA-001 evasion attempts.
- **Runtime confidence monitoring** with thresholds, hysteresis, and rate-limiting to resist COL-001 manipulation and stabilize decision logic.

## Integration guidance
Applying these defences within milliAmpere2’s perception stack clarifies attacker capabilities, constrains test scenarios, and keeps mitigations traceable to ATLAS techniques for auditability and continuous improvement.

## Code: ATLAS mapping utilities
A small Python module models the ATLAS threats, defences, and perception nodes so teams can generate mitigation plans directly from code.

### Installation
The package is self contained and uses the Python standard library. From the repository root you can install it in editable mode for local experimentation:

```bash
pip install -e .
```

If you want to run the YOLOv5-powered analyzer, install the optional dependencies:

```bash
pip install -e .[yolo]
```

### Usage (library)
The `atlas_mapping` package exposes the milliAmpere2 profile and a planner to bind tactics and defences to perception nodes.

```python
from atlas_mapping import MILLIAMPERE2_PROFILE, build_mitigation_plan
from atlas_mapping.planner import plan_summaries

plans = build_mitigation_plan(
    nodes=MILLIAMPERE2_PROFILE["nodes"],
    threats=MILLIAMPERE2_PROFILE["threats"],
    defences=MILLIAMPERE2_PROFILE["defences"],
)

print(plan_summaries(plans))
```

The resulting output lists each perception node, its responsibilities, mapped threats, and the defences that address those tactics. You can extend the profile with additional nodes or defences by importing the dataclasses from `atlas_mapping.models` and appending to the profile structures before calling `build_mitigation_plan`.

### Run against real maritime images (CLI)
1. Install the YOLO extras (once):
   ```bash
   pip install -e .[yolo]
   ```

2. Run the analyzer on an image:
   ```bash
   python -m atlas_mapping.image_analyzer path/to/maritime_image.jpg
   ```

3. (Optional) Choose a different YOLOv5 model or device (CPU/GPU):
   ```bash
   python -m atlas_mapping.image_analyzer path/to/maritime_image.jpg --model yolov5m --device cuda
   ```

The CLI loads a YOLOv5 model via `torch.hub`, performs detections on the supplied image, and prints the detections alongside the ATLAS mitigation plans for the milliAmpere2 perception stack. It works with any maritime image file path you provide.

### Try it interactively in Jupyter
Open the ready-made notebook to explore the mapping and analyzer outputs cell by cell:

```bash
jupyter notebook notebooks/maritime_atlas_demo.ipynb
```

The notebook includes:
- A mitigation-plan summary built from the milliAmpere2 profile.
- A dry-run analyzer backend that emits deterministic detections so you can see text output without YOLO installed.
- An optional YOLOv5 cell you can enable once `pip install -e .[yolo]` is complete and you have a real maritime image path.

### Tests
Run the lightweight test suite with `pytest`:

```bash
python -m pytest
```
