"""Image-level analyzer that binds YOLO detections to ATLAS mitigation plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Protocol, Sequence

from .planner import build_mitigation_plan, plan_summaries
from .profiles import MILLIAMPERE2_PROFILE
from .models import MitigationPlan


@dataclass(frozen=True)
class Detection:
    """Represents a single detection from a vision model."""

    label: str
    confidence: float
    bbox: Sequence[float]

    def as_text(self) -> str:
        coords = ", ".join(f"{value:.1f}" for value in self.bbox)
        return f"{self.label} ({self.confidence:.2f}) at [{coords}]"


class DetectionBackend(Protocol):
    """Protocol for pluggable detection engines."""

    def predict(self, image_path: str) -> Iterable[Detection]:
        """Run detection on an image path and yield detections."""


class UltralyticsYolov5Backend:
    """Thin wrapper around the Ultralytics YOLOv5 hub model for real images.

    This requires optional dependencies: `torch` and `pandas`. The backend loads
    the specified YOLOv5 variant via `torch.hub.load` and converts detections to
    `Detection` objects suitable for the analyzer.
    """

    def __init__(self, model_name: str = "yolov5s", device: str = "cpu") -> None:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - executed only when optional dep missing
            raise ImportError(
                "The Ultralytics backend requires torch. Install optional dependencies via "
                "`pip install atlas-mapping[yolo]` or install torch manually."
            ) from exc

        self.model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
        if device:
            self.model.to(device)

    def predict(self, image_path: str) -> Iterable[Detection]:  # pragma: no cover - exercised in real runs
        results = self.model(image_path)
        frames = results.pandas().xyxy  # type: ignore[attr-defined]
        detections: List[Detection] = []
        for frame in frames:
            for _, row in frame.iterrows():
                detections.append(
                    Detection(
                        label=str(row["name"]),
                        confidence=float(row["confidence"]),
                        bbox=[float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])],
                    )
                )
        return detections


@dataclass(frozen=True)
class AnalysisResult:
    """Bundle detection outputs with the relevant mitigation plans."""

    image_path: Path
    detections: List[Detection]
    plans: List[MitigationPlan]

    def summary(self) -> str:
        detection_lines = "\n".join(d.as_text() for d in self.detections) or "(no detections)"
        return (
            f"Image: {self.image_path}\n"
            f"Detections:\n{detection_lines}\n\n"
            f"Mitigation plans:\n{plan_summaries(self.plans)}"
        )


class MaritimeImageAnalyzer:
    """Runs YOLO detections and pairs them with ATLAS defences."""

    def __init__(self, backend: DetectionBackend, profile=MILLIAMPERE2_PROFILE) -> None:
        self.backend = backend
        self.profile = profile
        self.plans = build_mitigation_plan(
            nodes=self.profile["nodes"],
            threats=self.profile["threats"],
            defences=self.profile["defences"],
        )

    def analyze_image(self, image_path: str | Path) -> AnalysisResult:
        path = Path(image_path)
        detections = list(self.backend.predict(str(path)))
        return AnalysisResult(image_path=path, detections=detections, plans=self.plans)


def main() -> None:  # pragma: no cover - exercised by users via CLI
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLOv5 on maritime imagery and report ATLAS mitigations.")
    parser.add_argument("image", help="Path to an image file to analyze")
    parser.add_argument("--model", default="yolov5s", help="YOLOv5 model variant to load (default: yolov5s)")
    parser.add_argument("--device", default="cpu", help="Device passed to the YOLO model (cpu or cuda)")
    args = parser.parse_args()

    backend = UltralyticsYolov5Backend(model_name=args.model, device=args.device)
    analyzer = MaritimeImageAnalyzer(backend=backend)
    result = analyzer.analyze_image(args.image)
    print(result.summary())


if __name__ == "__main__":  # pragma: no cover
    main()
