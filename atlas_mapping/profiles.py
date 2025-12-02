"""Reference profiles for mapping ATLAS tactics to perception components."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .models import ATLASTactic, AttackVector, DefenseMeasure, PerceptionNode

EXP_001 = ATLASTactic(
    identifier="EXP-001",
    name="Adversarial Perturbations",
    description="Small input manipulations that push misclassification of maritime objects.",
)

EXP_002 = ATLASTactic(
    identifier="EXP-002",
    name="Physical Patch Attacks",
    description="Printed overlays on hulls or markers that divert bounding boxes or suppress detections.",
)

MEA_001 = ATLASTactic(
    identifier="MEA-001",
    name="Targeted Model Evasion",
    description="Trajectories or profiles crafted to slip past thresholds for class or confidence.",
)

COL_001 = ATLASTactic(
    identifier="COL-001",
    name="Confidence Manipulation",
    description="Inputs or timing cues that inflate/deflate detection confidence values.",
)


def build_profile(nodes: Iterable[PerceptionNode]) -> Dict[str, List[AttackVector]]:
    """Generate threat sets keyed by perception node."""

    profile: Dict[str, List[AttackVector]] = {}
    for node in nodes:
        if "YOLO" in node.name:
            profile[node.name] = [
                AttackVector(EXP_001, goal="Induce false detections via pixel-level noise", entry_points=["camera frames"]),
                AttackVector(EXP_002, goal="Hide vessels using physical patches", entry_points=["camera frames", "dock signage"]),
                AttackVector(MEA_001, goal="Approach along trajectories outside detector priors", entry_points=["camera frames"]),
                AttackVector(COL_001, goal="Destabilize confidence for avoidance logic", entry_points=["image stream", "timing"]),
            ]
        else:
            profile[node.name] = [
                AttackVector(MEA_001, goal="Misroute fused tracks", entry_points=["fusion bus"]),
                AttackVector(COL_001, goal="Spoof health/latency to impact confidence gating", entry_points=["diagnostics", "network timing"]),
            ]
    return profile


def _defence_list() -> List[DefenseMeasure]:
    return [
        DefenseMeasure(
            title="Adversarial training and augmentation",
            description="Expose YOLO models to perturbations and patch-like artifacts during training.",
            linked_tactics=[EXP_001, EXP_002],
        ),
        DefenseMeasure(
            title="Patch and anomaly detection",
            description="Flag localized saliency spikes or unnatural textures before inference.",
            linked_tactics=[EXP_002],
        ),
        DefenseMeasure(
            title="Cross-sensor consistency",
            description="Fuse radar/vision to constrain MEA-001 evasions and confidence swings.",
            linked_tactics=[MEA_001, COL_001],
        ),
        DefenseMeasure(
            title="Runtime confidence monitoring",
            description="Apply hysteresis, debounce, and rate limits on confidence-driven actions.",
            linked_tactics=[COL_001],
        ),
    ]


def default_defences() -> List[DefenseMeasure]:
    return list(_defence_list())


VISION_NODE = PerceptionNode(
    name="YOLOv5 Maritime Detector",
    responsibilities=["object detection", "collision avoidance", "docking assistance"],
)

FUSION_NODE = PerceptionNode(
    name="Sensor Fusion",
    responsibilities=["cross-sensor correlation", "confidence gating", "track confirmation"],
)

MILLIAMPERE2_PROFILE = {
    "nodes": [VISION_NODE, FUSION_NODE],
    "defences": default_defences(),
    "threats": build_profile([VISION_NODE, FUSION_NODE]),
}
