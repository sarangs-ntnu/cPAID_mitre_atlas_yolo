from atlas_mapping import MILLIAMPERE2_PROFILE, build_mitigation_plan
from atlas_mapping.profiles import COL_001, EXP_001, EXP_002, MEA_001


def test_plan_includes_nodes_and_threats():
    plans = build_mitigation_plan(
        nodes=MILLIAMPERE2_PROFILE["nodes"],
        threats=MILLIAMPERE2_PROFILE["threats"],
        defences=MILLIAMPERE2_PROFILE["defences"],
    )

    names = [plan.node.name for plan in plans]
    assert "YOLOv5 Maritime Detector" in names
    assert "Sensor Fusion" in names

    vision_plan = next(plan for plan in plans if plan.node.name == "YOLOv5 Maritime Detector")
    tactic_ids = {threat.tactic.identifier for threat in vision_plan.threats}
    assert {"EXP-001", "EXP-002", "MEA-001", "COL-001"} == tactic_ids


def test_matched_defences_cover_tactics():
    plans = build_mitigation_plan(
        nodes=MILLIAMPERE2_PROFILE["nodes"],
        threats=MILLIAMPERE2_PROFILE["threats"],
        defences=MILLIAMPERE2_PROFILE["defences"],
    )
    vision_plan = next(plan for plan in plans if plan.node.name == "YOLOv5 Maritime Detector")

    exp1_defences = {d.title for d in vision_plan.matched_defences(EXP_001)}
    assert "Adversarial training and augmentation" in exp1_defences

    exp2_defences = {d.title for d in vision_plan.matched_defences(EXP_002)}
    assert {"Adversarial training and augmentation", "Patch and anomaly detection"} == exp2_defences

    mea1_defences = {d.title for d in vision_plan.matched_defences(MEA_001)}
    assert "Cross-sensor consistency" in mea1_defences

    col1_defences = {d.title for d in vision_plan.matched_defences(COL_001)}
    assert {"Cross-sensor consistency", "Runtime confidence monitoring"} == col1_defences
