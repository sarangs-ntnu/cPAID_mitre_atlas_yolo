from atlas_mapping.image_analyzer import AnalysisResult, Detection, MaritimeImageAnalyzer


class _StubBackend:
    def predict(self, image_path: str):
        return [Detection(label="boat", confidence=0.91, bbox=[0, 1, 2, 3])]


def test_analyzer_wraps_backend_and_plan():
    analyzer = MaritimeImageAnalyzer(backend=_StubBackend())
    result: AnalysisResult = analyzer.analyze_image("sample.jpg")

    assert result.image_path.name == "sample.jpg"
    assert len(result.detections) == 1
    assert result.detections[0].label == "boat"

    plan_node_names = {plan.node.name for plan in result.plans}
    assert "YOLOv5 Maritime Detector" in plan_node_names

    summary_text = result.summary()
    assert "boat (0.91)" in summary_text
    assert "YOLOv5 Maritime Detector" in summary_text
