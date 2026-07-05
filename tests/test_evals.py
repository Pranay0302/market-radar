"""evals: the golden set passes and the quality metrics stay strong."""

from marketradar.evals import run_evals


def test_all_correctness_checks_pass():
    report = run_evals()
    failed = [c.name for c in report.checks if not c.passed]
    assert not failed, f"failed checks: {failed}"
    assert report.passed


def test_quality_metrics_hold():
    m = run_evals().metrics
    assert m["mapping_accuracy"] == 1.0
    assert m["traction_recall"] == 1.0
    assert m["aspect_f1"] >= 0.5