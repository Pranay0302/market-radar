"""pipeline: end-to-end output, restructure alert, audit, tenant isolation."""

import os

import pytest

from marketradar import data_gen
from marketradar.audit import TenantIsolationError, TenantScope
from marketradar.pipeline import run_pipeline


def test_pipeline_produces_valid_recommendations():
    result = run_pipeline("acme-pc")
    assert result.recommendations
    for sr in result.recommendations:
        assert sr.eval_badge == "PASS"
        assert not sr.violations
        assert sr.recommendation.rationale
        assert sr.recommendation.chosen.feasible


def test_pipeline_flags_the_restructure():
    result = run_pipeline("acme-pc")
    assert any(a.kind == "restructure" for a in result.alerts)


def test_pipeline_writes_an_audit_trail():
    result = run_pipeline("acme-pc")
    assert os.path.exists(result.audit_path)


def test_isolation_blocks_cross_tenant_access():
    scope = TenantScope(data_gen.generate())
    with pytest.raises(TenantIsolationError):
        scope.assert_isolated("acme-pc", ["GLOBEX-00"])
    with pytest.raises(TenantIsolationError):
        scope.own_portfolio("intruder-co")