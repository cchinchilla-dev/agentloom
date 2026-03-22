"""Tests for cost tracker."""

from __future__ import annotations

import pytest

from agentloom.observability.cost_tracker import CostTracker


class TestCostTracker:
    def test_empty_summary(self) -> None:
        ct = CostTracker()
        s = ct.summary()
        assert s.total_cost_usd == 0.0
        assert s.total_tokens == 0
        assert len(s.entries) == 0

    def test_single_entry(self) -> None:
        ct = CostTracker()
        ct.record(
            "step1",
            "gpt-4o-mini",
            "openai",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.01,
        )
        s = ct.summary()
        assert s.total_cost_usd == 0.01
        assert s.total_prompt_tokens == 100
        assert s.total_completion_tokens == 50
        assert s.total_tokens == 150

    def test_multiple_entries_aggregate(self) -> None:
        ct = CostTracker()
        ct.record("s1", "gpt-4o-mini", "openai", cost_usd=0.01)
        ct.record("s2", "gpt-4o-mini", "openai", cost_usd=0.02)
        ct.record("s3", "claude-haiku-4-5-20251001", "anthropic", cost_usd=0.03)
        s = ct.summary()
        assert s.total_cost_usd == pytest.approx(0.06)
        assert s.cost_by_provider["openai"] == pytest.approx(0.03)
        assert s.cost_by_provider["anthropic"] == pytest.approx(0.03)
        assert s.cost_by_model["gpt-4o-mini"] == pytest.approx(0.03)

    def test_cost_by_step(self) -> None:
        ct = CostTracker()
        ct.record("classify", "gpt-4o-mini", "openai", cost_usd=0.005)
        ct.record("respond", "gpt-4o-mini", "openai", cost_usd=0.010)
        s = ct.summary()
        assert s.cost_by_step["classify"] == pytest.approx(0.005)
        assert s.cost_by_step["respond"] == pytest.approx(0.010)

    def test_reset(self) -> None:
        ct = CostTracker()
        ct.record("s1", "m", "p", cost_usd=1.0)
        ct.reset()
        s = ct.summary()
        assert s.total_cost_usd == 0.0
        assert len(s.entries) == 0

    def test_entries_are_copies(self) -> None:
        ct = CostTracker()
        ct.record("s1", "m", "p", cost_usd=0.5)
        s = ct.summary()
        assert len(s.entries) == 1
        assert s.entries[0].step_id == "s1"
