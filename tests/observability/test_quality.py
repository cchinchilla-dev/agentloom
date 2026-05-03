"""Tests for quality annotations (#59)."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentloom.core.results import (
    QualityAnnotation,
    WorkflowResult,
    WorkflowStatus,
)
from agentloom.observability.quality import (
    emit_quality_annotation,
    emit_quality_annotations,
)
from agentloom.observability.schema import SpanAttr


def _result() -> WorkflowResult:
    return WorkflowResult(
        workflow_name="wf",
        status=WorkflowStatus.SUCCESS,
        run_id="run-xyz",
    )


class TestWorkflowResultAnnotate:
    def test_annotate_appends_and_returns(self) -> None:
        result = _result()
        annotation = result.annotate("answer", quality_score=4.5, source="human")

        assert isinstance(annotation, QualityAnnotation)
        assert result.annotations == [annotation]
        assert annotation.target == "answer"
        assert annotation.quality_score == 4.5
        assert annotation.source == "human"
        assert annotation.metadata == {}

    def test_annotate_captures_extra_metadata(self) -> None:
        result = _result()
        result.annotate(
            "summary",
            quality_score=3.2,
            source="llm_judge",
            reviewer="alice",
            confidence=0.8,
        )
        assert result.annotations[0].metadata == {
            "reviewer": "alice",
            "confidence": 0.8,
        }

    def test_multiple_annotations_accumulate(self) -> None:
        result = _result()
        result.annotate("answer", quality_score=4.0)
        result.annotate("answer", quality_score=3.5, source="second_rater")
        assert len(result.annotations) == 2


class TestEmitQualityAnnotation:
    def test_emit_creates_span_with_run_id_and_scores(self) -> None:
        tracing = MagicMock()
        span = MagicMock()
        tracing.start_span.return_value = span

        annotation = QualityAnnotation(
            target="answer",
            quality_score=4.5,
            source="human_feedback",
            metadata={"reviewer": "alice"},
        )
        emit_quality_annotation(
            annotation,
            tracing,
            run_id="run-xyz",
            workflow_name="wf-demo",
        )

        name_args, kwargs = tracing.start_span.call_args
        assert name_args[0] == "quality:answer"
        attrs = kwargs["attributes"]
        assert attrs[SpanAttr.WORKFLOW_RUN_ID] == "run-xyz"
        assert attrs[SpanAttr.WORKFLOW_NAME] == "wf-demo"
        assert attrs[SpanAttr.QUALITY_SCORE] == 4.5
        assert attrs[SpanAttr.QUALITY_SOURCE] == "human_feedback"
        assert attrs["agentloom.quality.target"] == "answer"
        assert attrs["agentloom.quality.metadata.reviewer"] == "alice"
        tracing.end_span.assert_called_once_with(span)

    def test_emit_no_tracer_is_noop(self) -> None:
        # Should not raise.
        annotation = QualityAnnotation(target="x", quality_score=1.0)
        emit_quality_annotation(annotation, None, run_id="r", workflow_name="w")
        emit_quality_annotation(annotation, 0, run_id="r", workflow_name="w")

    def test_emit_all_iterates(self) -> None:
        tracing = MagicMock()
        tracing.start_span.return_value = MagicMock()

        result = _result()
        result.annotate("answer", quality_score=4.0)
        result.annotate("summary", quality_score=3.5, source="llm_judge")
        emit_quality_annotations(result, tracing)

        assert tracing.start_span.call_count == 2
        span_names = {call.args[0] for call in tracing.start_span.call_args_list}
        assert span_names == {"quality:answer", "quality:summary"}


class TestEngineSetsRunIdOnResult:
    """The engine must plumb its run_id into the returned WorkflowResult
    so downstream annotate() calls emit under the correct trace."""

    async def test_run_id_propagates(self, mock_gateway) -> None:
        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.models import (
            StepDefinition,
            StepType,
            WorkflowConfig,
            WorkflowDefinition,
        )

        workflow = WorkflowDefinition(
            name="wf-run-id",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={},
            steps=[StepDefinition(id="s", type=StepType.LLM_CALL, prompt="hi")],
        )
        engine = WorkflowEngine(
            workflow=workflow, provider_gateway=mock_gateway, run_id="explicit-run"
        )
        result = await engine.run()
        assert result.run_id == "explicit-run"


class TestAutoEmitFromAnnotate:
    """``result.annotate()`` must auto-emit a quality span when a tracing
    context is wired — that's the contract issue #59 promises in its
    example. Manual ``emit_quality_annotation`` calls remain available for
    offline / replay paths but are no longer the only way to publish."""

    def test_annotate_with_tracing_attached_emits_span(self) -> None:
        from unittest.mock import MagicMock

        from agentloom.core.results import WorkflowResult, WorkflowStatus

        tracing = MagicMock()
        result = WorkflowResult(
            workflow_name="wf",
            status=WorkflowStatus.SUCCESS,
            run_id="run-x",
        )
        result.attach_tracing(tracing)
        annotation = result.annotate("answer", quality_score=4.5, source="human_feedback")

        # Span emitted with canonical attrs.
        tracing.start_span.assert_called_once()
        name, kwargs = (
            tracing.start_span.call_args.args[0],
            tracing.start_span.call_args.kwargs,
        )
        assert name == "quality:answer"
        attrs = kwargs["attributes"]
        assert attrs["workflow.run_id"] == "run-x"
        assert attrs["workflow.name"] == "wf"
        assert attrs["agentloom.quality.score"] == 4.5
        assert attrs["agentloom.quality.source"] == "human_feedback"
        # Annotation also recorded on the result for offline consumers.
        assert annotation in result.annotations

    def test_observer_exposes_tracing_via_public_property(self) -> None:
        # The engine wires ``result.attach_tracing`` from
        # ``observer.tracing`` (public) — not ``observer._tracing``
        # (private). Guards against re-introducing the private access.
        from unittest.mock import MagicMock

        from agentloom.observability.observer import WorkflowObserver

        sentinel = MagicMock()
        observer = WorkflowObserver(tracing=sentinel)
        assert observer.tracing is sentinel

    def test_annotate_without_tracing_only_records_annotation(self) -> None:
        # Offline / replay path: no tracer wired → no span emitted, but
        # ``result.annotations`` still grows. Keeps ``annotate()`` safe in
        # tests and pure-data scenarios.
        from agentloom.core.results import WorkflowResult, WorkflowStatus

        result = WorkflowResult(workflow_name="wf", status=WorkflowStatus.SUCCESS)
        result.annotate("answer", quality_score=3.0)
        assert len(result.annotations) == 1

    def test_engine_attaches_tracing_so_annotate_just_works(self, mock_gateway) -> None:
        # End-to-end: a workflow run with an observer wired returns a
        # result whose ``annotate()`` immediately publishes a span. This is
        # the exact pattern the issue's example shows.
        from unittest.mock import MagicMock

        import anyio

        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.models import (
            StepDefinition,
            StepType,
            WorkflowConfig,
            WorkflowDefinition,
        )
        from agentloom.observability.observer import WorkflowObserver

        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)

        workflow = WorkflowDefinition(
            name="wf-auto-emit",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={},
            steps=[StepDefinition(id="s", type=StepType.LLM_CALL, prompt="hi")],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=mock_gateway, observer=observer)
        result = anyio.run(engine.run)

        # Reset start_span to focus on the post-run annotate emission;
        # workflow / step / provider spans were created during run().
        tracing.start_span.reset_mock()
        result.annotate("answer", quality_score=4.7, source="llm_judge")
        tracing.start_span.assert_called_once()
        assert tracing.start_span.call_args.args[0] == "quality:answer"
