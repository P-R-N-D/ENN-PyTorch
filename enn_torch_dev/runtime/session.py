from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from enn_torch_dev.data import KVBatch

from .history import RuntimeHistorySummary, RuntimePassHistory
from .orchestration import ConservativeRuntimeOrchestrator, RuntimePassResult
from .source_factory import RuntimePassSourceFactory
from .summary import RuntimePassSummary, summarize_runtime_pass


@dataclass(frozen=True, slots=True)
class RuntimeSessionRecord:
    """One yielded record from a bounded conservative runtime session."""

    pass_index: int
    pass_result: RuntimePassResult
    pass_summary: RuntimePassSummary
    history_summary: RuntimeHistorySummary


class ConservativeRuntimeSession:
    """Run finite runtime passes lazily with bounded pass and history limits."""

    def __init__(
        self,
        orchestrator: ConservativeRuntimeOrchestrator,
        history: RuntimePassHistory,
        *,
        max_passes: int,
    ) -> None:
        if not isinstance(orchestrator, ConservativeRuntimeOrchestrator):
            raise TypeError(
                "ConservativeRuntimeSession.orchestrator must be a "
                "ConservativeRuntimeOrchestrator."
            )
        if not isinstance(history, RuntimePassHistory):
            raise TypeError(
                "ConservativeRuntimeSession.history must be a RuntimePassHistory."
            )
        if not isinstance(max_passes, int) or isinstance(max_passes, bool):
            raise TypeError("ConservativeRuntimeSession.max_passes must be an integer.")
        if max_passes <= 0:
            raise ValueError("ConservativeRuntimeSession.max_passes must be positive.")

        self.orchestrator = orchestrator
        self.history = history
        self.max_passes = max_passes

    def run_passes(
        self,
        pass_sources: Iterable[Iterable[KVBatch]],
    ) -> Iterator[RuntimeSessionRecord]:
        if isinstance(pass_sources, KVBatch):
            raise TypeError(
                "ConservativeRuntimeSession.run_passes expects an iterable of "
                "finite KVBatch iterables."
            )
        if not isinstance(pass_sources, Iterable):
            raise TypeError(
                "ConservativeRuntimeSession.run_passes expects an iterable of "
                "finite KVBatch iterables."
            )
        return self._run_passes(iter(pass_sources))

    def _execute_pass(
        self,
        pass_index: int,
        source: Iterable[KVBatch],
    ) -> RuntimeSessionRecord:
        pass_result = self.orchestrator.run_pass(source)
        pass_summary = summarize_runtime_pass(pass_result)
        history_summary = self.history.append_summary(pass_summary)
        return RuntimeSessionRecord(
            pass_index=pass_index,
            pass_result=pass_result,
            pass_summary=pass_summary,
            history_summary=history_summary,
        )

    def _run_passes(
        self,
        pass_sources: Iterator[Iterable[KVBatch]],
    ) -> Iterator[RuntimeSessionRecord]:
        for pass_index in range(self.max_passes):
            try:
                source = next(pass_sources)
            except StopIteration:
                return

            record = self._execute_pass(pass_index, source)
            try:
                yield record
            finally:
                del source
                del record

    def run_factory(
        self,
        source_factory: RuntimePassSourceFactory,
    ) -> Iterator[RuntimeSessionRecord]:
        if not isinstance(source_factory, RuntimePassSourceFactory):
            raise TypeError(
                "ConservativeRuntimeSession.source_factory must provide "
                "create_pass_source(pass_index)."
            )
        return self._run_factory(source_factory)

    def _run_factory(
        self,
        source_factory: RuntimePassSourceFactory,
    ) -> Iterator[RuntimeSessionRecord]:
        for pass_index in range(self.max_passes):
            source = source_factory.create_pass_source(pass_index)
            if isinstance(source, KVBatch) or not isinstance(source, Iterable):
                raise TypeError(
                    "RuntimePassSourceFactory.create_pass_source must return an "
                    "iterable of KVBatch."
                )

            record = self._execute_pass(pass_index, source)
            try:
                yield record
            finally:
                del source
                del record
