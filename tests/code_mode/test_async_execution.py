"""Lifecycle contracts for `execution_mode='async'` code mode on a standard asyncio loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, TypeVar

import pytest
from pydantic_ai import RunContext, Tool
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_monty import MountDir, OsFunction

from pydantic_ai_harness import CodeMode
from pydantic_ai_harness._monty_exec import PrintCapture
from pydantic_ai_harness.code_mode import CodeModeToolset
from pydantic_ai_harness.code_mode._toolset import (  # pyright: ignore[reportPrivateUsage]
    _async_execution,
    _AsyncMontyRunState,
)

pytestmark = pytest.mark.anyio

T = TypeVar('T')


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


_entered_toolsets: list[CodeModeToolset[Any]] = []


@pytest.fixture(autouse=True)
async def _close_direct_toolsets(anyio_backend: str) -> AsyncIterator[None]:
    """Close toolsets entered by the lower-level `call_tool` tests."""
    yield
    while _entered_toolsets:
        toolset = _entered_toolsets.pop()
        await toolset.__aexit__(None, None, None)


async def build_ctx(deps: T, toolset: CodeModeToolset[T]) -> RunContext[T]:
    """Enter the toolset and build a `RunContext` with a prepared `ToolManager`."""
    await toolset.__aenter__()
    _entered_toolsets.append(toolset)
    ctx = RunContext[T](
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        pending_messages=[],
    )
    ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(ctx)
    return ctx


def no_os(_function: OsFunction, _args: tuple[object, ...], _kwargs: dict[str, object]) -> object:
    raise AssertionError('Unsupported OS handler must not run')  # pragma: no cover


class TestAsyncExecution:
    async def test_syntax_error_is_a_model_retry(self) -> None:
        async def value() -> int:
            raise AssertionError('Invalid code must not execute tools')  # pragma: no cover

        wrapper = CodeModeToolset(FunctionToolset(tools=[value]), execution_mode='async')
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        # A fresh session type-checks its first feed, so a parse failure surfaces as the
        # checker's invalid-syntax diagnostic; on a warm session it is a plain syntax error.
        with pytest.raises(ModelRetry, match='(Syntax|Type) error in code'):
            await wrapper.call_tool('run_code', {'code': 'def ('}, ctx, tool)
        # The reset left the session usable for the retry.
        result = await wrapper.call_tool('run_code', {'code': '40 + 2'}, ctx, tool)
        assert result.return_value == 42

    async def test_callback_scheduled_after_completion_cannot_start_tool(self) -> None:
        """A callback Monty schedules as the feed settles must not start tool work."""
        callbacks: list[Any] = []

        class DeferredCallbackSession:
            async def feed_run(
                self,
                code: str,
                *,
                external_lookup: dict[str, Any],
                print_callback: object,
                skip_type_check: bool,
            ) -> object:
                callbacks.append(external_lookup['value']())
                return 42

        async def dispatch(name: str, kwargs: dict[str, Any]) -> Any:
            raise AssertionError('A late callback must not start tool work')  # pragma: no cover

        result = await _async_execution(
            DeferredCallbackSession(),  # pyright: ignore[reportArgumentType]
            '42',
            run_state=_AsyncMontyRunState(),
            dispatch=dispatch,
            callable_defs={'value': None},  # pyright: ignore[reportArgumentType] -- only keys are read
            skip_type_check=True,
            capture=PrintCapture(),
        )
        assert result == 42
        assert len(callbacks) == 1
        with pytest.raises(asyncio.CancelledError):
            await callbacks[0]

    async def test_named_tools_keep_validation_and_errors(self) -> None:
        async def value(number: int) -> int:
            if number < 0:
                raise ModelRetry('number must be positive')
            return number

        wrapper = CodeModeToolset(FunctionToolset(tools=[Tool(value, name='get-value')]), execution_mode='async')
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        result = await wrapper.call_tool('run_code', {'code': 'await get_value(number=3)'}, ctx, tool)
        assert result.return_value == 3
        assert next(iter(result.metadata['tool_calls'].values())).tool_name == 'get-value'
        with pytest.raises(ModelRetry, match='does not accept positional arguments'):
            await wrapper.call_tool('run_code', {'code': 'await get_value(3)'}, ctx, tool)
        with pytest.raises(ModelRetry, match='number must be positive'):
            await wrapper.call_tool('run_code', {'code': 'await get_value(number=-1)'}, ctx, tool)

    @pytest.mark.parametrize('execution_mode', ['snapshot', 'async'])
    async def test_results_context_metadata_and_restart(self, execution_mode: Literal['snapshot', 'async']) -> None:
        scope = ContextVar('scope', default='missing')

        async def value(number: int) -> ToolReturn:
            assert scope.get() == 'request'
            return ToolReturn(number * 2, metadata={'source': 'test'})

        wrapper = CodeMode[None](execution_mode=execution_mode).get_wrapper_toolset(FunctionToolset(tools=[value]))
        assert isinstance(wrapper, CodeModeToolset)
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        token = scope.set('request')
        try:
            result = await wrapper.call_tool(
                'run_code',
                {
                    'code': 'import asyncio\nvalues = await asyncio.gather(value(number=2), value(number=4))\nprint("done")\nvalues'
                },
                ctx,
                tool,
            )
        finally:
            scope.reset(token)
        assert result.return_value == {'output': 'done\n', 'result': [4, 8]}
        assert len(result.metadata['tool_calls']) == 2
        assert [r.metadata for r in result.metadata['tool_returns'].values()] == [{'source': 'test'}] * 2
        assert (await wrapper.call_tool('run_code', {'code': 'values[0]'}, ctx, tool)).return_value == 4
        with pytest.raises(ModelRetry, match='not defined|unresolved-reference'):
            await wrapper.call_tool('run_code', {'code': 'values[0]', 'restart': True}, ctx, tool)
        assert (await wrapper.call_tool('run_code', {'code': '6 * 7'}, ctx, tool)).return_value == 42

    async def test_cpu_loop_stays_off_the_event_loop_and_cancels(self) -> None:
        async def ready() -> int:
            started.set()
            return 1

        started = asyncio.Event()
        wrapper = CodeModeToolset(FunctionToolset(tools=[ready]), execution_mode='async')
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        task = asyncio.create_task(
            wrapper.call_tool(
                'run_code',
                {'code': 'x = await ready()\nwhile True:\n    x += 1'},
                ctx,
                tool,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        # This timer can fire only if the VM leaves the event loop free while computing.
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        # The cancelled turn discarded its worker; the next call starts a fresh REPL.
        result = await wrapper.call_tool('run_code', {'code': '40 + 2'}, ctx, tool)
        assert result.return_value == 42

    async def test_toolset_exit_interrupts_live_cpu_feed(self) -> None:
        """Teardown owns a feed the run abandoned mid-compute.

        pydantic-ai closes a run's toolsets before the in-flight tool-call tasks observe
        cancellation, so `__aexit__` is the first code positioned to stop a CPU-burning
        VM. It must interrupt the feed at once rather than sit on the busy worker until
        its `max_duration_secs` backstop aborts the snippet.
        """
        started = asyncio.Event()

        async def ready() -> int:
            started.set()
            return 1

        wrapper = CodeModeToolset(FunctionToolset(tools=[ready]), execution_mode='async')
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        task = asyncio.create_task(
            wrapper.call_tool(
                'run_code',
                {'code': 'x = await ready()\nwhile True:\n    x += 1'},
                ctx,
                tool,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        await asyncio.sleep(0.05)  # the VM is now computing, with no suspension point ahead
        _entered_toolsets.remove(wrapper)
        loop = asyncio.get_running_loop()
        exit_started = loop.time()
        await wrapper.__aexit__(None, None, None)
        elapsed = loop.time() - exit_started
        assert elapsed < 2, f'toolset exit waited {elapsed:.1f}s on a live feed instead of interrupting it'
        # The abandoned call unwinds as cancelled once its feed is interrupted.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

    @pytest.mark.parametrize('finish', ['cancel', 'cancel_twice', 'error', 'return'])
    async def test_nested_calls_are_drained(self, finish: str) -> None:
        started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        active: set[int] = set()
        finished: list[int] = []

        async def nested(number: int) -> None:
            active.add(number)
            if len(active) == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await release_cleanup.wait()
                active.remove(number)
                finished.append(number)

        wrapper = CodeModeToolset(FunctionToolset(tools=[nested]), execution_mode='async')
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        endings = {
            'cancel': 'await asyncio.gather(a, b)',
            'cancel_twice': 'await asyncio.gather(a, b)',
            'error': '1 / 0',
            'return': '42',
        }
        # Give both dispatched tasks time to start before ending the snippet.
        code = (
            'import asyncio\na = nested(number=1)\nb = nested(number=2)\n'
            'for i in range(100000):\n    x = i + 1\n' + endings[finish]
        )
        task = asyncio.create_task(wrapper.call_tool('run_code', {'code': code}, ctx, tool))
        await asyncio.wait_for(started.wait(), timeout=5)
        if finish.startswith('cancel'):
            task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        assert not task.done()
        if finish == 'cancel_twice':
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release_cleanup.set()
        if finish.startswith('cancel'):
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        elif finish == 'error':
            with pytest.raises(ModelRetry, match='division by zero'):
                await asyncio.wait_for(task, timeout=5)
        else:
            assert (await asyncio.wait_for(task, timeout=5)).return_value == 42
        assert active == set()
        assert sorted(finished) == [1, 2]
        await asyncio.sleep(0.02)
        assert sorted(finished) == [1, 2]

    async def test_nested_call_budget_is_enforced(self) -> None:
        calls = 0

        async def value() -> int:
            nonlocal calls
            calls += 1
            return calls

        wrapper = CodeModeToolset(FunctionToolset(tools=[value]), execution_mode='async', max_tool_calls=2)
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        with pytest.raises(ModelRetry, match='allows 2 nested tool calls'):
            await wrapper.call_tool(
                'run_code',
                {'code': 'for i in range(3):\n    x = await value()\nx'},
                ctx,
                tool,
            )
        assert calls == 2

    @pytest.mark.parametrize('configuration', ['mount', 'os'])
    async def test_unsupported_host_access_fails_at_run_start(self, configuration: str, tmp_path: Path) -> None:
        async def value() -> int:
            raise AssertionError('Unsupported execution mode must not start tool work')  # pragma: no cover

        wrapper = CodeModeToolset(
            FunctionToolset(tools=[value]),
            execution_mode='async',
            mount=MountDir(virtual_path='/work', host_path=tmp_path) if configuration == 'mount' else None,
            os_access=no_os if configuration == 'os' else None,
        )
        with pytest.raises(UserError, match='Async code mode requires'):
            await wrapper.__aenter__()

    @pytest.mark.parametrize('configuration', ['sequential', 'global_sequential'])
    async def test_sequential_configurations_fail_at_call(self, configuration: str) -> None:
        async def value() -> int:
            raise AssertionError('Unsupported execution mode must not start tool work')  # pragma: no cover

        wrapper = CodeModeToolset(
            FunctionToolset(tools=[Tool(value, sequential=configuration == 'sequential')]),
            execution_mode='async',
        )
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        with ToolManager.parallel_execution_mode('sequential' if configuration == 'global_sequential' else 'parallel'):
            with pytest.raises(UserError, match='Async code mode requires'):
                await wrapper.call_tool('run_code', {'code': '1'}, ctx, tool)

    async def test_eager_and_async_are_mutually_exclusive(self) -> None:
        capability = CodeMode[None](eager=True, execution_mode='async')
        with pytest.raises(UserError, match='not compatible with eager execution'):
            capability.get_wrapper_toolset(FunctionToolset(tools=[]))
