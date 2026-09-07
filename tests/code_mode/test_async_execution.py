"""Lifecycle contracts for code mode on a standard asyncio loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from pathlib import Path
from typing import Literal

import pytest
from pydantic_ai import RunContext, Tool
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_monty import Monty, MontySyntaxError, MountDir, OsFunction

from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.code_mode import CodeModeToolset

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def build_ctx(deps: None, toolset: CodeModeToolset[None]) -> RunContext[None]:
    ctx = RunContext(deps=deps, model=TestModel(), usage=RunUsage(), prompt=None, messages=[])
    ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(ctx)
    return ctx


def no_os(_function: OsFunction, _args: tuple[object, ...], _kwargs: dict[str, object]) -> object:
    raise AssertionError('Unsupported OS handler must not run')  # pragma: no cover


class TestAsyncExecution:
    async def test_parser_syntax_exception_is_a_model_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(MontySyntaxError) as syntax_error:
            Monty('def (')

        def parse(*_args: object, **_kwargs: object) -> None:
            raise syntax_error.value

        async def value() -> int:
            raise AssertionError('Invalid code must not execute tools')  # pragma: no cover

        # Parser diagnostics have used both MontySyntaxError and MontyTypingError
        # across Monty versions. Preserve the caller's retry contract for both.
        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.Monty', parse)
        wrapper = CodeModeToolset(FunctionToolset(tools=[value]), execution_mode='async')
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        with pytest.raises(ModelRetry, match='Syntax error in code'):
            await wrapper.call_tool('run_code', {'code': 'invalid'}, ctx, tool)

    async def test_callback_scheduled_after_completion_cannot_start_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        callbacks: list[Coroutine[object, object, object]] = []

        class DeferredCallbackRepl:
            async def feed_run_async(
                self,
                code: str,
                *,
                external_functions: dict[str, Callable[..., Coroutine[object, object, object]]],
                print_callback: object,
            ) -> object:
                callbacks.append(external_functions['value']())
                return 42

        async def value() -> int:
            raise AssertionError('A late callback must not start tool work')  # pragma: no cover

        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.MontyRepl', DeferredCallbackRepl)
        wrapper = CodeModeToolset(FunctionToolset(tools=[value]), execution_mode='async')
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        result = await wrapper.call_tool('run_code', {'code': '42'}, ctx, tool)
        assert result.return_value == 42
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

    async def test_cpu_cancellation_restores_repl(self) -> None:
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
        await asyncio.wait_for(started.wait(), timeout=2)
        # This timer can fire only if the VM yields the event loop while computing.
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        result = await wrapper.call_tool('run_code', {'code': '40 + 2'}, ctx, tool)
        assert result.return_value == 42

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
        # Give both native-created callbacks time to start before ending the snippet.
        code = (
            'import asyncio\na = nested(number=1)\nb = nested(number=2)\n'
            'for i in range(100000):\n    x = i + 1\n' + endings[finish]
        )
        task = asyncio.create_task(wrapper.call_tool('run_code', {'code': code}, ctx, tool))
        await asyncio.wait_for(started.wait(), timeout=2)
        if finish.startswith('cancel'):
            task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        assert not task.done()
        if finish == 'cancel_twice':
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release_cleanup.set()
        if finish.startswith('cancel'):
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        elif finish == 'error':
            with pytest.raises(ModelRetry, match='division by zero'):
                await asyncio.wait_for(task, timeout=2)
        else:
            assert (await asyncio.wait_for(task, timeout=2)).return_value == 42
        assert active == set()
        assert sorted(finished) == [1, 2]
        await asyncio.sleep(0.02)
        assert sorted(finished) == [1, 2]

    @pytest.mark.parametrize('configuration', ['mount', 'os', 'sequential', 'global_sequential'])
    async def test_unsupported_configurations_fail(self, configuration: str, tmp_path: Path) -> None:
        async def value() -> int:
            raise AssertionError('Unsupported execution mode must not start tool work')  # pragma: no cover

        wrapper = CodeModeToolset(
            FunctionToolset(tools=[Tool(value, sequential=configuration == 'sequential')]),
            execution_mode='async',
            mount=MountDir('/work', tmp_path) if configuration == 'mount' else None,
            os_access=no_os if configuration == 'os' else None,
        )
        ctx = await build_ctx(None, wrapper)
        tool = (await wrapper.get_tools(ctx))['run_code']
        with ToolManager.parallel_execution_mode('sequential' if configuration == 'global_sequential' else 'parallel'):
            with pytest.raises(UserError, match='Async code mode requires'):
                await wrapper.call_tool('run_code', {'code': '1'}, ctx, tool)
