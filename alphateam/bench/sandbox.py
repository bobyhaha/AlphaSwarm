"""Running agent-written Python for benchmarks that need analysis code.

Each call runs in a fresh subprocess (isolated mode) inside a workspace directory, with a CPU-time limit, a wall-clock
timeout and truncated output. Files written to the workspace persist between calls, which is how an agent keeps state.
This is a guard against runaway code, not a security boundary: run it only on your own machine.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PRELUDE = """
import resource, sys
try:
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
except Exception:
    pass
sys.argv = ['agent']
"""

_slots: asyncio.Semaphore | None = None


def _semaphore(n: int) -> asyncio.Semaphore:
    global _slots
    if _slots is None:
        _slots = asyncio.Semaphore(n)
    return _slots


def clip_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit * 2 // 3]
    tail = text[-limit // 3:]
    return f"{head}\n…[{len(text) - len(head) - len(tail)} characters cut]…\n{tail}"


async def run_python(code: str, workdir: Path, setup: str = "", timeout: float = 60, limit: int = 4000,
                     python: str | None = None, concurrency: int = 12) -> tuple[bool, str]:
    """Run setup + code in workdir. Returns (finished without error, combined stdout/stderr, clipped)."""
    workdir.mkdir(parents=True, exist_ok=True)
    script = PRELUDE.format(cpu=int(timeout)) + setup + "\n" + code
    path = workdir / ".agent_run.py"
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(workdir), "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"}  # discourage network use
    async with _semaphore(concurrency):
        path.write_text(script)
        proc = await asyncio.create_subprocess_exec(
            python or sys.executable, "-I", path.name, cwd=workdir, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout + 5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"[stopped: the code ran longer than {timeout:.0f}s]"
    text = out.decode("utf-8", "replace").replace(str(workdir), ".")
    ok = proc.returncode == 0
    if not text.strip():
        text = "[no output; print what you want to see]" if ok else f"[exited with code {proc.returncode}]"
    return ok, clip_output(text, limit)
