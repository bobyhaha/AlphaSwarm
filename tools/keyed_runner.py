#!/usr/bin/env python3
"""Hold the OpenRouter key in memory and start the paid jobs this project queues.

  .venv/bin/python tools/keyed_runner.py                    # asks for the key at a hidden prompt
  .venv/bin/python tools/keyed_runner.py --key-pipe PATH    # or takes it once through a named pipe at PATH

The key is given once and lives only in this process and in the jobs it starts: never in a file, a log or a command
line (a named pipe passes it through memory and is removed once read). A job is a file runs_recipes/queue/<name>.job.json holding {"argv": [...], "log": "<path inside the project>"}.
Only this project's own Python tools can be started (argv[0] a Python interpreter, argv[1] a script under tools/).
<name>.stop asks a running job to stop; <name>.done holds how it ended. Ctrl-C, or closing this tab, stops every job.
The machine is kept from idle sleep while a job runs (caffeinate); a closed lid still sleeps, so keep it open.
Only one runner serves the queue at a time (runs_recipes/queue/runner.pid): two would start every job twice.
"""
import argparse
import getpass
import json
import os
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "runs_recipes" / "queue"
LOCK = QUEUE / "runner.pid"
PYTHONS = {str((ROOT / ".venv/bin/python").resolve()), str(ROOT / ".venv/bin/python"), "python3", "/usr/bin/python3"}


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def usage(key: str) -> float:
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return float(json.load(r)["data"]["usage"])


def from_pipe(path: Path) -> str:
    """The key through a named pipe: it passes through memory only, never a file, a command line or the screen."""
    path = path.expanduser()
    if path.exists() and not stat.S_ISFIFO(path.stat().st_mode):
        sys.exit(f"{path} exists and is not a pipe; nothing was read")
    if not path.exists():
        os.mkfifo(path, 0o600)
    print(f"{stamp()} waiting for the key on the pipe {path}", flush=True)
    try:
        with open(path) as f:  # waits here until the sender opens the pipe
            return f.read().strip()
    finally:
        path.unlink(missing_ok=True)


def one_runner():
    """Two runners on one queue would start every job twice: refuse to be the second."""
    QUEUE.mkdir(parents=True, exist_ok=True)
    try:
        other = int(LOCK.read_text())
        os.kill(other, 0)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        pass
    except PermissionError:
        sys.exit(f"another runner seems to be running; stop it first (or delete {LOCK.relative_to(ROOT)} if it is stale)")
    else:
        if other != os.getpid():
            sys.exit(f"another runner is already running (pid {other}); stop it first "
                     f"(or delete {LOCK.relative_to(ROOT)} if it is stale)")
    LOCK.write_text(str(os.getpid()))


def ask_key(pipe: Path | None = None) -> str:
    key = from_pipe(pipe) if pipe else os.environ.get("OPENROUTER_API_KEY", "").strip()
    for _ in range(3):
        if not key:
            key = getpass.getpass("OpenRouter API key (hidden, press Enter when pasted): ").strip()
        if not key:
            continue
        try:
            used = usage(key)
        except urllib.error.HTTPError as exc:
            print(f"the key was refused ({exc.code}); try again")
            key = ""
            continue
        except Exception as exc:  # no network: keep the key, the jobs will say so themselves
            print(f"could not reach OpenRouter to check the key ({exc.__class__.__name__}); carrying on")
            return key
        led = json.loads((ROOT / "spend.json").read_text())
        spent = used - led["baseline_usage"]
        print(f"key accepted. Research ledger: spent ${spent:.2f} of ${led['cap_usd']:.2f}; ${led['cap_usd'] - spent:.2f} left")
        return key
    sys.exit("no usable key given")


def allowed(job: dict) -> str | None:
    argv = job.get("argv")
    if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(x, str) for x in argv):
        return "argv must be a list of strings: an interpreter and a script"
    if argv[0] not in PYTHONS:
        return "argv[0] must be this project's Python"
    script = (ROOT / argv[1]).resolve() if not os.path.isabs(argv[1]) else Path(argv[1]).resolve()
    if script.suffix != ".py" or not script.is_file() or (ROOT / "tools").resolve() not in script.parents:
        return "argv[1] must be a script under this project's tools/"
    log = (ROOT / str(job.get("log", ""))).resolve()
    if not str(job.get("log", "")) or ROOT.resolve() not in log.parents:
        return "log must be a path inside the project"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--key-pipe", type=Path, help="take the key once through this named pipe instead of the hidden prompt")
    args = ap.parse_args()
    one_runner()
    key = ask_key(args.key_pipe)
    env = dict(os.environ, OPENROUTER_API_KEY=key, PYTHONUNBUFFERED="1")
    running: dict[str, subprocess.Popen] = {}
    asked: dict[str, float] = {}
    seen: set[str] = set()

    def stop_all(*_):
        for name, p in running.items():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            (QUEUE / f"{name}.done").write_text(json.dumps({"exit": None, "stopped": "the runner was closed", "ended": time.time()}))
        LOCK.unlink(missing_ok=True)
        print(f"\n{stamp()} runner closed; {len(running)} job(s) stopped")
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop_all)
    print(f"{stamp()} ready: waiting for jobs in {QUEUE.relative_to(ROOT)} (Ctrl-C stops everything)")
    while True:
        for path in sorted(QUEUE.glob("*.job.json")):
            name = path.name[:-len(".job.json")]
            if name in seen or (QUEUE / f"{name}.done").exists():
                continue
            seen.add(name)
            try:
                job = json.loads(path.read_text())
                why = allowed(job)
            except Exception as exc:
                job, why = {}, f"unreadable job: {exc}"
            if why:
                (QUEUE / f"{name}.done").write_text(json.dumps({"exit": None, "error": why, "ended": time.time()}))
                print(f"{stamp()} refused {name}: {why}")
                continue
            log = (ROOT / job["log"]).resolve()
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "ab") as out:
                p = subprocess.Popen(job["argv"], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            running[name] = p
            try:
                subprocess.Popen(["caffeinate", "-i", "-w", str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass
            print(f"{stamp()} started {name}: {' '.join(job['argv'][1:])[:160]}  (log {job['log']})")
        for name, p in list(running.items()):
            code = p.poll()
            if code is not None:
                del running[name]
                (QUEUE / f"{name}.done").write_text(json.dumps({"exit": code, "ended": time.time()}))
                print(f"{stamp()} ended {name}: exit {code}")
                continue
            if (QUEUE / f"{name}.stop").exists():
                first = asked.setdefault(name, time.time())
                try:
                    os.killpg(p.pid, signal.SIGKILL if time.time() - first > 20 else signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(1)


if __name__ == "__main__":
    main()
