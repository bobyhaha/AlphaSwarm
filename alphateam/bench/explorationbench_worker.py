"""Subprocess worker for the ExplorationBench world: runs the official engines and grading code.

Agent-written AlienCode is executed here, never in the alphateam process, with the interpreter in hardened mode (the
mode the hosted evaluation uses; see `sandboxes/code/execution.py`), a CPU-time limit, and the parent's wall-clock
timeout. Everything that decides a verdict is the official code from `ext/explorationbench` (Apache-2.0), imported
under the v2 protocol flags: probe limits and feedback truncation (`run_eval._exec_tool_call`), answer extraction and
grading (`run_eval._run_single_test`, `harness._run_single_heldout`), prompts, worked examples and task sets.

Protocol: one JSON object on stdin, {"jobs": [{"op": ..., ...}, ...]}; one JSON list of results on stdout.
"""
from __future__ import annotations

import contextlib
import json
import os
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "ext" / "explorationbench" / "explorationbench"
for key, value in {"ALIENCODE_PROTOCOL_V2": "1", "ALIENLOGIC_PROTOCOL_V2": "1", "EVAL_TOOL_MODE": "1",
                   "ALIENCODE_HARDENED": "1", "ALIENLOGIC_EPISODE": "public_demo"}.items():
    os.environ[key] = value
sys.path.insert(0, str(ROOT))

_code = _logic = None


def code():
    """The official AlienCode harness module (prompts, limits, extraction, grading)."""
    global _code
    if _code is None:
        from sandboxes.code import execution
        execution.HARDENED = True
        from sandboxes.code import run_eval
        _code = run_eval
    return _code


def logic():
    """The official AlienLogic harness with the demo episode bound, as sandboxes/logic/run_eval.py does."""
    global _logic
    if _logic is None:
        from sandboxes.logic import episodes, harness
        harness.EPISODES = episodes.EPISODES
        harness.get_episode = episodes.get_episode
        harness.HeldoutTheorem = episodes.HeldoutTheorem
        harness.SeedExample = episodes.SeedExample
        harness.EpisodeSpec = episodes.EpisodeSpec
        episode = harness._apply_v2_task_set(episodes.PUBLIC_DEMO)
        _logic = (harness, episode)
    return _logic


# ---------------------------------------------------------------------------- AlienCode

def code_setup(_job):
    r = code()
    from sandboxes.code import protocol_v2
    from sandboxes.code.execution import REFERENCE_MANUAL, alien_exec
    return {"manual": REFERENCE_MANUAL,
            "demo_block": protocol_v2.demo_block(alien_exec),
            "system": r._initial_system_prompt(),
            "unsubmitted_retry": r._UNSUBMITTED_ANSWER_RETRY,
            "silent_retry": r._SILENT_ANSWER_RETRY,
            "tests": [{"id": t["id"], "question_type": t["question_type"], "difficulty": t["difficulty"],
                       "prompt": r._build_test_prompt(t)} for t in r.TEST_TASKS],
            "limits": {"calls_per_round": r.MAX_TOOL_CALLS_PER_ROUND, "units_per_call": r.MAX_EXPLORE_TESTS,
                       "loop_weight": r._LOOP_WEIGHT, "code_lines": r.MAX_CODE_LINES,
                       "output_lines": r.MAX_OUTPUT_LINES, "output_chars": r.MAX_OUTPUT_CHARS,
                       "rounds": r.N_EXPLORE_LOOPS}}


def code_probe(job):
    """One run_aliencode call, as `run_eval._exec_tool_call` handles a model's call (v2: 5 units per call)."""
    r = code()
    from sandboxes.code.execution import alien_exec
    src = str(job.get("code") or "")
    units = r._count_show_weighted(src)
    cap = r.MAX_EXPLORE_TESTS
    if units > cap:
        return {"rejected": True, "units": units,
                "output": (f'[ToolError] 这次调用需要 {units} 个观测值，超过单次上限 {cap}（EMIT 的每个参数算一个观测值，'
                           f'循环内每个按 {r._LOOP_WEIGHT}× 计数）。\n本次没有执行任何代码，也没有返回任何观测。'
                           '请把实验拆成多次调用后重试。')}
    src, warns = r._enforce_code_limits(src)
    payload = r._truncate_output(alien_exec(src))
    for w in warns:
        payload += f"\n[注意] {w}"
    return {"rejected": False, "units": units, "output": payload}


def code_unanswered(job):
    r = code()
    task = next(t for t in r.TEST_TASKS if t["id"] == job["task"])
    return {"unanswered": bool(r._unanswered(job["response"], task))}


def code_grade(job):
    """The grading branch of `run_eval._run_single_test`, applied to one answer."""
    r = code()
    from sandboxes.code.execution import alien_exec
    task = next(t for t in r.TEST_TASKS if t["id"] == job["task"])
    response = job["response"] or ""
    if task["question_type"] == "predict":
        prediction = r.extract_prediction(response, tool_protocol=r.TOOL_MODE)
        actual = r.predict_answer(task)
        correct = r.values_equal(prediction, actual) if prediction else False
        out = prediction
    else:
        src = r.extract_code(response)
        out = alien_exec(src) if src else "(无代码)"
        if task.get("test_cases"):
            passed, detail = r._run_test_cases(src, task)
            if task["expected"] is not None:
                correct = r.values_equal(out, task["expected"]) and passed
            else:
                correct = "[AlienError]" not in out and out != "(无代码)" and passed
            out += f"\n[TestCases {'PASS' if passed else 'FAIL'}]\n{detail}"
        elif task["expected"] is not None:
            correct = r.values_equal(out, task["expected"])
        else:
            correct = "[AlienError]" not in out and out != "(无代码)"
    return {"correct": bool(correct), "output": str(out)[:400]}


def code_references(_job):
    """Reference answers shipped with the demo world, formatted as an answerer would reply (for tests)."""
    r = code()
    refs = json.loads((ROOT / "sandboxes" / "code" / "reference_solutions.json").read_text())
    out = {}
    for t in r.TEST_TASKS:
        if t["question_type"] == "predict":
            out[t["id"]] = f"预测输出：{r.predict_answer(t)}"
        elif t["id"] in refs:
            out[t["id"]] = f"```aliencode\n{refs[t['id']]['reference']}\n```"
    return out


# ---------------------------------------------------------------------------- AlienLogic

def logic_setup(_job):
    h, ep = logic()
    from sandboxes.logic import protocol_v2
    from sandboxes.logic.engine import REFERENCE_MANUAL
    demo = protocol_v2.demo_block(h._demo_checker(ep), ep.seed_examples)
    system = h.SYSTEM_PROMPT_TEMPLATE.format(manual=REFERENCE_MANUAL) + (  # as harness.run_eval builds it under v2
        "\n\n─────────────────────────────────────────\n" + demo + "\n─────────────────────────────────────────\n")
    tests = []
    for t in ep.heldout_theorems:
        prompt = h._heldout_prompt(t)
        tests.append({"id": t.id, "role": t.role, "provable": t.alien_provable, "prompt": prompt,
                      "retry_prompt": h._heldout_retry_prompt(t.id, prompt)})
    return {"manual": REFERENCE_MANUAL, "demo_block": demo, "system": system, "tests": tests,
            "limits": {"calls_per_round": h.MAX_PROBES_PER_ROUND, "total": h.MAX_TOTAL_PROBES,
                       "rounds": h.N_EXPLORE_LOOPS}}


def logic_probe(job):
    """One check_proof call, as `harness._exec_probe_call` handles it (opaque diagnostic)."""
    h, ep = logic()
    from sandboxes.logic.engine import proof_diagnostic
    text = str(job.get("proof") or "")
    stripped = h.extract_first_proof(f"```alienlogic_proof\n{text}\n```")
    diag = proof_diagnostic(stripped or text, ep.alien_rules, hide_reason_detail=True)
    return {"accepted": bool(diag.get("accepted")), "output": h._format_diagnostic(diag)}


def logic_unanswered(job):
    h, _ = logic()
    return {"unanswered": not h._graded_answer(job["response"] or "")}


def logic_grade(job):
    """The grading branch of `harness._run_single_heldout` (asymmetric scoring of unprovable theorems)."""
    h, ep = logic()
    from sandboxes.logic.engine import proof_diagnostic
    theorem = next(t for t in ep.heldout_theorems if t.id == job["task"])
    response = job["response"] or ""
    proof_text = h.extract_first_proof(response)
    unprovable = h.extract_unprovable(response)
    if proof_text:
        diag = proof_diagnostic(proof_text, ep.alien_rules, hide_reason_detail=False,
                                expected_goal=theorem.goal, expected_premises=theorem.premises)
    elif unprovable:
        diag = {"accepted": False, "reason_class": "UNPROVABLE_CLAIM", "reason_id": "MODEL_DECLINED"}
    else:
        diag = {"accepted": False, "reason_class": "PARSE_ERROR", "reason_id": "NO_PROOF_BLOCK"}
    if theorem.alien_provable:
        correct = bool(diag.get("accepted"))
    else:
        correct = (not diag.get("accepted")) and bool(unprovable)
    return {"correct": correct, "output": f"{diag.get('reason_class')}/{diag.get('reason_id')}"}


def logic_references(_job):
    h, ep = logic()
    return {t.id: (f"```alienlogic_proof\n{t.reference_alien_proof}\n```" if t.alien_provable
                   else "ANSWER: UNPROVABLE\nAND_E2 is refused in this world.") for t in ep.heldout_theorems}


OPS = {f.__name__: f for f in (code_setup, code_probe, code_unanswered, code_grade, code_references,
                               logic_setup, logic_probe, logic_unanswered, logic_grade, logic_references)}


def main():
    cpu = int(os.environ.get("EB_CPU_SECONDS", "60"))
    with contextlib.suppress(Exception):
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    request = json.loads(sys.stdin.read() or "{}")
    results = []
    with contextlib.redirect_stdout(sys.stderr):  # the harness prints progress; stdout carries only the result
        for job in request.get("jobs", []):
            try:
                results.append(OPS[job["op"]](job))
            except Exception as exc:  # one bad job must not lose the others
                results.append({"error": f"{type(exc).__name__}: {exc}"})
    sys.__stdout__.write(json.dumps(results, ensure_ascii=False))
    sys.__stdout__.flush()


if __name__ == "__main__":
    main()
