"""SciGym world: experiments on the true system, agent code with SciGym's globals, submission checks and grading."""
import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.bench import make_world  # noqa: E402
from alphateam.bench.scigym import DATA  # noqa: E402
from alphateam.kernel import Kernel  # noqa: E402
from alphateam.metrics import measure  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.runner import run  # noqa: E402

CFG = {"tasks": 8, "home": 2, "eval_at": [], "world_opts": {"experiments": "3", "rounds": "3"}, "agents": 6}


def world(n=6, tmp=Path("/tmp/alphateam_scigym"), **cfg):
    shutil.rmtree(tmp, ignore_errors=True)
    k = Kernel(n, max_actions=4)
    k.env = make_world("scigym", CFG | cfg, n, seed=3)
    k.env.workdir = str(tmp)
    k.env.setup(k)
    return k


def act(k, aid, name, **args):
    return asyncio.run(k.act(aid, name, args))


def test_tasks_hide_the_system():
    k = world()
    env = k.env
    assert len(env.tasks) == 8 and env.n_super >= 1
    t = env.tasks["T1"]
    ok, msg = act(k, "A1", "open_task", task="T1")
    assert ok and "Incomplete SBML model" in msg and "listOfSpecies" in msg
    assert t.family not in msg and "BIOMD" not in msg  # the BioModels id is not shown
    assert "listOfReactions" not in t.meta["index"]["partial"] or t.meta["index"]["n_partial_reactions"] >= 0


def test_experiment_code_and_submission():
    k = world()
    env = k.env
    t = env.tasks["T1"]
    sid = next(s["id"] for s in t.meta["index"]["species"] if not s["constant"] and not s["boundary"])
    ok, msg = act(k, "A1", "experiment", task="T1")
    assert ok and "exp1" in msg and "Time" in msg, msg
    ok, msg = act(k, "A1", "experiment", task="T1", concentrations={sid: 0.5}, to="global")
    assert ok and f"{sid}=0.5" in msg, msg
    ok, msg = act(k, "A1", "experiment", task="T1", concentrations={"nope": 1})
    assert not ok and "unknown species" in msg
    # A2 sees only the experiment posted globally
    ok, out = act(k, "A2", "run_code", task="T1", code="print(sorted(experiment_history)); import libsbml; "
                  "print(libsbml.readSBMLFromString(input_sbml_string).getModel().getNumSpecies())")
    assert ok and "['exp2']" in out, out
    ok, out = act(k, "A1", "run_code", task="T1", code="df = simulate(input_sbml_string); print(sorted(experiment_history), "
                  "list(df.columns)[-1]); shared_variables.add('n', len(df)); final_sbml = input_sbml_string")
    assert ok and "['exp1', 'exp2'] Time" in out and "final_sbml saved" in out, out
    ok, out = act(k, "A1", "run_code", task="T1", code="print(shared_variables.access('n'))")
    assert ok and out.strip().isdigit(), out
    assert act(k, "A1", "experiment", task="T1")[0]
    ok, msg = act(k, "A1", "experiment", task="T1")
    assert not ok and "budget" in msg
    ok, msg = act(k, "A1", "submit", task="T1", sbml="<sbml>broken")
    assert not ok and "does not load" in msg
    ok, msg = act(k, "A1", "submit", task="T1", file="final_sbml.xml")
    assert ok and "0 reactions" in msg, msg
    ok, msg = act(k, "A3", "submit", task="T1", file="../A1/final_sbml.xml")
    assert not ok


def test_grading_partial_vs_truth():
    k = world()
    env = k.env
    t1, t2 = env.tasks["T1"], env.tasks["T2"]
    assert act(k, "A1", "submit", task="T1", sbml=t1.meta["index"]["partial"])[0]
    truth = (DATA / t2.family / "truth.xml").read_text()
    assert act(k, "A2", "submit", task="T2", sbml=truth)[0]
    asyncio.run(env.finalize(k))
    assert env.task_scores["T1"] == 0.0 and env.task_scores["T2"] == 1.0
    d1, d2 = env.details["T1"], env.details["T2"]
    assert 0 < d1["observe_smape"] <= 1 and d2["observe_smape"] < 1e-6 and d2["rpm_f1"] == 1.0
    assert d2["perturb_rounds"] > 0 and d2["perturb_smape"] < 1e-3, d2
    m = measure(k)
    assert abs(m["score"] - 1 / 8) < 1e-3 and m["coverage"] == 0.25
    env.metric = "smape"
    assert env.headline(d1) == 1 - d1["observe_smape"]


def test_mock_run(tmp=Path("/tmp/alphateam_scigym_run")):
    shutil.rmtree(tmp, ignore_errors=True)
    k = Kernel(10, max_actions=4)
    k.env = make_world("scigym", CFG | {"agents": 10}, 10, seed=1)
    k.env.setup(k)
    data = asyncio.run(run(k, MockBackend(0), 2, tmp, {"condition": "self_org", "backend": "mock"}, quiet=True))
    m = data["snapshots"][-1]["metrics"]
    assert m["score"] == 0.0 and m["coverage"] > 0  # unchanged incomplete models recover no reactions
    assert data["world"]["details"] and (tmp / "report.md").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
