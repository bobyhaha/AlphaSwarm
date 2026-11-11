"""SciGym simulation and grading, run in .venv-bench (libsbml, libsedml, roadrunner, tellurium).

Reimplements the parts of SciGym (github.com/h4duan/SciGym, NeurIPS 2025) the world needs, following its code:
- loading an SBML model: scigym/data/sbml.py SBML.load_sbml_from_string_or_file (local parameters made global,
  initial assignments and function definitions expanded, unset parameters set to 0);
- the time course: SBML.load_sedml_from_string_or_file (the uniform time course with the latest end time) and
  scigym/data/simulator.py (integrator chosen from the SED-ML algorithm, CVODE when unsupported; species concentrations);
- experiments: scigym/actions/experiment.py + SBML.change_initial_concentration (the TRUE model with new initial
  concentrations; only species of the incomplete model; not constant species);
- metrics: scigym/eval/evaluator.py and eval/utils.py (observation SMAPE, exact reaction recovery with and without
  modifiers, typed species-interaction F1) and SBML.eval_add_noise_to_initial_concentrations (perturbed conditions).

Used two ways: as a JSON-in/JSON-out command (`python scigym_worker.py < request.json`) by the world, and imported by
agent code (`simulate(sbml_string)`) inside the sandbox.
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings

import libsbml
import libsedml
import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------------------------- loading

def load_sbml(text: str):
    """SciGym SBML.load_sbml_from_string_or_file. Returns (document, model); raises ValueError on SBML errors."""
    doc = libsbml.readSBMLFromString(text)
    for i in range(doc.getNumErrors()):
        err = doc.getError(i)
        if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            raise ValueError(err.getMessage().strip())
    conv = libsbml.SBMLLocalParameterConverter()
    if conv.setDocument(doc) != libsbml.LIBSBML_OPERATION_SUCCESS or conv.convert() != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise ValueError("could not convert local parameters to global ones")
    if not doc.expandInitialAssignments():
        raise ValueError("could not expand initial assignments")
    if not doc.expandFunctionDefinitions():
        raise ValueError("could not expand function definitions")
    model = doc.getModel()
    if model is None:
        raise ValueError("the document has no model")
    for i in range(model.getNumParameters()):
        p = model.getParameter(i)
        if not p.isSetValue():
            p.setValue(0.0)
    return doc, model


def to_string(doc) -> str:
    return libsbml.writeSBMLToString(doc)


def load_timecourse(sedml_path: str):
    """SciGym SBML.load_sedml_from_string_or_file: the uniform time course with the latest end time."""
    doc = libsedml.readSedMLFromFile(sedml_path)
    if doc.getNumErrors() > 0:
        raise ValueError(doc.getErrorLog().toString())
    best = None
    sims = doc.getListOfSimulations()
    for i in range(sims.size()):
        s = sims.get(i)
        if s.getTypeCode() != libsedml.SEDML_SIMULATION_UNIFORMTIMECOURSE:
            continue
        if best is None or s.getOutputEndTime() > best.getOutputEndTime():
            best = s
    if best is None:  # SciGym's fallback
        best = doc.createUniformTimeCourse()
        best.setInitialTime(0)
        best.setOutputStartTime(0)
        best.setOutputEndTime(10)
        best.setNumberOfSteps(51)
        best.createAlgorithm().setKisaoID("KISAO:0000019")
    return best.clone()


def species_ids(model) -> list[str]:
    """Floating species first, then boundary species (SciGym SBML.get_species_ids)."""
    floating, boundary = [], []
    for i in range(model.getNumSpecies()):
        s = model.getSpecies(i)
        (boundary if s.getBoundaryCondition() else floating).append(s.getId())
    return floating + boundary


def reaction_ids(model) -> list[str]:
    return [model.getReaction(i).getId() for i in range(model.getNumReactions())]


# ---------------------------------------------------------------------------------------------- simulation

def _roadrunner(sbml_text: str):
    import tellurium as te
    return te.loadSBMLModel(sbml_text)


def _setup_integrator(sim, rr):
    """scigym/data/simulator.py Simulator._setup_integrator (algorithm from SED-ML; CVODE if unsupported)."""
    from tellurium.sedml.tesedml import SEDMLCodeFactory, getKisaoStringFromVal, getKisaoValFromString
    alg = sim.getAlgorithm()
    if alg is None:
        alg = sim.createAlgorithm()
        alg.setKisaoID(getKisaoStringFromVal(19))
    kisao = getKisaoValFromString(alg.getKisaoID())
    if not SEDMLCodeFactory.isSupportedAlgorithmForSimulationType(kisao=kisao, simType=libsedml.SEDML_SIMULATION_UNIFORMTIMECOURSE):
        alg = sim.createAlgorithm()
        alg.setKisaoID(getKisaoStringFromVal(19))
        kisao = 19
    name = SEDMLCodeFactory.getIntegratorNameForKisaoID(kisao)
    if not name:
        raise RuntimeError(f"no integrator for KISAO {kisao}")
    rr.setIntegrator(name)
    if name == "gillespie":
        rr.integrator.setValue("variable_step_size", False)
    if kisao == 288:
        rr.integrator.setValue("stiff", True)
    elif kisao == 280:
        rr.integrator.setValue("stiff", False)
    for par in alg.getListOfAlgorithmParameters():
        try:
            pkey = SEDMLCodeFactory.algorithmParameterToParameterKey(par)
            if pkey and pkey.key != "conserved_moiety_analysis":
                rr.integrator.setValue(pkey.key, pkey.value)
        except Exception:
            pass
    if getattr(rr, "conservedMoietyAnalysis", False):
        rr.conservedMoietyAnalysis = False


def run(sbml_text: str, sim, observed: list[str] | None = None, brackets: bool = True):
    """scigym/data/simulator.py run_simulation: returns (time, {species: values}) for the observed species."""
    doc, model = load_sbml(sbml_text)
    ids = species_ids(model)
    observed = [s for s in (observed or ids) if s in ids]
    rr = _roadrunner(to_string(doc))
    _setup_integrator(sim, rr)
    rr.resetSeed()
    rr.resetToOrigin()
    cols = [f"[{s}]" for s in observed]
    rr.timeCourseSelections = ["time"] + cols
    t0, start, end = sim.getInitialTime(), sim.getOutputStartTime(), sim.getOutputEndTime()
    steps = sim.getNumberOfSteps()
    if abs(start - t0) > 1e-6:
        rr.simulate(start=t0, end=start, points=2)
    res = rr.simulate(start=start, end=end, steps=steps)
    names = list(res.colnames)
    data = np.array(res)
    out = {}
    for s, c in zip(observed, cols):
        if c in names:
            out[c if brackets else s] = data[:, names.index(c)].tolist()
    return data[:, 0].tolist(), out


def with_initial_concentrations(sbml_text: str, values: dict[str, float]) -> str:
    """SciGym SBML.change_initial_concentration via roadrunner, then the model as roadrunner writes it back."""
    rr = _roadrunner(sbml_text)
    _, model = load_sbml(sbml_text)
    for sid, v in values.items():
        s = model.getSpecies(sid)
        if s is None:
            raise ValueError(f"species {sid} not found")
        if s.getConstant():
            raise ValueError(f"cannot modify a constant species {sid}")
        rr.setInitConcentration(sid, float(v))  # roadrunner >= 2.9 has no forceRegenerate argument
    return str(rr.getCurrentSBML())


# ---------------------------------------------------------------------------------------------- metrics

def reaction_hashes(model):
    """scigym/eval/utils.py get_reaction_hash / extract_reaction_hashes: (with modifiers, without modifiers)."""
    rpm, rp = set(), set()
    for i in range(model.getNumReactions()):
        r = model.getReaction(i)
        re_ = frozenset(r.getReactant(j).getSpecies() for j in range(r.getNumReactants()))
        pr = frozenset(r.getProduct(j).getSpecies() for j in range(r.getNumProducts()))
        mo = frozenset(r.getModifier(j).getSpecies() for j in range(r.getNumModifiers()))
        rp.add((re_, pr))
        rpm.add((re_, pr, mo))
    return rpm, rp


def recovery(true_m, inco_m, pred_m) -> dict:
    """scigym/eval/evaluator.py evaluate_exact_reaction_recovery: precision/recall/F1 of the added reactions against
    the missing ones, lenient (reactants+products, 'rp') and strict (+modifiers, 'rpm')."""
    t_rpm, t_rp = reaction_hashes(true_m)
    i_rpm, i_rp = reaction_hashes(inco_m)
    p_rpm, p_rp = reaction_hashes(pred_m)
    out = {}
    for tag, t, i, p in (("rp", t_rp, i_rp, p_rp), ("rpm", t_rpm, i_rpm, p_rpm)):
        missing, added = t - i, p - i
        prec = rec = f1 = 0.0
        if missing and added:
            found = added & missing
            rec, prec = len(found) / len(missing), len(found) / len(added)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out |= {f"{tag}_precision": prec, f"{tag}_recall": rec, f"{tag}_f1": f1, f"{tag}_missing": len(missing),
                f"{tag}_added": len(added)}
    return out


def typed_edges(model) -> dict:
    """scigym/eval/utils.py extract_typed_species_edges."""
    e = {"reactant_product": set(), "reactant_modifier": set(), "modifier_product": set()}
    for i in range(model.getNumReactions()):
        r = model.getReaction(i)
        re_ = [r.getReactant(j).getSpecies() for j in range(r.getNumReactants())]
        pr = [r.getProduct(j).getSpecies() for j in range(r.getNumProducts())]
        mo = [r.getModifier(j).getSpecies() for j in range(r.getNumModifiers())]
        e["reactant_product"] |= {(a, b) for a in re_ for b in pr}
        e["reactant_modifier"] |= {(a, b) for a in re_ for b in mo}
        e["modifier_product"] |= {(a, b) for a in mo for b in pr}
    return e


def interaction_f1(true_m, pred_m) -> dict:
    """scigym/eval/utils.py evaluate_typed_species_interaction_f1 (None where the true model has no such edges)."""
    t, p = typed_edges(true_m), typed_edges(pred_m)
    out = {}
    for k in t:
        tp, fp, fn = len(t[k] & p[k]), len(p[k] - t[k]), len(t[k] - p[k])
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"{k}_f1"] = None if not t[k] else f1
    return out


def smape(a: dict, b: dict) -> float:
    """scigym/eval/utils.py compute_dict_smape: species aligned by sorted name; 1 if the shapes differ."""
    keys = sorted(set(a) | set(b))
    A = np.asarray([a[k] for k in keys if k in a], dtype=float)
    B = np.asarray([b[k] for k in keys if k in b], dtype=float)
    if A.shape != B.shape:
        return 1.0
    num, den = np.abs(A - B), np.abs(A) + np.abs(B)
    vals = np.zeros_like(A)
    nz = den != 0
    vals[nz] = num[nz] / den[nz]
    v = float(vals.sum() / A.size)
    return 1.0 if math.isnan(v) else v


def grade(true_xml: str, partial_xml: str, pred_xml: str, sedml: str, rounds: int = 10, noise: float = 0.2,
          seed: int = 0) -> dict:
    sim = load_timecourse(sedml)
    _, true_m = load_sbml(true_xml)
    _, inco_m = load_sbml(partial_xml)
    try:
        _, pred_m = load_sbml(pred_xml)
    except Exception as exc:
        return {"valid": False, "error": f"unreadable SBML: {exc}"[:300]}
    m = {"valid": True}
    m |= recovery(true_m, inco_m, pred_m)
    m |= interaction_f1(true_m, pred_m)
    try:
        _, true_traj = run(true_xml, sim)
        _, pred_traj = run(pred_xml, sim)
        m["observe_smape"] = smape(true_traj, pred_traj)
    except Exception as exc:
        m["observe_smape"] = 1.0
        m["simulation_error"] = str(exc)[:300]
    # Perturbed initial conditions (SBML.eval_add_noise_to_initial_concentrations): the same proportional noise on the
    # incomplete model's non-constant, non-boundary species, applied to the true and the predicted model. SciGym's
    # evaluate_smape_noise compares against the UNperturbed true trajectory; we compare perturbed with perturbed.
    rng = np.random.default_rng(seed)
    floating = []
    for i in range(inco_m.getNumSpecies()):
        s = inco_m.getSpecies(i)
        if not (s.getConstant() or s.getBoundaryCondition()):
            floating.append(s.getId())
    try:
        base = _roadrunner(partial_xml).getFloatingSpeciesConcentrationsNamedArray()
        init = dict(zip(base.colnames, base.tolist()[0]))
    except Exception:
        init = {}
    vals = []
    for _ in range(rounds if init else 0):
        new = {}
        for sid in floating:
            c = float(init.get(sid, 0.0))
            new[sid] = max(c + rng.normal(0, noise * max(c, 1e-10)), 0.0)  # utils/main.py perturb_concentration_proportional
        try:
            _, tt = run(with_initial_concentrations(true_xml, new), sim)
        except Exception:
            continue
        try:
            _, pt = run(with_initial_concentrations(pred_xml, {k: v for k, v in new.items() if pred_m.getSpecies(k)}), sim)
            vals.append(smape(tt, pt))
        except Exception:
            vals.append(1.0)
    m["perturb_smape"] = float(np.mean(vals)) if vals else None
    m["perturb_rounds"] = len(vals)
    return m


# ---------------------------------------------------------------------------------------------- for agent code

_SEDML = os.environ.get("SCIGYM_SEDML", "")


def simulate(sbml_string: str):
    """Simulates an SBML model and returns a pandas DataFrame with column 'Time' and one column per species."""
    import pandas as pd
    if not _SEDML:
        raise RuntimeError("simulate() is only available inside a SciGym task workspace")
    time, data = run(sbml_string, load_timecourse(_SEDML), brackets=False)
    return pd.DataFrame({**data, "Time": time})


# ---------------------------------------------------------------------------------------------- JSON commands

def _describe(d: str) -> dict:
    true_xml = open(os.path.join(d, "truth.xml")).read()
    partial_xml = open(os.path.join(d, "partial.xml")).read()
    sim = load_timecourse(os.path.join(d, "truth.sedml"))
    tdoc, true_m = load_sbml(true_xml)
    pdoc, inco_m = load_sbml(partial_xml)
    rr = _roadrunner(partial_xml)
    conc = {}
    for arr in (rr.getFloatingSpeciesConcentrationsNamedArray(), rr.getBoundarySpeciesConcentrationsNamedArray()):
        conc |= dict(zip(arr.colnames, [float(x) for x in arr.tolist()[0]])) if arr.size else {}
    species = []
    for sid in species_ids(inco_m):
        s = inco_m.getSpecies(sid)
        species.append({"id": sid, "init": conc.get(sid), "constant": bool(s.getConstant()), "boundary": bool(s.getBoundaryCondition())})
    _, t_rp = reaction_hashes(true_m)
    _, i_rp = reaction_hashes(inco_m)
    return {"partial": to_string(pdoc), "species": species, "n_true_reactions": true_m.getNumReactions(),
            "n_partial_reactions": inco_m.getNumReactions(), "n_missing": len(t_rp - i_rp),
            "n_true_species": true_m.getNumSpecies(), "end_time": sim.getOutputEndTime(), "steps": sim.getNumberOfSteps()}


def handle(req: dict) -> dict:
    op = req["op"]
    if op == "describe":  # index the systems in a dataset folder
        out = {}
        for d in req["dirs"]:
            try:
                out[d] = _describe(d)
            except Exception as exc:
                out[d] = {"error": str(exc)[:300]}
        return out
    if op == "experiment":
        d = req["dir"]
        true_xml = open(os.path.join(d, "truth.xml")).read()
        _, inco_m = load_sbml(open(os.path.join(d, "partial.xml")).read())
        allowed = set(species_ids(inco_m))
        conc = req.get("concentrations") or {}
        bad = [s for s in conc if s not in allowed]
        if bad:
            return {"ok": False, "error": f"unknown species {bad}; species are {sorted(allowed)}"}
        for s, v in conc.items():
            if not isinstance(v, (int, float)) or v < 0 or math.isnan(v):
                return {"ok": False, "error": f"concentration of {s} must be a non-negative number"}
        try:
            xml = with_initial_concentrations(true_xml, conc) if conc else true_xml
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            time, data = run(xml, load_timecourse(os.path.join(d, "truth.sedml")), observed=sorted(allowed), brackets=False)
        except Exception:
            return {"ok": False, "error": "the experiment could not be run with these settings; make sure it is sensible"}
        import pandas as pd
        df = pd.DataFrame({**data, "Time": time})
        df.to_csv(req["out"], index=False, float_format="%.6g")
        return {"ok": True, "rows": len(df), "cols": list(df.columns)}
    if op == "check":  # can the submitted model be read and simulated with the task's time course?
        try:
            _, m = load_sbml(req["sbml"])
            run(req["sbml"], load_timecourse(req["sedml"]))
            return {"ok": True, "species": m.getNumSpecies(), "reactions": m.getNumReactions()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:400]}
    if op == "grade":
        d = req["dir"]
        return grade(open(os.path.join(d, "truth.xml")).read(), open(os.path.join(d, "partial.xml")).read(),
                     req["sbml"], os.path.join(d, "truth.sedml"), req.get("rounds", 10), req.get("noise", 0.2), req.get("seed", 0))
    raise ValueError(f"unknown op {op}")


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        result = handle(request)
    except Exception as exc:
        result = {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"[:500]}
    sys.stdout.write("\n@@RESULT@@" + json.dumps(result) + "\n")
    sys.stdout.flush()
