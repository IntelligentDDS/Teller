"""GMM-based candidate localization and compact trace slicing.

This module implements the lightweight triage stage described in the paper's
candidate-localization section. In practice, we found that TPE already gives
the RCA model a strong compact trace representation, and adding this
localization stage brings only limited additional gains. Therefore, this module
can be treated as optional in practical reproduction or deployment: the rest of
the TELLER pipeline can still achieve sufficiently strong results without
running this stage.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover
    GaussianMixture = None


_NOISE = re.compile(r",\s*(op_id|seq)\s*=\s*\d+", re.IGNORECASE)
_AT_LOC = re.compile(r"\s+at\s+.*$")
_COMM_HINTS = ("nccl", "allreduce", "all_reduce", "reduce_scatter", "all_gather", "broadcast")


@dataclass
class CandidateCfg:
    k: int = 2
    q: float = 0.25
    threshold: float = 0.65
    top_k: int = 8
    center_mass: float = 0.7
    sample_cap: int = 5000
    random_state: int = 42
    beta: float = 0.35
    gamma: float = 0.25
    tau_ms: float = 2.0
    kappa: float = 0.25
    ancestor_hops: int = 2
    max_slice_nodes: int = 80


@dataclass
class Node:
    node_id: int
    step: int
    parent: int
    kind: str
    name: str
    start: int
    end: int
    dur_ns: int
    rt_ns: int = 0
    drv_ns: int = 0
    ker_ns: int = 0
    rt_count: int = 0
    drv_count: int = 0
    ker_count: int = 0
    bytes_moved: float = 0.0
    comm: bool = False

    @property
    def family(self) -> str:
        return f"{self.kind}:{norm_name(self.name)}"

    def feat(self) -> list[float]:
        return [
            self.dur_ns / 1e6,
            self.rt_ns / 1e6,
            self.drv_ns / 1e6,
            self.ker_ns / 1e6,
            float(self.rt_count),
            float(self.drv_count),
            float(self.ker_count),
            math.log1p(max(self.bytes_moved, 0.0)),
            1.0 if self.comm else 0.0,
        ]


@dataclass
class FamModel:
    gmm: Any
    center_ids: list[int]
    median: Any
    scale: Any


class DiagGmm:
    def __init__(self, weights: list[float], means: list[list[float]], vars_: list[list[float]]):
        self.weights_ = weights
        self.means_ = means
        self.vars_ = vars_

    @classmethod
    def fit(cls, rows: list[list[float]], k: int, seed: int, rounds: int = 25) -> "DiagGmm":
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        dim = len(rows[0])
        order = sorted(range(len(rows)), key=lambda i: sum(rows[i]))
        if k == 1:
            means = [[sum(r[j] for r in rows) / len(rows) for j in range(dim)]]
        else:
            means = [rows[order[0]][:], rows[order[-1]][:]]
            for _ in range(2, k):
                means.append(rows[rng.randrange(len(rows))][:])
        weights = [1.0 / k] * k
        vars_ = [[1.0] * dim for _ in range(k)]

        for _ in range(rounds):
            resp = cls(weights, means, vars_).predict_proba(rows)
            mass = [sum(r[c] for r in resp) for c in range(k)]
            for c in range(k):
                denom = max(mass[c], 1e-12)
                weights[c] = denom / len(rows)
                for j in range(dim):
                    means[c][j] = sum(resp[i][c] * rows[i][j] for i in range(len(rows))) / denom
                for j in range(dim):
                    var = sum(resp[i][c] * (rows[i][j] - means[c][j]) ** 2 for i in range(len(rows))) / denom
                    vars_[c][j] = max(var, 1e-6)
        return cls(weights, means, vars_)

    def predict_proba(self, rows: Any) -> list[list[float]]:
        out: list[list[float]] = []
        for row in rows:
            logs = []
            for w, mean, var in zip(self.weights_, self.means_, self.vars_):
                val = math.log(max(w, 1e-12))
                for x, m, v in zip(row, mean, var):
                    val += -0.5 * (math.log(2.0 * math.pi * v) + ((float(x) - m) ** 2) / v)
                logs.append(val)
            top = max(logs)
            exps = [math.exp(v - top) for v in logs]
            denom = sum(exps) or 1.0
            out.append([v / denom for v in exps])
        return out


def norm_name(name: str) -> str:
    s = str(name or "unknown").strip()
    s = _AT_LOC.sub("", s)
    s = _NOISE.sub("", s)
    return s.rstrip(" ,") or "unknown"


def _duration(obj: dict[str, Any]) -> int:
    if "start" in obj and "end" in obj:
        return max(0, int(obj.get("end") or 0) - int(obj.get("start") or 0))
    return max(0, int(obj.get("duration") or 0))


def _start_end(obj: dict[str, Any]) -> tuple[int, int]:
    if "start" in obj and "end" in obj:
        start = int(obj.get("start") or 0)
        end = int(obj.get("end") or start)
        return start, max(start, end)
    start = int(obj.get("gpu_start") or 0)
    return start, start + _duration(obj)


def _bytes(obj: dict[str, Any]) -> float:
    total = 0.0
    for key, val in obj.items():
        low = str(key).lower()
        if any(token in low for token in ("byte", "size", "bytes")) and isinstance(val, (int, float)):
            total += float(val)
    return total


def _is_comm(name: str) -> bool:
    low = str(name or "").lower()
    return any(hint in low for hint in _COMM_HINTS)


def flatten_trace(trace: dict[str, Any]) -> list[Node]:
    nodes: list[Node] = []
    children: dict[int, list[int]] = defaultdict(list)

    def add(step: int, parent: int, kind: str, obj: dict[str, Any], name_key: str = "op_name") -> int:
        start, end = _start_end(obj)
        name = obj.get(name_key) or obj.get("name") or kind.lower()
        node = Node(
            node_id=len(nodes),
            step=step,
            parent=parent,
            kind=kind,
            name=norm_name(name),
            start=start,
            end=end,
            dur_ns=max(0, end - start),
            bytes_moved=_bytes(obj),
            comm=_is_comm(str(name)),
        )
        nodes.append(node)
        if parent >= 0:
            children[parent].append(node.node_id)
        return node.node_id

    for step_i, step in enumerate(trace.get("steps") or []):
        sid = add(step_i, -1, "STEP", step, "step_name")
        for fe in step.get("torch_frontend_ops") or []:
            fid = add(step_i, sid, "FE", fe)
            for be in fe.get("torch_backend_ops") or []:
                bid = add(step_i, fid, "BE", be)
                for rt in be.get("runtime_calls") or []:
                    rid = add(step_i, bid, "RT", rt, "name")
                    for ker in rt.get("kernels") or []:
                        add(step_i, rid, "KERNEL", ker, "name")
                for drv in be.get("driver_calls") or []:
                    did = add(step_i, bid, "DRIVER", drv, "name")
                    for ker in drv.get("kernels") or []:
                        add(step_i, did, "KERNEL", ker, "name")

    for node in reversed(nodes):
        for child_id in children.get(node.node_id, []):
            child = nodes[child_id]
            node.rt_ns += child.rt_ns + (child.dur_ns if child.kind == "RT" else 0)
            node.drv_ns += child.drv_ns + (child.dur_ns if child.kind == "DRIVER" else 0)
            node.ker_ns += child.ker_ns + (child.dur_ns if child.kind == "KERNEL" else 0)
            node.rt_count += child.rt_count + (1 if child.kind == "RT" else 0)
            node.drv_count += child.drv_count + (1 if child.kind == "DRIVER" else 0)
            node.ker_count += child.ker_count + (1 if child.kind == "KERNEL" else 0)
            node.bytes_moved += child.bytes_moved
            node.comm = node.comm or child.comm
    return nodes


def _percentile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _prep_py(rows: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    x = [[math.log1p(max(float(v), 0.0)) for v in row] for row in rows]
    dim = len(x[0])
    median = [_percentile([row[j] for row in x], 50) for j in range(dim)]
    scale = []
    for j in range(dim):
        col = [row[j] for row in x]
        val = _percentile(col, 75) - _percentile(col, 25)
        scale.append(val if val >= 1e-9 else 1.0)
    z = [[(row[j] - median[j]) / scale[j] for j in range(dim)] for row in x]
    return z, median, scale


def _prep_matrix(rows: list[list[float]]) -> tuple[Any, Any, Any]:
    if np is None:
        return _prep_py(rows)
    x = np.asarray(rows, dtype=float)
    x = np.log1p(np.maximum(x, 0.0))
    median = np.median(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    scale = q75 - q25
    scale[scale < 1e-9] = 1.0
    return (x - median) / scale, median, scale


def _standardize(rows: list[list[float]], model: FamModel) -> Any:
    if np is None:
        x = [[math.log1p(max(float(v), 0.0)) for v in row] for row in rows]
        return [
            [(row[j] - model.median[j]) / model.scale[j] for j in range(len(row))]
            for row in x
        ]
    x = np.asarray(rows, dtype=float)
    x = np.log1p(np.maximum(x, 0.0))
    return (x - model.median) / model.scale


def _center_ids(weights: Any, mass: float) -> list[int]:
    if np is None:
        order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        picked: list[int] = []
        total = 0.0
        for idx in order:
            picked.append(int(idx))
            total += float(weights[idx])
            if total >= mass:
                break
        return picked or [max(range(len(weights)), key=lambda i: weights[i])]
    order = np.argsort(weights)[::-1]
    picked: list[int] = []
    total = 0.0
    for idx in order:
        picked.append(int(idx))
        total += float(weights[idx])
        if total >= mass:
            break
    return picked or [int(np.argmax(weights))]


def fit_models(nodes: list[Node], cfg: CandidateCfg) -> dict[str, FamModel]:
    by_family: dict[str, list[list[float]]] = defaultdict(list)
    for node in nodes:
        if node.kind == "STEP":
            continue
        by_family[node.family].append(node.feat())

    rng = np.random.default_rng(cfg.random_state) if np is not None else random.Random(cfg.random_state)
    models: dict[str, FamModel] = {}
    for family, rows in by_family.items():
        if len(rows) < 2:
            continue
        if cfg.sample_cap and len(rows) > cfg.sample_cap:
            if np is not None:
                idx = rng.choice(len(rows), size=cfg.sample_cap, replace=False)
                rows = [rows[int(i)] for i in idx]
            else:
                rows = rng.sample(rows, cfg.sample_cap)
        x, median, scale = _prep_matrix(rows)
        unique_count = len(np.unique(x, axis=0)) if np is not None else len({tuple(r) for r in x})
        k = max(1, min(cfg.k, len(rows), unique_count))
        if GaussianMixture is not None and np is not None:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-6,
                random_state=cfg.random_state,
            )
            gmm.fit(x)
        else:
            gmm = DiagGmm.fit(x, k=k, seed=cfg.random_state)
        models[family] = FamModel(gmm=gmm, center_ids=_center_ids(gmm.weights_, cfg.center_mass), median=median, scale=scale)
    return models


def _rho(nodes: list[Node], models: dict[str, FamModel]) -> dict[int, float]:
    grouped: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        if node.kind != "STEP" and node.family in models:
            grouped[node.family].append(node)

    out: dict[int, float] = {}
    for family, items in grouped.items():
        model = models[family]
        x = _standardize([n.feat() for n in items], model)
        prob = model.gmm.predict_proba(x)
        if np is not None and not isinstance(prob, list):
            center_prob = prob[:, model.center_ids].sum(axis=1)
        else:
            center_prob = [sum(row[i] for i in model.center_ids) for row in prob]
        for node, val in zip(items, center_prob):
            out[node.node_id] = float(val)
    return out


def _pick_candidates(nodes: list[Node], scores: dict[int, float], cfg: CandidateCfg) -> list[dict[str, Any]]:
    by_step_family: dict[tuple[int, str], list[Node]] = defaultdict(list)
    for node in nodes:
        if node.node_id in scores:
            by_step_family[(node.step, node.family)].append(node)

    cand: list[dict[str, Any]] = []
    for (step, family), items in by_step_family.items():
        vals = sorted(scores[n.node_id] for n in items)
        take = max(1, int(math.floor(len(vals) * cfg.q)))
        suspiciousness = 1.0 - (sum(vals[:take]) / take)
        if suspiciousness < cfg.threshold:
            continue
        ranked = sorted(items, key=lambda n: scores[n.node_id])[:take]
        cand.append(
            {
                "step": step,
                "family": family,
                "score": suspiciousness,
                "nodes": [n.node_id for n in ranked],
                "names": sorted({n.name for n in ranked})[:5],
            }
        )
    cand.sort(key=lambda x: (-x["score"], x["step"], x["family"]))
    return cand[: cfg.top_k]


def _slice(nodes: list[Node], cand: dict[str, Any], scores: dict[int, float], cfg: CandidateCfg) -> dict[str, Any]:
    node_map = {n.node_id: n for n in nodes}
    step_nodes = [n for n in nodes if n.step == cand["step"]]
    chosen: set[int] = set(cand["nodes"])
    rel: dict[int, float] = defaultdict(float)
    for nid in chosen:
        rel[nid] = 1.0

    by_parent: dict[int, list[int]] = defaultdict(list)
    for node in step_nodes:
        if node.parent >= 0:
            by_parent[node.parent].append(node.node_id)

    for nid in list(chosen):
        cur = node_map[nid]
        rel[cur.parent] = max(rel[cur.parent], cfg.beta) if cur.parent >= 0 else rel[cur.parent]
        for child in by_parent.get(nid, []):
            rel[child] = max(rel[child], cfg.beta)
        for other in step_nodes:
            if other.node_id == nid:
                continue
            dt_ns = max(0, max(other.start - cur.end, cur.start - other.end))
            rel[other.node_id] = max(rel[other.node_id], cfg.gamma * math.exp(-(dt_ns / 1e6) / cfg.tau_ms))
            if cur.comm and other.comm:
                rel[other.node_id] = max(rel[other.node_id], cfg.beta)

        parent = cur.parent
        hops = 0
        while parent >= 0 and hops < cfg.ancestor_hops:
            chosen.add(parent)
            rel[parent] = max(rel[parent], cfg.beta)
            parent = node_map[parent].parent
            hops += 1

    chosen |= {nid for nid, val in rel.items() if val >= cfg.kappa}
    ranked = sorted(chosen, key=lambda nid: (-rel.get(nid, 0.0), node_map[nid].start, nid))
    keep = set(ranked[: cfg.max_slice_nodes])

    out_nodes = []
    for nid in sorted(keep, key=lambda x: (node_map[x].start, x)):
        node = node_map[nid]
        out_nodes.append(
            {
                "id": nid,
                "parent": node.parent if node.parent in keep else -1,
                "kind": node.kind,
                "name": node.name,
                "family": node.family,
                "duration_ms": round(node.dur_ns / 1e6, 6),
                "rho": round(scores.get(nid, 1.0), 6),
                "relevance": round(rel.get(nid, 0.0), 6),
            }
        )
    edges = [{"src": n.parent, "dst": n.node_id} for n in step_nodes if n.node_id in keep and n.parent in keep]
    return {"step": cand["step"], "candidate": cand, "nodes": out_nodes, "edges": edges}


def localize_trace(trace: dict[str, Any], models: dict[str, FamModel], cfg: CandidateCfg | None = None) -> dict[str, Any]:
    cfg = cfg or CandidateCfg()
    nodes = flatten_trace(trace)
    scores = _rho(nodes, models)
    candidates = _pick_candidates(nodes, scores, cfg)
    return {
        "trace_id": trace.get("trace_id"),
        "request_id": trace.get("request_id"),
        "engine": trace.get("engine"),
        "candidates": candidates,
        "slices": [_slice(nodes, cand, scores, cfg) for cand in candidates],
    }
