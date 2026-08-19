"""
Solver-statistics recorder.

Living in the shared core rather than in either platform layer is deliberate:
the numbers reported for the aerial and the legged robot are then produced by
the same code path, which is what makes a cross-platform claim admissible.

Everything IPOPT exposes through `sol.stats()` is captured, in particular the
per-callback timings (`t_proc_nlp_f`, `..._grad_f`, `..._jac_g`, `..._hess_l`)
that quantify how much of a solve is spent evaluating derivatives — the
practical counterpart of the cost analysis of Ch. 5 of the course notes.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SolveRecord:
    step: int
    success: bool
    status: str
    iterations: int
    cost: float
    solve_ms: float
    total_ms: float
    build_ms: float
    rebuilt: bool
    t_nlp_f_ms: float = 0.0
    t_nlp_grad_f_ms: float = 0.0
    t_nlp_jac_g_ms: float = 0.0
    t_nlp_hess_l_ms: float = 0.0
    extra: dict = field(default_factory=dict)


def _ms(stats: dict, key: str) -> float:
    return float(stats.get(key, 0.0)) * 1e3


class SolveRecorder:
    """Accumulates per-solve records and summarises them."""

    def __init__(self, label: str, deadline_ms: float | None = None):
        self.label = label
        self.deadline_ms = deadline_ms
        self.records: list[SolveRecord] = []

    # ------------------------------------------------------------------
    def add(self, result, step: int | None = None, **extra) -> SolveRecord:
        st = result.stats or {}
        rec = SolveRecord(
            step=len(self.records) if step is None else step,
            success=bool(result.success),
            status=result.status,
            iterations=int(result.iterations),
            cost=float(result.cost),
            solve_ms=float(result.solve_ms),
            total_ms=float(result.total_ms),
            build_ms=float(result.build_ms),
            rebuilt=bool(result.rebuilt),
            t_nlp_f_ms=_ms(st, "t_proc_nlp_f"),
            t_nlp_grad_f_ms=_ms(st, "t_proc_nlp_grad_f"),
            t_nlp_jac_g_ms=_ms(st, "t_proc_nlp_jac_g"),
            t_nlp_hess_l_ms=_ms(st, "t_proc_nlp_hess_l"),
            extra=dict(extra),
        )
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------------
    @staticmethod
    def _pct(values, q: float) -> float:
        if not values:
            return float("nan")
        s = sorted(values)
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[idx]

    def summary(self) -> dict:
        """Aggregate statistics.

        Solve time is reported as a distribution, not as a mean: what decides
        whether the controller is admissible in real time is the tail, so p95,
        max and the deadline-miss rate are the quantities that matter.
        """
        if not self.records:
            return {"label": self.label, "n": 0}

        t = [r.total_ms for r in self.records]
        s = [r.solve_ms for r in self.records]
        it = [r.iterations for r in self.records if r.iterations >= 0]
        ok = [r for r in self.records if r.success]

        out = {
            "label": self.label,
            "n": len(self.records),
            "success_rate": len(ok) / len(self.records),
            "iter_mean": statistics.fmean(it) if it else float("nan"),
            "iter_max": max(it) if it else -1,
            "solve_ms_mean": statistics.fmean(s),
            "solve_ms_p50": self._pct(s, 0.50),
            "solve_ms_p95": self._pct(s, 0.95),
            "solve_ms_max": max(s),
            "total_ms_mean": statistics.fmean(t),
            "total_ms_p95": self._pct(t, 0.95),
            "total_ms_max": max(t),
            "build_ms_total": sum(r.build_ms for r in self.records),
            "n_rebuilds": sum(1 for r in self.records if r.rebuilt),
            "cost_mean": statistics.fmean([r.cost for r in self.records if r.cost < float("inf")] or [float("nan")]),
        }
        if self.deadline_ms is not None:
            out["deadline_ms"] = self.deadline_ms
            out["deadline_miss_rate"] = sum(1 for v in t if v > self.deadline_ms) / len(t)

        # share of solve time spent inside derivative callbacks
        tot = sum(s) or 1.0
        out["frac_f"] = sum(r.t_nlp_f_ms for r in self.records) / tot
        out["frac_grad_f"] = sum(r.t_nlp_grad_f_ms for r in self.records) / tot
        out["frac_jac_g"] = sum(r.t_nlp_jac_g_ms for r in self.records) / tot
        out["frac_hess_l"] = sum(r.t_nlp_hess_l_ms for r in self.records) / tot

        statuses: dict[str, int] = {}
        for r in self.records:
            statuses[r.status] = statuses.get(r.status, 0) + 1
        out["status_counts"] = statuses
        return out

    # ------------------------------------------------------------------
    def to_csv(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(r) for r in self.records]
        for row in rows:
            row["extra"] = json.dumps(row["extra"])
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["step"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def to_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True))
        return path

    def __len__(self) -> int:
        return len(self.records)


def compare(summaries, keys=("solve_ms_mean", "solve_ms_p95", "iter_mean", "success_rate")):
    """Render a list of summaries as a markdown table, ready for the report."""
    header = "| variant | " + " | ".join(keys) + " |"
    sep = "|" + "---|" * (len(keys) + 1)
    lines = [header, sep]
    for s in summaries:
        cells = []
        for k in keys:
            v = s.get(k, float("nan"))
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append(f"| {s.get('label', '?')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
