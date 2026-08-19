"""
The platform-agnostic trajectory optimal control problem.

    min   sum_{k=0}^{N-1} [ ||x_k - x_ref_k||_Q^2 + ||u_k||_R^2
                            + r_jerk ||u_k - u_{k-1}||^2 + J_obs(p_k) ]
          + ||x_N - x_ref_N||_{Q_T}^2 + J_obs(p_N) + J_slack
    s.t.  x_{k+1} = f_Sigma(x_k, u_k)          <- platform
          x_0     = x_hat(t)
          u_k in U_Sigma                       <- platform
          x_k in X_Sigma                       <- platform
          (optional) terminal equilibrium constraint

Only the three lines marked "platform" depend on the robot; they are supplied by
a :class:`~trajopt_core.models.base.MotionModel`.  The same class instance
therefore drives both the aerial and the legged platform, which is the operative
form of the claim made in the report.

Build-once (parametric) formulation
-----------------------------------
The CasADi graph is constructed a single time.  Everything that changes between
control cycles — initial state, reference trajectory, occupancy grid, selected
obstacle points and their normals — enters through `Opti.parameter` objects and
is refreshed with `set_value`.  The sparsity pattern of the NLP is therefore
constant, which is exactly what lets IPOPT reuse its symbolic factorisation
structure.  `build_count` is exposed so the report can state the number rather
than assert it.

Multiple shooting
-----------------
Both the states and the inputs are decision variables, with the dynamics imposed
as equality constraints (Sec. 7.2.2 of the course notes).  Compared with a
condensed single-shooting formulation this trades a larger variable count for a
block-banded KKT matrix and for the guarantee that the model is never integrated
in open loop for more than one step.  `structure()` reports the resulting
sparsity so the trade-off can be quantified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from trajopt_core.mpc.config import OCPConfig
from trajopt_core.mpc.obstacles import ObstacleContext
from trajopt_core.mpc.reference import PathReference, build_path_reference


@dataclass
class OCPResult:
    """Outcome of a single solve."""

    success: bool
    x_pred: np.ndarray            # (N+1, NX)
    u_opt: np.ndarray             # (N, NU)
    x_ref: np.ndarray             # (N+1, NX)
    cost: float
    solve_ms: float               # wall time of Opti.solve()
    total_ms: float               # wall time of TrajectoryOCP.solve()
    build_ms: float               # graph construction time inside this call (0 if cached)
    iterations: int
    status: str
    rebuilt: bool
    stats: dict = field(default_factory=dict)

    @property
    def u0(self) -> np.ndarray:
        return self.u_opt[0]

    def planar_trajectory(self, model) -> np.ndarray:
        i, j = model.PLANAR_IDX
        return self.x_pred[:, [i, j]]


class TrajectoryOCP:
    """Receding-horizon trajectory optimiser, parametric in the perception input."""

    def __init__(self, model, cfg: OCPConfig | None = None, obstacle_terms=None):
        self.model = model
        self.cfg = cfg or OCPConfig()
        self.terms = list(obstacle_terms or [])

        # propagate the limits from the config to the model if it has none of its own
        if self.cfg.limits is not None:
            self.model.limits = self.cfg.limits

        self._opti = None
        self._X = None
        self._U = None
        self._p_x0 = None
        self._p_xref = None
        self._cost_expr = None

        self.build_count = 0
        self._prev_x = None       # (N+1, NX) warm start
        self._prev_u = None       # (N, NU)
        self._last_valid_x0 = None
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def requires_rebuild(self) -> bool:
        """True if any obstacle term bakes data into the graph."""
        return any(t.requires_rebuild for t in self.terms)

    @property
    def is_parametric(self) -> bool:
        return not self.requires_rebuild

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build(self) -> float:
        t0 = time.perf_counter()
        model, cfg = self.model, self.cfg
        N, dt = cfg.N, cfg.dt
        nx, nu = model.NX, model.NU

        opti = ca.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        p_x0 = opti.parameter(nx)
        p_xref = opti.parameter(nx, N + 1)

        for term in self.terms:
            term.declare(opti, N, model)

        q = model.state_weight_vector(cfg.weights)
        r = model.input_weight_vector(cfg.weights)
        Q = np.diag(q)
        Q_T = np.diag(q * cfg.weights.q_terminal)
        R = np.diag(r)

        i, j = model.PLANAR_IDX

        cost = 0
        for k in range(N):
            e = X[:, k] - p_xref[:, k]
            cost = cost + ca.mtimes([e.T, Q, e])

            uk = U[:, k]
            cost = cost + ca.mtimes([uk.T, R, uk])

            if k > 0:
                du = U[:, k] - U[:, k - 1]
                cost = cost + cfg.weights.r_jerk * ca.dot(du, du)

            p_k = ca.vertcat(X[i, k], X[j, k])
            for term in self.terms:
                cost = cost + term.stage_cost(k, p_k)
                term.add_constraints(opti, k, p_k)

        # terminal stage
        e_T = X[:, N] - p_xref[:, N]
        cost = cost + ca.mtimes([e_T.T, Q_T, e_T])
        p_N = ca.vertcat(X[i, N], X[j, N])
        for term in self.terms:
            cost = cost + term.stage_cost(N, p_N)
            term.add_constraints(opti, N, p_N)
            cost = cost + term.global_cost()

        opti.minimize(cost)

        # --- dynamics (multiple shooting) --------------------------------
        for k in range(N):
            x_next = ca.vertcat(*model.step(X[:, k], U[:, k], dt))
            opti.subject_to(X[:, k + 1] == x_next)

        # --- initial condition -------------------------------------------
        opti.subject_to(X[:, 0] == p_x0)

        # --- input box ----------------------------------------------------
        lb, ub = model.input_bounds()
        for k in range(N):
            for m in range(nu):
                opti.subject_to(opti.bounded(lb[m], U[m, k], ub[m]))

        # --- platform state constraints -----------------------------------
        # Imposed from k = 1 onwards only.  The initial column is pinned by the
        # equality X[:,0] == x0, so it is not a degree of freedom: adding a state
        # constraint there cannot influence the solution, but it CAN render the
        # whole problem infeasible whenever the measured state momentarily sits
        # outside the admissible set — which is routine, since the estimate comes
        # from a real inner loop that overshoots.  Dropping it is the cheapest
        # correct way to keep the feasibility set of the FHOCP from collapsing
        # for reasons that have nothing to do with the decision variables.
        model.add_state_constraints(opti, X[:, 1:])

        # --- optional terminal equilibrium --------------------------------
        vel_idx = getattr(model, "VEL_IDX", None)
        if cfg.terminal_zero_velocity and vel_idx:
            for v in vel_idx:
                opti.subject_to(X[v, N] == 0.0)

        p_opts, s_opts = cfg.solver.casadi_options()
        try:
            opti.solver("ipopt", p_opts, s_opts)
        except Exception:                                   # pragma: no cover
            p_opts = dict(p_opts, expand=False)
            opti.solver("ipopt", p_opts, s_opts)

        self._opti, self._X, self._U = opti, X, U
        self._p_x0, self._p_xref = p_x0, p_xref
        self._cost_expr = cost
        self.build_count += 1

        # a rebuild invalidates the warm-start buffers (different graph objects)
        self._prev_x = None
        self._prev_u = None

        return (time.perf_counter() - t0) * 1e3

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(
        self,
        state,
        path,
        points_xy=None,
        grid=None,
        z_ref: float | None = None,
    ) -> OCPResult:
        """
        Run one receding-horizon solve.

        Parameters
        ----------
        state     : (NX,) current state estimate
        path      : list of (x, y) or (x, y, z) A* waypoints, world frame
        points_xy : (M, 2) LiDAR hits used by the point-based obstacle terms
        grid      : FixedGaussianGridMap used by the grid obstacle term
        z_ref     : overrides cfg.z_ref for this solve
        """
        t_start = time.perf_counter()
        model, cfg = self.model, self.cfg
        N, nx, nu = cfg.N, model.NX, model.NU

        # --- sanitise the initial state ----------------------------------
        x0 = np.asarray(state, dtype=float).ravel()
        if x0.size != nx:
            raise ValueError(f"expected a state of length {nx}, got {x0.size}")
        if not np.isfinite(x0).all():
            x0 = (
                self._last_valid_x0.copy()
                if self._last_valid_x0 is not None
                else np.zeros(nx)
            )
        self._last_valid_x0 = x0.copy()

        # --- reference ----------------------------------------------------
        ref: PathReference = build_path_reference(
            model=model,
            state=x0,
            path=path,
            N=N,
            dt=cfg.dt,
            v_ref=cfg.v_ref,
            z_ref=cfg.z_ref if z_ref is None else z_ref,
            anchor_first_to_state=cfg.anchor_first_to_state,
            resample_ds=cfg.path_resample_ds,
            smooth_window=cfg.path_smooth_window,
        )

        # --- linearisation trajectory for the SCA half-spaces -------------
        i, j = model.PLANAR_IDX
        if self._prev_x is not None:
            lin_traj = self._prev_x[:, [i, j]]
        else:
            lin_traj = ref.p_ref

        ctx = ObstacleContext(
            robot_xy=np.array([x0[i], x0[j]], dtype=float),
            points_xy=None if points_xy is None else np.asarray(points_xy, dtype=float),
            grid=grid,
            lin_traj=lin_traj,
        )

        # --- build (once, unless a term bakes data into the graph) --------
        build_ms = 0.0
        rebuilt = False
        if self._opti is None or self.requires_rebuild:
            for term in self.terms:
                term.prepare(ctx)
            build_ms = self._build()
            rebuilt = True

        opti = self._opti

        # --- push parameter values ----------------------------------------
        opti.set_value(self._p_x0, x0)
        opti.set_value(self._p_xref, ref.x_ref.T)
        for term in self.terms:
            term.set_values(opti, ctx)

        # --- warm start ----------------------------------------------------
        if cfg.solver.warm_start and self._prev_x is not None and self._prev_u is not None:
            opti.set_initial(self._X, self._prev_x.T)
            opti.set_initial(self._U, self._prev_u.T)
        elif cfg.solver.cold_start:
            opti.set_initial(self._X, np.zeros((nx, N + 1)))
            opti.set_initial(self._U, np.zeros((nu, N)))
        else:
            # The reference trajectory is computed anyway and is already a very
            # good guess, so it — not zero — is the meaningful baseline against
            # which warm starting has to justify itself.
            opti.set_initial(self._X, ref.x_ref.T)
            opti.set_initial(self._U, np.zeros((nu, N)))

        # --- solve, optionally re-linearising the half-spaces --------------
        n_sca = max(1, int(cfg.sca_iterations))
        t_solve = time.perf_counter()
        sol, success, iter_total = None, False, 0
        for it in range(n_sca):
            if it > 0:
                # re-linearise around the trajectory just computed
                try:
                    ctx.lin_traj = np.array(sol.value(self._X), dtype=float).T[:, [i, j]]
                except Exception:       # pragma: no cover
                    break
                for term in self.terms:
                    term.set_values(opti, ctx)
                opti.set_initial(opti.x, sol.value(opti.x))
            try:
                sol = opti.solve()
                success = True
            except RuntimeError:
                sol = opti.debug        # best iterate available on failure
                success = False
            try:
                iter_total += int(sol.stats().get("iter_count", 0))
            except Exception:           # pragma: no cover
                pass
            if not success:
                break
        solve_ms = (time.perf_counter() - t_solve) * 1e3

        try:
            stats = dict(sol.stats())
        except Exception:               # pragma: no cover
            stats = {}
        stats["sca_iterations"] = n_sca
        # `iterations` aggregates every interior-point iteration spent in this
        # control cycle, across all SCA re-linearisations.
        iterations = iter_total if iter_total > 0 else int(stats.get("iter_count", -1))
        status = str(stats.get("return_status", "unknown"))

        try:
            cost_val = float(sol.value(self._cost_expr))
        except Exception:               # pragma: no cover
            cost_val = float("inf")

        # --- extract, guarding against NaN --------------------------------
        try:
            X_opt = np.array(sol.value(self._X), dtype=float)
            U_opt = np.array(sol.value(self._U), dtype=float)
            if not (np.isfinite(X_opt).all() and np.isfinite(U_opt).all()):
                raise ValueError("non-finite solution")
            x_pred = X_opt.T
            u_seq = U_opt.T if U_opt.ndim == 2 else U_opt.reshape(N, nu)
            self._prev_x = np.vstack([x_pred[1:], x_pred[-1:]])
            self._prev_u = np.vstack([u_seq[1:], u_seq[-1:]])
            self._consecutive_failures = 0 if success else self._consecutive_failures + 1
        except Exception:
            # Never let a bad solve poison the next warm start.
            success = False
            self._consecutive_failures += 1
            prev_x, prev_u = self._prev_x, self._prev_u
            self._prev_x = None
            self._prev_u = None
            if prev_x is not None and prev_u is not None:
                x_pred = np.vstack([prev_x[1:], prev_x[-1:]])
                u_seq = np.vstack([prev_u[1:], prev_u[-1:]])
            else:
                x_pred = ref.x_ref.copy()
                u_seq = np.zeros((N, nu))

        total_ms = (time.perf_counter() - t_start) * 1e3
        return OCPResult(
            success=success,
            x_pred=x_pred,
            u_opt=u_seq,
            x_ref=ref.x_ref,
            cost=cost_val,
            solve_ms=solve_ms,
            total_ms=total_ms,
            build_ms=build_ms,
            iterations=iterations,
            status=status,
            rebuilt=rebuilt,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Introspection — used to auto-generate the report tables
    # ------------------------------------------------------------------
    def structure(self) -> dict:
        """
        Size and sparsity of the NLP.

        Reports the non-zero count and density of the constraint Jacobian and of
        the Lagrangian Hessian, which is the quantitative form of the
        single-shooting vs multiple-shooting argument.
        """
        if self._opti is None:
            self._build()
        opti = self._opti
        x, g, f = opti.x, opti.g, opti.f

        jac = ca.jacobian(g, x).sparsity()
        lam = ca.MX.sym("lam", g.shape[0])
        hess = ca.hessian(f + ca.dot(lam, g), x)[0].sparsity()

        n_var = int(x.shape[0])
        n_con = int(g.shape[0])
        jac_nnz, hess_nnz = int(jac.nnz()), int(hess.nnz())
        return {
            "model": self.model.name,
            "N": self.cfg.N,
            "dt": self.cfg.dt,
            "n_variables": n_var,
            "n_constraints": n_con,
            "n_parameters": int(opti.p.shape[0]),
            "jac_nnz": jac_nnz,
            "jac_density": jac_nnz / max(1, n_con * n_var),
            "hess_nnz": hess_nnz,
            "hess_density": hess_nnz / max(1, n_var * n_var),
            "build_count": self.build_count,
            "is_parametric": self.is_parametric,
            "obstacle_terms": [t.describe() for t in self.terms],
        }

    def reset_warm_start(self) -> None:
        self._prev_x = None
        self._prev_u = None
