from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:
    linprog = None
    _HAS_SCIPY = False


@dataclass(frozen=True)
class OptimizationConfig:
    """
    flex: max reducible ratio per category (0..1)
    pain: weight (cost) per 1 EGP reduction (lower => prefer reducing this category)
    """
    flex: Dict[str, float]
    pain: Dict[str, float]


DEFAULT_CONFIG = OptimizationConfig(
    flex={
                                   
        "Food": 0.15,
        "Transport": 0.10,
        "Shopping": 0.35,
        "Entertainment": 0.40,
        "Bills": 0.00,                         
        "Health": 0.10,
        "Education": 0.10,
    },
    pain={
                                                          
        "Entertainment": 0.6,
        "Shopping": 0.7,
        "Food": 1.0,
        "Transport": 1.2,
        "Bills": 999.0,                             
        "Health": 2.0,
        "Education": 2.0,
    },
)


def optimize_reductions(
    forecast_by_category: Dict[str, float],
    required_cut: float,
    cfg: OptimizationConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    Solve a Linear Program:
      minimize   Σ pain_i * r_i
      s.t.       0 <= r_i <= flex_i * forecast_i
                 Σ r_i >= required_cut

    Returns:
      {
        "status": "ok" | "no_solver" | "infeasible",
        "required_cut": float,
        "achieved_cut": float,
        "reductions": {cat: r_i},
        "new_budgets": {cat: forecast_i - r_i},
      }
    """
    required_cut = float(max(0.0, required_cut))

             
    if required_cut <= 1e-9:
        return {
            "status": "ok",
            "required_cut": 0.0,
            "achieved_cut": 0.0,
            "reductions": {k: 0.0 for k in forecast_by_category},
            "new_budgets": {k: float(v) for k, v in forecast_by_category.items()},
        }

    cats = list(forecast_by_category.keys())
    n = len(cats)

                          
    caps = np.zeros(n, dtype=float)
    costs = np.zeros(n, dtype=float)

    for i, c in enumerate(cats):
        v = float(max(0.0, forecast_by_category.get(c, 0.0)))
        flex = float(cfg.flex.get(c, 0.0))
        caps[i] = v * max(0.0, min(1.0, flex))
        costs[i] = float(cfg.pain.get(c, 1.0))

                                                      
    max_possible = float(np.sum(caps))
    if max_possible + 1e-9 < required_cut:
                    
        return {
            "status": "infeasible",
            "required_cut": required_cut,
            "achieved_cut": max_possible,
            "reductions": {cats[i]: float(caps[i]) for i in range(n)},
            "new_budgets": {cats[i]: float(forecast_by_category[cats[i]] - caps[i]) for i in range(n)},
            "note": "Even maximum reducible amounts cannot meet required_cut.",
        }

                                    
    if not _HAS_SCIPY or linprog is None:
        return {
            "status": "no_solver",
            "required_cut": required_cut,
            "achieved_cut": 0.0,
            "reductions": {},
            "new_budgets": {},
            "note": "scipy is not available. Add scipy to requirements.txt to enable LP solver.",
        }

                  
                                                        
    A_ub = np.array([[-1.0] * n], dtype=float)
    b_ub = np.array([-required_cut], dtype=float)

    bounds: Tuple[Tuple[float, float], ...] = tuple((0.0, float(caps[i])) for i in range(n))

    res = linprog(
        c=costs,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not res.success or res.x is None:
        return {
            "status": "infeasible",
            "required_cut": required_cut,
            "achieved_cut": 0.0,
            "reductions": {},
            "new_budgets": {},
            "note": f"linprog failed: {getattr(res, 'message', 'unknown')}",
        }

    r = np.maximum(0.0, res.x)
    achieved = float(np.sum(r))

    reductions = {cats[i]: float(round(r[i], 2)) for i in range(n) if r[i] > 0.01}
    new_budgets = {cats[i]: float(round(float(forecast_by_category[cats[i]]) - r[i], 2)) for i in range(n)}

    return {
        "status": "ok",
        "required_cut": float(round(required_cut, 2)),
        "achieved_cut": float(round(achieved, 2)),
        "reductions": reductions,
        "new_budgets": new_budgets,
        "meta": {
            "max_possible_cut": float(round(max_possible, 2)),
            "solver": "scipy.optimize.linprog(highs)",
        },
    }