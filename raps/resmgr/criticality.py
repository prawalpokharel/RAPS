"""
Criticality-aware node placement ("--node-placement cei").

A job lost to a failure costs more when the failed node, or the failure domain it belongs to, carries a large
share of the running work, and more again when that node is one the operator has marked critical. This
placement scores each candidate by how the system's load spread would look after the job lands there:

    J = (1 - lam) * (H(p) / log m  -  g * Phi)  -  lam * E

    p    load shares over the m placement units (nodes, or failure domains such as an edge switch)
    H    Shannon entropy of p: higher when work is spread, so one failure takes a smaller share
    Phi  centrality-weighted load share, sum(c_i * p_i) / max(c): lower when critical units carry less work
    E    normalised node power from RAPS's power model

and takes the candidate with the highest J. The weights c are 1.0 for critical units and 0.2 for the rest (up to
0.6 with structural centrality such as switch betweenness), and lam = 0.35, g = 0.5 are the defaults of the Centrality-Entropy Index (CEI) study this follows (Pokharel,
"Self-Tuning Governance of the Energy-Resilience Tradeoff in GPU Cluster Scheduling", IEMCON 2026).

With ``hard_tier=True`` the critical units are used only when no other unit has room, and J chooses among the
rest. Scores within ``TIE_TOL`` of the best count as equal and the lowest id wins, so the choice does not
depend on floating-point summation order.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_LAMBDA = 0.35
DEFAULT_G = 0.5
CRITICAL_WEIGHT = 1.0
BASE_WEIGHT = 0.2
STRUCTURAL_SPAN = 0.4
TIE_TOL = 1e-9


class CriticalityScorer:
    """Scores placement units (nodes or failure domains) by the CEI objective.

    Parameters
    ----------
    num_units : int
        Number of placement units.
    critical_units : iterable of int, optional
        Units the operator marks critical (weight ``CRITICAL_WEIGHT``).
    unit_weights : array-like, optional
        Structural centrality per unit in [0, 1], for example edge-switch betweenness. Other units get
        ``BASE_WEIGHT + STRUCTURAL_SPAN * weight`` (0.2 to 0.6), so a unit marked critical (1.0) still stands
        out on a symmetric topology where every unit has the same structural weight.
    unit_power : callable, optional
        ``unit_power(load_array) -> power_array``; defaults to a load-proportional proxy.
    """

    def __init__(self, num_units, critical_units=(), unit_weights=None, lam=DEFAULT_LAMBDA, g=DEFAULT_G,
                 hard_tier=False, unit_power=None, peak_power=1.0):
        self.num_units = int(num_units)
        self.lam = float(lam)
        self.g = float(g)
        self.hard_tier = bool(hard_tier)
        self.unit_power = unit_power
        self.peak_power = float(peak_power) if peak_power else 1.0
        self.critical = np.zeros(self.num_units, dtype=bool)
        self.set_weights(unit_weights)
        self.set_critical(critical_units)

    def set_weights(self, unit_weights=None):
        base = np.full(self.num_units, BASE_WEIGHT)
        if unit_weights is not None:
            w = np.asarray(unit_weights, dtype=float)
            if w.shape != (self.num_units,):
                raise ValueError(f"unit_weights needs {self.num_units} entries, got {w.shape}")
            base = BASE_WEIGHT + STRUCTURAL_SPAN * np.clip(w, 0.0, 1.0)
        self._structural = base
        self._refresh()

    def set_critical(self, critical_units=()):
        self.critical = np.zeros(self.num_units, dtype=bool)
        idx = [int(u) for u in critical_units]
        if idx:
            self.critical[idx] = True
        self._refresh()

    def _refresh(self):
        if not hasattr(self, "_structural"):
            return
        c = self._structural.copy()
        c[self.critical] = CRITICAL_WEIGHT
        self.weights = c
        self._c_max = float(c.max()) if c.size else 1.0

    def scores(self, load, candidates, demand):
        """CEI objective after adding ``demand`` to each candidate unit (vectorised over candidates)."""
        load = np.asarray(load, dtype=float)
        cand = np.asarray(candidates, dtype=int)
        d = float(demand)
        total = load.sum() + d
        m = self.num_units
        if total <= 0 or m < 2:
            return np.zeros(cand.size)
        # entropy H = log T - S / T with S = sum l log l, updated for the one unit that changes
        xlogx = np.where(load > 0, load * np.log(np.where(load > 0, load, 1.0)), 0.0)
        s_total = xlogx.sum()
        new = load[cand] + d
        s_new = s_total - xlogx[cand] + new * np.log(new)
        h = (math.log(total) - s_new / total) / math.log(m)
        phi = ((self.weights * load).sum() + self.weights[cand] * d) / (total * self._c_max)
        if self.lam > 0:
            if self.unit_power is not None:
                base = np.asarray(self.unit_power(load), dtype=float)
                after = np.asarray(self.unit_power(new), dtype=float)
                e = (base.sum() - base[cand] + after) / (m * self.peak_power)
            else:
                e = np.full(cand.size, total / (m * self.peak_power))
        else:
            e = 0.0
        return (1.0 - self.lam) * (h - self.g * phi) - self.lam * e

    def pick(self, load, candidates, demand):
        """Best candidate unit for ``demand``, or None when there is no candidate."""
        cand = np.asarray(sorted(int(u) for u in candidates), dtype=int)
        if cand.size == 0:
            return None
        if self.hard_tier:
            ordinary = cand[~self.critical[cand]]
            if ordinary.size:
                cand = ordinary
        j = self.scores(load, cand, demand)
        best = j.max()
        return int(cand[np.flatnonzero(j >= best - TIE_TOL)[0]])


def fattree_domain_weights(k, total_nodes):
    """Edge-switch betweenness on RAPS's fat-tree, restricted to the hosts that exist, scaled to max 1.

    Returns one weight per edge switch (failure domain of ``k // 2`` consecutive node ids).
    """
    import networkx as nx
    from raps.network.fat_tree import build_fattree, node_id_to_host_name

    g = build_fattree(k, total_nodes)
    hosts = {node_id_to_host_name(n, k) for n in range(total_nodes)}
    g.remove_nodes_from([n for n in list(g) if n.startswith("h_") and n not in hosts])
    per_domain = k // 2
    num_domains = math.ceil(total_nodes / per_domain)
    edge_names = [node_id_to_host_name(d * per_domain, k).replace("h_", "e_", 1).rsplit("_", 1)[0]
                  for d in range(num_domains)]
    bc = nx.betweenness_centrality(g.subgraph(nx.node_connected_component(g, next(iter(hosts)))))
    w = np.array([bc.get(e, 0.0) for e in edge_names], dtype=float)
    return w / w.max() if w.max() > 0 else w
