"""Criticality-aware placement ("cei"): the score, whole-node packing by failure domain, and multitenant nodes."""
import math

import numpy as np
import pytest

from raps.job import Job, job_dict
from raps.policy import PolicyType
from raps.resmgr import make_resource_manager
from raps.resmgr.criticality import CriticalityScorer, fattree_domain_weights
from raps.resmgr.default import ExclusiveNodeResourceManager
from raps.resmgr.multitenant import MultiTenantResourceManager
from raps.schedulers.multitenant import Scheduler as MultiTenantScheduler

pytestmark = pytest.mark.nodata

GPU_CONFIG = {"CPUS_PER_NODE": 2, "CORES_PER_CPU": 20, "GPUS_PER_NODE": 8}


def make_job(id, nodes=1, gpus=0, cores=1):
    job = Job(job_dict(nodes_required=nodes, name=str(id), account="", id=id, cpu_trace=0, gpu_trace=0,
                       ntx_trace=0, nrx_trace=0, submit_time=0, time_limit=3600, start_time=None,
                       end_time=None, expected_run_time=3600))
    job.gpu_units_required = gpus
    job.cpu_cores_required = cores
    return job


def direct_objective(load, weights, lam, g, unit, demand):
    """J written out term by term, for comparison with the vectorised version."""
    load = np.array(load, dtype=float)
    load[unit] += demand
    p = load / load.sum()
    q = p[p > 0]
    h = -(q * np.log(q)).sum() / math.log(len(load))
    phi = (weights * p).sum() / weights.max()
    e = load.sum() / len(load)   # load-proportional power proxy with peak 1
    return (1 - lam) * (h - g * phi) - lam * e


def test_score_matches_the_formula_term_by_term():
    load = [3, 0, 5, 1, 0, 8]
    s = CriticalityScorer(6, critical_units=[2])
    got = s.scores(load, [0, 1, 2, 3, 4, 5], 2)
    want = [direct_objective(load, s.weights, s.lam, s.g, u, 2) for u in range(6)]
    assert np.allclose(got, want, atol=1e-12)
    assert s.weights[2] == 1.0 and s.weights[0] == pytest.approx(0.2)


def test_score_prefers_spreading_and_avoids_critical_units():
    s = CriticalityScorer(4, critical_units=[1])
    assert s.pick([4, 0, 0, 0], [0, 1, 2, 3], 1) == 2   # an empty, ordinary unit; ties go to the lowest id
    assert s.pick([0, 0, 4, 4], [0, 1], 1) == 0          # same load, but unit 1 is critical


def test_hard_tier_uses_critical_units_only_when_nothing_else_fits():
    s = CriticalityScorer(3, critical_units=[0], hard_tier=True, g=0.0)
    assert s.pick([0, 9, 9], [0, 1, 2], 1) == 1          # unit 0 would spread better, but it is critical
    assert s.pick([0, 9, 9], [0], 1) == 0
    assert s.pick([0, 0, 0], [], 1) is None


def test_near_ties_go_to_the_lowest_id():
    s = CriticalityScorer(5)
    load = np.array([1.0, 1.0 + 1e-13, 1.0, 1.0, 1.0])
    assert s.pick(load, [4, 3, 2], 1) == 2


def test_exclusive_cei_without_topology_keeps_jobs_off_critical_nodes():
    rm = ExclusiveNodeResourceManager(10, [], node_placement="cei", critical_nodes=[0, 1, 2])
    a, b, c = make_job("a", 4), make_job("b", 3), make_job("c", 2)
    for j in (a, b, c):
        rm.assign_nodes_to_job(j, 0, PolicyType.FCFS)
    assert a.scheduled_nodes == [3, 4, 5, 6] and b.scheduled_nodes == [7, 8, 9]
    assert c.scheduled_nodes == [0, 1]                         # critical nodes only once nothing else is free


def fat_tree_rm(**kw):
    config = {"TOPOLOGY": "fat-tree", "FATTREE_K": 8}   # 4 hosts per edge switch, 128 hosts in all
    return ExclusiveNodeResourceManager(64, [], config, node_placement="cei", **kw)


def test_exclusive_cei_packs_each_job_into_the_fewest_edge_switches():
    rm = fat_tree_rm()
    assert rm.domain_size == 4 and rm.criticality.num_units == 16
    big, small = make_job("big", 6), make_job("small", 3)
    rm.assign_nodes_to_job(big, 0, PolicyType.FCFS)
    rm.assign_nodes_to_job(small, 0, PolicyType.FCFS)
    assert len({n // 4 for n in big.scheduled_nodes}) == 2      # 6 nodes need two switches, not more
    assert len({n // 4 for n in small.scheduled_nodes}) == 1
    assert not {n // 4 for n in small.scheduled_nodes} & {n // 4 for n in big.scheduled_nodes}   # spread out
    for j in (big, small):
        rm.free_nodes_from_job(j)
    assert rm.available_nodes == list(range(64))


def test_exclusive_cei_avoids_critical_switches():
    rm = fat_tree_rm(critical_nodes=[0])                        # marks the whole first edge switch critical
    jobs = [make_job(i, 4) for i in range(15)]
    for j in jobs:
        rm.assign_nodes_to_job(j, 0, PolicyType.FCFS)
    used = {n // 4 for j in jobs for n in j.scheduled_nodes}
    assert 0 not in used and len(used) == 15


def test_fattree_weights_are_scaled_and_cover_every_switch():
    w = fattree_domain_weights(8, 64)
    assert w.shape == (16,) and w.max() == pytest.approx(1.0) and (w >= 0).all()


def test_multitenant_first_is_unchanged_and_random_is_reproducible():
    rm = MultiTenantResourceManager(4, [], GPU_CONFIG)
    assert rm.node_placement == "first" and rm.select_node(make_job("x", gpus=2)) == 0

    def pick(seed):
        r = MultiTenantResourceManager(50, [], GPU_CONFIG, node_placement="random", seed=seed)
        return [r.select_node(make_job(i, gpus=1)) for i in range(5)]
    assert pick(3) == pick(3) and pick(3) != pick(4)


def test_multitenant_cei_spreads_gpus_and_avoids_critical_nodes():
    rm = MultiTenantResourceManager(4, [], GPU_CONFIG, node_placement="cei", critical_nodes=[1])
    placed = []
    for i in range(3):
        j = make_job(i, gpus=4)
        rm.assign_nodes_to_job(j, 0, rm.select_node(j))
        placed.append(j.scheduled_nodes[0])
    assert placed == [0, 2, 3]
    full = make_job("full", gpus=8)
    assert rm.select_node(full) == 1                            # only the critical node still has 8 free GPUs


def test_multitenant_scheduler_honours_the_placement():
    rm = make_resource_manager(4, [], {**GPU_CONFIG, "multitenant": True}, node_placement="cei")
    sched = MultiTenantScheduler(config=GPU_CONFIG, policy="fcfs", bfpolicy="firstfit", resource_manager=rm)
    queue, running = [make_job(i, gpus=2) for i in range(4)], []
    sched.schedule(queue, running, current_time=0)
    assert sorted(j.scheduled_nodes[0] for j in running) == [0, 1, 2, 3]   # first-fit would stack all on node 0


def test_factory_rejects_unknown_placements_for_both_managers():
    with pytest.raises(ValueError):
        make_resource_manager(4, [], {**GPU_CONFIG, "multitenant": True}, node_placement="shotgun")
    with pytest.raises(ValueError):
        make_resource_manager(4, [], {}, node_placement="shotgun")
