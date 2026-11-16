"""Node power when several jobs share a node (multitenant systems) sums their utilisation."""
import numpy as np
import pytest

from raps.power import PowerManager, compute_node_power

pytestmark = pytest.mark.nodata

CONFIG = {"CPUS_PER_NODE": 2, "GPUS_PER_NODE": 8, "POWER_CPU_IDLE": 90, "POWER_CPU_MAX": 270,
          "POWER_GPU_IDLE": 50, "POWER_GPU_MAX": 250, "POWER_MEM": 74.26, "POWER_NIC": 20, "NICS_PER_NODE": 4,
          "POWER_NVME": 30, "SIVOC_LOSS_CONSTANT": 13, "SIVOC_EFFICIENCY": 0.98, "SC_SHAPE": (1, 1, 4)}


def manager():
    pm = PowerManager.__new__(PowerManager)
    pm.config = CONFIG
    pm.power_func = compute_node_power
    idle, idle_loss = compute_node_power(0, 0, 0, CONFIG)
    pm.power_state = np.full(CONFIG["SC_SHAPE"], idle)
    pm.sivoc_loss = np.full(CONFIG["SC_SHAPE"], idle_loss)
    return pm


def test_jobs_sharing_a_node_add_up():
    pm = manager()
    pm.update_power_state([[0], [0], [2]], cpu_util=[0.2, 0.3, 0.1], gpu_util=[2.0, 3.0, 4.0], net_util=[0, 0, 0])
    shared, _ = compute_node_power(0.5, 5.0, 0, CONFIG)
    alone, _ = compute_node_power(0.1, 4.0, 0, CONFIG)
    idle, _ = compute_node_power(0, 0, 0, CONFIG)
    assert pm.power_state[0, 0, 0] == pytest.approx(shared)      # before: only the second job's 3 GPUs counted
    assert pm.power_state[0, 0, 2] == pytest.approx(alone)
    assert pm.power_state[0, 0, 1] == pytest.approx(idle)


def test_sums_are_capped_at_the_node_and_one_job_per_node_is_unchanged():
    pm = manager()
    pm.update_power_state([[1], [1]], cpu_util=[1.5, 1.5], gpu_util=[6.0, 6.0], net_util=[0, 0])
    assert pm.power_state[0, 0, 1] == pytest.approx(compute_node_power(2, 8, 0, CONFIG)[0])
    pm2 = manager()
    pm2.update_power_state([[0, 1], [3]], cpu_util=[1.0, 0.5], gpu_util=[4.0, 8.0], net_util=[0, 0])
    assert pm2.power_state[0, 0, 0] == pytest.approx(compute_node_power(1.0, 4.0, 0, CONFIG)[0])
    assert pm2.power_state[0, 0, 3] == pytest.approx(compute_node_power(0.5, 8.0, 0, CONFIG)[0])
