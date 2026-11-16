import random
from raps.job import JobState
from raps.policy import PolicyType
from raps.resmgr.criticality import CriticalityScorer, fattree_domain_weights


class ExclusiveNodeResourceManager:
    """
    Legacy exclusive-node resource manager: allocates and frees full nodes.
    """

    PLACEMENTS = ("first", "random", "cei")

    def __init__(self, total_nodes, down_nodes, config=None, node_placement="first", seed=None,
                 critical_nodes=()):
        if node_placement not in self.PLACEMENTS:
            raise ValueError(f"node_placement must be one of {self.PLACEMENTS}, not {node_placement!r}")
        self.node_placement = node_placement
        self._rng = random.Random(seed)   # its own stream: placing jobs must not shift the workload's random numbers
        self.total_nodes = total_nodes
        self.down_nodes = set(down_nodes)
        self.config = config or {}
        self.criticality = None
        if node_placement == "cei":
            self._setup_criticality(critical_nodes)

        # Determine per-node capacities
        cfg = self.config
        if 'CPUS_PER_NODE' in cfg and 'CORES_PER_CPU' in cfg:
            total_cpu = cfg['CPUS_PER_NODE'] * cfg['CORES_PER_CPU']
        else:
            total_cpu = cfg.get('CORES_PER_NODE', cfg.get('CPUS_PER_NODE', 1))
        total_gpu = cfg.get('GPUS_PER_NODE', 0)

        # Build unified node list so engine can inspect resource_manager.nodes
        self.nodes = []
        for i in range(self.total_nodes):
            is_down = i in self.down_nodes
            self.nodes.append({
                'id': i,
                'total_cpu_cores':     total_cpu,
                'available_cpu_cores': 0 if is_down else total_cpu,
                'total_gpu_units':     total_gpu,
                'available_gpu_units': 0 if is_down else total_gpu,
                'is_down':             is_down
            })

        # Available nodes list for allocation/frees
        self.available_nodes = [n['id'] for n in self.nodes if not n['is_down']]
        # System utilization history (time, util%)
        self.sys_util_history = []

    def assign_nodes_to_job(self, job, current_time, policy, node_id=None):
        """Assigns full nodes to a job (replay or count-based)."""
        # Ensure enough free nodes
        if len(self.available_nodes) < job.nodes_required:
            raise ValueError(f"Not enough available nodes to schedule job {job.id}",
                             f"{len(self.available_nodes)} < {job.nodes_required}")

        if policy == PolicyType.REPLAY and job.scheduled_nodes:
            # Telemetry replay: use the exact nodes
            self.available_nodes = [n for n in self.available_nodes if n not in job.scheduled_nodes]
        elif self.node_placement == "random":
            # Scattered allocation: any N of the free nodes
            picked = set(self._rng.sample(self.available_nodes, job.nodes_required))
            job.scheduled_nodes = sorted(picked)
            self.available_nodes = [n for n in self.available_nodes if n not in picked]
        elif self.node_placement == "cei":
            # Criticality-aware: fewest failure domains, chosen by the CEI objective (see criticality.py)
            picked = set(self._pick_cei(job.nodes_required))
            job.scheduled_nodes = sorted(picked)
            self.available_nodes = [n for n in self.available_nodes if n not in picked]
        else:
            # Count-based allocation: take the first N free nodes
            job.scheduled_nodes = self.available_nodes[:job.nodes_required]
            self.available_nodes = self.available_nodes[job.nodes_required:]

        # Mark job running
        job.start_time = current_time
        if job.expected_run_time:
            job.end_time = current_time + job.expected_run_time  # This may be an assumption!
        job.current_state = JobState.RUNNING

    def _setup_criticality(self, critical_nodes=()):
        """Failure domains are edge switches on a fat-tree (k/2 consecutive nodes), single nodes otherwise."""
        cfg = self.config
        weights = None
        if cfg.get("TOPOLOGY") == "fat-tree" and cfg.get("FATTREE_K"):
            self.domain_size = int(cfg["FATTREE_K"]) // 2
            weights = fattree_domain_weights(int(cfg["FATTREE_K"]), self.total_nodes)
        else:
            self.domain_size = 1
        num_domains = -(-self.total_nodes // self.domain_size)
        critical = {int(n) // self.domain_size for n in critical_nodes}
        self.criticality = CriticalityScorer(num_domains, critical_units=sorted(critical), unit_weights=weights)

    def _pick_cei(self, count):
        """Pack ``count`` whole nodes into as few failure domains as possible; the CEI score picks which."""
        size = self.domain_size
        num_domains = self.criticality.num_units
        free = [[] for _ in range(num_domains)]
        for n in sorted(self.available_nodes):
            free[n // size].append(n)
        busy = [0.0] * num_domains
        for n in range(self.total_nodes):
            if n not in self.down_nodes:
                busy[n // size] += 1.0
        for d, nodes in enumerate(free):
            busy[d] -= len(nodes)
        picked, remaining = [], count
        while remaining > 0:
            counts = [len(f) for f in free]
            fits = [d for d in range(num_domains) if counts[d] >= remaining]
            if fits:
                candidates, take = fits, remaining
            else:
                most = max(counts)
                candidates, take = [d for d in range(num_domains) if counts[d] == most], most
            d = self.criticality.pick(busy, candidates, take)
            picked.extend(free[d][:take])
            free[d] = free[d][take:]
            busy[d] += take
            remaining -= take
        return picked

    def free_nodes_from_job(self, job):
        """Frees the full nodes previously allocated to a job."""
        if getattr(job, 'scheduled_nodes', None):
            for n in job.scheduled_nodes:
                if n not in self.available_nodes:
                    self.available_nodes.append(n)
                else:
                    # Already free — log instead of raising
                    print(f"[WARN] Tried to free node {n}, but it was already available")
                    print(f"Atempting to free node {n} after completion of job {job.id}. " +
                                     "Node is already free (in available nodes)!")
            self.available_nodes = sorted(self.available_nodes)

    def update_system_utilization(self, current_time, running_jobs):
        """
        Computes system utilization as percentage of non-down nodes that are active.

        Parameters:
        - current_time: simulation time
        - running_jobs: list of currently running Job objects
        """
        # Number of active nodes is length of running_jobs
        num_active = len(running_jobs)
        total_operational = self.total_nodes - len(self.down_nodes)
        util = (num_active / total_operational) * 100 if total_operational else 0
        self.sys_util_history.append((current_time, util))
        return util
        # """
        # Computes system utilization as percentage of non-down nodes that are active.
        # """
        # total_operational = self.total_nodes - len(self.down_nodes)
        # util = (num_active_nodes / total_operational) * 100 if total_operational else 0
        # self.sys_util_history.append((current_time, util))
        # return util

    def node_failure(self, mtbf):
        return []
        # Node failure not working!
        #  """Simulate node failure using Weibull distribution."""
        #  shape_parameter = 1.5
        #  scale_parameter = mtbf * 3600  # Convert to seconds

        #  # Create a NumPy array of node indices, excluding down nodes
        #  all_nodes = np.array(sorted(set(range(self.total_nodes)) - set(self.down_nodes)))

        #  # Sample the Weibull distribution for all nodes at once
        #  random_values = weibull_min.rvs(shape_parameter, scale=scale_parameter, size=all_nodes.size)

        #  # Identify nodes that have failed
        #  failure_threshold = 0.1
        #  failed_nodes_mask = random_values < failure_threshold
        #  newly_downed_nodes = all_nodes[failed_nodes_mask]

        #  # Update available and down nodes
        #  for node_index in newly_downed_nodes:
        #      if node_index in self.available_nodes:
        #          self.available_nodes.remove(node_index)
        #      self.down_nodes.add(str(node_index))

        #  return newly_downed_nodes.tolist()
