import numpy as np
import matplotlib.pyplot as plt
import random
from math import gcd, gamma
from functools import reduce
import heapq
import time as a_time


# ==============================================================================
# SECTION 1: TASK GENERATION AND EDF SIMULATION (UNCHANGED FROM PREVIOUS VERSION)
# ==============================================================================

def generate_tasks(num_tasks, total_utilization):
    utilizations = []
    remaining_util = total_utilization
    if remaining_util <= 0: return []

    for i in range(1, num_tasks):
        if num_tasks - i <= 0: continue
        next_util = remaining_util * random.random() ** (1 / (num_tasks - i))
        utilizations.append(remaining_util - next_util)
        remaining_util = next_util
    if remaining_util > 0: utilizations.append(remaining_util)

    tasks = []
    for i, util in enumerate(utilizations):
        period = random.choice([10, 20, 40, 50, 100, 200])
        execution = util * period
        if execution < 1: execution = 1
        tasks.append({
            'id': i, 'execution': execution, 'period': period,
            'deadline': period, 'utilization': execution / period if period > 0 else 0
        })
    return tasks


def edf_schedule_on_core(tasks_on_core, hyperperiod):
    if not tasks_on_core: return {}, 0
    if sum(t['utilization'] for t in tasks_on_core) > 1: return {}, float('inf')

    time = 0
    ready_queue, job_arrivals = [], []
    for task in tasks_on_core:
        if task['period'] > 0:
            for i in range(hyperperiod // task['period']):
                arrival_time = i * task['period']
                job_arrivals.append((arrival_time, task['execution'], arrival_time + task['deadline'], task['id']))
    job_arrivals.sort()

    job_results = {task['id']: {'finish_times': [], 'deadlines': []} for task in tasks_on_core}
    current_job = None

    while time < hyperperiod * 2:
        while job_arrivals and job_arrivals[0][0] <= time:
            arrival, exec_time, deadline, task_id = job_arrivals.pop(0)
            heapq.heappush(ready_queue, (deadline, exec_time, task_id))

        if current_job is None and ready_queue:
            deadline, exec_time, task_id = heapq.heappop(ready_queue)
            current_job = {'deadline': deadline, 'remaining_exec': exec_time, 'task_id': task_id}

        if current_job:
            current_job['remaining_exec'] -= 1
            if current_job['remaining_exec'] <= 0:
                task_id = current_job['task_id']
                job_results[task_id]['finish_times'].append(time + 1)
                job_results[task_id]['deadlines'].append(current_job['deadline'])
                current_job = None

        time += 1
        if not current_job and not ready_queue and not job_arrivals: break

    latest_finish = max((ft for res in job_results.values() for ft in res['finish_times']), default=0)
    return job_results, latest_finish


def calculate_qos(finish_time, deadline):
    if finish_time <= deadline: return 100
    if deadline <= 0: return 0
    if finish_time >= 2 * deadline: return 0
    return 100 * (1 - (finish_time - deadline) / deadline)


def get_solution_fitness(solution, tasks, num_cores):
    if len(solution) == 0: return 0
    core_assignments = {i: [] for i in range(num_cores)}
    for task_idx, core_idx in enumerate(solution):
        core_assignments[core_idx].append(tasks[task_idx])

    if not tasks: return 0
    periods = [t['period'] for t in tasks if t['period'] > 0]
    if not periods: return 0
    hyperperiod = reduce(lambda a, b: a * b // gcd(a, b) if a > 0 and b > 0 else a or b, periods, 1)

    all_task_qos = []
    for core_id in range(num_cores):
        tasks_on_core = core_assignments[core_id]
        job_results, _ = edf_schedule_on_core(tasks_on_core, hyperperiod)
        for task in tasks_on_core:
            if task['id'] in job_results and job_results[task['id']]['finish_times']:
                qos_values = [calculate_qos(ft, dl) for ft, dl in
                              zip(job_results[task['id']]['finish_times'], job_results[task['id']]['deadlines'])]
                avg_qos = np.mean(qos_values) if qos_values else 0
                all_task_qos.append(avg_qos)
            else:
                all_task_qos.append(0)

    return np.mean(all_task_qos) if all_task_qos else 0


# ==============================================================================
# SECTION 2: METAHEURISTIC ALGORITHMS (GA AND CUCKOO SEARCH)
# ==============================================================================

class GeneticScheduler:
    def __init__(self, tasks, num_cores, pop_size=50, elite_frac=0.2, mutation_rate=0.1, generations=100):
        self.tasks = tasks
        self.num_cores = num_cores
        self.pop_size = pop_size
        self.elite_size = int(elite_frac * pop_size)
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.fitness_cache = {}

    def _fitness(self, solution):
        sol_tuple = tuple(solution)
        if sol_tuple in self.fitness_cache:
            return self.fitness_cache[sol_tuple]
        fitness_val = get_solution_fitness(solution, self.tasks, self.num_cores)
        self.fitness_cache[sol_tuple] = fitness_val
        return fitness_val

    def initialize_population(self):
        return [np.random.randint(0, self.num_cores, len(self.tasks)) for _ in range(self.pop_size)]

    def evolve(self):
        if not self.tasks or self.num_cores == 0: return []
        population = self.initialize_population()

        for _ in range(self.generations):
            fitnesses = [self._fitness(p) for p in population]
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            new_population = [population[i] for i in elite_indices]

            total_fitness = sum(fitnesses)
            if total_fitness > 0:
                probs = [f / total_fitness for f in fitnesses]
                parent_indices = np.random.choice(len(population), size=self.pop_size - self.elite_size, p=probs,
                                                  replace=True)
                parents = [population[i] for i in parent_indices]

                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        p1, p2 = parents[i], parents[i + 1]
                        point = random.randint(1, len(p1) - 1) if len(p1) > 1 else 1
                        c1 = np.concatenate((p1[:point], p2[point:]))
                        for j in range(len(c1)):
                            if random.random() < self.mutation_rate: c1[j] = random.randint(0, self.num_cores - 1)
                        new_population.append(c1)

            population = new_population[:self.pop_size]

        final_fitnesses = [self._fitness(p) for p in population]
        return population[np.argmax(final_fitnesses)]


class CuckooScheduler:
    def __init__(self, tasks, num_cores, n_nests=50, pa=0.25, beta=1.5, generations=100):
        self.tasks = tasks
        self.num_cores = num_cores
        self.n_nests = n_nests
        self.pa = pa
        self.beta = beta
        self.generations = generations
        self.n_tasks = len(tasks)
        self.fitness_cache = {}

    def _fitness(self, nest):
        nest_tuple = tuple(nest)
        if nest_tuple in self.fitness_cache: return self.fitness_cache[nest_tuple]
        fitness_val = get_solution_fitness(nest, self.tasks, self.num_cores)
        self.fitness_cache[nest_tuple] = fitness_val
        return fitness_val

    def _levy_flight_step(self):
        sigma_u = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / (
                    gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        sigma_v = 1
        u = np.random.normal(0, sigma_u, 1)
        v = np.random.normal(0, sigma_v, 1)
        step = u / (np.abs(v) ** (1 / self.beta))
        return step

    def run(self):
        if not self.tasks or self.num_cores == 0: return []
        nests = [np.random.randint(0, self.num_cores, self.n_tasks) for _ in range(self.n_nests)]
        fitnesses = np.array([self._fitness(nest) for nest in nests])

        best_nest_idx = np.argmax(fitnesses)
        best_nest = nests[best_nest_idx]
        best_fitness = fitnesses[best_nest_idx]

        for _ in range(self.generations):
            step_size = 0.01 * self._levy_flight_step() * (best_nest - nests[random.randint(0, self.n_nests - 1)])
            new_nest = nests[random.randint(0, self.n_nests - 1)].copy()

            n_changes = int(np.linalg.norm(step_size) / self.n_tasks * 100) + 1
            n_changes = min(n_changes, self.n_tasks)

            indices_to_change = random.sample(range(self.n_tasks), n_changes)
            for idx in indices_to_change:
                new_nest[idx] = random.randint(0, self.num_cores - 1)

            f_new = self._fitness(new_nest)
            j = random.randint(0, self.n_nests - 1)
            if f_new > fitnesses[j]:
                nests[j] = new_nest
                fitnesses[j] = f_new

            if f_new > best_fitness:
                best_fitness = f_new
                best_nest = new_nest

            sorted_indices = np.argsort(fitnesses)
            n_abandon = int(self.pa * self.n_nests)
            for k in range(n_abandon):
                idx_to_abandon = sorted_indices[k]
                nests[idx_to_abandon] = np.random.randint(0, self.num_cores, self.n_tasks)
                fitnesses[idx_to_abandon] = self._fitness(nests[idx_to_abandon])

        return nests[np.argmax(fitnesses)]


# ==============================================================================
# SECTION 3: SIMULATION AND VISUALIZATION
# ==============================================================================

def run_phase_two_simulation(num_runs_per_config=3):
    configurations = [
        (8, 0.5), (8, 0.75), (8, 1.0),
        (16, 0.5), (16, 0.75), (16, 1.0),
        (32, 0.5), (32, 0.75), (32, 1.0)
    ]
    results = {}

    for cores, util_per_core in configurations:
        print(f"Running Config: {cores} Cores, {util_per_core} Util/Core...")
        ga_runs, cs_runs = [], []
        for i in range(num_runs_per_config):
            tasks = generate_tasks(num_tasks=int(4 * cores), total_utilization=cores * util_per_core)
            if not tasks: continue

            ga_scheduler = GeneticScheduler(tasks, cores)
            ga_assignment = ga_scheduler.evolve()
            if len(ga_assignment) > 0:
                core_utils = [sum(tasks[i]['utilization'] for i, c in enumerate(ga_assignment) if c == core_id) for
                              core_id in range(cores)]
                ga_runs.append(get_solution_fitness(ga_assignment, tasks, cores))

            cs_scheduler = CuckooScheduler(tasks, cores)
            cs_assignment = cs_scheduler.run()
            if len(cs_assignment) > 0:
                core_utils = [sum(tasks[i]['utilization'] for i, c in enumerate(cs_assignment) if c == core_id) for
                              core_id in range(cores)]
                cs_runs.append(get_solution_fitness(cs_assignment, tasks, cores))

        results[(cores, util_per_core)] = {'GA': ga_runs, 'CS': cs_runs}
    return results


def visualize_comparison_results(results):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    util_levels = sorted(list(set(k[1] for k in results.keys())))

    for algo in ['GA', 'CS']:
        for cores in [8, 16, 32]:
            avg_qos = []
            for util in util_levels:
                if (cores, util) in results:
                    qos_values = results[(cores, util)][algo]
                    avg_qos.append(np.mean(qos_values) if qos_values else np.nan)
                else:
                    avg_qos.append(np.nan)

            line_style = '-' if algo == 'GA' else '--'
            ax.plot(util_levels, avg_qos, marker='o', linestyle=line_style, label=f'{cores} Cores - {algo}')

    ax.set_title('Algorithm Comparison: Average System QoS vs. System Load', fontsize=16)
    ax.set_xlabel('Target Utilization per Core', fontsize=12)
    ax.set_ylabel('Average System QoS (%)', fontsize=12)
    ax.set_ylim(0, 101)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.savefig('phase_two_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    start_time = a_time.time()
    simulation_results = run_phase_two_simulation(num_runs_per_config=5)
    if simulation_results:
        visualize_comparison_results(simulation_results)
        print("Phase Two simulation finished successfully.")
        print(f"Results saved to 'phase_two_comparison.png'.")
    else:
        print("Simulation did not produce results.")
    print(f"Total execution time: {a_time.time() - start_time:.2f} seconds.")