import numpy as np
import matplotlib.pyplot as plt
import random
from math import gcd, gamma
from functools import reduce
import heapq
import time as a_time
import os


# ==============================================================================
# SECTION 1: TASK GENERATION AND EDF SIMULATION
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


def get_full_metrics_for_solution(solution, tasks, num_cores):
    if len(solution) == 0: return {}
    core_assignments = {i: [] for i in range(num_cores)}
    for task_idx, core_idx in enumerate(solution):
        core_assignments[core_idx].append(tasks[task_idx])

    if not tasks: return {}
    periods = [t['period'] for t in tasks if t['period'] > 0]
    if not periods: return {}
    hyperperiod = reduce(lambda a, b: a * b // gcd(a, b) if a > 0 and b > 0 else a or b, periods, 1)

    per_task_qos_list, all_core_utils = [], [sum(t['utilization'] for t in core_assignments[i]) for i in
                                             range(num_cores)]
    total_makespan = 0

    for core_id in range(num_cores):
        tasks_on_core = core_assignments[core_id]
        job_results, core_makespan = edf_schedule_on_core(tasks_on_core, hyperperiod)
        total_makespan = max(total_makespan, core_makespan)
        for task in tasks_on_core:
            if task['id'] in job_results and job_results[task['id']]['finish_times']:
                qos_values = [calculate_qos(ft, dl) for ft, dl in
                              zip(job_results[task['id']]['finish_times'], job_results[task['id']]['deadlines'])]
                avg_qos = np.mean(qos_values) if qos_values else 0
                per_task_qos_list.append({'id': task['id'], 'qos': avg_qos})
            else:
                per_task_qos_list.append({'id': task['id'], 'qos': 0})

    per_task_qos_list.sort(key=lambda x: x['id'])
    system_qos_val = np.mean([item['qos'] for item in per_task_qos_list]) if per_task_qos_list else 0

    return {'core_utils': all_core_utils, 'makespan': total_makespan, 'per_task_qos': per_task_qos_list,
            'system_qos': system_qos_val}


def get_solution_fitness(solution, tasks, num_cores):
    metrics = get_full_metrics_for_solution(solution, tasks, num_cores)
    return metrics.get('system_qos', 0)


# ==============================================================================
# SECTION 2: METAHEURISTIC ALGORITHMS
# ==============================================================================

class GeneticScheduler:
    def __init__(self, tasks, num_cores, pop_size=25, elite_frac=0.2, mutation_rate=0.1, generations=30):
        self.tasks = tasks;
        self.num_cores = num_cores;
        self.pop_size = pop_size
        self.elite_size = int(elite_frac * pop_size);
        self.mutation_rate = mutation_rate
        self.generations = generations;
        self.fitness_cache = {}

    def _fitness(self, solution):
        sol_tuple = tuple(solution)
        if sol_tuple in self.fitness_cache: return self.fitness_cache[sol_tuple]
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
    def __init__(self, tasks, num_cores, n_nests=25, pa=0.25, beta=1.5, generations=30):
        self.tasks = tasks;
        self.num_cores = num_cores;
        self.n_nests = n_nests
        self.pa = pa;
        self.beta = beta;
        self.generations = generations
        self.n_tasks = len(tasks);
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
        u = np.random.normal(0, sigma_u, 1);
        v = np.random.normal(0, 1, 1)
        return u / (np.abs(v) ** (1 / self.beta))

    def run(self):
        if not self.tasks or self.num_cores == 0: return []
        nests = [np.random.randint(0, self.num_cores, self.n_tasks) for _ in range(self.n_nests)]
        fitnesses = np.array([self._fitness(nest) for nest in nests])
        best_nest = nests[np.argmax(fitnesses)];
        best_fitness = np.max(fitnesses)
        for _ in range(self.generations):
            step = self._levy_flight_step()
            step_size = 0.01 * step * (best_nest - nests[random.randint(0, self.n_nests - 1)])
            new_nest = nests[random.randint(0, self.n_nests - 1)].copy()
            n_changes = min(int(np.linalg.norm(step_size)) + 1, self.n_tasks)
            indices_to_change = random.sample(range(self.n_tasks), n_changes)
            for idx in indices_to_change: new_nest[idx] = random.randint(0, self.num_cores - 1)
            f_new = self._fitness(new_nest)
            j = random.randint(0, self.n_nests - 1)
            if f_new > fitnesses[j]: nests[j], fitnesses[j] = new_nest, f_new
            if f_new > best_fitness: best_fitness, best_nest = f_new, new_nest
            n_abandon = int(self.pa * self.n_nests)
            if n_abandon > 0:
                sorted_indices = np.argsort(fitnesses)
                for k in range(n_abandon):
                    idx_to_abandon = sorted_indices[k]
                    nests[idx_to_abandon] = np.random.randint(0, self.num_cores, self.n_tasks)
                    fitnesses[idx_to_abandon] = self._fitness(nests[idx_to_abandon])
        return nests[np.argmax(fitnesses)]


# ==============================================================================
# SECTION 3: SIMULATION AND VISUALIZATION
# ==============================================================================

def run_phase_two_simulation(num_runs_per_config=3):
    configurations = [(8, 0.5), (8, 0.75), (8, 1.0), (16, 0.5), (16, 0.75), (16, 1.0), (32, 0.5), (32, 0.75), (32, 1.0)]
    results = {}
    for cores, util_per_core in configurations:
        print(f"Running Config: {cores} Cores, {util_per_core} Util/Core...")
        config_results = {'GA': [], 'CS': []}
        for i in range(num_runs_per_config):
            tasks = generate_tasks(num_tasks=int(4 * cores), total_utilization=cores * util_per_core)
            if not tasks: continue

            ga_scheduler = GeneticScheduler(tasks, cores)
            ga_assignment = ga_scheduler.evolve()
            if len(ga_assignment) > 0: config_results['GA'].append(
                {'tasks': tasks, 'metrics': get_full_metrics_for_solution(ga_assignment, tasks, cores)})

            cs_scheduler = CuckooScheduler(tasks, cores)
            cs_assignment = cs_scheduler.run()
            if len(cs_assignment) > 0: config_results['CS'].append(
                {'tasks': tasks, 'metrics': get_full_metrics_for_solution(cs_assignment, tasks, cores)})

        results[(cores, util_per_core)] = config_results
    return results


def visualize_detailed_results(results, algo_name, filename):
    fig, axs = plt.subplots(3, 2, figsize=(18, 24))
    fig.suptitle(f'Detailed Results for {algo_name}', fontsize=20, y=0.95)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    util_levels = sorted(list(set(k[1] for k in results.keys())))

    ax = axs[0, 0]
    sample_config_key = (16, 0.75)
    if sample_config_key in results and results[sample_config_key][algo_name]:
        last_run_metrics = results[sample_config_key][algo_name][-1]['metrics']
        task_qos_data = last_run_metrics.get('per_task_qos', [])
        if task_qos_data:
            ax.plot([t['id'] for t in task_qos_data], [t['qos'] for t in task_qos_data], 'o', alpha=0.7)
    ax.set_title('Individual Task QoS (Sample Run)');
    ax.set_xlabel('Task ID');
    ax.set_ylabel('Task QoS (%)');
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axs[0, 1]
    for cores in [8, 16, 32]:
        avg_qos_vals = [np.mean(
            [run['metrics']['system_qos'] for run in results.get((cores, u), {}).get(algo_name, [])]) if results.get(
            (cores, u), {}).get(algo_name) else np.nan for u in util_levels]
        ax.plot(util_levels, avg_qos_vals, '-o', label=f'{cores} cores')
    ax.set_title('Average System QoS vs. System Load');
    ax.set_xlabel('Target Utilization per Core');
    ax.set_ylabel('Average QoS (%)');
    ax.set_ylim(0, 101);
    ax.legend();
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axs[1, 0]
    for cores in [8, 16, 32]:
        avg_makespan_vals = [np.mean(
            [run['metrics']['makespan'] for run in results.get((cores, u), {}).get(algo_name, [])]) if results.get(
            (cores, u), {}).get(algo_name) else np.nan for u in util_levels]
        ax.plot(util_levels, avg_makespan_vals, '-o', label=f'{cores} cores')
    ax.set_title('Average Makespan vs. System Load');
    ax.set_xlabel('Target Utilization per Core');
    ax.set_ylabel('Makespan');
    ax.legend();
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axs[1, 1]
    all_core_utils = [util for cfg_res in results.values() for run in cfg_res[algo_name] for util in
                      run['metrics']['core_utils']]
    if all_core_utils: ax.hist(all_core_utils, bins=25, edgecolor='black')
    ax.set_title('Overall Core Utilization Distribution');
    ax.set_xlabel('Core Utilization');
    ax.set_ylabel('Frequency');
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axs[2, 0]
    configs_str = [f'{c[0]}c/{c[1]}u' for c in results.keys()]
    avg_sched_vals = [
        np.mean([run['metrics']['system_qos'] for run in results.get(cfg, {}).get(algo_name, [])]) if results.get(cfg,
                                                                                                                  {}).get(
            algo_name) else 0 for cfg in results.keys()]
    ax.bar(configs_str, avg_sched_vals)
    ax.set_title('System Schedulability (Avg. QoS)');
    ax.set_ylabel('Average QoS (%)');
    ax.tick_params(axis='x', rotation=45);
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax = axs[2, 1]
    ax.axis('off')
    if sample_config_key in results and results[sample_config_key][algo_name]:
        sample_tasks = results[sample_config_key][algo_name][-1]['tasks'][:6]
        task_data = [[t['id'], f"{t['execution']:.2f}", t['period'], f"{t['utilization']:.3f}"] for t in sample_tasks]
        col_labels = ['ID', 'Exec', 'Period', 'Util']
        table = ax.table(cellText=task_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False);
        table.set_fontsize(10);
        table.scale(1.1, 1.2)
        ax.set_title('Sample Task Parameters', pad=20)

    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_comparison_results(results, filename):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    util_levels = sorted(list(set(k[1] for k in results.keys())))

    for algo in ['GA', 'CS']:
        for cores in [8, 16, 32]:
            avg_qos = [np.mean(
                [run['metrics']['system_qos'] for run in results.get((cores, u), {}).get(algo, [])]) if results.get(
                (cores, u), {}).get(algo) else np.nan for u in util_levels]
            line_style = '-' if algo == 'GA' else '--'
            marker = 'o' if algo == 'GA' else 's'
            ax.plot(util_levels, avg_qos, marker=marker, linestyle=line_style, label=f'{cores} Cores - {algo}')

    ax.set_title('Algorithm Comparison: Average System QoS vs. System Load', fontsize=16)
    ax.set_xlabel('Target Utilization per Core', fontsize=12)
    ax.set_ylabel('Average System QoS (%)', fontsize=12)
    ax.set_ylim(0, 101);
    ax.legend();
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    start_time = a_time.time()
    simulation_results = run_phase_two_simulation(num_runs_per_config=3)

    if simulation_results:
        comparison_filename = os.path.join(RESULTS_DIR, 'phase_two_comparison.png')
        visualize_comparison_results(simulation_results, comparison_filename)
        print(f"Generated: {comparison_filename}")

        ga_details_filename = os.path.join(RESULTS_DIR, 'phase_two_GA_details.png')
        visualize_detailed_results(simulation_results, 'GA', ga_details_filename)
        print(f"Generated: {ga_details_filename}")

        cs_details_filename = os.path.join(RESULTS_DIR, 'phase_two_CS_details.png')
        visualize_detailed_results(simulation_results, 'CS', cs_details_filename)
        print(f"Generated: {cs_details_filename}")
    else:
        print("Simulation did not produce results.")

    print(f"Total execution time: {a_time.time() - start_time:.2f} seconds.")