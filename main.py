import numpy as np
import matplotlib.pyplot as plt
import random
from math import gcd
from functools import reduce
import heapq


def generate_tasks(num_tasks, total_utilization):
    utilizations = []
    remaining_util = total_utilization
    if remaining_util <= 0:
        return []

    for i in range(1, num_tasks):
        if num_tasks - i <= 0: continue
        next_util = remaining_util * random.random() ** (1 / (num_tasks - i))
        utilizations.append(remaining_util - next_util)
        remaining_util = next_util
    utilizations.append(remaining_util)

    tasks = []
    for i, util in enumerate(utilizations):
        period = random.choice([10, 20, 40, 50, 100, 200])
        execution = util * period
        if execution < 1: execution = 1
        deadline = period
        tasks.append({
            'id': i,
            'execution': execution,
            'period': period,
            'deadline': deadline,
            'utilization': execution / period if period > 0 else 0
        })
    return tasks


class GeneticScheduler:
    def __init__(self, tasks, num_cores, pop_size=50, elite_frac=0.2, mutation_rate=0.1, generations=100):
        self.tasks = tasks
        self.num_cores = num_cores
        self.pop_size = pop_size
        self.elite_size = int(elite_frac * pop_size)
        self.mutation_rate = mutation_rate
        self.generations = generations

    def initialize_population(self):
        return [np.random.randint(0, self.num_cores, len(self.tasks)) for _ in range(self.pop_size)]

    def fitness(self, chromosome):
        core_utils = np.zeros(self.num_cores)
        for task_idx, core_idx in enumerate(chromosome):
            core_utils[core_idx] += self.tasks[task_idx]['utilization']

        overload_penalty = sum(max(0, util - 1) * 100 for util in core_utils)
        balance_penalty = np.std(core_utils) * 10
        return 1 / (1 + overload_penalty + balance_penalty)

    def select_parents(self, population, fitnesses):
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return [random.choice(population) for _ in range(len(population) - self.elite_size)]

        probs = [f / total_fitness for f in fitnesses]
        parent_indices = np.random.choice(
            len(population),
            size=len(population) - self.elite_size,
            p=probs,
            replace=True
        )
        return [population[i] for i in parent_indices]

    def crossover(self, parent1, parent2):
        if len(parent1) <= 1: return parent1, parent2
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.randint(0, self.num_cores - 1)
        return chromosome

    def evolve(self):
        if not self.tasks or self.num_cores == 0: return []
        population = self.initialize_population()

        for _ in range(self.generations):
            fitnesses = [self.fitness(chromo) for chromo in population]

            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            new_population = [population[i] for i in elite_indices]

            parents = self.select_parents(population, fitnesses)
            random.shuffle(parents)

            children = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self.crossover(parents[i], parents[i + 1])
                    children.append(self.mutate(child1))
                    children.append(self.mutate(child2))
            new_population.extend(children)

            population = new_population[:self.pop_size]

        final_fitnesses = [self.fitness(chromo) for chromo in population]
        return population[np.argmax(final_fitnesses)]


def edf_schedule_on_core(tasks_on_core, hyperperiod):
    if not tasks_on_core: return {}, 0

    if sum(t['utilization'] for t in tasks_on_core) > 1:
        job_results = {}
        for task in tasks_on_core:
            job_results[task['id']] = {'finish_times': [], 'deadlines': []}
            num_jobs = hyperperiod // task['period'] if task['period'] > 0 else 0
            for i in range(num_jobs):
                job_results[task['id']]['finish_times'].append(float('inf'))
                job_results[task['id']]['deadlines'].append((i + 1) * task['period'])
        return job_results, float('inf')

    time = 0
    ready_queue = []
    job_arrivals = []
    job_counter = 0

    for task in tasks_on_core:
        if task['period'] > 0:
            for i in range(hyperperiod // task['period']):
                arrival_time = i * task['period']
                absolute_deadline = arrival_time + task['deadline']
                job_arrivals.append((arrival_time, task['execution'], absolute_deadline, task['id'], job_counter))
                job_counter += 1
    job_arrivals.sort()

    job_results = {task['id']: {'finish_times': [], 'deadlines': []} for task in tasks_on_core}
    current_job = None

    while time < hyperperiod * 2:
        while job_arrivals and job_arrivals[0][0] <= time:
            arrival, exec_time, deadline, task_id, job_id = job_arrivals.pop(0)
            heapq.heappush(ready_queue, (deadline, exec_time, task_id, job_id))

        if current_job is None and ready_queue:
            deadline, exec_time, task_id, job_id = heapq.heappop(ready_queue)
            current_job = {'deadline': deadline, 'remaining_exec': exec_time, 'task_id': task_id, 'job_id': job_id}

        if current_job:
            current_job['remaining_exec'] -= 1
            if current_job['remaining_exec'] <= 0:
                task_id = current_job['task_id']
                job_results[task_id]['finish_times'].append(time + 1)
                job_results[task_id]['deadlines'].append(current_job['deadline'])
                current_job = None

        time += 1
        if not current_job and not ready_queue and not job_arrivals: break

    latest_finish_time = max((finish for res in job_results.values() for finish in res['finish_times']), default=0)
    return job_results, latest_finish_time


def calculate_qos(finish_time, deadline):
    if finish_time <= deadline: return 100
    if deadline <= 0: return 0
    if finish_time >= 2 * deadline: return 0
    return 100 * (1 - (finish_time - deadline) / deadline)


def calculate_metrics_with_edf(tasks, assignment, num_cores):
    core_assignments = {i: [] for i in range(num_cores)}
    for task_idx, core_idx in enumerate(assignment):
        core_assignments[core_idx].append(tasks[task_idx])

    if not tasks: return {}
    periods = [t['period'] for t in tasks if t['period'] > 0]
    if not periods: return {}
    hyperperiod = reduce(lambda a, b: a * b // gcd(a, b) if a > 0 and b > 0 else a or b, periods, 1)

    per_task_qos_list = []
    all_core_utils = [sum(t['utilization'] for t in core_assignments[i]) for i in range(num_cores)]
    total_makespan = 0

    for core_id in range(num_cores):
        tasks_on_core = core_assignments[core_id]
        job_results, core_makespan = edf_schedule_on_core(tasks_on_core, hyperperiod)
        total_makespan = max(total_makespan, core_makespan)

        for task in tasks_on_core:
            task_id = task['id']
            if task_id in job_results and job_results[task_id]['finish_times']:
                avg_qos = np.mean([
                    calculate_qos(ft, dl) for ft, dl
                    in zip(job_results[task_id]['finish_times'], job_results[task_id]['deadlines'])
                ]) if job_results[task_id]['finish_times'] else 0
                per_task_qos_list.append({'id': task_id, 'qos': avg_qos})
            else:
                per_task_qos_list.append({'id': task_id, 'qos': 0})

    per_task_qos_list.sort(key=lambda x: x['id'])
    system_qos_val = np.mean([item['qos'] for item in per_task_qos_list]) if per_task_qos_list else 0

    return {'core_utils': all_core_utils, 'makespan': total_makespan, 'per_task_qos': per_task_qos_list,
            'system_qos': system_qos_val}


def run_simulation(num_runs_per_config=3):
    configurations = [
        (8, 0.25), (8, 0.5), (8, 0.75), (8, 1.0),
        (16, 0.25), (16, 0.5), (16, 0.75), (16, 1.0),
        (32, 0.25), (32, 0.5), (32, 0.75), (32, 1.0)
    ]

    final_results = {}
    for cores, util_per_core in configurations:
        print(f"Running: {cores} cores, util/core: {util_per_core}...")

        all_runs_data = []
        for i in range(num_runs_per_config):
            total_util = cores * util_per_core
            tasks = generate_tasks(num_tasks=int(4 * cores), total_utilization=total_util)
            if not tasks: continue

            scheduler = GeneticScheduler(tasks, cores)
            assignment = scheduler.evolve()

            if len(assignment) > 0:
                metrics = calculate_metrics_with_edf(tasks, assignment, cores)
                all_runs_data.append({'tasks': tasks, 'metrics': metrics})

        if all_runs_data:
            final_results[(cores, util_per_core)] = {'runs': all_runs_data}
    return final_results


def visualize_results(results, num_runs=3):
    fig, axs = plt.subplots(3, 2, figsize=(18, 24))
    fig.suptitle('Phase One Results: Genetic Algorithm with EDF Simulation', fontsize=20)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    run_colors = plt.cm.viridis(np.linspace(0, 1, num_runs))
    util_levels = [0.25, 0.5, 0.75, 1.0]

    ax = axs[0, 0]
    sample_config_key = (16, 0.75)
    if sample_config_key in results:
        for i, run_data in enumerate(results[sample_config_key]['runs']):
            task_qos_data = run_data['metrics']['per_task_qos']
            if task_qos_data:
                ax.plot([t['id'] for t in task_qos_data], [t['qos'] for t in task_qos_data], 'o', alpha=0.7,
                        color=run_colors[i], label=f'Run {i + 1}')
    ax.set_title('Task QoS for Each Run (Sample Config: 16c, u=0.75)')
    ax.set_xlabel('Task ID');
    ax.set_ylabel('Task QoS (%)');
    ax.legend();
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axs[0, 1]
    for cores in [8, 16, 32]:
        avg_qos_vals = []
        for util in util_levels:
            if (cores, util) in results:
                runs = results[(cores, util)]['runs']
                all_qos = [r['metrics']['system_qos'] for r in runs]
                avg_qos_vals.append(np.mean(all_qos))
                for i, qos_val in enumerate(all_qos):
                    ax.plot(util, qos_val, 'o', color=run_colors[i], alpha=0.3)
            else:
                avg_qos_vals.append(np.nan)
        ax.plot(util_levels, avg_qos_vals, '-o', label=f'{cores} cores (Avg)')
    ax.set_title('Average System QoS vs. System Load (with individual runs)');
    ax.set_xlabel('Target Utilization per Core');
    ax.set_ylabel('System QoS (%)');
    ax.set_ylim(0, 101);
    ax.legend();
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axs[1, 0]
    for cores in [8, 16, 32]:
        avg_makespan_vals = []
        for util in util_levels:
            if (cores, util) in results:
                runs = results[(cores, util)]['runs']
                all_makespans = [r['metrics']['makespan'] for r in runs]
                avg_makespan_vals.append(np.mean(all_makespans))
                for i, ms_val in enumerate(all_makespans):
                    ax.plot(util, ms_val, 'o', color=run_colors[i], alpha=0.3)
            else:
                avg_makespan_vals.append(np.nan)
        ax.plot(util_levels, avg_makespan_vals, '-o', label=f'{cores} cores (Avg)')
    ax.set_title('Average Makespan vs. System Load (with individual runs)');
    ax.set_xlabel('Target Utilization per Core');
    ax.set_ylabel('Makespan (time units)');
    ax.legend();
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axs[1, 1]
    all_core_utils = [util for data in results.values() for run in data['runs'] for util in
                      run['metrics']['core_utils']]
    ax.hist(all_core_utils, bins=25, edgecolor='black')
    ax.set_title('Overall Core Utilization Distribution (All Runs)');
    ax.set_xlabel('Core Utilization');
    ax.set_ylabel('Frequency');
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axs[2, 0]
    configs_str = [f'{c[0]}c/{c[1]}u' for c in results.keys()]
    avg_sched_vals = [np.mean([r['metrics']['system_qos'] for r in d['runs']]) for d in results.values()]
    ax.bar(configs_str, avg_sched_vals)
    ax.set_title('System Schedulability (Average System QoS)');
    ax.set_ylabel('Average QoS (%)');
    ax.tick_params(axis='x', rotation=45);
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax = axs[2, 1]
    ax.axis('off')
    table_data = []
    if sample_config_key in results:
        for i, run_data in enumerate(results[sample_config_key]['runs']):
            for task in run_data['tasks'][:2]:
                table_data.append(
                    [i + 1, task['id'], f"{task['execution']:.2f}", task['period'], f"{task['utilization']:.3f}"])
    if table_data:
        col_labels = ['Run', 'ID', 'Exec', 'Period', 'Util']
        table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False);
        table.set_fontsize(10);
        table.scale(1.1, 1.2)
        ax.set_title('Sample Tasks (Config: 16c, u=0.75)', pad=20)

    plt.savefig('phase_one_results_all_runs.png', bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    NUM_RUNS = 3
    simulation_results = run_simulation(num_runs_per_config=NUM_RUNS)
    if simulation_results:
        visualize_results(simulation_results, num_runs=NUM_RUNS)
        print("Simulation finished. Check 'phase_one_results_all_runs.png'.")
    else:
        print("Simulation did not produce any results.")