import numpy as np
import matplotlib.pyplot as plt
import random
from math import gcd
from functools import reduce
from collections import defaultdict

def generate_tasks(num_tasks, total_utilization):
    utilizations = []
    remaining_util = total_utilization

    for i in range(1, num_tasks):
        next_util = remaining_util * random.random() ** (1 / (num_tasks - i))
        utilizations.append(remaining_util - next_util)
        remaining_util = next_util
    utilizations.append(remaining_util)

    tasks = []
    for i, util in enumerate(utilizations):
        period = random.choice([10, 20, 40, 50, 100])
        execution = util * period
        deadline = period
        tasks.append({
            'id': i,
            'execution': execution,
            'period': period,
            'deadline': deadline,
            'utilization': util
        })
    return tasks

class GeneticScheduler:
    def __init__(self, tasks, num_cores, pop_size=50, elite=0.2, mutation_rate=0.1, generations=100):
        self.tasks = tasks
        self.num_cores = num_cores
        self.pop_size = pop_size
        self.elite = int(elite * pop_size)
        self.mutation_rate = mutation_rate
        self.generations = generations

    def initialize_population(self):
        return [np.random.randint(0, self.num_cores, len(self.tasks))
                for _ in range(self.pop_size)]

    def fitness(self, chromosome):
        core_utils = np.zeros(self.num_cores)
        for task_idx, core_idx in enumerate(chromosome):
            core_utils[core_idx] += self.tasks[task_idx]['utilization']

        overload_penalty = sum(max(0, util - 1) * 100 for util in core_utils)
        balance = np.std(core_utils) * 10
        return 1 / (1 + overload_penalty + balance)

    def select_parents(self, population, fitnesses):
        total_fitness = sum(fitnesses)
        probs = [f / total_fitness for f in fitnesses]
        parents = np.random.choice(
            range(len(population)),
            size=len(population) - self.elite,
            p=probs
        )
        return [population[i] for i in parents]

    def crossover(self, parent1, parent2):
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
        population = self.initialize_population()

        for _ in range(self.generations):
            fitnesses = [self.fitness(chromo) for chromo in population]

            elite_indices = np.argsort(fitnesses)[-self.elite:]
            new_population = [population[i] for i in elite_indices]

            parents = self.select_parents(population, fitnesses)
            random.shuffle(parents)

            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self.crossover(parents[i], parents[i + 1])
                    new_population += [self.mutate(child1), self.mutate(child2)]

            population = new_population

        fitnesses = [self.fitness(chromo) for chromo in population]
        return population[np.argmax(fitnesses)]

def calculate_metrics(tasks, assignment, num_cores):
    core_utils = [0] * num_cores
    for task_idx, core_idx in enumerate(assignment):
        core_utils[core_idx] += tasks[task_idx]['utilization']

    hyperperiod = reduce(lambda a, b: a * b // gcd(a, b),
                         [t['period'] for t in tasks], 1)
    makespan = hyperperiod * max(core_utils)

    task_qos = [100 if tasks[i]['utilization'] <= 1 else 0
                for i in range(len(tasks))]
    system_qos = 100 if all(u <= 1 for u in core_utils) else 0

    return {
        'core_utils': core_utils,
        'makespan': makespan,
        'task_qos': task_qos,
        'system_qos': system_qos,
        'hyperperiod': hyperperiod
    }

def run_simulation():
    configurations = [
        (8, 0.25), (8, 0.5), (8, 0.75), (8, 1.0),
        (16, 0.25), (16, 0.5), (16, 0.75), (16, 1.0),
        (32, 0.25), (32, 0.5), (32, 0.75), (32, 1.0)
    ]

    results = {}

    for cores, util_per_core in configurations:
        total_util = cores * util_per_core
        tasks = generate_tasks(3 * cores, total_util)

        scheduler = GeneticScheduler(tasks, cores)
        assignment = scheduler.evolve()

        metrics = calculate_metrics(tasks, assignment, cores)
        results[(cores, util_per_core)] = {
            'tasks': tasks,
            'assignment': assignment,
            'metrics': metrics
        }

    return results

def visualize_results(results):
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    plt.subplots_adjust(hspace=0.5)

    system_qos_data = {8: [], 16: [], 32: []}
    makespan_data = {8: [], 16: [], 32: []}
    core_util_data = []

    for (cores, util_per_core), data in results.items():
        metrics = data['metrics']
        tasks = data['tasks']

        axs[0, 0].plot(
            [t['id'] for t in tasks],
            metrics['task_qos'],
            'o',
            label=f'{cores} cores, u={util_per_core}'
        )

        system_qos_data[cores].append(metrics['system_qos'])

        makespan_data[cores].append(metrics['makespan'])

        for i, util in enumerate(metrics['core_utils']):
            core_util_data.append({
                'cores': cores,
                'util_per_core': util_per_core,
                'util': util
            })

    for cores, qos_vals in system_qos_data.items():
        axs[0, 1].plot(
            [0.25, 0.5, 0.75, 1.0],
            qos_vals,
            '-o',
            label=f'{cores} cores'
        )
    axs[0, 1].set_title('System QoS')
    axs[0, 1].set_xlabel('Utilization per Core')
    axs[0, 1].set_ylabel('QoS (%)')
    axs[0, 1].legend()

    for cores, makespan_vals in makespan_data.items():
        axs[1, 0].plot(
            [0.25, 0.5, 0.75, 1.0],
            makespan_vals,
            '-o',
            label=f'{cores} cores'
        )
    axs[1, 0].set_title('Makespan Comparison')
    axs[1, 0].set_xlabel('Utilization per Core')
    axs[1, 0].set_ylabel('Makespan')
    axs[1, 0].legend()

    core_util_vals = [d['util'] for d in core_util_data]
    axs[1, 1].hist(core_util_vals, bins=20)
    axs[1, 1].set_title('Core Utilization Distribution')
    axs[1, 1].set_xlabel('Utilization')
    axs[1, 1].set_ylabel('Frequency')

    schedulability = [
        (config, data['metrics']['system_qos'])
        for config, data in results.items()
    ]
    configs = [f'{c[0]}c/{c[1]}u' for c in results.keys()]
    schedulability_vals = [s[1] for s in schedulability]
    axs[2, 0].bar(configs, schedulability_vals)
    axs[2, 0].set_title('System Schedulability')
    axs[2, 0].set_ylabel('QoS (%)')
    plt.xticks(rotation=45)

    sample_tasks = list(results.values())[0]['tasks'][:5]
    task_data = [[t['id'], t['execution'], t['period'], t['deadline'], t['utilization']]
                 for t in sample_tasks]
    axs[2, 1].axis('off')
    axs[2, 1].table(
        cellText=task_data,
        colLabels=['ID', 'Exec', 'Period', 'Deadline', 'Util'],
        loc='center'
    )
    axs[2, 1].set_title('Sample Task Parameters')

    plt.savefig('phase_one_results.png', bbox_inches='tight')
    plt.close()

    return results

if __name__ == "__main__":
    results = run_simulation()
    final_results = visualize_results(results)