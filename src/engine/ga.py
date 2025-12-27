import json
import os
import random
import logging
import wandb
from typing import List, Dict, Any
from dataclasses import asdict

from ..core.search_space import SearchSpace, Genome
from ..core.evaluator import Evaluator, FitnessMetrics
from .pareto import get_pareto_front
from .operators import crossover_uniform, mutate_random_gene

logger = logging.getLogger(__name__)

class GAEngine:
    def __init__(self, 
                 search_space: SearchSpace, 
                 evaluator: Evaluator,
                 config: Dict[str, Any],
                 tracking_config: Dict[str, Any] = None,
                 output_dir: str = "./data/logs"):
        
        self.space = search_space
        self.evaluator = evaluator
        self.cfg = config
        self.tracking_cfg = tracking_config or {"enabled": False}
        self.output_dir = output_dir
        
        self.population: List[Genome] = []
        self.fitness_cache: Dict[str, FitnessMetrics] = {}
        self.history: List[Dict] = []
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize WandB if enabled
        if self.tracking_cfg.get("enabled", False):
            wandb.init(
                project=self.tracking_cfg.get("project", "empas"),
                config={**self.cfg, "search_space": self.space.__class__.__name__},
                tags=self.tracking_cfg.get("tags", []),
                dir=self.output_dir
            )

    def _get_fitness_str(self, genome: Genome) -> str:
        return str(genome.genes)

    def evaluate_population(self, population: List[Genome]) -> List[FitnessMetrics]:
        fitnesses = []
        for genome in population:
            key = self._get_fitness_str(genome)
            if key in self.fitness_cache:
                fitnesses.append(self.fitness_cache[key])
            else:
                fit = self.evaluator.evaluate(genome)
                self.fitness_cache[key] = fit
                fitnesses.append(fit)
        return fitnesses

    def select_tournament(self, population: List[Genome], fitnesses: List[FitnessMetrics]) -> Genome:
        indices = random.sample(range(len(population)), self.cfg['tournament_size'])
        best_idx = indices[0]
        for idx in indices[1:]:
            if self._dominates(fitnesses[idx], fitnesses[best_idx]):
                best_idx = idx
            elif not self._dominates(fitnesses[best_idx], fitnesses[idx]):
                if random.random() < 0.5:
                    best_idx = idx
        return population[best_idx]

    def _dominates(self, f1: FitnessMetrics, f2: FitnessMetrics) -> bool:
        from .pareto import dominates
        return dominates(f1, f2)

    def step(self, generation: int):
        logger.info(f"--- Generation {generation} ---")
        
        # 1. Evaluate
        fitnesses = self.evaluate_population(self.population)
        
        # 2. Pareto Front
        pareto_indices = get_pareto_front(fitnesses)
        elites = [self.population[i] for i in pareto_indices]
        
        # 3. Stats & Logging
        losses = [f.validation_loss for f in fitnesses]
        vrams = [f.vram_peak_mb for f in fitnesses]
        
        stats = {
            "gen": generation,
            "min_loss": min(losses),
            "avg_loss": sum(losses) / len(losses),
            "min_vram": min(vrams),
            "avg_vram": sum(vrams) / len(vrams),
            "num_elites": len(elites)
        }
        
        logger.info(f"Stats: Loss={stats['min_loss']:.4f}, VRAM={stats['min_vram']:.0f}MB, Elites={stats['num_elites']}")
        self.history.append(stats)
        
        # WandB Logging
        if self.tracking_cfg.get("enabled", False):
            # Log metrics
            wandb.log(stats, step=generation)
            
            # Create a Scatter Plot (Loss vs VRAM)
            table_data = []
            for genome, fit in zip(self.population, fitnesses):
                is_pareto = genome in elites
                table_data.append([fit.validation_loss, fit.vram_peak_mb, fit.latency_ms, str(genome.genes), is_pareto])
            
            table = wandb.Table(data=table_data, columns=["loss", "vram", "latency", "genes", "is_pareto"])
            
            wandb.log({
                "population_plot": wandb.plot.scatter(
                    table, "vram", "loss", title=f"Gen {generation}: Loss vs VRAM"
                )
            }, step=generation)

        # 4. Evolution (Selection, Crossover, Mutation)
        next_pop = []
        
        # Elitism
        num_elites_to_keep = min(len(elites), self.cfg['pop_size'] // 2)
        random.shuffle(elites)
        next_pop.extend(elites[:num_elites_to_keep])
        
        while len(next_pop) < self.cfg['pop_size']:
            p1 = self.select_tournament(self.population, fitnesses)
            p2 = self.select_tournament(self.population, fitnesses)
            
            if random.random() < self.cfg['crossover_rate']:
                c1, c2 = crossover_uniform(p1, p2)
            else:
                c1, c2 = p1, p2
                
            c1 = mutate_random_gene(c1, self.space, self.cfg['mutation_rate'])
            c2 = mutate_random_gene(c2, self.space, self.cfg['mutation_rate'])
            
            next_pop.append(c1)
            if len(next_pop) < self.cfg['pop_size']:
                next_pop.append(c2)
        
        self.population = next_pop

    def run(self):
        logger.info("Initializing population...")
        self.population = [self.space.sample() for _ in range(self.cfg['pop_size'])]
        
        for gen in range(1, self.cfg['n_generations'] + 1):
            self.step(gen)
            
        if self.tracking_cfg.get("enabled", False):
            wandb.finish()
        logger.info("Search Complete.")