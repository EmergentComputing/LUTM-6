"""NumPy island GA with gap-based tournaments and neutral child-first elitism."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from uuid import uuid4

import numpy as np

from machine import TABLE_HASH, INTERFACE_VERSION
from utils import (Dataset, Evaluation, SimulationConfig, SYMBOL_IDS, MATCH, DENOM,
                   FAILED, EXACT, SUCCESS, TIME, P_SPACE, X_SPACE, integer,
                   pack_programs, program_string, validate_population)


@dataclass(frozen=True)
class GAConfig:
    islands: int = 4
    population_per_island: int = 256
    alphabet: str = "01#"
    elite_fraction: float = .03
    cross_island_fraction: float = .02
    cross_island_interval: int = 1200
    random_immigrant_fraction: float = .15
    delta_thresholds: tuple = (.05, .20)
    winner_probabilities: tuple = (.53, .58)
    base_winner_probability: float = .80
    p_crossover: float = .50
    crossover_segments: int = 3
    crossover_min_length: int = 1
    crossover_max_length: int = 20
    p_insert: float = .04
    p_delete: float = .06
    p_symbol_mutation: float = .02
    k_penalty: float = .50
    seed: object = 4
    stop_first_exact: bool = True

    def __post_init__(self):
        object.__setattr__(self, "delta_thresholds", tuple(self.delta_thresholds))
        object.__setattr__(self, "winner_probabilities", tuple(self.winner_probabilities))
        integer("islands", self.islands, 1)
        integer("population_per_island", self.population_per_island, 2)
        integer("cross_island_interval", self.cross_island_interval, 1)
        integer("crossover_segments", self.crossover_segments, 0)
        integer("crossover_min_length", self.crossover_min_length, 1)
        integer("crossover_max_length", self.crossover_max_length, self.crossover_min_length)
        if self.alphabet not in ("01#", "B01#"):
            raise ValueError("alphabet must be '01#' or 'B01#'")
        for name in ("elite_fraction", "cross_island_fraction", "random_immigrant_fraction",
                     "base_winner_probability", "p_crossover", "p_insert", "p_delete", "p_symbol_mutation"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        thresholds = np.asarray(self.delta_thresholds, dtype=np.float64)
        probabilities = np.asarray(self.winner_probabilities, dtype=np.float64)
        if (thresholds.ndim != 1 or probabilities.shape != thresholds.shape or
                np.any(~np.isfinite(thresholds)) or np.any(thresholds <= 0) or
                np.any(np.diff(thresholds) <= 0)):
            raise ValueError("delta thresholds must be finite, positive, increasing and match probabilities")
        if (np.any(~np.isfinite(probabilities)) or np.any(probabilities < .5) or
                np.any(probabilities > 1) or self.base_winner_probability < .5):
            raise ValueError("winner probabilities must be in [0.5,1]")
        if self.p_insert + self.p_delete > 1:
            raise ValueError("p_insert + p_delete must not exceed 1")
        if not np.isfinite(self.k_penalty) or self.k_penalty < 0:
            raise ValueError("k_penalty must be finite and nonnegative")
        if self.seed is not None:
            integer("seed", self.seed, 0, 2**64 - 1)
        if type(self.stop_first_exact) is not bool:
            raise ValueError("stop_first_exact must be a bool")
        if self.islands == 1 and self.cross_island_fraction != 0:
            raise ValueError("a one-island run needs cross_island_fraction=0")
        if sum(self.slot_counts(True)[:3]) > self.population_per_island:
            raise ValueError("elite, migrant and immigrant slots exceed island population")

    def slot_counts(self, migration):
        n = self.population_per_island
        elite = max(1, int(self.elite_fraction * n))
        migrant = int(self.cross_island_fraction * n) if migration else 0
        immigrant = int(self.random_immigrant_fraction * n)
        return elite, migrant, immigrant, n - elite - migrant - immigrant

    @property
    def symbols(self):
        return np.array([SYMBOL_IDS[s] for s in self.alphabet], dtype=np.uint8)


def winner_probability(delta, config):
    delta = np.asarray(delta, dtype=np.float64)
    index = np.searchsorted(config.delta_thresholds, delta, side="left")
    probabilities = np.array(config.winner_probabilities + (config.base_winner_probability,))
    return np.where(delta == 0, .5, probabilities[index])


def rank_population(fitness, is_child):
    index = np.broadcast_to(np.arange(fitness.shape[1]), fitness.shape)
    return np.lexsort((index, -is_child.astype(np.int8), -fitness), axis=1)


def random_programs(rng, shape, width, symbols):
    lengths = rng.integers(0, width + 1, size=shape, dtype=np.int32)
    values = symbols[rng.integers(0, len(symbols), size=(*shape, width))]
    values[np.arange(width) >= lengths[..., None]] = 0
    return values, lengths


def tournament(rng, fitness, count, config):
    i, n = fitness.shape
    contenders = rng.integers(0, n, size=(i, count, 2))
    scores = fitness[np.arange(i)[:, None, None], contenders]
    best = (scores[..., 1] > scores[..., 0]).astype(np.int64)
    probability = winner_probability(np.abs(scores[..., 1] - scores[..., 0]), config)
    chosen = np.where(rng.random((i, count)) < probability, best, 1 - best)
    return np.take_along_axis(contenders, chosen[..., None], axis=2)[..., 0]


def crossover(rng, parent1, length1, parent2, length2, config):
    child = parent1.copy()
    overlap = np.minimum(length1, length2)
    crossing = (rng.random(length1.shape) < config.p_crossover) & (overlap > 0)
    low = np.minimum(config.crossover_min_length, np.maximum(1, overlap))
    high = np.minimum(config.crossover_max_length, np.maximum(1, overlap))
    coordinates = np.arange(child.shape[-1])
    for _ in range(config.crossover_segments):
        length = low + (rng.random(low.shape) * (high - low + 1)).astype(np.int32)
        start = (rng.random(low.shape) * (np.maximum(1, overlap) - length + 1)).astype(np.int32)
        mask = crossing[..., None] & (coordinates >= start[..., None]) & (coordinates < (start + length)[..., None])
        np.copyto(child, parent2, where=mask)
    return child, length1.copy()


def mutate(rng, values, lengths, config):
    """Structural insert/delete/stay first, then independent active-locus changes."""
    flat = values.reshape(-1, values.shape[-1])
    sizes = lengths.reshape(-1)
    draw = rng.random(len(sizes))
    inserting = (draw < config.p_insert) & (sizes < values.shape[-1])
    deleting = (draw >= config.p_insert) & (draw < config.p_insert + config.p_delete) & (sizes > 0)
    inserted_symbols = rng.choice(config.symbols, size=len(sizes))
    rows = np.flatnonzero(inserting)
    flat[rows, sizes[rows]] = inserted_symbols[rows]
    sizes[inserting] += 1
    sizes[deleting] -= 1
    rows = np.flatnonzero(deleting)
    flat[rows, sizes[rows]] = 0
    active = np.arange(flat.shape[1]) < sizes[:, None]
    changing = active & (rng.random(flat.shape) < config.p_symbol_mutation)
    rank = np.full(4, -1, dtype=np.int32)
    rank[config.symbols] = np.arange(len(config.symbols))
    offset = rng.integers(1, len(config.symbols), size=flat.shape)
    replacement = config.symbols[(rank[flat] + offset) % len(config.symbols)]
    np.copyto(flat, replacement, where=changing)
    return values, lengths


def reproduce(values, lengths, evaluation, is_child, generation, config, rng):
    """Preserve local elites; sample unchanged migrants from whole other islands."""
    islands, n, width = values.shape
    fitness = evaluation.fitness.reshape(islands, n)
    order = rank_population(fitness, is_child)
    e, m, r, children = config.slot_counts(generation % config.cross_island_interval == 0)
    new = np.zeros_like(values)
    sizes = np.zeros_like(lengths)
    child_flags = np.zeros_like(is_child)
    island = np.arange(islands)[:, None]
    new[:, :e], sizes[:, :e] = values[island, order[:, :e]], lengths[island, order[:, :e]]
    position = e
    if m:
        source = rng.integers(0, islands - 1, size=(islands, m))
        source += source >= island
        # Independent uniform draws with replacement; no donor fitness filter.
        source_index = rng.integers(0, n, size=(islands, m))
        new[:, position:position + m] = values[source, source_index]
        sizes[:, position:position + m] = lengths[source, source_index]
        position += m
    if r:
        new[:, position:position + r], sizes[:, position:position + r] = random_programs(rng, (islands, r), width, config.symbols)
        position += r
    if children:
        a = tournament(rng, fitness, children, config)
        b = tournament(rng, fitness, children, config)
        offspring, child_lengths = crossover(rng, values[island, a], lengths[island, a],
                                            values[island, b], lengths[island, b], config)
        mutate(rng, offspring, child_lengths, config)
        new[:, position:], sizes[:, position:] = offspring, child_lengths
        child_flags[:, position:] = True
    return new, sizes, child_flags


def _atomic_npz(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="checkpoint-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as file:
            # A JSON scalar is loadable with allow_pickle=False.
            np.savez_compressed(file, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_history(history):
    return [{k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in row.items()}
            for row in history]


class Trainer:
    """A committed generation is never mutated while its successor is evaluated."""

    def __init__(self, simulator, dataset, config=None, test_dataset=None, run_id=None,
                 runs_directory="runs", initial_programs=None):
        self.simulator, self.dataset = simulator, dataset
        self.config = GAConfig() if config is None else config
        self.test_dataset = test_dataset
        simulator.config.check_dataset(dataset)
        if test_dataset is not None:
            simulator.config.check_dataset(test_dataset)
        self.run_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
        if not self.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in self.run_id):
            raise ValueError("run_id may contain only ASCII letters, digits, '-' and '_'")
        self.runs_directory = Path(runs_directory)
        self.rng = np.random.default_rng(self.config.seed)
        self.generation, self.elapsed = 0, 0.0
        shape = (self.config.islands, self.config.population_per_island)
        self.population, self.lengths = random_programs(self.rng, shape, simulator.config.max_program_length, self.config.symbols)
        # Explicit optional seeds are not used by default; useful for a chosen experiment.
        if initial_programs is not None:
            values, lengths = pack_programs(initial_programs, simulator.config.max_program_length)
            if len(values) > self.population.size // self.population.shape[-1]:
                raise ValueError("more seed programs than population slots")
            active = np.arange(values.shape[1]) < lengths[:, None]
            if np.any(~np.isin(values[active], self.config.symbols)):
                raise ValueError("seed programs contain symbols outside the search alphabet")
            self.population.reshape(-1, values.shape[1])[:len(values)] = values
            self.lengths.reshape(-1)[:len(values)] = lengths
        self.is_child = np.zeros(shape, dtype=bool)
        start = perf_counter()
        self.evaluation = self._evaluate(self.population, self.lengths)
        self.elapsed += perf_counter() - start
        self.best = None
        self.history = []
        self._update_best()
        self.record_diagnostics()

    @property
    def checkpoint_path(self):
        return self.runs_directory / self.run_id / "checkpoint.npz"

    @property
    def solved(self):
        return bool(np.any(self.evaluation.exact))

    def _evaluate(self, population, lengths):
        return self.simulator.evaluate(population.reshape(-1, population.shape[-1]), lengths.ravel(),
                                       self.dataset, self.config.k_penalty)

    def _update_best(self):
        fitness = self.evaluation.fitness
        i = int(np.argmax(fitness))
        if self.best is None or fitness[i] > self.best["fitness"]:
            self.best = dict(program=self.population.reshape(-1, self.population.shape[-1])[i].copy(),
                             length=int(self.lengths.ravel()[i]), fitness=float(fitness[i]),
                             accuracy=float(self.evaluation.accuracy[i]),
                             invalid_fraction=float(self.evaluation.invalid_fraction[i]),
                             exact=bool(self.evaluation.exact[i]), generation=self.generation)

    def step(self):
        if self.config.stop_first_exact and self.solved:
            return False
        rng_state = deepcopy(self.rng.bit_generator.state)
        start = perf_counter()
        try:
            values, lengths, flags = reproduce(self.population, self.lengths, self.evaluation,
                                               self.is_child, self.generation + 1, self.config, self.rng)
            evaluation = self._evaluate(values, lengths)
            previous = (self.population, self.lengths, self.is_child, self.evaluation,
                        self.generation, self.elapsed, self.best)
            try:
                self.population, self.lengths, self.is_child = values, lengths, flags
                self.evaluation = evaluation
                self.generation += 1
                self.elapsed += perf_counter() - start
                self._update_best()
            except BaseException:
                (self.population, self.lengths, self.is_child, self.evaluation,
                 self.generation, self.elapsed, self.best) = previous
                raise
        except BaseException:
            self.rng.bit_generator.state = rng_state
            raise
        return True

    def champions(self):
        order = rank_population(self.evaluation.fitness.reshape(self.lengths.shape), self.is_child)
        flat = np.arange(self.config.islands) * self.config.population_per_island + order[:, 0]
        return flat

    def record_diagnostics(self):
        if self.history and self.history[-1]["generation"] == self.generation:
            return self.history[-1]
        ev = self.evaluation
        champion = self.champions()
        best = int(np.argmax(ev.fitness))
        row = dict(generation=self.generation, elapsed=self.elapsed,
                   current_best=float(ev.fitness[best]), global_best=self.best["fitness"],
                   best_accuracy=float(ev.accuracy[best]),
                   champion_invalid=float(ev.invalid_fraction[champion].mean()),
                   population_invalid=float(ev.invalid_fraction.mean()), exact_count=int(ev.exact.sum()))
        metrics = {"champion_fitness": ev.fitness[champion],
                   "time": ev.resource_mean(TIME)[champion],
                   "p_space": ev.resource_mean(P_SPACE)[champion],
                   "x_space": ev.resource_mean(X_SPACE)[champion]}
        for metric, values in metrics.items():
            finite = values[np.isfinite(values)]
            row[metric + "_mean"] = float(finite.mean()) if len(finite) else float("nan")
            for quantile in (20, 80):
                row[f"{metric}_q{quantile}"] = float(np.quantile(finite, quantile / 100)) if len(finite) else float("nan")
        self.history.append(row)
        return row

    def print_progress(self):
        ev = self.evaluation
        i = int(np.argmax(ev.fitness))
        current = program_string(self.population.reshape(-1, self.population.shape[-1])[i], self.lengths.ravel()[i])
        archive = program_string(self.best["program"], self.best["length"])
        print(f"gen={self.generation} current={ev.fitness[i]:.6f} global={self.best['fitness']:.6f} "
              f"accuracy={ev.accuracy[i]:.4f} invalid={ev.invalid_fraction[i]:.4f} "
              f"population_invalid={ev.invalid_fraction.mean():.4f} exact={int(ev.exact.sum())} "
              f"length={len(current)} elapsed={self.elapsed:.1f}s\n"
              f"  current: {current}\n  best ever: {archive}", flush=True)

    def run(self, generations, print_every=10, diagnostics_every=10, checkpoint_every=100):
        """Run up to this many ADDITIONAL generations, or first exact if enabled."""
        integer("generations", generations)
        for name, value in (("print_every", print_every), ("diagnostics_every", diagnostics_every),
                            ("checkpoint_every", checkpoint_every)):
            integer(name, value, 1)
        self.print_progress()
        try:
            for _ in range(generations):
                if not self.step():
                    break
                if self.generation % diagnostics_every == 0:
                    self.record_diagnostics()
                if self.generation % print_every == 0 or (self.config.stop_first_exact and self.solved):
                    self.print_progress()
                if self.generation % checkpoint_every == 0:
                    self.save()
                if self.config.stop_first_exact and self.solved:
                    break
        except KeyboardInterrupt:
            print("Stopped; saving the last fully evaluated generation.", flush=True)
        if self.config.stop_first_exact and self.solved:
            self.record_diagnostics()
        # Do not add off-cadence terminal history points: stopping/resuming at
        # arbitrary generations must not alter the diagnostic trajectory.
        if self.generation % print_every != 0:
            self.print_progress()
        self.save()
        return self

    def evaluate_best(self, dataset=None):
        dataset = self.test_dataset if dataset is None else dataset
        if dataset is None:
            raise ValueError("provide a dataset or configure test_dataset")
        values = self.best["program"][None, :]
        lengths = np.array([self.best["length"]], dtype=np.int32)
        return self.simulator.evaluate(values, lengths, dataset, self.config.k_penalty)

    def save(self, path=None):
        best = {k: v for k, v in self.best.items() if k != "program"}
        metadata = dict(schema=1, table_hash=TABLE_HASH, interface=INTERFACE_VERSION,
                        run_id=self.run_id, simulation=asdict(self.simulator.config), ga=asdict(self.config),
                        train=asdict(self.dataset), test=asdict(self.test_dataset) if self.test_dataset else None,
                        generation=self.generation, elapsed=self.elapsed, best=best,
                        history=_json_history(self.history), rng=self.rng.bit_generator.state)
        arrays = dict(population=self.population, lengths=self.lengths, is_child=self.is_child,
                      totals=self.evaluation.totals, best_program=self.best["program"])
        _atomic_npz(self.checkpoint_path if path is None else path, arrays, metadata)
        return self.checkpoint_path if path is None else Path(path)

    @classmethod
    def resume(cls, path, simulator, runs_directory=None):
        """The saved configs/datasets are authoritative; caller chooses only backend/capacity."""
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
            if (meta["schema"] != 1 or meta["table_hash"] != TABLE_HASH or meta["interface"] != INTERFACE_VERSION):
                raise ValueError("checkpoint schema, machine or interface does not match")
            if SimulationConfig(**meta["simulation"]) != simulator.config:
                raise ValueError("simulator configuration differs from the locked checkpoint")
            obj = cls.__new__(cls)
            obj.simulator, obj.config = simulator, GAConfig(**meta["ga"])
            obj.dataset = Dataset(**meta["train"])
            obj.test_dataset = Dataset(**meta["test"]) if meta["test"] is not None else None
            simulator.config.check_dataset(obj.dataset)
            if obj.test_dataset is not None:
                simulator.config.check_dataset(obj.test_dataset)
            obj.run_id = meta["run_id"]
            obj.runs_directory = Path(runs_directory) if runs_directory is not None else Path(path).resolve().parent.parent
            obj.population = saved["population"].copy()
            obj.lengths = saved["lengths"].copy()
            obj.is_child = saved["is_child"].copy()
            shape = (obj.config.islands, obj.config.population_per_island)
            if obj.population.shape != (*shape, simulator.config.max_program_length) or obj.lengths.shape != shape:
                raise ValueError("checkpoint population shapes do not match config")
            validate_population(obj.population.reshape(-1, obj.population.shape[-1]), obj.lengths.ravel(), simulator.config)
            active = np.arange(obj.population.shape[-1]) < obj.lengths[..., None]
            if np.any(~np.isin(obj.population[active], obj.config.symbols)):
                raise ValueError("checkpoint genes do not match search alphabet")
            totals = saved["totals"].copy()
            if (totals.dtype != np.uint64 or totals.shape != (np.prod(shape), 8) or
                    np.any(totals[:, DENOM] == 0) or np.any(totals[:, MATCH] > totals[:, DENOM]) or
                    np.any(totals[:, FAILED] + totals[:, SUCCESS] != len(obj.dataset.inputs)) or
                    np.any(totals[:, EXACT] > totals[:, SUCCESS])):
                raise ValueError("checkpoint evaluation counters are inconsistent")
            if obj.is_child.dtype != bool or obj.is_child.shape != shape:
                raise ValueError("invalid checkpoint child flags")
            obj.evaluation = Evaluation(totals, len(obj.dataset.inputs), obj.config.k_penalty)
            obj.best = dict(meta["best"], program=saved["best_program"].copy())
            validate_population(obj.best["program"][None, :], np.array([obj.best["length"]], dtype=np.int32), simulator.config)
            obj.generation, obj.elapsed = meta["generation"], meta["elapsed"]
            integer("checkpoint generation", obj.generation)
            obj.history = [{k: float("nan") if value is None else value for k, value in row.items()}
                           for row in meta["history"]]
            obj.rng = np.random.default_rng()
            obj.rng.bit_generator.state = meta["rng"]
        return obj


def checkpoint_config(path):
    """Read locked simulation settings before constructing a resume backend."""
    with np.load(path, allow_pickle=False) as saved:
        return SimulationConfig(**json.loads(str(saved["metadata"]))["simulation"])
