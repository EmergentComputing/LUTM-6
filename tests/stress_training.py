"""Controlled small/medium and one 40k-population exercise, not a benchmark suite."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evolution import Trainer, GAConfig, random_programs
from numpy_backend import NumpySimulator
from taichi_backend import initialize_cuda, TaichiSimulator
from utils import Dataset, SimulationConfig, validate_population


def stress():
    data = Dataset(["0", "11", "101"], ["1", "00", "010"])
    c = SimulationConfig(data_left_budget=16, data_right_budget=83, t_max=8000)
    gpu = TaichiSimulator(c, capacity=2048)
    g = GAConfig(islands=4, population_per_island=10000, cross_island_interval=2, stop_first_exact=False)
    v, l = random_programs(np.random.default_rng(12), (2048,), 128, g.symbols)
    preflight = Dataset(["101"], ["010"])
    gpu.evaluate(v, l, preflight)  # compile separately from timing
    start = perf_counter()
    gpu.evaluate(v, l, preflight)
    seconds = perf_counter() - start
    print(f"CUDA preflight: 2048 programs, 8000-transition cap, one input: {seconds:.3f}s including transfer", flush=True)
    if seconds > 1:
        raise RuntimeError("preflight too slow for the planned Windows stress run; no larger run launched")
    with tempfile.TemporaryDirectory() as directory:
        for label, config, generations in (("small", replace(g, islands=2, population_per_island=64), 5),
                                           ("medium", replace(g, population_per_island=512), 6),
                                           ("40k", g, 2)):
            start = perf_counter()
            trainer = Trainer(gpu, data, config, runs_directory=directory)
            trainer.run(generations, print_every=generations, diagnostics_every=1, checkpoint_every=generations)
            flat, lengths = trainer.population.reshape(-1, 128), trainer.lengths.ravel()
            validate_population(flat, lengths, c)
            sample = np.random.default_rng(8).choice(len(flat), size=min(32, len(flat)), replace=False)
            expected = NumpySimulator(c, 17).evaluate(flat[sample], lengths[sample], data, config.k_penalty)
            np.testing.assert_array_equal(expected.totals, trainer.evaluation.totals[sample])
            resumed = Trainer.resume(trainer.checkpoint_path, gpu)
            np.testing.assert_array_equal(resumed.population, trainer.population)
            print(f"{label}: {len(flat):,} programs, {c.data_size} X cells, generations={trainer.generation}; "
                  f"CPU sample and saved state agree; elapsed={perf_counter()-start:.2f}s", flush=True)


if __name__ == "__main__":
    initialize_cuda()
    stress()
