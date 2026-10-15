from copy import deepcopy
from dataclasses import replace
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evolution import (GAConfig, Trainer, winner_probability, rank_population, random_programs,
                       tournament, crossover, mutate, reproduce, checkpoint_config)
from numpy_backend import NumpySimulator
from utils import Dataset, Evaluation, SimulationConfig, validate_population


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        self.c = SimulationConfig(max_program_length=20, program_right_budget=22, data_right_budget=40, t_max=250)
        self.g = GAConfig(islands=3, population_per_island=12, cross_island_interval=2,
                          cross_island_fraction=.2, random_immigrant_fraction=.2, stop_first_exact=False)
        self.data = Dataset(["0", "11", "101"], ["1", "00", "010"])

    def test_thresholds(self):
        g = GAConfig()
        values = [0., np.nextafter(0., 1.), .05, np.nextafter(.05, 1.), .2, np.nextafter(.2, 1.)]
        np.testing.assert_array_equal(winner_probability(values, g), [.5, .53, .53, .58, .58, .8])

    def test_neutral_priority(self):
        order = rank_population(np.array([[.5, .5, .6, .5]]), np.array([[False, True, False, True]]))
        np.testing.assert_array_equal(order, [[2, 1, 3, 0]])

    def test_tournaments(self):
        # Sampling permits contenders/parents to coincide; exact ties are unbiased.
        rng = np.random.default_rng(4)
        fitness = np.array([[0., 1.]])
        g = replace(self.g, delta_thresholds=(), winner_probabilities=(), base_winner_probability=1.)
        result = tournament(rng, fitness, 100000, g)
        self.assertAlmostEqual(np.mean(result == 1), .75, delta=.005)
        tied = tournament(rng, np.zeros((1, 2)), 100000, g)
        self.assertAlmostEqual(tied.mean(), .5, delta=.005)

    def test_crossover(self):
        a = np.full((200, 8), 1, dtype=np.uint8)
        b = np.full_like(a, 2)
        la, lb = np.full(200, 8, dtype=np.int32), np.full(200, 3, dtype=np.int32)
        g = replace(self.g, p_crossover=1., crossover_segments=1, crossover_min_length=10, crossover_max_length=20)
        child, length = crossover(np.random.default_rng(3), a, la, b, lb, g)
        np.testing.assert_array_equal(length, la)
        np.testing.assert_array_equal(child[:, :3], b[:, :3])
        np.testing.assert_array_equal(child[:, 3:], a[:, 3:])
        g = replace(g, crossover_min_length=2, crossover_max_length=2)
        child, _ = crossover(np.random.default_rng(3), a, la, b, la, g)
        self.assertTrue(np.all(np.sum(child == 2, axis=1) == 2))
        for row in child:
            positions = np.flatnonzero(row == 2)
            self.assertEqual(positions[1] - positions[0], 1)
        child, _ = crossover(np.random.default_rng(3), a, la, b, np.zeros_like(lb), g)
        np.testing.assert_array_equal(child, a)
        child, _ = crossover(np.random.default_rng(3), a, la, b, lb, replace(g, p_crossover=0.))
        np.testing.assert_array_equal(child, a)

    def test_mutation_edges_and_padding(self):
        v = np.array([[1, 2, 3, 0], [0, 0, 0, 0], [1, 2, 3, 1]], dtype=np.uint8)
        l = np.array([3, 0, 4], dtype=np.int32)
        g = replace(self.g, p_insert=0., p_delete=1., p_symbol_mutation=0.)
        out, lengths = mutate(np.random.default_rng(2), v.copy(), l.copy(), g)
        np.testing.assert_array_equal(out, [[1, 2, 0, 0], [0, 0, 0, 0], [1, 2, 3, 0]])
        np.testing.assert_array_equal(lengths, [2, 0, 3])
        g = replace(g, p_insert=1., p_delete=0.)
        out, lengths = mutate(np.random.default_rng(2), v.copy(), l.copy(), g)
        np.testing.assert_array_equal(lengths, [4, 1, 4])
        np.testing.assert_array_equal(out[0, :3], v[0, :3])
        np.testing.assert_array_equal(out[2], v[2])
        g = replace(g, p_insert=0., p_symbol_mutation=1.)
        out, _ = mutate(np.random.default_rng(2), v.copy(), l.copy(), g)
        active = np.arange(4) < l[:, None]
        self.assertTrue(np.all(out[active] != v[active]))
        self.assertTrue(np.all(out[~active] == 0))

    def test_population_and_migration(self):
        rng = np.random.default_rng(91)
        g = replace(self.g, alphabet="B01#", elite_fraction=.25,
                    cross_island_fraction=.25, random_immigrant_fraction=0.,
                    p_insert=0., p_delete=0., p_symbol_mutation=0., p_crossover=0.)
        v, l = random_programs(rng, (3, 12), 20, g.symbols)
        # Distinct integer lengths identify each island's elites without genotype parsing.
        l[:] = np.arange(12)[None, :] + 1
        for i in range(3):
            v[i] = i + 1
        v[np.arange(20) >= l[..., None]] = 0
        totals = np.tile(np.array([1, 2, 0, 0, 1, 3, 4, 5], dtype=np.uint64), (36, 1))
        ev = Evaluation(totals, 1, .5)
        out, lengths, flags = reproduce(v, l, ev, np.zeros((3, 12), dtype=bool), 2, g, rng)
        for i in range(3):
            self.assertTrue(np.all(out[i, 3:6, 0] != i + 1))
        self.assertTrue(np.all(~flags[:, :6]))
        self.assertTrue(np.all(flags[:, 6:]))
        validate_population(out.reshape(-1, 20), lengths.ravel(), self.c)

    def test_resume_and_diagnostic_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Trainer(NumpySimulator(self.c, 11), self.data, self.g, runs_directory=directory)
            for _ in range(3):
                a.step(); a.record_diagnostics()
            a.save()
            self.assertEqual(checkpoint_config(a.checkpoint_path), self.c)
            b = Trainer.resume(a.checkpoint_path, NumpySimulator(self.c, 17))
            for _ in range(4):
                a.evaluate_best(self.data)  # consumes no training RNG
                a.step(); b.step()
                a.record_diagnostics(); b.record_diagnostics()
            np.testing.assert_array_equal(a.population, b.population)
            np.testing.assert_array_equal(a.lengths, b.lengths)
            np.testing.assert_array_equal(a.is_child, b.is_child)
            np.testing.assert_array_equal(a.evaluation.totals, b.evaluation.totals)
            np.testing.assert_array_equal(a.best["program"], b.best["program"])
            self.assertEqual(a.rng.bit_generator.state, b.rng.bit_generator.state)
            self.assertEqual([{k: v for k, v in r.items() if k != "elapsed"} for r in a.history],
                             [{k: v for k, v in r.items() if k != "elapsed"} for r in b.history])
            with self.assertRaises(ValueError):
                Trainer.resume(a.checkpoint_path, NumpySimulator(replace(self.c, t_max=251)))

    def test_interrupt_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(NumpySimulator(self.c), self.data, self.g, runs_directory=directory)
            population, rng = trainer.population.copy(), deepcopy(trainer.rng.bit_generator.state)
            evaluate = trainer._evaluate
            def interrupt(*args):
                raise KeyboardInterrupt
            trainer._evaluate = interrupt
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.run(5)
            np.testing.assert_array_equal(trainer.population, population)
            self.assertEqual(trainer.generation, 0)
            self.assertEqual(trainer.rng.bit_generator.state, rng)
            resumed = Trainer.resume(trainer.checkpoint_path, NumpySimulator(self.c))
            trainer._evaluate = evaluate
            trainer.step(); resumed.step()
            np.testing.assert_array_equal(trainer.population, resumed.population)

    def test_exact_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            g = replace(self.g, stop_first_exact=True)
            trainer = Trainer(NumpySimulator(self.c), Dataset(["01"], ["01"]), g,
                              runs_directory=directory, initial_programs=[""])
            self.assertTrue(trainer.solved)
            self.assertFalse(trainer.step())
            self.assertEqual(trainer.generation, 0)
            # An exact candidate first arriving in the next evaluated generation.
            trainer = Trainer(NumpySimulator(self.c), self.data, g, runs_directory=directory)
            trainer.evaluation.totals[:, 3] = 0
            def exact(*args):
                ev = Evaluation(trainer.evaluation.totals.copy(), 3, .5)
                ev.totals[0] = [3, 3, 0, 3, 3, 6, 3, 12]
                return ev
            trainer._evaluate = exact
            self.assertTrue(trainer.step())
            self.assertFalse(trainer.step())
            self.assertEqual(trainer.generation, 1)

    def test_bad_configs(self):
        for changes in (dict(p_insert=.9, p_delete=.2), dict(islands=1),
                        dict(delta_thresholds=(.2, .05)), dict(winner_probabilities=(.4, .6)),
                        dict(elite_fraction=.9, random_immigrant_fraction=.9), dict(alphabet="01")):
            with self.assertRaises(ValueError):
                GAConfig(**changes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
