"""CUDA is required here: unavailable CUDA is an error, never a silent CPU test."""

from dataclasses import replace
from itertools import product
import tempfile
import unittest

import numpy as np
import taichi as ti

from evolution import GAConfig, Trainer, random_programs
from numpy_backend import NumpySimulator, decode_buffers
from programs import TASKS, load_program, compile_blocks, CODE, HALT
from machine import serialize, decode, SYMBOLS
from taichi_backend import initialize_cuda, TaichiSimulator
from test_contract import check_reference
from utils import Dataset, SimulationConfig, pack_programs


@ti.kernel
def readout_fixtures(sim: ti.template(), heads: ti.types.ndarray(dtype=ti.i32, ndim=1),
                     targets: ti.types.ndarray(dtype=ti.u8, ndim=2), n: ti.i32):
    for i in range(n):
        length, matches = sim._readout(i, heads[i], 0, 1, targets, 0)
        sim.meta[i, 9] = length


class CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_cuda()

    def compare(self, a, b):
        self.assertEqual(a.outputs, b.outputs)
        for name in a.__dataclass_fields__:
            if name != "outputs":
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)

    def test_full_state_and_partition(self):
        rng = np.random.default_rng(56)
        patterns = ["", "B", "BBB", "1B", "BB#", "B" * 200]
        patterns += ["".join(rng.choice(list("B01#"), int(n))) for n in rng.integers(0, 201, 149)]
        patterns += [load_program(n) for n in ("bit_not", "parity", "append0", "append1")]
        c = SimulationConfig(max_program_length=200, program_right_budget=202,
                             data_left_budget=2, data_right_budget=24, t_max=1600)
        v, l = pack_programs(patterns, 200)
        cpu, gpu = NumpySimulator(c, 37), TaichiSimulator(c, 31)
        for x in ("", "0", "00101"):
            a, b = cpu.simulate(v, l, x, True), gpu.simulate(v, l, x, True)
            self.compare(a, b)
            check_reference(self, b, patterns, x, c)
        dataset = Dataset(["", "01", "11", "01"], ["0", "1", "", "1"])
        a, b = cpu.evaluate(v, l, dataset), gpu.evaluate(v, l, dataset)
        np.testing.assert_array_equal(a.totals, b.totals)
        permutation = rng.permutation(len(v))
        d = gpu.evaluate(v[permutation], l[permutation], dataset)
        np.testing.assert_array_equal(d.totals, a.totals[permutation])

    def test_known_computations(self):
        names = ("identity", "bit_not", "parity", "append0", "append1", "reverse",
                 "compiled_reverse", "increment", "times_two")
        patterns = [load_program(n) for n in names]
        width = max(map(len, patterns))
        c = SimulationConfig(max_program_length=width, program_right_budget=width + 2,
                             data_left_budget=32, data_right_budget=512, t_max=5_000_000)
        v, l = pack_programs(patterns, width)
        gpu = TaichiSimulator(c, capacity=len(names))
        for x in ("", "0", "1", "00101", "10110", "11001"):
            result = gpu.simulate(v, l, x, True)
            self.assertTrue(np.all(result.valid), [result.row(i) for i in range(len(names))])
            self.assertEqual(result.outputs, tuple(TASKS[n](x) for n in names))
        # Correspondence also when the budget ends on the exact halting transition.
        p, length = pack_programs([load_program("bit_not")], width)
        exact_t = int(gpu.simulate(p, length, "10110").T[0])
        short = TaichiSimulator(replace(c, t_max=exact_t), capacity=1)
        self.assertEqual(short.simulate(p, length, "10110").outputs, ("01001",))
        # Wider popcount cases are covered by the CPU physical validator. Keep
        # display-GPU validation kernels short, not tens of millions of steps.
        pattern = load_program("popcount")
        pop_c = replace(c, max_program_length=len(pattern), program_right_budget=len(pattern) + 1)
        p, length = pack_programs([pattern], len(pattern))
        pop = TaichiSimulator(pop_c, 1)
        for x in ("", "1"):
            self.assertEqual(pop.simulate(p, length, x).outputs, (TASKS["popcount"](x),))

    def test_evolution_backend_equivalence(self):
        c = SimulationConfig(max_program_length=40, program_right_budget=42,
                             data_left_budget=4, data_right_budget=40, t_max=500)
        g = GAConfig(islands=3, population_per_island=25, cross_island_interval=2,
                     cross_island_fraction=.1, stop_first_exact=False, alphabet="B01#")
        dataset = Dataset(["", "0", "11", "101"], ["", "1", "00", "010"])
        with tempfile.TemporaryDirectory() as directory:
            a = Trainer(NumpySimulator(c, 17), dataset, g, runs_directory=directory)
            b = Trainer(TaichiSimulator(c, 23), dataset, g, runs_directory=directory)
            for _ in range(6):
                np.testing.assert_array_equal(a.population, b.population)
                np.testing.assert_array_equal(a.evaluation.totals, b.evaluation.totals)
                self.assertEqual(a.rng.bit_generator.state, b.rng.bit_generator.state)
                a.step(); b.step()
            b.save()
            resumed = Trainer.resume(b.checkpoint_path, NumpySimulator(c, 29))
            resumed.step(); b.step()
            np.testing.assert_array_equal(b.population, resumed.population)
            np.testing.assert_array_equal(b.evaluation.totals, resumed.evaluation.totals)

    def test_uint32_limit_and_partial_function(self):
        c = SimulationConfig(t_max=2**32 - 1)
        v, l = pack_programs([""], c.max_program_length)
        self.assertEqual(TaichiSimulator(c, 1).simulate(v, l, "1").T[0], 2)
        source = {(0, a): (HALT if a == "1" else 0, a, 0) for a in CODE}
        pattern = serialize(compile_blocks(source))
        c = SimulationConfig(max_program_length=len(pattern), program_right_budget=len(pattern) + 1, t_max=3000)
        v, l = pack_programs([pattern], len(pattern))
        gpu, cpu = TaichiSimulator(c, 1), NumpySimulator(c, 1)
        for bits in ("1", "10", "0", ""):
            a, b = gpu.simulate(v, l, bits, True), cpu.simulate(v, l, bits, True)
            self.compare(a, b)
            if bits.startswith("1"):
                self.assertEqual(a.outputs, (bits,))
            else:
                self.assertEqual(a.reason[0], 5)
                self.assertFalse(a.halted[0])

    def test_decoder_on_finite_halt_buffers(self):
        # Synthetic HALTED tapes exercise the decoder independently of execution,
        # including exterior lookahead that is not a physical-head crossing.
        c = SimulationConfig(max_program_length=4, program_right_budget=5, data_left_budget=2, data_right_budget=6)
        n = 201
        gpu = TaichiSimulator(c, n)
        rng = np.random.default_rng(133)
        tapes = rng.integers(0, 4, size=(n, c.data_size), dtype=np.uint8)
        tapes[0] = 0
        tapes[0, -1] = 2  # final cell leader 1 + implicit exterior B -> decoded 0
        heads = rng.integers(0, c.data_size, size=n, dtype=np.int32)
        heads[0] = c.data_size - 1
        expected = tuple(decode({j: SYMBOLS[s] for j, s in enumerate(row) if s}, int(h)) for row, h in zip(tapes, heads))
        self.assertEqual(expected[0], "0")
        output, length = decode_buffers(tapes, heads, np.ones(n, dtype=bool))
        cpu = tuple("".join(map(str, row[:k])) for row, k in zip(output, length))
        self.assertEqual(cpu, expected)
        gpu.x.from_numpy(tapes)
        target = ti.ndarray(ti.u8, shape=(1, 1))
        target.from_numpy(np.zeros((1, 1), dtype=np.uint8))
        readout_fixtures(gpu, heads, target, n)
        length = gpu.meta.to_numpy()[:, 9]
        output = gpu.output.to_numpy()
        self.assertEqual(tuple("".join(map(str, row[:k])) for row, k in zip(output, length)), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
