"""Golden interface tests and independent dictionary/bounded-array correspondence."""

from dataclasses import replace
from itertools import product
import unittest

import numpy as np

from machine import Machine, SYMBOLS, TABLE, TABLE_HASH, encode_input, decode, serialize
from numpy_backend import NumpySimulator
from programs import TASKS, load_program, witnesses, compile_blocks, CODE, HALT
from utils import Dataset, SimulationConfig, pack_programs, SYMBOL_IDS, boundary_reasons, Evaluation


def bounded_reference(pattern, bits, config):
    machine = Machine(pattern, bits)
    why = 0
    for _ in range(config.t_max):
        machine.step()
        # Deliberately scalar and independent of the vectorized boundary helper.
        if machine.hp < -config.program_left_budget:
            why = 1
        elif machine.hp > config.program_right_budget:
            why = 2
        elif machine.hd < -config.data_left_budget:
            why = 3
        elif machine.hd > config.data_right_budget:
            why = 4
        if why or machine.state == "H":
            break
    else:
        why = 5
    return machine, why


def check_reference(test, result, patterns, bits, config):
    for i, pattern in enumerate(patterns):
        ref, why = bounded_reference(pattern, bits, config)
        actual = result.row(i)
        expected = ref.run(0)
        test.assertEqual(actual["reason"], ("ok", "program_left", "program_right", "data_left", "data_right", "timeout")[why])
        for key in ("T", "state", "halted", "program_head", "data_head", "program_space", "data_space"):
            test.assertEqual(actual[key], expected[key], (i, pattern, bits, key))
        test.assertEqual(actual["output"], expected["output"] if why == 0 else None)
        for attr, tape, low, high in (("program_tape", ref.p, -config.program_left_budget, config.program_right_budget),
                                      ("data_tape", ref.d, -config.data_left_budget, config.data_right_budget)):
            np.testing.assert_array_equal(getattr(result, attr)[i], [SYMBOL_IDS[tape.get(j, "B")] for j in range(low, high + 1)])


class ContractTests(unittest.TestCase):
    def test_table(self):
        self.assertEqual(len(TABLE), 144)
        self.assertEqual(TABLE_HASH, "88ac3e58f4525dccd286fd723808f051687fbccc1ac02d0fb4e41b8e5f2eba7a")
        self.assertTrue(all(r.p == p for (_, p, _), r in TABLE.items()))
        self.assertTrue(all(r.dp == r.dd == 0 for r in TABLE.values() if r.state == "H"))

    def test_codecs(self):
        for x, encoded in (("", "B1"), ("0", "1BB1"), ("1", "11B1"),
                           ("010", "1B111BB1"), ("0011", "1B1B1111B1")):
            self.assertEqual(encode_input(x), encoded)
            self.assertEqual(decode(dict(enumerate(encoded)), 0), x)
        self.assertEqual(decode({4: "1"}, 4), "0")  # exterior blank payload
        self.assertEqual(decode({0: "1", 1: "#"}, 0), "")
        self.assertEqual(decode({0: "1", 1: "1", 2: "B", 3: "B"}, 0), "1")

    def test_scalar_self_extends(self):
        m = Machine(load_program("bit_not"), "010101" * 7)
        r = m.run(100_000)
        self.assertEqual(r["output"], TASKS["bit_not"]("010101" * 7))
        self.assertGreater(m.x_left, 0)  # visited blanks need not remain nonblank
        self.assertGreater(max(m.d), 80)
        self.assertGreater(r["program_space"][0], 0)  # visited exterior blank

    def test_register_and_serialize(self):
        for name, pattern in witnesses().items():
            self.assertEqual(load_program(name), pattern)
        for name in TASKS:
            self.assertIsInstance(load_program(name), str)
        for bad in (False, True, "0"):
            with self.assertRaises(ValueError):
                serialize([((bad, 1, 0), None)])

    def test_validation(self):
        for pair in (([], []), (["1"], []), (["x"], ["0"]), (["0"], ["B"])):
            with self.assertRaises(ValueError):
                Dataset(*pair)
        self.assertEqual(Dataset(["", "0", "0"], ["", "1", "1"]).inputs, ("", "0", "0"))
        for kwargs in (dict(t_max=0), dict(t_max=2**32), dict(program_left_budget=0),
                       dict(program_right_budget=127), dict(t_max=True)):
            with self.assertRaises(ValueError):
                SimulationConfig(**kwargs)
        self.assertEqual(SimulationConfig(t_max=2**32 - 1).t_max, 4_294_967_295)
        with self.assertRaises(ValueError):
            SimulationConfig(data_right_budget=2).check_dataset(Dataset(["0"], ["0"]))

    def test_boundary_precedence(self):
        c = SimulationConfig()
        np.testing.assert_array_equal(boundary_reasons(np.array([-9, 137, 0, 0, -8, 136]),
                                                       np.array([-33, 256, -33, 256, -32, 255]), c),
                                      [1, 2, 3, 4, 0, 0])


class NumpyTests(unittest.TestCase):
    def test_raw_full_state(self):
        rng = np.random.default_rng(104)
        patterns = ["".join(x) for n in range(4) for x in product(SYMBOLS, repeat=n)]
        patterns += ["".join(rng.choice(SYMBOLS, size=int(n))) for n in rng.integers(4, 161, 100)]
        patterns += [load_program(n) for n in ("bit_not", "parity", "append0", "append1")]
        c = SimulationConfig(max_program_length=160, program_right_budget=162, data_left_budget=3,
                             data_right_budget=18, t_max=1500)
        values, lengths = pack_programs(patterns, 160)
        simulator = NumpySimulator(c, capacity=37)
        for bits in ("", "0", "10110"):
            check_reference(self, simulator.simulate(values, lengths, bits, tapes=True), patterns, bits, c)

    def test_known_outputs(self):
        names = ("identity", "bit_not", "parity", "append0", "append1")
        c = SimulationConfig(t_max=5000)
        values, lengths = pack_programs([load_program(n) for n in names], 128)
        for bits in ("", "0", "1", "00101", "10111001"):
            r = NumpySimulator(c).simulate(values, lengths, bits)
            self.assertEqual(r.outputs, tuple(TASKS[n](bits) for n in names))

    def test_exact_budget_and_failures(self):
        c = SimulationConfig(t_max=1)
        values, lengths = pack_programs([""], 128)
        self.assertEqual(NumpySimulator(c).simulate(values, lengths, "1").reason[0], 5)
        r = NumpySimulator(replace(c, t_max=2)).simulate(values, lengths, "1")
        self.assertEqual(r.T[0], 2)
        self.assertEqual(r.outputs, ("1",))
        # Blank visits count as used space; left crossing is counted at the move.
        p = serialize([((0, -1, 0), (1, -1, 0))])
        v, l = pack_programs([p], 128)
        r = NumpySimulator(replace(c, data_left_budget=1, t_max=100)).simulate(v, l, "")
        self.assertEqual(r.reason[0], 3)
        self.assertEqual(r.data_left[0], 2)
        self.assertIsNone(r.outputs[0])
        p = serialize([((0, 1, 0), (1, 1, 0))])
        v, l = pack_programs([p], 128)
        r = NumpySimulator(replace(c, data_right_budget=2, t_max=200)).simulate(v, l, "")
        self.assertEqual(r.reason[0], 4)
        self.assertEqual(r.data_head[0], 3)

    def test_scoring_and_target_independence(self):
        c = SimulationConfig(t_max=2)
        values, lengths = pack_programs([""], 128)
        simulator = NumpySimulator(c)
        ev = simulator.evaluate(values, lengths, Dataset(["11"], ["1100"]))
        self.assertEqual(ev.accuracy[0], .5)
        self.assertEqual(ev.fitness[0], .5)
        ev = simulator.evaluate(values, lengths, Dataset(["", "11"], ["", "1100"]))
        self.assertEqual(ev.accuracy[0], 3 / 5)  # micro, not average-of-case accuracies
        self.assertFalse(ev.exact[0])
        ev = simulator.evaluate(values, lengths, Dataset(["11"], ["1"]))
        self.assertEqual(ev.accuracy[0], .5)  # full output length penalizes extra bits
        before = simulator.simulate(values, lengths, "11", tapes=True)
        simulator.evaluate(values, lengths, Dataset(["11"], ["00000000000"]))
        after = simulator.simulate(values, lengths, "11", tapes=True)
        np.testing.assert_array_equal(before.data_tape, after.data_tape)
        ev = NumpySimulator(replace(c, t_max=1)).evaluate(values, lengths, Dataset(["1"], ["1"]))
        self.assertEqual(ev.accuracy[0], 0)
        self.assertEqual(ev.fitness[0], -.5)
        self.assertTrue(np.isnan(ev.resource_mean(5)[0]))

    def test_partition_and_explicit_blank_length(self):
        c = SimulationConfig(t_max=1200)
        patterns = ["", "BBB", "1B", "0#1", load_program("bit_not"), "B" * 128]
        v, l = pack_programs(patterns, 128)
        dataset = Dataset(["", "010", "1", "010"], ["", "101", "0", "101"])
        a = NumpySimulator(c, 2).evaluate(v, l, dataset)
        b = NumpySimulator(c, 20).evaluate(v, l, dataset)
        np.testing.assert_array_equal(a.totals, b.totals)
        permutation = np.array([4, 1, 3, 0, 5, 2])
        d = NumpySimulator(c, 4).evaluate(v[permutation], l[permutation], dataset)
        np.testing.assert_array_equal(d.totals, a.totals[permutation])
        r = NumpySimulator(c).simulate(v, l, "")
        self.assertEqual(r.program_right[1], 0)  # explicit BBB is not nonblank support
        self.assertEqual(l[1], 3)  # genotype length has not been trimmed

    def test_compiler_write_move_then_halt(self):
        for direction, output in ((-1, ""), (0, "00"), (1, "0")):
            source = {(0, a): (HALT, "0", direction) for a in CODE}
            pattern = serialize(compile_blocks(source))
            c = SimulationConfig(max_program_length=len(pattern), program_right_budget=len(pattern) + 1, t_max=10000)
            v, l = pack_programs([pattern], len(pattern))
            result = NumpySimulator(c).simulate(v, l, "10", True)
            self.assertEqual(result.outputs, (output,))
            self.assertEqual(result.data_head[0], 2 * direction)
            np.testing.assert_array_equal(result.data_tape[0, c.data_left_budget:c.data_left_budget + 2], [2, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
