"""Explicit CPU validation of long literal witnesses, including square.

The Numba loop below executes ONE PHYSICAL TABLE ROW per iteration. It is a
test-only CPU accelerator, not a source-TM shortcut or a trainer backend.
Its fixed test arena raises on crossing; the public scalar stays unbounded.
"""

import argparse
from itertools import product
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from numba import njit

# Allow the documented `python tests/validate_known.py` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machine import Machine, encode_input
from programs import (TASKS, CODE, HALT, compile_blocks, increment_source, times_two_source,
                      popcount_source, square_source, load_program)
from utils import SYMBOL_IDS, dense_table


@njit
def _physical(p, x, write, dp, dx, nxt, origin, budget):
    hp, hx, state = 1, origin, 0
    count = np.uint64(0)
    while state != 9 and count < budget:
        a, b = p[hp], x[hx]
        x[hx] = write[state, a, b]
        hp += dp[state, a, b]
        hx += dx[state, a, b]
        state = nxt[state, a, b]
        count += np.uint64(1)
        if hp < 0 or hp >= len(p) or hx < 0 or hx >= len(x):
            raise ValueError("physical validation arena crossed")
    if state != 9:
        raise ValueError("physical validation time budget exhausted")
    return state, hp - 1, hx - origin, count


def run_physical(pattern, bits, budget=20_000_000_000, arena=32768):
    p = np.zeros(len(pattern) + 2, dtype=np.uint8)
    p[1:len(pattern) + 1] = [SYMBOL_IDS[s] for s in pattern]
    x = np.zeros(arena, dtype=np.uint8)
    origin = arena // 2
    encoded = encode_input(bits)
    if origin + len(encoded) >= arena:
        raise ValueError("test input does not fit validation arena")
    x[origin:origin + len(encoded)] = [SYMBOL_IDS[s] for s in encoded]
    state, hp, hx, count = _physical(p, x, *dense_table(), origin, np.uint64(budget))
    output, pos = [], hx + origin
    while 0 <= pos < arena and x[pos] == 2:
        payload = int(x[pos + 1]) if pos + 1 < arena else 0
        if payload not in (0, 2):
            break
        output.append(str(int(payload == 2)))
        pos += 2
    return "".join(output), int(count), hp, hx, x


def run_source(table, names, bits, audit_square=False):
    """Independent literal four-symbol TM; optionally check every square pass."""
    tape, q, h, count, passes = dict(enumerate(bits + "#")), 0, 0, 0, 0
    final_checked = False
    budget = max(10000, 256 * (len(bits) + 1)**2)
    n = int(bits or "0", 2)
    while q != HALT:
        if count >= budget or names[q] == "trap":
            raise AssertionError(("source trap/timeout", bits, count, names[q]))
        if audit_square and (names[q] == "select" or (names[q] == "output_seek_end" and not final_checked)):
            if tape.get(h - 1) != "#":
                raise AssertionError("missing square left boundary")
            values, cursor, column = [0, 0, 0], h, 0
            while tape.get(cursor) != "#":
                for track in range(3):
                    value = tape.get(cursor + track, "B")
                    if value not in "01":
                        raise AssertionError("malformed square column")
                    values[track] += int(value) << column
                column += 1
                cursor += 3
            expected = [n >> passes, n << passes, n * (n % (1 << passes))]
            if values != expected:
                raise AssertionError((bits, passes, values, expected))
            if names[q] == "select":
                passes += 1
            else:
                final_checked = True
        q, write, direction = table[q, tape.get(h, "B")]
        if write == "B":
            tape.pop(h, None)
        else:
            tape[h] = write
        h += direction
        count += 1
    output = []
    while tape.get(h, "B") in "01":
        output.append(tape[h])
        h += 1
    return "".join(output), count


def validate_sources():
    rng = np.random.default_rng(20261015)
    samples = ["".join(s) for n in range(9) for s in product("01", repeat=n)]
    samples += ["".join(rng.choice(list("01"), size=n)) for n in (16, 31, 64, 127, 256)]
    for name, build in (("increment", increment_source), ("times_two", times_two_source),
                        ("popcount", popcount_source), ("square", square_source)):
        table, names = build()
        for bits in samples:
            result, _ = run_source(table, names, bits, audit_square=name == "square")
            if result != TASKS[name](bits):
                raise AssertionError((name, bits, result))
        print(f"{name}: {len(samples)} source inputs passed; maximum width 256", flush=True)


def validate_literals():
    # Cross-check the test accelerator against the unbounded public scalar.
    for name, bits in (("identity", "0101"), ("bit_not", "00101"), ("reverse", "10110"),
                       ("compiled_reverse", "01"), ("increment", "111"), ("times_two", "00101")):
        pattern = load_program(name)
        output, count, hp, hx, tape = run_physical(pattern, bits)
        scalar = Machine(pattern, bits)
        result = scalar.run(count)
        if (result["output"], result["T"], scalar.hp, scalar.hd) != (output, count, hp, hx):
            raise AssertionError(("accelerator/scalar mismatch", name, bits))
        expected = np.zeros(len(tape), dtype=np.uint8)
        for pos, symbol in scalar.d.items():
            expected[pos + len(tape) // 2] = SYMBOL_IDS[symbol]
        np.testing.assert_array_equal(tape, expected)
    for name, samples in (
        ("compiled_reverse", ("", "00101", "10110010")),
        ("increment", ("", "000", "00101", "1" * 32, "10100101")),
        ("times_two", ("", "000", "00101", "1" * 32, "10100101")),
        ("popcount", ("", "00101", "10110010")),
        ("square", ("", "0", "1", "10", "101", "110", "00110")),
    ):
        pattern = load_program(name)
        for bits in samples:
            start = perf_counter()
            output, count, *_ = run_physical(pattern, bits)
            if output != TASKS[name](bits):
                raise AssertionError((name, bits, output, TASKS[name](bits)))
            print(f"{name} x={bits!r} y={output!r} T={count:,} seconds={perf_counter()-start:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", action="store_true", help="also run expensive actual literal programs on CPU")
    parser.add_argument("--task", choices=tuple(TASKS), help="enumerate input lengths with the unbounded scalar")
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=10_000_000)
    args = parser.parse_args()
    if args.task is not None:
        if not 0 <= args.min_length <= args.max_length:
            parser.error("require 0 <= min-length <= max-length")
        pattern = load_program(args.task)
        for length in range(args.min_length, args.max_length + 1):
            for word in product("01", repeat=length):
                bits = "".join(word)
                result = Machine(pattern, bits).run(args.timeout)
                if not result["halted"] or result["output"] != TASKS[args.task](bits):
                    raise AssertionError((args.task, bits, result))
                print(f"x={bits!r} output={result['output']!r} T={result['T']:,}", flush=True)
    else:
        validate_sources()
        if args.physical:
            validate_literals()


if __name__ == "__main__":
    main()
