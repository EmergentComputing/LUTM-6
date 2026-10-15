"""Shared representations, bounded-run contract, fitness and plotting.

Physical IDs: B=0, 0=1, 1=2, #=3. Output padding 2 is NOT a tape ID.
"""

from dataclasses import dataclass
import numpy as np

from machine import SYMBOLS, STATES, H, TABLE, encode_input

SYMBOL_IDS = {symbol: i for i, symbol in enumerate(SYMBOLS)}
STATE_NAMES = STATES + (H,)
BLANK, ZERO, ONE, MARK = range(4)
HALT = len(STATES)
OUTPUT_PAD = 2
REASONS = ("ok", "program_left", "program_right", "data_left", "data_right", "timeout")
# Compact per-program totals, accumulated over input cases.
MATCH, DENOM, FAILED, EXACT, SUCCESS, TIME, P_SPACE, X_SPACE = range(8)


def integer(name, value, minimum=0, maximum=2**31 - 2):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


@dataclass(frozen=True)
class SimulationConfig:
    max_program_length: int = 128
    program_left_budget: int = 8
    program_right_budget: int = 136
    data_left_budget: int = 32
    data_right_budget: int = 255
    t_max: int = 20_000

    def __post_init__(self):
        integer("max_program_length", self.max_program_length, 1)
        for name in ("program_left_budget", "program_right_budget", "data_left_budget", "data_right_budget"):
            integer(name, getattr(self, name), 1)
        integer("t_max", self.t_max, 1, 2**32 - 1)
        if self.program_right_budget < self.max_program_length:
            raise ValueError("program_right_budget must include the blank after the maximum program")
        for size in (self.program_size, self.data_size):
            integer("tape size", size, 3, 2**31 - 4)

    @property
    def program_size(self):
        return self.program_left_budget + self.program_right_budget + 1

    @property
    def data_size(self):
        return self.data_left_budget + self.data_right_budget + 1

    @property
    def output_capacity(self):
        return (self.data_size + 1) // 2

    def check_dataset(self, dataset):
        max_target = max(map(len, dataset.targets))
        integer("maximum target length", max_target)
        if self.data_right_budget < 2 * max(map(len, dataset.inputs)) + 2:
            raise ValueError("data_right_budget must fit E(x) and its adjacent blank for every input")
        if len(dataset.inputs) * max(self.t_max, self.program_size, self.data_size, self.output_capacity, max_target) > 2**64 - 1:
            raise ValueError("dataset aggregate counters would overflow uint64")


@dataclass(frozen=True)
class Dataset:
    inputs: tuple
    targets: tuple

    def __post_init__(self):
        # Freeze caller-owned lists; preserve order and repetitions.
        if isinstance(self.inputs, str) or isinstance(self.targets, str):
            raise ValueError("provide lists/tuples of strings, not one string")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "targets", tuple(self.targets))
        if not self.inputs or len(self.inputs) != len(self.targets):
            raise ValueError("inputs and targets must have the same nonzero number of strings")
        for value in self.inputs + self.targets:
            if not isinstance(value, str) or any(bit not in "01" for bit in value):
                raise ValueError("each input and target must be a binary string; empty strings are allowed")

    def arrays(self):
        encoded = [encode_input(x) for x in self.inputs]
        data = np.zeros((len(encoded), max(map(len, encoded))), dtype=np.uint8)
        targets = np.full((len(encoded), max(1, max(map(len, self.targets)))), OUTPUT_PAD, dtype=np.uint8)
        for i, (x, target) in enumerate(zip(encoded, self.targets)):
            data[i, :len(x)] = [SYMBOL_IDS[s] for s in x]
            targets[i, :len(target)] = [int(s) for s in target]
        return data, targets, np.array(list(map(len, self.targets)), dtype=np.int32)


def pack_programs(patterns, width=None):
    if isinstance(patterns, str):
        raise ValueError("provide a list/tuple of programs, not one string")
    patterns = tuple(patterns)
    if not patterns:
        raise ValueError("provide at least one program")
    for p in patterns:
        if not isinstance(p, str) or any(s not in SYMBOL_IDS for s in p):
            raise ValueError("programs must use B,0,1,#")
    if width is None:
        width = max(1, max(map(len, patterns)))
    integer("width", width, 1)
    if max(map(len, patterns)) > width:
        raise ValueError("program exceeds width")
    values = np.zeros((len(patterns), width), dtype=np.uint8)
    lengths = np.array(list(map(len, patterns)), dtype=np.int32)
    for i, p in enumerate(patterns):
        values[i, :len(p)] = [SYMBOL_IDS[s] for s in p]
    return values, lengths


def program_string(values, length):
    return "".join(SYMBOLS[int(s)] for s in values[:int(length)])


def validate_population(values, lengths, config):
    if not isinstance(values, np.ndarray) or values.dtype != np.uint8 or values.ndim != 2:
        raise ValueError("programs must be a 2D uint8 array")
    if values.shape[1] != config.max_program_length or not len(values):
        raise ValueError("program array width must equal max_program_length and batch must be nonempty")
    if not isinstance(lengths, np.ndarray) or lengths.dtype != np.int32 or lengths.shape != (len(values),):
        raise ValueError("lengths must be an int32 vector matching the batch")
    if np.any(values > MARK) or np.any(lengths < 0) or np.any(lengths > values.shape[1]):
        raise ValueError("invalid program symbols or lengths")
    if np.any(values[np.arange(values.shape[1])[None, :] >= lengths[:, None]] != BLANK):
        raise ValueError("inactive program padding must be B")


def dense_table():
    write = np.zeros((9, 4, 4), dtype=np.uint8)
    moves = [np.zeros((9, 4, 4), dtype=np.int32) for _ in range(3)]
    dp, dx, state = moves
    for (s, p, x), r in TABLE.items():
        i = (STATES.index(s), SYMBOL_IDS[p], SYMBOL_IDS[x])
        write[i], dp[i], dx[i], state[i] = SYMBOL_IDS[r.d], r.dp, r.dd, STATE_NAMES.index(r.state)
    return write, dp, dx, state


def boundary_reasons(hp, hx, config):
    """Heads are logical coordinates; precedence is part of the contract."""
    return np.select((hp < -config.program_left_budget, hp > config.program_right_budget,
                      hx < -config.data_left_budget, hx > config.data_right_budget),
                     (1, 2, 3, 4), default=0).astype(np.uint8)


@dataclass
class BatchResult:
    outputs: tuple
    state: np.ndarray
    reason: np.ndarray
    T: np.ndarray
    program_head: np.ndarray
    data_head: np.ndarray
    program_left: np.ndarray
    program_right: np.ndarray
    data_left: np.ndarray
    data_right: np.ndarray
    program_tape: object = None
    data_tape: object = None

    @property
    def halted(self):
        return self.state == HALT

    @property
    def valid(self):
        return self.reason == 0

    def row(self, index=0):
        return dict(output=self.outputs[index], halted=bool(self.halted[index]),
                    reason=REASONS[self.reason[index]], state=STATE_NAMES[self.state[index]],
                    T=int(self.T[index]), program_head=int(self.program_head[index]),
                    data_head=int(self.data_head[index]),
                    program_space=(int(self.program_left[index]), int(self.program_right[index]),
                                   int(self.program_left[index] + self.program_right[index] + 1)),
                    data_space=(int(self.data_left[index]), int(self.data_right[index]),
                                int(self.data_left[index] + self.data_right[index] + 1)))

    @classmethod
    def concatenate(cls, results):
        fields = {}
        for name in cls.__dataclass_fields__:
            values = [getattr(r, name) for r in results]
            fields[name] = (sum(values, ()) if name == "outputs" else
                            None if values[0] is None else np.concatenate(values))
        return cls(**fields)


@dataclass
class Evaluation:
    totals: np.ndarray
    cases: int
    k_penalty: float

    @property
    def accuracy(self):
        return self.totals[:, MATCH] / self.totals[:, DENOM]

    @property
    def invalid_fraction(self):
        return self.totals[:, FAILED] / self.cases

    @property
    def fitness(self):
        return self.accuracy - self.k_penalty * self.invalid_fraction

    @property
    def exact(self):
        return self.totals[:, EXACT] == self.cases

    def resource_mean(self, column):
        return np.divide(self.totals[:, column], self.totals[:, SUCCESS],
                         out=np.full(len(self.totals), np.nan), where=self.totals[:, SUCCESS] != 0)


class BatchBackend:
    """Small common orchestration; backend methods implement physical execution."""

    def __init__(self, config, capacity):
        integer("capacity", capacity, 1)
        self.config, self.capacity = config, capacity

    def evaluate(self, values, lengths, dataset, k_penalty=0.5):
        validate_population(values, lengths, self.config)
        if not np.isfinite(k_penalty) or k_penalty < 0:
            raise ValueError("k_penalty must be finite and nonnegative")
        self.config.check_dataset(dataset)
        self._prepare(dataset)
        totals = np.empty((len(values), 8), dtype=np.uint64)
        for start in range(0, len(values), self.capacity):
            end = min(start + self.capacity, len(values))
            self._load(values[start:end])
            self._reset_totals(end - start)
            for case in range(len(dataset.inputs)):
                self._case(case, end - start, capture=False, tapes=False)
            totals[start:end] = self._get_totals(end - start)
        return Evaluation(totals, len(dataset.inputs), float(k_penalty))

    def simulate(self, values, lengths, input_bits, tapes=False):
        validate_population(values, lengths, self.config)
        dataset = Dataset((input_bits,), ("",))
        self.config.check_dataset(dataset)
        self._prepare(dataset)
        results = []
        for start in range(0, len(values), self.capacity):
            end = min(start + self.capacity, len(values))
            self._load(values[start:end])
            self._reset_totals(end - start)
            results.append(self._case(0, end - start, capture=True, tapes=tapes))
        return BatchResult.concatenate(results)


def plot_history(history):
    """Plot recorded diagnostics without touching the trainer or its RNG."""
    import matplotlib.pyplot as plt
    if not history:
        raise ValueError("no recorded diagnostics")
    x = [r["generation"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for key in ("current_best", "global_best", "champion_fitness_mean"):
        axes[0, 0].plot(x, [r[key] for r in history], label=key)
    axes[0, 0].fill_between(x, [r["champion_fitness_q20"] for r in history],
                          [r["champion_fitness_q80"] for r in history], alpha=.2)
    for key in ("best_accuracy", "champion_invalid", "population_invalid"):
        axes[0, 1].plot(x, [r[key] for r in history], label=key)
    for ax, metric in ((axes[1, 0], "time"), (axes[1, 1], "p_space"), (axes[1, 1], "x_space")):
        ax.plot(x, [r[metric + "_mean"] for r in history], label=metric)
        ax.fill_between(x, [r[metric + "_q20"] for r in history],
                        [r[metric + "_q80"] for r in history], alpha=.2)
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=.25)
        ax.set_xlabel("generation")
    fig.tight_layout()
    return fig
