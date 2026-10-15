"""Exact physical transitions, vectorized across programs with active masks."""

import numpy as np

from utils import (BatchBackend, BatchResult, HALT, ONE, OUTPUT_PAD, dense_table,
                   boundary_reasons)


def decode_buffers(tapes, heads, valid):
    """Final-head readout; exterior lookahead reads B without moving any head."""
    n, size = tapes.shape
    output = np.full((n, (size + 1) // 2), OUTPUT_PAD, dtype=np.uint8)
    lengths = np.zeros(n, dtype=np.int32)
    positions = heads.astype(np.int64, copy=True)
    live = np.flatnonzero(valid)
    while len(live):
        pos = positions[live]
        leaders = np.zeros(len(live), dtype=np.uint8)
        payload = np.zeros(len(live), dtype=np.uint8)
        inside = (pos >= 0) & (pos < size)
        leaders[inside] = tapes[live[inside], pos[inside]]
        inside2 = (pos + 1 >= 0) & (pos + 1 < size)
        payload[inside2] = tapes[live[inside2], pos[inside2] + 1]
        keep = (leaders == ONE) & ((payload == 0) | (payload == ONE))
        live, payload = live[keep], payload[keep]
        output[live, lengths[live]] = (payload == ONE).astype(np.uint8)
        lengths[live] += 1
        positions[live] += 2
    return output, lengths


class NumpySimulator(BatchBackend):
    def __init__(self, config, capacity=4096):
        super().__init__(config, capacity)
        self.lut = dense_table()

    def _prepare(self, dataset):
        self.dataset = dataset
        self.inputs, self.targets, self.target_lengths = dataset.arrays()

    def _load(self, values):
        c = self.config
        self.p = np.zeros((len(values), c.program_size), dtype=np.uint8)
        self.p[:, c.program_left_budget:c.program_left_budget + values.shape[1]] = values
        self.initial_p_right = np.max(np.where(values != 0, np.arange(values.shape[1]), 0), axis=1).astype(np.int32)

    def _reset_totals(self, n):
        self.totals = np.zeros((n, 8), dtype=np.uint64)

    def _get_totals(self, n):
        return self.totals.copy()

    def _case(self, case, n, capture, tapes):
        c = self.config
        x = np.zeros((n, c.data_size), dtype=np.uint8)
        encoded_length = 2 * len(self.dataset.inputs[case]) + 2
        x[:, c.data_left_budget:c.data_left_budget + encoded_length] = self.inputs[case, :encoded_length]
        hp = np.full(n, c.program_left_budget, dtype=np.int32)
        hx = np.full(n, c.data_left_budget, dtype=np.int32)
        state = np.zeros(n, dtype=np.int32)
        count = np.zeros(n, dtype=np.uint32)
        reason = np.zeros(n, dtype=np.uint8)
        p_left = np.zeros(n, dtype=np.int32)
        p_right = self.initial_p_right.copy()
        x_left = np.zeros(n, dtype=np.int32)
        x_right = np.full(n, encoded_length - 1, dtype=np.int32)
        active = np.arange(n)
        write, move_p, move_x, next_state = self.lut
        while len(active):
            at = (state[active], self.p[active, hp[active]], x[active, hx[active]])
            x[active, hx[active]] = write[at]
            hp[active] += move_p[at]
            hx[active] += move_x[at]
            state[active] = next_state[at]
            count[active] += np.uint32(1)
            pl = hp[active] - c.program_left_budget
            xl = hx[active] - c.data_left_budget
            p_left[active] = np.maximum(p_left[active], -pl)
            p_right[active] = np.maximum(p_right[active], pl)
            x_left[active] = np.maximum(x_left[active], -xl)
            x_right[active] = np.maximum(x_right[active], xl)
            why = boundary_reasons(pl, xl, c)
            timed_out = (why == 0) & (state[active] != HALT) & (count[active] == c.t_max)
            why[timed_out] = 5
            reason[active] = why
            active = active[(why == 0) & (state[active] != HALT)]

        valid = reason == 0
        output, lengths = decode_buffers(x, hx, valid)

        target_length = int(self.target_lengths[case])
        overlap = min(target_length, c.output_capacity)
        matches = np.sum(output[:, :overlap] == self.targets[case, :overlap], axis=1, dtype=np.uint64)
        if target_length == 0:
            matches[valid & (lengths == 0)] = 1
        matches[~valid] = 0
        den = np.maximum(1, np.maximum(lengths, target_length)).astype(np.uint64)
        den[~valid] = max(1, target_length)
        exact = valid & (lengths == target_length) & (matches == den)
        # With no output or target, the explicit empty-string match is exact too.
        p_span = p_left.astype(np.uint64) + p_right.astype(np.uint64) + 1
        x_span = x_left.astype(np.uint64) + x_right.astype(np.uint64) + 1
        self.totals += np.column_stack((matches, den, ~valid, exact, valid,
                                       np.where(valid, count, 0), np.where(valid, p_span, 0),
                                       np.where(valid, x_span, 0))).astype(np.uint64)
        if not capture:
            return None
        outputs = tuple("".join(map(str, row[:length])) if ok else None
                        for row, length, ok in zip(output, lengths, valid))
        return BatchResult(outputs, state, reason, count, hp - c.program_left_budget,
                           hx - c.data_left_budget, p_left, p_right, x_left, x_right,
                           self.p.copy() if tapes else None, x if tapes else None)
