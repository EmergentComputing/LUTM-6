"""CUDA physical interpreter. One thread per program; cases run serially.

Initialization is explicit and has no CPU fallback. No step chunking, packed
program integers, fitness cache, or source-level execution shortcut is used.
"""

import numpy as np
import taichi as ti

from utils import BatchBackend, BatchResult, dense_table


def initialize_cuda():
    ti.init(arch=ti.cuda, enable_fallback=False, offline_cache=False,
            default_ip=ti.i32, default_fp=ti.f64)


@ti.data_oriented
class TaichiSimulator(BatchBackend):
    def __init__(self, config, capacity=4096):
        super().__init__(config, capacity)
        if ti.lang.impl.get_runtime().prog is None:
            raise RuntimeError("call initialize_cuda() before creating a TaichiSimulator")
        if ti.lang.impl.current_cfg().arch != ti.cuda:
            raise RuntimeError("TaichiSimulator requires CUDA; CPU fallback is not supported")
        self.p_size, self.x_size = config.program_size, config.data_size
        self.p_origin, self.x_origin = config.program_left_budget, config.data_left_budget
        self.out_size = config.output_capacity
        self.p = ti.field(ti.u8, shape=(capacity, self.p_size))
        self.x = ti.field(ti.u8, shape=(capacity, self.x_size))
        self.initial_p_right = ti.field(ti.i32, shape=capacity)
        self.totals = ti.field(ti.u64, shape=(capacity, 8))
        self.meta = ti.field(ti.i64, shape=(capacity, 10))
        self.output = ti.field(ti.u8, shape=(capacity, self.out_size))
        self.write = ti.field(ti.u8, shape=(9, 4, 4))
        self.move_p = ti.field(ti.i32, shape=(9, 4, 4))
        self.move_x = ti.field(ti.i32, shape=(9, 4, 4))
        self.next_state = ti.field(ti.i32, shape=(9, 4, 4))
        for field, array in zip((self.write, self.move_p, self.move_x, self.next_state), dense_table()):
            field.from_numpy(array)
        self.dataset = None

    def _prepare(self, dataset):
        if self.dataset == dataset:
            return
        inputs, targets, lengths = dataset.arrays()
        self.inputs = ti.ndarray(ti.u8, shape=inputs.shape)
        self.targets = ti.ndarray(ti.u8, shape=targets.shape)
        self.input_lengths = ti.ndarray(ti.i32, shape=len(lengths))
        self.target_lengths = ti.ndarray(ti.i32, shape=len(lengths))
        self.inputs.from_numpy(inputs)
        self.targets.from_numpy(targets)
        self.input_lengths.from_numpy(np.array([2 * len(x) + 2 for x in dataset.inputs], dtype=np.int32))
        self.target_lengths.from_numpy(lengths)
        self.dataset = dataset

    def _load(self, values):
        p = np.zeros((self.capacity, self.p_size), dtype=np.uint8)
        p[:len(values), self.p_origin:self.p_origin + values.shape[1]] = values
        self.p.from_numpy(p)
        support = np.zeros(self.capacity, dtype=np.int32)
        support[:len(values)] = np.max(np.where(values != 0, np.arange(values.shape[1]), 0), axis=1)
        self.initial_p_right.from_numpy(support)

    @ti.kernel
    def _reset_totals(self, n: ti.i32):
        for i, j in ti.ndrange(n, 8):
            self.totals[i, j] = ti.u64(0)

    def _get_totals(self, n):
        return self.totals.to_numpy()[:n].copy()

    @ti.func
    def _readout(self, i, hx, case, capture, targets: ti.template(), target_len):
        output_len = 0
        matches = ti.u64(0)
        position, reading = ti.i64(hx), 1
        while reading == 1:
            leader, payload = 0, 0
            if 0 <= position < self.x_size:
                leader = ti.i32(self.x[i, ti.i32(position)])
            if 0 <= position + 1 < self.x_size:
                payload = ti.i32(self.x[i, ti.i32(position + 1)])
            if leader == 2 and (payload == 0 or payload == 2):
                bit = ti.i32(payload == 2)
                if output_len < target_len:
                    if bit == targets[case, output_len]:
                        matches += ti.u64(1)
                if capture == 1:
                    self.output[i, output_len] = ti.u8(bit)
                output_len += 1
                position += 2
            else:
                reading = 0
        if output_len == 0 and target_len == 0:
            matches = ti.u64(1)
        return output_len, matches

    @ti.kernel
    def _execute(self, n: ti.i32, case: ti.i32, budget: ti.u32, capture: ti.i32,
                 inputs: ti.types.ndarray(dtype=ti.u8, ndim=2),
                 targets: ti.types.ndarray(dtype=ti.u8, ndim=2),
                 input_lengths: ti.types.ndarray(dtype=ti.i32, ndim=1),
                 target_lengths: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(n):
            input_len = input_lengths[case]
            target_len = target_lengths[case]
            for j in range(self.x_size):
                value = ti.u8(0)
                if self.x_origin <= j < self.x_origin + input_len:
                    value = inputs[case, j - self.x_origin]
                self.x[i, j] = value
            hp, hx, state, reason = self.p_origin, self.x_origin, 0, 0
            count = ti.u32(0)
            pl, pr, xl, xr = 0, self.initial_p_right[i], 0, input_len - 1
            while state != 9 and reason == 0 and count < budget:
                p = ti.i32(self.p[i, hp])
                x = ti.i32(self.x[i, hx])
                self.x[i, hx] = self.write[state, p, x]
                hp += self.move_p[state, p, x]
                hx += self.move_x[state, p, x]
                state = self.next_state[state, p, x]
                count += ti.u32(1)
                pl = ti.max(pl, self.p_origin - hp)
                pr = ti.max(pr, hp - self.p_origin)
                xl = ti.max(xl, self.x_origin - hx)
                xr = ti.max(xr, hx - self.x_origin)
                if hp < 0:
                    reason = 1
                elif hp >= self.p_size:
                    reason = 2
                elif hx < 0:
                    reason = 3
                elif hx >= self.x_size:
                    reason = 4
                elif state != 9 and count == budget:
                    reason = 5
            output_len = 0
            matches = ti.u64(0)
            den = ti.u64(ti.max(1, target_len))
            if reason == 0:
                output_len, matches = self._readout(i, hx, case, capture, targets, target_len)
                den = ti.u64(ti.max(1, ti.max(output_len, target_len)))
            valid = reason == 0
            exact = valid and output_len == target_len and matches == den
            self.totals[i, 0] += matches
            self.totals[i, 1] += den
            self.totals[i, 2] += ti.u64(not valid)
            self.totals[i, 3] += ti.u64(exact)
            self.totals[i, 4] += ti.u64(valid)
            if valid:
                self.totals[i, 5] += ti.u64(count)
                self.totals[i, 6] += ti.u64(pl) + ti.u64(pr) + ti.u64(1)
                self.totals[i, 7] += ti.u64(xl) + ti.u64(xr) + ti.u64(1)
            if capture == 1:
                self.meta[i, 0], self.meta[i, 1], self.meta[i, 2] = state, reason, ti.i64(count)
                self.meta[i, 3], self.meta[i, 4] = hp - self.p_origin, hx - self.x_origin
                self.meta[i, 5], self.meta[i, 6] = pl, pr
                self.meta[i, 7], self.meta[i, 8], self.meta[i, 9] = xl, xr, output_len

    def _case(self, case, n, capture, tapes):
        self._execute(n, case, np.uint32(self.config.t_max), int(capture), self.inputs,
                      self.targets, self.input_lengths, self.target_lengths)
        if not capture:
            return None
        meta = self.meta.to_numpy()[:n]
        output = self.output.to_numpy()[:n]
        outputs = tuple("".join(map(str, row[:length])) if reason == 0 else None
                        for row, length, reason in zip(output, meta[:, 9], meta[:, 1]))
        return BatchResult(outputs, meta[:, 0].astype(np.int32), meta[:, 1].astype(np.uint8),
                           meta[:, 2].astype(np.uint32), *[meta[:, j].astype(np.int32) for j in range(3, 9)],
                           self.p.to_numpy()[:n].copy() if tapes else None,
                           self.x.to_numpy()[:n].copy() if tapes else None)
