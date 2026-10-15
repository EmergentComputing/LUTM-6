"""Direct writable-tape interpreter: two tapes, nine active states, 144 rows.

All raw patterns execute. Source-TM serialization is a witness construction,
not a parser or admission requirement. No imports from older LUTM projects.
"""

from dataclasses import dataclass
from itertools import product
from hashlib import sha256

B = "B"
SYMBOLS = (B, "0", "1", "#")
STATES = ("D", "S", "W", "M", "K", "C", "R", "J", "U")
H = "H"


@dataclass(frozen=True)
class Rule:
    p: str
    d: str
    dp: int
    dd: int
    state: str


def transition(state, p, d):
    if state == "D":
        if d in ("1", "#"):
            return Rule(p, d, 0, 0, "S")
        if p in "01":
            return Rule(p, B if p == "0" else "1", 1, 0, "M")
        return Rule(p, d, 0, 0, H)
    if state == "S":
        if p == "#":
            return Rule(p, d, 1, 0, "W")
        if p == B:
            return Rule(p, d, 0, 0, H)
        return Rule(p, d, 1, 0, "S")
    if state == "W":
        if p in "01":
            return Rule(p, B if p == "0" else "1", 1, 0, "M")
        return Rule(p, d, 0, 0, H)
    if state == "M":
        if p in "01":
            return Rule(p, d, 1, -1 if p == "0" else 1, "K")
        return Rule(p, d, 0, 0, H)
    if state == "K":
        # Mark the new source-head cell while retaining its underlying bit.
        return Rule(p, "#" if d in ("1", "#") else "0", 0, 0, "C")
    if state == "C":
        if p in "01":
            return Rule(p, d, 1, 1, "C")
        return Rule(p, d, -1, 0, "R")
    if state == "R":
        if p == B:
            return Rule(p, d, 1, 0, "J")
        return Rule(p, d, -1, 0, "R")
    if state == "J":
        if d in ("0", "#"):
            return Rule(p, B if d == "0" else "1", 0, 0, "D")
        return Rule(p, d, 0, -1, "U")
    if state == "U":
        if p == "#":
            return Rule(p, d, 1, 0, "J")
        if p == B:
            return Rule(p, d, 0, 0, H)
        return Rule(p, d, 1, 0, "U")
    raise ValueError(f"unknown physical state: {state}")


TABLE = {(s, p, d): transition(s, p, d)
         for s, p, d in product(STATES, SYMBOLS, SYMBOLS)}

TABLE_HASH = sha256(("\n".join(
    "|".join(map(str, (s, p, d, r.p, r.d, r.dp, r.dd, r.state)))
    for (s, p, d), r in TABLE.items()) + "\n").encode("ascii")).hexdigest()
INTERFACE_VERSION = "pairs-1B-11-B1-final-head-v1"


def encode_input(bits):
    """Fixed block code: virtual 0=1B, 1=11, end-marker=B1, blank=BB."""
    if not isinstance(bits, str) or any(c not in "01" for c in bits):
        raise ValueError("input must be a binary string")
    return "".join("1" + (B if bit == "0" else "1") for bit in bits) + B + "1"


def decode(tape, head):
    """Read leading-1 pairs; every reachable finite halted tape has a value."""
    output = []
    while tape.get(head, B) == "1":
        value = tape.get(head + 1, B)
        if value not in (B, "1"):
            break
        output.append("0" if value == B else "1")
        head += 2
    return "".join(output)


class Machine:
    def __init__(self, pattern, bits):
        if not isinstance(pattern, str) or any(c not in SYMBOLS for c in pattern):
            raise ValueError("pattern must use the physical alphabet B,0,1,#")
        self.p = {i: c for i, c in enumerate(pattern) if c != B}
        self.d = {i: c for i, c in enumerate(encode_input(bits)) if c != B}
        self.hp = self.hd = self.steps = 0
        self.state = "D"
        self.p_left = self.x_left = 0
        self.p_right = max(self.p, default=0)
        self.x_right = max(self.d, default=0)

    def step(self):
        if self.state == H:
            raise RuntimeError("cannot step a halted machine")
        rule = TABLE[self.state, self.p.get(self.hp, B), self.d.get(self.hd, B)]
        if rule.d == B:
            self.d.pop(self.hd, None)
        else:
            self.d[self.hd] = rule.d
        # All rules preserve the program symbol.
        self.hp += rule.dp
        self.hd += rule.dd
        self.state = rule.state
        self.steps += 1
        self.p_left = max(self.p_left, -self.hp)
        self.p_right = max(self.p_right, self.hp)
        self.x_left = max(self.x_left, -self.hd)
        self.x_right = max(self.x_right, self.hd)

    def run(self, budget):
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        for _ in range(budget):
            if self.state == H:
                break
            self.step()
        return {"halted": self.state == H,
                "output": decode(self.d, self.hd) if self.state == H else None,
                "T": self.steps, "state": self.state,
                "program_head": self.hp, "data_head": self.hd,
                "program_space": (self.p_left, self.p_right, self.p_left + self.p_right + 1),
                "data_space": (self.x_left, self.x_right, self.x_left + self.x_right + 1)}


def serialize(rules):
    """Witness only: each state has actions for logical zero (B), then one.

    Action None halts; otherwise (write_bit, direction, target_state).
    Direction is -1 or +1. A target counts 2*target_state records.
    """
    if not rules:
        raise ValueError("empty source table")
    chunks = []
    for row in rules:
        if len(row) != 2:
            raise ValueError("source states need two actions")
        for action in row:
            if action is None:
                chunks.append("#")
                continue
            write, direction, target = action
            if type(write) is not int or write not in (0, 1) or type(direction) is not int or direction not in (-1, 1):
                raise ValueError("source actions need a bit and a left/right move")
            if type(target) is not int or not 0 <= target < len(rules):
                raise ValueError("source target out of range")
            chunks.append(str(write) + str(int(direction == 1)) + "1" * (2 * target) + "#")
    return "".join(chunks)
