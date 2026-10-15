"""Length-independent witnesses; all run through the single fixed interface."""

from collections import deque

from machine import B, serialize

CODE = {B: "00", "0": "10", "1": "11", "#": "01"}
DECODE = {value: key for key, value in CODE.items()}
HALT = "halt"


def compile_blocks(transitions, start=0):
    """Four-symbol TM -> binary TM -> fixed physical program.

    The source input is literal x#; the public E(x) is already its FIXED
    two-bit block encoding. No source-dependent input preprocessing exists.
    Source transitions: (q,a) -> (next_q, write_symbol, -1/0/+1).
    The source halts with its head on the first output symbol.
    """
    states = {q for q, _ in transitions}
    if start != HALT and start not in states:
        raise ValueError("unknown source start state")
    for q in states:
        for symbol in CODE:
            if (q, symbol) not in transitions:
                raise ValueError("source table must be total on B,0,1,#")
            target, write, direction = transitions[q, symbol]
            if target not in states | {HALT} or write not in CODE or direction not in (-1, 0, 1):
                raise ValueError("invalid source action")

    def action(controller, bit):
        phase, *args = controller
        if phase == "enter":
            q = args[0]
            if q == HALT:
                return None
            return bit, 1, ("second", q, bit)
        if phase == "second":
            q, first = args
            symbol = DECODE[str(first) + str(bit)]
            target, write, direction = transitions[q, symbol]
            u, v = map(int, CODE[write])
            return v, -1, ("left", target, u, direction)
        if phase == "left":
            target, u, direction = args
            return u, -1 if direction == -1 else 1, ("bridge", target, direction)
        if phase == "bridge":
            target, direction = args
            return bit, 1 if direction == 1 else -1, ("enter", target)
        raise AssertionError(controller)

    initial = ("enter", start)
    indexes, pending, rules = {initial: 0}, deque([initial]), []
    while pending:
        controller = pending.popleft()
        row = []
        for bit in (0, 1):
            item = action(controller, bit)
            if item is None:
                row.append(None)
                continue
            write, direction, target = item
            if target not in indexes:
                indexes[target] = len(indexes)
                pending.append(target)
            row.append((write, direction, indexes[target]))
        rules.append(tuple(row))
    return rules


def not_rules():
    # Scan leaders, flip payload bits; rewind by pairs to the left blank.
    return [((0, -1, 2), (1, 1, 1)),
            ((1, 1, 0), (0, 1, 0)),
            ((0, -1, 3), (1, -1, 3)),
            ((0, 1, 4), (1, -1, 2)),
            ((0, 1, 5), (1, 1, 5)),
            (None, None)]


def parity_rules():
    return [((1, 1, 4), (1, 1, 2)),
            ((1, 1, 5), (1, 1, 3)),
            ((0, 1, 0), (1, 1, 1)),
            ((0, 1, 1), (1, 1, 0)),
            ((0, -1, 6), (0, -1, 6)),
            ((1, -1, 6), (1, -1, 6)),
            (None, None)]


def append_rules(bit):
    if bit not in (0, 1):
        raise ValueError("append bit must be zero or one")
    return [((1, 1, 2), (1, 1, 1)),
            ((0, 1, 0), (1, 1, 0)),
            ((bit, -1, 3), (bit, -1, 3)),
            ((0, 1, 5), (1, -1, 4)),
            ((0, -1, 3), (1, -1, 3)),
            ((0, 1, 6), (1, 1, 6)),
            (None, None)]


def reverse_source():
    t = {}
    for a in CODE:
        t[0, a] = (1, a, -1)
        t[1, a] = (2, "#", 1)
        t[2, a] = (3, a, -1) if a == "#" else (2, a, 1)
        t[3, a] = ((9, a, 1) if a == "#" else
                   (3, a, -1) if a == B else (4 + int(a), B, 1))
        for bit in (0, 1):
            t[4 + bit, a] = (6 + bit, a, 1) if a == "#" else (4 + bit, a, 1)
            t[6 + bit, a] = (8, str(bit), -1) if a == B else (6 + bit, a, 1)
        t[8, a] = (3, a, -1) if a == "#" else (8, a, -1)
        t[9, a] = (HALT, a, 1) if a == "#" else (9, a, 1)
    return t


def reverse_rules():
    """Native pair-level reversal, with marked left and right boundaries.

    Remove the last remaining input pair and append its bit beyond the right
    boundary. Empty pairs mark consumed input. At completion return to output.
    """
    table = {
        "scan": ((0, -1, "take"), (1, 1, "scan_payload")),
        "take": ((0, -1, "lead0"), (1, -1, "lead1")),
        "lead0": ((0, -1, "take"), (0, 1, "seek_payload0")),
        "lead1": ((0, 1, "finish_skip"), (0, 1, "erase1")),
        "erase1": ((0, 1, "seek_lead1"), (0, 1, "seek_lead1")),
        "back_lead": ((0, -1, "take"), (1, -1, "back_payload")),
        "finish_payload": ((0, 1, "finish_lead"), (1, 1, "halt")),
        "halt": (None, None),
    }
    for name, direction, target in (
            ("init", -1, "mark"), ("scan_payload", 1, "scan"),
            ("back_payload", -1, "back_lead"),
            ("finish_skip", 1, "finish_lead"),
            ("finish_lead", 1, "finish_payload")):
        table[name] = tuple((bit, direction, target) for bit in (0, 1))
    table["mark"] = ((1, 1, "scan"), (1, 1, "scan"))
    for bit in (0, 1):
        table[f"seek_lead{bit}"] = tuple((b, 1, f"seek_payload{bit}") for b in (0, 1))
        table[f"seek_payload{bit}"] = ((0, 1, f"seek_lead{bit}"), (1, 1, f"append_lead{bit}"))
        table[f"append_lead{bit}"] = ((1, 1, f"append_payload{bit}"), (1, 1, f"append_skip{bit}"))
        table[f"append_skip{bit}"] = tuple((b, 1, f"append_lead{bit}") for b in (0, 1))
        table[f"append_payload{bit}"] = tuple((bit, -1, "back_lead") for _ in (0, 1))
    pending, indexes, rules = deque(["init"]), {"init": 0}, []
    while pending:
        row = []
        for action in table[pending.popleft()]:
            if action is None:
                row.append(None)
                continue
            bit, direction, target = action
            if target not in indexes:
                indexes[target] = len(indexes)
                pending.append(target)
            row.append((bit, direction, indexes[target]))
        rules.append(tuple(row))
    return rules


def witnesses():
    return {"identity": "", "bit_not": serialize(not_rules()),
            "parity": serialize(parity_rules()),
            "append0": serialize(append_rules(0)),
            "append1": serialize(append_rules(1)),
            "reverse": serialize(reverse_rules()),
            "compiled_reverse": serialize(compile_blocks(reverse_source()))}



class Builder:
    def __init__(self):
        self.rows = {}

    def add(self, name, actions):
        if name in self.rows:
            raise ValueError(f"duplicate source state {name}")
        self.rows[name] = {a: actions.get(a, ("trap", a, 0)) for a in CODE}

    def move(self, name, distance, target):
        if distance == 0:
            self.add(name, {a: (target, a, 0) for a in CODE})
            return
        direction = 1 if distance > 0 else -1
        for i in range(abs(distance)):
            here = name if i == 0 else f"{name}.{i}"
            nxt = target if i + 1 == abs(distance) else f"{name}.{i+1}"
            self.add(here, {a: (nxt, a, direction) for a in CODE})

    def build(self):
        self.add("trap", {a: ("trap", a, 0) for a in CODE})
        names = tuple(self.rows)
        ids = {name: i for i, name in enumerate(names)}
        transitions = {}
        for name, row in self.rows.items():
            for a, (target, write, direction) in row.items():
                if target != HALT and target not in ids:
                    raise ValueError(f"undefined source state {target}")
                transitions[ids[name], a] = (HALT if target == HALT else ids[target], write, direction)
        return transitions, names


def square_source():
    """Binary grade-school n*n; canonical output, empty and leading zeros allowed.

    Input x# becomes # [A_i M_i R_i]* #, LSB first. A is shifted right,
    M left, and selected M is added to R. Columns grow on carry, not by a
    precomputed length budget. Final R is emitted MSB first, trimming zeros.
    """
    b = Builder()
    b.add("zero_scan", {"0": ("zero_scan", "0", 1),
                        "1": ("rewind_input", "1", -1),
                        "#": (HALT, "0", 0)})
    b.add("rewind_input", {a: ("rewind_input", a, -1) for a in "01"} |
          {B: ("scan_input", "#", 1)})
    b.add("scan_input", {a: ("scan_input", a, 1) for a in "01"} |
          {"#": ("write_work_end", "#", 1)})
    b.add("write_work_end", {B: ("to_input_last", "#", -1)})
    b.move("to_input_last", -1, "take_input")
    b.add("take_input", {B: ("take_input", B, -1),
                         "0": ("seek_sep0", B, 1),
                         "1": ("seek_sep1", B, 1),
                         "#": ("find_work_start", "#", 1)})
    b.add("find_work_start", {B: ("find_work_start", B, 1),
                             "#": ("select", "#", 1)})
    for bit in (0, 1):
        s = str(bit)
        b.add(f"seek_sep{bit}", {a: (f"seek_sep{bit}", a, 1) for a in (B, "0", "1")} |
              {"#": (f"append_a{bit}", "#", 1)})
        b.add(f"append_a{bit}", {a: (f"append_a{bit}", a, 1) for a in "01"} |
              {"#": (f"append_m{bit}", s, 1)})
        b.add(f"append_m{bit}", {B: (f"append_r{bit}", s, 1)})
        b.add(f"append_r{bit}", {B: ("append_end", "0", 1)})
    b.add("append_end", {B: ("return_sep", "#", -1)})
    b.add("return_sep", {a: ("return_sep", a, -1) for a in "01"} |
          {"#": ("take_input", "#", -1)})

    # Select and divide A by two using one-cell-column lookahead.
    b.add("select", {str(s): (f"first_look{s}", str(s), 1) for s in (0, 1)})
    for selected in (0, 1):
        b.move(f"first_look{selected}", 2, f"look{selected}0")
        for any_one in (0, 1):
            tag = f"{selected}{any_one}"
            actions = {str(v): (f"backwrite{selected}{any_one|v}{v}", str(v), -1)
                       for v in (0, 1)}
            actions["#"] = (f"last_back{tag}", "#", -1)
            b.add(f"look{tag}", actions)
            b.move(f"last_back{tag}", -2, f"last_zero{tag}")
            b.add(f"last_zero{tag}", {a: (f"rewind_add{tag}", "0", -1) for a in "01"})
            b.add(f"rewind_add{tag}", {a: (f"rewind_add{tag}", a, -1) for a in "01"} |
                  {"#": (f"add_a{selected}{any_one}00", "#", 1)})
        for any_one in (0, 1):
            for value in (0, 1):
                tag = f"{selected}{any_one}{value}"
                b.move(f"backwrite{tag}", -2, f"write_a{tag}")
                b.add(f"write_a{tag}", {a: (f"next_look{tag}", str(value), 1) for a in "01"})
                b.move(f"next_look{tag}", 5, f"look{selected}{any_one}")

    # One sweep: R += selected*M, M <<= 1. Carry plus previous M are finite bits.
    for selected in (0, 1):
        for remaining in (0, 1):
            for previous in (0, 1):
                for carry in (0, 1):
                    tag = f"{selected}{remaining}{previous}{carry}"
                    actions = {a: (f"add_m{tag}", a, 1) for a in "01"}
                    actions["#"] = ((f"grow_m{tag}", "0", 1) if previous or carry else
                                    (f"pass_return{remaining}", "#", -1))
                    b.add(f"add_a{tag}", actions)
                    b.add(f"add_m{tag}", {str(old): (f"add_r{selected}{remaining}{old}{carry}",
                                                   str(previous), 1) for old in (0, 1)})
                    b.add(f"add_r{tag}", {
                        str(r): (f"add_a{selected}{remaining}{previous}{(r+selected*previous+carry)//2}",
                                 str((r+selected*previous+carry) % 2), 1) for r in (0, 1)})
                    b.add(f"grow_m{tag}", {B: (f"grow_r{tag}", str(previous), 1)})
                    b.add(f"grow_r{tag}", {B: (f"grow_end{remaining}", str(carry), 1)})
    for remaining in (0, 1):
        b.add(f"grow_end{remaining}", {B: (f"pass_return{remaining}", "#", -1)})
        destination = "select" if remaining else "output_seek_end"
        b.add(f"pass_return{remaining}", {a: (f"pass_return{remaining}", a, -1) for a in "01"} |
              {"#": (destination, "#", 1)})

    # Extract R from high to low. One temporary B is a return cursor in the array.
    b.add("output_seek_end", {a: ("output_seek_end", a, 1) for a in "01"} |
          {"#": ("output_read0", "#", -1)})
    for emitted in (0, 1):
        b.add(f"output_read{emitted}", {
            "0": ((f"output_skip{emitted}", "0", -1) if not emitted else ("out_seek0", B, 1)),
            "1": ("out_seek1", B, 1),
            "#": (f"finish_seek{emitted}", "#", 1)})
        b.move(f"output_skip{emitted}", -2, f"output_read{emitted}")
        b.add(f"finish_seek{emitted}", {a: (f"finish_seek{emitted}", a, 1) for a in "01"} |
              {"#": (HALT if emitted else "emit_zero", "#", 1)})
    b.add("emit_zero", {B: (HALT, "0", 0)})
    for value in (0, 1):
        b.add(f"out_seek{value}", {a: (f"out_seek{value}", a, 1) for a in "01"} |
              {"#": (f"out_append{value}", "#", 1)})
        b.add(f"out_append{value}", {a: (f"out_append{value}", a, 1) for a in "01"} |
              {B: (f"out_back{value}", str(value), -1)})
        b.add(f"out_back{value}", {a: (f"out_back{value}", a, -1) for a in "01"} |
              {"#": (f"out_cursor{value}", "#", -1)})
        b.add(f"out_cursor{value}", {a: (f"out_cursor{value}", a, -1) for a in "01"} |
              {B: ("out_next", str(value), -1)})
    b.move("out_next", -2, "output_read1")
    return b.build()


def increment_source():
    b = Builder()
    b.add("scan", {a: ("scan", a, 1) for a in "01"} |
          {"#": ("carry", B, -1)})
    b.add("carry", {"1": ("carry", "0", -1),
                    "0": ("left", "1", -1), B: ("left", "1", -1)})
    b.add("left", {a: ("left", a, -1) for a in "01"} |
          {B: ("trim", B, 1)})
    b.add("trim", {"0": ("trim", B, 1), "1": (HALT, "1", 0)})
    return b.build()


def popcount_source():
    """Canonical binary count of ones; fixed-width counter allocated internally."""
    b = Builder()
    b.add("start", {a: ("mark_left", a, -1) for a in CODE})
    b.add("mark_left", {B: ("seek_separator", "#", 1)})
    b.add("seek_separator", {a: ("seek_separator", a, 1) for a in "01"} |
          {"#": ("initial_counter", "#", 1)})
    b.add("initial_counter", {B: ("back_separator", "0", -1)})
    b.add("back_separator", {"#": ("rewind_alloc", "#", -1)})
    b.add("rewind_alloc", {a: ("rewind_alloc", a, -1) for a in "01"} |
          {"#": ("allocate", "#", 1)})
    b.add("allocate", {str(bit): (f"alloc_seek{bit}", B, 1) for bit in (0, 1)} |
          {"#": ("rewind_count", "#", -1)})
    for bit in (0, 1):
        tag = str(bit)
        b.add(f"alloc_seek{bit}", {a: (f"alloc_seek{bit}", a, 1) for a in "01"} |
              {"#": (f"alloc_counter{bit}", "#", 1)})
        b.add(f"alloc_counter{bit}", {a: (f"alloc_counter{bit}", a, 1) for a in "01"} |
              {B: (f"alloc_back{bit}", "0", -1)})
        b.add(f"alloc_back{bit}", {a: (f"alloc_back{bit}", a, -1) for a in "01"} |
              {"#": (f"alloc_cursor{bit}", "#", -1)})
        b.add(f"alloc_cursor{bit}", {a: (f"alloc_cursor{bit}", a, -1) for a in "01"} |
              {B: ("allocate", tag, 1)})
    b.add("rewind_count", {a: ("rewind_count", a, -1) for a in "01"} |
          {"#": ("count", "#", 1)})
    b.add("count", {"0": ("count", B, 1), "1": ("count_seek", B, 1),
                    "#": ("trim", "#", 1)})
    b.add("count_seek", {a: ("count_seek", a, 1) for a in "01"} |
          {"#": ("counter_end", "#", 1)})
    b.add("counter_end", {a: ("counter_end", a, 1) for a in "01"} |
          {B: ("carry", B, -1)})
    b.add("carry", {"1": ("carry", "0", -1), "0": ("return_sep", "1", -1)})
    b.add("return_sep", {a: ("return_sep", a, -1) for a in "01"} |
          {"#": ("return_cursor", "#", -1)})
    b.add("return_cursor", {a: ("return_cursor", a, -1) for a in "01"} |
          {B: ("count", B, 1)})
    b.add("trim", {"0": ("trim", B, 1), "1": (HALT, "1", 0), B: (HALT, "0", 0)})
    return b.build()


def times_two_source():
    """Strip leading zeros, append zero; zero/empty input returns canonical 0."""
    b = Builder()
    b.add("trim", {"0": ("trim", B, 1), "1": ("scan", "1", 1),
                   "#": (HALT, "0", 0)})
    b.add("scan", {a: ("scan", a, 1) for a in "01"} |
          {"#": ("left", "0", -1)})
    b.add("left", {a: ("left", a, -1) for a in "01"} |
          {B: (HALT, B, 1)})
    return b.build()


# Independent task oracles; no oracle is ever consulted by physical execution.
TASKS = {
    "identity": lambda x: x,
    "bit_not": lambda x: "".join("1" if c == "0" else "0" for c in x),
    "parity": lambda x: str(x.count("1") % 2),
    "append0": lambda x: x + "0",
    "append1": lambda x: x + "1",
    "reverse": lambda x: x[::-1],
    "compiled_reverse": lambda x: x[::-1],
    "increment": lambda x: bin(int(x or "0", 2) + 1)[2:],
    "times_two": lambda x: bin(2 * int(x or "0", 2))[2:],
    "popcount": lambda x: bin(x.count("1"))[2:],
    "square": lambda x: bin(int(x or "0", 2) ** 2)[2:],
}


def load_program(name):
    """Read a curated literal and verify its register length/hash (not execution syntax)."""
    import csv
    from hashlib import sha256
    from pathlib import Path
    folder = Path(__file__).resolve().parent / "known_programs"
    with (folder / "manifest.csv").open(encoding="ascii", newline="") as file:
        records = {r["name"]: r for r in csv.DictReader(file)}
    record = records[name]
    pattern = (folder / record["file"]).read_text(encoding="ascii").removesuffix("\n")
    if (len(pattern) != int(record["length"]) or any(s not in "B01#" for s in pattern) or
            sha256(pattern.encode("ascii")).hexdigest() != record["sha256"]):
        raise ValueError(f"damaged known program: {name}")
    return pattern


def export_register():
    """Reproduce the checked-in literal data from the finite source algorithms."""
    import csv
    from hashlib import sha256
    from pathlib import Path
    folder = Path(__file__).resolve().parent / "known_programs"
    folder.mkdir(exist_ok=True)
    patterns = witnesses()
    for name, build in (("increment", increment_source), ("times_two", times_two_source),
                        ("popcount", popcount_source), ("square", square_source)):
        transitions, _ = build()
        patterns[name] = serialize(compile_blocks(transitions))
    with (folder / "manifest.csv").open("w", encoding="ascii", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("name", "file", "length", "sha256"), lineterminator="\n")
        writer.writeheader()
        for name, pattern in patterns.items():
            filename = name + ".txt"
            (folder / filename).write_text(pattern + "\n", encoding="ascii", newline="\n")
            writer.writerow(dict(name=name, file=filename, length=len(pattern),
                                 sha256=sha256(pattern.encode("ascii")).hexdigest()))
            print(f"{name}: {len(pattern):,} symbols", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-register", action="store_true")
    arguments = parser.parse_args()
    if arguments.export_register:
        export_register()
    else:
        parser.print_help()
