# Implementation validation

Completed on Windows, 2026-10-15, using conda `slackenv`:
Python 3.10.13, NumPy 1.26.4, Taichi 1.7.3, Numba 0.61.2,
Matplotlib 3.10.3, nbclient 0.11.0 and JupyterLab 4.6.3.
CUDA device: NVIDIA GeForce GTX 1650, 4 GiB VRAM.
No package, driver, global Git setting or older project source was changed.

## Physical machine and interface

All 144 transition rows match the original LUTM-6 exactly. Canonical table hash:

```text
88ac3e58f4525dccd286fd723808f051687fbccc1ac02d0fb4e41b8e5f2eba7a
```

A development-only comparison with the original scalar matched **621 complete
configurations**: program/data dictionaries, state, both heads and actual T.
No old-project import or path is needed by the delivered code or tests.
Original machine.py, programs.py and validation.py source hashes remained unchanged.

The final core command, `python run_tests.py`, completed **27 tests, all passed,
none skipped**, in 29.847 seconds. Its coverage includes:

- Golden pair encodings, empty strings, leading zeros, marker termination and
  the exact table fingerprint.
- Self-extending scalar tapes and visited blank space.
- Full state/head/T/tape comparisons against the dictionary reference for short
  exhaustive and randomly generated raw patterns. Internal B genes and explicit
  trailing-B lengths are included; patterns wider than 62 symbols are tested.
- Native NOT, parity and append outputs, plus native/generic reverse,
  increment, ×2 and popcount on the CUDA backend.
  The final suite caps those kernels at five million transitions and uses only
  short popcount examples; wider physical popcount validation runs on CPU.
- Exact-budget halts; timeout and left/right data crossings; no credit for
  unfinished output; deterministic boundary precedence.
- Both program-boundary predicates. With sufficient padding, ordinary execution
  cannot cross the read-only program boundary; no fabricated reachable example
  is claimed. All H-entering rows have zero head displacement, so simultaneous
  H-entry/crossing is likewise unreachable for this table from an in-bounds state.
- Full decoded length for scoring, missing/extra bits, empty versus empty,
  length-weighted aggregation, soft failure penalties and undefined resource means.
- Independent input resets; target-independent execution; microbatch splitting,
  final partial batches and population permutation.
- **201 synthetic finite halted buffers** comparing the scalar, NumPy and CUDA
  decoders, including a final-cell leader 1 with an exterior blank payload. These
  are decoder fixtures, not claims that every fixture is reachable from initialization.
- Compiler write/move completion before halt, all three virtual movement
  directions, and a partial function with successful and bounded nonhalting branches.
- Acceptance of the uint32 maximum budget on an immediately halting program;
  no four-billion-transition CUDA experiment was needed or attempted.
- Tournament threshold endpoints/adjacent floats, neutral elite priority,
  segment bounds and inherited lengths, structural/categorical mutation,
  inactive padding, migration source exclusion and exact-stop behavior.
- Identical evolving populations, counters and RNG states with NumPy versus
  CUDA simulation, including checkpoint continuation on the other backend.
- Continuous versus resumed evolution, diagnostics/held-out RNG independence,
  generation-zero stopping and interruption rollback with a saved committed state.

No incorrect physical output or transition correspondence was found in these
completed checks. This is finite implementation evidence, not a formal proof.

## Arithmetic and long literal programs

`python tests/validate_known.py --physical` passed.

For each of increment, ×2, popcount and square, the independent four-symbol source
simulator checked **516 inputs**: every binary string through length eight and
five additional strings of lengths 16, 31, 64, 127 and 256. The square test also
checks its arithmetic loop invariant at each pass and before extraction.
These wide-input results are **source-machine** checks, not long physical runs.

The test-only CPU physical runner executes one canonical physical transition
per loop iteration. It was checked against the public unbounded scalar on six
selected native/compiled cases, including the complete final data tape.
The independent task oracles never write the real execution tape.

Selected actual square-literal results:

| Literal input | Decoded output | Physical T |
|---|---|---:|
| empty | `0` | 1,408 |
| `0` | `0` | 2,566 |
| `1` | `1` | 196,542,532 |
| `10` | `100` | 996,929,339 |
| `101` | `11001` | 2,141,210,538 |
| `110` | `100100` | 2,482,001,531 |
| `00110` | `100100` | 2,758,401,687 |

Thus nonzero, asymmetric and leading-zero inputs were checked, not only zero
or all-ones inputs. These used the same **1,449,178-symbol** program, encoder
and decoder. Literal SHA-256, excluding the file's terminating newline:

```text
dd2b47f14499a7de895d3fb9ef92e753f2639e9ab04313d43adc56f400258b66
```

Additional CPU literal checks covered generic reverse, increment (including a
32-one carry chain), ×2, and popcount. Arithmetic output is canonical binary;
append0 instead preserves literal input bits and appends zero.

Arbitrary-length witness claims rest on algorithmic invariants, not this finite
list. Some larger square inputs need more than uint32 T and are intentionally
outside the bounded production backends' budget range. Square was not run on CUDA.

## Training and usability

`python tests/stress_training.py` completed one controlled larger exercise,
preceded by small and medium runs:

| Population | New generations | X arena | T_max | Inputs |
|---:|---:|---:|---:|---:|
| 128 | 5 | 100 cells | 8,000 | 3 |
| 2,048 | 6 | 100 cells | 8,000 | 3 |
| 40,000 | 2 | 100 cells | 8,000 | 3 |

Each also evaluated generation zero. CUDA capacity was 2,048, independent of
population size. The warmed one-input preflight took **0.007 seconds**, including
transfers. The 40k trainer initialization plus two generations took about
**1.2 seconds**; the complete exercise including a NumPy sample and checkpoint
inspection took **3.51 seconds**. These are observed timings, not benchmark promises.
All sampled CPU counts and saved populations agreed. No watchdog setting changed.
This large exercise preceded the decoder-helper extraction; the final 27-test
suite verifies the same execution and GA contracts after that refactor.

Both committed notebooks executed in order: seven code cells in simulation,
nine in training/evaluation, including plotting, held-out inspection and export.
They remain output-free on disk. The default 300-generation demonstration ran
from random initialization: best training accuracy 0.75, held-out accuracy about
0.429; no exact NOT solution was found. This validates the workflow, not a claim
of successful arbitrary-length synthesis or a favorable search landscape.
Generated example runs are ignored by Git and are not curated program seeds.

## Errors encountered and their resolution

1. The restricted Windows process launcher initially failed with a DLL relocation
   error before Python started. Running the same installed conda environment with
   the required permissions worked. No environment modification was made.
2. One new scalar-space test incorrectly demanded negative nonblank cells on the
   final tape. NOT visits negative blanks but need not leave nonblank symbols
   there. The test now checks visited-head space; the simulator was unchanged.
3. One new square invariant test checked every visit to a scanning state instead
   of entry to that phase. Restricting it to phase entry fixed the audit. No
   arithmetic or transition rule changed.
4. A one-off multiline command was rejected by the installed Conda CLI before
   Python ran. No project operation occurred. A notebook metadata patch format
   was also rejected atomically; normal update patches were then used.
5. Extracting CUDA readout into a testable function initially omitted its
   `ti.template()` array annotation. Five CUDA tests stopped at compilation with
   the same cause; no physical run or wrong output was produced. The annotation
   was corrected, and the final full 27-test suite passed. A fixture's uint8 fill
   warning was removed with an explicitly typed input array.
6. The first notebook harness left kernel shutdown to parent-process exit, which
   emitted shutdown warnings after successful execution. The harness now closes
   each kernel explicitly; notebook semantics were not changed.

The previously identified source serializer Boolean-stringification defect was
not carried over: construction-time bits/directions are checked as actual
integers. This changes validation of the optional serializer, **not** the UTM
table, input encoding or runtime behavior of any raw program.

These validation results were recorded before the initial publication to
`EmergentComputing/LUTM-6`. Publishing does not change the machine or test scope.
Universal plasticity remains a separate future direction outside this repository.
