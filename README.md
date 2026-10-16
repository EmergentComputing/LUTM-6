# LUTM-6: a fixed-interface Turing machine for program search

LUTM stands for **Latent Universal Turing Machine**. This repository provides a
small, fixed machine and tools for discovering programs that transform binary
strings. You supply input–target examples; an island genetic algorithm searches
for a program that produces the targets. The machine's rules never change.

LUTM-6 has **two tapes, four symbols, nine running states, one halt state and
144 transition rows**. It includes an easy-to-inspect Python reference, NumPy
batching, CUDA batching through Taichi, a resumable genetic algorithm, notebooks,
and known programs for tasks including NOT, reverse, increment and square.

Research context: [*Emergent Models: Intelligence from Tiny Substrates*](https://arxiv.org/abs/2608.14019)
introduces the fixed-rule, fixed-interface latent-universality framework that
motivates this project. This repository develops the particular LUTM-6 machine.

## Why a fixed interface matters

Classical universality says that a suitable universal machine can simulate any
algorithm when given an appropriate description. Here we additionally care about
keeping the **data interface fixed while the algorithm changes**.

For example, a reverse program and a square program receive the same input string
through exactly the same encoder, at the same location. The encoder does not know
which program will run—not even that program's length. The same decoder reads
both results. A program may build whatever private working representation it needs
*during execution*, but the experiment does not prepare that representation for it.

This separates the algorithm from data injection and readout. It lets a search
algorithm try many raw programs against the same examples without changing the
interface for each candidate or task. Latent universality is not a stronger class
of computability than Turing universality; it emphasizes this universal interface.

The intended requirement is length-independent: for each computable binary-string
function, one finite program should work on inputs of arbitrary length, not merely
up to a compiled size limit. An evolved program can still overfit finite examples;
that is a search/generalization issue, distinct from the machine's expressivity.

**Status:** [CONSTRUCTION.md](CONSTRUCTION.md) gives an explicit constructive
argument for this machine's fixed-interface universality. The argument is not
formally verified or independently certified. The tests establish finite
implementation evidence, not a universality theorem or a proof that every learned
program generalizes. The construction uses only finite initial patterns on blank
tapes, not an infinite periodic background.

## What you can do

- Run any raw program pattern on a binary input and inspect its output, actual
  transition count, final heads, tapes and used space.
- Evaluate many programs in parallel on CPU or GPU using the same semantics.
- Search from random programs with an island GA, monitor progress, inspect
  island champions, and evaluate on held-out examples.
- Save a training run and resume its exact evolutionary state later.
- Study known length-independent algorithms and the optional witness compiler.

Start with the two notebooks:

| Notebook | Purpose |
|---|---|
| [01_simulate.ipynb](01_simulate.ipynb) | Run a single program, inspect tapes, run a batch, or check known tasks over an input-length range. |
| [02_train_eval.ipynb](02_train_eval.ipynb) | Edit examples and hyperparameters, train, plot progress, evaluate, export a program and resume checkpoints. |

## Installation and first run

The tested setup is **Windows, Python 3.10, conda `slackenv`**. CUDA execution needs
an NVIDIA CUDA-capable GPU and a working driver. CPU simulation does not need a GPU.

Clone the repository in PowerShell:

```powershell
git clone https://github.com/EmergentComputing/LUTM-6.git
cd LUTM-6
```

For a new environment:

```powershell
conda create -n slackenv python=3.10
conda activate slackenv
python -m pip install -r requirements.txt
python -m ipykernel install --user --name slackenv --display-name "Python (slackenv)"
jupyter lab
```

If you already use `slackenv` with these dependencies, activate it and run
`jupyter lab`; there is no need to recreate it. Select its kernel in the notebook.

In the simulation notebook, choose `scalar`, `numpy` or `cuda`. In the training
notebook, choose `numpy` or `cuda`, edit the input–target lists and budgets, then
run the cells in order. CUDA is explicit: there is **no automatic CPU fallback**.
Restart the notebook kernel before reinitializing Taichi/CUDA.

## How the machine works

### 1. Two tapes, one fixed transition table

The program tape **P** holds the candidate pattern. The writable data tape **X**
holds the input and becomes the working/output tape. Both are theoretically
bi-infinite and otherwise filled with `B`, the blank symbol. Their common physical
alphabet is `B,0,1,#`.

Both heads start at coordinate zero, in control state D. Each transition reads
the two scanned symbols and current state, writes a data symbol, moves the heads
and chooses the next state. Every row preserves the scanned program symbol:
**P is read-only throughout execution**, not restored afterward.

The program head is not restricted to a single forward scan. The table supports
rescanning and data-dependent control flow. For the optional compiled witnesses,
the machine selects a source instruction from the scanned data bit, performs its
write/move, then locates the next source instruction using a temporary data marker
and program scans. All of that work is performed by physical transitions inside
the machine; there is no external instruction executor.

Compiled instruction syntax is only one useful way to build programs. Raw searched
patterns need no well-formed instruction structure. There is no parser or reject
state: a pattern can halt, loop or produce an unintended result through the same
complete transition table. The GA does not compile or repair its candidates.

### 2. Write the input independently of the program

Write the raw program from coordinate zero on P. Independently, encode the input
on X from coordinate zero using:

| Binary input | Physical symbols on X |
|---|---|
| bit `0` | `1B` |
| bit `1` | `11` |
| end of input | `B1` |

For example:

```text
input ""     -> B1
input "0"    -> 1B | B1
input "010"  -> 1B | 11 | 1B | B1
```

The bars above are explanatory, not tape symbols. All other cells on both tapes
are B. There is no program-dependent offset, scratch prefix or input conversion.
Physical `0` and `#` on X are available as temporary machine markers; initial
input uses only `B` and `1`.

### 3. Halt and read the output from the final data head

The machine halts only when its control enters H. Starting at its **final X-head**,
decode consecutive pairs: `11` means one, `1B` means zero, and any other pair ends
the output. For example, `11 | 1B | 11 | BB` reads as `101`.

Empty output is valid. Neither head must return to zero; no separator must remain
at a prescribed location; scratch need not be erased. Garbage before the final
head or after the terminating pair is irrelevant. No output is credited before
halt. In particular, B as the payload of `1B` is a real zero, not a missing bit.

The full rules, compiler layout and simulation invariants are explained in
[CONSTRUCTION.md](CONSTRUCTION.md).

## Run a program from Python

The scalar reference uses sparse, self-extending tapes and Python-integer time:

```python
from machine import Machine
from programs import load_program

machine = Machine(load_program("bit_not"), "00101")
result = machine.run(1_000_000)
print(result["output"])  # 11010
print(result)            # actual T, halt/state, heads and used-space metrics
```

`run(n)` executes at most **n additional physical transitions**. If it times out,
you can call it again to continue that same computation. Replace the registered
literal with any raw string over `B,0,1,#`, including the empty program.

For a NumPy batch, use fixed-width arrays with explicit active lengths:

```python
from utils import SimulationConfig, pack_programs
from numpy_backend import NumpySimulator

config = SimulationConfig()
programs, lengths = pack_programs(["", load_program("bit_not")], config.max_program_length)
simulator = NumpySimulator(config)
batch = simulator.simulate(programs, lengths, "00101", tapes=True)
print(batch.row(0))  # identity, from the empty program
print(batch.row(1))  # NOT
```

For CUDA, replace the simulator construction with:

```python
from taichi_backend import initialize_cuda, TaichiSimulator

initialize_cuda()  # once per Python/notebook session
simulator = TaichiSimulator(config, capacity=2048)
```

The batch APIs are identical. Each input starts a fresh computation. Programs
are uint8 symbol arrays, not packed integers: there is no 62-bit program limit.
`capacity` is the simulator's microbatch size, separate from the total population.
Programs run in parallel; input cases and microbatches are evaluated serially.
CUDA training transfers compact statistics, not full tapes for every case.

### Theoretical tapes versus finite experiments

| Backend | Space | Per-computation time counter |
|---|---|---|
| Scalar Python | Self-extending sparse tapes | Python integer |
| NumPy | Explicitly padded P and X arrays | uint32 |
| Taichi/CUDA | The same bounded tape semantics | uint32 |

The bounded backends expose independent inclusive intervals
`[-left_budget, +right_budget]` for each tape. Initial content and adjacent blank
padding must fit. `max_program_length` is a genotype capacity, not an input offset.

Each transition writes, moves and increments T before checking boundaries.
The boundary cell is allowed; **crossing either side on either tape fails**.
Otherwise, halting on transition `t_max` succeeds; reaching that limit while still
running is a timeout. Failed cases return `None` and receive no partial-output
credit. There is no separate invalid-decoding condition.

`t_max` can be 1 through **4,294,967,295**; long CUDA kernels on display GPUs may
be unsuitable even within that numeric range. The implementation uses one bounded
kernel per input, not TM-step chunking, and changes no watchdog settings.

Used space is tracked separately for P and X. It includes initial nonblank content
and all head visits, including blank cells: `left_used + right_used + 1` is the
historical used span. Unvisited padding does not count. Decoder lookahead is not
head movement; untouched exterior cells read as B.

## Evolve programs from examples

The training notebook is the main workflow. Supply nonempty, equal-length lists of binary
input/target strings; individual empty strings are allowed. For example, NOT:

```python
train_inputs  = ["0", "1", "01", "110", "00101"]
train_targets = ["1", "0", "10", "001", "11010"]
```

No task function is needed for training. Targets are used only for scoring and
never affect initialization, physical execution or halting. Optional held-out
pairs never feed selection. Repeated training examples retain their supplied weight.

### Fitness

On a successfully halted case, count equal bits at aligned prediction/target
positions. Divide by the larger length, at least one. Missing and extra bits are
wrong: prediction `11` against target `1100` scores 2/4. Empty versus empty counts
as one match. A failed case has zero matches and denominator
`max(1, target_length)`.

Sum matches and denominators across cases **before dividing**, then apply:

```text
fitness = length-weighted bit accuracy - k_penalty * failed_case_fraction
```

The default penalty is 0.5. Failures are soft penalties, not hard exclusions from
selection. There is no length/time/space penalty within the budgets. Exact means
successful literal equality on every training case. `stop_first_exact=True`
stops on the first evaluated generation containing an exact program, including
generation zero. A perfect finite fit is not an arbitrary-length guarantee.

### Island GA and neutral turnover

Each island keeps its own population. Two parents are chosen by **independent
tournaments of two**, sampled with replacement within that island. The probability
of selecting the better tournament contender depends on their absolute fitness gap:

| Fitness gap | Better-contender probability |
|---|---:|
| exactly zero | 0.50 |
| above zero through 0.05 | 0.53 |
| above 0.05 through 0.20 | 0.58 |
| above 0.20 | 0.80 |

Thresholds and probabilities are configurable. This is discrete, temperature-like
selection—not a post-mutation Metropolis acceptance rule.

Each child starts as parent 1, inheriting its length. With configurable probability,
coordinate-aligned segments are copied from parent 2's shared active prefix.
Segment lengths/starts are sampled uniformly over feasible positions; overlaps are
allowed. Mutation then applies one insert/delete/stay choice, followed by independent
per-locus symbol mutations. Insertion appends a random symbol; deletion removes the
final active symbol. A blocked structural mutation is a no-op. A locus mutation
always changes to a different allowed symbol, and can affect a newly inserted locus.

The default search alphabet is `01#`; choose `B01#` to include active blanks.
Inactive array padding is B too, but explicit lengths preserve the distinction.
Initial programs and immigrants have random lengths and symbols. No compiled
program seeds are used unless explicitly supplied through `initial_programs`.

Elites are ranked by fitness, preferring **new children on exact ties**, then stable
index. This permits neutral turnover without a diversity metric or deduplication
rule. On migration generations, each migrant's donor island is chosen uniformly
from all other islands, then its individual is chosen uniformly from that island's
**whole population**, with replacement and no fitness or validity filter. Migrants
are copied unchanged; receiving islands keep their own elites. Migration fraction
and interval are configurable. Random immigrants are fresh programs. The copied
best-ever archive updates only on strict improvement
and is not continually reinserted into the population.

The full-population donor rule replaces the earlier elite-only policy. Resuming
an older checkpoint with this code uses the new migration rule on subsequent
migration generations; it does not reproduce the older policy's trajectory.

The GA runs in NumPy; either batched simulator can evaluate it. The notebook prints
progress during training and exposes per-island champions. Charts show current/global
best, champion means and q20/q80 bands, accuracy, invalid rates for champions and the
whole population, and successful-halt time/space means. No successful cases means
an undefined resource mean (NaN), not zero. Diagnostic cadence defaults to ten
generations and does not change the training RNG.

### Checkpoint and resume

Runs save to `runs/<run_id>/checkpoint.npz` with atomic replacement. A checkpoint
contains the population, explicit lengths, child flags, evaluation counters,
best-ever archive, history, train/test pairs, configs, exact NumPy RNG state and
machine/interface fingerprints. Loading does not require pickle.

```python
from evolution import Trainer, checkpoint_config
from numpy_backend import NumpySimulator

path = "runs/YOUR_RUN_ID/checkpoint.npz"
trainer = Trainer.resume(path, NumpySimulator(checkpoint_config(path)))
trainer.run(100)  # up to 100 additional generations
```

Saved configurations and datasets are authoritative on resume. You can choose
another backend or microbatch capacity without changing the evolutionary trajectory.
Use a new run for a different experiment. Ctrl+C in `run()` saves the last fully
evaluated generation and its RNG state, discarding partial offspring evaluation.
A running CUDA kernel may finish before Python handles the interrupt.

## Known programs, checks and limitations

[known_programs/manifest.csv](known_programs/manifest.csv) indexes literal program
files with lengths and SHA-256 hashes. `programs.py` supplies task oracles and
optional finite witness compilers. The register includes identity, NOT, parity,
append 0/1, native and generic reverse, increment, ×2, popcount and square.
Arithmetic interprets empty input as zero and returns canonical binary, including
on leading-zero inputs. Append0 preserves literal input bits and is therefore a
different task from numeric ×2.

The square witness is a length-independent algorithm, but its compiled program is
**1,449,178 physical symbols** long and very expensive. It demonstrates a computation,
not an efficient squaring implementation or a useful default training seed. Use
the CPU physical validator for its selected long runs, not a huge CUDA timeout.
The test-only Numba runner executes one physical transition per iteration; it
does not skip to source-level results or repair outputs.

From the repository folder in `slackenv`:

```powershell
python run_tests.py                         # core scalar/NumPy/CUDA/GA tests
python tests/validate_known.py              # arithmetic source algorithms/invariants
python tests/validate_known.py --physical   # also selected long literal CPU runs
python tests/validate_known.py --task bit_not --min-length 0 --max-length 5
python tests/stress_training.py             # explicit controlled 40k exercise
python tests/validate_notebooks.py          # execute notebooks without saving their outputs
```

`python run_tests.py --known --stress --notebooks` runs all validation groups.
The core suite includes CUDA tests, so that command requires a working CUDA GPU
even though normal scalar/NumPy use does not. CUDA unavailability is an error,
never a silent fallback. The stress script checks a warmed microbatch duration
before attempting its larger run.

[VALIDATION.md](VALIDATION.md) records the actual completed checks: 27 passing core
tests, original-machine comparisons, arithmetic checks including asymmetric square
inputs, and a controlled 40,000-program exercise. It also records encountered
errors and their resolutions. The default short GA demonstration did not find an
exact NOT solution; its purpose is a usable starting workflow, not a search-success
claim. Larger budgets and different hyperparameters are research choices.

This version does not implement self-modification, persistent macrosteps, a web
frontend or a program miner. Favorable evolutionary search behavior is an empirical
question, separate from fixed-interface expressivity.

## Repository guide

| File or directory | Responsibility |
|---|---|
| `machine.py` | Canonical transition table, fixed encoder/decoder and unbounded scalar reference. |
| `numpy_backend.py`, `taichi_backend.py` | Bounded batched physical execution. |
| `evolution.py` | Island GA, diagnostics, best archive and resumable checkpoints. |
| `utils.py` | Shared configs, datasets, representations, results and plotting. |
| `programs.py`, `known_programs/` | Optional witness builders, independent task oracles and literal register. |
| `tests/`, `run_tests.py` | Correctness and controlled validation commands. |
| `CONSTRUCTION.md` | Construction and fixed-interface universality argument, with explicit proof limits. |
| `IMPLEMENTATION_PLAN.md` | Detailed semantic contract and implementation rationale. |

The project is self-contained: it imports no older LUTM folders. Generated
`runs/`, caches and notebook checkpoints are excluded from Git.

## Note

Implemented with AI assistance and tested on Windows in conda `slackenv` (Python
3.10.13, NumPy 1.26.4, Taichi 1.7.3, NVIDIA GTX 1650). Linux has not been tested.
Tests and the construction argument may still contain errors; reports through
GitHub issues and independent scrutiny are welcome. Licensed under [MIT](LICENSE).
