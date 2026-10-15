# LUTM-6 evolution: frozen specification and implementation plan

Date: 2026-10-15.

Status: **implemented; see VALIDATION.md for completed verification**. This document records
the agreed behavior and the engineering plan for a clean, standalone research
project. Implementation was subsequently authorized. The user later authorized
publication to `https://github.com/EmergentComputing/LUTM-6`; the no-push clauses
below record the original milestone boundary, not a current publication block.

The semantic contract below is frozen. Module organization and performance
choices may be simplified during implementation, but must not silently change
the machine, interface, fitness, or evolutionary rules.

## 1. Objective and scope

Create `LUTM-6-evolution` beside the existing LUTM folders. It will evolve raw
program-tape patterns for the current LUTM-6 using a temperature-like,
fitness-gap-based island GA. The physical transition table remains fixed.

Use the successful LUTM-1 evolutionary algorithm as the behavioral reference,
particularly `LUTM-1/island_ga.py` and the user's edited
`04_temperature_island_ga.ipynb`, rather than blindly copying old defaults.
The old v2 files are historical precursors, not the primary reference.

Deliverables:

- A self-contained implementation with no runtime imports from older folders.
- An unbounded scalar Python reference, a bounded batched NumPy simulator,
  and a bounded Taichi/CUDA simulator.
- A vectorized NumPy island GA using either explicitly selected batch backend.
- Human-facing simulation and training/evaluation notebooks.
- Known-program examples, focused automated tests, documentation, and resumable
  generation-boundary checkpoints.
- A local Git repository on `main`, with clean ignore/line-ending conventions.

Not included in this milestone: evolving the transition table, self-modification
of the read-only program tape, GUI/web server, program miner, PCFG/motif learning,
another universality construction, Lean work, automatic GPU fallback, or GitHub
publication. These require separate requests.

## 2. Frozen physical machine and fixed interface

### 2.1 Machine

- Two theoretically bi-infinite tapes: program P and data X.
- Common physical alphabet: `B,0,1,#`, with B the blank.
- Nine active control states and one halt state H; 144 active/read-pair rows.
- Both heads start at logical coordinate zero; initial control is D.
- P is read-only because every existing rule preserves its program symbol.
- X is writable. Both heads can move left and right according to the table.
- Halting is solely entry into H, not an external check on tape appearance.

The exact existing table is authoritative. Its canonical SHA-256 is:

```text
88ac3e58f4525dccd286fd723808f051687fbccc1ac02d0fb4e41b8e5f2eba7a
```

Fingerprint serialization: iterate state in existing STATES order, p in SYMBOLS,
d in SYMBOLS; join `state,p,d,rule.p,rule.d,rule.dp,rule.dd,rule.state` with `|`
per row; join rows with LF and include the final LF; hash ASCII bytes.

### 2.2 Initialization

- Write the candidate program from coordinate zero on P.
- Write the SAME E(x) from coordinate zero on X for every candidate and task:
  input 0 becomes `1B`, input 1 becomes `11`, and append `B1`.
- All other cells on both tapes are B.
- Empty input and empty program are permitted.
- E and data placement never receive the program, its length, state count,
  search alphabet, or algorithm.
- Task targets are supervision for scoring only. They must not affect any
  transition, write, movement, halting decision, or tape initialization.

Array origins reflect the configured left budgets; this is a storage offset,
not a change of logical head coordinates or input placement.

### 2.3 Halting and readout

After a successful bounded halt, start at the final DATA-head coordinate:

- `11` decodes to output bit 1.
- `1B` decodes to output bit 0.
- Any other pair ends the output prefix.
- Advance two cells after each decoded bit.

No requirement to return either head to zero. No requirement to preserve a
separator at any coordinate, clean scratch, or compare P with its initial value.
P happens to be preserved by the machine rules, not by a final acceptance test.

Every finite halted tape has a decoded value, possibly empty. There is NO
invalid-decoding flag, empty-output penalty, malformed-instruction rejection,
missing-separator check, or marker-cleanup requirement.

An apparent output prefix before H is not a completed result. Failed bounded
executions do not receive partial-output credit.

Examples against target `1100`:

- Decoded `11` has two matches out of four positions: accuracy 0.5.
- Physical `11|11|BB` decodes to `11`.
- Physical `11|BB` decodes to `1`, giving accuracy 0.25.
- A payload B in `1B` is a real decoded zero; it is not a missing bit.

## 3. Finite experimental budgets, without changing the theoretical machine

Expose independent inclusive coordinate intervals:

```text
P: [-program_left_budget, +program_right_budget]
X: [-data_left_budget,    +data_right_budget]
```

Both tape buffers contain blank padding on both sides of their initial content.
Maximum program length is a separate GA/genotype capacity, not the data origin.
P must contain the whole maximum-width program region and at least its adjacent
blank on each side. X must contain E(x) and adjacent blank padding for every
configured input. Reject configurations that cannot initialize those tapes.

Budget behavior:

1. Execute one exact physical transition from an in-bounds configuration.
2. Count that transition and update both heads and their visited extrema.
3. Crossing ANY left/right boundary on EITHER tape fails that case.
4. Otherwise, entry into H succeeds, including on transition T_max.
5. Otherwise, reaching T_max fails with timeout.

The boundary cell itself is allowed. There is no near-edge failure margin,
clamping, wrapping, reflective wall, or write-before-boundary workaround.
If a transition both enters H and crosses a boundary, the resource failure
takes precedence for fitness. Preserve the actual control/head/T in diagnostics;
`state == H` remains a physical fact even if that run violated its budget.

Store a single primary failure reason using deterministic precedence:
P-left, P-right, X-left, X-right, then timeout. One input contributes at most
one failure penalty, regardless of simultaneous conditions.

For this read-only table, sufficient program padding normally makes P-boundary
failure unreachable from a standard initial configuration. It is nevertheless
checked, rather than silently exempting the program head. In particular, the
ordinary rewind visit to P coordinate -1 is allowed by proper padding.

Decoder lookahead is NOT movement of a TM head. Following a successful bounded
run, cells outside X's buffer are known untouched blanks. Readout must preserve
the unbounded decoder in this case: a leader 1 in the last allocated cell can
pair with the exterior B to yield zero, then terminate on exterior blanks.
Do not truncate the output or falsely penalize this as a head crossing.

Use uint32 for each computation's transition count; enforce
`1 <= T_max <= 4,294,967,295`. Head coordinates/buffer sizes use checked int32
ranges, with room for the one-step crossing test. No integer counter may wrap.
The scalar reference uses Python integers and a self-extending dictionary tape.

## 4. Program representation and mutation

### 4.1 Genotypes

- Dense uint8 population array with logical shape
  `(islands, population_per_island, max_program_length)`.
- Explicit int32 length per individual, including when the search alphabet
  contains B.
- Active program is a LEFT-ALIGNED PREFIX at coordinate zero. Unused capacity
  to its right is B; padding is not an active gene.
- No old left-padded suffix representation or inferred start position.
- Search alphabet is configurable as `01#` or `B01#`; default `01#`.
  Both modes run the identical four-symbol physical machine and interface.
- In four-symbol mode, internal and trailing active B genes remain legitimate
  genotype positions. Do not trim them or infer length from nonblank support.
  Different genotypes can intentionally describe the same physical tape.

Random initialization and immigrants sample lengths uniformly from
`0..max_program_length`, inclusive, and active symbols uniformly from the chosen
search alphabet. No automatic compiled-program seeding or syntax enforcement.

### 4.2 Mutation

For each child, draw one exclusive structural action:

- append a uniformly sampled allowed symbol, with probability p_insert;
- delete the final active symbol, with probability p_delete;
- otherwise retain length.

An unavailable insertion at maximum length or deletion at zero is a no-op,
not a resampled or redirected action. Require p_insert + p_delete <= 1.

Afterwards, independently mutate each active locus with probability
`p_symbol_mutation`. Replace it uniformly with a DIFFERENT allowed symbol.
This includes mutations to/from '#' and, in four-symbol mode, B. Padding does
not mutate. Newly inserted active loci participate in this second stage.

Carry the previous numeric settings initially: p_insert=0.04, p_delete=0.06,
point-mutation probability=0.02. The name changes from bit flip to symbol
mutation because the alphabet is no longer binary. No explicit length penalty.

## 5. Frozen fitness

Keep the old length-weighted normalized positional accuracy, not edit distance
and not the unweighted mean of per-input accuracies. Missing AND extra output
positions count as mismatches; blanks are not implicit zero-fill.

For each successful case j, with decoded y_j and binary target t_j:

```text
denominator_j = max(1, len(y_j), len(t_j))
matches_j = matching aligned bits over their overlap
if both strings are empty: matches_j = 1
```

For a failed case, output is undefined rather than a provisional prediction:

```text
denominator_j = max(1, len(t_j))
matches_j = 0
```

Then, over the training set:

```text
accuracy = sum(matches_j) / sum(denominator_j)
failure_fraction = failed_cases / number_of_cases
fitness = accuracy - k_penalty * failure_fraction
```

Default k_penalty=0.5, configurable and nonnegative. This is a SOFT penalty
on the same scale as accuracy: failed programs remain eligible for selection,
elitism and migration according to their resulting fitness. No hard filtering,
negative-infinity assignment, or extra time/space/length penalty is introduced.

The empty/empty matching unit is the explicit extension needed because the
old tasks rejected empty targets. All nonempty-target behavior is retained.

Exactness is computed directly as successful halt plus string equality on
EVERY training case, not inferred by approximate comparison of fitness to 1.
Default `stop_first_exact=True`; an exact generation-zero population can stop
immediately. Perfect training fit is not a claim of arbitrary-length correctness.

Inputs/targets are two equally sized binary-string lists, with a nonempty
training set; empty individual strings are allowed. Preserve order and supplied
multiplicity without silently deduplicating or reweighting cases. Optional
held-out inputs/targets use the same validation and execution protocol.

## 6. Frozen island GA

### 6.1 Local parent selection

Two independent local tournaments select parent 1 and parent 2. Each tournament
samples two contenders WITH replacement from the same island. Parent 2 is not
the loser of parent 1's tournament. Contenders or parents may coincide.

Let delta be the absolute FITNESS difference between contenders. It is not
the child's improvement over a parent, and there is no Metropolis acceptance
or neutral-hill-climber step in this GA.

| Computed fitness gap | Probability of selecting better contender |
|---|---:|
| exactly zero | 0.50 |
| 0 < delta <= 0.05 | 0.53 |
| 0.05 < delta <= 0.20 | 0.58 |
| delta > 0.20 | 0.80 |

Thresholds, interval probabilities and base probability are configurable.
Preserve inclusive upper thresholds and exact computed-fitness ties as in the
latest implementation. Use float64 NumPy fitness derived from integer counts;
do not silently introduce epsilon tie bands or continuous-temperature schedules.

### 6.2 Segment crossover

- Base probability per child: initially 0.50.
- Start with a copy of parent 1; child length always equals parent 1's length
  BEFORE subsequent mutation.
- Choose a configured number of segments, initially 3.
- Segment-length limits initially 1..20, uniformly sampled over the feasible
  inclusive range for each segment.
- Use the common coordinate-aligned PREFIX of length min(len(parent1),len(parent2)),
  adapting the old shared suffix to the new coordinate-zero program convention.
- As in the previous algorithm, clip the length limits to that overlap for
  short parents; zero overlap means no copied segment.
- Sample starts uniformly among positions where the selected length fits.
- Copy parent-2 symbols at the SAME coordinates. Overlap among segments is fine.
- Do not copy padding into active genes or extend/shrink the child via crossover.

### 6.3 Neutral elitism

Within each evaluated island, rank by decreasing fitness; on exact ties prefer
new children over carried individuals. Preserve the previous stable index tie
break after that. No Hamming diversity test, genotype deduplication, or novelty
bonus is added.

Flags mean the same as before: offspring are children; carried elites, migrants
and fresh immigrants are not. Reset flags each generation. This permits neutral
turnover without forcing replacement by a worse-fitness child.

Maintain a separate immutable-copy best-ever archive, updated on strict fitness
improvement. It is reporting/checkpoint state, not a mechanism that continually
reinserts an old genotype and prevents neutral population drift.

### 6.4 Migration and immigrants

- Population size per island remains constant.
- Elite count is max(1, floor(elite_fraction * island_population)), as before.
- Migrant and immigrant counts are floor(fraction * island_population).
- Remaining slots contain children.
- At scheduled new-generation indices divisible by the migration interval,
  sample source islands uniformly excluding the destination, then source elites
  uniformly. Sampling with replacement is allowed.
- Other generations have no migration. Random immigrants use fresh random
  genotypes, not saved programs or the best-ever archive.
- Validate slot totals before training. One-island runs require zero migration.

Reuse the user's threshold/crossover/mutation/penalty settings rather than the
older module defaults. Notebook population size, budgets and generation count
will be deliberately modest initially, not automatically copied from the old
160,000-individual, 16,000-generation experiment.

Baseline evolutionary defaults carried from the edited research notebook:

| Setting | Value |
|---|---:|
| elite_fraction | 0.03 |
| cross_island_fraction | 0.02 |
| cross_island_interval | 1,200 generations |
| random_immigrant_fraction | 0.15 |
| p_crossover | 0.50 |
| crossover_segments | 3 |
| crossover segment length | 1..20 |
| p_insert / p_delete | 0.04 / 0.06 |
| p_symbol_mutation | 0.02 |
| k_penalty | 0.50 |
| RNG seed | 4, configurable including None |
| stop_first_exact | True |

The default notebook is a small demonstration, not the user's previous large
experiment: propose 4 islands x 256 programs, 300 generations, maximum program
length 128, P budgets 8/136, X budgets 32/255, T_max=20,000, and a small NOT
training task with separate held-out strings. Show a plainly labeled migration
interval override of 30 for this short demonstration; the inherited research
default remains 1,200. Print/diagnostic cadence is 10 and checkpoint cadence 100.
All these experiment values are editable; none changes the machine/interface.
Use a separate explicit configuration for longer known example programs.

## 7. Batch simulation and performance design

- Scalar reference: exact self-extending tape and explicit transition stepping.
- NumPy backend: active masks and batched table/tape operations; not a Python
  loop invoking the scalar simulator once for every population member.
- CUDA backend: Taichi, explicit CUDA selection, no automatic CPU fallback.
- Programs are parallel; input cases are evaluated serially using the same
  loaded program batch. Each case starts fresh heads/control and E(x).
- Keep GPU arrays allocated and reusable. Upload each population microbatch
  once per generation, not once per input. Upload fixed datasets once.
- Training decoding/scoring can run on the GPU, but targets affect scoring
  only. Accumulate integer match/denominator/failure/exact counts per program;
  convert to float64 fitness in NumPy so both backends share identical arithmetic.
- Use sufficiently wide aggregate counters; individual transition counts remain
  uint32. Avoid overflow when summing many cases.
- Transfer compact per-program statistics during training, not whole tape/output
  arrays for every input. Detailed evaluation explicitly requests outputs/tapes.
- Population size and GPU microbatch capacity are separate. Serial microbatches
  must give the same results regardless of partitioning or final partial batch.
- No bit packing, semantic macrostep shortcut, speculative kernel variants,
  or fitness caching in the first implementation.
- Use the working trainer's bounded per-input kernel scheduling initially;
  do NOT introduce TM-step chunking in this milestone. Measure kernel duration
  under modest limits before large runs. No Windows watchdog/driver setting changes.

Full decoded length is needed even when it exceeds target length. Never score
only a target-sized prefix. The bounded data tape gives a safe full-output
capacity of ceil(data_tape_size/2), including an implicit exterior B payload.
Output arrays contain binary 0/1 values and an unrelated padding sentinel,
such as 2; this sentinel is not a physical tape symbol or a prediction bit.

## 8. Results, diagnostics and state consistency

Detailed simulation results provide output or None, physical halted/control
state, failure reason, actual T, both head coordinates, and space metrics.
An optional detailed request can return full final tapes for inspection/tests.

For EACH tape separately:

```text
left_used  = maximum of 0, initial nonblank left extent, all visited -head positions
right_used = maximum of 0, initial nonblank right extent, all visited head positions
span_used  = left_used + right_used + 1
```

All written cells have been visited, so initial support plus head extrema
captures historical used space without repeated full scans. Visited blanks
count. Unvisited padding does not. In four-symbol mode, initial P support is
the actual nonblank support, NOT the explicit genotype length.

Progress printed DURING training includes generation, current/global best
fitness, accuracy, failed-case fraction, exact-program count, program length,
best program and elapsed time. Also expose per-island champions and scalar history.

Charts include current/global best fitness and island-champion mean with q20/q80
bands; accuracy and failure rates; resource summaries for island champions.
Resource summaries use successful halted cases only. If none halted, report
undefined/NaN rather than a misleading zero. Keep P and X space distinguishable.
Failure rates cover both champions and the full population. Rich diagnostics
default to every 10 generations; ordinary training progress has its own cadence.

Held-out evaluation is optional and never feeds selection. Diagnostic choices,
plotting, checkpoint cadence, or test datasets must not consume the training RNG
or change the evolving population. Do not persist TM work tapes across inputs.

## 9. Training API and checkpoints

Expose a straightforward trainer with one-generation stepping and a run loop.
Generation zero is an evaluated random population. A generation commits only
after its full new population has been evaluated successfully.

Checkpoints are atomic overwrite within a distinct `runs/<run_id>/` folder,
at configurable completed-generation intervals, initially every 100 generations,
and at normal termination. A clean Ctrl+C saves the last fully committed state,
not a half-evaluated population. A running CUDA kernel may finish before Python
can handle the interrupt; do not promise instantaneous GPU cancellation.

Use arrays plus JSON metadata, without pickle-dependent loading. Save:

- Full genotypes, explicit lengths and child flags.
- Completed generation, current fitness/exact counts, and history.
- Best-ever program and its metrics.
- Training and held-out input/target lists, including order/multiplicity.
- Complete simulator/GA configuration, alphabet and representation versions.
- NumPy RNG kind and exact state, not just the original seed.
- Machine-table fingerprint, fixed-interface version and checkpoint schema version.

Retain the committed RNG state as well as committed population state so Ctrl+C
during reproduction cannot advance the RNG of the saved generation. Temporary
checkpoint files are written in the destination directory, closed, and atomically
replaced in a Windows-safe way. Errors are raised; do not silently skip saving
or substitute a partial checkpoint.

Resume validates metadata/shapes/signatures and restores all selection-relevant
state. Structural configuration and datasets remain locked for that resumed run.
Continuing the same run must match uninterrupted evolution, excluding wall-clock
timings. Different experiment settings require a new run, not silent mutation of
the checkpoint. No live server or frontend hyperparameter editor is included.

## 10. Proposed standalone layout

```text
LUTM-6-evolution/
  README.md
  CONSTRUCTION.md
  IMPLEMENTATION_PLAN.md
  LICENSE
  requirements.txt
  .gitignore
  .gitattributes
  machine.py
  numpy_backend.py
  taichi_backend.py
  evolution.py
  utils.py
  programs.py
  known_programs/               # literal .txt programs and manifest.csv
  01_simulate.ipynb
  02_train_eval.ipynb
  tests/
  runs/                       # generated, ignored by Git
```

Keep core rules/reference in machine.py, batching in the two backend modules,
GA rules in evolution.py, shared data/config/codec/checkpoint/plot helpers in
utils.py, and a compact register of verified native examples in programs.py.
Avoid coupling mutation/selection operators to a concrete Taichi simulator.
Do not add a generic plugin framework, compatibility layer, or web stack.

CONSTRUCTION.md will be self-contained and accurately state proof status.
Do not copy old experiment logs or leave dangling references to old modules.
The current machine is supported by an explicit constructive argument, not
independent mathematical certification or a proof-assistant theorem.

If the source-table serializer is included, do not carry over the known
Boolean-bit stringification bug: validated bit values must serialize as 0/1.
Test this separately; do not alter physical transition rules to accommodate it.

## 11. Human-facing notebooks

### 01_simulate.ipynb

Short ordered cells for imports/backend selection; program and input; bounds;
execution; readable output/status/T/head/space; optional tape window. Provide
verified native examples for identity, NOT, parity, append and reverse. Support
batch execution without a fixed 62-bit program-width restriction.

### 02_train_eval.ipynb

Short ordered cells for train/test lists; simulation and GA parameters; explicit
backend initialization; training/resume; charts; champion inspection and held-out
evaluation; checkpoint path/program export. Printing must occur while training,
not only after completion. Put parameters together in editable cells.

These are research notebooks, not automated test reports. Do not litter their
main workflow with asserts, backend-comparison loops or repeated validation cells.
Necessary invalid-configuration errors belong in the Python API. Start committed
notebooks without large execution outputs or private run artifacts.

## 12. Ordered implementation and verification gates

### Phase A: project skeleton and canonical reference

Create the fresh folder only after implementation approval. Initialize local
Git/main. Add Windows/slackenv instructions, MIT licensing consistent with the
existing project, LF-normalized source/notebooks, and ignored run/cache artifacts.
Implement/copy the exact scalar machine and public interface, with the same table
fingerprint. Place this plan inside the new project as IMPLEMENTATION_PLAN.md.

Gate: table fingerprint; golden codec examples; empty/leading-zero cases;
known programs; no old-project runtime imports. Original folders stay unchanged.

### Phase B: bounded NumPy batch backend

Implement both-tape buffers/budgets, exact per-transition counts, decoder,
failure reasons, detailed results and compact scoring. Cross-check development
results against the old unbounded scalar evaluator as well as the new reference.
The final standalone tests must not need the old evaluator's path.

Gate: full state/head/tape correspondence for random raw patterns and known
computations; exact-budget halts; X-boundary crossings; simultaneous halt/crossing
precedence; P-budget predicates and padding; successful boundary lookahead;
empty outputs; missing/extra bits; target-independent execution; input reset;
batch permutation/splitting invariance. Do not invent impossible ordinary P
crossings if sufficient read-only program padding prevents them.

### Phase C: Taichi/CUDA backend

Initialize CUDA explicitly and build a small dense LUT from the canonical table.
Keep buffers resident and implement the same bounded semantics and result contract.
First validate small cases before any large population run.

Gate: match scalar/NumPy on every meaningful returned field, including physical
T, heads, failure precedence, empty outputs and whole final tapes in diagnostic
mode. Test widths beyond 62, mixed program lengths, internal B genes and partial
microbatches. Matching wrong outputs on two backends is not sufficient: also
check verified task results and golden interface examples.

### Phase D: vectorized island evolution

Implement backend-independent NumPy operators and trainer state/checkpoints.
Reuse tested evolutionary logic, adapted explicitly to prefix alignment and
categorical mutation, without importing old simulator assumptions.

Gate: threshold endpoints/adjacent floats; ties; independent parents; neutral
elite priority; fixed child length before mutation; segment mask/bounds; structural
mutation endpoints; inactive padding; migration excludes destination; immigrant
randomness; exact-stop at generation zero and later; no hard exclusion of failed
programs; continuous-versus-resumed population/RNG/history equivalence.

### Phase E: usable notebooks and proportionate end-to-end checks

Exercise a small and a medium island run, actual progress printing, evaluation,
save/resume and clean interruption. Run one controlled larger stress population
around 40,000 programs, a roughly 100-cell data arena and T_max around 8,000,
with compatible independent program padding and a small task set. GPU microbatch
capacity may be smaller than the total population. Check memory/kernel durations
first; do not repeatedly launch large stress runs or modify watchdog settings.

Use modest reference samples for expensive cross-checks. Do not repeat the old
billion-transition square list: it is unrelated to validating the new trainer's
core batching/GA behavior. A finite learned fit is never reported as an
arbitrary-length theorem.

Gate: all standalone tests pass; notebooks execute in order; runtime is measured
separately from first JIT compilation; all errors and semantic changes (if any)
are reported. No unexpected changes in original folders, dependencies, drivers,
global Git settings, or saved user runs.

## 13. Repository hygiene and handoff

- Run inside conda `slackenv`; do not install/upgrade packages without need.
- Use existing NumPy/Taichi versions initially; record tested versions. Add
  plotting/notebook dependencies only where used.
- Keep tests in tests/ and offer one documented command, preferably
  `python -m unittest discover -s tests -v`.
- Ignore __pycache__, .ipynb_checkpoints, generated checkpoints/runs and local
  caches. Do not ignore curated known-program sources or useful documentation.
- Set repository-local line-ending rules; do not modify global Git configuration.
- Include a short AI-assistance/platform note at the END of the README, not an
  alarming introductory warning. State Windows testing and actual proof limits.
- Inspect git status/diff and whitespace before handoff. No automatic commit,
  remote creation, push, or publication without the requested approval.
- Report every encountered error, especially any semantic mismatch. If a mismatch
  occurs, stop the affected backend/phase, isolate the cause against the scalar
  contract, and make one clear correction rather than adding alternate semantics
  or output repairs to force agreement.

## 14. Completion criteria

The milestone is complete when a user can open the two notebooks in slackenv,
simulate arbitrary raw candidates, train the specified island GA using NumPy or
CUDA batching, inspect meaningful progress and outputs, evaluate unseen pairs,
and resume a saved run, all with the fixed audited LUTM-6 interface and no old
folder dependencies. Correctness tests and one controlled larger stress run
must pass, with measured scope and any remaining limitations clearly documented.

The later known-program addition includes literal square/reverse/arithmetic
witnesses and CPU validation scripts. Universal plasticity remains outside this
repository. See README.md for usage and VALIDATION.md for actual test scope.
