# LUTM-6: fixed-interface universality construction and proof audit

This document concerns the EXACT `machine.py` table and `programs.py` compiler
in this folder: two bi-infinite tapes, alphabet `B,0,1,#`, nine active states,
one halt state, and 144 active/read-pair rows. The argument below is a
constructive mathematical proof at the level of finite-control procedures and
simulation invariants. It is not a Lean proof or an independent certification.
Finite tests support implementation correspondence; they are not its logical basis.

The statement is about arbitrary finite binary strings, including empty strings
and leading zeros, and equality of PARTIAL functions, not only successful runs.
No finite time, space, integer-width, or input-length bound is part of the
mathematical machine.

## 1. Fixed interface and data alphabet

Physical symbols are `B,0,1,#`. At simulated binary-TM boundaries the data tape
contains ONLY B and 1. Logical zero is physical B. The two spare symbols 0 and
# are temporary markers, preserving an overwritten B or 1 respectively.

Public input is always the fixed pair code `0 -> 1B`, `1 -> 11`, followed by
`B1`, from coordinate zero. Both heads start at zero. Program information is
never consulted during input preparation. All other cells are blank.

At physical halt, read consecutive pairs starting at the data head: `1B` gives
0, `11` gives 1, and anything else ends the prefix. This total readout does
not classify halted patterns as invalid. Raw patterns can halt in any position;
the witnesses below guarantee the intended position and representation.

Formally, a configuration contains control, both head coordinates, and both
tapes. Each tape is a map from the integers to the alphabet, with finite
nonblank support. Initialization writes p on tape P and E(x) on tape X, both
from zero, with state D and both heads zero. The two writes act on DIFFERENT
tapes: combining them does not shift, inspect, or preprocess either argument.

The halting predicate is ONLY `state == H`, independent of either head or tape.
The decoder is applied ONLY after halting. A well-looking output prefix during
an infinite run does not constitute a result. Each physical transition writes
at most one data cell and moves heads by at most one; therefore finite support
is preserved at every finite time. The decoder terminates on every reachable
halted configuration: infinitely many successive leading-1 pairs would require
infinite support. Thus D is total on the relevant configuration space.

Write Phi_p(x) for this decoded value if H is reached, and leave Phi_p(x)
undefined otherwise. Timeouts in Python experiments are not part of Phi.

The read-only P invariant follows by checking that every row writes its read
program symbol unchanged. For an entirely total transition function, add
absorbing H self-loops; execution stops at first entry to H and never uses them.

## 2. Binary source TM records

Each binary source state q has two records, logical zero first, then one:

```text
step: write_bit direction_bit 1^(2*target_state) #
halt: #
direction 0 = left, 1 = right
```

Every record has exactly one delimiter; there is no parsing requirement at
runtime. This format constructs a useful witness subset of raw patterns.
An arbitrary pattern can take other paths through the same complete table.

For precision, a binary source has N >= 1 states numbered 0,...,N-1 and
two actions per state. An action is either halt, or (w,d,t) with w in {0,1},
d in {-1,+1}, and 0 <= t < N. Binary zero is physical B, NOT physical 0.
Let a_j be the start of record j, j=0,...,2N-1, and a_(2N)=P the program
length. Every record length is positive, and the serialized program has NO
internal B. Outside [0,P-1] the program tape is blank. State q starts at a_(2q).

This restriction is an assumption on compiled witnesses only, not on the
machine's accepted candidate patterns. In particular, the serializer's Python
checks are construction-time checks, not a runtime program parser.

## 3. One source step: exact physical correspondence

Boundary invariant: state D, program head at the first record of source state
q, data head at its simulated position h, data symbols only B/1, program
unchanged, and finite tape support.

For source zero, D executes the first record's write directly. For source one,
D enters S, which skips exactly one record and enters W to execute its write.
A selected # halts without changing data. Otherwise M reads the direction and
moves the data head one source cell left/right.

K marks this NEW head cell: B becomes 0, and 1 becomes #. No other source data
changes. C walks the data head right by exactly 2*target_state cells while
reading the unary target. It does not modify the crossed cells. R rewinds
the program to its left exterior blank and moves onto its first record.

J now inspects the data head. A temporary 0/# means the saved position has
been reached: restore B/1 and enter D. Otherwise move the data head left one
cell, skip one program record with U, and repeat J.

There is exactly one marker; all crossed cells contain B/1. Hence precisely
2*target_state records are skipped, and the marker is reached after finitely
many repetitions. It restores the new source-head cell EXACTLY, including
its former bit. The full source tape, head, and target control state now agree.

This also handles target zero without special preparation. Every physical
phase is finite: target length, program length and each record length are
finite, and all witness delimiters exist. Thus there is no infinite physical
stuttering inside a source step. Induction proves exact correspondence for
arbitrary finite source runs and preservation of source divergence.

The count walk can cross input or output cells: it READS them, does not erase
or rewrite them. There is no preallocated scratch or program-length padding.

### 3.1 Exact return-loop invariant

Let b be the read bit, let (w,d,t) be the selected nonhalting action, put
r=2t, and let h'=h+d. Immediately after K, the data tape equals the source
successor tape except at h', which contains the unique temporary marker.
C makes exactly r right moves, stopping at the record's delimiter. R makes
no data move and returns the program head to a_0. At the j-th J inspection,
for 0 <= j <= r, the following assertions hold:

- the data head is at h'+r-j;
- the program head is at a_j;
- only h' is marked; all other data cells have their exact successor B/1 value.

For j<r the inspected cell is unmarked. J moves left once, and U skips exactly
record j, including its terminating delimiter. This establishes the next
inspection. For j=r the marker is found, its underlying B/1 is restored, and
the machine enters D with program head a_r=a_(2t) and data head h'.

The skip is always in range: only records 0,...,r-1 are skipped, and
r <= 2N-2. U may skip halt records too; '#' is a halt instruction only when
SELECTED by D/W, not when passed by U. No intermediate phase can reach H on
these well-formed witnesses. In particular, S finds the first record's '#',
M finds a direction bit, and U finds each required delimiter before an exterior B.

### 3.2 Finite completion and halting equivalence

The phases terminate for explicit finite reasons: S/U scan finite records;
C consumes r unary symbols; R moves left through a finite B-free program
prefix; J/U repeat exactly r times. A convenient conservative bound is

```text
one nonhalting binary step: T <= 3*P + 4*N + 8
```

For example, selection costs at most P+2; M/K cost 2; C at most r+1;
R at most P+1; all J/U work at most r+P+1. Their sum is at most
3P+2r+7, below the displayed bound. No tape-extent assumption is involved.

More strongly, the exact cost for a nonhalting action is

```text
ell = a_(2q+1) - a_(2q)
T = a_(2q+b) + a_r + 3*r + 8 + b*(ell+1),   r=2t.
```

For b=0 a selected halt record takes one physical transition. For b=1 it
takes ell+2: enter S, skip the first record, then halt in W. Neither case
changes the data tape or data head. These counts include the transition into H.

Induct on completed binary steps. The initial boundary invariant holds. The
step lemma preserves it, and a selected source halt gives a finite physical
halt with identical tape/head. Conversely, the only possible physical halt
on such an execution is a selected source halt: the nonhalting-step argument
excludes all other H entries. An infinite source execution therefore induces
an infinite physical execution, with infinitely many positive-length macrosteps
and no possibility of infinite internal stuttering or accidental fall-off halting.

## 4. Universal fixed interface by a local block compiler

Take an ordinary four-symbol TM over virtual `B,0,1,#`, starting on literal
`x#`, whose halt head points to its output's first bit. Its output is the
consecutive virtual 0/1 block, stopping at B or #. Section 9 gives a constructive
normalization argument that this family realizes all partial computable
binary-string functions, instead of leaving that property as an unexplained
premise. It is not necessary for every raw candidate to be in this normal form.

Use the SAME block code for every source machine:

```text
virtual B -> 00       physical BB
virtual 0 -> 10       physical 1B
virtual 1 -> 11       physical 11
virtual # -> 01       physical B1
```

Here binary zero is physical B. All four two-bit words are accounted for,
and the virtual quiescent background maps to the physical blank background.
The public E(x) is EXACTLY this fixed encoding of x#. No program description
or state is encoded into the input region, and no bootstrap is missing.

The binary controller stores the virtual source state in its finite control
and operates on pairs. At a virtual boundary the head is at its block's left
bit, at physical position 2h:

1. read the first bit, preserve it, move right, remember it in control;
2. read the second bit; select the virtual transition; write the replacement
   block's right bit and move left;
3. write the replacement block's left bit; move left if the virtual move is
   left, otherwise right;
4. preserve the scanned bit; move left for virtual left/stay, right for
   virtual right; enter the new virtual source state.

The resulting head is at 2h-2, 2h, or 2h+2 respectively. Other virtual blocks
are unchanged. Each virtual transition uses four binary-source transitions.
Source halt is entered only after all writes and positioning finish.

`compile_blocks` materializes this finite construction. State count and program
length can depend on the source algorithm. The pair code and input placement
do not. It is an optional witness compiler, not an inference-time service.

### 4.1 Block simulation invariant, including halt

Let beta map the ENTIRE virtual tape to two-bit blocks, with physical B
representing bit zero. At an enter(q) boundary, physical head=2h and the
binary tape is exactly beta(T), including all blanks and all garbage. Reading
two bits determines one of the four source symbols uniquely. Phases 2 and 3
write precisely its replacement block; phase 4 preserves the scanned bit.
In particular, a VIRTUAL '#' delimiter or cursor is the PHYSICAL pair B1,
not a physical '#'. A virtual literal zero is the physical pair 1B, not a
physical '0'. Thus arbitrarily many virtual markers cannot collide with the
unique physical 0/# jump marker. The two symbol layers must not be conflated.

The head trajectories, relative to 2h, are:

| virtual move | four binary moves | final offset |
|---|---|---:|
| left | right, left, left, left | -2 |
| stay | right, left, right, left | 0 |
| right | right, left, right, right | +2 |

Hence even a stay move is correctly implemented without binary stay actions.
Every nonhalting virtual transition produces exactly four nonhalting binary
actions. A transition TARGETING source halt still completes all four, including
its final write and movement. Only the subsequent enter(HALT) boundary has
two halt actions. Source initial halt is handled by that same boundary.
No intermediate phase halts. The generated controller set is finite: source
state, remembered bit, replacement bit and direction each range over finite sets.
The breadth-first numbering starts its initial controller at binary state zero;
state names or unreachable source states do not change the interface.

Induction proves full-tape/head equality at every virtual boundary and halt
if and only if source halt. A divergent virtual run has infinitely many
four-action blocks; it cannot be converted to a binary halt or internal loop.

### 4.2 Decoder compatibility, not merely an internal simulation

Suppose the virtual machine halts at h with output y: positions h,...,
h+|y|-1 are its literal bits, and the next symbol is B or '#'. On beta(T),
starting at 2h, each bit is precisely 1B or 11. The terminating source B
becomes BB, and '#' becomes B1; both have leading B and stop the decoder.
Thus D(beta(T),2h)=y. If y is empty, the FIRST pair stops the readout and
returns empty. Garbage before the head or after the terminator is irrelevant.

Alignment is maintained by the compiler, not imposed as a runtime acceptance
condition. Arbitrary raw programs may halt on odd coordinates or markers;
the same decoder still returns a finite string. No final-coordinate test,
separator-preservation test, marker-cleanup validator, or success flag is
inserted into the machine's halting predicate.

## 5. Universality composition

For any partial computable g on binary strings, choose a four-symbol source
TM computing it in output normal form. Apply the fixed block compiler and
serialize its finite binary controller. This gives one finite physical
pattern p_g. On EVERY finite x, start with the same E(x) at coordinate zero.

The binary block invariant and direct-interpreter invariant imply:

- source halting gives physical halting with the same decoded output;
- source divergence gives physical divergence, since every simulated step
  completes in finite physical time and no witness halt occurs otherwise.

Therefore the proposed construction satisfies

$$
\forall g\text{ partial computable}\;\exists p_g\text{ finite}\;
\forall x\in\{0,1\}^*: \Phi_{p_g}(x)=g(x),
$$

including equality of domains. Initial support is finite and both tapes have
ordinary blank backgrounds. One p_g works at all input lengths.

This conclusion uses section 9's source normalization, sections 3 and 4's
two simulation lemmas, and section 4.2's readout lemma. It is not the invalid
inference "a universal internal subsystem implies any chosen interface is
universal". The public input is already the correct fixed block image of the
normalized source input.

For defined g(x), all initialization, source computation, output formatting,
block steps and physical steps finish. For undefined g(x), initialization
finishes, the source computation never enters its formatting/halt phase,
and both simulators preserve that nonhalting behavior. Thus

```text
H is reached on (p_g,E(x))  iff  x is in dom(g).
```

The initial support is finite even when the compiled program is large. There
is no periodic background, infinite preformatted workspace, or input-length
cutoff. Program-dependent extra space is CREATED DURING execution. It is not
part of E, the initial data tape, or the initial head placement.

This is a constructive proof argument requiring independent audit, not a claim
of formal certification. Testing bounded examples cannot establish the
induction, normalization, or divergence claims by itself.

## 6. Efficiency and limitations

One binary-source step uses O(P+q') physical transitions, where P is the
finite program length and q' the target index. No scan of the whole input or
work tape is required. Each block-compiled virtual step uses four such steps.
For a fixed finite source algorithm with t steps, physical time is O(t), with
a program-dependent constant. Space overhead of jump counting is at most
twice the maximum target index beyond the simulated source head, not a
function of input length, and is not preinitialized.

Absolute unary targets still enlarge programs and constants. This design does
not promise efficient simulation uniformly in source-program size. Two tapes
and 144 rows are chosen to improve useful computation costs without a huge
control table. Native compact algorithms can be much cheaper than generic
block-compiled ones. Program search friendliness remains an empirical question.

## 7. Native witness invariants

NOT: scan pair leaders, flip each payload bit, stop on the end leader B, then
rewind by pairs to the blank pair left of the input. Leaders remain 1. The
halt head is restored to zero. Empty input returns empty. The binary source
takes 4*n+4 transitions plus its halt, independent of any chosen maximum n.

Parity: finite control holds the parity while scanning input pairs. At the
end marker, write a result pair 1B/11 and halt on its leader. Old input is
ignored by the fixed head-relative decoder. No length bound occurs.

Append: scan input pairs, replace the end marker by a payload pair containing
the appended bit, and rewind by pairs. The next leader is blank, so the fixed
decoder stops after the new bit. Exact preservation of an input terminator at
halt is not an interface requirement.

Reverse: an ordinary source TM puts a left marker before the input, repeatedly
removes its last remaining bit and appends that bit to output beyond the
right marker. When no input remains, it halts on the output's first symbol.
The two markers delimit an arbitrary finite input; neither depends on program
length. The fixed block compiler implements this algorithm on the same machine.

The faster native reverse implements the same idea directly on pairs. It
creates the left boundary B1 at positions -2,-1 while retaining the original
right boundary B1. Unprocessed bits have pairs 1B/11; consumed bits become BB.
It searches left from the right boundary for the last unconsumed pair, remembers
its bit in finite control, erases that pair, and appends 1B/11 beyond the right
boundary. Encountering the left B1 boundary means all input is consumed.
It then traverses the erased BB pairs to the right B1 boundary and halts on
the first output pair. Each iteration removes exactly one unprocessed bit,
so the algorithm terminates for every finite input without a built-in bound.
The output is the extracted last bits in sequence, hence the reversed input.
Both boundary markers are ordinary writable binary pairs, not immutable tape
features. Temporary physical jump markers remain a different symbol layer.

## 8. More complex compiled witnesses

The [known-program validation](VALIDATION.md) includes generic
reverse, increment with unbounded carry chains, binary
popcount, and a binary shift-and-add square. The square source maintains
multiplier, shifted operand and sum columns created internally from raw x#.
Its invariant and final formatting provide an algorithmic arbitrary-length
argument separate from the generic compiler/interpreter correspondence.

The source-dependent three-track layout is a runtime computation, not an input
encoding change. The external E(x) stays the same fixed two-bit code for every
one of these programs, including empty and leading-zero inputs. Exact full-tape
checks at binary and virtual-source boundaries supplement, but do not replace,
the mathematical simulation invariants above.

## 9. Explicit source normal form without an algorithm-aware input encoder

This section fills the previously compressed source-family lemma. It is a
finite-control construction, not an additional external preprocessing stage.

Use the ordinary one-tape definition of a partial computable binary-string
function: some finite-alphabet TM M starts on literal x with blank exterior,
and, if it halts, has its head at a literal finite binary result, followed by
a nonbinary symbol (one can use blank). The tape alphabet Gamma is finite,
contains distinct blank/0/1, and may otherwise be arbitrary. An ordinary
output cleanup routine may be included in M; no resource bound is assumed.
Missing transition entries can be replaced by explicit halt-preserving
transitions, so only a distinguished halt control is needed.

We construct a four-symbol machine N_M which instead starts on literal x#.
Its externally prepared input is IDENTICAL for every M.

### 9.1 A private, finite, dense block representation

Choose k >= 1 with 2^k >= |Gamma| and an injective code

```text
c : Gamma -> {0,1}^k,     c(blank) = 0^k.
```

These digits are literal nonblank four-symbol tape symbols 0 and 1. Outside
an allocated finite interval there is still physical B. This is NOT an infinite
background of coded blank blocks. The integer k and every word of length k
can be remembered by finite control because M is fixed.

Represent logical source cell j by the k-cell block whose left edge is

```text
b(j) = -1 - k*(j+1).
```

Logical right is physical left. A represented source head j corresponds to
the four-symbol head at b(j). All allocated blocks form one dense interval;
its exterior is B. A block represents a blank by k literal zeros, not by B.

### 9.2 Internal bootstrap from raw x#, including empty x

Starting at zero, N_M writes a temporary '#' at -1. Raw x# lies to its right;
the initially empty coded work interval grows to its left.

Repeat these tape-local operations:

1. From the boundary, scan right across consumed B cells to the next raw
   bit or the original input '#'.
2. For a bit a, remember a in finite control and erase that raw cell to B.
   Scan left to the boundary '#', then left across the dense coded interval
   to its first exterior B.
3. Write c(a) right-to-left into k new cells, extending the interval left.
   Scan right through coded bits back to the boundary, and repeat.
4. On reaching the original input '#', erase it and scan left through the
   consumed raw-input blanks to the boundary. If the coded interval is
   empty, create one c(blank) block to its left. Erase the boundary and move
   k cells left to b(0).

After i consumed input bits, their codes occupy exactly cells b(0),...,b(i-1)
as blocks; all consumed raw positions are B. The two '#' symbols cannot be
confused: the work interval contains only 0/1, and the consumed raw interval
contains only B. Each iteration consumes one previously unconsumed bit, so
there are |x| iterations, but the controller NEVER needs to know or store |x|.
Every individual sweep is finite. Empty input follows step 4 immediately.

The result is precisely M's literal input represented in finite dense blocks,
with state M's start and logical head zero. No input delimiter is passed to M.
All original raw-input cells and both temporary delimiters have been erased.

### 9.3 Simulating one arbitrary-alphabet source step

At block b(j), read k literal bits and remember the word in finite control;
return to its left edge. Look up M's transition, write the k replacement bits,
and return to the block edge. For a logical move d in {-1,0,+1}, move -k*d
physical cells. If the destination is outside the allocated interval (read B),
write a new c(blank) block there and return to its left edge. Otherwise the
existing block is used. Since moves are to adjacent blocks, density is preserved.

Reading, rewriting and moving each take a fixed finite number of local
operations depending only on k. The simulated tape/head/control agree exactly
after every step, including erasures and moves beyond either old edge.
Only finitely many new cells are allocated per step.

If M diverges, this step procedure repeats forever. There is no transition to
the final formatter unless M's halt control is reached.

### 9.4 Returning literal output and positioning the halt head

Suppose M halts with its head on the first symbol of result y. All represented
cells lie in the dense interval, and its exterior is B. There are no literal
'#' symbols in the representation: even a logical Gamma '#' is encoded in bits.

First mark the current block's first bit with '#', retaining that bit in finite
control. Scan right to the interval's first exterior B, and write '#' there as
an output boundary. Scan left to the cursor '#', restore its bit, and resume at
that block edge. The two markers are distinct, even for a rightmost-block head.

For each successive logical binary output symbol:

1. Read its code, remember its literal output bit and original first code bit,
   and mark its block edge with '#'.
2. Scan right to the output boundary, pass it, scan across the already written
   literal output to the first B, and append the remembered output bit.
3. Scan left across the literal output to the boundary; pass it and continue
   left across coded bits to the cursor '#'. Restore the saved code bit and
   move k cells LEFT to the next logical output cell.

If that cell is exterior B, or its decoded Gamma symbol is nonbinary, stop
copying. There are finitely many allocated blocks, so this phase is finite.
Finally scan right to the output boundary, move one cell right, and enter halt.
The output is literal y, followed by B; the head is at its first bit, or at B
for empty y. Its right-side region was initially blank and is disjoint from
all coded source blocks. No unread output is overwritten.

### 9.5 Why these are genuine finite-control procedures

The controller stores only M's state, a word of at most k bits, finitely many
flags, and a counter ranging up to k. Unbounded work is represented by tape
positions and scans, not by host integers or hidden registers. A scan-until-symbol
is an ordinary finite-state loop, and a k-step move is k control states. Every
procedure can therefore be expanded to a finite four-symbol TM transition table.
Unexpected-symbol branches can be filled with an ordinary nonhalting loop;
they are unreachable under the valid-input invariant. No external parser is used.
The expansion is effective from M's finite transition table: enumerate its
finite control/word/phase combinations and unroll the bounded k-step routines.
It never runs M during compilation, consults x, or decides whether M will halt.

Thus N_M is finite, accepts the SAME x# input for every M, preserves the exact
halting domain, and returns literal y at its halt head. Its internal code width
k may depend on M because the bootstrap is part of the algorithm running INSIDE
the machine. The external pair code, input placement and head initialization
never change. This distinction is essential: preparing c(x) OUTSIDE N_M would
violate our requirement, but computing it internally does not.

The normalizer is specified constructively here. A general transition-table generator for this
normalizer is NOT currently implemented. The existing `compile_blocks` generator
starts from a four-symbol source table and is implemented and tested separately.
This boundary is explicit: the normalization argument is not a physical execution
or a machine-checked proof of finite-control compilation. This repository's
executable witness compiler begins with a four-symbol source machine.

## 10. Adversarial audit: objections and scope

| Potential obstruction | Why it does not invalidate the construction |
|---|---|
| Input preparation needs program length | E receives x only; P and X are separate tapes, both starting at zero. |
| Encoded blank requires infinite nonblank background | beta(B)=BB; the section 9 dense code allocates only finitely many blocks internally. |
| Jump counting overwrites input/output | Only one marked cell is changed and restored; crossed cells are read only. |
| A jump sees a data '#' and halts | H depends on selected program records; the unique data marker is tested by J and restored. |
| A skipped halt record halts | S/U treat '#' as a delimiter; only D/W select halt records on well-formed executions. |
| Source halt occurs before its last write/move | All four binary actions finish before enter(HALT). |
| A stay move or negative position breaks alignment | The four-action head trajectories and bi-infinite coordinates cover both. |
| Empty output cannot halt or decode | The source may halt on B/#; the decoder immediately returns empty. |
| Output must preserve input or program syntax | No data preservation is required; only the fixed final readout matters. |
| An infinite source step hides a physical halt | Every macrostep has a finite bound; H is excluded between source boundaries. |
| A valid-looking prefix makes a divergent run defined | Output is read ONLY after H. |
| Compiled witness syntax rejects searched patterns | There is no physical parser/reject state; structured witnesses establish existence only. |
| Finite tests prove arbitrary lengths | They do not. Induction and finite-control algorithms establish the mathematical claims. |
| Read-only program tape prevents universality | A fixed description can control arbitrary writable work-tape computations. Self-editing is not required. |

No claim of universal behavior for EVERY raw p is made or needed. Universality
requires a suitable finite p_g for each g. Other p may halt immediately, loop,
or produce unintended values. They are still ordinary programs in the same
partial-function family. Favorable search landscapes are a separate empirical
question, not a consequence of the theorem.

## 11. Assurance status

The table-level interpreter, implemented block compiler, readout lemma, and
source-normalization construction now form an explicit end-to-end argument
with no KNOWN interface/halting obstruction. The normalizer's finite-control
expansion is described rather than materialized; the proof is not a complete
formal derivation checked by a proof assistant. Independent scrutiny may find
an error in either the argument or its implementation correspondence.

[VALIDATION.md](VALIDATION.md) records reproducible tests, exact coverage,
and issues encountered while implementing the experimental backends. Passing those tests does not
replace the universal quantifiers or establish mathematical divergence from
a timeout.
