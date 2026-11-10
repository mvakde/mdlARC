# Write a useful variation review

You are reviewing a synthetic ARC puzzle generator so that a subsequent coding agent
can improve its correctness and diversity. Write the kind of concrete, perceptive
review the human wrote in the six examples. The deliverable is a review, not a new
production generator.

## Explore and learn

Start by reading `examples/README.md` and all six examples: their original solved
puzzles in `task.txt`, summaries, original generators, human reviews, and final
generators. Compare the code before and after each review and representative original
and synthetic grids as compact digit rows. Learn the human's priorities and level of
specificity. Do not read historical implementation/coverage reports for these examples
or write a recap of the demonstrations. The final generators
show how reviews were acted on; their passing tests do not make every implementation
choice or generalization ground truth. Example-specific numbers (such as 1/14 or 1/7)
are illustrations, not mandatory quotas for every family.

The variations in these six examples teach what variation means; they are not an
exhaustive set of variation axes. Use any relevant common variations they demonstrate,
but you are expected and strongly encouraged to find additional variations specific
to the target task, including ones absent from all six examples.

Explore `target/` and `target/provenance.json`. All saved seeds are in `target/samples/`.
Read `target/task.txt` and the summary, inspect the generator, and compare original
and generated puzzles as digits. Sample `.txt` files sit beside their `.json` files.
Each grid has one digit per cell and one row per line, with train/test, pair, input,
and output labels. Read across multiple seeds and across the pairs within each seed.
Look at distributions over the full saved collection as useful, then inspect
representative and unusual cases in this compact text format. Generate additional seeds
or boundary cases wherever that helps resolve an uncertainty. There is no prescribed
number or sequence of tool calls. Spend effort where the puzzle warrants it.

Use ordinary shell/Python tools, search source code, create diagnostic scripts, and
test hypotheses. Do not render or view images. Keep JSON for Python analysis; when
showing grids to the model, use compact digit rows rather than JSON arrays.
`tools/inspect_puzzles.py show <file.json>` prints labeled digit grids, including for
new diagnostic samples. Add `--split train --pair 1` to inspect one pair, or pass
several selected files together. The helper also provides sampling and descriptive
statistics, and you may write task-specific tools. `environment.json` identifies a Python interpreter
with NumPy. `scratch/` is provided for experiments; organize other working files as
useful. The original repository path is available in `environment.json` if further
context is needed. Preserve the staged source evidence and examples; experiment on
copies when edits help investigate a hypothesis.

## What to look for

The original solved task is primary evidence of the invariant transformation. The
summary is an interpretation and may be wrong or overly narrow; the generator is an
implementation attempt. Diagnose mistakes in either, explaining the source evidence.

For shape-to-color or pattern-to-label tasks, preserve the identity of the patterns
being classified. A statistic that separates the observed shapes, such as component
count or area, does not establish equivalence for unseen shapes. Do not recommend
an additional topology/counting regime simply because synthetic examples can teach
it. Repair this kind of drift before requesting more variety. A finite shape
vocabulary and repeated layouts can be essential to a lookup task.

Find missing variation that preserves the puzzle's rule and makes it unambiguously
solvable from its training demonstrations and test input. Look for unnecessary fixed
colors, dimensions, object vocabularies, positions, shapes, counts, orientations,
sampling ranges, layout templates, and correlations. Look for wrong outputs,
degenerate shapes, impossible cases, duplicated examples, hidden conventions, and
shortcuts that replace the intended reasoning. Name the task-specific objects and
mechanisms instead of pasting a generic checklist.

Separate properties that can vary between pairs in one puzzle, properties learned
once per puzzle that can vary between seeds, and meaningful train/test regimes.
Identify which conventions must remain consistent and how training reveals them.
Allow compatible variation axes to combine and include a useful spectrum of simple,
moderately varied, and strongly varied cases. Maximum variation in every example is
not the objective. Do not force every category to contain a complaint when evidence
does not support one, or turn incidental source details into mandatory invariants.

For a demonstrated defect, reproduce it when practical and identify the relevant
seed, train/test pair, source comparison, or code location. Distinguish a measured
sampling gap from a proven impossibility in the generator. Finite samples cannot
prove a feature is absent from all seeds. For proposed extensions, explain why they
preserve the rule and which ambiguity constraints they need. Label unresolved
interpretations honestly. A passing self-test or distinct grid hashes alone do not
establish rule fidelity or meaningful structural diversity.

If sampling failed, read the recorded failure and investigate the code. That failure
is evidence to include, not a reason to invent sample observations or skip the task.

## Deliverables

Write only `review.md` as the deliverable, in natural prose/bullets in the spirit of
the human notes. Give the next coding agent clear corrections and concrete variation requests with their
scope and constraints. Keep it focused on this family. There is no required heading
template, length, quota of findings, JSON response, or code-fence format. It is valid
to explain that a suspected issue was not supported.

Put brief seed/pair or code references and any material uncertainty beside the
findings they support. Keep useful diagnostic scripts/results in the workspace,
but do not write `evidence.md`, a demonstration recap, or a separate audit narrative.
The review must stand alone for regeneration. Before finishing, check that its claims
match the evidence and its requests preserve learnability. Summarize completion
briefly in your final message. The runner only checks file integrity and completeness,
not semantics.
