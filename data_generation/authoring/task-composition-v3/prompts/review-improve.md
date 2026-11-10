Review and improve this new ARC composition in ONE model call, in two strictly
ordered phases. Complete phase 1 before beginning any improved generator creation.
There is no separate model call between the phases. If resuming with a completed
review-complete.json checkpoint, preserve that review and continue phase 2 directly.
Deliver review.md, generator.py,
task_spec.md and review_coverage.md.

PHASE 1 — REVIEW ONLY
Read source_spec.md, composition.md, generator_v1.py, task.txt, parents/ and the
few solved samples/ supplied here. Inspect representative and unusual pairs; do
not mechanically read 42 puzzles. You may generate targeted diagnostic cases into
scratch/ when needed. validation_v1.json records checks over 42 seeds; passing
those checks is not independent semantic proof. Use compact digit-row .txt grids
for inspection and JSON for analysis. tools/inspect_puzzles.py can show, sample and
describe puzzles. Do not render or view images. Preserve staged evidence/tools.

Study the FOUR worked human review examples listed in examples/README.md: their
originals, summaries, before/after generators, reviews and selected samples. They
are references for grounded reasoning and specificity, not a ceiling on creativity
or a checklist. Strongly seek fresh, task-specific diversity beyond every example.
Explore unusual compatible interactions between variation axes, broader object
vocabularies, geometry, roles, placement, counts, orientations and train/test
regimes. Remove unnecessary correlations and repetitive construction templates.
Keep a useful spectrum of simple through rich cases and make both parent concepts
matter to the answer. Go beyond the demonstrated diversity wherever the rule allows.

source_spec.md describes the intended invented family; code and solved labels can
contain errors. Parent puzzles establish lineage, not incidental layout constraints.
Find concrete output defects, construction failures, hidden conventions, weak
teaching examples, duplicate examples, ambiguities and shortcuts. Separate invariants,
puzzle-level parameters, pair-level properties and sampling regimes. Give each
extension's supported range, scope, constraints and compatible combinations.
Preserve the composed rule and meaningful contributions from both parents. Do not
replace it with an unrelated task. Explain any necessary clarification or correction.

Write a complete, standalone review.md with actionable corrections and ambitious
compatible diversity requests. Cite seed/pair or code references. Distinguish
reproduced defects, observed sampling gaps and unverified hypotheses. Finite samples
cannot prove a variation is impossible across all seeds. No approval/rejection
verdict, independent solver or blind solving is required.

Do NOT create generator.py, task_spec.md or review_coverage.md in phase 1 (including
candidate improved generators in scratch/). Diagnostic scripts testing the unchanged
v1 are allowed. When review.md is fully finished, run:

    python finish_review.py

using the Python executable in environment.json. This records the completed review
hash and creates the editable starting generator.py and task_spec.md. Do not edit
review.md or review-complete.json after this checkpoint. If interrupted, complete
or resume this barrier before entering phase 2. Do not skip it or write the checkpoint
by hand. A host check pins the completed review across any execution repair turns.

PHASE 2 — IMPLEMENT THE COMPLETED REVIEW
Only now read your finished review and implement every compatible correction and
variation request in generator.py. You may redesign construction while preserving
one coherent rule, both parent contributions and the author's connecting concepts.
Correct proven errors, clarify the final rule in task_spec.md, and explain material
corrections or clarifications in review_coverage.md. Incompatible requests need a
specific rule-grounded explanation. Preserve all phase-1 evidence and the review.

For every review request record its scope (invariant/puzzle-level/pair-level/regime),
implemented range and sampling mechanism, compatible combinations, measured results
or concrete seed/pair inspections, constraints and anything unverified. Measure
actual returned grids, not just sampled internal parameters. Distinct hashes and
agreement with your own solver are not independent semantic correctness evidence.

Implement using only Python's standard library and NumPy:
    generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict
    solve(input_grid: np.ndarray) -> np.ndarray
    validate_generated_puzzle(puzzle: dict) -> None
    self_test() -> None

Return exactly train/test lists of input/output pairs. Integer grids have dimensions
1..30 and colors 0..9. Honor explicit counts: 2..10 train, 1..4 test, train > test;
illegal counts raise ValueError. Use np.random.default_rng(seed), deterministic,
bounded construction and fallbacks. Each seed yields one coherent puzzle with
unique inputs and no exact parent original/selected variation pairs. Teach its
conventions and provide configurations that distinguish simpler shortcuts.

solve derives outputs from input alone, without seeds, cached answers, hidden state
or generation metadata. Preserve this contract across the expanded family. The
validator checks returned grids and meaningful family invariants. self_test tests
counts, determinism, transformations and requested structural variety over 42 seeds;
keep it bounded. Exported code must not do file/network I/O, spawn processes, use
eval/exec or depend on packages besides NumPy.

Run python verify.py with the supplied interpreter after improvement. The host
checks 42 recorded seeds, all legal explicit counts, illegal counts, determinism,
duplicate/source copies, the generator-owned solver, validator and self_test. These
are local execution checks, not independent approval. Keep all saved JSON compact,
without indentation. Final response briefly reports changes and measured/unverified
coverage; the edited files are the deliverables.
