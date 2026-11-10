Create one new ARC task family using concepts from both parents A and B.

Read parents/A/ and parents/B/: original puzzles (task.txt), selected variation puzzles (variation.txt), summaries, both generators,
the review, final-generator coverage, and provenance. Reviews describe requested
changes to v1; coverage is the v2 author's account, not independent approval.
Use the final generators as references. You may execute them to understand their
families, but invent a new rule rather than chaining whole-puzzle generators.

Study both the original and the selected generated variation of each parent.
variation_selection.json records the seed, mode and selection rationale. Read grids
as compact digit rows in task.txt and variation.txt; use compact JSON for analysis.
The selected variation is evidence of the generator's broader family, not a new
constraint on what you may compose.

Think about the concepts in both parents, and actively think beyond them. You are
encouraged to introduce independent ARC concepts, especially concepts that connect
the two inherited mechanisms into a coherent new transformation. Explain the added
concepts and why they help. Both parents must still contribute meaningfully.

Develop ONE composition idea and record it in composition.md. Explain each parent's
contribution, their interaction and any new connecting concepts. Do not produce a
menu of three proposals. Be bold and creative: discover family-specific interactions
beyond obvious sequential chaining, familiar layout templates and cosmetic changes.
Aim for a diverse new rule with broad compatible structural variation. Creativity
must preserve coherent, inferable behavior and a meaningful role for both parents.
Avoid independent side-by-side puzzles. You may change source layouts and conventions.

Write task_spec.md for this idea. Include:
- rules_summary, input_generation, variation_guidance, train_test_shift,
  solution_steps, key_insight, and puzzle_concepts;
- exact contributions of A/B and their interaction;
- input constraints, color/marker conventions, ties and boundary behavior;
- what varies between puzzles versus between examples in a single puzzle;
- plausible shortcuts/alternative rules and examples that distinguish them;
- how to check that omitting either parent concept changes the answer.

Implement generator.py using Python's standard library and NumPy only:

    generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict
    solve(input_grid: np.ndarray) -> np.ndarray

generate_puzzle returns exactly {"train": [...], "test": [...]}, where every pair
has exactly "input" and "output", each an integer 2D NumPy array or nested list.
All grids have dimensions 1..30 and colors 0..9. Support 2..10 training examples
and 1..4 test examples with num_train > num_test; raise ValueError for illegal
counts. Use np.random.default_rng(seed). A seed must reproduce the whole puzzle.
Every pair shares one coherent rule and conventions, and inputs are distinct
within the puzzle. Do not copy exact parent pairs. Vary structures, geometry,
palette roles and counts where the new rule allows it, rather than just colors.

solve derives the output entirely from its input, with no seed, latent state,
stored examples, or dependence on generation. Any convention required for solving
must be fixed by the family or visibly encoded in the input. Construct inputs
with a meaningful train/test shift while keeping one rule. The demonstrations
must expose both concepts and distinguish simpler alternatives; include cases
where a shortcut that ignores either contribution gives a different answer.
Document why the demonstrations suffice; do not add difficulty without evidence.

Use scratch/ freely for experiments. Run `python verify.py` with the Python
executable given below. That verifier checks structure and your solve function;
later a fresh variation/error review will inspect your full code and solved samples,
then that same reviewer session will expand and correct the generator after completing its review.

Deliver composition.md, task_spec.md and generator.py. Do not edit parents/, this
instruction file, AGENTS.md, or verify.py. If the pair is fundamentally unsuitable,
explain why rather than manufacturing arbitrary exceptions.

Also implement validate_generated_puzzle(puzzle) and self_test(). Check meaningful
family-specific invariants and demonstrate variation across 42 seeds. Keep self_test
bounded. Your input-only solve function must continue to support all family members.
The specification defines the intended new rule; your seed-0 puzzle is an example.
The host checks all legal explicit counts and rejects illegal counts. No independent
solver or blind model prediction is used. Keep stored JSON compact (no indentation);
use .txt digit grids to inspect puzzles. Record unresolved limitations honestly.
