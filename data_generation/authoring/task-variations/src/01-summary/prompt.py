"""Deterministic prompt assembly for ARC-2 rule summaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .config import RunnerConfig
from .data import Demonstration, PuzzleSource, ensure_not_demo_target, load_demonstrations, load_puzzle


OUTPUT_CONTRACT = """<rules_summary>
Describe the transformation as a general rule using semantic roles rather
than particular colors, orientations, sizes, or example-specific layouts.
When shape or pattern identity determines a categorical output, preserve the
demonstrated shape vocabulary and its mapping explicitly. Do not replace exact
shape recognition with a coincidentally matching count or connectivity statistic.
</rules_summary>

<input_generation>
Describe the objects and relationships that every valid input must contain.
Include the constraints needed to make the puzzle unambiguous.
</input_generation>

<variation_guidance>
Explain what may change across valid instances. Cover only relevant dimensions,
such as orientation, rotation, size, object count, layout, spacing, palette
complexity, and noise. Describe colors by their roles. Literal color identities
may change unless a particular color is semantically meaningful.
</variation_guidance>

<train_test_shift>
Explain how the test examples differ systematically from the training examples.
State whether the test introduces a new orientation, larger sizes, more objects,
a different layout, a new combination of familiar factors, reassigned color
roles, or another controlled shift. End by stating what generalization the
solver or generator must preserve.
</train_test_shift>

<solution_steps>
Give the ordered solution procedure in role-based language.
</solution_steps>

<key_insight>
State the single most important reasoning insight.
</key_insight>

<puzzle_concepts>
- concept tags
</puzzle_concepts>"""


@dataclass(frozen=True)
class PromptBuild:
    prompt_text: str
    prompt_sha256: str
    prompt_version: str
    target: PuzzleSource
    demonstrations: Tuple[Demonstration, ...]


def _format_grid(grid: Sequence[Sequence[int]], role: str) -> str:
    colors = sorted({cell for row in grid for cell in row})
    lines = [
        "- %s height: %d" % (role, len(grid)),
        "- %s width: %d" % (role, len(grid[0])),
        "- %s unique colors: %s" % (role, colors),
        "- %s grid:" % role,
    ]
    lines.extend("".join(str(cell) for cell in row) for row in grid)
    return "\n".join(lines)


def format_puzzle(puzzle: PuzzleSource) -> str:
    sections: List[str] = []
    for split in ("train", "test"):
        pairs: List[str] = []
        for index, pair in enumerate(puzzle.data[split], 1):
            pairs.append(
                "<pair_%d>\n<input>\n%s\n</input>\n<output>\n%s\n</output>\n</pair_%d>"
                % (
                    index,
                    _format_grid(pair["input"], "Input"),
                    _format_grid(pair["output"], "Output"),
                    index,
                )
            )
        sections.append("<%s>\n%s\n</%s>" % (split, "\n".join(pairs), split))
    return "\n".join(sections)


def _render(target: PuzzleSource, demos: Sequence[Demonstration], prompt_version: str,
            target_dataset: str = "ARC-2") -> str:
    demo_blocks: List[str] = []
    for index, demo in enumerate(demos, 1):
        demo_blocks.append(
            "<demonstration_%d id=\"%s\" source_split=\"%s\">\n<puzzle>\n%s\n</puzzle>\n"
            "<reference_summary>\n%s\n</reference_summary>\n</demonstration_%d>"
            % (
                index,
                demo.puzzle.puzzle_id,
                demo.puzzle.source_split,
                format_puzzle(demo.puzzle),
                demo.summary_text.strip(),
                index,
            )
        )

    if target_dataset == "ARC-2":
        introduction = "You are producing a high-fidelity rule summary for a solved ARC-2 puzzle.\n"
        split_explanation = (
            "The target is distinct from the demonstrations. The source_split attribute says whether a task "
            "comes from the ARC-2 training or evaluation corpus; it is separate from the solved <train> and "
            "<test> examples inside every task. "
        )
    else:
        introduction = "You are producing a high-fidelity rule summary for a solved %s puzzle.\n" % target_dataset
        split_explanation = (
            "The target is from %s. Its source_split=\"training\" is the local artifact group for this "
            "dataset. The six reference demonstrations are from ARC-2 evaluation. These source labels are "
            "separate from the solved <train> and <test> examples inside every task. " % target_dataset
        )

    return (
        "%s"
        "All puzzle evidence and all reference demonstrations are embedded in this user message. "
        "Do not use tools, browse, or open files. Analyze only the grids and summaries below.\n\n"
        "%sTreat color identities as semantic roles unless "
        "the evidence makes a literal identity meaningful. Training and test pairs are labeled explicitly; "
        "use those labels when describing the controlled train/test shift.\n\n"
        "Distinguish the demonstrated rule from hypotheses that merely fit the examples. "
        "For shape-to-label tasks, shape identity can be the semantic role itself: preserve "
        "the demonstrated masks when the evidence supports a lookup. Matching component "
        "counts, areas, or other statistics do not establish that unseen shapes share a label. "
        "Do not invent a broader rule or extra reasoning steps just to enable variation.\n\n"
        "<target_puzzle id=\"%s\" source_split=\"%s\">\n%s\n</target_puzzle>\n\n"
        "Here are exactly six complete solved demonstrations in fixed order. Each demonstration includes "
        "its solved training/test grids and its human-authored seven-section summary.\n\n"
        "%s\n\n"
        "## Required Output\n\n"
        "Return only the following seven XML sections, exactly once each and in exactly this order. "
        "Do not include analysis, Markdown fences, prose outside the sections, or any other XML tags.\n\n"
        "%s\n"
        % (
            introduction,
            split_explanation,
            target.puzzle_id,
            target.source_split,
            format_puzzle(target),
            "\n\n".join(demo_blocks),
            OUTPUT_CONTRACT,
        )
    )


def build_prompt(
    config: RunnerConfig, puzzle_id: str, source_split: str = "auto"
) -> PromptBuild:
    target = load_puzzle(config, puzzle_id, source_split)
    ensure_not_demo_target(config, puzzle_id, target.source_split)
    demos = load_demonstrations(config)
    expected_ids = [spec.puzzle_id for spec in config.fixed_demonstrations]
    actual_ids = [demo.puzzle.puzzle_id for demo in demos]
    if actual_ids != expected_ids:
        raise ValueError("Fixed demonstration order changed unexpectedly")

    prompt_text = _render(target, demos, config.prompt_version, config.target_dataset)
    positions = [prompt_text.index('id="%s"' % puzzle_id) for puzzle_id in actual_ids]
    if positions != sorted(positions):
        raise ValueError("Fixed demonstrations are not in deterministic order")
    for demo in demos:
        if prompt_text.count(demo.summary_text.strip()) != 1:
            raise ValueError("Manual summary %s was not included exactly once" % demo.puzzle.puzzle_id)
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return PromptBuild(prompt_text, prompt_hash, config.prompt_version, target, demos)
