"""Deterministically render the task-plus-summary generator prompt."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence

from .config import RunnerConfig
from .data import PuzzleSource, SummarySource, load_puzzle, load_summary


TASK_PLACEHOLDER = "{TASK_JSON}"
SUMMARY_PLACEHOLDER = "{SEVEN_SECTION_SUMMARY}"


@dataclass(frozen=True)
class PromptBuild:
    prompt_text: str
    prompt_sha256: str
    prompt_version: str
    template_sha256: str
    target: PuzzleSource
    summary: SummarySource


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


def build_prompt(config: RunnerConfig, puzzle_id: str, source_split: str = "auto") -> PromptBuild:
    target = load_puzzle(config, puzzle_id, source_split)
    summary = load_summary(config, target)
    try:
        template_raw = config.prompt_template_path.read_bytes()
        template = template_raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Could not read prompt template %s: %s" % (config.prompt_template_path, exc)) from exc
    if template.count(TASK_PLACEHOLDER) != 1:
        raise ValueError("Prompt template must contain {TASK_JSON} exactly once")
    if template.count(SUMMARY_PLACEHOLDER) != 1:
        raise ValueError("Prompt template must contain {SEVEN_SECTION_SUMMARY} exactly once")

    task_text = format_puzzle(target)
    prompt_text = template.replace(TASK_PLACEHOLDER, task_text).replace(
        SUMMARY_PLACEHOLDER, summary.text.strip()
    )
    prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return PromptBuild(
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        prompt_version=config.prompt_version,
        template_sha256=hashlib.sha256(template_raw).hexdigest(),
        target=target,
        summary=summary,
    )
