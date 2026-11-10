"""Render source, reviewed, or newly generated ARC grids without extra packages.

Examples in a staged workspace:
  python render.py task.json --output scratch/source.png
  python render.py reviewed_samples.json --output scratch/reviewed.png
  python render.py --seed 123 --output scratch/candidate.png
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path


COLORS = ("000000", "0074D9", "FF4136", "2ECC40", "FFDC00", "AAAAAA", "F012BE", "FF851B", "7FDBFF", "870C25")


def _pairs(task):
    return [(split, index, pair) for split in ("train", "test")
            for index, pair in enumerate(task[split])]


def render_png(task, cell_size=12):
    """One input/output pair per row, training first. Return PNG bytes."""
    rows = _pairs(task)
    if not rows:
        raise ValueError("task has no pairs")
    gap = 16
    left_width = max(len(p["input"][0]) for _, _, p in rows) * cell_size
    right_width = max(len(p["output"][0]) for _, _, p in rows) * cell_size
    width = left_width + right_width + gap * 3
    height = sum(max(len(p["input"]), len(p["output"])) * cell_size + gap for _, _, p in rows) + gap
    pixels = [bytearray(bytes((238, 240, 244)) * width) for _ in range(height)]
    palette = [bytes.fromhex(color) for color in COLORS]
    y = gap
    for split, _, pair in rows:
        for role, x in (("input", gap), ("output", left_width + gap * 2)):
            for r, row in enumerate(pair[role]):
                for c, color in enumerate(row):
                    for dy in range(cell_size):
                        line = pixels[y + r * cell_size + dy]
                        for dx in range(cell_size):
                            pos = (x + c * cell_size + dx) * 3
                            line[pos:pos+3] = palette[int(color)] if dx and dy else bytes((70, 74, 80))
        # Blue margin identifies training; orange identifies test rows.
        marker = bytes.fromhex("0074D9" if split == "train" else "FF851B")
        for yy in range(y, y + max(len(pair["input"]), len(pair["output"])) * cell_size):
            pixels[yy][0:9] = marker * 3
        y += max(len(pair["input"]), len(pair["output"])) * cell_size + gap

    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value) & 0xffffffff)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(b"\0" + row for row in pixels))) + chunk(b"IEND", b""))


def source_svg(task):
    """A labeled, text-backed source rendering that can be pinned in the manifest."""
    rows = _pairs(task)
    cell, gap = 12, 30
    left = max(len(p["input"][0]) for _, _, p in rows) * cell
    width = left + max(len(p["output"][0]) for _, _, p in rows) * cell + gap * 3
    height = sum(max(len(p["input"]), len(p["output"])) * cell + gap for _, _, p in rows) + gap
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#eef0f4"/>']
    y = gap
    for split, index, pair in rows:
        for role, x in (("input", gap), ("output", left + gap * 2)):
            parts.append(f'<text x="{x}" y="{y-6}" font-family="sans-serif" font-size="12">{split} {index+1} {role}</text>')
            for r, row in enumerate(pair[role]):
                for c, color in enumerate(row):
                    parts.append(f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell}" height="{cell}" fill="#{COLORS[int(color)]}" stroke="#464a50" stroke-width="0.5"/>')
        y += max(len(pair["input"]), len(pair["output"])) * cell + gap
    return "\n".join(parts + ["</svg>\n"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="task.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", default="scratch/render.png")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    scratch = root / "scratch"
    scratch.mkdir(exist_ok=True)
    output = (root / args.output).resolve()
    if scratch.resolve() not in output.parents:
        raise ValueError("render output must be inside scratch/")
    if args.seed is not None:
        namespace = {"__name__": "render_generator"}
        code = (root / "generator.py").read_text(encoding="utf-8")
        exec(compile(code, "generator.py", "exec"), namespace)
        task = namespace["generate_puzzle"](args.seed)
        tasks = [(f"seed-{args.seed}", task)]
    else:
        path = (root / args.input).resolve()
        if root not in path.parents:
            raise ValueError("input must be inside the puzzle workspace")
        value = json.loads(path.read_text(encoding="utf-8"))
        tasks = [(f"seed-{entry['seed']}", entry["task"]) for entry in value["samples"]] if "samples" in value else [("source", value)]
    output.parent.mkdir(parents=True, exist_ok=True)
    for label, task in tasks:
        target = output if len(tasks) == 1 else output.with_name(output.stem + "-" + label + output.suffix)
        target.write_bytes(render_png(task))
        target.with_suffix(".svg").write_text(source_svg(task), encoding="utf-8")
        print(target)
    print("Input left, output right; pairs in source order. Blue margin: train, orange: test. SVG companion has labels.")


if __name__ == "__main__":
    main()
