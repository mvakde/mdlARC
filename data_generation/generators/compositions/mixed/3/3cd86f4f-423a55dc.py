"""Lossless foreground-anchored row shear on a dimension-padded canvas.

Only NumPy and the standard library are used. No files are read or written by
generation, validation, or self_test; all retained diagnostics are host-owned.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 218115083, 17650613, 870183660, 785874213, 111586796,
              256669421, 110598117, 219116004, 232451833, 742354612, 593681777,
              599798174, 799736101, 719448775, 600779420, 723248500, 320600937,
              741063625, 515257347, 267980323, 875137183, 153568143, 627247725,
              615110763, 753642919, 312293787, 767365018, 696983732, 699850722,
              833777469, 421210688, 388077282, 188791223, 859936366, 724698489,
              1774978, 225062359, 143721475, 898737010, 379703808, 96650266)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return a


def _roles(a):
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 2 or counts[0] == counts[1]:
        raise ValueError("input needs two colors and a unique majority background")
    bg_index = int(np.argmax(counts))
    return int(colors[bg_index]), int(colors[1 - bg_index])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive colors, global lowest occupied row, dimensions, and offsets locally.

    Occupied rows are copied as full vectors. Rows below the anchor are blank:
    they are left blank instead of attempting an out-of-bounds background copy.
    """
    a = _grid(input_grid)
    height, width = a.shape
    if height + width - 1 > 30:
        raise ValueError("padded output would exceed 30 columns")
    background, _ = _roles(a)
    occupied = np.flatnonzero(np.any(a != background, axis=1))
    anchor = int(occupied[-1])
    out = np.full((height, height + width - 1), background, dtype=np.int8)
    for row in occupied:
        offset = height - 1 - (anchor - int(row))
        out[row, offset:offset + width] = a[row]
    return out


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_digest(puzzle):
    # Compact serialization is bounded by fourteen small pairs; keep only a hash.
    serial = {split: [{role: np.asarray(pair[role]).tolist()
                       for role in ("input", "output")} for pair in puzzle[split]]
              for split in ("train", "test")}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def _mask(rng, height, width, kind):
    """Make horizontal slices; no row starts empty before explicit gap insertion."""
    mask = np.zeros((height, width), dtype=bool)
    if kind == 0:
        mask[:] = True
    elif kind == 1:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif kind == 2:
        # Variable bars joined by a visible common spine.
        spine = int(rng.integers(width))
        for r in range(height):
            lo = int(rng.integers(spine + 1))
            hi = int(rng.integers(spine + 1, width + 1))
            mask[r, lo:hi] = True
    elif kind == 3:
        # Staggered intervals, with independently changing thickness.
        center = int(rng.integers(width))
        for r in range(height):
            previous = center
            center = int(np.clip(center + rng.integers(-1, 2), 0, width - 1))
            lo = max(0, min(center, previous) - int(rng.integers(3)))
            hi = min(width, max(center, previous) + int(rng.integers(1, 4)))
            mask[r, lo:hi] = True
    elif kind == 4:
        # Sparse side branches; repeated rows and asymmetry are intentional.
        spine = int(rng.integers(width))
        mask[:, spine] = True
        for r in range(height):
            if rng.random() < 0.65:
                tip = int(rng.integers(width))
                mask[r, min(tip, spine):max(tip, spine) + 1] = True
    elif kind == 5:
        # Uneven hollow slices, including within-row gaps.
        for r in range(height):
            lo = int(rng.integers(width))
            hi = int(rng.integers(lo, width))
            mask[r, lo] = mask[r, hi] = True
    else:
        # Repeated arbitrary row vector: its holes must travel intact.
        columns = rng.choice(width, size=int(rng.integers(1, width + 1)), replace=False)
        mask[:, columns] = True
    if rng.random() < 0.5:
        mask = mask[:, ::-1].copy()
    return mask


def _shape(rng, split, index, mode, train):
    if split == "train":
        if index < 2:
            h = int(rng.integers(7, 14))
            w = int(rng.integers(4, min(12, 31 - h) + 1))
        else:
            h = int(rng.integers(1, 21))
            w = int(rng.integers(1, min(18, 30 - h) + 1))
            if h * w < 3:
                w = 3
        return h, w
    if mode == "tall":
        h = int(rng.integers(max(p["input"].shape[0] for p in train) + 1, 31))
        return h, int(rng.integers(1, 32 - h))
    if mode == "wide":
        w = int(rng.integers(max(p["input"].shape[1] for p in train) + 1, 31))
        return int(rng.integers(1, 32 - w)), w
    if mode == "boundary":
        h = int(rng.integers(1, 31))
        return h, 31 - h
    h = int(rng.integers(8, 24))
    return h, int(rng.integers(1, min(12, 31 - h) + 1))


def _input(rng, height, width, background, foreground, witness, big_gap):
    # This function may return a too-dense candidate; the caller rejects it.
    if witness:
        span = int(rng.integers(4, min(7, height - 1) + 1))
        patch_width = int(rng.integers(3, width + 1))
        top = int(rng.integers(0, height - span))  # at least one blank bottom row
        left = width - patch_width
        gap = 1
    elif big_gap:
        gap = int(rng.integers(2, 5))
        span = gap + int(rng.integers(3, min(6, height - gap) + 1))
        patch_width = int(rng.integers(1, width + 1))
        top = int(rng.integers(0, height - span + 1))
        left = int(rng.integers(0, width - patch_width + 1))
    else:
        span = int(rng.integers(1, height + 1))
        patch_width = int(rng.integers(1, width + 1))
        top = int(rng.integers(0, height - span + 1))
        left = int(rng.integers(0, width - patch_width + 1))
        gap = int(span >= 4 and rng.random() < 0.28)
    kind = int(rng.integers(7))
    mask = _mask(rng, span, patch_width, kind)
    if gap:
        gap_start = int(rng.integers(1, span - gap))
        mask[gap_start:gap_start + gap] = False
    if witness:
        # A natural endpoint on the anchor slice proves widening preserves cells.
        mask[-1, -1] = True
    a = np.full((height, width), background, dtype=np.int8)
    a[top:top + span, left:left + patch_width][mask] = foreground
    return a


def _colors(rng, palette_mode, base_background, base_foreground):
    if palette_mode == "fixed":
        return base_background, base_foreground
    bg = base_background if palette_mode == "shared_background" else int(rng.integers(10))
    fg = int(rng.choice([c for c in range(10) if c != bg]))
    return bg, fg


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate distinct pairs under one fixed rule, with four shift profiles."""
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(("tall", "wide", "boundary", "gap")))
    palette_mode = str(rng.choice(("fixed", "shared_background", "variable")))
    base_background = int(rng.integers(10))
    base_foreground = int(rng.choice([c for c in range(10) if c != base_background]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            bg, fg = _colors(rng, palette_mode, base_background, base_foreground)
            for _ in range(192):
                h, w = _shape(rng, split, index, mode, puzzle["train"])
                inp = _input(rng, h, w, bg, fg, split == "train" and index == 0,
                             split == "test" and mode == "gap")
                mass = int(np.count_nonzero(inp == fg))
                if not 0 < 2 * mass < inp.size or _signature(inp) in seen:
                    continue
                seen.add(_signature(inp))
                puzzle[split].append({"input": inp, "output": solve(inp)})
                break
            else:
                raise RuntimeError("bounded unique-input sampling exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(a, model):
    """Finite, explicit shortcut checks; these are not an independent solver."""
    h, w = a.shape
    bg, fg = _roles(a)
    occupied = np.flatnonzero(np.any(a == fg, axis=1))
    bottom, top = int(occupied[-1]), int(occupied[0])
    ranks = {int(r): i for i, r in enumerate(occupied)}
    component_bottom = {}
    if model == "component_anchor":
        remaining = {tuple(int(v) for v in rc) for rc in np.argwhere(a == fg)}
        while remaining:
            start = remaining.pop()
            component, stack = {start}, [start]
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (rr + dr, cc + dc)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            anchor = max(rr for rr, _ in component)
            component_bottom.update({rc: anchor for rc in component})
    out = np.full((h, h + w - 1), bg, dtype=np.int8)
    for r, c in np.argwhere(a == fg):
        if model == "canvas_bottom":
            offset = int(r)
        elif model == "A_alone":
            offset = h - 1 - int(r)
        elif model == "rigid":
            offset = h - 1
        elif model == "object_top":
            offset = h - 1 + int(r) - top
        elif model == "double_rate":
            offset = h - 1 - 2 * (bottom - int(r))
        elif model == "occupied_rank":
            offset = h - 1 - (len(occupied) - 1 - ranks[int(r)])
        elif model == "B_without_padding":
            offset = -(bottom - int(r))
        elif model == "component_anchor":
            offset = h - 1 - (component_bottom[(int(r), int(c))] - int(r))
        elif model == "crop_base":
            offset = int(r) - top
        else:
            raise ValueError("unknown alternative")
        cc = int(c) + offset
        if 0 <= cc < out.shape[1]:
            out[int(r), cc] = fg
    return out


_ALTERNATIVES = ("canvas_bottom", "A_alone", "rigid", "object_top",
                 "double_rate", "occupied_rank", "B_without_padding",
                 "component_anchor", "crop_base")


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    h, w = a.shape
    assert h + w - 1 <= 30 and out.shape == (h, h + w - 1)
    bg, fg = _roles(a)
    rows, cols = np.nonzero(a == fg)
    anchor = int(rows.max())
    # Check using cell coordinates, independently of solve's row-vector copies.
    expected = np.full(out.shape, bg, dtype=np.int8)
    targets = cols + h - 1 - (anchor - rows)
    assert np.all(targets >= 0) and np.all(targets < out.shape[1])
    expected[rows, targets] = fg
    assert np.array_equal(out, expected), "output violates the saved affine rule"
    assert int(np.count_nonzero(out == fg)) == len(rows), "foreground was lost or duplicated"
    for r in range(h):
        source_cols = np.flatnonzero(a[r] == fg)
        dest_cols = np.flatnonzero(out[r] == fg)
        assert len(source_cols) == len(dest_cols)
        if len(source_cols):
            offset = h - 1 - (anchor - r)
            assert np.array_equal(dest_cols - source_cols, np.full(len(source_cols), offset))
            assert np.array_equal(out[r, offset:offset + w], a[r])
        else:
            assert np.all(out[r] == bg), "blank rows must stay blank"
    occupied = np.flatnonzero(np.any(a == fg, axis=1))
    internal = np.any(a == fg, axis=1)[occupied[0]:occupied[-1] + 1]
    gaps, current = [], 0
    for bit in internal:
        current = 0 if bit else current + 1
        gaps.append(current)
    return {"background": bg, "foreground": fg, "anchor": anchor,
            "span": int(occupied[-1] - occupied[0] + 1),
            "occupied_rows": len(occupied), "max_gap": max(gaps),
            "bottom_margin": h - 1 - anchor, "extended": bool(np.any(targets >= w))}


def validate_generated_puzzle(puzzle):
    """Validate both pair semantics and required demonstration/shift evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            fact = _inspect_pair(pair)
            signature = _signature(pair["input"])
            assert signature not in seen, "inputs must be distinct within a puzzle"
            seen.add(signature)
            facts[split].append(fact)
    assert any(f["bottom_margin"] > 0 and f["max_gap"] >= 1
               and f["occupied_rows"] >= 3 and f["extended"] for f in facts["train"])
    for model in _ALTERNATIVES:
        assert any(not np.array_equal(_alternative(np.asarray(p["input"]), model), p["output"])
                   for p in puzzle["train"]), f"training fails to distinguish {model}"
    # One of the four visible, measurable shift contracts must hold for all tests.
    max_h = max(np.asarray(p["input"]).shape[0] for p in puzzle["train"])
    max_w = max(np.asarray(p["input"]).shape[1] for p in puzzle["train"])
    max_output_w = max(np.asarray(p["output"]).shape[1] for p in puzzle["train"])
    max_gap = max(f["max_gap"] for f in facts["train"])
    assert (all(np.asarray(p["input"]).shape[0] > max_h for p in puzzle["test"])
            or all(np.asarray(p["input"]).shape[1] > max_w for p in puzzle["test"])
            or (max_output_w < 30 and all(sum(np.asarray(p["input"]).shape) - 1 == 30
                                         for p in puzzle["test"]))
            or all(f["max_gap"] > max_gap for f in facts["test"]))


def _metamorphic_checks(a):
    """Check color equivariance, losslessness and the two margin dependencies."""
    out = solve(a)
    bg, fg = _roles(a)
    # A bijection of all ten colors must commute with this geometric operation.
    permutation = (np.arange(10) * 3 + 7) % 10
    assert np.array_equal(solve(permutation[a]), permutation[out])
    h, w = a.shape
    rows, _ = np.nonzero(a == fg)
    bottom = int(rows.max())
    recovered = np.full(a.shape, bg, dtype=np.int8)
    for r in np.unique(rows):
        offset = h - 1 - (bottom - int(r))
        recovered[r] = out[r, offset:offset + w]
    assert np.array_equal(recovered, a)
    if h + w <= 30:
        # Bottom padding raises H but leaves r0 fixed: every slice shifts right 1.
        below = np.vstack((a, np.full((1, w), bg, dtype=np.int8)))
        expected = np.full((h + 1, out.shape[1] + 1), bg, dtype=np.int8)
        expected[:h, 1:] = out
        assert np.array_equal(solve(below), expected)
        # Top padding preserves all relative depths but raises the H-1 base.
        above = np.vstack((np.full((1, w), bg, dtype=np.int8), a))
        expected[:] = bg
        expected[1:, 1:] = out
        assert np.array_equal(solve(above), expected)
    if bottom < h - 1:
        # At fixed H, vertical translation changes r and r0 equally.
        shifted = np.full(a.shape, bg, dtype=np.int8)
        shifted[1:] = a[:-1]
        expected = np.full(out.shape, bg, dtype=np.int8)
        expected[1:] = out[:-1]
        assert np.array_equal(solve(shifted), expected)


def self_test():
    """Bounded 42-seed audit plus explicit count/domain and mutation checks.

    No payloads, logs, execution traces, or files are retained. The returned
    compact-JSON-compatible summary lets the caller preserve measured evidence.
    """
    shapes, normalized, backgrounds, foregrounds, digests = set(), set(), set(), set(), set()
    spans, gaps, margins, masses, width_limits = set(), set(), set(), set(), 0
    relations, palette_profiles = set(), set()
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        digests.add(_puzzle_digest(puzzle))
        max_h = max(p["input"].shape[0] for p in puzzle["train"])
        max_w = max(p["input"].shape[1] for p in puzzle["train"])
        train_gap = max(_inspect_pair(p)["max_gap"] for p in puzzle["train"])
        palette_profiles.add(tuple(sorted({_roles(p["input"]) for p in puzzle["train"] + puzzle["test"]})))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                f = _inspect_pair(pair)
                pristine = a.copy()
                assert np.array_equal(solve(a), out) and np.array_equal(a, pristine)
                _metamorphic_checks(a)
                pairs += 1
                shapes.add(a.shape)
                normalized.add(_signature((a == f["foreground"]).astype(np.int8)))
                backgrounds.add(f["background"])
                foregrounds.add(f["foreground"])
                spans.add(f["span"])
                gaps.add(f["max_gap"])
                margins.add(f["bottom_margin"])
                masses.add(int(np.count_nonzero(a == f["foreground"])))
                width_limits += out.shape[1] == 30
                if split == "test":
                    if a.shape[0] > max_h:
                        relations.add("taller")
                    if a.shape[1] > max_w:
                        relations.add("wider")
                    if f["max_gap"] > train_gap:
                        relations.add("larger_gap")
                    if out.shape[1] == 30:
                        relations.add("width_limit")
    assert len(digests) == 42 and len(shapes) >= 60 and len(normalized) >= 160
    assert len(backgrounds) >= 8 and len(foregrounds) >= 8
    assert len(spans) >= 12 and {0, 1, 2, 3, 4} <= gaps
    assert len(margins) >= 10 and len(masses) >= 30 and width_limits >= 5
    assert relations == {"taller", "wider", "larger_gap", "width_limit"}
    assert len(palette_profiles) >= 25
    count_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1701 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
            count_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicit boundary and degenerate members of the input-only rule domain.
    for h, w in ((1, 30), (30, 1), (2, 29), (29, 2), (15, 16), (1, 3), (3, 1)):
        a = np.full((h, w), 8, dtype=np.int8)
        a[0, 0] = 0
        if h * w > 6:
            a[h // 2, w - 1] = 0
        p = {"input": a, "output": solve(a)}
        _inspect_pair(p)
        _metamorphic_checks(a)
    # The semantic validator must reject a lost cell, not just agree with solve.
    p = generate_puzzle(0)
    pair = p["train"][0]
    bg, fg = _roles(pair["input"])
    r, c = np.argwhere(pair["output"] == fg)[0]
    pair["output"][r, c] = bg
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a deleted foreground cell")
    return {"seeds": len(HOST_SEEDS), "pairs": pairs, "distinct_puzzles": len(digests),
            "input_shapes": len(shapes), "normalized_inputs": len(normalized),
            "backgrounds": len(backgrounds), "foregrounds": len(foregrounds),
            "spans": len(spans), "max_internal_gaps": sorted(gaps),
            "bottom_margins": len(margins), "foreground_counts": len(masses),
            "width_30_outputs": width_limits, "test_shift_types": sorted(relations),
            "palette_profiles": len(palette_profiles), "legal_count_calls": count_calls,
            "semantic_verification": "generator-owned invariants and finite shortcut checks only"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
