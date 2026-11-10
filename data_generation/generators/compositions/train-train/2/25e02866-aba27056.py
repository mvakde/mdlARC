"""Shattered window: fuse congruent panel layers, then cut the emitted light
pattern of the reassembled open frame out of the carrier sheet.

Parent A (25e02866): separated congruent square panels are transparent layers
sharing one carrier colour; their accent cells are superimposed at unchanged
local coordinates to give one fused square.

Parent B (aba27056): an incomplete rectangle with one contiguous gap (with
intact non-corner flanks) is filled inside and across the gap, emits one
perpendicular ray per gap cell, and two diverging diagonal rays from the frame
cells flanking the gap, all clipped at the boundary.

Composition: every panel holds only a fragment of one rectangle outline.  No
single panel contains a readable open frame; the frame and its unique opening
exist only in the fused layer.  The fused square is then activated with B's
stencil, and the activated cells are cut out of the carrier, i.e. painted with
the canvas (background) colour of the input, so the fill colour is visibly
encoded in every input.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

# Opening direction names, indexed by the np.rot90 count applied to the
# canonical (opening faces up) panel pattern.
_SIDES = ("up", "left", "down", "right")
_OUTWARD = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# --------------------------------------------------------------------------
# Input-only solver
# --------------------------------------------------------------------------

def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                comp = []
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(comp)
    return comps


def _panels_for_background(grid: np.ndarray, bg: int):
    """Return sorted panel origins and side if every non-bg component is a
    filled square of one common size (at least two of them), else None."""
    comps = _components(grid != bg)
    if len(comps) < 2:
        return None
    origins, side = [], None
    for comp in comps:
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h != w or h * w != len(comp):
            return None
        if side is None:
            side = h
        elif side != h:
            return None
        origins.append((r0, c0))
    return sorted(origins), side


def _segment(grid: np.ndarray):
    """Find the canvas colour and the congruent square panels."""
    counts = Counter(int(v) for v in grid.flat)
    candidates = []
    for color, _ in counts.most_common():
        found = _panels_for_background(grid, color)
        if found is not None:
            candidates.append((color, found))
    if not candidates:
        raise ValueError("no canvas colour separates congruent square panels")
    bg, (origins, side) = candidates[0]
    panels = [grid[r:r + side, c:c + side] for r, c in origins]
    return bg, panels, origins, side, len(candidates)


def _fuse(panels: list[np.ndarray]):
    """Parent A: overlay the panels' accent cells on one carrier square."""
    side = panels[0].shape[0]
    counts = Counter(int(v) for p in panels for v in p.flat)
    carrier = counts.most_common(1)[0][0]
    fused = np.full((side, side), carrier, dtype=int)
    for p in panels:
        mask = p != carrier
        fused[mask] = p[mask]
    return fused, carrier


def _parse_frame(fused: np.ndarray, carrier: int):
    """Locate the open frame in the fused square: bounding box, gap side and
    gap cells (ordered along the side)."""
    frame = fused != carrier
    rows, cols = np.nonzero(frame)
    if rows.size == 0:
        raise ValueError("empty fused layer")
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    sides = {
        "up": [(top, c) for c in range(left, right + 1)],
        "down": [(bottom, c) for c in range(left, right + 1)],
        "left": [(r, left) for r in range(top, bottom + 1)],
        "right": [(r, right) for r in range(top, bottom + 1)],
    }
    open_sides = [(name, [p for p in cells if not frame[p]])
                  for name, cells in sides.items()]
    open_sides = [(n, g) for n, g in open_sides if g]
    if len(open_sides) != 1:
        raise ValueError("fused frame must have exactly one open side")
    name, gap = open_sides[0]
    return (int(top), int(bottom), int(left), int(right)), name, gap


def _activate(fused: np.ndarray, carrier: int, paint: int) -> np.ndarray:
    """Parent B stencil on the fused square, painted with `paint`."""
    (top, bottom, left, right), side, gap = _parse_frame(fused, carrier)
    out = fused.copy()
    h, w = out.shape
    out[top + 1:bottom, left + 1:right] = paint
    for r, c in gap:
        out[r, c] = paint
    dr, dc = _OUTWARD[side]
    for r, c in gap:
        y, x = r + dr, c + dc
        while 0 <= y < h and 0 <= x < w:
            out[y, x] = paint
            y, x = y + dr, x + dc
    # flanking frame cells immediately beyond each end of the gap
    ar, ac = (0, 1) if side in ("up", "down") else (1, 0)  # along the side
    first, last = gap[0], gap[-1]
    flanks = [((first[0] - ar, first[1] - ac), (-ar, -ac)),
              ((last[0] + ar, last[1] + ac), (ar, ac))]
    for (fr, fc), (sr, sc) in flanks:
        y, x = fr + dr, fc + dc
        while 0 <= y < h and 0 <= x < w:
            out[y, x] = paint
            y, x = y + dr + sr, x + dc + sc
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    bg, panels, _, _, _ = _segment(grid)
    fused, carrier = _fuse(panels)
    return _activate(fused, carrier, bg)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _canonical_outline(top, left, depth, length):
    """Clockwise perimeter of a depth x length rectangle starting at its
    top-left corner."""
    bottom, right = top + depth - 1, left + length - 1
    cells = [(top, c) for c in range(left, right + 1)]
    cells += [(r, right) for r in range(top + 1, bottom + 1)]
    cells += [(bottom, c) for c in range(right - 1, left - 1, -1)]
    cells += [(r, left) for r in range(bottom - 1, top, -1)]
    return cells


def _sample_frame(rng, side, demo):
    """Canonical frame (opening on the top side, facing up) inside a
    side x side panel.  Returns (top, left, depth, length, flank_l, gap)."""
    for _ in range(400):
        length = int(rng.integers(5, side + 1))
        depth = int(rng.integers(3, side + 1))
        gap = int(rng.integers(1, length - 3))
        spare = length - 2 - gap  # flank cells shared between both ends (>=2)
        flank_l = int(rng.integers(1, spare))
        clearance_style = rng.random()
        max_top = side - depth
        if clearance_style < 0.12 and not demo:
            top = 0
        elif clearance_style < 0.55:
            top = max_top
        else:
            top = int(rng.integers(0, max_top + 1))
        left = int(rng.integers(0, side - length + 1))
        if demo:
            cl = left + flank_l
            cr = left + flank_l + gap + 1
            if top < 2 or min(top, cl + 1) < 2 or min(top, side - cr) < 2:
                continue
        return top, left, depth, length, flank_l, gap
    return None


def _build_layers(rng, side, count, demo):
    """Canonical label array: -1 carrier, i >= 0 fragment owned by panel i."""
    frame = None
    while frame is None:
        frame = _sample_frame(rng, side, demo)
    top, left, depth, length, flank_l, gap = frame
    outline = _canonical_outline(top, left, depth, length)
    # gap occupies outline indices (1 + flank_l) .. (flank_l + gap)
    start = flank_l + gap + 1
    path = outline[start:] + outline[:1 + flank_l]
    n = len(path)
    pieces = min(n, count + int(rng.integers(0, 4)))
    cuts = sorted(rng.choice(np.arange(1, n), size=pieces - 1, replace=False).tolist())
    bounds = [0] + cuts + [n]
    owners = list(rng.permutation(count)) + [int(rng.integers(count))
                                            for _ in range(pieces - count)]
    owners = [owners[i] for i in rng.permutation(pieces)]
    labels = np.full((side, side), -1, dtype=int)
    for piece in range(pieces):
        for idx in range(bounds[piece], bounds[piece + 1]):
            labels[path[idx]] = owners[piece]
    return labels


def _overlap(a, b, side):
    ar, ac = a
    br, bc = b
    # keep at least one background cell between squares (no 8-adjacency)
    return not (ar + side < br or br + side < ar or ac + side < bc or bc + side < ac)


def _place(rng, count, side):
    styles = ["scatter"]
    if count * side + (count - 1) <= 30:
        styles += ["row", "column"]
    if count * side + (count - 1) <= 30 and side <= 9:
        styles += ["diagonal", "antidiagonal"]
    if count >= 3 and 2 * side + 1 <= 30:
        styles += ["block"]
    style = styles[int(rng.integers(len(styles)))]
    positions = None
    if style in ("row", "column", "diagonal", "antidiagonal"):
        max_gap = (30 - count * side) // (count - 1)
        gaps = [int(rng.integers(1, min(max_gap, 4) + 1)) for _ in range(count - 1)]
        while count * side + sum(gaps) > 30:
            gaps[int(np.argmax(gaps))] -= 1
        along = [0]
        for g in gaps:
            along.append(along[-1] + side + g)
        if style == "row":
            positions = [(int(rng.integers(0, 4)), a) for a in along]
        elif style == "column":
            positions = [(a, int(rng.integers(0, 4))) for a in along]
        elif style == "diagonal":
            positions = [(a, a) for a in along]
        else:
            positions = [(a, along[-1] - a) for a in along]
    elif style == "block":
        g1, g2 = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        slots = [(0, 0), (0, side + g2), (side + g1, 0), (side + g1, side + g2)]
        chosen = sorted(rng.choice(4, size=count, replace=False).tolist())
        positions = [slots[i] for i in chosen]
    else:
        for _ in range(60):
            height = int(rng.integers(min(30, 2 * side + 2), 31))
            width = int(rng.integers(min(30, 2 * side + 2), 31))
            if height < side or width < side:
                continue
            pos = []
            for _ in range(count):
                for _ in range(200):
                    cand = (int(rng.integers(0, height - side + 1)),
                            int(rng.integers(0, width - side + 1)))
                    if all(not _overlap(cand, p, side) for p in pos):
                        pos.append(cand)
                        break
                else:
                    break
            if len(pos) == count:
                positions = pos
                break
        if positions is None:
            style = "block"
            slots = [(0, 0), (0, side + 1), (side + 1, 0), (side + 1, side + 1)]
            positions = slots[:count]
    min_r = min(p[0] for p in positions)
    min_c = min(p[1] for p in positions)
    positions = [(r - min_r, c - min_c) for r, c in positions]
    span_h = max(r for r, _ in positions) + side
    span_w = max(c for _, c in positions) + side
    m = [int(rng.integers(0, max(1, min(4, 30 - span_h) + 1))) for _ in range(2)]
    while span_h + m[0] + m[1] > 30:
        m[int(np.argmax(m))] -= 1
    n = [int(rng.integers(0, max(1, min(4, 30 - span_w) + 1))) for _ in range(2)]
    while span_w + n[0] + n[1] > 30:
        n[int(np.argmax(n))] -= 1
    positions = [(r + m[0], c + n[0]) for r, c in positions]
    return positions, span_h + m[0] + m[1], span_w + n[0] + n[1], style


def _make_pair(rng, spec, bg):
    count, side, rot, demo = spec["count"], spec["side"], spec["rot"], spec["demo"]
    labels = _build_layers(rng, side, count, demo)
    labels = np.rot90(labels, rot).copy()
    others = [c for c in range(10) if c != bg]
    carrier = int(rng.choice(others))
    pool = [c for c in others if c != carrier]
    if rng.random() < 0.2:
        accents = [int(rng.choice(pool))] * count
    else:
        accents = [int(c) for c in rng.choice(pool, size=count, replace=False)]
    positions, height, width, style = _place(rng, count, side)
    order = rng.permutation(count)  # which layer goes to which location
    grid = np.full((height, width), bg, dtype=int)
    for slot, (r, c) in enumerate(positions):
        layer = int(order[slot])
        panel = np.full((side, side), carrier, dtype=int)
        panel[labels == layer] = accents[layer]
        grid[r:r + side, c:c + side] = panel
    return {"input": grid, "output": solve(grid)}, style


def _sample_specs(rng, num_train, num_test):
    regime = ["free", "held_side", "more_panels", "bigger", "classic"][int(rng.integers(5))]
    train, test = [], []

    def sides_with_two(n, exclude=None):
        allowed = [r for r in range(4) if r != exclude]
        while True:
            s = [int(rng.choice(allowed)) for _ in range(n)]
            if len(set(s)) >= 2:
                return s

    held = int(rng.integers(4)) if regime in ("held_side", "classic") else None
    train_rots = sides_with_two(num_train, held)
    if regime == "more_panels":
        train_counts = [int(rng.integers(2, 4)) for _ in range(num_train)]
        test_counts = [max(train_counts) + 1 if i == 0 else int(rng.integers(2, 5))
                       for i in range(num_test)]
    elif regime == "classic":
        train_counts = [int(rng.integers(2, 4)) for _ in range(num_train)]
        test_counts = [4] + [int(rng.integers(2, 5)) for _ in range(num_test - 1)]
    else:
        train_counts = [int(rng.integers(2, 5)) for _ in range(num_train)]
        test_counts = [int(rng.integers(2, 5)) for _ in range(num_test)]
    if regime in ("bigger", "classic"):
        train_sides = [int(rng.integers(5, 9)) for _ in range(num_train)]
        hi = max(train_sides) + 1
        test_sides = [int(rng.integers(hi, 13)) if i == 0 else int(rng.integers(5, 13))
                      for i in range(num_test)]
    else:
        train_sides = [int(rng.integers(5, 13)) for _ in range(num_train)]
        test_sides = [int(rng.integers(5, 13)) for _ in range(num_test)]
    train_sides[0] = max(train_sides[0], 6)
    for i in range(num_train):
        train.append({"count": train_counts[i], "side": train_sides[i],
                      "rot": train_rots[i], "demo": i == 0})
    for i in range(num_test):
        if held is not None and i == 0:
            rot = held
        else:
            rot = int(rng.integers(4))
        test.append({"count": test_counts[i], "side": test_sides[i],
                     "rot": rot, "demo": False})
    return regime, train, test


def _backgrounds(rng, n_train, n_test):
    while True:
        train = [int(rng.integers(10)) for _ in range(n_train)]
        if len(set(train)) >= 2:
            break
    test = [int(rng.integers(10)) for _ in range(n_test)]
    return train, test


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        regime, train_specs, test_specs = _sample_specs(rng, num_train, num_test)
        train_bg, test_bg = _backgrounds(rng, num_train, num_test)
        pairs, seen = [], set()
        ok = True
        for spec, bg in zip(train_specs + test_specs, train_bg + test_bg):
            for _ in range(20):
                pair, _ = _make_pair(rng, spec, bg)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    pairs.append(pair)
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError("could not generate a valid puzzle")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _analyze(grid: np.ndarray) -> dict:
    """Structural facts recovered from an input alone (used for checks)."""
    bg, panels, origins, side, n_candidates = _segment(grid)
    fused, carrier = _fuse(panels)
    (top, bottom, left, right), open_side, gap = _parse_frame(fused, carrier)
    return {"bg": bg, "panels": panels, "origins": origins, "side": side,
            "candidates": n_candidates, "carrier": carrier, "fused": fused,
            "box": (top, bottom, left, right), "open": open_side, "gap": gap}


def _diagonal_lengths(info):
    side = info["side"]
    top, bottom, left, right = info["box"]
    gap, open_side = info["gap"], info["open"]
    (r0, c0), (r1, c1) = gap[0], gap[-1]
    if open_side == "up":
        return min(top, c0), min(top, side - c1 - 1), top
    if open_side == "down":
        clear = side - 1 - bottom
        return min(clear, c0), min(clear, side - c1 - 1), clear
    if open_side == "left":
        return min(left, r0), min(left, side - r1 - 1), left
    clear = side - 1 - right
    return min(clear, r0), min(clear, side - r1 - 1), clear


def _check_pair(inp, out):
    assert inp.ndim == 2 and out.ndim == 2
    assert all(1 <= d <= 30 for d in inp.shape + out.shape)
    assert inp.min() >= 0 and inp.max() <= 9
    info = _analyze(inp)
    assert info["candidates"] == 1, "canvas colour must be unique"
    panels, side, carrier, bg = info["panels"], info["side"], info["carrier"], info["bg"]
    assert 2 <= len(panels) <= 4 and 5 <= side <= 12
    # panels separated by at least one canvas cell including diagonals
    for i, a in enumerate(info["origins"]):
        for b in info["origins"][:i]:
            assert not _overlap(a, b, side)
    seen = np.zeros((side, side), dtype=bool)
    for p in panels:
        vals = Counter(int(v) for v in p.flat)
        assert vals.most_common(1)[0][0] == carrier
        assert vals.most_common(1)[0][1] > side * side // 2, "carrier must dominate"
        accent = p != carrier
        assert accent.any(), "every panel contributes a fragment"
        assert not (accent & seen).any(), "fragments are disjoint"
        seen |= accent
        assert bg not in vals
    # fused layer is exactly one rectangle outline with one contiguous gap
    top, bottom, left, right = info["box"]
    assert bottom - top >= 2 and right - left >= 2
    ring = np.zeros((side, side), dtype=bool)
    ring[top:bottom + 1, left:right + 1] = True
    ring[top + 1:bottom, left + 1:right] = False
    assert not (seen & ~ring).any(), "accent cells lie only on the outline"
    gap = info["gap"]
    along = [g[1] for g in gap] if info["open"] in ("up", "down") else [g[0] for g in gap]
    assert along == list(range(along[0], along[0] + len(along))), "gap contiguous"
    lo, hi = (left, right) if info["open"] in ("up", "down") else (top, bottom)
    assert along[0] - lo >= 2 and hi - along[-1] >= 2, "non-corner flanks on both ends"
    # no single panel is the whole frame
    for p in panels:
        assert int((p != carrier).sum()) < int(seen.sum())
    expected = _activate(info["fused"], carrier, bg)
    assert out.shape == (side, side)
    assert np.array_equal(out, expected)
    # both parents matter: plain overlay and any single panel differ
    assert not np.array_equal(out, info["fused"])
    for p in panels:
        assert not np.array_equal(out, p)
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    keys = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            key = (inp.shape, inp.tobytes())
            assert key not in keys, "inputs must be distinct"
            keys.add(key)
            infos[split].append(_check_pair(inp, out))
    tr = infos["train"]
    assert len({i["bg"] for i in tr}) >= 2, "training shows the fill colour varying"
    assert len({i["open"] for i in tr}) >= 2, "training shows two opening sides"
    first = tr[0]
    d1, d2, clear = _diagonal_lengths(first)
    assert clear >= 2 and d1 >= 2 and d2 >= 2, "first pair demonstrates the full stencil"


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def _assert_value_error(a, b):
    try:
        generate_puzzle(0, num_train=a, num_test=b)
    except ValueError:
        return
    raise AssertionError(f"illegal counts accepted: {a}/{b}")


def self_test() -> None:
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4, 5)):
        _assert_value_error(a, b)
    sides, counts, sizes, layouts, regimes = Counter(), Counter(), Counter(), Counter(), set()
    clip_kinds = Counter()
    signatures = set()
    held_out = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        for split in ("train", "test"):
            for p, q in zip(puzzle[split], again[split]):
                assert np.array_equal(p["input"], q["input"])
                assert np.array_equal(p["output"], q["output"])
        signatures.add(tuple(p["input"].tobytes() for p in puzzle["train"] + puzzle["test"]))
        train_open = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                sides[(split, info["open"])] += 1
                counts[(split, len(info["panels"]))] += 1
                sizes[(split, info["side"])] += 1
                d1, d2, clear = _diagonal_lengths(info)
                kind = ("edge" if clear == 0 else
                        "both_full" if d1 == clear and d2 == clear else
                        "clipped")
                clip_kinds[(split, kind)] += 1
                if split == "train":
                    train_open.add(info["open"])
                elif info["open"] not in train_open:
                    held_out += 1
                # Shortcut checks: overlay only / fixed fill colour
                fused = info["fused"]
                assert not np.array_equal(pair["output"], fused)
        # a fixed fill colour (first training canvas) fails somewhere in train
        first_bg = _analyze(puzzle["train"][0]["input"])["bg"]
        assert any(_analyze(p["input"])["bg"] != first_bg for p in puzzle["train"])
    assert len(signatures) == 42
    for split in ("train", "test"):
        assert len({s for (sp, s) in sides if sp == split}) == 4, sides
        assert len({c for (sp, c) in counts if sp == split}) == 3, counts
    assert len({s for (_, s) in sizes}) >= 6, sizes
    assert held_out >= 3
    assert {k for (_, k) in clip_kinds} >= {"edge", "both_full", "clipped"}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(7, num_train=a, num_test=b))


if __name__ == "__main__":
    self_test()
    print("self_test passed")
