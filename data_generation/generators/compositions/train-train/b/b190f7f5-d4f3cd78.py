"""Leaking stencil mosaic.

Composition of b190f7f5 (stencil x colored-layout block substitution) and
d4f3cd78 (gapped frame: fill the cavity, leak a ray through the gap).

Input: two equal h x w panels (3 <= h, w <= 5) joined horizontally or
vertically, in either order.  One panel (the stencil) contains only the
background and one marker color forming a one-cell-thick rectangular frame with
exactly one missing non-corner boundary cell (the aperture).  The other panel
(the layout) uses the background plus payload colors, never the marker.

Output: an (h*h) x (w*w) block matrix.  Each background layout cell becomes an
all-background block.  Each layout cell of color p becomes a copy of the
stencil whose frame keeps the marker color while its interior and aperture are
filled with p.  After all blocks are placed, every filled block leaks a ray of
its color out through its aperture, perpendicular to the aperture side,
across background cells until the canvas edge or the first non-background cell
(the frame of the next copied block in that direction).
"""
from __future__ import annotations

import numpy as np

SIDES = ("top", "bottom", "left", "right")
DIRS = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
MODES = ("unseen_side", "unseen_arrangement", "larger_scale", "mixed")


# ----------------------------------------------------------------------------
# Shared analysis (used by solve and validation)
# ----------------------------------------------------------------------------

def _frame_of(panel):
    """Return (bg, marker, top, left, bottom, right, gap_r, gap_c, side) if the
    panel is exactly a background plus one gapped rectangular frame, else None."""
    colors = np.unique(panel)
    if len(colors) != 2:
        return None
    found = []
    for marker in colors:
        bg = int(colors[0] if colors[1] == marker else colors[1])
        rr, cc = np.nonzero(panel == marker)
        top, bottom, left, right = rr.min(), rr.max(), cc.min(), cc.max()
        if bottom - top < 2 or right - left < 2:
            continue
        ring = np.zeros(panel.shape, dtype=bool)
        ring[top, left:right + 1] = True
        ring[bottom, left:right + 1] = True
        ring[top:bottom + 1, left] = True
        ring[top:bottom + 1, right] = True
        cells = panel == marker
        if np.any(cells & ~ring):
            continue
        missing = np.argwhere(ring & ~cells)
        if len(missing) != 1:
            continue
        gr, gc = (int(v) for v in missing[0])
        if (gr in (top, bottom)) and (gc in (left, right)):
            continue  # corner gap is not an aperture
        if gr == top:
            side = "top"
        elif gr == bottom:
            side = "bottom"
        elif gc == left:
            side = "left"
        else:
            side = "right"
        found.append((bg, int(marker), int(top), int(left), int(bottom), int(right), gr, gc, side))
    if len(found) != 1:
        return None
    return found[0]


def _split(grid):
    """Split into (panel_a, panel_b, axis).  The panel-size limits make the
    split axis unique."""
    H, W = grid.shape
    if 3 <= H <= 5 and W % 2 == 0 and 3 <= W // 2 <= 5:
        return grid[:, : W // 2], grid[:, W // 2:], "horizontal"
    if 3 <= W <= 5 and H % 2 == 0 and 3 <= H // 2 <= 5:
        return grid[: H // 2, :], grid[H // 2:, :], "vertical"
    raise ValueError("input is not two equal 3..5 panels")


def _decode(grid):
    a, b, axis = _split(np.asarray(grid, dtype=int))
    fa, fb = _frame_of(a), _frame_of(b)
    if (fa is None) == (fb is None):
        raise ValueError("exactly one panel must be a gapped frame stencil")
    if fa is not None:
        return a, fa, b, axis, 0
    return b, fb, a, axis, 1


def _render(stencil_info, layout, h, w):
    bg, marker, top, left, bottom, right, gr, gc, side = stencil_info
    out = np.full((h * h, w * w), bg, dtype=int)
    rays = []
    for R in range(h):
        for C in range(w):
            p = int(layout[R, C])
            if p == bg:
                continue
            r0, c0 = R * h, C * w
            out[r0 + top:r0 + bottom + 1, c0 + left:c0 + right + 1] = marker
            out[r0 + top + 1:r0 + bottom, c0 + left + 1:c0 + right] = p
            out[r0 + gr, c0 + gc] = p
            rays.append((r0 + gr, c0 + gc, p))
    dr, dc = DIRS[side]
    for r, c, p in rays:
        r, c = r + dr, c + dc
        while 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and out[r, c] == bg:
            out[r, c] = p
            r, c = r + dr, c + dc
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    stencil, info, layout, _axis, _pos = _decode(grid)
    h, w = stencil.shape
    if np.any(layout == info[1]):
        raise ValueError("layout must not contain the stencil marker")
    return _render(info, layout, h, w)


# ----------------------------------------------------------------------------
# Ray statistics (for demonstration guarantees)
# ----------------------------------------------------------------------------

def _ray_stats(layout, bg, side):
    """For every colored layout cell, count empty layout cells crossed before
    the next colored cell in the ray direction, and whether it is blocked."""
    dr, dc = DIRS[side]
    h, w = layout.shape
    stats = []
    for R in range(h):
        for C in range(w):
            if layout[R, C] == bg:
                continue
            r, c, empties, blocked = R + dr, C + dc, 0, False
            while 0 <= r < h and 0 <= c < w:
                if layout[r, c] != bg:
                    blocked = True
                    break
                empties += 1
                r, c = r + dr, c + dc
            stats.append((empties, blocked))
    return stats


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _resolve_counts(num_train, num_test):
    if isinstance(num_train, bool) or isinstance(num_test, bool):
        raise ValueError("counts must be integers")
    if not isinstance(num_train, (int, np.integer)) or not isinstance(num_test, (int, np.integer)):
        raise ValueError("counts must be integers")
    if not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("illegal train/test counts")
    return int(num_train), int(num_test)


def _make_stencil(rng, h, w, side, bg, marker):
    for _ in range(200):
        fh = int(rng.integers(3, h + 1))
        fw = int(rng.integers(3, w + 1))
        top = int(rng.integers(0, h - fh + 1))
        left = int(rng.integers(0, w - fw + 1))
        bottom, right = top + fh - 1, left + fw - 1
        if side == "top":
            gr, gc = top, int(rng.integers(left + 1, right))
        elif side == "bottom":
            gr, gc = bottom, int(rng.integers(left + 1, right))
        elif side == "left":
            gr, gc = int(rng.integers(top + 1, bottom)), left
        else:
            gr, gc = int(rng.integers(top + 1, bottom)), right
        panel = np.full((h, w), bg, dtype=int)
        panel[top, left:right + 1] = marker
        panel[bottom, left:right + 1] = marker
        panel[top:bottom + 1, left] = marker
        panel[top:bottom + 1, right] = marker
        panel[gr, gc] = bg
        info = _frame_of(panel)
        if info is not None and info[1] == marker and info[8] == side:
            return panel
    raise RuntimeError("could not build stencil")


def _make_layout(rng, h, w, bg, payload, side, need_cross_block):
    for _ in range(400):
        k = int(rng.integers(1, min(3, len(payload)) + 1))
        colors = [int(c) for c in rng.choice(payload, size=k, replace=False)]
        density = float(rng.uniform(0.3, 0.7))
        mask = rng.random((h, w)) < density
        layout = np.full((h, w), bg, dtype=int)
        idx = np.argwhere(mask)
        if len(idx) < 2 or len(idx) > h * w - 2:
            continue
        for (r, c) in idx:
            layout[r, c] = colors[int(rng.integers(0, k))]
        present = set(np.unique(layout).tolist()) - {bg}
        if present != set(colors):
            continue
        stats = _ray_stats(layout, bg, side)
        if not any(b for _, b in stats):
            continue
        if not any(e >= 1 for e, _ in stats):
            continue
        if need_cross_block and not any(e >= 1 and b for e, b in stats):
            continue
        if _frame_of(layout) is not None:
            continue
        return layout
    return None


def _assemble(stencil, layout, axis, stencil_first):
    parts = [stencil, layout] if stencil_first else [layout, stencil]
    return np.concatenate(parts, axis=1 if axis == "horizontal" else 0)


def _pair(rng, h, w, side, axis, stencil_first, bg, marker, payload, need_cross_block):
    for _ in range(50):
        stencil = _make_stencil(rng, h, w, side, bg, marker)
        layout = _make_layout(rng, h, w, bg, payload, side, need_cross_block)
        if layout is None:
            continue
        grid = _assemble(stencil, layout, axis, stencil_first)
        return {"input": grid, "output": solve(grid)}
    raise RuntimeError("could not build pair")


def _plan(rng, mode, num_train, num_test):
    """Per-pair (h, w, side, axis, stencil_first) plans implementing the shift."""
    dims_all = [3, 4, 5]
    train_axis = str(rng.choice(["horizontal", "vertical"]))
    plans_train, plans_test = [], []
    held_side = str(rng.choice(SIDES))
    train_sides_pool = [s for s in SIDES if s != held_side] if mode == "unseen_side" else list(SIDES)
    # Training sides: at least two different sides.
    order = list(rng.permutation(train_sides_pool))
    train_sides = [str(order[i % len(order)]) if i < len(order) else str(rng.choice(train_sides_pool))
                   for i in range(num_train)]
    train_sides = [str(s) for s in rng.permutation(train_sides)]
    for i in range(num_train):
        if mode == "larger_scale":
            h, w = int(rng.choice([3, 4])), int(rng.choice([3, 4]))
        else:
            h, w = int(rng.choice(dims_all)), int(rng.choice(dims_all))
        if mode == "unseen_arrangement":
            axis = train_axis
        else:
            axis = str(rng.choice(["horizontal", "vertical"]))
        plans_train.append((h, w, train_sides[i], axis, bool(rng.integers(0, 2))))
    for j in range(num_test):
        if mode == "larger_scale":
            h, w = int(rng.choice(dims_all)), int(rng.choice(dims_all))
            if max(h, w) < 5:
                if rng.random() < 0.5:
                    h = 5
                else:
                    w = 5
        else:
            h, w = int(rng.choice(dims_all)), int(rng.choice(dims_all))
        if mode == "unseen_side":
            side = held_side
        else:
            side = str(rng.choice(SIDES))
        if mode == "unseen_arrangement":
            axis = "vertical" if train_axis == "horizontal" else "horizontal"
        else:
            axis = str(rng.choice(["horizontal", "vertical"]))
        plans_test.append((h, w, side, axis, bool(rng.integers(0, 2))))
    return plans_train, plans_test


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _resolve_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(MODES))
    bg = 0 if rng.random() < 0.7 else int(rng.integers(1, 10))
    marker = int(rng.choice([c for c in range(10) if c != bg]))
    payload = np.array([c for c in range(10) if c not in (bg, marker)])
    plans_train, plans_test = _plan(rng, mode, num_train, num_test)
    seen = set()
    train, test = [], []
    cross_block_index = int(rng.integers(0, num_train))
    for split, plans, out in (("train", plans_train, train), ("test", plans_test, test)):
        for i, (h, w, side, axis, first) in enumerate(plans):
            need = split == "train" and i == cross_block_index
            for _ in range(100):
                pair = _pair(rng, h, w, side, axis, first, bg, marker, payload, need)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise RuntimeError("could not build distinct pair")
            out.append({"input": pair["input"].tolist(), "output": pair["output"].tolist()})
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


# ----------------------------------------------------------------------------
# Validation and self-test
# ----------------------------------------------------------------------------

def _pair_facts(pair):
    grid = np.asarray(pair["input"], dtype=int)
    out = np.asarray(pair["output"], dtype=int)
    stencil, info, layout, axis, pos = _decode(grid)
    bg, marker = info[0], info[1]
    h, w = stencil.shape
    if out.shape != (h * h, w * w):
        raise AssertionError("output shape must be (h*h, w*w)")
    if np.any(layout == marker):
        raise AssertionError("layout contains the marker color")
    if not np.array_equal(solve(grid), out):
        raise AssertionError("solve disagrees with stored output")
    stats = _ray_stats(layout, bg, info[8])
    return {"bg": bg, "marker": marker, "side": info[8], "axis": axis, "pos": pos,
            "shape": (h, w), "stats": stats,
            "payload": sorted(set(np.unique(layout).tolist()) - {bg})}


def _shortcut_outputs(grid):
    """Outputs of simpler alternative rules, for distinguishing checks."""
    stencil, info, layout, _a, _p = _decode(np.asarray(grid, dtype=int))
    bg, marker, top, left, bottom, right, gr, gc, side = info
    h, w = stencil.shape
    plain = np.full((h * h, w * w), bg, dtype=int)  # parent A alone
    for R in range(h):
        for C in range(w):
            if layout[R, C] != bg:
                blk = plain[R * h:(R + 1) * h, C * w:(C + 1) * w]
                blk[stencil == marker] = layout[R, C]
    # Rays that ignore obstacles (overpaint only background) to the canvas edge.
    full = _render(info, layout, h, w)
    no_block = full.copy()
    dr, dc = DIRS[side]
    for R in range(h):
        for C in range(w):
            p = int(layout[R, C])
            if p == bg:
                continue
            r, c = R * h + gr + dr, C * w + gc + dc
            while 0 <= r < h * h and 0 <= c < w * w:
                no_block[r, c] = p
                r, c = r + dr, c + dc
    # Fill only, no ray at all.
    fill_only = np.full((h * h, w * w), bg, dtype=int)
    for R in range(h):
        for C in range(w):
            p = int(layout[R, C])
            if p == bg:
                continue
            r0, c0 = R * h, C * w
            fill_only[r0 + top:r0 + bottom + 1, c0 + left:c0 + right + 1] = marker
            fill_only[r0 + top + 1:r0 + bottom, c0 + left + 1:c0 + right] = p
            fill_only[r0 + gr, c0 + gc] = p
    # Rays confined to the block that owns them (no travel across blocks).
    local = fill_only.copy()
    for R in range(h):
        for C in range(w):
            p = int(layout[R, C])
            if p == bg:
                continue
            blk = local[R * h:(R + 1) * h, C * w:(C + 1) * w]
            r, c = gr + dr, gc + dc
            while 0 <= r < h and 0 <= c < w and blk[r, c] == bg:
                blk[r, c] = p
                r, c = r + dr, c + dc
    return {"parent_a_only": plain, "rays_ignore_blocks": no_block,
            "rays_local_to_block": local, "fill_without_ray": fill_only}


def validate_generated_puzzle(puzzle) -> None:
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("puzzle must have exactly train/test")
    train, test = puzzle["train"], puzzle["test"]
    if not (2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)):
        raise AssertionError("illegal counts")
    seen = set()
    facts = []
    for pair in train + test:
        if set(pair) != {"input", "output"}:
            raise AssertionError("pair keys")
        key = np.asarray(pair["input"], dtype=int)
        k = (key.shape, key.tobytes())
        if k in seen:
            raise AssertionError("duplicate input")
        seen.add(k)
        facts.append(_pair_facts(pair))
    if len({(f["bg"], f["marker"]) for f in facts}) != 1:
        raise AssertionError("background/marker roles must be fixed per puzzle")
    tf = facts[:len(train)]
    if len({f["side"] for f in tf}) < 2:
        raise AssertionError("training must show at least two aperture sides")
    for f, pair in zip(facts, train + test):
        if not any(b for _, b in f["stats"]):
            raise AssertionError("each pair needs a blocked ray")
        if not any(e >= 1 for e, _ in f["stats"]):
            raise AssertionError("each pair needs a ray crossing an empty block")
        out = np.asarray(pair["output"], dtype=int)
        for name, alt in _shortcut_outputs(pair["input"]).items():
            if np.array_equal(alt, out):
                raise AssertionError(f"shortcut {name} reproduces a pair")
    if not any(e >= 1 and b for f in tf for e, b in f["stats"]):
        raise AssertionError("training needs a ray that crosses an empty block and is then blocked")


def self_test() -> None:
    sides, shapes, axes, positions, modes_seen, bgs, ncolors = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert puzzle == generate_puzzle(seed)
        modes_seen.add(str(np.random.default_rng(seed).choice(MODES)))
        for pair in puzzle["train"] + puzzle["test"]:
            f = _pair_facts(pair)
            sides.add(f["side"])
            shapes.add(f["shape"])
            axes.add(f["axis"])
            positions.add(f["pos"])
            bgs.add(f["bg"])
            ncolors.add(len(f["payload"]))
    assert sides == set(SIDES), sides
    assert len(shapes) >= 7, shapes
    assert axes == {"horizontal", "vertical"} and positions == {0, 1}
    assert modes_seen == set(MODES), modes_seen
    assert len(bgs) >= 2 and ncolors >= {1, 2, 3}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(7, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test ok")
