"""Compass-key compartments: a composition of a79310a0 and ef26cbf6.

A divider lattice splits the grid into a key bank and a pattern bank joined by
aligned lanes (parent B).  Each key compartment has odd dimensions and holds one
colored key cell.  The key's color is painted onto the placeholder shape in the
aligned pattern compartment (parent B), and the key's offset from the exact
centre of its own compartment is a one-cell displacement vector: the shape is
rigidly translated by that vector, vacated cells revert to background and the
old/new footprints may overlap (parent A, generalised from "down one" to any of
the eight king moves or no move).  Dividers, keys and background are unchanged.
"""
from __future__ import annotations

import json

import numpy as np

OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
LAYOUTS = [("V", 0), ("V", 1), ("H", 0), ("H", 1)]  # (main divider, key bank index)
REGIMES = ("unseen_direction", "unseen_layout", "scale", "familiar")


# ---------------------------------------------------------------------------
# Input-only solver
# ---------------------------------------------------------------------------

def _segments(lines, length):
    bounds, start = [], 0
    for line in sorted(lines) + [length]:
        if line > start:
            bounds.append((start, line))
        start = line + 1
    return bounds


def _interpretations(grid):
    """Return every structurally valid reading of the lattice in ``grid``."""
    g = np.asarray(grid)
    h, w = g.shape
    rows = [r for r in range(h) if np.all(g[r] == g[r, 0])]
    cols = [c for c in range(w) if np.all(g[:, c] == g[0, c])]
    line_colors = {int(g[r, 0]) for r in rows} | {int(g[0, c]) for c in cols}
    if len(line_colors) != 1 or not rows or not cols:
        return []
    divider = line_colors.pop()
    mask = np.ones_like(g, dtype=bool)
    mask[rows, :] = False
    mask[:, cols] = False
    if np.any(g[mask] == divider):
        return []
    values, counts = np.unique(g[mask], return_counts=True)
    order = np.argsort(-counts)
    if len(order) > 1 and counts[order[0]] == counts[order[1]]:
        return []
    bg = int(values[order[0]])
    rseg, cseg = _segments(rows, h), _segments(cols, w)
    results = []
    for main in ("V", "H"):
        if main == "V":
            if len(cseg) != 2 or len(rseg) < 2:
                continue
            banks = [[(rs, cseg[b]) for rs in rseg] for b in (0, 1)]
        else:
            if len(rseg) != 2 or len(cseg) < 2:
                continue
            banks = [[(rseg[b], cs) for cs in cseg] for b in (0, 1)]
        for key_bank in (0, 1):
            lanes, ok, placeholder, big = [], True, None, False
            for kc, pc in zip(banks[key_bank], banks[1 - key_bank]):
                (kr0, kr1), (kc0, kc1) = kc
                (pr0, pr1), (pc0, pc1) = pc
                kh, kw = kr1 - kr0, kc1 - kc0
                ksub = g[kr0:kr1, kc0:kc1]
                psub = g[pr0:pr1, pc0:pc1]
                kcells = np.argwhere(ksub != bg)
                pcells = np.argwhere(psub != bg)
                if kh % 2 == 0 or kw % 2 == 0 or len(kcells) != 1 or len(pcells) == 0:
                    ok = False
                    break
                pcols = set(int(v) for v in psub[psub != bg])
                if len(pcols) != 1:
                    ok = False
                    break
                pcol = pcols.pop()
                if placeholder is None:
                    placeholder = pcol
                if pcol != placeholder:
                    ok = False
                    break
                kr, kcc = (int(v) for v in kcells[0])
                key_color = int(ksub[kr, kcc])
                if key_color == placeholder:
                    ok = False
                    break
                big = big or len(pcells) >= 2
                lanes.append({
                    "key_box": (kr0, kr1, kc0, kc1), "pat_box": (pr0, pr1, pc0, pc1),
                    "key_pos": (kr0 + kr, kc0 + kcc), "key_color": key_color,
                    "offset": (kr - (kh - 1) // 2, kcc - (kw - 1) // 2),
                    "cells": [(pr0 + int(a), pc0 + int(b)) for a, b in pcells],
                })
            if ok and big and all(lane["key_color"] != placeholder for lane in lanes):
                results.append({"main": main, "key_bank": key_bank, "bg": bg, "divider": divider,
                                "placeholder": placeholder, "lanes": lanes})
    return results


def _render(grid, info, move=True, recolor=True):
    out = np.array(grid, dtype=int).copy()
    bg = info["bg"]
    for lane in info["lanes"]:
        r0, r1, c0, c1 = lane["pat_box"]
        dr, dc = lane["offset"] if move else (0, 0)
        color = lane["key_color"] if recolor else info["placeholder"]
        for r, c in lane["cells"]:
            out[r, c] = bg
        for r, c in lane["cells"]:
            nr, nc = r + dr, c + dc
            if r0 <= nr < r1 and c0 <= nc < c1:
                out[nr, nc] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    readings = _interpretations(grid)
    if not readings:
        return grid.copy()
    return _render(grid, readings[0])


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def _crop(mask):
    rr, cc = np.nonzero(mask)
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]


def _grow(rng, h, w, size):
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    tries = 0
    while len(cells) < size and tries < 400:
        tries += 1
        r, c = list(cells)[int(rng.integers(len(cells)))]
        dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][int(rng.integers(4))]
        if 0 <= r + dr < h and 0 <= c + dc < w:
            cells.add((r + dr, c + dc))
    m = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        m[r, c] = True
    return _crop(m)


def _shape(rng, h, w, kind):
    """Return a cropped boolean mask that fits inside h x w (h, w >= 1)."""
    if kind == "single" or h * w == 1:
        return np.ones((1, 1), dtype=bool)
    if kind == "line":
        if w >= 2 and (h < 2 or rng.random() < 0.5):
            return np.ones((1, int(rng.integers(2, w + 1))), dtype=bool)
        return np.ones((int(rng.integers(2, h + 1)), 1), dtype=bool)
    if kind == "rect":
        a = int(rng.integers(1, h + 1))
        b = int(rng.integers(1, w + 1))
        if a * b == 1:
            b = min(2, w) if w >= 2 else 1
            a = 2 if b == 1 else a
        return np.ones((a, b), dtype=bool)
    if kind == "ring" and h >= 3 and w >= 3:
        a, b = int(rng.integers(3, h + 1)), int(rng.integers(3, w + 1))
        m = np.ones((a, b), dtype=bool)
        m[1:-1, 1:-1] = False
        return m
    if kind in ("L", "T") and h >= 2 and w >= 2:
        a, b = int(rng.integers(2, h + 1)), int(rng.integers(2, w + 1))
        m = np.zeros((a, b), dtype=bool)
        if kind == "L":
            m[:, 0] = True
            m[-1, :] = True
        else:
            m[0, :] = True
            m[:, b // 2] = True
        k = int(rng.integers(4))
        m = np.rot90(m, k)
        if m.shape[0] > h or m.shape[1] > w:
            m = np.rot90(m, 1)
        if m.shape[0] > h or m.shape[1] > w:
            m = np.rot90(m, -1)
            m = m[:h, :w]
        if rng.random() < 0.5:
            m = m[:, ::-1]
        return _crop(m)
    if kind == "sparse" and h * w >= 4:
        for _ in range(30):
            m = rng.random((h, w)) < rng.uniform(0.3, 0.6)
            if m.sum() >= 2:
                return _crop(m)
    size = int(rng.integers(2, max(3, int(h * w * 0.7)) + 1))
    return _grow(rng, h, w, min(size, h * w))


SHAPE_KINDS = ["blob", "blob", "line", "rect", "ring", "L", "T", "sparse", "single"]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _roles(rng, reserved):
    pool = [c for c in range(10) if c != reserved]
    bg = 0 if rng.random() < 0.4 and 0 in pool else int(rng.choice(pool))
    rest = [c for c in pool if c != bg]
    div, ph = (int(v) for v in rng.choice(rest, size=2, replace=False))
    return bg, div, ph


def _make_pair(rng, layout, roles, reserved, forced, n_lanes, big, allow_reserved):
    """Build one pair in the canonical V frame, then transpose for H layouts."""
    main, key_bank = layout
    bg, div, ph = roles
    key_pool = [c for c in range(10) if c not in (bg, div, ph) and (allow_reserved or c != reserved)]
    span_choices = [3, 5, 7] if big else [3, 3, 5, 5, 7]
    max_len = 30
    spans = []
    for _ in range(n_lanes):
        spans.append(int(rng.choice(span_choices)))
    while sum(spans) + n_lanes - 1 > max_len:
        i = int(np.argmax(spans))
        spans[i] -= 2
    kd = int(rng.choice([3, 3, 5]))
    pd = int(rng.integers(5, 12)) if big else int(rng.integers(3, 9))
    # offsets
    offsets = list(forced) + [OFFSETS[int(rng.integers(9))] for _ in range(n_lanes - len(forced))]
    rng.shuffle(offsets)
    if all(o == (0, 0) for o in offsets):
        nz = [o for o in OFFSETS if o != (0, 0)]
        idx = [i for i in range(n_lanes)]
        offsets[int(rng.choice(idx))] = nz[int(rng.integers(len(nz)))]
    # key colors: at least two distinct, repeats allowed
    if allow_reserved and reserved is not None and reserved in key_pool:
        colors = [int(rng.choice(key_pool)) for _ in range(n_lanes)]
        colors[int(rng.integers(n_lanes))] = reserved
    else:
        colors = [int(rng.choice(key_pool)) for _ in range(n_lanes)]
    if len(set(colors)) < 2:
        others = [c for c in key_pool if c != colors[0]]
        colors[int(rng.integers(1, n_lanes))] = int(rng.choice(others))
    H = sum(spans) + n_lanes - 1
    W = kd + 1 + pd
    g = np.full((H, W), bg, dtype=int)
    if key_bank == 0:
        kcol0, dcol, pcol0 = 0, kd, kd + 1
    else:
        pcol0, dcol, kcol0 = 0, pd, pd + 1
    g[:, dcol] = div
    r = 0
    multi = False
    shapes_info = []
    for i in range(n_lanes):
        span = spans[i]
        if i > 0:
            g[r - 1, :] = div
        dr, dc = offsets[i]
        # key: lane span and kd are odd; offset is inside the central 3x3
        g[r + (span - 1) // 2 + dr, kcol0 + (kd - 1) // 2 + dc] = colors[i]
        ah, aw = span - abs(dr), pd - abs(dc)
        kind = SHAPE_KINDS[int(rng.integers(len(SHAPE_KINDS)))]
        if i == n_lanes - 1 and not multi:
            kind = "blob"
        m = _shape(rng, ah, aw, kind)
        if m.sum() >= 2:
            multi = True
        mh, mw = m.shape
        rlo, rhi = max(0, -dr), span - mh - max(0, dr)
        clo, chi = max(0, -dc), pd - mw - max(0, dc)
        pr = int(rng.integers(rlo, rhi + 1))
        pc = int(rng.integers(clo, chi + 1))
        sub = g[r + pr:r + pr + mh, pcol0 + pc:pcol0 + pc + mw]
        sub[m] = ph
        shapes_info.append((m, (dr, dc)))
        r += span + 1
    if main == "H":
        g = g.T.copy()
    return g, offsets, shapes_info


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _pair_facts(inp):
    readings = _interpretations(inp)
    _check(len(readings) == 1, "input lattice reading is not unique")
    info = readings[0]
    offs = [lane["offset"] for lane in info["lanes"]]
    overlap = clear_only = False
    for lane in info["lanes"]:
        dr, dc = lane["offset"]
        if (dr, dc) == (0, 0):
            continue
        cells = set(lane["cells"])
        moved = {(r + dr, c + dc) for r, c in cells}
        if cells & moved:
            overlap = True
        else:
            clear_only = True
    return info, offs, overlap, clear_only


def _layout_of(info):
    return (info["main"], info["key_bank"])


def _category(o):
    if o == (0, 0):
        return "zero"
    if o[1] == 0:
        return "vertical"
    if o[0] == 0:
        return "horizontal"
    return "diagonal"


def _counts(num_train, num_test):
    for name, v in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("need 2<=num_train<=10, 1<=num_test<=4 and num_train>num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _attempt in range(200):
        puzzle = _attempt_puzzle(rng, num_train, num_test)
        if puzzle is None:
            continue
        try:
            validate_generated_puzzle(puzzle)
        except ValueError:
            continue
        return puzzle
    raise RuntimeError("generation failed")


def _attempt_puzzle(rng, num_train, num_test):
    regime = REGIMES[int(rng.integers(len(REGIMES)))]
    local_palette = rng.random() < 0.4
    reserved = int(rng.integers(10)) if rng.random() < 0.5 else None
    base_roles = _roles(rng, reserved)

    def roles_for(i):
        if local_palette and i > 0:
            return _roles(rng, reserved)
        return base_roles

    # training layouts: first two differ
    layouts = [LAYOUTS[int(i)] for i in rng.permutation(4)[:2]]
    layouts += [LAYOUTS[int(rng.integers(4))] for _ in range(num_train - 2)]
    if regime == "unseen_layout":
        hidden = LAYOUTS[int(rng.integers(4))]
        for i, lay in enumerate(layouts):
            if lay == hidden:
                choices = [l for l in LAYOUTS if l != hidden]
                layouts[i] = choices[int(rng.integers(3))]
        if len(set(layouts[:2])) < 2:
            return None
    # forced offset categories spread over training
    cats = {
        "zero": [(0, 0)],
        "vertical": [(-1, 0), (1, 0)],
        "horizontal": [(0, -1), (0, 1)],
        "diagonal": [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    }
    forced = [[] for _ in range(num_train)]
    for name in ("zero", "vertical", "horizontal", "diagonal"):
        opts = cats[name]
        forced[int(rng.integers(num_train))].append(opts[int(rng.integers(len(opts)))])
    hidden_dir = None
    if regime == "unseen_direction":
        nz = [o for o in OFFSETS if o != (0, 0)]
        hidden_dir = nz[int(rng.integers(len(nz)))]
        if any(hidden_dir in f for f in forced):
            return None
    train = []
    seen = set()
    max_train_lanes, max_train_cells = 0, 0
    for i in range(num_train):
        need = max(2, len(forced[i]))
        n = int(rng.integers(need, 6))
        if regime == "scale":
            n = int(rng.integers(need, max(need, 4) + 1))
        pair = None
        for _ in range(30):
            g, offs, _ = _make_pair(rng, layouts[i], roles_for(i), reserved, forced[i], n,
                                    big=False, allow_reserved=False)
            if hidden_dir is not None and hidden_dir in offs:
                continue
            key = g.tobytes() + bytes(g.shape)
            if key in seen:
                continue
            pair = g
            break
        if pair is None:
            return None
        seen.add(pair.tobytes() + bytes(pair.shape))
        max_train_lanes = max(max_train_lanes, n)
        max_train_cells = max(max_train_cells, pair.size)
        train.append({"input": pair, "output": solve(pair)})
    test = []
    for i in range(num_test):
        pair = None
        for _ in range(40):
            if regime == "unseen_layout":
                lay = hidden
            elif regime == "familiar":
                lay = layouts[int(rng.integers(num_train))]
            else:
                lay = LAYOUTS[int(rng.integers(4))]
            big = regime == "scale"
            n = int(rng.integers(max(2, max_train_lanes + 1 if big else 2), (7 if big else 6)))
            n = min(n, 7)
            forced_t = [hidden_dir] if hidden_dir is not None else []
            g, offs, _ = _make_pair(rng, lay, roles_for(num_train + i) if not local_palette else _roles(rng, reserved),
                                    reserved, forced_t, n, big=big,
                                    allow_reserved=reserved is not None and i == 0)
            if regime == "scale" and g.size <= max_train_cells:
                continue
            if regime == "familiar" and not set(offs) <= {o for f in forced for o in f} | set(
                    lane["offset"] for p in train for lane in _interpretations(p["input"])[0]["lanes"]):
                continue
            key = g.tobytes() + bytes(g.shape)
            if key in seen:
                continue
            pair = g
            break
        if pair is None:
            return None
        seen.add(pair.tobytes() + bytes(pair.shape))
        test.append({"input": pair, "output": solve(pair)})
    return {"train": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in train],
            "test": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in test]}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_generated_puzzle(puzzle) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4, "bad counts")
    _check(len(puzzle["train"]) > len(puzzle["test"]), "num_train must exceed num_test")
    seen = set()
    train_offsets, train_layouts = set(), set()
    overlap_seen = clear_seen = False
    train_key_colors = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            _check(inp.ndim == 2 and 1 <= inp.shape[0] <= 30 and 1 <= inp.shape[1] <= 30, "bad size")
            _check(inp.shape == out.shape, "output shape differs")
            _check(inp.min() >= 0 and inp.max() <= 9 and out.min() >= 0 and out.max() <= 9, "bad colors")
            key = json.dumps(inp.tolist())
            _check(key not in seen, "duplicate input")
            seen.add(key)
            info, offs, overlap, clear_only = _pair_facts(inp)
            _check(len(info["lanes"]) >= 2, "need at least two lanes")
            _check(all(abs(a) <= 1 and abs(b) <= 1 for a, b in offs), "key offset beyond one cell")
            _check(any(o != (0, 0) for o in offs), "pair has no moving shape")
            _check(len({l["key_color"] for l in info["lanes"]}) >= 2, "need two key colors")
            roles = {info["bg"], info["divider"], info["placeholder"]}
            _check(len(roles) == 3, "roles collide")
            _check(all(l["key_color"] not in roles for l in info["lanes"]), "key color reuses a role")
            for lane in info["lanes"]:
                r0, r1, c0, c1 = lane["pat_box"]
                dr, dc = lane["offset"]
                for r, c in lane["cells"]:
                    _check(r0 <= r + dr < r1 and c0 <= c + dc < c1, "shifted shape leaves compartment")
            expected = solve(inp)
            _check(np.array_equal(expected, out), "output disagrees with rule")
            # shortcuts that ignore a parent must give different answers
            _check(not np.array_equal(_render(inp, info, move=False), out), "recolor-only shortcut matches")
            _check(not np.array_equal(_render(inp, info, recolor=False), out), "move-only shortcut matches")
            if split == "train":
                train_offsets.update(offs)
                train_layouts.add(_layout_of(info))
                overlap_seen |= overlap
                clear_seen |= clear_only or overlap
                train_key_colors.update(l["key_color"] for l in info["lanes"])
    cats = {_category(o) for o in train_offsets}
    _check(cats == {"zero", "vertical", "horizontal", "diagonal"}, "training misses an offset category")
    _check(len(train_layouts) >= 2, "training must show two layouts")
    _check(overlap_seen and clear_seen, "training must show overlap and clearing")
    # a uniform "move everything down one" shortcut must fail on training
    _check(train_offsets != {(1, 0)}, "down-only shortcut")


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def self_test() -> None:
    layouts, offsets, sizes, regimes_seen, palettes = set(), set(), set(), set(), set()
    lane_counts, sigs = set(), set()
    unseen_dir = unseen_layout = test_only_color = local_palette = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        sigs.add(json.dumps(p))
        tr_offs, tr_lays, tr_colors, tr_roles = set(), set(), set(), set()
        for pair in p["train"]:
            info = _interpretations(np.asarray(pair["input"]))[0]
            tr_offs.update(l["offset"] for l in info["lanes"])
            tr_lays.add(_layout_of(info))
            tr_colors.update(l["key_color"] for l in info["lanes"])
            tr_roles.add((info["bg"], info["divider"], info["placeholder"]))
        local_palette += len(tr_roles) > 1
        for split in ("train", "test"):
            for pair in p[split]:
                g = np.asarray(pair["input"])
                info = _interpretations(g)[0]
                layouts.add(_layout_of(info))
                offsets.update(l["offset"] for l in info["lanes"])
                sizes.add(g.shape)
                lane_counts.add(len(info["lanes"]))
                palettes.add((info["bg"], info["divider"], info["placeholder"]))
                if split == "test":
                    offs = {l["offset"] for l in info["lanes"]}
                    unseen_dir += bool(offs - tr_offs)
                    unseen_layout += _layout_of(info) not in tr_lays
                    test_only_color += bool({l["key_color"] for l in info["lanes"]} - tr_colors)
    assert len(sigs) == 42
    assert layouts == set(LAYOUTS), layouts
    assert set(offsets) == set(OFFSETS), offsets
    assert len(sizes) >= 40, len(sizes)
    assert lane_counts >= {2, 3, 4, 5}, lane_counts
    assert len(palettes) >= 20, len(palettes)
    assert unseen_dir >= 5 and unseen_layout >= 5 and test_only_color >= 5 and local_palette >= 5, (
        unseen_dir, unseen_layout, test_only_color, local_palette)
    # determinism and counts
    assert json.dumps(generate_puzzle(7, 3, 2)) == json.dumps(generate_puzzle(7, 3, 2))
    for a, b in ((2, 1), (10, 4), (5, 4)):
        q = generate_puzzle(11, a, b)
        assert len(q["train"]) == a and len(q["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test ok")
