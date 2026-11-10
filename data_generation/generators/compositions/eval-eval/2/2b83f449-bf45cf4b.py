"""Switch-tile loom: a blueprint assembles a globally interacting pipe network.

Only NumPy and the standard library are used. No parent module is imported.
The solver reads one input and uses no puzzle-specific learned state.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _box(binary):
    loc = np.argwhere(binary)
    if len(loc) == 0:
        raise ValueError("empty encoding")
    r, c = loc.min(axis=0)
    rr, cc = loc.max(axis=0) + 1
    return int(r), int(rr), int(c), int(cc)


def _crop(a, box):
    r, rr, c, cc = box
    return a[r:rr, c:cc].copy()


def _separated(a, b):
    # Exclusive upper bounds: '<' leaves at least one blank line.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _template(h, w, y, x, bg, track, divider):
    """Canonical four-port tile; arms leave clearance around the switch."""
    if not (7 <= h <= 10 and 7 <= w <= 10
            and 3 <= y <= h - 4 and 3 <= x <= w - 4):
        raise ValueError("tile dimensions or switch center outside grammar")
    a = np.full((h, w), bg, dtype=np.int8)
    a[y - 1, x - 1:x + 2] = track
    a[y + 1, x - 1:x + 2] = track
    a[y, x - 1:x + 2] = divider
    a[:y, x - 1] = track
    a[0, x] = track
    a[y - 1, x + 1:] = track
    a[y, -1] = track
    a[y + 1, :x] = track
    a[y, 0] = track
    a[y + 1:, x + 1] = track
    a[-1, x] = track
    return a


def _symmetries(a):
    return [np.rot90(a if not mirror else np.fliplr(a), k)
            for mirror in (False, True) for k in range(4)]


def _tile_info(stamp, bg):
    colors, counts = np.unique(stamp[stamp != bg], return_counts=True)
    if len(colors) != 3:
        raise ValueError("tile must have track, divider and bead colors")
    track = int(colors[np.argmax(counts)])
    possibilities = []
    for divider in map(int, colors):
        if divider == track:
            continue
        bead = next(iter(set(map(int, colors)) - {track, divider}))
        if not 1 <= np.count_nonzero(stamp == bead) <= 3:
            continue
        loc = np.argwhere(stamp == divider)
        if len(loc) != 3:
            continue
        y, x = map(int, loc.mean(axis=0))
        horizontal = np.all(loc[:, 0] == y) and set(loc[:, 1]) == {x - 1, x, x + 1}
        vertical = np.all(loc[:, 1] == x) and set(loc[:, 0]) == {y - 1, y, y + 1}
        if not (horizontal or vertical):
            continue
        if not (1 <= y < stamp.shape[0] - 1 and 1 <= x < stamp.shape[1] - 1):
            continue
        if np.any(stamp[y - 1:y + 2, x - 1:x + 2] == bead):
            continue
        plain = stamp.copy()
        plain[plain == bead] = track
        matched = False
        for t in _symmetries(plain):
            d = np.argwhere(t == divider)
            cy, cx = map(int, d.mean(axis=0))
            if not np.all(d[:, 0] == cy):
                continue
            try:
                candidate = _template(*t.shape, cy, cx, bg, track, divider)
            except ValueError:
                continue
            if np.array_equal(candidate, t):
                matched = True
                break
        if matched:
            possibilities.append({"track": track, "divider": divider, "bead": bead,
                                  "center": (y, x), "horizontal": bool(horizontal)})
    if len(possibilities) != 1:
        raise ValueError("tile roles and grammar must have one reading")
    return possibilities[0]


def _read_input(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)
            and np.count_nonzero(a == bg) > a.size / 2):
        raise ValueError("input needs an evident uniform background rim")
    colors = set(map(int, np.unique(a)))
    if len(colors) != 5:
        raise ValueError("input must have exactly five color roles")
    candidates = []
    for signal in sorted(colors - {bg}):
        mb = _box(a == signal)
        sb = _box((a != bg) & (a != signal))
        if not _separated(mb, sb):
            continue
        mask_crop, stamp = _crop(a, mb), _crop(a, sb)
        if not set(map(int, np.unique(mask_crop))) <= {bg, signal} or signal in stamp:
            continue
        mask = mask_crop == signal
        if not (all(1 <= n <= 4 for n in mask.shape)
                and all(m * s <= 30 for m, s in zip(mask.shape, stamp.shape))):
            continue
        try:
            tile = _tile_info(stamp, bg)
        except ValueError:
            continue
        candidates.append(dict(tile, background=bg, signal=signal, mask=mask,
                               stamp=stamp, mask_box=mb, stamp_box=sb))
    if len(candidates) != 1:
        raise ValueError("encodings must have a unique visible interpretation")
    return candidates[0]


def _expand(mask, tile, bg):
    sh, sw = tile.shape
    result = np.full((mask.shape[0] * sh, mask.shape[1] * sw), bg, dtype=np.int8)
    for r, c in np.argwhere(mask):
        result[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = tile
    return result


def _turn_tile(stamp, info):
    result = stamp.copy()
    result[result == info["bead"]] = info["track"]
    y, x = info["center"]
    result[y - 1:y + 2, x - 1:x + 2] = info["track"]
    if info["horizontal"]:
        result[y - 1:y + 2, x] = info["divider"]
    else:
        result[y, x - 1:x + 2] = info["divider"]
    return result


def _paths(binary):
    remaining = set(map(tuple, np.argwhere(binary)))
    paths = []
    while remaining:
        start = min(remaining)
        component, pending = set(), [start]
        while pending:
            cell = pending.pop()
            if cell in component:
                continue
            component.add(cell)
            r, c = cell
            pending.extend((r + dr, c + dc) for dr, dc in _DIRS
                           if (r + dr, c + dc) in remaining
                           and (r + dr, c + dc) not in component)
        adjacency = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRS
                         if (p[0] + dr, p[1] + dc) in component] for p in component}
        ends = sorted(p for p, ns in adjacency.items() if len(ns) == 1)
        if len(ends) != 2 or any(len(ns) not in (1, 2) for ns in adjacency.values()):
            raise ValueError("network must consist of simple paths")
        ordered, previous = [ends[0]], None
        while ordered[-1] != ends[1]:
            choices = [p for p in adjacency[ordered[-1]] if p != previous]
            if len(choices) != 1:
                raise ValueError("ambiguous path")
            previous = ordered[-1]
            ordered.append(choices[0])
        if len(ordered) != len(component):
            raise ValueError("cycle in path")
        paths.append(ordered)
        remaining.difference_update(component)
    return sorted(paths, key=lambda p: p[0])


def _settle(structure, beads, track, bead_color):
    paths = _paths(structure == track)
    answer, details = structure.copy(), []
    covered = set()
    for path in paths:
        members = beads.intersection(path)
        covered.update(members)
        count = len(members)
        if count > len(path):
            raise ValueError("beads exceed path capacity")
        if path[0][0] == path[-1][0]:
            targets = members
        else:
            inward = path if path[0][0] > path[-1][0] else path[::-1]
            targets = set(inward[:count])
        for p in targets:
            answer[p] = bead_color
        details.append({"path": path, "sources": members, "targets": targets, "count": count})
    if covered != beads:
        raise ValueError("beads must lie on track")
    return answer, details


def _compute(info, turn=True):
    mask, stamp, bg = info["mask"], info["stamp"], info["background"]
    raw = _expand(mask, stamp, bg)
    beads = set(map(tuple, np.argwhere(raw == info["bead"])))
    tile = _turn_tile(stamp, info) if turn else stamp.copy()
    tile[tile == info["bead"]] = info["track"]
    structure = _expand(mask, tile, bg)
    answer, details = _settle(structure, beads, info["track"], info["bead"])
    return answer, details, raw, structure


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire output from a single visible input."""
    info = _read_input(input_grid)
    return _compute(info)[0]


def _alternatives(info):
    answer, details, raw, _ = _compute(info)
    unturned = _compute(info, turn=False)[0]
    single_info = dict(info, mask=np.ones((1, 1), dtype=bool))
    local = _compute(single_info)[0]
    isolated = _expand(info["mask"], local, info["background"])
    b = info["bead"]
    return {
        "unchanged_expansion": not np.array_equal(answer, raw),
        "unturned_settlement": not np.array_equal(answer == b, unturned == b),
        "isolated_settlement": not np.array_equal(answer == b, isolated == b),
        "single_motion": any(d["count"] == 1 and d["sources"] != d["targets"] for d in details),
        "packing": any(d["count"] >= 2 for d in details),
    }


def _make_tile(rng, shape, bead_count, rotation, mirror):
    h, w = shape
    y, x = int(rng.integers(3, h - 3)), int(rng.integers(3, w - 3))
    a = _template(h, w, y, x, 0, 1, 2)
    info = {"bead": 3, "track": 1, "divider": 2, "center": (y, x), "horizontal": True}
    eligible = set(map(tuple, np.argwhere(a == 1)))
    eligible = {p for p in eligible if abs(p[0] - y) > 1 or abs(p[1] - x) > 1}
    if bead_count == 2:
        # A bead on each local path forces merged populations at occupied seams.
        channels = _paths(_turn_tile(a, info) == 1)
        choices = []
        for path in channels:
            pool = sorted(eligible.intersection(path))
            choices.append(pool[int(rng.integers(len(pool)))])
    else:
        pool = sorted(eligible)
        choices = [pool[int(k)] for k in rng.choice(len(pool), bead_count, replace=False)]
    for p in choices:
        a[p] = 3
    return np.rot90(np.fliplr(a) if mirror else a, rotation).copy()


def _mask(rng, shape, family):
    h, w = shape
    a = np.zeros(shape, dtype=bool)
    if family == "L":
        a[:] = True
        a[0, -1] = False
    elif family == "full":
        a[:] = True
    elif family == "ring":
        a[0] = a[-1] = True
        a[:, 0] = a[:, -1] = True
        if min(shape) >= 3 and rng.random() < .55:
            corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
            for k in rng.choice(4, int(rng.integers(1, 3)), replace=False):
                a[corners[int(k)]] = False
    elif family == "diagonal":
        rr, cc = np.indices(shape)
        a = (rr + cc) % 2 == 0
    elif family == "stair":
        r = c = 0
        a[r, c] = True
        while r < h - 1 or c < w - 1:
            if r < h - 1 and (c == w - 1 or rng.random() < .5):
                r += 1
            else:
                c += 1
            a[r, c] = True
    else:
        a = rng.random(shape) < float(rng.uniform(.25, .85))
    # Tight bounds remain visible even for islands or blank internal lines.
    if not a[0].any():
        a[0, int(rng.integers(w))] = True
    if not a[-1].any():
        a[-1, int(rng.integers(w))] = True
    if not a[:, 0].any():
        a[int(rng.integers(h)), 0] = True
    if not a[:, -1].any():
        a[int(rng.integers(h)), -1] = True
    if rng.random() < .5:
        a = np.flipud(a)
    if rng.random() < .5:
        a = np.fliplr(a)
    return a.copy()


def _layout(rng, mask, stamp, palette):
    bg, signal = palette[0], palette[4]
    mcrop = np.where(mask, signal, bg).astype(np.int8)
    sh, sw = stamp.shape
    mh, mw = mask.shape
    horizontal, reverse = bool(rng.integers(2)), bool(rng.integers(2))
    gap = int(rng.integers(1, 5))
    h = (max(mh, sh) if horizontal else mh + sh + gap) + 2 + int(rng.integers(5))
    w = (mw + sw + gap if horizontal else max(mw, sw)) + 2 + int(rng.integers(5))
    while h * w <= 2 * (int(mask.sum()) + int(np.count_nonzero(stamp != bg))):
        if h <= w:
            h += 1
        else:
            w += 1
    a = np.full((h, w), bg, dtype=np.int8)
    if horizontal:
        left = int(rng.integers(1, w - mw - sw - gap))
        mc, sc = (left + sw + gap, left) if reverse else (left, left + mw + gap)
        mr, sr = int(rng.integers(1, h - mh)), int(rng.integers(1, h - sh))
    else:
        top = int(rng.integers(1, h - mh - sh - gap))
        mr, sr = (top + sh + gap, top) if reverse else (top, top + mh + gap)
        mc, sc = int(rng.integers(1, w - mw)), int(rng.integers(1, w - sw))
    a[mr:mr + mh, mc:mc + mw] = mcrop
    a[sr:sr + sh, sc:sc + sw] = stamp
    return a


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))  # network growth / tile growth / topology
    varied = bool(rng.random() < .7)
    fixed_palette = np.array(rng.permutation(10)[:5], dtype=np.int8)
    first_rotation = int(rng.integers(4))
    puzzle, seen, first_shape = {"train": [], "test": []}, set(), None
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            witness = index if split == "train" and index < 2 else None
            for _attempt in range(100):
                if witness == 0:
                    mshape, family, bead_count = (2, 2), "L", 1
                elif witness == 1:
                    mshape = (2, 3) if rng.random() < .5 else (3, 2)
                    family, bead_count = "full", 2
                elif split == "test" and mode == 0:
                    mshape = ((4, 3), (3, 4), (4, 4))[int(rng.integers(3))]
                    family, bead_count = str(rng.choice(["full", "ring", "stair", "random"])), int(rng.integers(1, 4))
                elif split == "test" and mode == 2:
                    mshape, family, bead_count = (3, 3), "ring", int(rng.integers(1, 4))
                else:
                    mshape = (int(rng.integers(1 if split == "train" else 2, 4)),
                              int(rng.integers(1 if split == "train" else 2, 4)))
                    if split == "train" and mode == 2 and min(mshape) == 3:
                        # Reserve a genuinely surrounded hole for the query.
                        mshape = (2, 3) if rng.random() < .5 else (3, 2)
                    family = str(rng.choice(["full", "ring", "diagonal", "stair", "random"]))
                    bead_count = int(rng.integers(1, 4))
                low, high = (9, 11) if split == "test" and mode == 1 else ((7, 9) if mode == 1 else (7, 11))
                rotation = first_rotation if witness == 0 else (first_rotation + 1) % 4 if witness == 1 else int(rng.integers(4))
                # Choose displayed dimensions independently, subject to products.
                sh = int(rng.integers(low, min(high, 30 // mshape[0] + 1)))
                sw = int(rng.integers(low, min(high, 30 // mshape[1] + 1)))
                if witness == 1 and (sh, sw) == first_shape:
                    continue
                canonical_shape = (sw, sh) if rotation % 2 else (sh, sw)
                tile = _make_tile(rng, canonical_shape, bead_count, rotation, bool(rng.integers(2)))
                mask = _mask(rng, mshape, family)
                palette = np.array(rng.permutation(10)[:5], dtype=np.int8) if varied else fixed_palette
                colored = palette[tile]
                inp = _layout(rng, mask, colored, palette)
                try:
                    info = _read_input(inp)
                    out = _compute(info)[0]
                    alternatives = _alternatives(info)
                except ValueError:
                    continue
                if witness == 0 and not (alternatives["single_motion"] and alternatives["unturned_settlement"]):
                    continue
                if witness == 1 and not (alternatives["packing"] and alternatives["isolated_settlement"]):
                    continue
                if split == "test" and not alternatives["isolated_settlement"]:
                    continue
                key = _key(inp)
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append({"input": inp, "output": out})
                if witness == 0:
                    first_shape = (sh, sw)
                break
            else:
                raise RuntimeError("bounded construction failed")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _endpoint_exposed(p, info):
    sh, sw = info["stamp"].shape
    mask = info["mask"]
    r, c = map(int, p)
    mr, mc, lr, lc = r // sh, c // sw, r % sh, c % sw
    # Endpoints must face outside the blueprint or an inactive neighbor.
    for condition, nr, nc in ((lr == 0, mr - 1, mc), (lr == sh - 1, mr + 1, mc),
                              (lc == 0, mr, mc - 1), (lc == sw - 1, mr, mc + 1)):
        if condition and (not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]) or not mask[nr, nc]):
            return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos, coverage = set(), {"train": [], "test": []}, {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _read_input(inp)
            expected, paths, raw, structure = _compute(info)
            assert np.array_equal(out, expected), "incorrect output"
            assert out.shape == tuple(m * s for m, s in zip(info["mask"].shape, info["stamp"].shape))
            bg, bead, divider, track = (info[k] for k in ("background", "bead", "divider", "track"))
            assert np.count_nonzero(out == bead) == int(info["mask"].sum()) * np.count_nonzero(info["stamp"] == bead)
            assert np.count_nonzero(out == divider) == 3 * int(info["mask"].sum())
            assert info["signal"] not in out
            assert np.all((out == bead) | (out == structure))
            assert np.array_equal(out != bg, structure != bg), "background holes changed"
            for details in paths:
                path, sources, targets = details["path"], details["sources"], details["targets"]
                assert path[0][0] != path[-1][0], "tile grammar should have no level route"
                assert all(_endpoint_exposed(p, info) for p in (path[0], path[-1]))
                assert len(sources) == len(targets)
                inward = path if path[0][0] > path[-1][0] else path[::-1]
                assert targets == set(inward[:len(sources)]), "not consecutively packed"
                assert sources == {p for p in path if raw[p] == bead}
                assert targets == {p for p in path if out[p] == bead}
            sh, sw = info["stamp"].shape
            for r, c in np.argwhere(~info["mask"]):
                assert np.all(out[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] == bg)
            alternatives = _alternatives(info)
            if split == "train":
                for k, v in alternatives.items():
                    coverage[k] = coverage.get(k, False) or v
            else:
                assert alternatives["isolated_settlement"], "test does not require global assembly"
            infos[split].append(info)
    assert all(coverage.values()), "training fails a mechanism or shortcut witness"
    train, test = infos["train"], infos["test"]
    assert len({i["mask"].shape for i in train}) >= 2
    assert len({i["stamp"].shape for i in train}) >= 2
    assert {i["horizontal"] for i in train} == {False, True}
    assert any(np.any(~i["mask"]) for i in train)
    assert any(all(not np.array_equal(i["stamp"], t)
                   for t in _symmetries(i["stamp"])[1:]) for i in train), "missing tile orientation witness"
    # Recover at least one declared shift from grids, not sampled mode labels.
    network_growth = (all(max(i["mask"].shape) <= 3 for i in train)
                      and all(4 in i["mask"].shape for i in test))
    tile_growth = (all(max(i["stamp"].shape) <= 8 for i in train)
                   and all(min(i["stamp"].shape) >= 9 for i in test))
    topology = all(min(i["mask"].shape) <= 2 for i in train) and all(i["mask"].shape == (3, 3) and not i["mask"][1, 1]
                   and i["mask"][0, 1] and i["mask"][1, 0]
                   and i["mask"][1, 2] and i["mask"][2, 1] for i in test)
    assert network_growth or tile_growth or topology, "missing visible train/test shift"


def _features(pair):
    i = _read_input(pair["input"])
    _, paths, _, _ = _compute(i)
    mb, sb = i["mask_box"], i["stamp_box"]
    layout = ("left" if mb[3] < sb[2] else "right") if (mb[3] < sb[2] or sb[3] < mb[2]) else ("above" if mb[1] < sb[0] else "below")
    normalized_tile = np.zeros(i["stamp"].shape, dtype=np.int8)
    for code, key in enumerate(("background", "track", "divider", "bead")):
        normalized_tile[i["stamp"] == i[key]] = code
    return {"mask_shape": i["mask"].shape, "tile_shape": i["stamp"].shape,
            "input_shape": np.asarray(pair["input"]).shape,
            "output_shape": np.asarray(pair["output"]).shape,
            "mask": _key(i["mask"]), "tile": _key(normalized_tile),
            "palette": tuple(i[k] for k in ("background", "track", "divider", "bead", "signal")),
            "beads_per_tile": int(np.count_nonzero(i["stamp"] == i["bead"])),
            "active": int(i["mask"].sum()), "paths": len(paths),
            "max_population": max(d["count"] for d in paths), "layout": layout,
            "axis": i["horizontal"], "center": i["center"]}


def self_test():
    """Bounded semantic, API, mutation and 42-seed structural checks."""
    records, fingerprints = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solver mutated input"
                records.append(_features(pair))
    assert len(fingerprints) == 42
    for feature, minimum in (("mask", 40), ("tile", 100), ("input_shape", 50),
                             ("output_shape", 35), ("tile_shape", 12), ("palette", 60)):
        assert len({r[feature] for r in records}) >= minimum, feature
    assert {r["beads_per_tile"] for r in records} == {1, 2, 3}
    assert {r["layout"] for r in records} == {"left", "right", "above", "below"}
    assert {r["axis"] for r in records} == {False, True}
    assert max(r["active"] for r in records) >= 12
    assert max(r["max_population"] for r in records) >= 5
    assert any(r["tile_shape"][0] != r["tile_shape"][1] for r in records)
    for role in range(5):
        assert {r["palette"][role] for r in records} == set(range(10))
    p, q = generate_puzzle(3127, 5, 3), generate_puzzle(3127, 5, 3)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for seed in (-1, True, 3.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # Exhaust every nonempty 2x3 mask, including crops with internal blank lines.
    rng = np.random.default_rng(7821)
    stamp = _make_tile(rng, (7, 9), 3, 0, False)
    info = dict(_tile_info(stamp, 0), background=0, stamp=stamp)
    for bits in itertools.product((False, True), repeat=6):
        mask = np.array(bits, dtype=bool).reshape(2, 3)
        if not mask.any():
            continue
        answer, paths, raw, _ = _compute(dict(info, mask=mask))
        assert np.count_nonzero(answer == 3) == np.count_nonzero(raw == 3)
        assert all(d["path"][0][0] != d["path"][-1][0] for d in paths)
    damaged = generate_puzzle(0)
    out = damaged["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(fingerprints),
            "distinct_masks": len({r["mask"] for r in records}),
            "distinct_tiles": len({r["tile"] for r in records}),
            "input_shapes": len({r["input_shape"] for r in records}),
            "output_shapes": len({r["output_shape"] for r in records}),
            "max_path_population": max(r["max_population"] for r in records)}


if __name__ == "__main__":
    # Optional local report, kept compact and confined to scratch/.
    report = self_test()
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report, separators=(",", ":")))
