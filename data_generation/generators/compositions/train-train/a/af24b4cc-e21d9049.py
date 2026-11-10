"""Periodic rendezvous: noisy tile clocks generate a color-agreement lattice.

Only the standard library and NumPy are used.  The solver reads gutters, local
modes, the two complete words and their phases from the input itself.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be 2D with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.tobytes()


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _rendezvous(height, width, gutter, empty, horizontal, vertical, hs, vs):
    """The words run on *tile indices*, independent of physical tile sizes."""
    h = np.asarray(horizontal, dtype=np.int8)[
        (np.arange(width) - hs) % len(horizontal)]
    v = np.asarray(vertical, dtype=np.int8)[
        (np.arange(height) - vs) % len(vertical)]
    out = np.full((height + 2, width + 2), gutter, dtype=np.int8)
    out[1:-1, 1:-1] = np.where(v[:, None] == h[None, :], h[None, :], empty)
    return out


def _read_input(input_grid):
    """Validate and decode a family member without consulting any output."""
    a = _grid(input_grid)
    gutter = int(a[0, 0])
    _require(bool(np.all(a[0] == gutter) and np.all(a[-1] == gutter)
                  and np.all(a[:, 0] == gutter) and np.all(a[:, -1] == gutter)),
             "a complete outer gutter is required")
    gr = np.flatnonzero(np.all(a == gutter, axis=1))
    gc = np.flatnonzero(np.all(a == gutter, axis=0))
    heights = tuple(int(x) for x in np.diff(gr) - 1)
    widths = tuple(int(x) for x in np.diff(gc) - 1)
    height, width = len(heights), len(widths)
    _require(5 <= height <= 9 and 5 <= width <= 9, "tile board must be 5..9 on each axis")
    _require(all(x in (2, 3) for x in heights + widths), "tile bands must be 2 or 3 cells")
    modes = np.empty((height, width), dtype=np.int8)
    first_pixels = np.empty_like(modes)
    patches = {}
    plurality = uniform_signal = 0
    minorities = set()
    for r in range(height):
        for c in range(width):
            patch = a[gr[r] + 1:gr[r + 1], gc[c] + 1:gc[c + 1]]
            _require(bool(np.all(patch != gutter)), "gutter color inside tile")
            values, counts = np.unique(patch, return_counts=True)
            _require(1 <= len(values) <= 3, "a tile uses one to three colors")
            best = int(np.argmax(counts))
            _require(np.count_nonzero(counts == counts[best]) == 1, "tied local mode")
            modes[r, c] = values[best]
            first_pixels[r, c] = patch[0, 0]
            patches[r, c] = (patch, values, counts, best)
    vals, counts = np.unique(modes, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "blank tile mode is ambiguous")
    empty = int(vals[np.argmax(counts)])
    _require(np.count_nonzero(modes == empty) > modes.size // 2,
             "blank tiles must be a strict majority of the tile board")
    mask = modes != empty
    rr = np.flatnonzero(mask.sum(axis=1) > 1)
    cc = np.flatnonzero(mask.sum(axis=0) > 1)
    _require(len(rr) == len(cc) == 1, "expected one horizontal and one vertical word")
    row, col = int(rr[0]), int(cc[0])
    hc, vr = np.flatnonzero(mask[row]), np.flatnonzero(mask[:, col])
    hp, vp = len(hc), len(vr)
    _require(3 <= hp <= 5 and 3 <= vp <= 5, "visible words must have lengths 3..5")
    _require(hp < width and vp < height, "both axes must have room to repeat")
    hs, vs = int(hc[0]), int(vr[0])
    _require(np.array_equal(hc, np.arange(hs, hs + hp))
             and np.array_equal(vr, np.arange(vs, vs + vp)), "word fragments must be contiguous")
    expected_mask = np.zeros_like(mask)
    expected_mask[row, hs:hs + hp] = True
    expected_mask[vs:vs + vp, col] = True
    _require(np.array_equal(mask, expected_mask), "extra nonblank tiles")
    hword = tuple(int(x) for x in modes[row, hs:hs + hp])
    vword = tuple(int(x) for x in modes[vs:vs + vp, col])
    _require(len(set(hword)) == hp and len(set(vword)) == vp,
             "each visible word must have distinct colors")
    _require(hword[col - hs] == vword[row - vs], "intersection phases disagree")
    signals = set(hword) | set(vword)
    _require(len(signals) <= 7, "retain at least one available noise-only color")
    _require(len(set(hword) & set(vword)) >= 2, "at least two shared clock colors are required")
    noisy = 0
    for (r, c), (patch, values, pc, best) in patches.items():
        if not mask[r, c]:
            _require(bool(np.all(patch == empty)), "blank tiles must be uniform")
        else:
            _require(empty not in values, "blank color inside a signal tile")
            uniform_signal += len(values) == 1
            noisy += len(values) > 1
            plurality += 2 * int(pc[best]) < patch.size
            minorities.update(int(v) for i, v in enumerate(values) if i != best)
    hi, vi = col - hs, row - vs
    shape = ("L" if hi in (0, hp - 1) and vi in (0, vp - 1) else
             "cross" if 0 < hi < hp - 1 and 0 < vi < vp - 1 else "T")
    rotations = {hword[i:] + hword[:i] for i in range(hp)}
    relation = ("unequal" if hp != vp else "partial" if set(hword) != set(vword) else
                "same" if hword == vword else "rotation" if vword in rotations else "permuted")
    return dict(gutter=gutter, empty=empty, height=height, width=width,
                heights=heights, widths=widths, modes=modes, first_pixels=first_pixels,
                mask=mask, row=row, col=col, hs=hs, vs=vs,
                horizontal=hword, vertical=vword, hp=hp, vp=vp, hi=hi, vi=vi,
                shape=shape, relation=relation, plurality=int(plurality),
                noisy=int(noisy), uniform=int(uniform_signal),
                input_only_noise=bool(minorities - signals),
                role_overlap=bool(minorities & signals))


def _from_info(info, *, horizontal=None, vertical=None, hs=None, vs=None):
    return _rendezvous(info["height"], info["width"], info["gutter"], info["empty"],
                       info["horizontal"] if horizontal is None else horizontal,
                       info["vertical"] if vertical is None else vertical,
                       info["hs"] if hs is None else hs,
                       info["vs"] if vs is None else vs)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the compact, bordered rendezvous lattice of this input's clocks."""
    return _from_info(_read_input(input_grid))


def _ablation_evidence(info, out):
    """Concrete wrong-rule witnesses, measured from observed inputs and outputs."""
    r, c, hs, vs = (info[k] for k in ("row", "col", "hs", "vs"))
    hp, vp = info["hp"], info["vp"]
    first = info["first_pixels"]
    pixel_rule = _from_info(info, horizontal=first[r, hs:hs + hp],
                           vertical=first[vs:vs + vp, c])
    finite = np.full_like(out, info["gutter"])
    finite[1:-1, 1:-1] = info["empty"]
    finite[vs + 1:vs + vp + 1, hs + 1:hs + hp + 1] = out[
        vs + 1:vs + vp + 1, hs + 1:hs + hp + 1]
    # Incorrectly derive the vertical clock from the horizontal clock, while
    # aligning its shared color at the observed intersection.
    cloned_v = tuple(info["horizontal"][(j - info["vi"] + info["hi"]) % hp]
                     for j in range(hp))
    cloned_vs = r - info["vi"]
    cloned = _from_info(info, vertical=cloned_v, vs=cloned_vs)
    off_axis = np.ones((info["height"], info["width"]), dtype=bool)
    off_axis[r, :] = False
    off_axis[:, c] = False
    return dict(local_mode=not np.array_equal(pixel_rule, out),
                repetition=not np.array_equal(finite, out),
                phase=not np.array_equal(_from_info(info, hs=0, vs=0), out),
                independent_axes=not np.array_equal(cloned, out),
                off_axis=bool(np.any((out[1:-1, 1:-1] != info["empty"]) & off_axis)))


def _words(rng, available, kind):
    hp = int(rng.integers(3, 6))
    vp = hp
    if kind == "unequal":
        vp = _pick(rng, [p for p in (3, 4, 5) if p != hp])
    h = tuple(int(x) for x in rng.choice(available, hp, replace=False))
    if kind == "same":
        return h, h
    if kind == "rotation":
        shift = int(rng.integers(1, hp))
        return h, h[shift:] + h[:shift]
    if kind == "permuted":
        # Swapping two entries can never be a rotation for length >= 3.
        v = list(h)
        i, j = [int(x) for x in rng.choice(hp, 2, replace=False)]
        v[i], v[j] = v[j], v[i]
        return h, tuple(v)
    overlap_min = max(2, hp + vp - 7)
    overlap_max = min(hp, vp) - 1
    overlap = int(rng.integers(overlap_min, overlap_max + 1))
    common = [int(x) for x in rng.choice(h, overlap, replace=False)]
    outside = [x for x in available if x not in h]
    extra = [int(x) for x in rng.choice(outside, vp - overlap, replace=False)]
    return h, tuple(int(x) for x in rng.permutation(common + extra))


def _bands(rng, n, required=None):
    bands = [int(x) for x in rng.integers(2, 4, n)]
    if required is not None:
        bands[required] = 3
        other = _pick(rng, [i for i in range(n) if i != required])
        bands[other] = 2
    while sum(bands) + n + 1 > 30:
        candidates = [i for i in range(n) if bands[i] == 3 and i != required]
        bands[_pick(rng, candidates)] = 2
    return bands


def _noise_patch(rng, height, width, mode, available, signals,
                 *, special=False, uniform=False, hard=False):
    area = height * width
    patch = np.full((height, width), mode, dtype=np.int8)
    if uniform:
        return patch
    if special:
        _require(area == 9, "plurality witness needs a 3x3 tile")
        counts = (4, 3, 2)
        others = [_pick(rng, [x for x in available if x not in signals]),
                  _pick(rng, sorted(signals - {mode}))]
    else:
        choices = [x for x in available if x != mode]
        if rng.random() < 0.35:
            candidates = [(m, n, area - m - n)
                          for m in range(2, area)
                          for n in range(1, area - m)
                          if m > n and m > area - m - n]
            counts = _pick(rng, candidates)
            others = [int(x) for x in rng.choice(choices, 2, replace=False)]
        else:
            noise = ((area - 1) // 2 if hard or rng.random() < 0.4
                     else int(rng.integers(1, max(2, area // 3 + 1))))
            counts = (area - noise, noise)
            others = [_pick(rng, choices)]
    style = _pick(rng, ["scatter", "edge", "cluster", "stripe"])
    cells = [(r, c) for r in range(height) for c in range(width)]
    order = [int(x) for x in rng.permutation(area)]
    if style == "edge":
        order.sort(key=lambda i: min(cells[i][0], cells[i][1],
                                      height - 1 - cells[i][0], width - 1 - cells[i][1]))
    elif style == "cluster":
        ar, ac = _pick(rng, cells)
        order.sort(key=lambda i: abs(cells[i][0] - ar) + abs(cells[i][1] - ac))
    elif style == "stripe":
        axis = int(rng.integers(2))
        order.sort(key=lambda i: cells[i][axis])
    if special:
        # Make point sampling fail visibly while preserving the 4:3:2 mode.
        order = [0, area // 2] + [i for i in order if i not in (0, area // 2)]
    cursor = 0
    for color, count in zip(others, counts[1:]):
        for i in order[cursor:cursor + count]:
            patch[cells[i]] = color
        cursor += count
    return patch


def _start(rng, length, period):
    return _pick(rng, [0, length - period, int(rng.integers(length - period + 1))])


def _make_pair(rng, kind, *, witness=False, test_shift=False, hard=False,
               excluded_gutter=None, excluded_empty=None):
    gutter = _pick(rng, [x for x in range(10) if x != excluded_gutter])
    empty = _pick(rng, [x for x in range(10) if x not in (gutter, excluded_empty)])
    available = [x for x in range(10) if x not in (gutter, empty)]
    h, v = _words(rng, available, kind)
    hp, vp = len(h), len(v)
    height = int(rng.integers(max(5, vp + 1), 8))
    width = int(rng.integers(max(5, hp + 1), 8))
    if test_shift:
        if rng.integers(2):
            height, width = int(rng.integers(8, 10)), int(rng.integers(max(5, hp + 1), 10))
        else:
            height, width = int(rng.integers(max(5, vp + 1), 10)), int(rng.integers(8, 10))
    hs, vs = _start(rng, width, hp), _start(rng, height, vp)
    matches = [(i, j) for i in range(hp) for j in range(vp) if h[i] == v[j]]
    groups = {}
    for i, j in matches:
        shape = ("L" if i in (0, hp - 1) and j in (0, vp - 1) else
                 "cross" if 0 < i < hp - 1 and 0 < j < vp - 1 else "T")
        groups.setdefault(shape, []).append((i, j))
    hi, vi = _pick(rng, groups[_pick(rng, list(groups))])
    row, col = vs + vi, hs + hi
    modes = np.full((height, width), empty, dtype=np.int8)
    modes[row, hs:hs + hp] = h
    modes[vs:vs + vp, col] = v
    positions = [tuple(int(x) for x in p) for p in np.argwhere(modes != empty)]
    common = set(h) & set(v)
    special = _pick(rng, [p for p in positions if int(modes[p]) in common and p != (row, col)])
    heights = _bands(rng, height, special[0] if witness else None)
    widths = _bands(rng, width, special[1] if witness else None)
    # Mix clean and noisy tokens in the first witness. Other pairs may have no
    # uniform signal tokens, or several; blank tiles are always uniform.
    uniform_positions = set()
    if witness:
        uniform_positions.add(_pick(rng, [p for p in positions if p != special]))
    else:
        for p in positions:
            if rng.random() < 0.22:
                uniform_positions.add(p)
    gr = np.cumsum([0] + [x + 1 for x in heights])
    gc = np.cumsum([0] + [x + 1 for x in widths])
    inp = np.full((int(gr[-1]) + 1, int(gc[-1]) + 1), gutter, dtype=np.int8)
    signals = set(h) | set(v)
    for r in range(height):
        for c in range(width):
            if modes[r, c] == empty:
                patch = np.full((heights[r], widths[c]), empty, dtype=np.int8)
            else:
                patch = _noise_patch(rng, heights[r], widths[c], int(modes[r, c]),
                                     available, signals, special=witness and (r, c) == special,
                                     uniform=(r, c) in uniform_positions, hard=hard)
            inp[gr[r] + 1:gr[r + 1], gc[c] + 1:gc[c + 1]] = patch
    out = _rendezvous(height, width, gutter, empty, h, v, hs, vs)
    return {"input": inp, "output": out}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    used = set()
    first_info = None
    kinds = ["same", "rotation", "permuted", "partial", "unequal"]
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            mandatory = split == "train" and i < 2
            kind = ("permuted" if i == 0 else "unequal") if mandatory else _pick(rng, kinds)
            for attempt in range(256):
                pair = _make_pair(rng, kind, witness=mandatory,
                                  test_shift=split == "test" and i == 0,
                                  hard=split == "test" or rng.random() < 0.35,
                                  excluded_gutter=first_info["gutter"] if mandatory and i == 1 else None,
                                  excluded_empty=first_info["empty"] if mandatory and i == 1 else None)
                info = _read_input(pair["input"])
                evidence = _ablation_evidence(info, pair["output"])
                if _key(pair["input"]) in used:
                    continue
                if not evidence["repetition"] or not evidence["off_axis"]:
                    continue
                if mandatory and (not all(evidence.values()) or not info["plurality"]):
                    continue
                if mandatory and i == 1 and (info["height"], info["width"]) == (
                        first_info["height"], first_info["width"]):
                    continue
                puzzle[split].append(pair)
                used.add(_key(pair["input"]))
                if split == "train" and i == 0:
                    first_info = info
                break
            else:
                raise RuntimeError("bounded construction could not find a distinct witnessed pair")
    # Witnesses are requirements on the demonstrations, not on their order.
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal example counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _read_input(inp)
            _require(out.shape == (info["height"] + 2, info["width"] + 2), "wrong compression size")
            _require(np.array_equal(_from_info(info), out), "wrong rendezvous output")
            # Check each output cell's meaning explicitly, including unmatched
            # clocks, clipping, and the one-cell border.
            g, e = info["gutter"], info["empty"]
            _require(bool(np.all(out[0] == g) and np.all(out[-1] == g)
                          and np.all(out[:, 0] == g) and np.all(out[:, -1] == g)), "wrong border")
            for r in range(info["height"]):
                for c in range(info["width"]):
                    h = info["horizontal"][(c - info["hs"]) % info["hp"]]
                    v = info["vertical"][(r - info["vs"]) % info["vp"]]
                    _require(int(out[r + 1, c + 1]) == (h if h == v else e), "invalid color coincidence")
            evidence = _ablation_evidence(info, out)
            _require(evidence["repetition"] and evidence["off_axis"], "periodic lattice is not visible")
            info["evidence"] = evidence
            (training if split == "train" else testing).append(info)
    _require(len({a["gutter"] for a in training}) >= 2
             and len({a["empty"] for a in training}) >= 2, "training must demonstrate varying color roles")
    _require(len({(a["height"], a["width"]) for a in training}) >= 2, "training board sizes must vary")
    _require(any(a["relation"] == "permuted" for a in training), "missing independent same-palette demonstration")
    _require(any(a["hp"] != a["vp"] and set(a["horizontal"]) != set(a["vertical"])
                 for a in training), "missing unequal, partially shared clock demonstration")
    for name in ("local_mode", "repetition", "phase", "independent_axes", "off_axis"):
        _require(any(a["evidence"][name] for a in training), f"training does not distinguish {name}")
    _require(any(a["plurality"] and a["input_only_noise"] and a["role_overlap"] for a in training),
             "missing local plurality/color-role witness")
    _require(any(a["uniform"] for a in training), "missing uniform signal token")
    _require(any(len(set(a["heights"])) > 1 and len(set(a["widths"])) > 1 for a in training),
             "missing unequal physical band sizes")
    _require(any(max(a["height"], a["width"]) > max(max(b["height"], b["width"]) for b in training)
                 for a in testing), "missing longer test recurrence")


def _puzzle_key(puzzle):
    return tuple((s, tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                           for p in puzzle[s])) for s in ("train", "test"))


def self_test():
    """Bounded tests and input-derived coverage over 42 deterministic seeds."""
    shapes, layouts, periods, relations, geometries, colors, structures = (set() for _ in range(7))
    puzzle_keys = set()
    counts = Counter()
    l_corners = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterministic generation")
        puzzle_keys.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = pair["input"].copy()
                _require(np.array_equal(solve(before), pair["output"]), "input-only solver mismatch")
                _require(np.array_equal(before, pair["input"]), "solver mutated input")
                a = _read_input(pair["input"])
                shapes.add(pair["input"].shape)
                layouts.add((a["heights"], a["widths"]))
                periods.add((a["hp"], a["vp"]))
                relations.add(a["relation"])
                geometries.add(a["shape"])
                colors.update((a["gutter"], a["empty"]))
                roles = {}
                normalized = tuple(roles.setdefault(int(x), len(roles)) for x in pair["input"].flat)
                structures.add((pair["input"].shape, normalized))
                counts[split + "_pairs"] += 1
                for feature in ("plurality", "input_only_noise", "role_overlap"):
                    counts[feature + "_pairs"] += bool(a[feature])
                counts["edge_intersections"] += (a["row"] in (0, a["height"] - 1)
                                                  or a["col"] in (0, a["width"] - 1))
                counts["zero_is_signal"] += 0 in set(a["horizontal"]) | set(a["vertical"])
                if a["shape"] == "L":
                    l_corners.add((a["hi"] == 0, a["vi"] == 0))
                # Recolor all ten roles and transpose: both are rule-preserving
                # transformations, independently of the generation schedule.
                perm = np.roll(np.arange(10, dtype=np.int8), 3)
                _require(np.array_equal(solve(perm[before]), perm[pair["output"]]), "palette covariance failed")
                _require(np.array_equal(solve(before.T), pair["output"].T), "transpose covariance failed")
    _require(len(puzzle_keys) == 42 and len(shapes) >= 30 and len(layouts) >= 100,
             "structural variation collapsed")
    _require(len(periods) == 9 and len(relations) == 5, "clock variation collapsed")
    _require(geometries == {"cross", "L", "T"} and len(l_corners) == 4, "intersection variation collapsed")
    _require(len(colors) == 10 and counts["zero_is_signal"] > 0 and counts["edge_intersections"] > 0,
             "color or boundary variation collapsed")
    # All legal explicit count combinations, including the minimum evidence case.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7300 + 10 * nt + ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {nt}/{ne}")
    return dict(seeds=42, distinct_puzzles=len(puzzle_keys), input_shapes=len(shapes),
                band_layouts=len(layouts), color_normalized_structures=len(structures),
                period_pairs=sorted(periods), relations=sorted(relations),
                intersection_shapes=sorted(geometries), l_corners=len(l_corners), **counts)
