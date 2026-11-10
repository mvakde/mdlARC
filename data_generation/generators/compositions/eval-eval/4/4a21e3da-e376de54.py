"""Conserved offcuts: a middle-length cut and marker-defined collective routing.

Only NumPy and the standard library are used. Generation draws labels directly
from sampled geometry; solve reconstructs all geometry from its input alone.
There is no file I/O, global random state, example cache, or parent dependency.
"""
from __future__ import annotations

import hashlib

import numpy as np


HOST_SEEDS = (0, 874513993, 443780214, 489286348, 251008851, 327371623,
              54484809, 124423370, 895278624, 257873783, 935124327, 521830624,
              820089351, 723529492, 986835843, 226762136, 890397193, 4255224,
              546419856, 696423169, 506682727, 262369013, 259621881, 350850992,
              637648556, 971053521, 51747762, 887782212, 271935427, 861054431,
              778426404, 54951770, 191585256, 188629427, 518721884, 964237859,
              301739035, 103850709, 418460811, 421311635, 661686139, 723405299)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or any(n < 1 or n > 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def _orient(g, turns, mirror=False):
    return np.rot90(np.fliplr(g) if mirror else g, turns).copy()


def _parse(input_grid):
    """Reconstruct a canonical top-marker comb, using no generation metadata."""
    g = _grid(input_grid)
    h, w = g.shape
    if min(h, w) < 5:
        raise ValueError("canvas too small for this family's visible grammar")
    rim = np.concatenate((g[0], g[-1], g[1:-1, 0], g[1:-1, -1]))
    values, counts = np.unique(rim, return_counts=True)
    bg = int(values[np.argmax(counts)])
    border = np.zeros(g.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    markers = np.argwhere(border & (g != bg))
    if len(markers) != 1:
        raise ValueError("expected one border marker")
    r, c = map(int, markers[0])
    if r in (0, h-1) and c in (0, w-1):
        raise ValueError("corner markers are ambiguous")
    # Positive np.rot90 maps top to left, bottom, then right.
    turns = 0 if r == 0 else 1 if c == 0 else 2 if r == h-1 else 3
    x = np.rot90(g, -turns).copy()
    mark_position = np.argwhere((x != bg) & np.rot90(border, -turns))[0]
    s = int(mark_position[1])
    mark = int(x[0, s])
    if np.count_nonzero(x == mark) != 1:
        raise ValueError("marker color must occur only at its border cell")
    mask = (x != bg) & (x != mark)
    rr, cc = np.nonzero(mask)
    if not len(rr) or np.any((rr == 0) | (rr == x.shape[0]-1)
                              | (cc == 0) | (cc == x.shape[1]-1)):
        raise ValueError("comb must be nonempty and strictly interior")
    cols = sorted(map(int, np.unique(cc)))
    n = len(cols)
    if n not in (3, 5, 7, 9) or min(np.diff(cols)) < 2:
        raise ValueError("expected an odd separated array of 3/5/7/9 strokes")
    if not cols[0] < s < cols[-1] or s in cols:
        raise ValueError("marker axis must lie in an internal gap")
    lengths, colors, anchors = [], [], []
    for col in cols:
        rows = np.flatnonzero(mask[:, col])
        color = int(x[rows[0], col])
        if (len(rows) != int(rows[-1]-rows[0]+1)
                or np.any(x[rows, col] != color)):
            raise ValueError("teeth must be continuous monochrome lines")
        lengths.append(len(rows))
        colors.append(color)
        anchors.append(int(rows[0]))
    if len(set(anchors)) != 1 or anchors[0] < 2:
        raise ValueError("teeth must have a common interior anchor baseline")
    reference = lengths[n//2]
    if reference < 2 or max(lengths) <= reference:
        raise ValueError("expected reference >=2 and at least one long tooth")
    return {"grid": x, "turns": turns, "bg": bg, "mark": mark,
            "seam": s, "cols": cols, "lengths": lengths, "colors": colors,
            "anchor": anchors[0], "reference": reference}


def _tails(info, reference=None, seam=None):
    length = info["reference"] if reference is None else reference
    axis = info["seam"] if seam is None else seam
    a = info["anchor"]
    groups = [[], []]
    for c, span, color in zip(info["cols"], info["lengths"], info["colors"]):
        groups[int(c > axis)].extend((r, c, color) for r in range(a+length, a+span))
    return groups


def _put_group(out, cells, right):
    if not cells:
        return
    dr = out.shape[0]-1-max(r for r, c, color in cells)
    dc = (out.shape[1]-1-max(c for r, c, color in cells) if right
          else -min(c for r, c, color in cells))
    for r, c, color in cells:
        out[r+dr, c+dc] = color


def _render(info, reference=None, tail_mode="collective", seam=None):
    """Render parsed geometry; alternate parameters are bounded ablation checks."""
    length = info["reference"] if reference is None else int(reference)
    out = np.full_like(info["grid"], info["bg"])
    a, s = info["anchor"], info["seam"]
    for col, color in zip(info["cols"], info["colors"]):
        out[a:a+length, col] = color
    out[:a+length, s] = info["mark"]
    groups = _tails(info, length, seam)
    if tail_mode == "collective":
        for right, cells in enumerate(groups):
            _put_group(out, cells, bool(right))
    elif tail_mode == "individual":
        for right, cells in enumerate(groups):
            for col in sorted({c for r, c, color in cells}):
                _put_group(out, [p for p in cells if p[1] == col], bool(right))
    elif tail_mode != "drop":
        raise ValueError("unknown offcut mode")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use the visible middle length and seam; return a new grid."""
    info = _parse(input_grid)
    return _orient(_render(info), info["turns"])


def _sample_gaps(rng, count, equal, witness):
    gaps = [2] * (count-1)
    if witness:
        # An odd extra at an outer gap makes the unique geometric middle
        # differ from the ordinal middle, while leaving room for margins.
        extra = int(rng.choice([v for v in (5, 7, 9) if sum(gaps)+v <= 26]))
        gaps[0 if rng.integers(2) else -1] += extra
    elif equal:
        gaps = [int(rng.integers(2, min(6, 26//(count-1))+1))] * (count-1)
    else:
        budget = int(rng.integers(1, min(12, 26-sum(gaps))+1))
        for _ in range(budget):
            gaps[int(rng.integers(count-1))] += 1
        if len(set(gaps)) == 1:
            gaps[0] += 1
            gaps[-1] -= 1
    return gaps


def _sample_lengths(rng, count, reference, capacity, witness, gap_index):
    lengths = [reference] * count
    others = [i for i in range(count) if i != count//2]
    if witness:
        # All but one noncentral tooth are long: reference != length median.
        # The short tooth is at the large group's far end, so a long tooth
        # necessarily crosses the incorrect center-seam partition.
        short = count-1 if gap_index == 0 else 0
        for i in others:
            lengths[i] = int(rng.integers(reference+1, capacity+1))
        lengths[short] = int(rng.integers(1, reference))
        long_indices = [i for i in others if i != short]
        # Distinct excess heights demonstrate collective vertical alignment.
        large_group = [i for i in long_indices
                       if (i > gap_index if gap_index == 0 else i <= gap_index)]
        lengths[large_group[0]] = reference+1
        lengths[large_group[-1]] = reference+2
    else:
        mode = int(rng.integers(3))  # mixed, trimming-only, one trim
        order = [int(i) for i in rng.permutation(others)]
        if mode != 2:
            for i in others:
                state = int(rng.integers(3))
                if state == 0 and mode == 0:
                    lengths[i] = int(rng.integers(1, reference))
                elif state != 1:
                    lengths[i] = int(rng.integers(reference+1, capacity+1))
        lengths[order[0]] = int(rng.integers(reference+1, capacity+1))
        if mode == 0:
            lengths[order[1]] = int(rng.integers(1, reference))
    return lengths


def _draw_from_recipe(shape, bg, mark, seam, cols, lengths, colors, anchor):
    """Ground truth from sampled lists, without calling the input parser/solve."""
    inp = np.full(shape, bg, dtype=np.int8)
    out = np.full(shape, bg, dtype=np.int8)
    reference = lengths[len(cols)//2]
    left, right = [], []
    for col, span, color in zip(cols, lengths, colors):
        inp[anchor:anchor+span, col] = color
        out[anchor:anchor+reference, col] = color
        target = left if col < seam else right
        target.extend((r, col, color) for r in range(anchor+reference, anchor+span))
    inp[0, seam] = mark
    out[:anchor+reference, seam] = mark
    # Direct bounding-box raster placement, kept separate from _render.
    for is_right, group in enumerate((left, right)):
        if not group:
            continue
        rows, columns, _ = zip(*group)
        top = shape[0]-(max(rows)-min(rows)+1)
        side = shape[1]-(max(columns)-min(columns)+1) if is_right else 0
        for r, c, color in group:
            rr, cc = top+r-min(rows), side+c-min(columns)
            assert out[rr, cc] == bg, "offcut collision"
            out[rr, cc] = color
    return {"input": inp, "output": out}


def _sample_pair(rng, count, reference, turns, palette_mode, equal,
                 witness=False, seam_end=None, internal_seam=False, bg_avoid=None):
    mirror = bool(rng.integers(2))
    gaps = _sample_gaps(rng, count, equal, witness)
    span = sum(gaps)
    width = int(rng.integers(span+3, 31))
    first_col = int(rng.integers(1, width-span-1))
    cols = [first_col+int(v) for v in np.cumsum([0]+gaps)]
    if witness or seam_end is not None:
        # Specify the *visible* end after reflection, so the two essential
        # demonstrations really do put the seam at opposite ends.
        k = 0 if (int(seam_end) ^ int(mirror)) == 0 else count-2
    elif internal_seam:
        k = int(rng.integers(1, count-2))
    else:
        k = int(rng.integers(count-1))
    seam = int(rng.integers(cols[k]+1, cols[k+1]))
    capacity = int(rng.integers(reference+2, min(23, reference+12)+1))
    lengths = _sample_lengths(rng, count, reference, capacity, witness, k)
    height = int(rng.integers(max(lengths)+3, 31))
    anchor = int(rng.integers(2, height-max(lengths)))
    bg = int(rng.choice([c for c in range(10) if c != bg_avoid]))
    mark = int(rng.choice([c for c in range(10) if c != bg]))
    available = [c for c in range(10) if c not in (bg, mark)]
    size = (1 if palette_mode == "mono" else
            int(rng.integers(2, min(count, 8)+1)) if palette_mode == "multi" else
            int(rng.integers(1, min(count, 8)+1)))
    palette = [int(c) for c in rng.choice(available, size, replace=False)]
    # Every sampled color appears, then unbalanced repetition is allowed.
    colors = palette + [int(rng.choice(palette)) for _ in range(count-size)]
    rng.shuffle(colors)
    pair = _draw_from_recipe((height, width), bg, mark, seam, cols, lengths, colors, anchor)
    return {k: _orient(g, turns, mirror) for k, g in pair.items()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    ntrain, ntest = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # count, direction, or seam transfer
    sides = [int(v) for v in rng.permutation(4)]
    common = bool(rng.integers(2))
    outer_end = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_ref = first_count = first_bg = None
    for split, count_examples in (("train", ntrain), ("test", ntest)):
        for index in range(count_examples):
            training = split == "train"
            witness = training and index == 0
            if regime == 0:
                count = 5 if witness else 3 if training and index == 1 else int(rng.choice((3, 5) if training else (7, 9)))
            else:
                choices = (5, 7) if witness else (5, 7, 9) if regime == 2 and not training else (3, 5, 7, 9)
                if training and index == 1:
                    choices = tuple(c for c in choices if c != first_count)
                count = int(rng.choice(choices))
            choices = [v for v in range(2, 13) if not (training and index == 1 and v == first_ref)]
            reference = int(rng.choice(choices))
            if regime == 1:
                turns = sides[index % 2] if training else sides[2+index % 2]
            else:
                turns = sides[0] if common else sides[index % 4]
            palette = "mono" if witness else "multi" if training and index == 1 else "any"
            equal = False if witness else True if training and index == 1 else bool(rng.integers(2))
            # Opposite outer seams in the first two demonstrations establish
            # an independent axis, including with only two training examples.
            seam_end = (outer_end if witness else 1-outer_end if training and index == 1
                        else int(rng.integers(2)) if training and regime == 2 else None)
            for _ in range(24):
                pair = _sample_pair(rng, count, reference, turns, palette, equal,
                                    witness, seam_end, regime == 2 and not training,
                                    first_bg if training and index == 1 else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-input proposals exhausted")
            if witness:
                first_ref, first_count = reference, count
                first_bg = _parse(pair["input"])["bg"]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    cols, lengths, length = info["cols"], info["lengths"], info["reference"]
    distances = [abs(2*c-cols[0]-cols[-1]) for c in cols]
    geometric = distances.index(min(distances))
    groups = _tails(info)
    group_heights = [[span-length for col, span in zip(cols, lengths)
                      if span > length and int(col > info["seam"]) == side]
                     for side in range(2)]
    return {"count": len(cols), "reference": length,
            "gaps": tuple(np.diff(cols).tolist()), "shape": info["grid"].shape,
            "side": info["turns"], "bg": info["bg"],
            "palette": len(set(info["colors"])),
            "seam_gap": sum(c < info["seam"] for c in cols)-1,
            "extend": sum(span < length for span in lengths),
            "trim": sum(span > length for span in lengths),
            "nonmedian": sorted(lengths)[len(cols)//2] != length,
            "geometric_witness": (distances.count(min(distances)) == 1
                                  and lengths[geometric] != length),
            "group_teeth": tuple(len({c for r, c, color in g}) for g in groups),
            "collective_height_witness": any(len(set(v)) >= 2 for v in group_heights),
            "structure": (tuple(c-cols[0] for c in cols), tuple(lengths),
                          info["seam"]-cols[0])}


def _check_material(inp, out, info):
    """Check count/prefix/ray invariants separately from whole-output equality."""
    x = info["grid"]
    y = np.rot90(out, -info["turns"])
    a, length, seam = info["anchor"], info["reference"], info["seam"]
    assert np.all(y[:a+length, seam] == info["mark"])
    assert np.count_nonzero(y == info["mark"]) == a+length
    assert np.all(y[a+length:, seam] == info["bg"])
    for col, span, color in zip(info["cols"], info["lengths"], info["colors"]):
        assert np.all(y[a:a+length, col] == color)
        assert np.array_equal(x[a:a+min(span, length), col], y[a:a+min(span, length), col])
    middle = info["cols"][len(info["cols"])//2]
    assert np.array_equal(x[a:a+length, middle], y[a:a+length, middle])
    assert set(np.unique(y)) == set(np.unique(x))
    for color in set(info["colors"]):
        added = sum(max(0, length-span) for span, c in zip(info["lengths"], info["colors"]) if c == color)
        assert np.count_nonzero(y == color) == np.count_nonzero(x == color)+added
    # Tail bounding boxes touch the two far edges, with one vector per group.
    for right, cells in enumerate(_tails(info)):
        if not cells:
            continue
        dr = y.shape[0]-1-max(r for r, c, color in cells)
        dc = y.shape[1]-1-max(c for r, c, color in cells) if right else -min(c for r, c, color in cells)
        assert min(r+dr for r, c, color in cells) > a+length-1
        assert all(y[r+dr, c+dc] == color for r, c, color in cells)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(inp)
            assert np.array_equal(solve(inp), out), "incorrect conserved-offcut transformation"
            assert not np.array_equal(inp, out)
            _check_material(inp, out, info)
            if split == "train":
                train.append((info, _features(info)))
    fs = [f for info, f in train]
    assert len({f["count"] for f in fs}) > 1
    assert len({f["reference"] for f in fs}) > 1
    assert any(f["palette"] == 1 for f in fs) and any(f["palette"] > 1 for f in fs)
    assert any(len(set(f["gaps"])) == 1 for f in fs)
    assert any(f["seam_gap"] == 0 for f in fs)
    assert any(f["seam_gap"] == f["count"]-2 for f in fs)
    assert any(f["extend"] and f["trim"] and f["nonmedian"] and f["geometric_witness"] for f in fs)
    assert any(min(f["group_teeth"]) > 0 and max(f["group_teeth"]) >= 2 for f in fs)
    assert any(f["collective_height_witness"] for f in fs)
    # Mutants keep all other mechanisms intact, isolating each concept.
    assert any(not np.array_equal(_render(i, tail_mode="drop"), _render(i)) for i, f in train)
    assert any(not np.array_equal(_render(i, reference=sorted(i["lengths"])[len(i["cols"])//2]), _render(i)) for i, f in train)
    assert any(not np.array_equal(_render(i, tail_mode="individual"), _render(i)) for i, f in train)
    assert any(not np.array_equal(_render(i, seam=i["cols"][len(i["cols"])//2]), _render(i)) for i, f in train)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                g = _grid(pair[name])
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded streaming checks; returns small measurements and writes nothing."""
    rng = np.random.default_rng(117021)
    shapes, structures, signatures = set(), set(), set()
    sides = {"train": set(), "test": set()}
    counts = {"train": set(), "test": set()}
    references, backgrounds, palettes = set(), set(), set()
    pairs = offcut_groups = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = _digest(puzzle)
        assert digest == _digest(generate_puzzle(seed)), "seed nondeterminism"
        signatures.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                f = _features(info)
                sides[split].add(f["side"])
                counts[split].add(f["count"])
                shapes.add(pair["input"].shape)
                structures.add(f["structure"])
                references.add(f["reference"])
                backgrounds.add(f["bg"])
                palettes.add(f["palette"])
                pairs += 1
                offcut_groups += sum(n > 0 for n in f["group_teeth"])
        first = puzzle["train"][0]
        inp, out = first["input"], first["output"]
        for turns in range(4):
            for mirror in (False, True):
                assert np.array_equal(solve(_orient(inp, turns, mirror)), _orient(out, turns, mirror))
        permutation = rng.permutation(10).astype(np.int8)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        i = _parse(inp)
        canonical = i["grid"].copy()
        mid = len(i["cols"])//2
        col, a, length = i["cols"][mid], i["anchor"], i["reference"]
        # Only the central tooth changes. Original first witness has tails of
        # at least two cells, so either perturbation remains a valid member.
        if length > 2:
            canonical[a+length-1, col] = i["bg"]
        else:
            canonical[a+length, col] = i["colors"][mid]
        altered = solve(canonical)
        assert not np.array_equal(altered, _render(i))
        assert np.count_nonzero(altered == i["mark"]) != a+length
        # Move only the seam to the opposite outer gap. Ignore the guide in
        # this comparison so that a trivial marker displacement cannot pass.
        moved = i["grid"].copy()
        moved[0, i["seam"]] = i["bg"]
        k = len(i["cols"])-2 if i["seam"] < i["cols"][1] else 0
        moved[0, i["cols"][k]+1] = i["mark"]
        before, after = _render(i), solve(moved)
        before[before == i["mark"]] = i["bg"]
        after[after == i["mark"]] = i["bg"]
        assert not np.array_equal(before, after), "seam did not affect recovered payload"
        for ref in (min(i["lengths"]), max(i["lengths"]), i["lengths"][0], i["lengths"][-1]):
            assert not np.array_equal(_render(i, reference=ref), _render(i))
    assert len(signatures) == 42
    assert min(len(s) for s in sides.values()) == 4
    assert all(s == {3, 5, 7, 9} for s in counts.values())
    assert len(shapes) >= 50 and len(structures) >= 150
    assert references == set(range(2, 13)) and backgrounds == set(range(10))
    assert {1, 2, 3, 4}.issubset(palettes)
    legal = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(271828, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            legal += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal explicit counts accepted")
    broken = generate_puzzle(9)
    i = _parse(broken["test"][0]["input"])
    y = np.rot90(broken["test"][0]["output"], -i["turns"]).copy()
    y[i["anchor"], i["cols"][len(i["cols"])//2]] = i["bg"]
    broken["test"][0]["output"] = _orient(y, i["turns"])
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged reference prefix")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "canonical_structures": len(structures), "grid_shapes": len(shapes),
            "reference_lengths": sorted(references), "backgrounds": sorted(backgrounds),
            "payload_palette_sizes": sorted(palettes), "nonempty_offcut_groups": offcut_groups,
            "legal_count_combinations": legal, "dihedral_checks": 42*8,
            "palette_checks": 42, "reference_counterfactuals": 42,
            "payload_seam_counterfactuals": 42,
            "semantic_verification": "generator-owned checks only"}
