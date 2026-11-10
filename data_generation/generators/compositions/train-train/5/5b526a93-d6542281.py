"""Consensus stencils: retrieve damaged header icons, then intersect their ink.

All inference uses the input. No parent module, saved task, or generator state is
read by solve(). The two exemplars and all registration information are visible.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


HOST_SEEDS = (
    0, 94359050, 671977222, 543856916, 892097231, 485485361, 573743780,
    771180004, 913610208, 513154382, 581061490, 866067722, 245975080,
    598328891, 114833163, 59327335, 704311109, 185851696, 735997752,
    809599387, 496580584, 828410959, 518894030, 687532345, 23070515,
    385256996, 995160637, 416491225, 76290302, 357607155, 115770193,
    744609170, 393626491, 739704053, 496096059, 691755226, 922180720,
    943996705, 455130321, 156437411, 841528764, 323042304,
)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _border(h, w):
    return {(r, c) for r in range(h) for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)}


def _frame(grid, origin, shape, color):
    r0, c0 = origin
    for r, c in _border(*shape):
        grid[r0 + r, c0 + c] = color


def _ink(grid, origin, mapping):
    """mapping coordinates are relative to the inside of the frame."""
    r0, c0 = origin
    for (r, c), color in mapping.items():
        grid[r0 + 1 + r, c0 + 1 + c] = color


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for pos in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if pos in remaining:
                    remaining.remove(pos)
                    component.add(pos)
                    todo.append(pos)
        result.append(component)
    return result


def _normalize(mapping):
    mr = min(r for r, c in mapping)
    mc = min(c for r, c in mapping)
    return tuple(sorted((r - mr, c - mc, int(v)) for (r, c), v in mapping.items()))


def _placements(template, inside_shape):
    h = max(r for r, c, v in template) + 1
    w = max(c for r, c, v in template) + 1
    for dr in range(inside_shape[0] - h + 1):
        for dc in range(inside_shape[1] - w + 1):
            yield {(r + dr, c + dc): v for r, c, v in template}


def _embeds(small, large):
    """Does a colored translation of small fit within large's foreground?"""
    lookup = {(r, c): v for r, c, v in large}
    ar, ac, av = small[0]
    return any(all(lookup.get((sr + r - ar, sc + c - ac)) == sv
                       for sr, sc, sv in small)
               for r, c, v in large if v == av)


def _agreement(a, b):
    return {p: v for p, v in a.items() if b.get(p) == v}


def _analyze(input_grid):
    """Recover frames, maximal exemplar motifs, and each unique translation."""
    inp = _grid(input_grid)
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        raise ValueError("input background must occupy a strict majority")
    candidates = []
    for color in values:
        color = int(color)
        if color == bg:
            continue
        origins, shapes = [], set()
        for comp in _components(map(tuple, np.argwhere(inp == color))):
            top = min(r for r, c in comp)
            left = min(c for r, c in comp)
            h = max(r for r, c in comp) - top + 1
            w = max(c for r, c in comp) - left + 1
            if (h not in (5, 6) or w not in (5, 6)
                    or comp != {(top + r, left + c) for r, c in _border(h, w)}):
                break
            origins.append((int(top), int(left)))
            shapes.add((int(h), int(w)))
        else:
            if len(origins) >= 5 and len(shapes) == 1:
                candidates.append((color, sorted(origins), next(iter(shapes))))
    if len(candidates) != 1:
        raise ValueError("expected one color of identical intact header frames")
    frame_color, origins, shape = candidates[0]
    contents = {}
    for r0, c0 in origins:
        contents[r0, c0] = {(r, c): int(inp[r0 + 1 + r, c0 + 1 + c])
                            for r in range(shape[0] - 2)
                            for c in range(shape[1] - 2)
                            if inp[r0 + 1 + r, c0 + 1 + c] != bg}
    empty = [p for p in origins if not contents[p]]
    if len(empty) != 1:
        raise ValueError("one empty frame must mark the crossing of the arms")
    ar, ac = anchor = empty[0]
    rows = sorted({r for r, c in origins})
    cols = sorted({c for r, c in origins})
    if len(rows) < 3 or len(cols) < 3:
        raise ValueError("both axes must have at least three positions")
    if set(origins) != {(ar, c) for c in cols} | {(r, ac) for r in rows}:
        raise ValueError("frames must be exactly two complete perpendicular arms")
    if (any(b - a < shape[0] + 1 for a, b in zip(rows, rows[1:]))
            or any(b - a < shape[1] + 1 for a, b in zip(cols, cols[1:]))):
        raise ValueError("frames need at least one background cell between them")
    # Foreground elsewhere would be neither a header nor an instruction.
    footprint = np.zeros(inp.shape, dtype=bool)
    for r, c in origins:
        footprint[r:r + shape[0], c:c + shape[1]] = True
    if np.any((inp != bg) & ~footprint):
        raise ValueError("unexpected foreground outside the header cards")
    signatures = sorted({_normalize(m) for m in contents.values() if m})
    templates = [s for s in signatures
                 if not any(len(t) > len(s) and _embeds(s, t) for t in signatures)]
    if len(templates) != 2 or any(not 6 <= len(t) <= 8 for t in templates):
        raise ValueError("expected two maximal visible exemplars of 6..8 pixels")
    restored, types = {}, {}
    inside_shape = shape[0] - 2, shape[1] - 2
    options = [(i, m) for i, t in enumerate(templates)
               for m in _placements(t, inside_shape)]
    for p, clue in contents.items():
        if p == anchor:
            continue
        matches = [(i, m) for i, m in options
                   if all(m.get(q) == v for q, v in clue.items())]
        if len(matches) != 1:
            raise ValueError("header fragment must select one motif and translation")
        types[p], restored[p] = matches[0]
    return {"input": inp, "background": bg, "frame_color": frame_color,
            "shape": shape, "rows": rows, "cols": cols, "anchor": anchor,
            "contents": contents, "restored": restored, "types": types,
            "templates": templates, "options": options}


def _from_analysis(info, *, restore=True, operator="agreement", complete=True):
    out = info["input"].copy()
    ar, ac = info["anchor"]
    headers = info["restored"] if restore else info["contents"]
    if restore:
        for p, mapping in info["restored"].items():
            _ink(out, p, mapping)
    if complete:
        for r in info["rows"]:
            if r == ar:
                continue
            for c in info["cols"]:
                if c == ac:
                    continue
                a, b = headers[r, ac], headers[ar, c]
                if operator == "agreement":
                    mapping = _agreement(a, b)
                elif operator == "row":
                    mapping = a
                elif operator == "union":
                    mapping = {**a, **b}
                elif operator == "occupancy":
                    mapping = {p: v for p, v in a.items() if p in b}
                else:
                    raise ValueError("unknown operator")
                _frame(out, (r, c), info["shape"], info["frame_color"])
                _ink(out, (r, c), mapping)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore each damaged header; put equal-colored overlap at every crossing."""
    return _from_analysis(_analyze(input_grid))


def _sample_templates(rng, palette):
    all_coords = [(r, c) for r in range(3) for c in range(3)]
    result = []
    for _ in range(2000):
        n = int(rng.integers(6, 9))
        coords = sorted(all_coords[int(i)] for i in rng.choice(9, n, replace=False))
        if {r for r, c in coords} != {0, 1, 2} or {c for r, c in coords} != {0, 1, 2}:
            continue
        colors = list(palette) * 2
        colors += [int(rng.choice(palette)) for _ in range(n - len(colors))]
        rng.shuffle(colors)
        mapping = dict(zip(coords, colors))
        if result:
            other = {(r, c): v for r, c, v in result[0]}
            if (len(_agreement(mapping, other)) < 2
                    or sum(p in other and other[p] != v for p, v in mapping.items()) < 2
                    or not (mapping.keys() - other.keys())
                    or not (other.keys() - mapping.keys())):
                continue
        result.append(tuple((r, c, int(mapping[r, c])) for r, c in coords))
        if len(result) == 2:
            return result
    return None


def _clue(rng, truth, options, mode, joint=False):
    cells = sorted(truth)
    n = len(cells)
    sizes = list(range(2, n))
    if mode == "near":
        sizes.reverse()
    elif mode == "middle":
        sizes.sort(key=lambda k: abs(k - n / 2))
    for size in sizes:
        choices = list(combinations(cells, size))
        rng.shuffle(choices)
        for subset in choices:
            if joint and any(sum(m.get(p) == truth[p] for _, m in options) < 2
                             for p in subset):
                continue
            clue = {p: truth[p] for p in subset}
            if sum(all(m.get(p) == v for p, v in clue.items())
                   for _, m in options) == 1:
                return clue
    return None


def _axis(rng, size, count):
    slack = 30 - (count * size + count - 1)
    if slack < 0:
        raise ValueError("axis cannot fit")
    # Allocate a random amount of slack among gaps and asymmetric margins.
    extra = int(rng.integers(0, min(slack, 8) + 1))
    additions = [0] * (count + 1)
    for _ in range(extra):
        additions[int(rng.integers(count + 1))] += 1
    positions = [additions[0]]
    for i in range(count - 1):
        positions.append(positions[-1] + size + 1 + additions[i + 1])
    return positions, positions[-1] + size + additions[-1]


def _make_pair(rng, shape, counts, anchor_mode, test, easy=False):
    bg, fc, *palette = [int(x) for x in rng.choice(10, 4 + int(rng.integers(2)), replace=False)]
    templates = _sample_templates(rng, palette)
    if templates is None:
        return None
    rows, h = _axis(rng, shape[0], counts[0])
    cols, w = _axis(rng, shape[1], counts[1])
    if anchor_mode == "corner":
        ar, ac = rows[int(rng.choice((0, -1)))], cols[int(rng.choice((0, -1)))]
    elif anchor_mode == "plus":
        ar, ac = int(rng.choice(rows[1:-1])), int(rng.choice(cols[1:-1]))
    else:
        ar, ac = int(rng.choice(rows)), int(rng.choice(cols))
    vertical = [(r, ac) for r in rows if r != ar]
    horizontal = [(ar, c) for c in cols if c != ac]
    types = {}
    for arm in (vertical, horizontal):
        assignment = [0, 1] + [int(rng.integers(2)) for _ in arm[2:]]
        rng.shuffle(assignment)
        types.update(zip(arm, assignment))
    inside = shape[0] - 2, shape[1] - 2
    options = [(i, m) for i, t in enumerate(templates) for m in _placements(t, inside)]
    truth = {}
    for p, typ in types.items():
        possible = [m for i, m in options if i == typ]
        truth[p] = possible[0] if easy else possible[int(rng.integers(len(possible)))]
    if test:
        arm = vertical if rng.integers(2) else horizontal
        full = {next(p for p in arm if types[p] == i) for i in range(2)}
    else:
        typ = int(rng.integers(2))
        full = {next(p for p in vertical if types[p] == typ),
                next(p for p in horizontal if types[p] != typ)}
        if not easy:
            full.update(p for p in types if rng.random() < 0.15)
    inp = np.full((h, w), bg, dtype=np.int8)
    for p in [(ar, ac)] + list(types):
        _frame(inp, p, shape, fc)
    have_joint = False
    for p in types:
        if p in full:
            clue = truth[p]
        else:
            mode = "sparse" if test else ("sparse", "middle", "near")[int(rng.integers(3))]
            want_joint = shape == (6, 6) and not have_joint
            clue = _clue(rng, truth[p], options, mode, joint=want_joint)
            if clue is None:
                return None
            have_joint |= all(sum(m.get(q) == v for _, m in options) >= 2
                              for q, v in clue.items())
        _ink(inp, p, clue)
    # Build the intended answer from construction truth, independently of parser.
    out = inp.copy()
    for p, mapping in truth.items():
        _ink(out, p, mapping)
    for r, _ in vertical:
        for _, c in horizontal:
            _frame(out, (r, c), shape, fc)
            _ink(out, (r, c), _agreement(truth[r, ac], truth[ar, c]))
    try:
        info = _analyze(inp)
        _check_pair(info, out)
    except (ValueError, AssertionError):
        return None
    if shape == (6, 6) and not have_joint:
        return None
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            test = split == "test"
            easy = not test and index == 0
            if easy:
                shape, counts, mode = (5, 5), (3, 3), "corner"
            elif not test and index == 1:
                shape = (6, 6)
                counts = [(3, 3), (3, 4), (4, 3), (4, 4)][int(rng.integers(4))]
                mode = "plus"
            else:
                shape = (int(rng.integers(5, 7)), int(rng.integers(5, 7)))
                maxima = [min(5 if test else 4, 31 // (s + 1)) for s in shape]
                counts = tuple(int(rng.integers(3, maximum + 1)) for maximum in maxima)
                mode = "any"
            for _ in range(250):
                pair = _make_pair(rng, shape, counts, mode, test, easy)
                if pair is not None and _key(pair["input"]) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                raise RuntimeError("unable to construct an unambiguous consensus scene")
    return puzzle


def _check_pair(info, output_grid):
    inp, out = info["input"], _grid(output_grid)
    assert inp.shape == out.shape
    bg, fc = info["background"], info["frame_color"]
    templates = [{(r, c): v for r, c, v in t} for t in info["templates"]]
    assert all(max(r for r, c in t) == max(c for r, c in t) == 2 for t in templates)
    palettes = [set(t.values()) for t in templates]
    assert palettes[0] == palettes[1] and len(palettes[0]) in (2, 3)
    assert bg not in palettes[0] and fc not in palettes[0]
    assert all(list(t.values()).count(v) >= 2 for t in templates for v in palettes[0])
    a, b = templates
    assert len(_agreement(a, b)) >= 2
    assert sum(p in b and b[p] != v for p, v in a.items()) >= 2
    assert a.keys() - b.keys() and b.keys() - a.keys()
    assert np.array_equal(_from_analysis(info), out), "not the exact consensus closure"
    changed = inp != out
    assert np.any(changed) and np.all(inp[changed] == bg), "only add foreground"
    assert np.all(out[inp != bg] == inp[inp != bg])
    ar, ac = info["anchor"]
    # Both motif types occur on each arm: the product includes same and mixed pairs.
    for arm in ([p for p in info["types"] if p[1] == ac],
                [p for p in info["types"] if p[0] == ar]):
        assert {info["types"][p] for p in arm} == {0, 1}
    partials = [p for p, m in info["restored"].items()
                if len(info["contents"][p]) < len(m)]
    assert len(partials) >= 2
    assert all(len(info["contents"][p]) >= 2 for p in partials)
    # Ablations: each inherited mechanism changes the answer, including interiors.
    assert not np.array_equal(_from_analysis(info, complete=False), out)
    raw = _from_analysis(info, restore=False)
    interiors = np.zeros(inp.shape, dtype=bool)
    nonempty = proper = conflicts = 0
    for r in info["rows"]:
        if r == ar:
            continue
        for c in info["cols"]:
            if c == ac:
                continue
            a, b = info["restored"][r, ac], info["restored"][ar, c]
            common = _agreement(a, b)
            nonempty += bool(common)
            proper += bool(common) and len(common) < min(len(a), len(b))
            conflicts += any(p in b and b[p] != v for p, v in a.items())
            interiors[r + 1:r + info["shape"][0] - 1,
                      c + 1:c + info["shape"][1] - 1] = True
            # Each new boundary is complete; backgrounds at disagreements remain.
            assert all(out[r + dr, c + dc] == fc for dr, dc in _border(*info["shape"]))
    assert nonempty >= 2 and proper >= 1 and conflicts >= 1
    assert np.any(raw[interiors] != out[interiors]), "restoration must affect new icons"
    if info["shape"] == (6, 6):
        phases = {(min(r for r, c in m), min(c for r, c in m))
                  for m in info["restored"].values()}
        assert len(phases) >= 2, "teach that an icon can move inside its frame"
        canonical = out.copy()
        for r in info["rows"]:
            if r == ar:
                continue
            for c in info["cols"]:
                if c == ac:
                    continue
                a = {(rr, cc): v for rr, cc, v in _normalize(info["restored"][r, ac])}
                b = {(rr, cc): v for rr, cc, v in _normalize(info["restored"][ar, c])}
                canonical[r + 1:r + 5, c + 1:c + 5] = bg
                _ink(canonical, (r, c), _agreement(a, b))
        assert np.any(canonical[interiors] != out[interiors]), "placement must affect consensus"
    for alternative in ("row", "union", "occupancy"):
        assert not np.array_equal(_from_analysis(info, operator=alternative), out)
    return {"partials": len(partials), "nonempty": nonempty, "proper": proper}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _count(len(puzzle["train"]), "num_train", 2, 10)
    _count(len(puzzle["test"]), "num_test", 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            info = _analyze(pair["input"])
            key = _key(info["input"])
            assert key not in seen
            seen.add(key)
            _check_pair(info, pair["output"])
            ar, ac = info["anchor"]
            full = [p for p, m in info["restored"].items()
                    if len(info["contents"][p]) == len(m)]
            if split == "test":
                assert len(full) == 2
                assert all(r == ar for r, c in full) or all(c == ac for r, c in full)
            elif index < 2:
                assert any(r == ar for r, c in full) and any(c == ac for r, c in full)
            if split == "train" and index == 1:
                assert ar not in (info["rows"][0], info["rows"][-1])
                assert ac not in (info["cols"][0], info["cols"][-1])
                assert info["shape"] == (6, 6)
            if info["shape"] == (6, 6):
                # At least one fragment needs joint colored offsets: every one
                # of its cells alone admits two or more restorations.
                assert any(len(clue) < len(info["restored"][p])
                           and all(sum(m.get(q) == v for _, m in info["options"]) >= 2
                                   for q, v in clue.items())
                           for p, clue in info["contents"].items() if p != info["anchor"])


def self_test():
    """Bounded semantic, metamorphic, count, reproducibility and diversity checks."""
    shapes, frames, sizes, anchors, palettes, signatures = set(), set(), set(), set(), set(), set()
    offsets, gaps, densities = set(), set(), set()
    disconnected = joint = empty_crossings = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                frames.add(info["shape"])
                sizes.add((len(info["rows"]), len(info["cols"])))
                ar, ac = info["anchor"]
                anchors.add((info["rows"].index(ar), info["cols"].index(ac)))
                palettes.add((info["background"], info["frame_color"]))
                for axis in (info["rows"], info["cols"]):
                    gaps.update(b - a for a, b in zip(axis, axis[1:]))
                for p, m in info["restored"].items():
                    offsets.add((min(r for r, c in m), min(c for r, c in m)))
                    clue = info["contents"][p]
                    if len(clue) < len(m):
                        densities.add((len(clue), len(m)))
                        disconnected += len(_components(clue)) > 1
                        joint += all(sum(t.get(q) == v for _, t in info["options"]) >= 2
                                     for q, v in clue.items())
                empty_crossings += sum(not _agreement(info["restored"][r, ac],
                                                      info["restored"][ar, c])
                                       for r in info["rows"] if r != ar
                                       for c in info["cols"] if c != ac)
        # Geometry and palette commute with the intended transformation.
        pair = puzzle["train"][1]
        for transform in (lambda a: a[::-1], lambda a: a[:, ::-1], lambda a: a.T,
                          lambda a: (a.astype(int) + 3) % 10):
            assert np.array_equal(solve(transform(pair["input"])), transform(pair["output"]))
    assert len(signatures) == 42 and len(shapes) >= 30
    assert len(frames) == 4 and len(sizes) >= 6 and len(anchors) >= 8
    assert len(palettes) >= 40 and len(offsets) == 4 and len(gaps) >= 4
    assert len(densities) >= 10 and disconnected >= 20 and joint >= 42 and empty_crossings >= 1
    a, b = generate_puzzle(314159, 5, 3), generate_puzzle(314159, 5, 3)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s])
               for k in ("input", "output"))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(7000 + 10 * train + test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "canvas_shapes": len(shapes),
            "frame_shapes": len(frames), "lattice_sizes": len(sizes),
            "palette_roles": len(palettes), "clue_densities": len(densities),
            "disconnected_fragments": disconnected, "joint_fragments": joint,
            "empty_consensus_crossings": empty_crossings}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
