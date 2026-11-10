"""Compass docking establishes the endpoints of an equal, full-slice gap split.

Only NumPy and the standard library are used. Construction and validation do
not read evidence files. No intermediate execution traces are saved.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


_DIR = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
_SEEDS = (0, 292049457, 512686777, 508017965, 193030554, 307962688,
          392290941, 337539902, 88344680, 937218457, 625918330, 969374826,
          744776060, 433734799, 68747220, 195145974, 199084651, 762431180,
          217070175, 595866197, 42419110, 788546079, 341691619, 849233861,
          752205790, 691132761, 973815259, 777464300, 960800601, 400681212,
          384427058, 962850600, 561758239, 731928654, 257484892, 276773866,
          926192602, 339131960, 156237880, 163497988, 919054467, 23714793)


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or a.dtype.kind not in "iu" or np.any((a < 0) | (a > 9))):
        raise ValueError("grid must be an integer 1..30 by 1..30 array of colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _box(points):
    return (min(r for r, c in points), max(r for r, c in points) + 1,
            min(c for r, c in points), max(c for r, c in points) + 1)


def _move(cells, origin):
    return {(r + origin[0], c + origin[1]): v for (r, c), v in cells.items()}


def _d4(cells, t):
    result = {}
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result[r, c] = v
    r0, _, c0, _ = _box(result)
    return _move(result, (-r0, -c0))


def _signature(cells):
    return tuple(sorted(_d4(cells, 0).items()))


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in remaining:
                        remaining.remove(p)
                        group.add(p)
                        stack.append(p)
        result.append(group)
    return result


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _frame(a):
    """Read four bands independently of which adjacent band owns a corner."""
    h, w = a.shape
    north, south = int(a[0, w // 2]), int(a[-1, w // 2])
    west, east = int(a[h // 2, 0]), int(a[h // 2, -1])
    r0, r1 = 0, h
    while r0 < h and a[r0, w // 2] == north:
        r0 += 1
    while r1 > r0 and a[r1 - 1, w // 2] == south:
        r1 -= 1
    if r1 - r0 < 5:
        raise ValueError("missing framed interior")
    c0, c1 = 0, w
    while c0 < w and a[h // 2, c0] == west:
        c0 += 1
    while c1 > c0 and a[h // 2, c1 - 1] == east:
        c1 -= 1
    if c1 - c0 < 5 or min(r0, h - r1, c0, w - c1) < 1:
        raise ValueError("invalid frame")
    bands = dict(N=north, E=east, S=south, W=west)
    if len(set(bands.values())) != 4:
        raise ValueError("compass band identities must be distinct")
    if not (np.all(a[:r0, c0:c1] == north) and np.all(a[r1:, c0:c1] == south)
            and np.all(a[r0:r1, :c0] == west) and np.all(a[r0:r1, c1:] == east)):
        raise ValueError("frame sides must be constant bands")
    bg = int(a[r0, c0])
    if bg in bands.values():
        raise ValueError("interior corner must identify background")
    return (r0, r1, c0, c1), bands, bg


def _core(cells, band_colors, reg):
    body_colors = set(cells.values()) - band_colors - {reg}
    if len(body_colors) != 1:
        raise ValueError("a pad needs one body color")
    body = next(iter(body_colors))
    core = {p for p, v in cells.items() if v not in band_colors}
    r0, r1, c0, c1 = _box(core)
    if len(core) != (r1 - r0) * (c1 - c0):
        raise ValueError("pad core must be a solid rectangle")
    facing = {}
    for (r, c), v in cells.items():
        if v not in band_colors:
            continue
        options = [d for d, ok in (("N", r == r0 - 1 and c0 <= c < c1),
                                   ("S", r == r1 and c0 <= c < c1),
                                   ("W", c == c0 - 1 and r0 <= r < r1),
                                   ("E", c == c1 and r0 <= r < r1)) if ok]
        if len(options) != 1:
            raise ValueError("labels must protrude from a core side")
        facing[r, c] = options[0]
    directions = set(facing.values())
    if (len(directions) != 2 or directions in ({"N", "S"}, {"E", "W"})
            or len({cells[p] for p in facing}) != 2):
        raise ValueError("each pad needs two adjacent compass labels")
    for d in directions:
        if len({cells[p] for p in facing if facing[p] == d}) != 1:
            raise ValueError("one compass identity per side")
    return (r0, r1, c0, c1), body, facing


def _parse(a):
    rect, bands, bg = _frame(a)
    r0, r1, c0, c1 = rect
    interior = a[r0:r1, c0:c1]
    separators = []
    for r in range(r0, r1):
        line = a[r, c0:c1]
        if line[0] != bg and np.all(line == line[0]):
            separators.append(("V", r, int(line[0])))
    for c in range(c0, c1):
        line = a[r0:r1, c]
        if line[0] != bg and np.all(line == line[0]):
            separators.append(("H", c, int(line[0])))
    if len(separators) > 1:
        raise ValueError("at most one one-cell midpoint divider")
    sep = separators[0] if separators else None
    base = a.copy()
    base[r0:r1, c0:c1] = bg
    mask = np.zeros(a.shape, dtype=bool)
    mask[r0:r1, c0:c1] = interior != bg
    if sep:
        axis, coordinate, color = sep
        if color in set(bands.values()) | {bg}:
            raise ValueError("separator needs its own color")
        if axis == "H":
            mask[r0:r1, coordinate] = False
            base[r0:r1, coordinate] = color
        else:
            mask[coordinate, c0:c1] = False
            base[coordinate, c0:c1] = color
    sources, targets = [], []
    for points in _components(mask):
        cells = {p: int(a[p]) for p in points}
        if len(set(cells.values())) == 1:
            targets.append(cells)
        else:
            sources.append(cells)
    if len(sources) != 2 or len(targets) != 2:
        raise ValueError("expected two composite pads and two detached stencils")
    target_colors = {v for target in targets for v in target.values()}
    if len(target_colors) != 1:
        raise ValueError("stencils must share a registration color")
    reg = next(iter(target_colors))
    band_colors = set(bands.values())
    if reg in band_colors | {bg}:
        raise ValueError("registration color has a separate role")
    cores = [_core(s, band_colors, reg) for s in sources]
    bodies = {v[1] for v in cores}
    if len(bodies) != 2 or bodies & (band_colors | {reg, bg}):
        raise ValueError("pads must have distinct body colors")
    if sep and sep[2] in bodies | {reg}:
        raise ValueError("separator must be distinct from all other roles")
    for s in sources:
        if reg not in s.values():
            raise ValueError("every pad must contain registration marks")
    return dict(a=a, rect=rect, bands=bands, bg=bg, reg=reg, sep=sep,
                sources=sources, targets=targets, base=base)


def _placements(source, info, compass=True):
    choices, seen = [], set()
    r0, r1, c0, c1 = info["rect"]
    band_colors = set(info["bands"].values())
    for t in range(8):
        cells = _d4(source, t)
        _, _, directions = _core(cells, band_colors, info["reg"])
        if compass and any(cells[p] != info["bands"][d] for p, d in directions.items()):
            continue
        marks = {p for p, v in cells.items() if v == info["reg"]}
        anchor = min(marks)
        for j, target in enumerate(info["targets"]):
            if len(target) != len(marks):
                continue
            first = min(target)
            delta = first[0] - anchor[0], first[1] - anchor[1]
            placed = _move(cells, delta)
            if {p for p, v in placed.items() if v == info["reg"]} != set(target):
                continue
            if any(not (r0 <= r < r1 and c0 <= c < c1) for r, c in placed):
                continue
            if any(info["base"][p] != info["bg"] for p in placed):
                continue
            signature = j, tuple(sorted(placed.items()))
            if signature not in seen:
                seen.add(signature)
                box, body, facing = _core(placed, band_colors, info["reg"])
                choices.append(dict(target=j, cells=placed, box=box, color=body,
                                    facing=set(facing.values()), transform=t))
    return choices


def _geometry(x, y):
    """Return a common axis, ordered pads, and the open interval."""
    a, b = x["box"], y["box"]
    if a[:2] == b[:2] and (a[3] + 2 <= b[2] or b[3] + 2 <= a[2]):
        near, far = sorted((x, y), key=lambda p: p["box"][2])
        return "H", near, far, near["box"][3], far["box"][2]
    if a[2:] == b[2:] and (a[1] + 2 <= b[0] or b[1] + 2 <= a[0]):
        near, far = sorted((x, y), key=lambda p: p["box"][0])
        return "V", near, far, near["box"][1], far["box"][0]
    return None


def _dockings(info, compass=True, limit=2):
    options = [_placements(s, info, compass) for s in info["sources"]]
    found = {}
    for x in options[0]:
        for y in options[1]:
            if x["target"] == y["target"] or x["cells"].keys() & y["cells"].keys():
                continue
            geometry = _geometry(x, y)
            if geometry is None:
                continue
            axis, near, far, start, stop = geometry
            if compass:
                outward = ("W", "E") if axis == "H" else ("N", "S")
                if outward[0] not in near["facing"] or outward[1] not in far["facing"]:
                    continue
                common = near["facing"] & far["facing"]
                if len(common) != 1 or common & set(outward):
                    continue
            cells = {**x["cells"], **y["cells"]}
            # Labels and stencils cannot leave ink in the propagation interval.
            if any(start <= (c if axis == "H" else r) < stop for r, c in cells):
                continue
            sep = info["sep"]
            if sep and (sep[0] != axis or (stop - start) % 2 != 1
                        or sep[1] != (start + stop) // 2):
                continue
            signature = tuple(sorted(cells.items()))
            found[signature] = dict(axis=axis, near=near, far=far, start=start,
                                    stop=stop, gap=stop - start, cells=cells)
            if limit is not None and len(found) >= limit:
                return list(found.values())
    return list(found.values())


def _render(info, dock, fill=True):
    out = info["base"].copy()
    for p, v in dock["cells"].items():
        out[p] = v
    if fill:
        r0, r1, c0, c1 = info["rect"]
        n, start, stop = dock["gap"] // 2, dock["start"], dock["stop"]
        ca, cb = dock["near"]["color"], dock["far"]["color"]
        if dock["axis"] == "H":
            out[r0:r1, start:start + n] = ca
            out[r0:r1, stop - n:stop] = cb
        else:
            out[start:start + n, c0:c1] = ca
            out[stop - n:stop, c0:c1] = cb
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve by input-only segmentation, D4 registration, and gap bisection."""
    info = _parse(_grid(input_grid))
    answers = _dockings(info)
    if len(answers) != 1:
        raise ValueError("input must determine exactly one docked pixel map")
    return _render(info, answers[0])


def _pattern(rng, span, depth, size):
    points = {(int(rng.integers(span)), int(rng.integers(depth)))}
    while len(points) < size:
        frontier = sorted({(r + dr, c + dc) for r, c in points
                           for dr, dc in _DIR.values()
                           if 0 <= r + dr < span and 0 <= c + dc < depth} - points)
        points.add(frontier[int(rng.integers(len(frontier)))])
    return points


def _pad(rng, span, depth, body, reg, bands, side, transverse, pattern, simple):
    # Canonical construction is horizontal; the whole scene can be transposed.
    cells = {(r, c): body for r in range(span) for c in range(depth)}
    cells.update({p: reg for p in pattern})
    for d in (side, transverse):
        length = span if d in ("W", "E") else depth
        count = 1 if simple else int(rng.integers(1, min(2, length) + 1))
        k = length // 2 if simple else int(rng.integers(length - count + 1))
        for v in range(k, k + count):
            p = {"W": (v, -1), "E": (v, depth),
                 "N": (-1, v), "S": (span, v)}[d]
            cells[p] = bands[d]
    return cells


def _construct(rng, mode, axis, split, teaching=False):
    colors = [int(v) for v in rng.permutation(10)]
    bg, reg, b0, b1, north, east, south, west, divider = colors[:9]
    bands = dict(N=north, E=east, S=south, W=west)
    simple = mode == "even" and teaching
    span = int(rng.choice([3, 5])) if simple else int(rng.integers(3, 7))
    depths = (3, 3) if simple else tuple(int(v) for v in rng.integers(2, 4, size=2))
    possible_gaps = range(4, 10) if split == "train" else range(10, 14)
    gaps = [g for g in possible_gaps if g % 2 == (mode != "even")]
    gap = gaps[int(rng.integers(len(gaps)))]
    margins = [int(v) for v in rng.integers(1, 4, size=2)]
    left = margins[0] + 1
    right = left + depths[0] + gap
    length = right + depths[1] + 1 + margins[1]
    cross = int(rng.integers(max(14, span + 6), 23))
    tr = int(rng.integers(3, cross - span - 1))
    thick = [int(v) for v in rng.integers(1, 3, size=4)]
    tn, te, ts, tw = thick
    h, w = cross + tn + ts, length + tw + te
    if max(h, w) > 30:
        return None
    base = np.full((h, w), bg, dtype=np.int8)
    base[:tn] = north
    base[-ts:] = south
    base[tn:-ts, :tw] = west
    base[tn:-ts, -te:] = east
    common = str(rng.choice(["N", "S"]))
    if simple:
        patterns = [{(span // 2, 1)}] * 2
    else:
        patterns = [_pattern(rng, span, d, int(rng.integers(1, min(4, span * d // 2) + 1)))
                    for d in depths]
        if depths[0] == depths[1] and rng.random() < .3:
            patterns[1] = patterns[0].copy()
    finals = []
    for i, (d, body, side, axial) in enumerate(zip(depths, (b0, b1), ("W", "E"), (left, right))):
        pad = _pad(rng, span, d, body, reg, bands, side, common, patterns[i], simple)
        finals.append(_move(pad, (tn + tr, tw + axial)))
    start, stop = tw + left + depths[0], tw + right
    if mode == "marked":
        base[tn:h - ts, (start + stop) // 2] = divider
    inp = base.copy()
    targets = {p: reg for f in finals for p, v in f.items() if v == reg}
    for p, v in targets.items():
        inp[p] = v
    blocked = _halo(targets) | set(zip(*np.nonzero(base != bg)))
    sources = []
    for i, final in enumerate(finals):
        transforms = [int(t) for t in rng.permutation(8)]
        if teaching:
            transforms = [t for t in transforms if (0 < t < 4 if i == 0 else t >= 4)]
        placed = None
        for t in transforms:
            tile = _d4(final, t)
            ph, pw = _box(tile)[1], _box(tile)[3]
            origins = [(r, c) for r in range(tn + 1, h - ts - ph)
                       for c in range(tw + 1, w - te - pw)]
            rng.shuffle(origins)
            for origin in origins:
                trial = _move(tile, origin)
                if not trial.keys() & blocked:
                    placed = trial
                    break
            if placed is not None:
                break
        if placed is None:
            return None
        sources.append(placed)
        blocked.update(_halo(placed))
        for p, v in placed.items():
            inp[p] = v
    raw = [dict(box=_core(s, set(bands.values()), reg)[0]) for s in sources]
    if _geometry(*raw) is not None:
        return None  # Source positions themselves must not be usable endpoints.
    out = base.copy()
    for final in finals:
        for p, v in final.items():
            out[p] = v
    # Construction uses a per-cell distance comparison, independent of slicing.
    for c in range(start, stop):
        near_distance, far_distance = c - start + 1, stop - c
        if near_distance != far_distance:
            out[tn:h - ts, c] = b0 if near_distance < far_distance else b1
    if axis == "V":
        inp, out = inp.T.copy(), out.T.copy()
    try:
        info = _parse(inp)
        docks = _dockings(info)
        if len(docks) != 1 or not np.array_equal(_render(info, docks[0]), out):
            return None
        if simple and not _compass_changes_fill(info, docks[0]):
            return None
    except ValueError:
        return None
    return {"input": inp, "output": out}


def _compass_changes_fill(info, correct):
    key = (correct["axis"], correct["start"], correct["stop"],
           correct["near"]["color"], correct["far"]["color"])
    for alternative in _dockings(info, compass=False, limit=None):
        other = (alternative["axis"], alternative["start"], alternative["stop"],
                 alternative["near"]["color"], alternative["far"]["color"])
        if key != other:
            return True
    return False


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train, 2, 10, "num_train")
    ns = _integer(num_test, 1, 4, "num_test")
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    modes = ["even", "marked"] + (["odd"] if nt >= 3 else [])
    modes += [str(rng.choice(["even", "odd", "marked"])) for _ in range(nt - len(modes))]
    rng.shuffle(modes)
    first_axis = str(rng.choice(["H", "V"]))
    axes = [first_axis, "V" if first_axis == "H" else "H"]
    axes += [str(rng.choice(["H", "V"])) for _ in range(nt - 2)]
    tests = [str(rng.choice(["even", "marked"] if nt == 2 else ["even", "odd", "marked"]))
             for _ in range(ns)]
    plan = [("train", m, a) for m, a in zip(modes, axes)]
    plan += [("test", m, str(rng.choice(["H", "V"]))) for m in tests]
    puzzle, used = {"train": [], "test": []}, set()
    taught_even = False
    for split, mode, axis in plan:
        teaching = split == "train" and mode == "even" and not taught_even
        for _ in range(80):
            pair = _construct(rng, mode, axis, split, teaching)
            if pair is not None and _key(pair["input"]) not in used:
                break
        else:
            raise RuntimeError("bounded construction could not find a unique scene")
        if teaching:
            taught_even = True
        used.add(_key(pair["input"]))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _parse(inp)
    docks = _dockings(info)
    assert len(docks) == 1, "docking must determine one complete pixel map"
    dock = docks[0]
    r0, r1, c0, c1 = info["rect"]
    raw = [dict(box=_core(s, set(info["bands"].values()), info["reg"])[0])
           for s in info["sources"]]
    assert _geometry(*raw) is None, "source positions must not already be opposed"
    placed = _render(info, dock, fill=False)
    # Validate every slice by distances, including frame and outer margins.
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            expected = int(placed[r, c])
            k = c if dock["axis"] == "H" else r
            if r0 <= r < r1 and c0 <= c < c1 and dock["start"] <= k < dock["stop"]:
                da, db = k - dock["start"] + 1, dock["stop"] - k
                if da != db:
                    expected = dock["near" if da < db else "far"]["color"]
            assert out[r, c] == expected, "full-slice distance partition is wrong"
    assert np.array_equal(solve(inp), out)
    assert not np.array_equal(placed, out), "B must add nonempty fill"
    assert not np.array_equal(inp, out)
    for p, v in dock["cells"].items():
        assert out[p] == v, "the complete transformed pads must survive"
    if info["sep"]:
        axis, k, color = info["sep"]
        line = out[r0:r1, k] if axis == "H" else out[k, c0:c1]
        assert np.all(line == color), "midpoint separator ink must survive"
    # Every colored gap slice extends beyond the pad's transverse span.
    b = dock["near"]["box"]
    cross = r1 - r0 if dock["axis"] == "H" else c1 - c0
    span = b[1] - b[0] if dock["axis"] == "H" else b[3] - b[2]
    assert cross > span
    moves = []
    for source in info["sources"]:
        body = _core(source, set(info["bands"].values()), info["reg"])[1]
        final = next(p["cells"] for p in (dock["near"], dock["far"]) if p["color"] == body)
        ts = [t for t in range(8) if _signature(_d4(source, t)) == _signature(final)]
        assert ts
        moves.append("translation" if 0 in ts else "rotation" if min(ts) < 4 else "reflection")
    mode = "marked" if info["sep"] else "odd" if dock["gap"] % 2 else "even"
    return dict(info=info, dock=dock, mode=mode, axis=dock["axis"],
                gap=dock["gap"], shape=inp.shape, bg=info["bg"],
                depths=tuple((p["box"][3] - p["box"][2] if dock["axis"] == "H"
                              else p["box"][1] - p["box"][0])
                             for p in (dock["near"], dock["far"])), moves=moves)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen
            seen.add(key)
            records[split].append(_inspect_pair(pair))
    train = records["train"]
    assert {x["axis"] for x in train} == {"H", "V"}
    assert {"even", "marked"} <= {x["mode"] for x in train}
    if nt >= 3:
        assert "odd" in {x["mode"] for x in train}
    assert any(_compass_changes_fill(x["info"], x["dock"]) for x in train)
    assert {"rotation", "reflection"} <= {m for x in train for m in x["moves"]}
    assert max(x["gap"] for x in train) < min(x["gap"] for x in records["test"])
    if nt == 2:
        assert all(x["mode"] in {"even", "marked"} for x in records["test"])


def self_test():
    """Bounded streaming checks; return compact, reproducible diversity counts."""
    shapes, structures, palettes, backgrounds, moves, depths = (set() for _ in range(6))
    axes, modes, fingerprints, patterns = set(), Counter(), set(), set()
    pairs = 0
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                record = _inspect_pair(pair)
                a, info, dock = pair["input"], record["info"], record["dock"]
                digest.update(bytes(a.shape) + a.tobytes() + pair["output"].tobytes())
                shapes.add(a.shape)
                roles = {}
                structures.add((a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)))
                palettes.add(tuple(int(c) for c in np.unique(a)))
                backgrounds.add(info["bg"])
                moves.update(record["moves"])
                depths.update(record["depths"])
                axes.add((split, dock["axis"]))
                modes[record["mode"]] += 1
                pairs += 1
                for target in info["targets"]:
                    patterns.add(min(_signature(_d4({p: 1 for p in target}, t)) for t in range(8)))
                # No mutation, and all 10 color identities can be permuted.
                if seed == 0:
                    before = a.copy()
                    permutation = np.array([7, 3, 9, 1, 6, 0, 8, 2, 5, 4], dtype=np.int8)
                    assert np.array_equal(solve(permutation[a]), permutation[pair["output"]])
                    assert np.array_equal(a, before)
                    for t in range(8):
                        # Array geometry and compass roles move together.
                        x = np.fliplr(a) if t >= 4 else a
                        y = np.fliplr(pair["output"]) if t >= 4 else pair["output"]
                        assert np.array_equal(solve(np.rot90(x, t % 4)), np.rot90(y, t % 4))
        fingerprints.add(digest.hexdigest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 45 and len(structures) >= 180
    assert backgrounds == set(range(10)) and len(palettes) >= 8
    assert moves >= {"rotation", "reflection", "translation"}
    # There are only nine free connected polyominoes of sizes one through four.
    assert depths == {2, 3} and len(patterns) >= 8
    assert axes == {(s, a) for s in ("train", "test") for a in ("H", "V")}
    assert set(modes) == {"even", "odd", "marked"}
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9000 + nt * 10 + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=pairs, shapes=len(shapes), structures=len(structures),
                palettes=len(palettes), registration_shapes=len(patterns),
                gap_modes=dict(modes), unique_puzzles=len(fingerprints))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
