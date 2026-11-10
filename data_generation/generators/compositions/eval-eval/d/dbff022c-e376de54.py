"""Calibrated hatches: positional length normalization changes enclosure.

No parent imports, file reads, persistent state, or generation metadata in solve.
Run this file for a bounded, in-memory 42-seed self-test.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected an integer ARC grid of size 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, steps=EIGHT):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w and mask[y, x] and not seen[y, x]:
                    seen[y, x] = True
                    stack.append((y, x))
        result.append(cells)
    return result


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("Background must be the unique most frequent input color")
    return int(colors[np.argmax(counts)])


def _key_and_objects(a, bg):
    """The isolated, solid multicolor edge strip is the only key candidate."""
    candidates, objects = [], []
    for cells in _components(a != bg):
        ys, xs = zip(*cells)
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        sh, sw = y1 - y0 + 1, x1 - x0 + 1
        colors = {int(a[q]) for q in cells}
        if len(colors) == 1:
            objects.append(cells)
            continue
        if len(cells) != sh * sw:
            raise ValueError("The key must be a solid rectangle")
        lanes = []
        if sh == 2 and 3 <= sw <= 5:
            if y0 == 0:
                lanes.append((a[y0, x0:x1+1], a[y1, x0:x1+1], "top"))
            if y1 == a.shape[0] - 1:
                lanes.append((a[y1, x0:x1+1], a[y0, x0:x1+1], "bottom"))
        if sw == 2 and 3 <= sh <= 5:
            if x0 == 0:
                lanes.append((a[y0:y1+1, x0], a[y0:y1+1, x1], "left"))
            if x1 == a.shape[1] - 1:
                lanes.append((a[y0:y1+1, x1], a[y0:y1+1, x0], "right"))
        for outer, inner, side in lanes:
            sources, fills = list(map(int, outer)), list(map(int, inner))
            if len(set(sources)) == len(sources):
                candidates.append((dict(zip(sources, fills)), cells, side))
    if len(candidates) != 1 or len(objects) not in (3, 5, 7):
        raise ValueError("Expected one directed key and 3, 5, or 7 separated containers")
    mapping, cells, side = candidates[0]
    key_mask = np.zeros(a.shape, dtype=bool)
    for q in cells:
        key_mask[q] = True
    labels = np.full(a.shape, -1, dtype=np.int8)
    for i, points in enumerate(objects):
        for q in points:
            labels[q] = i
    return mapping, key_mask, labels, len(objects), side


def _transform(a, turns, mirror):
    a = np.rot90(a, turns)
    return np.fliplr(a) if mirror else a


def _undo(a, turns, mirror):
    return np.rot90(np.fliplr(a) if mirror else a, -turns).copy()


def _canonical_tanks(a, labels, count, bg):
    """Recognize top hatches, left anchors, and solid-bottom container bodies."""
    tanks = []
    for i in range(count):
        ys, xs = np.nonzero(labels == i)
        y, bottom, x = int(ys.min()), int(ys.max()), int(xs.min())
        if bottom - y < 2:
            return None
        color = int(a[ys[0], xs[0]])
        gate_x = xs[ys == y]
        length = len(gate_x)
        if length < 2 or set(map(int, gate_x)) != set(range(x, x + length)):
            return None
        body_x = xs[ys > y]
        width = int(body_x.max()) - x + 1
        if width < 3 or not np.all(a[y+1:bottom+1, x] == color):
            return None
        if not np.all(a[bottom, x:x+width] == color):
            return None
        # The body is rectangular material with top-opening cavities carved out.
        body = a[y+1:bottom+1, x:x+width]
        if np.any((body != bg) & (body != color)):
            return None
        if np.any(xs > x + max(width, length) - 1):
            return None
        holes = _components(body == bg, FOUR)
        if not holes or any(not any(r == 0 for r, c in hole) for hole in holes):
            return None
        # Every individual body's cavity touches its hatch row; no hidden holes.
        tanks.append({"row": y, "bottom": bottom, "anchor": x, "width": width,
                      "length": length, "color": color, "body_holes": holes})
    tanks.sort(key=lambda t: t["row"])
    if len({t["anchor"] for t in tanks}) != 1:
        return None
    if any(a["bottom"] + 1 >= b["row"] for a, b in zip(tanks, tanks[1:])):
        return None
    if len({t["length"] for t in tanks}) == 1:
        return None
    ref = tanks[count // 2]["length"]
    if any(t["anchor"] + ref > a.shape[1] for t in tanks):
        return None
    return tanks


def _normalize(a, tanks, bg, reference=None):
    out = a.copy()
    ref = tanks[len(tanks) // 2]["length"] if reference is None else reference
    for t in tanks:
        y, x = t["row"], t["anchor"]
        out[y, x:x+t["length"]] = bg
        if x + ref > a.shape[1]:
            raise ValueError("Normalized hatch would leave the canvas")
        if np.any(out[y, x:x+ref] != bg):
            raise ValueError("Normalized hatch would overwrite another object")
        out[y, x:x+ref] = t["color"]
    return out


def _fill(a, bg, mapping, key_mask):
    out, regions = a.copy(), []
    h, w = a.shape
    for cells in _components(a == bg, FOUR):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        boundary = set()
        for r, c in cells:
            for dr, dc in FOUR:
                y, x = r + dr, c + dc
                if a[y, x] != bg:
                    if key_mask[y, x]:
                        raise ValueError("The key cannot bound a cavity")
                    boundary.add(int(a[y, x]))
        if len(boundary) != 1:
            raise ValueError("An enclosed region must have one wall color")
        color = boundary.pop()
        regions.append((cells, color))
        if color in mapping:
            for q in cells:
                out[q] = mapping[color]
    return out, regions


def _interpret(input_grid):
    a = _grid(input_grid)
    bg = _background(a)
    mapping, key_mask, labels, count, side = _key_and_objects(a, bg)
    candidates = []
    for turns in range(4):
        for mirror in (False, True):
            g = _transform(a, turns, mirror)
            ids = _transform(labels, turns, mirror)
            tanks = _canonical_tanks(g, ids, count, bg)
            if tanks is None:
                continue
            km = _transform(key_mask, turns, mirror)
            try:
                normalized = _normalize(g, tanks, bg)
                filled, regions = _fill(normalized, bg, mapping, km)
            except ValueError:
                continue
            prediction = _undo(filled, turns, mirror)
            candidates.append((prediction, {"canonical": g, "normalized": normalized,
                               "regions": regions, "tanks": tanks, "background": bg,
                               "mapping": mapping, "key_mask": km, "side": side,
                               "turns": turns, "mirror": mirror}))
    if not candidates:
        raise ValueError("No consistent hatch/body/anchor interpretation")
    if any(not np.array_equal(candidates[0][0], out) for out, _ in candidates[1:]):
        raise ValueError("Ambiguous hatch direction or anchor side")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer everything from this input; equalize hatches, then fill new cavities."""
    return _interpret(input_grid)[0]


def _features(input_grid):
    out, a = _interpret(input_grid)
    g, n, ts = a["canonical"], a["normalized"], a["tanks"]
    bg, mapping, km = a["background"], a["mapping"], a["key_mask"]
    before, before_regions = _fill(g, bg, mapping, km)
    after, regions = _fill(n, bg, mapping, km)
    gate_mask = np.zeros(g.shape, dtype=bool)
    for t in ts:
        gate_mask[t["row"], :] = True
    before_cells = {q for cells, color in before_regions for q in cells if color in mapping}
    after_cells = {q for cells, color in regions for q in cells if color in mapping}
    ref = ts[len(ts)//2]["length"]
    positions = [t["row"] for t in ts]
    lengths = [t["length"] for t in ts]
    d = [abs(2*p - positions[0] - positions[-1]) for p in positions]
    nearest = [i for i, value in enumerate(d) if value == min(d)]
    geometric = (len(nearest) == 1 and nearest[0] != len(ts)//2
                 and lengths[nearest[0]] != ref)
    irregular = any(len(cells) != (max(r for r,c in cells)-min(r for r,c in cells)+1)
                    * (max(c for r,c in cells)-min(c for r,c in cells)+1)
                    for cells, color in regions if color in mapping)
    vented = any(n[t["row"]+1, t["anchor"]+t["width"]-1] == bg
                 and ref >= t["width"]-1 and t["color"] in mapping for t in ts)
    chain = any(color in mapping and mapping[color] in mapping
                and mapping[mapping[color]] != mapping[color] for cells, color in regions)
    self_map = any(color in mapping and mapping[color] == color for cells, color in regions)
    inert = any(color not in mapping for cells, color in regions)
    a.update({"output": out, "reference": ref, "lengths": tuple(lengths),
              "count": len(ts), "positions": tuple(positions), "geometric": geometric,
              "nonmedian": sorted(lengths)[len(ts)//2] != ref,
              "sealed": len(after_cells-before_cells), "opened": len(before_cells-after_cells),
              "fill_changes_without_B": int(np.count_nonzero((before != after) & ~gate_mask)),
              "missing_A_changes": int(np.count_nonzero(after != n)),
              "missing_lookup_changes": any(color in mapping and mapping[color] != color
                                             for cells, color in regions),
              "irregular": irregular, "vented": vented, "chain": chain,
              "self_map": self_map, "inert": inert,
              "multi": any(len(t["body_holes"]) > 1 for t in ts),
              "bay_counts": tuple(len(t["body_holes"]) for t in ts),
              "gaps": tuple(b-a for a,b in zip(positions,positions[1:]))})
    return a


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(name + " outside its legal range")
    return int(value)


def _body(rng, height, width, bays, vent=False, irregular=False, first_cut=None):
    """Return wall mask and known connected cavity masks, without solving a grid."""
    if first_cut is not None:
        cuts = [0, first_cut, width-1]
    else:
        # Distribute at least two columns per bay, leaving nonempty mouths.
        gaps = [2] * bays
        for _ in range(width-1-sum(gaps)):
            gaps[int(rng.integers(bays))] += 1
        cuts = [0] + list(map(int, np.cumsum(gaps)))
    wall = np.zeros((height, width), dtype=bool)
    wall[1:] = True
    cavities = []
    for index, (left, right) in enumerate(zip(cuts, cuts[1:])):
        cells = []
        depth = int(rng.integers(1, height-1))
        if irregular and index == 0:
            depth = height-2
        lo, hi = left+1, right-1
        previous = (lo, hi)
        for r in range(1, depth+1):
            a, b = lo, hi
            if r > 1 and hi > lo:
                if irregular and index == 0 and r == 2:
                    a += 1
                elif rng.random() < .75:
                    point = int(rng.integers(previous[0], previous[1]+1))
                    a = int(rng.integers(lo, point+1))
                    b = int(rng.integers(point, hi+1))
            previous = (a, b)
            cells.extend((r, c) for c in range(a, b+1))
        is_vented = vent and index == len(cuts)-2
        if is_vented:
            cells.append((1, width-1))
        for q in cells:
            wall[q] = False
        cavities.append((cells, right, is_vented))
    return wall, cavities


def _make_pair(rng, reference, count, lesson, transform):
    k = int(rng.integers(3, 6))
    sources = list(range(1, k+1))
    # Color 9 is available for an absent-from-both-lanes inert container.
    fills = {c: int(rng.integers(1, 9)) for c in sources}
    fills[1], fills[2] = (2 if lesson == 0 or rng.random() < .5 else k+1), k+1
    fills[3] = 3 if lesson == 0 or rng.random() < .35 else int(rng.integers(1, 9))
    key_order = list(map(int, rng.permutation(sources)))
    side = str(rng.choice(("top", "bottom", "left", "right")))
    heights, gaps = [3]*count, [1]*(count-1)
    height_budget = 30 - (3 if side in ("top", "bottom") else 0)
    if lesson == 0:
        # A deliberately unequal outer gap defeats geometric midpoint selection.
        gaps[0 if rng.random() < .5 else -1] += int(rng.choice((5, 7)))
    else:
        spare = height_budget - sum(heights) - sum(gaps)
        if lesson == 1:
            heights[0] += 1
            spare -= 1
        for _ in range(int(rng.integers(min(1, spare), min(10, spare)+1))):
            candidates = [(0,i) for i,h in enumerate(heights) if h < 6]
            candidates += [(1,i) for i,gap in enumerate(gaps) if gap < 5]
            which, i = candidates[int(rng.integers(len(candidates)))]
            (heights if which == 0 else gaps)[i] += 1
    middle = count//2
    lengths = [reference]*count
    others = [i for i in range(count) if i != middle]
    rng.shuffle(others)
    short_i, long_i = others[:2]
    mode = "mixed" if lesson in (0,1) else str(rng.choice(("mixed", "extend", "trim", "one_extend", "one_trim")))
    majority_short = bool(rng.integers(2))
    for i in others:
        if mode in ("one_extend", "one_trim"):
            continue
        if lesson != 0 and rng.random() < .20:
            continue
        short = majority_short if lesson == 0 else (mode == "extend" or (mode == "mixed" and rng.random() < .5))
        lengths[i] = int(rng.integers(2, reference-1)) if short else int(rng.integers(reference+2, reference+7))
    if mode in ("mixed", "extend", "one_extend"):
        lengths[short_i] = int(rng.integers(2, reference-1))
    if mode in ("mixed", "trim", "one_trim"):
        lengths[long_i] = int(rng.integers(reference+2, reference+7))
    if lesson == 1:
        short_i, long_i = 0, count-1
        lengths[short_i], lengths[long_i] = 2, reference+int(rng.integers(2, 7))
    pivot = short_i if lengths[short_i] < reference else long_i
    colors = [int(rng.choice(sources + [k+1,9])) for _ in range(count)]
    colors[middle], colors[pivot] = 2, 1
    special = {}
    if lesson == 0:
        colors[short_i] = colors[long_i] = 1
        remaining = [i for i in others if i not in (short_i,long_i)]
        colors[remaining[0]], colors[remaining[1]] = 3, k+1
        special[short_i], special[long_i] = (reference+1,1,False), (reference+2,1,False)
        # All four noncentral lengths differ from the reference; 3 vs 1 signs.
        for i in remaining:
            lengths[i] = int(rng.integers(2,reference-1)) if majority_short else int(rng.integers(reference+2,reference+7))
    elif lesson == 1:
        colors[-1] = 9
        special[pivot] = (reference+1,2,True)
    else:
        width = reference+1 if lengths[pivot] < reference else reference+2
        special[pivot] = (width, int(rng.integers(1, min(3,(width-1)//2)+1)), False)
    bodies, cavity_records, widths = [], [], []
    for i in range(count):
        if i in special:
            width, bays, vent = special[i]
        else:
            low = 5 if lesson == 2 and i == middle else 3
            width = int(rng.integers(low, reference+2))
            max_bays = min(3, (width-1)//2)
            bays = int(rng.integers(1,max_bays+1))
            if lesson == 2 and i == middle:
                bays = int(rng.integers(2,max_bays+1))
            vent = i != middle and lesson != 0 and rng.random() < .25
        if lesson == 0:
            bays, vent = 1, False
        if lesson == 1 and i == count-1:
            width, bays, vent = int(rng.integers(3,reference+2)), 1, False
        cut = int(rng.integers(3,reference-1)) if lesson == 1 and i == pivot else None
        body, cavities = _body(rng, heights[i], width, bays, vent,
                               irregular=lesson == 1 and i == pivot, first_cut=cut)
        bodies.append(body)
        cavity_records.append(cavities)
        widths.append(width)
    core_h = sum(heights) + sum(gaps)
    core_w = max(max(widths), max(lengths), reference)
    reserve_h = 3 if side in ("top", "bottom") else 0
    reserve_w = 3 if side in ("left", "right") else 0
    min_h, min_w = core_h+reserve_h, core_w+reserve_w
    h = int(rng.integers(min_h, min(30,min_h+5)+1))
    w = int(rng.integers(min_w, min(30,min_w+8)+1))
    # Sparse surrounding space makes background detection visible and color-free.
    body_ink = sum(int(np.count_nonzero(b)) for b in bodies) + sum(lengths) + 2*k
    if h*w < body_ink*2:
        w = 30
    row = (3 if side == "top" else 0) + int(rng.integers(h-min_h+1))
    col = (3 if side == "left" else 0) + int(rng.integers(w-min_w+1))
    inp = np.zeros((h,w), dtype=np.int8)
    out = inp.copy()
    for i, (body, length, color, cavities) in enumerate(zip(bodies,lengths,colors,cavity_records)):
        hh, ww = body.shape
        for grid, span in ((inp,length),(out,reference)):
            grid[row:row+hh,col:col+ww][body] = color
            grid[row,col:col+span] = color
        # Construct labels directly from known mouths, vents and cavity cells.
        # This does not call solve, its parser, or its flood-fill routine.
        if color in fills:
            for cells, threshold, vent in cavities:
                if reference >= threshold and not vent:
                    for r,c in cells:
                        out[row+r,col+c] = fills[color]
        row += heights[i] + (gaps[i] if i < count-1 else 0)
    axis = w if side in ("top", "bottom") else h
    offset = int(rng.integers(axis-k+1))
    for grid in (inp,out):
        for j, source in enumerate(key_order):
            p = offset+j
            if side == "top":
                grid[0,p], grid[1,p] = source, fills[source]
            elif side == "bottom":
                grid[-1,p], grid[-2,p] = source, fills[source]
            elif side == "left":
                grid[p,0], grid[p,1] = source, fills[source]
            else:
                grid[p,-1], grid[p,-2] = source, fills[source]
    palette = rng.permutation(10).astype(np.int8)
    turns, mirror = transform//2, bool(transform%2)
    return {role: _transform(palette[grid], turns, mirror).copy()
            for role, grid in (("input",inp),("output",out))}


def _fingerprint(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    transforms = list(map(int,rng.permutation(8)))
    references = list(map(int,rng.permutation(np.arange(5,13))))
    transfer = bool(rng.integers(2))
    puzzle, seen = {"train":[], "test":[]}, set()
    for split, total in (("train",num_train),("test",num_test)):
        for index in range(total):
            lesson = index if split == "train" and index < 2 else 2
            count = 5 if lesson == 0 else 3 if lesson == 1 else int(rng.choice((3,5,7)))
            if split == "test":
                reference = references[6+index%2]
            elif lesson < 2:
                reference = references[index]
            else:
                reference = int(rng.choice(references[:6]))
            if split == "train":
                transform = transforms[index%2] if index < 2 else int(rng.choice(transforms[:4]))
            else:
                transform = transforms[4+index%4] if transfer else int(rng.choice(transforms[:2]))
            for _ in range(48):
                pair = _make_pair(rng,reference,count,lesson,transform)
                signature = _fingerprint(pair["input"])
                if signature in seen:
                    continue
                try:
                    f = _features(pair["input"])
                except ValueError:
                    continue
                if not np.array_equal(f["output"],pair["output"]):
                    raise AssertionError("Constructed labels disagree with parsed enclosure")
                if not f["missing_A_changes"] or not f["fill_changes_without_B"]:
                    continue
                if lesson == 0 and not all(f[name] for name in
                        ("nonmedian","geometric","sealed","opened","chain","self_map","inert")):
                    continue
                if lesson == 1 and not all(f[name] for name in ("multi","vented","irregular","inert")):
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("Bounded hatch construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            signature = _fingerprint(inp)
            assert signature not in seen, "Duplicate input"
            seen.add(signature)
            f = _features(inp)
            assert np.array_equal(f["output"],out), "Incorrect hatch or fill result"
            assert f["missing_A_changes"] > 0, "No mapped cavity demonstrates parent A"
            assert f["fill_changes_without_B"] > 0, "Parent B must change cavity filling"
            assert f["missing_lookup_changes"], "The directed lookup must affect an answer"
            # Only editable hatch rows and enclosed background may change.
            g, n, km = f["canonical"], f["normalized"], f["key_mask"]
            filled, _ = _fill(n,f["background"],f["mapping"],km)
            assert np.array_equal(g[km],filled[km]), "Key changed"
            for t in f["tanks"]:
                y,x = t["row"],t["anchor"]
                assert np.all(n[y,x:x+f["reference"]] == t["color"])
                assert np.array_equal(g[y+1:t["bottom"]+1],n[y+1:t["bottom"]+1])
                common = min(t["length"],f["reference"])
                assert np.array_equal(g[y,x:x+common],n[y,x:x+common])
            assert np.array_equal(n[n != f["background"]],filled[n != f["background"]])
            records[split].append(f)
    train = records["train"]
    assert len({f["count"] for f in train}) > 1
    assert len({f["reference"] for f in train}) > 1
    assert {f["reference"] for f in train}.isdisjoint(
        {f["reference"] for f in records["test"]}), "Test reference lengths must be withheld"
    for name in ("nonmedian","geometric","sealed","opened","chain","self_map",
                 "inert","multi","vented","irregular"):
        assert any(f[name] for f in train), "Missing training distinction: " + name
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                shape, data = _fingerprint(pair[role])
                h.update(bytes(shape))
                h.update(data)
    return h.hexdigest()


def self_test():
    """One puzzle at a time: 42 seeds, metamorphic checks, and damaged labels."""
    shapes, references, counts, backgrounds, sides, directions = set(),set(),set(),set(),set(),set()
    signatures, layouts, bay_counts, patterns = set(),set(),set(),set()
    rng = np.random.default_rng(9471)
    seed_zero = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _digest(puzzle)
        signatures.add(signature)
        if seed in (0,17,41):
            assert signature == _digest(generate_puzzle(seed))
        if seed == 0:
            seed_zero = puzzle
        for split in ("train","test"):
            for pair in puzzle[split]:
                f = _features(pair["input"])
                shapes.add(pair["input"].shape)
                references.add(f["reference"])
                counts.add(f["count"])
                backgrounds.add(f["background"])
                sides.add(f["side"])
                directions.add((f["turns"],f["mirror"]))
                bay_counts.update(f["bay_counts"])
                layouts.add((f["count"],f["reference"],f["gaps"],f["lengths"],f["bay_counts"]))
                patterns.add((any(n < f["reference"] for n in f["lengths"]),
                              any(n > f["reference"] for n in f["lengths"])))
        # All eight cardinal hatch/body arrangements and arbitrary color roles.
        pair = puzzle["test"][0]
        palette = rng.permutation(10).astype(np.int8)
        assert np.array_equal(solve(palette[pair["input"]]),palette[pair["output"]])
        for turn in range(4):
            for mirror in (False,True):
                inp = _transform(pair["input"],turn,mirror)
                out = _transform(pair["output"],turn,mirror)
                assert np.array_equal(solve(inp),out)
        # Counterfactual one-step lookup: the same geometry gets different fills.
        f = _features(puzzle["train"][0]["input"])
        recursive = f["mapping"].copy()
        for c in recursive:
            if recursive[c] in f["mapping"]:
                recursive[c] = f["mapping"][recursive[c]]
        altered, _ = _fill(f["normalized"],f["background"],recursive,f["key_mask"])
        correct, _ = _fill(f["normalized"],f["background"],f["mapping"],f["key_mask"])
        assert not np.array_equal(altered,correct)
    assert len(signatures) == 42 and len(shapes) >= 40 and len(layouts) >= 150
    assert counts == {3,5,7} and len(references) >= 8
    assert backgrounds == set(range(10)) and len(sides) == 4 and len(directions) == 8
    assert bay_counts == {1,2,3} and {(True,True),(True,False),(False,True)} <= patterns
    for a,b in ((2,1),(10,1),(10,4)):
        assert len(generate_puzzle(510+a+b,a,b)["train"]) == a
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(3.5,1),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    broken = {s:[{k:v.copy() for k,v in p.items()} for p in seed_zero[s]] for s in ("train","test")}
    broken["test"][0]["output"][0,0] = (int(broken["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError,ValueError):
        pass
    else:
        raise AssertionError("Damaged output accepted")
    return {"seeds":42,"distinct_puzzles":len(signatures),"shapes":len(shapes),
            "structural_signatures":len(layouts),"references":sorted(references),
            "container_counts":sorted(counts),"bay_counts":sorted(bay_counts),
            "backgrounds":sorted(backgrounds),"key_sides":sorted(sides),
            "hatch_arrangements":len(directions),"metamorphic_checks":42*9,
            "independent_solver":False}


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",",":")))
