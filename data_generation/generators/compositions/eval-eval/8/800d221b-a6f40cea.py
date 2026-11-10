"""Occluded voting territories. Input-only decoding; no parent imports."""
from itertools import combinations
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    tr, te = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= tr <= 10 and 1 <= te <= 4 and tr > te):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return tr, te


def _bbox(points):
    rows, cols = zip(*points)
    return min(rows), max(rows), min(cols), max(cols)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _neighbors(points):
    return {(y + dy, x + dx) for y, x in points for dy, dx in _DIRS}


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            y, x = stack.pop()
            for dy, dx in _DIRS:
                q = y + dy, x + dx
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grid must be a 1..30 integer rectangle with colors 0..9")
    return a


def _collinear(a, b):
    """More than one shared collinear perimeter cell obscures circuit identity."""
    t, z, l, r = a
    u, v, c, d = b
    return (bool({t, z} & {u, v}) and min(r, d) - max(l, c) >= 1
            or bool({l, r} & {c, d}) and min(z, v) - max(t, u) >= 1)


def _read_input(input_grid):
    """Recover color roles, panels, contacts and complete boxes from pixels."""
    g = _grid(input_grid)
    h, w = g.shape
    bg = int(g[0, 0])
    assert np.all(g[0] == bg) and np.all(g[-1] == bg)
    assert np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)
    colors = tuple(map(int, np.unique(g)))
    points = {c: {tuple(map(int, p)) for p in np.argwhere(g == c)} for c in colors}
    frames = []
    for c in colors:
        if c == bg:
            continue
        box = _bbox(points[c])
        t, b, l, r = box
        if b - t < 2 or r - l < 2 or points[c] != _perimeter(box):
            continue
        inside = g[t + 1:b, l + 1:r]
        fill = int(inside[0, 0])
        if fill != c and np.all(inside == fill):
            frames.append((box, c, fill))
    assert len(frames) == 1, "require one complete opaque window"
    frame, frame_color, fill = frames[0]
    masked = _area(frame)
    t, b, l, r = frame
    interior = _area((t + 1, b - 1, l + 1, r - 1))
    remaining = [c for c in colors if c not in (bg, frame_color, fill)]
    interpretations = []
    for signals in combinations(remaining, 2):
        circuit_colors = [c for c in remaining if c not in signals]
        if not 2 <= len(circuit_colors) <= 5:
            continue
        panels = _components(np.isin(g, signals))
        if not 2 <= len(panels) <= 4:
            continue
        try:
            panel_boxes, labels, margins = [], [], []
            for group in panels:
                box = _bbox(group)
                assert group == _area(box) and not group & masked
                n0 = sum(int(g[p]) == signals[0] for p in group)
                n1 = len(group) - n0
                assert min(n0, n1) >= 1 and n0 != n1
                panel_boxes.append(box)
                labels.append(signals[int(n1 > n0)])
                margins.append(n0 - n1)
            boxes = [_bbox(points[c]) for c in circuit_colors]
            rings = [_perimeter(box) for box in boxes]
            panel_union = set().union(*panels)
            expected = set().union(*rings) - masked
            observed = set().union(*(points[c] for c in circuit_colors))
            assert expected == observed
            assert not any(_collinear(a, z) for a, z in combinations(boxes, 2))
            owners = []
            for color, box, ring in zip(circuit_colors, boxes, rings):
                y0, y1, x0, x1 = box
                assert 1 <= y0 < y1 < h - 1 and 1 <= x0 < x1 < w - 1
                assert y1 - y0 >= 2 and x1 - x0 >= 2
                assert points[color] <= ring and len(ring & interior) >= 2
                assert not ring & panel_union
                # All four extrema must survive the opaque mask AND crossings.
                assert _bbox(points[color]) == box
                for p in ring - masked:
                    actual = int(g[p])
                    assert actual in circuit_colors
                    assert p in rings[circuit_colors.index(actual)]
                touches = {i for i, panel in enumerate(panels)
                           if _neighbors(points[color]) & panel}
                full_touches = {i for i, panel in enumerate(panels)
                                if _neighbors(ring - masked) & panel}
                assert len(touches) == 1 and touches == full_touches
                owners.append(touches.pop())
            assert set(owners) == set(range(len(panels)))
            assert set(labels) == set(signals)
            interpretations.append(dict(background=bg, frame=frame,
                frame_color=frame_color, fill=fill, signals=signals,
                panels=panels, panel_boxes=panel_boxes, labels=labels,
                margins=margins, circuit_colors=circuit_colors,
                boxes=boxes, rings=rings, owners=owners, shape=g.shape))
        except AssertionError:
            continue
    assert len(interpretations) == 1, "require one panel/circuit interpretation"
    return interpretations[0]


def _fields(info):
    """Circuit, distinct-panel and pooled-raw-ballot vote fields."""
    t, b, l, r = info["frame"]
    shape = b - t - 1, r - l - 1
    score = np.zeros(shape, dtype=np.int16)
    coverage = np.zeros(shape, dtype=np.int16)
    raw = np.zeros(shape, dtype=np.int16)
    panel_masks = [np.zeros(shape, dtype=bool) for _ in info["panels"]]
    for box, owner in zip(info["boxes"], info["owners"]):
        y0, y1, x0, x1 = box
        ys = slice(max(y0, t + 1) - t - 1, min(y1, b - 1) - t)
        xs = slice(max(x0, l + 1) - l - 1, min(x1, r - 1) - l)
        label = info["labels"][owner]
        sign = 1 if label == info["signals"][0] else -1
        score[ys, xs] += sign
        coverage[ys, xs] += 1
        raw[ys, xs] += info["margins"][owner]
        panel_masks[owner][ys, xs] = True
    panel_score = np.zeros(shape, dtype=np.int16)
    for owner, mask in enumerate(panel_masks):
        panel_score[mask] += 1 if info["labels"][owner] == info["signals"][0] else -1
    return score, coverage, panel_score, raw


def _answer(info):
    score = _fields(info)[0]
    out = np.full(score.shape, info["fill"], dtype=np.int8)
    out[score > 0] = info["signals"][0]
    out[score < 0] = info["signals"][1]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Every convention is fixed geometry or visible color/contact evidence."""
    return _answer(_read_input(input_grid))


def _rand(rng, lo, hi):
    if lo > hi:
        return None
    return int(rng.integers(lo, hi + 1))


def _panel_boxes(rng, frame, shape, sides, witness):
    t, b, l, r = frame
    h, w = shape
    boxes = []
    for i, side in enumerate(sides):
        if witness and i == 0:
            ph, pw = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        elif side in (0, 1):
            ph, pw = int(rng.integers(2, 5)), int(rng.integers(3, 6))
        else:
            ph, pw = int(rng.integers(3, 6)), int(rng.integers(2, 5))
        if side == 0:
            y = _rand(rng, 1, t - ph - 1)
            x = _rand(rng, l + 1, r - pw)
        elif side == 1:
            y = _rand(rng, b + 2, h - ph - 1)
            x = _rand(rng, l + 1, r - pw)
        elif side == 2:
            y = _rand(rng, t + 1, b - ph)
            x = _rand(rng, 1, l - pw - 1)
        else:
            y = _rand(rng, t + 1, b - ph)
            x = _rand(rng, r + 2, w - pw - 1)
        if y is None or x is None:
            return None
        boxes.append((y, y + ph - 1, x, x + pw - 1))
    return boxes


def _anchored_box(rng, panel, h, w):
    """Sample a rectangle with one side immediately beside a panel."""
    pt, pb, pl, pr = panel
    anchor = int(rng.integers(8))
    if anchor < 4:
        y = (pt - 1, pt - 1, pb + 1, pb + 1)[anchor]
        if anchor % 2 == 0:
            t, b = y, _rand(rng, y + 2, h - 2)
        else:
            t, b = _rand(rng, 1, y - 2), y
        l, r = sorted(map(int, rng.choice(np.arange(1, w - 1), 2, replace=False)))
    else:
        x = (pl - 1, pl - 1, pr + 1, pr + 1)[anchor - 4]
        if anchor % 2 == 0:
            l, r = x, _rand(rng, x + 2, w - 2)
        else:
            l, r = _rand(rng, 1, x - 2), x
        t, b = sorted(map(int, rng.choice(np.arange(1, h - 1), 2, replace=False)))
    if None in (t, b, l, r):
        return None
    if not (1 <= t and b <= h - 2 and 1 <= l and r <= w - 2
            and b - t >= 2 and r - l >= 2):
        return None
    return t, b, l, r


def _circuits(rng, frame, shape, panels, weights, focus):
    h, w = shape
    mask = _area(frame)
    t, b, l, r = frame
    inside = _area((t + 1, b - 1, l + 1, r - 1))
    panel_cells = [_area(p) for p in panels]
    union = set().union(*panel_cells)
    halos = [_neighbors(p) for p in panel_cells]
    # Candidate pools and bounded backtracking avoid letting the first circuit
    # consume the only compatible contact/bounds for a shared panel's siblings.
    order = sorted(range(len(panels)), key=lambda i: -weights[i])
    pools = {}
    for owner in order:
        candidates, seen = [], set()
        for attempt in range(1400):
            box = _anchored_box(rng, panels[owner], h, w)
            if box is None or box in seen:
                continue
            seen.add(box)
            y0, y1, x0, x1 = box
            if focus is not None and not (y0 <= focus[0] <= y1 and x0 <= focus[1] <= x1):
                continue
            ring = _perimeter(box)
            visible = ring - mask
            if ring & union or len(ring & inside) < 2 or not visible:
                continue
            if _bbox(visible) != box or not visible & halos[owner]:
                continue
            if any(visible & halo for i, halo in enumerate(halos) if i != owner):
                continue
            candidates.append(box)
            if len(candidates) == 28:
                break
        if len(candidates) < weights[owner]:
            return None
        pools[owner] = candidates
    owners = [i for i in order for _ in range(weights[i])]
    boxes, indices, budget = [], [], [12000]

    def search(depth):
        if depth == len(owners):
            return True
        owner = owners[depth]
        start = indices[-1] + 1 if depth and owners[depth - 1] == owner else 0
        for index in range(start, len(pools[owner])):
            budget[0] -= 1
            if budget[0] < 0:
                return False
            box = pools[owner][index]
            if any(_collinear(box, old) for old in boxes):
                continue
            boxes.append(box)
            indices.append(index)
            if search(depth + 1):
                return True
            boxes.pop()
            indices.pop()
        return False

    return (boxes, owners) if search(0) else None


def _panel_pattern(rng, box, majority, minority, strength=None):
    t, b, l, r = box
    shape = b - t + 1, r - l + 1
    size = shape[0] * shape[1]
    if strength == "weak":
        count = (size - 1) // 2
    elif strength == "strong":
        count = 1
    else:
        count = int(rng.integers(1, (size - 1) // 2 + 1))
    style = int(rng.integers(3))
    if style == 0:
        indices = rng.permutation(size)
    elif style == 1:
        yy, xx = np.indices(shape)
        cy, cx = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
        indices = np.argsort((abs(yy - cy) + abs(xx - cx) + rng.random(shape) * .1).ravel())
    else:
        indices = np.arange(size).reshape(shape)
        if rng.integers(2):
            indices = indices.T
        indices = indices.ravel()
    flat = np.full(size, majority, dtype=np.int8)
    flat[indices[:count]] = minority
    return flat.reshape(shape)


def _construction_answer(frame, fill, signals, boxes, owners, labels):
    """Truth from complete scene records, before the circuits are masked."""
    t, b, l, r = frame
    out = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    for y in range(t + 1, b):
        for x in range(l + 1, r):
            votes = [labels[owner] for (y0, y1, x0, x1), owner in zip(boxes, owners)
                     if y0 <= y <= y1 and x0 <= x <= x1]
            a, z = votes.count(signals[0]), votes.count(signals[1])
            if a != z:
                out[y - t - 1, x - l - 1] = signals[int(z > a)]
    return out


def _make_pair(rng, palette, kind, split):
    witness = kind == "witness"
    tied_demo = kind == "tie"
    for layout_attempt in range(180):
        oh, ow = map(int, rng.integers(6, 11, size=2) if split == "train"
                     else rng.integers(11, 15, size=2))
        up, down, left, right = map(int, rng.integers(5, 8, size=4))
        h, w = oh + 2 + up + down, ow + 2 + left + right
        if h > 30 or w > 30:
            continue
        frame = up, up + oh + 1, left, left + ow + 1
        if witness:
            sides, weights = [0, 2, 3], [3, 1, 1]
        elif tied_demo:
            sides = list(map(int, rng.choice(4, 2, replace=False)))
            weights = [2, 2]
        else:
            nc = int(rng.integers(3 if split == "test" else 2, 6))
            npanels = int(rng.integers(2, min(4, nc) + 1))
            sides = list(map(int, rng.choice(4, npanels, replace=False)))
            weights = [1] * npanels
            for _ in range(nc - npanels):
                weights[int(rng.integers(npanels))] += 1
        panels = _panel_boxes(rng, frame, (h, w), sides, witness)
        if panels is None:
            continue
        if witness:
            p = panels[0]
            focus = (int(rng.integers(up + 2, up + oh - 1)),
                     int(rng.integers(p[2], p[3] + 1)))
        elif tied_demo:
            focus = (int(rng.integers(up + 2, up + oh - 1)),
                     int(rng.integers(left + 2, left + ow - 1)))
        else:
            focus = None
        routed = _circuits(rng, frame, (h, w), panels, weights, focus)
        if routed is None:
            continue
        boxes, owners = routed
        bg, border, special_fill, s0, s1 = map(int, palette[:5])
        signals = (s0, s1)
        fill = bg if witness else special_fill if tied_demo else int(rng.choice([bg, special_fill]))
        first_label = int(rng.choice(signals))
        other = next(c for c in signals if c != first_label)
        if witness:
            labels = [first_label, other, other]
        elif tied_demo:
            labels = [first_label, other]
        else:
            labels = list(map(int, rng.choice(signals, len(panels))))
            if len(set(labels)) == 1:
                labels[int(rng.integers(len(labels)))] = next(c for c in signals if c != labels[0])
        patterns = [_panel_pattern(rng, box, label, next(c for c in signals if c != label),
                                  "weak" if witness and i == 0 else "strong" if witness else None)
                    for i, (box, label) in enumerate(zip(panels, labels))]
        circuit_colors = list(map(int, palette[5:5 + len(boxes)]))
        inp = np.full((h, w), bg, dtype=np.int8)
        # A crossing can display either identity. Rectangular bounds survive it.
        for i in rng.permutation(len(boxes)):
            for p in _perimeter(boxes[int(i)]):
                inp[p] = circuit_colors[int(i)]
        for box, pattern in zip(panels, patterns):
            t, b, l, r = box
            inp[t:b + 1, l:r + 1] = pattern
        t, b, l, r = frame
        inp[t:b + 1, l:r + 1] = fill
        for p in _perimeter(frame):
            inp[p] = border
        out = _construction_answer(frame, fill, signals, boxes, owners, labels)
        try:
            info = _read_input(inp)
            assert sorted(info["boxes"]) == sorted(boxes)
            assert np.array_equal(_answer(info), out)
            score, cover, panel_score, raw = _fields(info)
            assert np.any(score != 0)
            if witness:
                assert np.any((score * panel_score < 0) & (score * raw < 0) & (cover == 5))
            if tied_demo:
                assert set(signals) <= set(map(int, np.unique(out)))
                assert np.any((score == 0) & (cover == 4))
        except AssertionError:
            continue
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        inp, out = np.rot90(inp, turns), np.rot90(out, turns)
        if flip:
            inp, out = np.fliplr(inp), np.fliplr(out)
        return {"input": inp.copy(), "output": out.copy()}
    raise RuntimeError(f"bounded search exhausted for {split}/{kind} voting circuits")


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    tr, te = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .35)
    base_palette = rng.permutation(10)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", tr), ("test", te)):
        for i in range(count):
            kind = "witness" if split == "train" and i == 0 else "tie" if split == "train" and i == 1 else "general"
            for _ in range(12):
                palette = base_palette if stable else rng.permutation(10)
                pair = _make_pair(rng, palette, kind, split)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate inputs after bounded retries")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _read_input(inp)
            expected = _answer(info)
            assert np.array_equal(expected, out), "incorrect spatial circuit majority"
            assert all(a < z for a, z in zip(out.shape, inp.shape))
            score, cover, panel_score, raw = _fields(info)
            assert np.any(score != 0), "omitting reconstruction must change the answer"
            assert np.any(out != info["fill"])
            # Replacing each panel's vote with its minority changes every
            # non-tied signal pixel, without altering any rectangle geometry.
            opposite = np.full(out.shape, info["fill"], dtype=np.int8)
            opposite[score > 0] = info["signals"][1]
            opposite[score < 0] = info["signals"][0]
            assert not np.array_equal(opposite, out)
            features[split].append((info, score, cover, panel_score, raw))
    assert any(np.any((score * ps < 0) & (score * raw < 0) & (cover == 5))
               for info, score, cover, ps, raw in features["train"]), "missing shared-panel witness"
    assert any(np.any((score == 0) & (cover == 4))
               and set(info["signals"]) <= set(map(int, np.unique(_answer(info))))
               for info, score, cover, ps, raw in features["train"]), "missing covered tie and both winners"
    # Enlarging the crop is the deliberate transfer axis; other counts overlap.
    assert max(max(_answer(info).shape) for info, *_ in features["train"]) <= 10
    assert min(min(_answer(info).shape) for info, *_ in features["test"]) >= 11


def self_test():
    """Bounded 42-seed checks, diversity, count contract, and counterfactuals."""
    shapes, output_shapes, layouts, palettes, topologies = set(), set(), set(), set(), set()
    panel_sizes, margins, circuit_counts, panel_counts, hidden_sizes = set(), set(), set(), set(), set()
    ties, gaps, crossings, pair_count = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                score, cover, _, _ = _fields(info)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                layouts.add((info["shape"], info["frame"], tuple(info["boxes"]), tuple(info["panel_boxes"])))
                palettes.add(tuple([info["background"], info["frame_color"], *info["signals"]]))
                weights = tuple(sorted(info["owners"].count(i) for i in range(len(info["panels"]))))
                topologies.add(weights)
                panel_sizes.update(map(len, info["panels"]))
                margins.update(map(abs, info["margins"]))
                circuit_counts.add(len(info["boxes"]))
                panel_counts.add(len(info["panels"]))
                mask = _area(info["frame"])
                hidden_sizes.update(len(ring & mask) for ring in info["rings"])
                ties += int(np.any((score == 0) & (cover > 0)))
                gaps += int(np.any(cover == 0))
                crossings += int(any(a & b for a, b in combinations(info["rings"], 2)))
                pair_count += 1
                # Pose and color covariance test on one pair per puzzle.
                if split == "test":
                    perm = np.random.default_rng(10000 + seed).permutation(10)
                    assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
                    assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
    assert len(layouts) == pair_count and len(shapes) >= 40 and len(output_shapes) >= 35
    assert len(palettes) >= 45 and len(topologies) >= 7
    assert circuit_counts == {2, 3, 4, 5} and panel_counts == {2, 3, 4}
    assert len(panel_sizes) >= 8 and len(margins) >= 8 and len(hidden_sizes) >= 15
    assert ties >= 42 and gaps >= 15 and crossings >= 100
    # All legal explicit counts, including maximum 14 total pairs.
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(2000 + tr * 10 + te, tr, te)
            assert len(p["train"]) == tr and len(p["test"]) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(981, 2, 1), generate_puzzle(981, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    b["test"][0]["output"][0, 0] = (int(b["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(b)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, pairs=pair_count, input_shapes=len(shapes), output_shapes=len(output_shapes),
                layouts=len(layouts), role_palettes=len(palettes), weights=sorted(topologies),
                panel_areas=sorted(panel_sizes), local_margins=sorted(margins),
                hidden_perimeter_sizes=sorted(hidden_sizes), covered_tie_pairs=ties,
                uncovered_pairs=gaps, crossing_pairs=crossings)
