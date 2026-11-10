"""Census-powered comb: connected objects grow teeth, slack absorbs motion.

Only NumPy and the standard library are used. No files are read or written by
the generator, solver, validator or bounded self-test.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 401501430, 849308005, 254206613, 800513968, 420279525,
              458485625, 811425155, 654937654, 901655024, 207258074,
              71348378, 447970650, 136267798, 984213362, 369124696,
              948427443, 797111981, 65573772, 672076076, 146540458,
              455007736, 141505081, 355892602, 732389306, 494348715,
              670788497, 753177329, 391059063, 974170946, 845606083,
              226759504, 745482244, 734386045, 336732831, 451074616,
              983090758, 263697905, 286470281, 779687481, 425831195,
              678726729)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train")
    b = _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _check(bool(np.all((0 <= a) & (a <= 9))), "colors outside 0..9")
    return a


def _parts(grid, bg):
    """Four-connected foreground regions, allowing colors to join in the press."""
    seen = np.zeros(grid.shape, dtype=bool)
    regions = []
    for r, c in np.argwhere(grid != bg):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, cells = [(r, c)], []
        while todo:
            rr, cc = todo.pop()
            cells.append((rr, cc))
            for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                        and not seen[nr, nc] and grid[nr, nc] != bg):
                    seen[nr, nc] = True
                    todo.append((nr, nc))
        regions.append(cells)
    return regions


def _box(cells):
    rr, cc = zip(*cells)
    return int(min(rr)), int(max(rr)), int(min(cc)), int(max(cc))


def _scaffold(shape, bg, structure, top, bottom, left, right, columns, labels):
    a = np.full(shape, bg, dtype=np.int8)
    a[top, left:right+1] = structure
    a[bottom, left:right+1] = structure
    a[top:bottom+1, left] = structure
    a[top:bottom+1, right] = structure
    for col, color in zip(columns, labels):
        a[bottom, col] = color
    return a


def _read(input_grid):
    """Recover orientation, sockets, shafts, ordered pins and census from pixels."""
    original = _grid(input_grid)
    values, frequencies = np.unique(original, return_counts=True)
    _check(np.count_nonzero(frequencies == frequencies.max()) == 1, "ambiguous background")
    bg = int(values[np.argmax(frequencies)])
    regions = _parts(original, bg)
    _check(bool(regions), "missing press")
    press = max(regions, key=len)
    mask = np.zeros(original.shape, dtype=bool)
    for r, c in press:
        mask[r, c] = True
    readings = []
    for turn in range(4):
        a = np.rot90(original, turn)
        rotated_mask = np.rot90(mask, turn)
        top, bottom, left, right = _box(np.argwhere(rotated_mask))
        s = int(a[top, left])
        if bottom-top < 5 or right-left < 6 or s == bg:
            continue
        if not (np.all(a[top, left:right+1] == s)
                and np.all(a[top:bottom+1, left] == s)
                and np.all(a[top:bottom+1, right] == s)):
            continue
        floor = a[bottom, left:right+1]
        if np.any(floor == bg):
            continue
        columns = (np.flatnonzero(floor != s) + left).tolist()
        if not (2 <= len(columns) <= 6 and columns[0] >= left+2
                and columns[-1] <= right-2 and min(np.diff(columns)) >= 2):
            continue
        try:
            labels = [int(a[bottom, col]) for col in columns]
            heights, stacks, gaps = [], [], []
            expected = _scaffold(a.shape, bg, s, top, bottom, left, right, columns, labels)
            depth = bottom-top-1
            for col in columns:
                h = 0
                while bottom-h-1 > top and a[bottom-h-1, col] == s:
                    h += 1
                _check(1 <= h <= depth-1, "missing or overgrown initial tooth")
                segment = a[top+1:bottom-h, col]
                occupied = np.flatnonzero(segment != bg)
                _check(len(occupied) > 0 and np.array_equal(occupied, np.arange(len(occupied))),
                       "pin must start at roof and be contiguous")
                seq = tuple(int(v) for v in segment[:len(occupied)])
                _check(seq[-1] != s, "pin tip must distinguish pin from tooth")
                heights.append(h)
                stacks.append(seq)
                gaps.append(depth-h-len(seq))
                expected[bottom-h:bottom, col] = s
                expected[top+1:top+1+len(seq), col] = seq
            _check(np.array_equal(a[rotated_mask], expected[rotated_mask]), "malformed press")
            _check(np.array_equal(expected != bg, rotated_mask), "extra cells in press")
            external = _parts(np.where(rotated_mask, bg, a), bg)
            objects = []
            for cells in external:
                colors = {int(a[r, c]) for r, c in cells}
                _check(len(colors) == 1 and 1 <= len(cells) <= 16, "external objects must be monochrome, area 1..16")
                objects.append((colors.pop(), cells))
            # A one-cell moat around the press and every token is explicit input syntax.
            labels_grid = np.full(a.shape, -1, dtype=int)
            labels_grid[rotated_mask] = 0
            for i, (_, cells) in enumerate(objects, 1):
                for r, c in cells:
                    labels_grid[r, c] = i
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                x = labels_grid[max(0, dr):a.shape[0]+min(0, dr), max(0, dc):a.shape[1]+min(0, dc)]
                y = labels_grid[max(0, -dr):a.shape[0]+min(0, -dr), max(0, -dc):a.shape[1]+min(0, -dc)]
                _check(not np.any((x >= 0) & (y >= 0) & (x != y)), "objects touch")
            counts = {color: sum(c == color for c, _ in objects) for color in set(labels)}
            areas = {color: sum(len(cells) for c, cells in objects if c == color) for color in set(labels)}
            _check(all(h+counts[c] <= depth-1 for h, c in zip(heights, labels)), "no pin survival space")
            readings.append(dict(grid=a, turn=turn, background=bg, structure=s,
                                 top=top, bottom=bottom, left=left, right=right,
                                 columns=columns, labels=labels, heights=heights,
                                 stacks=stacks, gaps=gaps, counts=counts, areas=areas,
                                 objects=objects, depth=depth))
        except AssertionError:
            continue
    _check(len(readings) == 1, "press must have exactly one directed interpretation")
    return readings[0]


def _simulate(d, counts=None, force_contact=False, freeze_pins=False):
    """Stepwise growth/pushing on input-derived sequences, then reverse rotation."""
    counts = d["counts"] if counts is None else counts
    bg, s, top, bottom = d["background"], d["structure"], d["top"], d["bottom"]
    result = _scaffold(d["grid"].shape, bg, s, top, bottom, d["left"], d["right"],
                       d["columns"], d["labels"])
    for col, color, h, seq in zip(d["columns"], d["labels"], d["heights"], d["stacks"]):
        tip = bottom-h
        pin = [(top+1+j, value) for j, value in enumerate(seq)]
        for _ in range(counts[color]):
            tip -= 1
            if not freeze_pins and pin[-1][0] >= tip:
                pin = [(r-1, v) for r, v in pin if r-1 > top]
                _check(bool(pin), "entire pin expelled")
        if force_contact and not freeze_pins:
            delta = tip-1-pin[-1][0]
            pin = [(r+delta, value) for r, value in pin if r+delta > top]
        for r, value in pin:
            result[r, col] = value
        result[tip:bottom, col] = s
    return np.rot90(result, -d["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The complete rule depends only on visible input cells."""
    return _simulate(_read(input_grid))


def _shape(rng, area=None):
    area = int(rng.integers(1, 9)) if area is None else int(area)
    if area <= 5 and rng.random() < .3:
        cells = {(0, c) for c in range(area)}
    else:
        cells = {(1, 1)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _place(rng, base, box, specs):
    a = base.copy()
    occupied = np.zeros(a.shape, dtype=bool)
    t, b, l, r = box
    occupied[max(0, t-1):b+2, max(0, l-1):r+2] = True
    for color, cells in sorted(specs, key=lambda p: -len(p[1])):
        sh, sw = max(r for r, c in cells)+1, max(c for r, c in cells)+1
        candidates = np.ones((a.shape[0]-sh+1, a.shape[1]-sw+1), dtype=bool)
        for dr, dc in cells:
            candidates &= ~occupied[dr:dr+candidates.shape[0], dc:dc+candidates.shape[1]]
        rr, cc = np.nonzero(candidates)
        if not len(rr):
            return None
        j = int(rng.integers(len(rr)))
        r0, c0 = int(rr[j]), int(cc[j])
        for dr, dc in cells:
            r, c = r0+dr, c0+dc
            a[r, c] = color
            occupied[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return a


def _make_pair(rng, labels, counts, bg, s, distractors, forced, teaching, turn, mirror):
    n = len(labels)
    depth = int(rng.integers(8 if teaching else 7, 13))
    steps = rng.integers(2, 5, n-1).tolist()
    # Arbitrary separated lane spacings; bounded width even with six lanes.
    il, ir = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    relative = [il]
    for step in steps:
        relative.append(relative[-1]+int(step))
    width = relative[-1]+ir+1
    heights, gaps, stacks = [], [], []
    palette = [c for c in range(10) if c != bg]
    tips = [c for c in palette if c != s]
    duplicate_seen = 0
    for i, color in enumerate(labels):
        h = int(rng.integers(1, min(4, depth-counts[color]-1)+1))
        if teaching:
            h = min(h, depth-6)
        gap = int(rng.integers(0, min(4, depth-h-2)+1))
        if teaching and color == teaching[0]:
            gap = (0 if duplicate_seen == 0 else 2) if teaching[1] == 1 else (0 if duplicate_seen == 0 else 1)
            duplicate_seen += 1
        elif teaching:
            gap = 1 if counts[color] else int(rng.integers(0, 3))
        length = depth-h-gap
        a, b = [int(v) for v in rng.choice(tips, 2, replace=False)]
        mode = int(rng.integers(3))
        if teaching or mode == 0:
            seq = [a if j % 2 == 0 else b for j in range(length)]
            if teaching and duplicate_seen == 1 and teaching[1] == 1:
                seq[0] = s
        elif mode == 1:
            prefix = int(rng.integers(length))
            seq = [s]*prefix + [a]*(length-prefix)
        else:
            seq = rng.choice(palette, length).astype(int).tolist()
            seq[-1] = a
        heights.append(h)
        gaps.append(gap)
        stacks.append(tuple(seq))
    specs = []
    for color in sorted(set(labels)):
        areas = forced.get(color, [None]*counts[color])
        specs.extend((color, _shape(rng, area)) for area in areas)
    specs.extend((color, _shape(rng)) for color in distractors)
    _check(len(specs) <= 16, "token packing budget")
    top, left = int(rng.integers(0, 4)), int(rng.integers(0, min(3, 30-width)+1))
    bottom, right = top+depth+1, left+width-1
    shape = (int(rng.integers(max(bottom+7, 22), 31)),
             int(rng.integers(max(right+1, 20), 31)))

    def render(shape, t, l, output=False):
        b, r = t+depth+1, l+width-1
        columns = [l+c for c in relative]
        a = _scaffold(shape, bg, s, t, b, l, r, columns, labels)
        for col, color, h, seq, gap in zip(columns, labels, heights, stacks, gaps):
            grown = h+counts[color] if output else h
            shift = max(0, counts[color]-gap) if output else 0
            kept = seq[shift:]
            a[t+1:t+1+len(kept), col] = kept
            a[b-grown:b, col] = s
        return a

    inp = None
    for attempt in range(8):
        if attempt == 4:
            shape = (30, 30)
        base = render(shape, top, left)
        inp = _place(rng, base, (top, bottom, left, right), specs)
        if inp is not None:
            break
    if inp is None:
        # Finite guaranteed fallback. Up to 16 area<=8 tokens in separated 3x3
        # boxes below a <=14-row housing; at least 21 boxes are available.
        shape, top, left = (30, 30), 0, 0
        bottom, right = depth+1, width-1
        inp = render(shape, top, left)
        slots = [(r, c) for r in range(bottom+2, 28, 4) for c in range(0, 28, 4)]
        _check(len(slots) >= len(specs), "finite fallback capacity")
        order = rng.permutation(len(slots))
        for (color, cells), index in zip(specs, order):
            r0, c0 = slots[int(index)]
            for j in range(len(cells)):
                inp[r0+j//3, c0+j % 3] = color
    out = render(shape, top, left, output=True)
    if mirror:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {"input": np.rot90(inp, turn).copy(), "output": np.rot90(out, turn).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    seed = _integer(seed, "seed")
    rng = np.random.default_rng(seed)
    anchor, y_color, z_color, novel = [int(c) for c in rng.choice(10, 4, replace=False)]
    size_transfer = bool(rng.integers(2))
    mixed = bool(rng.random() < .75)
    base_turn = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(nt+nq):
        train = index < nt
        teaching = None
        forced = {}
        if index < 2:
            other = y_color if index == 0 else z_color
            labels = [anchor, other, anchor]
            rng.shuffle(labels)
            counts = {anchor: index+1, other: 2 if index == 0 else 0}
            forced = {anchor: [4] if index == 0 else [1, 3], other: [2, 4] if index == 0 else []}
            teaching = (anchor, index+1)
            excluded = {anchor, y_color, z_color, novel}
        else:
            excluded = {novel} if not train else set()
        bg, s = [int(c) for c in rng.choice([c for c in range(10) if c not in excluded], 2, replace=False)]
        if index >= 2:
            n = int(rng.integers(2, 5)) if train else int(rng.integers(5, 7) if size_transfer else rng.integers(2, 6))
            pool = [c for c in range(10) if c not in (bg, s) and (not train or c != novel)]
            k = int(rng.integers(1, min(n, 4)+1))
            colors = [int(c) for c in rng.choice(pool, k, replace=False)]
            if not train and novel not in colors:
                colors[0] = novel
            labels = colors + [int(c) for c in rng.choice(colors, n-k)]
            rng.shuffle(labels)
            max_count = 3 if size_transfer else (2 if train else 5)
            counts = {c: int(rng.integers(max_count+1)) for c in colors}
            if not train and not size_transfer:
                counts[novel] = int(rng.integers(3, 6))
            while sum(counts.values()) > 12:
                color = max(counts, key=counts.get)
                counts[color] -= 1
        dpool = [c for c in range(10) if c not in (bg, *set(labels))]
        distractors = [int(c) for c in rng.choice(dpool, int(rng.integers(0, 4)))]
        if index == 1:
            distractors = [y_color] + distractors[:2]
        if not sum(counts.values()) and not distractors:
            distractors = [int(rng.choice(dpool))]
        turn = base_turn if not mixed else (base_turn if index == 0 else (base_turn+2) % 4 if index == 1 else int(rng.integers(4)))
        for _ in range(12):
            pair = _make_pair(rng, labels, counts, bg, s, distractors, forced, teaching, turn, bool(rng.integers(2)))
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                break
        else:
            raise RuntimeError("bounded distinct-input construction exhausted")
        seen.add(key)
        puzzle["train" if train else "test"].append(pair)
    return puzzle


def _metrics(d):
    shifts = [max(0, d["counts"][c]-g) for c, g in zip(d["labels"], d["gaps"])]
    return dict(shifts=shifts,
                stationary=any(v == 0 for v in shifts), moving=any(v > 0 for v in shifts),
                slack=any(0 < d["counts"][c] <= g for c, g in zip(d["labels"], d["gaps"])),
                repeated_effect=any(c == cc and a != b for c, a in zip(d["labels"], shifts)
                                    for cc, b in zip(d["labels"], shifts)),
                ordered=any(v > 0 and len(set(seq[v:])) >= 2 for v, seq in zip(shifts, d["stacks"])))


def validate_generated_puzzle(puzzle) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _check(inp.shape == out.shape, "shape changed")
            key = (inp.shape, inp.astype(np.int8).tobytes())
            _check(key not in seen, "duplicate input")
            seen.add(key)
            d = _read(inp)
            _check(np.array_equal(_simulate(d), out), "growth, contact, order, roof or erasure mismatch")
            _check(not np.array_equal(inp, out), "identity pair")
            details[split].append(d)
    training = details["train"]
    witnesses = [(c, d["counts"][c], d["areas"][c]) for d in training for c in set(d["labels"])]
    _check(any(c == cc and a == aa and n != nn for c, n, a in witnesses for cc, nn, aa in witnesses), "missing equal-area/different-count evidence")
    _check(any(n == nn and a != aa for c, n, a in witnesses for cc, nn, aa in witnesses), "missing same-count/different-area evidence")
    stats = [_metrics(d) for d in training]
    for feature in ("stationary", "moving", "slack", "repeated_effect", "ordered"):
        _check(any(m[feature] for m in stats), "missing teaching evidence: "+feature)
    _check({0, 1, 2} <= {v for d in training for v in d["counts"].values()}, "zero/one/two evidence")
    categories = {c for d in training for c in d["labels"]}
    distractors = {c for d in training for c, _ in d["objects"] if c not in d["labels"]}
    _check(bool(categories & distractors), "missing local role change")
    _check(any(set(d["labels"])-categories for d in details["test"]), "new test category missing")
    # Counterfactual ablations are allowed to violate the survival bound: they
    # are wrong candidate answers, never additional members of the family.
    _check(any(not np.array_equal(_simulate(d, {c: 0 for c in d["counts"]}), _simulate(d)) for d in training), "count concept inert")
    _check(any(not np.array_equal(_simulate(d, freeze_pins=True), _simulate(d)) for d in training), "pin mechanics inert")
    _check(any(not np.array_equal(_simulate(d, force_contact=True), _simulate(d)) for d in training), "slack concept inert")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """42 fixed seeds; bounded streaming metrics, equivariance and mutation checks."""
    signatures, shapes, lanes, directions, masks, backgrounds = set(), set(), set(), set(), set(), set()
    counts, gaps, growth_profiles = set(), set(), set()
    pairs = moving = stationary = repeated = ordered = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_digest(p))
        for split in ("train", "test"):
            for pair in p[split]:
                d = _read(pair["input"])
                m = _metrics(d)
                pairs += 1
                moving += m["moving"]
                stationary += m["stationary"]
                repeated += m["repeated_effect"]
                ordered += m["ordered"]
                shapes.add(pair["input"].shape)
                lanes.add(len(d["labels"]))
                directions.add(d["turn"])
                backgrounds.add(d["background"])
                counts.update(d["counts"].values())
                gaps.update(d["gaps"])
                growth_profiles.add(tuple(d["heights"]))
                for color, cells in d["objects"]:
                    t, b, l, r = _box(cells)
                    masks.add(tuple(sorted((rr-t, cc-l) for rr, cc in cells)))
        # Shape and role equivariance on one pair per seed, no stored grids.
        pair = p["train"][seed % len(p["train"])]
        mapping = np.array([(3*c+seed) % 10 for c in range(10)], dtype=np.int8)
        for k in range(4):
            inp = np.rot90(np.fliplr(mapping[pair["input"]]), k)
            expected = np.rot90(np.fliplr(mapping[pair["output"]]), k)
            _check(np.array_equal(solve(inp), expected), "rotation/reflection/recoloring")
    _check(len(signatures) == 42, "cross-seed duplicates")
    _check(lanes == set(range(2, 7)) and directions == set(range(4)), "lane/orientation coverage")
    _check(backgrounds == set(range(10)), "background-role coverage")
    _check(counts >= set(range(6)) and gaps >= set(range(5)), "count/slack coverage")
    _check(len(shapes) >= 35 and len(masks) >= 60 and len(growth_profiles) >= 80, "structural diversity")
    _check(min(moving, stationary, repeated, ordered) >= 42, "mechanical variation")
    _check(_digest(generate_puzzle(0)) == _digest(generate_puzzle(0)), "nondeterminism")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(13579, a, b)
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"] = bad["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted unchanged tokens and pins")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures), shapes=len(shapes),
                object_masks=len(masks), initial_profiles=len(growth_profiles),
                moving_pairs=moving, stationary_pairs=stationary,
                repeated_color_different_motion_pairs=repeated, ordered_survival_pairs=ordered)
