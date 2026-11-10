"""Addressed stripe fields: unique plan-row matching controls a Voronoi field.

Only NumPy and the standard library are used. No files, seeds, examples, or
generation state are consulted by solve(). No execution traces are persisted.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, cells = [(r, c)], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for a, b in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= a < h and 0 <= b < w and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    todo.append((a, b))
        result.append(cells)
    return result


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _matches(plan, sequence):
    n = len(sequence)
    return [(r, c) for r in range(plan.shape[0])
            for c in range(plan.shape[1]-n+1)
            if tuple(int(x) for x in plan[r, c:c+n]) == sequence]


def _recover(value):
    """Parse all roles and geometry from pixels, before doing any filling."""
    inp = _grid(value)
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[bg]) == 1, "background is ambiguous")
    singles, squares, keys = [], [], []
    for cells in _components(inp != bg):
        points = np.asarray(cells)
        r, c = (int(x) for x in points.min(axis=0))
        b, e = (int(x)+1 for x in points.max(axis=0))
        region = inp[r:b, c:e]
        if len(cells) == 1:
            singles.append((r, c))
        elif (region.shape == (2, 3) and len(cells) == 6
              and np.all(region[0] == region[0, 0])
              and np.all(region[1] == region[1, 0])
              and region[0, 0] != region[1, 0]):
            keys.append((r, c, b, e, int(region[0, 0]), int(region[1, 0])))
        else:
            side = b-r
            _require(2 <= side <= 5 and e-c == side and len(cells) == side*side
                     and np.all(region == region[0, 0]), "unexpected component")
            squares.append((r, c, side, int(region[0, 0])))
    _require(len(keys) == 1 and len(singles) >= 4 and len(squares) >= 2,
             "need one key, a miniature, and at least two squares")
    key = keys[0]
    guide, junction = key[4:]
    points = np.asarray(singles)
    mr, mc = (int(x) for x in points.min(axis=0))
    mb, me = (int(x)+1 for x in points.max(axis=0))
    _require(np.all((points-(mr, mc)) % 2 == 0), "miniature pitch must be two")
    _require(all(np.any(np.diff(np.unique(points[:, d])) == 2) for d in (0, 1)),
             "miniature needs adjacent pitch witnesses")
    plan = inp[mr:mb:2, mc:me:2].copy()
    h, w = plan.shape
    _require(2 <= h <= 4 and 3 <= w <= 7, "unsupported miniature dimensions")
    sampled = np.zeros((mb-mr, me-mc), dtype=bool)
    sampled[::2, ::2] = True
    _require(np.all(inp[mr:mb, mc:me][~sampled] == bg), "nonempty swatch gutter")
    _require(np.count_nonzero(plan != bg) == len(singles), "unexpected singleton")
    palette = set(int(x) for x in plan.flat)-{bg}
    _require(not ({guide, junction} & (palette | {bg})), "color roles overlap")
    squares.sort(key=lambda s: s[1])
    side = squares[0][2]
    _require(len({s[0] for s in squares}) == 1 and all(s[2] == side for s in squares),
             "squares must form one horizontal equal-size run")
    deltas = np.diff([s[1] for s in squares])
    pitch = int(deltas[0])
    _require(np.all(deltas == pitch) and 1 <= pitch-side <= 4, "invalid square stride")
    seq = tuple(s[3] for s in squares)
    matches = _matches(plan, seq)
    _require(len(matches) == 1, "seed sequence must have exactly one plan-row match")
    r, c = matches[0]
    br, bc = squares[0][0]-r*pitch, squares[0][1]-c*pitch
    height, width = (h-1)*pitch+side, (w-1)*pitch+side
    rect = (br, bc, br+height, bc+width)
    _require(0 <= br < br+height <= inp.shape[0] and 0 <= bc < bc+width <= inp.shape[1],
             "full lattice does not fit")
    mini = (mr, mc, mb, me)
    _require(not _overlap(rect, mini) and not _overlap(rect, key[:4])
             and not _overlap(mini, key[:4]), "structures overlap")
    row = plan[r]
    occupied = np.flatnonzero(row != bg)
    labels = tuple(int(row[j]) for j in occupied)
    _require(len(labels) >= 2 and len(set(labels)) >= 2, "need two distinct source labels")
    middle = (side-1)//2
    positions = tuple(int(j)*pitch+middle for j in occupied)
    guide_y = r*pitch+middle
    # Reconstruct the complete input, including all expected empty target cells.
    rebuilt = np.full(inp.shape, bg, dtype=np.int8)
    rebuilt[mr:mb:2, mc:me:2] = plan
    rebuilt[key[0], key[1]:key[3]] = guide
    rebuilt[key[0]+1, key[1]:key[3]] = junction
    for y, x, s, color in squares:
        rebuilt[y:y+s, x:x+s] = color
    _require(np.array_equal(inp, rebuilt), "unexplained pixels")
    target = np.full((height, width), bg, dtype=np.int8)
    for k, color in enumerate(seq):
        target[r*pitch:r*pitch+side, (c+k)*pitch:(c+k)*pitch+side] = color
    _require(np.array_equal(inp[br:br+height, bc:bc+width], target), "target contains extra pixels")
    return dict(background=bg, guide=guide, junction=junction, plan=plan,
                miniature=mini, key=key[:4], rect=rect, base=(br, bc),
                side=side, pitch=pitch, match=(r, c), sequence=seq,
                positions=positions, labels=labels, guide_y=guide_y,
                shape=inp.shape, field_shape=(height, width))


def _field(height, width, positions, labels, guide_y, guide, junction, invert=True):
    """Nearest-distance implementation used by solve and the sampled renderer."""
    out = np.empty((height, width), dtype=np.int8)
    for x in range(width):
        distances = [abs(x-p) for p in positions]
        minimum = min(distances)
        nearest = {labels[i] for i, d in enumerate(distances) if d == minimum}
        if minimum == 0 and invert:
            out[:, x] = junction
            out[guide_y, x] = guide
        elif len(nearest) > 1:
            out[:, x] = junction
        else:
            out[:, x] = next(iter(nearest))
            out[guide_y, x] = junction
    return out


def _interval_field(data):
    """Validation uses adjacent intervals, with no nearest-distance search."""
    h, w = data["field_shape"]
    pos, labels = data["positions"], data["labels"]
    values = np.empty(w, dtype=np.int8)
    values[:pos[0]] = labels[0]
    for i, (left, right) in enumerate(zip(pos, pos[1:])):
        for x in range(left+1, right):
            if 2*x < left+right:
                values[x] = labels[i]
            elif 2*x > left+right:
                values[x] = labels[i+1]
            else:
                values[x] = labels[i] if labels[i] == labels[i+1] else data["junction"]
    values[pos[-1]+1:] = labels[-1]
    values[list(pos)] = data["junction"]
    out = np.tile(values, (h, 1))
    out[data["guide_y"], :] = data["junction"]
    out[data["guide_y"], list(pos)] = data["guide"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    d = _recover(inp)
    out = inp.copy()
    r, c, b, e = d["rect"]
    out[r:b, c:e] = _field(*d["field_shape"], d["positions"], d["labels"],
                            d["guide_y"], d["guide"], d["junction"])
    return out


def _options(kind, regime, avoid=None):
    options = []
    for h in range(2, 5):
        for w in range(3, 8):
            for side in range(2, 6):
                if avoid and (side == avoid["side"] or w == avoid["plan"].shape[1]):
                    continue
                if kind in ("even", "odd") and side > 4:
                    continue
                if kind == "test" and w < (5 if regime == "sparse" else 4):
                    continue
                if kind == "test" and regime == "scale" and side < 4:
                    continue
                for gutter in range(1, 5):
                    p = side+gutter
                    if kind == "even" and p % 2:
                        continue
                    if kind == "odd" and p % 2 != 1:
                        continue
                    target = ((h-1)*p+side, (w-1)*p+side)
                    module = (2*h+2, 2*w-1)
                    if max(target) > 30:
                        continue
                    for axis in (0, 1):
                        if target[axis]+module[axis]+1 <= 30:
                            options.append((h, w, side, gutter, axis))
    return options


def _schedule(rng, w, palette, bg, kind, regime):
    a, b = palette[:2]
    if kind in ("even", "odd"):
        row = rng.choice(palette, w).astype(np.int8)
        row[:3] = (a, a, b)
        if w > 3 and rng.random() < .5:
            row[3:] = bg
        if rng.integers(2):
            row = row[::-1].copy()
        return row
    minimum = 3 if kind == "test" else 2
    count = int(rng.integers(minimum, w+1))
    if kind == "test" and regime == "sparse":
        count = min(count, w-1)
    indices = sorted(int(x) for x in rng.choice(w, count, replace=False))
    mode = int(rng.integers(6))
    labels = [int(x) for x in rng.choice(palette, count)]
    labels[:2] = (a, b)
    if mode == 0:
        labels = [palette[i % len(palette)] for i in range(count)]
    elif mode == 1 and count >= 3:
        labels[:3] = (a, a, b)
    elif mode == 2 and count >= 3:
        labels[:3] = (a, b, a)
    elif mode == 3 and count >= 4:
        labels[:4] = (a, a, b, b)
    elif mode == 4 and count >= 4:
        labels[:4] = (a, a, a, b)
    if rng.integers(2):
        labels.reverse()
    row = np.full(w, bg, dtype=np.int8)
    row[indices] = labels
    return row


def _plan(rng, h, w, palette, bg, kind, regime):
    partial = kind in ("even", "odd", "test")
    for _ in range(100):
        plan = rng.choice(palette, (h, w)).astype(np.int8)
        plan[rng.random((h, w)) < .17] = bg
        r = int(rng.integers(h))
        plan[r] = _schedule(rng, w, palette, bg, kind, regime)
        if any(np.all(edge == bg) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
            continue
        occupied = np.argwhere(plan != bg)
        if not all(np.any(np.diff(np.unique(occupied[:, d])) == 1) for d in (0, 1)):
            continue
        sources = int(np.count_nonzero(plan[r] != bg))
        choices = []
        for length in range(2, w+1):
            if partial and length >= sources:
                continue
            for c in range(w-length+1):
                seq = tuple(int(x) for x in plan[r, c:c+length])
                if bg not in seq and len(_matches(plan, seq)) == 1:
                    choices.append((r, c, length))
        if choices:
            if kind == "test" and regime == "ordinary":
                shortest = min(x[2] for x in choices)
                choices = [x for x in choices if x[2] == shortest]
            match = choices[int(rng.integers(len(choices)))]
            return plan, match
    # Constructive fallback has one unique [a,a] segment and an unseen b.
    # It preserves both parity anchors and partial-source teaching.
    plan = np.full((h, w), palette[1], dtype=np.int8)
    r = int(rng.integers(h))
    plan[r, :3] = (palette[0], palette[0], palette[1])
    if kind == "test" and regime == "sparse":
        plan[r, -1] = bg
    return plan, (r, 0, 2)


def _place(rng, h, w, side, pitch, axis):
    target, module = ((h-1)*pitch+side, (w-1)*pitch+side), (2*h+2, 2*w-1)
    gap = int(rng.integers(1, min(3, 30-target[axis]-module[axis])+1))
    required = [max(target[d], module[d]) for d in (0, 1)]
    required[axis] = target[axis]+module[axis]+gap
    mode = float(rng.random())
    if mode < .12:
        shape = (30, 30)
    elif mode < .30:
        n = int(rng.integers(max(required), 31))
        shape = (n, n)
    else:
        shape = tuple(int(rng.integers(n, min(30, n+5)+1)) for n in required)
    base, control = [0, 0], [0, 0]
    offset = int(rng.integers(shape[axis]-required[axis]+1))
    if rng.integers(2):
        base[axis], control[axis] = offset, offset+target[axis]+gap
    else:
        control[axis], base[axis] = offset, offset+module[axis]+gap
    other = 1-axis
    base[other] = int(rng.integers(shape[other]-target[other]+1))
    control[other] = int(rng.integers(shape[other]-module[other]+1))
    mini = tuple(control)
    key = (control[0]+2*h, control[1]+int(rng.integers(module[1]-2)))
    return shape, mini, key, tuple(base)


def _make_pair(rng, kind, regime, avoid=None):
    options = _options(kind, regime, avoid)
    _require(bool(options), "no feasible generation geometry")
    h, w, side, gutter, axis = options[int(rng.integers(len(options)))]
    colors = [int(x) for x in rng.permutation(10)]
    if avoid and colors[0] == avoid["background"]:
        colors[0], colors[1] = colors[1], colors[0]
    bg, guide, junction = colors[:3]
    palette = colors[3:3+int(rng.integers(2, 8))]
    plan, (r, c, length) = _plan(rng, h, w, palette, bg, kind, regime)
    pitch = side+gutter
    shape, mini, key, base = _place(rng, h, w, side, pitch, axis)
    inp = np.full(shape, bg, dtype=np.int8)
    mr, mc = mini
    inp[mr:mr+2*h:2, mc:mc+2*w:2] = plan
    ky, kx = key
    inp[ky, kx:kx+3] = guide
    inp[ky+1, kx:kx+3] = junction
    br, bc = base
    for k in range(length):
        y, x = br+r*pitch, bc+(c+k)*pitch
        inp[y:y+side, x:x+side] = plan[r, c+k]
    height, width = (h-1)*pitch+side, (w-1)*pitch+side
    middle = (side-1)//2
    occupied = np.flatnonzero(plan[r] != bg)
    positions = tuple(int(j)*pitch+middle for j in occupied)
    labels = tuple(int(plan[r, j]) for j in occupied)
    out = inp.copy()
    out[br:br+height, bc:bc+width] = _field(height, width, positions, labels,
                                         r*pitch+middle, guide, junction)
    return {"input": inp, "output": out}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    _require(nt > ns, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("ordinary", "sparse", "scale")))
    puzzle, seen = {"train": [], "test": []}, set()
    first = None
    for split, kinds in (("train", ["even", "odd"]+["free"]*(nt-2)),
                         ("test", ["test"]*ns)):
        for kind in kinds:
            for _ in range(24):
                pair = _make_pair(rng, kind, regime, first if kind == "odd" else None)
                key = _key(pair["input"])
                if key in seen:
                    continue
                d = _recover(pair["input"])
                seen.add(key)
                puzzle[split].append(pair)
                if kind == "even":
                    first = d
                break
            else:
                raise RuntimeError("bounded distinct-input construction failed")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(d):
    return {(a == b, (y-x) % 2) for a, b, x, y in
            zip(d["labels"], d["labels"][1:], d["positions"], d["positions"][1:])}


def _seed_only_field(d):
    c = d["match"][1]
    positions = tuple((c+k)*d["pitch"]+(d["side"]-1)//2
                      for k in range(len(d["sequence"])))
    return _field(*d["field_shape"], positions, d["sequence"], d["guide_y"],
                  d["guide"], d["junction"])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid example counts")
    seen, teaching, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "changed canvas shape")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _recover(inp)
            r, c, b, e = d["rect"]
            expected = inp.copy()
            field = _interval_field(d)
            expected[r:b, c:e] = field
            _require(np.array_equal(out, expected), "incorrect stripe field or changed preserved cell")
            _require(not np.array_equal(inp, out), "identity pair")
            _require(not np.any(field == d["background"]), "background left inside target")
            _require(np.all(field[d["guide_y"], list(d["positions"])] == d["guide"]),
                     "source crossings must retain guide color")
            gcount = int(np.count_nonzero(field == d["guide"]))
            _require(gcount == len(d["positions"]), "guide color outside source crossings")
            ordinary = np.ones(field.shape[1], dtype=bool)
            ordinary[list(d["positions"])] = False
            _require(np.all(field[d["guide_y"], ordinary] == d["junction"]),
                     "ordinary crossings must use junction color")
            if split == "train":
                teaching |= _features(d)
                training.append(d)
            else:
                _require(len(d["sequence"]) < len(d["labels"]), "tests need an unseen source")
    _require({(False, 0), (True, 0), (False, 1), (True, 1)} <= teaching,
             "training must expose both equality relations and both distance parities")
    for feature in ("side", "background"):
        _require(len({d[feature] for d in training}) >= 2, "training must vary "+feature)
    _require(len({d["plan"].shape[1] for d in training}) >= 2, "training must vary plan width")
    _require(any(not np.array_equal(_seed_only_field(d), _interval_field(d)) for d in training),
             "matching must expose a source absent from the enlarged clue")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def _pattern(labels):
    mapping = {}
    return tuple(mapping.setdefault(x, len(mapping)) for x in labels)


def self_test():
    """Stream 42 seeds, retaining only hashes/features; no files are written."""
    fingerprints = set()
    features = {k: set() for k in ("shape", "plan_shape", "side", "gutter", "match",
                                  "length", "source_count", "pattern", "seams", "background")}
    unseen = sparse = equal_seed = longer = nonconsecutive = 0
    longer_resolves = mutated_sources = pitch_ablations = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                d = _recover(inp)
                assert np.array_equal(solve(inp.copy()), out)
                assert np.array_equal(solve(inp.tolist()), out)
                field = _interval_field(d)
                # Parent A must matter on every example.
                uninverted = _field(*d["field_shape"], d["positions"], d["labels"],
                                    d["guide_y"], d["guide"], d["junction"], invert=False)
                assert not np.array_equal(field, uninverted)
                if len(d["sequence"]) < len(d["labels"]):
                    assert not np.array_equal(field, _seed_only_field(d))
                    unseen += 1
                if len(d["sequence"]) > 2 and len(_matches(d["plan"], d["sequence"][:2])) > 1:
                    longer_resolves += 1
                # Miniature pitch two cannot replace the measured large pitch:
                # its inferred target extent always differs in both dimensions.
                wrong_extent = tuple((n-1)*2+d["side"] for n in d["plan"].shape)
                assert wrong_extent != d["field_shape"]
                pitch_ablations += 1
                positions, labels = d["positions"], d["labels"]
                seams = sum(a != b and (y-x) % 2 == 0 for a, b, x, y in
                            zip(labels, labels[1:], positions, positions[1:]))
                for k, value in dict(shape=d["shape"], plan_shape=d["plan"].shape,
                                     side=d["side"], gutter=d["pitch"]-d["side"],
                                     match=d["match"], length=len(d["sequence"]),
                                     source_count=len(labels), pattern=_pattern(labels),
                                     seams=seams, background=d["background"]).items():
                    features[k].add(value)
                sparse += int(np.any(d["plan"][d["match"][0]] == d["background"]))
                equal_seed += int(len(set(d["sequence"])) == 1)
                longer += int(len(d["sequence"]) > 2)
                nonconsecutive += int(any(labels[i] == labels[j] and
                                         any(x != labels[i] for x in labels[i+1:j])
                                         for i in range(len(labels)) for j in range(i+2, len(labels))))
        # Local palette changes must preserve the rule, including the key.
        pair = puzzle["test"][0]
        mapping = np.random.default_rng(seed+10000).permutation(10).astype(np.int8)
        assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
        # Change a source not present in the seed, preserving the unique match.
        # Check the effect in the field, excluding the edited miniature pixel.
        d = _recover(pair["input"])
        active = d["plan"][d["match"][0]]
        first_col = d["match"][1]
        hidden = [int(j) for j in np.flatnonzero(active != d["background"])
                  if not first_col <= j < first_col+len(d["sequence"])]
        changed = False
        for j in hidden:
            for color in range(10):
                if color in {d["background"], d["guide"], d["junction"], int(active[j])}:
                    continue
                altered = pair["input"].copy()
                mr, mc = d["miniature"][:2]
                altered[mr+2*d["match"][0], mc+2*j] = color
                try:
                    prediction = solve(altered)
                except ValueError:
                    continue
                r, c, b, e = d["rect"]
                assert not np.array_equal(prediction[r:b, c:e], pair["output"][r:b, c:e])
                mutated_sources += 1
                changed = True
                break
            if changed:
                break
        assert changed, "could not exhibit dependence on an unseen selected-row label"
    assert len(fingerprints) == 42
    for name, minimum in (("shape", 30), ("plan_shape", 8), ("side", 4), ("gutter", 4),
                          ("match", 10), ("length", 3), ("source_count", 4),
                          ("pattern", 12), ("seams", 3), ("background", 9)):
        assert len(features[name]) >= minimum, (name, features[name])
    assert unseen >= 126 and sparse >= 10 and equal_seed >= 3 and longer >= 5 and nonconsecutive >= 5
    assert longer_resolves >= 3 and mutated_sources == 42 and pitch_ablations == 210
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(123456, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    assert _fingerprint(generate_puzzle(7931, 10, 4)) == _fingerprint(generate_puzzle(7931, 10, 4))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Direct semantic fixture distinguishes the two kinds of midpoint and edges.
    fixture = dict(field_shape=(3, 13), positions=(2, 6, 10), labels=(4, 4, 5),
                   guide_y=1, guide=8, junction=9)
    field = _interval_field(fixture)
    assert np.array_equal(field[:, 4], [4, 9, 4])
    assert np.all(field[:, 8] == 9)
    assert np.array_equal(field[:, 2], [9, 8, 9])
    assert np.array_equal(field[:, 0], [4, 9, 4])
    assert np.array_equal(field[:, -1], [5, 9, 5])
    # Ambiguous plan matching must be rejected, before considering canvas fit.
    p = generate_puzzle(8, 2, 1)
    pair = p["test"][0]
    d = _recover(pair["input"])
    ambiguous = pair["input"].copy()
    other = (d["match"][0]+1) % d["plan"].shape[0]
    mr, mc = d["miniature"][:2]
    for k, color in enumerate(d["sequence"]):
        ambiguous[mr+2*other, mc+2*(d["match"][1]+k)] = color
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous sequence accepted")
    damaged = {s: [{k: v.copy() for k, v in q.items()} for q in p[s]] for s in p}
    d = _recover(damaged["test"][0]["input"])
    r, c = d["base"]
    damaged["test"][0]["output"][r, c] = d["background"]
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted field accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(fingerprints),
                distinct_features={k: len(v) for k, v in features.items()},
                partial_clues=unseen, sparse_schedules=sparse, equal_color_seeds=equal_seed,
                longer_seeds=longer, longer_resolves_ambiguous_prefix=longer_resolves,
                changed_unseen_source=mutated_sources, wrong_pitch_changes_extent=pitch_ablations,
                nonconsecutive_repeats=nonconsecutive)
