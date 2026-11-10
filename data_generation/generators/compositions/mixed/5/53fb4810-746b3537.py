"""Complete hub-attached cyclic words, then quotient adjacent equal vectors.

No parent module, sampled metadata, or stored example is used by solve().
The small grids and bounded retries keep all diagnostics in memory.
"""

from itertools import permutations
import hashlib
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    todo = set(zip(*np.nonzero(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in todo:
                    todo.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _mask(h, w, kind):
    a = np.ones((h, w), dtype=bool)
    if kind == "clipped":
        a[0, 0] = a[0, -1] = a[-1, 0] = a[-1, -1] = False
    elif kind == "cross":
        a[:] = False
        th = h - (4 if h >= 5 else 2)
        tw = w - (4 if w >= 5 else 2)
        a[(h-th)//2:(h+th)//2, :] = True
        a[:, (w-tw)//2:(w+tw)//2] = True
    return a


def _hub_kind(a):
    h, w = a.shape
    if not (3 <= h <= 9 and 3 <= w <= 9):
        return None
    for kind in ("rectangle", "clipped", "cross"):
        if np.array_equal(a, _mask(h, w, kind)):
            return kind
    return None


def _coords(box, direction, width, distance):
    t, l, h, w = box
    if direction < 2:
        r = t - distance if direction == 0 else t + h - 1 + distance
        return [(r, l + (w - width) // 2 + j) for j in range(width)]
    c = l - distance if direction == 2 else l + w - 1 + distance
    return [(t + (h - width) // 2 + j, c) for j in range(width)]


def _maximum(shape, box, direction):
    t, l, h, w = box
    return (t, shape[0] - t - h, l, shape[1] - l - w)[direction]


def _dilate(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _parse_color(a, bg, hc, groups):
    hubs, claimed = [], set()
    for cells in groups:
        rs, cs = zip(*cells)
        box = (min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
        t, l, h, w = box
        kind = _hub_kind(a[t:t+h, l:l+w] == hc)
        if kind is None:
            raise ValueError("not a hub color")
        hubs.append((box, cells, kind))
        claimed.update(cells)
    strips = []
    for box, cells, kind in hubs:
        choices = []
        for direction in range(4):
            maximum = _maximum(a.shape, box, direction)
            if maximum < 1:
                continue
            transverse = box[3] if direction < 2 else box[2]
            face = _coords(box, direction, transverse, 1)
            active = [j for j, p in enumerate(face) if int(a[p]) not in (bg, hc)]
            if not active:
                continue
            width = len(active)
            start = (transverse - width) // 2
            if (not 1 <= width <= 4 or transverse - width < 2
                    or (transverse - width) % 2
                    or active != list(range(start, start + width))):
                raise ValueError("uncentered face")
            dr, dc = _DIRS[direction]
            if not all((r-dr, c-dc) in cells
                       for r, c in _coords(box, direction, width, 1)):
                raise ValueError("detached seed")
            choices.append((direction, width, maximum))
        if len(choices) != 1:
            raise ValueError("each hub needs exactly one attached strip")
        direction, width, maximum = choices[0]
        word, original, blank_seen = [], set(), False
        for distance in range(1, maximum + 1):
            points = _coords(box, direction, width, distance)
            values = tuple(int(a[p]) for p in points)
            if all(v not in (bg, hc) for v in values) and not blank_seen:
                word.append(values)
                original.update(points)
            elif all(v == bg for v in values):
                blank_seen = True
            else:
                raise ValueError("gapped strip, partial slice, or obstructed ray")
        if not word or original & claimed:
            raise ValueError("missing or overlapping strip")
        claimed.update(original)
        strips.append(dict(box=box, cells=cells, kind=kind, direction=direction,
                           width=width, maximum=maximum, word=word, original=original))
    if claimed != set(zip(*np.nonzero(a != bg))):
        raise ValueError("unattached material")
    # The family excludes all crossings, including through completed material.
    rays = set()
    for s in strips:
        ray = {p for d in range(1, s["maximum"] + 1)
               for p in _coords(s["box"], s["direction"], s["width"], d)}
        if ray & rays or ray & set.union(*(x[1] for x in hubs)):
            raise ValueError("intersecting rays")
        rays.update(ray)
    return strips


def _complete(value):
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= a.size // 2:
        raise ValueError("background must occupy more than half of the input")
    interpretations = []
    for hc in np.unique(a):
        hc = int(hc)
        if hc == bg:
            continue
        try:
            strips = _parse_color(a, bg, hc, _components(a == hc))
            interpretations.append((hc, strips))
        except ValueError:
            pass
    if len(interpretations) != 1:
        raise ValueError("ambiguous or missing hub interpretation")
    hc, strips = interpretations[0]
    completed = a.copy()
    for s in strips:
        word = s["word"]
        # The entire visible rectangle is a cyclic word. ABA continues ABA,
        # never BAB; integer copies of a word give exactly the same continuation.
        for d in range(len(word) + 1, s["maximum"] + 1):
            for p, v in zip(_coords(s["box"], s["direction"], s["width"], d),
                            word[(d - 1) % len(word)]):
                completed[p] = v
    return completed, dict(background=bg, hub_color=hc, strips=strips)


def _edges(a):
    return (np.all(a[1:] == a[:-1], axis=1),
            np.all(a[:, 1:] == a[:, :-1], axis=0))


def _collapse(a):
    """Both masks are computed on the SAME completed canvas."""
    rows, cols = _edges(a)
    return a[np.ix_(np.r_[True, ~rows], np.r_[True, ~cols])].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    completed, _ = _complete(input_grid)
    return _collapse(completed)


def _recurs(a, axis, background, hub_color):
    vectors = a if axis == 0 else a.T
    payload_vectors = [row.tobytes() for row in vectors
                       if np.any((row != background) & (row != hub_color))]
    return len(set(payload_vectors)) < len(payload_vectors)


def _features(inp, out):
    completed, info = _complete(inp)
    assert np.array_equal(out, _collapse(completed)), "wrong completion or vector collapse"
    r, c = _edges(completed)
    ir, ic = _edges(inp)
    info.update(born_rows=int(np.count_nonzero(r & ~ir)),
                born_columns=int(np.count_nonzero(c & ~ic)),
                deleted_rows=int(r.sum()), deleted_columns=int(c.sum()),
                recurring_rows=_recurs(out, 0, info["background"], info["hub_color"]),
                recurring_columns=_recurs(out, 1, info["background"], info["hub_color"]),
                changed_cells=int(np.count_nonzero(completed != inp)))
    assert info["changed_cells"] > 0, "must teach actual continuation"
    assert r.any() and c.any(), "both compression axes must affect every answer"
    assert not np.array_equal(_collapse(inp), out), "omitting A must change answer"
    assert out.shape[0] < completed.shape[0] and out.shape[1] < completed.shape[1]
    # Deleting representatives cannot create new inequalities or lose a
    # distinction: test order independence by genuinely sequential deletion.
    row_first = completed[np.r_[True, ~r], :]
    row_first = row_first[:, np.r_[True, ~_edges(row_first)[1]]]
    col_first = completed[:, np.r_[True, ~c]]
    col_first = col_first[np.r_[True, ~_edges(col_first)[0]], :]
    assert np.array_equal(row_first, out) and np.array_equal(col_first, out)
    assert not any(x.any() for x in _edges(out)), "output is not maximally run-collapsed"
    assert set(np.unique(out)) == set(np.unique(completed)), "lost a color"
    return info


def _prefix_alternative(inp):
    """Diagnostic rival: infer a short period even if it does not divide word length."""
    completed, info = _complete(inp)
    for s in info["strips"]:
        word = s["word"]
        period = next(p for p in range(1, len(word) + 1)
                      if all(v == word[i % p] for i, v in enumerate(word)))
        for d in range(len(word) + 1, s["maximum"] + 1):
            for p, v in zip(_coords(s["box"], s["direction"], s["width"], d),
                            word[(d-1) % period]):
                completed[p] = v
    return _collapse(completed)


def _word(rng, width, length, payload):
    # A vector word has repeated endpoints and at least one different interior
    # slice. This creates a new equality at its completed seam if unobstructed.
    a = rng.choice(payload, size=width).astype(np.int8)
    b = a.copy()
    j = int(rng.integers(width))
    b[j] = rng.choice([v for v in payload if v != a[j]])
    word = np.array([a.copy() for _ in range(length)], dtype=np.int8)
    word[1] = b
    if length > 3:
        for i in range(2, length - 1):
            word[i] = (a if rng.random() < .45 else b if rng.random() < .65
                       else rng.choice(payload, size=width))
    return word


def _spec(rng, shape, direction, width, length, payload, *, box=None, word=None):
    transverse = width + 2 * int(rng.integers(1, 3))
    longitudinal = int(rng.integers(3, 8))
    h, w = ((longitudinal, transverse) if direction < 2 else (transverse, longitudinal))
    if box is None:
        box = (int(rng.integers(shape[0] - h + 1)),
               int(rng.integers(shape[1] - w + 1)), h, w)
    t, l, h, w = box
    maximum = _maximum(shape, box, direction)
    if maximum <= length:
        return None
    kind = str(rng.choice(("rectangle", "clipped", "cross")))
    transverse = w if direction < 2 else h
    if kind == "cross" and transverse - (4 if transverse >= 5 else 2) < width:
        kind = "clipped"
    mask = _mask(h, w, kind)
    cells = {(t + int(r), l + int(c)) for r, c in zip(*np.nonzero(mask))}
    hub_box = {(r, c) for r in range(t, t+h) for c in range(l, l+w)}
    original = {p for d in range(1, length + 1)
                for p in _coords(box, direction, width, d)}
    ray = {p for d in range(1, maximum + 1)
           for p in _coords(box, direction, width, d)}
    if word is None:
        word = _word(rng, width, length, payload)
    return dict(box=box, kind=kind, cells=cells, hub_box=hub_box, original=original,
                ray=ray, direction=direction, width=width, maximum=maximum, word=word)


def _compatible(a, others):
    if a is None:
        return False
    for b in others:
        if (a["ray"] & b["ray"]
                or _dilate(a["hub_box"]) & (b["hub_box"] | b["ray"])
                or _dilate(b["hub_box"]) & a["ray"]
                or _dilate(a["original"]) & b["original"]):
            return False
    return True


def _paint(shape, specs, bg, hc):
    inp = np.full(shape, bg, dtype=np.int8)
    completed = inp.copy()
    for s in specs:
        for p in s["cells"]:
            inp[p] = completed[p] = hc
        for d in range(1, s["maximum"] + 1):
            for p, v in zip(_coords(s["box"], s["direction"], s["width"], d),
                            s["word"][(d - 1) % len(s["word"])]):
                completed[p] = v
                if d <= len(s["word"]):
                    inp[p] = v
    return {"input": inp, "output": _collapse(completed)}


def _witness(pair, both):
    try:
        f = _features(pair["input"], pair["output"])
    except (ValueError, AssertionError):
        return False
    axes = (f["born_rows"] > 0, f["born_columns"] > 0)
    return (all(axes) if both else any(axes)) and f["recurring_rows"] and f["recurring_columns"]


def _aligned_scene(rng, count, width_limit, length, payload, bg, hc, min_width=1):
    """Constructive escape path: aligned words expose full-vector seam runs.

    Four collision-free outward rays fit, even at width four. This is also a
    sampled layout, rather than a latent exception to the transformation.
    """
    specs = []
    nv = (count + 1) // 2
    nh = count // 2
    for direction, n in ((0, nv), (2, nh)):
        for i in range(n):
            width = int(rng.integers(min_width if not specs else 1, width_limit + 1))
            offset = 16 + i * 8
            box = ((10, offset, 5, width + 2) if direction == 0
                   else (offset, 10, width + 2, 5))
            s = _spec(rng, (30, 30), direction, width, length, payload, box=box)
            assert _compatible(s, specs)
            specs.append(s)
    # Shared slice-class order makes equal rows/columns global even when
    # distinct strips carry distinct vector colors and transverse widths.
    for direction in (0, 2):
        group = [s for s in specs if s["direction"] == direction]
        classes = np.zeros(length, dtype=int)
        classes[1] = 1
        if length > 3:
            classes[2:-1] = rng.integers(0, 2, size=length-3)
        for s in group:
            ab = _word(rng, s["width"], 3, payload)[:2]
            s["word"] = ab[classes]
    pair = _paint((30, 30), specs, bg, hc)
    turns = int(rng.integers(4))
    mirror = bool(rng.integers(2))
    pair = {k: np.ascontiguousarray(np.rot90(v, turns)) for k, v in pair.items()}
    if mirror:
        pair = {k: np.ascontiguousarray(np.fliplr(v)) for k, v in pair.items()}
    assert _witness(pair, both=True)
    return pair


def _make_pair(rng, count, width_limit, length, both, min_width=1):
    palette = [int(v) for v in rng.permutation(10)]
    bg, hc = palette[:2]
    payload = palette[2:2 + int(rng.integers(2, 9))]
    if rng.random() < .18:
        return _aligned_scene(rng, count, width_limit, length, payload, bg, hc, min_width)
    # Generic collision-checked placement over the full canvas. The pixel-based
    # acceptance condition, not an assumed local repeat, establishes causality.
    for _ in range(45):
        shape = (int(rng.integers(18, 31)), int(rng.integers(18, 31)))
        directions = [int(rng.integers(2)), 2 + int(rng.integers(2))]
        directions += [int(rng.integers(4)) for _ in range(count - 2)]
        rng.shuffle(directions)
        specs = []
        for direction in directions:
            for _ in range(45):
                width = int(rng.integers(min_width if not specs else 1, width_limit + 1))
                candidate = _spec(rng, shape, direction, width, length, payload)
                if _compatible(candidate, specs):
                    specs.append(candidate)
                    break
            else:
                break
        if len(specs) == count:
            pair = _paint(shape, specs, bg, hc)
            if _witness(pair, both):
                return pair
    return _aligned_scene(rng, count, width_limit, length, payload, bg, hc, min_width)


def _counts(a, b):
    for name, value, low, high in (("num_train", a, 2, 10), ("num_test", b, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} out of range")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def _make_unique(pair, used):
    if _key(pair["input"]) not in used:
        return pair
    colors = np.unique(pair["input"]).tolist()
    # At most 13 exclusions; injective color assignments give at least 5040
    # candidates with the four colors mandatory in every input.
    for values in permutations(range(10), len(colors)):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = values
        candidate = {k: mapping[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in used:
            return candidate
    raise RuntimeError("finite recoloring space exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # count, wider vectors, longer cyclic words
    puzzle, used = {"train": [], "test": []}, set()
    for i in range(a):
        length = 3 if i == 0 else int(rng.integers(3, 6))
        count = 2 if i < 2 or regime == 0 else int(rng.integers(2, 4))
        width_limit = 2 if regime == 1 else int(rng.integers(2, 5))
        pair = _make_unique(_make_pair(rng, count, width_limit, length, i == 0,
                                      min_width=2 if i == 1 else 1), used)
        used.add(_key(pair["input"]))
        puzzle["train"].append(pair)
    rng.shuffle(puzzle["train"])
    for _ in range(b):
        count = int(rng.integers(3, 5)) if regime == 0 else int(rng.integers(2, 5))
        width_limit = 4
        length = int(rng.integers(6, 8)) if regime == 2 else int(rng.integers(3, 6))
        pair = _make_unique(_make_pair(rng, count, width_limit, length, True,
                                      min_width=3 if regime == 1 else 1), used)
        used.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, training, testing, prefix_witness = set(), [], [], False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            f = _features(inp, out)
            assert f["born_rows"] or f["born_columns"], "no continuation-caused vector equality"
            assert f["recurring_rows"] and f["recurring_columns"], "need retained separated vectors"
            assert 2 <= len(f["strips"]) <= 4
            assert {s["direction"] // 2 for s in f["strips"]} == {0, 1}
            if split == "train":
                training.append(f)
                prefix_witness |= not np.array_equal(_prefix_alternative(inp), out)
            else:
                testing.append(f)
                assert f["born_rows"] and f["born_columns"], "test must exercise both causal axes"
    assert any(f["born_rows"] and f["born_columns"] for f in training)
    assert any(len(s["word"]) == 3 and s["word"][0] == s["word"][-1]
               and s["word"][0] != s["word"][1]
               for f in training for s in f["strips"]), "teach whole ABA word"
    assert prefix_witness, "demonstrate why a nondividing prefix period is wrong"
    assert any(s["width"] >= 2 for f in training for s in f["strips"]), "teach vector slices"
    # Reconstruct the curriculum from pixels; a sampled regime label is not evidence.
    assert any(max(feature(f) for f in testing) > max(feature(f) for f in training)
               for feature in (lambda f: len(f["strips"]),
                               lambda f: max(s["width"] for s in f["strips"]),
                               lambda f: max(len(s["word"]) for s in f["strips"]))), "no structural test shift"


def self_test():
    """42 seeds, streamed: no puzzle files, logs, traces, or retained corpus."""
    shapes, structures, palettes, directions, counts, widths, lengths, kinds = (set() for _ in range(8))
    fingerprints = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp), out)
                completed, f = _complete(inp)
                rows, cols = _edges(completed)
                assert not np.array_equal(completed[np.r_[True, ~rows], :], out)
                assert not np.array_equal(completed[:, np.r_[True, ~cols]], out)
                shapes.add(inp.shape)
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).digest()))
                palettes.add(tuple(np.unique(inp)))
                counts.add(len(f["strips"]))
                for s in f["strips"]:
                    directions.add(s["direction"])
                    widths.add(s["width"])
                    lengths.add(len(s["word"]))
                    kinds.add(s["kind"])
                digest.update(repr(inp.shape).encode())
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 20 and len(structures) >= 100 and len(palettes) >= 30
    assert directions == {0, 1, 2, 3} and counts == {2, 3, 4}
    assert widths == {1, 2, 3, 4} and lengths >= {3, 4, 5, 6, 7}
    assert kinds == {"rectangle", "clipped", "cross"}
    # This two-axis quotient restricts exactly to B's rule on solid stripe
    # canvases, including a separated returning color; it extends B on scenes.
    bands = np.repeat(np.array([4, 7, 4, 9], dtype=np.int8), [2, 1, 3, 2])
    stripes = np.tile(bands, (5, 1))
    assert np.array_equal(_collapse(stripes), np.array([[4, 7, 4, 9]], dtype=np.int8))
    assert np.array_equal(_collapse(stripes.T), np.array([[4], [7], [4], [9]], dtype=np.int8))
    p = generate_puzzle(42, 2, 1)
    q = generate_puzzle(42, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for split in p
               for x, y in zip(p[split], q[split]) for k in x)
    for a in range(2, 11):
        for b in range(1, min(4, a-1) + 1):
            p = generate_puzzle(100 + a * 5 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (4, 1.0),
                 (None, 1), (4, None), ("4", 1), (4, "1"),
                 (np.bool_(True), 1), (4, np.bool_(True))):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(9, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError("incorrect label accepted")
