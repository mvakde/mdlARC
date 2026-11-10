"""Fractured blueprint registration: jointly pack and register, then enlarge.

Only standard-library modules and NumPy are used. No file I/O or retained traces
occur in generation, validation, or self_test. All searches have finite bounds.
"""
from functools import lru_cache
from itertools import combinations
from math import isqrt
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError("counts must be integers")
    a, b = int(num_train), int(num_test)
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _DIRS:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(np.array(cells, dtype=int))
    return result


def _normal(coords):
    a = np.array(coords, dtype=int)
    return tuple(sorted(map(tuple, (a - a.min(axis=0)).tolist())))


def _rotate(shape, turns):
    a = np.array(shape, dtype=int)
    for _ in range(turns % 4):
        a = np.column_stack((a[:, 1], -a[:, 0]))
    return _normal(a)


def _dims(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _asymmetric(shape):
    return len({_rotate(shape, k) for k in range(4)}) == 4


def _mask(coords, n):
    return sum(1 << (int(r) * n + int(c)) for r, c in coords)


def _placements(shape, n, corner, anchor=False):
    result = {}
    corner_cells = (0, n - 1, n * (n - 1), n * n - 1)
    for k in range(1 if anchor else 4):
        a = np.array(_rotate(shape, k), dtype=int)
        h, w = a.max(axis=0) + 1
        if h > n or w > n:
            continue
        a += (0 if corner < 2 else n - h, 0 if corner % 2 == 0 else n - w)
        m = _mask(a, n)
        hits = [(m >> t) & 1 for t in corner_cells]
        if sum(hits) == 1 and hits[corner]:
            result.setdefault(m, k)
    return tuple(result.items())


@lru_cache(maxsize=128)
def _pack(shapes, n):
    """Every distinct corner-colored packing, collapsing interchangeable pieces.

    A piece that occupies a specified extreme corner has a fixed translation
    once rotated. Thus 3! * 4**3 is an upper bound on the non-anchor trials.
    The bounded cache contains geometry only and has no effect on answers.
    """
    if len(shapes) != 4 or sum(map(len, shapes)) != n * n:
        return ()
    table = {(i, c): _placements(shapes[i], n, c)
             for i in range(1, 4) for c in range(1, 4)}
    found = set()

    def visit(corner, remaining, occupied, masks):
        if corner == 4:
            if occupied == (1 << (n * n)) - 1:
                found.add(tuple(masks))
            return
        for i in remaining:
            for m, k in table[i, corner]:
                if not m & occupied:
                    visit(corner + 1, tuple(j for j in remaining if j != i),
                          occupied | m, masks + [m])

    for m, k in _placements(shapes[0], n, 0, anchor=True):
        visit(1, (1, 2, 3), m, [m])
    return tuple(sorted(found))


def _owner(masks, n):
    a = np.full((n, n), -1, dtype=np.int8)
    for corner, m in enumerate(masks):
        for t in range(n * n):
            if (m >> t) & 1:
                a.flat[t] = corner
    _require(bool(np.all(a >= 0)), "incomplete packing")
    return a


def _sequence(a, axis, r, c, length):
    return tuple(int(a[r + (axis == 0) * k, c + (axis == 1) * k]) for k in range(length))


def _matches(plans, axis, sequence):
    n, length = len(plans[0]), len(sequence)
    result = []
    for j, a in enumerate(plans):
        for r in range(n - (length - 1 if axis == 0 else 0)):
            for c in range(n - (length - 1 if axis == 1 else 0)):
                if _sequence(a, axis, r, c, length) == sequence:
                    result.append((j, r, c))
    return result


def _clue_kind(start, length, n):
    return "full" if length == n else "end" if start == 0 or start + length == n else "interior"


def _clues(plans, wanted, axis, kind, require_anchor=False):
    n = len(plans[0])
    found = {}
    for j, a in enumerate(plans):
        for length in range(3, n + 1):
            for r in range(n - (length - 1 if axis == 0 else 0)):
                for c in range(n - (length - 1 if axis == 1 else 0)):
                    seq = _sequence(a, axis, r, c, length)
                    if len(set(seq)) >= 3:
                        found.setdefault(seq, []).append((j, r, c, length))
    return [(r, c, length, seq) for seq, hits in found.items() if len(hits) == 1
            for j, r, c, length in hits if j == wanted and (not require_anchor or 0 in seq)
            and _clue_kind(r if axis == 0 else c, length, n) == kind]


def _partition(rng, n, axis, fallback=False, asymmetric=False):
    """Complementary lower pieces have a reversible, palindromic seam.

    This is a sampling construction, never a solver assumption. Transposition
    changes which chamber contains the reversible pair and exchanges TR/BL.
    """
    h = 2 if fallback else int(rng.integers(2, n - 2))
    m = n - h
    top = np.full(h, n - 2) if fallback else rng.integers(2, n - 1, size=h)
    if fallback and asymmetric:
        top[-1] -= 1
    if fallback:
        lower = [1 + min(i, m - 1 - i) for i in range(m)]
    else:
        half = rng.integers(1, n, size=(m + 1) // 2)
        lower = np.concatenate((half, half[-(2 if m % 2 else 1)::-1])).tolist()
    if len(set(lower)) < 2:
        return None
    a = np.empty((n, n), dtype=np.int8)
    for r in range(h):
        a[r, :top[r]], a[r, top[r]:] = 0, 1
    for i, r in enumerate(range(h, n)):
        a[r, :lower[i]], a[r, lower[i]:] = 2, 3
    if axis == 1:
        a = np.array([0, 2, 1, 3], dtype=np.int8)[a.T]
    if np.count_nonzero(a == 0) < 6:
        return None
    return a


def _key_sites(owner):
    return [(r, c) for r in range(len(owner) - 1) for c in range(len(owner) - 1)
            if np.all(owner[r:r + 2, c:c + 2] == 0)]


def _layout(rng, dims):
    """Five reserved boxes: four pieces and the entire future enlarged square.

    Random shelf groupings may put fragments on several sides of the witness.
    The last fallback is guaranteed for the supported dimensions at gap one.
    """
    chosen = None
    for attempt in range(81):
        transpose = bool(rng.integers(2))
        ds = [(w, h) if transpose else (h, w) for h, w in dims]
        gap = int(rng.integers(1, 4)) if attempt < 80 else 1
        order = [int(x) for x in rng.permutation(5)]
        if attempt == 80:
            # Small pieces on one shelf, the reserved target on the other.
            order = [int(x) for x in rng.permutation(4)] + [4]
            cut = 8
        else:
            cut = int(rng.integers(16))
        groups, group = [], []
        for j, i in enumerate(order):
            group.append(i)
            if j == 4 or (cut >> j) & 1:
                groups.append(group)
                group = []
        widths = [sum(ds[i][1] for i in g) + gap * (len(g) - 1) for g in groups]
        heights = [max(ds[i][0] for i in g) for g in groups]
        height, width = sum(heights) + gap * (len(groups) - 1), max(widths)
        if height > 30 or width > 30:
            continue
        offsets, row = [None] * 5, 0
        for g, hh, ww in zip(groups, heights, widths):
            col = int(rng.integers(width - ww + 1))
            for i in g:
                rr = row + int(rng.integers(hh - ds[i][0] + 1))
                offsets[i] = (col, rr) if transpose else (rr, col)
                col += ds[i][1] + gap
            row += hh + gap
        shape = (width, height) if transpose else (height, width)
        chosen = offsets, shape
        break
    _require(chosen is not None, "layout fallback did not fit")
    offsets, extent = chosen
    pads = []
    for d in extent:
        before = int(rng.integers(min(3, 30 - d) + 1))
        after = int(rng.integers(min(4, 30 - d - before) + 1))
        pads.append((before, after))
    shape = tuple(extent[d] + sum(pads[d]) for d in (0, 1))
    return [(r + pads[0][0], c + pads[1][0]) for r, c in offsets], shape


def _paint(grid, plan, base, side, stride, key):
    for (r, c), corner in np.ndenumerate(plan):
        y, x = base[0] + r * stride, base[1] + c * stride
        grid[y:y + side, x:x + side] = key[int(corner)]


def _make_pair(rng, n, axis, kind, side=None, avoid_background=None, teaching=False,
               avoid_key=None, asymmetric=False):
    # At most 81 profiles; the last is a constructive, verified profile for each
    # supported n/axis/clue class. No retries, grids, or traces are saved to disk.
    for attempt in range(81):
        owner = _partition(rng, n, axis, fallback=attempt == 80, asymmetric=asymmetric)
        if owner is None:
            continue
        shapes = tuple(_normal(np.argwhere(owner == i)) for i in range(4))
        if asymmetric and not _asymmetric(shapes[0]):
            continue
        packings = _pack(shapes, n)
        expected_masks = tuple(_mask(np.argwhere(owner == i), n) for i in range(4))
        if len(packings) < 2 or expected_masks not in packings:
            continue
        plans = [_owner(masks, n) for masks in packings]
        choices = _clues(plans, packings.index(expected_masks), axis, kind, teaching)
        if choices:
            break
    else:
        raise AssertionError("constructive profile lacked an informative clue")
    r, c, length, seq = choices[int(rng.integers(len(choices)))]
    geometry = [(s, g) for s in (2, 3) for g in (1, 2, 3)
                if (side is None or s == side) and (n - 1) * (s + g) + s <= 23]
    s, gutter = geometry[int(rng.integers(len(geometry)))]
    stride, extent = s + gutter, (n - 1) * (s + gutter) + s
    turns = [0] + ([1, 2, 3] if teaching else [int(x) for x in rng.integers(0, 4, size=3)])
    if teaching:
        turns[1:] = [int(x) for x in rng.permutation(turns[1:])]
    oriented = [_rotate(shape, k) for shape, k in zip(shapes, turns)]
    offsets, canvas = _layout(rng, [_dims(a) for a in oriented] + [(extent, extent)])
    colors = [int(x) for x in rng.permutation(10)]
    if colors[0] == avoid_background:
        colors[0], colors[-1] = colors[-1], colors[0]
    bg, key = colors[0], colors[1:5]
    if tuple(key) == avoid_key:
        key[0], key[1] = key[1], key[0]
    pool = colors[5:]
    count = int(rng.integers(1, 5))
    carriers = [pool[0]] + [int(rng.choice(pool[:count])) for _ in range(3)]
    inp = np.full(canvas, bg, dtype=np.int8)
    for shape, offset, color in zip(oriented, offsets, carriers):
        cells = np.array(shape) + offset
        inp[cells[:, 0], cells[:, 1]] = color
    kr, kc = _key_sites(owner)[int(rng.integers(len(_key_sites(owner))))]
    ar, ac = offsets[0]
    inp[ar + kr:ar + kr + 2, ac + kc:ac + kc + 2] = np.array(key).reshape(2, 2)
    base = offsets[4]
    for k, corner in enumerate(seq):
        y = base[0] + (r + (axis == 0) * k) * stride
        x = base[1] + (c + (axis == 1) * k) * stride
        inp[y:y + s, x:x + s] = key[corner]
    out = inp.copy()
    _paint(out, owner, base, s, stride, key)
    return {"input": inp, "output": out}


def _parse(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > inp.size // 2, "background must occupy a majority")
    comps = _components(inp != bg)
    anchors = []
    for cells in comps:
        values, freq = np.unique(inp[cells[:, 0], cells[:, 1]], return_counts=True)
        if len(values) == 5 and sum(freq == 1) == 4 and sum(freq > 1) == 1:
            carrier = int(values[freq > 1][0])
            points = cells[inp[cells[:, 0], cells[:, 1]] != carrier]
            r, c = points.min(axis=0)
            if set(map(tuple, points)) == {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}:
                anchors.append((cells, carrier, inp[r:r + 2, c:c + 2].copy()))
    _require(len(anchors) == 1, "exactly one readable anchor required")
    anchor, carrier, key2 = anchors[0]
    key = tuple(int(x) for x in key2.flat)
    _require(len(set(key)) == 4 and bg not in key and carrier not in key, "legend palette")
    pieces, squares, carriers = [anchor], [], [carrier]
    for cells in comps:
        if np.array_equal(cells, anchor):
            continue
        values = np.unique(inp[cells[:, 0], cells[:, 1]])
        _require(len(values) == 1, "plain pieces and seed squares must be uniform")
        color = int(values[0])
        if color not in key:
            pieces.append(cells)
            carriers.append(color)
            continue
        r, c = cells.min(axis=0)
        h, w = cells.max(axis=0) - (r, c) + 1
        _require(h == w and h >= 2 and len(cells) == h * h, "seed must be a solid square")
        squares.append((int(r), int(c), int(h), key.index(color)))
    _require(len(pieces) == 4 and len(squares) >= 3, "four fragments and at least three squares required")
    for a, b in combinations(comps, 2):
        distance = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2).min()
        _require(distance >= 2, "objects need a background-cell gap")
    total = sum(map(len, pieces))
    n = isqrt(total)
    _require(n * n == total, "fragment area must be square")
    shapes = tuple(_normal(cells) for cells in pieces)
    axis = 1 if len({s[0] for s in squares}) == 1 else 0
    _require(len({s[1 - axis] for s in squares}) == 1, "seed must lie on one axis")
    squares.sort(key=lambda a: a[axis])
    side = squares[0][2]
    _require(all(s[2] == side for s in squares), "uniform seed scale required")
    steps = np.diff([s[axis] for s in squares])
    stride = int(steps[0])
    _require(np.all(steps == stride) and stride > side, "uniform positive seed gutter required")
    seq = tuple(s[3] for s in squares)
    _require(len(set(seq)) >= 3, "seed must cross at least three corner colors")
    return dict(background=bg, key=key, shapes=shapes, pieces=pieces, carriers=carriers,
                n=n, axis=axis, side=side, stride=stride, sequence=seq, squares=squares)


def _analyze(input_grid):
    inp = _grid(input_grid)
    a = _parse(inp)
    n, axis = a["n"], a["axis"]
    packings = _pack(a["shapes"], n)
    _require(len(packings) >= 2, "the fragments must be geometrically ambiguous")
    plans = [_owner(masks, n) for masks in packings]
    matches = _matches(plans, axis, a["sequence"])
    # Uniqueness is checked before considering canvas edges or occupied cells.
    _require(len(matches) == 1, "witness must select one assembly and one plan position")
    j, r, c = matches[0]
    plan = plans[j]
    s, stride = a["side"], a["stride"]
    y = a["squares"][0][0] - r * stride
    x = a["squares"][0][1] - c * stride
    extent = (n - 1) * stride + s
    _require(0 <= y and 0 <= x and y + extent <= inp.shape[0] and x + extent <= inp.shape[1],
             "complete enlarged square must fit")
    for cells in a["pieces"]:
        near = ((cells[:, 0] >= y - 1) & (cells[:, 0] <= y + extent) &
                (cells[:, 1] >= x - 1) & (cells[:, 1] <= x + extent))
        _require(not near.any(), "fragments must be separate from the full enlarged square")
    # All reserved target pixels except supplied seed squares start as background.
    expected_input = np.full((extent, extent), a["background"], dtype=np.int8)
    for yy, xx, ss, corner in a["squares"]:
        expected_input[yy - y:yy - y + ss, xx - x:xx - x + ss] = a["key"][corner]
    _require(np.array_equal(inp[y:y + extent, x:x + extent], expected_input), "unexpected target marks")
    out = inp.copy()
    _paint(out, plan, (y, x), s, stride, a["key"])
    a.update(plan=plan, packings=packings, selected=j, match=(r, c), base=(y, x),
             kind=_clue_kind(r if axis == 0 else c, len(a["sequence"]), n), output=out)
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from visible input pixels alone; never access labels."""
    return _analyze(input_grid)["output"].copy()


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, metas = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            _require(np.array_equal(out, a["output"]), "incorrect completion or changed evidence")
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "nontrivial in-place completion")
            _require(np.all(out[inp != a["background"]] == inp[inp != a["background"]]), "evidence changed")
            # A-ablation: same exact pieces/key allow a different colored answer.
            alt = next(m for i, m in enumerate(a["packings"]) if i != a["selected"])
            _require(not np.array_equal(_owner(alt, a["n"]), a["plan"]), "assembly ambiguity was cosmetic")
            # A common B-ablation uses the legend as four rectangular quadrants.
            # An irregular seam in every accepted blueprint contradicts it.
            n = a["n"]
            quadrant_matches = any(np.array_equal(a["plan"],
                np.where(np.indices((n, n))[0] < yy, 0, 2) +
                np.where(np.indices((n, n))[1] < xx, 0, 1))
                for yy in range(1, n) for xx in range(1, n))
            _require(not quadrant_matches, "straight quadrant shortcut was sufficient")
            metas[split].append(a)
    train = metas["train"]
    _require({a["axis"] for a in train} == {0, 1}, "training must demonstrate both clue axes")
    _require({a["kind"] for a in train} >= {"full", "end"}, "training must teach scale and partial registration")
    _require(len({a["side"] for a in train}) >= 2, "training must teach scale inference")
    _require(len({a["background"] for a in train}) >= 2, "background must be a local role")
    _require(len({a["key"] for a in train}) >= 2, "training must show changing legends")
    _require(any(_asymmetric(a["shapes"][0]) for a in train), "training must establish anchor orientation")
    _require(all(a["kind"] == "interior" for a in metas["test"]), "test registration shift missing")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # growth, shared size range, shrink
    train_sizes = (6, 7) if regime == 2 else (5, 6)
    sizes = [int(x) for x in rng.permutation(train_sizes)]
    axes = [int(x) for x in rng.permutation(2)]
    scales = [2 if n == 7 else 3 for n in sizes] if 7 in sizes else [int(x) for x in rng.permutation([2, 3])]
    result, seen, first_bg, first_key = {"train": [], "test": []}, set(), None, None
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            if split == "train":
                n = sizes[i] if i < 2 else int(rng.choice(train_sizes))
                axis = axes[i] if i < 2 else int(rng.integers(2))
                kind = ("full", "end")[i] if i < 2 else str(rng.choice(["full", "end", "interior"]))
                side = scales[i] if i < 2 else None
            else:
                n = 7 if regime == 0 else 5 if regime == 2 else int(rng.choice(train_sizes))
                axis, kind, side = int(rng.integers(2)), "interior", None
            for attempt in range(16):
                pair = _make_pair(rng, n, axis, kind, side, first_bg if split == "train" and i == 1 else None,
                                  teaching=split == "train" and i == 0,
                                  avoid_key=first_key if split == "train" and i == 1 else None,
                                  asymmetric=split == "train" and i < 2 and n == 6)
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise AssertionError("unable to produce distinct inputs")
            seen.add(signature)
            result[split].append(pair)
            if split == "train" and i == 0:
                first = _parse(pair["input"])
                first_bg, first_key = first["background"], first["key"]
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed audit, with structural rather than only palette diversity."""
    signatures, records, witness_groups = set(), [], {}
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "input-only solve")
                a["split"] = split
                records.append(a)
                # Hold the entire abstract run and registration fixed; different
                # inferred blueprints prove that the run alone cannot set seams.
                group = (a["n"], a["axis"], a["match"], a["sequence"])
                witness_groups.setdefault(group, set()).add(a["plan"].tobytes())
    _require(len(signatures) == 42, "seed diversity")
    _require(any(len(v) > 1 for v in witness_groups.values()), "shape-sensitive witness collision missing")
    for split in ("train", "test"):
        rows = [a for a in records if a["split"] == split]
        for name, minimum in (("n", 3), ("axis", 2), ("side", 2), ("stride", 3),
                              ("background", 8), ("match", 8), ("base", 20)):
            _require(len({a[name] for a in rows}) >= minimum, "diversity collapsed: " + name)
        _require(len({a["plan"].tobytes() for a in rows}) >= 20, "seam diversity")
        _require(all(len(a["packings"]) >= 2 for a in rows), "missing joint disambiguation")
    _require(_signature(generate_puzzle(9137, 10, 4)) == _signature(generate_puzzle(9137, 10, 4)), "determinism")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exhaust the constructive profile fallbacks, separate from random success.
    for n in (5, 6, 7):
        for axis in (0, 1):
            for asymmetric in ((False,) if n == 5 else (False, True)):
                owner = _partition(np.random.default_rng(0), n, axis, fallback=True, asymmetric=asymmetric)
                shapes = tuple(_normal(np.argwhere(owner == i)) for i in range(4))
                packs = _pack(shapes, n)
                masks = tuple(_mask(np.argwhere(owner == i), n) for i in range(4))
                plans = [_owner(m, n) for m in packs]
                for kind in ("full", "end", "interior"):
                    _require(bool(_clues(plans, packs.index(masks), axis, kind)), "fallback clue")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"] = damaged["test"][0]["output"].copy()
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
