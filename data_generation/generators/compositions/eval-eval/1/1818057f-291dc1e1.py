"""Cross-registered catalogue: local landmarks define directed ribbon views.

Only Python/NumPy are needed. Generation and self_test do no filesystem I/O.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    """Return a proper-rotation-normalized input and its visible roles."""
    grid = _grid(grid)
    if min(grid.shape) < 7:
        raise ValueError("the coordinate key needs dimensions of at least seven")
    candidates = []
    for turn in range(4):
        g = np.rot90(grid, turn)
        h, w = g.shape
        for up in (False, True):
            edge, far = (h - 1, 0) if up else (0, h - 1)
            highlight = int(g[edge, 0])
            guide, bg = int(g[far, 0]), int(g[far, -1])
            if len({highlight, guide, bg}) != 3:
                continue
            slow = np.delete(g[:, 0], edge)
            fast = np.where(np.arange(1, w) % 2, guide, bg)
            if not (np.all(slow == guide) and np.array_equal(g[edge, 1:], fast)
                    and np.all(np.delete(g[:, -1], edge) == bg)
                    and np.all(g[far, 1:] == bg)):
                continue
            body = g.copy()
            body[edge, :] = bg
            body[:, 0] = bg
            colors = set(map(int, np.unique(body))) - {bg}
            if len(colors) != 1 or colors & {guide, highlight}:
                continue
            ink = colors.pop()
            mask = body == ink
            # One clear interior cell between drawing and any outer side.
            if (mask[:2].any() or mask[-2:].any()
                    or mask[:, :2].any() or mask[:, -2:].any()):
                continue
            candidates.append({"grid": body, "mask": mask, "up": up,
                               "rotation": (-turn) % 4, "background": bg,
                               "ink": ink, "highlight": highlight, "guide": guide})
    if len(candidates) != 1:
        raise ValueError("input must have exactly one striped/solid coordinate key")
    return candidates[0]


def _plus(mask):
    centers = np.zeros_like(mask, dtype=bool)
    centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                          & mask[2:, 1:-1] & mask[1:-1, :-2]
                          & mask[1:-1, 2:])
    union = centers.copy()
    union[:-1] |= centers[1:]
    union[1:] |= centers[:-1]
    union[:, :-1] |= centers[:, 1:]
    union[:, 1:] |= centers[:, :-1]
    return centers, union


def _describe(input_grid):
    d = _key(input_grid)
    mask = d["mask"]
    center_mask, union = _plus(mask)
    pins = [tuple(map(int, p)) for p in np.argwhere(center_mask)]
    if not 1 <= len(pins) <= 10:
        raise ValueError("a catalogue requires one through ten plus centers")
    pins.sort(key=lambda p: (-p[0] if d["up"] else p[0], p[1]))
    painted = d["grid"].copy()
    painted[union] = d["highlight"]
    entries = []
    for r, c in pins:
        a, b = c, c + 1
        while a > 0 and mask[r, a - 1]:
            a -= 1
        while b < mask.shape[1] and mask[r, b]:
            b += 1
        entries.append({"pin": (r, c), "start": a, "end": b,
                        "anchor": c - a, "block": painted[r-1:r+2, a:b].copy()})
    left = max(e["anchor"] for e in entries)
    right = max(e["end"] - e["pin"][1] - 1 for e in entries)
    if left + 1 + right > 30:
        raise ValueError("joint pin-relative reaches exceed thirty columns")
    d.update(entries=entries, union=union, centers=center_mask, left=left, right=right)
    return d


def _pack(entries, background, *, geometric=False, flip=False):
    if geometric:
        width = max(e["block"].shape[1] for e in entries)
        left = None
    else:
        left = max(e["anchor"] for e in entries)
        width = left + 1 + max(e["block"].shape[1] - e["anchor"] - 1 for e in entries)
    result = np.full((3 * len(entries), width), background, dtype=np.int8)
    for i, e in enumerate(entries):
        block = e["block"][::-1] if flip else e["block"]
        offset = (width - block.shape[1]) // 2 if geometric else left - e["anchor"]
        result[3*i:3*i+3, offset:offset+block.shape[1]] = block
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; do not infer anything from generated pairs."""
    d = _describe(input_grid)
    return _pack(d["entries"], d["background"])


def _reference(mask, roles, up):
    """Construction-side scalar specification; no production detection helper."""
    bg, ink, guide, highlight = roles
    h, w = mask.shape
    pins, marked = [], set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cross = ((r, c), (r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if all(mask[p] for p in cross):
                pins.append((r, c))
                marked.update(cross)
    if not 1 <= len(pins) <= 10:
        raise ValueError("construction pin budget")
    pins.sort(key=lambda p: (-p[0] if up else p[0], p[1]))
    views = []
    for r, c in pins:
        start = c
        while start and mask[r, start-1]:
            start -= 1
        stop = c
        while stop + 1 < w and mask[r, stop+1]:
            stop += 1
        views.append((r, c, start, stop))
    left = max(c - start for r, c, start, stop in views)
    right = max(stop - c for r, c, start, stop in views)
    if left + 1 + right > 30:
        raise ValueError("construction width budget")
    output = np.full((3*len(views), left+1+right), bg, dtype=np.int8)
    for i, (r, c, start, stop) in enumerate(views):
        for rr in range(r-1, r+2):
            for cc in range(start, stop+1):
                value = highlight if (rr, cc) in marked else ink if mask[rr, cc] else bg
                output[3*i+rr-r+1, left+cc-c] = value
    return output


def _paint(mask, roles, rotation, up):
    bg, ink, guide, highlight = map(int, roles)
    grid = np.full(mask.shape, bg, dtype=np.int8)
    grid[mask] = ink
    edge = -1 if up else 0
    grid[:, 0] = guide
    grid[edge, 1:] = np.where(np.arange(1, mask.shape[1]) % 2, guide, bg)
    grid[edge, 0] = highlight
    return {"input": np.rot90(grid, rotation).copy(),
            "output": _reference(mask, roles, up)}


def _run(mask, row, col, length, pins, rng):
    """A full center run; side intersections prescribe its exact plus centers."""
    mask[row, col:col+length] = True
    states = rng.integers(0, 3, size=length)
    mask[row-1, col:col+length] = states == 1
    mask[row+1, col:col+length] = states == 2
    for p in pins:
        mask[row-1:row+2, col+p] = True
    # An asymmetric extra at an endpoint is legal and can never be a new pin.
    mask[row-1, col+length-1] = True
    mask[row+1, col+length-1] = False


def _shared_scene(rng):
    h, w = int(rng.integers(14, 27)), int(rng.integers(14, 29))
    mask = np.zeros((h, w), dtype=bool)
    length = int(rng.integers(6, min(10, w-4)+1))
    r = int(rng.integers(3, h-6))
    c = int(rng.integers(2, w-length-1))
    _run(mask, r, c, length, [1, 2], rng)
    # Detached T: a deliberate three-arm near miss, ignored by extraction.
    t = int(rng.integers(2, w-4))
    mask[r+4, t:t+3] = True
    mask[r+3, t+1] = True
    return mask


def _raster_scene(rng):
    h, w = int(rng.integers(18, 30)), int(rng.integers(18, 30))
    mask = np.zeros((h, w), dtype=bool)
    r1, r2 = int(rng.integers(3, 6)), int(rng.integers(h-6, h-3))
    _run(mask, r1, 2, 6, [1], rng)
    _run(mask, r1, w-6, 3, [1], rng)
    _run(mask, r2, int(rng.integers(3, 6)), 7, [4], rng)
    # A second-row pin contributes a highlighted arm inside the first pin's
    # crop while its own center is outside that crop. Suppress incidental pins
    # between them, so this distinction remains easy to read.
    mask[r1+2, 3:6] = True
    mask[r1+1:r1+4, 4] = True
    mask[r1-1, 4] = False
    mask[r1+1, 2] = mask[r1+1, 5] = False
    return mask


def _gaps(rng, extra, size):
    weights = rng.dirichlet(np.ones(size) * float(rng.choice([.4, 1., 3.])))
    return rng.multinomial(extra, weights).tolist()


def _band_scene(rng):
    lanes = int(rng.integers(1, 6))
    minimum = 4 + 3*lanes + lanes - 1
    h, w = int(rng.integers(minimum, 31)), int(rng.integers(9, 31))
    mask = np.zeros((h, w), dtype=bool)
    vg = _gaps(rng, h-minimum, lanes+1)
    top, runs = 2+vg[0], []
    for lane in range(lanes):
        count = int(rng.integers(1, min(4, (w-3)//4)+1))
        available = w-4-(count-1)-3*count
        extras = int(rng.integers(available+1))
        lengths = [3+x for x in _gaps(rng, extras, count)]
        hg = _gaps(rng, available-extras, count+1)
        col = 2+hg[0]
        for j, length in enumerate(lengths):
            runs.append((top+1, col, length))
            col += length + 1 + hg[j+1]
        top += 4 + vg[lane+1]
    choices = [(i, p) for i, (_, _, length) in enumerate(runs) for p in range(1, length-1)]
    n = int(rng.integers(1, min(10, len(choices))+1))
    selected = {choices[int(i)] for i in rng.choice(len(choices), n, replace=False)}
    for i, (r, c, length) in enumerate(runs):
        _run(mask, r, c, length, [p for j, p in selected if j == i], rng)
    return mask


def _scatter_scene(rng):
    h, w = map(int, rng.integers(9, 31, size=2))
    mask = np.zeros((h, w), dtype=bool)
    for _ in range(int(rng.integers(1, 7))):
        r, c = int(rng.integers(3, h-3)), int(rng.integers(3, w-3))
        for p in ((r, c), (r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            mask[p] = True
    stamps = (
        ((0, -1), (0, 0), (0, 1), (1, 0)),  # T
        ((-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)),  # X
        tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)),
        tuple((a, b) for a in (0, 1, 2) for b in (0, 1, 2, 3)),
    )
    for _ in range(int(rng.integers(1, 13))):
        r, c = int(rng.integers(2, h-2)), int(rng.integers(2, w-2))
        if rng.random() < .45:
            offsets = stamps[int(rng.integers(len(stamps)))]
            turn = int(rng.integers(4))
            for a, b in offsets:
                for _ in range(turn):
                    a, b = -b, a
                rr, cc = r+a, c+b
                if 2 <= rr < h-2 and 2 <= cc < w-2:
                    mask[rr, cc] = True
        else:
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            for _ in range(int(rng.integers(1, 9))):
                if 2 <= r < h-2 and 2 <= c < w-2:
                    mask[r, c] = True
                if rng.random() < .3:
                    dr, dc = -dc, dr
                r, c = r+dr, c+dc
    return mask


def _ordinary_scene(rng):
    for _ in range(32):
        mask = _band_scene(rng) if rng.random() < .55 else _scatter_scene(rng)
        try:
            _reference(mask, (0, 1, 2, 3), False)
        except ValueError:
            continue
        return mask
    # Small shared-pin masks always meet both output budgets.
    return _shared_scene(rng)


def _near_miss(mask):
    t = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(mask, (3, 3))
    return any(bool((windows == np.rot90(t, k)).all(axis=(-1, -2)).any()) for k in range(4))


def _evidence_flags(d):
    entries, bg = d["entries"], d["background"]
    target = _pack(entries, bg)
    groups = {}
    for e in entries:
        groups.setdefault(e["pin"][0], []).append(e)
    groups = list(groups.values())  # Already in directed slow order.
    alternatives = {
        "slow_reverse": [e for group in groups[::-1] for e in group],
        "fast_reverse": [e for group in groups for e in group[::-1]],
        "fast_major": sorted(entries, key=lambda e: (e["pin"][1],
                                    -e["pin"][0] if d["up"] else e["pin"][0])),
    }
    intervals = {(e["pin"][0], e["start"], e["end"]) for e in entries}
    # Simulate the plausible destructive row-order detector.
    remaining = d["mask"].copy()
    greedy = 0
    for r in range(1, remaining.shape[0]-1):
        for c in range(1, remaining.shape[1]-1):
            cross = ((r, c), (r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if all(remaining[p] for p in cross):
                greedy += 1
                for p in cross:
                    remaining[p] = False
    flags = {
        "shared": len(intervals) < len(entries),
        "greedy": greedy < len(entries),
        "selective": any(np.any(e["block"] == d["ink"]) for e in entries),
        "registration": not np.array_equal(target, _pack(entries, bg, geometric=True)),
        "asymmetric": not np.array_equal(target, _pack(entries, bg, flip=True)),
        "near_miss": _near_miss(d["mask"]),
        "global_union": any(not np.array_equal(
            _plus(d["mask"][e["pin"][0]-1:e["pin"][0]+2, e["start"]:e["end"]])[1],
            d["union"][e["pin"][0]-1:e["pin"][0]+2, e["start"]:e["end"]]) for e in entries),
    }
    # Compare against recoloring every ink cell connected to a real plus.
    component = d["union"].copy()
    todo = [tuple(map(int, p)) for p in np.argwhere(component)]
    while todo:
        r, c = todo.pop()
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (0 <= rr < component.shape[0] and 0 <= cc < component.shape[1]
                    and d["mask"][rr, cc] and not component[rr, cc]):
                component[rr, cc] = True
                todo.append((rr, cc))
    alternative_grid = d["grid"].copy()
    alternative_grid[component] = d["highlight"]
    component_entries = []
    for e in entries:
        r, _ = e["pin"]
        component_entries.append(dict(e, block=alternative_grid[r-1:r+2, e["start"]:e["end"]]))
    flags["component"] = not np.array_equal(target, _pack(component_entries, bg))
    flags.update({name: not np.array_equal(target, _pack(order, bg))
                  for name, order in alternatives.items()})
    return flags


def _key_bytes(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _new_pair(rng, scene_kind, frame, used):
    scene = {"shared": _shared_scene, "raster": _raster_scene, "ordinary": _ordinary_scene}[scene_kind]
    for _ in range(32):
        mask = scene(rng)
        roles = tuple(map(int, rng.choice(10, 4, replace=False)))
        pair = _paint(mask, roles, *frame)
        if _key_bytes(pair["input"]) not in used:
            return pair
    # At most 13 earlier inputs; 7*7 distinct reserved marker assignments
    # suffice for this same mask without modifying any semantic convention.
    bg, ink, guide, highlight = roles
    for new_guide in range(10):
        if new_guide in (bg, ink):
            continue
        for new_highlight in range(10):
            if new_highlight in (bg, ink, new_guide):
                continue
            pair = _paint(mask, (bg, ink, new_guide, new_highlight), *frame)
            if _key_bytes(pair["input"]) not in used:
                return pair
    raise AssertionError("finite uniqueness bound exceeded")


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    k, up = int(rng.integers(4)), bool(rng.integers(2))
    taught = [(k, up), ((k+1) % 4, not up)]
    held = [(k, not up), ((k+1) % 4, up)]
    puzzle, used = {"train": [], "test": []}, set()
    for i in range(num_train):
        frame = taught[i] if i < 2 else taught[int(rng.integers(2))]
        kind = ("shared", "raster")[i] if i < 2 else "ordinary"
        pair = _new_pair(rng, kind, frame, used)
        used.add(_key_bytes(pair["input"]))
        puzzle["train"].append(pair)
    for _ in range(num_test):
        pair = _new_pair(rng, "ordinary", held[int(rng.integers(2))], used)
        used.add(_key_bytes(pair["input"]))
        puzzle["test"].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, descriptors = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            identity = _key_bytes(inp)
            assert identity not in seen, "duplicate input"
            seen.add(identity)
            d = _describe(inp)
            descriptors[split].append(d)
            entries, bg = d["entries"], d["background"]
            assert out.shape == (3*len(entries), d["left"]+1+d["right"])
            assert np.array_equal(out, _pack(entries, bg)), "wrong crop, order, union, or registration"
            # Verify each local copy, its padding, and its designated pin.
            for i, e in enumerate(entries):
                block, left = e["block"], d["left"]-e["anchor"]
                assert np.array_equal(out[3*i:3*i+3, left:left+block.shape[1]], block)
                assert np.all(out[3*i:3*i+3, :left] == bg)
                assert np.all(out[3*i:3*i+3, left+block.shape[1]:] == bg)
                assert out[3*i+1, d["left"]] == d["highlight"]
            scalar = _reference(d["mask"], (bg, d["ink"], d["guide"], d["highlight"]), d["up"])
            assert np.array_equal(out, scalar), "scalar construction disagrees"
    flags = [_evidence_flags(d) for d in descriptors["train"]]
    assert all(any(f[name] for f in flags) for name in flags[0]), "missing training distinction"
    # The witnesses really show asymmetric thickness in both taught frames.
    frames = {(d["rotation"], d["up"]) for d in descriptors["train"]}
    assert len(frames) == 2
    assert any(((k+1) % 4, not s) in frames for k, s in frames)
    for frame in frames:
        assert any((d["rotation"], d["up"]) == frame and _evidence_flags(d)["asymmetric"]
                   for d in descriptors["train"])
    rotations = {k for k, s in frames}
    for d in descriptors["test"]:
        frame = (d["rotation"], d["up"])
        assert frame not in frames and frame[0] in rotations, "query must recombine the taught frame factors"


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _must_fail(call):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError("invalid example was accepted")


def self_test():
    """Bounded streaming 42-seed audit, explicit counts, and semantic regressions."""
    fingerprints, input_shapes, output_shapes, structures = set(), set(), set(), set()
    counts, frames, palettes, population_vectors = set(), set(), set(), set()
    shared = irregular = extra_width = overlap = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _describe(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                input_shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                counts.add(len(d["entries"]))
                frames.add((d["rotation"], d["up"]))
                palettes.add(tuple(d[x] for x in ("background", "ink", "guide", "highlight")))
                roles = {}
                normalized = [roles.setdefault(int(v), len(roles)) for v in pair["input"].flat]
                structures.add((pair["input"].shape, bytes(normalized)))
                rows = {}
                for e in d["entries"]:
                    rows[e["pin"][0]] = rows.get(e["pin"][0], 0) + 1
                population_vectors.add(tuple(rows.values()))
                shared += _evidence_flags(d)["shared"]
                irregular += len(rows) >= 3
                extra_width += pair["output"].shape[1] > max(e["block"].shape[1] for e in d["entries"])
                overlap += int(d["union"].sum()) < 5*len(d["entries"])
    assert len(fingerprints) == 42
    assert len(input_shapes) >= 70 and len(output_shapes) >= 35 and len(structures) >= 180
    assert len(palettes) >= 100 and len(population_vectors) >= 15
    assert frames == {(k, s) for k in range(4) for s in (False, True)}
    assert min(counts) == 1 and max(counts) >= 9
    assert shared >= 42 and overlap >= 42 and extra_width >= 42 and irregular >= 10
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    # All 30 legal explicit count combinations; retain only one at a time.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(9000+10*a+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        _must_fail(lambda a=a, b=b: generate_puzzle(7, a, b))
    for seed in (-1, True, 1.5):
        _must_fail(lambda seed=seed: generate_puzzle(seed))

    # Explicit pin registration: width seven, pins at 1 and 4 -> output width ten.
    mask = np.zeros((9, 13), dtype=bool)
    mask[4, 3:10] = True
    mask[3:6, 4] = mask[3:6, 7] = True
    roles = (0, 2, 5, 8)
    for up in (False, True):
        expected = _reference(mask, roles, up)
        assert expected.shape == (6, 10)
        for rotation in range(4):
            pair = _paint(mask, roles, rotation, up)
            assert np.array_equal(solve(pair["input"]), expected)
            for turn in range(4):
                assert np.array_equal(solve(np.rot90(pair["input"], turn)), expected)
    # Adjacent centers in a 3x4 block: simultaneous union has eight cells.
    small = np.ones((3, 4), dtype=bool)
    centers, union = _plus(small)
    assert int(centers.sum()) == 2 and int(union.sum()) == 8
    # All possible 3x3 neighborhoods, plus clipped no-wrap boundary behavior.
    for bits in range(512):
        a = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        cs, us = _plus(a)
        yes = bool(a[1, 1] and a[0, 1] and a[2, 1] and a[1, 0] and a[1, 2])
        assert int(cs.sum()) == int(yes) and int(us.sum()) == 5*int(yes)
    clipped = np.zeros((4, 5), dtype=bool)
    clipped[0, 1:4] = True
    clipped[1, 2] = clipped[-1, 2] = True
    assert not _plus(clipped)[0].any()
    # A crop contains a highlighted arm of a center OUTSIDE that crop.
    mask = np.zeros((10, 13), dtype=bool)
    mask[3, 3:8] = True
    mask[2:5, 4] = True
    mask[5, 4:7] = True
    mask[4:7, 5] = True
    pair = _paint(mask, roles, 1, False)
    assert np.array_equal(solve(pair["input"]), pair["output"])
    d = _describe(pair["input"])
    assert len({e["pin"][0] for e in d["entries"]}) == 2
    assert d["entries"][0]["pin"] == (3, 4) and d["entries"][1]["pin"] == (5, 5)
    assert d["entries"][0]["block"][2, 2] == roles[3]
    # A fixed arbitrary color bijection commutes with the rule.
    permutation = np.array([9, 4, 0, 8, 2, 7, 1, 6, 3, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # Reject damaged labels, damaged key, duplicate inputs, and no-pin drawings.
    p = generate_puzzle(317, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    _must_fail(lambda: validate_generated_puzzle(p))
    p = generate_puzzle(318, 2, 1)
    p["test"][0] = p["train"][0]
    _must_fail(lambda: validate_generated_puzzle(p))
    broken = _paint(mask, roles, 0, False)["input"]
    broken[0, 1] = roles[3]
    _must_fail(lambda: solve(broken))
    no_pin = np.zeros((9, 10), dtype=bool)
    no_pin[4, 3:7] = True
    no_pin[3, 5] = True
    g = np.full(no_pin.shape, roles[0], dtype=np.int8)
    g[no_pin] = roles[1]
    g[:, 0] = roles[2]
    g[0, 1:] = np.where(np.arange(1, g.shape[1]) % 2, roles[2], roles[0])
    g[0, 0] = roles[3]
    _must_fail(lambda: solve(g))
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "color_normalized_inputs": len(structures), "palettes": len(palettes),
            "pin_counts": sorted(counts), "population_vectors": len(population_vectors),
            "shared_ribbon_pairs": shared, "overlap_pairs": overlap,
            "wider_than_any_ribbon_pairs": extra_width, "irregular_row_pairs": irregular,
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
