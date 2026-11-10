"""Register a guide-defined rectangle, then mark its empty 3x3-window union.

No parent module, file, seed, or training pair is used by solve().
Generation and self-tests do not write artifacts or retain a seed sweep.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
NAMES = ("identity", "r90", "r180", "r270", "flip_lr", "flip_ud",
         "transpose", "anti_transpose")
SIDE_MAPS = (
    ("T", "R", "B", "L"), ("L", "T", "R", "B"),
    ("B", "L", "T", "R"), ("R", "B", "L", "T"),
    ("T", "L", "B", "R"), ("B", "R", "T", "L"),
    ("L", "B", "R", "T"), ("R", "T", "L", "B"),
)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _edge(a, side):
    return {"T": lambda: a[0], "R": lambda: a[:, -1],
            "B": lambda: a[-1], "L": lambda: a[:, 0]}[side]()


def _adjacent(sides):
    return len(sides) == 2 and any(set(sides) == {SIDES[i], SIDES[(i + 1) % 4]}
                                   for i in range(4))


def _components(a, color):
    seen = np.zeros(a.shape, dtype=bool)
    for rr, cc in np.argwhere(a == color):
        pos = int(rr), int(cc)
        if seen[pos]:
            continue
        seen[pos] = True
        todo, cells = [pos], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                y, x = r + dr, c + dc
                if (0 <= y < a.shape[0] and 0 <= x < a.shape[1]
                        and not seen[y, x] and a[y, x] == color):
                    seen[y, x] = True
                    todo.append((y, x))
        yield cells


def _segment(a, side, bounds):
    r, c, h, w = bounds
    if side == "T":
        return a[r - 1, c:c + w]
    if side == "B":
        return a[r + h, c:c + w]
    if side == "L":
        return a[r:r + h, c - 1]
    return a[r:r + h, c + w]


def _decode(input_grid):
    a = _grid(input_grid)
    if min(a.shape) < 7 or np.any(a == 1):
        raise ValueError("highlight 1 is absent from inputs; frame must surround guides")
    outer = {}
    for side in SIDES:
        core = _edge(a, side)[1:-1]
        if len(core) and core[0] >= 2 and np.all(core == core[0]):
            outer[side] = int(core[0])
    if not _adjacent(outer) or len(set(outer.values())) != 2:
        raise ValueError("need two distinct complete adjacent outer controls")
    H, W = a.shape
    bars = []
    for color in sorted(outer.values()):
        interior = [cells for cells in _components(a, color)
                    if not any(r in (0, H - 1) or c in (0, W - 1) for r, c in cells)]
        if len(interior) != 1:
            raise ValueError("each control identity must have exactly one interior guide")
        cells = interior[0]
        rr, cc = zip(*cells)
        if len(cells) < 4 or (len(set(rr)) != 1 and len(set(cc)) != 1):
            raise ValueError("guides must be straight bars of length at least four")
        bars.append((min(rr), min(cc), max(rr), max(cc), color))
    horiz = [b for b in bars if b[0] == b[2]]
    vert = [b for b in bars if b[1] == b[3]]
    if len(horiz) != 1 or len(vert) != 1:
        raise ValueError("need one horizontal and one vertical guide")
    hr, c, _, cend, hc = horiz[0]
    r, vc, rend, _, vcolor = vert[0]
    h, w = rend - r + 1, cend - c + 1
    hs = "T" if hr == r - 1 else "B" if hr == r + h else None
    vs = "L" if vc == c - 1 else "R" if vc == c + w else None
    if hs is None or vs is None:
        raise ValueError("guide endpoints do not define adjacent rectangle sides")
    guides, bounds = {hs: hc, vs: vcolor}, (r, c, h, w)
    r0, r1 = r - (hs == "T"), r + h + (hs == "B")
    c0, c1 = c - (vs == "L"), c + w + (vs == "R")
    if r0 < 2 or c0 < 2 or r1 > H - 2 or c1 > W - 2:
        raise ValueError("guides need a clear background moat inside the frame")
    ring = np.concatenate((a[r0 - 1, c0 - 1:c1 + 1], a[r1, c0 - 1:c1 + 1],
                           a[r0:r1, c0 - 1], a[r0:r1, c1]))
    if np.any(ring != 0):
        raise ValueError("nonzero guide moat")
    payload = a[r:r + h, c:c + w].copy()
    if set(np.unique(payload)) & set(outer.values()):
        raise ValueError("marker colors cannot occur in the payload")
    for side in SIDES:
        if np.any(_segment(a, side, bounds) != guides.get(side, 0)):
            raise ValueError("unexpected material directly outside payload")
    outside = a.copy()
    outside[r0:r1, c0:c1] = 0
    if np.any(outside[2:-2, 2:-2]):
        raise ValueError("only the one guide-defined payload is allowed away from frame")
    transforms = [t for t, mapping in enumerate(SIDE_MAPS)
                  if all(outer.get(mapping[SIDES.index(s)]) == color
                         for s, color in guides.items())]
    if len(transforms) != 1:
        raise ValueError("orientation is not unique")
    # Orthogonal side normals determine a signed permutation matrix.
    matrix = np.zeros((2, 2), dtype=int)
    for side, color in guides.items():
        target = next(s for s, v in outer.items() if v == color)
        matrix += np.outer(NORMALS[target], NORMALS[side])
    return dict(payload=payload, bounds=bounds, guides=guides, outer=outer,
                transform=transforms[0], matrix=matrix)


def _orient(a, transform):
    if transform < 4:
        return np.rot90(a, transform).copy()
    if transform == 4:
        return np.fliplr(a).copy()
    if transform == 5:
        return np.flipud(a).copy()
    if transform == 6:
        return a.T.copy()
    return np.rot90(a.T, 2).copy()


def _coordinate_orient(a, matrix):
    coordinates = matrix @ np.indices(a.shape).reshape(2, -1)
    coordinates -= coordinates.min(axis=1, keepdims=True)
    result = np.empty(tuple(coordinates.max(axis=1) + 1), dtype=np.int8)
    result[coordinates[0], coordinates[1]] = a.ravel()
    return result


def _windows(a):
    h, w = a.shape
    counts = np.zeros((max(0, h - 2), max(0, w - 2)), dtype=np.int8)
    for dr in range(3):
        for dc in range(3):
            counts += (a[dr:dr + counts.shape[0], dc:dc + counts.shape[1]] == 0)
    return [tuple(map(int, p)) for p in np.argwhere(counts == 9)]


def _union(shape, starts):
    mask = np.zeros(shape, dtype=bool)
    for r, c in starts:
        mask[r:r + 3, c:c + 3] = True
    return mask


def _highlight(a):
    out = a.copy()
    out[_union(a.shape, _windows(a))] = 1
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read local marker correspondence; crop, orient, then union empty windows."""
    info = _decode(input_grid)
    registered = _coordinate_orient(info["payload"], info["matrix"])
    return _highlight(registered)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _palette(rng):
    colors = [int(c) for c in rng.permutation(np.arange(2, 10))]
    return colors[:2], colors[2:2 + int(rng.integers(1, 5))], colors[-1]


def _stamp(rng, a, reserved, shape, colors, *, spur=False, miss=False, interior=False):
    h, w = a.shape
    ph, pw = shape
    candidates = []
    for r in range(int(interior), h - ph + 1 - int(interior)):
        for c in range(int(interior), w - pw + 1 - int(interior)):
            y0, y1 = max(0, r - 1), min(h, r + ph + 1)
            x0, x1 = max(0, c - 1), min(w, c + pw + 1)
            if not np.any(reserved[y0:y1, x0:x1]):
                candidates.append((r, c, y0, y1, x0, x1))
    if not candidates:
        return False
    r, c, y0, y1, x0, x1 = _pick(rng, candidates)
    a[y0:y1, x0:x1] = rng.choice(colors, size=(y1 - y0, x1 - x0))
    a[r:r + ph, c:c + pw] = 0
    if miss:
        a[r + int(rng.integers(ph)), c + int(rng.integers(pw))] = _pick(rng, colors)
    if spur:
        options = []
        if r > 0:
            options += [(r - 1, c + j) for j in range(pw)]
        if r + ph < h:
            options += [(r + ph, c + j) for j in range(pw)]
        if c > 0:
            options += [(r + i, c - 1) for i in range(ph)]
        if c + pw < w:
            options += [(r + i, c + pw) for i in range(ph)]
        if options:
            a[_pick(rng, options)] = 0
    reserved[y0:y1, x0:x1] = True
    return True


def _payload(rng, shape, colors, patches, blank_side, witness):
    # At most 100 small in-memory attempts; no retained failed canvases.
    for _ in range(100):
        a = rng.choice(colors, size=shape).astype(np.int8)
        a[rng.random(shape) < float(rng.uniform(.12, .49))] = 0
        reserved = np.zeros(shape, dtype=bool)
        if blank_side is not None:
            thickness = 1 if witness else int(rng.integers(1, 3))
            for d in range(thickness):
                line = {"T": (d, slice(None)), "B": (-1 - d, slice(None)),
                        "L": (slice(None), d), "R": (slice(None), -1 - d)}[blank_side]
                a[line], reserved[line] = 0, True
        success = True
        for i, patch in enumerate(patches):
            if not _stamp(rng, a, reserved, patch, colors,
                          spur=(i == 1 if witness else bool(rng.integers(2))),
                          interior=witness):
                success = False
                break
        if not success or not _stamp(rng, a, reserved, (3, 3), colors, miss=True):
            continue
        if witness and len({_key(_highlight(_orient(a, t))) for t in range(8)}) != 8:
            continue
        return a
    raise RuntimeError("bounded payload packing exhausted")


def _embed(rng, payload, inner, markers, noise, transform):
    h, w = payload.shape
    top, right, bottom, left = [int(rng.integers(3, 6)) for _ in range(4)]
    H, W = min(30, h + top + bottom), min(30, w + left + right)
    # Every guide has room for its one-cell moat, even on a 30-cell canvas.
    r = int(rng.integers(3, H - h - 2))
    c = int(rng.integers(3, W - w - 2))
    a = np.zeros((H, W), dtype=np.int8)
    guides = dict(zip(inner, markers))
    outer = {SIDE_MAPS[transform][SIDES.index(s)]: color for s, color in guides.items()}
    protected = np.zeros((H, W), dtype=bool)
    r0, r1 = r - ("T" in guides), r + h + ("B" in guides)
    c0, c1 = c - ("L" in guides), c + w + ("R" in guides)
    protected[r0 - 1:r1 + 1, c0 - 1:c1 + 1] = True
    if rng.random() < .65:
        for side in SIDES:
            if side not in outer:
                line = _edge(a, side)
                step = int(rng.integers(2, 6))
                line[1 + int(rng.integers(step)):len(line) - 1:step] = noise
    for side in rng.permutation(list(outer)):
        _edge(a, str(side))[:] = outer[str(side)]
    if rng.random() < .5:
        for side, color in outer.items():
            length = W if side in ("T", "B") else H
            for p in rng.choice(np.arange(2, length - 2), size=2, replace=False):
                pos = {"T": (1, p), "B": (H - 2, p),
                       "L": (p, 1), "R": (p, W - 2)}[side]
                if not protected[pos]:
                    a[pos] = color
    a[r:r + h, c:c + w] = payload
    for side, color in guides.items():
        _segment(a, side, (r, c, h, w))[:] = color
    return {"input": a, "output": _highlight(_orient(payload, transform))}


def _arrangement(info):
    return frozenset(info["guides"]), frozenset(info["outer"])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ne = _integer(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .25)
    palette = _palette(rng)
    h, w = int(rng.integers(12, 20)), int(rng.integers(12, 20))
    if h == w:
        w += 1
    inner_index = int(rng.integers(4))
    inner = (SIDES[inner_index], SIDES[(inner_index + 1) % 4])
    transform = int(_pick(rng, (1, 3)))
    blank_side = _pick(rng, SIDES)
    patches = [(3, 3), _pick(rng, ((3, 4), (4, 3)))]
    markers, textures, noise = palette
    payload = _payload(rng, (h, w), textures, patches, blank_side, True)
    first = _embed(rng, payload, inner, markers, noise, transform)
    second = {k: v.copy() for k, v in first.items()}
    replacements = []
    for color, other in zip(markers, markers[::-1]):
        for component in _components(first["input"], color):
            H, W = first["input"].shape
            if any(r in (0, H - 1) or c in (0, W - 1) for r, c in component):
                replacements.extend((pos, other) for pos in component)
    for pos, color in replacements:
        second["input"][pos] = color
    t2 = _decode(second["input"])["transform"]
    second["output"] = _highlight(_orient(payload, t2))
    puzzle = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    train_arrangements = {_arrangement(_decode(first["input"]))}
    for split, count in (("train", nt - 2), ("test", ne)):
        for _ in range(count):
            for attempt in range(30):
                markers, textures, noise = palette if stable else _palette(rng)
                possibilities = []
                for j in range(4):
                    sides = (SIDES[j], SIDES[(j + 1) % 4])
                    for t in range(8):
                        outer = frozenset(SIDE_MAPS[t][SIDES.index(s)] for s in sides)
                        if split == "train" or (frozenset(sides), outer) not in train_arrangements:
                            possibilities.append((sides, t))
                sides, t = _pick(rng, possibilities)
                n = int(rng.integers(1, 5))
                low = (9, 12, 15, 18)[n - 1]
                shape = (int(rng.integers(low, 25)), int(rng.integers(low, 25)))
                if rng.random() < .2:
                    shape = (shape[0], shape[0])
                patches = [_pick(rng, ((3, 3), (3, 4), (4, 3), (4, 4),
                                      (3, 5), (5, 3), (4, 5), (5, 4), (5, 5))) for _ in range(n)]
                side = _pick(rng, SIDES) if rng.random() < .45 else None
                try:
                    content = _payload(rng, shape, textures, patches, side, False)
                except RuntimeError:
                    continue
                pair = _embed(rng, content, sides, markers, noise, t)
                key = _key(pair["input"])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_arrangements.add(_arrangement(_decode(pair["input"])))
                break
            else:
                raise RuntimeError("bounded pair generation exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(pair):
    info = _decode(pair["input"])
    a = info["payload"]
    starts = _windows(a)
    mask = _union(a.shape, starts)
    info.update(starts=starts, mask=mask,
                overlap=int(mask.sum()) < 9 * len(starts),
                blank_edge=any(np.all(_edge(a, s) == 0) for s in SIDES),
                partial_component=any(any(mask[p] for p in cells)
                                      and any(not mask[p] for p in cells)
                                      for cells in _components(a, 0)),
                near_miss=any(np.count_nonzero(a[r:r + 3, c:c + 3] == 0) == 8
                              and not np.any(mask[r:r + 3, c:c + 3])
                              for r in range(a.shape[0] - 2) for c in range(a.shape[1] - 2)))
    return info


def validate_generated_puzzle(puzzle):
    """Check returned-grid semantics, training witnesses, and held-out geometry."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            f = _facts(pair)
            info[split].append(f)
            registered = _orient(f["payload"], f["transform"])
            assert np.array_equal(registered, _coordinate_orient(f["payload"], f["matrix"]))
            # Scalar enumeration is a second check of the vector window scanner.
            mask = np.zeros(registered.shape, dtype=bool)
            for r in range(registered.shape[0] - 2):
                for c in range(registered.shape[1] - 2):
                    if all(registered[r + dr, c + dc] == 0 for dr in range(3) for dc in range(3)):
                        mask[r:r + 3, c:c + 3] = True
            assert np.any(mask) and np.any((registered == 0) & ~mask)
            assert out.shape == registered.shape
            assert np.array_equal(out != registered, mask)
            assert np.all(out[mask] == 1)
            assert np.array_equal(out[~mask], registered[~mask])
            assert np.array_equal(solve(inp), out)
            # Removing A gives a lossless crop with no highlights: always wrong.
            assert not np.array_equal(out, registered)
    a, b = info["train"][:2]
    assert a["bounds"] == b["bounds"] and a["guides"] == b["guides"]
    assert np.array_equal(a["payload"], b["payload"])
    assert set(a["outer"]) == set(b["outer"])
    assert all(a["outer"][s] != b["outer"][s] for s in a["outer"])
    assert a["transform"] in (1, 3) and b["transform"] >= 4
    assert len({_key(_highlight(_orient(a["payload"], t))) for t in range(8)}) == 8
    assert not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"])
    assert a["overlap"] and a["partial_component"] and a["near_miss"] and a["blank_edge"]
    assert any(len(cells) == 9 and all(a["mask"][p] for p in cells)
               for cells in _components(a["payload"], 0))
    # Removing B's orientation cannot fit the contrast; tight cropping loses a strip.
    assert not np.array_equal(_highlight(a["payload"]), puzzle["train"][0]["output"])
    arrangements = {_arrangement(f) for f in info["train"]}
    assert all(_arrangement(f) not in arrangements for f in info["test"])


def self_test():
    """Bounded 42-seed sweep; return aggregate observations, save no grids."""
    shapes, payload_shapes, palettes, windows, areas, signatures = (set() for _ in range(6))
    symmetries = {"train": set(), "test": set()}
    counts = {"overlap_pairs": 0, "blank_edge_pairs": 0, "partial_components": 0,
              "near_miss_pairs": 0, "pairs": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _facts(pair)
                digest.update(pair["input"].tobytes())
                shapes.add(pair["input"].shape)
                payload_shapes.add(f["payload"].shape)
                palettes.add(tuple(sorted(f["outer"].values())))
                windows.add(len(f["starts"]))
                areas.add(int(f["mask"].sum()))
                symmetries[split].add(f["transform"])
                counts["pairs"] += 1
                for name, flag in (("overlap_pairs", "overlap"), ("blank_edge_pairs", "blank_edge"),
                                   ("partial_components", "partial_component"), ("near_miss_pairs", "near_miss")):
                    counts[name] += int(f[flag])
        signatures.add(digest.digest())
        if seed < 2:
            other = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], other[s]) for k in a)
    assert len(signatures) == 42 and len(shapes) >= 30 and len(payload_shapes) >= 30
    assert len(palettes) >= 15 and len(windows) >= 10 and len(areas) >= 10
    assert symmetries["train"] == symmetries["test"] == set(range(8))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(10000 + 10 * nt + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0)
    for kind in ("cell", "crop", "guide"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        pair = bad["test"][0]
        if kind == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif kind == "crop":
            pair["output"] = pair["output"][1:].copy()
        else:
            f = _decode(pair["input"])
            line = _segment(pair["input"], next(iter(f["guides"])), f["bounds"])
            line[len(line) // 2] = 0
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("corrupted puzzle accepted")
    # Explicitly distinguish whole-window union from clipped-neighborhood centers.
    patch = np.zeros((3, 4), dtype=np.int8)
    assert np.all(_highlight(patch) == 1)
    no_fit = np.full((4, 5), 2, dtype=np.int8)
    no_fit[:2] = 0
    assert np.array_equal(_highlight(no_fit), no_fit)
    return dict(seeds=42, **counts, distinct_input_shapes=len(shapes),
                distinct_payload_shapes=len(payload_shapes), marker_pairs=len(palettes),
                window_counts=sorted(windows), union_areas=sorted(areas),
                symmetries={s: [NAMES[t] for t in sorted(v)] for s, v in symmetries.items()})
