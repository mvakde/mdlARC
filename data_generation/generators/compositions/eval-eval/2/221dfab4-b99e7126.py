"""Clocked stencil printer: the blueprint and the printing clock locate each other.

Only NumPy and the standard library are used. No evidence reads or artifact writes
occur here. All semantic parameters, including both ink colors, are input-visible.
"""
from functools import lru_cache
import hashlib

import numpy as np


_SHAPES = tuple((h, w) for h in range(2, 5) for w in range(2, 5))


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _check(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _forward(a, k, flip):
    b = np.rot90(a, k)
    return np.fliplr(b) if flip else b


def _backward(a, k, flip):
    return np.rot90(np.fliplr(a) if flip else a, -k).copy()


def _header(a):
    """Find the unique frame view with the edge bar pointing to the right."""
    candidates = []
    for k in range(4):
        for flip in (False, True):
            g = _forward(a, k, flip)
            d = int(g[0, 0])
            if not (np.all(g[-1] == d) and np.all(g[:, 0] == d)
                    and np.all(g[:, -1] == d)):
                continue
            cells = np.flatnonzero(g[0] != d)
            if len(cells) != 3 or cells[0] == 0 or cells[-1] == g.shape[1] - 1:
                continue
            if not np.array_equal(cells, np.arange(cells[0], cells[-1] + 1)):
                continue
            colors = g[0, cells]
            if np.all(colors[:-1] == colors[0]) and colors[-1] != colors[0]:
                candidates.append((g.copy(), k, flip, d, int(colors[0]),
                                   int(colors[-1]), int(cells[0]), len(cells)))
    _check(len(candidates) == 1, "need one directed, two-color edge bar")
    return candidates[0]


def _runs(indices):
    result = []
    for x in map(int, indices):
        if result and x == result[-1][1] + 1:
            result[-1] = result[-1][0], x
        else:
            result.append((x, x))
    return result


def _lattice(g, divider, start, length):
    repaired = g.copy()
    repaired[0, start:start + length] = divider
    rr = _runs(np.flatnonzero(np.all(repaired == divider, axis=1)))
    cc = _runs(np.flatnonzero(np.all(repaired == divider, axis=0)))
    _check(len(rr) >= 3 and len(cc) >= 3, "missing lattice")
    _check(rr[0][0] == cc[0][0] == 0 and rr[-1][1] == g.shape[0] - 1
           and cc[-1][1] == g.shape[1] - 1, "incomplete outer frame")
    thickness = {b - a + 1 for a, b in rr + cc}
    heights = {rr[i + 1][0] - rr[i][1] - 1 for i in range(len(rr) - 1)}
    widths = {cc[i + 1][0] - cc[i][1] - 1 for i in range(len(cc) - 1)}
    _check(len(thickness) == len(heights) == len(widths) == 1, "irregular lattice")
    t, h, w = thickness.pop(), heights.pop(), widths.pop()
    _check(t in (1, 2) and (h, w) in _SHAPES, "unsupported tile geometry")
    nr, nc = len(rr) - 1, len(cc) - 1
    _check(nr >= h and nc >= w, "blueprint cannot fit")
    return h, w, t, nr, nc


def _phase_stencil(mask):
    """-1 means inactive, 0 ordinary ink, 1 every third occupied position."""
    phase = np.full(mask.shape, -1, dtype=np.int8)
    points = [tuple(map(int, p)) for p in np.argwhere(mask)]
    for i, p in enumerate(points):
        phase[p] = int(i % 3 == 2)
    return points, phase


def _origins(mask, phase, nr, nc, clues, colored=True):
    h, w = mask.shape
    result = []
    for r in range(nr - h + 1):
        for c in range(nc - w + 1):
            if all(0 <= y - r < h and 0 <= x - c < w and mask[y - r, x - c]
                   and (not colored or int(phase[y - r, x - c]) == tint)
                   for y, x, tint in clues):
                result.append((r, c))
    return result


def _decode(value):
    a = _grid(value)
    g, k, flip, d, seed, accent, start, length = _header(a)
    geometry = _lattice(g, d, start, length)
    h, w, t, nr, nc = geometry
    ordinary = glyph_mask = background = None
    clues = []
    ordinary_count = 0
    for r in range(nr):
        for c in range(nc):
            y, x = t + r * (h + t), t + c * (w + t)
            tile = g[y:y + h, x:x + w]
            _check(not np.any(tile == d), "divider color inside a tile")
            ink = (tile == seed) | (tile == accent)
            if ink.any():
                _check(3 <= int(ink.sum()) < h * w, "invalid glyph mask")
                ink_colors = np.unique(tile[ink])
                back_colors = np.unique(tile[~ink])
                _check(len(ink_colors) == len(back_colors) == 1, "glyph must have uniform ink and background")
                if glyph_mask is None:
                    glyph_mask, background = ink.copy(), int(back_colors[0])
                _check(np.array_equal(ink, glyph_mask) and int(back_colors[0]) == background,
                       "glyph copies disagree")
                clues.append((r, c, int(ink_colors[0] == accent)))
            else:
                if ordinary is None:
                    ordinary = tile.copy()
                _check(np.array_equal(tile, ordinary), "ordinary tiles disagree")
                ordinary_count += 1
    _check(glyph_mask is not None and ordinary is not None, "need ordinary and glyph tiles")
    _check(ordinary_count > len(clues) >= 2, "need several clues and a strict ordinary majority")
    _check(2 <= len(np.unique(ordinary)) <= 3 and background in ordinary,
           "invalid ordinary tile palette")
    _check(len({d, seed, accent, background}) == 4, "colliding palette roles")
    points, phase = _phase_stencil(glyph_mask)
    _check(len(clues) < len(points), "input must be incomplete")
    candidates = _origins(glyph_mask, phase, nr, nc, clues)
    _check(len(candidates) == 1, "colored clues must register exactly one stencil")
    return dict(canonical=g, k=k, flip=flip, divider=d, seed=seed, accent=accent,
                header_start=start, header_length=length, geometry=geometry,
                ordinary=ordinary, background=background, mask=glyph_mask,
                points=points, phase=phase, origin=candidates[0], clues=clues,
                geometric_origins=_origins(glyph_mask, phase, nr, nc, clues, False))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the clock, blueprint, translation, and whole-tile completion from input."""
    f = _decode(input_grid)
    g = f["canonical"].copy()
    h, w, t, _, _ = f["geometry"]
    ro, co = f["origin"]
    supplied = {(r, c) for r, c, _ in f["clues"]}
    for r, c in f["points"]:
        if (ro + r, co + c) not in supplied:
            ink = f["accent"] if f["phase"][r, c] else f["seed"]
            tile = np.where(f["mask"], ink, f["background"])
            y, x = t + (ro + r) * (h + t), t + (co + c) * (w + t)
            g[y:y + h, x:x + w] = tile
    return _backward(g, f["k"], f["flip"])


@lru_cache(maxsize=13)
def _codes(n):
    values = np.arange(1, 1 << n, dtype=np.uint32)
    counts = np.fromiter((int(x).bit_count() for x in values), dtype=np.uint8)
    values.flags.writeable = counts.flags.writeable = False
    return values, counts


def _fragment_options(mask, geometry, origin):
    h, w, _, nr, nc = geometry
    points, phase = _phase_stencil(mask)
    values, counts = _codes(len(points))
    limit = min(len(points) - 1, (nr * nc - 1) // 2)
    valid = values[(counts >= 2) & (counts <= limit)].copy()
    geom_covers = []
    for r in range(nr - h + 1):
        for c in range(nc - w + 1):
            if (r, c) == origin:
                continue
            cover = geom = 0
            for i, (y, x) in enumerate(points):
                yy, xx = origin[0] + y - r, origin[1] + x - c
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                    geom |= 1 << i
                    if phase[yy, xx] == phase[y, x]:
                        cover |= 1 << i
            valid = valid[(valid & cover) != valid]
            geom_covers.append(geom)
    ambiguous = np.zeros(len(valid), dtype=bool)
    for cover in geom_covers:
        ambiguous |= (valid & cover) == valid
    return points, phase, valid, ambiguous


def _asymmetric(mask):
    return all(a.shape != mask.shape or not np.array_equal(a, mask)
               for i, a in enumerate([np.rot90(mask, k) for k in range(4)]
                                      + [np.fliplr(np.rot90(mask, k)) for k in range(4)])
               if i != 0)


def _witness_bits(mask):
    points, phase = _phase_stencil(mask)
    bits = {name: 0 for name in ("seed", "accent", "carry", "slots", "snake")}
    for i, (r, c) in enumerate(points):
        bits["accent" if i % 3 == 2 else "seed"] |= 1 << i
        row_rank = sum(mask[r, :c])
        if bool(int(row_rank) % 3 == 2) != bool(phase[r, c]):
            bits["carry"] |= 1 << i
        if bool((r * mask.shape[1] + c) % 3 == 2) != bool(phase[r, c]):
            bits["slots"] |= 1 << i
        previous = int(mask[:r].sum())
        within = int(mask[r, c + 1:].sum()) if r % 2 else int(mask[r, :c].sum())
        if bool((previous + within) % 3 == 2) != bool(phase[r, c]):
            bits["snake"] |= 1 << i
    return bits


def _choose_fragment(rng, mask, geometry, origin, mode):
    points, phase, codes, coupled = _fragment_options(mask, geometry, origin)
    if mode == "clock_location":
        codes = codes[coupled]
    if mode == "teaching":
        if len(points) < 6 or not _asymmetric(mask):
            return None
        bits = _witness_bits(mask)
        full = (1 << len(points)) - 1
        keep = []
        for value in codes:
            code = int(value)
            missing = full ^ code
            visible = [i for i in range(len(points)) if code & (1 << i)]
            visible_only_wrong = any((i % 3 == 2) != (j % 3 == 2)
                                     for j, i in enumerate(visible))
            if (code.bit_count() >= 3 and code & bits["seed"] and code & bits["accent"]
                    and visible_only_wrong and all(missing & bits[name] for name in bits)):
                keep.append(code)
        codes = np.asarray(keep, dtype=np.uint32)
    if mode == "test":
        small = codes[np.fromiter((int(c).bit_count() <= 4 for c in codes), bool)]
        if len(small):
            codes = small
    if not len(codes):
        return None
    sizes = np.fromiter((int(c).bit_count() for c in codes), np.uint8)
    size = int(rng.choice(np.unique(sizes)))
    code = int(rng.choice(codes[sizes == size]))
    return [(origin[0] + r, origin[1] + c, int(phase[r, c]))
            for i, (r, c) in enumerate(points) if code & (1 << i)]


def _sample_geometry(rng, mode):
    shapes = [s for s in _SHAPES if s[0] * s[1] >= (8 if mode == "teaching" else
                                                   6 if mode == "clock_location" else 4)]
    h, w = shapes[int(rng.integers(len(shapes)))]
    t = int(rng.choice([1, 2], p=[.75, .25]))
    def axis(n):
        maximum = (30 - t) // (n + t)
        return maximum if rng.random() < .4 else int(rng.integers(n, maximum + 1))
    nr, nc = axis(h), axis(w)
    if nr * nc == 4:
        nr += 1
    return h, w, t, nr, nc


def _sample_mask(rng, shape):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    mode = int(rng.integers(8))
    if mode == 0:
        mask[:, (0, -1)] = True
        mask[-1] = True
    elif mode == 1:
        mask[(0, -1), :] = True
        mask[:, (0, -1)] = True
    elif mode == 2:
        mask = np.indices(shape).sum(axis=0) % 2 == 0
    else:
        count = int(rng.integers(3, h * w))
        mask.flat[rng.choice(h * w, count, replace=False)] = True
    if not 3 <= int(mask.sum()) < mask.size:
        mask[:] = False
        mask.flat[rng.choice(mask.size, 3, replace=False)] = True
    return mask


def _paint(geometry, palette, ordinary, mask, origin, positions, start, length):
    h, w, t, nr, nc = geometry
    divider, seed, accent, background = palette[:4]
    grid = np.full((nr * (h + t) + t, nc * (w + t) + t), divider, dtype=np.int8)
    selected = {(r, c): tint for r, c, tint in positions}
    for r in range(nr):
        for c in range(nc):
            if (r, c) in selected:
                ink = accent if selected[r, c] else seed
                tile = np.where(mask, ink, background)
            else:
                tile = ordinary
            y, x = t + r * (h + t), t + c * (w + t)
            grid[y:y + h, x:x + w] = tile
    grid[0, start:start + length - 1] = seed
    grid[0, start + length - 1] = accent
    return grid


def _assemble(rng, geometry, palette, mask, origin, clues, k, flip):
    h, w, t, _, nc = geometry
    ncolors = int(rng.choice([2, 3], p=[.8, .2]))
    colors = np.asarray([palette[3], *palette[4:4 + ncolors - 1]], dtype=np.int8)
    labels = np.concatenate((np.arange(ncolors), rng.integers(ncolors, size=h * w - ncolors)))
    rng.shuffle(labels)
    ordinary = colors[labels.reshape(h, w)]
    # Ensure a visible witness that copying only ink would leave a wrong pixel.
    inactive = np.argwhere(~mask)
    if not np.any(ordinary[~mask] != palette[3]):
        r, c = inactive[int(rng.integers(len(inactive)))]
        yy, xx = np.argwhere(ordinary == palette[4])[0]
        ordinary[yy, xx] = palette[3]
        ordinary[r, c] = palette[4]
    width = nc * (w + t) + t
    length = 3  # The visible clock word itself: seed, seed, accent.
    start = int(rng.integers(1, width - length))
    points, phase = _phase_stencil(mask)
    complete = [(origin[0] + r, origin[1] + c, int(phase[r, c])) for r, c in points]
    args = (geometry, palette, ordinary, mask, origin)
    a = _paint(*args, clues, start, length)
    b = _paint(*args, complete, start, length)
    return {"input": _backward(a, k, flip), "output": _backward(b, k, flip)}


def _new_pair(rng, mode, palette, forbidden, used, k, flip):
    for attempt in range(96 + 510):
        if attempt < 96:
            geometry = _sample_geometry(rng, mode)
            mask = _sample_mask(rng, geometry[:2])
            origin = (int(rng.integers(geometry[3] - geometry[0] + 1)),
                      int(rng.integers(geometry[4] - geometry[1] + 1)))
        else:
            # Finite fallback over 510 binary designs, not a fixed glyph alphabet.
            geometry = (3, 3, 1, 7, 7)
            code = attempt - 95
            mask = np.array([bool(code & (1 << i)) for i in range(9)]).reshape(3, 3)
            if not 3 <= int(mask.sum()) < 9:
                continue
            origin = (1, 2)
        if _key(mask) in forbidden:
            continue
        clues = _choose_fragment(rng, mask, geometry, origin, mode)
        if clues is None:
            continue
        pair = _assemble(rng, geometry, palette, mask, origin, clues, k, flip)
        if _key(pair["input"]) not in used:
            return pair
    raise RuntimeError("bounded stencil construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} outside legal range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    stable_palette = bool(rng.random() < .25)
    base_palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    base_k, base_flip = int(rng.integers(4)), bool(rng.integers(2))
    orientation_regime = int(rng.integers(3))
    puzzle, used, training_masks = {"train": [], "test": []}, set(), set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            mode = ("teaching" if index == 0 else "clock_location" if index == 1 else "normal") if split == "train" else "test"
            palette = base_palette if stable_palette else tuple(map(int, rng.choice(10, 6, replace=False)))
            if orientation_regime == 0:
                k, flip = base_k, base_flip
            elif orientation_regime == 1:
                k, flip = (base_k + (split == "test")) % 4, base_flip
            else:
                k, flip = int(rng.integers(4)), bool(rng.integers(2))
            forbidden = training_masks if split == "test" or index == 1 else set()
            pair = _new_pair(rng, mode, palette, forbidden, used, k, flip)
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
            if split == "train":
                training_masks.add(_key(_decode(pair["input"])["mask"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    _check(a.shape == b.shape, "shape changed")
    f = _decode(a)
    h, w, t, nr, nc = f["geometry"]
    ro, co = f["origin"]
    output = _forward(b, f["k"], f["flip"])
    # Validate tiles directly, rather than invoking the solver's missing-tile painter.
    untiled = np.ones(f["canonical"].shape, dtype=bool)
    for r in range(nr):
        for c in range(nc):
            yy, xx = r - ro, c - co
            active = 0 <= yy < h and 0 <= xx < w and f["mask"][yy, xx]
            expected = f["ordinary"]
            if active:
                ink = f["accent"] if f["phase"][yy, xx] else f["seed"]
                expected = np.where(f["mask"], ink, f["background"])
            y, x = t + r * (h + t), t + c * (w + t)
            _check(np.array_equal(output[y:y + h, x:x + w], expected), "incorrect whole-tile completion")
            untiled[y:y + h, x:x + w] = False
    _check(np.array_equal(output[untiled], f["canonical"][untiled]), "frame or divider changed")
    _check(np.array_equal(solve(a), b), "input-only solve disagrees")
    _check(not np.array_equal(a, b), "identity output")
    supplied = {(r - ro, c - co) for r, c, _ in f["clues"]}
    missing_code = sum(1 << i for i, p in enumerate(f["points"]) if p not in supplied)
    bits = _witness_bits(f["mask"])
    f["evidence"] = {name: bool(missing_code & bit) for name, bit in bits.items()}
    visible = [i for i, p in enumerate(f["points"]) if p in supplied]
    f["evidence"]["visible_only"] = any((i % 3 == 2) != (j % 3 == 2)
                                         for j, i in enumerate(visible))
    f["background_changes"] = bool(np.any(f["ordinary"][~f["mask"]] != f["background"]))
    return f


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid counts")
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            f = _pair_info(pair)
            key = _key(np.asarray(pair["input"]))
            _check(key not in seen, "duplicate input")
            seen.add(key)
            info[split].append(f)
    training = info["train"]
    masks = {_key(f["mask"]) for f in training}
    _check(len(masks) >= 2, "training needs different blueprints")
    _check(any(len(f["geometric_origins"]) > 1 for f in training), "training must teach clock-based registration")
    for name in ("seed", "accent", "carry", "slots", "snake", "visible_only"):
        _check(any(f["evidence"][name] for f in training), f"training lacks {name} witness")
    _check(any(f["background_changes"] for f in training), "training lacks whole-tile-copy witness")
    _check(any(len(f["points"]) >= 6 for f in training), "training must show a repeated clock")
    _check(any(_asymmetric(f["mask"]) for f in training), "training must establish stencil orientation")
    _check(all(_key(f["mask"]) not in masks for f in info["test"]), "tests must use new blueprints")


def _puzzle_key(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds, count structural axes, and exercise corruption/boundaries."""
    signatures, shapes, masks, geometries, palettes, orientations, fragments = (set() for _ in range(7))
    coupled = carry = whole = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        if seed in (0, 41):
            _check(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _pair_info(pair)
                shapes.add(pair["input"].shape)
                masks.add(_key(f["mask"]))
                geometries.add(f["geometry"])
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                orientations.add((f["k"], f["flip"]))
                ro, co = f["origin"]
                fragments.add((_key(f["mask"]), tuple((r - ro, c - co) for r, c, _ in f["clues"])))
                coupled += len(f["geometric_origins"]) > 1
                carry += f["evidence"]["carry"]
                whole += f["background_changes"]
                pairs += 1
    _check(len(signatures) == 42 and len(masks) >= 70 and len(shapes) >= 25, "insufficient structural diversity")
    _check(len(geometries) >= 40 and len(palettes) >= 30 and len(orientations) == 8, "missing variation axes")
    _check(coupled >= 42 and carry >= 42 and whole == pairs, "missing family interactions")
    # Every legal explicit count and representative type/range errors.
    count_calls = 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(901 + nt * 10 + ns, nt, ns)
            _check(len(p["train"]) == nt and len(p["test"]) == ns, "wrong explicit counts")
            count_calls += 1
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # One conditional glyph, with multiple clue selections and all eight orientations.
    rng = np.random.default_rng(833)
    mask = np.array([[0, 1, 0], [1, 1, 1]], dtype=bool)
    geometry, origin, palette = (2, 3, 1, 7, 7), (3, 1), (0, 1, 2, 3, 4, 5)
    seen_fragments = set()
    for i in range(32):
        clues = _choose_fragment(rng, mask, geometry, origin, "normal")
        seen_fragments.add(tuple(clues))
        pair = _assemble(rng, geometry, palette, mask, origin, clues, i % 4, bool(i // 4 % 2))
        _pair_info(pair)
    _check(len(seen_fragments) >= 4, "fixed-glyph fragments do not vary")
    # Actual 30x30 pixel boundary, rectangular tiles and two-cell divider bands.
    mask = np.array([[1, 1], [0, 1]], dtype=bool)
    clues = _choose_fragment(rng, mask, (2, 2, 2, 7, 7), (2, 4), "normal")
    _check(clues is not None, "boundary fixture has no fragment")
    pair = _assemble(rng, (2, 2, 2, 7, 7), palette, mask, (2, 4), clues, 0, False)
    _check(pair["input"].shape == (30, 30), "wrong boundary fixture")
    _pair_info(pair)
    # Equal full-subset matches remain ambiguous even after color constraints.
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool)
    ambiguous = _assemble(rng, (3, 3, 1, 7, 7), palette, mask, (2, 2),
                          [(2, 2, 0), (2, 3, 0)], 0, False)
    try:
        solve(ambiguous["input"])
    except ValueError:
        pass
    else:
        raise AssertionError("a phase-consistent registration tie was accepted")
    p = generate_puzzle(0)
    for damage in ("divider", "background", "tint"):
        pair = {k: v.copy() for k, v in p["test"][0].items()}
        f = _decode(pair["input"])
        g = _forward(pair["output"], f["k"], f["flip"]).copy()
        if damage == "divider":
            g[-1, -1] = f["seed"]
        else:
            ro, co = f["origin"]
            supplied = {(r, c) for r, c, _ in f["clues"]}
            r, c = next(p for p in f["points"] if (ro + p[0], co + p[1]) not in supplied)
            h, w, t, _, _ = f["geometry"]
            local = np.argwhere((f["ordinary"] != f["background"]) & ~f["mask"]) if damage == "background" else np.argwhere(f["mask"])
            yy, xx = map(int, local[0])
            y, x = t + (ro + r) * (h + t) + yy, t + (co + c) * (w + t) + xx
            g[y, x] = f["ordinary"][yy, xx] if damage == "background" else (f["seed"] if g[y, x] == f["accent"] else f["accent"])
        pair["output"] = _backward(g, f["k"], f["flip"])
        try:
            _pair_info(pair)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{damage} corruption accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "blueprints": len(masks), "geometries": len(geometries),
            "palettes": len(palettes), "directions": len(orientations),
            "fragments": len(fragments), "clock_resolved_pairs": coupled,
            "carry_witness_pairs": carry, "whole_tile_witness_pairs": whole,
            "legal_count_calls": count_calls, "conditional_fragments": len(seen_fragments)}
