"""Welded stamp constellations: registered, overlapping multicolor substitution.

Only standard Python and NumPy are used. All solving information is in pixels.
See task_spec.md for the semantic rule and the stricter sampling safeguards.
"""

from itertools import product
import numpy as np


def _choose(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return array.astype(np.int8, copy=False)


def _key(array):
    return array.shape, array.astype(np.int8).tobytes()


def _components(binary):
    remaining = set(map(tuple, np.argwhere(binary)))
    result = []
    while remaining:
        stack = [remaining.pop()]
        component = []
        while stack:
            r, c = stack.pop()
            component.append((int(r), int(c)))
            for dr, dc in product((-1, 0, 1), repeat=2):
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(component)
    return result


def _asymmetric(array):
    variants = [f(np.rot90(array, k)) for k in range(4)
                for f in (lambda a: a, np.fliplr)]
    return all(not np.array_equal(array, v) for v in variants[1:])


def _decode(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(counts.max()) <= grid.size // 2:
        raise ValueError("background must be a strict majority")
    components = _components(grid != bg)
    large = [component for component in components if len(component) > 1]
    if len(large) != 1:
        raise ValueError("expected one compact stamp and isolated map points")
    cells = np.array(large[0], dtype=int)
    tr, tc = map(int, cells.min(axis=0))
    br, bc = map(int, cells.max(axis=0))
    stamp = grid[tr:br + 1, tc:bc + 1].copy()
    h, w = stamp.shape
    payload = set(map(int, np.unique(stamp))) - {bg}
    if min(h, w) < 2 or len(payload) < 2:
        raise ValueError("stamp must span both axes and be multicolor")
    dots = [component[0] for component in components if len(component) == 1]
    shared = [p for p in dots if int(grid[p]) in payload]
    ordinary = [p for p in dots if int(grid[p]) not in payload]
    if len(shared) != 1 or not ordinary:
        raise ValueError("map must have exactly one stamp-colored anchor")
    marker_colors = {int(grid[p]) for p in ordinary}
    if len(marker_colors) != 1:
        raise ValueError("ordinary map points must share a color absent from stamp")
    marker = marker_colors.pop()
    register = int(grid[shared[0]])
    pr, pc = map(int, np.min(dots, axis=0))
    er, ec = map(int, np.max(dots, axis=0))
    if not (er + 1 < tr or br + 1 < pr or ec + 1 < tc or bc + 1 < pc):
        raise ValueError("stamp and whole map need a blank separating line")
    if any((r - pr) % 2 or (c - pc) % 2 for r, c in dots):
        raise ValueError("map points must lie on the fixed pitch-two lattice")
    sites = frozenset(((r - pr) // 2, (c - pc) // 2) for r, c in dots)
    mh, mw = (er - pr) // 2 + 1, (ec - pc) // 2 + 1
    anchor = ((shared[0][0] - pr) // 2, (shared[0][1] - pc) // 2)
    top, left = tr - anchor[0] * (h - 1), tc - anchor[1] * (w - 1)
    height, width = mh * (h - 1) + 1, mw * (w - 1) + 1
    if not (0 <= top and top + height <= grid.shape[0]
            and 0 <= left and left + width <= grid.shape[1]):
        raise ValueError("registered assembly must fit without clipping")
    return dict(grid=grid, bg=bg, marker=marker, register=register, stamp=stamp,
                sites=sites, anchor=anchor, map_shape=(mh, mw),
                stamp_pos=(tr, tc), top=top, left=left, extent=(height, width),
                map_pos=(pr, pc), dots=tuple(dots))


def _pixel_contributions(info, r, c, recolor=True):
    """Inverse registration: at most four stamps can cover one target cell."""
    stamp = info["stamp"]
    h, w = stamp.shape
    sr, sc = h - 1, w - 1
    rr, cc = r - info["top"], c - info["left"]
    incoming = []
    # floor division plus the previous site covers all possible one-cell seams.
    for mr in (rr // sr - 1, rr // sr):
        for mc in (cc // sc - 1, cc // sc):
            site = mr, mc
            a, b = rr - mr * sr, cc - mc * sc
            if site in info["sites"] and 0 <= a < h and 0 <= b < w:
                raw = int(stamp[a, b])
                painted = (info["marker"] if recolor and site == info["anchor"]
                           and raw == info["register"] else raw)
                incoming.append((site, raw, painted))
    return incoming


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, registration, and every weld from the input alone."""
    info = _decode(input_grid)
    output = np.full_like(info["grid"], info["bg"])
    height, width = info["extent"]
    for r in range(info["top"], info["top"] + height):
        for c in range(info["left"], info["left"] + width):
            ink = {painted for _, _, painted in _pixel_contributions(info, r, c)
                   if painted != info["bg"]}
            if ink:
                output[r, c] = next(iter(ink)) if len(ink) == 1 else info["marker"]
    return output


def _render(shape, bg, marker, register, stamp, sites, anchor, top, left,
            recolor=True, step=None, monochrome=False, opaque=False):
    """Forward bit-set accumulation, separate from solve's inverse indexing.

    Optional switches implement documented alternative rules for checks only.
    """
    h, w = stamp.shape
    sh, sw = (h - 1, w - 1) if step is None else step
    bits = np.zeros(shape, dtype=np.uint16)
    for mr, mc in sorted(sites):
        r0, c0 = top + mr * sh, left + mc * sw
        for a, b in np.ndindex(stamp.shape):
            color = int(stamp[a, b])
            if monochrome and color != bg:
                color = register
            if recolor and (mr, mc) == anchor and color == register:
                color = marker
            if color == bg and not opaque:
                continue
            r, c = r0 + a, c0 + b
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                bits[r, c] |= np.uint16(1 << color)
    output = np.full(shape, bg, dtype=np.int8)
    conflict = (bits != 0) & ((bits & (bits - np.uint16(1))) != 0)
    for color in range(10):
        output[bits == (1 << color)] = color
    output[conflict] = marker
    return output


def _from_info(info, **options):
    return _render(info["grid"].shape, info["bg"], info["marker"], info["register"],
                   info["stamp"], info["sites"], info["anchor"], info["top"],
                   info["left"], **options)


def _observations(info):
    counts = dict(conflict=0, agreement=0, transparent=0, four_way=0)
    h, w = info["extent"]
    bg, anchor = info["bg"], info["anchor"]
    for r in range(info["top"], info["top"] + h):
        for c in range(info["left"], info["left"] + w):
            incoming = _pixel_contributions(info, r, c)
            ink = [painted for _, _, painted in incoming if painted != bg]
            ordinary_only = all(site != anchor for site, _, _ in incoming)
            if ordinary_only and len(ink) > 1:
                counts["conflict" if len(set(ink)) > 1 else "agreement"] += 1
            if ink and len(set(ink)) == 1 and any(raw == bg for _, raw, _ in incoming):
                counts["transparent"] += 1
            if len(incoming) == 4:
                counts["four_way"] += 1
    return counts


def _stamp(rng, shape, bg, payload, witness):
    h, w = shape
    r, c = np.indices(shape)
    n = len(payload)
    kind = str(_choose(rng, ("random", "diagonal", "bands", "rings", "tile")))
    if witness:
        kind = "random"
    if kind == "diagonal":
        labels = (r + c * int(rng.integers(1, n + 1))) % n
    elif kind == "bands":
        labels = (r if rng.integers(2) else c) % n
    elif kind == "rings":
        labels = np.minimum.reduce((r, c, h - 1 - r, w - 1 - c)) % n
    elif kind == "tile":
        tile = rng.integers(n, size=(min(2, h), min(3, w)))
        labels = tile[r % tile.shape[0], c % tile.shape[1]]
    else:
        labels = rng.integers(n, size=shape)
    stamp = np.asarray(payload, dtype=np.int8)[labels].copy()
    # Payload evidence is distributed anywhere, never a compulsory palette row.
    positions = rng.choice(h * w, size=n, replace=False)
    stamp.flat[positions] = payload
    holes = int(rng.integers(0, max(2, h * w // 4 + 1)))
    if witness:
        holes = max(2, holes)
    for index in rng.permutation(h * w)[:holes]:
        old = int(stamp.flat[index])
        stamp.flat[index] = bg
        support = stamp != bg
        if (not all(np.any(side) for side in (support[0], support[-1], support[:, 0], support[:, -1]))
                or len(_components(support)) != 1
                or len(set(map(int, np.unique(stamp))) - {bg}) < 2):
            stamp.flat[index] = old
    return stamp


def _layout(rng, shape, gap=False, witness=False):
    mh, mw = shape
    if witness:
        mask = np.ones(shape, dtype=bool)
        # Keep a nonanchor 2x2 block available and expose an inactive site.
        missing = _choose(rng, ((0, 0), (0, mw - 1), (mh - 1, 0), (mh - 1, mw - 1)))
        mask[missing] = False
    else:
        mode = str(_choose(rng, ("scatter", "path", "ring", "dense", "full")))
        if mode == "path":
            mask = np.zeros(shape, dtype=bool)
            r = c = 0
            mask[r, c] = True
            while r < mh - 1 or c < mw - 1:
                choices = ([(1, 0)] if r < mh - 1 else []) + ([(0, 1)] if c < mw - 1 else [])
                dr, dc = _choose(rng, choices)
                r, c = r + dr, c + dc
                mask[r, c] = True
        elif mode == "ring":
            r, c = np.indices(shape)
            mask = (r == 0) | (r == mh - 1) | (c == 0) | (c == mw - 1)
        elif mode == "full":
            mask = np.ones(shape, dtype=bool)
        else:
            mask = rng.random(shape) < (.8 if mode == "dense" else float(rng.uniform(.15, .6)))
        # Tight bounds do not rely on connectivity or density.
        mask[0, 0] = mask[-1, -1] = True
        # Unannounced blank bands would undermine the train/test safeguard.
        # Scattered islands remain possible without dropping an entire axis band.
        for r in range(mh):
            if not np.any(mask[r]):
                mask[r, int(rng.integers(mw))] = True
        for c in range(mw):
            if not np.any(mask[:, c]):
                mask[int(rng.integers(mh)), c] = True
    if gap:
        mask[int(rng.integers(2, mh - 1)), :] = False
        mask[:, int(rng.integers(2, mw - 1))] = False
        mask[:2, :2] = True  # both unit lattice pitches remain visible
        mask[0, -1] = mask[-1, 0] = mask[-1, -1] = True
    return frozenset(map(tuple, np.argwhere(mask)))


def _place(rng, stamp, sites, anchor, map_shape, bg, marker, register):
    h, w = stamp.shape
    mh, mw = map_shape
    ah, aw = mh * (h - 1) + 1, mw * (w - 1) + 1
    ph, pw = 2 * mh - 1, 2 * mw - 1
    tr, tc = anchor[0] * (h - 1), anchor[1] * (w - 1)
    rr, cc = np.meshgrid(np.arange(-ph - 1, ah + 2), np.arange(-pw - 1, aw + 2), indexing="ij")
    minr, minc = np.minimum(rr, 0), np.minimum(cc, 0)
    maxr, maxc = np.maximum(rr + ph, ah), np.maximum(cc + pw, aw)
    valid = ((rr + ph < tr) | (tr + h < rr) | (cc + pw < tc) | (tc + w < cc))
    valid &= (maxr - minr <= 30) & (maxc - minc <= 30)
    choices = np.argwhere(valid)
    if not len(choices):
        raise ValueError("no room for this separated encoding")
    if rng.random() < .4:
        areas = (maxr - minr)[valid] * (maxc - minc)[valid]
        choices = choices[areas <= np.quantile(areas, .2)]
    i, j = _choose(rng, choices)
    pr, pc = int(rr[i, j]), int(cc[i, j])
    base_h, base_w = int(maxr[i, j] - minr[i, j]), int(maxc[i, j] - minc[i, j])
    padding = int(rng.integers(3))
    gh = base_h if padding == 0 else 30 if padding == 1 else int(rng.integers(base_h, 31))
    gw = base_w if padding == 0 else 30 if padding == 1 else int(rng.integers(base_w, 31))
    foreground = int(np.count_nonzero(stamp != bg)) + len(sites)
    while gh * gw <= 2 * foreground:
        if gh < 30 and (gh <= gw or gw == 30):
            gh += 1
        else:
            gw += 1
    top = int(rng.integers(gh - base_h + 1)) - int(minr[i, j])
    left = int(rng.integers(gw - base_w + 1)) - int(minc[i, j])
    grid = np.full((gh, gw), bg, dtype=np.int8)
    grid[top + tr:top + tr + h, left + tc:left + tc + w] = stamp
    for mr, mc in sites:
        grid[top + pr + 2 * mr, left + pc + 2 * mc] = register if (mr, mc) == anchor else marker
    output = _render(grid.shape, bg, marker, register, stamp, sites, anchor, top, left)
    return {"input": grid, "output": output}


def _shapes(rng, shift, train, index, extensions, previous):
    gap = extensions and (not train or index == 1 or (index > 1 and rng.random() < .4))
    if shift == 0:
        maps = [(a, b) for a in range(1, 5 if train else 8) for b in range(1, 5 if train else 8)
                if train or max(a, b) >= 5]
        stamps = list(product(range(2, 6), repeat=2))
    elif shift == 1:
        maps = list(product(range(1, 5), repeat=2))
        stamps = [(a, b) for a in range(2, 5 if train else 7) for b in range(2, 5 if train else 7)
                  if train or max(a, b) >= 5]
    elif shift == 2:
        maps = [(a, b) for a in range(1, 7) for b in range(1, 7)
                if (a < b if train else a > b)]
        stamps = list(product(range(2, 6), repeat=2))
    else:
        maps = list(product(range(2, 5 if train else 7), repeat=2))
        stamps = list(product(range(2 if train else 4, 5 if train else 7), repeat=2))
        if not train:
            maps = [m for m in maps if max(m) >= 5]
            stamps = [s for s in stamps if max(s) >= 5]
    if train and index == 0:
        maps = [m for m in maps if min(m) >= 2 and max(m) <= 4
                and m[0] * m[1] >= 6 and m[0] != m[1]]
        stamps = [s for s in stamps if min(s) >= 3 and s[0] != s[1]]
        gap = False
    if train and index == 1:
        maps = [m for m in maps if min(m) >= 2]
        stamps = [s for s in stamps if s[0] != previous["stamp"].shape[0]
                  and s[1] != previous["stamp"].shape[1]]
    if gap:
        maps = [m for m in maps if min(m) >= 4]
    maps = [m for m in maps if m[0] * m[1] >= 2]
    options = [(m, [s for s in stamps if m[0] * (s[0] - 1) + 1 <= 27
                   and m[1] * (s[1] - 1) + 1 <= 27]) for m in maps]
    options = [(m, ss) for m, ss in options if ss]
    m, ss = _choose(rng, options)
    return m, _choose(rng, ss), gap


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(4))
    extensions = shift == 3 or rng.random() < .55
    stable = rng.random() < .25
    palette = rng.permutation(10)
    puzzle = {"train": [], "test": []}
    seen = set()
    first = None
    for split, count in (("train", num_train), ("test", num_test)):
        train = split == "train"
        for index in range(count):
            witness = train and index == 0
            for attempt in range(180):
                mshape, sshape, gap = _shapes(rng, shift, train, index, extensions, first)
                colors = palette if stable else rng.permutation(10)
                bg, marker = map(int, colors[:2])
                payload = list(map(int, colors[2:2 + int(rng.integers(2, 5))]))
                stamp = _stamp(rng, sshape, bg, payload, witness)
                visible = sorted(set(map(int, np.unique(stamp))) - {bg})
                register = int(_choose(rng, visible))
                sites = _layout(rng, mshape, gap, witness)
                anchors = sorted(sites)
                if witness:
                    anchors = [a for a in anchors if a != ((mshape[0] - 1) // 2, (mshape[1] - 1) // 2)]
                if train and index == 1:
                    anchors = [a for a in anchors if a[0] != first["anchor"][0] and a[1] != first["anchor"][1]]
                if not anchors:
                    continue
                anchor = _choose(rng, anchors)
                try:
                    pair = _place(rng, stamp, sites, anchor, mshape, bg, marker, register)
                    info = _decode(pair["input"])
                except ValueError:
                    continue
                if _key(pair["input"]) in seen:
                    continue
                if witness:
                    obs = _observations(info)
                    if (not _asymmetric(stamp) or not all(obs[k] > 0 for k in ("conflict", "agreement", "transparent"))
                            or np.array_equal(pair["output"], _from_info(info, recolor=False))
                            or np.array_equal(pair["output"], _from_info(info, opaque=True))):
                        continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if witness:
                    first = info
                break
            else:
                raise RuntimeError("bounded construction could not find an informative distinct example")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            assert inp.shape == out.shape
            assert np.array_equal(solve(inp), out), "incorrect registered welds"
            assert np.array_equal(_from_info(info), out), "forward/inverse disagreement"
            assert not np.array_equal(inp, out)
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            infos[split].append(info)
    train, test = infos["train"], infos["test"]
    assert len({i["stamp"].shape[0] for i in train}) >= 2
    assert len({i["stamp"].shape[1] for i in train}) >= 2
    assert len({i["anchor"][0] for i in train}) >= 2
    assert len({i["anchor"][1] for i in train}) >= 2
    assert any(_asymmetric(i["stamp"]) for i in train)
    assert any(i["map_shape"][0] != i["map_shape"][1] for i in train)
    assert any(len(i["sites"]) < np.prod(i["map_shape"]) for i in train)
    assert any(i["anchor"] != ((i["map_shape"][0] - 1) // 2, (i["map_shape"][1] - 1) // 2) for i in train)
    observations = [_observations(i) for i in train]
    assert all(any(o[k] for o in observations) for k in ("conflict", "agreement", "transparent"))
    for axis in (0, 1):
        assert any(any((r + (axis == 0), c + (axis == 1)) in i["sites"] for r, c in i["sites"]) for i in train)
        has_gap = lambda i: len({s[axis] for s in i["sites"]}) < i["map_shape"][axis]
        if any(has_gap(i) for i in test):
            assert any(has_gap(i) for i in train), "test gap needs a demonstrated coordinate convention"
    # Explicit concept ablations. Both must change demonstrated answers.
    assert any(not np.array_equal(_from_info(i, recolor=False), solve(i["grid"])) for i in train)
    assert any(not np.array_equal(_from_info(i, monochrome=True), solve(i["grid"])) for i in train)
    assert any(not np.array_equal(_from_info(i, opaque=True), solve(i["grid"])) for i in train)
    return None


def self_test():
    """Bounded 42-seed semantic/diversity audit, plus API and mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprints = set()
    records = []
    for puzzle in puzzles:
        fingerprint = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                fingerprint.append((_key(pair["input"]), _key(pair["output"])))
                records.append(info)
        fingerprints.add(tuple(fingerprint))
    assert len(fingerprints) == 42
    repeat = generate_puzzle(17)
    assert all(np.array_equal(a[k], b[k]) for s in repeat
               for a, b in zip(repeat[s], puzzles[17][s]) for k in ("input", "output"))
    assert len({i["grid"].shape for i in records}) >= 45
    assert len({i["stamp"].shape for i in records}) >= 18
    assert len({i["map_shape"] for i in records}) >= 20
    assert len({i["anchor"] for i in records}) >= 18
    assert len({(i["map_shape"], tuple(sorted(i["sites"]))) for i in records}) >= 100
    assert {i["bg"] for i in records} == set(range(10))
    assert {i["marker"] for i in records} == set(range(10))
    assert any(i["map_shape"][0] == 1 for i in records)
    assert any(i["map_shape"][1] == 1 for i in records)
    assert any(min(i["stamp"].shape) == 2 for i in records)
    assert any(max(i["stamp"].shape) == 6 for i in records)
    assert max(len(i["sites"]) for i in records) >= 20
    assert any(len({s[0] for s in i["sites"]}) < i["map_shape"][0]
               and len({s[1] for s in i["sites"]}) < i["map_shape"][1] for i in records)
    assert sum(_observations(i)["four_way"] > 0 for i in records) >= 20
    assert any(any(pair["output"][p] != _decode(pair["input"])["bg"]
                   for p in _decode(pair["input"])["dots"])
               for puzzle in puzzles for split in puzzle for pair in puzzle[split])
    for counts in ((2, 1), (10, 4), (5, 3)):
        validate_generated_puzzle(generate_puzzle(1234, *counts))
    # Semantic sizes beyond the sampler, including gaps in one-dimensional maps.
    # These are authored regression fixtures, never a bank used by generation.
    regression_rng = np.random.default_rng(919)
    for sshape, mshape, sites, anchor in (
            ((8, 3), (1, 3), {(0, 0), (0, 2)}, (0, 2)),
            ((3, 8), (3, 1), {(0, 0), (2, 0)}, (0, 0)),
            ((2, 2), (9, 2), {(0, 0), (0, 1), (8, 0), (8, 1)}, (8, 1))):
        rr, cc = np.indices(sshape)
        stamp = np.where((rr + cc) % 3 == 0, 3, 2).astype(np.int8)
        stamp[1, 1] = 0
        pair = _place(regression_rng, stamp, sites, anchor, mshape, 0, 1, 2)
        assert np.array_equal(solve(pair["input"]), pair["output"])
    # Move only the exceptional dot; keep stamp, occupancy, and canvas fixed.
    counterfactual = np.zeros((20, 20), dtype=np.int8)
    stamp = np.array([[2, 3, 2, 3], [3, 0, 2, 2], [2, 2, 3, 3]], dtype=np.int8)
    counterfactual[8:11, 8:12] = stamp
    for r, c in product(range(2), range(3)):
        counterfactual[1 + 2*r, 1 + 2*c] = 1
    counterfactual[1, 1] = 2
    moved = counterfactual.copy()
    moved[1, 1], moved[3, 5] = 1, 2
    assert np.array_equal(solve(counterfactual), _from_info(_decode(counterfactual)))
    assert np.array_equal(solve(moved), _from_info(_decode(moved)))
    assert not np.array_equal(solve(counterfactual), solve(moved))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("42-seed self-test passed")
