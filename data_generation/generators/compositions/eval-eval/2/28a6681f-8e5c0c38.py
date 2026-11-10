"""Symmetry locks colored solids; their unmatched material melts and settles.

Only NumPy and the standard library are used. Generation constructs complete
liquid levels; solve reads only its argument and has no generation metadata.
"""
import numpy as np


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or a.dtype.kind not in "iu" or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= a.size // 2:
        raise ValueError("background must be a strict majority")
    return bg


def _split_material(a):
    """Infer independent axes, then classify solid versus molten cells."""
    bg = _background(a)
    rigid = np.full_like(a, bg)
    axes, excess = {}, []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        mask = a == color
        scores = np.zeros(2 * a.shape[1] - 1, dtype=int)
        for row in mask.astype(int):
            scores += np.convolve(row, row)
        winners = np.flatnonzero(scores == scores.max())
        if len(winners) != 1:
            raise ValueError("reflection support has a tied maximum")
        axis = int(winners[0])  # twice the column coordinate
        axes[color] = axis
        for r, c in np.argwhere(mask):
            partner = axis - int(c)
            if 0 <= partner < a.shape[1] and mask[r, partner]:
                rigid[r, c] = color
            else:
                excess.append((int(r), int(c), color))
    return bg, rigid, axes, excess


def _pour(rigid, drops, bg, reverse=False, top_first=False):
    """Local downward-first pouring; alternate orders are validation probes."""
    a = rigid.copy()
    h, w = a.shape
    ordered = sorted(drops, key=lambda p: (p[0] if top_first else -p[0],
                                           -p[1] if reverse else p[1]))
    for r, c, color in ordered:
        while r >= 0 and a[r, c] != bg and rigid[r, c] == bg:
            r -= 1
        if r < 0 or a[r, c] != bg:
            raise ValueError("source displaced beneath a rigid ceiling or beyond canvas")
        for _ in range(h + 1):
            if r + 1 < h and a[r + 1, c] == bg:
                r += 1
                continue
            lo = hi = c
            while lo and a[r, lo - 1] == bg:
                lo -= 1
            while hi + 1 < w and a[r, hi + 1] == bg:
                hi += 1
            exits = [] if r + 1 == h else [j for j in range(lo, hi + 1)
                                          if a[r + 1, j] == bg]
            if not exits:
                a[r, c] = color
                break
            c = min(exits, key=lambda j: (abs(j - c), -j if reverse else j))
            r += 1
        else:
            raise ValueError("pour failed to terminate")
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _array(input_grid)
    bg, rigid, _, drops = _split_material(a)
    return _pour(rigid, drops, bg)


def _parts(mask):
    """Four-connected components as coordinate sets."""
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        part, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    part.add(p)
                    todo.append(p)
        result.append(part)
    return result


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    if cells:
        rows, cols = zip(*cells)
        a[rows, cols] = True
    return a


def _branches(pool):
    rows = np.arange(pool.shape[0])[:, None]
    top = int(np.nonzero(pool)[0].min())
    return max([1] + [len(_parts(pool & (rows >= r)))
                      for r in range(top + 1, pool.shape[0])])


def _basins(solid):
    """Complete, closed levels; external space is never a receiving vessel."""
    h, w = solid.shape
    rows = np.arange(h)[:, None]
    pools = []
    for level in range(2, h - 1):
        for cells in _parts(~solid & (rows >= level)):
            if not 2 <= len(cells) <= 85 or not any(r == level for r, _ in cells):
                continue
            if any(r == h - 1 or c in (0, w - 1) for r, c in cells):
                continue
            pool = _mask(solid.shape, cells)
            if not np.any(pool[level] & ~solid[level - 1]):
                continue
            pools.append(pool)
    return pools


def _catchments(solid, pools):
    """Destination bitsets for every possible dry downward/lateral route.

    A pool absorbs routes. Unsupported cells fall first. Supported cells can
    seek any downward outlet in their free horizontal interval. Zero denotes
    a dry dead end; the extra bit denotes an unassigned destination.
    """
    h, w = solid.shape
    unknown = 1 << len(pools)
    destinations = np.zeros((h, w), dtype=np.int64)
    absorbing = np.zeros((h, w), dtype=np.int64)
    for i, pool in enumerate(pools):
        absorbing[pool] = 1 << i
    for r in range(h - 1, -1, -1):
        intervals = []
        lo = 0
        while lo < w:
            if solid[r, lo]:
                lo += 1
                continue
            hi = lo + 1
            while hi < w and not solid[r, hi]:
                hi += 1
            intervals.append(range(lo, hi))
            lo = hi
        for interval in intervals:
            possible = 0
            for c in interval:
                possible |= int(absorbing[r, c])
                if r + 1 < h and not solid[r + 1, c]:
                    possible |= int(destinations[r + 1, c])
            for c in interval:
                if absorbing[r, c]:
                    destinations[r, c] = absorbing[r, c]
                elif r + 1 < h and not solid[r + 1, c]:
                    destinations[r, c] = destinations[r + 1, c]
                else:
                    destinations[r, c] = possible or unknown
    return destinations


def _geometry(rng, pillars, parity=None, joined=False, simple=False):
    """A mirrored outer vessel and independently mirrored interior supports."""
    h = int(rng.integers(17, 29))
    w = int(rng.integers(19 if pillars == 2 else 13, 31))
    span = int(rng.integers(17 if pillars == 2 else 11, w + 1))
    if parity is not None and (span - 1) % 2 != parity:
        span -= 1
    left = int(rng.integers(0, w - span + 1))
    right = left + span - 1
    floor = h - 1 - int(rng.integers(0, 3))
    rim = int(rng.integers(4, min(9, floor - 5)))
    shell = np.zeros((h, w), dtype=bool)
    # Rectangular, tapered, lobed, stepped and irregular wall profiles all
    # preserve the same axis. Changes are connected with horizontal shoulders.
    style = 0 if simple else int(rng.integers(5))
    maximum_inset = min(3, (span - 9) // 2)
    inset = 0
    for r in range(rim, floor):
        previous = inset
        progress = (r - rim) / max(1, floor - rim - 1)
        if style == 1:
            if rng.random() < .4:
                inset = int(rng.integers(maximum_inset + 1))
        elif style == 2:
            inset = int(round(maximum_inset * progress))
        elif style == 3:
            inset = int(round(maximum_inset * abs(2 * progress - 1)))
        elif style == 4:
            inset = maximum_inset if (r - rim) % 5 < 2 else 0
        shell[r, left + min(previous, inset):left + max(previous, inset) + 1] = True
        shell[r, right - max(previous, inset):right - min(previous, inset) + 1] = True
    shell[floor, left:right + 1] = True
    shoulder = 0 if simple else int(rng.integers(0, 3))
    if shoulder:
        shell[floor - shoulder:floor, left + 1:left + 3] = True
        shell[floor - shoulder:floor, right - 2:right] = True
    if parity == 1 or (not simple and rng.random() < .6):
        # Disconnected but mirrored fragments are rigid too. Some act as dry
        # deflectors above the mouth; an integral-axis singleton locks itself.
        row = int(rng.integers(1, rim - 1))
        distance = int(rng.integers(2, min(5, span // 2)))
        col = (left + right) // 2 - distance
        for offset in range(int(rng.integers(1, 3))):
            shell[row, col + offset] = True
            shell[row, left + right - col - offset] = True
        if parity != 0 and (left + right) % 2 == 0 and rng.random() < .35:
            shell[max(0, row - 1), (left + right) // 2] = True
    if parity == 0:
        # Teach self-support explicitly before tests can show isolated axis
        # cells. No shell or optional fragment occupies the next axis cell.
        shell[0, (left + right) // 2] = True
    cores = [shell]
    axes = [left + right]
    available = list(range(left + 4, right - 3))
    rng.shuffle(available)
    centers = []
    for index in range(pillars):
        width = int(rng.integers(2, 5))
        # The second demonstration's moving pedestal teaches half axes.
        if parity == 1 and index == 0:
            width = int(rng.choice([2, 4]))
        choices = [c for c in available if c + width <= right - 2
                   and all(c + width + 1 < lo or hi + 1 < c for lo, hi in centers)
                   and not (parity == 1 and index == 0
                            and 2 * c + width - 1 == left + right)]
        if not choices:
            return None
        col = choices[0]
        centers.append((col, col + width - 1))
        height = int(rng.integers(1, 3) if joined else rng.integers(4, min(9, floor - rim)))
        core = np.zeros((h, w), dtype=bool)
        for r in range(floor - height, floor):
            trim = 0 if simple or width <= 2 else int(rng.integers(0, 2))
            # Full base guarantees contact and a real separating ridge.
            if r == floor - 1 or r == floor - height:
                trim = 0
            core[r, col + trim:col + width - trim] = True
        if np.any(core & np.logical_or.reduce(cores)):
            return None
        if not simple and rng.random() < .3:
            # A detached symmetric cap shares this color's axis and must stay.
            cap_row = floor - height - 2
            if cap_row >= rim and not np.any(shell[cap_row, col:col + width]):
                core[cap_row, col:col + width] = True
        cores.append(core)
        axes.append(2 * col + width - 1)
    return cores, axes


def _choose_pools(rng, solid, mode):
    candidates = _basins(solid)
    rng.shuffle(candidates)
    if mode == "joined":
        candidates = [p for p in candidates if _branches(p) >= 2]
        if not candidates:
            return None
        candidates.sort(key=lambda p: int(p.sum()))
        index = int(rng.integers(min(3, len(candidates))))
        return [candidates[index]]
    # Separate full levels below a ridge, rather than two arbitrary crop slots.
    candidates = [p for p in candidates if _branches(p) == 1 and int(p.sum()) <= 28]
    candidates.sort(key=lambda p: int(p.sum()))
    if not candidates:
        return None
    if mode == "single":
        index = int(rng.integers(min(6, len(candidates))))
        return [candidates[index]]
    firsts = candidates[:12]
    rng.shuffle(firsts)
    for p in firsts:
        for q in candidates:
            if not np.any(p & q):
                # Four-neighbor separation is needed to exclude mixing.
                near = p.copy()
                near[1:] |= p[:-1]
                near[:-1] |= p[1:]
                near[:, 1:] |= p[:, :-1]
                near[:, :-1] |= p[:, 1:]
                if not np.any(near & q):
                    return [p, q]
    return None


def _source_cells(rng, eligible, pool, axis, used, style, volume):
    points = [tuple(p) for p in np.argwhere(eligible).tolist()
              if 2 * p[1] != axis and tuple(p) not in used]
    if len(points) < volume:
        return None
    anchor = points[int(rng.integers(len(points)))]
    scores = {}
    for r, c in points:
        if style == 0:  # clustered excess, including short attached branches
            value = abs(r - anchor[0]) + abs(c - anchor[1])
        elif style == 1:  # dispersed droplets
            value = 0
        elif style == 2:  # low sources and some already occupied final cells
            value = 0 if pool[r, c] else 2
        else:  # horizontal/vertical streaks
            value = abs(r - anchor[0]) * 3 + abs(c - anchor[1]) * .2
        scores[r, c] = value + float(rng.random())
    selected = set()
    for p in sorted(points, key=lambda p: scores[p]):
        mirror = (p[0], axis - p[1])
        if mirror in selected or mirror in used:
            continue
        selected.add(p)
        if len(selected) == volume:
            break
    if len(selected) != volume or all(pool[p] for p in selected):
        return None
    return selected


def _make_pair(rng, pillars, mode, anchor):
    for attempt in range(240):
        bg, *colors = map(int, rng.choice(10, pillars + 2, replace=False))
        parity = 0 if anchor == 0 else 1 if anchor == 1 else None
        geometry = _geometry(rng, pillars, parity, mode == "joined", attempt >= 160)
        if geometry is None:
            continue
        cores, axes = geometry
        solid = np.logical_or.reduce(cores)
        pools = _choose_pools(rng, solid, mode)
        if pools is None:
            continue
        if anchor == 1 and len({int(np.nonzero(p)[0].min()) for p in pools}) != 2:
            continue
        # First demonstration melts outer-wall material. The next melts two
        # colors, proving that geometry, not a water-color convention, decides.
        if anchor == 0:
            material_indices = [0]
        elif anchor == 1:
            material_indices = [1, 0]
        else:
            material_indices = list(map(int, rng.choice(len(cores), len(pools),
                                                       replace=False)))
        if any(i and int(p.sum()) >= int(cores[i].sum()) - 2
               for i, p in zip(material_indices, pools)):
            continue
        terrain = np.full(solid.shape, bg, dtype=np.int8)
        for color, core in zip(colors, cores):
            terrain[core] = color
        destinations = _catchments(solid, pools)
        inp, expected = terrain.copy(), terrain.copy()
        used_by_color = {color: set() for color in colors}
        occupied = set()
        ok = True
        for j, (pool, index) in enumerate(zip(pools, material_indices)):
            color, axis = colors[index], axes[index]
            eligible = destinations == (1 << j)
            for r, c in occupied:
                eligible[r, c] = False
            source = _source_cells(rng, eligible, pool, axis, used_by_color[color],
                                   int(rng.integers(4)), int(pool.sum()))
            if source is None:
                ok = False
                break
            used_by_color[color] |= source
            occupied |= source
            for p in source:
                inp[p] = color
            expected[pool] = color
        if not ok:
            continue
        try:
            found_bg, found_core, found_axes, drops = _split_material(inp)
            if (found_bg != bg or found_axes != dict(zip(colors, axes))
                    or not np.array_equal(found_core, terrain)):
                continue
            if anchor == 1:
                # Informative demonstrations are construction requirements,
                # not fortunate observations about a few sampled seeds.
                bbox_shortcut_fails = False
                for color, axis in found_axes.items():
                    cols = np.nonzero(inp == color)[1]
                    bbox_shortcut_fails |= int(cols.min() + cols.max()) != axis
                if not bbox_shortcut_fails:
                    continue
                final_asymmetry = False
                for color, axis in found_axes.items():
                    for r, c in np.argwhere(expected == color):
                        reflected = axis - int(c)
                        if not 0 <= reflected < inp.shape[1] or expected[r, reflected] != color:
                            final_asymmetry = True
                            break
                if not final_asymmetry:
                    continue
            if all(np.array_equal(_pour(terrain, drops, bg, rev, order), expected)
                   for rev in (False, True) for order in (False, True)):
                if np.array_equal(inp, expected):
                    continue
                # Joining must demonstrate lateral redistribution, not merely
                # vertical compression of independent source columns.
                if anchor == 0:
                    color = colors[0]
                    src = (inp == color) & ~solid
                    dst = (expected == color) & ~solid
                    if np.array_equal(src.sum(axis=0), dst.sum(axis=0)):
                        continue
                return {"input": inp, "output": expected}
        except ValueError:
            continue
    raise RuntimeError("bounded vessel/source construction exhausted")


def _count(value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError("illegal example count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10)
    num_test = _count(num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            anchor = index if split == "train" and index < 2 else None
            mode = ("joined" if anchor == 0 else "separate" if anchor == 1
                    else str(rng.choice(["joined", "separate", "single"])))
            for _ in range(16):
                pair = _make_pair(rng, 1 if split == "train" else 2, mode, anchor)
                key = _key(pair["input"])
                if anchor == 1:
                    first_volume = len(_split_material(puzzle["train"][0]["input"])[3])
                    if len(_split_material(pair["input"])[3]) == first_volume:
                        continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _inspect(inp, out):
    bg, rigid, axes, drops = _split_material(inp)
    solid = rigid != bg
    assert inp.shape == out.shape and drops and np.any(inp != out)
    assert np.array_equal(out[solid], rigid[solid]), "locked material moved"
    assert np.array_equal(np.bincount(inp.ravel(), minlength=10),
                          np.bincount(out.ravel(), minlength=10)), "material lost"
    liquid = (out != bg) & ~solid
    pools = [_mask(inp.shape, cells) for cells in _parts(liquid)]
    assert pools
    routes = _catchments(solid, pools)
    assigned = set()
    levels = []
    branches = []
    foreign_support = False
    for j, pool in enumerate(pools):
        colors = np.unique(out[pool])
        assert len(colors) == 1, "different liquids mix"
        color = int(colors[0])
        top = int(np.nonzero(pool)[0].min())
        free_below = ~solid & (np.arange(inp.shape[0])[:, None] >= top)
        closure = next(p for p in _parts(free_below) if tuple(np.argwhere(pool)[0]) in p)
        assert np.array_equal(_mask(inp.shape, closure), pool), "incomplete or leaking level"
        supply = {(r, c) for r, c, k in drops if routes[r, c] == 1 << j and k == color}
        assert len(supply) == int(pool.sum()), "wrong or ambiguous catchment volume"
        assigned |= supply
        levels.append(top)
        branches.append(_branches(pool))
        for r, c in np.argwhere(pool):
            for rr, cc in ((r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]:
                    if solid[rr, cc] and rigid[rr, cc] != color:
                        foreign_support = True
    assert assigned == {(r, c) for r, c, _ in drops}, "unassigned molten material"
    for reverse in (False, True):
        for top_first in (False, True):
            assert np.array_equal(_pour(rigid, drops, bg, reverse, top_first), out)
    assert not np.array_equal(rigid, out), "parent A has no effect"
    # Ablate B by making every foreground cell liquid and leaving no terrain.
    all_drops = [(int(r), int(c), int(inp[r, c])) for r, c in np.argwhere(inp != bg)]
    try:
        no_b = _pour(np.full_like(inp, bg), all_drops, bg)
        assert not np.array_equal(no_b, out), "parent B has no effect"
    except ValueError:
        pass  # No-terrain pouring can overflow; this also differs from output.
    src_cols = np.zeros(inp.shape[1], dtype=int)
    for _, c, _ in drops:
        src_cols[c] += 1
    bbox_wrong = False
    components = []
    axis_singletons = 0
    for color, axis in axes.items():
        columns = np.nonzero(inp == color)[1]
        bbox_wrong |= int(columns.min() + columns.max()) != axis
        pieces = _parts(rigid == color)
        components.append(len(pieces))
        axis_singletons += sum(len(piece) == 1 and 2 * next(iter(piece))[1] == axis
                               for piece in pieces)
    asymmetric_result = any(any(out[r, axis - c] != color
                               if 0 <= axis - c < inp.shape[1] else True
                               for r, c in np.argwhere(out == color))
                            for color, axis in axes.items())
    return {"shape": tuple(inp.shape), "background": bg, "colors": len(axes),
            "axes": tuple(axes.values()), "parities": tuple(a % 2 for a in axes.values()),
            "volume": len(drops), "pools": len(pools), "levels": len(set(levels)),
            "branches": max(branches), "mobile_colors": len({k for _, _, k in drops}),
            "foreign_support": foreign_support,
            "lateral": not np.array_equal(src_cols, liquid.sum(axis=0)),
            "retained_liquid": sum(bool(liquid[r, c] and out[r, c] == k) for r, c, k in drops),
            "source_low": min(r for r, _, _ in drops) >= inp.shape[0] // 2,
            "bbox_axis_wrong": bool(bbox_wrong),
            "rigid_components": max(components),
            "axis_singletons": axis_singletons,
            "asymmetric_result": asymmetric_result,
            "normalized": _key(np.where(inp == bg, 0, np.where(solid, 1, 2)))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _count(len(puzzle["train"]), 2, 10)
    _count(len(puzzle["test"]), 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            stats = _inspect(inp, out)
            assert np.array_equal(solve(inp.copy()), out)
            assert stats["colors"] == (2 if split == "train" else 3)
            records[split].append(stats)
    assert records["train"][0]["branches"] >= 2
    assert records["train"][0]["lateral"]
    assert records["train"][0]["axis_singletons"] >= 1
    assert records["train"][1]["pools"] == 2
    assert records["train"][1]["mobile_colors"] == 2
    assert records["train"][1]["foreign_support"]
    assert records["train"][1]["bbox_axis_wrong"]
    assert records["train"][1]["asymmetric_result"]
    assert records["train"][1]["rigid_components"] >= 2
    assert records["train"][1]["levels"] == 2
    assert len(set(records["train"][1]["axes"])) == 2
    assert records["train"][0]["volume"] != records["train"][1]["volume"]
    # Both axis kinds are demonstrated by material that actually moves.
    demonstrated = set()
    for pair in puzzle["train"]:
        _, _, axes, drops = _split_material(np.asarray(pair["input"]))
        demonstrated |= {axes[k] % 2 for _, _, k in drops}
    assert demonstrated == {0, 1}


def self_test():
    records, structures, signatures = [], set(), set()
    for seed in range(42):
        # Vary legal counts while keeping this audit bounded.
        a, b = 2 + seed % 9, 1 + seed % min(4, 1 + seed % 9)
        puzzle = generate_puzzle(seed, a, b)
        signature = tuple(_key(p["input"]) for s in puzzle for p in puzzle[s])
        signatures.add(signature)
        for split in puzzle:
            for p in puzzle[split]:
                item = _inspect(p["input"], p["output"])
                structures.add(item.pop("normalized"))
                records.append(item)
        if seed < 3:
            again = generate_puzzle(seed, a, b)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in p)
    assert len(signatures) == 42 and len(structures) >= 200
    assert len({r["shape"] for r in records}) >= 45
    assert {r["background"] for r in records} == set(range(10))
    assert len({r["volume"] for r in records}) >= 25
    assert {r["colors"] for r in records} == {2, 3}
    assert {r["mobile_colors"] for r in records} == {1, 2}
    assert sum(r["branches"] > 1 for r in records) >= 42
    assert sum(r["foreign_support"] for r in records) >= 80
    assert sum(r["levels"] > 1 for r in records) >= 5
    assert sum(r["retained_liquid"] > 0 for r in records) >= 20
    assert any(r["source_low"] for r in records)
    assert sum(len(set(r["axes"])) > 1 for r in records) >= 80
    assert sum(r["rigid_components"] > 1 for r in records) >= 50
    assert sum(r["bbox_axis_wrong"] for r in records) >= 42
    assert sum(r["asymmetric_result"] for r in records) >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(314, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # A validator must reject plausible but semantically wrong outputs.
    for defect in ("delete", "move_rigid", "extra"):
        bad = generate_puzzle(717, 2, 1)
        p = bad["test"][0]
        bg, rigid, _, _ = _split_material(p["input"])
        solid = rigid != bg
        liquid = (p["output"] != bg) & ~solid
        if defect == "delete":
            r, c = np.argwhere(liquid)[0]
            p["output"][r, c] = bg
        elif defect == "move_rigid":
            r, c = np.argwhere(solid)[0]
            p["output"][r, c] = bg
        else:
            r, c = np.argwhere(p["output"] == bg)[0]
            p["output"][r, c] = int(p["output"][liquid][0])
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("semantic mutation was accepted")
