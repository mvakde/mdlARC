"""Phase-locked self-stencil: an origin-dependent coarse map over a phase lattice.

Only NumPy and the standard library are used. solve() reads one input; it has no
seed, training examples, file access, or generation metadata. See task_spec.md.
"""
from functools import lru_cache

import numpy as np


_SHAPES = tuple((h, w) for h in range(2, 5) for w in range(2, 5))


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or
                not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and i == result[-1][1] + 1:
            result[-1] = (result[-1][0], i)
        else:
            result.append((i, i))
    return result


def _full_box(mask):
    return bool(mask[0].any() and mask[-1].any() and
                mask[:, 0].any() and mask[:, -1].any())


def _asymmetric(mask):
    transformed = [mask[::-1], mask[:, ::-1], mask[::-1, ::-1]]
    if mask.shape[0] == mask.shape[1]:
        transformed += [np.rot90(mask), np.rot90(mask, 3), mask.T,
                        mask.T[::-1, ::-1]]
    return all(not np.array_equal(mask, m) for m in transformed)


def _phase_tiles(base, lattice, steps):
    nr, nc = lattice
    h, w = base.shape
    a, b = steps
    ys = (np.arange(h)[None, None, :, None] +
          a * np.arange(nr)[:, None, None, None]) % h
    xs = (np.arange(w)[None, None, None, :] +
          b * np.arange(nc)[None, :, None, None]) % w
    return base[ys, xs]


def _footprints(tiles):
    """Each possible origin uses THAT tile's mask, never a global fixed mask."""
    nr, nc, h, w = tiles.shape
    result = []
    for r in range(nr - h + 1):
        for c in range(nc - w + 1):
            local = tiles[r, c]
            if _full_box(local):
                active = frozenset((r + int(y), c + int(x))
                                   for y, x in np.argwhere(local))
                result.append(((r, c), active))
    return result


def _paint(tiles, thickness, roles, highlighted):
    divider, carrier, ground, highlight = roles
    nr, nc, h, w = tiles.shape
    out = np.full((nr * (h + thickness) + thickness,
                   nc * (w + thickness) + thickness), divider, dtype=np.int8)
    for r in range(nr):
        for c in range(nc):
            y, x = thickness + r * (h + thickness), thickness + c * (w + thickness)
            accent = highlight if (r, c) in highlighted else ground
            out[y:y+h, x:x+w] = np.where(tiles[r, c], accent, carrier)
    return out


def _decode(value):
    """Decode geometry, roles and all local masks using only the input."""
    grid = _grid(value)
    divider = int(grid[0, 0])
    rg = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    cg = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    if (len(rg) < 3 or len(cg) < 3 or rg[0][0] != 0 or cg[0][0] != 0 or
            rg[-1][1] != grid.shape[0] - 1 or cg[-1][1] != grid.shape[1] - 1):
        raise ValueError("missing complete framed divider lattice")
    thicknesses = {b-a+1 for a, b in rg + cg}
    heights = {rg[i+1][0] - rg[i][1] - 1 for i in range(len(rg)-1)}
    widths = {cg[i+1][0] - cg[i][1] - 1 for i in range(len(cg)-1)}
    if len(thicknesses) != 1 or len(heights) != 1 or len(widths) != 1:
        raise ValueError("irregular divider spacing")
    t, h, w = thicknesses.pop(), heights.pop(), widths.pop()
    nr, nc = len(rg)-1, len(cg)-1
    if t not in (1, 2) or (h, w) not in _SHAPES or nr < h or nc < w:
        raise ValueError("unsupported lattice geometry")
    tile_values, common = [], None
    for r in range(nr):
        row = []
        for c in range(nc):
            y, x = rg[r][1]+1, cg[c][1]+1
            tile = grid[y:y+h, x:x+w]
            colors = set(map(int, np.unique(tile)))
            if divider in colors or len(colors) != 2:
                raise ValueError("each interior must have a carrier and one accent")
            common = colors if common is None else common & colors
            row.append(tile)
        tile_values.append(row)
    if len(common) != 1:
        raise ValueError("carrier must be the one color shared by every tile")
    carrier = next(iter(common))
    tiles = np.array(tile_values) != carrier
    accents = {}
    for r in range(nr):
        for c in range(nc):
            accent_values = np.unique(tile_values[r][c][tiles[r, c]])
            if len(accent_values) != 1:
                raise ValueError("a tile motif cannot mix ground and highlight")
            accent = int(accent_values[0])
            accents.setdefault(accent, set()).add((r, c))
    if len(accents) != 2:
        raise ValueError("need ground tiles and highlighted clue tiles")
    ordered = sorted(accents, key=lambda color: len(accents[color]), reverse=True)
    ground, highlight = ordered
    supplied = accents[highlight]
    if not 2 <= len(supplied) < len(accents[ground]):
        raise ValueError("at least two clues and a strict ground-tile majority required")
    base = tiles[0, 0]
    # Progression is visible. Equivalent step descriptions are harmless: solve
    # uses the actual local masks, not a guessed hidden step or starting phase.
    steps = [(a, b) for a in range(1, h) for b in range(1, w)
             if np.array_equal(_phase_tiles(base, (nr, nc), (a, b)), tiles)]
    if not steps or np.array_equal(tiles[1, 0], base) or np.array_equal(tiles[0, 1], base):
        raise ValueError("motif phases must advance visibly on both axes")
    candidates = [(o, s) for o, s in _footprints(tiles) if supplied < s]
    if len(candidates) != 1:
        raise ValueError("clues must determine exactly one strict stencil completion")
    origin, complete = candidates[0]
    mask = tiles[origin]
    if not 3 <= int(mask.sum()) < h*w:
        raise ValueError("blueprint must be a proper mask with at least three cells")
    return {"grid": grid, "tiles": tiles, "thickness": t,
            "roles": (divider, carrier, ground, highlight), "steps": steps,
            "origin": origin, "mask": mask, "supplied": frozenset(supplied),
            "complete": complete, "missing": complete - supplied}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    out = info["grid"].copy()
    nr, nc, h, w = info["tiles"].shape
    t = info["thickness"]
    highlight = info["roles"][3]
    for r, c in info["missing"]:
        y, x = t + r * (h + t), t + c * (w + t)
        out[y:y+h, x:x+w][info["tiles"][r, c]] = highlight
    return out


@lru_cache(maxsize=15)
def _subset_codes(n):
    codes = np.arange(1, 1 << n, dtype=np.uint32)
    sizes = np.zeros(len(codes), dtype=np.uint8)
    for i in range(n):
        sizes += ((codes >> i) & 1).astype(np.uint8)
    codes.flags.writeable = sizes.flags.writeable = False
    return codes, sizes


def _valid_fragments(tiles, footprints, origin, split, anchor_status=None):
    complete = dict(footprints)[origin]
    points = sorted(complete)
    n = len(points)
    nr, nc = tiles.shape[:2]
    codes, sizes = _subset_codes(n)
    lower = max(2, (n+1)//2) if split == "train" else 2
    upper = min(n-1, (nr*nc-1)//2)
    if split == "test":
        upper = min(upper, (n-1)//2)
    codes = codes[(sizes >= lower) & (sizes <= upper)].copy()
    if anchor_status is not None:
        bit = 1 << points.index(origin) if origin in complete else 0
        if anchor_status and not bit:
            return points, codes[:0]
        codes = codes[(codes & bit != 0) if anchor_status else (codes & bit == 0)]
    for other, support in footprints:
        if other != origin:
            covered = sum(1 << i for i, p in enumerate(points) if p in support)
            # Subsets of another full footprint are ambiguous, even if that
            # footprint already equals the clue set (would predict no change).
            codes = codes[(codes & covered) != codes]
    return points, codes


def _witnesses(tiles, origin, supplied, complete):
    missing = complete - supplied
    # Every possible single supplied template fails at some destination.
    copy_fails = all(any(not np.array_equal(tiles[q], tiles[p]) for q in missing)
                     for p in supplied)
    ys, xs = zip(*supplied)
    bbox = frozenset((r, c) for r in range(min(ys), max(ys)+1)
                     for c in range(min(xs), max(xs)+1))
    rectangle_fails = complete != supplied | bbox
    origin_phase_matters = not np.array_equal(tiles[origin], tiles[0, 0])
    return copy_fails, rectangle_fails, origin_phase_matters


def _geometric_origins(tiles, supplied):
    """Windows containing the clues before inspecting any microscopic mask."""
    nr, nc, h, w = tiles.shape
    return [(r, c) for r in range(nr-h+1) for c in range(nc-w+1)
            if all(r <= y < r+h and c <= x < c+w for y, x in supplied)]


def _geometry(rng, shape):
    h, w = shape
    t = int(rng.choice((1, 2), p=(.75, .25)))
    largest = ((30-t)//(h+t), (30-t)//(w+t))
    def choose(side, limit):
        lo = side + 1 if limit > side else side
        return limit if rng.random() < .35 else int(rng.integers(lo, limit+1))
    return t, (choose(h, largest[0]), choose(w, largest[1]))


def _random_mask(rng, shape, split):
    h, w = shape
    lo = 5 if split == "test" else 3
    if h*w <= lo:
        return None
    n = int(rng.integers(lo, h*w))
    for _ in range(24):
        result = np.zeros(shape, dtype=bool)
        result.flat[rng.choice(h*w, n, replace=False)] = True
        if _full_box(result):
            return result
    # Finite construction, with broad extra-cell choices, not a mask vocabulary.
    result = np.zeros(shape, dtype=bool)
    result[0, 0] = result[h-1, 0] = result[h-1, w-1] = True
    free = np.flatnonzero(~result.ravel())
    result.flat[rng.choice(free, n-3, replace=False)] = True
    return result


def _assemble(tiles, t, roles, supplied, complete):
    return {"input": _paint(tiles, t, roles, supplied),
            "output": _paint(tiles, t, roles, complete)}


def _make_pair(rng, split, index, roles, used, train_masks, first_count):
    # All attempts are bounded. Each candidate enumerates the complete finite
    # translation set and all clue subsets before selecting a random valid one.
    for _ in range(96):
        shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
        target = _random_mask(rng, shape, split)
        if target is None:
            continue
        if split == "train" and index == 0 and not _asymmetric(target):
            continue
        if split == "train" and index == 1 and int(target.sum()) == first_count:
            continue
        if split == "test" and _key(target) in train_masks:
            continue
        t, lattice = _geometry(rng, shape)
        h, w = shape
        origin = (int(rng.integers(lattice[0]-h+1)),
                  int(rng.integers(lattice[1]-w+1)))
        steps = (int(rng.integers(1, h)), int(rng.integers(1, w)))
        base = np.roll(target, (steps[0]*origin[0], steps[1]*origin[1]), axis=(0, 1))
        tiles = _phase_tiles(base, lattice, steps)
        if np.array_equal(tiles[1, 0], base) or np.array_equal(tiles[0, 1], base):
            continue
        if split == "train" and index == 0 and np.array_equal(base, target):
            continue
        footprints = _footprints(tiles)
        anchor_status = (index == 0) if split == "train" and index < 2 else None
        points, codes = _valid_fragments(tiles, footprints, origin, split, anchor_status)
        if not len(codes):
            continue
        complete = dict(footprints)[origin]
        sizes = np.array([int(code).bit_count() for code in codes])
        size_order = rng.permutation(np.unique(sizes))
        # Examine at most 32 randomized fragments per geometry. A different
        # geometry/mask is cheaper than checking thousands of poor witnesses.
        choices = []
        for size in size_order:
            group = codes[sizes == size]
            choices.extend(map(int, rng.choice(group, min(8, len(group)), replace=False)))
        for code in choices[:32]:
            supplied = frozenset(p for i, p in enumerate(points) if code & (1 << i))
            copy_fails, rectangle_fails, _ = _witnesses(tiles, origin, supplied, complete)
            if not copy_fails or not rectangle_fails:
                continue
            if split == "train" and index == 1 and len(_geometric_origins(tiles, supplied)) < 2:
                continue
            pair = _assemble(tiles, t, roles, supplied, complete)
            if _key(pair["input"]) not in used:
                return pair
    # A known constructive geometry is a termination backstop, not the ordinary
    # sampling domain. Preferred masks are followed by the full finite mask
    # space, so prior training designs cannot exhaust the test fallback.
    strings = ("110100011", "110001101", "101100011", "111100001")
    preferred = [np.array([s == "1" for s in text]).reshape(3, 3) for text in strings]
    exhaustive = [np.array([(code >> bit) & 1 for bit in range(9)], dtype=bool).reshape(3, 3)
                  for code in range(1, 511)]
    for target in preferred + exhaustive:
        n = int(target.sum())
        if not _full_box(target) or n < (5 if split == "test" else 3):
            continue
        if split == "train" and index == 0 and not _asymmetric(target):
            continue
        if split == "train" and index == 1 and n == first_count:
            continue
        if split == "test" and _key(target) in train_masks:
            continue
        for origin in ((1, 2), (2, 1), (1, 1)):
            base = np.roll(target, (2*origin[0], origin[1]), axis=(0, 1))
            tiles = _phase_tiles(base, (5, 6), (2, 1))
            if np.array_equal(tiles[1, 0], base) or np.array_equal(tiles[0, 1], base):
                continue
            if split == "train" and index == 0 and np.array_equal(base, target):
                continue
            footprints = _footprints(tiles)
            status = (index == 0) if split == "train" and index < 2 else None
            points, codes = _valid_fragments(tiles, footprints, origin, split, status)
            complete = dict(footprints)[origin]
            for code in codes:
                supplied = frozenset(p for i, p in enumerate(points) if int(code) & (1 << i))
                if not all(_witnesses(tiles, origin, supplied, complete)[:2]):
                    continue
                if split == "train" and index == 1 and len(_geometric_origins(tiles, supplied)) < 2:
                    continue
                pair = _assemble(tiles, 1, roles, supplied, complete)
                if _key(pair["input"]) not in used:
                    return pair
    raise RuntimeError("bounded construction exhausted; no input was emitted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.random() < .3)
    base_roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    puzzle, used, train_masks = {"train": [], "test": []}, set(), set()
    first_count = None
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            roles = base_roles if stable_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
            if not stable_palette and split == "train" and index == 1 and roles[3] == base_roles[3]:
                roles = roles[1:] + roles[:1]
            pair = _make_pair(rng, split, index, roles, used, train_masks, first_count)
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
            info = _decode(pair["input"])
            if split == "train":
                train_masks.add(_key(info["mask"]))
                if index == 0:
                    first_count = int(info["mask"].sum())
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    info = _decode(pair["input"])
    out = _grid(pair["output"])
    inp = info["grid"]
    assert inp.shape == out.shape, "output dimensions must be preserved"
    tiles = info["tiles"]
    nr, nc, h, w = tiles.shape
    t = info["thickness"]
    divider, carrier, ground, highlight = info["roles"]
    interiors = np.zeros(inp.shape, dtype=bool)
    for r in range(nr):
        for c in range(nc):
            y, x = t+r*(h+t), t+c*(w+t)
            interiors[y:y+h, x:x+w] = True
            tile = out[y:y+h, x:x+w]
            occupancy = tiles[r, c]
            assert np.all(tile[~occupancy] == carrier), "carrier holes must survive"
            accent = highlight if (r, c) in info["complete"] else ground
            assert np.all(tile[occupancy] == accent), "wrong local phase or coarse footprint"
            if (r, c) not in info["missing"]:
                assert np.array_equal(tile, inp[y:y+h, x:x+w]), "unaffected tile changed"
    assert np.all(out[~interiors] == divider), "dividers must survive"
    n = int(info["mask"].sum())
    assert np.count_nonzero(out == highlight) == n*n, "n active tiles each contain n motif pixels"
    changed = inp != out
    assert np.count_nonzero(changed) == len(info["missing"])*n
    assert np.all(inp[changed] == ground) and np.all(out[changed] == highlight)
    assert np.array_equal(solve(inp), out)
    return info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            info = _check_pair(pair)
            key = _key(info["grid"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(info)
            n, supplied = int(info["mask"].sum()), len(info["supplied"])
            assert (2*supplied >= n) if split == "train" else (2*supplied < n)
            assert all(_witnesses(info["tiles"], info["origin"], info["supplied"], info["complete"])[:2])
    first, second = infos["train"][:2]
    assert _asymmetric(first["mask"]), "first demonstration must expose orientation"
    assert _witnesses(first["tiles"], first["origin"], first["supplied"], first["complete"])[2]
    assert first["origin"] in first["supplied"]
    assert second["origin"] not in second["supplied"]
    assert len(_geometric_origins(second["tiles"], second["supplied"])) >= 2
    assert int(first["mask"].sum()) != int(second["mask"].sum())
    train_masks = {_key(i["mask"]) for i in infos["train"]}
    assert all(_key(i["mask"]) not in train_masks for i in infos["test"])


def _fingerprint(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed variation audit, count checks and semantic corruptions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    masks, shapes, geometries, origins, phases, fragments, palettes = (set() for _ in range(7))
    clue_counts, missing_counts, densities, outputs = (set() for _ in range(4))
    roles = [set() for _ in range(4)]
    placements = {"train": set(), "test": set()}
    rect = disconnected = nozero = 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _check_pair(pair)
                nr, nc, h, w = i["tiles"].shape
                r, c = i["origin"]
                masks.add(_key(i["mask"]))
                shapes.add((h, w))
                geometries.add((nr, nc, h, w, i["thickness"]))
                origins.add((r, c))
                phases.update((h, w, a, b) for a, b in i["steps"])
                fragments.add(((h, w), tuple(sorted((y-r, x-c) for y, x in i["supplied"]))))
                palettes.add(tuple(sorted(i["roles"])))
                for j, value in enumerate(i["roles"]):
                    roles[j].add(value)
                clue_counts.add(len(i["supplied"]))
                missing_counts.add(len(i["missing"]))
                densities.add(int(i["mask"].sum()))
                outputs.add(_key(pair["output"]))
                placements[split].add((r == 0, r+h == nr, c == 0, c+w == nc))
                rect += h != w
                nozero += 0 not in i["roles"]
                supplied = i["supplied"]
                disconnected += any(all(abs(y-a)+abs(x-b) != 1 for a, b in supplied-{(y, x)})
                                    for y, x in supplied)
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert len(masks) >= 80 and len(outputs) >= 190
    assert shapes == set(_SHAPES) and len(geometries) >= 60
    assert len(phases) >= 15 and len(fragments) >= 80 and len(palettes) >= 50
    assert len(clue_counts) >= 5 and len(missing_counts) >= 6 and len(densities) >= 8
    assert all(len(colors) == 10 for colors in roles)
    assert rect >= 60 and disconnected >= 30 and nozero >= 50
    for split in ("train", "test"):
        assert any(not any(p) for p in placements[split]), "interior placement absent"
        assert all(any(p[side] for p in placements[split]) for side in range(4))
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(900+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2),
                 (3, 4), (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Input-only inference also supports sparse clues in what was a train pair,
    # and dense clues in what was a test pair: split is not a solving parameter.
    for i in (_decode(puzzles[0]["train"][0]["input"]),
              _decode(puzzles[0]["test"][0]["input"])):
        for split in ("train", "test"):
            points, codes = _valid_fragments(i["tiles"], _footprints(i["tiles"]), i["origin"], split)
            if len(codes):
                code = int(codes[0])
                clues = frozenset(p for j, p in enumerate(points) if code & (1 << j))
                pair = _assemble(i["tiles"], i["thickness"], i["roles"], clues, i["complete"])
                _check_pair(pair)
    # Four meaningful output failures: broken divider, lost hole, exact copy,
    # and an extra tile outside the self-stencil footprint.
    source = puzzles[0]["train"][0]
    i = _decode(source["input"])
    nr, nc, h, w = i["tiles"].shape
    t = i["thickness"]
    for kind in ("divider", "solid", "copy", "extra"):
        bad = {k: v.copy() for k, v in source.items()}
        out = bad["output"]
        if kind == "divider":
            out[0, 0] = i["roles"][1]
        elif kind == "copy":
            q = next(iter(i["supplied"]))
            tile = np.where(i["tiles"][q], i["roles"][3], i["roles"][1])
            for r, c in i["missing"]:
                out[t+r*(h+t):t+r*(h+t)+h, t+c*(w+t):t+c*(w+t)+w] = tile
        else:
            r, c = (next(iter(i["missing"])) if kind == "solid" else
                    next((r, c) for r in range(nr) for c in range(nc) if (r, c) not in i["complete"]))
            tile = out[t+r*(h+t):t+r*(h+t)+h, t+c*(w+t):t+c*(w+t)+w]
            if kind == "solid":
                tile[:] = i["roles"][3]
            else:
                tile[i["tiles"][r, c]] = i["roles"][3]
        try:
            _check_pair(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted {kind} corruption")
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    ambiguous = _paint(_phase_tiles(mask, (7, 7), (1, 1)), 1, (8, 7, 2, 4), {(2, 2), (2, 3)})
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted ambiguous origin")
    return {"seeds": 42, "pairs": 210, "blueprint_masks": len(masks),
            "tile_shapes": len(shapes), "geometries": len(geometries),
            "phase_progressions": len(phases), "fragments": len(fragments),
            "palettes": len(palettes), "outputs": len(outputs),
            "clue_counts": sorted(clue_counts), "missing_counts": sorted(missing_counts),
            "semantic_verification": "generator-owned; no independent solver"}
