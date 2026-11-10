"""Compositional generator for the corner-only scaffold adjacency task."""

import numpy as np


_SOURCE_ROWS = (
    ("7777777777777777", "7777777777777777", "7777777777777777",
     "7777777777777777", "7777766766777777", "7777677677677777",
     "7777766266777777", "7777677677677777", "7777766266777777",
     "7777677677677777", "7777777777777777", "7777777777777777",
     "7777777777777777", "7777777777777777", "7777777777777777",
     "7777777777777777"),
    ("7777777777777777", "7777777777777777", "7777777777777777",
     "7777777777777777", "7777777777777777", "7777777447777777",
     "7777774774777777", "7777747117477777", "7777747117477777",
     "7777774774777777", "7777777447777777", "7777777777777777",
     "7777777777777777", "7777777777777777", "7777777777777777",
     "7777777777777777"),
    ("7777777777777777", "7777777777777777", "7777777777777777",
     "7777777777777777", "7777777777777777", "7777777777777777",
     "7777771771777777", "7777777997777777", "7777777997777777",
     "7777771771777777", "7777777777777777", "7777777777777777",
     "7777777777777777", "7777777777777777", "7777777777777777",
     "7777777777777777"),
)
_SOURCE_INPUTS = tuple(np.array([[int(x) for x in row] for row in rows],
                                dtype=np.int8) for rows in _SOURCE_ROWS)
_EXTENDED = ("diamond", "stagger", "zigzag", "irregular", "straight", "composite")
_ALL_FAMILIES = _EXTENDED + ("guides",)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _counts(seed, num_train, num_test):
    if num_train is not None and (not _integer(num_train) or not 2 <= num_train <= 10):
        raise ValueError("num_train must be an integer from 2 to 10")
    if num_test is not None and (not _integer(num_test) or not 1 <= num_test <= 4):
        raise ValueError("num_test must be an integer from 1 to 4")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _neighbors(mask):
    p = np.pad(mask.astype(np.int8), 1)
    diagonal = p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
    orthogonal = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return diagonal, orthogonal


def _solve_with_roles(grid, background, scaffold, ink):
    diagonal, orthogonal = _neighbors(grid == scaffold)
    result = grid.copy()
    result[(grid == background) & (diagonal > 0) & (orthogonal == 0)] = ink
    return result


def _bbox(cells):
    return (min(r for r, _ in cells), max(r for r, _ in cells),
            min(c for _, c in cells), max(c for _, c in cells))


def _normalize(scaffold, ink):
    r0, _, c0, _ = _bbox(scaffold | ink)
    return ({(r-r0, c-c0) for r, c in scaffold},
            {(r-r0, c-c0) for r, c in ink})


def _orient(scaffold, ink, rng):
    rotation = int(rng.integers(4))
    flip = bool(rng.integers(2))
    def point(p):
        r, c = p
        if flip:
            c = -c
        for _ in range(rotation):
            r, c = c, -r
        return r, c
    return _normalize({point(p) for p in scaffold}, {point(p) for p in ink})


def _motif(rng, family):
    scaffold = set()
    if family == "diamond":
        rr, cc = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        corners = [(0, cc), (rr, 2*cc), (2*rr, cc), (rr, 0)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            steps = max(abs(a[0]-b[0]), abs(a[1]-b[1]))
            for k in range(steps+1):
                scaffold.add((round(a[0]+(b[0]-a[0])*k/steps),
                              round(a[1]+(b[1]-a[1])*k/steps)))
        if rng.random() < .55:
            for c in range(cc-1, cc+2):
                scaffold.add((rr, c))
    elif family == "stagger":
        h, w = int(rng.integers(7, 15)), int(rng.integers(8, 17))
        period = int(rng.integers(3, 5))
        for j, r in enumerate(range(0, h, 2)):
            for c in range((j % 2), w, period):
                if rng.random() < .85:
                    scaffold.add((r, c))
                    if c+1 < w and rng.random() < .8:
                        scaffold.add((r, c+1))
    elif family == "zigzag":
        h, w = int(rng.integers(8, 16)), int(rng.integers(9, 17))
        amplitude = int(rng.integers(2, min(5, w//2)))
        for r in range(h):
            phase = r % (2*amplitude)
            shift = phase if phase <= amplitude else 2*amplitude-phase
            for c in {1+shift, w-2-shift}:
                scaffold.add((r, c))
                if r % 3 == 0 and c+1 < w:
                    scaffold.add((r, c+1))
    elif family == "irregular":
        h, w = int(rng.integers(9, 16)), int(rng.integers(9, 17))
        top, bottom = 1, h-2
        for c in range(1, w-1):
            next_top = max(1, min(3, top + int(rng.integers(-1, 2))))
            next_bottom = max(h-4, min(h-2, bottom + int(rng.integers(-1, 2))))
            scaffold.update((r, c) for r in range(min(top, next_top), max(top, next_top)+1))
            scaffold.update((r, c) for r in range(min(bottom, next_bottom), max(bottom, next_bottom)+1))
            top, bottom = next_top, next_bottom
        scaffold.update((r, 1) for r in range(1, h-1))
        scaffold.update((r, w-2) for r in range(1, h-1))
        for _ in range(int(rng.integers(1, 5))):
            c = int(rng.integers(2, w-2))
            r = int(rng.integers(3, h-3))
            scaffold.add((r, c))
    elif family == "straight":
        h, w = int(rng.integers(7, 15)), int(rng.integers(8, 17))
        variant = int(rng.integers(4))
        if variant == 0:
            scaffold = {(r, c) for r in range(h) for c in range(w)
                        if r in (0, h-1) or c in (0, w-1)}
            for r in range(2, h-2, 3):
                scaffold.update((r, c) for c in range(0, w, 2))
        elif variant == 1:
            scaffold = {(r, c) for r in range(h) for c in range(w)
                        if c in (0, w-1) or r in (0, h-1, h//2)}
        elif variant == 2:
            scaffold = {(r, c) for r in range(h) for c in range(w)
                        if r in (0, h//2, h-1) or c in (0, w//2, w-1)}
        else:
            scaffold = {(h//2, c) for c in range(w)}
            for j, c in enumerate(range(0, w, 3)):
                rows = range(0, h//2) if j % 2 else range(h//2+1, h)
                scaffold.update((r, c) for r in rows)
    elif family == "guides":
        h, w = int(rng.integers(6, 12)), int(rng.integers(6, 12))
        scaffold = {(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)}
        for _ in range(int(rng.integers(0, 8))):
            side = int(rng.integers(4))
            if side == 0:
                scaffold.add((0, int(rng.integers(1, w-1))))
            elif side == 1:
                scaffold.add((h-1, int(rng.integers(1, w-1))))
            elif side == 2:
                scaffold.add((int(rng.integers(1, h-1)), 0))
            else:
                scaffold.add((int(rng.integers(1, h-1)), w-1))
        if rng.random() < .45:
            scaffold.add((1, 0))
    elif family == "composite":
        left = _motif(rng, str(rng.choice(_EXTENDED[:5])))
        right = _motif(rng, str(rng.choice(_EXTENDED[:5])))
        left = {(r, c) for r, c in left if r < 10 and c < 9}
        right = {(r+int(rng.integers(0, 4)), c+11) for r, c in right
                 if r < 10 and c < 9}
        scaffold = left | right
    return scaffold


def _ink_for(scaffold, rng):
    r0, r1, c0, c1 = _bbox(scaffold)
    if r1-r0 < 3 or c1-c0 < 3:
        raise ValueError("scaffold too narrow")
    candidates = [(r, c) for r in range(r0+1, r1) for c in range(c0+1, c1)
                  if (r, c) not in scaffold and
                  min(max(abs(r-sr), abs(c-sc)) for sr, sc in scaffold) <= 4]
    if len(candidates) < 6:
        raise ValueError("no ink room")
    center_r, center_c = (r0+r1)/2, (c0+c1)/2
    candidates.sort(key=lambda p: (abs(p[0]-center_r)+abs(p[1]-center_c), p))
    pool = candidates[:min(len(candidates), max(12, len(candidates)//2))]
    mode = int(rng.integers(4))
    if mode in (0, 1):
        count = int(rng.integers(2, 7))
        order = rng.permutation(len(pool))
        ink = set()
        for i in order:
            p = pool[int(i)]
            if mode == 0 and any(abs(p[0]-q[0])+abs(p[1]-q[1]) < 2 for q in ink):
                continue
            ink.add(p)
            if len(ink) == count:
                break
        return ink
    if mode == 2:
        shapes = [(2, 2), (2, 3), (3, 2), (1, 3), (3, 1)]
        shape = shapes[int(rng.integers(len(shapes)))]
        starts = []
        available = set(candidates)
        for r, c in pool:
            cells = {(r+dr, c+dc) for dr in range(shape[0]) for dc in range(shape[1])}
            if cells <= available:
                starts.append(cells)
        if starts:
            return starts[int(rng.integers(len(starts)))]
    # Bent and partially separated cue clusters.
    anchor = pool[int(rng.integers(min(8, len(pool))))]
    ink = {anchor}
    for _ in range(int(rng.integers(2, 6))):
        options = [p for p in candidates if p not in ink and
                   min(abs(p[0]-q[0])+abs(p[1]-q[1]) for q in ink) <= 2]
        if options:
            ink.add(options[int(rng.integers(len(options)))])
    return ink


def _embed(scaffold, ink, rng, colors):
    scaffold, ink = _orient(scaffold, ink, rng)
    _, r1, _, c1 = _bbox(scaffold | ink)
    local_h, local_w = r1+1, c1+1
    if local_h > 27 or local_w > 27:
        raise ValueError("motif too large")
    crop_mode = int(rng.integers(3))
    limits = ((0, 3), (2, 9), (6, 17))[crop_mode]
    def extra(local):
        lo, hi = limits
        hi = min(hi, 30-local)
        lo = min(lo, hi)
        return int(rng.integers(lo, hi+1))
    eh, ew = extra(local_h), extra(local_w)
    top = int(rng.integers(eh+1))
    left = int(rng.integers(ew+1))
    background, scaffold_color, ink_color = colors
    grid = np.full((local_h+eh, local_w+ew), background, dtype=np.int8)
    for r, c in scaffold:
        grid[top+r, left+c] = scaffold_color
    for r, c in ink:
        grid[top+r, left+c] = ink_color
    return grid, _solve_with_roles(grid, *colors)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    sizes = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for dy, dx in ((1,0), (-1,0), (0,1), (0,-1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _input_roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3
    background = int(colors[int(np.argmax(counts))])
    foreground = [int(x) for x in colors if x != background]
    def area(color):
        rr, cc = np.nonzero(grid == color)
        return (int(rr.max()-rr.min()+1) * int(cc.max()-cc.min()+1),
                int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max()))
    a, b = area(foreground[0]), area(foreground[1])
    assert a[0] != b[0]
    scaffold, ink = (foreground[0], foreground[1]) if a[0] > b[0] else (foreground[1], foreground[0])
    outer, inner = area(scaffold), area(ink)
    assert outer[1] < inner[1] and inner[2] < outer[2]
    assert outer[3] < inner[3] and inner[4] < outer[4]
    return background, scaffold, ink


def _features(grid, output):
    background, scaffold, ink = _input_roles(grid)
    sm, im = grid == scaffold, grid == ink
    diagonal, orthogonal = _neighbors(sm)
    eligible = (grid == background) & (diagonal > 0) & (orthogonal == 0)
    veto = (grid == background) & (diagonal > 0) & (orthogonal > 0)
    assert np.array_equal(output, _solve_with_roles(grid, background, scaffold, ink))
    assert np.count_nonzero(eligible) > 0
    sr, sc = np.nonzero(sm)
    corners = {(int(sr.min())-1, int(sc.min())-1),
               (int(sr.min())-1, int(sc.max())+1),
               (int(sr.max())+1, int(sc.min())-1),
               (int(sr.max())+1, int(sc.max())+1)}
    changed = set(zip(*np.nonzero(eligible)))
    edge = bool(np.any(sm[0]) or np.any(sm[-1]) or np.any(sm[:,0]) or np.any(sm[:,-1]))
    return {"colors": (background, scaffold, ink), "added": len(changed),
            "veto": int(veto.sum()), "double": int(((grid == background) & (diagonal >= 2) & (orthogonal == 0)).sum()),
            "mixed": int(veto.sum()), "existing_qualifying": int((im & (diagonal > 0) & (orthogonal == 0)).sum()),
            "noncorner": bool(changed - corners), "edge": edge,
            "scaffold_components": _components(sm), "ink_components": _components(im),
            "scaffold_count": int(sm.sum()), "ink_count": int(im.sum())}


def _palette(rng, palette_mode, stable_colors, index, section):
    if palette_mode == "stable":
        return stable_colors
    if palette_mode == "swap":
        bg, sc, ink = stable_colors
        return (bg, ink, sc) if index % 2 else (bg, sc, ink)
    if palette_mode == "varied" or section == "train":
        return tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    # A transfer can reuse a training color as scaffold and reveal a new ink.
    bg, sc, ink = stable_colors
    available = [x for x in range(10) if x not in stable_colors]
    return bg, ink, int(rng.choice(available))


def _pair(rng, family, colors, witness=False, require_veto=False):
    for _ in range(240):
        try:
            scaffold = _motif(rng, family)
            ink = _ink_for(scaffold, rng)
            grid, output = _embed(scaffold, ink, rng, colors)
            if any(grid.shape == source.shape and np.array_equal(grid, source)
                   for source in _SOURCE_INPUTS):
                continue
            f = _features(grid, output)
            if witness and not (f["noncorner"] and f["added"] >= 6 and f["veto"] > 0 and f["double"] > 0):
                continue
            if require_veto and f["veto"] == 0:
                continue
            return {"input": grid, "output": output}
        except (AssertionError, ValueError):
            pass
    raise ValueError("could not construct pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    shift = ("stay", "transfer", "mixed")[int(rng.integers(3))]
    palette_mode = ("stable", "swap", "varied", "novel")[int(rng.integers(4))]
    stable_colors = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    stay_family = str(rng.choice(_EXTENDED))
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for attempt in range(50):
                child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                if shift == "stay":
                    family = stay_family
                elif shift == "transfer":
                    family = str(child.choice(_EXTENDED)) if section == "train" else "guides"
                else:
                    family = str(child.choice(_ALL_FAMILIES))
                if section == "train" and index == 0 and family == "guides":
                    family = str(child.choice(_EXTENDED))
                colors = _palette(child, palette_mode, stable_colors, index, section)
                try:
                    pair = _pair(child, family, colors,
                                 witness=(section == "train" and index == 0),
                                 require_veto=(section == "test" and index == 0 and shift != "transfer"))
                except ValueError:
                    continue
                key = (pair["input"].shape, pair["input"].tobytes())
                if key in seen:
                    continue
                seen.add(key)
                puzzle[section].append(pair)
                break
            else:
                raise ValueError("could not construct unique puzzle")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_features = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            x, y = pair["input"], pair["output"]
            for grid in (x, y):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert x.shape == y.shape
            assert not any(x.shape == src.shape and np.array_equal(x, src) for src in _SOURCE_INPUTS)
            key = (x.shape, x.tobytes())
            pair_key = (key, y.tobytes())
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            f = _features(x, y)
            assert f["scaffold_count"] >= 4 and f["ink_count"] >= 1
            if section == "train":
                train_features.append(f)
    assert any(f["noncorner"] and f["added"] >= 6 and f["veto"] > 0 and f["double"] > 0
               for f in train_features)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, structures, colors = set(), set(), set()
    seen_families = {"isolated": 0, "extended": 0}
    varied_backgrounds = 0
    separated_ink = 0
    boundary = 0
    veto_test = 0
    diagonal_only = isolated_test = double_test = noncorner_test = 0
    existing_ink = edge_additions = nonrectangular_ink = 0
    familiar_test_ink = novel_test_ink = 0
    stable_palettes = role_swaps = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        backgrounds = set()
        palettes = set()
        train_ink_colors = set()
        train_scaffold_colors = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                x, y = pair["input"], pair["output"]
                f = _features(x, y)
                backgrounds.add(f["colors"][0])
                palettes.add(f["colors"])
                if section == "train":
                    train_ink_colors.add(f["colors"][2])
                    train_scaffold_colors.add(f["colors"][1])
                else:
                    familiar_test_ink += f["colors"][2] in train_ink_colors
                    novel_test_ink += f["colors"][2] not in train_ink_colors
                shapes.add(x.shape)
                colors.add(f["colors"])
                sc = x == f["colors"][1]
                rr, cc = np.nonzero(sc)
                crop = sc[rr.min():rr.max()+1, cc.min():cc.max()+1]
                structures.add((crop.shape, crop.tobytes()))
                seen_families["isolated" if max(f["scaffold_components"]) == 1 else "extended"] += 1
                separated_ink += len(f["ink_components"]) > 1
                boundary += f["edge"]
                diagonal_only += max(f["scaffold_components"]) <= 2
                existing_ink += f["existing_qualifying"] > 0
                added = x != y
                edge_additions += bool(np.any(added[0]) or np.any(added[-1]) or
                                       np.any(added[:, 0]) or np.any(added[:, -1]))
                ink_mask = x == f["colors"][2]
                ir, ic = np.nonzero(ink_mask)
                nonrectangular_ink += len(ir) != (int(ir.max()-ir.min()+1) * int(ic.max()-ic.min()+1))
                if section == "test":
                    veto_test += f["veto"] > 0
                    double_test += f["double"] > 0
                    noncorner_test += f["noncorner"]
                    isolated_test += max(f["scaffold_components"]) == 1
        varied_backgrounds += len(backgrounds) > 1
        stable_palettes += len(palettes) == 1
        role_swaps += bool(train_ink_colors & train_scaffold_colors)
    assert len(shapes) >= 40 and len(structures) >= 90 and len(colors) >= 30
    assert all(seen_families.values()) and separated_ink >= 15
    assert boundary >= 20 and veto_test >= 15 and varied_backgrounds >= 2
    assert diagonal_only >= 20 and isolated_test >= 2
    assert double_test >= 15 and noncorner_test >= 20
    assert existing_ink >= 15 and edge_additions >= 20
    assert nonrectangular_ink >= 20
    assert familiar_test_ink >= 5 and novel_test_ink >= 5
    assert stable_palettes >= 2 and role_swaps >= 2
    for seed, nt, ntest in ((5001, 2, 1), (5002, 10, 1), (5003, 10, 4),
                            (6001, None, 1), (6002, None, 4),
                            (6003, 2, None), (6004, 10, None)):
        p = generate_puzzle(seed, nt, ntest)
        validate_generated_puzzle(p)
        if nt is not None:
            assert len(p["train"]) == nt
        if ntest is not None:
            assert len(p["test"]) == ntest
        q = generate_puzzle(seed, nt, ntest)
        assert all(np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
                   for section in ("train", "test") for a, b in zip(p[section], q[section]))
    assert not np.array_equal(generate_puzzle(100)["train"][0]["input"],
                              generate_puzzle(101)["train"][0]["input"])
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 3}, {"num_train": True},
                   {"num_test": False}):
        try:
            generate_puzzle(9000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
