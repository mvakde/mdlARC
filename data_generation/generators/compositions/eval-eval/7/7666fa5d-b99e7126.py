"""Fit a self-replicating tile stencil inside a finite diagonal enclosure.

No source files, saved examples, global RNG, or generation metadata are used by
solve. Generation and self_test write no files. All search bounds are explicit.
"""
from itertools import combinations
import hashlib

import numpy as np


_TILE_SHAPES = ((2, 3), (3, 2), (3, 3))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _normal(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(v), len(roles)) for v in a.flat)


def _runs(indices):
    groups = []
    for i in map(int, indices):
        if groups and i == groups[-1][1] + 1:
            groups[-1] = (groups[-1][0], i)
        else:
            groups.append((i, i))
    return groups


def _strokes(points, slope):
    """Return maximal straight runs, or None if any run is a singleton."""
    lines = {}
    for r, c in points:
        lines.setdefault(int(c - slope*r), []).append(int(r))
    strokes = []
    for q, rows in lines.items():
        for low, high in _runs(sorted(rows)):
            if low == high:
                return None
            strokes.append((q, 2*low + slope*q, 2*high + slope*q))
    return sorted(strokes)


def _region(shape, strokes, slope, rails, alternative=""):
    """Continuous-interval sweep; coordinates refer to logical tile centers."""
    rr, cc = np.indices(shape)
    q, t = cc - slope*rr, rr + slope*cc
    answer = np.zeros(shape, dtype=bool)
    for level in range(int(t.min()), int(t.max()) + 1):
        active = [a for a in strokes if alternative == "infinite" or a[1] <= level <= a[2]]
        if alternative == "pixel_centers":
            active = [a for a in active if (level - slope*a[0]) % 2 == 0]
        if len(active) >= 2:
            answer |= (t == level) & (q > min(a[0] for a in active)) & (q < max(a[0] for a in active))
    answer[rails] = False
    return answer


def _pair_region(shape, strokes, slope, rails, consecutive=False):
    """Geometric union used by validation, separate from the sweep above."""
    rr, cc = np.indices(shape)
    q, t = cc - slope*rr, rr + slope*cc
    answer = np.zeros(shape, dtype=bool)
    allowed_pairs = None
    if consecutive:
        # Consecutive ranks within each interval-interaction component.
        remaining = set(range(len(strokes)))
        allowed_pairs = set()
        while remaining:
            component, todo = set(), [remaining.pop()]
            while todo:
                i = todo.pop()
                component.add(i)
                neighbors = {j for j in remaining
                             if max(strokes[i][1], strokes[j][1]) <= min(strokes[i][2], strokes[j][2])}
                remaining -= neighbors
                todo.extend(neighbors)
            ordered = sorted(component)
            allowed_pairs.update(zip(ordered, ordered[1:]))
    for i, a in enumerate(strokes):
        for j in range(i + 1, len(strokes)):
            if consecutive and (i, j) not in allowed_pairs:
                continue
            b = strokes[j]
            answer |= ((q > a[0]) & (q < b[0])
                       & (t >= max(a[1], b[1])) & (t <= min(a[2], b[2])))
    answer[rails] = False
    return answer


def _points(mask, origin):
    return {(int(r) + origin[0], int(c) + origin[1]) for r, c in np.argwhere(mask)}


def _placements(mask, shape, clues, allowed):
    """All same-orientation fitting translations; ties are never scored away."""
    h, w = mask.shape
    local = np.argwhere(mask)
    result = []
    for r in range(shape[0] - h + 1):
        for c in range(shape[1] - w + 1):
            if not all(0 <= y-r < h and 0 <= x-c < w and mask[y-r, x-c] for y, x in clues):
                continue
            absolute = local + (r, c)
            if np.all(allowed[absolute[:, 0], absolute[:, 1]]):
                result.append((r, c))
    return result


def _asymmetric(mask):
    target = _key(mask)
    others = [np.rot90(mask, k) for k in range(1, 4)]
    others += [np.fliplr(np.rot90(mask, k)) for k in range(4)]
    return all(_key(a) != target for a in others)


def _decode(grid):
    """Infer lattice, ordinary motif, glyph, and rails from this input alone."""
    divider = int(grid[0, 0])
    row_runs = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    col_runs = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    assert len(row_runs) >= 3 and len(col_runs) >= 3, "missing divider lattice"
    assert row_runs[0][0] == col_runs[0][0] == 0
    assert row_runs[-1][1] == grid.shape[0]-1 and col_runs[-1][1] == grid.shape[1]-1
    thicknesses = {b-a+1 for a, b in row_runs + col_runs}
    assert len(thicknesses) == 1
    d = thicknesses.pop()
    heights = {row_runs[i+1][0] - row_runs[i][1] - 1 for i in range(len(row_runs)-1)}
    widths = {col_runs[i+1][0] - col_runs[i][1] - 1 for i in range(len(col_runs)-1)}
    assert len(heights) == len(widths) == 1
    h, w = heights.pop(), widths.pop()
    assert (h, w) in _TILE_SHAPES and d in (1, 2)
    nr, nc = len(row_runs)-1, len(col_runs)-1
    rails = np.zeros((nr, nc), dtype=bool)
    rail_colors = {}
    records = {}
    for r in range(nr):
        for c in range(nc):
            y, x = d + r*(h+d), d + c*(w+d)
            tile = grid[y:y+h, x:x+w]
            assert divider not in tile, "divider color inside a tile"
            if np.all(tile == tile[0, 0]):
                rails[r, c] = True
                rail_colors[(r, c)] = int(tile[0, 0])
            else:
                records.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
    assert len(records) == 2, "need ordinary and exceptional nonuniform tiles"
    ordered = sorted(records.values(), key=lambda v: len(v[1]), reverse=True)
    ordinary, ordinary_positions = ordered[0]
    glyph, clues = ordered[1]
    assert len(ordinary_positions) > len(clues) >= 1
    ordinary_colors = set(map(int, np.unique(ordinary)))
    assert len(ordinary_colors) in (2, 3)
    extra = set(map(int, np.unique(glyph))) - ordinary_colors
    assert len(extra) == 1
    ink = extra.pop()
    mask = glyph == ink
    assert 4 <= int(mask.sum()) < h*w
    assert len(np.unique(glyph[~mask])) == 1 and int(glyph[~mask][0]) in ordinary_colors
    active = np.argwhere(mask)
    assert tuple(active.min(axis=0)) == (0, 0) and tuple(active.max(axis=0)) == (h-1, w-1)
    assert set(rail_colors.values()).isdisjoint(ordinary_colors | {ink, divider})
    candidates = []
    for slope in (-1, 1):
        strokes = _strokes(rail_colors, slope)
        if strokes is not None and len(strokes) >= 2:
            if all(len({rail_colors[p] for p in rail_colors if p[1]-slope*p[0] == q and lo <= p[0]+slope*p[1] <= hi}) == 1
                   for q, lo, hi in strokes):
                candidates.append((slope, strokes))
    assert len(candidates) == 1, "rails must have one recognizable slope"
    slope, strokes = candidates[0]
    return dict(geometry=(h, w, d, nr, nc), divider=divider, rails=rails,
                rail_colors=rail_colors, slope=slope, strokes=strokes,
                ordinary=ordinary, glyph=glyph, clues=set(clues), mask=mask, ink=ink)


def _copy_at(grid, info, positions):
    output = grid.copy()
    h, w, d, _, _ = info["geometry"]
    for r, c in positions:
        y, x = d + r*(h+d), d + c*(w+d)
        output[y:y+h, x:x+w] = info["glyph"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    info = _decode(grid)
    enclosure = _region(info["rails"].shape, info["strokes"], info["slope"], info["rails"])
    origins = _placements(info["mask"], enclosure.shape, info["clues"], enclosure)
    if len(origins) != 1:
        raise ValueError("the complete stencil must have exactly one enclosed registration")
    active = _points(info["mask"], origins[0])
    return _copy_at(grid, info, active - info["clues"])


def _geometry(rng, stable=None):
    if stable is not None:
        return stable
    h, w = _TILE_SHAPES[int(rng.integers(len(_TILE_SHAPES)))]
    d = int(rng.choice([1, 2], p=[.8, .2]))
    nr = int(rng.integers(5, (30-d)//(h+d) + 1))
    nc = int(rng.integers(5, (30-d)//(w+d) + 1))
    return h, w, d, nr, nc


def _scene(rng, geometry, count, visible_count, witness, forbidden_mask=None, order_slope=1):
    """Sample geometry, then subsets of its admissible stencil patches."""
    h, w, _, nr, nc = geometry
    interior = nr >= 7 and nc >= 7 and rng.random() < .50
    for _ in range(160):
        # A compact rail assembly can float within a larger coarse canvas.
        # This adds actual interior placement, not merely extra pixel padding.
        bh, bw, top, left = nr, nc, 0, 0
        if interior:
            bh, bw = int(rng.integers(5, nr-1)), int(rng.integers(5, nc-1))
            top, left = int(rng.integers(1, nr-bh)), int(rng.integers(1, nc-bw))
        elif rng.random() < .45:
            bh, bw = int(rng.integers(5, nr+1)), int(rng.integers(5, nc+1))
            top, left = int(rng.integers(nr-bh+1)), int(rng.integers(nc-bw+1))
        qs = sorted(map(int, rng.choice(np.arange(-bh+2, bw-1), count, replace=False)))
        if min(np.diff(qs)) < 2:
            continue
        rail_points = []
        for q in qs:
            legal = [(r+top, r+q+left) for r in range(bh) if 0 <= r+q < bw]
            length = int(rng.integers(2, len(legal)+1))
            start = int(rng.integers(len(legal)-length+1))
            rail_points.extend(legal[start:start+length])
        strokes = _strokes(rail_points, 1)
        if _strokes(rail_points, -1) is not None:
            continue
        rails = np.zeros((nr, nc), dtype=bool)
        rails[tuple(zip(*rail_points))] = True
        enclosure = _region((nr, nc), strokes, 1, rails)
        patches = [(r, c) for r in range(nr-h+1) for c in range(nc-w+1)
                   if int(enclosure[r:r+h, c:c+w].sum()) >= 4]
        if not patches:
            continue
        alternative = None
        if witness == "skip":
            alternative = _pair_region((nr, nc), strokes, 1, rails, consecutive=True)
        elif witness == "finite":
            alternative = _region((nr, nc), strokes, 1, rails, "pixel_centers")
        for patch_index in rng.permutation(len(patches))[:8]:
            r, c = patches[int(patch_index)]
            patch = enclosure[r:r+h, c:c+w]
            mask = patch & (rng.random((h, w)) < rng.uniform(.70, 1.01))
            if int(mask.sum()) < 4 or mask.all() or _key(mask) == forbidden_mask:
                continue
            local = np.argwhere(mask)
            if tuple(local.min(0)) != (0, 0) or tuple(local.max(0)) != (h-1, w-1):
                continue
            if witness == "finite" and not _asymmetric(mask):
                continue
            if witness == "finite" and (patch.all() or not np.any(~mask & ~rails[r:r+h, c:c+w])):
                # Teach active-footprint fitting rather than requiring the
                # whole stencil window to fit, or copying its rectangle.
                continue
            if int(mask.sum()) >= int(enclosure.sum()):
                continue  # Copying the whole enclosure must be visibly wrong.
            sizes = [visible_count] if visible_count else [2, 3]
            sizes = [k for k in sizes if k < len(local)]
            rng.shuffle(sizes)
            for k in sizes:
                fragments = list(combinations(range(len(local)), k))
                for fi in rng.permutation(len(fragments))[:24]:
                    chosen = fragments[int(fi)]
                    clues = {tuple(map(int, p)) for p in local[list(chosen)] + (r, c)}
                    if _placements(mask, (nr, nc), clues, enclosure) != [(r, c)]:
                        continue
                    free_origins = _placements(mask, (nr, nc), clues, ~rails)
                    if len(free_origins) < 2:
                        continue  # Actual geometric ablation, keeping rails as obstacles.
                    ordered = sorted(free_origins, key=lambda p: (p[0], order_slope*p[1]))
                    if (witness == "finite" and (r, c) == ordered[-1]
                            or witness == "skip" and (r, c) == ordered[0]):
                        continue
                    active = _points(mask, (r, c))
                    missing = active - clues
                    if alternative is not None and all(alternative[p] for p in missing):
                        continue
                    if witness == "finite":
                        infinite = _region((nr, nc), strokes, 1, rails, "infinite")
                        if len(_placements(mask, (nr, nc), clues, infinite)) < 2:
                            continue
                    return rails, mask, (r, c), clues
    return None


# Only bounded-search fallbacks. The ordinary sampler above uses arbitrary
# endpoints and arbitrary ink subsets, not a stencil or rail-layout catalog.
_FALLBACKS = (
    (((1, 0), (2, 1), (3, 2), (4, 3), (0, 4), (1, 5), (2, 6)),
     ("011", "111", "001"), (1, 2), ((2, 4), (3, 4))),
    (((3, 0), (4, 1), (5, 2), (6, 3), (5, 4), (6, 5), (1, 3), (2, 4), (3, 5), (4, 6)),
     ("101", "001", "001"), (3, 1), ((3, 3), (4, 3))),
)


def _reflect_scene(scene):
    rails, mask, origin, clues = scene
    nr, nc = rails.shape
    h, w = mask.shape
    return (rails[:, ::-1].copy(), mask[:, ::-1].copy(),
            (origin[0], nc-w-origin[1]), {(r, nc-1-c) for r, c in clues})


def _ordinary(rng, shape, colors, forbidden=None, code=None):
    for attempt in range(32):
        n = int(np.prod(shape))
        if code is not None:
            labels = np.array([(code >> i) & 1 for i in range(n)]).reshape(shape)
        else:
            labels = np.concatenate((np.arange(len(colors)), rng.integers(len(colors), size=n-len(colors))))
            rng.shuffle(labels)
            labels = labels.reshape(shape)
        tile = np.array(colors, dtype=np.int8)[labels]
        if len(np.unique(tile)) >= 2 and _normal(tile) != forbidden:
            return tile
        code = None
    raise AssertionError("ordinary motif sampling exhausted")


def _assemble(rng, geometry, scene, palette, forbidden_motif=None, code=None):
    rails, mask, origin, clues = scene
    h, w, d, nr, nc = geometry
    divider, ink = palette[:2]
    rail_palette, ordinary_colors = palette[2], palette[3]
    ordinary = _ordinary(rng, (h, w), ordinary_colors, forbidden_motif, code)
    backgrounds = [v for v in ordinary_colors if np.any(ordinary[~mask] != v)]
    background = int(rng.choice(backgrounds))
    glyph = np.where(mask, ink, background).astype(np.int8)
    grid = np.full((nr*(h+d)+d, nc*(w+d)+d), divider, dtype=np.int8)
    slope = 1 if _strokes(map(tuple, np.argwhere(rails)), 1) is not None else -1
    strokes = _strokes(map(tuple, np.argwhere(rails)), slope)
    color_order = list(map(int, rng.permutation(rail_palette)))
    stroke_order = [strokes[int(i)] for i in rng.permutation(len(strokes))]
    stroke_colors = {stroke: color_order[i % len(color_order)] for i, stroke in enumerate(stroke_order)}
    for r in range(nr):
        for c in range(nc):
            y, x = d + r*(h+d), d + c*(w+d)
            tile = glyph if (r, c) in clues else ordinary
            if rails[r, c]:
                q, t = c-slope*r, r+slope*c
                stroke = next(a for a in strokes if a[0] == q and a[1] <= t <= a[2])
                tile = stroke_colors[stroke]
            grid[y:y+h, x:x+w] = tile
    full = _points(mask, origin)
    output = grid.copy()
    for r, c in full - clues:
        y, x = d + r*(h+d), d + c*(w+d)
        output[y:y+h, x:x+w] = glyph
    return {"input": grid, "output": output}


def _palette(rng):
    colors = list(map(int, rng.permutation(10)))
    rail_count = 2 if rng.random() < .35 else 1
    ordinary_count = 3 if rng.random() < .25 else 2
    return (colors[0], colors[1], tuple(colors[2:2+rail_count]),
            tuple(colors[2+rail_count:2+rail_count+ordinary_count]))


def _make_pair(rng, stable_geometry, palette, slope, visible, witness,
               seen, forbidden_mask=None, forbidden_motif=None):
    for _ in range(12):
        geometry = _geometry(rng, stable_geometry)
        count = 3 if witness == "skip" else (2 if witness == "finite" else int(rng.choice([2, 3, 4], p=[.4, .4, .2])))
        scene = _scene(rng, geometry, count, visible, witness,
                       forbidden_mask if slope == 1 else None, slope)
        if scene is None:
            continue
        if slope == -1:
            scene = _reflect_scene(scene)
        if _key(scene[1]) == forbidden_mask:
            continue
        pair = _assemble(rng, geometry, scene, palette, forbidden_motif)
        if _key(pair["input"]) not in seen:
            return pair
    # Guaranteed finite reserve: two asymmetric designs and 510 ordinary motifs,
    # against at most 13 preceding inputs. Fallbacks use the same family rule.
    rails0, rows, origin0, clues0 = _FALLBACKS[witness == "skip"]
    rails = np.zeros((7, 7), dtype=bool)
    rails[tuple(zip(*rails0))] = True
    mask = np.array([[int(v) for v in row] for row in rows], dtype=bool)
    scene = (rails, mask, origin0, set(clues0))
    if visible == 1:
        scene = (rails, mask, origin0, {clues0[0]})
    if slope == -1:
        scene = _reflect_scene(scene)
    # Transposition changes the asymmetric design, retaining slope and the
    # first/last-registration counterexample of these particular reserves.
    if _key(scene[1]) == forbidden_mask:
        ra, ma, (r, c), cl = scene
        scene = (ra.T.copy(), ma.T.copy(), (c, r), {(x, y) for y, x in cl})
    fallback_palette = (palette[0], palette[1], palette[2], palette[3][:2])
    for code in range(1, 511):
        pair = _assemble(rng, (3, 3, 1, 7, 7), scene, fallback_palette, forbidden_motif, code)
        if _key(pair["input"]) not in seen:
            return pair
    raise AssertionError("finite uniqueness reserve exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    fixed_palette = _palette(rng) if rng.random() < .25 else None
    stable_geometry = None
    if rng.random() < .30:
        h, w = _TILE_SHAPES[int(rng.integers(3))]
        stable_geometry = (h, w, 1, int(rng.integers(6, (29//(h+1))+1)),
                           int(rng.integers(6, (29//(w+1))+1)))
    base_slope = int(rng.choice([-1, 1]))
    puzzle, seen, train_info = {"train": [], "test": []}, set(), []
    for split, number in (("train", nt), ("test", ne)):
        for index in range(number):
            witness = "finite" if split == "train" and index == 0 else "skip" if split == "train" and index == 1 else ""
            slope = base_slope * (-1 if split == "train" and index == 1 else 1)
            if index >= 2 or split == "test":
                slope = int(rng.choice([-1, 1]))
            palette = fixed_palette or _palette(rng)
            if fixed_palette is None and split == "train" and index == 1:
                # Teach both color-independent rail participation and ink roles.
                if palette[1] == train_info[0]["ink"]:
                    palette = (palette[1], palette[0], palette[2], palette[3])
                if len(palette[2]) == 1:
                    used_colors = {palette[0], palette[1], *palette[2], *palette[3]}
                    extra = int(rng.choice([v for v in range(10) if v not in used_colors]))
                    palette = (palette[0], palette[1], (*palette[2], extra), palette[3])
            pair = _make_pair(rng, stable_geometry, palette, slope,
                              1 if split == "test" else 0, witness, seen,
                              _key(train_info[0]["mask"]) if split == "train" and index == 1 else None,
                              _normal(train_info[0]["ordinary"]) if split == "train" and index == 1 else None)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train":
                train_info.append(_decode(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, output = _grid(pair["input"]), _grid(pair["output"])
    assert grid.shape == output.shape
    info = _decode(grid)
    rails, mask, clues = info["rails"], info["mask"], info["clues"]
    enclosure = _pair_region(rails.shape, info["strokes"], info["slope"], rails)
    assert np.array_equal(enclosure, _region(rails.shape, info["strokes"], info["slope"], rails))
    origins = _placements(mask, rails.shape, clues, enclosure)
    assert len(origins) == 1, "ambiguous or impossible enclosed stencil"
    active = _points(mask, origins[0])
    assert clues < active, "at least one complete tile must be added"
    assert len(active) < int(enclosure.sum()), "whole-enclosure filling shortcut"
    assert len(_placements(mask, rails.shape, clues, ~rails)) >= 2, "A must resolve genuine placement ambiguity"
    assert np.any(info["ordinary"][~mask] != info["glyph"][~mask]), "full-tile copying must matter"
    expected = _copy_at(grid, info, active-clues)
    assert np.array_equal(output, expected), "incorrect whole-tile completion"
    assert np.array_equal(solve(grid), expected)
    missing = active-clues
    r0, c0 = origins[0]
    h, w = mask.shape
    infinite = _region(rails.shape, info["strokes"], info["slope"], rails, "infinite")
    pixel = _region(rails.shape, info["strokes"], info["slope"], rails, "pixel_centers")
    consecutive = _pair_region(rails.shape, info["strokes"], info["slope"], rails, consecutive=True)
    info.update(origin=origins[0], enclosure=enclosure, active=active, missing=missing,
                finite_witness=len(_placements(mask, rails.shape, clues, infinite)) > 1,
                pixel_witness=any(not pixel[p] for p in missing),
                skip_witness=any(not consecutive[p] for p in missing),
                rectangle_witness=bool(np.any(~mask & ~rails[r0:r0+h, c0:c0+w])),
                window_witness=not bool(enclosure[r0:r0+h, c0:c0+w].all()),
                no_a_candidates=len(_placements(mask, rails.shape, clues, ~rails)),
                top_left_wrong=origins[0] != _placements(mask, rails.shape, clues, ~rails)[0],
                bottom_right_wrong=origins[0] != _placements(mask, rails.shape, clues, ~rails)[-1])
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _key(pair["input"])
            assert key not in seen, "duplicate inputs"
            seen.add(key)
            if split == "train":
                train.append(info)
    assert len({_key(i["mask"]) for i in train}) >= 2
    assert len({_normal(i["ordinary"]) for i in train}) >= 2
    assert {i["slope"] for i in train} == {-1, 1}
    assert any(_asymmetric(i["mask"]) for i in train), "missing orientation evidence"
    for witness in ("finite_witness", "pixel_witness", "skip_witness", "rectangle_witness", "window_witness"):
        assert any(i[witness] for i in train), f"missing {witness} training evidence"
    assert any(i["top_left_wrong"] for i in train) and any(i["bottom_right_wrong"] for i in train)


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming audit: 42 seeds, no stored grids or execution traces."""
    fingerprints, shapes, masks, motifs, layouts, palettes = set(), set(), set(), set(), set(), set()
    slopes, tile_shapes, thicknesses, stroke_counts, clue_counts, ink_counts = set(), set(), set(), set(), set(), set()
    metrics = dict(puzzles=42, pairs=0, finite_witnesses=0, pixel_witnesses=0,
                   skip_witnesses=0, rectangle_witnesses=0, window_witnesses=0,
                   multicolor_rails=0, ternary_ordinary=0,
                   edge_rails=0, interior_rails=0, masks_with_inactive_cells=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        fingerprints.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _analyze_pair(pair)
                assert len(f["clues"]) == 1 if split == "test" else len(f["clues"]) in (2, 3)
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"]) and np.array_equal(before, pair["input"])
                assert np.array_equal(solve(before.tolist()), pair["output"])
                h, w, d, nr, nc = f["geometry"]
                shapes.add(pair["input"].shape)
                masks.add(_key(f["mask"]))
                motifs.add(_normal(f["ordinary"]))
                layouts.add((f["rails"].shape, f["rails"].tobytes(), f["origin"], tuple(sorted(f["clues"]))))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                slopes.add(f["slope"])
                tile_shapes.add((h, w))
                thicknesses.add(d)
                stroke_counts.add(len(f["strokes"]))
                clue_counts.add(len(f["clues"]))
                ink_counts.add(len(f["active"]))
                metrics["pairs"] += 1
                for name in ("finite", "pixel", "skip", "rectangle", "window"):
                    metrics[name+"_witnesses"] += int(f[name+"_witness"])
                metrics["multicolor_rails"] += int(len(set(f["rail_colors"].values())) > 1)
                metrics["ternary_ordinary"] += int(len(np.unique(f["ordinary"])) == 3)
                edge = any(r in (0, nr-1) or c in (0, nc-1) for r, c in f["rail_colors"])
                metrics["edge_rails"] += int(edge)
                metrics["interior_rails"] += int(not edge)
                # An inactive ink pixel inside the ink bounding box distinguishes a rectangle.
                metrics["masks_with_inactive_cells"] += int(not f["mask"].all())
    assert len(fingerprints) == 42 and len(layouts) >= 150
    assert len(shapes) >= 15 and len(masks) >= 25 and len(motifs) >= 50 and len(palettes) >= 25
    assert slopes == {-1, 1} and tile_shapes == set(_TILE_SHAPES) and thicknesses == {1, 2}
    assert {2, 3, 4} <= stroke_counts and {1, 2, 3} <= clue_counts and len(ink_counts) >= 3
    assert metrics["multicolor_rails"] > 0 and metrics["ternary_ordinary"] > 0
    assert metrics["interior_rails"] > 0
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(8241, nt, ne)
        assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicitly exercise both finite fallback scenes and both slopes.
    for fallback_index, fallback in enumerate(_FALLBACKS):
        points, rows, origin, clues = fallback
        ra = np.zeros((7, 7), dtype=bool)
        ra[tuple(zip(*points))] = True
        ma = np.array([[int(v) for v in row] for row in rows], dtype=bool)
        for slope in (-1, 1):
            scene = (ra, ma, origin, set(clues))
            if slope == -1:
                scene = _reflect_scene(scene)
            pair = _assemble(np.random.default_rng(64), (3, 3, 1, 7, 7), scene,
                             (0, 1, (2, 3), (4, 5)), code=26)
            info = _analyze_pair(pair)
            assert info["skip_witness" if fallback_index else "finite_witness"]
    # Reject wrong labels, including an ink-only copy that omits background changes.
    p = generate_puzzle(0, 2, 1)
    pair = p["test"][0]
    f = _analyze_pair(pair)
    h, w, d, _, _ = f["geometry"]
    r, c = next(iter(f["missing"]))
    dy, dx = np.argwhere((f["ordinary"] != f["glyph"]) & ~f["mask"])[0]
    y, x = d+r*(h+d)+int(dy), d+c*(w+d)+int(dx)
    pair["output"][y, x] = pair["input"][y, x]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("ink-only copying accepted")
    metrics.update(distinct_puzzles=len(fingerprints), canvas_shapes=len(shapes),
                   ink_masks=len(masks), ordinary_motifs=len(motifs), coarse_layouts=len(layouts),
                   palettes=len(palettes), tile_shapes=sorted(tile_shapes),
                   thicknesses=sorted(thicknesses), stroke_counts=sorted(stroke_counts),
                   clue_counts=sorted(clue_counts), ink_counts=sorted(ink_counts))
    return metrics
