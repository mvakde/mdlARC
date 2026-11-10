"""Rooted stencil mosaics: identity alignment, binary exchange, geodesic phase.

All decoding conventions are visible or fixed by the family. No parent generator
is imported. Public solve takes only a single input grid.
"""
from collections import deque
import itertools
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
CW = np.array([[0, 1], [-1, 0]], dtype=int)
IDENTITY = np.eye(2, dtype=int)
NEIGHBORS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _check(bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, component = deque([start]), []
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr, dc in NEIGHBORS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        result.append(component)
    return result


def _distances(component, root):
    allowed = set(component)
    distances = {root: 0}
    queue = deque([root])
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS:
            p = (r + dr, c + dc)
            if p in allowed and p not in distances:
                distances[p] = distances[(r, c)] + 1
                queue.append(p)
    return distances


def _root_info(stencil, background):
    values = [int(v) for v in np.unique(stencil) if v != background]
    _check(len(values) == 2 and np.any(stencil == background), "stencil needs holes and two ink roles")
    components = _components(stencil != background)
    _check(1 <= len(components) <= 4 and all(len(c) >= 3 for c in components), "invalid rooted components")
    candidates = [v for v in values if all(sum(stencil[p] == v for p in c) == 1 for c in components)]
    _check(len(candidates) == 1, "each component must have one uniquely identifiable root")
    root_color = candidates[0]
    roots, distance = [], np.full(stencil.shape, -1, dtype=int)
    for component in components:
        root = next(p for p in component if stencil[p] == root_color)
        roots.append(root)
        for p, d in _distances(component, root).items():
            distance[p] = d
    return root_color, components, roots, distance


def _bar_description(component, color):
    rr = [p[0] for p in component]
    cc = [p[1] for p in component]
    if len(component) < 2:
        return None
    if min(rr) == max(rr) and max(cc) - min(cc) + 1 == len(component):
        return ("H", min(rr), min(cc), max(cc), int(color))
    if min(cc) == max(cc) and max(rr) - min(rr) + 1 == len(component):
        return ("V", min(cc), min(rr), max(rr), int(color))
    return None


def _rectangle(horizontal, vertical):
    _, hr, hc0, hc1, hcolor = horizontal
    _, vc, vr0, vr1, vcolor = vertical
    row, col, height, width = vr0, hc0, vr1 - vr0 + 1, hc1 - hc0 + 1
    if hr == row - 1:
        hs = "T"
    elif hr == row + height:
        hs = "B"
    else:
        return None
    if vc == col - 1:
        vs = "L"
    elif vc == col + width:
        vs = "R"
    else:
        return None
    return (row, col, height, width), {hs: hcolor, vs: vcolor}


def _paint_bar(grid, bounds, side, color):
    r, c, h, w = bounds
    if side == "T":
        grid[r - 1, c:c + w] = color
    elif side == "B":
        grid[r + h, c:c + w] = color
    elif side == "L":
        grid[r:r + h, c - 1] = color
    else:
        grid[r:r + h, c + w] = color


def _matching_matrix(source, target):
    matrix = np.zeros((2, 2), dtype=int)
    for side, color in source.items():
        matching_side = next(s for s, v in target.items() if v == color)
        matrix += np.outer(NORMALS[matching_side], NORMALS[side])
    _check(np.array_equal(matrix.T @ matrix, IDENTITY), "guides do not specify a symmetry")
    return matrix


def _decode_input(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    _check(all(np.all(edge == background) for edge in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "objects need a clear outer margin")
    # A guide identity occurs in exactly two isolated complete straight bars.
    bar_colors = {}
    for color in np.unique(grid):
        if color == background:
            continue
        components = _components(grid == color)
        bars = [_bar_description(c, color) for c in components]
        if len(bars) == 2 and all(b is not None for b in bars):
            bar_colors[int(color)] = bars
    matches = []
    for marker_pair in itertools.combinations(sorted(bar_colors), 2):
        bars = bar_colors[marker_pair[0]] + bar_colors[marker_pair[1]]
        rectangles = []
        for hi, horizontal in enumerate(bars):
            if horizontal[0] != "H":
                continue
            for vi, vertical in enumerate(bars):
                if vertical[0] != "V" or horizontal[-1] == vertical[-1]:
                    continue
                recovered = _rectangle(horizontal, vertical)
                if recovered is None:
                    continue
                bounds, guides = recovered
                r, c, h, w = bounds
                if min(r, c) < 2 or r + h > grid.shape[0] - 2 or c + w > grid.shape[1] - 2:
                    continue
                payload = grid[r:r + h, c:c + w].copy()
                if set(np.unique(payload)) & set(marker_pair):
                    continue
                rectangles.append((bounds, guides, payload, frozenset((hi, vi))))
        for source in rectangles:
            sb, sg, tile, si = source
            n = tile.shape[0]
            if tile.shape != (n, n) or not 2 <= n <= 5 or len(np.unique(tile)) != 2 or background in tile:
                continue
            for target in rectangles:
                tb, tg, stencil, ti = target
                if si & ti or si | ti != frozenset(range(4)):
                    continue
                if min(stencil.shape) < 3 or max(np.array(stencil.shape) * n) > 30:
                    continue
                try:
                    root_color, components, roots, distance = _root_info(stencil, background)
                except ValueError:
                    continue
                # The complete bounding boxes, including unused guide sides,
                # are disjoint. Reconstruct every pixel to exclude extra ink.
                sr, sc, sh, sw = sb
                tr, tc, th, tw = tb
                disjoint = (sr + sh + 1 <= tr - 1 or tr + th + 1 <= sr - 1 or
                            sc + sw + 1 <= tc - 1 or tc + tw + 1 <= sc - 1)
                if not disjoint:
                    continue
                expected = np.full(grid.shape, background, int)
                for bounds, guides, payload, _ in (source, target):
                    r, c, h, w = bounds
                    expected[r:r + h, c:c + w] = payload
                    for side, color in guides.items():
                        _paint_bar(expected, bounds, side, color)
                if not np.array_equal(grid, expected):
                    continue
                matches.append(dict(tile=tile, stencil=stencil, background=background,
                                    source_guides=sg, target_guides=tg, source_bounds=sb,
                                    target_bounds=tb, root_color=root_color, roots=roots,
                                    components=components, distance=distance,
                                    matrix=_matching_matrix(sg, tg)))
    _check(len(matches) == 1, "input must have exactly one template/stencil interpretation")
    return matches[0]


def _coordinate_transform(tile, matrix):
    coords = np.indices(tile.shape).reshape(2, -1)
    target = matrix @ coords
    target -= target.min(axis=1, keepdims=True)
    result = np.empty(tuple(target.max(axis=1) + 1), dtype=int)
    result[target[0], target[1]] = tile.ravel()
    return result


def _predict(info, behavior="rule"):
    tile, stencil, bg = info["tile"], info["stencil"], info["background"]
    colors = np.unique(tile)
    tile = tile.copy() if behavior == "no_exchange" else int(colors.sum()) - tile
    matrix = info["matrix"]
    if behavior == "identity_base":
        matrix = IDENTITY
    elif behavior == "inverse_base":
        matrix = matrix.T
    elif behavior == "swapped_match":
        guides = info["target_guides"]
        vals = list(guides.values())
        matrix = _matching_matrix(info["source_guides"], {s: vals[1 - vals.index(v)] for s, v in guides.items()})
    base = _coordinate_transform(tile, matrix)
    turns = [np.rot90(base, -d).copy() for d in range(4)]
    if behavior in ("phase_before_match", "whole_mosaic"):
        turns = [_coordinate_transform(np.rot90(tile, -d), matrix if behavior == "phase_before_match" else IDENTITY)
                 for d in range(4)]
    n, (h, w) = len(tile), stencil.shape
    out = np.full((n * h, n * w), bg, int)
    for component, root in zip(info["components"], info["roots"]):
        for r, c in component:
            d = int(info["distance"][r, c])
            if behavior == "uniform_phase":
                d = 0
            elif behavior == "manhattan":
                d = abs(r - root[0]) + abs(c - root[1])
            elif behavior == "ccw_phase":
                d = -d
            out[r * n:(r + 1) * n, c * n:(c + 1) * n] = turns[d % 4]
    if behavior == "fill_holes":
        for r, c in np.argwhere(stencil == bg):
            out[r * n:(r + 1) * n, c * n:(c + 1) * n] = base
    if behavior == "whole_mosaic":
        out = _coordinate_transform(out, matrix)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from the two visible guide-defined objects."""
    return _predict(_decode_input(input_grid))


def _numpy_orientation(tile, matrix):
    # Forward generation uses array operations; solve uses signed coordinates.
    operations = ((IDENTITY, lambda a: a),
                  (CW, lambda a: np.rot90(a, -1)),
                  (-IDENTITY, lambda a: np.rot90(a, 2)),
                  (-CW, lambda a: np.rot90(a, 1)),
                  (np.diag([1, -1]), np.fliplr),
                  (np.diag([-1, 1]), np.flipud),
                  (np.array([[0, 1], [1, 0]]), lambda a: a.T),
                  (np.array([[0, -1], [-1, 0]]), lambda a: np.rot90(a.T, 2)))
    return next(f(tile).copy() for m, f in operations if np.array_equal(m, matrix))


def _construct_output(tile, stencil, bg, source_guides, target_guides):
    _, components, roots, _ = _root_info(stencil, bg)
    colors = np.unique(tile)
    base = _numpy_orientation(int(colors.sum()) - tile, _matching_matrix(source_guides, target_guides))
    n, (h, w) = len(tile), stencil.shape
    out = np.full((h * n, w * n), bg, int)
    for component, root in zip(components, roots):
        for (r, c), d in _distances(component, root).items():
            out[r * n:(r + 1) * n, c * n:(c + 1) * n] = np.rot90(base, -d)
    return out


def _adjacent(index):
    return SIDES[index], SIDES[(index + 1) % 4]


def _corner(guides):
    return next(i for i in range(4) if set(_adjacent(i)) == set(guides))


def _moved_guides(guides, matrix):
    return {next(s for s in SIDES if tuple(matrix @ NORMALS[side]) == NORMALS[s]): color
            for side, color in guides.items()}


def _sample_tile(rng, n, colors, witness=False):
    for _ in range(128):
        bits = rng.random((n, n)) < rng.uniform(.22, .78)
        if not bits.any() or bits.all():
            continue
        if witness:
            matrices = [np.linalg.matrix_power(CW, d) @ m for d in range(4)
                        for m in (IDENTITY, np.diag([1, -1]))]
            variants = [_key(_numpy_orientation(bits, m)) for m in matrices]
            if len(set(variants)) != 8 or bits.sum() * 2 == bits.size:
                continue
        return np.where(bits, colors[0], colors[1]).astype(int)
    # A fully asymmetric binary 3x3, with no color-swapping symmetry.
    bits = np.zeros((n, n), bool)
    if n == 2:
        bits[0, 1] = True
    else:
        bits[0, 0] = bits[0, 1] = bits[1, 0] = bits[-1, 1] = True
    return np.where(bits, colors[0], colors[1]).astype(int)


def _raw_mask(rng, h, w, density):
    style = int(rng.integers(6))
    mask = np.zeros((h, w), bool)
    if style == 0:
        mask = rng.random((h, w)) < density
    elif style == 1:  # Wandering strokes can branch and intersect.
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(int(rng.integers(5, 2 * h * w))):
                mask[r, c] = True
                dr, dc = NEIGHBORS[int(rng.integers(4))]
                r, c = int(np.clip(r + dr, 0, h - 1)), int(np.clip(c + dc, 0, w - 1))
    elif style == 2:  # A maze has genuine geodesic detours, including loops.
        mask[::2, ::2] = True
        for r in range(h):
            for c in range(w):
                if (r % 2) != (c % 2) and rng.random() < .62:
                    mask[r, c] = True
    elif style == 3:  # Irregular bars and isolated branches.
        horizontal = bool(rng.integers(2))
        for i in range(0, h if horizontal else w, 2):
            lo, hi = sorted(map(int, rng.integers(0, w if horizontal else h, size=2)))
            if horizontal:
                mask[i, lo:hi + 1] = True
            else:
                mask[lo:hi + 1, i] = True
        mask |= rng.random((h, w)) < .08
    elif style == 4:  # Partial rings, solid regions, and holes.
        for _ in range(int(rng.integers(1, 4))):
            r0, r1 = sorted(map(int, rng.integers(h, size=2)))
            c0, c1 = sorted(map(int, rng.integers(w, size=2)))
            mask[r0:r1 + 1, c0:c1 + 1] = True
            if r1 - r0 >= 2 and c1 - c0 >= 2:
                mask[r0 + 1:r1, c0 + 1:c1] = False
    else:  # A continuous meander through alternating rows.
        for r in range(0, h, 2):
            mask[r, :] = True
            if r + 1 < h:
                mask[r + 1, -1 if (r // 2) % 2 == 0 else 0] = True
        mask &= rng.random((h, w)) > .04
    return mask


def _sample_stencil(rng, n, bg, ink, root_color, density, witness=False):
    maximum = 30 // n
    for _ in range(220):
        low = 6 if witness else 3
        h, w = map(int, rng.integers(low, maximum + 1, size=2))
        mask = _raw_mask(rng, h, w, density)
        if witness or rng.random() < .45:
            side = str(rng.choice(SIDES))
            if side == "T":
                mask[0] = False
            elif side == "B":
                mask[-1] = False
            elif side == "L":
                mask[:, 0] = False
            else:
                mask[:, -1] = False
        components = [c for c in _components(mask) if len(c) >= 3]
        if len(components) > 4:
            chosen = rng.choice(len(components), 4, replace=False)
            components = [components[int(i)] for i in chosen]
        if not components or (witness and len(components) < 2):
            continue
        roots, detour, far = [], False, False
        for component in components:
            root = component[int(rng.integers(len(component)))]
            distance = _distances(component, root)
            roots.append(root)
            detour |= any((d - abs(p[0] - root[0]) - abs(p[1] - root[1])) % 4 != 0 for p, d in distance.items())
            far |= max(distance.values()) >= 4
        clean = np.full((h, w), bg, int)
        for component, root in zip(components, roots):
            for p in component:
                clean[p] = ink
            clean[root] = root_color
        if np.all(clean != bg) or (witness and not (detour and far)):
            continue
        return clean
    # Conservative winding witness. Random symmetry and padding position still
    # vary its geometry; this fallback is not the normal family distribution.
    h = w = 6 if witness else max(3, min(6, maximum))
    clean = np.full((h, w), bg, int)
    if h >= 6:
        clean[:4, 0] = clean[:4, 2] = ink
        clean[3, :3] = ink
        clean[5, :3] = ink
        clean[0, 2] = clean[5, 1] = root_color
    else:
        clean[1, :] = ink
        clean[1, 1] = root_color
    return np.rot90(clean, int(rng.integers(4))).copy()


def _palette(rng):
    colors = list(map(int, rng.permutation(10)))
    bg, ma, mb, ta, tb = colors[:5]
    ink = int(rng.choice((ta, tb))) if rng.random() < .5 else colors[5]
    root_options = [c for c in (ta, tb, colors[6]) if c != ink]
    root = int(rng.choice(root_options))
    return bg, (ma, mb), (ta, tb), ink, root


def _pack(rng, tile, stencil, bg, source_guides, target_guides, preferred):
    objects = [(tile, source_guides), (stencil, target_guides)]
    if rng.integers(2):
        objects.reverse()
    horizontal = (preferred == "horizontal") if rng.random() < .65 else bool(rng.integers(2))
    gap = int(rng.integers(1, 5))
    heights = [a.shape[0] + 2 for a, _ in objects]
    widths = [a.shape[1] + 2 for a, _ in objects]
    min_h = max(heights) + 2 if horizontal else sum(heights) + gap + 2
    min_w = sum(widths) + gap + 2 if horizontal else max(widths) + 2
    H = int(rng.integers(min_h, min(30, min_h + 6) + 1))
    W = int(rng.integers(min_w, min(30, min_w + 6) + 1))
    grid = np.full((H, W), bg, int)
    if horizontal:
        left = int(rng.integers(1, W - sum(widths) - gap))
        boxes = [(int(rng.integers(1, H - h)), c)
                 for h, c in zip(heights, (left, left + widths[0] + gap))]
    else:
        top = int(rng.integers(1, H - sum(heights) - gap))
        boxes = [(r, int(rng.integers(1, W - w)))
                 for w, r in zip(widths, (top, top + heights[0] + gap))]
    for (a, guides), (r, c) in zip(objects, boxes):
        bounds = r + 1, c + 1, *a.shape
        grid[r + 1:r + 1 + a.shape[0], c + 1:c + 1 + a.shape[1]] = a
        for side, color in guides.items():
            _paint_bar(grid, bounds, side, color)
    return grid


def _make_pair(rng, params, n, source_corner, target_corner, witness=False, quarter=None):
    bg, markers, tile_colors, ink, root = _palette(rng)
    tile = _sample_tile(rng, n, tile_colors, witness)
    stencil = _sample_stencil(rng, n, bg, ink, root, params["density"], witness)
    source_guides = dict(zip(_adjacent(source_corner), markers))
    if quarter is not None:
        target_guides = _moved_guides(source_guides, np.linalg.matrix_power(CW, quarter))
    else:
        order = markers if rng.integers(2) else markers[::-1]
        target_guides = dict(zip(_adjacent(target_corner), order))
    inp = _pack(rng, tile, stencil, bg, source_guides, target_guides, params["layout"])
    out = _construct_output(tile, stencil, bg, source_guides, target_guides)
    return {"input": inp, "output": out}


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    params = dict(density=float(rng.uniform(.35, .65)), layout=str(rng.choice(("horizontal", "vertical"))))
    n = int(rng.choice((3, 4)))
    source_corner, quarter = int(rng.integers(4)), int(rng.choice((1, 3)))
    first = _make_pair(rng, params, n, source_corner, 0, witness=True, quarter=quarter)
    info = _decode_input(first["input"])
    first_corner = _corner(info["target_guides"])
    held_corner = int(rng.choice([i for i in range(4) if i != first_corner]))
    second_input = first["input"].copy()
    vals = list(info["target_guides"].values())
    second_guides = {s: vals[1 - vals.index(v)] for s, v in info["target_guides"].items()}
    for side, color in second_guides.items():
        _paint_bar(second_input, info["target_bounds"], side, color)
    second_output = _construct_output(info["tile"], info["stencil"], info["background"],
                                      info["source_guides"], second_guides)
    puzzle = {"train": [first, {"input": second_input, "output": second_output}], "test": []}
    seen = {_key(first["input"]), _key(second_input)}
    for split, count in (("train", nt - 2), ("test", nq)):
        for i in range(count):
            for _ in range(32):
                tile_size = int(rng.choice([v for v in range(2, 6) if split != "test" or i != 0 or v != n]))
                src = int(rng.integers(4))
                target = held_corner if split == "test" else int(rng.choice([j for j in range(4) if j != held_corner]))
                pair = _make_pair(rng, params, tile_size, src, target)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


ALTERNATIVES = ("no_exchange", "uniform_phase", "manhattan", "ccw_phase", "identity_base",
                "inverse_base", "swapped_match", "fill_holes", "phase_before_match", "whole_mosaic")


def validate_generated_puzzle(puzzle):
    """Check visible extraction, local cell semantics, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode_input(inp)
            infos[split].append(info)
            tile, stencil, bg = info["tile"], info["stencil"], info["background"]
            n = len(tile)
            assert out.shape == tuple(np.array(stencil.shape) * n), "wrong expanded extent"
            assert np.array_equal(solve(inp), out), "incorrect output"
            vals, counts = np.unique(tile, return_counts=True)
            for r in range(stencil.shape[0]):
                for c in range(stencil.shape[1]):
                    block = out[r * n:(r + 1) * n, c * n:(c + 1) * n]
                    if stencil[r, c] == bg:
                        assert np.all(block == bg), "transparent hole was painted"
                        continue
                    assert np.array_equal(np.unique(block), vals)
                    assert [np.count_nonzero(block == v) for v in vals] == list(counts[::-1]), "binary exchange histogram"
                    d = int(info["distance"][r, c])
                    matrix = np.linalg.matrix_power(CW, d % 4) @ info["matrix"]
                    corners = np.array([[0, 0], [0, n - 1], [n - 1, 0], [n - 1, n - 1]]).T
                    offset = (matrix @ corners).min(axis=1)
                    # Forward cell correspondence, separate from solve's block transform.
                    for y in range(n):
                        for x in range(n):
                            yy, xx = matrix @ (y, x) - offset
                            assert block[yy, xx] == int(vals.sum()) - tile[y, x], "wrong geodesic orientation"
            assert set(np.unique(out)) == {bg, *map(int, vals)}
    train = infos["train"]
    for behavior in ALTERNATIVES:
        assert any(not np.array_equal(_predict(info, behavior), np.asarray(pair["output"]))
                   for info, pair in zip(train, puzzle["train"])), f"training does not defeat {behavior}"
    assert any(np.array_equal(t["matrix"], CW) or np.array_equal(t["matrix"], -CW) for t in train)
    assert any(round(np.linalg.det(t["matrix"])) == -1 for t in train)
    assert any(len(t["components"]) >= 2 and t["distance"].max() >= 4 for t in train)
    assert any(any(np.all(edge == t["background"]) for edge in
                   (t["stencil"][0], t["stencil"][-1], t["stencil"][:, 0], t["stencil"][:, -1])) for t in train)
    a, b = train[:2]
    assert np.array_equal(a["tile"], b["tile"]) and np.array_equal(a["stencil"], b["stencil"])
    assert a["source_guides"] == b["source_guides"] and set(a["target_guides"]) == set(b["target_guides"])
    assert a["target_guides"] != b["target_guides"], "missing identity contrast"
    assert not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"])
    training_corners = {_corner(t["target_guides"]) for t in train}
    assert all(_corner(t["target_guides"]) not in training_corners for t in infos["test"]), "missing side-arrangement shift"


def _features(info):
    distances = info["distance"]
    degrees = [sum((r + dr, c + dc) in set(component) for dr, dc in NEIGHBORS)
               for component in info["components"] for r, c in component]
    edges = sum(degrees) // 2
    cells = sum(map(len, info["components"]))
    return dict(tile_size=len(info["tile"]), stencil_shape=tuple(info["stencil"].shape),
                components=len(info["components"]), max_distance=int(distances.max()),
                branches=sum(d >= 3 for d in degrees), cycles=edges - cells + len(info["components"]),
                symmetry=tuple(info["matrix"].ravel()), source_corner=_corner(info["source_guides"]),
                target_corner=_corner(info["target_guides"]))


def self_test():
    """Bounded 42-seed checks; statistics concern returned pixels, not parameters."""
    signatures, shapes, normalized, features, palettes = set(), set(), set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((_key(pair["input"]), _key(pair["output"]))
                          for split in ("train", "test") for pair in puzzle[split])
        signatures.add(signature)
        if seed < 3:
            repeated = generate_puzzle(seed)
            assert signature == tuple((_key(p["input"]), _key(p["output"]))
                                      for s in ("train", "test") for p in repeated[s])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode_input(pair["input"])
                f = _features(info)
                f["split"] = split
                features.append(f)
                shapes.add(pair["input"].shape)
                palettes.add(tuple(np.unique(pair["input"])))
                roles = {}
                normalized.add((info["stencil"].shape, tuple(roles.setdefault(int(v), len(roles)) for v in info["stencil"].ravel())))
    assert len(signatures) == 42 and len(shapes) >= 35 and len(normalized) >= 90
    assert {f["tile_size"] for f in features} == {2, 3, 4, 5}
    assert {f["components"] for f in features} >= {1, 2, 3, 4}
    assert len({f["stencil_shape"] for f in features}) >= 40
    assert max(f["max_distance"] for f in features) >= 15
    assert any(f["branches"] >= 3 for f in features) and any(f["cycles"] >= 2 for f in features)
    assert len({f["symmetry"] for f in features}) == 8
    assert len({f["source_corner"] for f in features}) == len({f["target_corner"] for f in features}) == 4
    assert len(palettes) >= 20
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(4200 + 17 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    original = generate_puzzle(0)
    for kind in ("output", "guide", "root", "extent"):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]] for s in original}
        pair = damaged["test"][0]
        info = _decode_input(pair["input"])
        if kind == "output":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif kind == "extent":
            pair["output"] = pair["output"][1:].copy()
        elif kind == "root":
            r, c, _, _ = info["target_bounds"]
            rr, cc = info["roots"][0]
            ink = next(v for v in np.unique(info["stencil"]) if v not in (info["background"], info["root_color"]))
            pair["input"][r + rr, c + cc] = ink
        else:
            r, c, h, w = info["source_bounds"]
            side = next(iter(info["source_guides"]))
            p = {"T": (r - 1, c), "B": (r + h, c), "L": (r, c - 1), "R": (r, c + w)}[side]
            pair["input"][p] = info["background"]
        try:
            validate_generated_puzzle(damaged)
        except (ValueError, AssertionError):
            continue
        raise AssertionError(f"{kind} corruption accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                normalized_stencils=len(normalized), stencil_shapes=len({f["stencil_shape"] for f in features}),
                symmetries=len({f["symmetry"] for f in features}), max_distance=max(f["max_distance"] for f in features),
                branched_pairs=sum(f["branches"] > 0 for f in features), cyclic_pairs=sum(f["cycles"] > 0 for f in features))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
