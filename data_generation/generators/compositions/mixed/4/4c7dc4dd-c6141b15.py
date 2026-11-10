"""Reference-selected reciprocal arrangement exchange. No parent imports or I/O.

The complete pair supplies the anchor topology, including a midpoint that a
drawn line alone cannot reveal. Panel locations never supply transformation
direction. Construction uses coordinates; solve reads only returned pixels.
"""
from itertools import combinations
import hashlib

import numpy as np


_MOTIFS = (
    frozenset({(0, 0)}),
    frozenset({(0, -1), (0, 0), (0, 1)}),
    frozenset({(-1, 0), (0, 0), (1, 0)}),
    frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)}),
    frozenset({(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)}),
    frozenset({(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)}),
    frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1)}),
)
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b) or not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 train, 1..4 test, train > test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer) or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grid must be a 1..30 by 1..30 integer array of colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _components(cells, diagonal=True):
    unseen = set(cells)
    result = []
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(frozenset(component))
    return sorted(result, key=min)


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = max(abs(dr), abs(dc))
    if not n or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise ValueError("distinct endpoints must be horizontal, vertical or 45 degree aligned")
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return frozenset((a[0] + i * sr, a[1] + i * sc) for i in range(n + 1))


def _figure(centers):
    return frozenset().union(*(_line(a, b) for a, b in combinations(centers, 2)))


def _stamps(centers, motif):
    return frozenset((r + dr, c + dc) for r, c in centers for dr, dc in motif)


def _apart(a, b):
    # At least one background cell even diagonally: both states parse cleanly.
    return not any((r + dr, c + dc) in b for r, c in a
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _hull(cells):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    points = sorted(cells)
    halves = []
    for sequence in (points, points[::-1]):
        half = []
        for p in sequence:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.append(half[:-1])
    return tuple(sorted(set(halves[0] + halves[1])))


def _center_kind(centers):
    centers = tuple(sorted(centers))
    _figure(centers)  # Checks all edge slopes, even for a collinear triple.
    if len(centers) == 2:
        return "pair"
    if len(centers) != 3:
        raise ValueError("only two or three glyph centers are supported")
    a, m, b = centers
    if (2 * m[0], 2 * m[1]) == (a[0] + b[0], a[1] + b[1]):
        return "midpoint"
    if len(_hull(centers)) != 3:
        raise ValueError("a collinear triple must be equally spaced")
    distances = sorted((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                       for p, q in combinations(centers, 2))
    if distances[0] != distances[1] or sum(distances[:2]) != distances[2]:
        raise ValueError("three noncollinear centers must form a right isosceles triangle")
    return "triangle"


def _edge_info(cells):
    points = sorted(cells)
    if len(points) < 4:
        raise ValueError("edges must have at least four cells")
    try:
        if _line(points[0], points[-1]) == cells:
            return "line", (points[0], points[-1])
    except ValueError:
        pass
    corners = _hull(cells)
    if len(corners) == 3 and _center_kind(corners) == "triangle" and _figure(corners) == cells:
        return "triangle", corners
    raise ValueError("edge role is not an exact segment or triangle perimeter")


def _glyph_info(cells):
    components = _components(cells)
    if len(components) not in (2, 3):
        raise ValueError("glyph role needs two or three separate copies")
    centers, motifs = [], []
    for component in components:
        rows, cols = zip(*component)
        if (min(rows) + max(rows)) % 2 or (min(cols) + max(cols)) % 2:
            raise ValueError("glyph bounding boxes must be odd in both axes")
        center = ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2)
        motif = frozenset((r - center[0], c - center[1]) for r, c in component)
        if (0, 0) not in motif:
            raise ValueError("glyph center must be occupied")
        centers.append(center)
        motifs.append(motif)
    if len(set(motifs)) != 1:
        raise ValueError("glyphs must be exact translated copies")
    centers = tuple(sorted(centers))
    return centers, motifs[0], _center_kind(centers)


def _analyze(panel, bg):
    colors = set(map(int, np.unique(panel))) - {bg}
    if len(colors) != 2:
        raise ValueError("occupied panel must have exactly two foreground roles")
    candidates = []
    for glyph_color in sorted(colors):
        edge_color = next(iter(colors - {glyph_color}))
        glyphs = frozenset(map(tuple, np.argwhere(panel == glyph_color).tolist()))
        edges = frozenset(map(tuple, np.argwhere(panel == edge_color).tolist()))
        try:
            centers, motif, topology = _glyph_info(glyphs)
            edge_kind, anchors = _edge_info(edges)
            if not _apart(glyphs, edges):
                continue
            candidates.append(dict(bg=bg, edge=edge_color, glyph=glyph_color,
                                   centers=centers, motif=motif, topology=topology,
                                   edge_kind=edge_kind, anchors=anchors, shape=panel.shape))
        except ValueError:
            continue
    if len(candidates) != 1:
        raise ValueError("panel foreground roles are ambiguous or invalid")
    return candidates[0]


def _paint(shape, bg, edge, glyph, edge_cells, centers, motif):
    glyph_cells = _stamps(centers, motif)
    if not _apart(edge_cells, glyph_cells):
        raise ValueError("roles collide or touch")
    if len(_components(glyph_cells)) != len(centers):
        raise ValueError("glyph copies touch")
    if any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in edge_cells | glyph_cells):
        raise ValueError("structure would clip at the canvas boundary")
    result = np.full(shape, bg, dtype=np.int8)
    for r, c in edge_cells:
        result[r, c] = edge
    for r, c in glyph_cells:
        result[r, c] = glyph
    return result


def _forward(a):
    if a["edge_kind"] != "line":
        raise ValueError("forward exchange starts with a single segment")
    return _paint(a["shape"], a["bg"], a["edge"], a["glyph"],
                  _figure(a["centers"]), a["anchors"], a["motif"])


def _infer(reference, bg):
    analyses = [_analyze(p, bg) for p in reference]
    if any(analyses[0][k] != analyses[1][k] for k in ("edge", "glyph", "motif", "shape")):
        raise ValueError("reference must preserve its palette roles and glyph shape")
    models = set()
    for i in range(2):
        try:
            if np.array_equal(_forward(analyses[i]), reference[1 - i]):
                models.add(analyses[i]["topology"])
        except ValueError:
            pass
    if len(models) != 1:
        raise ValueError("complete pair does not uniquely demonstrate an exchange topology")
    return next(iter(models)), analyses


def _predict(a, model):
    if a["topology"] == model and a["edge_kind"] == "line":
        return _forward(a)
    if a["topology"] != "pair":
        raise ValueError("occupied target does not correspond to either reference state")
    if model == "midpoint" and a["edge_kind"] == "line":
        p, q = a["anchors"]
        if (p[0] + q[0]) % 2 or (p[1] + q[1]) % 2:
            raise ValueError("midpoint does not fall on a cell; no rounding is allowed")
        middle = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
        anchors = (p, middle, q)
    elif model == "triangle" and a["edge_kind"] == "triangle":
        anchors = a["anchors"]
    else:
        raise ValueError("target edge topology does not match the demonstrated relation")
    return _paint(a["shape"], a["bg"], a["edge"], a["glyph"],
                  _figure(a["centers"]), anchors, a["motif"])


def _extract(grid):
    # Frames use dedicated colors; identify actual connected outlines, not
    # an expected location, a literal color, or a guessed most common color.
    panels = []
    for color in map(int, np.unique(grid)):
        cells = frozenset(map(tuple, np.argwhere(grid == color).tolist()))
        for component in _components(cells, diagonal=False):
            rows, cols = zip(*component)
            r, c, bottom, right = min(rows), min(cols), max(rows), max(cols)
            h, w = bottom - r + 1, right - c + 1
            if h < 7 or w < 7:
                continue
            perimeter = frozenset((y, x) for y in range(r, bottom + 1)
                                  for x in range(c, right + 1)
                                  if y in (r, bottom) or x in (c, right))
            if component == perimeter:
                panels.append((color, r, c, h, w))
    if len(panels) != 4 or len({p[3:] for p in panels}) != 1:
        raise ValueError("need exactly four equally sized framed panels")
    groups = {}
    outside = np.ones(grid.shape, bool)
    for p in panels:
        color, r, c, h, w = p
        if not outside[r:r + h, c:c + w].all():
            raise ValueError("panels overlap")
        outside[r:r + h, c:c + w] = False
        groups.setdefault(color, []).append((p, grid[r + 1:r + h - 1, c + 1:c + w - 1].copy()))
    if len(groups) != 2 or any(len(g) != 2 for g in groups.values()):
        raise ValueError("matching frame colors must define two groups of two")
    blank = [(color, p, a) for color, group in groups.items() for p, a in group
             if np.all(a == a[0, 0])]
    if len(blank) != 1:
        raise ValueError("exactly one panel must be blank")
    target_color, blank_box, blank_panel = blank[0]
    bg = int(blank_panel[0, 0])
    if bg in groups or not np.all(grid[outside] == bg):
        raise ValueError("blank and outside must share the neutral background")
    reference_color = next(color for color in groups if color != target_color)
    reference = [a for _, a in groups[reference_color]]
    occupied = next(a for p, a in groups[target_color] if p != blank_box)
    if any(color in np.unique(a) for color in groups for group in groups.values() for _, a in group):
        raise ValueError("frame colors are reserved for outlines")
    return dict(reference=reference, occupied=occupied, bg=bg, panels=panels,
                groups=groups, reference_color=reference_color, target_color=target_color,
                blank_box=blank_box)


def _details(input_grid):
    scene = _extract(_grid(input_grid))
    model, reference = _infer(scene["reference"], scene["bg"])
    occupied = _analyze(scene["occupied"], scene["bg"])
    if occupied["motif"] != reference[0]["motif"]:
        raise ValueError("reference and target must show the same glyph shape")
    return scene, model, occupied


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the blank companion from the complete input, without state or I/O."""
    _, model, occupied = _details(input_grid)
    return _predict(occupied, model)


def _translate(rng, points, motif, shape):
    low = [-min(p[d] for p in points) - min(p[d] for p in motif) for d in (0, 1)]
    high = [shape[d] - 1 - max(p[d] for p in points) - max(p[d] for p in motif) for d in (0, 1)]
    if any(lo > hi for lo, hi in zip(low, high)):
        return None
    offset = [int(rng.integers(lo, hi + 1)) for lo, hi in zip(low, high)]
    return tuple(sorted((r + offset[0], c + offset[1]) for r, c in points))


def _sample_centers(rng, shape, motif, topology):
    if topology == "triangle":
        length = int(rng.integers(3, 8))
        if rng.integers(2):
            points = [(0, 0), (length, 0), (0, length)]
        else:
            points = [(0, 0), (length, length), (length, -length)]
        for _ in range(int(rng.integers(4))):
            points = [(c, -r) for r, c in points]
    else:
        dr, dc = _DIRECTIONS[int(rng.integers(4))]
        if topology == "midpoint":
            step = int(rng.integers(2, 6))
            points = [(i * step * dr, i * step * dc) for i in range(3)]
        else:
            length = int(rng.integers(4, 10))
            points = [(0, 0), (length * dr, length * dc)]
    return _translate(rng, points, motif, shape)


def _sample_exchange(rng, shape, motif, topology, rail_direction=None):
    """Construct both complementary states from coordinates, not from solve."""
    for _ in range(2400):
        centers = _sample_centers(rng, shape, motif, topology)
        if centers is None:
            continue
        d = int(rng.integers(4)) if rail_direction is None else rail_direction
        dr, dc = _DIRECTIONS[d]
        length = int(rng.integers(4, 10))
        endpoints = _translate(rng, ((0, 0), (length * dr, length * dc)), motif, shape)
        if endpoints is None:
            continue
        try:
            source = _paint(shape, 0, 1, 2, _line(*endpoints), centers, motif)
            partner = _paint(shape, 0, 1, 2, _figure(centers), endpoints, motif)
            return source, partner, centers, endpoints
        except ValueError:
            continue
    raise RuntimeError("bounded geometry sampler exhausted")


def _without_middle(source, centers, motif):
    result = source.copy()
    for p in _stamps((sorted(centers)[1],), motif):
        result[p] = 0
    return result


def _palette(rng):
    colors = list(map(int, rng.permutation(10)))
    bg, frame_r, frame_t, re, rg, te, tg = colors[:7]
    mode = int(rng.integers(3))
    if mode == 0:
        te, tg = re, rg
    elif mode == 1:
        te, tg = rg, re
    return bg, frame_r, frame_t, (bg, re, rg), (bg, te, tg)


def _outline(grid, box, color):
    r, c, h, w = box
    grid[r, c:c + w] = grid[r + h - 1, c:c + w] = color
    grid[r:r + h, c] = grid[r:r + h, c + w - 1] = color


def _carrier(rng, reference, occupied, palette):
    bg, fr, ft, ref_map, target_map = palette
    h, w = occupied.shape
    ph, pw = h + 2, w + 2
    height, width = int(rng.integers(2 * ph + 1, 31)), int(rng.integers(2 * pw + 1, 31))
    row_slack = rng.multinomial(height - 2 * ph - 1, [1 / 3] * 3)
    col_slack = rng.multinomial(width - 2 * pw - 1, [1 / 3] * 3)
    rows = (int(row_slack[0]), int(row_slack[0] + ph + 1 + row_slack[1]))
    cols = (int(col_slack[0]), int(col_slack[0] + pw + 1 + col_slack[1]))
    slots = [(r, c, ph, pw) for r in rows for c in cols]
    permutation = list(map(int, rng.permutation(4)))
    contents = [reference[0], reference[1], occupied, np.zeros_like(occupied)]
    grid = np.full((height, width), bg, dtype=np.int8)
    boxes = []
    for i, logical in enumerate(contents):
        box = slots[permutation[i]]
        boxes.append(box)
        r, c, _, _ = box
        _outline(grid, box, fr if i < 2 else ft)
        mapping = np.array(ref_map if i < 2 else target_map, dtype=np.int8)
        grid[r + 1:r + h + 1, c + 1:c + w + 1] = mapping[logical]
    return grid, boxes


def _twin_demonstrations(rng, train_directions):
    shape = (int(rng.integers(11, 13)), int(rng.integers(11, 13)))
    motif = _MOTIFS[int(rng.integers(3, len(_MOTIFS)))]
    reference = _sample_exchange(rng, shape, motif, "midpoint")
    direction = int(rng.choice(train_directions))
    target = _sample_exchange(rng, shape, motif, "midpoint", direction)
    # Coordinates differ between groups, so copying the demonstrated answer
    # cannot substitute for transferring the arrangement rule.
    for _ in range(16):
        if not np.array_equal(reference[0], target[0]):
            break
        target = _sample_exchange(rng, shape, motif, "midpoint", direction)
    palette = _palette(rng)
    order = int(rng.integers(2))
    refs = [reference[0], reference[1]] if order == 0 else [reference[1], reference[0]]
    grid, boxes = _carrier(rng, refs, target[1], palette)
    first = {"input": grid, "output": np.array(palette[4], dtype=np.int8)[target[0]]}
    second_grid = grid.copy()
    r, c, _, _ = boxes[order]
    ref_two = _without_middle(reference[0], reference[2], motif)
    second_grid[r + 1:r + 1 + shape[0], c + 1:c + 1 + shape[1]] = np.array(palette[3], dtype=np.int8)[ref_two]
    target_two = _without_middle(target[0], target[2], motif)
    second = {"input": second_grid, "output": np.array(palette[4], dtype=np.int8)[target_two]}
    twins = [first, second]
    if rng.integers(2):
        twins.reverse()
    return twins


def _random_pair(rng, directions, model, reverse):
    shape = (int(rng.integers(11, 13)), int(rng.integers(11, 13)))
    motif = _MOTIFS[int(rng.integers(len(_MOTIFS)))]
    ref = _sample_exchange(rng, shape, motif, model)
    target = _sample_exchange(rng, shape, motif, model, int(rng.choice(directions)))
    for _ in range(16):
        if not np.array_equal(ref[0], target[0]):
            break
        target = _sample_exchange(rng, shape, motif, model, int(rng.choice(directions)))
    refs = list(ref[:2])
    if rng.integers(2):
        refs.reverse()
    # The two-glyph exchange is an involution. Choose its displayed state so
    # the held-out direction refers to the visible two-glyph baseline, just as
    # it does in the reverse midpoint/triangle cases.
    occupied, answer = (target[1], target[0]) if reverse or model == "pair" else target[:2]
    palette = _palette(rng)
    grid, _ = _carrier(rng, refs, occupied, palette)
    return {"input": grid, "output": np.array(palette[4], dtype=np.int8)[answer]}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    direction_order = list(map(int, rng.permutation(4)))
    train_directions, test_directions = direction_order[:2], direction_order[2:]
    puzzle = {"train": _twin_demonstrations(rng, train_directions), "test": []}
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, count, directions in (("train", num_train - 2, train_directions),
                                      ("test", num_test, test_directions)):
        for i in range(count):
            # The shortest set proves reference dependence; larger sets also
            # demonstrate the noncollinear construction before testing it.
            if split == "train" and i == 0:
                model, reverse = "triangle", bool(rng.integers(2))
            elif split == "test" and i == 0:
                model, reverse = "midpoint", True
            else:
                pool = ("pair", "midpoint", "triangle") if num_train >= 3 else ("pair", "midpoint")
                model, reverse = str(rng.choice(pool)), bool(rng.integers(2))
            for _ in range(12):
                pair = _random_pair(rng, directions, model, reverse)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded duplicate rejection exhausted")
    return puzzle


def _direction(points):
    a, b = sorted(points)[0], sorted(points)[-1]
    dr, dc = b[0] - a[0], b[1] - a[1]
    return 0 if dr == 0 else 1 if dc == 0 else 2 if dr * dc > 0 else 3


def _rail_direction(a):
    # Original single segment, whether currently drawn or represented by its
    # two endpoint glyphs. This is observable in both complementary states.
    return _direction(a["centers"] if a["topology"] == "pair" else a["anchors"])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert _key(input_grid) not in seen, "duplicate input"
            seen.add(_key(input_grid))
            scene, model, a = _details(input_grid)
            expected = _predict(a, model)
            assert np.array_equal(expected, output), "wrong missing companion"
            assert output.shape == scene["occupied"].shape, "local canvas size must be preserved"
            assert not np.array_equal(scene["occupied"], output), "degenerate unchanged exchange"
            out = _analyze(output, scene["bg"])
            assert (out["edge"], out["glyph"], out["motif"]) == (a["edge"], a["glyph"], a["motif"])
            # Both states must form the same demonstrated reciprocal relation.
            target_model, _ = _infer([scene["occupied"], output], scene["bg"])
            assert target_model == model
            # Erasure and exact movement of both structures are independently
            # expressed as their recovered endpoint/center sets.
            if a["topology"] == model and a["edge_kind"] == "line":
                assert set(out["centers"]) == set(a["anchors"])
                expected_edge = _figure(a["centers"])
            else:
                assert set(out["anchors"]) == set(a["centers"])
                expected_edge = _figure(a["centers"])
            actual_edge = frozenset(map(tuple, np.argwhere(output == a["edge"]).tolist()))
            assert actual_edge == expected_edge
            assert not np.array_equal(output, scene["reference"][0])
            assert not np.array_equal(output, scene["reference"][1])
            records[split].append((scene, model, a))
    # A reference-blind predictor cannot explain these two training answers.
    assert any(m == "pair" and n == "midpoint" and b["topology"] == "pair"
               and np.array_equal(s["occupied"], t["occupied"])
               for s, m, a in records["train"] for t, n, b in records["train"])
    train_dirs = {_rail_direction(a) for _, _, a in records["train"]}
    test_dirs = {_rail_direction(a) for _, _, a in records["test"]}
    assert train_dirs.isdisjoint(test_dirs), "target rail directions must be held out"
    assert records["test"][0][1] == "midpoint" and records["test"][0][2]["topology"] == "pair"


def self_test():
    """42 puzzles streamed in memory; no traces, files, caches or retained runs."""
    signatures, shapes, palettes, motifs, structures = set(), set(), set(), set(), set()
    models, directions, layouts, restoration = set(), set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        duplicate = generate_puzzle(seed)
        assert all(np.array_equal(p[k], q[k]) for s in puzzle
                   for p, q in zip(puzzle[s], duplicate[s]) for k in p)
        digest = hashlib.sha256()
        for split in puzzle:
            for pair in puzzle[split]:
                scene, model, a = _details(pair["input"])
                models.add(model)
                shapes.add(pair["input"].shape)
                palettes.add((a["bg"], a["edge"], a["glyph"]))
                motifs.add(a["motif"])
                directions.add(_rail_direction(a))
                layouts.add(tuple(p[:3] for p in scene["panels"]))
                # Geometry rather than recoloring: this normalizes role colors.
                structures.add((a["shape"], model, a["centers"], a["anchors"], a["motif"]))
                restoration += model == "midpoint" and a["topology"] == "pair"
                digest.update(pair["input"].tobytes())
        signatures.add(digest.hexdigest())
        # Swap the two complete reference members while leaving target intact.
        original = puzzle["test"][0]
        changed = original["input"].copy()
        scene, _, _ = _details(changed)
        group = scene["groups"][scene["reference_color"]]
        for (p, _), (_, other) in zip(group, group[::-1]):
            _, r, c, h, w = p
            changed[r + 1:r + h - 1, c + 1:c + w - 1] = other
        assert np.array_equal(solve(changed), original["output"])
    assert len(signatures) == 42 and len(shapes) >= 10 and len(palettes) >= 35
    assert len(motifs) == len(_MOTIFS) and len(structures) >= 130 and len(layouts) >= 100
    assert models == {"pair", "midpoint", "triangle"} and directions == set(range(4))
    assert restoration >= 84
    # Wrong labels must be rejected, and omitting either inherited mechanism
    # changes the constructed twin answers rather than just their carrier.
    puzzle = generate_puzzle(91, 2, 1)
    first, second = puzzle["train"]
    assert not np.array_equal(first["output"], second["output"])
    scenes = [_details(pair["input"]) for pair in puzzle["train"]]
    assert np.array_equal(scenes[0][0]["occupied"], scenes[1][0]["occupied"])
    # Only one reference interior changes: neither the target, the frames,
    # nor the other reference member can secretly encode the answer.
    changed = first["input"] != second["input"]
    changed_members = 0
    for p, _ in scenes[0][0]["groups"][scenes[0][0]["reference_color"]]:
        _, r, c, h, w = p
        interior = changed[r + 1:r + h - 1, c + 1:c + w - 1]
        changed_members += bool(interior.any())
        interior[:] = False
    assert changed_members == 1 and not changed.any()
    # The answers differ by precisely the restored middle glyph. Their
    # edge masks agree, ruling out a second unrelated operation in the twin.
    midpoint_index = next(i for i, (_, model, _) in enumerate(scenes) if model == "midpoint")
    scene, _, a = scenes[midpoint_index]
    p, q = a["anchors"]
    middle = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
    expected_difference = _stamps((middle,), a["motif"])
    actual_difference = frozenset(map(tuple, np.argwhere(first["output"] != second["output"]).tolist()))
    assert actual_difference == expected_difference
    assert np.array_equal(first["output"] == a["edge"], second["output"] == a["edge"])
    for pair in puzzle["train"]:
        scene, _, a = _details(pair["input"])
        assert not np.array_equal(scene["occupied"], pair["output"])
        # Omitting either half of B fails separately, as does keeping source
        # debris while adding the new structures.
        assert not np.array_equal(scene["occupied"] == a["glyph"], pair["output"] == a["glyph"])
        assert not np.array_equal(scene["occupied"] == a["edge"], pair["output"] == a["edge"])
        assert np.any((scene["occupied"] != a["bg"]) & (pair["output"] == a["bg"]))
    puzzle["test"][0]["output"][0, 0] = (int(puzzle["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
