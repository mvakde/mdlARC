import numpy as np


# Exact source inputs, including shape; no palette or dimension exclusions.
_SOURCE_INPUTS = (('11111111111111111111111', '14444444444411133111881', '14111111111411133111111', '14111111111411111111111', '14111111111411111144111', '14444444444411111144111', '11111111111111111111111', '18888888888811111111311', '18111111111811144111311', '18111111111811111111111', '18111111111811111111111', '18888888888811111111111', '11111111111111112221111', '12222222222211112221111', '12111111111211111111111', '12111111111211111111111', '12111111111211188111111', '12222222222211188111111', '11111111111111111111111', '11111111111111111181111', '11111111111881111181111', '11111111111111111111111', '11111111111111111111111'), ('88888888888888888888888', '88888888888888884888668', '88111111111118844488888', '88188888888818884888888', '88188888888818888884888', '88188888888818888444888', '88111111111118888848888', '88888888888888888888888', '88666666666668822888338', '88688888888868822888388', '88688888888868888888888', '88688888888868888188888', '88666666666668881188888', '88888888888888888888888', '88444444444448888888888', '88488888888848888888888', '88488888888848888888888', '88488888888848868888888', '88444444444448868668888', '88888888888888888688888', '88888888888888888888888', '88888888888888888888888', '88888888888888888888888'), ('33333333333333333333333', '32222222222233331333333', '32333333333233311133333', '32333333333233311133333', '32333333333233331333333', '32222222222233333333333', '33333333333333333333333', '34444444444433383331133', '34333333333433888311113', '34333333333433383331133', '34333333333433333333333', '34444444444433333333333', '33333333333333333333333', '31111111111133333433333', '31333333333133334443333', '31333333333133333433333', '31333333333133333333333', '31111111111133333344333', '33333333333333333344333', '33333333333333333333333'), ('88888888888888888888888', '82222222222288818884488', '82888888888288111884488', '82888888888288818888888', '82888888888288888888888', '82222222222288888828888', '88888888888888882222288', '84444444444488888828888', '84888888888488888888888', '84888888888488888888888', '84888888888488888888488', '84444444444488228884488', '88888888888888228888888', '83333333333388888888888', '83888888888388888888888', '83888888888388884888558', '83888888888388884488888', '83333333333388888888888', '88888888888888888888888', '81111111111188888888448', '81888888888188888888448', '81888888888188838888888', '81888888888188333888888', '81111111111188838888888', '88888888888888888889988', '88888888888888888889988', '88888888888888888888888'))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

def _integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)

def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        num_train = _integer("num_train", num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        num_test = _integer("num_test", num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = num_test + 1
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        low = num_test + 1
        num_train = low + seed % (11 - low)
    elif num_test is None:
        high = min(4, num_train - 1)
        num_test = 1 + seed % high

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test

def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    short_side: int,
    long_side: int,
    color: int,
) -> None:
    r1 = r0 + short_side - 1
    c1 = c0 + long_side - 1

    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color

def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    maximum = int(counts.max())

    _check(
        int(np.count_nonzero(counts == maximum)) == 1,
        "background mode is not unique",
    )

    return int(values[int(np.argmax(counts))])

def _components(grid: np.ndarray, background: int) -> list[dict]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components = []

    for r in range(height):
        for c in range(width):
            if visited[r, c] or int(grid[r, c]) == background:
                continue

            color = int(grid[r, c])
            visited[r, c] = True
            stack = [(r, c)]
            cells = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))

                for nr, nc in (
                    (rr - 1, cc),
                    (rr + 1, cc),
                    (rr, cc - 1),
                    (rr, cc + 1),
                ):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            components.append(
                {
                    "color": color,
                    "cells": cells,
                    "box": (
                        min(rows),
                        max(rows),
                        min(cols),
                        max(cols),
                    ),
                }
            )

    return components

def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            if not (
                np.array_equal(
                    pair_a["input"],
                    pair_b["input"],
                )
                and np.array_equal(
                    pair_a["output"],
                    pair_b["output"],
                )
            ):
                return False

    return True

def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            pair["input"].shape,
            pair["input"].tobytes(),
            pair["output"].tobytes(),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )

def _shape(rng, area=None, simple=False):
    """Connected, filled polyominoes; geometry has no access to object count."""
    if area is None:
        area = int(rng.choice((1, 1, 2, 2, 3, 4))) if simple else int(
            rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16),
                       p=(.08, .13, .13, .13, .10, .09, .07, .07,
                          .05, .04, .05, .03, .03)))
    kind = int(rng.integers(5))
    if kind == 0 and area <= 8:
        cells = {(0, c) for c in range(area)}
    elif kind == 1:
        width = int(rng.integers(2, min(5, area) + 1)) if area > 1 else 1
        cells = {(i // width, i % width) for i in range(area)}
    elif kind == 2 and area <= 11:
        arm = int(rng.integers(1, area + 1))
        cells = {(0, c) for c in range(arm)} | {(r, 0) for r in range(area-arm+1)}
    else:
        cells = {(0, 0)}
        # A cross seed makes branches common without fixing one mask vocabulary.
        if kind == 3 and area >= 5:
            cells |= {(0, 1), (0, -1), (1, 0), (-1, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _frame(component):
    r0, r1, c0, c1 = component['box']
    h, w = r1-r0+1, c1-c0+1
    if min(h, w) not in (5, 7, 9) or max(h, w) <= min(h, w):
        return False
    perimeter = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                 if r in (r0, r1) or c in (c0, c1)}
    return set(component['cells']) == perimeter


def _inspect(grid):
    """Recover all semantic roles and counts from visible cells only."""
    bg = _mode_color(grid)
    parts = _components(grid, bg)
    frames = [p for p in parts if _frame(p)]
    objects = [p for p in parts if not _frame(p)]
    _check(1 <= len(frames) <= 5, 'expected one through five hollow frames')
    _check(bool(objects), 'field must be visible, including in all-zero cases')
    colors = [f['color'] for f in frames]
    _check(len(set(colors)) == len(colors), 'frame colors must be distinct')
    axes = {int(f['box'][1]-f['box'][0] > f['box'][3]-f['box'][2]) for f in frames}
    _check(len(axes) == 1, 'frames must share a long-axis orientation')
    vertical = axes.pop()
    centers = [(f['box'][0]+f['box'][1])/2 if vertical else
               (f['box'][2]+f['box'][3])/2 for f in frames]
    obj_centers = [sum(p[0 if vertical else 1] for p in o['cells']) / len(o['cells'])
                   for o in objects]
    if min(obj_centers) > max(centers):
        direction = 'down' if vertical else 'right'
    elif max(obj_centers) < min(centers):
        direction = 'up' if vertical else 'left'
    else:
        raise AssertionError('external objects do not identify one field-facing end')

    # Every object/frame has a background moat, even between different colors.
    labels = np.full(grid.shape, -1, dtype=int)
    for i, p in enumerate(parts):
        for r, c in p['cells']:
            labels[r, c] = i
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = labels[max(0, dr):grid.shape[0]+min(0, dr),
                   max(0, dc):grid.shape[1]+min(0, dc)]
        b = labels[max(0, -dr):grid.shape[0]+min(0, -dr),
                   max(0, -dc):grid.shape[1]+min(0, -dc)]
        _check(not np.any((a >= 0) & (b >= 0) & (a != b)), 'components touch')
    for f in frames:
        r0, r1, c0, c1 = f['box']
        _check(np.all(grid[r0+1:r1, c0+1:c1] == bg), 'nonempty frame interior')
    for o in objects:
        _check(1 <= len(o['cells']) <= 16, 'unsupported external object size')
    counts = [sum(o['color'] == f['color'] for o in objects) for f in frames]
    areas = [sum(len(o['cells']) for o in objects if o['color'] == f['color'])
             for f in frames]
    dims = [(f['box'][1]-f['box'][0]+1, f['box'][3]-f['box'][2]+1) for f in frames]
    capacities = [(max(d)-3)//2 for d in dims]
    _check(all(n <= cap for n, cap in zip(counts, capacities)), 'count exceeds capacity')
    return dict(background=bg, frames=frames, objects=objects, direction=direction,
                frame_colors=tuple(colors), counts=tuple(counts), areas=tuple(areas),
                capacities=tuple(capacities), dimensions=tuple(dims))


def _reconstruct(grid, detail=None):
    detail = _inspect(grid) if detail is None else detail
    result = np.full_like(grid, detail['background'])
    for f, count in zip(detail['frames'], detail['counts']):
        for r, c in f['cells']:
            result[r, c] = f['color']
        r0, r1, c0, c1 = f['box']
        for i in range(1, count+1):
            direction = detail['direction']
            r = r1-2*i if direction == 'down' else r0+2*i if direction == 'up' else (r0+r1)//2
            c = c1-2*i if direction == 'right' else c0+2*i if direction == 'left' else (c0+c1)//2
            result[r, c] = f['color']
    return result


def _layout(rng, counts, simple, teaching, spacious=False):
    n = len(counts)
    shorts = [5]*n
    # Heights, gaps, and margins compete only for actual available canvas space.
    budget = 30 - (6*n-1)
    if not simple and not spacious:
        for i in rng.permutation(n):
            add = int(rng.choice([x for x in (0, 2, 4) if x <= budget]))
            shorts[int(i)] += add
            budget -= add
    uniform = simple or rng.random() < .30
    longs = [max(s+1, 7, 2*c+3) for s, c in zip(shorts, counts)]
    for i in range(n):
        if teaching and i < 2:
            longs[i] = max(longs[i], 2*counts[i]+5)
        if not spacious:
            longs[i] = min(21, longs[i] + int(rng.choice((0, 0, 1, 2, 3, 4))))
    if uniform:
        shorts = [min(shorts)]*n
        longs = [max(longs)]*n
    gaps = [1]*(n-1)
    extra = 30 - sum(shorts) - sum(gaps)
    if not spacious:
        for i in range(n-1):
            add = int(rng.integers(min(extra, 3)+1))
            gaps[i] += add
            extra -= add
    top = int(rng.integers(min(extra, 4)+1)) if not spacious else 0
    extra -= top
    bottom = int(rng.integers(min(extra, 6)+1)) if not spacious else extra
    h = sum(shorts) + sum(gaps) + top + bottom
    h = min(30, max(h, 9))
    left = int(rng.integers(4)) if not spacious else 0
    offsets = [0 if uniform or spacious else int(rng.integers(3)) for _ in counts]
    boxes = []
    r = top
    for i, (s, long) in enumerate(zip(shorts, longs)):
        c = min(left+offsets[i], 23-long)
        boxes.append((r, c, s, long))
        r += s + (gaps[i] if i < n-1 else 0)
    return h, boxes


def _place(rng, h, w, boxes, colors, bg, specs, wrap, greedy=False):
    grid = np.full((h, w), bg, dtype=np.int8)
    occupied = np.zeros((h, w), dtype=bool)
    for (r, c, short, long), color in zip(boxes, colors):
        _draw_frame(grid, r, c, short, long, color)
        occupied[max(0, r-1):min(h, r+short+1), max(0, c-1):min(w, c+long+1)] = True
    edge = max(c+long-1 for r, c, short, long in boxes)
    center = max(c+(long-1)/2 for r, c, short, long in boxes)
    lower = max(int(center)+1, edge-3) if wrap else edge+2
    occupied[:, :lower] = True
    # Largest objects first; candidates are every legal pixel, not lattice slots.
    ordered = sorted(range(len(specs)), key=lambda i: (-len(specs[i][1]), float(rng.random())))
    for index in ordered:
        color, shape = specs[index]
        sh = 1+max(r for r, c in shape)
        sw = 1+max(c for r, c in shape)
        if sh > h or sw > w:
            return None
        candidates = np.ones((h-sh+1, w-sw+1), dtype=bool)
        for dr, dc in shape:
            candidates &= ~occupied[dr:dr+h-sh+1, dc:dc+w-sw+1]
        rr, cc = np.nonzero(candidates)
        if not len(rr):
            return None
        j = 0 if greedy else int(rng.integers(len(rr)))
        r0, c0 = int(rr[j]), int(cc[j])
        for dr, dc in shape:
            r, c = r0+dr, c0+dc
            grid[r, c] = color
            occupied[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
    return grid


def _make_pair(rng, colors, counts, bg, distractors, rotation, simple, forced):
    specs = []
    for i, (color, count) in enumerate(zip(colors, counts)):
        for j in range(count):
            specs.append((color, _shape(rng, forced[i][j] if i in forced else None, simple)))
    specs += [(color, _shape(rng, simple=simple)) for color in distractors]
    teaching = bool(forced)
    wrap = rng.random() < .65
    h, boxes = _layout(rng, counts, simple, teaching)
    edge = max(c+long for r, c, short, long in boxes)
    area = sum(len(shape) for color, shape in specs)
    # Independently choose compact or airy fields; actual packing can enlarge
    # a tight proposal without coupling any object's geometry to its count.
    packing_factor = float(rng.uniform(1.4, 4.2))
    field = max(4, int(np.ceil((area*packing_factor + len(specs)*3)/h)))
    w = min(30, edge+field+int(rng.integers(1, 6)))
    # Enlarge the canvas if the independently drawn shapes require more space.
    grid = None
    for attempt in range(12):
        grid = _place(rng, h, w, boxes, colors, bg, specs, wrap)
        if grid is not None:
            break
        if attempt < 5:
            h, w = min(30, h+3), min(30, w+3)
        else:
            h, boxes = _layout(rng, counts, simple, teaching, spacious=True)
            h, w, wrap = 30, 30, True
    if grid is None:
        # Guaranteed roomy fallback: preserve teaching areas, simplify only
        # optional geometry, minimize the bank, then scan free placements.
        specs = []
        for i, (color, count) in enumerate(zip(colors, counts)):
            for j in range(count):
                area = forced[i][j] if i in forced else 1
                specs.append((color, tuple((r, 0) for r in range(area))))
        specs += [(color, ((0, 0),)) for color in distractors]
        boxes = [(6*i, 0, 5, max(7, 2*n+3, 2*n+5 if teaching and i < 2 else 0))
                 for i, n in enumerate(counts)]
        h, w = 30, 30
        grid = _place(rng, h, w, boxes, colors, bg, specs, True, greedy=True)
        _check(grid is not None, 'roomy deterministic packing failed')
    output = np.full_like(grid, bg)
    for (r, c, short, long), color, count in zip(boxes, colors, counts):
        _draw_frame(output, r, c, short, long, color)
        for i in range(count):
            output[r+short//2, c+long-3-2*i] = color
    return {'input': np.rot90(grid, rotation).copy(),
            'output': np.rot90(output, rotation).copy()}


def _count_row(rng, n, maximum, allowed=None):
    values = list(range(maximum+1)) if allowed is None else sorted(allowed)
    draw = float(rng.random())
    if draw < .10:
        return [0]*n
    if draw < .24:
        return [int(rng.choice([x for x in values if x*n <= 14]))]*n
    result = [int(rng.choice(values)) for _ in range(n)]
    while sum(result) > 14:
        i = int(np.argmax(result))
        result[i] = max(x for x in values if x < result[i])
    return result


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer('seed', seed)
    nt, nq = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed % (1 << 64))
    # Only the color-match/count/erase/field-relative pitch-two convention is
    # shared. These sampled styles and teaching regimes never encode an answer.
    simple = rng.random() < .20
    mixed_orientation = rng.random() < .78
    base_rotation = int(rng.integers(4))
    # Opposite sides rule out both fixed screen direction and fixed coordinate
    # sign before a perpendicular orientation is held out at test time.
    turn = 2
    regime = int(rng.choice(4, p=(.23, .18, .19, .40)))
    base_categories = int(rng.integers(2, 5))
    unseen_counts = rng.random() < .27
    familiar_counts = not unseen_counts and rng.random() < .45
    color_order = [int(x) for x in rng.permutation(10)]
    anchor, novel = color_order[:2]
    collision_area = int(rng.integers(3, 5 if simple else 9))
    other_area = int(rng.choice([x for x in range(2, 5 if simple else 11) if x != collision_area]))
    split_area = int(rng.integers(1, collision_area))
    split_other = int(rng.integers(1, other_area))
    puzzle = {'train': [], 'test': []}
    previous_bg = None
    first_palette = None
    observed_counts = set()
    seen = set()
    for index in range(nt+nq):
        train = index < nt
        local_index = index if train else index-nt
        split = 'train' if train else 'test'
        if regime == 0:
            n = 3 if train else 4
        elif regime == 1:
            n = base_categories
        elif regime == 2:
            n = int(rng.integers(3, 6)) if train else int(rng.integers(1, 3))
        else:
            n = int(rng.integers(2 if index < 2 else 1, 6))
        bg = int(rng.choice([c for c in range(10) if c != previous_bg
                            and not (index < 2 and c == anchor)
                            and not (not train and local_index == 0 and c == novel)
                            and not (index == 1 and c == first_palette[1])]))
        pool = [c for c in range(10) if c != bg and (not train or c != novel)]
        forced = {}
        if index < 2:
            # Keep one literal color across two incompatible area/count examples.
            pool.remove(anchor)
            if index == 1 and first_palette[1] in pool:
                pool.remove(first_palette[1])
            colors = [anchor] + [int(x) for x in rng.choice(pool, n-1, replace=False)]
        else:
            colors = [int(x) for x in rng.choice(pool, n, replace=False)]
        if not train and local_index == 0 and novel not in colors:
            colors[0] = novel
        maximum = 3 if train and unseen_counts else 4 if simple else 6
        allowed = observed_counts | {0} if not train and familiar_counts else None
        counts = _count_row(rng, n, maximum, allowed)
        if index == 0:
            counts[:2] = [1, 2]
            forced = {0: [collision_area], 1: [split_other, other_area-split_other]}
            first_palette = colors[:]
        elif index == 1:
            counts[0] = 2
            forced = {0: [split_area, collision_area-split_area]}
            # In two-pair puzzles this provides the complete 1,2 progression.
            counts[1] = 1
        elif not train and local_index == 0 and unseen_counts:
            counts[0] = int(rng.integers(4, 8))
        while sum(counts) > 14:
            i = max(range(2 if index < 2 else 1, n), key=lambda j: counts[j])
            counts[i] -= 1
        dpool = [c for c in range(10) if c != bg and c not in colors]
        nd = int(rng.integers(0, 7))
        if not sum(counts):
            nd = max(1, nd)
        palette_size = int(rng.integers(1, min(4, len(dpool))+1))
        dcolors = [int(x) for x in rng.choice(dpool, palette_size, replace=False)]
        distractors = [int(rng.choice(dcolors)) for _ in range(nd)]
        if index == 1:
            distractors = [first_palette[1]] + distractors[:5]
        if mixed_orientation:
            rotation = base_rotation if index == 0 else (base_rotation+turn)%4 if index == 1 else int(rng.integers(4))
            if not train and local_index == 0:
                rotation = int(rng.choice([x for x in range(4) if x not in
                                          (base_rotation, (base_rotation+turn)%4)]))
        else:
            rotation = base_rotation
        for attempt in range(8):
            pair = _make_pair(rng, colors, counts, bg, distractors, rotation, simple, forced)
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen and not _source_copy(pair['input']):
                break
        else:
            raise AssertionError('failed to produce a distinct scene')
        seen.add(key)
        puzzle[split].append(pair)
        previous_bg = bg
        if train:
            observed_counts.update(counts)
    validate_generated_puzzle(puzzle)
    return puzzle


def _source_copy(grid):
    return tuple(''.join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS


def _evidence(details):
    examples = [(c, n, a) for d in details for c, n, a in
                zip(d['frame_colors'], d['counts'], d['areas'])]
    collision = any(c == cc and a == aa and n != nn for c, n, a in examples
                    for cc, nn, aa in examples)
    independence = any(n == nn and a != aa for c, n, a in examples
                       for cc, nn, aa in examples)
    informative = {d['direction'] for d in details if any(
        0 < n < cap for n, cap in zip(d['counts'], d['capacities']))}
    return collision, independence, informative


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    _check(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
           and len(puzzle['train']) > len(puzzle['test']), 'pair counts')
    seen = set()
    details = {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            for g in pair.values():
                _check(isinstance(g, np.ndarray) and g.ndim == 2, 'grid schema')
                _check(all(1 <= n <= 30 for n in g.shape), 'grid bounds')
                _check(np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9)), 'colors')
            inp, out = pair['input'], pair['output']
            _check(inp.shape == out.shape, 'dimensions changed')
            key = (inp.shape, tuple(int(x) for x in inp.flat))
            _check(key not in seen, 'duplicate input')
            seen.add(key)
            _check(not _source_copy(inp), 'exact source copy')
            d = _inspect(inp)
            _check(np.array_equal(out, _reconstruct(inp, d)), 'wrong tally/erasure/frame output')
            _check(not np.array_equal(inp, out), 'unchanged example')
            details[split].append(d)
    train, test = details['train'], details['test']
    collision, independence, informative = _evidence(train)
    _check(collision and independence, 'training permits an area lookup shortcut')
    _check({1, 2}.issubset({n for d in train for n in d['counts']}), 'unary progression missing')
    _check(len({d['background'] for d in train}) >= 2, 'training backgrounds must vary')
    _check(len({frozenset(d['frame_colors']) for d in train}) >= 2, 'training frame palettes must vary')
    categories = {c for d in train for c in d['frame_colors']}
    discards = {o['color'] for d in train for o in d['objects'] if o['color'] not in d['frame_colors']}
    _check(bool(categories & discards), 'local color-role switching missing')
    all_directions = {d['direction'] for d in train+test}
    signs = {d in ('right', 'down') for d in informative}
    _check((len(informative) >= 2 and len(signs) == 2) or all_directions <= informative,
           'mixed/novel field sides need informative training on different sides')
    _check(any(set(d['frame_colors'])-categories for d in test), 'test must exercise a new local category')


def puzzle_metrics(puzzle):
    """Task features reconstructed from grids for self-test and review reports."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _inspect(pair['input'])
            shapes = []
            for o in d['objects']:
                r0, r1, c0, c1 = o['box']
                shapes.append(tuple(sorted((r-r0, c-c0) for r, c in o['cells'])))
            counted = [o for o in d['objects'] if o['color'] in d['frame_colors']]
            discarded = [o for o in d['objects'] if o['color'] not in d['frame_colors']]
            vertical = d['direction'] in ('up', 'down')
            axis = 0 if vertical else 2
            low = min(f['box'][axis] for f in d['frames'])
            high = max(f['box'][axis+1] for f in d['frames'])
            wrap = any(any(low <= p[0 if vertical else 1] <= high for p in o['cells']) for o in d['objects'])
            shorts = [min(x) for x in d['dimensions']]
            longs = [max(x) for x in d['dimensions']]
            bank_axis = 2 if vertical else 0
            ordered = sorted(d['frames'], key=lambda f: f['box'][bank_axis])
            gaps = [b['box'][bank_axis]-a['box'][bank_axis+1]-1 for a, b in zip(ordered, ordered[1:])]
            offsets = [f['box'][axis] for f in d['frames']]
            records.append(dict(split=split, shape=pair['input'].shape,
                background=d['background'], direction=d['direction'], frames=len(d['frames']),
                palette=d['frame_colors'], counts=d['counts'], total_areas=d['areas'],
                capacities=d['capacities'], shorts=shorts, longs=longs,
                unequal=len(set(d['dimensions'])) > 1, offsets=offsets, gaps=gaps,
                wrapped=wrap, all_zero=max(d['counts']) == 0,
                all_equal=len(set(d['counts'])) == 1,
                full=sum(n == cap for n, cap in zip(d['counts'], d['capacities'])),
                distractors=len(discarded), distractor_colors=sorted({o['color'] for o in discarded}),
                counted_areas=[len(o['cells']) for o in counted],
                discarded_areas=[len(o['cells']) for o in discarded],
                object_boxes=[o['box'] for o in d['objects']], masks=shapes,
                repeated_shapes=len(shapes) > len(set(shapes)),
                repeated_within_color=any(
                    len([m for o, m in zip(d['objects'], shapes) if o['color'] == c]) >
                    len({m for o, m in zip(d['objects'], shapes) if o['color'] == c})
                    for c in d['frame_colors']),
                within_color_sizes=any(len({len(o['cells']) for o in counted if o['color'] == c}) > 1
                                       for c in d['frame_colors']),
                field_density=sum(len(o['cells']) for o in d['objects']) /
                    (pair['input'].size-sum(h*w for h, w in d['dimensions'])),
                frame_boxes=[f['box'] for f in d['frames']]))
    return records


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [r for p in puzzles for r in puzzle_metrics(p)]
    _check(len({_puzzle_signature(p) for p in puzzles}) == 42, 'cross-seed repetition')
    _check(len({len(p['train']) for p in puzzles}) >= 4, 'train count diversity')
    _check(len({len(p['test']) for p in puzzles}) >= 3, 'test count diversity')
    _check({r['background'] for r in records} == set(range(10)), 'background coverage')
    _check({r['frames'] for r in records} == set(range(1, 6)), 'category-count coverage')
    _check({r['direction'] for r in records} == {'up', 'down', 'left', 'right'}, 'field sides')
    _check({s for r in records for s in r['shorts']} == {5, 7, 9}, 'short sides')
    _check(len({s for r in records for s in r['longs']}) >= 10, 'frame lengths')
    _check(len({r['shape'] for r in records}) >= 60, 'canvas diversity')
    _check(min(min(r['shape']) for r in records) <= 12 and max(max(r['shape']) for r in records) == 30, 'canvas range')
    _check(len({tuple(m) for r in records for m in r['masks']}) >= 100, 'object vocabulary')
    _check(max(a for r in records for a in r['counted_areas']) >= 14, 'large objects')
    _check(max(max(r['counts']) for r in records) >= 6, 'large counts')
    _check({r['distractors'] for r in records} >= set(range(7)), 'distractor counts')
    _check(max(len(r['distractor_colors']) for r in records) >= 3, 'distractor palettes')
    for key in ('unequal', 'wrapped', 'repeated_shapes', 'repeated_within_color', 'within_color_sizes'):
        _check(sum(r[key] for r in records) >= 20, 'missing variation: '+key)
    for split in ('train', 'test'):
        selected = [r for r in records if r['split'] == split]
        _check(any(r['all_zero'] for r in selected), 'missing zero '+split)
        _check(any(r['all_equal'] and r['frames'] > 1 for r in selected), 'missing equal '+split)
        _check(any(r['full'] for r in selected), 'missing full '+split)
    _check(len({(o[0]%3, o[2]%3) for r in records for o in r['object_boxes']}) == 9, 'placement lattice')
    _check(any(len(set(r['gaps'])) > 1 for r in records), 'unequal gaps')
    _check(any(len(set(r['offsets'])) > 1 for r in records), 'offset banks')
    _check(len({a for r in records for a in r['counted_areas']} &
               {a for r in records for a in r['discarded_areas']}) >= 10,
           'distractor and counted area distributions do not overlap')
    _check(sum(r['unequal'] and max(r['counted_areas'] or [0]) >= 8
               and r['wrapped'] for r in records) >= 10, 'combined variation')
    _check(min(r['field_density'] for r in records) < .05 and max(r['field_density'] for r in records) > .25, 'density spectrum')
    mixed, constant, novel_count, familiar_count, contraction, expansion, equal = (0,)*7
    for p in puzzles:
        validate_generated_puzzle(p)
        rs = puzzle_metrics(p)
        tr, te = [r for r in rs if r['split'] == 'train'], [r for r in rs if r['split'] == 'test']
        sides = {r['direction'] for r in rs}
        mixed += len(sides) > 1
        constant += len(sides) == 1
        tc, qc = {n for r in tr for n in r['counts']}, {n for r in te for n in r['counts']}
        novel_count += bool(qc-tc)
        familiar_count += qc <= tc
        contraction += min(r['frames'] for r in te) < min(r['frames'] for r in tr)
        expansion += max(r['frames'] for r in te) > max(r['frames'] for r in tr)
        equal += {r['frames'] for r in te} == {r['frames'] for r in tr}
        _check(len({r['shape'] for r in rs}) > 1, 'constant canvas within puzzle')
    _check(min(mixed, constant, novel_count, familiar_count, contraction, expansion, equal) >= 3,
           'train/test regime spectrum')
    _check(_puzzles_equal(generate_puzzle(137, 7, 3), generate_puzzle(137, 7, 3)), 'determinism')
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(1003, tr, te)
        _check(tr is None or len(p['train']) == tr, 'requested train count')
        _check(te is None or len(p['test']) == te, 'requested test count')
        validate_generated_puzzle(p)
    for args in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
                 {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(0, **args)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Negative checks exercise semantic validation rather than sampler metadata.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return
    raise AssertionError('validator accepted an unerased object field')
