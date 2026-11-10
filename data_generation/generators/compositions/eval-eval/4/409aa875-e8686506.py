"""Crossfire jigsaw: projected color constraints, exact inventory, tile impacts.

No parent modules, files, training labels, seeds, or generation state are read by
solve. Only the standard library and NumPy are used. All searches are bounded;
an exhausted search raises an error rather than claiming uniqueness.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import hashlib

import numpy as np


BACKGROUND = 0
COLLISION = 1
SCAFFOLD = 5
DISTANCE = 5
PAYLOAD = (2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
KING = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))
MAX_SEARCH_NODES = 12000
MAX_PAIR_ATTEMPTS = 160
MAX_ARROWS = 14


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "Grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "Integer cells required")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "Colors must be 0..9")
    return grid.astype(np.int8, copy=True)


def _normalize(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _components(cells, neighbors=ORTHOGONAL):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(frozenset(component))
    return result


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return frozenset((r + a, c + b) for a, b in offsets)


def _decode_arrow(cells):
    matches = [(p, d) for p in sorted(cells) for d in DIRECTIONS
               if _arrow(p, d) == cells]
    _require(len(matches) == 1, "An arrow must be an unambiguous V or L triomino")
    return matches[0]


def _separated(cells, occupied):
    return not any((r + dr, c + dc) in occupied
                   for r, c in cells for dr, dc in KING)


def _read_scene(input_grid):
    grid = _grid(input_grid)
    _require(not np.any(grid == COLLISION), "Collision color is output-only")
    rr, cc = np.where(grid == SCAFFOLD)
    _require(len(rr) > 0, "Missing scaffold")
    top, left, bottom, right = int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1
    board = grid[top:bottom, left:right].copy()
    _require(min(board.shape) >= 3, "Board too small")
    _require(all(np.all(edge == SCAFFOLD) for edge in
                 (board[0], board[-1], board[:, 0], board[:, -1])),
             "The board has a closed scaffold border")
    holes = frozenset(map(tuple, np.argwhere(board == BACKGROUND).tolist()))
    _require(bool(holes), "No missing region")
    # At least one axis of the negative-space mask is bilateral. Arrows may
    # break pixel symmetry; their positions are directional evidence.
    mask = board == BACKGROUND
    _require(np.array_equal(mask, mask[::-1]) or np.array_equal(mask, mask[:, ::-1]),
             "The missing-space mask must be bilateral")
    arrow_cells = set(map(tuple, np.argwhere((board != BACKGROUND) & (board != SCAFFOLD)).tolist()))
    arrows, pins, counts = [], {}, {}
    for component in _components(arrow_cells, KING):
        colors = {int(board[p]) for p in component}
        _require(len(component) == 3 and len(colors) == 1, "Malformed arrow component")
        color = next(iter(colors))
        _require(color in PAYLOAD, "Invalid arrow color")
        tip, direction = _decode_arrow(component)
        endpoint = tuple(tip[a] + DISTANCE * direction[a] for a in (0, 1))
        _require(endpoint in holes, "Every shot must end in missing space")
        _require(endpoint not in pins or pins[endpoint] == color,
                 "Different piece colors cannot claim the same cell")
        pins[endpoint] = color
        counts[endpoint] = counts.get(endpoint, 0) + 1
        arrows.append((color, tip, direction, endpoint, component))
    _require(bool(arrows), "Missing arrows")
    exterior = grid.copy()
    exterior[top:bottom, left:right] = BACKGROUND
    loose = set(map(tuple, np.argwhere(exterior != BACKGROUND).tolist()))
    pieces = []
    for component in _components(loose):
        colors = {int(exterior[p]) for p in component}
        _require(len(colors) == 1, "Each exterior piece is monochrome")
        color = next(iter(colors))
        _require(color in PAYLOAD, "Invalid exterior piece color")
        pieces.append((color, _normalize(component)))
    pieces.sort()
    _require(2 <= len(pieces) <= 7, "Need 2..7 inventory pieces")
    _require(len({c for c, _ in pieces}) == len(pieces),
             "Within a scene each piece has its own color")
    _require(sum(len(s) for _, s in pieces) == len(holes), "Inventory area does not match missing area")
    _require(set(pins.values()) <= {c for c, _ in pieces}, "Shot has no matching inventory color")
    return {"grid": grid, "board": board, "bbox": (top, left, bottom, right),
            "holes": holes, "arrows": arrows, "pieces": pieces, "pins": pins,
            "counts": counts, "collisions": frozenset(p for p, n in counts.items() if n >= 2)}


def _assemblies(board, pieces, pins, collisions):
    """Return at most two distinct (assembled, impacted) colored completions.

    A piece may cover an addressed cell only if its ORIGINAL color matches.
    Collision recoloring occurs after placement, from the unchanged shot list.
    The finite inventory and negative space participate in the SAME search as
    projected constraints; this is not a call to either parent's solver.
    """
    holes = tuple(map(tuple, np.argwhere(board == BACKGROUND).tolist()))
    if sum(len(s) for _, s in pieces) != len(holes):
        return (), 0
    indices = {p: i for i, p in enumerate(holes)}
    by_cell = [[] for _ in holes]
    for index, (color, shape) in enumerate(pieces):
        sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
        for top in range(board.shape[0] - sh + 1):
            for left in range(board.shape[1] - sw + 1):
                cells = tuple((top + r, left + c) for r, c in shape)
                if not all(p in indices and pins.get(p, color) == color for p in cells):
                    continue
                ids = [indices[p] for p in cells]
                bits = sum(1 << i for i in ids)
                impact = COLLISION if any(p in collisions for p in cells) else color
                before = sum((color + 1) << (4 * i) for i in ids)
                after = sum((impact + 1) << (4 * i) for i in ids)
                move = (index, bits, before, after)
                for i in ids:
                    by_cell[i].append(move)
    nodes = 0

    @lru_cache(maxsize=MAX_SEARCH_NODES)
    def search(remaining, unused):
        nonlocal nodes
        nodes += 1
        if nodes > MAX_SEARCH_NODES:
            raise ValueError("Assembly search bound exceeded; uniqueness unproved")
        if remaining == 0:
            return ((0, 0),) if unused == 0 else ()
        best, bits = None, remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            options = [m for m in by_cell[i]
                       if (unused & (1 << m[0])) and (m[1] & remaining) == m[1]]
            if not options:
                return ()
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for index, mask, before, after in best:
            for a, b in search(remaining ^ mask, unused ^ (1 << index)):
                answers.add((a | before, b | after))
                if len(answers) == 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search((1 << len(holes)) - 1, (1 << len(pieces)) - 1)
    results = []
    for codes_pair in codes:
        grids = []
        for code in codes_pair:
            result = board.copy()
            for p, i in indices.items():
                result[p] = ((code >> (4 * i)) & 15) - 1
            grids.append(result)
        results.append(tuple(grids))
    return tuple(results), nodes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve a scene using only its visible pixels and fixed family conventions."""
    scene = _read_scene(input_grid)
    answers, _ = _assemblies(scene["board"], scene["pieces"], scene["pins"], scene["collisions"])
    _require(len(answers) == 1, "Projected inventory must have one colored assembly and impact result")
    return answers[0][1].copy()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _hole_mask(rng, kind):
    if kind == 0:
        h, w = int(rng.integers(3, 7)), int(rng.integers(4, 8))
        return {(r, c) for r in range(h) for c in range(w)}, (h, w)
    if kind == 1:
        h, w = int(rng.integers(4, 8)), int(rng.integers(5, 8))
        cells = {(r, c) for r in range(h) for c in range(w)}
        # Independent upper/lower corner notches keep left-right symmetry.
        for rows in (range(int(rng.integers(1, 3))),
                     range(h - int(rng.integers(0, 3)), h)):
            depth = int(rng.integers(1, min(3, (w - 1) // 2) + 1))
            for r in rows:
                for c in range(depth):
                    cells.discard((r, c))
                    cells.discard((r, w - 1 - c))
        return cells, (h, w)
    h, w = int(rng.integers(5, 8)), int(rng.integers(4, 8))
    divider = int(rng.integers(2, h - 1))
    cells = {(r, c) for r in range(h) for c in range(w) if r != divider}
    return cells, (h, w)


def _repeat_shape(rng):
    cells = {(0, 0)}
    for _ in range(int(rng.integers(1, 5))):
        options = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in ORTHOGONAL} - cells)
        cells.add(_choose(rng, options))
    return _normalize(cells)


def _partition(rng, holes, count):
    """Two repeated procedural pieces plus a connected region-grown remainder."""
    shape = _repeat_shape(rng)
    placements = []
    for r, c in sorted(holes):
        cells = frozenset((r + a, c + b) for a, b in shape)
        if cells <= holes:
            placements.append(cells)
    if len(placements) < 2:
        return None
    for _ in range(70):
        a, b = rng.choice(len(placements), size=2, replace=False)
        first, second = placements[int(a)], placements[int(b)]
        if first & second:
            continue
        remaining = holes - first - second
        components = _components(remaining)
        if not components or len(components) > count - 2 or len(remaining) < count - 2:
            continue
        seeds = [_choose(rng, sorted(c)) for c in components]
        others = sorted(remaining - set(seeds))
        if count - 2 > len(seeds):
            for index in rng.choice(len(others), size=count - 2 - len(seeds), replace=False):
                seeds.append(others[int(index)])
        regions = [{p} for p in seeds]
        free = set(remaining) - set(seeds)
        while free:
            frontier = [(i, p) for i, region in enumerate(regions)
                        for p in sorted({(r + dr, c + dc) for r, c in region
                                         for dr, dc in ORTHOGONAL} & free)]
            if not frontier:
                return None
            i, p = _choose(rng, frontier)
            regions[i].add(p)
            free.remove(p)
        return [first, second] + [frozenset(r) for r in regions]
    return None


def _arrow_options(cells, holes, board_shape):
    options = []
    h, w = board_shape
    for endpoint in sorted(cells):
        for direction in DIRECTIONS:
            tip = tuple(endpoint[i] - DISTANCE * direction[i] for i in (0, 1))
            motif = _arrow(tip, direction)
            if all(1 <= r < h - 1 and 1 <= c < w - 1 and (r, c) not in holes for r, c in motif):
                options.append((endpoint, tip, direction, motif))
    return options


def _install_arrows(rng, tiles, holes, board_shape, unpinned, convergence, second_collision):
    options = [_arrow_options(tile, holes, board_shape) for tile in tiles]
    arrows, occupied = [], set()

    def add(i, option):
        nonlocal occupied
        endpoint, tip, direction, motif = option
        arrows.append((i, endpoint, tip, direction, motif))
        occupied |= motif

    def group(i, number):
        choices = []
        for p in sorted(tiles[i]):
            choices.extend(combinations([o for o in options[i] if o[0] == p], number))
        for j in rng.permutation(len(choices)):
            selected = choices[int(j)]
            used = set(occupied)
            for option in selected:
                if not _separated(option[3], used):
                    break
                used |= option[3]
            else:
                for option in selected:
                    add(i, option)
                return True
        return False

    # Tile 0 and tile 1 share a shape but DIFFERENT colors. The former is
    # collided; the latter receives two distinct singles and must keep its color.
    if not group(0, convergence):
        return None
    pairs = [(a, b) for a, b in combinations(options[1], 2) if a[0] != b[0]]
    for j in rng.permutation(len(pairs)):
        a, b = pairs[int(j)]
        if _separated(a[3], occupied) and _separated(b[3], occupied | a[3]):
            add(1, a)
            add(1, b)
            break
    else:
        return None
    other = [i for i in range(2, len(tiles)) if i not in unpinned]
    if second_collision:
        if not other or not group(other[0], 2):
            return None
        other = other[1:]
    for i in rng.permutation(other):
        i = int(i)
        choices = [o for o in options[i] if _separated(o[3], occupied)]
        if not choices:
            return None
        add(i, _choose(rng, choices))
    return arrows, occupied, options


def _scatter(rng, board, tiles, colors):
    h, w = board.shape
    for attempt in range(14):
        ch = int(rng.integers(max(24, h + 6), 31)) if attempt < 10 else 30
        cw = int(rng.integers(max(24, w + 6), 31)) if attempt < 10 else 30
        top = int(rng.integers(ch - h + 1))
        left = int(rng.integers(cw - w + 1))
        canvas = np.zeros((ch, cw), dtype=np.int8)
        canvas[top:top + h, left:left + w] = board
        blocked = np.zeros((ch, cw), dtype=bool)
        blocked[max(0, top - 1):min(ch, top + h + 1),
                max(0, left - 1):min(cw, left + w + 1)] = True
        ordered = sorted(range(len(tiles)), key=lambda i: (-len(tiles[i]), i))
        for i in ordered:
            shape = _normalize(tiles[i])
            sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
            choices = [(r, c) for r in range(ch - sh + 1) for c in range(cw - sw + 1)
                       if all(not blocked[r + a, c + b] for a, b in shape)]
            if not choices:
                break
            r, c = _choose(rng, choices)
            for a, b in shape:
                rr, cc = r + a, c + b
                canvas[rr, cc] = colors[i]
                for dr, dc in KING:
                    if 0 <= rr + dr < ch and 0 <= cc + dc < cw:
                        blocked[rr + dr, cc + dc] = True
        else:
            return canvas
    return None


def _pair(rng, palette, test, convergence, second_collision, kind):
    for _ in range(MAX_PAIR_ATTEMPTS):
        count = int(rng.integers(5, 8) if test else rng.integers(4, 7))
        local_holes, extent = _hole_mask(rng, kind)
        if len(local_holes) < count + 5:
            continue
        tiles = _partition(rng, frozenset(local_holes), count)
        if tiles is None:
            continue
        # Generous, independently varied board axes admit several orientations.
        h = int(rng.integers(max(17, extent[0] + 12), 23))
        w = int(_choose(rng, [n for n in range(max(17, extent[1] + 12), 23)
                              if n % 2 == extent[1] % 2]))
        dr = (h - extent[0]) // 2
        dc = (w - extent[1]) // 2
        tiles = [frozenset((r + dr, c + dc) for r, c in t) for t in tiles]
        holes = frozenset().union(*tiles)
        unpinned = set(map(int, rng.choice(range(2, count), size=2 if test else 1, replace=False)))
        installed = _install_arrows(rng, tiles, holes, (h, w), unpinned, convergence, second_collision)
        if installed is None:
            continue
        arrows, occupied, options = installed
        colors = list(map(int, palette[:count]))
        pieces = [(colors[i], _normalize(tile)) for i, tile in enumerate(tiles)]
        true_before = np.full((h, w), SCAFFOLD, dtype=np.int8)
        board = true_before.copy()
        for i, tile in enumerate(tiles):
            for p in tile:
                board[p] = BACKGROUND
                true_before[p] = colors[i]

        def render_arrows():
            visible = board.copy()
            before = true_before.copy()
            pins, counts = {}, {}
            for i, endpoint, _, _, motif in arrows:
                pins[endpoint] = colors[i]
                counts[endpoint] = counts.get(endpoint, 0) + 1
                for p in motif:
                    visible[p] = colors[i]
                    before[p] = colors[i]
            collisions = frozenset(p for p, n in counts.items() if n >= 2)
            return visible, before, pins, collisions

        # If a pin has several possible positions inside a piece, use the
        # visible geometry to add a distinguishing single shot. Never address
        # the withheld pieces: these must be placed by inventory reasoning.
        for _ in range(MAX_ARROWS):
            visible, before, pins, collisions = render_arrows()
            try:
                answers, _ = _assemblies(visible, pieces, pins, collisions)
            except ValueError:
                break
            if len(answers) == 1:
                if not np.array_equal(answers[0][0], before):
                    break
                expected = before.copy()
                for i, tile in enumerate(tiles):
                    if tile & collisions:
                        for p in tile:
                            expected[p] = COLLISION
                if not np.array_equal(expected, answers[0][1]):
                    break
                canvas = _scatter(rng, visible, tiles, colors)
                if canvas is None:
                    break
                turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
                if reflect:
                    canvas, expected = canvas[:, ::-1], expected[:, ::-1]
                return {"input": np.rot90(canvas, turns).copy(),
                        "output": np.rot90(expected, turns).copy()}
            if not answers or len(arrows) >= MAX_ARROWS:
                break
            wrong = next((a for a, _ in answers if not np.array_equal(a, before)), None)
            if wrong is None:
                break
            choices = [(i, option) for i, tile in enumerate(tiles) if i not in unpinned
                       for option in options[i]
                       if option[0] not in pins and before[option[0]] != wrong[option[0]]
                       and _separated(option[3], occupied)]
            if not choices:
                break
            i, option = _choose(rng, choices)
            endpoint, tip, direction, motif = option
            arrows.append((i, endpoint, tip, direction, motif))
            occupied |= motif
    raise RuntimeError("Bounded construction could not find a uniquely addressed jigsaw")


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("Example counts must be integers")
        if not low <= int(value) <= high:
            raise ValueError("Example count outside allowed range")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.permutation(PAYLOAD)))
    exchange = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    # Two essential demonstrations distinguish two/three-way convergence from
    # two DISTINCT single endpoints on one piece. Their order is shuffled.
    training_convergence = [2, 3] + [int(rng.integers(2, 4)) for _ in range(num_train - 2)]
    rng.shuffle(training_convergence)
    for split, number in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(number):
            test = split == "test"
            convergence = int(rng.integers(2, 4)) if test else training_convergence[index]
            second_collision = bool(test and rng.random() < .45)
            kind = int(rng.integers(3))
            if split == "train" and index < 2:
                # Swap the fates of the same two original piece colors across
                # the essential demonstrations. A literal-color burn shortcut
                # must disagree with a visible training answer.
                colors = palette if index == 0 else (palette[1], palette[0]) + palette[2:]
            elif exchange:
                colors = tuple(map(int, rng.permutation(PAYLOAD)))
            else:
                offset = int(rng.integers(len(palette)))
                colors = palette[offset:] + palette[:offset]
            for _ in range(6):
                pair = _pair(rng, colors, test, convergence, second_collision, kind)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not produce distinct inputs")
    return puzzle


def _analyze_pair(pair):
    scene = _read_scene(pair["input"])
    output = _grid(pair["output"])
    answers, nodes = _assemblies(scene["board"], scene["pieces"], scene["pins"], scene["collisions"])
    _require(len(answers) == 1, "Assembly is not unique")
    before, expected = answers[0]
    _require(np.array_equal(output, expected), "Incorrect output")
    fixed = scene["board"] != BACKGROUND
    _require(np.array_equal(output[fixed], scene["board"][fixed]), "Fixed cells changed")
    _require(not np.any(output == BACKGROUND), "Unfilled board cells")
    pinned_colors = set(scene["pins"].values())
    burned, distinct_singles = set(), set()
    for color, shape in scene["pieces"]:
        cells = frozenset(map(tuple, np.argwhere(before == color).tolist())) - frozenset(
            p for a in scene["arrows"] if a[0] == color for p in a[4])
        _require(_normalize(cells) == shape, "Piece was not translated intact")
        hits = [p for p, c in scene["pins"].items() if c == color]
        if any(scene["counts"][p] >= 2 for p in hits):
            burned.add(color)
            _require(all(output[p] == COLLISION for p in cells), "Collision did not recolor whole piece")
        else:
            _require(all(output[p] == color for p in cells), "Uncollided piece changed color")
            if len(hits) >= 2:
                distinct_singles.add(color)
    shapes = [s for _, s in scene["pieces"]]
    interchangeable = [(a, b) for a, b in combinations(scene["pieces"], 2)
                       if a[1] == b[1] and (a[0] not in burned or b[0] not in burned)]
    _require(bool(interchangeable), "Need an answer-changing interchangeable pair without arrow constraints")
    _require(any(len(s) >= 2 for _, s in scene["pieces"]), "No shape propagation")
    _require(bool(distinct_singles) and bool(burned), "Need collision versus separate-single evidence")
    _require(any(len({int(before[p]) for p in comp}) >= 2
                 for comp in _components(scene["holes"])),
             "Need several inventory pieces sharing a connected missing region")
    # A simple endpoint-painting shortcut necessarily leaves an unaddressed
    # tile empty, whereas the inventory completes it.
    endpoint_only = scene["board"].copy()
    for p, color in scene["pins"].items():
        endpoint_only[p] = COLLISION if p in scene["collisions"] else color
    _require(not np.array_equal(endpoint_only, output), "Assembly contribution is vacuous")
    return {"scene": scene, "before": before, "nodes": nodes,
            "unpinned": len(scene["pieces"]) - len(pinned_colors),
            "burned": burned, "distinct_singles": distinct_singles,
            "shapes": shapes, "gap_components": len(_components(scene["holes"]))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, taught, burnt_train, kept_train = set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            analysis = _analyze_pair({"input": inp, "output": out})
            _require(4 <= len(analysis["scene"]["pieces"]) <= 7, "Generated inventory range")
            _require(analysis["unpinned"] == (2 if split == "test" else 1), "Missing inventory reasoning shift")
            if split == "train":
                taught.update(n for n in analysis["scene"]["counts"].values() if n >= 2)
                burnt_train |= analysis["burned"]
                kept_train |= analysis["distinct_singles"]
    _require({2, 3} <= taught, "Demonstrate both two- and three-way endpoint convergence")
    _require(len(burnt_train & kept_train) >= 2,
             "Two colors must each appear both collided and preserved in training")


def _fixture():
    """Small hand-constructed regression, not used to generate puzzles."""
    board = np.full((19, 18), SCAFFOLD, dtype=np.int8)
    board[8:10, 7:11] = BACKGROUND
    arrows = [(2, (8, 2), (0, 1)), (2, (13, 7), (-1, 0)),
              (3, (8, 14), (0, -1)), (3, (4, 10), (1, 0))]
    for color, tip, direction in arrows:
        for p in _arrow(tip, direction):
            board[p] = color
    inp = np.zeros((23, 26), dtype=np.int8)
    inp[1:20, 1:19] = board
    inp[1:3, 22:24] = 2
    inp[7:9, 22:24] = 3
    out = board.copy()
    out[8:10, 7:9] = COLLISION
    out[8:10, 9:11] = 3
    return inp, out


def self_test():
    """Stream 42 seeds; retain only sets/counters, never a trace or sample bank."""
    inp, expected = _fixture()
    original = inp.copy()
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(inp, original)
    assert np.all(expected[8, 3:7] == SCAFFOLD), "Paths are not painted"
    for turns in range(4):
        for reflected in (False, True):
            a, b = (inp[:, ::-1], expected[:, ::-1]) if reflected else (inp, expected)
            assert np.array_equal(solve(np.rot90(a, turns)), np.rot90(b, turns))
    # Removing projected constraints makes the two equal shapes swappable.
    scene = _read_scene(inp)
    alternatives, _ = _assemblies(scene["board"], scene["pieces"], {}, frozenset())
    assert len(alternatives) == 2
    assert not np.array_equal(alternatives[0][0], alternatives[1][0])
    for direction in DIRECTIONS:
        assert _decode_arrow(_arrow((10, 10), direction)) == ((10, 10), direction)
    signatures, input_shapes, board_shapes, piece_shapes = set(), set(), set(), set()
    directions, piece_counts, gap_counts = set(), set(), set()
    normalized_layouts, convergences = set(), set()
    total = max_nodes = max_arrows = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair)
                s = a["scene"]
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                digest.update(str(pair["input"].shape).encode())
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
                input_shapes.add(pair["input"].shape)
                board_shapes.add(pair["output"].shape)
                piece_shapes.update(a["shapes"])
                piece_counts.add(len(s["pieces"]))
                gap_counts.add(a["gap_components"])
                directions.update(arrow[2] for arrow in s["arrows"])
                convergences.update(s["counts"].values())
                normalized_layouts.add((s["board"].shape, (s["board"] == BACKGROUND).tobytes()))
                max_nodes = max(max_nodes, a["nodes"])
                max_arrows = max(max_arrows, len(s["arrows"]))
                total += 1
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42
    assert len(input_shapes) >= 25 and len(board_shapes) >= 15
    assert len(piece_shapes) >= 60 and len(normalized_layouts) >= 60
    assert directions == set(DIRECTIONS) and piece_counts == {4, 5, 6, 7}
    assert {1, 2} <= gap_counts and {1, 2, 3} <= convergences
    # Palette roles 0/1/5 remain fixed; payload and address colors permute together.
    mapping = np.arange(10, dtype=np.int8)
    for a, b in zip(PAYLOAD, PAYLOAD[::-1]):
        mapping[a] = b
    assert np.array_equal(solve(mapping[inp]), mapping[expected])
    for a, b in ((2, 1), (10, 4)):
        p = generate_puzzle(120, a, b)
        validate_generated_puzzle(p)
        q = generate_puzzle(120, a, b)
        assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                   for k in ("input", "output"))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    broken = inp.copy()
    broken[1, 22] = BACKGROUND
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing inventory cell accepted")
    return {"seeds": 42, "pairs": total, "distinct_puzzles": len(signatures),
            "input_shapes": len(input_shapes), "board_shapes": len(board_shapes),
            "oriented_piece_shapes": len(piece_shapes), "negative_space_layouts": len(normalized_layouts),
            "piece_counts": sorted(piece_counts), "gap_component_counts": sorted(gap_counts),
            "directions": len(directions), "endpoint_multiplicities": sorted(convergences),
            "maximum_search_nodes": max_nodes, "maximum_arrows": max_arrows}
