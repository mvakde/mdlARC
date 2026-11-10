"""Keyed deletion in a communicating tank of half-submerged floats.

Only NumPy and the standard library are used.  No examples, seeds or generation
metadata are consulted by solve().  See task_spec.md for the universal physics.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return grid.astype(np.int8, copy=True)


def _parts(mask):
    """Four-connected components, represented as coordinate lists."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        parts.append(cells)
    return parts


def _decode(input_grid):
    """Read the liquid key, selectors, grounded tank, and complete rectangles.

    A block is (top, left, height, width, color).  Empty column gaps make
    repeated identities unambiguous without latent joints or hidden endpoints.
    """
    grid = _grid(input_grid)
    h, w = grid.shape
    if h < 10 or w < 7:
        raise ValueError("tank too small")
    bg, water = int(grid[2, 0]), int(grid[0, 0])
    if bg == water or not np.all(grid[2] == bg):
        raise ValueError("missing blank separator")
    if not np.all(grid[:2, :2] == water) or not np.all(grid[1, 2:] == bg):
        raise ValueError("liquid key must be a 2 by 2 square at upper left")
    if not np.all(grid[0, 2:4] == bg):
        raise ValueError("liquid key must be separated from selectors")
    controls = tuple(map(int, grid[0, 4:][grid[0, 4:] != bg]))
    if not 1 <= len(controls) <= 2 or len(set(controls)) != len(controls) or water in controls:
        raise ValueError("expected one or two distinct deletion swatches")
    wall_columns = np.flatnonzero(grid[3] != bg)
    if len(wall_columns) != 2:
        raise ValueError("tank mouth must show its two walls")
    left, right = map(int, wall_columns)
    terrain = int(grid[3, left])
    if right - left < 4 or terrain in (bg, water, *controls) or grid[3, right] != terrain:
        raise ValueError("invalid tank identity or width")
    if not np.all(grid[3:, [left, right]] == terrain):
        raise ValueError("tank side wall is incomplete")
    if not np.all(grid[3:, :left] == bg) or not np.all(grid[3:, right + 1:] == bg):
        raise ValueError("scene must lie inside the tank")
    floors = []
    for c in range(left + 1, right):
        rows = np.flatnonzero(grid[4:, c] == terrain) + 4
        if not len(rows) or not np.array_equal(rows, np.arange(int(rows[0]), h)):
            raise ValueError("terrain must be grounded with no overhangs")
        floors.append(int(rows[0]))
    floors = np.asarray(floors, dtype=int)
    blocks = []
    body = grid[4:, left + 1:right]
    for color in map(int, np.unique(body)):
        if color in (bg, water, terrain):
            continue
        for cells in _parts(body == color):
            r0, r1 = min(r for r, c in cells) + 4, max(r for r, c in cells) + 4
            c0 = min(c for r, c in cells) + left + 1
            c1 = max(c for r, c in cells) + left + 1
            bh, bw = r1 - r0 + 1, c1 - c0 + 1
            if bh < 2 or bh % 2 or len(cells) != bh * bw:
                raise ValueError("floats must be complete rectangles of even height")
            blocks.append((r0, c0, bh, bw, color))
    blocks.sort(key=lambda b: b[1])
    if not blocks:
        raise ValueError("input needs at least one float")
    for a, b in zip(blocks, blocks[1:]):
        if a[1] + a[3] >= b[1]:
            raise ValueError("floats need at least one empty column between footprints")
    volume = int(np.count_nonzero(body == water))
    if volume < 1:
        raise ValueError("tank needs liquid, apart from its key")
    return grid, bg, water, terrain, controls, left, right, floors, blocks, volume


def _equilibrium(decoded, controls=None):
    """Joint volume/displacement equation, followed by rigid translation.

    V + D = sum(F[c] - s), where s is the water surface, F is each
    column's first terrain row, and D is HALF the surviving float area.
    """
    grid, bg, water, terrain, command, left, right, floors, blocks, volume = decoded
    command = set(command if controls is None else controls)
    survivors = [b for b in blocks if b[4] not in command]
    displacement = sum(b[2] // 2 * b[3] for b in survivors)
    width = right - left - 1
    numerator = int(floors.sum()) - volume - displacement
    surface, remainder = divmod(numerator, width)
    if remainder or surface < 4 or surface >= int(floors.min()):
        raise ValueError("volume does not give a complete shared horizontal level")
    final = []
    for row, col, height, span, color in survivors:
        top = surface - height // 2
        if top < row:
            raise ValueError("input floats must begin at or above their final heights")
        if surface + height // 2 >= int(floors[col - left - 1:col - left - 1 + span].min()):
            raise ValueError("a float needs liquid clearance beneath it")
        final.append((top, col, height, span, color))
    out = np.full(grid.shape, bg, dtype=np.int8)
    out[:3] = grid[:3]
    out[3:][grid[3:] == terrain] = terrain
    for c, floor in enumerate(floors, left + 1):
        out[surface:int(floor), c] = water
    for row, col, height, span, color in final:
        out[row:row + height, col:col + span] = color
    if np.count_nonzero(out[4:, left + 1:right] == water) != volume:
        raise AssertionError("liquid accounting failed")
    return out, surface, final


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the visible input and the family's fixed conventions."""
    return _equilibrium(_decode(input_grid))[0]


def _gaps(rng, total, bins):
    """A bounded random weak composition (no rejection or hidden layout bank)."""
    return rng.multinomial(total, np.full(bins, 1 / bins)).tolist()


def _widths(rng, count, budget, maximum=5):
    widths = [1] * count
    for _ in range(int(rng.integers(0, max(0, budget - count) + 1))):
        eligible = [i for i in range(count) if widths[i] < maximum]
        if not eligible:
            break
        widths[int(rng.choice(eligible))] += 1
    return widths


def _palette(rng, bg, water):
    colors = [c for c in range(10) if c not in (bg, water)]
    rng.shuffle(colors)
    return colors


def _container(rng, width, maximum_half, extra_surface_row=0, minimum_surface=4):
    minimum_surface = max(maximum_half + 5, minimum_surface)
    minimum_height = max(15, 2 * maximum_half + 10 + extra_surface_row,
                         minimum_surface + maximum_half + 4 + extra_surface_row)
    h = int(rng.integers(minimum_height, 31))
    surface = int(rng.integers(minimum_surface, h - maximum_half - 2 - extra_surface_row))
    minimum_floor = surface + extra_surface_row + maximum_half + 1
    style = int(rng.integers(4))
    if style == 0:
        floors = np.full(width, int(rng.integers(minimum_floor, h)))
    elif style == 1:
        floors = rng.integers(minimum_floor, h, size=width)
    elif style == 2:
        floors = np.clip(np.rint(np.linspace(minimum_floor, h - 1, width)), minimum_floor, h - 1).astype(int)
        if rng.integers(2):
            floors = floors[::-1].copy()
    else:
        values = [int(rng.integers(minimum_floor, h))]
        for _ in range(1, width):
            values.append(int(np.clip(values[-1] + rng.integers(-2, 3), minimum_floor, h - 1)))
        floors = np.array(values, dtype=int)
    margin_budget = 30 - width - 2
    ml = int(rng.integers(0, min(3, margin_budget) + 1))
    mr = int(rng.integers(0, min(3, margin_budget - ml) + 1))
    return h, width + ml + mr + 2, ml, floors, surface


def _scene(rng, bg, water, terrain, widths, halves, colors, width, extra_surface_row=0, minimum_surface=4):
    h, canvas_width, left, floors, surface = _container(
        rng, width, max(halves), extra_surface_row, minimum_surface)
    right = left + width + 1
    grid = np.full((h, canvas_width), bg, dtype=np.int8)
    grid[:2, :2] = water
    grid[3:, [left, right]] = terrain
    for c, floor in enumerate(floors, left + 1):
        grid[int(floor):, c] = terrain
    spare = width - sum(widths) - (len(widths) - 1)
    gaps = _gaps(rng, spare, len(widths) + 1)
    col = left + 1 + gaps[0]
    blocks = []
    # Reordering widths, heights and identities prevents a fixed selected slot.
    order = rng.permutation(len(widths))
    for j, i in enumerate(order):
        span, half, color = widths[i], halves[i], colors[i]
        top = int(rng.integers(4, surface - half))
        grid[top:top + 2 * half, col:col + span] = color
        blocks.append((top, col, 2 * half, span, color))
        col += span + 1 + gaps[j + 1]
    return grid, left, right, floors, surface, blocks


def _add_source(rng, grid, bg, water, left, right, volume):
    eligible = np.zeros(grid.shape, dtype=bool)
    eligible[4:, left + 1:right] = grid[4:, left + 1:right] == bg
    points = np.argwhere(eligible)
    if volume > len(points):
        raise AssertionError("insufficient visible supply space")
    style = int(rng.integers(5))
    jitter = rng.random(len(points))
    anchor = points[int(rng.integers(len(points)))]
    dr, dc = np.abs(points[:, 0] - anchor[0]), np.abs(points[:, 1] - anchor[1])
    if style == 0:  # scattered droplets
        scores = jitter
    elif style == 1:  # compact patch
        scores = dr + dc + jitter
    elif style == 2:  # horizontal supply band
        scores = 4 * dr + .15 * dc + jitter
    elif style == 3:  # narrow vertical supply
        scores = 4 * dc + .15 * dr + jitter
    else:  # already low liquid, which the descending floats may push upward
        scores = -points[:, 0] + jitter
    chosen = points[np.argsort(scores)[:volume]]
    grid[chosen[:, 0], chosen[:, 1]] = water


def _command(rng, grid, bg, colors):
    grid[0, 4:] = bg
    positions = rng.choice(np.arange(4, grid.shape[1]), size=len(colors), replace=False)
    for c, color in zip(positions, colors):
        grid[0, c] = color


def _pair(grid):
    return {"input": grid, "output": solve(grid)}


def _teaching_pair(rng, bg, water):
    """Matched scenes: deleting W cells of displacement lowers the level by one.

    Three deletable floats use two identities (one repeated).  The two survivors
    have unequal heights.  Both selectors are present in the deletion scene.
    """
    choices = [(half, total_width) for half in range(2, 5)
               for total_width in range(4, 9)
               if 14 <= half * total_width <= 24
               and half * total_width >= total_width + 6]
    half, selected_width = choices[int(rng.integers(len(choices)))]
    width = half * selected_width
    # Three positive spans with exactly the chosen total width.
    cuts = sorted(map(int, rng.choice(np.arange(1, selected_width), size=2, replace=False)))
    widths = [cuts[0], cuts[1] - cuts[0], selected_width - cuts[1]]
    widths += _widths(rng, 2, min(8, width - selected_width - 4), maximum=4)
    survivor_halves = list(map(int, rng.choice(np.arange(1, 5), size=2, replace=False)))
    halves = [half] * 3 + survivor_halves
    terrain, selected_a, selected_b, survivor_a, survivor_b, absent = _palette(rng, bg, water)[:6]
    colors = [selected_a, selected_a, selected_b, survivor_a, survivor_b]
    grid, left, right, floors, surface, blocks = _scene(
        rng, bg, water, terrain, widths, halves, colors, width, extra_surface_row=1)
    displacement = sum(b[2] // 2 * b[3] for b in blocks)
    volume = int(floors.sum()) - width * surface - displacement
    _add_source(rng, grid, bg, water, left, right, volume)
    keep, delete = grid.copy(), grid.copy()
    _command(rng, keep, bg, [absent])
    _command(rng, delete, bg, [selected_a, selected_b])
    return [_pair(keep), _pair(delete)]


def _random_pair(rng, bg, water, test=False):
    count = int(rng.integers(6, 8)) if test else int(rng.integers(2, 6))
    width = int(rng.integers(max(10, 2 * count + 1), 27))
    widths = _widths(rng, count, width - count + 1, maximum=5)
    halves = list(map(int, rng.integers(1, 7 if test else 6, size=count)))
    # Tall narrow sticks and broad shallow rafts are both legal rectangles.
    if test:
        halves[0], halves[1] = 1, int(rng.integers(3, 7))
    palette = _palette(rng, bg, water)
    terrain, absent = palette[:2]
    identities = palette[2:2 + int(rng.integers(2, min(5, count) + 1))]
    colors = list(map(int, rng.choice(identities, size=count)))
    if test:
        # At least three survivors and a real deletion; reused identities remain
        # possible without accidentally deleting the whole scene.
        colors[0] = identities[0]
        for j in range(1, 4):
            colors[j] = identities[1]
        candidates = sorted(c for c in set(colors) if colors.count(c) <= count - 3)
        doubles = [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:]
                   if colors.count(a) + colors.count(b) <= count - 3]
        if doubles and rng.random() < .5:
            selected = list(doubles[int(rng.integers(len(doubles)))])
        else:
            selected = [int(rng.choice(candidates))]
            if rng.random() < .5:
                selected.append(absent)
    else:
        selected = [absent] if rng.random() < .2 else [int(rng.choice(sorted(set(colors))))]
        remaining = sorted(set(colors) - set(selected))
        if rng.random() < .45:
            selected.append(int(rng.choice(remaining)) if remaining and rng.random() < .7 else absent)
        selected = list(dict.fromkeys(selected))
    total_half_area = sum(n * m for n, m in zip(halves, widths))
    surviving_half_area = sum(n * m for n, m, c in zip(halves, widths, colors) if c not in selected)
    # Input free area is sum(F)-4W-2*total_half_area.  Output liquid is
    # sum(F)-sW-surviving_half_area.  Leave at least W spare source cells,
    # including when many large input blocks will be deleted in the output.
    minimum_surface = 5 + (2 * total_half_area - surviving_half_area + width - 1) // width
    grid, left, right, floors, surface, blocks = _scene(
        rng, bg, water, terrain, widths, halves, colors, width, minimum_surface=minimum_surface)
    displacement = sum(b[2] // 2 * b[3] for b in blocks if b[4] not in selected)
    volume = int(floors.sum()) - width * surface - displacement
    _add_source(rng, grid, bg, water, left, right, volume)
    _command(rng, grid, bg, selected)
    return _pair(grid)


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(value) or not low <= int(value) <= high:
            raise ValueError("illegal pair count")
    num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    bg, water = map(int, rng.choice(10, size=2, replace=False))
    puzzle = {"train": _teaching_pair(rng, bg, water), "test": []}
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, target in (("train", num_train), ("test", num_test)):
        while len(puzzle[split]) < target:
            for _ in range(12):
                pair = _random_pair(rng, bg, water, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    decoded = _decode(pair["input"])
    inp, bg, water, terrain, controls, left, right, floors, blocks, volume = decoded
    out = _grid(pair["output"])
    expected, surface, final = _equilibrium(decoded)
    assert inp.shape == out.shape and np.array_equal(expected, out)
    assert np.array_equal(inp[:3], out[:3]), "legend and selectors must survive"
    assert np.array_equal(inp[3:] == terrain, out[3:] == terrain), "tank terrain changed"
    assert np.count_nonzero(out[4:, left + 1:right] == water) == volume
    selected = [b for b in blocks if b[4] in controls]
    survivors = [b for b in blocks if b[4] not in controls]
    final_mask = np.zeros(inp.shape, dtype=bool)
    for before, after in zip(survivors, final):
        r, c, height, span, color = after
        assert after[1:] == before[1:] and r >= before[0]
        assert r + height // 2 == surface, "a float is not half submerged"
        assert np.all(out[r:r + height, c:c + span] == color)
        final_mask[r:r + height, c:c + span] = True
    assert not any(np.any(out[4:] == b[4]) for b in selected)
    tank = np.zeros(inp.shape, dtype=bool)
    for c, floor in enumerate(floors, left + 1):
        tank[surface:int(floor), c] = True
    assert np.array_equal((out == water) & (np.indices(out.shape)[0] >= 4), tank & ~final_mask)
    # Open clearance below every float guarantees one shared communicating body.
    water_mask = out == water
    water_mask[:4] = False
    assert len(_parts(water_mask)) == 1, "output liquid must communicate"
    # Ignoring liquid support would drop each survivor onto the uneven dry floor.
    dry_tops = [int(floors[b[1] - left - 1:b[1] - left - 1 + b[3]].min()) - b[2] for b in survivors]
    assert all(dry > b[0] for dry, b in zip(dry_tops, final)), "water has no effect on float placement"
    # Twice-submerged area must equal total survivor area, independently of colors.
    submerged = sum(b[3] * (b[0] + b[2] - surface) for b in final)
    area = sum(b[2] * b[3] for b in survivors)
    assert 2 * submerged == area
    assert volume + submerged == int(floors.sum()) - (right - left - 1) * surface
    return {"shape": inp.shape, "background": bg, "water": water, "terrain": terrain,
            "surface": surface, "volume": volume, "blocks": len(blocks),
            "survivors": len(final), "deleted": len(selected),
            "commands": len(controls), "displacement": submerged,
            "halves": tuple(sorted({b[2] // 2 for b in blocks})),
            "widths": tuple(sorted({b[3] for b in blocks})),
            "floor_relief": int(np.ptp(floors)),
            "falls": tuple(sorted({a[0] - b[0] for b, a in zip(survivors, final)})),
            "source_components": len(_parts((inp[4:, left + 1:right] == water))),
            "source_rows": (int(np.argwhere(inp[4:, left + 1:right] == water)[:, 0].min()) + 4,
                            int(np.argwhere(inp[4:, left + 1:right] == water)[:, 0].max()) + 4)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, records = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            record = _inspect_pair(pair)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            roles.add((record["background"], record["water"]))
            records[split].append(record)
            if split == "test":
                assert record["blocks"] >= 6 and record["deleted"] >= 1 and record["survivors"] >= 3
            else:
                assert record["blocks"] <= 5
    assert len(roles) == 1
    # Locate the matched demonstrations by their visible lower scene, not order.
    witnessed = False
    for i, first in enumerate(puzzle["train"]):
        a = _decode(first["input"])
        if any(b[4] in a[4] for b in a[8]):
            continue
        for j, second in enumerate(puzzle["train"]):
            if i == j or not np.array_equal(np.asarray(first["input"])[1:], np.asarray(second["input"])[1:]):
                continue
            b = _decode(second["input"])
            removed = [x for x in b[8] if x[4] in b[4]]
            if len(removed) != 3 or len(b[4]) != 2:
                continue
            assert sum(x[2] // 2 * x[3] for x in removed) == a[6] - a[5] - 1
            assert records["train"][j]["surface"] == records["train"][i]["surface"] + 1
            before_final = {x[4]: x for x in _equilibrium(a)[2] if x[4] not in b[4]}
            after_final = _equilibrium(b)[2]
            assert len(before_final) == len(after_final) == 2
            assert all(x[0] == before_final[x[4]][0] + 1 for x in after_final)
            # Actual counterfactual: ignoring deletion changes surviving positions
            # and liquid positions, even if selected pixels were erased afterward.
            no_deletion, _, _ = _equilibrium(b, controls=())
            assert not np.array_equal(no_deletion[4:] == b[2], np.asarray(second["output"])[4:] == b[2])
            witnessed = True
    assert witnessed, "missing coupled counterfactual demonstrations"


def self_test():
    """Bounded 42-seed variation, invariants, replay, and count coverage.

    Retains only hashes and aggregate feature sets, never a gallery of arrays.
    Returns descriptive counts; these are authored checks, not external judging.
    """
    signatures, shapes, roles, volumes, surfaces = set(), set(), set(), set(), set()
    widths, halves, block_counts, reliefs, falls = set(), set(), set(), set(), set()
    features = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _inspect_pair(pair)
                shapes.add(f["shape"])
                roles.add((f["background"], f["water"]))
                volumes.add(f["volume"])
                surfaces.add(f["surface"])
                widths.update(f["widths"])
                halves.update(f["halves"])
                block_counts.add(f["blocks"])
                reliefs.add(f["floor_relief"])
                falls.update(f["falls"])
                features["pairs"] += 1
                features["deleted_pairs"] += f["deleted"] > 0
                features["multi_selector_pairs"] += f["commands"] > 1
                features["unequal_falls"] += len(f["falls"]) > 1
                features["disconnected_sources"] += f["source_components"] > 1
                features["low_sources"] += f["source_rows"][0] >= f["shape"][0] // 2
                # Input-only invariance to any renaming of color identities.
                permutation = np.random.default_rng(seed + 9000).permutation(10)
                assert np.array_equal(solve(permutation[np.asarray(pair["input"])]), permutation[np.asarray(pair["output"])])
                assert np.array_equal(solve(np.asarray(pair["input"]).tolist()), pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 35 and len(roles) >= 20
    assert len(volumes) >= 50 and len(surfaces) >= 8
    assert widths >= {1, 2, 3, 4, 5} and halves == set(range(1, 7))
    assert block_counts >= {2, 3, 4, 5, 6, 7}
    assert len(reliefs) >= 5 and len(falls) >= 8
    assert features["unequal_falls"] >= 80 and features["disconnected_sources"] >= 50
    assert features["low_sources"] >= 5
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(2026, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    # Regression from the count sweep: many large deletions must leave enough
    # input free space to draw the conserved output water volume visibly.
    generate_puzzle(443455678, 10, 1)
    generate_puzzle(443455678, 10, 4)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 0.5, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(shapes),
            "background_liquid_assignments": len(roles), "volumes": len(volumes),
            "surfaces": len(surfaces), "block_counts": sorted(block_counts),
            "half_heights": sorted(halves), "widths": sorted(widths),
            "floor_reliefs": sorted(reliefs), "fall_distances": sorted(falls),
            "features": dict(features), "independent_semantic_verification": False}
