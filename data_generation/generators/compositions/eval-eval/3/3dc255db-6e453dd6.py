"""Fuel-limited docking. Only NumPy and the Python standard library are used.

The public solver reads pixels only. Generation constructs labels from explicit
sampled cells and expenditure, then checks them against that input-only reader.
"""
from __future__ import annotations

import copy
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = {tuple(map(int, row)) for row in np.argwhere(mask)}
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _regions(walls):
    h, w = walls.shape
    enclosed, opened = [], []
    for cells in _components(~walls):
        target = opened if any(r in (0, h - 1) or c in (0, w - 1)
                               for r, c in cells) else enclosed
        target.append(cells)
    return enclosed, opened


def _leading(walls, enclosed):
    active = []
    for region in enclosed:
        contact = {r for r, c in region
                   if c == walls.shape[1] - 2 and walls[r, c + 1]}
        if not contact:
            continue
        rows = {r for r, c in region}
        _require(contact == rows, "partially leading tanks are outside the family")
        _require((max(rows) - min(rows) + 1) % 2 == 1,
                 "eligible tanks need a unique integer midpoint")
        active.append(region)
    _require(len(active) <= 1, "a body has more than one eligible tank")
    return active[0] if active else set()


def _scene(grid):
    """A rail and its blank side visibly identify all scene conventions."""
    candidates = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        h, w = a.shape
        for col in range(1, w - 1):
            material, bg = int(a[0, col]), int(a[0, -1])
            if (material != bg and np.all(a[:, col] == material)
                    and np.all(a[:, col + 1:] == bg)
                    and np.any(a[:, :col] == material)):
                candidates.append((a, turn, bg, material, col))
    _require(len(candidates) == 1, "rail and projection direction must be unique")
    a, turn, bg, material, rail = candidates[0]
    objects, owned = [], set()
    for cells in _components(a[:, :rail] == material):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        patch = a[r0:r1 + 1, c0:c1 + 1].copy()
        walls = np.zeros(patch.shape, dtype=bool)
        for r, c in cells:
            walls[r - r0, c - c0] = True
        _require(np.array_equal(walls, patch == material), "nested wall components")
        enclosed, opened = _regions(walls)
        active = _leading(walls, enclosed)
        fuel = {p for p in active if patch[p] != bg}
        colors = {int(patch[p]) for p in fuel}
        _require(len(colors) <= 1, "eligible fuel must have one color")
        center = ((min(r for r, c in active) + max(r for r, c in active)) // 2
                  if active else patch.shape[0] // 2)
        bbox = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        _require(not (owned & bbox), "overlapping object ownership")
        owned.update(bbox)
        objects.append(dict(row=r0, end_row=r1, col=c0, end_col=c1,
                            patch=patch, walls=walls, enclosed=enclosed, opened=opened,
                            active=active, fuel=fuel, count=len(fuel), center=center,
                            color=next(iter(colors)) if colors else None,
                            gap=rail - 1 - c1))
    _require(objects, "no wall objects")
    objects.sort(key=lambda o: o['row'])
    _require(all(b['row'] > x['end_row'] + 1 for x, b in zip(objects, objects[1:])),
             "objects require separate transverse bands")
    payload = set(map(tuple, np.argwhere((a[:, :rail] != bg)
                                        & (a[:, :rail] != material))))
    _require(payload <= owned, "loose exterior payload is outside the family")
    return a, turn, bg, material, rail, objects


def _transition(grid, mode="rule"):
    """Counterfactual modes are used only to audit demonstration strength."""
    a, turn, bg, material, rail, objects = _scene(grid)
    out = np.full(a.shape, bg, dtype=np.int8)
    out[:, rail] = material
    for obj in objects:
        patch = obj['patch'].copy()
        fuel = set(obj['fuel'])
        center, color = obj['center'], obj['color']
        if mode == "all_payload":
            fuel = set(map(tuple, np.argwhere((patch != bg) & (patch != material))))
        elif mode in ("ignore_enclosure", "ignore_extreme"):
            regions = obj['opened'] if mode == "ignore_enclosure" else obj['enclosed']
            for region in regions:
                if mode == "ignore_extreme" or any(c == patch.shape[1] - 2 for r, c in region):
                    fuel.update(p for p in region if patch[p] != bg)
        if fuel and color is None:
            color = int(patch[min(fuel)])
        budget = len(fuel)
        if mode == "components" and fuel:
            mask = np.zeros(patch.shape, dtype=bool)
            for p in fuel:
                mask[p] = True
            budget = len(_components(mask, diagonal=True))
        for p in fuel:
            patch[p] = bg
        gap = obj['gap']
        motion = gap if mode == "unlimited_alignment" else min(budget, gap)
        start = obj['col'] + motion
        out[obj['row']:obj['end_row'] + 1, start:start + patch.shape[1]] = patch
        length = max(0, budget - gap)
        if mode == "edge_ray" and length:
            length = a.shape[1] - rail - 1
        if length:
            stop = min(a.shape[1], rail + 1 + length)
            out[obj['row'] + center, rail + 1:stop] = color
    return np.rot90(out, -turn).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer exclusively from this input's visible geometry."""
    return _transition(_grid(input_grid))


def _pick(rng, cells, quantity, pattern):
    cells = set(cells)
    _require(0 <= quantity <= len(cells), "payload exceeds its region")
    if not quantity:
        return set()
    ordered = sorted(cells)
    if pattern == "random":
        return {ordered[int(i)] for i in rng.choice(len(ordered), quantity, replace=False)}
    if pattern == "sparse":
        # Starting on a diameter endpoint ensures the N=2 anchor is disconnected.
        distance = lambda p, q: max(abs(p[0] - q[0]), abs(p[1] - q[1]))
        diameter = max(distance(p, q) for p in ordered for q in ordered)
        endpoints = [p for p in ordered if any(distance(p, q) == diameter for q in ordered)]
        selected = {endpoints[int(rng.integers(len(endpoints)))]}
        while len(selected) < quantity:
            choices = sorted(cells - selected)
            scores = [min(distance(p, q) for q in selected) for p in choices]
            best = max(scores)
            candidates = [p for p, score in zip(choices, scores) if score == best]
            selected.add(candidates[int(rng.integers(len(candidates)))])
        return selected
    selected = {ordered[int(rng.integers(len(ordered)))]}
    while len(selected) < quantity:
        frontier = sorted(p for p in cells - selected
                          if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in selected))
        _require(frontier, "compact payload region is disconnected")
        selected.add(frontier[int(rng.integers(len(frontier)))])
    return selected


def _erode(rng, walls):
    target = {p for hole in _regions(walls)[0] for p in hole}
    h, w = walls.shape
    for _ in range(int(rng.integers(0, 6))):
        side = int(rng.integers(3))
        p = (int(rng.integers(h)), 0) if side == 0 else (
            0 if side == 1 else h - 1, int(rng.integers(w - 1)))
        if not walls[p]:
            continue
        trial = walls.copy()
        trial[p] = False
        if (len(_components(trial)) == 1 and np.all(trial.any(axis=1))
                and np.all(trial.any(axis=0))
                and {p for hole in _regions(trial)[0] for p in hole} == target):
            walls = trial
    return walls


def _body(rng, height, dual=True, anchor=False, long_tank=False, openable=False):
    width = int(rng.integers(9, 13)) if anchor else int(rng.integers(8, 14) if dual else rng.integers(4, 11))
    walls = np.ones((height, width), dtype=bool)
    heights = list(range(1, height - 1, 2))
    tank_h = height - 2 if long_tank else int(rng.choice([x for x in heights if x >= 3] if anchor else heights))
    y0 = int(rng.integers(1, height - tank_h))
    split = int(rng.integers(3, min(5, width - 5) + 1)) if dual else 0
    upper = width - 4 if anchor else width - 2
    lower = split + 1 if dual else 1
    left = int(rng.integers(lower, upper + 1))
    stepped = rng.random() < .55
    reservoir = set()
    for r in range(y0, y0 + tank_h):
        back = int(rng.integers(left, upper + 1)) if stepped else left
        reservoir.update((r, c) for c in range(back, width - 1))
    for p in reservoir:
        walls[p] = False
    if dual:
        if not anchor and height >= 7 and rng.random() < .35:
            intervals = [(1, 2), (height - 2, height - 1)]
        else:
            cargo_h = 3 if anchor else int(rng.integers(1, height - 1))
            cargo_y = int(rng.integers(1, height - cargo_h))
            intervals = [(cargo_y, cargo_y + cargo_h)]
        for lo, hi in intervals:
            walls[lo:hi, 1:split] = False
    # Make a doorway before erosion. A closed loop with eroded shoulders can
    # acquire branches that would detach when the loop is opened later.
    if not anchor and not openable:
        walls = _erode(rng, walls)
    if not anchor and rng.random() < .3:
        extension = int(rng.integers(1, 4))
        extended = np.zeros((height, width + extension), dtype=bool)
        extended[:, extension:] = walls
        row = int(rng.choice([0, height - 1]))
        extended[row, :extension + 1] = True
        reservoir = {(r, c + extension) for r, c in reservoir}
        walls = extended
    return walls, reservoir


def _opened(rng, walls, reservoir):
    rows = [r for r, c in reservoir]
    for side in rng.permutation(3):
        choices = (sorted(c for r, c in reservoir if r == min(rows)) if side == 0 else
                   sorted(c for r, c in reservoir if r == max(rows)) if side == 1 else
                   sorted(set(rows)))
        for index in rng.permutation(len(choices)):
            value = choices[int(index)]
            result = walls.copy()
            if side == 0:
                result[:min(rows), value] = False
            elif side == 1:
                result[max(rows) + 1:, value] = False
            else:
                result[value, -1] = False
            if len(_components(result)) == 1:
                return result
    raise AssertionError("no connected doorway construction")


def _recessed(rng, walls):
    h, w = walls.shape
    extension = int(rng.integers(1, 4))
    result = np.zeros((h, w + extension), dtype=bool)
    result[:, :w] = walls
    # Attach at a wall that erosion has not removed.
    rows = np.flatnonzero(walls[:, -1])
    row = int(rng.choice([int(rows[0]), int(rows[-1])]))
    result[row, w:] = True
    return result


def _spec(rng, walls, reservoir, quantity, gap=None, pattern="random", full=False, cargo_count=None):
    walls = walls.copy()
    enclosed, _ = _regions(walls)
    active = _leading(walls, enclosed)
    tokens = {p: 2 for p in _pick(rng, reservoir, quantity, pattern)}
    for hole in enclosed:
        if hole & reservoir or len(hole) < 2:
            continue
        q = min(cargo_count, len(hole) - 1) if cargo_count is not None else int(rng.integers(1, min(4, len(hole) - 1) + 1))
        tokens.update({p: 3 for p in _pick(rng, hole, q, str(rng.choice(["random", "compact", "sparse"])))})
    fuel = {p for p in active if p in tokens}
    center = ((min(r for r, c in active) + max(r for r, c in active)) // 2 if active else 0)
    return dict(walls=walls, tokens=tokens, fuel=fuel, count=len(fuel),
                center=center, gap=gap, full=full)


def _random_specs(rng, training, first_test=False):
    count = int(rng.choice([1, 1, 2, 2, 3, 3, 4, 5]))
    budget = 30 - (count - 1) - 2
    specs = []
    for index in range(count):
        available = min(9, budget - 3 * (count - index - 1))
        if first_test and index == 0:
            height = int(rng.choice([7, 9]))
            walls, tank = _body(rng, height, anchor=True, long_tank=True)
            quantity = int(rng.integers(9, min(12, len(tank) - 1) + 1))
            specs.append(_spec(rng, walls, tank, quantity, gap=int(rng.integers(1, 5)),
                               pattern="compact", full=True))
        else:
            height = int(rng.choice(list(range(3, available + 1, 2))))
            kind = str(rng.choice(["closed", "closed", "closed", "open", "recessed", "solid"]))
            if kind == "solid":
                width = int(rng.integers(2, 9))
                walls = np.ones((height, width), dtype=bool)
                if rng.random() < .6:
                    spine = int(rng.integers(width))
                    for row in range(height):
                        lo = int(rng.integers(spine + 1))
                        hi = int(rng.integers(spine, width))
                        walls[row, :lo] = walls[row, hi + 1:] = False
                    cols = np.flatnonzero(walls.any(axis=0))
                    walls = walls[:, cols[0]:cols[-1] + 1]
                specs.append(_spec(rng, walls, set(), 0))
            else:
                walls, tank = _body(rng, height, dual=rng.random() < .65,
                                    openable=kind == "open")
                if kind == "open":
                    walls = _opened(rng, walls, tank)
                    walls = _erode(rng, walls)
                elif kind == "recessed":
                    walls = _recessed(rng, walls)
                limit = min(8 if training else 20, len(tank) - 1)
                quantity = (0 if limit == 0 else
                            int(rng.integers(0 if kind == "closed" else 1, limit + 1)))
                specs.append(_spec(rng, walls, tank, quantity,
                                   pattern=str(rng.choice(["random", "compact", "sparse"]))))
        budget -= height
    return specs


def _render(rng, specs, palette, turn, matched=False):
    """Direct labels use explicit sampled fuel cells, not solve or scene parsing."""
    specs = copy.deepcopy(specs)
    rng.shuffle(specs)
    bg, material = palette[:2]
    min_band = max([1] + [max(0, s['count'] - s['gap']) + 1
                          for s in specs if s['full']])
    max_rail = 29 - min_band
    for s in specs:
        if s['gap'] is None:
            s['gap'] = int(rng.integers(0, min(8, max_rail - s['walls'].shape[1]) + 1))
    minimum = max(s['walls'].shape[1] + s['gap'] for s in specs)
    _require(minimum <= max_rail, "ray and body cannot fit together")
    rail = int(rng.integers(minimum, max_rail + 1))
    band = int(rng.integers(min_band, 30 - rail))
    base = sum(s['walls'].shape[0] for s in specs) + len(specs) - 1 + 2
    height = int(rng.integers(base, 31))
    width = rail + 1 + band
    extras = rng.multinomial(height - base, np.full(len(specs) + 1, 1 / (len(specs) + 1)))
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    inp[:, rail] = out[:, rail] = material
    row = 1 + int(extras[0])
    for i, s in enumerate(specs):
        h, w = s['walls'].shape
        color = palette[2] if matched else int(rng.choice(palette[2:]))
        cargo_color = palette[3] if matched else color if rng.random() < .5 else int(rng.choice(palette[2:]))
        patch = np.full((h, w), bg, dtype=np.int8)
        patch[s['walls']] = material
        for p, role in s['tokens'].items():
            patch[p] = color if role == 2 else cargo_color
        start = rail - w - s['gap']
        inp[row:row + h, start:start + w] = patch
        moved = patch.copy()
        for p in s['fuel']:
            moved[p] = bg
        travel = min(s['count'], s['gap'])
        out[row:row + h, start + travel:start + travel + w] = moved
        remaining = max(0, s['count'] - s['gap'])
        out[row + s['center'], rail + 1:min(width, rail + 1 + remaining)] = color
        row += h + 1 + int(extras[i + 1])
    return {"input": np.rot90(inp, turn).copy(), "output": np.rot90(out, turn).copy()}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, int(rng.integers(4, 8)), replace=False)))
    palette_policy = str(rng.choice(["fixed", "local", "local"]))
    orientation_policy = str(rng.choice(["fixed", "mixed", "rotation"]))
    base_turn = int(rng.integers(4))
    test_turn = (base_turn + int(rng.integers(1, 4))) % 4
    walls, tank = _body(rng, int(rng.choice([5, 7])), anchor=True)
    high = int(rng.integers(6, 9))
    anchors = [
        _spec(rng, walls, tank, 2, 3, "compact", cargo_count=2),
        _spec(rng, walls, tank, high, 3, "compact", full=True, cargo_count=2),
        _spec(rng, walls, tank, 3, 3, "random", cargo_count=2),
        _spec(rng, walls, tank, 2, 0, "sparse", full=True, cargo_count=2),
        _spec(rng, _opened(rng, walls, tank), tank, 2, 3, "compact", cargo_count=2),
        _spec(rng, _recessed(rng, walls), tank, 2, 3, "compact", cargo_count=2),
    ]
    rng.shuffle(anchors)
    plans = [anchors[:3], anchors[3:]]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for index in range(count):
            matched = split == "train" and index < 2
            first_test = split == "test" and index == 0
            for _ in range(20):
                specs = plans[index] if matched else _random_specs(rng, split == "train", first_test)
                local_palette = palette if matched or palette_policy == "fixed" else list(map(int, rng.choice(10, int(rng.integers(4, 8)), replace=False)))
                turn = (int(rng.integers(4)) if orientation_policy == "mixed" else
                        test_turn if split == "test" and orientation_policy == "rotation" else base_turn)
                pair = _render(rng, specs, local_palette, turn, matched)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError("failed to construct a distinct input")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(pairs):
    regimes, full_lengths, repeated, counts_by_body = set(), set(), {}, {}
    opened_loaded = recessed_loaded = cargo_counterexample = False
    records = []
    for pair in pairs:
        grid = _grid(pair['input'])
        a, turn, bg, mat, rail, objects = _scene(grid)
        observed = np.rot90(_grid(pair['output']), turn)
        for o in objects:
            n, d = o['count'], o['gap']
            records.append(o)
            if n:
                regimes.add("docked" if d == 0 else "short" if n < d else "exact" if n == d else "surplus")
                fuel_mask = np.zeros(o['patch'].shape, dtype=bool)
                for p in o['fuel']:
                    fuel_mask[p] = True
                repeated.setdefault(n, set()).add(len(_components(fuel_mask, diagonal=True)))
                signature = (o['walls'].shape, o['walls'].tobytes(), d)
                counts_by_body.setdefault(signature, set()).add(n)
                remainder = max(0, n - d)
                if 0 < remainder < a.shape[1] - rail - 1:
                    ray = observed[o['row'] + o['center'], rail + 1:]
                    _require(np.count_nonzero(ray == o['color']) == remainder, "training ray count")
                    full_lengths.add(remainder)
            payload_cells = set(map(tuple, np.argwhere((o['patch'] != bg) & (o['patch'] != mat))))
            opened_loaded |= any(region & payload_cells and any(c == o['patch'].shape[1] - 2 for r, c in region)
                                 for region in o['opened'])
            recessed_loaded |= bool(not o['active'] and any(hole & payload_cells for hole in o['enclosed']))
            cargo_counterexample |= bool(n and len(payload_cells) > n and n < d <= len(payload_cells))
    _require(regimes == {"docked", "short", "exact", "surplus"}, "training misses an expenditure regime")
    _require(len(full_lengths) >= 2, "training admits a fixed or edge-length ray")
    _require(any(len(v) > 1 for v in repeated.values()), "no count repeated with different connectivity")
    _require(any(len(v) > 1 for v in counts_by_body.values()), "count confounded with body or gap")
    _require(opened_loaded and recessed_loaded and cargo_counterexample, "missing topology/cargo contrasts")
    for mode in ("components", "all_payload", "ignore_enclosure", "ignore_extreme", "unlimited_alignment", "edge_ray"):
        _require(any(not np.array_equal(_transition(_grid(p['input']), mode), _grid(p['output']))
                     for p in pairs), f"training admits shortcut {mode}")
    return records


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal pair counts")
    seen, split_counts = set(), {'train': [], 'test': []}
    test_full_large = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "shape changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a, turn, bg, mat, rail, objects = _scene(inp)
            target = np.rot90(out, turn)
            _require(np.all(target[:, rail] == mat), "rail changed")
            expected_payload = 0
            for o in objects:
                n_fuel, gap = o['count'], o['gap']
                _require(gap >= 0, "body crossed the rail")
                _require(np.all(o['patch'][o['walls']] == mat), "wall material")
                for region in o['enclosed']:
                    _require(any(o['patch'][p] == bg for p in region), "loaded cavity is not visibly hollow")
                travel = min(n_fuel, gap)
                patch = o['patch'].copy()
                for p in o['fuel']:
                    patch[p] = bg
                translated = target[o['row']:o['end_row'] + 1,
                                    o['col'] + travel:o['col'] + travel + patch.shape[1]]
                _require(np.array_equal(translated, patch), "body or cargo did not move rigidly")
                cargo = np.count_nonzero((patch != bg) & (patch != mat))
                visible = min(max(0, n_fuel - gap), a.shape[1] - rail - 1)
                expected_payload += int(cargo) + visible
                split_counts[split].append(n_fuel)
                test_full_large |= (split == 'test' and n_fuel >= 9 and gap > 0
                                    and 0 < n_fuel - gap < a.shape[1] - rail - 1)
            _require(np.count_nonzero(target == mat) == np.count_nonzero(a == mat), "walls not conserved")
            _require(np.count_nonzero((target != bg) & (target != mat)) == expected_payload,
                     "fuel expenditure or cargo conservation failed")
            _require(np.array_equal(solve(inp), out), "output differs from input-only rule")
    _evidence(puzzle['train'])
    _require(max(split_counts['train']) <= 8 and test_full_large,
             "missing demonstrated small-budget to larger-budget shift")
    return True


def _fixtures():
    """Small explicit expected grids; these are not parent pairs."""
    inp = np.zeros((8, 15), dtype=np.int8)
    inp[:, 9] = 1
    ring = np.ones((5, 5), dtype=np.int8)
    ring[1:4, 1:4] = 0
    inp[1:6, 1:6] = ring
    inp[2, 2:4] = 3
    shortage = np.zeros_like(inp)
    shortage[:, 9] = 1
    shortage[1:6, 3:8] = ring
    yield inp, shortage
    surplus = inp.copy()
    surplus[2:4, 2:5] = 3
    expected = np.zeros_like(inp)
    expected[:, 9] = 1
    expected[1:6, 4:9] = ring
    expected[3, 10:13] = 3
    yield surplus, expected
    opened = inp.copy()
    opened[1, 4] = 0
    yield opened, opened.copy()
    recessed = inp.copy()
    recessed[1, 6:8] = 1
    yield recessed, recessed.copy()
    clipped = np.zeros((8, 15), dtype=np.int8)
    clipped[:, 13] = 1
    clipped[1:6, 8:13] = ring
    clipped[2:4, 9:12] = 3
    expected = np.zeros_like(clipped)
    expected[:, 13] = 1
    expected[1:6, 8:13] = ring
    expected[3, 14] = 3
    yield clipped, expected


def self_test():
    for inp, expected in _fixtures():
        for turn in range(4):
            for offset in (0, 4):
                a = (np.rot90(inp, turn) + offset) % 10
                b = (np.rot90(expected, turn) + offset) % 10
                _require(np.array_equal(solve(a), b), "direct geometry/expenditure fixture")
                saved = a.copy()
                solve(a)
                _require(np.array_equal(a, saved), "solver mutated input")
    fingerprints, shapes, body_shapes, palettes, dirs = set(), set(), set(), set(), set()
    counts, quantities, gaps, movements, tank_spans = set(), set(), set(), set(), set()
    regimes, clipped_count, silent_count, irregular, stepped = set(), 0, 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        _require(_fingerprint(p) == _fingerprint(generate_puzzle(seed)), "seed not reproducible")
        for split in ('train', 'test'):
            for pair in p[split]:
                inp = _grid(pair['input'])
                a, turn, bg, mat, rail, objects = _scene(inp)
                shapes.add(inp.shape)
                palettes.add((bg, mat))
                dirs.add(turn)
                counts.add(len(objects))
                silent_count += np.array_equal(pair['input'], pair['output'])
                for o in objects:
                    n, d = o['count'], o['gap']
                    quantities.add(n)
                    gaps.add(d)
                    movements.add(min(n, d))
                    body_shapes.add((o['walls'].shape, o['walls'].tobytes()))
                    irregular += not (o['walls'][0].all() and o['walls'][-1].all()
                                      and o['walls'][:, 0].all() and o['walls'][:, -1].all())
                    if o['active']:
                        rows = sorted({r for r, c in o['active']})
                        tank_spans.add(len(rows))
                        stepped += len({min(c for r, c in o['active'] if r == row) for row in rows}) > 1
                    if n:
                        regimes.add("docked" if d == 0 else "short" if n < d else "exact" if n == d else "surplus")
                        clipped_count += n - d > a.shape[1] - rail - 1
    _require(len(fingerprints) == 42, "cross-seed duplication")
    _require(len(shapes) >= 40 and len(body_shapes) >= 100, "structural variation collapsed")
    _require(dirs == {0, 1, 2, 3} and len(palettes) >= 25, "direction/palette variation collapsed")
    _require({1, 2, 3, 4, 5} <= counts and max(quantities) >= 12, "count/quantity variation collapsed")
    _require({1, 3, 5, 7} <= tank_spans and len(gaps) >= 7 and len(movements) >= 6,
             "tank or expenditure variation collapsed")
    _require(regimes == {"docked", "short", "exact", "surplus"} and clipped_count > 0
             and silent_count > 0 and irregular > 30 and stepped > 20,
             "missing boundary, inactive, or irregular cases")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(99173, a, b)
            _require(len(p['train']) == a and len(p['test']) == b, "explicit counts ignored")
    for a, b in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    corrupt = copy.deepcopy(generate_puzzle(0))
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an incorrect label")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), grid_shapes=len(shapes),
                wall_geometries=len(body_shapes), directions=sorted(dirs),
                visible_role_palettes=len(palettes), object_counts=sorted(counts),
                fuel_quantities=sorted(quantities), gaps=sorted(gaps), movements=sorted(movements),
                tank_spans=sorted(tank_spans), clipped_exports=int(clipped_count),
                unchanged_pairs=int(silent_count), irregular_objects=int(irregular),
                stepped_tanks=int(stepped), independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
