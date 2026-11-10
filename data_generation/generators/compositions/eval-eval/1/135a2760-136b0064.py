"""Periodic tape repair in coordinates supplied by a colored turtle program.

No files, examples, seed, or generator state are consulted by solve().
All experiments and reports are opt-in; normal generation performs no I/O.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


# A four-cell triangle points towards its tip. Entries are N, E, S, W.
_ARROWS = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=bool),
)
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_HOST_SEEDS = (0,683912169,588877495,722668098,646084039,730898175,328072864,994717887,51209947,719177608,399732026,217331512,162444648,782833396,733255732,981434683,928800386,725923592,89292869,344850719,154638679,471995446,260590192,42370399,192856025,753568727,681477590,242562805,288195648,904977163,629656862,151981234,213747752,447326025,57943385,407043534,192204463,795647343,555785274,743690403,353168647,1088051)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=False)


def _decode_tile(block, background, structure):
    colors = [int(c) for c in np.unique(block) if c != background]
    if len(colors) != 1 or colors[0] == structure:
        raise ValueError("instruction must have one accent")
    ink = colors[0]
    directions = [i for i, arrow in enumerate(_ARROWS)
                  if np.array_equal(block[:3, :3] == ink, arrow)]
    tally = int(np.count_nonzero(block[:, 4] == ink))
    if len(directions) != 1 or not 1 <= tally <= 4:
        raise ValueError("invalid arrow or tally")
    expected = np.full((4, 5), background, dtype=np.int8)
    expected[:3, :3][_ARROWS[directions[0]]] = ink
    expected[:tally, 4] = ink
    if not np.array_equal(block, expected):
        raise ValueError("instruction contains extra marks or a broken tally")
    return directions[0], tally + 1, ink


def _read_input(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    if h < 11 or (h - 1) % 5 or w < 15:
        raise ValueError("invalid framed program dimensions")
    structure = int(a[0, 0])
    if not (np.all(a[0] == structure) and np.all(a[-1] == structure)
            and np.all(a[:, 0] == structure) and np.all(a[:, -1] == structure)):
        raise ValueError("missing outer frame")
    background = int(a[4, 1])  # The arrow's fourth row is always empty.
    if background == structure:
        raise ValueError("padding and structure must differ")
    bands = (h - 1) // 5
    candidates = []
    for lanes in (1, 2, 3):
        divider = 6 * lanes
        if w - divider < 7:
            continue
        if not all(np.all(a[:, x] == structure) for x in range(0, divider + 1, 6)):
            continue
        if not all(np.all(a[y, :divider + 1] == structure) for y in range(0, h, 5)):
            continue
        pane = a[:, divider:]
        # Blank inner margins distinguish the canvas from a shorter program prefix.
        if not (np.all(pane[1, 1:-1] == background)
                and np.all(pane[-2, 1:-1] == background)
                and np.all(pane[1:-1, 1] == background)
                and np.all(pane[1:-1, -2] == background)):
            continue
        try:
            commands = [_decode_tile(a[5*r+1:5*r+5, 6*c+1:6*c+6],
                                     background, structure)
                        for c in range(lanes) for r in range(bands)]
        except ValueError:
            continue
        accents = {color for _, _, color in commands}
        if background in accents or structure in accents:
            continue
        others = [int(c) for c in np.unique(a)
                  if c not in accents | {background, structure}]
        if len(others) != 1:
            continue
        marker = others[0]
        marks = np.argwhere(pane == marker)
        if marks.shape != (1, 2) or np.count_nonzero(a == marker) != 1:
            continue
        start = tuple(int(v) for v in marks[0])
        try:
            coords, inks, stroke_ids = _trace(commands, start, pane.shape)
        except ValueError:
            continue
        values = pane[coords[:, 0], coords[:, 1]]
        if not np.all((values == structure) | (values == inks)):
            continue
        if any(int(c) not in accents | {background, structure, marker} for c in pane.flat):
            continue
        candidates.append(dict(pane=pane.copy(), structure=structure,
                               background=background, marker=marker, start=start,
                               commands=commands, coords=coords, inks=inks,
                               stroke_ids=stroke_ids, lanes=lanes, bands=bands,
                               divider=divider, bits=(values != structure).astype(np.int8)))
    if len(candidates) != 1:
        raise ValueError("input has no unique program and canvas")
    return candidates[0]


def _trace(commands, start, shape):
    r, c = start
    if not (2 <= r < shape[0] - 2 and 2 <= c < shape[1] - 2):
        raise ValueError("seed is outside the active canvas")
    occupied = {start}
    coords, inks, stroke_ids = [], [], []
    for j, (direction, distance, ink) in enumerate(commands):
        dr, dc = _DIRECTIONS[direction]
        for _ in range(distance):
            r, c = r + dr, c + dc
            if not (2 <= r < shape[0] - 2 and 2 <= c < shape[1] - 2):
                raise ValueError("route leaves the active canvas")
            if (r, c) in occupied:
                raise ValueError("route revisits a cell")
            occupied.add((r, c))
            coords.append((r, c))
            inks.append(ink)
            stroke_ids.append(j)
    return (np.asarray(coords, dtype=int), np.asarray(inks, dtype=np.int8),
            np.asarray(stroke_ids, dtype=int))


def _recover(bits):
    """All supported periodic repairs must agree, including redundant periods."""
    bits = np.asarray(bits, dtype=np.int8)
    n = len(bits)
    candidates = []
    for period in range(1, n // 4 + 1):
        unit = np.empty(period, dtype=np.int8)
        for phase in range(period):
            group = bits[phase::period]
            ones = int(group.sum())
            if 2 * ones == len(group):
                break
            unit[phase] = int(2 * ones > len(group))
        else:
            clean = unit[np.arange(n) % period]
            if (len(np.unique(unit)) == 2
                    and np.count_nonzero(clean != bits) <= max(1, n // 8)):
                candidates.append((period, unit, clean))
    if not candidates:
        raise ValueError("no majority-supported repeating tape")
    first = candidates[0]
    if any(not np.array_equal(first[2], candidate[2]) for candidate in candidates[1:]):
        raise ValueError("ambiguous periodic repair")
    return first


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    _, _, clean = _recover(info["bits"])
    answer = info["pane"].copy()
    coords = info["coords"]
    answer[coords[:, 0], coords[:, 1]] = np.where(clean, info["inks"], info["structure"])
    return answer


def _raster_indices(coords):
    return np.lexsort((coords[:, 1], coords[:, 0]))


def _different_coordinates(commands, coords, period):
    order = _raster_indices(coords)
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(len(order))
    local = np.concatenate([np.arange(distance) for _, distance, _ in commands])
    return (np.any(ranks % period != np.arange(len(order)) % period)
            and np.any(local % period != np.arange(len(order)) % period))


def _walk(rng, count, height, width, minimum, period, search_budget=2500,
          required_directions=frozenset()):
    """Bounded randomized search; a feasible folded snake is the finite fallback.

    Coordinates here use the interior canvas; retain one blank cell around it.
    """
    corners = [(1, 1), (1, width-2), (height-2, 1), (height-2, width-2)]
    starts = [corners[int(i)] for i in rng.permutation(4)]
    starts += [(int(rng.integers(1, height-1)), int(rng.integers(1, width-1)))
               for _ in range(4)]
    budget = [search_budget]

    def search(pos, occupied, program, path):
        budget[0] -= 1
        remaining = count - len(program)
        if len(path) + 5 * remaining < minimum:
            return None
        if not remaining:
            coords = np.asarray(path, dtype=int)
            if (len(path) >= minimum and len({d % 2 for d, _, _ in program}) == 2
                    and required_directions <= {d for d, _, _ in program}
                    and _different_coordinates(program, coords, period)):
                return program, coords
            return None
        if budget[0] <= 0:
            return None
        for choice in rng.permutation(16):
            direction, offset = divmod(int(choice), 4)
            distance = offset + 2
            dr, dc = _DIRECTIONS[direction]
            cells = [(pos[0] + k*dr, pos[1] + k*dc) for k in range(1, distance+1)]
            if any(not (1 <= r < height-1 and 1 <= c < width-1)
                   or (r, c) in occupied for r, c in cells):
                continue
            result = search(cells[-1], occupied | set(cells),
                            program + [(direction, distance, 0)], path + cells)
            if result is not None:
                return result
            if budget[0] <= 0:
                break
        return None

    for start in starts:
        result = search(start, {start}, [], [])
        if result is not None:
            return start, result[0]
        if budget[0] <= 0:
            break

    # All generation layouts fit this constructive family, including (2,1).
    if count == 4:
        raw = [(1, 5, 0), (2, 3, 0), (3, 5, 0), (2, 3, 0)]
    else:
        verticals = count // 2
        distances = [2] * verticals
        extra = max(0, minimum - (5 * ((count+1)//2) + 2 * verticals))
        for j in range(verticals):
            add = min(3, extra)
            distances[j] += add
            extra -= add
        if extra or sum(distances) > height-3:
            raise RuntimeError("infeasible fallback geometry")
        raw = []
        for j in range(count):
            raw.append(((1 if (j//2) % 2 == 0 else 3), 5, 0)
                       if j % 2 == 0 else (2, distances[j//2], 0))
    # Reflections give all compass directions without changing route feasibility.
    for reflection in rng.permutation(4):
        flip_y, flip_x = bool(reflection // 2), bool(reflection % 2)
        start = (height-2 if flip_y else 1, width-2 if flip_x else 1)
        program = [((0 if d == 2 else 2) if flip_y and d % 2 == 0 else
                    (4-d) if flip_x and d % 2 else d, distance, ink)
                   for d, distance, ink in raw]
        # Trace in interior coordinates, offsetting to framed coordinates.
        framed = [(d, n, 1) for d, n, _ in program]
        coords, _, _ = _trace(framed, (start[0]+1, start[1]+1), (height+2, width+2))
        if (len(coords) >= minimum and _different_coordinates(program, coords, period)
                and required_directions <= {d for d, _, _ in program}):
            return start, program
    raise RuntimeError("fallback lacks a coordinate counterexample")


def _signal(rng, commands, coords, period):
    n = len(coords)
    local = np.concatenate([np.arange(distance) for _, distance, _ in commands])
    order = _raster_indices(coords)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)

    def consider(unit, deterministic=False):
        clean = unit[np.arange(n) % period]
        if (np.array_equal(unit[local % period], clean)
                or np.array_equal(unit[ranks % period], clean)):
            return None
        zero_phases = np.flatnonzero(unit == 0)
        one_phases = np.flatnonzero(unit == 1)
        if not len(zero_phases) or not len(one_phases):
            return None
        phases = [int(zero_phases[0] if deterministic else rng.choice(zero_phases)),
                  int(one_phases[0] if deterministic else rng.choice(one_phases))]
        if not deterministic and period >= 3 and n // 8 >= 3 and rng.integers(2):
            phases.append(int(rng.choice([p for p in range(period) if p not in phases])))
        noisy = clean.copy()
        for j, phase in enumerate(phases):
            copies = np.arange(phase, n, period)
            # At least one corrupted cell is in the first period: copying it fails.
            at = int(copies[0] if j == 0 else copies[-1] if deterministic else rng.choice(copies))
            noisy[at] ^= 1
        try:
            recovered_period, _, repaired = _recover(noisy)
        except ValueError:
            return None
        if recovered_period == period and np.array_equal(repaired, clean):
            return noisy, clean
        return None

    for _ in range(64):
        unit = rng.integers(0, 2, size=period, dtype=np.int8)
        result = consider(unit)
        if result is not None:
            return result
    for word in range(1, (1 << period)-1):
        unit = np.array([(word >> j) & 1 for j in range(period)], dtype=np.int8)
        result = consider(unit, deterministic=True)
        if result is not None:
            return result
    raise RuntimeError("finite signal construction exhausted")


def _make_pair(rng, bands, lanes, period, minimum, roles, required_directions=frozenset()):
    structure, background, marker = roles[:3]
    h = 5 * bands + 1
    inner_h = h - 2
    max_width = 28 - 6 * lanes
    inner_w = int(rng.integers(8, max_width + 1))
    start, commands = _walk(rng, bands*lanes, inner_h, inner_w, minimum, period,
                            required_directions=required_directions)
    palette = [int(c) for c in rng.choice(roles[3:], size=int(rng.integers(2, 5)), replace=False)]
    colors = [palette[j % len(palette)] for j in range(len(commands))]
    rng.shuffle(colors)
    # Neighboring instructions deliberately sometimes share an ink.
    if len(commands) >= 6:
        colors[1] = colors[0]
    if len(set(colors)) < 2:
        colors[-1] = next(c for c in palette if c != colors[0])
    commands = [(d, n, color) for (d, n, _), color in zip(commands, colors)]
    pane = np.full((h, inner_w+2), background, dtype=np.int8)
    pane[[0, -1], :] = structure
    pane[:, [0, -1]] = structure
    seed = (start[0]+1, start[1]+1)
    pane[seed] = marker
    coords, inks, _ = _trace(commands, seed, pane.shape)
    noisy, clean = _signal(rng, commands, coords, period)
    pane[coords[:, 0], coords[:, 1]] = np.where(noisy, inks, structure)
    occupied = {seed} | {tuple(p) for p in coords.tolist()}
    free = [(r, c) for r in range(2, h-2) for c in range(2, inner_w)
            if (r, c) not in occupied]
    route_cells = occupied - {seed}
    adjacent = [p for p in free if any((p[0]+dr, p[1]+dc) in route_cells
                                     for dr, dc in _DIRECTIONS)]
    # At least one off-route mark touches the route; connectivity alone is insufficient.
    marks = [adjacent[int(rng.integers(len(adjacent)))]]
    available = [p for p in free if p != marks[0]]
    marks += [available[int(i)] for i in rng.choice(len(available), size=3, replace=False)]
    for p in marks:
        pane[p] = int(rng.choice([structure] + sorted(set(colors))))
    output = pane.copy()
    output[coords[:, 0], coords[:, 1]] = np.where(clean, inks, structure)
    divider = 6 * lanes
    inp = np.full((h, divider + inner_w + 2), background, dtype=np.int8)
    inp[:, divider:] = pane
    inp[::5, :divider+1] = structure
    inp[:, :divider+1:6] = structure
    for index, (direction, distance, ink) in enumerate(commands):
        lane, band = divmod(index, bands)
        block = inp[5*band+1:5*band+5, 6*lane+1:6*lane+6]
        block[:3, :3][_ARROWS[direction]] = ink
        block[:distance-1, 4] = ink
    return {"input": inp, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = [int(v) for v in rng.permutation(10)]
    periods = [int(v) for v in rng.permutation(np.arange(2, 6))]
    if periods[0] == 5:
        periods[0], periods[1] = periods[1], periods[0]
    puzzle = {"train": [], "test": []}
    max_train_length = 0
    training_directions = set()
    for j in range(nt + nq):
        if j < nt:
            bands, lanes = ((2, 2) if j == 0 else (2, 3) if j == 1 else
                            ((2, 2), (2, 3), (3, 2))[int(rng.integers(3))])
            period = periods[j % 4]
            if bands * lanes == 4 and period == 5:
                bands, lanes = 3, 2
            minimum = max(16, 4 * period)
            split = "train"
        else:
            bands, lanes = ((4, 2), (5, 2), (3, 3), (4, 3), (5, 3))[int(rng.integers(5))]
            period = periods[int(rng.integers(min(4, nt)))]
            minimum = max(4 * period, max_train_length + 2)
            split = "test"
        # Rotate literal roles per example, preserving a single visible grammar.
        pair_roles = roles[j % 10:] + roles[:j % 10]
        if j >= 10:
            # Distinct visible (frame,padding) roles guarantee distinct inputs
            # even at the maximum fourteen pairs, without random duplicate retries.
            pair_roles[1], pair_roles[2] = pair_roles[2], pair_roles[1]
        needed = {0, 1, 2, 3} - training_directions if j == 1 else frozenset()
        pair = _make_pair(rng, bands, lanes, period, minimum, pair_roles, needed)
        puzzle[split].append(pair)
        if split == "train":
            info = _read_input(pair["input"])
            max_train_length = max(max_train_length, len(info["bits"]))
            training_directions.update(d for d, _, _ in info["commands"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    return grid.shape, grid.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, descriptors = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _read_input(inp)
            assert out.shape == info["pane"].shape
            coords, inks = info["coords"], info["inks"]
            period, unit, clean = _recover(info["bits"])
            assert 2 <= period <= 5 and len(clean) >= 4*period
            output_values = out[coords[:, 0], coords[:, 1]]
            assert np.array_equal(output_values, np.where(clean, inks, info["structure"]))
            route_mask = np.zeros(out.shape, dtype=bool)
            route_mask[coords[:, 0], coords[:, 1]] = True
            assert np.array_equal(out[~route_mask], info["pane"][~route_mask])
            errors = np.flatnonzero(info["bits"] != clean)
            assert 2 <= len(errors) <= min(3, len(clean)//8)
            assert len(set(int(i % period) for i in errors)) == len(errors)
            assert set(int(v) for v in clean[errors]) == {0, 1}
            assert any(i < period for i in errors)
            local = np.concatenate([np.arange(n) for _, n, _ in info["commands"]])
            assert not np.array_equal(unit[local % period], clean), "phase reset shortcut survives"
            ranks = np.empty(len(clean), dtype=int)
            ranks[_raster_indices(coords)] = np.arange(len(clean))
            assert not np.array_equal(unit[ranks % period], clean), "raster shortcut survives"
            offroute = (~route_mask) & (info["pane"] != info["background"])
            offroute[[0, -1], :] = False
            offroute[:, [0, -1]] = False
            offroute[info["start"]] = False
            assert np.count_nonzero(offroute) == 4
            assert any(offroute[r+dr, c+dc] for r, c in coords
                       for dr, dc in _DIRECTIONS)
            assert len(set(int(v) for v in inks)) >= 2
            assert np.array_equal(solve(inp), out)
            info.update(period=period, length=len(clean), errors=len(errors))
            descriptors[split].append(info)
    train, test = descriptors["train"], descriptors["test"]
    assert len({(d["structure"], d["background"]) for d in train + test}) == len(train) + len(test)
    assert len({d["period"] for d in train}) >= 2
    assert len({d["lanes"] for d in train}) >= 2
    assert len({d["structure"] for d in train}) >= 2
    assert len({d["background"] for d in train}) >= 2
    assert {direction for d in train for direction, _, _ in d["commands"]} == {0, 1, 2, 3}
    assert len({n for d in train for _, n, _ in d["commands"]}) >= 2
    assert min(len(d["commands"]) for d in test) > max(len(d["commands"]) for d in train)
    assert min(d["length"] for d in test) > max(d["length"] for d in train)
    assert {d["period"] for d in test} <= {d["period"] for d in train}


def self_test():
    """42 puzzles, streamed statistics, deterministic and semantic mutations.

    No grids, logs, execution traces, or JSON files are saved. A single puzzle
    and bounded sets of hashes/descriptors are retained at any time.
    """
    puzzle_hashes, shapes, paths, periods, directions, distances = set(), set(), set(), set(), set(), set()
    command_counts, tape_lengths, palettes = set(), set(), set()
    pairs = 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                period, _, _ = _recover(info["bits"])
                periods.add(period)
                pairs += 1
                for grid in pair.values():
                    digest.update(bytes(grid.shape))
                    digest.update(grid.tobytes())
                shapes.add((pair["input"].shape, pair["output"].shape))
                paths.add(tuple((d, n) for d, n, _ in info["commands"]))
                directions.update(d for d, _, _ in info["commands"])
                distances.update(n for _, n, _ in info["commands"])
                command_counts.add(len(info["commands"]))
                tape_lengths.add(len(info["bits"]))
                palettes.add((info["structure"], info["background"], info["marker"]))
        puzzle_hashes.add(digest.hexdigest())
    assert len(puzzle_hashes) == 42
    assert len(paths) >= 180 and len(shapes) >= 15
    assert periods == {2, 3, 4, 5} and directions == {0, 1, 2, 3}
    assert distances == {2, 3, 4, 5}
    assert len(command_counts) >= 6 and len(tape_lengths) >= 20 and len(palettes) >= 42
    fallback_cases = 0
    for bands, lanes in ((2, 2), (2, 3), (3, 2), (4, 2), (5, 2), (3, 3), (4, 3), (5, 3)):
        for width in sorted({8, 28 - 6*lanes}):
            for period in range(2, 6):
                if bands*lanes == 4 and period == 5:
                    continue
                requirements = (frozenset(),) if bands*lanes == 4 else (
                    frozenset(), frozenset((0, 1)), frozenset((0, 3)),
                    frozenset((2, 1)), frozenset((2, 3)))
                for needed in requirements:
                    rng = np.random.default_rng(fallback_cases)
                    minimum = max(16, 4*period) if bands*lanes <= 6 else 32
                    start, commands = _walk(rng, bands*lanes, 5*bands-1, width,
                                            minimum, period, search_budget=0,
                                            required_directions=needed)
                    coords, _, _ = _trace([(d, n, 1) for d, n, _ in commands],
                                          (start[0]+1, start[1]+1), (5*bands+1, width+2))
                    noisy, clean = _signal(rng, commands, coords, period)
                    assert np.array_equal(_recover(noisy)[2], clean)
                    fallback_cases += 1
    # Count boundaries and deterministic whole-puzzle arrays.
    a, b = generate_puzzle(1225, 10, 4), generate_puzzle(1225, 10, 4)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    generate_puzzle(19, 2, 1)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    for damage in ("route", "outside", "glyph"):
        mutated = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
        pair = mutated["test"][0]
        info = _read_input(pair["input"])
        if damage == "route":
            r, c = info["coords"][0]
            pair["output"][r, c] = info["background"]
        elif damage == "outside":
            pair["output"][1, 1] = info["structure"]
        else:
            pair["input"][1:4, 1:4] = info["background"]
        try:
            validate_generated_puzzle(mutated)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("semantic mutation accepted")
    return {"seeds":42,"pairs":pairs,"distinct_puzzles":len(puzzle_hashes),
            "distinct_routes":len(paths),"shape_pairs":len(shapes),
            "periods":sorted(periods),"directions":sorted(directions),
            "distances":sorted(distances),"command_counts":sorted(command_counts),
            "tape_lengths":sorted(tape_lengths),"role_palettes":len(palettes),
            "forced_fallback_cases":fallback_cases}
