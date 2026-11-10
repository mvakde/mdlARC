"""Locator-addressed 3x3 payloads with endpoint-only diagonal cap emission.

Only NumPy and the standard library are used. No files, examples, RNG state, or
training labels are consulted by solve(). See task_spec.md for the full domain.
"""
from __future__ import annotations

import hashlib
import numpy as np


LOCATOR = 9
POSITIONS = tuple((r, c) for r in range(3) for c in range(3))
DIAGONALS = (((0, 0), (1, 1), (2, 2)),
             ((0, 2), (1, 1), (2, 0)))
VECTORS = frozenset(((-1, -1), (-1, 1), (1, -1), (1, 1)))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and result[-1][1] == i:
            result[-1] = (result[-1][0], i + 1)
        else:
            result.append((i, i + 1))
    return result


def _axis_starts(indices, extent):
    bands = _runs(indices)
    if len(bands) != 2:
        raise ValueError("expected two divider bands per axis")
    (a, b), (c, d) = bands
    if b - a not in (1, 3, 5) or d - c != b - a or c - b != 3:
        raise ValueError("invalid 3x3-panel lattice")
    if not (0 <= a - 3 <= 5 and 0 <= extent - d - 3 <= 5):
        raise ValueError("invalid outer checkerboard margin")
    return (a - 3, b, d)


def _panel(info, position):
    r, c = position
    return (slice(info["rows"][r], info["rows"][r] + 3),
            slice(info["cols"][c], info["cols"][c] + 3))


def _cap_vector(diagonal, locator):
    r, c = locator
    distance = r - c if diagonal == 0 else r + c - 2
    if distance == 0:
        raise ValueError("locator cannot lie on an active cap")
    sign = 1 if distance > 0 else -1
    return (-sign, sign) if diagonal == 0 else (-sign, -sign)


def _caps(payload, locator, background):
    result = []
    for i, cells in enumerate(DIAGONALS):
        color = int(payload[cells[0]])
        if (color not in (*background, LOCATOR)
                and all(int(payload[cell]) == color for cell in cells)):
            result.append((i, color, _cap_vector(i, locator)))
    return result


def _parse(input_grid):
    grid = _grid(input_grid)
    rr = np.flatnonzero(np.all(grid == grid[:, :1], axis=1))
    cc = np.flatnonzero(np.all(grid == grid[:1, :], axis=0))
    rows = _axis_starts(rr, grid.shape[0])
    cols = _axis_starts(cc, grid.shape[1])
    mask = np.zeros(grid.shape, dtype=bool)
    mask[rr, :] = True
    mask[:, cc] = True
    values = np.unique(grid[mask])
    if len(values) != 1 or int(values[0]) == LOCATOR:
        raise ValueError("divider bands must have one non-locator color")
    divider = int(values[0])
    if np.any(grid[~mask] == divider):
        raise ValueError("divider color is reserved for divider bands")
    parity = np.indices(grid.shape).sum(axis=0) & 1
    background = []
    for p in (0, 1):
        cells = grid[(parity == p) & ~mask]
        counts = np.bincount(cells.astype(int), minlength=10)
        color = int(counts.argmax())
        if counts[color] <= len(cells) // 2 or color in (divider, LOCATOR):
            raise ValueError("checkerboard phase must have a strict majority")
        background.append(color)
    if background[0] == background[1]:
        raise ValueError("checkerboard colors must differ")
    background = tuple(background)
    base = np.asarray(background, dtype=np.int8)[parity]
    base[mask] = divider
    info = dict(rows=rows, cols=cols, divider=divider, mask=mask,
                background=background, base=base)
    locations = np.argwhere(grid == LOCATOR)
    if locations.shape != (1, 2):
        raise ValueError("expected exactly one locator of color 9")
    r, c = map(int, locations[0])
    sr = [i for i, a in enumerate(rows) if a <= r < a + 3]
    sc = [i for i, a in enumerate(cols) if a <= c < a + 3]
    if len(sr) != 1 or len(sc) != 1:
        raise ValueError("locator must be inside a panel")
    source = (sr[0], sc[0])
    address = (r - rows[source[0]], c - cols[source[1]])
    payload = grid[_panel(info, source)].copy()
    panel_mask = np.zeros(grid.shape, dtype=bool)
    for position in POSITIONS:
        panel_mask[_panel(info, position)] = True
    if np.any(grid[~panel_mask] != base[~panel_mask]):
        raise ValueError("outer margins must contain only checkerboard")
    info.update(source=source, address=address, payload=payload,
                caps=_caps(payload, address, background), panel_mask=panel_mask)
    return info


def _paths(info, destination=None):
    """Walk physical grid coordinates, skipping paint on transparent dividers."""
    if destination is None:
        destination = info["address"]
    r0, c0 = info["rows"][destination[0]], info["cols"][destination[1]]
    h, w = info["base"].shape
    paths = []
    for diagonal, color, (vr, vc) in info["caps"]:
        for er, ec in (DIAGONALS[diagonal][0], DIAGONALS[diagonal][-1]):
            path = []
            r, c = r0 + er + vr, c0 + ec + vc
            while 0 <= r < h and 0 <= c < w:
                if not info["mask"][r, c]:
                    path.append((r, c))
                r, c = r + vr, c + vc
            paths.append((color, (vr, vc), path))
    return paths


def _compose(info, *, relocate=True, emit=True):
    destination = info["address"] if relocate else info["source"]
    output = info["base"].copy()
    output[_panel(info, destination)] = info["payload"]
    if emit:
        claimed = {}
        for color, _, path in _paths(info, destination):
            for cell in path:
                if cell in claimed and claimed[cell] != color:
                    raise ValueError("conflicting cap rays")
                claimed[cell] = color
                output[cell] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode, clear, relocate unchanged, then emit only the moved cap endpoints."""
    return _compose(_parse(input_grid))


def _geometric_prediction(info):
    """Closed-form ray membership; shares parsing but not the walking renderer."""
    output = info["base"].copy()
    output[_panel(info, info["address"])] = info["payload"]
    rows, cols = np.indices(output.shape)
    r0, c0 = info["rows"][info["address"][0]], info["cols"][info["address"][1]]
    for diagonal, color, (vr, vc) in info["caps"]:
        for er, ec in (DIAGONALS[diagonal][0], DIAGONALS[diagonal][-1]):
            dr = (rows - r0 - er) * vr
            dc = (cols - c0 - ec) * vc
            ray = (dr == dc) & (dr >= 1) & ~info["mask"]
            output[ray] = color
    return output


def _counts(rng, num_train, num_test):
    def check(value, lo, hi):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError("illegal example count")
        return int(value)
    a = None if num_train is None else check(num_train, 2, 10)
    b = None if num_test is None else check(num_test, 1, 4)
    if b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1)) if a is not None else int(rng.integers(1, 5))
    if a is None:
        a = int(rng.integers(max(2, b + 1), 11))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _available_diagonals(address):
    return tuple(i for i, cells in enumerate(DIAGONALS) if address not in cells)


def _choose_diagonals(rng, address, allowed=VECTORS, permit_empty=True):
    available = [i for i in _available_diagonals(address)
                 if _cap_vector(i, address) in allowed]
    if not available:
        return ()
    if permit_empty and rng.random() < .15:
        return ()
    count = int(rng.integers(1, len(available) + 1))
    return tuple(sorted(map(int, rng.choice(available, count, replace=False))))


def _blank(gutters, margins, colors):
    gh, gw = gutters
    top, bottom, left, right = margins
    rows = tuple(top + i * (3 + gh) for i in range(3))
    cols = tuple(left + i * (3 + gw) for i in range(3))
    shape = (9 + 2 * gh + top + bottom, 9 + 2 * gw + left + right)
    parity = np.indices(shape).sum(axis=0) & 1
    grid = np.asarray(colors[:2], dtype=np.int8)[parity]
    mask = np.zeros(shape, dtype=bool)
    for start in rows[:2]:
        mask[start + 3:start + 3 + gh, :] = True
    for start in cols[:2]:
        mask[:, start + 3:start + 3 + gw] = True
    grid[mask] = colors[2]
    return grid, dict(rows=rows, cols=cols, mask=mask)


def _draw_pair(rng, gutters, colors, source, address, diagonals, rich=False):
    margins = tuple(map(int, rng.integers(0, 6, size=4)))
    grid, layout = _blank(gutters, margins, colors)
    background = tuple(colors[:2])
    palette = [c for c in range(9) if c not in colors]
    rng.shuffle(palette)
    cap_color = palette[0]
    accents = palette[1:]
    payload = grid[_panel(layout, source)].copy()
    cap_cells = {p for i in diagonals for p in DIAGONALS[i]}
    for p in cap_cells:
        payload[p] = cap_color
    payload[address] = LOCATOR
    free = [p for p in POSITIONS if p not in cap_cells and p != address]
    # Rich teaching payloads keep one checker cell and at least two accents.
    count = min(2, len(free) - 1) if rich else int(rng.integers(0, min(3, len(free) - 1) + 1))
    for j in map(int, rng.choice(len(free), size=count, replace=False)):
        payload[free[j]] = int(rng.choice(accents))
    # Three randomly repeated accents can accidentally complete an unwanted
    # cap in an intended capless payload. Break that line without touching any
    # requested cap, locator, or background cell, rather than failing a seed.
    for i, color, _ in _caps(payload, address, background):
        if i not in diagonals:
            endpoint = next(p for p in (DIAGONALS[i][0], DIAGONALS[i][-1])
                            if p not in cap_cells and p != address)
            payload[endpoint] = next(c for c in accents if c != color)
    assert {i for i, _, _ in _caps(payload, address, background)} == set(diagonals)

    others = [p for p in POSITIONS if p != source]
    rng.shuffle(others)
    # Every scene has a complete cap decoy and a broken near-cap. Neither has 9.
    for index, position in enumerate(others):
        panel = grid[_panel(layout, position)]
        if index < 2:
            cells = DIAGONALS[int(rng.integers(2))]
            color = int(rng.choice(palette))
            for p in cells:
                panel[p] = color
            if index == 1:
                panel[cells[int(rng.integers(3))]] = int(_pick(rng, [c for c in palette if c != color]))
        else:
            count = int(rng.choice([0, 0, 1, 1, 2, 3, 4]))
            for cell in map(int, rng.choice(9, size=count, replace=False)):
                panel.flat[cell] = int(rng.choice(palette))
    grid[_panel(layout, source)] = payload
    # Maintain observable strict-majority checkerboard phases by removing only
    # optional random distractor marks. The two explicit cap decoys are kept.
    parity = np.indices(grid.shape).sum(axis=0) & 1
    for p in (0, 1):
        phase = (parity == p) & ~layout["mask"]
        while int(np.count_nonzero(grid[phase] == background[p])) <= int(phase.sum()) // 2:
            removable = []
            for pos in others[2:]:
                r0, c0 = layout["rows"][pos[0]], layout["cols"][pos[1]]
                removable.extend((r0 + r, c0 + c) for r, c in POSITIONS
                                 if parity[r0 + r, c0 + c] == p
                                 and grid[r0 + r, c0 + c] != background[p])
            assert removable, "mandatory teaching objects exceeded checkerboard budget"
            grid[_pick(rng, removable)] = background[p]
    info = _parse(grid)
    assert info["source"] == source and info["address"] == address
    return {"input": grid, "output": _compose(info)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    a, b = _counts(rng, num_train, num_test)
    gutters = (int(rng.choice([1, 3, 5])), int(rng.choice([1, 3, 5])))
    fixed_colors = tuple(map(int, rng.choice(9, size=3, replace=False)))
    varied_palette = bool(rng.integers(2))
    corner = _pick(rng, ((0, 0), (0, 2), (2, 0), (2, 2)))
    first, second = (corner[0], 1), (1, corner[1])
    if rng.random() < .5:
        first, second = second, first
    edge_shift = a >= 4 and rng.random() < .5
    held_address = (2 - second[0], 2 - second[1]) if edge_shift else corner
    train_specs = [(first, (0, 1), True)]
    first_vectors = {_cap_vector(i, first) for i in (0, 1)}
    second_diag = next(i for i in (0, 1) if _cap_vector(i, second) not in first_vectors)
    train_specs.append((second, (second_diag,), False))
    if a >= 3:
        opposite = (2 - first[0], 2 - first[1])
        train_specs.append((opposite, (0, 1), False))
    if edge_shift:
        # Expose the missing row/column in a corner before asking for the new
        # edge-middle recombination. Both cap counts are already demonstrated.
        if held_address[0] == 1:
            address = (int(rng.choice([0, 2])), held_address[1])
        else:
            address = (held_address[0], int(rng.choice([0, 2])))
        train_specs.append((address, _choose_diagonals(rng, address, permit_empty=False), False))
    for _ in range(a - len(train_specs)):
        address = _pick(rng, [p for p in POSITIONS if p != held_address])
        train_specs.append((address, _choose_diagonals(rng, address), False))
    sources = []
    for i, (address, _, _) in enumerate(train_specs):
        pool = [p for p in POSITIONS if (i >= 2 or p != address)
                and (i != 1 or p != sources[0])]
        sources.append(_pick(rng, pool))
    # Pair order carries no convention; witness roles are randomized.
    order = list(map(int, rng.permutation(a)))
    train_specs = [(*train_specs[i], sources[i]) for i in order]
    demonstrated = {_cap_vector(i, address) for address, ds, _, _ in train_specs for i in ds}
    trained_rows = {x[0][0] for x in train_specs}
    trained_cols = {x[0][1] for x in train_specs}
    has_empty = any(not x[1] for x in train_specs)
    pool = [p for p in POSITIONS if p[0] in trained_rows and p[1] in trained_cols
            and p != held_address and (p != (1, 1) or has_empty)]
    rng.shuffle(pool)
    test_addresses = [held_address] + pool[:b - 1]
    assert len(test_addresses) == b
    test_specs = []
    for index, address in enumerate(test_addresses):
        ds = _choose_diagonals(rng, address, demonstrated,
                               permit_empty=has_empty and address != held_address)
        used_sources = {s for ad, _, _, s in train_specs if ad == address}
        source_pool = [p for p in POSITIONS if p not in used_sources
                       and (index != 0 or p != address)]
        test_specs.append((address, ds, False, _pick(rng, source_pool or POSITIONS)))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for address, diagonals, rich, source in specs:
            for _ in range(16):
                colors = tuple(map(int, rng.choice(9, 3, replace=False))) if varied_palette else fixed_colors
                pair = _draw_pair(rng, gutters, colors, source, address, diagonals, rich)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _middle_probes(info):
    r0, c0 = info["rows"][info["address"][0]], info["cols"][info["address"][1]]
    occupied = np.zeros(info["base"].shape, dtype=bool)
    occupied[_panel(info, info["address"])] = True
    rays = {p for _, _, path in _paths(info) for p in path}
    h, w = info["base"].shape
    result = set()
    for _, _, (vr, vc) in info["caps"]:
        r, c = r0 + 1 + vr, c0 + 1 + vc
        while 0 <= r < h and 0 <= c < w:
            if not occupied[r, c] and not info["mask"][r, c] and (r, c) not in rays:
                result.add((r, c))
            r, c = r + vr, c + vc
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(out, _compose(info))
            assert np.array_equal(out, _geometric_prediction(info))
            assert not np.array_equal(inp, out)
            assert np.array_equal(out[info["mask"]], inp[info["mask"]])
            assert np.array_equal(out[_panel(info, info["address"])], info["payload"])
            assert np.count_nonzero(out == LOCATOR) == 1
            rays = {p for _, _, path in _paths(info) for p in path}
            intact = np.ones(inp.shape, dtype=bool)
            intact[_panel(info, info["address"])] = False
            for p in rays:
                intact[p] = False
            assert np.array_equal(out[intact], info["base"][intact])
            assert any(np.any(inp[_panel(info, p)] != info["base"][_panel(info, p)])
                       for p in POSITIONS if p != info["source"])
            # False emitters in discarded panels make ownership observable.
            assert any(any(len({int(inp[_panel(info, p)][q]) for q in cells}) == 1
                               and int(inp[_panel(info, p)][cells[0]]) not in info["background"]
                           for cells in DIAGONALS)
                       for p in POSITIONS if p != info["source"])
            for p in _middle_probes(info):
                assert out[p] == info["base"][p], "the cap middle must not continue"
            if rays:
                assert not np.array_equal(out, _compose(info, emit=False))
            if info["source"] != info["address"]:
                assert not np.array_equal(out, _compose(info, relocate=False))
            features[split].append(info)
    train = features["train"]
    assert len({f["source"] for f in train}) >= 2
    assert len({f["address"][0] for f in train}) >= 2
    assert len({f["address"][1] for f in train}) >= 2
    assert any(f["address"][0] != f["address"][1] for f in train)
    assert {len(f["caps"]) for f in train} >= {1, 2}
    assert sum(f["source"] != f["address"] and bool(f["caps"]) for f in train) >= 2
    assert any(_middle_probes(f) for f in train)
    assert any(any(f["payload"][p] not in (*f["background"], LOCATOR)
                       and p not in {q for i, _, _ in f["caps"] for q in DIAGONALS[i]}
                   for p in POSITIONS) for f in train)
    directions = {v for f in train for _, _, v in f["caps"]}
    addresses = {f["address"] for f in train}
    rows, cols = {p[0] for p in addresses}, {p[1] for p in addresses}
    assert any(f["address"] not in addresses for f in features["test"])
    for f in features["test"]:
        assert f["address"][0] in rows and f["address"][1] in cols
        assert {v for _, _, v in f["caps"]} <= directions
    assert features["test"][0]["caps"], "held-out address must emit"
    assert features["test"][0]["source"] != features["test"][0]["address"]
    assert len({f["address"] for f in features["test"]}) == b
    # Odd gutter widths preserve checkerboard phase under a whole-panel move.
    assert len({(f["rows"][1] - f["rows"][0], f["cols"][1] - f["cols"][0])
                for s in features.values() for f in s}) == 1


def _manual_regressions():
    # Expected ray coordinates are hand-enumerated, not produced by a renderer.
    inp, layout = _blank((1, 1), (0, 0, 0, 0), (0, 1, 8))
    payload = np.array([[2, 9, 0], [1, 2, 1], [0, 4, 2]], dtype=np.int8)
    inp[_panel(layout, (2, 2))] = payload
    inp[0, 0] = 3
    expected, _ = _blank((1, 1), (0, 0, 0, 0), (0, 1, 8))
    expected[_panel(layout, (0, 1))] = payload
    for p in ((2, 2), (4, 0), (4, 4), (6, 2), (8, 0)):
        expected[p] = 2
    for k in range(4):
        for reflect in (False, True):
            x, y = np.rot90(inp, k), np.rot90(expected, k)
            if reflect:
                x, y = np.fliplr(x), np.fliplr(y)
            assert np.array_equal(solve(x), y)
    # A centered locator admits no complete diagonal cap; copy alone is valid.
    inp[_panel(layout, (2, 2))] = np.array([[0, 1, 0], [1, 9, 3], [0, 1, 0]], dtype=np.int8)
    expected, _ = _blank((1, 1), (0, 0, 0, 0), (0, 1, 8))
    expected[_panel(layout, (1, 1))] = inp[_panel(layout, (2, 2))]
    assert np.array_equal(solve(inp), expected)


def self_test():
    """Bounded, streaming 42-seed sweep plus count and mutation regressions."""
    _manual_regressions()
    shapes, roles, sources, addresses, cap_counts, directions = set(), set(), set(), set(), set(), set()
    geometries, signatures, lengths = set(), set(), set()
    test_cap_counts, test_directions, test_addresses = set(), set(), set()
    stationary = capless = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        repeat = generate_puzzle(seed)
        assert all(_key(p[role]) == _key(q[role]) for split in ("train", "test")
                   for p, q in zip(puzzle[split], repeat[split]) for role in ("input", "output"))
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                inp = pair["input"]
                saved = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, saved), "solve mutated its input"
                f = _parse(inp)
                shapes.add(inp.shape)
                roles.add((*f["background"], f["divider"]))
                sources.add(f["source"])
                addresses.add(f["address"])
                cap_counts.add(len(f["caps"]))
                capless += not f["caps"]
                stationary += f["source"] == f["address"]
                directions.update(v for _, _, v in f["caps"])
                if split == "test":
                    test_cap_counts.add(len(f["caps"]))
                    test_directions.update(v for _, _, v in f["caps"])
                    test_addresses.add(f["address"])
                lengths.update(len(path) for _, _, path in _paths(f))
                geometries.add((inp.shape, f["rows"], f["cols"], f["source"], f["address"],
                                tuple(i for i, _, _ in f["caps"])))
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 35 and len(roles) >= 35
    assert sources == set(POSITIONS) and addresses == set(POSITIONS)
    assert cap_counts == {0, 1, 2} and directions == VECTORS
    assert test_cap_counts == {1, 2} and test_directions == VECTORS
    assert len(test_addresses) >= 6
    assert stationary >= 3 and capless >= 3 and len(geometries) >= 180
    assert max(lengths) >= 8 and len(lengths) >= 8
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(4000 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((None, 1), (None, 4), (2, None), (10, None), (None, None)):
        p = generate_puzzle(53, a, b)
        assert a is None or len(p["train"]) == a
        assert b is None or len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, "1")):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0)
    pair = damaged["test"][0]
    info = _parse(pair["input"])
    ray_cell = next(p for _, _, path in _paths(info) for p in path)
    pair["output"][ray_cell] = info["base"][ray_cell]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a removed endpoint-ray cell")
    return dict(seeds=42, pairs=pairs, shapes=len(shapes), role_triples=len(roles),
                source_positions=len(sources), addresses=len(addresses),
                cap_counts=sorted(cap_counts), directions=sorted(directions),
                test_cap_counts=sorted(test_cap_counts), test_addresses=sorted(test_addresses),
                structural_signatures=len(geometries), stationary_pairs=stationary,
                capless_pairs=capless, ray_lengths=sorted(lengths))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
