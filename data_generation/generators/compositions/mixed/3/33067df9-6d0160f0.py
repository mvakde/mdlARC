"""Exact-span rectangle decoding routed by a panel-local address.

The family fixes the sampling phase to panel rows 0 and 2, and fixes locator
color 4. Neither solving nor generation reads files or uses parent modules.
"""
from hashlib import sha256
from itertools import product

import numpy as np


LOCATOR = 4
POSITIONS = tuple(product(range(3), repeat=2))
SPANS = tuple((left, right) for left in range(3) for right in range(left + 1, 4))
LATTICE = np.zeros((11, 11), dtype=bool)
LATTICE[[3, 7], :] = True
LATTICE[:, [3, 7]] = True


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside its permitted range")
    return value


def _counts(num_train, num_test):
    nt = _integer("num_train", num_train, 2, 10)
    ns = _integer("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _panel(position):
    row, col = position
    return slice(4 * row, 4 * row + 3), slice(4 * col, 4 * col + 3)


def _blank(bg, divider):
    out = np.full((11, 11), bg, dtype=np.int8)
    out[LATTICE] = divider
    return out


def _grid(value):
    grid = np.asarray(value)
    if grid.shape != (11, 11) or not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("Expected an integer 11 by 11 grid")
    if np.any((grid < 0) | (grid > 9)):
        raise ValueError("Colors must be 0..9")
    return grid


def _runs(row, bg):
    """Return maximal (color, inclusive left, exclusive right) runs."""
    runs = []
    col = 0
    while col < len(row):
        color = int(row[col])
        end = col + 1
        while end < len(row) and int(row[end]) == color:
            end += 1
        if color != bg:
            runs.append((color, col, end))
        col = end
    return runs


def _render(code, bg):
    """Render complete run stacks as rectangles on the local 3 by 3 canvas."""
    out = np.full((3, 3), bg, dtype=np.int8)
    rows = [_runs(row, bg) for row in code]
    for row, runs in enumerate(rows):
        for color, left, right in runs:
            run = (color, left, right)
            if row and run in rows[row - 1]:
                continue
            depth = 2 if row == 0 and run in rows[1] else 1
            out[2 * row:2 * row + 2 * depth - 1, left:right] = color
    return out


def _bridge_row(code, bg):
    """Cell-context check, without run extraction or rectangle rendering."""
    result = np.full(3, bg, dtype=np.int8)
    for col in range(3):
        color = int(code[0, col])
        if color == bg or color != int(code[1, col]):
            continue
        boundaries = []
        for row in range(2):
            left, right = col, col + 1
            while left and code[row, left - 1] == color:
                left -= 1
            while right < 3 and code[row, right] == color:
                right += 1
            boundaries.append((left, right))
        if boundaries[0] == boundaries[1]:
            result[col] = color
    return result


def _features(code, bg):
    upper, lower = (_runs(row, bg) for row in code)
    exact = set(upper) & set(lower)
    unequal = any(a == b and (l, r) != (s, t) and max(l, s) < min(r, t)
                  for a, l, r in upper for b, s, t in lower)
    return {
        "exact": bool(exact),
        "wide_exact": any(r - l >= 2 for _, l, r in exact),
        "partial": unequal,
        "horizontal": any(r - l >= 2 for _, l, r in upper + lower),
        "rectangles": len(upper) + len(lower) - len(exact),
    }


def _parse(input_grid):
    grid = _grid(input_grid)
    divider = int(grid[3, 3])
    if divider == LOCATOR or not np.all(grid[LATTICE] == divider):
        raise ValueError("Invalid divider lattice")
    if np.any(grid[~LATTICE] == divider):
        raise ValueError("Divider color is reserved for the lattice")
    counts = np.bincount(grid[~LATTICE].astype(np.int64), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= 40 or bg == LOCATOR:
        raise ValueError("Panel background must occupy a strict majority of cells")
    locations = np.argwhere(grid == LOCATOR)
    if locations.shape != (1, 2):
        raise ValueError("Exactly one locator of color 4 is required")
    row, col = map(int, locations[0])
    source, address = (row // 4, col // 4), (row % 4, col % 4)
    if address not in POSITIONS:
        raise ValueError("Locator lies on a divider")
    payload = grid[_panel(source)].copy()
    payload[address] = bg
    if not np.all(payload[1] == bg):
        raise ValueError("The payload middle row holds only background and locator")
    code = payload[::2].copy()
    if not np.any(code != bg):
        raise ValueError("At least one encoded rectangle is required")
    return grid, bg, divider, source, address, code


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive both the rectangle pattern and its destination from this input."""
    _, bg, divider, _, address, code = _parse(input_grid)
    out = _blank(bg, divider)
    out[_panel(address)] = _render(code, bg)
    return out


def _paint(code, row, span, color, bg):
    left, right = span
    code[row, left:right] = color
    if left and code[row, left - 1] == color:
        code[row, left - 1] = bg
    if right < 3 and code[row, right] == color:
        code[row, right] = bg


def _sample_code(rng, palette, bg, address, role):
    density = float(_pick(rng, (0.2, 0.4, 0.65, 0.85, 1.0)))
    code = rng.choice(palette, size=(2, 3)).astype(np.int8)
    code[rng.random((2, 3)) > density] = bg
    hole = None if address[0] == 1 else (address[0] // 2, address[1])
    if hole is not None:
        code[hole] = bg
    kind = role if role != "free" else _pick(rng, ("exact", "partial", "random", "random"))
    color = int(_pick(rng, palette))
    if kind in ("wide_exact", "exact"):
        spans = [s for s in SPANS if (kind != "wide_exact" or s[1] - s[0] >= 2)
                 and (hole is None or not s[0] <= hole[1] < s[1])]
        span = _pick(rng, spans)
        for row in range(2):
            _paint(code, row, span, color, bg)
    elif kind == "partial":
        choices = [(a, b) for a in SPANS for b in SPANS
                   if a != b and max(a[0], b[0]) < min(a[1], b[1])
                   and (hole is None or not (a, b)[hole[0]][0] <= hole[1] < (a, b)[hole[0]][1])]
        first, second = _pick(rng, choices)
        _paint(code, 0, first, color, bg)
        _paint(code, 1, second, color, bg)
    if hole is not None:
        code[hole] = bg
    if not np.any(code != bg):
        cells = [(r, c) for r in range(2) for c in range(3) if (r, c) != hole]
        code[_pick(rng, cells)] = color
    return code


def _specs(rng, nt, ns):
    # A two-cell-wide repeated span must fit without covering the locator.
    exact_address = _pick(rng, [p for p in POSITIONS if p[0] == 1 or p[1] != 1])
    partial_address = _pick(rng, [p for p in POSITIONS
                                 if p[0] != exact_address[0] and p[1] != exact_address[1]
                                 and (p[0] != p[1] or exact_address[0] != exact_address[1])])
    addresses = [exact_address, partial_address]
    if nt >= 3:
        remaining_row = next(r for r in range(3) if r not in (exact_address[0], partial_address[0]))
        remaining_col = next(c for c in range(3) if c not in (exact_address[1], partial_address[1]))
        addresses.append((remaining_row, remaining_col))
    rows, cols = {r for r, _ in addresses}, {c for _, c in addresses}
    address_pool = [(r, c) for r in sorted(rows) for c in sorted(cols)]
    held_out = _pick(rng, [p for p in address_pool if p not in addresses])
    while len(addresses) < nt:
        addresses.append(_pick(rng, [p for p in address_pool if p != held_out]))
    roles = ["wide_exact", "partial"] + ["free"] * (nt - 2)
    sources = []
    for i, address in enumerate(addresses):
        eligible = [p for p in POSITIONS if i >= 2 or (p != address and p not in sources)]
        sources.append(_pick(rng, eligible))
    train = [(sources[i], addresses[i], roles[i]) for i in rng.permutation(nt)]
    remainder = [p for p in address_pool if p != held_out]
    rng.shuffle(remainder)
    # Novel address combinations are preferred; the first test is guaranteed novel.
    remainder.sort(key=lambda p: p in addresses)
    test_addresses = [held_out] + remainder[:ns - 1]
    combinations = set(zip(sources, addresses))
    test = []
    for i, address in enumerate(test_addresses):
        eligible = [p for p in POSITIONS if (p, address) not in combinations
                    and (i != 0 or p != address)]
        test.append((_pick(rng, eligible), address, "free"))
    return train, test


def _draw_pair(rng, bg, divider, source, address, role):
    available = [c for c in range(10) if c not in (bg, divider, LOCATOR)]
    palette = [int(c) for c in rng.choice(available, size=int(rng.integers(1, 6)), replace=False)]
    singleton = int(_pick(rng, [c for c in available if c not in palette]))
    code = _sample_code(rng, palette, bg, address, role)
    inp = _blank(bg, divider)
    payload = np.full((3, 3), bg, dtype=np.int8)
    payload[::2] = code
    payload[address] = LOCATOR
    inp[_panel(source)] = payload
    # A structurally identical unmarked code makes shape alone insufficient.
    decoy = _pick(rng, [p for p in POSITIONS if p != source])
    decoy_panel = inp[_panel(decoy)]
    decoy_panel[::2] = code
    cells = [(4 * r + y, 4 * c + x) for r, c in POSITIONS if (r, c) not in (source, decoy)
             for y in range(3) for x in range(3)]
    order = [int(i) for i in rng.permutation(len(cells))]
    inp[cells[order[0]]] = singleton
    # At most 38/81 panel cells are marked: background remains input-visible.
    for index in order[1:1 + int(rng.integers(25))]:
        inp[cells[index]] = int(_pick(rng, palette))
    out = _blank(bg, divider)
    out[_panel(address)] = _render(code, bg)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    bg, divider = map(int, rng.choice([c for c in range(10) if c != LOCATOR], size=2, replace=False))
    train_specs, test_specs = _specs(rng, nt, ns)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for source, address, role in specs:
            for _ in range(16):
                pair = _draw_pair(rng, bg, divider, source, address, role)
                key = pair["input"].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not generate a distinct input in 16 attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, roles = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, bg, divider, source, address, code = _parse(pair["input"])
            out = _grid(pair["output"])
            key = inp.astype(np.int8).tobytes()
            assert key not in seen, "Duplicate input"
            seen.add(key)
            roles.add((bg, divider))
            assert np.all(out[LATTICE] == inp[LATTICE])
            assert not np.any(out == LOCATOR), "Locator must be consumed"
            decoded = out[_panel(address)]
            assert np.array_equal(decoded[::2], code), "Coarse rows must be retained"
            assert np.array_equal(decoded[1], _bridge_row(code, bg)), "Incorrect exact-span bridging"
            for position in POSITIONS:
                if position != address:
                    assert np.all(out[_panel(position)] == bg), "Unselected panel was not cleared"
            assert set(map(int, np.unique(decoded))) <= set(map(int, np.unique(code))) | {bg}
            assert np.array_equal(solve(inp), out)
            assert not np.array_equal(inp, out)
            decoys = [inp[_panel(p)] for p in POSITIONS if p != source]
            assert any(np.array_equal(panel[::2], code) and np.all(panel[1] == bg) for panel in decoys)
            counts = np.bincount(inp.ravel().astype(np.int64), minlength=10)
            assert any(counts[c] == 1 for c in range(10) if c != LOCATOR), "Missing incidental singleton"
            analyses[split].append((source, address, code, _features(code, bg)))
    assert len(roles) == 1, "Background and divider conventions vary within this puzzle"
    train = analyses["train"]
    assert len({x[0] for x in train}) >= 2
    assert len({x[1][0] for x in train}) >= 2 and len({x[1][1] for x in train}) >= 2
    assert any(r != c for _, (r, c), _, _ in train), "Transpose must be distinguishable"
    assert any(f["wide_exact"] and source != address for source, address, _, f in train)
    assert any(f["partial"] and source != address for source, address, _, f in train)
    train_addresses = {x[1] for x in train}
    tests = analyses["test"]
    assert tests[0][1] not in train_addresses
    assert tests[0][0] != tests[0][1]
    assert len({x[1] for x in tests}) == len(tests)
    assert all(address[0] in {x[1][0] for x in train} and address[1] in {x[1][1] for x in train}
               for _, address, _, _ in tests)
    combinations = {(x[0], x[1]) for x in train}
    assert all((x[0], x[1]) not in combinations for x in tests)


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _normalized(code, bg):
    mapping = {bg: 0}
    return tuple(mapping.setdefault(int(c), len(mapping)) for c in code.flat)


def self_test():
    """Bounded in-memory checks; return descriptive, non-independent metrics."""
    # All 729 two-row codes over background plus two labels, including empty codes.
    for values in product((0, 1, 2), repeat=6):
        code = np.array(values, dtype=np.int8).reshape(2, 3)
        decoded = _render(code, 0)
        assert np.array_equal(decoded[::2], code)
        assert np.array_equal(decoded[1], _bridge_row(code, 0))
    # Explicit witnesses against per-column equality and naive source copying.
    assert np.array_equal(_render(np.array([[1, 1, 0], [1, 1, 0]]), 0)[1], [1, 1, 0])
    assert np.array_equal(_render(np.array([[1, 1, 0], [1, 1, 1]]), 0)[1], [0, 0, 0])
    assert np.array_equal(_render(np.array([[1, 0, 1], [1, 1, 1]]), 0)[1], [0, 0, 0])
    layouts, puzzles, roles, palettes = set(), set(), set(), set()
    addresses, sources, rectangle_counts, occupancies = set(), set(), set(), set()
    exact = partial = zero_payload = stationary = blank_row = pairs = 0
    for seed in range(42):
        nt = 2 + seed % 9
        ns = 1 + (seed // 3) % min(4, nt - 1)
        puzzle = generate_puzzle(seed, nt, ns)
        assert _equal_puzzles(puzzle, generate_puzzle(seed, nt, ns))
        # Only compact fingerprints, never retained copies of the full sweep.
        digest = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
        puzzles.add(digest.digest())
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _, bg, divider, source, address, code = _parse(pair["input"])
                f = _features(code, bg)
                layouts.add(_normalized(code, bg))
                roles.add((bg, divider))
                palettes.add(tuple(sorted(set(map(int, code.flat)) - {bg})))
                addresses.add(address)
                sources.add(source)
                rectangle_counts.add(f["rectangles"])
                occupancies.add(int(np.count_nonzero(code != bg)))
                exact += f["exact"]
                partial += f["partial"]
                zero_payload += bg != 0 and bool(np.any(code == 0))
                stationary += source == address
                blank_row += bool(np.any(np.all(code == bg, axis=1)))
                pairs += 1
        # Same code, different control coordinates: content stays fixed while
        # the destination changes. Include all nine coordinates where possible.
        pair = puzzle["train"][0]
        inp, bg, divider, source, address, code = _parse(pair["input"])
        for new_address in POSITIONS:
            if new_address[0] != 1 and code[new_address[0] // 2, new_address[1]] != bg:
                continue
            moved = inp.copy()
            moved[_panel(source)][address] = bg
            moved[_panel(source)][new_address] = LOCATOR
            prediction = solve(moved)
            expected = _blank(bg, divider)
            expected[_panel(new_address)] = pair["output"][_panel(address)]
            assert np.array_equal(prediction, expected)
        # Relocating the complete sparse source without changing its local
        # marker coordinate cannot change the answer. Swap, rather than erase,
        # the two panels so background visibility remains unchanged.
        for new_source in POSITIONS:
            moved = inp.copy()
            old_panel = inp[_panel(source)].copy()
            moved[_panel(source)] = inp[_panel(new_source)]
            moved[_panel(new_source)] = old_panel
            before = moved.copy()
            assert np.array_equal(solve(moved), pair["output"])
            assert np.array_equal(moved, before), "solve mutated its argument"
        # Operational ablations: sparse-copy and source-only routing fail on
        # an exact witness; cellwise bridging fails on an unequal-span witness.
        for pair in puzzle["train"]:
            inp, bg, divider, source, address, code = _parse(pair["input"])
            f = _features(code, bg)
            if f["wide_exact"] and source != address:
                sparse = np.full((3, 3), bg, dtype=np.int8)
                sparse[::2] = code
                assert not np.array_equal(sparse, pair["output"][_panel(address)])
                at_source = _blank(bg, divider)
                at_source[_panel(source)] = _render(code, bg)
                assert not np.array_equal(at_source, pair["output"])
            if f["partial"]:
                naive = np.where((code[0] == code[1]) & (code[0] != bg), code[0], bg)
                assert not np.array_equal(naive, pair["output"][_panel(address)][1])
    assert len(puzzles) == 42 and len(layouts) >= 30 and len(roles) >= 20
    assert len(palettes) >= 20 and len(addresses) == len(sources) == 9
    assert len(rectangle_counts) >= 4 and len(occupancies) >= 5
    assert min(exact, partial) >= 42 and min(zero_payload, stationary, blank_row) > 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(123456, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (4.0, 1), (True, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(puzzles),
            "normalized_payloads": len(layouts), "background_divider_roles": len(roles),
            "payload_palettes": len(palettes), "addresses": len(addresses),
            "sources": len(sources), "rectangle_counts": sorted(rectangle_counts),
            "occupied_slot_counts": sorted(occupancies), "exact_pairs": exact,
            "partial_overlap_pairs": partial, "stationary_pairs": stationary,
            "zero_payload_pairs": zero_payload, "blank_coarse_row_pairs": blank_row,
            "exhaustive_small_codes": 729, "independent_semantic_review": False}
