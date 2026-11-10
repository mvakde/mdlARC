"""Locator-addressed symmetry transport with stationary instruction markers.

Only NumPy and the standard library are used. No imports or reads from parents.
The two control colors are family constants; every other role is input-derived.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


LOCATOR = 4
MARKER = 2
POSITIONS = tuple(itertools.product(range(3), repeat=2))
ADDRESSES = tuple(p for p in POSITIONS if p != (1, 1))
HOST_SEEDS = (
    0, 423923661, 760389085, 400595731, 9222334, 676753169, 844442316,
    222731871, 744022251, 366128939, 102146851, 310842912, 454661231,
    860894449, 211724432, 640832070, 551066042, 375980218, 578039495,
    62381767, 857904848, 639024680, 486469069, 790174016, 266666946,
    892327979, 154275580, 151515732, 322738723, 586826287, 135922729,
    238229437, 937662446, 335176070, 523166098, 487313929, 152581983,
    349627838, 738113039, 319025542, 470539505, 909295862,
)


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _array(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("Expected a 1..30 integer ARC grid with colors 0..9")
    return g


def _mask(gaps):
    gh, gw = gaps
    mask = np.zeros((9 + 2 * gh, 9 + 2 * gw), dtype=bool)
    for k in (1, 2):
        mask[k * 3 + (k - 1) * gh:k * (3 + gh), :] = True
        mask[:, k * 3 + (k - 1) * gw:k * (3 + gw)] = True
    return mask


def _panel(position, gaps):
    r, c = position
    gh, gw = gaps
    return (slice(r * (3 + gh), r * (3 + gh) + 3),
            slice(c * (3 + gw), c * (3 + gw) + 3))


def _blank(gaps, background, divider):
    mask = _mask(gaps)
    g = np.full(mask.shape, background, dtype=np.int8)
    g[mask] = divider
    return g


def _transform(tile, kind):
    if kind == "H":
        return np.flipud(tile).copy()
    if kind == "V":
        return np.fliplr(tile).copy()
    if kind == "P":
        return np.rot90(tile, 2).copy()
    raise ValueError("Unknown symmetry")


def _destination(cell, kind):
    r, c = cell
    if kind == "H":
        return 2 - r, c
    if kind == "V":
        return r, 2 - c
    if kind == "P":
        return 2 - r, 2 - c
    raise ValueError("Unknown symmetry")


def _marker_kind(cells):
    if cells == [(1, 1)]:
        return "P"
    if len(cells) not in (2, 3):
        raise ValueError("One central singleton or one central straight segment required")
    for kind, axis in (("H", 0), ("V", 1)):
        other = [p[1 - axis] for p in cells]
        if (all(p[axis] == 1 for p in cells)
                and other == list(range(min(other), max(other) + 1))):
            return kind
    raise ValueError("Marker must lie on a central panel axis")


def _read(input_grid):
    """Decode all conventions from one input, without labels or generation state."""
    g = _array(input_grid)
    if any(n not in (11, 13, 15) for n in g.shape):
        raise ValueError("Three 3x3 panels per axis with 1..3-cell dividers required")
    gaps = tuple((n - 9) // 2 for n in g.shape)
    mask = _mask(gaps)
    colors = np.unique(g[mask])
    if len(colors) != 1 or int(colors[0]) in (LOCATOR, MARKER):
        raise ValueError("Divider lattice must have one non-control color")
    divider = int(colors[0])
    if np.any(g[~mask] == divider):
        raise ValueError("Divider color is reserved for the lattice")
    counts = np.bincount(g[~mask].astype(np.int64), minlength=10)
    background = int(counts.argmax())
    if counts[background] <= 40 or background in (LOCATOR, MARKER):
        raise ValueError("Background must occupy a strict majority of the 81 panel cells")
    locator_cells = np.argwhere(g == LOCATOR)
    if locator_cells.shape != (1, 2):
        raise ValueError("Exactly one locator required")
    row, col = map(int, locator_cells[0])
    source = (row // (3 + gaps[0]), col // (3 + gaps[1]))
    address = (row % (3 + gaps[0]), col % (3 + gaps[1]))
    if address not in ADDRESSES or source == address:
        raise ValueError("Address must be a noncentral panel distinct from the source")
    region = _panel(source, gaps)
    tile = g[region].copy()
    cells = sorted(map(tuple, np.argwhere(tile == MARKER).tolist()))
    if int(np.count_nonzero(g == MARKER)) != len(cells):
        raise ValueError("All symmetry-marker cells must belong to the locator's panel")
    kind = _marker_kind(cells)
    data = tile.copy()
    data[(data == LOCATOR) | (data == MARKER)] = background
    if not 2 <= np.count_nonzero(data != background) <= 7:
        raise ValueError("Payload must contain 2..7 data cells")
    if not any(np.any(g[_panel(p, gaps)] != background)
               for p in POSITIONS if p != source):
        raise ValueError("At least one marked distractor panel required")
    return {"grid": g, "gaps": gaps, "mask": mask, "background": background,
            "divider": divider, "source": source, "address": address,
            "kind": kind, "markers": cells, "data": data}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Keep the controls; transport the complete data field in its panel frame."""
    info = _read(input_grid)
    g = info["grid"]
    out = _blank(info["gaps"], info["background"], info["divider"])
    controls = (g == LOCATOR) | (g == MARKER)
    out[controls] = g[controls]
    out[_panel(info["address"], info["gaps"])] = _transform(info["data"], info["kind"])
    return out


def _pixel_output(info, *, address=None, kind=None):
    """Construction/check by explicit coordinates; solve uses whole-array flips."""
    address = info["address"] if address is None else address
    kind = info["kind"] if kind is None else kind
    out = _blank(info["gaps"], info["background"], info["divider"])
    g = info["grid"]
    controls = (g == LOCATOR) | (g == MARKER)
    out[controls] = g[controls]
    region = out[_panel(address, info["gaps"])]
    for r, c in POSITIONS:
        if info["data"][r, c] != info["background"]:
            region[_destination((r, c), kind)] = info["data"][r, c]
    return out


def _marker_options(kind, address):
    if kind == "P":
        return [[(1, 1)]]
    runs = ((0, 1), (1, 2), (0, 1, 2))
    candidates = [[(1, x) if kind == "H" else (x, 1) for x in run] for run in runs]
    return [cells for cells in candidates if address not in cells]


def _orientation_witness(data, kind, background):
    # Reject mere translation of an unchanged cropped shape to the opposite bbox.
    rows, cols = np.nonzero(data != background)
    cropped = data[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return not np.array_equal(cropped, _transform(cropped, kind))


def _distinct_orientations(data):
    variants = [data] + [_transform(data, k) for k in ("H", "V", "P")]
    return len({x.tobytes() for x in variants}) == 4


def _data_tile(rng, background, palette, address, markers, kind):
    free = [p for p in POSITIONS if p != address and p not in markers]
    # Tiny procedural motifs: scattered marks, diagonal growth, or a cut block.
    # Panel containment owns all pieces, so connectivity need not survive controls.
    for attempt in range(64):
        count = int(rng.integers(2, len(free) + 1))
        style = int(rng.integers(3))
        if style == 0:
            chosen = [free[int(i)] for i in rng.choice(len(free), count, replace=False)]
        elif style == 1:
            chosen = [free[int(rng.integers(len(free)))]]
            while len(chosen) < count:
                frontier = [p for p in free if p not in chosen and any(
                    max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1 for q in chosen)]
                pool = frontier or [p for p in free if p not in chosen]
                chosen.append(pool[int(rng.integers(len(pool)))])
        else:
            corner = POSITIONS[int(rng.integers(9))]
            order = [free[int(i)] for i in rng.permutation(len(free))]
            order.sort(key=lambda p: abs(p[0] - corner[0]) + abs(p[1] - corner[1]))
            chosen = order[:count]
        data = np.full((3, 3), background, dtype=np.int8)
        for cell in chosen:
            data[cell] = int(rng.choice(palette))
        if _distinct_orientations(data) and _orientation_witness(data, kind, background):
            return data
    # Finite exhaustive fallback, with no relaxation of the intended invariants.
    for count in range(3, len(free) + 1):
        for chosen in itertools.combinations(free, count):
            data = np.full((3, 3), background, dtype=np.int8)
            for i, cell in enumerate(chosen):
                data[cell] = palette[i % len(palette)]
            if _distinct_orientations(data) and _orientation_witness(data, kind, background):
                return data
    raise RuntimeError("No asymmetric payload fits the chosen controls")


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _plans(rng, a, b):
    kinds = ["P", _choice(rng, ["H", "V"])] if a == 2 else ["P", "H", "V"]
    active = tuple(kinds)
    kinds += [_choice(rng, active) for _ in range(a - len(kinds))]
    rng.shuffle(kinds)
    # Distinct rows/columns teach the two address coordinates independently.
    schedules = [list(zip(rows, cols))[:min(3, a)]
                 for rows in itertools.permutations(range(3))
                 for cols in itertools.permutations(range(3))
                 if all(p in ADDRESSES for p in list(zip(rows, cols))[:min(3, a)])]
    first = _choice(rng, schedules)
    addresses = list(first)
    for i in range(len(addresses), a):
        # The fourth demo crosses an earlier address with another operation.
        candidates = [p for p in addresses if (kinds[i], p) not in set(zip(kinds[:i], addresses))]
        if i != 3 or not candidates:
            candidates = [p for p in ADDRESSES if (kinds[i], p) not in set(zip(kinds[:i], addresses))]
        addresses.append(_choice(rng, candidates or list(ADDRESSES)))
    train = []
    for kind, address in zip(kinds, addresses):
        pool = [p for p in POSITIONS if p != address]
        if train and len(train) == 1:
            old_source, old_address, _ = train[0]
            pool = [p for p in pool if p != old_source and
                    (address[0] - p[0], address[1] - p[1]) !=
                    (old_address[0] - old_source[0], old_address[1] - old_source[1])]
        train.append((_choice(rng, pool), address, kind))
    known_rows = {p[0] for p in addresses}
    known_cols = {p[1] for p in addresses}
    trained = set(zip(kinds, addresses))
    pool = [(k, p) for k in active for p in ADDRESSES
            if p[0] in known_rows and p[1] in known_cols and (k, p) not in trained]
    test = []
    for _ in range(b):
        novel_address = [(k, p) for k, p in pool if p not in addresses]
        kind, address = _choice(rng, novel_address if novel_address and rng.random() < .75 else pool)
        pool.remove((kind, address))
        sources = [p for p in POSITIONS if p != address
                   and (p, address) not in {(s, d) for s, d, _ in train}]
        novel_source = [p for p in sources if p not in {s for s, _, _ in train}]
        source = _choice(rng, novel_source if novel_source and rng.random() < .6 else sources)
        test.append((source, address, kind))
    return train, test


def _draw_pair(rng, roles, gaps, plan):
    background, divider = roles
    source, address, kind = plan
    colors = [c for c in range(10) if c not in (background, divider, LOCATOR, MARKER)]
    palette = [int(c) for c in rng.choice(colors, int(rng.integers(1, 7)), replace=False)]
    markers = _choice(rng, _marker_options(kind, address))
    data = _data_tile(rng, background, palette, address, markers, kind)
    g = _blank(gaps, background, divider)
    tile = data.copy()
    tile[address] = LOCATOR
    for cell in markers:
        tile[cell] = MARKER
    g[_panel(source, gaps)] = tile
    positions = [p for p in POSITIONS if p != source]
    rng.shuffle(positions)
    active = int(rng.integers(1, 9))
    budget = 40 - int(np.count_nonzero(tile != background))
    for i, position in enumerate(positions[:active]):
        # Reserve at least one mark for every remaining active distractor.
        cap = min(9, budget - (active - i - 1))
        count = int(rng.integers(1, cap + 1))
        if rng.random() < .35:
            count = 1
        panel = g[_panel(position, gaps)]
        for index in rng.choice(9, count, replace=False):
            panel.flat[int(index)] = int(rng.choice(palette))
        budget -= count
    info = {"grid": g, "gaps": gaps, "background": background, "divider": divider,
            "source": source, "address": address, "kind": kind, "data": data}
    return {"input": g, "output": _pixel_output(info)}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate distinct examples under one fixed set of visible conventions."""
    a, b = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    available = [c for c in range(10) if c not in (LOCATOR, MARKER)]
    roles = tuple(int(c) for c in rng.choice(available, 2, replace=False))
    gaps = tuple(int(x) for x in rng.integers(1, 4, size=2))
    plans = _plans(rng, a, b)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, split_plans in zip(("train", "test"), plans):
        for plan in split_plans:
            for _ in range(16):
                pair = _draw_pair(rng, roles, gaps, plan)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check actual grids, exact causal contributions, witnesses, and split shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, conventions = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _array(pair["input"]), _array(pair["output"])
            assert g.shape == out.shape
            key = _key(g)
            assert key not in seen, "Repeated input"
            seen.add(key)
            info = _read(g)
            infos[split].append(info)
            conventions.add((info["gaps"], info["background"], info["divider"]))
            assert np.array_equal(out, _pixel_output(info)), "Incorrect transport"
            assert np.array_equal(out, solve(g)), "Input-only solve disagrees"
            controls = (g == LOCATOR) | (g == MARKER)
            assert np.array_equal(g[controls], out[controls]), "Controls must stay fixed"
            assert np.array_equal(g == LOCATOR, out == LOCATOR)
            assert np.array_equal(g == MARKER, out == MARKER)
            assert np.array_equal(g[info["mask"]], out[info["mask"]])
            assert _distinct_orientations(info["data"]), "Symmetry choice is invisible"
            assert _orientation_witness(info["data"], info["kind"], info["background"])
            # Omission A: a source-staying transformed field gives a different grid.
            assert not np.array_equal(out, _pixel_output(info, address=info["source"]))
            # Omission B: untransformed relocation gives a different data field.
            assert not np.array_equal(info["data"], _transform(info["data"], info["kind"]))
            target = out[_panel(info["address"], info["gaps"])]
            assert np.array_equal(np.sort(target.ravel()), np.sort(info["data"].ravel()))
            source = out[_panel(info["source"], info["gaps"])]
            assert not np.any((source != info["background"]) & (source != MARKER) & (source != LOCATOR))
            assert not np.array_equal(g, out)
    assert len(conventions) == 1, "Puzzle conventions changed"
    train, test = infos["train"], infos["test"]
    kinds = {i["kind"] for i in train}
    assert "P" in kinds and bool(kinds & {"H", "V"})
    assert a == 2 or kinds == {"P", "H", "V"}
    assert len({i["source"] for i in train}) >= 2
    assert len({i["address"][0] for i in train}) >= min(a, 3)
    assert len({i["address"][1] for i in train}) >= min(a, 3)
    assert len({(i["address"][0] - i["source"][0], i["address"][1] - i["source"][1])
                for i in train}) >= 2, "A fixed displacement explains training"
    trained = {(i["kind"], i["address"]) for i in train}
    moves = {(i["source"], i["address"]) for i in train}
    assert len({(i["kind"], i["address"]) for i in test}) == b
    for i in test:
        assert i["kind"] in kinds
        assert (i["kind"], i["address"]) not in trained
        assert (i["source"], i["address"]) not in moves
        assert i["address"][0] in {t["address"][0] for t in train}
        assert i["address"][1] in {t["address"][1] for t in train}


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    """42 required seeds, all explicit counts, metamorphic and corruption checks.

    No files, traces, global caches, or unbounded diagnostics are written.
    Only one puzzle and its deterministic repeat are retained at a time.
    """
    shapes, roles, sources, addresses, marker_lengths = set(), set(), set(), set(), set()
    structures, data_counts, distractor_counts, palettes = set(), set(), set(), set()
    kinds, pair_count, crossed, crop_offsets, diagonal = set(), 0, 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _equal(puzzle, generate_puzzle(seed)), "Nondeterministic puzzle"
        train_info = [_read(p["input"]) for p in puzzle["train"]]
        crossed += len({i["address"] for i in train_info}) < len(train_info)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                info = _read(inp)
                assert np.array_equal(solve(inp), out)
                assert np.array_equal(inp, before), "solve mutated its input"
                assert np.array_equal(solve(inp.T.copy()), out.T)
                assert np.array_equal(solve(inp[::-1, ::-1].copy()), out[::-1, ::-1])
                shapes.add(inp.shape)
                roles.add((info["background"], info["divider"]))
                sources.add(info["source"])
                addresses.add(info["address"])
                marker_lengths.add(len(info["markers"]))
                kinds.add(info["kind"])
                count = int(np.count_nonzero(info["data"] != info["background"]))
                data_counts.add(count)
                distractor_counts.add(sum(np.any(inp[_panel(p, info["gaps"])] != info["background"])
                                          for p in POSITIONS if p != info["source"]))
                palettes.add(tuple(sorted(set(map(int, np.unique(info["data"]))) - {info["background"]})))
                normalized, mapping = [], {}
                for cell in inp.flat:
                    normalized.append(mapping.setdefault(int(cell), len(mapping)))
                structures.add(hashlib.sha256(bytes(normalized)).hexdigest())
                transformed = _transform(info["data"], info["kind"])
                rr, cc = np.nonzero(transformed != info["background"])
                crop_offsets += int(rr.min() > 0 or cc.min() > 0)
                cells = set(map(tuple, np.argwhere(info["data"] != info["background"]).tolist()))
                diagonal += int(any((r + 1, c + 1) in cells and (r + 1, c) not in cells
                                    and (r, c + 1) not in cells for r, c in cells))
                pair_count += 1
    assert len(shapes) == 9 and len(roles) >= 20
    assert sources == set(POSITIONS) and addresses == set(ADDRESSES)
    assert kinds == {"H", "V", "P"} and marker_lengths == {1, 2, 3}
    assert len(structures) >= 200 and len(data_counts) >= 5
    assert distractor_counts == set(range(1, 9)) and len(palettes) >= 30
    assert crossed == 42 and crop_offsets >= 10 and diagonal >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(20261023 + 31 * a + b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Reject wrong symmetry, source-staying labels, additive copies, moved controls.
    good = generate_puzzle(0)
    info = _read(good["train"][0]["input"])
    corruptions = [_pixel_output(info, kind=k) for k in ("H", "V", "P") if k != info["kind"]]
    corruptions.append(_pixel_output(info, address=info["source"]))
    additive = good["train"][0]["output"].copy()
    source = _panel(info["source"], info["gaps"])
    original = good["train"][0]["input"][source]
    extra = additive[source]
    extra[info["data"] != info["background"]] = original[info["data"] != info["background"]]
    corruptions.append(additive)
    lost_control = good["train"][0]["output"].copy()
    lost_control[good["train"][0]["input"] == LOCATOR] = info["background"]
    corruptions.append(lost_control)
    for wrong in corruptions:
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in good[s]] for s in good}
        bad["train"][0]["output"] = wrong
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("Accepted corrupted labels")
    return {"seeds": 42, "pairs": pair_count, "grid_shapes": len(shapes),
            "role_pairs": len(roles), "normalized_inputs": len(structures),
            "payload_cell_counts": sorted(data_counts), "marker_lengths": sorted(marker_lengths),
            "distractor_panel_counts": sorted(map(int, distractor_counts)),
            "payload_palettes": len(palettes), "crossed_demo_puzzles": crossed,
            "crop_offset_pairs": crop_offsets, "diagonal_gap_pairs": diagonal,
            "legal_count_combinations": 30, "semantic_checks": "generator-owned only"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
