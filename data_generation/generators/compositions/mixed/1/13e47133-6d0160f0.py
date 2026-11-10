"""Contour cargo: diagonal-key painting followed by exact local-address transport.

The outer array is always 3x3. Panels are 5..9 cells on each axis; the
unique color-9 locator occupies local rows/columns 0..2. No filesystem I/O.
"""
from __future__ import annotations

from collections import deque
import hashlib

import numpy as np


LOCATOR = 9
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
POSITIONS = tuple((r, c) for r in range(3) for c in range(3))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _require(not isinstance(value, (bool, np.bool_)) and
             isinstance(value, (int, np.integer)), name + " must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), name + " out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _lattice(h, w):
    mask = np.zeros((3 * h + 2, 3 * w + 2), dtype=bool)
    mask[h, :] = mask[2 * h + 1, :] = True
    mask[:, w] = mask[:, 2 * w + 1] = True
    return mask


def _panel(position, h, w):
    r, c = position
    return slice(r * (h + 1), r * (h + 1) + h), slice(c * (w + 1), c * (w + 1) + w)


def _blank(h, w, background, divider):
    result = np.full((3 * h + 2, 3 * w + 2), background, dtype=np.int8)
    result[_lattice(h, w)] = divider
    return result


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        region = np.zeros(mask.shape, dtype=bool)
        queue = deque([(sr, sc)])
        seen[sr, sc] = True
        while queue:
            r, c = queue.popleft()
            region[r, c] = True
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(region)
    return result


def _depths(mask):
    """BFS distance from the eight-neighbor boundary, boundary depth zero."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=np.int8)
    queue = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc])
               for dr, dc in DIR8):
            depth[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and depth[rr, cc] < 0:
                depth[rr, cc] = depth[r, c] + 1
                queue.append((rr, cc))
    return depth


def _erode(mask, diagonal=True):
    """Iterative erosion, used for validation and metric counterexamples."""
    h, w = mask.shape
    current = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int8)
    level = 0
    while np.any(current):
        padded = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in (DIR8 if diagonal else DIR4):
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~interior] = level
        current = interior
        level += 1
    return depth


def _paths(mask, depth):
    h, w = mask.shape

    def inside(r, c):
        return 0 <= r < h and 0 <= c < w and mask[r, c]

    result = []
    for r, c in np.argwhere(mask & (depth == 0)):
        r, c = int(r), int(c)
        for dr in (-1, 1):
            for dc in (-1, 1):
                if inside(r - dr, c) or inside(r, c - dc):
                    continue
                # A degenerate corner can supply a one-entry key.
                cells = []
                k = 0
                while inside(r + dr * k, c + dc * k) and depth[r + dr * k, c + dc * k] == k:
                    cells.append((r + dr * k, c + dc * k))
                    k += 1
                if cells:
                    result.append((tuple(cells), (dr, dc)))
    return result


def _key(panel, region, depth, background):
    """The last non-neutral cue terminates the key; neutral gaps are entries."""
    cues = [tuple(map(int, p)) for p in np.argwhere(region & (panel != background) & (panel != LOCATOR))]
    if not cues:
        return (background,), ()
    length = max(int(depth[p]) for p in cues) + 1
    readings = []
    for cells, direction in _paths(region, depth):
        if len(cells) < length or any(p not in cells[:length] for p in cues):
            continue
        # The marker cannot obscure even a neutral key entry.
        if any(panel[p] == LOCATOR for p in cells[:length]):
            continue
        period = tuple(int(panel[p]) for p in cells[:length])
        if period[-1] != background:
            readings.append((period, direction))
    periods = {p for p, _ in readings}
    _require(len(periods) == 1, "region key is absent, non-diagonal, or ambiguous")
    return next(iter(periods)), tuple(sorted({d for _, d in readings}))


def _paint(panel, background, divider, *, erosion=False):
    result = panel.copy()
    records = []
    for region in _components(panel != divider):
        depth = _erode(region) if erosion else _depths(region)
        period, directions = _key(panel, region, depth, background)
        result[region] = np.asarray(period, dtype=np.int8)[depth[region] % len(period)]
        records.append((region, depth, period, directions))
    result[panel == LOCATOR] = LOCATOR
    return result, records


def _read(input_grid):
    grid = _grid(input_grid)
    gh, gw = grid.shape
    _require((gh - 2) % 3 == 0 and (gw - 2) % 3 == 0, "not a 3x3 panel array")
    h, w = (gh - 2) // 3, (gw - 2) // 3
    _require(5 <= h <= 9 and 5 <= w <= 9, "panel dimensions must be 5..9")
    lattice = _lattice(h, w)
    values = np.unique(grid[lattice])
    _require(len(values) == 1 and values[0] != LOCATOR, "lattice must be monochrome")
    divider = int(values[0])
    counts = np.bincount(grid[~lattice].astype(int), minlength=10)
    counts[divider] = counts[LOCATOR] = 0
    candidates = np.flatnonzero(counts == counts.max())
    _require(len(candidates) == 1 and counts[candidates[0]] > 0, "background must be the unique panel mode")
    background = int(candidates[0])
    locations = np.argwhere(grid == LOCATOR)
    _require(locations.shape == (1, 2), "exactly one color-9 locator required")
    r, c = map(int, locations[0])
    _require(not lattice[r, c], "locator on lattice")
    source = (r // (h + 1), c // (w + 1))
    address = (r % (h + 1), c % (w + 1))
    _require(address in POSITIONS, "locator local coordinates must both be 0..2")
    payload = grid[_panel(source, h, w)].copy()
    return grid, h, w, lattice, background, divider, source, address, payload


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, h, w, lattice, bg, divider, source, address, payload = _read(input_grid)
    filled, _ = _paint(payload, bg, divider)
    output = _blank(h, w, bg, divider)
    output[_panel(address, h, w)] = filled
    return output


def _wall_mask(rng, h, w, kind):
    transpose = bool(rng.integers(2))
    a, b = (w, h) if transpose else (h, w)
    mask = np.zeros((a, b), dtype=bool)
    if kind == "notch":
        r = int(rng.integers(2, a - 2))
        length = int(rng.integers(2, b // 2 + 1))
        mask[r, :length] = True
    elif kind == "split":
        cut = int(rng.integers(3, b - 3))
        mask[:, cut] = True
    elif kind == "box":
        top = int(rng.integers(1, a - 5))
        bottom = int(rng.integers(top + 4, a))
        far = int(rng.integers(3, b - 2))
        mask[top, :far + 1] = mask[bottom, :far + 1] = True
        mask[top:bottom + 1, far] = True
    elif kind == "step":
        cut = b // 2
        top = int(rng.integers(1, a - 3))
        bottom = int(rng.integers(top + 2, a))
        mask[:, cut] = True
        mask[top + 1:bottom, cut] = False
        mask[top:bottom + 1, cut + 1] = True
    else:
        _require(kind == "plain", "unknown geometry")
    if transpose:
        mask = mask.T
    if rng.integers(2):
        mask = mask[::-1]
    if rng.integers(2):
        mask = mask[:, ::-1]
    return mask.copy()


def _metric_witness(panel, filled, records):
    """Actual colored cells differ from both rectangular rings and L1 erosion."""
    for region, depth, period, _ in records:
        rr, cc = np.nonzero(region)
        if len(period) < 2 or len(rr) == (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1):
            continue
        eligible = region & (panel != LOCATOR)
        box = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
        box_image = np.full(panel.shape, -1, dtype=np.int8)
        box_image[region] = np.asarray(period)[box % len(period)]
        l1 = _erode(region, diagonal=False)
        l1_image = np.full(panel.shape, -1, dtype=np.int8)
        l1_image[region] = np.asarray(period)[l1[region] % len(period)]
        if np.any(filled[eligible] != box_image[eligible]) and np.any(filled[eligible] != l1_image[eligible]):
            return True
    return False


def _draw_payload(rng, h, w, bg, divider, address, kind, focus):
    walls = _wall_mask(rng, h, w, kind)
    _require(not walls[address], "locator overlaps a wall")
    raw = np.full((h, w), bg, dtype=np.int8)
    raw[walls] = divider
    filled = raw.copy()
    palette = [c for c in range(9) if c not in (bg, divider)]
    regions = _components(~walls)
    options = []
    for region in regions:
        depth = _depths(region)
        paths = [(cells, direction) for cells, direction in _paths(region, depth)]
        options.append((region, depth, paths))
    if focus == "repeat":
        choices = [(i, cells, direction) for i, (_, _, paths) in enumerate(options)
                   for cells, direction in paths if len(cells) >= 4 and direction[0] == 1
                   and address not in cells[:3]]
        length, gap = 3, 1
    elif focus == "metric":
        choices = [(i, cells, direction) for i, (_, _, paths) in enumerate(options)
                   for cells, direction in paths if len(cells) >= 2 and direction[0] == -1
                   and address not in cells[:2]]
        length, gap = 2, 0
    else:
        choices = [(i, cells, direction) for i, (_, _, paths) in enumerate(options)
                   for cells, direction in paths if len(cells) >= 2 and address not in cells[:2]]
        length, gap = 2, None
    _require(bool(choices), "no valid main key")
    main_i, main_path, _ = choices[int(rng.integers(len(choices)))]
    for i, (region, depth, paths) in enumerate(options):
        if i == main_i:
            key_length, key_gap, cells = length, gap, main_path
        else:
            if rng.random() < 0.25 or not paths:
                continue
            cells, _ = paths[int(rng.integers(len(paths)))]
            key_length = int(rng.integers(1, min(3, len(cells)) + 1))
            if address in cells[:key_length]:
                continue
            key_gap = int(rng.integers(key_length - 1)) if key_length > 1 and rng.random() < 0.35 else None
        # Unforced main keys may be longer, including single-pair three-color cycles.
        if i == main_i and focus == "mixed":
            key_length = int(rng.integers(2, min(3, len(cells)) + 1))
            if address in cells[:key_length]:
                key_length = 2
            key_gap = int(rng.integers(key_length - 1)) if rng.random() < 0.35 else None
        period = [int(c) for c in rng.choice(palette, key_length, replace=False)]
        if key_gap is not None:
            period[key_gap] = bg
        for p, color in zip(cells[:key_length], period):
            raw[p] = color
        filled[region] = np.asarray(period, dtype=np.int8)[depth[region] % key_length]
    raw[address] = filled[address] = LOCATOR
    _require(not np.array_equal(raw, filled), "painting must visibly complete the payload")
    decoded, records = _paint(raw, bg, divider)
    _require(np.array_equal(decoded, filled), "key construction did not decode")
    if focus == "metric":
        _require(_metric_witness(raw, filled, records), "no metric witness")
    return raw, filled


def _draw_pair(rng, source, address, kind, focus, bg, divider):
    low = 7 if focus in ("repeat", "metric") or kind in ("split", "box", "step") else 5
    h, w = map(int, rng.integers(low, 10, size=2))
    raw, filled = _draw_payload(rng, h, w, bg, divider, address, kind, focus)
    grid = _blank(h, w, bg, divider)
    others = [p for p in POSITIONS if p != source]
    rng.shuffle(others)
    palette = [c for c in range(9) if c not in (bg, divider)]
    for i, position in enumerate(others):
        panel = grid[_panel(position, h, w)]
        if i < 2:
            # Valid-looking A keys prevent diagonal-pattern selection replacing B.
            corner = int(rng.integers(4))
            r, dr = (0, 1) if corner < 2 else (h - 1, -1)
            c, dc = (0, 1) if corner % 2 == 0 else (w - 1, -1)
            colors = rng.choice(palette, 2, replace=False)
            panel[r, c], panel[r + dr, c + dc] = colors
        elif i == 2:
            panel.flat[int(rng.integers(h * w))] = int(rng.choice(palette))
        else:
            density = float(rng.choice((0.0, 0.04, 0.12, 0.28, 0.43)))
            count = int(round(h * w * density))
            if count:
                cells = rng.choice(h * w, count, replace=False)
                panel.flat[cells] = rng.choice(palette, count)
            if rng.random() < 0.3:
                panel[_wall_mask(rng, h, w, "notch")] = divider
    grid[_panel(source, h, w)] = raw
    output = _blank(h, w, bg, divider)
    output[_panel(address, h, w)] = filled
    # Role interpretation must come from the returned input alone.
    _require(_read(grid)[4] == bg, "background not observable")
    return {"input": grid, "output": output}


def _schedule(rng, nt, ne):
    rows, cols = rng.permutation(3), rng.permutation(3)
    if rows[0] == cols[0] and rows[1] == cols[1]:
        cols[0], cols[1] = cols[1], cols[0]
    addresses = [(int(rows[i]), int(cols[i])) for i in range(min(3, nt))]
    addresses += [POSITIONS[int(rng.integers(9))] for _ in range(nt - len(addresses))]
    sources = []
    for i, address in enumerate(addresses):
        eligible = [p for p in POSITIONS if (i >= 2 or p != address) and
                    (i != 1 or p != sources[0])]
        sources.append(eligible[int(rng.integers(len(eligible)))])
    pool = [(r, c) for r in sorted({p[0] for p in addresses})
            for c in sorted({p[1] for p in addresses})]
    rng.shuffle(pool)
    novel = [p for p in pool if p not in addresses]
    familiar = [p for p in pool if p in addresses]
    tests = (novel + familiar)[:ne]
    test_sources = []
    for address in tests:
        eligible = [p for p in POSITIONS if (p, address) not in set(zip(sources, addresses))]
        unseen = [p for p in eligible if p not in sources]
        pool_sources = unseen if unseen and rng.random() < 0.65 else eligible
        test_sources.append(pool_sources[int(rng.integers(len(pool_sources)))])
    return sources, addresses, test_sources, tests


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return bytes(a.shape) + a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ne = _integer(num_test, "num_test", 1, 4)
    _require(nt > ne, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    sources, addresses, test_sources, test_addresses = _schedule(rng, nt, ne)
    kinds = ["plain", "notch", "split", str(rng.choice(("box", "step")))]
    kinds = kinds[:nt] + [str(rng.choice(("plain", "notch", "split", "box", "step")))
                          for _ in range(max(0, nt - 4))]
    test_kinds = [str(rng.choice(sorted(set(kinds)))) for _ in range(ne)]
    backgrounds = [int(c) for c in rng.permutation(9)]
    puzzle, used = {"train": [], "test": []}, set()
    for i in range(nt + ne):
        is_train = i < nt
        j = i if is_train else i - nt
        source = sources[j] if is_train else test_sources[j]
        address = addresses[j] if is_train else test_addresses[j]
        kind = kinds[j] if is_train else test_kinds[j]
        focus = "repeat" if i == 0 else "metric" if i == 1 else "mixed"
        bg = backgrounds[i % 9]
        divider = backgrounds[(i + 1) % 9]
        for _ in range(64):
            try:
                pair = _draw_pair(rng, source, address, kind, focus, bg, divider)
                signature = _signature(pair["input"])
                _require(signature not in used, "duplicate input")
            except ValueError:
                continue
            used.add(signature)
            puzzle["train" if is_train else "test"].append(pair)
            break
        else:
            raise RuntimeError("bounded construction exhausted")
    # Demonstration order carries no convention.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    grid, h, w, lattice, bg, divider, source, address, payload = _read(pair["input"])
    out = _grid(pair["output"])
    assert out.shape == grid.shape
    filled, records = _paint(payload, bg, divider, erosion=True)
    expected = _blank(h, w, bg, divider)
    expected[_panel(address, h, w)] = filled
    assert np.array_equal(out, expected), "contours, transport, or clearing incorrect"
    assert np.array_equal(out[lattice], grid[lattice]), "external lattice changed"
    assert np.count_nonzero(out == LOCATOR) == 1
    assert np.array_equal(out[_panel(address, h, w)] == divider, payload == divider)
    assert any(np.any(grid[_panel(p, h, w)] != bg) for p in POSITIONS if p != source)
    assert not np.array_equal(payload, filled), "A has no effect"
    assert not np.array_equal(grid, out)
    assert set(map(int, np.unique(out))) <= set(map(int, np.unique(grid)))
    for region, depth, _, _ in records:
        assert np.array_equal(depth, _depths(region)), "BFS/erosion mismatch"
    return {"h": h, "w": w, "bg": bg, "divider": divider, "source": source,
            "address": address, "payload": payload, "filled": filled, "records": records}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    analyses, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for value in pair.values():
                _grid(value)
            signature = _signature(pair["input"])
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            analyses[split].append(_analyze(pair))
    train, test = analyses["train"], analyses["test"]
    assert len({a["bg"] for a in train}) >= 2
    assert len({a["divider"] for a in train + test}) >= 2
    assert len({a["source"] for a in train}) >= 2
    assert len({a["address"][0] for a in train}) >= 2
    assert len({a["address"][1] for a in train}) >= 2
    assert sum(a["source"] != a["address"] for a in train) >= 2, "B needs two transport witnesses"
    assert any(r != c for r, c in (a["address"] for a in train)), "no row/column order witness"
    assert any(_metric_witness(a["payload"], a["filled"], a["records"]) for a in train)
    assert any(len(p) == 3 and p[1] == a["bg"] and int(d.max()) >= 3
               for a in train for _, d, p, _ in a["records"]), "no repeating internal-neutral key"
    assert any(len(p) >= 2 and p[0] == a["bg"] for a in train for _, _, p, _ in a["records"])
    marker_hole_witness = False
    for a in train:
        for region, _, period, _ in a["records"]:
            if not np.any(region & (a["payload"] == LOCATOR)):
                continue
            open_without_marker = region & (a["payload"] != LOCATOR)
            wrong_depth = _erode(open_without_marker)
            wrong_colors = np.asarray(period)[wrong_depth[open_without_marker] % len(period)]
            marker_hole_witness |= bool(np.any(wrong_colors != a["filled"][open_without_marker]))
    assert marker_hole_witness, "no evidence that marker is transparent to erosion"
    assert len({a["address"] for a in test}) == ne
    rows, cols = {a["address"][0] for a in train}, {a["address"][1] for a in train}
    assert all(a["address"][0] in rows and a["address"][1] in cols for a in test)
    trained_moves = {(a["source"], a["address"]) for a in train}
    assert all((a["source"], a["address"]) not in trained_moves for a in test), "test move repeats training"
    trained_addresses = {a["address"] for a in train}
    available_addresses = {(r, c) for r in rows for c in cols} - trained_addresses
    if available_addresses:
        assert any(a["address"] in available_addresses for a in test), "missing address recombination shift"
    # Parent ablations: direct transport of sparse marks, or painting in place.
    for a in train:
        no_a = _blank(a["h"], a["w"], a["bg"], a["divider"])
        no_a[_panel(a["address"], a["h"], a["w"])] = a["payload"]
        full = _blank(a["h"], a["w"], a["bg"], a["divider"])
        full[_panel(a["address"], a["h"], a["w"])] = a["filled"]
        assert not np.array_equal(no_a, full)
        if a["source"] != a["address"]:
            no_b = _blank(a["h"], a["w"], a["bg"], a["divider"])
            no_b[_panel(a["source"], a["h"], a["w"])] = a["filled"]
            assert not np.array_equal(no_b, full)


def _puzzle_hash(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                digest.update(_signature(pair[role]))
    return digest.hexdigest()


def self_test():
    """42 seeds, one puzzle at a time; no samples, logs, or traces are saved."""
    shapes, walls, role_pairs, sources, destinations, periods, directions, hashes = (set() for _ in range(8))
    region_counts, novel_addresses = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_hash(puzzle) == _puzzle_hash(generate_puzzle(seed))
        hashes.add(_puzzle_hash(puzzle))
        trained = {_read(p["input"])[7] for p in puzzle["train"]}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                immutable_input = pair["input"].copy()
                immutable_input.setflags(write=False)
                assert np.array_equal(solve(immutable_input), pair["output"])
                assert np.array_equal(immutable_input, pair["input"]), "solver mutated its input"
                shapes.add((a["h"], a["w"]))
                walls.add(_signature((a["payload"] == a["divider"]).astype(np.int8)))
                role_pairs.add((a["bg"], a["divider"]))
                sources.add(a["source"])
                destinations.add(a["address"])
                region_counts.add(len(a["records"]))
                novel_addresses += split == "test" and a["address"] not in trained
                for _, _, period, orientation in a["records"]:
                    periods.add(len(period))
                    directions.update(orientation)
    assert len(hashes) == 42 and len(shapes) >= 12 and len(walls) >= 45
    assert len(role_pairs) >= 45 and len(sources) == len(destinations) == 9
    assert {1, 2} <= region_counts and {1, 2, 3} <= periods
    assert len(directions) == 4 and novel_addresses >= 30
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(913, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0, 2, 1)
    nested = {s: [{k: v.tolist() for k, v in p.items()} for p in original[s]] for s in original}
    validate_generated_puzzle(nested)
    assert np.array_equal(solve(nested["test"][0]["input"]), original["test"][0]["output"])
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    g, h, w, lattice, bg, *_ = _read(broken["test"][0]["input"])
    r, c = map(int, np.argwhere(lattice)[0])
    broken["test"][0]["output"][r, c] = bg
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt lattice accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, input-only solving and parent ablations")
