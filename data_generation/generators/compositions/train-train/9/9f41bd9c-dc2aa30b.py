"""Rank-loaded fringe: count rank is a rigid tile's drag distance.

Only standard-library modules and NumPy are used. Coordinates in helpers are
canonical (floor below); solve recognizes the orientation from the input.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0, 98082007, 984291457, 748039272, 851222857, 477618654,
              528935588, 249932243, 312772467, 121976077, 881059514,
              239284773, 732213434, 971295689, 977340742, 123585365,
              646025188, 272926631, 28037192, 44587806, 3541440,
              279850855, 230737888, 283429649, 382000546, 468698189,
              251763682, 254956750, 29548203, 341882855, 714154692,
              891912271, 138851478, 803089542, 434155986, 911637135,
              983665821, 649701375, 449924800, 540062210, 192117481,
              128402820)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _canonical_parse(a):
    """Read a floor-below input. No output, seed, or generation data is used."""
    height, width = a.shape
    floor = int(a[-1, 0])
    _require(np.all(a[-1] == floor), "no bottom floor")
    floor_at = height
    while floor_at > 0 and np.all(a[floor_at - 1] == floor):
        floor_at -= 1
    _require(0 < floor_at < height, "no scene above floor")
    # Floor-colored bars reveal every tile's full width, including empty tiles.
    bar_cells = np.argwhere(a[:floor_at] == floor)
    _require(len(bar_cells) > 0, "no carriers")
    bars = sorted(set(int(r) for r in bar_cells[:, 0]))
    number = len(bars)
    _require(3 <= number <= 5, "expected 3..5 tiles")
    cols = np.flatnonzero(a[bars[0]] == floor)
    m, anchor = len(cols), int(cols[0])
    _require(3 <= m <= 6 and width >= m + number - 1, "tile/canvas width")
    _require(np.array_equal(cols, np.arange(anchor, anchor + m)), "non-solid carrier")
    _require(anchor in (0, width - m) and width > m, "carrier must touch one side")
    side = 0 if anchor == 0 else 1  # 0: starts left; 1: starts right
    h = bars[1] - bars[0] - 2
    _require(1 <= h <= 3, "tile height")
    _require(bars == [bars[0] + j * (h + 2) for j in range(number)], "irregular bands")
    _require(bars[-1] + h + 1 == floor_at, "lowest tile must touch floor")
    background = int(a[bars[0], -1 if side == 0 else 0])
    # One blank row separates the head from the first carrier.
    head_end = bars[0] - 1
    _require(head_end >= 2 and np.all(a[head_end] == background), "head gap")
    token = int(a[head_end - 1, anchor])
    _require(len({background, token, floor}) == 3, "ambiguous token role")
    head_top = head_end - 1
    while head_top > 0 and a[head_top - 1, anchor] == token:
        head_top -= 1
    head_h = head_end - head_top
    _require(head_h in (1, 2), "head height")
    ink_row = head_top - 1
    _require(ink_row >= 0, "missing reservoir")
    ink = int(a[ink_row, anchor])
    _require(len({background, token, floor, ink}) == 4, "four distinct roles required")
    _require(set(map(int, np.unique(a))) == {background, token, floor, ink}, "extra colors")
    tiles = [a[r + 1:r + 1 + h, anchor:anchor + m].copy() for r in bars]
    _require(all(set(map(int, np.unique(t))) <= {background, token} for t in tiles),
             "binary payloads required")
    counts = [int(np.count_nonzero(t == token)) for t in tiles]
    _require(len(set(counts)) == number, "token-count ties are not allowed")
    _require(np.any(tiles[-1][-1] == token), "bottom tile needs a contact token")
    # Exact reconstruction rejects all extra cells and validates all bars/gaps.
    reconstructed = np.full_like(a, background)
    reconstructed[floor_at:] = floor
    reconstructed[ink_row, anchor:anchor + m] = ink
    reconstructed[head_top:head_end, anchor:anchor + m] = token
    for r, tile in zip(bars, tiles):
        reconstructed[r, anchor:anchor + m] = floor
        reconstructed[r + 1:r + h + 1, anchor:anchor + m] = tile
    _require(np.array_equal(a, reconstructed), "unexpected scene cells")
    ranks = [sum(other < c for other in counts) for c in counts]
    return dict(shape=a.shape, floor=floor, floor_at=floor_at, background=background,
                token=token, ink=ink, bars=bars, tiles=tiles, counts=counts,
                ranks=ranks, m=m, h=h, n=number, anchor=anchor, side=side,
                ink_row=ink_row, head_top=head_top, head_end=head_end)


def _parse(input_grid):
    a = _grid(input_grid)
    candidates = []
    for k in range(4):
        canonical = np.rot90(a, k)
        try:
            info = _canonical_parse(canonical)
        except (AssertionError, IndexError, ValueError):
            continue
        candidates.append((k, canonical, info))
    _require(len(candidates) == 1, "input must reveal one orientation and brush")
    return candidates[0]


def _render(info, *, lags=None, sweep=True, contact=True):
    """Render a parsed input's transformation (also used for counterfactuals)."""
    height, width = info["shape"]
    bg, token, floor = info["background"], info["token"], info["floor"]
    m, h, side = info["m"], info["h"], info["side"]
    destination = width - m if side == 0 else 0
    if not sweep:
        destination = info["anchor"]
    out = np.full((height, width), bg, dtype=np.int8)
    out[info["floor_at"]:] = floor
    out[info["ink_row"], destination:destination + m] = info["ink"]
    out[info["head_top"]:info["head_end"], destination:destination + m] = token
    lags = info["ranks"] if lags is None else list(lags)
    anchors = []
    for r, tile, lag in zip(info["bars"], info["tiles"], lags):
        start = destination + (-lag if side == 0 else lag) if sweep else destination
        _require(0 <= start <= width - m, "lag leaves canvas")
        anchors.append(start)
        out[r, start:start + m] = floor
        out[r + 1:r + 1 + h, start:start + m] = tile
    if sweep:
        occupied = np.flatnonzero(info["tiles"][-1][-1] == token)
        offset = int(occupied[-1] if side == 0 else occupied[0]) if contact else (m - 1 if side == 0 else 0)
        tip = anchors[-1] + offset
        if side == 0:
            out[info["floor_at"], :tip] = info["ink"]
        else:
            out[info["floor_at"], tip + 1:] = info["ink"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the output entirely from visible input cells."""
    k, _, info = _parse(input_grid)
    return np.rot90(_render(info), -k).copy()


def _make_tile(rng, count, h, m, bg, token, side, contact):
    tile = np.full(h * m, bg, dtype=np.int8)
    if not contact:
        if count:
            tile[rng.choice(h * m, size=count, replace=False)] = token
    else:
        # Preserve the selected count, place a bottom token, and leave the
        # forward edge empty. The actual token tip then differs from the bar.
        edge = h * m - 1 if side == 0 else (h - 1) * m
        # Opposite edge occupied, forward edge empty: even the minimum
        # demonstration has an asymmetric motif that cannot be mirrored.
        forced = (h - 1) * m if side == 0 else h * m - 1
        tile[forced] = token
        available = [i for i in range(h * m) if i not in (edge, forced)]
        if count > 1:
            tile[rng.choice(available, size=count - 1, replace=False)] = token
    return tile.reshape(h, m)


def _make_pair(rng, n, geometry, side, rotation, colors, bottom_mode):
    m, h = geometry
    area = m * h
    # Gaps force rank, and forbid raw token quantity as the displacement.
    for _ in range(200):
        upper = area if bottom_mode == "max" else area + 1
        counts = sorted(int(c) for c in rng.choice(upper, size=n, replace=False))
        if any(b - a > 1 for a, b in zip(counts, counts[1:])):
            break
    else:
        raise AssertionError("could not choose gapped counts")
    eligible = [j for j, c in enumerate(counts) if 1 <= c < area]
    if bottom_mode == "max":
        bottom_rank = n - 1
    elif bottom_mode == "middle":
        bottom_rank = int(rng.choice([j for j in eligible if 0 < j < n - 1]))
    else:
        bottom_rank = int(rng.choice(eligible))
    order = [j for j in range(n) if j != bottom_rank]
    rng.shuffle(order)
    order.append(bottom_rank)
    if order == list(range(n)) or order == list(range(n - 1, -1, -1)):
        order[0], order[1] = order[1], order[0]
    bg, token, floor, ink = colors
    head_h = int(rng.integers(1, 3))
    # Head ink row + head + head gap + module bands + intermodule gaps.
    base_height = 1 + head_h + 1 + n * (h + 1) + n - 1
    room = 30 - base_height
    top = int(rng.integers(0, min(3, room - 1) + 1))
    floor_rows = int(rng.integers(1, min(4, room - top) + 1))
    height = base_height + top + floor_rows
    minimum_width = m + n - 1
    width = minimum_width if rng.random() < 0.30 else int(rng.integers(minimum_width + 1, min(30, minimum_width + 13) + 1))
    anchor = 0 if side == 0 else width - m
    a = np.full((height, width), bg, dtype=np.int8)
    a[-floor_rows:] = floor
    a[top, anchor:anchor + m] = ink
    a[top + 1:top + 1 + head_h, anchor:anchor + m] = token
    first_bar = top + 1 + head_h + 1
    for j, rank in enumerate(order):
        row = first_bar + j * (h + 2)
        a[row, anchor:anchor + m] = floor
        tile = _make_tile(rng, counts[rank], h, m, bg, token, side, j == n - 1)
        a[row + 1:row + h + 1, anchor:anchor + m] = tile
    info = _canonical_parse(a)
    # Generation constructs its expected cells directly; solve reparses them.
    b = _render(info)
    return {"input": np.rot90(a, rotation).copy(), "output": np.rot90(b, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    first_side = int(rng.integers(2))
    broad = [(m, h) for m in range(3, 7) for h in (2, 3) if m * h >= 7]
    first = broad[int(rng.integers(len(broad)))]
    second = [g for g in broad if g != first][int(rng.integers(len(broad) - 1))]
    train_geometries = [first, second]
    train_ns = [3, 4]
    for _ in range(num_train - 2):
        n = int(rng.integers(3, 5))
        options = [(m, h) for m in range(3, 7) for h in range(1, 4) if m * h >= n + 1]
        train_geometries.append(options[int(rng.integers(len(options)))])
        train_ns.append(n)
    test_options = [g for g in train_geometries if g[0] * g[1] >= 7]
    puzzle, seen = {"train": [], "test": []}, set()
    previous_token = None
    for split, number in (("train", num_train), ("test", num_test)):
        for j in range(number):
            n = train_ns[j] if split == "train" else 5
            geom = train_geometries[j] if split == "train" else test_options[int(rng.integers(len(test_options)))]
            side = first_side if j == 0 else 1 - first_side if j == 1 else int(rng.integers(2))
            mode = "middle" if split == "train" and j < 2 else "max" if split == "test" and j == 0 else "any"
            for _ in range(100):
                colors = tuple(int(c) for c in rng.choice(10, size=4, replace=False))
                if colors[1] == previous_token:
                    continue
                pair = _make_pair(rng, n, geom, side, rotation, colors, mode)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise AssertionError("unable to make distinct inputs")
            previous_token = colors[1]
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_cells(a, b, info):
    """Cell-by-cell invariant check, without calling solve or _render."""
    width = a.shape[1]
    m, side = info["m"], info["side"]
    dest = width - m if side == 0 else 0
    bg, token, floor, ink = (info[k] for k in ("background", "token", "floor", "ink"))
    expected = {}
    for c in range(dest, dest + m):
        expected[info["ink_row"], c] = ink
        for r in range(info["head_top"], info["head_end"]):
            expected[r, c] = token
    module_starts = []
    for j, (r, tile) in enumerate(zip(info["bars"], info["tiles"])):
        lag = sum(c < info["counts"][j] for c in info["counts"])
        start = dest - lag if side == 0 else dest + lag
        module_starts.append(start)
        _require(0 <= start <= width - m, "tile crosses boundary")
        _require(np.all(b[r, start:start + m] == floor), "carrier changed")
        _require(np.array_equal(b[r + 1:r + 1 + info["h"], start:start + m], tile), "tile pattern/height changed")
        for c in range(m):
            expected[r, start + c] = floor
            for dr in range(info["h"]):
                expected[r + 1 + dr, start + c] = int(tile[dr, c])
    occupied = [c for c, v in enumerate(info["tiles"][-1][-1]) if v == token]
    tip = module_starts[-1] + (max(occupied) if side == 0 else min(occupied))
    for r in range(b.shape[0]):
        for c in range(width):
            value = floor if r >= info["floor_at"] else bg
            if r == info["floor_at"] and ((side == 0 and c < tip) or (side == 1 and c > tip)):
                value = ink
            value = expected.get((r, c), value)
            _require(int(b[r, c]) == value, "wrong changed-cell set")
    _require(b[info["floor_at"], tip] == floor, "painted directly under tip")
    _require(np.all(b[info["floor_at"] + 1:] == floor), "deeper floor changed")
    # The geometric endpoint must differ from a carrier-edge approximation.
    _require(not np.array_equal(b, _render(info, contact=False)), "contact shortcut not exposed")
    # Parent ablations: no ranks, original depth shear, no sweep.
    _require(not np.array_equal(b, _render(info, lags=[0] * info["n"])), "counting irrelevant")
    _require(not np.array_equal(b, _render(info, lags=range(info["n"]))), "depth shortcut not exposed")
    _require(not np.array_equal(b, _render(info, sweep=False)), "dragging irrelevant")


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "shape changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            k, canonical, info = _parse(a)
            _check_cells(canonical, np.rot90(b, k), info)
            sorted_counts = sorted(info["counts"])
            _require(any(y - x > 1 for x, y in zip(sorted_counts, sorted_counts[1:])), "ungapped quantities")
            _require(info["ranks"] not in (list(range(info["n"])), list(range(info["n"] - 1, -1, -1))), "trivial vertical order")
            records[split].append((k, info))
    tr, te = records["train"], records["test"]
    _require(len({k for k, _ in tr + te}) == 1, "orientation changes within puzzle")
    _require({x["side"] for _, x in tr} == {0, 1}, "both directions must be trained")
    _require(len({(x["m"], x["h"]) for _, x in tr}) >= 2, "train tile geometries")
    _require([tr[0][1]["n"], tr[1][1]["n"]] == [3, 4], "introductory fringe sizes")
    _require(all(x["n"] in (3, 4) for _, x in tr), "training fringe sizes")
    _require(all(x["n"] == 5 for _, x in te), "test must extrapolate rank range")
    _require(all((x["m"], x["h"]) in {(v["m"], v["h"]) for _, v in tr} for _, x in te), "unseen tile geometry")
    _require(te[0][1]["ranks"][-1] == 4, "first test must expose extended endpoint lag")
    _require(tr[0][1]["token"] != tr[1][1]["token"], "train visible token roles must vary")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _semantic_checks():
    """Hand-calculated and metamorphic checks of the connecting mechanism."""
    # Counts 8,1,5,3 -> ranks 3,0,2,1. Head destination is column 9.
    tiles = [np.array([[2, 2, 2, 2, 2], [2, 2, 2, 0, 0]], dtype=np.int8),
             np.array([[0, 2, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int8),
             np.array([[2, 0, 0, 2, 0], [0, 2, 2, 0, 2]], dtype=np.int8),
             np.array([[0, 2, 0, 0, 0], [2, 0, 2, 0, 0]], dtype=np.int8)]
    a = np.zeros((21, 14), dtype=np.int8)
    a[19:] = 6
    a[1, :5], a[2, :5] = 9, 2
    expected = np.zeros_like(a)
    expected[19:] = 6
    expected[1, 9:14], expected[2, 9:14] = 9, 2
    for row, start, tile in zip((4, 8, 12, 16), (6, 9, 7, 8), tiles):
        a[row, :5] = 6
        a[row + 1:row + 3, :5] = tile
        expected[row, start:start + 5] = 6
        expected[row + 1:row + 3, start:start + 5] = tile
    expected[19, :10] = 9  # bottom starts at 8; its forward token offset is 2
    for mirror in (False, True):
        x = np.fliplr(a) if mirror else a
        y = np.fliplr(expected) if mirror else expected
        for rotation in range(4):
            _require(np.array_equal(solve(np.rot90(x, rotation)), np.rot90(y, rotation)),
                     "hand-calculated rank/contact example failed")
    info = _canonical_parse(a)
    _require(not np.array_equal(expected, _render(info, lags=info["counts"])),
             "raw-count shortcut was not distinguished")
    # Change OTHER payload's count 1 -> 4. Bottom payload is identical, but
    # its rank becomes 0 and it moves one cell toward the destination.
    changed_counts = a.copy()
    changed_counts[9, [0, 2, 3]] = 2
    moved = solve(changed_counts)
    _require(np.array_equal(moved[17:19, 9:14], tiles[-1]), "global comparison ignored")
    _require(np.all(moved[19, :11] == 9) and moved[19, 11] == 6,
             "count mutation did not move endpoint")
    # Move a bottom token without changing its count: rank and bars stay put,
    # while the floor endpoint moves from 10 to 12.
    changed_footprint = a.copy()
    changed_footprint[18, 2], changed_footprint[18, 4] = 0, 2
    expected_footprint = expected.copy()
    expected_footprint[18, 10], expected_footprint[18, 12] = 0, 2
    expected_footprint[19, 10:12] = 9
    _require(np.array_equal(solve(changed_footprint), expected_footprint),
             "equal-count footprint mutation failed")
    tied = a.copy()
    tied[9, [0, 2]] = 2  # second and fourth payloads now both count 3
    try:
        solve(tied)
    except AssertionError:
        pass
    else:
        raise AssertionError("solver accepted count ties")
    # A valid tight-canvas family member with an empty tile and an empty stroke.
    edge = np.zeros((15, 5), dtype=np.int8)
    edge[0, :3], edge[1, :3], edge[14:] = 9, 2, 6
    for row in (3, 7, 11):
        edge[row, :3] = 6
    edge[4, 1] = 2  # rank 1
    edge[12:14, 0] = 2  # rank 2, bottom forwardmost contact offset 0
    edge_out = solve(edge)
    _require(np.all(edge_out[14] == 6), "empty stroke boundary is incorrect")
    _require(np.array_equal(edge_out[12:14, :3], edge[12:14, :3]), "edge tile clipped")


def self_test() -> dict:
    """Bounded 42-seed coverage plus count/API and semantic mutation checks."""
    shapes, structures, geometries, palettes, directions, rotations = (set() for _ in range(6))
    bottom_ranks, count_sets, floor_depths, head_heights = (set() for _ in range(4))
    tight = at_top = empty_tiles = full_tiles = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                k, a, info = _parse(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                shapes.add(pair["input"].shape)
                roles = {info[c]: j for j, c in enumerate(("background", "token", "floor", "ink"))}
                structures.add((a.shape, bytes(roles[int(v)] for v in a.flat)))
                geometries.add((info["m"], info["h"]))
                palettes.add(tuple(info[c] for c in ("background", "token", "floor", "ink")))
                directions.add(info["side"])
                rotations.add(k)
                bottom_ranks.add(info["ranks"][-1])
                count_sets.add(tuple(sorted(info["counts"])))
                floor_depths.add(a.shape[0] - info["floor_at"])
                head_heights.add(info["head_end"] - info["head_top"])
                tight += a.shape[1] == info["m"] + info["n"] - 1
                at_top += info["ink_row"] == 0
                empty_tiles += info["counts"].count(0)
                full_tiles += info["counts"].count(info["m"] * info["h"])
    _require(len(shapes) >= 40 and len(structures) >= 180, "insufficient structural variation")
    _require(len(geometries) >= 9 and len(palettes) >= 100, "insufficient geometry/palette variation")
    _require(directions == {0, 1} and rotations == set(range(4)), "orientation coverage")
    _require(bottom_ranks == set(range(5)), "bottom lag coverage")
    _require(len(count_sets) >= 80 and floor_depths == {1, 2, 3, 4} and head_heights == {1, 2}, "quantity/surface coverage")
    _require(tight >= 25 and at_top >= 20, "boundary coverage")
    _require(empty_tiles >= 10 and full_tiles >= 10, "empty/full payload coverage")
    _require(_same(generate_puzzle(123), generate_puzzle(123)), "nondeterministic")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(17, nt, ns)
            _require((len(p["train"]), len(p["test"])) == (nt, ns), "count mismatch")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _semantic_checks()
    # Mutate a real label: the validator must catch an altered floor cell.
    p = generate_puzzle(0, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    k, _, info = _parse(bad["train"][0]["input"])
    out = np.rot90(bad["train"][0]["output"], k)
    out[-1, -1] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator missed changed floor")
    return dict(seeds=42, pairs=210, shapes=len(shapes), normalized_structures=len(structures),
                tile_geometries=len(geometries), palettes=len(palettes), count_sets=len(count_sets),
                tight_canvases=tight, top_edge_heads=at_top,
                empty_tiles=empty_tiles, full_tiles=full_tiles,
                independent_solver=False)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
