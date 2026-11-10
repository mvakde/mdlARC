"""Telescoping docks: geometric key order is the coordinate system for sorting.

Only the gap between a piece's two rigid caps changes. Connector keys, cap
geometry, stem width, and payload hue belong to the destination piece. All
inference is input-only; no parent code or generation metadata is imported.
No files, execution traces, or unbounded caches are written by this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _cells(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= v <= 30 for v in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(bytes([len(puzzle[split])]))
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _roles(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > grid.size // 2, "background must dominate"
    cc = _components(_cells(grid != bg))
    assert 2 <= len(cc) <= 6
    palettes = [{int(grid[q]) for q in comp} for comp in cc]
    common = set.intersection(*palettes)
    assert len(common) == 1, "the common connector hue must be unique"
    marker = common.pop()
    roots = [i for i, palette in enumerate(palettes) if len(palette) == 3]
    assert len(roots) == 1 and all(len(p) in (2, 3) for p in palettes)
    root = cc[roots[0]]
    singletons = [c for c in palettes[roots[0]] - {marker}
                  if sum(grid[q] == c for q in root) == 1]
    assert len(singletons) == 1
    seal = singletons[0]
    assert np.count_nonzero(grid == seal) == 1
    return bg, marker, seal


def _run(xs):
    xs = sorted(xs)
    assert xs and xs == list(range(xs[0], xs[-1] + 1))
    return xs[0], len(xs)


def _decode(canonical, roles):
    """Read individual payloads even when marker cells join the output pieces."""
    bg, marker, seal = roles
    payload_mask = ~np.isin(canonical, roles)
    pieces = []
    for comp in _components(_cells(payload_mask)):
        colors = {int(canonical[q]) for q in comp}
        assert len(colors) == 1
        color = colors.pop()
        top = min(r for r, c in comp)
        bottom = max(r for r, c in comp)
        length = bottom - top - 1
        assert 1 <= length <= 7
        upper_left, upper_width = _run(c for r, c in comp if r == top)
        lower_left, lower_width = _run(c for r, c in comp if r == bottom)
        assert 4 <= upper_width <= 9 and 4 <= lower_width <= 9
        stem_left, width = _run(c for r, c in comp if r == top + 1)
        assert 1 <= width <= 3
        for left, rail_width in ((upper_left, upper_width), (lower_left, lower_width)):
            assert 1 <= stem_left - left <= 3
            assert 1 <= left + rail_width - stem_left - width <= 3
        expected = {(top, c) for c in range(upper_left, upper_left + upper_width)}
        expected |= {(bottom, c) for c in range(lower_left, lower_left + lower_width)}
        expected |= {(r, c) for r in range(top + 1, bottom)
                     for c in range(stem_left, stem_left + width)}
        assert expected == comp, "caps and stem must explain the entire payload"

        def pins(row, left, rail_width):
            if not 0 <= row < canonical.shape[0]:
                return ()
            result = tuple(c for c in range(left, left + rail_width)
                           if canonical[row, c] == marker)
            assert len(result) in (0, 2)
            if result:
                assert 2 <= result[1] - result[0] <= 6
            return result

        upper = pins(top - 1, upper_left, upper_width)
        lower = pins(bottom + 1, lower_left, lower_width)
        seal_cells = [(r, c) for r, c in _cells(canonical == seal)
                      if r == top - 1 and upper_left <= c < upper_left + upper_width]
        assert len(seal_cells) <= 1
        is_root = bool(seal_cells)
        assert bool(upper) != is_root
        pieces.append(dict(top=top, x=stem_left, length=length, width=width,
                           upper_left=upper_left - stem_left, upper_width=upper_width,
                           lower_left=lower_left - stem_left, lower_width=lower_width,
                           upper=tuple(c - stem_left for c in upper),
                           lower=tuple(c - stem_left for c in lower),
                           color=color, seal=(seal_cells[0][1] - stem_left if is_root else None)))
    assert 2 <= len(pieces) <= 6
    root_ids = [i for i, p in enumerate(pieces) if p["seal"] is not None]
    assert len(root_ids) == 1
    incoming = {}
    for i, p in enumerate(pieces):
        if p["upper"]:
            gap = p["upper"][1] - p["upper"][0]
            assert gap not in incoming, "each key must identify one successor"
            incoming[gap] = i
    chain, used = [], set()
    i = root_ids[0]
    while i not in used:
        used.add(i)
        p = pieces[i]
        chain.append(p)
        if not p["lower"]:
            break
        gap = p["lower"][1] - p["lower"][0]
        assert gap in incoming
        i = incoming[gap]
    assert len(chain) == len(pieces) and not chain[-1]["lower"]
    assert len(incoming) == len(chain) - 1
    # Every cell must belong to a decoded cap, stem, key, or the start seal.
    rebuilt = np.full(canonical.shape, bg, dtype=np.int8)
    for p in chain:
        mapping = _piece_cells(p, p["length"], roles)
        for (r, c), color in mapping.items():
            assert 0 <= r < rebuilt.shape[0] and 0 <= c < rebuilt.shape[1]
            assert rebuilt[r, c] in (bg, marker) and (rebuilt[r, c] == bg or color == marker)
            rebuilt[r, c] = color
    assert np.array_equal(rebuilt, canonical), "unexplained cells"
    return chain


def _analyze(input_grid):
    a = _grid(input_grid)
    roles = _roles(a)
    candidates = []
    for k in range(4):
        canonical = np.rot90(a, k)
        try:
            chain = _decode(canonical, roles)
        except (AssertionError, IndexError):
            continue
        candidates.append((canonical, roles, chain, k))
    assert len(candidates) == 1, "the seal must determine a unique growth direction"
    return candidates[0]


def _piece_cells(p, length, roles):
    _, marker, seal = roles
    row, x, color = p["top"], p["x"], p["color"]
    result = {}
    for offset, left, width in ((0, p["upper_left"], p["upper_width"]),
                                (length + 1, p["lower_left"], p["lower_width"])):
        result.update({(row + offset, x + c): color for c in range(left, left + width)})
    result.update({(row + r, x + c): color for r in range(1, length + 1)
                   for c in range(p["width"])})
    result.update({(row - 1, x + c): marker for c in p["upper"]})
    result.update({(row + length + 2, x + c): marker for c in p["lower"]})
    if p["seal"] is not None:
        result[row - 1, x + p["seal"]] = seal
    return result


def _layout(chain, lengths):
    placed = []
    for i, (p, length) in enumerate(zip(chain, lengths)):
        q = dict(p, length=int(length))
        if i:
            previous = placed[-1]
            q["top"] = previous["top"] + previous["length"] + 3
            q["x"] = previous["x"] + previous["lower"][0] - p["upper"][0]
        placed.append(q)
    return placed


def _paint(shape, chain, roles):
    result = np.full(shape, roles[0], dtype=np.int8)
    for p in chain:
        for (r, c), color in _piece_cells(p, p["length"], roles).items():
            assert 0 <= r < shape[0] and 0 <= c < shape[1], "assembly leaves canvas"
            assert result[r, c] == roles[0] or (result[r, c] == roles[1] == color)
            result[r, c] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    canonical, roles, chain, k = _analyze(input_grid)
    lengths = sorted(p["length"] for p in chain)
    out = _paint(canonical.shape, _layout(chain, lengths), roles)
    return np.rot90(out, -k).copy()


def _template(p):
    return tuple(p[key] for key in ("width", "upper_left", "upper_width", "lower_left",
                                    "lower_width", "upper", "lower", "color", "seal"))


def _check_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    canonical, roles, original, k = _analyze(inp)
    result = _decode(np.rot90(out, k), roles)
    assert [_template(p) for p in original] == [_template(p) for p in result]
    assert (original[0]["top"], original[0]["x"]) == (result[0]["top"], result[0]["x"])
    assert Counter(p["length"] for p in original) == Counter(p["length"] for p in result)
    lengths = [p["length"] for p in result]
    assert all(a <= b for a, b in zip(lengths, lengths[1:])), "lengths must increase along keys"
    owners = Counter()
    for i, p in enumerate(result):
        for (r, c), color in _piece_cells(p, p["length"], roles).items():
            if color == roles[1]:
                owners[r, c] += 1
        if i:
            prev = result[i - 1]
            assert p["top"] - 1 == prev["top"] + prev["length"] + 2
            assert tuple(p["x"] + c for c in p["upper"]) == tuple(prev["x"] + c for c in prev["lower"])
            assert (p["top"], p["x"]) != (original[i]["top"], original[i]["x"])
    assert set(owners.values()) == {2}
    assert np.count_nonzero(inp == roles[1]) == 2 * np.count_nonzero(out == roles[1])
    # Relational conditions above determine the output, without calling solve,
    # sorting the original list, or using the constructor's placement routine.
    return canonical, roles, original, result, k


def _features(pair):
    canonical, roles, chain, result, k = _check_pair(pair)
    lengths = [p["length"] for p in chain]
    widths = [p["width"] for p in chain]
    areas = [p["length"] * p["width"] for p in chain]
    full_areas = [a + p["upper_width"] + p["lower_width"] for a, p in zip(areas, chain)]
    return dict(count=len(chain), direction=k, shape=list(np.asarray(pair["input"]).shape),
                lengths=lengths, widths=widths,
                key_gaps=[p["lower"][1] - p["lower"][0] for p in chain[:-1]],
                cap_shapes=[(p["upper_left"], p["upper_width"], p["lower_left"], p["lower_width"]) for p in chain],
                changed_lengths=lengths != [p["length"] for p in result],
                reversal_fails=lengths[::-1] != [p["length"] for p in result],
                area_inversion=any(lengths[i] < lengths[j] and areas[i] > areas[j]
                                   for i in range(len(chain)) for j in range(len(chain))),
                full_area_inversion=any(lengths[i] < lengths[j] and full_areas[i] > full_areas[j]
                                        for i in range(len(chain)) for j in range(len(chain))),
                repeated_colors=len({p["color"] for p in chain}) < len(chain),
                ties=len(set(lengths)) < len(chain),
                lateral_shifts=[result[i]["x"] - result[i - 1]["x"] for i in range(1, len(chain))],
                spatial_order_differs=chain != sorted(chain, key=lambda p: (p["top"], p["x"])),
                roles=list(roles))


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def _rail(rng, width, gap):
    options = [(left, width + left + right) for left in range(1, 4) for right in range(1, 4)
               if width + left + right >= max(4, gap + 1)]
    left_pad, cap_width = options[int(rng.integers(len(options)))]
    left = -left_pad
    start = int(rng.integers(left, left + cap_width - gap)) if gap else 0
    return left, cap_width, (() if not gap else (start, start + gap))


def _scatter(rng, shape, chain, roles):
    root_map = _piece_cells(chain[0], chain[0]["length"], roles)
    if any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in root_map):
        return None
    maps = [_piece_cells(p, p["length"], roles) for p in chain]
    # Only one candidate canvas is retained; bounded retries write no logs.
    for restart in range(12):
        occupied = set(root_map)
        forbidden = {(r + dr, c + dc) for r, c in occupied for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        placed = [chain[0]]
        order = list(range(1, len(chain)))
        rng.shuffle(order)
        order.sort(key=lambda i: -len(maps[i]))
        for i in order:
            mapping = maps[i]
            minr, maxr = min(r for r, c in mapping), max(r for r, c in mapping)
            minc, maxc = min(c for r, c in mapping), max(c for r, c in mapping)
            for attempt in range(160):
                dr = int(rng.integers(shape[0] - maxr + minr)) - minr
                dc = int(rng.integers(shape[1] - maxc + minc)) - minc
                cells = {(r + dr, c + dc) for r, c in mapping}
                if cells & forbidden:
                    continue
                p = dict(chain[i], top=chain[i]["top"] + dr, x=chain[i]["x"] + dc)
                placed.append(p)
                forbidden |= {(r + a, c + b) for r, c in cells for a in (-1, 0, 1) for b in (-1, 0, 1)}
                break
            else:
                break
        if len(placed) == len(chain):
            return _paint(shape, placed, roles)
    return None


def _make_pair(rng, config, n, witness, test, direction):
    for attempt in range(96):
        cap = 4 if config["shift"] == "stem_extension" and not test else 7
        if config["shift"] == "chain_extension" and test:
            cap = 4
        if witness == 1:
            lengths = [int(v) for v in rng.choice(np.arange(1, cap + 1), n, replace=False)]
        elif witness == 2:
            a, b, c = sorted(int(v) for v in rng.choice(np.arange(1, cap + 1), 3, replace=False))
            lengths = [a, b, b, c]
        else:
            lengths = [int(v) for v in rng.integers(1, cap + 1, size=n)]
            # Fit long chains constructively instead of repeatedly drawing a
            # six-piece sum that exceeds its 12-cell stem budget.
            while sum(lengths) > 30 - 3 * n:
                choices = [i for i, v in enumerate(lengths) if v > 1]
                lengths[int(rng.choice(choices))] -= 1
        if sum(lengths) + 3 * n > 30:
            continue
        if test and config["shift"] == "stem_extension":
            if max(lengths) < 6:
                continue
        if len(set(lengths)) < 2:
            continue
        ordered = sorted(lengths)
        rng.shuffle(lengths)
        if witness and (lengths == ordered or lengths[::-1] == ordered or lengths[0] == min(lengths)):
            # A cyclic shift always supplies a non-reversal sort for these
            # 3-distinct-value witnesses and makes the marked piece telescope.
            lengths = ordered[1:] + ordered[:1]
        if lengths == ordered:
            lengths[0], lengths[-1] = lengths[-1], lengths[0]
        widths = [int(v) for v in rng.integers(1, 4, size=n)]
        if witness == 1:
            pairs = [(a, b) for a in range(n) for b in range(n)
                     if lengths[a] < lengths[b] < 3 * lengths[a]]
            if not pairs:
                continue
            short, long = pairs[int(rng.integers(len(pairs)))]
            widths[short], widths[long] = 3, 1
        roles = config["roles"] if config["fixed_roles"] else tuple(int(v) for v in rng.choice(10, 3, replace=False))
        pool = [c for c in range(10) if c not in roles]
        palette_size = int(rng.integers(2, min(n, 5) + 1))
        if witness == 2:
            palette_size = 2
        palette = [int(c) for c in rng.choice(pool, palette_size, replace=False)]
        colors = palette + [int(v) for v in rng.choice(palette, n - palette_size)]
        rng.shuffle(colors)
        gaps = [int(v) for v in rng.choice(np.arange(2, 7), n - 1, replace=False)]
        chain = []
        for i in range(n):
            ul, uw, upper = _rail(rng, widths[i], gaps[i - 1] if i else 0)
            ll, lw, lower = _rail(rng, widths[i], gaps[i] if i < n - 1 else 0)
            seal = int(rng.integers(ul, ul + uw)) if i == 0 else None
            chain.append(dict(top=0, x=0, length=lengths[i], width=widths[i],
                              upper_left=ul, upper_width=uw, lower_left=ll, lower_width=lw,
                              upper=upper, lower=lower, color=colors[i], seal=seal))
        assembled = _layout(chain, sorted(lengths))
        allcells = set().union(*(_piece_cells(p, p["length"], roles) for p in assembled))
        minr, maxr = min(r for r, c in allcells), max(r for r, c in allcells)
        minc, maxc = min(c for r, c in allcells), max(c for r, c in allcells)
        bh, bw = maxr - minr + 1, maxc - minc + 1
        if max(bh, bw) > 30:
            continue
        h = max(bh, int(rng.integers(18, 31)))
        w = max(bw, int(rng.integers(18, 31)))
        for enlarged in (False, True):
            if enlarged:
                h = w = 30
            dy = int(rng.integers(h - bh + 1)) - minr
            dx = int(rng.integers(w - bw + 1)) - minc
            output_chain = [dict(p, top=p["top"] + dy, x=p["x"] + dx) for p in assembled]
            # Only the start's upper cap is anchored. Its original lower cap
            # may be farther away, so also check the complete scattered root.
            source_chain = [dict(p, top=output_chain[i]["top"], x=output_chain[i]["x"]) for i, p in enumerate(chain)]
            inp = _scatter(rng, (h, w), source_chain, roles)
            if inp is None:
                continue
            out = _paint((h, w), output_chain, roles)
            pair = {"input": np.rot90(inp, -direction).copy(),
                    "output": np.rot90(out, -direction).copy()}
            try:
                f = _features(pair)
            except AssertionError:
                continue
            if not f["spatial_order_differs"]:
                continue
            if witness and (not f["changed_lengths"] or not f["reversal_fails"]):
                continue
            if witness and not any(f["lateral_shifts"]):
                continue
            if witness == 1 and not f["full_area_inversion"]:
                continue
            return pair
    raise RuntimeError("bounded telescoping-piece construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    config = dict(roles=tuple(int(v) for v in rng.choice(10, 3, replace=False)),
                  fixed_roles=bool(rng.random() < 0.6),
                  shift=str(rng.choice(["chain_extension", "stem_extension"])))
    d = int(rng.integers(4))
    axes = [d, (d + int(rng.choice([1, 3]))) % 4]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = i + 1 if split == "train" and i < 2 else 0
            if witness:
                n, direction = (3 if witness == 1 else 4), axes[i]
            else:
                n = int(rng.integers(5, 7)) if split == "test" and config["shift"] == "chain_extension" else int(rng.integers(2, 5))
                direction = int(rng.choice(axes))
            for attempt in range(16):
                pair = _make_pair(rng, config, n, witness, split == "test", direction)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            puzzle[split].append(pair)
            seen.add(key)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen
            seen.add(key)
            features[split].append(_features(pair))
    train, test = features["train"], features["test"]
    assert sum(f["changed_lengths"] and f["reversal_fails"] for f in train) >= 2
    assert any(f["area_inversion"] for f in train)
    assert any(f["full_area_inversion"] for f in train)
    assert any(f["ties"] and f["repeated_colors"] for f in train)
    assert len({f["count"] for f in train}) >= 2
    assert all(f["spatial_order_differs"] for f in train + test)
    assert {f["direction"] for f in test} <= {f["direction"] for f in train}
    train_max_count = max(f["count"] for f in train)
    train_max_length = max(max(f["lengths"]) for f in train)
    assert all(f["count"] > train_max_count or max(f["lengths"]) > train_max_length for f in test)


def self_test():
    """Streaming 42-seed checks, descriptive metrics, and semantic mutations.

    Peak retained grids: one 14-pair puzzle (<26 KB uint8), plus a repeat and
    one mutation. No samples/logs are persisted; work is bounded by fixed loops.
    """
    signatures, shapes, caps, roles, directions, gaps, counts, widths, lengths = (set() for _ in range(9))
    tallies = Counter()
    rows = []
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed == 0:
            first = puzzle
        digest = _digest(puzzle)
        assert digest == _digest(generate_puzzle(seed))
        signatures.add(digest)
        local_counts, local_lengths = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                f = _features(pair)
                shapes.add(tuple(f["shape"]))
                caps.update(f["cap_shapes"])
                roles.add(tuple(f["roles"]))
                directions.add(f["direction"])
                gaps.update(f["key_gaps"])
                counts.add(f["count"])
                widths.update(f["widths"])
                lengths.update(f["lengths"])
                local_counts.add(f["count"])
                local_lengths.update(f["lengths"])
                tallies["pairs"] += 1
                tallies["nonzero_lateral_steps"] += sum(v != 0 for v in f["lateral_shifts"])
                for name in ("area_inversion", "full_area_inversion", "ties", "repeated_colors", "spatial_order_differs"):
                    tallies[name] += int(f[name])
        rows.append(dict(seed=seed, sha256=digest, counts=sorted(local_counts), lengths=sorted(local_lengths)))
    assert len(signatures) == 42 and len(shapes) >= 30 and len(caps) >= 40
    assert counts == set(range(2, 7)) and widths == {1, 2, 3}
    assert lengths == set(range(1, 8)) and gaps == set(range(2, 7))
    assert directions == set(range(4)) and len(roles) >= 20
    assert tallies["nonzero_lateral_steps"] > 100
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(9100 + nt + nq, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    pair = next(p for p in first["train"] if _features(p)["changed_lengths"])
    canonical, role, chain, k = _analyze(pair["input"])
    omitted_a = np.rot90(_paint(canonical.shape, _layout(chain, [p["length"] for p in chain]), role), -k).copy()
    mutations = [omitted_a]
    # Omit B's exact geometric alignment while retaining A's correctly sorted
    # lengths and every piece's own cap/key geometry. One successor drifts by
    # a cell; no payload is distorted and the start is still fixed.
    misplaced = _layout(chain, sorted(p["length"] for p in chain))
    moved_cells = _piece_cells(misplaced[1], misplaced[1]["length"], role)
    dx = 1 if max(c for r, c in moved_cells) + 1 < canonical.shape[1] else -1
    misplaced[1]["x"] += dx
    mutations.append(np.rot90(_paint(canonical.shape, misplaced, role), -k).copy())
    bad = pair["output"].copy()
    pin = tuple(np.argwhere(bad == role[1])[0])
    bad[pin] = role[0]
    mutations.append(bad)
    shifted = np.full_like(pair["output"], role[0])
    shifted[1:] = pair["output"][:-1]
    mutations.append(shifted)
    for bad in mutations:
        assert not np.array_equal(bad, pair["output"])
        try:
            _check_pair({"input": pair["input"], "output": bad})
        except AssertionError:
            pass
        else:
            raise AssertionError("semantic mutation accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                cap_templates=len(caps), role_palettes=len(roles), directions=sorted(directions),
                piece_counts=sorted(counts), stem_widths=sorted(widths), stem_lengths=sorted(lengths),
                key_gaps=sorted(gaps), tallies=dict(tallies), seed_records=rows,
                rejected_mutations=["omit_length_sort", "misalign_successor", "erase_connector", "move_start"],
                limitations="Authored invariant and solver checks; no independent semantic solver or blind prediction.")
