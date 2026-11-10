"""Round-robin dealing of counted 2x2 tokens onto a growing-gap lane.

Composition of parent A (72207abc: cyclic colour continuation with gaps that
grow by one each step) and parent B (ff28f65a: count separate solid 2x2 squares
and write that count in unary into an ordered list of output sites).

Input: one edge line (the *lane*) is cut off from the rest of the grid (the
*field*) by a full divider line of one colour.  The lane holds a seed: one
cycle of k distinct colours at lane offsets 0, 1, 3, 6, ... (gaps 0, 1, 2, ...)
measured from the lane end where the seed sits.  The field holds solid 2x2
squares ("tokens") in seed colours, possibly diagonally touching, and possibly
other non-square debris.

Output: every 2x2 token is dealt onto the lane, one marker per token.  Turns
continue the seed's colour cycle (starting again at the first seed colour); a
colour whose tokens are used up is skipped.  Markers go at the continuing
triangular offsets (gap grows by one each placement).  Dealt tokens are erased
from the field; debris, divider and seed stay.
"""
from __future__ import annotations

import numpy as np

# Orientation = (edge holding the lane, whether the seed sits at the far end).
ORIENTATIONS = (
    ("top", False), ("top", True), ("bottom", False), ("bottom", True),
    ("left", False), ("left", True), ("right", False), ("right", True),
)
MAX_SIDE = 30

# Debris shapes: none contains a solid 2x2 block.
_DEBRIS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def tri(i: int) -> int:
    return i * (i + 1) // 2


# ----------------------------------------------------------------------------
# Shared rule helpers (pure functions of visible content)
# ----------------------------------------------------------------------------

def deal(seed, counts):
    """Round-robin over the seed cycle, skipping colours with no tokens left."""
    remaining = dict(counts)
    out, turn = [], 0
    total = sum(remaining.get(c, 0) for c in seed)
    while total > 0:
        colour = seed[turn % len(seed)]
        if remaining.get(colour, 0) > 0:
            out.append(colour)
            remaining[colour] -= 1
            total -= 1
        turn += 1
    return out


def _to_canonical(grid, orient):
    edge, reverse = orient
    g = np.asarray(grid)
    if edge == "bottom":
        g = g[::-1, :]
    elif edge == "left":
        g = g.T
    elif edge == "right":
        g = g[:, ::-1].T
    if reverse:
        g = g[:, ::-1]
    return np.ascontiguousarray(g)


def _from_canonical(grid, orient):
    edge, reverse = orient
    g = np.asarray(grid)
    if reverse:
        g = g[:, ::-1]
    if edge == "bottom":
        g = g[::-1, :]
    elif edge == "left":
        g = g.T
    elif edge == "right":
        g = g.T[:, ::-1]
    return np.ascontiguousarray(g)


def _components(field, bg):
    """4-connected single-colour components of the field: list of (colour, cells)."""
    h, w = field.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or field[r, c] == bg:
                continue
            colour = int(field[r, c])
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and field[ny, nx] == colour:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((colour, cells))
    return comps


def _is_square(cells):
    if len(cells) != 4:
        return False
    rs = {r for r, _ in cells}
    cs = {c for _, c in cells}
    return len(rs) == 2 and len(cs) == 2 and max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1


def _find_orientation(g):
    """Locate the divider line (index 1 or -2, full, uniform) and the lane edge."""
    h, w = g.shape
    found = []
    for edge, line in (("top", g[1, :] if h >= 3 else None), ("bottom", g[h - 2, :] if h >= 3 else None),
                       ("left", g[:, 1] if w >= 3 else None), ("right", g[:, w - 2] if w >= 3 else None)):
        if line is None:
            continue
        if np.all(line == line[0]):
            found.append(edge)
    return found


def parse(input_grid):
    """Return a dict describing the canonical view of a valid input."""
    g = np.asarray(input_grid, dtype=int)
    edges = _find_orientation(g)
    candidates = []
    for edge in edges:
        canon = _to_canonical(g, (edge, False))
        divider = int(canon[1, 0])
        rest = np.concatenate([canon[0], canon[2:].ravel()])
        vals, cnts = np.unique(rest, return_counts=True)
        bg = int(vals[np.argmax(cnts)])
        if divider == bg:
            continue
        lane = canon[0]
        occupied = [i for i in range(len(lane)) if lane[i] != bg]
        if not occupied:
            continue
        reverse = lane[0] == bg
        candidates.append((edge, reverse, bg, divider))
    if len(candidates) != 1:
        raise ValueError("Could not identify a unique lane/divider layout")
    edge, reverse, bg, divider = candidates[0]
    orient = (edge, bool(reverse))
    canon = _to_canonical(g, orient)
    lane = canon[0]
    occupied = [i for i in range(len(lane)) if lane[i] != bg]
    seed = [int(lane[i]) for i in occupied]
    field = canon[2:]
    comps = _components(field, bg)
    squares = [(col, cells) for col, cells in comps if _is_square(cells)]
    debris = [(col, cells) for col, cells in comps if not _is_square(cells)]
    return {"orient": orient, "canon": canon, "bg": bg, "divider": divider, "lane_positions": occupied,
            "seed": seed, "squares": squares, "debris": debris}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = parse(input_grid)
    canon = info["canon"].copy()
    bg, seed = info["bg"], info["seed"]
    counts = {}
    for colour, _ in info["squares"]:
        counts[colour] = counts.get(colour, 0) + 1
    sequence = deal(seed, counts)
    width = canon.shape[1]
    for j, colour in enumerate(sequence):
        pos = tri(len(seed) + j)
        if pos >= width:
            break
        canon[0, pos] = colour
    for _, cells in info["squares"]:
        for r, c in cells:
            canon[2 + r, c] = bg
    return _from_canonical(canon, info["orient"])


# ----------------------------------------------------------------------------
# Shortcut predictions used to prove the demonstrations are discriminating
# ----------------------------------------------------------------------------

def _render_lane(canon, seed_len, sequence, bg, erase_cells):
    out = canon.copy()
    for j, colour in enumerate(sequence):
        pos = tri(seed_len + j)
        if pos >= out.shape[1]:
            break
        out[0, pos] = colour
    for r, c in erase_cells:
        out[2 + r, c] = bg
    return out


def shortcut_outputs(input_grid):
    """Outputs of plausible simpler rules, in the input's own orientation."""
    info = parse(input_grid)
    canon, bg, seed = info["canon"], info["bg"], info["seed"]
    k = len(seed)
    counts = {}
    for colour, _ in info["squares"]:
        counts[colour] = counts.get(colour, 0) + 1
    total = sum(counts.values())
    sq_cells = [cell for _, cells in info["squares"] for cell in cells]
    alts = {}
    # Stop at the first colour that has no token left (instead of skipping it).
    rem, stop_seq, turn = dict(counts), [], 0
    while True:
        colour = seed[turn % k]
        if rem.get(colour, 0) <= 0:
            break
        stop_seq.append(colour)
        rem[colour] -= 1
        turn += 1
    alts["stop_at_exhausted"] = stop_seq
    # Ignore per-colour counts: place `total` markers cycling all seed colours.
    alts["plain_cycle_total"] = [seed[i % k] for i in range(total)]
    # Colour blocks in seed order instead of interleaving.
    alts["blocks_in_seed_order"] = [c for c in seed for _ in range(counts.get(c, 0))]
    # Parent A alone: continue the cycle to the lane boundary.
    fill, j = [], 0
    while tri(k + j) < canon.shape[1]:
        fill.append(seed[j % k])
        j += 1
    alts["fill_to_boundary"] = fill
    results = {name: _from_canonical(_render_lane(canon, k, seq, bg, sq_cells), info["orient"])
               for name, seq in alts.items()}
    # Count every object (debris included) by colour.
    all_counts = dict(counts)
    for colour, _ in info["debris"]:
        if colour in seed:
            all_counts[colour] = all_counts.get(colour, 0) + 1
    deb_cells = [cell for col, cells in info["debris"] if col in seed for cell in cells]
    results["count_all_objects"] = _from_canonical(
        _render_lane(canon, k, deal(seed, all_counts), bg, sq_cells + deb_cells), info["orient"])
    # Count with 8-connectivity: diagonally touching same-colour squares merge.
    results["eight_connected_count"] = _eight_connected_prediction(info)
    return results


def _eight_connected_prediction(info):
    canon, bg, seed = info["canon"], info["bg"], info["seed"]
    field = canon[2:]
    h, w = field.shape
    seen = np.zeros((h, w), dtype=bool)
    square_cells = {cell for _, cells in info["squares"] for cell in cells}
    counts, erase = {}, []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or field[r, c] == bg:
                continue
            colour = int(field[r, c])
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and field[ny, nx] == colour:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if all(cell in square_cells for cell in cells):
                # a diagonal chain of same-colour squares merges into one object
                counts[colour] = counts.get(colour, 0) + 1
                erase.extend(cells)
    return _from_canonical(_render_lane(canon, len(seed), deal(seed, counts), bg, erase), info["orient"])


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _random_counts(rng, k, total, need_discriminating):
    for _ in range(200):
        counts = [0] * k
        for _ in range(total):
            counts[int(rng.integers(k))] += 1
        seed = list(range(k))
        seq = deal(seed, dict(enumerate(counts)))
        plain = [i % k for i in range(total)]
        if need_discriminating == "full":
            blocks = [c for c in seed for _ in range(counts[c])]
            stop, rem, t = [], list(counts), 0
            while rem[t % k] > 0:
                stop.append(t % k)
                rem[t % k] -= 1
                t += 1
            if seq == plain or seq == blocks or seq == stop:
                continue
        elif need_discriminating == "plain":
            if seq == plain:
                continue
        elif seq == plain and rng.random() < 0.7:
            continue
        return counts
    return None


def _place_field(rng, depth, width, square_colours, debris_specs, diag_bias):
    field = np.full((depth, width), -1, dtype=int)
    placed = []  # (r, c, colour)
    for colour in square_colours:
        options = []
        if placed and rng.random() < diag_bias:
            same = [(r, c) for r, c, col in placed if col == colour] or [(r, c) for r, c, _ in placed]
            for r0, c0 in same:
                for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
                    options.append((r0 + dr, c0 + dc))
            rng.shuffle(options)
        free = [(r, c) for r in range(depth - 1) for c in range(width - 1)]
        rng.shuffle(free)
        ok = False
        for r, c in options + free:
            if not (0 <= r <= depth - 2 and 0 <= c <= width - 2):
                continue
            block = False
            for y in range(r - 1, r + 3):
                for x in range(c - 1, c + 3):
                    corner = y in (r - 1, r + 2) and x in (c - 1, c + 2)
                    if corner:
                        continue
                    if 0 <= y < depth and 0 <= x < width and field[y, x] != -1:
                        block = True
            if block:
                continue
            field[r:r + 2, c:c + 2] = colour
            placed.append((r, c, colour))
            ok = True
            break
        if not ok:
            return None
    for shape, colour in debris_specs:
        hh = max(p[0] for p in shape) + 1
        ww = max(p[1] for p in shape) + 1
        spots = [(r, c) for r in range(depth - hh + 1) for c in range(width - ww + 1)]
        rng.shuffle(spots)
        ok = False
        for r, c in spots:
            cells = [(r + dy, c + dx) for dy, dx in shape]
            clash = False
            for y, x in cells:
                for yy in range(y - 1, y + 2):
                    for xx in range(x - 1, x + 2):
                        if 0 <= yy < depth and 0 <= xx < width and field[yy, xx] != -1:
                            clash = True
            if clash:
                continue
            for y, x in cells:
                field[y, x] = colour
            ok = True
            break
        if not ok:
            return None
    return field


def _make_pair(rng, spec):
    """spec keys: k, total, orient, bg, divider, debris, discriminating, min_len, max_len."""
    k, total = spec["k"], spec["total"]
    bg, divider = spec["bg"], spec["divider"]
    palette = [c for c in range(10) if c not in (bg, divider)]
    seed = [int(x) for x in rng.choice(palette, size=k, replace=False)]
    counts = _random_counts(rng, k, total, spec["discriminating"])
    if counts is None:
        return None
    markers = k + total
    min_len = max(tri(markers) + 1, spec.get("min_len", 0), 8)
    max_len = min(MAX_SIDE, spec.get("max_len", MAX_SIDE))
    if min_len > max_len:
        return None
    length = int(rng.integers(min_len, max_len + 1))
    square_colours = [seed[i] for i in range(k) for _ in range(counts[i])]
    rng.shuffle(square_colours)
    debris_specs = []
    if spec["debris"]:
        n_deb = int(rng.integers(1, 4))
        for d in range(n_deb):
            shape = _DEBRIS[int(rng.integers(len(_DEBRIS)))]
            if d == 0:
                colour = seed[int(rng.integers(k))]
            else:
                pool = seed + [c for c in palette if c not in seed]
                colour = int(pool[int(rng.integers(len(pool)))])
            debris_specs.append((shape, colour))
    need_area = 9 * total + sum(len(s) * 5 for s, _ in debris_specs)
    min_depth = max(3, int(np.ceil(need_area / length)) + 1)
    max_depth = MAX_SIDE - 2
    if min_depth > max_depth:
        return None
    depth = int(rng.integers(min_depth, min(max_depth, min_depth + 7) + 1))
    field = _place_field(rng, depth, length, square_colours, debris_specs, spec.get("diag", 0.25))
    if field is None:
        return None
    field[field == -1] = bg
    canon = np.full((depth + 2, length), bg, dtype=int)
    canon[1, :] = divider
    canon[2:, :] = field
    for i, colour in enumerate(seed):
        canon[0, tri(i)] = colour
    grid = _from_canonical(canon, spec["orient"])
    out = solve(grid)
    return {"input": grid, "output": out, "k": k, "total": total, "counts": counts, "seed": seed,
            "orient": spec["orient"], "debris": bool(debris_specs)}


def _diag_touch_same_colour(info):
    sq = info["squares"]
    for i in range(len(sq)):
        for j in range(i + 1, len(sq)):
            if sq[i][0] != sq[j][0]:
                continue
            for r, c in sq[i][1]:
                for y, x in sq[j][1]:
                    if abs(r - y) == 1 and abs(c - x) == 1:
                        return True
    return False


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not isinstance(num_train, (int, np.integer)) or not isinstance(num_test, (int, np.integer)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal train/test counts")
    rng = np.random.default_rng(seed)
    for _ in range(400):
        puzzle = _attempt(rng, int(num_train), int(num_test))
        if puzzle is not None:
            return puzzle
    raise RuntimeError("generation failed")


def _attempt(rng, num_train, num_test):
    # Puzzle-level conventions
    pool_size = int(min(num_train, rng.choice([1, 1, 2, 2, 3])))
    first = ORIENTATIONS[0] if rng.random() < 0.3 else ORIENTATIONS[int(rng.integers(8))]
    pool = [first]
    while len(pool) < pool_size:
        o = ORIENTATIONS[int(rng.integers(8))]
        if o not in pool:
            pool.append(o)
    bg_mode = rng.choice(["zero", "fixed", "vary"], p=[0.45, 0.3, 0.25])
    fixed_bg = 0 if bg_mode == "zero" else int(rng.integers(10))
    fixed_div = int(rng.choice([c for c in range(10) if c != fixed_bg]))
    vary_div = rng.random() < 0.3
    debris_on = rng.random() < 0.5
    regime = str(rng.choice(["longer", "absent", "more_colours", "mixed"]))

    def colours():
        bg = fixed_bg if bg_mode != "vary" else int(rng.integers(10))
        div = fixed_div
        if vary_div or div == bg:
            div = int(rng.choice([c for c in range(10) if c != bg]))
        return bg, div

    # Train orientation list covering the pool
    orients = list(pool) + [pool[int(rng.integers(len(pool)))] for _ in range(num_train - len(pool))]
    rng.shuffle(orients)
    disc_index = int(rng.integers(num_train))

    train_k_cap = 3 if regime == "more_colours" else 4
    train_m_cap = 6 if regime == "longer" else 7
    train = []
    for i in range(num_train):
        for _ in range(60):
            if i == disc_index:
                k = int(rng.integers(3, train_k_cap + 1))
            else:
                k = int(rng.choice([2, 3, 3, 4][: (4 if train_k_cap == 4 else 3)]))
            max_total = train_m_cap - k
            if max_total < 2:
                continue
            total = int(rng.integers(2, max_total + 1))
            if i == disc_index:
                total = max(total, 3)
                if total > max_total:
                    continue
            bg, div = colours()
            spec = {"k": k, "total": total, "orient": orients[i], "bg": bg, "divider": div,
                    "debris": debris_on and (i == disc_index or rng.random() < 0.75),
                    "discriminating": "full" if i == disc_index else None, "diag": 0.35}
            pair = _make_pair(rng, spec)
            if pair is not None:
                train.append(pair)
                break
        else:
            return None
    if not any(p["k"] >= 3 for p in train):
        return None
    max_train_m = max(p["k"] + p["total"] for p in train)
    max_train_k = max(p["k"] for p in train)
    tests = []
    for _ in range(num_test):
        for _ in range(60):
            if regime == "longer":
                k = int(rng.choice([2, 3]))
                total = 7 - k
                if k + total <= max_train_m:
                    continue
            elif regime == "more_colours":
                k = 4
                total = int(rng.integers(2, 4))
                if k <= max_train_k:
                    continue
            else:
                k = int(rng.choice([3, 4])) if regime == "absent" else int(rng.choice([2, 3, 4]))
                total = int(rng.integers(2, 7 - k + 1))
            bg, div = colours()
            spec = {"k": k, "total": total, "orient": pool[int(rng.integers(len(pool)))], "bg": bg,
                    "divider": div, "debris": debris_on and rng.random() < 0.8,
                    "discriminating": "plain", "diag": 0.4}
            pair = _make_pair(rng, spec)
            if pair is None:
                continue
            if regime == "absent" and min(pair["counts"]) != 0:
                continue
            tests.append(pair)
            break
        else:
            return None
    keys = set()
    for p in train + tests:
        key = p["input"].tobytes() + bytes(p["input"].shape)
        if key in keys:
            return None
        keys.add(key)
    puzzle = {"train": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in train],
              "test": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in tests]}
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return None
    return puzzle


# ----------------------------------------------------------------------------
# Validation and self test
# ----------------------------------------------------------------------------

def _check_pair(pair):
    inp = np.asarray(pair["input"], dtype=int)
    out = np.asarray(pair["output"], dtype=int)
    assert inp.ndim == 2 and inp.shape == out.shape, "shape mismatch"
    assert all(1 <= s <= MAX_SIDE for s in inp.shape), "grid size"
    info = parse(inp)
    canon, bg, seed = info["canon"], info["bg"], info["seed"]
    k = len(seed)
    assert 2 <= k <= 4 and len(set(seed)) == k, "seed must be 2..4 distinct colours"
    assert info["lane_positions"] == [tri(i) for i in range(k)], "seed not on triangular offsets"
    assert info["divider"] not in seed and bg not in seed, "colour roles overlap"
    assert canon.shape[0] >= 5, "field too shallow"
    counts = {}
    for colour, _ in info["squares"]:
        assert colour in seed, "token colour absent from seed"
        counts[colour] = counts.get(colour, 0) + 1
    total = sum(counts.values())
    assert total >= 2, "need at least two tokens"
    assert tri(k + total) < canon.shape[1], "lane must leave spare room after the last marker"
    for colour, cells in info["debris"]:
        assert colour != info["divider"], "debris uses divider colour"
        arr = np.zeros(canon.shape[:2], dtype=bool)
        for r, c in cells:
            arr[r, c] = True
        assert not np.any(arr[:-1, :-1] & arr[1:, :-1] & arr[:-1, 1:] & arr[1:, 1:]), "debris contains 2x2"
    # Output check against the rule, written independently of solve's loop
    oc = _to_canonical(out, info["orient"])
    lane = oc[0]
    expected_positions = [tri(i) for i in range(k + total)]
    got_positions = [i for i in range(len(lane)) if lane[i] != bg]
    assert got_positions == expected_positions, "output lane positions"
    got = [int(lane[p]) for p in expected_positions]
    assert got[:k] == seed, "seed changed"
    for c in seed:
        assert got.count(c) == 1 + counts.get(c, 0), "token conservation"
    assert np.array_equal(oc[1], canon[1]), "divider changed"
    field_in, field_out = canon[2:], oc[2:]
    sq_mask = np.zeros(field_in.shape, dtype=bool)
    for _, cells in info["squares"]:
        for r, c in cells:
            sq_mask[r, c] = True
    assert np.all(field_out[sq_mask] == bg), "tokens not erased"
    assert np.array_equal(field_out[~sq_mask], field_in[~sq_mask]), "debris/background changed"
    assert np.array_equal(solve(inp), out), "solve mismatch"
    return info, counts, total


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            key = repr(np.asarray(pair["input"]).tolist())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info, counts, total = _check_pair(pair)
            infos[split].append((info, counts, total, pair))
    assert any(len(i["seed"]) >= 3 for i, _, _, _ in infos["train"]), "gap growth never visible in train"
    train_orients = {i["orient"] for i, _, _, _ in infos["train"]}
    for i, _, _, _ in infos["test"]:
        assert i["orient"] in train_orients, "test orientation unseen in training"

    def differs(pair, names):
        alts = shortcut_outputs(pair["input"])
        out = np.asarray(pair["output"])
        return {n for n in names if not np.array_equal(alts[n], out)}

    core = ["stop_at_exhausted", "plain_cycle_total", "blocks_in_seed_order", "fill_to_boundary"]
    beaten = set()
    for _, _, _, pair in infos["train"]:
        beaten |= differs(pair, core + ["count_all_objects"])
    for name in core:
        assert name in beaten, f"training never refutes shortcut {name}"
    beaten |= set().union(*[differs(pair, ["eight_connected_count"]) for _, _, _, pair in infos["train"]])
    for _, _, _, pair in infos["test"]:
        d = differs(pair, ["count_all_objects", "eight_connected_count"])
        assert d <= beaten, "test needs a debris/diagonal convention not demonstrated in training"
    for _, _, _, pair in infos["test"]:
        d = differs(pair, ["plain_cycle_total", "fill_to_boundary"])
        assert d == {"plain_cycle_total", "fill_to_boundary"}, "test solvable by a shortcut"
    return True


def self_test():
    stats = {"orient": set(), "k": set(), "bg": set(), "debris": 0, "diag": 0, "shapes": set(),
             "zero_colour": 0}
    for s in range(42):
        p = generate_puzzle(s)
        validate_generated_puzzle(p)
        assert p == generate_puzzle(s), "nondeterministic"
        for pair in p["train"] + p["test"]:
            inp = np.asarray(pair["input"])
            assert np.array_equal(solve(inp), np.asarray(pair["output"]))
            info = parse(inp)
            stats["orient"].add(info["orient"])
            stats["k"].add(len(info["seed"]))
            stats["bg"].add(info["bg"])
            stats["shapes"].add(inp.shape)
            stats["debris"] += bool(info["debris"])
            stats["diag"] += _diag_touch_same_colour(info)
            cols = {c for c, _ in info["squares"]}
            stats["zero_colour"] += any(c not in cols for c in info["seed"])
    assert len(stats["orient"]) == 8, stats["orient"]
    assert stats["k"] == {2, 3, 4}
    assert len(stats["bg"]) >= 5
    assert stats["debris"] > 20 and stats["diag"] > 5 and stats["zero_colour"] > 10
    assert len(stats["shapes"]) > 60
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 3), (4, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return True


if __name__ == "__main__":
    self_test()
    print("self_test ok")
