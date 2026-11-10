"""Key-operated electorate: ordered clipping changes labels and live port weights.

Only Python's standard library and NumPy are used. No parent code, saved pair,
seed or construction state is consulted by solve().
"""

from collections import Counter
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be in 0..9")
    return a


def _majority(values):
    counts = Counter(map(int, values))
    _require(bool(counts), "empty electorate")
    best = max(counts.values())
    winners = [c for c, n in counts.items() if n == best]
    _require(len(winners) == 1, "tied electorate")
    return winners[0]


def _optional_majority(values):
    try:
        return _majority(values)
    except ValueError:
        return None


def _components(mask):
    pending = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in pending:
                    pending.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _read_candidate(a, bg, structure, housing, comb):
    """Infer one upright interpretation solely from input pixels."""
    rs, cs = zip(*housing)
    t, b, l, r = min(rs), max(rs), min(cs), max(cs)
    _require(t >= 4 and b - t >= 4 and r - l >= 6, "housing scale")
    expected_u = {(t, c) for c in range(l, r + 1)}
    expected_u |= {(q, c) for q in range(t, b + 1) for c in (l, r)}
    _require(housing == expected_u, "housing must be an open-bottom U")

    e = max(q for q, c in comb)
    columns = sorted(c for q, c in comb if q == e - 1)
    _require(2 <= len(columns) <= 6, "tooth count")
    _require(all(y - x >= 3 for x, y in zip(columns, columns[1:])), "lane spacing")
    _require(columns[0] >= l + 2 and columns[-1] <= r - 2, "lane insets")
    _require({c for q, c in comb if q == e} == set(range(columns[0], columns[-1] + 1)), "comb base")
    _require(min(q for q, c in comb) > b, "external comb must be below housing")
    heights = []
    expected_comb = {(e, c) for c in range(columns[0], columns[-1] + 1)}
    for c in columns:
        q = e - 1
        while (q, c) in comb:
            q -= 1
        height = e - q - 1
        _require(1 <= height <= b - t - 2, "tooth height")
        heights.append(height)
        expected_comb |= {(q, c) for q in range(e - height, e)}
    _require(comb == expected_comb, "malformed comb")

    remaining = set(map(int, np.unique(a))) - {bg, structure}
    interpretations = []
    for socket in sorted(remaining):
        for cc in range(l, r - 1):
            collector = (t - 3, cc)
            if not np.all(a[t - 3:t, cc:cc + 3] == socket):
                continue
            signals = tuple(sorted(remaining - {socket}))
            if len(signals) != 2:
                continue
            try:
                sequences, capacities, labels = [], [], []
                covered = set(housing) | set(comb)
                covered |= {(q, c) for q in range(t - 3, t) for c in range(cc, cc + 3)}
                for c, height in zip(columns, heights):
                    seq = []
                    q = t + 1
                    while q < b and int(a[q, c]) in signals:
                        seq.append(int(a[q, c]))
                        covered.add((q, c))
                        q += 1
                    capacity = b - t - height - 1
                    _require(capacity <= len(seq) <= b - t - 1, "gap or cartridge length")
                    sequences.append(tuple(seq))
                    capacities.append(capacity)
                    labels.append(_majority(seq[-capacity:]))
                ports = []
                socket_cells = set(map(tuple, np.argwhere(a == socket))) - covered
                for q, c in sorted(socket_cells):
                    owners = [i for i, col in enumerate(columns) if abs(c - col) == 1]
                    _require(t + 1 <= q <= b - 2 and len(owners) == 1, "socket association")
                    ports.append((int(q), int(c), owners[0]))
                    covered.add((q, c))
                _require(set(i for q, c, i in ports) == set(range(len(columns))), "each lane needs a socket")
                _require(covered == set(map(tuple, np.argwhere(a != bg))), "unaccounted input cells")
                votes = [labels[i] for q, c, i in ports if q <= t + capacities[i]]
                winner = _majority(votes)
                interpretations.append(dict(
                    shape=a.shape, background=bg, structure=structure, socket=socket,
                    signals=signals, top=t, base=b, left=l, right=r, key_base=e,
                    columns=tuple(columns), heights=tuple(heights),
                    sequences=tuple(sequences), capacities=tuple(capacities),
                    labels=tuple(labels), ports=tuple(ports), collector=collector,
                    key_cells=tuple(sorted(comb)), winner=winner))
            except ValueError:
                continue
    _require(len(interpretations) == 1, "collector and voter roles are not unique")
    return interpretations[0]


def _read_input(input_grid):
    g = _grid(input_grid)
    corners = [int(g[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    _require(len(set(corners)) == 1, "background corners")
    bg = corners[0]
    _require(np.all(g[0] == bg) and np.all(g[-1] == bg)
             and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), "blank outer margin")
    values = tuple(map(int, np.unique(g)))
    _require(len(values) == 5, "five color roles required")
    candidates = []
    # Reflection needs no extra trials: left and right sockets obey the same rule.
    for turn in range(4):
        a = np.rot90(g, turn)
        for structure in values:
            if structure == bg:
                continue
            components = _components(a == structure)
            if len(components) != 2:
                continue
            for housing, comb in (components, components[::-1]):
                try:
                    f = _read_candidate(a, bg, structure, housing, comb)
                except ValueError:
                    continue
                f["turn"] = turn
                f["canonical"] = a
                candidates.append(f)
    _require(len(candidates) == 1, "input requires one geometric interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Move actual cells, then infer live contacts and votes; accepts input only."""
    f = _read_input(input_grid)
    a = f["canonical"]
    out = a.copy()
    bg, s, t, b = f["background"], f["structure"], f["top"], f["base"]
    for p in f["key_cells"]:
        out[p] = bg
    for c in f["columns"]:
        out[t + 1:b, c] = bg
    translation = b - f["key_base"]
    for q, c in f["key_cells"]:
        out[q + translation, c] = s
    local_labels = []
    for c, seq, capacity in zip(f["columns"], f["sequences"], f["capacities"]):
        shift = len(seq) - capacity
        moved = []
        for j, color in enumerate(seq):
            q = t + 1 + j - shift
            if q > t:
                out[q, c] = color
                moved.append(color)
        local_labels.append(_majority(moved))
    votes = []
    for q, c, owner in f["ports"]:
        if int(out[q, f["columns"][owner]]) in f["signals"]:
            out[q, c] = local_labels[owner]
            votes.append(local_labels[owner])
    cr, cc = f["collector"]
    out[cr + 1, cc + 1] = _majority(votes)
    return np.rot90(out, -f["turn"]).copy()


def _paint_fixed(f):
    a = np.full(f["shape"], f["background"], dtype=np.int8)
    t, b, l, r, s = (f[k] for k in ("top", "base", "left", "right", "structure"))
    a[t, l:r + 1] = s
    a[t:b + 1, l] = a[t:b + 1, r] = s
    cr, cc = f["collector"]
    a[cr:cr + 3, cc:cc + 3] = f["socket"]
    for q, c, owner in f["ports"]:
        a[q, c] = f["socket"]
    return a


def _render_input(f):
    a = _paint_fixed(f)
    e, cols, s = f["key_base"], f["columns"], f["structure"]
    a[e, cols[0]:cols[-1] + 1] = s
    for c, height, seq in zip(cols, f["heights"], f["sequences"]):
        a[e - height:e, c] = s
        a[f["top"] + 1:f["top"] + 1 + len(seq), c] = seq
    return a


def _construct_output(f):
    """Direct construction formula, independent of solve's pixel movement."""
    a = _paint_fixed(f)
    t, b, cols, s = f["top"], f["base"], f["columns"], f["structure"]
    a[b, cols[0]:cols[-1] + 1] = s
    capacities, labels = [], []
    for c, height, seq in zip(cols, f["heights"], f["sequences"]):
        k = b - t - height - 1
        capacities.append(k)
        suffix = seq[-k:]
        labels.append(_majority(suffix))
        a[t + 1:t + k + 1, c] = suffix
        a[b - height:b, c] = s
    votes = []
    for q, c, i in f["ports"]:
        if q - t <= capacities[i]:
            a[q, c] = labels[i]
            votes.append(labels[i])
    cr, cc = f["collector"]
    a[cr + 1, cc + 1] = _majority(votes)
    return a


def _geometry(rng, n, span, maximum_height):
    # Randomly allocate width slack among gaps, insets and outer margins.
    gaps = [3] * (n - 1)
    extras = [0] * (n + 3)
    caps = [3] * (n - 1) + [2, 2, 2, 2]
    budget = 30 - (3 * (n - 1) + 7)
    for i in rng.permutation(len(extras)):
        extras[i] = int(rng.integers(0, min(caps[i], budget) + 1))
        budget -= extras[i]
    gaps = [g + v for g, v in zip(gaps, extras[:n - 1])]
    il, ir, lm, rm = [base + extra for base, extra in zip((2, 2, 1, 1), extras[n - 1:])]
    cols = [lm + il]
    for gap in gaps:
        cols.append(cols[-1] + gap)
    right = cols[-1] + ir
    gap_cap = min(3, 24 - span - maximum_height)
    _require(gap_cap >= 1, "geometry cannot fit")
    external_gap = int(rng.integers(1, gap_cap + 1))
    margin_cap = min(3, 25 - span - maximum_height - external_gap)
    bottom_margin = int(rng.integers(1, margin_cap + 1))
    top_cap = min(6, 29 - span - maximum_height - external_gap - bottom_margin)
    t = int(rng.integers(4, top_cap + 1))
    b = t + span
    e = b + maximum_height + external_gap
    return dict(shape=(e + bottom_margin + 1, right + rm + 1), top=t, base=b,
                left=lm, right=right, key_base=e, columns=tuple(cols),
                collector=(t - 3, int(rng.integers(lm, right - 1))))


def _with_roles(f, roles):
    bg, structure, socket, x, y = map(int, roles)
    return dict(f, background=bg, structure=structure, socket=socket, signals=(x, y))


def _pair(f, pose):
    return {"input": _orient(_render_input(f), pose),
            "output": _orient(_construct_output(f), pose)}


def _comparison(rng, roles, pose, n, span):
    """Construct a counterfactual pair with a strict reversal and voting witness."""
    x, y = map(int, roles[3:])
    champion, flip = map(int, rng.choice(n, 2, replace=False))
    capacities = [int(rng.integers(4, span - 2)) for _ in range(n)]
    feasible = [k for k in range(3, min(5, span - 4) + 1)
                if k + (2 * (k // 2 + 1) - k + 1) <= span - 2]
    capacities[champion] = int(rng.choice(feasible))
    sequences = []
    still = int(rng.choice([i for i in range(n) if i != champion]))
    for i, k in enumerate(capacities):
        if i == champion:
            nx = k // 2 + 1
            tail = [x] * nx + [y] * (k - nx - 1)
            rng.shuffle(tail)
            tail.append(y)  # A minority contact distinguishes copying from a vote.
            minimum_prefix = 2 * nx - k + 1
            p = int(rng.integers(minimum_prefix, span - k - 1))
            sequences.append(tuple([y] * p + tail))
        else:
            length = k if i == still else int(rng.integers(k, span))
            sequences.append(tuple([y] * (length - 1) + [x]))
    second_capacities = capacities.copy()
    second_capacities[champion] = len(sequences[champion])  # Stationary, original y majority.
    second_capacities[flip] = 1  # The former minority at the tail now wins locally.
    heights = tuple(span - k - 1 for k in capacities)
    second_heights = tuple(span - k - 1 for k in second_capacities)
    f = _with_roles(_geometry(rng, n, span, max(*heights, *second_heights)), roles)
    ports = []
    for i, (col, k) in enumerate(zip(f["columns"], capacities)):
        if i == champion:
            weight = int(rng.integers(n + 1, min(2 * k, n + 3, 7) + 1))
            possible = [(q, side) for q in range(1, k + 1) for side in (-1, 1)]
            forced = (k, int(rng.choice((-1, 1))))
            possible.remove(forced)
            selected = [forced] + [possible[int(j)] for j in rng.choice(len(possible), weight - 1, replace=False)]
            lower = [(q, side) for q in range(k + 1, span - 1) for side in (-1, 1)]
            extra = int(rng.integers(1, min(8 - weight, len(lower), 2) + 1))
            awakening = (int(rng.integers(k + 1, len(sequences[i]) + 1)), int(rng.choice((-1, 1))))
            lower.remove(awakening)
            selected += [awakening] + [lower[int(j)] for j in rng.choice(len(lower), extra - 1, replace=False)]
        else:
            # The flip lane has a live socket that the companion key disconnects.
            selected = [(1, int(rng.choice((-1, 1))))]
            if i == flip:
                selected.append((int(rng.integers(2, k + 1)), int(rng.choice((-1, 1)))))
            lower = [(q, side) for q in range(k + 1, span - 1) for side in (-1, 1)]
            extra = int(rng.integers(1, min(len(lower), 3) + 1))
            selected += [lower[int(j)] for j in rng.choice(len(lower), extra, replace=False)]
        ports.extend((f["top"] + q, col + side, i) for q, side in selected)
    f.update(heights=heights, sequences=tuple(sequences), ports=tuple(sorted(ports)))
    g = dict(f, heights=second_heights)
    return [_pair(f, pose), _pair(g, pose)]


def _suffix_pattern(rng, length, winner, loser):
    minority = int(rng.integers(0, (length - 1) // 2 + 1))
    kind = int(rng.integers(3))
    seq = [loser] * minority + [winner] * (length - minority)
    if kind == 0:
        rng.shuffle(seq)
    elif kind == 1 and rng.random() < .5:
        seq.reverse()
    elif kind == 2:
        seq = [v for _ in range(minority) for v in (winner, loser)] + [winner] * (length - 2 * minority)
    return seq


def _ordinary_pair(rng, roles, pose, lane_range, span_range):
    for attempt in range(128):
        n = int(rng.integers(lane_range[0], lane_range[1] + 1))
        span = int(rng.integers(span_range[0], span_range[1] + 1))
        hcap = min(span - 2, 23 - span)
        lower_capacity = span - hcap - 1
        mode = int(rng.integers(5))
        capacities = rng.integers(lower_capacity, span - 1, n).tolist()
        if mode == 0:
            capacities = [capacities[0]] * n
        elif mode == 1:
            capacities.sort()
        elif mode == 2:
            capacities.sort(reverse=True)
        heights = tuple(span - k - 1 for k in capacities)
        labels = rng.integers(0, 2, n).tolist()
        if len(set(labels)) == 1:
            labels[int(rng.integers(n))] ^= 1
        x, y = map(int, roles[3:])
        sequences = []
        for k, height, label in zip(capacities, heights, labels):
            shift = int(rng.integers(0, height + 1))
            if rng.random() < .25:
                shift = 0
            suffix = _suffix_pattern(rng, k, (x, y)[label], (y, x)[label])
            prefix = rng.choice([x, y], shift).tolist()
            sequences.append(tuple(prefix + suffix))
        f = _with_roles(_geometry(rng, n, span, max(heights)), roles)
        ports = []
        for i, col in enumerate(f["columns"]):
            possible = [(q, side) for q in range(1, span - 1) for side in (-1, 1)]
            count = int(rng.integers(1, min(8, len(possible)) + 1))
            for j in rng.choice(len(possible), count, replace=False):
                q, side = possible[int(j)]
                ports.append((f["top"] + q, col + side, i))
        f.update(heights=heights, sequences=tuple(sequences), ports=tuple(sorted(ports)))
        try:
            return _pair(f, pose)
        except ValueError:  # Only empty or tied global elections need resampling.
            continue
    raise RuntimeError("bounded port-vote construction exhausted")


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
                 name + " must be an integer")
        _require(lo <= int(value) <= hi, name + " is out of range")
    _require(num_train > num_test, "num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # More lanes, changed pose, or deeper chamber.
    pose = int(rng.integers(8))
    mixed_pose = regime != 1 and bool(rng.integers(2))
    varying_palette = bool(rng.integers(2))
    roles = tuple(map(int, rng.choice(10, 5, replace=False)))
    lane_bounds = (2, 4) if regime == 0 else (2, 6)
    span_bounds = (7, 9) if regime == 2 else (7, 11)
    witness_n = int(rng.integers(3, (4 if regime == 0 else 5) + 1))
    witness_span = int(rng.integers(*[span_bounds[0], span_bounds[1] + 1]))
    train = _comparison(rng, roles, pose, witness_n, witness_span)
    seen = {_key(p["input"]) for p in train}
    puzzle = {"train": train, "test": []}
    for split, remaining in (("train", num_train - 2), ("test", num_test)):
        for index in range(remaining):
            pair_roles = tuple(map(int, rng.choice(10, 5, replace=False))) if varying_palette else roles
            pair_pose = int(rng.integers(8)) if mixed_pose else pose
            lanes, spans = lane_bounds, span_bounds
            if split == "test":
                if regime == 0:
                    lanes = (5, 6)
                elif regime == 1:
                    pair_pose = (pose // 4) * 4 + (pose % 4 + int(rng.integers(1, 4))) % 4
                else:
                    spans = (11, 13)
            for attempt in range(64):
                pair = _ordinary_pair(rng, pair_roles, pair_pose, lanes, spans)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness construction exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    return puzzle


def _measure(f):
    t = f["top"]
    weights = [0] * len(f["columns"])
    minorities = 0
    for q, c, i in f["ports"]:
        if q <= t + f["capacities"][i]:
            weights[i] += 1
            minorities += f["sequences"][i][-f["capacities"][i] + q - t - 1] != f["labels"][i]
    old_labels = [_optional_majority(seq) for seq in f["sequences"]]
    old_votes = [label for label, w in zip(old_labels, weights) for _ in range(w)]
    no_clipping = None if any(label is None and w for label, w in zip(old_labels, weights)) else _optional_majority(old_votes)
    surviving = [v for seq, k in zip(f["sequences"], f["capacities"]) for v in seq[-k:]]
    shifts = [len(seq) - k for seq, k in zip(f["sequences"], f["capacities"])]
    return dict(lanes=len(weights), span=f["base"] - t, heights=list(f["heights"]),
                capacities=list(f["capacities"]), lengths=list(map(len, f["sequences"])),
                shifts=shifts, weights=weights, labels=list(f["labels"]), winner=f["winner"],
                without_clipping=no_clipping, equal_cartridges=_optional_majority(f["labels"]),
                raw_survivors=_optional_majority(surviving),
                raw_input=_optional_majority(v for seq in f["sequences"] for v in seq),
                minority_contacts=int(minorities), active_ports=sum(weights),
                inactive_ports=len(f["ports"]) - sum(weights),
                changed_local_labels=sum(a != b for a, b in zip(old_labels, f["labels"]) if a is not None),
                stationary=sum(v == 0 for v in shifts), moving=sum(v > 0 for v in shifts),
                roles=[f["background"], f["structure"], f["socket"], *f["signals"]],
                gaps=list(map(int, np.diff(f["columns"]))),
                both_sides=any(any(c < f["columns"][i] for q, c, owner in f["ports"] if owner == i)
                               and any(c > f["columns"][i] for q, c, owner in f["ports"] if owner == i)
                               for i in range(len(weights))))


def _hardware_key(f):
    a = f["canonical"].copy()
    for p in f["key_cells"]:
        a[p] = f["background"]
    return _key(a)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "pair lists required")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_readings = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape must be preserved")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            f = _read_input(inp)
            canonical_out = np.rot90(out, f["turn"])
            _require(np.array_equal(solve(inp), out), "incorrect solved output")
            # Explicit invariants also check sequence, inactivity and preservation.
            allowed = set(f["key_cells"])
            for col, h, seq, k in zip(f["columns"], f["heights"], f["sequences"], f["capacities"]):
                _require(tuple(map(int, canonical_out[f["top"] + 1:f["top"] + k + 1, col])) == seq[-k:],
                         "surviving sequence changed")
                allowed |= {(q, col) for q in range(f["top"] + 1, f["base"] + 1)}
            allowed |= {(f["base"], c) for c in range(f["columns"][0], f["columns"][-1] + 1)}
            for q, c, i in f["ports"]:
                active = q <= f["top"] + f["capacities"][i]
                expected = f["labels"][i] if active else f["socket"]
                _require(int(canonical_out[q, c]) == expected, "wrong socket state")
                allowed.add((q, c))
            cr, cc = f["collector"]
            allowed.add((cr + 1, cc + 1))
            changed = set(map(tuple, np.argwhere(canonical_out != f["canonical"])))
            _require(changed <= allowed, "unaffected pixels changed")
            if split == "train":
                train_readings.append(f)
    measures = [_measure(f) for f in train_readings]
    witness = lambda m: (m["without_clipping"] is not None and m["without_clipping"] != m["winner"]
                         and m["equal_cartridges"] is not None and m["equal_cartridges"] != m["winner"]
                         and m["raw_survivors"] is not None and m["raw_survivors"] != m["winner"]
                         and m["raw_input"] is not None and m["raw_input"] != m["winner"]
                         and m["minority_contacts"] > 0 and m["inactive_ports"] > 0
                         and m["stationary"] > 0 and m["moving"] > 0)
    _require(any(witness(m) for m in measures), "missing coupled counterexample witness")
    compared = False
    for i, first in enumerate(train_readings):
        for second in train_readings[i + 1:]:
            if _hardware_key(first) == _hardware_key(second):
                state_a = [q <= first["top"] + first["capacities"][owner] for q, c, owner in first["ports"]]
                state_b = [q <= second["top"] + second["capacities"][owner] for q, c, owner in second["ports"]]
                gates_change = any(a and not b for a, b in zip(state_a, state_b)) and any(
                    b and not a for a, b in zip(state_a, state_b))
                compared |= (first["winner"] != second["winner"] and first["heights"] != second["heights"]
                             and gates_change)
    _require(compared, "missing same-hardware key comparison with reversed result")


def _normalized_structure(a):
    mapping = {}
    return a.shape, bytes(mapping.setdefault(int(v), len(mapping)) for v in a.flat)


def _encoded_puzzle(puzzle):
    return tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))


def self_test():
    """Bounded authored checks and input-derived variation over seeds 0..41."""
    # A small semantic fixture outside the sampled span range. Lane 0 starts
    # tied, clips to one voter, and drives two sockets; lane 1 stays still.
    # Expected pixels are specified directly rather than through the renderer.
    fixture = dict(shape=(14, 13), top=4, base=8, left=1, right=11,
                   key_base=11, columns=(4, 8), heights=(2, 1),
                   sequences=((3, 2), (3, 3)), collector=(1, 5),
                   ports=((5, 3, 0), (5, 5, 0), (6, 3, 0), (6, 9, 1)))
    fixture = _with_roles(fixture, (0, 5, 8, 2, 3))
    expected_rows = ["0000000000000", "0000088800000", "0000082800000",
                     "0000088800000", "0555555555550", "0502220030050",
                     "0508500033050", "0500500050050", "0500555550050"]
    expected_rows += ["0000000000000"] * 5
    expected = np.array([[int(v) for v in row] for row in expected_rows], dtype=np.int8)
    _require(np.array_equal(solve(_render_input(fixture)), expected), "small semantic fixture")

    # More sockets than the constructor samples remains a legal family member.
    wide = _with_roles(_geometry(np.random.default_rng(15), 2, 12, 7), (0, 5, 8, 2, 3))
    ports = [(wide["top"] + q, wide["columns"][0] + side, 0)
             for q in range(1, 11) for side in (-1, 1)]
    ports.append((wide["top"] + 1, wide["columns"][1] + 1, 1))
    wide.update(heights=(1, 7), sequences=((2,) * 10, (3,) * 4), ports=tuple(sorted(ports)))
    large_prediction = solve(_render_input(wide))
    _require(np.array_equal(large_prediction, _construct_output(wide)), "extended socket multiplicity")

    puzzles, shapes, structures, profiles, gaps, roles, lane_counts, spans = (set() for _ in range(8))
    measurements = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        _require(_encoded_puzzle(p) == _encoded_puzzle(generate_puzzle(seed)), "nondeterministic seed")
        puzzles.add(_encoded_puzzle(p))
        for split in ("train", "test"):
            for pair in p[split]:
                f = _read_input(pair["input"])
                m = _measure(f)
                measurements.append(m)
                shapes.add(pair["input"].shape)
                structures.add(_normalized_structure(pair["input"]))
                profiles.add(f["heights"])
                gaps.update(map(int, np.diff(f["columns"])))
                roles.add(tuple(m["roles"]))
                lane_counts.add(m["lanes"])
                spans.add(m["span"])
    _require(len(puzzles) == 42 and len(structures) >= 150, "insufficient structural diversity")
    _require(len(shapes) >= 40 and len(profiles) >= 80 and len(roles) >= 40, "insufficient variation")
    _require(lane_counts == set(range(2, 7)) and gaps == {3, 4, 5, 6}, "geometry variation")
    _require(spans.issuperset(range(7, 14)), "depth variation")
    _require(any(0 in m["weights"] for m in measurements), "missing disconnected cartridge")
    _require(any(len(set(m["heights"])) == 1 for m in measurements), "missing flat key")
    _require(any(min(m["capacities"]) == 1 for m in measurements), "missing one-cell electorate")
    _require(any(m["both_sides"] for m in measurements), "missing two-sided contacts")

    p = generate_puzzle(0, 2, 1)
    witness = next(pair for pair in p["train"]
                   if _measure(_read_input(pair["input"]))["without_clipping"] != _read_input(pair["input"])["winner"])
    # Recoloring and all eight whole-scene symmetries preserve the operation.
    mapping = np.array([(3 * c + 7) % 10 for c in range(10)], dtype=np.int8)
    for pose in range(8):
        inp, out = (_orient(mapping[witness[k]], pose) for k in ("input", "output"))
        _require(np.array_equal(solve(inp), out), "symmetry or recoloring failure")

    # Same multiset, changed order: placing opposition at the surviving tail matters.
    f = _read_input(witness["input"])
    champion = next(i for i, label in enumerate(f["labels"]) if label == f["winner"])
    a = f["canonical"].copy()
    col = f["columns"][champion]
    seq = f["sequences"][champion]
    ordered = sorted(seq, key=lambda color: color != f["winner"])
    a[f["top"] + 1:f["top"] + len(seq) + 1, col] = ordered
    altered = _read_input(a)
    _require(altered["labels"][champion] != f["labels"][champion], "cartridge order was ignored")

    # Wrong center and a wrong active socket must be rejected by validation.
    for target in ("center", "socket"):
        q = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = q["test"][0]
        f = _read_input(pair["input"])
        out = np.rot90(pair["output"], f["turn"]).copy()
        cr, cc = f["collector"]
        pos = (cr + 1, cc + 1) if target == "center" else next(
            (r, c) for r, c, i in f["ports"] if r <= f["top"] + f["capacities"][i])
        out[pos] = f["socket"]
        pair["output"] = np.rot90(out, -f["turn"]).copy()
        try:
            validate_generated_puzzle(q)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong " + target + " accepted")
    for counts in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(19, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=len(measurements), distinct_puzzles=len(puzzles),
                input_shapes=len(shapes), color_normalized_inputs=len(structures),
                key_profiles=len(profiles), palettes=len(roles), lanes=sorted(lane_counts),
                lane_gaps=sorted(gaps), chamber_spans=sorted(spans),
                stationary_lanes=sum(m["stationary"] for m in measurements),
                moving_lanes=sum(m["moving"] for m in measurements),
                inactive_sockets=sum(m["inactive_ports"] for m in measurements),
                minority_contacts=sum(m["minority_contacts"] for m in measurements))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
