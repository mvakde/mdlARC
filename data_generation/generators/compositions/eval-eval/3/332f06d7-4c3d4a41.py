"""A rail-guided square pushes roof pins until its footprint meets masonry.

The constructor uses a swept-envelope formula; solve reads pixels and moves
one rail step at a time. No examples, seeds, or construction state enter solve.
Only NumPy and the Python standard library are used. Nothing is written here.
"""

from collections import Counter
import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 250277546, 35696607, 64138691, 718290286, 350431969,
              133563681, 688189718, 457769400, 708617423, 740448364, 66695955,
              129972829, 113334784, 173135077, 649240029, 228429269, 364993560,
              666144880, 266815274, 306860989, 59375896, 334273582, 995706985,
              12451189, 131153007, 12754091, 707621471, 815961225, 237001158,
              57867376, 771165576, 95671832, 496576574, 86296730, 428990004,
              343728562, 711276312, 212636090, 128861155, 315174445, 361491728)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _counts(num_train, num_test):
    for value, lower, upper, name in ((num_train, 2, 10, "num_train"),
                                      (num_test, 1, 4, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if not lower <= value <= upper:
            raise ValueError(name + " is outside its legal range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((a >= 0) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _square(a, color):
    cells = np.argwhere(a == color)
    _require(len(cells) > 0, "absent square")
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    k = int(hi[0] - lo[0])
    _require(k >= 3 and k % 2 == 1 and hi[1] - lo[1] == k, "not an odd square")
    _require(len(cells) == k * k, "not a solid exclusive square")
    return int(lo[0]), int(lo[1]), k


def _neighbors(p, shape):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        q = p[0] + dr, p[1] + dc
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _read_rail(a, color, start):
    cells = {tuple(map(int, p)) for p in np.argwhere(a == color)}
    _require(len(cells) >= 5 and start in cells, "rail absent at square's foot")
    adjacency = {p: [q for q in _neighbors(p, a.shape) if q in cells] for p in cells}
    _require(all(len(v) in (1, 2) for v in adjacency.values()), "branched or isolated rail")
    ends = [p for p in cells if len(adjacency[p]) == 1]
    _require(len(ends) == 2 and start in ends, "square must start at a rail endpoint")
    path, previous, current = [], None, start
    while current is not None:
        path.append(current)
        following = [q for q in adjacency[current] if q != previous]
        previous, current = current, following[0] if following else None
    _require(len(path) == len(cells), "disconnected rail or loop")
    return path


def _box(anchor, k):
    return anchor[0] - k, anchor[1] - k // 2


def _fits(static, anchor, k):
    r, c = _box(anchor, k)
    return (r >= 0 and c >= 0 and r + k <= static.shape[0]
            and c + k <= static.shape[1] and not static[r:r+k, c:c+k].any())


def _read_canonical(a, bg, structure, mover, rail):
    mr, mc, k = _square(a, mover)
    path = _read_rail(a, rail, (mr + k, mc + k // 2))
    h, w = a.shape
    _require(all(0 < r < h - 1 and 0 < c < w - 1 for r, c in path), "rail crosses frame")
    pin_mask = np.zeros(a.shape, dtype=bool)
    pins = []
    excluded = {bg, structure, mover, rail}
    for col in range(1, w - 1):
        row = 1
        while row < h - 1 and int(a[row, col]) not in (bg, mover, rail):
            row += 1
        seq = tuple(map(int, a[1:row, col]))
        if not any(v not in excluded for v in seq):
            continue
        prefix = next(i for i, v in enumerate(seq) if v != structure)
        _require(all(v not in excluded for v in seq[prefix:]), "pin prefix must precede payload")
        _require(2 <= col <= w - 3, "pin too near side frame")
        _require(not pins or col - pins[-1][0] >= 2, "pin lanes must be separated")
        rail_rows = [r for r, c in path if c == col]
        _require(not rail_rows or len(seq) < min(rail_rows), "pin covers rail")
        pins.append((col, seq))
        pin_mask[1:row, col] = True
    _require(len(pins) >= 2, "need two or more roof pins")
    static = (a == structure) & ~pin_mask
    # Interior masonry consists of roof-rooted pillars, not arbitrary noise.
    for col in range(1, w - 1):
        occupied = np.flatnonzero(static[1:-1, col]) + 1
        _require(np.array_equal(occupied, np.arange(1, len(occupied) + 1)), "floating masonry")
    explained = (a == bg) | static | pin_mask | (a == rail) | (a == mover)
    _require(explained.all(), "unattached colored cells")
    _require(_fits(static | pin_mask, path[0], k), "initial square overlaps material")
    return dict(grid=a, background=bg, structure=structure, mover=mover, rail=rail,
                k=k, path=path, pins=pins, static=static, pin_mask=pin_mask)


def _parse(input_grid):
    a = _grid(input_grid)
    _require(min(a.shape) >= 7, "scene too small")
    structure = int(a[0, 0])
    _require(np.all(a[0] == structure) and np.all(a[-1] == structure)
             and np.all(a[:, 0] == structure) and np.all(a[:, -1] == structure), "missing frame")
    bg = int(a[1, 1])
    _require(bg != structure and all(int(a[r, c]) == bg
             for r in (1, a.shape[0] - 2) for c in (1, a.shape[1] - 2)), "empty corner convention")
    colors = set(map(int, np.unique(a))) - {bg, structure}
    movers = []
    for color in sorted(colors):
        try:
            _square(a, color)
            movers.append(color)
        except AssertionError:
            pass
    _require(len(movers) == 1, "unique solid square required")
    mover = movers[0]
    candidates = []
    for turns in range(4):
        canonical = np.rot90(a, turns).copy()
        for rail in sorted(colors - {mover}):
            try:
                f = _read_canonical(canonical, bg, structure, mover, rail)
            except AssertionError:
                continue
            f["turns"] = turns
            candidates.append(f)
    _require(len(candidates) == 1, "scene must identify one roof and rail")
    return candidates[0]


def _base_output(f):
    out = f["grid"].copy()
    out[out == f["mover"]] = f["background"]
    out[f["pin_mask"]] = f["background"]
    return out


def _finish(f, anchor, sequences):
    out = _base_output(f)
    for (col, _), seq in zip(f["pins"], sequences):
        out[1:1+len(seq), col] = seq
    r, c = _box(anchor, f["k"])
    out[r:r+f["k"], c:c+f["k"]] = f["mover"]
    return out


def _simulate(f):
    """Push literal ordered pixels at each accepted step, dropping roof overflow."""
    sequences = [list(seq) for _, seq in f["pins"]]
    anchor = f["path"][0]
    for proposed in f["path"][1:]:
        if not _fits(f["static"], proposed, f["k"]):
            break
        top, left = _box(proposed, f["k"])
        for i, (col, _) in enumerate(f["pins"]):
            if left <= col < left + f["k"]:
                # All stacks touch the roof. Each upward unit clips its first
                # cell; cells that never contact the mover stay stationary.
                while sequences[i] and len(sequences[i]) >= top:
                    sequences[i].pop(0)
        anchor = proposed
    return _finish(f, anchor, sequences)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input's frame, square, rail, masonry and pin pixels."""
    f = _parse(input_grid)
    return np.rot90(_simulate(f), -f["turns"]).copy()


def _reachable(f, hard_pins=False):
    static = f["static"] | f["pin_mask"] if hard_pins else f["static"]
    last = 0
    for i, anchor in enumerate(f["path"][1:], 1):
        if not _fits(static, anchor, f["k"]):
            break
        last = i
    return last


def _envelope(f, last):
    """Constructor/checker: minimum swept top per column, not a stack simulator."""
    limit = np.full(f["grid"].shape[1], f["grid"].shape[0], dtype=int)
    for anchor in f["path"][:last+1]:
        r, c = _box(anchor, f["k"])
        limit[c:c+f["k"]] = np.minimum(limit[c:c+f["k"]], r - 1)
    return limit


def _formula_output(f, last=None):
    if last is None:
        last = _reachable(f)
    capacity = _envelope(f, last)
    seqs = [seq[max(0, len(seq) - int(capacity[col])):] for col, seq in f["pins"]]
    return _finish(f, f["path"][last], seqs)


def _features(f):
    last = _reachable(f)
    cap, full = _envelope(f, last), _envelope(f, len(f["path"]) - 1)
    shifts, prefixes, suffixes = [], [], []
    for col, seq in f["pins"]:
        shift = max(0, len(seq) - int(cap[col]))
        shifts.append(shift)
        prefixes.append(next(i for i, v in enumerate(seq) if v != f["structure"]))
        suffixes.append(seq[shift:])
    path = f["path"]
    headings = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    end_top = _box(path[last], f["k"])[0]
    return dict(k=f["k"], shape=f["grid"].shape, turns=f["turns"],
                bends=sum(a != b for a, b in zip(headings, headings[1:])),
                reached_bends=sum(a != b for a, b in zip(headings[:last], headings[1:last])),
                pins=len(shifts), pillars=int(np.count_nonzero(f["static"][1, 1:-1])),
                length=len(path), travel=last, success=last == len(path) - 1,
                shifts=tuple(shifts), prefixes=tuple(prefixes),
                stationary=any(v == 0 for v in shifts), moving=any(v > 0 for v in shifts),
                lane_specific=len(set(shifts)) > 1,
                moving_prefix=any(0 < s < p for s, p in zip(shifts, prefixes)),
                payload_clipped=any(s > p for s, p in zip(shifts, prefixes)),
                ordered_suffix=any(s > 0 and len(set(seq) - {f["structure"]}) >= 2
                                   for s, seq in zip(shifts, suffixes)),
                history=any(s > 0 and cap[col] < end_top - 1
                            for s, (col, _) in zip(shifts, f["pins"])),
                downstream=any(cap[col] >= len(seq) > full[col] for col, seq in f["pins"]),
                hard_pin_stop=_reachable(f, True) < last,
                palette=(f["background"], f["structure"], f["mover"], f["rail"]))


def _route(rng, h, w, k, legs, witness, late_stop=False):
    first, end = k // 2 + 2, w - k // 2 - 3
    span = end - first
    minima = [max(2, k // 2 + 1)] + [2] * (legs - 1)
    if late_stop:
        minima[-1] = k // 2 + 5
    _require(sum(minima) <= span, "route does not fit")
    extra = rng.multinomial(span - sum(minima), np.ones(legs) / legs)
    lengths = [a + int(b) for a, b in zip(minima, extra)]
    low, high = k + 4, h - 3
    _require(high - low >= 3, "vertical range too small")
    if witness:
        peak = int(rng.integers(low, high - 1))
        rows = [int(rng.integers(peak + 1, high + 1)), peak,
                int(rng.integers(peak + 1, high + 1))]
    else:
        rows = [int(rng.integers(low, high + 1))]
        for _ in range(legs - 1):
            choices = [r for r in range(low, high + 1) if 0 < abs(r - rows[-1]) <= 4]
            rows.append(int(rng.choice(choices)))
    path, col = [(rows[0], first)], first
    for i, length in enumerate(lengths):
        for _ in range(length):
            col += 1
            path.append((rows[i], col))
        if i + 1 < legs:
            step = 1 if rows[i+1] > rows[i] else -1
            for row in range(rows[i] + step, rows[i+1] + step, step):
                path.append((row, col))
    return path


def _make_scene(rng, colors, k, legs, witness, blocked, late_stop=False):
    bg, structure, mover, rail = colors
    min_width = max(22 if k >= 5 else 20, k + 4 + 2 * legs + max(0, k // 2 - 1))
    if late_stop:
        min_width = max(min_width, 2 * k + 2 * legs + 5)
    w = int(rng.integers(min_width, 31))
    h = int(rng.integers(k + 10, 31))
    path = _route(rng, h, w, k, legs, bool(witness), late_stop)
    a = np.full((h, w), bg, dtype=np.int8)
    a[0] = a[-1] = structure
    a[:, 0] = a[:, -1] = structure
    static = a == structure
    first, end = path[0][1], path[-1][1]
    geometry = dict(grid=a, static=static, k=k, path=path)
    full_capacity = _envelope(geometry, len(path) - 1)
    min_rail = {c: min(r for r, x in path if x == c) for _, c in path}
    blocker = None
    if blocked:
        lower = first + k // 2 + 4
        if late_stop:
            last_rise_col = max(a[1] for a, b in zip(path, path[1:]) if a[1] == b[1])
            lower = max(lower, last_rise_col + k // 2 + 2)
        upper = end - 3
        _require(lower <= upper, "no room for stop and untouched suffix")
        blocker = int(rng.integers(lower, upper + 1))
        depth = min_rail[blocker] - 1
        a[1:depth+1, blocker] = structure
        static[1:depth+1, blocker] = True
    # Sample genuine passable ceiling restrictions as well as the main stop.
    forbidden = set(range(first-k//2, first+k//2+1))
    pillar_cols = {blocker} if blocker is not None else set()
    for col in rng.permutation(np.arange(first+k//2+1, end+k//2+1))[:int(rng.integers(0, 4))]:
        col = int(col)
        if col in pillar_cols or col in forbidden:
            continue
        depth = int(rng.integers(1, max(2, int(full_capacity[col]) + 1)))
        a[1:depth+1, col] = structure
        static[1:depth+1, col] = True
        pillar_cols.add(col)
    last = _reachable(geometry)
    _require(last > 1 and (last < len(path)-1) == blocked, "wrong construction outcome")
    capacity = _envelope(geometry, last)
    candidates = [c for c in range(first+k//2+1, end+k//2+1)
                  if c not in pillar_cols and 2 <= c <= w-3]
    reachable = [c for c in candidates if capacity[c] < min_rail.get(c, h - 2) - 1]
    downstream = [c for c in candidates if capacity[c] == h
                  and full_capacity[c] < min_rail.get(c, h - 2) - 1]
    chosen = []
    special = {}
    if witness == "sweep":
        past_peak = [c for c in reachable if capacity[c] < path[-1][0] - k - 1]
        _require(past_peak, "need a remembered high sweep")
        chosen.append(int(rng.choice(past_peak)))
        special[chosen[0]] = "prefix"
        other = [c for c in reachable if abs(c - chosen[0]) >= 2]
        _require(other, "need separate clipping lane")
        chosen.append(int(rng.choice(other)))
        special[chosen[1]] = "clip"
    elif witness == "stop":
        _require(reachable and downstream, "need reached and unreached pins")
        chosen.append(int(rng.choice(reachable)))
        special[chosen[0]] = "clip"
        other = [c for c in downstream if abs(c - chosen[0]) >= 2]
        _require(other, "need an untouched downstream pin")
        chosen.append(int(rng.choice(other)))
        special[chosen[1]] = "downstream"
    target = int(rng.integers(3, min(8, (w - k) // 2) + 1))
    for col in rng.permutation(candidates):
        col = int(col)
        if len(chosen) >= target:
            break
        if all(abs(col - c) >= 2 for c in chosen):
            chosen.append(col)
    _require(len(chosen) >= 3, "need at least three readable lanes")
    if witness == "sweep":
        ordinary = [c for c in chosen if c not in special]
        special[int(rng.choice(ordinary))] = "stationary"
    payload_palette = [int(v) for v in range(10) if v not in colors]
    n_payload = int(rng.integers(2, min(6, len(payload_palette)) + 1))
    payload_palette = [int(v) for v in rng.choice(payload_palette, n_payload, replace=False)]
    pins = []
    for col in sorted(chosen):
        kind = special.get(col, "random")
        limit = int(full_capacity[col] if kind == "downstream" else capacity[col])
        max_length = min_rail.get(col, h - 2) - 1
        if kind in ("prefix", "clip", "downstream"):
            shift = int(rng.integers(1, min(3, max_length - limit) + 1))
            length = limit + shift
        elif kind == "stationary":
            shift = 0
            length = int(rng.integers(2, min(limit, max_length) + 1))
        else:
            length = int(rng.integers(1, max_length + 1))
            shift = max(0, length - limit)
        if kind == "prefix":
            prefix = shift + 1
        elif kind == "clip":
            prefix = int(rng.integers(0, shift))
        else:
            prefix = int(rng.integers(0, length)) if rng.random() < 0.6 else 0
        payload_length = length - prefix
        runs = int(rng.integers(1, min(3, payload_length) + 1))
        cuts = sorted(map(int, rng.choice(np.arange(1, payload_length), runs - 1, replace=False)))
        bounds = [0, *cuts, payload_length]
        run_colors = [int(v) for v in rng.choice(payload_palette, runs)]
        payload = [color for i, color in enumerate(run_colors)
                   for _ in range(bounds[i+1] - bounds[i])]
        seq = [structure] * prefix + payload
        if kind == "prefix":
            seq[-2:] = payload_palette[:2]
        pins.append((col, tuple(seq)))
        a[1:length+1, col] = seq
    for point in path:
        a[point] = rail
    r, c = _box(path[0], k)
    a[r:r+k, c:c+k] = mover
    pin_mask = np.zeros(a.shape, dtype=bool)
    for col, seq in pins:
        pin_mask[1:1+len(seq), col] = True
    f = dict(grid=a, background=bg, structure=structure, mover=mover, rail=rail,
             path=path, k=k, pins=pins, static=static, pin_mask=pin_mask)
    out = _formula_output(f)
    code = int(rng.integers(8))
    pair = {"input": _orient(a, code), "output": _orient(out, code)}
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("size", "bends")))
    base_k = 3 if regime == "size" else int(rng.choice((3, 5)))
    base_colors = tuple(map(int, rng.choice(10, 4, replace=False)))
    variable_colors = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_colors = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = ("sweep", "stop")[index] if split == "train" and index < 2 else ""
            k = base_k if split == "train" or regime == "bends" else int(rng.choice((5, 7)))
            legs = 3 if witness else (int(rng.integers(1, 4)) if split == "train"
                                       else int(rng.integers(4, 7)) if regime == "bends"
                                       else int(rng.integers(1, 5)))
            blocked = witness == "stop" or (not witness and bool(rng.integers(2)))
            for _ in range(96):
                colors = tuple(map(int, rng.choice(10, 4, replace=False))) if variable_colors else base_colors
                if variable_colors and split == "train" and index == 1 and colors == first_colors:
                    continue
                try:
                    late_stop = split == "test" and regime == "bends" and blocked
                    pair = _make_scene(rng, colors, k, legs, witness, blocked, late_stop)
                    f = _parse(pair["input"])
                    feature = _features(f)
                    _require(np.array_equal(solve(pair["input"]), pair["output"]), "constructor mismatch")
                    if witness == "sweep":
                        _require(all(feature[v] for v in ("success", "history", "moving_prefix",
                                     "payload_clipped", "ordered_suffix", "stationary", "lane_specific")),
                                 "incomplete sweep demonstration")
                    if witness == "stop":
                        _require(not feature["success"] and feature["downstream"]
                                 and feature["moving"] and feature["hard_pin_stop"], "incomplete stop demonstration")
                    key = (pair["input"].shape, pair["input"].tobytes())
                    _require(key not in seen, "duplicate input")
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    first_colors = colors
                break
            else:
                raise RuntimeError("bounded scene construction exhausted")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split type")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            f = _parse(inp)
            canonical_out = np.rot90(out, f["turns"])
            last = _reachable(f)
            _require(np.array_equal(canonical_out, _formula_output(f, last)), "wrong swept envelope or scenery")
            _require(np.array_equal(solve(inp), out), "stepwise solver disagrees")
            _require(not np.array_equal(inp, out), "identity example")
            _require(np.all(canonical_out[f["static"]] == f["structure"]), "masonry changed")
            _square(canonical_out, f["mover"])
            if last < len(f["path"]) - 1:
                _require(not _fits(f["static"], f["path"][last+1], f["k"]), "not first collision")
            feature = _features(f)
            if split == "train":
                training.append(feature)
            else:
                tests.append(feature)
    _require(any(v["success"] and v["history"] and v["moving_prefix"] and v["ordered_suffix"]
                 and v["payload_clipped"] and v["stationary"] and v["lane_specific"] for v in training),
             "training lacks a complete sweep witness")
    _require(any(not v["success"] and v["travel"] > 0 and v["downstream"]
                 and v["moving"] and v["hard_pin_stop"] for v in training), "training lacks a stop witness")
    _require(all(v["k"] > max(t["k"] for t in training) for v in tests)
             or all(v["reached_bends"] > max(t["bends"] for t in training) for v in tests),
             "missing effective train/test shift")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _hand_regressions():
    """Literal expected suffixes on a rail that revisits columns from above.

    This return geometry is outside the random sampler's monotone staircases.
    Expected cells below are specified directly, without the envelope routine.
    """
    path = [(12, c) for c in range(3, 18)]
    path += [(r, 17) for r in range(11, 7, -1)]
    path += [(8, c) for c in range(16, 6, -1)]
    path += [(7, 7), (6, 7)] + [(6, c) for c in range(8, 13)]
    pins = ((5, (1, 1, 1, 1, 1, 5, 6, 6, 7, 7)),
            (9, (1, 5, 5, 6, 7)), (13, (6, 6, 7, 7, 5, 5, 6)))
    for blocked in (False, True):
        inp = np.zeros((15, 22), dtype=np.int8)
        inp[0] = inp[-1] = 1
        inp[:, 0] = inp[:, -1] = 1
        if blocked:
            inp[1:8, 15] = 1
        for col, seq in pins:
            inp[1:len(seq)+1, col] = seq
        for point in path:
            inp[point] = 3
        inp[9:12, 2:5] = 2
        expected = inp.copy()
        expected[9:12, 2:5] = 0
        for col, seq in pins:
            expected[1:len(seq)+1, col] = 0
        suffixes = ((1, 1, 1, 5, 6, 6, 7, 7), pins[1][1], pins[2][1]) if blocked else (
            (1, 1, 1, 5, 6, 6, 7, 7), (6, 7), (5, 6))
        for (col, _), seq in zip(pins, suffixes):
            expected[1:len(seq)+1, col] = seq
        if blocked:
            expected[5:8, 16:19] = 2
        else:
            expected[3:6, 11:14] = 2
        _require(np.array_equal(solve(inp), expected), "literal return-rail regression")
        f = _parse(inp)
        _require(np.array_equal(_simulate(dict(f, pins=list(reversed(f["pins"])))), expected),
                 "pin processing order changed result")
        for point in ((0, 0), (3, 9), (12, 13)):
            malformed = inp.copy()
            malformed[point] = 0
            try:
                solve(malformed)
            except AssertionError:
                continue
            raise AssertionError("broken frame, pin gap or split rail accepted")


def self_test():
    """Bounded streaming checks: 42 seeds, 30 legal count pairs and mutations."""
    _hand_regressions()
    shapes, structures, signatures, palettes, pin_counts, sizes, bends = (set() for _ in range(7))
    features = Counter()
    pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        _require(_fingerprint(p) == _fingerprint(generate_puzzle(seed)), "nondeterministic seed")
        signatures.add(_fingerprint(p))
        local = []
        for split in ("train", "test"):
            for pair in p[split]:
                f = _parse(pair["input"])
                v = _features(f)
                local.append((split, v))
                shapes.add(v["shape"])
                palettes.add(v["palette"])
                pin_counts.add(v["pins"])
                sizes.add(v["k"])
                bends.add(v["bends"])
                # Geometry signature discards literal colors and pin contents.
                normalized = np.zeros(f["grid"].shape, dtype=np.int8)
                normalized[f["static"]] = 1
                normalized[f["pin_mask"]] = 2
                normalized[f["grid"] == f["rail"]] = 3
                normalized[f["grid"] == f["mover"]] = 4
                structures.add((normalized.shape, normalized.tobytes()))
                for name in ("success", "stationary", "moving_prefix", "payload_clipped",
                             "ordered_suffix", "history", "downstream", "hard_pin_stop"):
                    features[name] += bool(v[name])
                features["stopped"] += not v["success"]
                features["rotation_" + str(v["turns"])] += 1
                if split == "test":
                    features["test_success" if v["success"] else "test_stop"] += 1
                pairs += 1
        tr = [v for s, v in local if s == "train"]
        te = [v for s, v in local if s == "test"]
        _require(all(v["k"] > max(t["k"] for t in tr) for v in te)
                 or all(v["reached_bends"] > max(t["bends"] for t in tr) for v in te), "missing test shift")
    _require(len(signatures) == 42 and len(structures) >= 190, "structural repetition")
    _require(len(shapes) >= 60 and len(palettes) >= 42, "shape or role-palette collapse")
    _require(sizes == {3, 5, 7} and len(pin_counts) >= 4, "size or lane count collapse")
    _require(0 in bends and max(bends) >= 10, "route complexity collapse")
    _require(all(features[name] > 0 for name in ("test_success", "test_stop",
                 "rotation_0", "rotation_1", "rotation_2", "rotation_3")), "outcome or rotation collapse")
    # All legal explicit requests, without saving any puzzles to disk.
    legal = 0
    for n in range(2, 11):
        for m in range(1, min(n - 1, 4) + 1):
            p = generate_puzzle(718, n, m)
            _require(len(p["train"]) == n and len(p["test"]) == m, "counts ignored")
            legal += 1
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0, 2, 1)
    for role in ("masonry", "pin", "square"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]] for s in original}
        pair = bad["train"][0]
        f = _parse(pair["input"])
        out = np.rot90(pair["output"], f["turns"]).copy()
        if role == "masonry":
            point = (0, 0)
        elif role == "pin":
            point = (1, f["pins"][0][0])
        else:
            point = tuple(np.argwhere(out == f["mover"])[0])
        out[point] = (int(out[point]) + 1) % 10
        pair["output"] = np.rot90(out, -f["turns"]).copy()
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("output mutation accepted: " + role)
    # Whole-scene transformations and a bijective relabeling exercise the
    # input-only role reader without construction metadata.
    pair = original["test"][0]
    mapping = np.array([(3 * c + 7) % 10 for c in range(10)], dtype=np.int8)
    for code in range(8):
        _require(np.array_equal(solve(_orient(mapping[pair["input"]], code)),
                                 _orient(mapping[pair["output"]], code)), "equivariance failure")
    return {"seeds": 42, "pairs": pairs, "legal_counts": legal,
            "structures": len(structures), "shapes": len(shapes), "palettes": len(palettes),
            "sizes": sorted(sizes), "pin_counts": sorted(pin_counts), "bends": sorted(bends),
            "features": dict(features), "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
