"""ARC family: charged carriers activate gapped outlet valves.

Only NumPy and the Python standard library are used. No parent code or examples
are loaded. The parser and solver use the supplied input alone.
"""
from __future__ import annotations

from collections import deque
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(int, copy=False)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _components(mask):
    seen = set()
    groups = []
    h, w = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        start = (int(r), int(c))
        if start in seen:
            continue
        todo, group = [start], []
        seen.add(start)
        while todo:
            p = todo.pop()
            group.append(p)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (p[0] + dr, p[1] + dc)
                if (0 <= q[0] < h and 0 <= q[1] < w
                        and mask[q] and q not in seen):
                    seen.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _perimeter(f):
    t, b, l, r = (f[k] for k in ("t", "b", "l", "r"))
    return ([(t, c) for c in range(l, r + 1)]
            + [(b, c) for c in range(l, r + 1)]
            + [(y, l) for y in range(t + 1, b)]
            + [(y, r) for y in range(t + 1, b)])


def _parse_vertical(a, background):
    """Parse a candidate normalization in which carriers are vertical."""
    h, w = a.shape
    _require(h >= 5 and w >= 5, "canvas too small")
    _require(np.array_equal(a[0], a[-1]), "carrier endpoints differ")
    endpoint_colors = set(map(int, a[0])) - {background}
    _require(len(endpoint_colors) == 1, "carrier color ambiguous")
    carrier = next(iter(endpoint_colors))
    rails = tuple(map(int, np.flatnonzero(a[0] == carrier)))
    _require(rails and rails[0] > 0 and rails[-1] < w - 1, "edge carrier")
    other = set(map(int, np.unique(a))) - {background, carrier}
    _require(len(other) == 2, "need frame and bead colors")
    singleton = [c for c in other if np.count_nonzero(a == c) == 1]
    _require(len(singleton) == 1, "need exactly one uniquely colored bead")
    signal = singleton[0]
    frame_color = next(iter(other - {signal}))
    bead = tuple(map(int, np.argwhere(a == signal)[0]))
    frames = []
    for group in _components(a == frame_color):
        ys, xs = zip(*group)
        t, b, l, r = min(ys), max(ys), min(xs), max(xs)
        _require(b - t >= 2 and r - l >= 2 and (r - l) % 2 == 0,
                 "frame needs nonempty cavity and odd width")
        _require(1 <= t and b < h - 1, "valve touches carrier endpoint border")
        x = (l + r) // 2
        _require(x in rails, "valve not centered on a carrier")
        f = dict(t=t, b=b, l=l, r=r, x=x)
        border = set(_perimeter(f))
        gaps = [p for p in border if a[p] != frame_color]
        _require(len(gaps) == 1 and a[gaps[0]] == background, "not one empty gap")
        gap = gaps[0]
        _require(t < gap[0] < b and gap[1] in (l, r), "outlet must face across carriers")
        _require(set(group) == border - {gap}, "frame has clutter")
        _require(all(abs(other_x - x) >= (r - l) // 2 + 2
                     for other_x in rails if other_x != x), "carrier too close to valve")
        f.update(gap=gap, d=1 if gap[1] == r else -1)
        frames.append(f)
    frames.sort(key=lambda f: f["t"])
    _require(frames, "no outlet valves")
    _require(all(v["t"] >= u["b"] + 2 for u, v in zip(frames, frames[1:])),
             "valves must occupy separated bands")
    seed_frames = [i for i, f in enumerate(frames)
                   if f["t"] < bead[0] < f["b"] and f["l"] < bead[1] < f["r"]
                   and bead[1] != f["x"]]
    _require(len(seed_frames) == 1, "bead must be in one cavity, off the carrier")
    expected = np.full(a.shape, background, dtype=int)
    expected[:, rails] = carrier
    for f in frames:
        for p in _perimeter(f):
            if p != f["gap"]:
                expected[p] = frame_color
    expected[bead] = signal
    _require(np.array_equal(expected, a), "unexpected input content")
    return dict(grid=a, background=background, carrier=carrier, signal=signal,
                frame_color=frame_color, rails=rails, frames=frames,
                seed_rail=frames[seed_frames[0]]["x"], bead=bead)


def _analyze(input_grid):
    a = _grid(input_grid)
    background = int(a[0, 0])
    _require(all(a[p] == background for p in
                 ((0, -1), (-1, 0), (-1, -1))), "background corners disagree")
    _require(len(np.unique(a)) == 4, "input must have four visible roles")
    candidates = []
    for transpose in (False, True):
        try:
            info = _parse_vertical(a.T if transpose else a, background)
        except AssertionError:
            continue
        info["transpose"] = transpose
        candidates.append(info)
    _require(len(candidates) == 1, "carrier orientation or roles ambiguous")
    return candidates[0]


def _targets(info, f, bilateral=False):
    return [x for x in info["rails"] if
            (x < f["l"] if f["d"] < 0 else x > f["r"])
            or (bilateral and x != f["x"])]


def _activate(info, propagate=True, bilateral=False):
    """Breadth-first carrier charging; never inspect or modify output pixels."""
    active_rails = {info["seed_rail"]}
    todo = deque(active_rails)
    while todo:
        x = todo.popleft()
        for f in info["frames"]:
            if f["x"] != x:
                continue
            for target in _targets(info, f, bilateral):
                if propagate and target not in active_rails:
                    active_rails.add(target)
                    todo.append(target)
    return {i for i, f in enumerate(info["frames"]) if f["x"] in active_rails}


def _render(info, active, stamp=True, bilateral=False):
    a = info["grid"]
    out = a.copy()
    color = info["signal"]
    for i in sorted(active):
        f = info["frames"][i]
        cavity = out[f["t"] + 1:f["b"], f["l"] + 1:f["r"]]
        # Inheritance from B: preserve the carrier stripe inside a valve.
        cavity[cavity != info["carrier"]] = color
        y, gap_x = f["gap"]
        directions = (-1, 1) if bilateral else (f["d"],)
        for d in directions:
            start = f["l"] if d < 0 else f["r"]
            if d == f["d"]:
                out[y, start] = color
            # The bilateral ablation preserves the intact opposite wall.
            for x in range(start + d, a.shape[1] if d > 0 else -1, d):
                if a[y, x] == info["background"]:
                    out[y, x] = color
            if stamp:
                for x in info["rails"]:
                    if (d < 0 and x < f["l"]) or (d > 0 and x > f["r"]):
                        out[y - 1:y + 2, x - 1:x + 2] = color
                        out[y, x] = info["carrier"]
    return out.T.copy() if info["transpose"] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the least-closure relay result, deriving all roles from input."""
    info = _analyze(input_grid)
    return _render(info, _activate(info))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _sample_pair(rng, rail_count, assignments, directions, seed_index,
                 rotation, wide=False, diagnostic=False):
    count = len(assignments)
    widths = [int(rng.choice((3, 5))) for _ in assignments]
    widths[seed_index] = int(rng.choice((7, 9))) if wide else 5
    if wide:
        for i in range(count):
            if i != seed_index:
                widths[i] = int(rng.choice((3, 5, 7, 9)))
    elif diagnostic:
        # Both cavity area and a carrier-preserving fill are visibly taught.
        widths = [5] * count
    half = max(widths) // 2
    spacing = half + 2
    minimum_w = 2 * half + 1 + spacing * (rail_count - 1)
    _require(minimum_w <= 30, "infeasible valve/carrier capacity")
    w = int(rng.integers(minimum_w, 31))
    extras = rng.multinomial(w - minimum_w, np.full(rail_count + 1, 1 / (rail_count + 1)))
    rails = [half + int(extras[0])]
    for i in range(1, rail_count):
        rails.append(rails[-1] + spacing + int(extras[i]))

    heights = [3] * count
    if diagnostic:
        heights[seed_index] = 5
    gaps = list(map(int, rng.integers(1, 3, size=max(0, count - 1))))
    top_margin, bottom_margin = map(int, rng.integers(1, 3, size=2))
    budget = 30 - sum(heights) - sum(gaps) - top_margin - bottom_margin
    for i in map(int, rng.permutation(count)):
        extra = int(rng.integers(0, min(9 - heights[i], budget) + 1))
        heights[i] += extra
        budget -= extra
    bottom_margin += int(rng.integers(0, budget + 1))
    h = sum(heights) + sum(gaps) + top_margin + bottom_margin
    background, carrier, frame_color, signal = map(int, rng.choice(10, 4, replace=False))
    a = np.full((h, w), background, dtype=int)
    a[:, rails] = carrier
    order = list(map(int, rng.permutation(count)))
    frames, row = {}, top_margin
    for position, i in enumerate(order):
        x, fh, fw = rails[assignments[i]], heights[i], widths[i]
        f = dict(t=row, b=row + fh - 1, l=x - fw // 2, r=x + fw // 2,
                 x=x, d=int(directions[i]))
        offset = int(rng.integers(1, fh - 1))
        if diagnostic and i == seed_index:
            offset = 1  # Clearly off-center on the at-least-five-cell frame.
        f["gap"] = (row + offset, f["l"] if f["d"] < 0 else f["r"])
        for p in _perimeter(f):
            if p != f["gap"]:
                a[p] = frame_color
        frames[i] = f
        row += fh + (gaps[position] if position < count - 1 else 0)
    f = frames[seed_index]
    bead_y = int(rng.integers(f["t"] + 1, f["b"]))
    bead_x = int(rng.choice([x for x in range(f["l"] + 1, f["r"]) if x != f["x"]]))
    a[bead_y, bead_x] = signal

    # Generation uses its geometric construction and a round-based fixed point,
    # whereas solve reconstructs frames and uses a carrier queue.
    ordered = [frames[i] for i in range(count)]
    info = dict(grid=a, background=background, carrier=carrier,
                frame_color=frame_color, signal=signal, rails=tuple(rails),
                frames=ordered, seed_rail=f["x"], transpose=False)
    charged = {f["x"]}
    while True:
        more = {x for valve in ordered if valve["x"] in charged
                for x in _targets(info, valve)}
        if more <= charged:
            break
        charged |= more
    active = {i for i, valve in enumerate(ordered) if valve["x"] in charged}
    out = _render(info, active)
    return {"input": np.rot90(a, rotation).copy(),
            "output": np.rot90(out, rotation).copy()}


def _random_layout(rng, rail_count, count):
    assignments = list(map(int, rng.integers(0, rail_count, size=count)))
    directions = list(map(int, rng.choice((-1, 1), size=count)))
    # Sometimes deliberately exercise carrier-wide activation of several valves.
    if rail_count >= 2 and count >= 3 and rng.random() < 0.6:
        assignments[0] = int(rng.integers(0, rail_count - 1))
        assignments[1] = assignments[2] = assignments[0] + 1
        directions[0] = 1
        directions[1] = -1 if rng.random() < 0.65 else 1
    return assignments, directions


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    expand_bus = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    train = []

    # Witness 1: a remote carrier is contacted, but the frame behind the seed
    # stays inactive. This defeats directionless broadcasting and fill-all rules.
    n = int(rng.integers(3, 5))
    d = int(rng.choice((-1, 1)))
    source = int(rng.integers(1, n - 1))
    dormant = int(rng.integers(0, source)) if d > 0 else int(rng.integers(source + 1, n))
    train.append(_sample_pair(rng, n, [source, dormant],
                              [d, int(rng.choice((-1, 1)))], 0, rotation, diagnostic=True))

    # Witness 2: a contact on another carrier activates two spatially separated
    # valves there; one sends a return signal behind the initial seed.
    n = int(rng.integers(3, 5))
    d = int(rng.choice((-1, 1)))
    source = int(rng.integers(1, n - 1))
    target = int(rng.integers(source + 1, n)) if d > 0 else int(rng.integers(0, source))
    assignments = [source, target, target]
    directions = [d, -d, int(rng.choice((-1, 1)))]
    if rng.random() < 0.5:
        behind = int(rng.integers(0, source)) if d > 0 else int(rng.integers(source + 1, n))
        assignments.append(behind)
        directions.append(int(rng.choice((-1, 1))))
    train.append(_sample_pair(rng, n, assignments, directions, 0,
                              (rotation + 1) % 4, diagnostic=True))
    while len(train) < num_train:
        n, count = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        assignments, directions = _random_layout(rng, n, count)
        train.append(_sample_pair(rng, n, assignments, directions, 0, int(rng.integers(4))))
    rng.shuffle(train)
    puzzle = {"train": train, "test": []}
    seen = {_key(p["input"]) for p in train}
    _require(len(seen) == num_train, "unexpected repeated training input")
    for _ in range(num_test):
        for attempt in range(50):
            n = int(rng.integers(5, 7)) if expand_bus else int(rng.integers(2, 5))
            count = int(rng.integers(1, 5))
            assignments, directions = _random_layout(rng, n, count)
            pair = _sample_pair(rng, n, assignments, directions, 0,
                                int(rng.integers(4)), wide=not expand_bus)
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                puzzle["test"].append(pair)
                break
        else:
            raise RuntimeError("could not produce a distinct test input")
    return puzzle


def _closure_by_matrix(info):
    """Validator's rail-level transitive closure, separate from solve's queue."""
    rails = info["rails"]
    index = {x: i for i, x in enumerate(rails)}
    reach = np.eye(len(rails), dtype=bool)
    for f in info["frames"]:
        for x in rails:
            if (f["d"] < 0 and x < f["l"]) or (f["d"] > 0 and x > f["r"]):
                reach[index[f["x"]], index[x]] = True
    for k in range(len(rails)):
        reach |= reach[:, k:k + 1] & reach[k:k + 1, :]
    charged = {rails[i] for i in np.flatnonzero(reach[index[info["seed_rail"]]])}
    return {i for i, f in enumerate(info["frames"]) if f["x"] in charged}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    witnesses = dict(inactive=False, propagated=False, repeated=False,
                     omit_a=False, omit_b=False, one_pass=False, off_center=False)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "output shape changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _analyze(a)
            analyses[split].append(info)
            active = _closure_by_matrix(info)
            _require(active == _activate(info), "closure methods disagree")
            _require(np.array_equal(out, _render(info, active)), "wrong relay result")
            _require(np.array_equal(out, solve(a)), "input-only solve disagrees")
            _require(not np.array_equal(a, out), "identity example")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(a))), "unencoded color")
            canonical_out = out.T if info["transpose"] else out
            for i, f in enumerate(info["frames"]):
                for p in _perimeter(f):
                    if p != f["gap"]:
                        _require(canonical_out[p] == info["frame_color"], "outline was overwritten")
                stripe = canonical_out[f["t"] + 1:f["b"], f["x"]]
                _require(np.all(stripe == info["carrier"]), "carrier stripe not preserved")
                if i not in active:
                    sl = np.s_[f["t"]:f["b"] + 1, f["l"]:f["r"] + 1]
                    _require(np.array_equal(canonical_out[sl], info["grid"][sl]), "inactive valve changed")
                else:
                    for x in _targets(info, f):
                        y = f["gap"][0]
                        ring = canonical_out[y - 1:y + 2, x - 1:x + 2].copy()
                        _require(ring[1, 1] == info["carrier"], "contact center lost")
                        ring[1, 1] = info["signal"]
                        _require(np.all(ring == info["signal"]), "incomplete contact ring")
            if split == "train":
                initial = _activate(info, propagate=False)
                witnesses["inactive"] |= len(active) < len(info["frames"])
                witnesses["propagated"] |= bool(active - initial)
                witnesses["repeated"] |= len({f["x"] for f in info["frames"]}) < len(info["frames"])
                witnesses["omit_a"] |= not np.array_equal(out, _render(info, _activate(info, bilateral=True), bilateral=True))
                witnesses["omit_b"] |= not np.array_equal(out, _render(info, initial, stamp=False))
                witnesses["one_pass"] |= not np.array_equal(out, _render(info, initial))
                witnesses["off_center"] |= any(i in active and 2 * f["gap"][0] != f["t"] + f["b"]
                                                for i, f in enumerate(info["frames"]))
    _require(all(witnesses.values()), f"missing training evidence: {witnesses}")
    max_rails = max(len(a["rails"]) for a in analyses["train"])
    max_width = max(f["r"] - f["l"] + 1 for a in analyses["train"] for f in a["frames"])
    train_orientations = {a["transpose"] for a in analyses["train"]}
    for a in analyses["test"]:
        _require(a["transpose"] in train_orientations, "unshown carrier orientation")
        _require(len(a["rails"]) > max_rails or
                 max(f["r"] - f["l"] + 1 for f in a["frames"]) > max_width,
                 "test must recombine the rule at a new carrier count or valve width")


def self_test():
    """Bounded 42-seed structural/causal sweep and API boundary checks."""
    shapes, layouts, palettes, rails, frame_counts, sizes, directions = (set() for _ in range(7))
    split_stats = {s: dict(pairs=0, inactive=0, propagated=0, repeated=0) for s in ("train", "test")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                active = _activate(info)
                initial = _activate(info, propagate=False)
                shapes.add(pair["input"].shape)
                canonical = np.empty(info["grid"].shape, dtype=np.int8)
                colors = (info["background"], info["carrier"], info["frame_color"], info["signal"])
                for role, color in enumerate(colors):
                    canonical[info["grid"] == color] = role
                layouts.add(_key(canonical))
                palettes.add(colors)
                rails.add(len(info["rails"]))
                frame_counts.add(len(info["frames"]))
                sizes.update((f["b"] - f["t"] + 1, f["r"] - f["l"] + 1) for f in info["frames"])
                directions.update((info["transpose"], f["d"]) for f in info["frames"])
                stats = split_stats[split]
                stats["pairs"] += 1
                stats["inactive"] += len(active) < len(info["frames"])
                stats["propagated"] += bool(active - initial)
                stats["repeated"] += len({f["x"] for f in info["frames"]}) < len(info["frames"])
                # Geometric equivariance checks have value beyond re-running solve.
                _require(np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"])), "rotation failure")
                _require(np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"])), "reflection failure")
    _require(len(shapes) >= 60 and len(layouts) >= 180, "insufficient geometric variation")
    _require(rails == set(range(1, 7)) and frame_counts == set(range(1, 5)), "missing count variation")
    _require(len(sizes) >= 18 and len(directions) == 4 and len(palettes) >= 150, "insufficient role/shape variation")
    p, q = generate_puzzle(741, 7, 3), generate_puzzle(741, 7, 3)
    _require(all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ("input", "output")), "nondeterminism")
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(952 + a + b, a, b)
        _require(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
        validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (4.0, 1), (True, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    return dict(seeds=42, pairs=sum(s["pairs"] for s in split_stats.values()),
                distinct_shapes=len(shapes), normalized_layouts=len(layouts),
                role_palettes=len(palettes), carrier_counts=sorted(rails),
                valve_counts=sorted(frame_counts), valve_sizes=len(sizes),
                direction_roles=len(directions), split_stats=split_stats)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
