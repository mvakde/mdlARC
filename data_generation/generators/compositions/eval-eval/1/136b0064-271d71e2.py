"""Ink-powered courier: arrow ink is simultaneously program, fuel, and trail.

Only standard-library modules and NumPy are used. No parent code or data is
loaded. Every semantic/color convention is fixed or recoverable from one input.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import numpy as np


DIRECTIONS = ("U", "L", "D", "R")
DELTA = {"U": (-1, 0), "L": (0, -1), "D": (1, 0), "R": (0, 1)}
TO_UP = {"U": 0, "L": 3, "D": 2, "R": 1}


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _need(not isinstance(value, (bool, np.bool_)) and
          isinstance(value, (int, np.integer)), f"{name} must be an integer")
    value = int(value)
    _need(value >= low and (high is None or value <= high), f"invalid {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _need(np.all((0 <= a) & (a <= 9)), "invalid grid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _masks(side):
    up = np.zeros((side, side), dtype=bool)
    center = (side - 1) // 2
    up[0, center] = True
    up[1, center - 1:center + 2] = True
    up[2:, center] = True
    return {d: np.rot90(up, k).copy() for k, d in enumerate(DIRECTIONS)}


def _scan(mask, direction, rotated=True):
    """Occupied coordinates, in the travel-up raster order, not mask order."""
    ids = np.arange(mask.size).reshape(mask.shape)
    turn = TO_UP[direction] if rotated else 0
    view_ids = np.rot90(ids, turn)
    view_mask = np.rot90(mask, turn)
    return [divmod(int(i), mask.shape[1]) for i in view_ids[view_mask]]


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, cells = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in DELTA.values():
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    stack.append(q)
        components.append(cells)
    return components


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _need(np.all(a[0] == bg) and np.all(a[-1] == bg) and
          np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "background border required")
    candidates = []
    for color in np.unique(a):
        color = int(color)
        if color == bg:
            continue
        for cells in _components(a == color):
            ys, xs = zip(*cells)
            t, b, l, r = min(ys), max(ys), min(xs), max(xs)
            if b - t < 5 or r - l < 5:
                continue
            block = a[t:b + 1, l:r + 1]
            if not (np.all(block[0] == color) and np.all(block[-1] == color)
                    and np.all(block[:, 0] == color) and np.all(block[:, -1] == color)):
                continue
            # A completely spent shaft can itself make an extra full line.
            # Infer the regular wall lattice from its outer dimensions rather
            # than treating every full line as a structural wall.
            for pitch in (5, 6):
                if (b - t) % pitch or (r - l) % pitch:
                    continue
                rows = np.arange(0, b - t + 1, pitch)
                cols = np.arange(0, r - l + 1, pitch)
                if np.all(block[rows] == color) and np.all(block[:, cols] == color):
                    candidates.append((color, (t, b, l, r), rows, cols))
    _need(len(candidates) == 1, "unique arrow tablet required")
    frame, bounds, wall_rows, wall_cols = candidates[0]
    t, b, l, r = bounds
    side = int(wall_rows[1] - wall_rows[0] - 1)
    masks = _masks(side)
    tokens, flat_states = [], []
    expected = np.full_like(a, bg)
    expected[t + wall_rows[:, None], np.arange(l, r + 1)] = frame
    expected[np.arange(t, b + 1)[:, None], l + wall_cols] = frame
    for ci in range(len(wall_cols) - 1):
        for ri in range(len(wall_rows) - 1):
            y, x = t + int(wall_rows[ri]) + 1, l + int(wall_cols[ci]) + 1
            tile = a[y:y + side, x:x + side]
            mask = tile != bg
            directions = [d for d in DIRECTIONS if np.array_equal(mask, masks[d])]
            _need(len(directions) == 1, "unrecognized intact arrow silhouette")
            direction = directions[0]
            live_hues = set(map(int, tile[mask])) - {frame}
            _need(len(live_hues) <= 1, "one live hue per arrow required")
            color = next(iter(live_hues)) if live_hues else None
            coordinates = [(y + dy, x + dx) for dy, dx in _scan(mask, direction)]
            states = [int(a[p]) == frame for p in coordinates]
            flat_states.extend(states)
            tokens.append(dict(direction=direction, color=color, coordinates=coordinates,
                               top=y, left=x, row=ri, column=ci, mask=mask))
            expected[y:y + side, x:x + side] = tile
    spent = sum(flat_states)
    _need(flat_states == [True] * spent + [False] * (len(flat_states) - spent),
          "spent cells must form the ordered prefix")
    outside = (a == frame)
    outside[t:b + 1, l:r + 1] = False
    structures = _components(outside)
    singles = [v for v in structures if len(v) == 1]
    squares = []
    for cells in structures:
        ys, xs = zip(*cells)
        if len(cells) == 4 and max(ys) - min(ys) == max(xs) - min(xs) == 1:
            squares.append(cells)
    _need(len(structures) == 2 and len(singles) == len(squares) == 1,
          "one singleton courier and one square dock required")
    start, dock = next(iter(singles[0])), squares[0]
    for y, x in {start} | dock:
        _need(not (t - 1 <= y <= b + 1 and l - 1 <= x <= r + 1),
              "objects need a gap from the tablet")
        expected[y, x] = frame
    _need(all(max(abs(y - start[0]), abs(x - start[1])) > 1 for y, x in dock),
          "courier needs a gap from the dock")
    _need(np.array_equal(expected, a), "unexplained input pixels")
    info = dict(grid=a, background=bg, frame=frame, bounds=bounds, side=side,
                bands=len(wall_rows) - 1, lanes=len(wall_cols) - 1,
                tokens=tokens, start=start, dock=dock, spent=spent,
                capacity=len(flat_states))
    # Validate even the unexecuted suffix. Arrival does not hide bad geometry.
    _walk(info)
    return info


def _units(info, order="column", rotated=True, reuse_spent=False):
    tokens = info["tokens"]
    if order == "row":
        tokens = sorted(tokens, key=lambda v: (v["row"], v["column"]))
    units = []
    a, frame = info["grid"], info["frame"]
    for token in tokens:
        if rotated:
            positions = token["coordinates"]
        else:
            positions = [(token["top"] + y, token["left"] + x)
                         for y, x in _scan(token["mask"], token["direction"], False)]
        for source in positions:
            if int(a[source]) != frame:
                units.append((source, token["direction"], int(a[source])))
            elif reuse_spent and token["color"] is not None:
                units.append((source, token["direction"], token["color"]))
    return units


def _walk(info, order="column", straight=False):
    units = _units(info, order)
    p = info["start"]
    path, used = [p], {p}
    h, w = info["grid"].shape
    t, b, l, r = info["bounds"]
    heading = units[0][1] if units else None
    for _, direction, _ in units:
        dy, dx = DELTA[heading if straight else direction]
        p = p[0] + dy, p[1] + dx
        _need(1 <= p[0] < h - 1 and 1 <= p[1] < w - 1, "route leaves border")
        _need(not (t <= p[0] <= b and l <= p[1] <= r), "route hits tablet")
        _need(p not in used, "route revisits a pixel")
        used.add(p)
        path.append(p)
    return path


def _execute(info, order="column", rotated=True, straight=False, ignore_dock=False,
             reuse_spent=False):
    out = info["grid"].copy()
    p, steps = info["start"], 0
    units = _units(info, order, rotated, reuse_spent)
    heading = units[0][1] if units else None
    h, w = out.shape
    t, b, l, r = info["bounds"]
    used = {p}
    for source, direction, color in units:
        dy, dx = DELTA[heading if straight else direction]
        q = p[0] + dy, p[1] + dx
        _need(1 <= q[0] < h - 1 and 1 <= q[1] < w - 1 and
              not (t <= q[0] <= b and l <= q[1] <= r) and q not in used,
              "alternative execution has invalid geometry")
        out[source] = info["frame"]
        if p not in info["dock"]:
            out[p] = color
        out[q] = info["frame"]
        used.add(q)
        p, steps = q, steps + 1
        if p in info["dock"] and not ignore_dock:
            break
    for cell in info["dock"]:
        out[cell] = info["frame"]
    return out, steps, p


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode and execute a valid family member using only this input."""
    return _execute(_parse(input_grid))[0]


def _opposite(direction):
    return DIRECTIONS[(DIRECTIONS.index(direction) + 2) % 4]


def _random_route(rng, allowed, start, lengths, required):
    """Bounded randomized backtracking over whole straight instructions."""
    path, directions = [start], []
    nodes = [0]

    def visit(index, used):
        nodes[0] += 1
        if nodes[0] > 180:
            return False
        if index == len(lengths):
            return required <= set(directions)
        if len(required - set(directions)) > len(lengths) - index:
            return False
        p = path[-1]
        for direction in rng.permutation(DIRECTIONS):
            direction = str(direction)
            if directions and direction == _opposite(directions[-1]):
                continue
            dy, dx = DELTA[direction]
            segment = [(p[0] + dy * j, p[1] + dx * j)
                       for j in range(1, lengths[index] + 1)]
            if any(q not in allowed or q in used for q in segment):
                continue
            directions.append(direction)
            path.extend(segment)
            if visit(index + 1, used | set(segment)):
                return True
            del path[-len(segment):]
            directions.pop()
        return False

    return (directions, path) if visit(0, {start}) else None


def _dock_options(point, allowed, tablet, start):
    t, b, l, r = tablet
    options = []
    for dy in (0, -1):
        for dx in (0, -1):
            y, x = point[0] + dy, point[1] + dx
            cells = {(y + j, x + k) for j in range(2) for k in range(2)}
            if (cells <= allowed and all(not (t - 1 <= q[0] <= b + 1 and
                                             l - 1 <= q[1] <= r + 1) for q in cells)
                    and all(max(abs(q[0] - start[0]), abs(q[1] - start[1])) > 1
                            for q in cells)):
                options.append(cells)
    return options


def _make_pair(rng, parameters, split, lesson=None, required=frozenset(), monochrome=False):
    side = parameters["side"]
    ink_count = side + 2
    for attempt in range(320):
        if split == "test":
            bands, lanes = ((3, 2) if rng.random() < .5 else (2, 3))
        elif lesson is not None:
            bands, lanes = 2, 2
        else:
            bands, lanes = ((2, 2), (3, 1), (1, 3), (2, 1), (1, 2))[int(rng.integers(5))]
        n = bands * lanes
        ph, pw = bands * (side + 1) + 1, lanes * (side + 1) + 1
        # Random asymmetric layouts, with a generous deterministic late budget.
        H = 30 if attempt >= 160 else int(rng.integers(max(19, ph + 4), 31))
        W = 30 if attempt >= 160 else int(rng.integers(max(19, pw + 4), 31))
        my = int(rng.integers(1, min(4, H - ph - 1)))
        mx = int(rng.integers(1, min(4, W - pw - 1)))
        top = my if rng.random() < .5 else H - my - ph
        left = mx if rng.random() < .5 else W - mx - pw
        tablet = (top, top + ph - 1, left, left + pw - 1)
        t, b, l, r = tablet
        allowed = {(y, x) for y in range(1, H - 1) for x in range(1, W - 1)
                   if not (t <= y <= b and l <= x <= r)}
        starts = sorted(q for q in allowed if not
                        (t - 1 <= q[0] <= b + 1 and l - 1 <= q[1] <= r + 1))
        start = starts[int(rng.integers(len(starts)))]
        if split == "test":
            regime = str(rng.choice(("dock", "fuel", "exact"), p=(.4, .4, .2)))
        else:
            regime = lesson or str(rng.choice(("dock", "fuel", "exact", "zero"),
                                             p=(.36, .36, .20, .08)))
        if regime == "zero":
            spent = n * ink_count
        elif split == "test":
            spent = int(rng.integers(0, ink_count - 1))
        elif lesson:
            spent = int(rng.integers(1, ink_count - 1))
        else:
            spent = int(rng.integers(0, max(1, (n - 1) * ink_count)))
        lengths = [max(0, ink_count - max(0, spent - i * ink_count)) for i in range(n)]
        active_indices = [i for i, length in enumerate(lengths) if length]
        active_lengths = [lengths[i] for i in active_indices]
        found = _random_route(rng, allowed, start, active_lengths, set(required))
        if found is None:
            continue
        active_dirs, path = found
        if lesson and len(set(active_dirs)) < 2:
            continue
        directions = [str(rng.choice(DIRECTIONS)) for _ in range(n)]
        for i, d in zip(active_indices, active_dirs):
            directions[i] = d
        cumulative = np.cumsum(active_lengths).tolist()
        unit_dirs = [d for d, count in zip(active_dirs, active_lengths) for _ in range(count)]
        candidates = []
        if regime == "dock":
            # Arrival after a turn, within a command, with a nonempty suffix.
            for j in range(2, len(path) - 2):
                if j in cumulative or len(set(unit_dirs[:j])) < 2:
                    continue
                if not set(required) <= set(unit_dirs[:j]):
                    continue
                for cells in _dock_options(path[j], allowed, tablet, start):
                    if not cells & set(path[:j]):
                        candidates.append((cells, j))
        elif regime == "exact":
            for cells in _dock_options(path[-1], allowed, tablet, start):
                if not cells & set(path[:-1]):
                    candidates.append((cells, len(path) - 1))
        elif regime == "fuel":
            if not active_dirs or len(set(active_dirs)) < 2:
                continue
            dy, dx = DELTA[active_dirs[-1]]
            for distance in range(2, 7):
                point = path[-1][0] + dy * distance, path[-1][1] + dx * distance
                for cells in _dock_options(point, allowed, tablet, start):
                    if not cells & set(path):
                        candidates.append((cells, None))
        else:
            for point in starts[::max(1, len(starts) // 20)]:
                for cells in _dock_options(point, allowed, tablet, start):
                    candidates.append((cells, None))
        if not candidates:
            continue
        dock, _ = candidates[int(rng.integers(len(candidates)))]
        palette = parameters["palette"].copy()
        if parameters["vary_roles"]:
            rng.shuffle(palette)
        bg, frame = palette[:2]
        live_palette = palette[2:2 + parameters["hues"]]
        colors = ([live_palette[0]] * n if monochrome else
                  [int(rng.choice(live_palette)) for _ in range(n)])
        if lesson and not monochrome:
            colors[0], colors[1] = live_palette[0], live_palette[1]
        a = np.full((H, W), bg, dtype=np.int8)
        for y in range(top, top + ph, side + 1):
            a[y, left:left + pw] = frame
        for x in range(left, left + pw, side + 1):
            a[top:top + ph, x] = frame
        masks = _masks(side)
        counter = 0
        for i, (direction, color) in enumerate(zip(directions, colors)):
            ci, ri = divmod(i, bands)
            y, x = top + ri * (side + 1) + 1, left + ci * (side + 1) + 1
            for dy, dx in _scan(masks[direction], direction):
                a[y + dy, x + dx] = frame if counter < spent else color
                counter += 1
        a[start] = frame
        for q in dock:
            a[q] = frame
        info = _parse(a)
        out = _execute(info)[0]
        pair = dict(input=a, output=out)
        if lesson == "dock" and not _reject_alternative([pair], rotated=False):
            continue
        return pair
    raise RuntimeError("bounded courier construction failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    _need(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    parameters = dict(side=int(rng.integers(4, 6)),
                      palette=list(map(int, rng.permutation(10))),
                      hues=int(rng.integers(2, 6)), vary_roles=bool(rng.integers(2)))
    d = int(rng.integers(4))
    first_dirs = {DIRECTIONS[d], DIRECTIONS[(d + 1) % 4]}
    second_dirs = {_opposite(v) for v in first_dirs}
    lessons = [("dock", first_dirs, True), ("fuel", second_dirs, False)]
    if rng.random() < .5:
        lessons.reverse()
    puzzle = dict(train=[], test=[])
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            lesson, required, mono = lessons[i] if split == "train" and i < 2 else (None, set(), False)
            # Finite uniqueness/teaching retries. No latent data enter solve.
            for _ in range(32):
                pair = _make_pair(rng, parameters, split, lesson, required, mono)
                if _key(pair["input"]) in seen:
                    continue
                if split == "train" and i == 1:
                    teaching = puzzle["train"] + [pair]
                    if not all(_reject_alternative(teaching, **kwargs) for kwargs in
                               (dict(order="row"), dict(rotated=False))):
                        continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded uniqueness/teaching construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _reject_alternative(pairs, **kwargs):
    for pair in pairs:
        try:
            alternative = _execute(_parse(pair["input"]), **kwargs)[0]
        except ValueError:
            return True
        if not np.array_equal(alternative, pair["output"]):
            return True
    return False


def _features(pair):
    info = _parse(pair["input"])
    out, moves, endpoint = _execute(info)
    units = _units(info)
    reached = endpoint in info["dock"]
    stop = ("zero" if not units else "exact" if reached and moves == len(units)
            else "dock" if reached else "fuel")
    return dict(shape=info["grid"].shape, side=info["side"], bands=info["bands"],
                lanes=info["lanes"], count=len(info["tokens"]), capacity=info["capacity"],
                spent=info["spent"], remaining=len(units), moves=moves, stop=stop,
                directions=tuple(v["direction"] for v in info["tokens"]),
                active_directions=tuple(d for _, d, _ in units[:moves]),
                colors=tuple(v["color"] for v in info["tokens"]),
                background=info["background"], frame=info["frame"], bounds=info["bounds"],
                start=info["start"], endpoint=endpoint)


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _need(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and
          len(puzzle["train"]) > len(puzzle["test"]), "invalid puzzle counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _need(a.shape == out.shape, "canvas must be preserved")
            _need(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _parse(a)
            expected, moves, endpoint = _execute(info)
            _need(np.array_equal(out, expected), "wrong courier transformation")
            t, b, l, r = info["bounds"]
            _need(np.array_equal(a[t:b + 1, l:r + 1] != info["background"],
                                 out[t:b + 1, l:r + 1] != info["background"]),
                  "arrow silhouettes and walls must be preserved")
            _need(all(out[p] == info["frame"] for p in info["dock"]), "dock must remain fixed")
            units = _units(info)
            for j, (source, _, color) in enumerate(units):
                _need(out[source] == (info["frame"] if j < moves else color),
                      "exact sparse counter advancement required")
            exterior = out.copy()
            exterior[t:b + 1, l:r + 1] = info["background"]
            live = {color for _, _, color in units}
            _need(sum(int(np.sum(exterior == c)) for c in live) == moves,
                  "one trail pixel must correspond to each consumed ink pixel")
            for color in live:
                _need(np.sum(a == color) == np.sum(out == color), "live hue must be conserved")
            _need(out[endpoint] == info["frame"], "terminal courier missing")
            features[split].append(_features(pair))
    all_features = features["train"] + features["test"]
    _need(len({f["side"] for f in all_features}) == 1, "arrow size changes within puzzle")
    training = features["train"]
    _need({"dock", "fuel"} <= {f["stop"] for f in training}, "both stops must be taught")
    _need(set(d for f in training for d in f["active_directions"]) == set(DIRECTIONS),
          "all four active directions must be taught")
    for stop in ("dock", "fuel"):
        _need(any(f["stop"] == stop and len(set(f["active_directions"])) >= 2 for f in training),
              "both stops need a turning demonstration")
    _need(any(f["stop"] == "dock" and f["remaining"] - f["moves"] >= 2 and
              (f["spent"] + f["moves"]) % (f["side"] + 2) != 0 for f in training),
          "arrival must expose partial-command stopping")
    _need(min(f["remaining"] for f in features["test"]) >
          max(f["remaining"] for f in training), "missing longer remaining-program shift")
    for kwargs in (dict(order="row"), dict(rotated=False), dict(straight=True),
                   dict(ignore_dock=True), dict(reuse_spent=True)):
        _need(_reject_alternative(puzzle["train"], **kwargs), f"ambiguous alternative {kwargs}")


def _puzzle_key(puzzle):
    return tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed audit plus API, cellwise, and corruption regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_features(p) for puzzle in puzzles for split in ("train", "test") for p in puzzle[split]]
    _need(len({_puzzle_key(p) for p in puzzles}) == 42, "seeds lack diversity")
    _need(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "nondeterministic puzzle")
    _need({f["side"] for f in records} == {4, 5}, "arrow-size coverage missing")
    _need(len({f["shape"] for f in records}) >= 35, "canvas variation collapsed")
    _need({f["stop"] for f in records} == {"dock", "fuel", "exact", "zero"}, "stop coverage missing")
    _need({f["count"] for f in records} >= {2, 3, 4, 6}, "program-count coverage missing")
    _need({(f["bands"], f["lanes"]) for f in records} >=
          {(2, 2), (3, 2), (2, 3), (1, 3), (3, 1)}, "lattice shape coverage missing")
    _need(any(f["spent"] == 0 for f in records), "fresh tablet missing")
    _need(any(0 < f["spent"] < f["capacity"] and f["spent"] >= f["side"] + 2
              for f in records), "fully spent leading glyph missing")
    _need(len({(f["background"], f["frame"]) for f in records}) >= 30, "role diversity missing")
    _need({(d, f["stop"]) for f in records for d in set(f["active_directions"])} >=
          {(d, stop) for d in DIRECTIONS for stop in ("dock", "fuel")}, "direction/stop coupling")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(101 + nt * 4 + nq, nt, nq)
            _need(len(p["train"]) == nt and len(p["test"]) == nq, "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Grid-derived unit transfer oracle: compare each departure to its own
    # source, independently of final-state rendering and aggregate counts.
    for puzzle in puzzles:
        for pair in puzzle["train"] + puzzle["test"]:
            info = _parse(pair["input"])
            path = _walk(info)
            units = _units(info)
            arrival = next((j for j, p in enumerate(path) if p in info["dock"]), len(units))
            for j, (source, _, color) in enumerate(units[:arrival]):
                _need(pair["output"][source] == info["frame"], "fuel tick regression")
                _need(pair["output"][path[j]] == color, "departure color regression")
    for corruption in ("trail", "tablet"):
        changed = deepcopy(puzzles[0])
        pair = changed["train"][0]
        info = _parse(pair["input"])
        q = info["start"] if corruption == "trail" else _units(info)[0][0]
        pair["output"][q] = info["background"]
        try:
            validate_generated_puzzle(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("semantic corruption accepted")
    # Small direct numerical example, not produced by the random constructor:
    # side-4 U arrow, first two pixels spent, four units left, two paid steps.
    mask = _masks(4)["U"]
    scan = _scan(mask, "U")
    _need(scan == [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)], "canonical scan regression")
    a = np.zeros((14, 12), dtype=np.int8)
    a[7:13, 1:7] = 8
    a[8:12, 2:6] = 0
    for j, (y, x) in enumerate(scan):
        a[8 + y, 2 + x] = 8 if j < 2 else 3
    a[5, 9] = 8
    a[2:4, 8:10] = 8
    wanted = a.copy()
    wanted[9, 3:5] = 8
    wanted[5, 9] = wanted[4, 9] = 3
    _need(np.array_equal(solve(a), wanted), "hand-authored partial arrival regression")
    return dict(seeds=42, pairs=len(records), shapes=len({f["shape"] for f in records}),
                sides=sorted({f["side"] for f in records}),
                stops=dict(Counter(f["stop"] for f in records)),
                counts=sorted({f["count"] for f in records}),
                role_assignments=len({(f["background"], f["frame"]) for f in records}))
