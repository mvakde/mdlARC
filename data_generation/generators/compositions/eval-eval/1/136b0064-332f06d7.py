"""A collision-edited glyph program driving an atomic square paintbrush.

Only standard-library modules and NumPy are used.  The public solver recovers
all roles, commands, and geometry from its single input grid.
"""

from collections import deque
import json
from numbers import Integral

import numpy as np


_DIRECTIONS = ((0, 1), (-1, 0), (0, -1), (1, 0))  # right, up, left, down
_RIGHT = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=bool)
_ARROWS = tuple(np.rot90(_RIGHT, j).copy() for j in range(4))
_HOST_SEEDS = (0, 766167826, 658888216, 306222137, 164635792, 569065092,
               799317503, 791492548, 24905961, 807416973, 554940201,
               260154565, 987641346, 180855451, 358813640, 225607480,
               887902733, 728650155, 384051042, 76945791, 517147489,
               216811863, 100718324, 926659417, 671724744, 394102297,
               190192169, 606805254, 359522125, 285966875, 461614852,
               874325993, 806136147, 483099333, 558299845, 643939591,
               887785640, 694280355, 847565673, 220660326, 362375401,
               459207511)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _square(mask):
    cells = np.argwhere(mask)
    _require(len(cells) > 0, "missing brush")
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    k = int(hi[0] - lo[0])
    _require(k == hi[1] - lo[1] and len(cells) == k * k,
             "brush must be one solid square")
    return tuple(map(int, lo)), k


def _glyph(direction, distance):
    mask = np.zeros((5, 5), dtype=bool)
    mask[:3, :3] = _ARROWS[_DIRECTIONS.index(tuple(direction))]
    mask[4, :distance] = True
    return mask


def _decode(block, background):
    occupied = block != background
    values = np.unique(block[occupied])
    _require(len(values) == 1, "glyph must have one ink color")
    candidates = [(direction, distance)
                  for direction in _DIRECTIONS for distance in range(1, 6)
                  if np.array_equal(occupied, _glyph(direction, distance))]
    _require(len(candidates) == 1, "invalid arrow or unary distance bar")
    direction, distance = candidates[0]
    return direction, distance, int(values[0])


def _read(input_grid):
    a = _grid(input_grid)
    # The divider color occurs exclusively in a single complete row/column.
    stripes = []
    for color in map(int, np.unique(a)):
        positions = np.argwhere(a == color)
        rs, cs = np.unique(positions[:, 0]), np.unique(positions[:, 1])
        if len(cs) == 1 and len(rs) == a.shape[0]:
            stripes.append(("left", int(cs[0]), color))
        if len(rs) == 1 and len(cs) == a.shape[1]:
            stripes.append(("above", int(rs[0]), color))
    _require(len(stripes) == 1, "missing or ambiguous full-length divider")
    placement, divider, separator = stripes[0]
    background = int(a[0, 0])
    _require(background != separator and divider >= 13,
             "invalid panel margin or divider")
    if placement == "left":
        panel, arena = a[:, :divider], a[:, divider + 1:]
    else:
        panel, arena = a[:divider, :], a[divider + 1:, :]
    _require(min(arena.shape) >= 3, "empty arena")
    ink = np.argwhere(panel != background)
    _require(len(ink) > 0 and np.all(ink >= 1), "invalid panel outer margin")
    bands = 1 + int((ink[:, 0].max() - 1) // 6)
    lanes = 1 + int((ink[:, 1].max() - 1) // 6)
    _require(2 <= bands <= 4 and 2 <= lanes <= 4, "unsupported slot rectangle")
    expected_span = 6 * (lanes if placement == "left" else bands) + 1
    _require(divider == expected_span, "divider does not bound the slot rectangle")
    _require(6 * bands + 1 <= panel.shape[0] and
             6 * lanes + 1 <= panel.shape[1], "missing panel margin")
    commands, slots = [], []
    expected_panel = np.full(panel.shape, background, dtype=np.int8)
    for lane in range(lanes):
        for band in range(bands):
            r, c = 1 + 6 * band, 1 + 6 * lane
            block = panel[r:r + 5, c:c + 5]
            direction, distance, color = _decode(block, background)
            expected_panel[r:r + 5, c:c + 5][_glyph(direction, distance)] = color
            commands.append((direction, distance, color))
            slots.append((band, lane))
    _require(np.array_equal(panel, expected_panel), "noise in instruction panel")
    wall = int(arena[0, 0])
    _require(wall not in (background, separator), "invalid wall color")
    _require(np.all(arena[0] == wall) and np.all(arena[-1] == wall) and
             np.all(arena[:, 0] == wall) and np.all(arena[:, -1] == wall),
             "arena must have a closed wall frame")
    others = set(map(int, np.unique(arena))) - {wall, background}
    _require(len(others) == 1, "arena must contain one brush role")
    marker = others.pop()
    _require(len({background, wall, separator, marker}) == 4, "roles must be distinct")
    _require(all(color not in (background, wall, separator, marker)
                 for _, _, color in commands), "ink conflicts with a structural role")
    start, k = _square(arena == marker)
    _require(1 <= k <= 4, "brush side must be 1..4")
    floor = arena != wall
    _require(_connected(floor), "floor must be connected")
    return {"arena": arena.copy(), "floor": floor, "background": background,
            "wall": wall, "separator": separator, "marker": marker,
            "start": start, "k": k, "commands": commands, "slots": slots,
            "bands": bands, "lanes": lanes, "placement": placement}


def _connected(mask):
    points = np.argwhere(mask)
    if not len(points):
        return False
    start = tuple(map(int, points[0]))
    seen, pending = {start}, deque([start])
    while pending:
        r, c = pending.popleft()
        for dr, dc in _DIRECTIONS:
            q = r + dr, c + dc
            if (0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1]
                    and mask[q] and q not in seen):
                seen.add(q)
                pending.append(q)
    return len(seen) == int(mask.sum())


def _fits(floor, position, k):
    r, c = position
    return (0 <= r and 0 <= c and r + k <= floor.shape[0]
            and c + k <= floor.shape[1]
            and bool(floor[r:r + k, c:c + k].all()))


def _attempt(floor, position, k, direction, distance):
    """Return all legal prefix placements, but do not commit them."""
    r, c = position
    dr, dc = direction
    prefix = [position]
    for j in range(1, distance + 1):
        q = r + dr * j, c + dc * j
        if not _fits(floor, q, k):
            return False, prefix
        prefix.append(q)
    return True, prefix


def _execute(info, variant="atomic"):
    """Input-derived execution; variants exist only for shortcut checks."""
    out = info["arena"].copy()
    out[out == info["marker"]] = info["background"]
    floor, k, position = info["floor"], info["k"], info["start"]
    written = np.zeros(out.shape, dtype=bool)
    commands = info["commands"]
    if variant == "row_major":
        commands = [command for _, command in sorted(zip(info["slots"], commands))]
    events = []
    for direction, distance, color in commands:
        before = position
        success, prefix = _attempt(floor, position, k, direction, distance)
        dr, dc = direction
        endpoint = position[0] + dr * distance, position[1] + dc * distance
        if variant == "point":
            success, point_prefix = _attempt(floor, position, 1, direction, distance)
            if success:
                prefix = point_prefix
        elif variant == "endpoint_only":
            success = _fits(floor, endpoint, k)
            if success:
                prefix = [(position[0] + dr * j, position[1] + dc * j)
                          for j in range(distance + 1)]
        elif variant == "clip_prefix" and not success:
            success = len(prefix) > 1
        events.append({"before": before, "direction": direction, "distance": distance,
                       "color": color, "success": success, "prefix": len(prefix) - 1})
        if not success:
            if variant == "stop_program":
                break
            continue
        paint_positions = [prefix[-1]] if variant == "endpoint_paint" else prefix
        for r, c in paint_positions:
            # Clipping only makes the point/endpoint shortcut renderable even
            # when its erroneous footprint extends out of bounds or into walls.
            r0, r1 = max(0, r), min(out.shape[0], r + k)
            c0, c1 = max(0, c), min(out.shape[1], c + k)
            if r1 <= r0 or c1 <= c0:
                continue
            region = out[r0:r1, c0:c1]
            if variant == "first_paint":
                region[~written[r0:r1, c0:c1]] = color
            else:
                region[:] = color
            written[r0:r1, c0:c1] = True
        position = prefix[-1]
    r, c = position
    r0, r1 = max(0, r), min(out.shape[0], r + k)
    c0, c1 = max(0, c), min(out.shape[1], c + k)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = info["marker"]
    return out, events, position


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode and execute solely from this input; never inspect training or RNG."""
    return _execute(_read(input_grid))[0]


def _compose(floor, start, k, commands, roles, placement, lanes, bands):
    background, wall, separator, marker = roles
    _require(len(commands) == lanes * bands, "slot count mismatch")
    floor = floor.copy()
    target_shape = list(floor.shape)
    if placement == "left":
        target_shape[0] = max(target_shape[0], 6 * bands + 1)
    else:
        target_shape[1] = max(target_shape[1], 6 * lanes + 1)
    floor = np.pad(floor, [(0, target_shape[i] - floor.shape[i]) for i in range(2)])
    arena = np.full(floor.shape, wall, dtype=np.int8)
    arena[floor] = background
    r, c = start
    _require(_fits(floor, start, k), "start overlaps a wall")
    arena[r:r + k, c:c + k] = marker
    ph, pw = 6 * bands + 1, 6 * lanes + 1
    if placement == "left":
        inp = np.full((arena.shape[0], pw + 1 + arena.shape[1]), background, dtype=np.int8)
        inp[:, pw] = separator
        inp[:, pw + 1:] = arena
    else:
        inp = np.full((ph + 1 + arena.shape[0], arena.shape[1]), background, dtype=np.int8)
        inp[ph] = separator
        inp[ph + 1:] = arena
    for j, (direction, distance, color) in enumerate(commands):
        lane, band = divmod(j, bands)
        r, c = 1 + 6 * band, 1 + 6 * lane
        inp[r:r + 5, c:c + 5][_glyph(direction, distance)] = color
    return {"input": inp, "output": solve(inp)}


def _calibration(rng):
    """One controlled two-cell throat; explicit destinations are tested below."""
    h, w = map(int, rng.integers(13, 17, size=2))
    floor = np.ones((h, w), dtype=bool)
    floor[0] = floor[-1] = False
    floor[:, 0] = floor[:, -1] = False
    floor[3, 7] = False
    # Extra terrain is outside every calibration stroke and its relevant
    # shortcut strokes. These are wall islands, not independent objects.
    for _ in range(int(rng.integers(0, 4))):
        previous_floor = floor.copy()
        r, c = int(rng.integers(9, h - 1)), int(rng.integers(9, w - 1))
        floor[r:min(h - 1, r + int(rng.integers(1, 3))),
              c:min(w - 1, c + int(rng.integers(1, 3)))] = False
        if not _connected(floor):
            floor = previous_floor
    program = [((0, 1), 4), ((1, 0), 3), ((0, -1), 3), ((-1, 0), 2)]
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    def transform(a):
        a = np.rot90(a, turns)
        return np.fliplr(a).copy() if reflect else a.copy()
    directions = []
    for (dr, dc), distance in program:
        for _ in range(turns):
            dr, dc = -dc, dr
        if reflect:
            dc = -dc
        directions.append(((dr, dc), distance))
    result = []
    for k in (2, 3):
        marker = np.zeros(floor.shape, dtype=bool)
        marker[1:1 + k, 2:2 + k] = True
        start, _ = _square(transform(marker))
        result.append((transform(floor), start, k, directions))
    return result


def _terrain(rng, h, w, k):
    """Sample open rooms with islands or alternating wall spurs and gaps."""
    for attempt in range(24):
        floor = np.ones((h, w), dtype=bool)
        floor[0] = floor[-1] = False
        floor[:, 0] = floor[:, -1] = False
        style = int(rng.integers(3))
        if style != 0:
            transpose = bool(rng.integers(2))
            work = floor.T if transpose else floor
            a, b = work.shape
            cuts = list(range(3, b - 3, int(rng.integers(3, 6))))
            for i, col in enumerate(cuts):
                gap = int(rng.integers(1, min(a - 3, k + 4) + 1))
                if i % 2:
                    work[gap + 1:, col:col + int(rng.integers(1, 3))] = False
                else:
                    work[:a - 1 - gap, col:col + int(rng.integers(1, 3))] = False
        for _ in range(int(rng.integers(1, 7 if style == 0 else 4))):
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            rh, cw = int(rng.integers(1, min(5, h - r - 1) + 1)), int(rng.integers(1, min(5, w - c - 1) + 1))
            floor[r:r + rh, c:c + cw] = False
        valid = np.lib.stride_tricks.sliding_window_view(floor, (k, k)).all(axis=(-1, -2))
        if _connected(floor) and int(valid.sum()) >= 8:
            return floor
    # Finite fallback is another legal arena, not a different transformation.
    floor = np.ones((h, w), dtype=bool)
    floor[0] = floor[-1] = False
    floor[:, 0] = floor[:, -1] = False
    floor[h // 2, w // 2] = False
    return floor


def _random_program(rng, floor, k, length, inks, split):
    valid = np.lib.stride_tricks.sliding_window_view(floor, (k, k)).all(axis=(-1, -2))
    choices = []
    for p in map(tuple, np.argwhere(valid)):
        if sum(_fits(floor, (p[0] + dr, p[1] + dc), k) for dr, dc in _DIRECTIONS) >= 2:
            choices.append(tuple(map(int, p)))
    if not choices:
        return None
    start = choices[int(rng.integers(len(choices)))]
    position, program, previous = start, [], None
    palette = list(map(int, rng.choice(inks, int(rng.integers(2, 7)), replace=False)))
    rejection_count = int(rng.integers(1 if split == "train" else 2,
                                       min(length - 2, max(2, length // 3)) + 1))
    rejected_indices = set(map(int, rng.choice(length, rejection_count, replace=False)))
    for index in range(length):
        options = []
        for direction in _DIRECTIONS:
            for distance in range(1, 6):
                success, prefix = _attempt(floor, position, k, direction, distance)
                options.append((direction, distance, success, prefix))
        want_failure = index in rejected_indices
        preferred = [t for t in options if t[2] != want_failure]
        if not preferred:
            preferred = options
        if want_failure:
            partial = [t for t in preferred if 0 < len(t[3]) - 1 < t[1]]
            if partial:
                preferred = partial
        else:
            long_turns = [t for t in preferred if t[1] >= 2 and t[0] != previous]
            if long_turns:
                preferred = long_turns
        direction, distance, success, prefix = preferred[int(rng.integers(len(preferred)))]
        color_options = [c for c in palette if split == "test" or not program or c != program[-1][2]]
        color = int(rng.choice(color_options))
        program.append((direction, distance, color))
        if success:
            position = prefix[-1]
        previous = direction
    if split == "test":
        # Repeated adjacent colors are legal; geometry still defines commands.
        j = int(rng.integers(1, length))
        program[j] = (program[j][0], program[j][1], program[j - 1][2])
    return start, program


def _random_pair(rng, k, roles, split, used):
    layouts = (("left", 2, 2), ("left", 2, 3), ("left", 3, 2),
               ("above", 2, 2), ("above", 3, 2), ("above", 2, 3)) if split == "train" else (
                   ("left", 2, 4), ("left", 3, 3), ("left", 3, 4),
                   ("above", 4, 2), ("above", 3, 3), ("above", 4, 3))
    inks = [c for c in range(10) if c not in roles]
    for attempt in range(64):
        placement, lanes, bands = layouts[int(rng.integers(len(layouts)))]
        if placement == "left":
            h = int(rng.integers(max(6 * bands + 1, k + 5), 31))
            w = int(rng.integers(k + 5, 30 - (6 * lanes + 1)))
        else:
            h = int(rng.integers(k + 5, 30 - (6 * bands + 1)))
            w = int(rng.integers(max(6 * lanes + 1, k + 5), 31))
        floor = _terrain(rng, h, w, k)
        sampled = _random_program(rng, floor, k, lanes * bands, inks, split)
        if sampled is None:
            continue
        start, commands = sampled
        pair = _compose(floor, start, k, commands, roles, placement, lanes, bands)
        info = _read(pair["input"])
        _, events, _ = _execute(info)
        if (sum(e["success"] for e in events) < 2
                or not any(not e["success"] and e["prefix"] > 0 for e in events)):
            continue
        signature = _key(pair["input"])
        if signature not in used:
            used.add(signature)
            return pair
    raise AssertionError("bounded arena/program construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    inks = list(map(int, rng.permutation([c for c in range(10) if c not in roles])))
    calibration = _calibration(rng)
    placements = ("left", "above") if bool(rng.integers(2)) else ("above", "left")
    puzzle, used = {"train": [], "test": []}, set()
    for j, ((floor, start, k, program), placement) in enumerate(zip(calibration, placements)):
        colors = [inks[(i + j + 1) % len(inks)] for i in range(4)]
        commands = [(direction, distance, color) for (direction, distance), color in zip(program, colors)]
        pair = _compose(floor, start, k, commands, roles, placement, 2, 2)
        puzzle["train"].append(pair)
        used.add(_key(pair["input"]))
    regime = int(rng.integers(4))
    for _ in range(nt - 2):
        k = int(rng.integers(2 if regime == 1 else 1, 4))
        puzzle["train"].append(_random_pair(rng, k, roles, "train", used))
    puzzle["train"] = [puzzle["train"][int(j)] for j in rng.permutation(nt)]
    for _ in range(nq):
        k = 4 if regime == 0 else 1 if regime == 1 else int(rng.integers(2, 4)) if regime == 2 else int(rng.integers(1, 5))
        puzzle["test"].append(_random_pair(rng, k, roles, "test", used))
    validate_generated_puzzle(puzzle)
    return puzzle


_SHORTCUTS = ("row_major", "point", "endpoint_only", "clip_prefix",
              "stop_program", "endpoint_paint", "first_paint")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal counts")
    seen, roles, training, testing = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _read(inp)
            expected, events, destination = _execute(info)
            _require(np.array_equal(out, expected), "incorrect transaction, sweep, overwrite, or final brush")
            _require(out.shape == info["arena"].shape, "output must be the entire arena")
            _require(np.array_equal(out == info["wall"], ~info["floor"]), "wall cells changed")
            final, k = _square(out == info["marker"])
            _require(k == info["k"] and final == destination, "brush footprint or destination changed")
            _require(_fits(info["floor"], final, k), "final brush overlaps a wall")
            _require(not np.array_equal(out, info["arena"]), "no painting or movement occurred")
            successful_inks = {e["color"] for e in events if e["success"]}
            allowed = {info["background"], info["wall"], info["marker"]} | successful_inks
            _require(set(map(int, np.unique(out))) <= allowed, "a rejected-only color was painted")
            _require(any(not e["success"] for e in events), "each example must reject a command")
            _require(sum(e["success"] for e in events) >= 2, "too few successful strokes")
            roles.add(tuple(info[r] for r in ("background", "wall", "separator", "marker")))
            (training if split == "train" else testing).append(info)
    _require(len(roles) == 1, "one puzzle must share structural colors")
    _require(min(len(i["commands"]) for i in testing) > max(len(i["commands"]) for i in training),
             "tests must have longer programs than all demonstrations")
    _require({i["placement"] for i in training} == {"left", "above"},
             "both panel placements must be demonstrated")
    _require({direction for i in training for direction, _, _ in i["commands"]} == set(_DIRECTIONS),
             "demonstrations must show every arrow")
    _require(any(not e["success"] and e["prefix"] > 0
                 for i in training for e in _execute(i)[1]),
             "training must expose a cancellable legal prefix")
    contrasts = []
    for small in training:
        for large in training:
            if small["k"] != 2 or large["k"] != 3:
                continue
            if (_key(small["floor"]) != _key(large["floor"])
                    or [(d, n) for d, n, _ in small["commands"]] !=
                       [(d, n) for d, n, _ in large["commands"]]):
                continue
            s, b = small["start"], large["start"]
            nested = all(b[j] <= s[j] and s[j] + 2 <= b[j] + 3 for j in range(2))
            patterns = [[e["success"] for e in _execute(i)[1]] for i in (small, large)]
            if nested and patterns[0] != patterns[1]:
                contrasts.append((small, large))
    _require(bool(contrasts), "missing controlled footprint-size contrast")
    _require({e["direction"] for i in training for e in _execute(i)[1] if e["success"]}
             == set(_DIRECTIONS), "each arrow must have a successful training execution")
    for shortcut in _SHORTCUTS:
        _require(any(not np.array_equal(_execute(i, shortcut)[0], _execute(i)[0]) for i in training),
                 "training admits shortcut: " + shortcut)
    # Geometry is independent of occurrence color, including in the minimum
    # two-example puzzle. Every direction receives different training colors.
    for direction in _DIRECTIONS:
        _require(len({color for i in training for d, _, color in i["commands"] if d == direction}) >= 2,
                 "direction can be inferred from a single ink color")


def puzzle_features(puzzle):
    """Small grid-derived measurements for reproducible variation audits."""
    result = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _read(pair["input"])
            _, events, destination = _execute(info)
            normalized = np.empty(info["arena"].shape, dtype=np.int8)
            for value, role in enumerate(("background", "wall", "marker")):
                normalized[info["arena"] == info[role]] = value
            result.append({"split": split, "input_shape": tuple(np.shape(pair["input"])),
                           "arena_shape": info["arena"].shape, "k": info["k"],
                           "placement": info["placement"], "lanes": info["lanes"], "bands": info["bands"],
                           "length": len(events), "distances": tuple(e["distance"] for e in events),
                           "accepted": sum(e["success"] for e in events),
                           "rejected": sum(not e["success"] for e in events),
                           "partial_rejections": sum(not e["success"] and e["prefix"] > 0 for e in events),
                           "destination": destination, "start": info["start"],
                           "wall_cells": int((~info["floor"]).sum()),
                           "roles": tuple(info[r] for r in ("background", "wall", "separator", "marker")),
                           "structure": (normalized.shape, normalized.tobytes().hex(),
                                         tuple((d, n) for d, n, _ in info["commands"]))})
    return result


def _fingerprint(puzzle):
    return tuple(_key(np.asarray(p[key])) for s in ("train", "test") for p in puzzle[s] for key in ("input", "output"))


def self_test():
    # Hand-specified world, destinations, and visible colors, not sampled labels.
    floor = np.ones((13, 13), dtype=bool)
    floor[0] = floor[-1] = False
    floor[:, 0] = floor[:, -1] = False
    floor[3, 7] = False
    commands = [((0, 1), 4, 1), ((1, 0), 3, 2), ((0, -1), 3, 3), ((-1, 0), 2, 4)]
    for k, destination in ((2, (1, 3)), (3, (2, 2))):
        pair = _compose(floor, (1, 2), k, commands, (0, 8, 9, 7), "left", 2, 2)
        expected = np.where(floor, 0, 8).astype(np.int8)
        if k == 2:
            expected[1:3, 2:8] = 1
            expected[1:3, 3:8] = 3
        else:
            expected[1:7, 2:5] = 2
            expected[2:7, 2:5] = 4
        r, c = destination
        expected[r:r + k, c:c + k] = 7
        _require(np.array_equal(pair["output"], expected), "hand-specified rollback/overwrite regression")
        _require(_square(pair["output"] == 7) == (destination, k), "hand-specified destination")
    puzzles = [generate_puzzle(seed) for seed in _HOST_SEEDS]
    rows = [r for p in puzzles for r in puzzle_features(p)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "seed variation collapsed")
    _require(len({r["structure"] for r in rows}) >= 120, "too little structural variation")
    _require(len({r["input_shape"] for r in rows}) >= 35, "too few shapes")
    _require({r["k"] for r in rows} == {1, 2, 3, 4}, "brush scale coverage")
    _require({r["length"] for r in rows} == {4, 6, 8, 9, 12}, "program length coverage")
    _require({n for r in rows for n in r["distances"]} == {1, 2, 3, 4, 5}, "unary distance coverage")
    _require({r["placement"] for r in rows} == {"left", "above"}, "panel orientation coverage")
    _require(len({r["wall_cells"] for r in rows}) >= 50, "terrain variation collapsed")
    for role in range(4):
        _require(len({r["roles"][role] for r in rows}) >= 9, "palette role coverage")
    for seed in (0, 766167826):
        p = generate_puzzle(seed, 10, 4)
        _require(_fingerprint(p) == _fingerprint(generate_puzzle(seed, 10, 4)), "nondeterminism")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            validate_generated_puzzle(generate_puzzle(0, n, m))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    original = puzzles[0]
    for damage in ("wall", "marker", "stroke", "bar"):
        bad = {s: [{key: value.copy() for key, value in p.items()} for p in original[s]]
               for s in ("train", "test")}
        pair = bad["test"][0]
        info = _read(pair["input"])
        if damage == "wall":
            pair["output"][0, 0] = info["background"]
        elif damage == "marker":
            p = tuple(np.argwhere(pair["output"] == info["marker"])[0])
            pair["output"][p] = info["background"]
        elif damage == "stroke":
            p = tuple(np.argwhere((pair["output"] != info["background"]) &
                                  (pair["output"] != info["wall"]) &
                                  (pair["output"] != info["marker"]))[0])
            pair["output"][p] = info["background"]
        else:
            pair["input"][5, 1] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("mutation accepted: " + damage)


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": len(_HOST_SEEDS)}, separators=(",", ":")))
