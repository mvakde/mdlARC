"""Reachable plus switches: frozen motif detection and keyed directed rays.

All semantic information is visible in an input. No parent module is imported.
The small construction lessons are witnesses, not restrictions of solve().
"""

from __future__ import annotations

import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W
PLUS = ((0, 0),) + DIRS


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _centers(field, colors):
    """A center must have all four immediate neighbors of its own keyed color."""
    result = {}
    for color in colors:
        m = field == color
        if min(m.shape) < 3:
            continue
        match = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1]
                 & m[1:-1, :-2] & m[1:-1, 2:])
        for r, c in np.argwhere(match):
            result[(int(r) + 1, int(c) + 1)] = int(color)
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    frame = int(a[0, 0])
    _require(np.all(a[0] == frame), "missing full top frame")
    bottoms = np.flatnonzero(np.all(a[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(a) - 1, "invalid key band")
    band = a[1:bottom]
    _require(np.all(band == band[0]), "bars must span the key interior height")
    row = band[0]
    _require(row[0] == row[-1] == frame, "missing key side walls")
    spans = []
    c = 0
    while c < len(row):
        if row[c] == frame:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != frame:
            c += 1
        spans.append((start, c))
    _require(bool(spans) and all(b - a >= 3 for a, b in spans), "unclear panels")
    bg = int(row[spans[0][0] + 1])
    turns = {}
    for start, end in spans:
        _require(np.all(row[start + 1:end - 1] == bg), "inconsistent panel background")
        left, right = int(row[start]), int(row[end - 1])
        _require((left == bg) != (right == bg), "each panel needs one side bar")
        color, side = (right, "R") if left == bg else (left, "L")
        _require(color not in turns and color not in (bg, frame), "conflicting key color")
        turns[color] = side
    field = a[bottom + 1:].copy()
    _require(frame not in np.unique(field), "frame color is reserved for the key")
    others = set(map(int, np.unique(field))) - {bg} - set(turns)
    _require(len(others) == 1, "source color must be unique and unkeyed")
    source = others.pop()
    positions = np.argwhere(field == source)
    _require(len(positions) == 1, "the emitter must be a single cell")
    _require(np.any(field == bg), "field needs some background")
    r, c = map(int, positions[0])
    return dict(field=field, background=bg, source=source, origin=(r, c),
                frame=frame, turns=turns, centers=_centers(field, turns),
                crop=bottom + 1, panels=tuple(end - start + 2 for start, end in spans))


def _walk(info, *, all_marks=False, straight=False, mapping=None, destructive=False,
          center_only=False, one_turn=False):
    """Finite directed-state traversal; options are explicit shortcut hypotheses.

    Production traversal consults only immutable input centers. A separate
    destructive option is used to verify that training exposes that mistake.
    """
    field = info["field"]
    h, w = field.shape
    out = field.copy()
    bg, source = info["background"], info["source"]
    centers = info["centers"]
    if all_marks:
        centers = {(int(r), int(c)): int(field[r, c])
                   for r, c in np.argwhere(np.isin(field, list(info["turns"])))}
    paths = np.zeros_like(field, dtype=bool)
    activated = np.zeros_like(field, dtype=bool)
    events, states, cycles, hits = [], [], [], set()
    for branch in range(4):
        r, c = info["origin"]
        d = branch
        seen = set()
        encounters = 0
        while 0 <= r < h and 0 <= c < w:
            state = (r, c, d)
            if state in seen:
                cycles.append(branch)
                break
            seen.add(state)
            states.append((branch, r, c, d))
            paths[r, c] = True
            color = centers.get((r, c))
            if destructive:
                color = int(out[r, c])
                valid = (color in info["turns"] and 0 < r < h - 1 and 0 < c < w - 1
                         and all(out[r + dr, c + dc] == color for dr, dc in PLUS))
                if not valid:
                    color = None
            if color is not None:
                hits.add((r, c))
                side = info["turns"][color]
                if mapping is not None:
                    side = mapping[side]
                nd = d if (straight or (one_turn and encounters >= 1)) else (d + (1 if side == "R" else -1)) % 4
                encounters += 1
                events.append((branch, r, c, d, nd, color, side))
                offsets = ((0, 0),) if (center_only or all_marks) else PLUS
                for dr, dc in offsets:
                    x, y = r + dr, c + dc
                    if 0 <= x < h and 0 <= y < w:
                        activated[x, y] = True
                        out[x, y] = source
                d = nd
            if field[r, c] == bg:
                out[r, c] = source
            dr, dc = DIRS[d]
            r, c = r + dr, c + dc
    return dict(output=out, paths=paths, activated=activated, events=events,
                states=states, cycles=tuple(cycles), hits=hits)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the cropped field, routing exclusively from this input's key/motifs."""
    return _walk(_decode(input_grid))["output"]


def _header(rng, width, height, turns):
    # In construction coordinates: 0 = visible background, 9 = frame.
    colors = list(map(int, rng.permutation(list(turns))))
    widths = np.full(len(colors), 5, dtype=int)
    widths += rng.multinomial(width - 5 * len(colors), np.full(len(colors), 1 / len(colors)))
    result = np.zeros((height, width), dtype=np.int8)
    result[[0, -1]] = 9
    left = 0
    for color, size in zip(colors, widths):
        result[:, left] = result[:, left + size - 1] = 9
        bar = left + 1 if turns[color] == "L" else left + size - 2
        result[1:-1, bar] = color
        left += int(size)
    return result


def _scene_info(field, turns):
    pos = np.argwhere(field == 1)
    _require(len(pos) == 1, "construction lost emitter")
    return dict(field=field, background=0, source=1, origin=tuple(map(int, pos[0])),
                turns=turns, centers=_centers(field, turns))


def _stamp(field, points, color, cap=.43):
    points = set(points)
    h, w = field.shape
    if not all(0 <= r < h and 0 <= c < w for r, c in points):
        return False
    if any(field[r, c] not in (0, color) for r, c in points):
        return False
    extra = sum(field[r, c] == 0 for r, c in points)
    if np.count_nonzero(field > 1) + extra > field.size * cap:
        return False
    for p in points:
        field[p] = color
    return True


def _plus(field, r, c, color):
    return _stamp(field, [(r + dr, c + dc) for dr, dc in PLUS], color)


def _decorate(rng, field, turns, richness, attempts=18):
    """Independent marks, near misses, strokes, blocks, and added diagonals."""
    h, w = field.shape
    for _ in range(attempts):
        if rng.random() > richness:
            continue
        color = int(rng.choice(list(turns)))
        if rng.random() < .5:
            states = _walk(_scene_info(field, turns))["states"]
            _, r, c, _ = states[int(rng.integers(len(states)))]
        else:
            r, c = int(rng.integers(h)), int(rng.integers(w))
        kind = int(rng.integers(7))
        if kind == 0:
            offsets = [(0, 0)]
        elif kind == 1:  # T: one missing arm.
            offsets = [(0, 0), (0, -1), (0, 1), (-1, 0)]
        elif kind == 2:
            offsets = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        elif kind == 3:
            offsets = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
        elif kind == 4:
            offsets = [(0, j) for j in range(int(rng.integers(2, 7)))]
        elif kind == 5:
            offsets = [(j, 0) for j in range(int(rng.integers(2, 5)))] + [(0, 1), (0, 2)]
        else:
            offsets = [(a, b) for a in range(int(rng.integers(2, 5)))
                       for b in range(int(rng.integers(2, 5)))]
        rotation = int(rng.integers(4))
        points = []
        for a, b in offsets:
            for _ in range(rotation):
                a, b = b, -a
            points.append((r + a, c + b))
        _stamp(field, points, color)
    return field


def _chain_lesson(rng):
    """Four recursive turns, a passed T, a passed arm, and an embedded plus."""
    top = int(rng.integers(2, 5))
    middle = top + int(rng.integers(4, 8))
    endrow = middle + int(rng.integers(3, 6))
    start = int(rng.integers(1, 4))
    first = start + int(rng.integers(9, 12))
    last = first + int(rng.integers(4, 7))
    field = np.zeros((endrow + int(rng.integers(2, 5)), last + int(rng.integers(3, 7))), dtype=np.int8)
    field[middle, start] = 1
    turns = {2: "L", 3: "R"}
    for r, c, color in ((middle, first, 2), (top, first, 3),
                        (top, last, 3), (endrow, last, 2), (middle - 1, start + 5, 3)):
        _require(_plus(field, r, c, color), "lesson stamp failed")
    _stamp(field, [(middle, start + j) for j in (1, 2, 3)] + [(middle - 1, start + 2)], 2)
    _stamp(field, [(middle, first + 2), (middle - 1, first + 1)], 2)
    return field, turns


def _overlap_lesson(rng, loop):
    """Adjacent centers are reached successively; optionally alongside a circuit."""
    if loop:
        top = int(rng.integers(2, 5))
        left = int(rng.integers(2, 5))
        right = left + int(rng.integers(9, 14))
        bottom = top + int(rng.integers(6, 10))
        sc = left + int(rng.integers(4, 8))
        br, bc = bottom + int(rng.integers(3, 5)), sc - 1
        field = np.zeros((br + int(rng.integers(5, 7)), right + int(rng.integers(3, 6))), dtype=np.int8)
        field[top, sc] = 1
        for r, c in ((top, left), (top, right), (bottom, right), (bottom, left)):
            _require(_plus(field, r, c, 3), "circuit stamp failed")
    else:
        br, bc = int(rng.integers(2, 6)), int(rng.integers(2, 9))
        sr = br + int(rng.integers(6, 11))
        field = np.zeros((sr + int(rng.integers(2, 5)), bc + int(rng.integers(9, 14))), dtype=np.int8)
        field[sr, bc + 2] = 1
        _require(_plus(field, sr, bc + 6, 3), "side switch stamp failed")
    field[br:br + 3, bc:bc + 4] = 2
    return field, {2: "L", 3: "R"}


def _orient(rng, field, turns, reflected):
    if reflected:
        field = field[:, ::-1].copy()
        turns = {color: ("R" if side == "L" else "L") for color, side in turns.items()}
    return np.rot90(field, int(rng.integers(4))).copy(), turns


def _enrich_lesson(rng, original, original_turns, kind, loop, richness):
    """Vary witness surroundings and additional topology without losing evidence.

    Rejection checks the returned geometry, not a sampled intention. The clean
    construction remains a bounded fallback, and is also an allowed scene.
    """
    for _ in range(18):
        field, turns = original.copy(), original_turns.copy()
        n = int(rng.integers(2, 6))
        for color in range(4, 2 + n):
            turns[color] = str(rng.choice(("L", "R")))
        for _ in range(int(rng.integers(0, 7))):
            route = _walk(_scene_info(field, turns))
            if rng.random() < .5:
                _, r, c, _ = route["states"][int(rng.integers(len(route["states"])))]
            else:
                r, c = int(rng.integers(1, field.shape[0] - 1)), int(rng.integers(1, field.shape[1] - 1))
            _plus(field, r, c, int(rng.choice(list(turns))))
        _decorate(rng, field, turns, richness, attempts=int(rng.integers(0, 24)))
        info = _scene_info(field, turns)
        route = _walk(info)
        if kind == 0:
            headings = {s: {e[3] for e in route["events"] if e[6] == s} for s in ("L", "R")}
            passed_arm = any((r, c) not in route["hits"] and
                             any(route["paths"][r + dr, c + dc] for dr, dc in DIRS)
                             for r, c in info["centers"])
            incomplete_on_path = any(0 < r < field.shape[0] - 1 and 0 < c < field.shape[1] - 1
                                     and field[r, c] in turns and (r, c) not in info["centers"]
                                     and sum(field[r + dr, c + dc] == field[r, c] for dr, dc in DIRS) == 3
                                     for _, r, c, _ in route["states"])
            recursive = not np.array_equal(_walk(info, one_turn=True)["output"], route["output"])
            okay = (all(len(x) >= 2 for x in headings.values()) and passed_arm and incomplete_on_path and recursive
                    and not np.array_equal(_all_activated(info, route), route["output"])
                    and not np.array_equal(_walk(info, all_marks=True)["output"], route["output"])
                    and not np.array_equal(_walk(info, straight=True)["output"], route["output"])
                    and not np.array_equal(_walk(info, center_only=True)["output"], route["output"])
                    and not np.array_equal(_component_answer(info, route), route["output"]))
        else:
            okay = (not np.array_equal(_walk(info, destructive=True)["output"], route["output"])
                    and (not loop or bool(route["cycles"])))
        if okay:
            return field, turns
    return original.copy(), original_turns.copy()


def _package(rng, field, turns, palette, key_count=None):
    turns = turns.copy()
    if key_count is None:
        key_count = int(rng.integers(2, 6))
    for color in range(2, 2 + key_count):
        turns.setdefault(color, str(rng.choice(("L", "R"))))
    if field.shape[1] < 5 * key_count:
        extra = 5 * key_count - field.shape[1]
        left = int(rng.integers(extra + 1))
        field = np.pad(field, ((0, 0), (left, extra - left)))
    hh = int(rng.integers(3, min(5, 30 - field.shape[0]) + 1))
    inp = np.vstack((_header(rng, field.shape[1], hh, turns), field))
    inp = palette[inp].astype(np.int8)
    return dict(input=inp, output=solve(inp))


def _random_scene(rng, h, w, n, target, richness, allow_cycle, force_cycle=False):
    field = np.zeros((h, w), dtype=np.int8)
    turns = {color: str(rng.choice(("L", "R"))) for color in range(2, 2 + n)}
    planted = 0
    if force_cycle:
        # A geometric circuit scaffold is then exposed to the same placement
        # and camouflage process as any other network. Final traversal must
        # confirm that a circuit survives, rather than trusting this intention.
        top, left = int(rng.integers(1, h - 5)), int(rng.integers(1, w - 5))
        bottom = int(rng.integers(top + 4, h - 1))
        right = int(rng.integers(left + 4, w - 1))
        side = str(rng.choice(("L", "R")))
        anchor_color = int(rng.choice(list(turns)))
        turns[anchor_color] = side
        colors = [c for c, s in turns.items() if s == side]
        for r, c in ((top, left), (top, right), (bottom, right), (bottom, left)):
            _require(_plus(field, r, c, int(rng.choice(colors))), "random circuit failed to fit")
        if rng.random() < .5:
            r, c = int(rng.choice((top, bottom))), int(rng.integers(left + 2, right - 1))
        else:
            r, c = int(rng.integers(top + 2, bottom - 1)), int(rng.choice((left, right)))
        field[r, c] = 1
        planted = 4
    else:
        field[int(rng.integers(h)), int(rng.integers(w))] = 1
    for _ in range(target * 32):
        if planted >= target:
            break
        info = _scene_info(field, turns)
        route = _walk(info)
        states = [s for s in route["states"] if 0 < s[1] < h - 1 and 0 < s[2] < w - 1
                  and field[s[1], s[2]] == 0]
        # Most placements alter a route. Some are genuine unreachable switches.
        if states and rng.random() < .78:
            indirect = [s for s in states if s[0] != s[3]]
            options = indirect if indirect and rng.random() < .7 else states
            _, r, c, _ = options[int(rng.integers(len(options)))]
        else:
            r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        color = int(rng.choice(list(turns)))
        trial = field.copy()
        if not _plus(trial, r, c, color):
            continue
        if not allow_cycle and _walk(_scene_info(trial, turns))["cycles"]:
            continue
        field, planted = trial, planted + 1
    _decorate(rng, field, turns, richness, attempts=int(rng.integers(12, 35)))
    return field, turns


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    base_palette = rng.permutation(10)
    stable = rng.random() < .25
    reflected = bool(rng.integers(2))
    loop = rng.random() < .38
    richness = float(rng.uniform(.15, .8))
    shift = str(rng.choice(("larger", "thin", "camouflage")))
    puzzle = dict(train=[], test=[])
    seen = set()

    def palette():
        return base_palette if stable else rng.permutation(10)

    # Two variable construction witnesses suffice even for a 2/1 request.
    for index in range(2):
        original, original_turns = _chain_lesson(rng) if index == 0 else _overlap_lesson(rng, loop)
        lesson_reflection = reflected if index == 0 else not reflected
        for attempt in range(24):
            if attempt < 20:
                field, turns = _enrich_lesson(rng, original, original_turns, index, loop, richness)
                field, turns = _orient(rng, field, turns, lesson_reflection)
            else:
                field, turns = original.copy(), original_turns.copy()
                if lesson_reflection:
                    field = field[:, ::-1].copy()
                    turns = {color: ("R" if side == "L" else "L") for color, side in turns.items()}
                field = np.rot90(field, attempt - 20).copy()
            pair = _package(rng, field, turns, palette(), len(turns))
            if index == 0 or not np.array_equal(_walk(_decode(pair["input"]), destructive=True)["output"], pair["output"]):
                break
        else:
            raise AssertionError("could not preserve the overlap witness after orientation")
        puzzle["train"].append(pair)
        seen.add(_key(pair["input"]))

    cycle_taught = any(_walk(_decode(p["input"]))["cycles"] for p in puzzle["train"])
    for split, count in (("train", nt - 2), ("test", nq)):
        for _ in range(count):
            force_cycle = split == "test" and cycle_taught and rng.random() < .5
            for attempt in range(80):
                hh_limit = 26  # All header choices still fit in 30 rows.
                n = int(rng.integers(2, 6))
                if split == "test" and shift == "larger":
                    h, w = int(rng.integers(23, hh_limit + 1)), int(rng.integers(27, 31))
                elif split == "test" and shift == "thin":
                    h, w = int(rng.integers(7, 11)), int(rng.integers(max(22, 5 * n), 31))
                else:
                    h, w = int(rng.integers(9, hh_limit + 1)), int(rng.integers(max(12, 5 * n), 31))
                target = int(rng.integers(2, 13 if split == "test" else 9))
                density = min(.98, richness + .3) if split == "test" and shift == "camouflage" else richness
                field, turns = _random_scene(rng, h, w, n, target, density, cycle_taught, force_cycle)
                pair = _package(rng, field, turns, palette(), n)
                info = _decode(pair["input"])
                route = _walk(info)
                if not route["hits"] or (route["cycles"] and not cycle_taught):
                    continue
                if force_cycle and not route["cycles"]:
                    continue
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    if split == "train":
                        cycle_taught |= bool(route["cycles"])
                    break
            else:
                # Finite construction fallback: an overlap lesson has reached
                # switches and thousands of geometry/palette/header variants.
                for retry in range(80):
                    field, turns = _overlap_lesson(rng, cycle_taught)
                    field, turns = _orient(rng, field, turns, bool(rng.integers(2)))
                    pair = _package(rng, field, turns, palette())
                    if _key(pair["input"]) not in seen:
                        seen.add(_key(pair["input"]))
                        puzzle[split].append(pair)
                        break
                else:
                    raise AssertionError("unique scene construction exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _all_activated(info, route):
    out = info["field"].copy()
    out[route["paths"] & (out == info["background"])] = info["source"]
    for r, c in info["centers"]:
        for dr, dc in PLUS:
            out[r + dr, c + dc] = info["source"]
    return out


def _component_answer(info, route):
    """Shortcut: recolor whole monochrome components containing reached centers."""
    field = info["field"]
    h, w = field.shape
    out = field.copy()
    out[route["paths"] & (field == info["background"])] = info["source"]
    seen = set()
    for start in route["hits"]:
        color = int(field[start])
        stack = [start]
        while stack:
            r, c = stack.pop()
            if (r, c) in seen:
                continue
            seen.add((r, c))
            out[r, c] = info["source"]
            for dr, dc in DIRS:
                x, y = r + dr, c + dc
                if 0 <= x < h and 0 <= y < w and field[x, y] == color and (x, y) not in seen:
                    stack.append((x, y))
    return out


def _features(pair):
    info = _decode(pair["input"])
    route = _walk(info)
    headings = {s: {e[3] for e in route["events"] if e[6] == s} for s in ("L", "R")}
    counts = {}
    for color in info["centers"].values():
        counts[color] = counts.get(color, 0) + 1
    return dict(shape=info["field"].shape, origin=info["origin"], keys=len(info["turns"]),
                centers=len(info["centers"]), reached=len(route["hits"]),
                turns=len(route["events"]), cycle=bool(route["cycles"]),
                max_branch_turns=max(sum(e[0] == b for e in route["events"]) for b in range(4)),
                unvisited=len(info["centers"]) - len(route["hits"]),
                repeated=any(count > 1 for count in counts.values()),
                headings=headings, crop=info["crop"], panels=info["panels"],
                edge_source=any(x == 0 or x == s - 1 for x, s in zip(info["origin"], info["field"].shape)),
                destructive=not np.array_equal(_walk(info, destructive=True)["output"], route["output"]))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    evidence = {"L": set(), "R": set()}
    witnesses = dict(no_a=False, no_b=False, reach=False, union=False, component=False,
                     overlap=False, recursion=False, cycle=False)
    fits = {(l, r) for l in ("L", "R") for r in ("L", "R")}
    test_cycle = test_destructive = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _key(inp)
            _require(signature not in seen, "duplicate inputs")
            seen.add(signature)
            info = _decode(inp)
            field = info["field"]
            route = _walk(info)
            _require(out.shape == field.shape, "output must omit exactly the key band")
            _require(np.array_equal(out, route["output"]), "wrong routed activation output")
            # Check the local motif definition using explicit coordinate tests.
            scanned = {(r, c): int(field[r, c]) for r in range(1, field.shape[0] - 1)
                       for c in range(1, field.shape[1] - 1)
                       if int(field[r, c]) in info["turns"]
                       and all(field[r + dr, c + dc] == field[r, c] for dr, dc in PLUS)}
            _require(scanned == info["centers"], "motif detector disagrees with local definition")
            change = route["activated"] | (route["paths"] & (field == info["background"]))
            _require(np.array_equal(out[~change], field[~change]), "unactivated marks must be preserved")
            _require(np.all(out[change] == info["source"]), "ray and activation color must come from source")
            _require(len(route["states"]) <= 16 * field.size, "directed-state termination bound")
            destructive = not np.array_equal(_walk(info, destructive=True)["output"], out)
            if split == "train":
                for e in route["events"]:
                    evidence[e[6]].add(e[3])
                for l, r in tuple(fits):
                    prediction = _walk(info, mapping={"L": l, "R": r})["output"]
                    if not np.array_equal(prediction, out):
                        fits.remove((l, r))
                witnesses["no_a"] |= not np.array_equal(_walk(info, all_marks=True)["output"], out)
                witnesses["no_b"] |= not np.array_equal(_walk(info, straight=True)["output"], out)
                witnesses["reach"] |= not np.array_equal(_all_activated(info, route), out)
                witnesses["union"] |= not np.array_equal(_walk(info, center_only=True)["output"], out)
                witnesses["component"] |= not np.array_equal(_component_answer(info, route), out)
                witnesses["overlap"] |= destructive
                witnesses["recursion"] |= not np.array_equal(_walk(info, one_turn=True)["output"], out)
                witnesses["cycle"] |= bool(route["cycles"])
            else:
                test_cycle |= bool(route["cycles"])
                test_destructive |= destructive
    _require(all(len(headings) >= 2 for headings in evidence.values()), "teach relative turns from multiple headings")
    _require(fits == {("L", "R")}, "key meanings must be uniquely demonstrated among four turn hypotheses")
    required = ("no_a", "no_b", "reach", "union", "component", "overlap", "recursion")
    _require(all(witnesses[k] for k in required), "missing teaching counterexamples: " + str([k for k in required if not witnesses[k]]))
    _require(not test_cycle or witnesses["cycle"], "test circuit needs a training circuit")
    _require(not test_destructive or witnesses["overlap"], "test overlap needs frozen-detection evidence")


def _fingerprint(puzzle):
    return tuple((s, _key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test() -> dict:
    """Bounded checks of 42 seeds, role variation, and targeted rule semantics."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fs = [_features(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "seed diversity")
    _require(_fingerprint(puzzles[17]) == _fingerprint(generate_puzzle(17)), "seed reproducibility")
    _require(len({f["shape"] for f in fs}) >= 50, "shape variation")
    _require(len({f["centers"] for f in fs}) >= 10, "switch-count variation")
    _require({2, 3, 4, 5} <= {f["keys"] for f in fs}, "key-count variation")
    _require(len({f["origin"] for f in fs}) >= 50, "emitter-position variation")
    _require(sum(f["cycle"] for f in fs) >= 5, "closed circuit coverage")
    _require(sum(_features(pair)["cycle"] for p in puzzles for pair in p["test"]) >= 2,
             "held-out circuit coverage")
    _require(sum(f["unvisited"] > 0 for f in fs) >= 30, "reachability coverage")
    _require(sum(f["destructive"] for f in fs) >= 42, "immutable geometry coverage")
    _require(any(f["edge_source"] for f in fs), "boundary emitter coverage")
    _require(any(min(f["shape"]) <= 8 for f in fs), "thin field coverage")
    _require(any(max(f["shape"]) == 30 for f in fs), "full width coverage")
    for p in puzzles:
        _require(len({_features(pair)["shape"] for s in p for pair in p[s]}) >= 2, "within-puzzle geometry")

    # Exhaust the 3x3 local binary vocabulary; no missing arm or diagonal can
    # substitute for one of the five cardinal cells.
    for bits in range(512):
        field = np.array([2 if bits & (1 << i) else 0 for i in range(9)], dtype=np.int8).reshape(3, 3)
        expected = all(field[1 + dr, 1 + dc] == 2 for dr, dc in PLUS)
        _require(((1, 1) in _centers(field, [2])) == expected, "local motif truth table")
    # A clipped candidate at an edge must not wrap; a complete edge-reaching
    # cross is legal. Adjacent centers in a 3x4 block must both be recognized.
    edge = np.zeros((5, 6), dtype=np.int8)
    _plus(edge, 1, 1, 2)
    _require((1, 1) in _centers(edge, [2]), "complete edge plus")
    edge[:] = 0
    edge[0, 1:4] = edge[1, 2] = edge[-1, 2] = 2
    _require(not _centers(edge, [2]), "clipped cross must not wrap")
    block = np.pad(np.full((3, 4), 2, dtype=np.int8), ((2, 2), (2, 2)))
    _require(set(_centers(block, [2])) == {(3, 3), (3, 4)}, "adjacent center detection")
    # Hand-enumerated closed circuit: the east ray travels around the square;
    # the west ray turns north at its top-left corner. Other rays remain straight.
    field = np.zeros((20, 20), dtype=np.int8)
    field[3, 7] = 1
    corners = ((3, 2), (3, 13), (10, 13), (10, 2))
    for r, c in corners:
        _plus(field, r, c, 2)
    header_rng = np.random.default_rng(1729)
    header = _header(header_rng, 20, 3, {2: "R", 3: "L"})
    mask = np.zeros_like(field, dtype=bool)
    mask[3, 2:14] = mask[10, 2:14] = True
    mask[0:11, 2] = mask[3:11, 13] = mask[:, 7] = True
    for r, c in corners:
        for dr, dc in PLUS:
            mask[r + dr, c + dc] = True
    expected = field.copy()
    expected[mask] = 1
    _require(np.array_equal(solve(np.vstack((header, field))), expected), "hand-enumerated circuit output")
    # Rotate the field while keeping the key upright: relative turns commute
    # with rotations, unlike an absolute north/east direction lookup.
    for turns in range(4):
        rotated = np.rot90(field, turns).copy()
        _require(np.array_equal(solve(np.vstack((header, rotated))), np.rot90(expected, turns)),
                 "relative-turn rotational equivariance")
    # No-match member: a keyed boundary mark is transparent and preserved.
    small = np.zeros((3, 10), dtype=np.int8)
    small[1, 4], small[2, 4] = 1, 2
    expected = small.copy()
    expected[1, :] = 1
    expected[0, 4] = 1
    header = _header(header_rng, 10, 3, {2: "L", 3: "R"})
    _require(np.array_equal(solve(np.vstack((header, small))), expected), "no-match member and boundary mark")
    mixed = np.zeros((8, 10), dtype=np.int8)
    mixed[5, 5] = 1
    _plus(mixed, 2, 5, 2)
    mixed[2, 6] = 3
    _require(not _centers(mixed, [2, 3]), "mixed-color arms do not make a switch")
    # Palette relabeling is equivariant, including assigning zero to the emitter.
    pair = puzzles[0]["train"][0]
    for offset in (1, 4, 7):
        relabel = (np.arange(10) + offset) % 10
        _require(np.array_equal(solve(relabel[pair["input"]]), relabel[pair["output"]]), "palette equivariance")
    before = pair["input"].copy()
    solve(pair["input"])
    _require(np.array_equal(before, pair["input"]), "solve must not mutate input")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(123456 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return dict(seeds=42, pairs=len(fs), shapes=len({f["shape"] for f in fs}),
                center_counts=sorted({f["centers"] for f in fs}),
                cyclic_pairs=sum(f["cycle"] for f in fs),
                destructive_counterexamples=sum(f["destructive"] for f in fs))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
