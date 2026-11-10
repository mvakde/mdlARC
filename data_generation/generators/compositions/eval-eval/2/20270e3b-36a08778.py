"""Live registered obstacles: rays summon inserts before choosing their flanks.

Only NumPy and the standard library are used. All recognition and dynamics in
solve are input-derived; generation has no role in interpreting a query.
"""
from collections import Counter
import json
import numpy as np


def _ensure(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _matrix(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _cells(a, color):
    return {tuple(map(int, p)) for p in np.argwhere(a == color)}


def _components(cells, diagonal=False):
    todo, parts = set(cells), []
    directions = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (dy or dx) and (diagonal or not (dy and dx))]
    while todo:
        start = min(todo)
        todo.remove(start)
        part, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in directions:
                q = y + dy, x + dx
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _shift(cells, dy, dx):
    return {(y + dy, x + dx) for y, x in cells}


def _bbox(cells):
    return (min(y for y, x in cells), max(y for y, x in cells),
            min(x for y, x in cells), max(x for y, x in cells))


def _arrow(y, x):
    # Two-cell stem, three-cell shoulder, one-cell tip: six cells total.
    return {(y, x), (y + 1, x), (y + 2, x - 1), (y + 2, x),
            (y + 2, x + 1), (y + 3, x)}


def _parse_canonical(a, bg, guide, marker, fg):
    """Recognize the downward scene, with no generated metadata."""
    arrows, tips = _components(_cells(a, guide)), []
    if not 1 <= len(arrows) <= 3:
        return None
    for part in arrows:
        t, b, l, r = _bbox(part)
        if b - t != 3 or r - l != 2 or part != _arrow(t, l + 1):
            return None
        tips.append((b, l + 1))
    if len({y for y, x in tips}) != 1:
        return None
    source_row = tips[0][0]
    source_top = source_row - 3
    bars = _components(_cells(a, marker))
    if not 2 <= len(bars) <= 8 or len(bars) % 2:
        return None
    lengths = Counter(len(bar) for bar in bars)
    if any(n != 2 for n in lengths.values()):
        return None
    all_fg = _cells(a, fg)
    objects = _components(all_fg | _cells(a, marker), diagonal=True)
    if len(objects) != len(bars):
        return None
    records = {}
    for bar in bars:
        t, b, l, r = _bbox(bar)
        length = len(bar)
        if t != b or r - l + 1 != length:
            return None
        owners = [part for part in objects if bar <= part]
        if len(owners) != 1 or len(owners[0] & _cells(a, marker)) != length:
            return None
        shape = owners[0] & all_fg
        if not shape:
            return None
        up = all((t - 1, x) in shape for x in range(l, r + 1))
        down = all((t + 1, x) in shape for x in range(l, r + 1))
        if up == down:
            return None
        if up:
            # Sockets are one contiguous transverse backing stroke.
            st, sb, sl, sr = _bbox(shape)
            if (st != t - 1 or sb != t - 1 or len(shape) != sr - sl + 1
                    or st <= source_row):
                return None
            kind = "socket"
        else:
            # Inserts can have ears beside their bar, but never upstream pixels.
            st, sb, sl, sr = _bbox(shape | bar)
            if st != t or sb >= source_top or any((t - 1, x) in shape
                                                   for x in range(l, r + 1)):
                return None
            kind = "insert"
        if kind in records.setdefault(length, {}):
            return None
        records[length][kind] = {"fg": shape, "bar": bar, "origin": (t, l)}
    if any(set(pair) != {"socket", "insert"} for pair in records.values()):
        return None
    pairs = {}
    h, w = a.shape
    for length, record in sorted(records.items()):
        socket, insert = record["socket"], record["insert"]
        sy, sx = socket["origin"]
        iy, ix = insert["origin"]
        translation = sy - 1 - iy, sx - ix
        docked = _shift(insert["fg"], *translation)
        potential = socket["fg"] | docked
        if not all(0 <= y < h and 0 <= x < w for y, x in potential):
            return None
        pairs[length] = {**record, "translation": translation,
                         "docked": docked, "potential": potential}
    # Different possible assemblies are separate objects, even before activation.
    # Rays can still merge in their common one-cell flank gaps.
    prior = set()
    for pair in pairs.values():
        expanded = {(y + dy, x + dx) for y, x in pair["potential"]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        if prior & expanded:
            return None
        prior |= pair["potential"]
    return {"grid": a, "colors": (bg, fg, marker, guide), "tips": sorted(tips),
            "source_row": source_row, "pairs": pairs, "foreground": all_fg,
            "guides": _cells(a, guide)}


def _recognize(grid):
    a = _matrix(grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("input must have four distinct, geometrically visible roles")
    bg = int(colors[np.argmax(counts)])
    choices = []
    remaining = [int(c) for c in colors if c != bg]
    for k in range(4):
        canonical = np.rot90(a, k).copy()
        for guide in remaining:
            for marker in remaining:
                if marker == guide:
                    continue
                fg = next(c for c in remaining if c not in (guide, marker))
                scene = _parse_canonical(canonical, bg, guide, marker, fg)
                if scene is not None:
                    scene["rotation"] = k
                    choices.append(scene)
    if len(choices) != 1:
        raise ValueError("input must uniquely identify arrows, paired bars and backing sides")
    return choices[0]


def _run(cells, y, x):
    left = right = x
    while (y, left - 1) in cells:
        left -= 1
    while (y, right + 1) in cells:
        right += 1
    return left, right


def _trace(scene, mutant=None):
    """Monotone live simulation; faulty variants are used only for evidence QA."""
    h, w = scene["grid"].shape
    bg, fg, marker, guide = scene["colors"]
    solids, ink = set(scene["foreground"]), set(scene["guides"])
    activated, activation_layers, events = set(), [], []
    pairs = scene["pairs"]

    def dock(key):
        pair = pairs[key]
        solids.difference_update(pair["insert"]["fg"])
        moved = pair["docked"]
        if mutant == "same_line":
            moved = _shift(moved, 1, 0)
        elif mutant == "xor":
            solids.symmetric_difference_update(moved)
            activated.add(key)
            return
        solids.update(q for q in moved if 0 <= q[0] < h and 0 <= q[1] < w)
        activated.add(key)

    if mutant == "all_dock":
        for key in pairs:
            dock(key)
    active = {x for y, x in scene["tips"]}
    simultaneous = multi_hit = clipped = interrupted = shared_flanks = 0
    for y in range(scene["source_row"] + 1, h):
        before = set(solids) if mutant == "branch_before" else None
        summons = [key for key, pair in pairs.items() if key not in activated
                   and any((y, x) in pair["socket"]["fg"] for x in active)]
        if mutant not in ("no_dock", "no_rays"):
            for key in summons:
                dock(key)
            if summons:
                activation_layers.append((y, tuple(summons)))
        if mutant == "no_rays":
            break
        barriers = before if before is not None else solids
        hits, following = {}, set()
        for x in sorted(active):
            if (y, x) in barriers and mutant != "straight":
                hits.setdefault(_run(barriers, y, x), []).append(x)
            else:
                following.add(x)
                if (y, x) not in solids:
                    ink.add((y, x))
        simultaneous += len(hits) > 1
        new_flanks = set()
        for (left, right), incident in sorted(hits.items()):
            events.append((y, left, right, tuple(incident)))
            multi_hit += len(incident) > 1
            clipped += left == 0 or right == w - 1
            interrupted += any((y - 1, x) in solids
                               for x in range(max(0, left - 1), min(w, right + 2)))
            for x in range(max(0, left - 1), min(w, right + 2)):
                if (y - 1, x) not in solids:
                    ink.add((y - 1, x))
            for x in (left - 1, right + 1):
                if 0 <= x < w:
                    shared_flanks += x in new_flanks or x in following
                    new_flanks.add(x)
                    following.add(x)
                    if (y, x) not in solids:
                        ink.add((y, x))
        active = following
    out = np.full((h, w), bg, dtype=np.int8)
    for y, x in ink - solids:
        out[y, x] = guide
    for y, x in solids:
        out[y, x] = fg
    if mutant == "erase_seams":
        for key in activated:
            sy, sx = pairs[key]["socket"]["origin"]
            for y in (sy - 1, sy):
                out[y, sx:sx + key] = bg
    stats = {"activated": tuple(sorted(activated)), "inert": tuple(sorted(set(pairs) - activated)),
             "activation_layers": activation_layers, "events": events,
             "simultaneous": simultaneous, "multi_hit": multi_hit, "clipped": clipped,
             "interrupted": interrupted, "shared_flanks": shared_flanks,
             "ordinary_overlap": sum(len(pairs[k]["socket"]["fg"] & pairs[k]["docked"])
                                     for k in activated),
             "marker_lengths": tuple(pairs), "insert_heights": tuple(
                 _bbox(p["insert"]["fg"])[1] - p["insert"]["origin"][0]
                 for p in pairs.values()),
             "final_foreground": solids, "new_guide_cells": len(ink - solids - scene["guides"])}
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the final live assembly and trace, using this input alone."""
    scene = _recognize(input_grid)
    result, _ = _trace(scene)
    return np.rot90(result, -scene["rotation"]).copy()


def _insert(rng, length, height, *, wings=None, growth=None, style=None):
    left, right = wings if wings is not None else tuple(map(int, rng.integers(0, 3, 2)))
    if left + right == 0:
        left = 1
    grow_left, grow_right = growth if growth is not None else tuple(map(int, rng.integers(0, 3, 2)))
    # Parking four inserts in two shelves remains feasible on a <=30 canvas.
    while length + left + right + grow_left + grow_right > 12:
        if grow_left:
            grow_left -= 1
        elif grow_right:
            grow_right -= 1
        elif right:
            right -= 1
        else:
            left -= 1
    bar = {(0, x) for x in range(length)}
    cells = {(0, x) for x in range(-left, 0)} | {(0, x) for x in range(length, length + right)}
    lo, hi = -left, length + right - 1
    cells.update((1, x) for x in range(lo, hi + 1))
    style = style or str(rng.choice(["slab", "hollow", "stair", "fork", "taper"]))
    last_lo, last_hi = lo, hi
    for y in range(2, height + 1):
        if style in ("stair", "slab"):
            a = lo - min(grow_left, y - 1)
            b = hi + min(grow_right, y - 1)
            cells.update((y, x) for x in range(a, b + 1))
        elif style in ("hollow", "fork"):
            a, b = lo - grow_left, hi + grow_right
            # Connect any widening to the preceding row by short elbows.
            cells.update((y, x) for x in range(a, last_lo + 1))
            cells.update((y, x) for x in range(last_hi, b + 1))
            if style == "fork" and y == height:
                cells.update((y, x) for x in range(a, b + 1))
        else:
            a = min(0, lo + y - 1)
            b = max(length - 1, hi - y + 1)
            cells.update((y, x) for x in range(a, b + 1))
        last_lo, last_hi = a, b
    return {"fg": cells, "bar": bar, "length": length, "height": height}


def _socket(length, left=0, right=0):
    return {"fg": {(-1, x) for x in range(-left, length + right)},
            "bar": {(0, x) for x in range(length)}}


def _potential(socket, insert):
    return socket["fg"] | _shift(insert["fg"], -1, 0)


def _park(rng, inserts, width):
    order = list(map(int, rng.permutation(len(inserts))))
    groups = [order[i:i + 2] for i in range(0, len(order), 2)]
    if len(order) == 2 and rng.random() < .25:
        groups = [[order[0]], [order[1]]]
    locations, row = {}, int(rng.integers(0, 2))
    for group in groups:
        sizes = [_bbox(inserts[i]["fg"] | inserts[i]["bar"]) for i in group]
        widths = [box[3] - box[2] + 1 for box in sizes]
        room = width - sum(widths)
        if room < len(group) - 1:
            return None
        gap = min(int(rng.integers(1, 4)), room) if len(group) > 1 else 0
        x = int(rng.integers(width - sum(widths) - gap + 1))
        bottom = row
        for i, box, item_width in zip(group, sizes, widths):
            locations[i] = (row - box[0], x - box[2])
            bottom = max(bottom, row + box[1] - box[0])
            x += item_width + gap
        row = bottom + int(rng.integers(2, 4))
    return locations, max(locations[i][0] + _bbox(inserts[i]["fg"] | inserts[i]["bar"])[1]
                          for i in locations)


def _active_at(columns, cells, start, stop, width):
    active = set(columns)
    for y in range(start, stop):
        following = set()
        for x in active:
            if (y, x) not in cells:
                following.add(x)
            else:
                a, b = _run(cells, y, x)
                following.update(c for c in (a - 1, b + 1) if 0 <= c < width)
        active = following
    return active


def _clear_potentials(potentials):
    used = set()
    for cells in potentials:
        expanded = {(y + dy, x + dx) for y, x in cells
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        if used & expanded:
            return False
        used |= cells
    return True


def _construct(rng, profile, palette, rotation):
    """Three compositional scheduling profiles, all under one live rule."""
    teaching0, teaching1 = profile == "teach_move", profile == "teach_feedback"
    test = profile == "query"
    n = 2 if teaching0 else 3 if teaching1 else 4 if test else int(rng.integers(1, 5))
    heights = [int(rng.integers(3, 5) if test else rng.integers(1, 3)) for _ in range(n)]
    if teaching0:
        heights[0] = 2
        lengths = list(map(int, rng.choice(np.arange(2, 6), n, replace=False)))
    elif teaching1:
        lengths = list(map(int, rng.choice(np.arange(2, 7), 2, replace=False))) + [1]
    else:
        lengths = list(map(int, rng.choice(np.arange(1, 7), n, replace=False)))
    inserts, sockets = [], []
    for i, (length, height) in enumerate(zip(lengths, heights)):
        if teaching0 and i == 0:
            inserts.append(_insert(rng, length, height, wings=(int(rng.integers(2, 4)), 1),
                                   growth=(0, int(rng.integers(1, 3))), style="stair"))
            sockets.append(_socket(length, 1, 0))
        elif teaching1:
            inserts.append(_insert(rng, length, height, wings=(2, 2), growth=(0, 0)))
            sockets.append(_socket(length))
        else:
            inserts.append(_insert(rng, length, height))
            sockets.append(_socket(length, int(rng.integers(0, 2)), int(rng.integers(0, 2))))
    width = int(rng.integers(26, 31) if n >= 3 else rng.integers(18, 31))
    parked = _park(rng, inserts, width)
    if parked is None:
        return None
    park_locations, park_bottom = parked
    arrow_top = park_bottom + 2
    tip = arrow_top + 3
    first_hit = tip + int(rng.integers(1, 3))
    origins, entries = [], []
    potentials = [_potential(s, i) for s, i in zip(sockets, inserts)]

    if teaching0:
        # Left clipping, a genuine foreground overlap, and a second collision
        # on the insert's wider downstream foot. The other pair stays parked.
        box = _bbox(potentials[0])
        x0 = -box[2]
        origins.append((first_hit + 1, x0))
        entries.append(x0 + (lengths[0] - 1) // 2)
        box = _bbox(potentials[1])
        x1 = width - 1 - box[3]
        origins.append((first_hit + 1, x1))
    elif teaching1:
        # Two assembled top runs separated by exactly one shared flank.
        widths = [_bbox(potentials[i])[3] - _bbox(potentials[i])[2] + 1 for i in (0, 1)]
        total = sum(widths) + 1
        start = int(rng.integers(1, width - total))
        x0 = start - _bbox(potentials[0])[2]
        x1 = start + widths[0] + 1 - _bbox(potentials[1])[2]
        origins = [(first_hit + 1, x0), (first_hit + 1, x1)]
        entries = [x0 + (lengths[0] - 1) // 2, x1 + (lengths[1] - 1) // 2]
        shared = start + widths[0]
        second_hit = first_hit + max(heights[:2]) + int(rng.integers(2, 4))
        origins.append((second_hit + 1, shared))
    else:
        first_count = min(2, n)
        boxes = [_bbox(potentials[i]) for i in range(first_count)]
        widths = [b[3] - b[2] + 1 for b in boxes]
        gap = int(rng.integers(1, 4)) if first_count == 2 else 0
        total = sum(widths) + gap
        if total > width:
            return None
        start = int(rng.integers(width - total + 1))
        for i, box, item_width in zip(range(first_count), boxes, widths):
            x = start - box[2]
            origins.append((first_hit + 1, x))
            entries.append(x + int(rng.integers(lengths[i])))
            start += item_width + gap
        first_solids = set.union(*[_shift(potentials[i], *origins[i]) for i in range(first_count)])
        second_hit = first_hit + max(heights[:first_count]) + int(rng.integers(2, 4))
        active = _active_at(entries, first_solids, tip + 1, second_hit, width)
        for i in range(first_count, n):
            box = _bbox(potentials[i])
            feasible = []
            for x in range(-box[2], width - box[3]):
                current = _shift(potentials[i], second_hit + 1, x)
                previous = [_shift(potentials[j], *origins[j]) for j in range(len(origins))]
                if _clear_potentials(previous + [current]):
                    hit = any(x <= c < x + lengths[i] for c in active)
                    feasible.append((x, hit))
            if not feasible:
                return None
            reached_options = [x for x, hit in feasible if hit]
            pool = reached_options if reached_options and rng.random() < .85 else [x for x, hit in feasible]
            origins.append((second_hit + 1, int(rng.choice(pool))))

    all_potential = [_shift(p, *origin) for p, origin in zip(potentials, origins)]
    if not _clear_potentials(all_potential):
        return None
    if any(not 1 <= x < width - 1 for x in entries) or any(abs(a - b) < 4
                                                         for a in entries for b in entries if a != b):
        return None
    bottom = max(_bbox(p)[1] for p in all_potential)
    height = bottom + 1 + int(rng.integers(0, 4))
    if height > 30:
        return None
    bg, fg, marker, guide = palette
    a = np.full((height, width), bg, dtype=np.int8)
    for socket, origin, insert, parked_origin in zip(sockets, origins, inserts,
                                                   [park_locations[i] for i in range(n)]):
        for obj, location in ((socket, origin), (insert, parked_origin)):
            for role, color in (("fg", fg), ("bar", marker)):
                for y, x in _shift(obj[role], *location):
                    if not 0 <= y < height or not 0 <= x < width or a[y, x] != bg:
                        return None
                    a[y, x] = color
    for x in entries:
        for y, c in _arrow(arrow_top, x):
            if a[y, c] != bg:
                return None
            a[y, c] = guide
    if rng.random() < .5:
        a = np.fliplr(a).copy()
    a = np.rot90(a, rotation).copy()
    try:
        scene = _recognize(a)
    except ValueError:
        return None
    out, stats = _trace(scene)
    if not stats["activated"] or stats["new_guide_cells"] == 0:
        return None
    if teaching0 and (len(stats["activated"]) != 1 or len(stats["inert"]) != 1
                      or not stats["ordinary_overlap"] or not stats["clipped"]
                      or not stats["interrupted"] or len(stats["events"]) < 2):
        return None
    if teaching1:
        if (len(stats["activated"]) != 3 or not stats["simultaneous"]
                or not stats["shared_flanks"] or len(stats["activation_layers"]) < 2):
            return None
        # Without the inserted wings, no ray hits the final singleton socket.
        singleton = scene["pairs"][1]["socket"]["fg"]
        _, absent = _trace(scene, "no_dock")
        y, x = next(iter(singleton))
        if any(ey == y and left <= x <= right for ey, left, right, columns in absent["events"]):
            return None
    if teaching0 or teaching1:
        for mutant in ("no_dock", "branch_before", "straight", "same_line", "erase_seams"):
            if np.array_equal(out, _trace(scene, mutant)[0]):
                return None
    return {"input": a, "output": np.rot90(out, -scene["rotation"]).copy()}


def _key(a):
    return (tuple(a.shape), a.tobytes())


def _integer(value, name, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent family with visible teaching and a height shift."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    mixed_palette = bool(rng.integers(2))
    orientation = int(rng.integers(4))
    mixed_orientation = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            profile = ("teach_move" if i == 0 else "teach_feedback" if i == 1 else "practice") if split == "train" else "query"
            local_palette = tuple(map(int, rng.choice(10, 4, replace=False))) if mixed_palette else palette
            local_rotation = int(rng.integers(4)) if mixed_orientation else orientation
            for attempt in range(240):
                pair = _construct(rng, profile, local_palette, local_rotation)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError(f"bounded construction exhausted for {profile}")
    validate_generated_puzzle(puzzle)
    return puzzle


_SHORTCUTS = ("no_dock", "all_dock", "branch_before", "straight", "same_line", "erase_seams", "xor")


def validate_generated_puzzle(puzzle):
    """Validate returned pixels, relocation, live reachability and teaching."""
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _ensure(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _ensure(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, training_stats, rejected = set(), [], set()
    train_heights, test_heights = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _ensure(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, o = _matrix(pair["input"]), _matrix(pair["output"])
            _ensure(a.shape == o.shape and not np.array_equal(a, o), "changed same-size output")
            key = _key(a)
            _ensure(key not in seen, "distinct inputs")
            seen.add(key)
            scene = _recognize(a)
            expected, stats = _trace(scene)
            actual = np.rot90(o, scene["rotation"])
            _ensure(np.array_equal(actual, expected), "incorrect live assembly or ray trace")
            bg, fg, marker, guide = scene["colors"]
            _ensure(not np.any(o == marker), "temporary markers must vanish")
            _ensure(set(map(int, np.unique(o))) == {bg, fg, guide}, "output roles")
            _ensure(all(actual[y, x] == guide for y, x in scene["guides"]), "source preservation")
            reconstructed = set()
            for length, record in scene["pairs"].items():
                reconstructed |= record["socket"]["fg"]
                activated = length in stats["activated"]
                reconstructed |= record["docked"] if activated else record["insert"]["fg"]
                if activated:
                    sy, sx = record["socket"]["origin"]
                    _ensure(all(actual[y, x] == fg for y in (sy - 1, sy)
                                for x in range(sx, sx + length)), "foreground over guide seams")
                    _ensure(all(actual[y, x] == bg for y, x in record["insert"]["fg"]),
                            "activated inserts are moved rather than copied")
                else:
                    _ensure(all(actual[y, x] == fg for y, x in record["insert"]["fg"]),
                            "unreached inserts stay parked")
            _ensure(_cells(actual, fg) == reconstructed, "exact union and relocation")
            _ensure(len(reconstructed) == len(scene["foreground"]) - stats["ordinary_overlap"],
                    "foreground conservation with union overlap")
            _ensure(stats["activated"] and stats["new_guide_cells"] > 0, "both mechanisms present")
            if split == "train":
                training_stats.append(stats)
                train_heights.extend(stats["insert_heights"])
                rejected.update(m for m in _SHORTCUTS if not np.array_equal(_trace(scene, m)[0], actual))
            else:
                test_heights.extend(stats["insert_heights"])
                for mutant in _SHORTCUTS:
                    _ensure(mutant in rejected or np.array_equal(_trace(scene, mutant)[0], actual),
                            f"training admits a test-changing shortcut: {mutant}")
    _ensure(any(s["inert"] for s in training_stats), "causal inactivity demonstration")
    _ensure(any(s["ordinary_overlap"] for s in training_stats), "union collision demonstration")
    _ensure(any(s["clipped"] for s in training_stats), "edge clipping demonstration")
    _ensure(any(s["simultaneous"] and s["shared_flanks"] for s in training_stats), "simultaneous shared flanks")
    _ensure(any(len(s["activation_layers"]) >= 2 for s in training_stats), "live assembly cascade")
    _ensure(any(s["interrupted"] for s in training_stats), "barrier precedence demonstration")
    _ensure(set(_SHORTCUTS) <= rejected, "training must reject the designated simpler rules")
    _ensure(max(train_heights) <= 2 < min(test_heights), "forward-extent train/test shift")


def self_test():
    """Bounded 42-seed coverage, mutation rejection, counts and determinism."""
    shapes, structures, palettes, rotations, counts, lengths, heights, fingerprints = (set() for _ in range(8))
    metrics = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, o = pair["input"], pair["output"]
                scene = _recognize(a)
                _, stats = _trace(scene)
                _ensure(np.array_equal(solve(a.copy()), o), "input-only prediction")
                normalized = np.zeros(scene["grid"].shape, dtype=np.int8)
                for role, color in enumerate(scene["colors"]):
                    normalized[scene["grid"] == color] = role
                structures.add(_key(normalized))
                fingerprint.append((_key(a), _key(o)))
                shapes.add(a.shape)
                palettes.add(scene["colors"])
                rotations.add(scene["rotation"])
                counts.add(len(scene["pairs"]))
                lengths.update(stats["marker_lengths"])
                heights.update(stats["insert_heights"])
                for feature in ("simultaneous", "shared_flanks", "clipped", "interrupted", "ordinary_overlap"):
                    metrics[feature] += bool(stats[feature])
                metrics["inert"] += bool(stats["inert"])
                metrics["cascade"] += len(stats["activation_layers"]) >= 2
        fingerprints.add(tuple(fingerprint))
    _ensure(len(fingerprints) == 42 and len(structures) >= 180, "structural diversity across 42 seeds")
    _ensure(len(shapes) >= 25 and len(palettes) >= 30, "shape and palette diversity")
    _ensure(rotations == set(range(4)) and counts == {1, 2, 3, 4}, "direction and assembly-count coverage")
    _ensure(lengths == set(range(1, 7)) and heights == {1, 2, 3, 4}, "registration and geometry variation")
    _ensure(all(metrics[f] >= 42 for f in metrics), "meaningful event coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(42 + nt * 5 + nq, nt, nq)
            _ensure(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(0), generate_puzzle(0)
    _ensure(all(np.array_equal(x[k], y[k]) for s in first for x, y in zip(first[s], second[s])
                for k in ("input", "output")), "whole-puzzle determinism")
    # Validator must reject labels produced by each meaningful faulty rule.
    for mutant in _SHORTCUTS:
        candidate = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
        for pair in candidate["train"]:
            scene = _recognize(pair["input"])
            wrong = np.rot90(_trace(scene, mutant)[0], -scene["rotation"]).copy()
            if not np.array_equal(wrong, pair["output"]):
                pair["output"] = wrong
                break
        try:
            validate_generated_puzzle(candidate)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted mutant {mutant}")
    return {"seeds_checked": 42, "distinct_puzzles": len(fingerprints),
            "normalized_inputs": len(structures), "input_shapes": len(shapes),
            "palettes": len(palettes), "assembly_counts": sorted(counts),
            "marker_lengths": sorted(lengths), "insert_heights": sorted(heights),
            "feature_pair_counts": dict(metrics)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
