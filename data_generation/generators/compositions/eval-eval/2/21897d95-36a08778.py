"""Color relays around islands. Standard library + NumPy, no parent imports.

The input reader reconstructs every convention. Generation plans are never
consulted by solve(), and geometry always refers to the original clean scene.
"""
from __future__ import annotations

from collections import defaultdict
import json
import numpy as np


HOST_SEEDS = (0, 779460406, 510320664, 644977552, 334350424, 242122638,
              511211426, 710384113, 469994061, 823290132, 97458184, 304186635,
              268558649, 17275999, 321619037, 364582108, 922494852, 953897850,
              682067585, 339684160, 335353930, 625226779, 438606662, 347959097,
              454081538, 762773046, 866586580, 645475299, 971299180, 77522461,
              704894064, 795746029, 636777205, 975867343, 862982547, 337430455,
              776596526, 280766834, 119379936, 497918492, 42151904, 732817986)
SHORTCUTS = ("no_paint", "no_exchange", "straight", "colorwide",
             "latest", "choose_one", "ignore_explicit")
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grid must be an integer matrix of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(clean, background):
    ids = np.full(clean.shape, -1, dtype=np.int16)
    colors, cells = [], []
    h, w = clean.shape
    for r, c in np.ndindex(clean.shape):
        if clean[r, c] == background or ids[r, c] >= 0:
            continue
        label, color = len(colors), int(clean[r, c])
        colors.append(color)
        pending, points = [(r, c)], []
        ids[r, c] = label
        while pending:
            y, x = pending.pop()
            points.append((y, x))
            for dy, dx in DIRECTIONS:
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and ids[yy, xx] < 0
                        and clean[yy, xx] == color):
                    ids[yy, xx] = label
                    pending.append((yy, xx))
        cells.append(points)
    return ids, colors, cells


def _run(ids, r, c):
    label = int(ids[r, c])
    a = b = c
    while a > 0 and ids[r, a - 1] == label:
        a -= 1
    while b + 1 < ids.shape[1] and ids[r, b + 1] == label:
        b += 1
    return label, a, b


def _relay(clean, background, ink, arrows, shortcut=None):
    """Layer simulation with payload sets and immutable original components.

    Shortcut modes are explicit wrong rules used only to measure demonstrations.
    """
    h, w = clean.shape
    ids, colors, cells = _components(clean, background)
    injections = defaultdict(lambda: defaultdict(set))
    for arrow in arrows:
        r, c = arrow["start"]
        payload = arrow["host"] if shortcut == "ignore_explicit" else arrow["payload"]
        injections[r][c].add(payload)
    trace = np.zeros(clean.shape, dtype=bool)
    active = defaultdict(set)
    assignments, first_depth, first_colors = {}, {}, {}
    events, incoming = [], []
    shared_flanks = conflicts = 0
    for r in range(h):
        for c, paints in injections[r].items():
            active[c].update(paints)
        incoming.append({c: set(p) for c, p in active.items()})
        following, hits = defaultdict(set), defaultdict(set)
        for c, paints in sorted(active.items()):
            if ids[r, c] < 0:
                trace[r, c] = True
                following[c].update(paints)
            else:
                hits[_run(ids, r, c)].update(paints)
        by_object = defaultdict(set)
        for (label, a, b), paints in hits.items():
            by_object[label].update(paints)
            events.append((r, a, b, label, tuple(sorted(paints))))
        for label, paints in by_object.items():
            if label not in first_depth:
                first_depth[label], first_colors[label] = r, set(paints)
                conflicts += len(paints) > 1
            if first_depth[label] == r or shortcut == "latest":
                assignments[label] = (min(paints) if shortcut == "choose_one" or len(paints) == 1
                                      else colors[label])
        flanks_this_layer = set()
        for (label, a, b), paints in sorted(hits.items()):
            leaving = set(paints) if shortcut == "no_exchange" else {colors[label]}
            if shortcut == "straight":
                # A deliberately wrong geometric alternative: terminate the
                # detour and keep incident positions moving straight forward.
                for c in active:
                    if a <= c <= b:
                        following[c].update(leaving)
                continue
            if r > 0:
                for c in range(max(0, a - 1), min(w, b + 2)):
                    if ids[r - 1, c] < 0:
                        trace[r - 1, c] = True
            for c in (a - 1, b + 1):
                if 0 <= c < w:
                    # The family's separation convention guarantees background.
                    _need(ids[r, c] < 0, "a flank is occupied by another island")
                    shared_flanks += c in flanks_this_layer or c in following
                    flanks_this_layer.add(c)
                    trace[r, c] = True
                    following[c].update(leaving)
        active = following
    out = clean.copy()
    if shortcut == "colorwide":
        global_assignments = {}
        for label in sorted(assignments, key=lambda x: (first_depth[x], x)):
            global_assignments.setdefault(colors[label], assignments[label])
        for label, color in enumerate(colors):
            out[ids == label] = global_assignments.get(color, color)
    elif shortcut != "no_paint":
        for label, payload in assignments.items():
            out[ids == label] = payload
    out[trace] = ink
    info = dict(clean=clean, background=background, ink=ink, arrows=arrows,
                ids=ids, colors=colors, cells=cells, trace=trace,
                assignments=assignments, first_depth=first_depth,
                first_colors=first_colors, events=events, incoming=incoming,
                shared_flanks=shared_flanks, conflicts=conflicts)
    return out, info


def _read(input_grid):
    """Find the unique T role/direction using only the input, then restore it."""
    raw = _grid(input_grid)
    values, counts = np.unique(raw, return_counts=True)
    if max(counts) <= raw.size / 2:
        raise ValueError("the background must occupy more than half the input")
    background = int(values[np.argmax(counts)])
    interpretations = []
    for rotation in range(4):
        g = np.rot90(raw, rotation).copy()
        h, w = g.shape
        for ink in map(int, values):
            if ink == background:
                continue
            arrows = []
            if h < 4 or w < 3:
                continue
            host_support = g[2:-1, :-2]
            candidate_mask = ((g[1:-2, :-2] == ink) & (g[1:-2, 2:] == ink)
                              & (g[2:-1, 1:-1] == ink)
                              & (host_support == g[2:-1, 2:])
                              & (host_support != background) & (host_support != ink)
                              & (g[3:, 1:-1] == background)
                              & (g[1:-2, 1:-1] != background))
            for y, x in np.argwhere(candidate_mask):
                r, c = int(y) + 1, int(x) + 1
                arms_tip = ((r, c - 1), (r, c + 1), (r + 1, c))
                host = int(g[r + 1, c - 1])
                explicit = g[r, c] != ink
                arrows.append(dict(center=(r, c), start=(r + 2, c), host=host,
                                   payload=int(g[r, c]) if explicit else host,
                                   explicit=bool(explicit),
                                   write={*arms_tip, (r, c)}))
            if not arrows:
                continue
            writes, covered = set(), set()
            valid = True
            clean = g.copy()
            for arrow in arrows:
                if writes & arrow["write"]:
                    valid = False
                    break
                writes.update(arrow["write"])
                covered.update(p for p in arrow["write"] if g[p] == ink)
                for p in arrow["write"]:
                    clean[p] = arrow["host"]
            if not valid or covered != set(map(tuple, np.argwhere(g == ink))):
                continue
            ids, colors, cells = _components(clean, background)
            for arrow in arrows:
                label = int(ids[arrow["center"]])
                if (label < 0 or any(ids[p] != label for p in arrow["write"])
                        or not any(r == 0 for r, c in cells[label])):
                    valid = False
                    break
                arrow["source"] = label
            # Neighboring non-background pixels of different fills would leave
            # no empty flank between islands, and are outside this family.
            for a, b in ((clean[:-1], clean[1:]), (clean[:, :-1], clean[:, 1:])):
                if np.any((a != background) & (b != background) & (a != b)):
                    valid = False
            if valid:
                interpretations.append((rotation, clean, background, ink, arrows))
    if len(interpretations) != 1:
        raise ValueError("input must have exactly one common T interpretation")
    return interpretations[0]


def _decode(input_grid, shortcut=None):
    rotation, clean, background, ink, arrows = _read(input_grid)
    result, info = _relay(clean, background, ink, arrows, shortcut)
    info["rotation"] = rotation
    return np.rot90(result, -rotation).copy(), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from one input; never mutate the argument."""
    return _decode(input_grid)[0]


def _put(grid, r, c, mask, color):
    h, w = grid.shape
    points = [(r + int(y), c + int(x)) for y, x in zip(*np.nonzero(mask))]
    if not points or any(not (0 <= y < h and 0 <= x < w) for y, x in points):
        return False
    for y, x in points:
        if grid[y, x] != 0:
            return False
        for dy, dx in DIRECTIONS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and grid[yy, xx] != 0:
                return False
    for y, x in points:
        grid[y, x] = color
    return True


def _shape(rng, height, width, kind=None):
    """Connected masks with genuine pixel-scale variation, not a block tiling."""
    kind = int(rng.integers(6)) if kind is None else kind
    mask = np.zeros((height, width), dtype=bool)
    if kind == 0:
        mask[:] = True
    elif kind == 1:  # Tapering, possibly asymmetrically.
        left, right = 0, width - 1
        for r in range(height):
            mask[r, left:right + 1] = True
            if left < right:
                if rng.random() < .5:
                    left += 1
                elif rng.random() < .7:
                    right -= 1
    elif kind == 2:  # Overlapping, moving row intervals form steps/concavities.
        left, right = 0, width - 1
        for r in range(height):
            mask[r, left:right + 1] = True
            new_left = int(np.clip(left + rng.integers(-1, 2), 0, right))
            new_right = int(np.clip(right + rng.integers(-1, 2), max(left, new_left), width - 1))
            left, right = new_left, new_right
    elif kind == 3:  # A hooked component can be encountered more than once.
        mask[0] = True
        mask[:, -1] = True
        mask[-1] = True
        if height > 2 and width > 2 and rng.random() < .5:
            mask[-1, 0] = False
    elif kind == 4:  # Slender stem and a crossbar/elbow.
        stem = int(rng.integers(width))
        mask[:, stem] = True
        mask[0] = True
        if rng.random() < .5:
            mask[-1, :stem + 1] = True
    else:  # Random connected growth inside a bounded box.
        start = (0, int(rng.integers(width)))
        mask[start] = True
        desired = int(rng.integers(max(1, height * width // 3), height * width + 1))
        frontier = {start}
        while int(mask.sum()) < desired:
            options = set()
            for y, x in frontier:
                options.update((y + dy, x + dx) for dy, dx in DIRECTIONS
                               if 0 <= y + dy < height and 0 <= x + dx < width
                               and not mask[y + dy, x + dx])
            if not options:
                break
            # Sorting makes seeded output independent of set iteration order.
            choices = sorted(options)
            point = choices[int(rng.integers(len(choices)))]
            mask[point] = True
            frontier.add(point)
    rows, cols = np.nonzero(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _source(grid, rng, center, color, payload=None, height=None):
    height = int(rng.integers(3, 6)) if height is None else height
    half = int(rng.integers(1, 3))
    mask = np.ones((height, 2 * half + 1), dtype=bool)
    # A reservoir's top shoulder may have a notch without damaging the T.
    if half == 2 and height >= 4 and rng.random() < .5:
        mask[0, 0] = False
    if not _put(grid, 0, center - half, mask, color):
        return None
    return (height - 2, center, payload)


def _annotate(clean, launches):
    inp = clean.copy()
    arrows = []
    for r, c, explicit_payload in launches:
        host = int(clean[r, c])
        writes = {(r, c - 1), (r, c), (r, c + 1), (r + 1, c)}
        for p in writes:
            inp[p] = 1
        if explicit_payload is not None:
            inp[r, c] = explicit_payload
        arrows.append(dict(center=(r, c), start=(r + 2, c), host=host,
                           payload=host if explicit_payload is None else explicit_payload,
                           explicit=explicit_payload is not None, write=writes))
    return inp, arrows


def _witness_exchange(rng):
    w = int(rng.integers(15, 24))
    c = int(rng.integers(5, w - 4))
    sh = int(rng.integers(3, 6))
    first = sh + int(rng.integers(2, 4))
    a, b = c - int(rng.integers(1, 3)), c + int(rng.integers(1, 3))
    hook_depth = int(rng.integers(2, 4))
    recipient_row = first + hook_depth + int(rng.integers(2, 4))
    h = int(rng.integers(max(17, recipient_row + 4), 25))
    clean = np.zeros((h, w), dtype=np.int8)
    launch = _source(clean, rng, c, 2, height=sh)
    _need(launch is not None, "witness source placement")
    hook = np.zeros((hook_depth + 1, b - a + 2), dtype=bool)
    hook[0, 1:] = True
    hook[:, -1] = True
    hook[-1] = True
    _need(_put(clean, first, a - 1, hook, 3), "hook placement")
    recipient = _shape(rng, int(rng.integers(1, 4)), 2, kind=int(rng.integers(2)))
    _need(_put(clean, recipient_row, a - 2, recipient, 4), "relay recipient placement")
    # This equal-fill island is upstream of the descendant and cannot be hit.
    _need(_put(clean, first, 0, np.ones((1, 2), dtype=bool), 3), "inert equal-fill placement")
    if rng.random() < .5:
        _put(clean, recipient_row, b + 1, _shape(rng, 2, 2, kind=1), 5)
    return clean, [launch]


def _witness_consensus(rng):
    w = int(rng.integers(18, 26))
    left = int(rng.integers(3, 6))
    right = int(rng.integers(w - 6, w - 3))
    middle = int(rng.integers(left + 2, right - 1))
    sh1, sh2 = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    first = max(sh1, sh2) + int(rng.integers(2, 4))
    body_height = int(rng.integers(1, 4))
    next_row = first + body_height + int(rng.integers(2, 4))
    h = int(rng.integers(max(17, next_row + 4), 25))
    clean = np.zeros((h, w), dtype=np.int8)
    l1 = _source(clean, rng, left, 2, height=sh1)
    l2 = _source(clean, rng, right, 3, payload=8, height=sh2)
    if l1 is None or l2 is None:
        return None
    a, b = left - int(rng.integers(0, 2)), right + int(rng.integers(0, 2))
    mask1 = _shape(rng, body_height, middle - a, kind=1)
    mask2 = _shape(rng, body_height, b - middle, kind=1)
    if not (_put(clean, first, a, mask1, 4)
            and _put(clean, first, middle + 1, mask2, 5)):
        return None
    merge_width = int(rng.choice([1, 3]))
    merge = _shape(rng, int(rng.integers(1, 4)), merge_width, kind=1)
    if not (_put(clean, next_row, middle - merge_width // 2, merge, 6)
            and _put(clean, next_row, a - 1, np.ones((2, 1), dtype=bool), 6)):
        return None
    return clean, [l1, l2]


def _geometry_step(clean, r, active):
    following = set()
    for c in active:
        if clean[r, c] == 0:
            following.add(c)
            continue
        a = b = c
        while a > 0 and clean[r, a - 1] == clean[r, c]:
            a -= 1
        while b + 1 < clean.shape[1] and clean[r, b + 1] == clean[r, c]:
            b += 1
        following.update(x for x in (a - 1, b + 1) if 0 <= x < clean.shape[1])
    return following


def _free_scene(rng, test, shift):
    if test and shift == "scale":
        h, w = int(rng.integers(25, 31)), int(rng.integers(24, 31))
    elif not test and shift == "scale":
        h, w = int(rng.integers(15, 23)), int(rng.integers(13, 24))
    else:
        h, w = int(rng.integers(17, 31)), int(rng.integers(13, 31))
    clean = np.zeros((h, w), dtype=np.int8)
    source_count = int(rng.integers(1, min(3, (w - 1) // 6) + 1))
    launches = []
    for c in rng.permutation(np.arange(2, w - 2)):
        c = int(c)
        if any(abs(c - x[1]) < 6 for x in launches):
            continue
        explicit = int(rng.integers(2, 10)) if rng.random() < .45 else None
        launch = _source(clean, rng, c, int(rng.integers(2, 10)), explicit)
        if launch is not None:
            launches.append(launch)
            if len(launches) == source_count:
                break
    if not launches:
        return None
    injections = defaultdict(set)
    for r, c, _ in launches:
        injections[r + 2].add(c)
    first_allowed = max(r + 2 for r, c, p in launches) + 1
    budget = int(rng.integers(4, 9) if test and shift != "orientation" else rng.integers(1, 7))
    event_rows = set(map(int, rng.choice(np.arange(first_allowed, h),
                                       min(budget, h - first_allowed), replace=False)))
    active = set()
    for r in range(h):
        active.update(injections[r])
        if r in event_rows:
            pivots = list(sorted(active))
            rng.shuffle(pivots)
            quota = int(rng.integers(1, min(3, len(pivots)) + 1)) if pivots else 0
            for c in pivots[:quota]:
                if clean[r, c] != 0:
                    continue
                for _ in range(8):
                    mh = int(rng.integers(1, min(5, h - r) + 1))
                    mw = int(rng.integers(1, min(w, 10) + 1))
                    mask = _shape(rng, mh, mw)
                    offsets = list(map(int, np.flatnonzero(mask[0])))
                    offset = offsets[int(rng.integers(len(offsets)))]
                    if _put(clean, r, c - offset, mask, int(rng.integers(2, 10))):
                        break
        active = _geometry_step(clean, r, active)
    # Extra objects may alter the eventual network: labels are derived only
    # after this step, and validation does not trust causal-placement intent.
    for _ in range(int(rng.integers(0, 8))):
        r, c = int(rng.integers(first_allowed, h)), int(rng.integers(w))
        mask = _shape(rng, int(rng.integers(1, min(5, h - r) + 1)),
                      int(rng.integers(1, min(w, 9) + 1)))
        _put(clean, r, c, mask, int(rng.integers(2, 10)))
    return clean, launches


def _make_pair(rng, mode, test, shift, palette, rotation, seen):
    for _ in range(160):
        if mode == "exchange":
            construction = _witness_exchange(rng)
        elif mode == "consensus":
            construction = _witness_consensus(rng)
        else:
            construction = _free_scene(rng, test, shift)
        if construction is None:
            continue
        clean, launches = construction
        if np.count_nonzero(clean == 0) <= clean.size / 2 + 12:
            continue
        inp, arrows = _annotate(clean, launches)
        out, planned = _relay(clean, 0, 1, arrows)
        if not any(planned["assignments"][i] != planned["colors"][i]
                   for i in planned["assignments"]):
            continue
        if mode == "exchange":
            required = ("no_paint", "no_exchange", "straight", "colorwide", "latest")
        elif mode == "consensus":
            required = ("choose_one", "ignore_explicit")
        else:
            required = ()
        if any(np.array_equal(_relay(clean, 0, 1, arrows, s)[0], out) for s in required):
            continue
        contacts = len(planned["assignments"])
        if test and contacts < (5 if shift == "composition" else 2):
            continue
        if not test and shift == "composition" and contacts > 4:
            continue
        if rng.random() < .5:
            inp, out = np.fliplr(inp), np.fliplr(out)
        inp = np.rot90(palette[inp], rotation).copy()
        out = np.rot90(palette[out], rotation).copy()
        if _key(inp) in seen:
            continue
        try:
            predicted, info = _decode(inp)
        except (ValueError, AssertionError):
            continue
        if not np.array_equal(predicted, out):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(["scale", "orientation", "composition"]))
    mixed_palette = bool(rng.integers(2))
    base_palette = rng.permutation(10).astype(np.int8)
    base_rotation = int(rng.integers(4))
    mixed_directions = bool(rng.integers(2))
    held_out = int(rng.integers(4))
    train_rotations = [k for k in range(4) if k != held_out]
    rng.shuffle(train_rotations)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            test = split == "test"
            mode = ("exchange" if i == 0 else "consensus") if not test and i < 2 else "free"
            palette = rng.permutation(10).astype(np.int8) if mixed_palette else base_palette
            if shift == "orientation":
                rotation = held_out if test else train_rotations[i % 3]
            elif mixed_directions:
                rotation = int(rng.integers(4))
            else:
                rotation = base_rotation
            pair = _make_pair(rng, mode, test, shift, palette, rotation, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _rejected_shortcuts(pair):
    expected = np.asarray(pair["output"])
    rotation, clean, bg, ink, arrows = _read(pair["input"])
    return {s for s in SHORTCUTS
            if not np.array_equal(np.rot90(_relay(clean, bg, ink, arrows, s)[0], -rotation), expected)}


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, rejected = set(), set()
    train_directions, train_depths, train_contacts = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape and _key(inp) not in seen, "shape/uniqueness")
            seen.add(_key(inp))
            predicted, info = _decode(inp)
            _need(np.array_equal(predicted, out), "complete input-derived prediction")
            canonical_out = np.rot90(out, info["rotation"])
            clean, ids = info["clean"], info["ids"]
            trace = info["trace"]
            _need(np.all(ids[trace] == -1), "trace may occupy only original background")
            _need(np.all(canonical_out[trace] == info["ink"]), "trace color")
            _need(np.all(canonical_out[(ids < 0) & ~trace] == info["background"]), "unused background")
            _need(not np.any(canonical_out[ids >= 0] == info["ink"]), "island annotations removed")
            changes = 0
            for label, color in enumerate(info["colors"]):
                expected = info["assignments"].get(label, color)
                _need(np.all(canonical_out[ids == label] == expected), "whole original component paint")
                changes += expected != color
                if label in info["first_colors"] and len(info["first_colors"][label]) > 1:
                    _need(expected == color, "disagreement must retain original fill")
            _need(changes > 0 and trace.any(), "both painting and propagation must be observable")
            shortcuts = _rejected_shortcuts(pair)
            if split == "train":
                rejected.update(shortcuts)
                train_directions.add(info["rotation"])
                train_depths.append(clean.shape[0])
                train_contacts.append(len(info["assignments"]))
            else:
                _need(shortcuts <= rejected, "test-changing shortcut fits all demonstrations")
                _need(info["rotation"] not in train_directions
                      or clean.shape[0] > max(train_depths)
                      or len(info["assignments"]) > max(train_contacts),
                      "test lacks an observable orientation, depth, or contact-count shift")
    _need(rejected == set(SHORTCUTS), "training must expose all seven core distinctions")


def _features(pair):
    _, i = _decode(pair["input"])
    contacted = set(i["assignments"])
    source_ids = {a["source"] for a in i["arrows"]}
    inert = set(range(len(i["colors"]))) - contacted - source_ids
    counts = defaultdict(int)
    for r, a, b, label, payloads in i["events"]:
        counts[label] += 1
    nonrect = 0
    for points in i["cells"]:
        rr, cc = zip(*points)
        nonrect += len(points) != (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)
    return dict(shape=list(np.asarray(pair["input"]).shape), direction=i["rotation"],
                background=i["background"], ink=i["ink"], sources=len(i["arrows"]),
                objects=len(i["colors"]), contacts=len(contacted), events=len(i["events"]),
                explicit=sum(a["explicit"] for a in i["arrows"]), inert=len(inert),
                nonrect=nonrect, consensus=i["conflicts"], shared=i["shared_flanks"],
                revisit=sum(n > 1 for n in counts.values()),
                clipped=sum(a == 0 or b == i["clean"].shape[1] - 1 for r, a, b, _, _ in i["events"]),
                repeat=len(set(i["colors"])) < len(i["colors"]))


def self_test():
    """Bounded checks and measured variation over exactly the 42 host seeds."""
    signatures, shapes, geometries = set(), set(), set()
    features, within = [], []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[k]) for s in ("train", "test")
                          for p in puzzle[s] for k in ("input", "output"))
        signatures.add(signature)
        if seed == 0:
            other = generate_puzzle(seed)
            _need(signature == tuple(_key(p[k]) for s in ("train", "test")
                                     for p in other[s] for k in ("input", "output")), "determinism")
        group = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = np.asarray(pair["input"]).copy()
                _need(np.array_equal(solve(before), pair["output"]), "input-only solve")
                _need(np.array_equal(before, pair["input"]), "solve mutated its argument")
                f = _features(pair)
                f.update(seed=int(seed), split=split)
                features.append(f)
                group.append(f)
                shapes.add(tuple(f["shape"]))
                _, info = _decode(pair["input"])
                geometries.add(_key(info["ids"]))
        within.append({k: len({str(f[k]) for f in group})
                       for k in ("shape", "sources", "objects", "direction", "background", "ink")})
    _need(len(signatures) == 42, "puzzle diversity")
    _need(len(shapes) >= 40 and len(geometries) >= 170, "structural diversity")
    _need({f["direction"] for f in features} == set(range(4)), "direction variation")
    _need({f["sources"] for f in features} >= {1, 2, 3}, "source count variation")
    _need({f["background"] for f in features} == set(range(10)), "background roles")
    _need({f["ink"] for f in features} == set(range(10)), "ink roles")
    for k in ("nonrect", "consensus", "shared", "revisit", "clipped", "inert", "explicit", "repeat"):
        _need(sum(bool(f[k]) for f in features) >= 3, "missing family variation: " + k)
    _need(all(f["objects"] > 1 and f["shape"] > 1 for f in within), "within-puzzle structure variation")
    # All explicit count combinations are exercised, including smallest curriculum.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(0, nt, nq)
            _need((len(p["train"]), len(p["test"])) == (nt, nq), "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(signatures),
                grid_shapes=len(shapes), component_geometries=len(geometries),
                features=features, within_puzzle=within,
                semantic_verification="generator-owned checks; no independent solver")


if __name__ == "__main__":
    report = self_test()
    print(json.dumps({k: v for k, v in report.items() if k not in ("features", "within_puzzle")},
                     separators=(",", ":")))
