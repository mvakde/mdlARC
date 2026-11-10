"""Coincidence walls: a coarse color field keys a native-period obstacle stencil.

Only standard-library modules and NumPy are used. No files are read by generation
or solving. Coordinates, roles, scale factors and entry direction are input-visible.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer matrices, size 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        first = remaining.pop()
        todo, cells = [first], [first]
        while todo:
            r, c = todo.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _entries(arena, background):
    """Return the unique rotation that puts all straight entry prefixes on top."""
    colors = [int(x) for x in np.unique(arena) if x != background]
    if len(colors) != 1:
        raise ValueError("arena must contain background and one guide color")
    guide = colors[0]
    choices = []
    for k in range(4):
        a = np.rot90(arena, k)
        columns = np.flatnonzero(a[0] == guide)
        if not len(columns) or np.any(np.diff(columns) < 2):
            continue
        mask = np.zeros(a.shape, dtype=bool)
        for c in columns:
            rows = np.flatnonzero(a[:, c] == guide)
            if not 2 <= len(rows) < a.shape[0] or not np.array_equal(rows, np.arange(len(rows))):
                break
            mask[:len(rows), c] = True
        else:
            if np.array_equal(mask, a == guide):
                choices.append(k)
    if len(choices) != 1:
        raise ValueError("entry stubs must show exactly one inward direction")
    return guide, choices[0]


def _parse(value):
    """Recognize three complete hollow frames; no frequency-based color guesses."""
    a = _grid(value)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("outer margin must expose the common background")
    candidates = []
    for frame in (int(x) for x in np.unique(a) if x != bg):
        parts = _components(a == frame)
        if len(parts) != 3:
            continue
        boxes, covered = [], np.zeros(a.shape, dtype=bool)
        for cells in parts:
            rows, cols = zip(*cells)
            r0, r1, c0, c1 = min(rows), max(rows)+1, min(cols), max(cols)+1
            if r1-r0 < 4 or c1-c0 < 4:
                break
            patch = a[r0:r1, c0:c1]
            rim = np.ones(patch.shape, dtype=bool)
            rim[1:-1, 1:-1] = False
            if not np.array_equal(patch == frame, rim):
                break
            if np.any(covered[r0:r1, c0:c1]):
                break
            covered[r0:r1, c0:c1] = True
            boxes.append((r0, r1, c0, c1))
        else:
            if np.all(a[~covered] == bg):
                candidates.append((frame, boxes))
    if len(candidates) != 1:
        raise ValueError("expected three unambiguous separated rectangular frames")
    frame, boxes = candidates[0]
    panels = [a[r0+1:r1-1, c0+1:c1-1].copy() for r0, r1, c0, c1 in boxes]
    sizes = [x.size for x in panels]
    arena_index = int(np.argmax(sizes))
    if sizes.count(sizes[arena_index]) != 1:
        raise ValueError("arena must be uniquely largest")
    arena = panels.pop(arena_index)
    if panels[0].shape != panels[1].shape:
        raise ValueError("the two recipe cards must have equal dimensions")
    full = [i for i, p in enumerate(panels) if np.all(p != bg)]
    if len(full) != 1:
        raise ValueError("carrier is full; stencil has transparent background holes")
    carrier, stencil = panels[full[0]], panels[1-full[0]]
    guide, rotation = _entries(arena, bg)
    if np.any(carrier == guide) or np.any(stencil == guide):
        raise ValueError("guide ink is reserved for entry stubs and routes")
    n, m = carrier.shape
    h, w = arena.shape
    if h % n or w % m or h < n or w < m:
        raise ValueError("arena must be an integral replication of the carrier")
    base, blocked = _fields(carrier, stencil, h, w, bg)
    if np.any(blocked[arena == guide]):
        raise ValueError("entry prefixes cannot start on a coincidence wall")
    return dict(background=bg, frame=frame, guide=guide, rotation=rotation,
                carrier=carrier, stencil=stencil, arena=arena, base=base, blocked=blocked)


def _fields(carrier, stencil, h, w, background, variant=None):
    n, m = carrier.shape
    rr, cc = np.arange(h)[:, None], np.arange(w)[None, :]
    sr, sc = rr*n//h, cc*m//w
    base = carrier[sr, sc]
    if variant == "tile_carrier_rows":
        base = carrier[rr % n, sc]
    elif variant == "tile_carrier_columns":
        base = carrier[sr, cc % m]
    if variant == "scale_stencil":
        stamp = stencil[sr, sc]
    else:
        stamp = stencil[(rr + int(variant == "phase_rows")) % n,
                        (cc + int(variant == "phase_columns")) % m]
    blocked = (stamp != background) & (stamp == base)
    if variant == "all_opaque":
        blocked = stamp != background
    elif variant == "mismatches":
        blocked = (stamp != background) & (stamp != base)
    elif variant == "ignore_walls":
        blocked = np.zeros((h, w), dtype=bool)
    return base.copy(), blocked


def _route(blocked, arena, guide, rotation, variant=None):
    """Layer propagation on immutable coincidence walls, with clipped flank sets."""
    wall = np.rot90(blocked, rotation).copy()
    stubs = np.rot90(arena == guide, rotation)
    if variant == "ignore_singletons":
        left = np.pad(wall[:, :-1], ((0, 0), (1, 0)))
        right = np.pad(wall[:, 1:], ((0, 0), (0, 1)))
        wall &= left | right
    # Wrong-rule probes also preserve visibly given entry pixels.
    wall[stubs] = False
    path = stubs.copy()
    h, w = wall.shape
    active = set(map(int, np.flatnonzero(stubs[0])))
    path[0, list(active)] = True
    events = hits_total = split_count = clipped = simultaneous = merges = 0
    reached = set()
    for r in range(1, h):
        hits, following = {}, set()
        for c in sorted(active):
            if not wall[r, c]:
                path[r, c] = True
                following.add(c)
                continue
            lo = hi = c
            while lo and wall[r, lo-1]:
                lo -= 1
            while hi+1 < w and wall[r, hi+1]:
                hi += 1
            hits.setdefault((lo, hi), []).append(c)
        if variant == "outline_all":
            c = 0
            while c < w:
                if not wall[r, c]:
                    c += 1
                    continue
                lo = c
                while c+1 < w and wall[r, c+1]:
                    c += 1
                hits.setdefault((lo, c), [])
                c += 1
        if variant == "one_per_layer":
            hits = dict(sorted(hits.items())[:1])
        simultaneous += len(hits) >= 2
        for (lo, hi), incident in hits.items():
            events += 1
            hits_total += len(incident)
            reached.add((r, lo, hi))
            split_count += lo > 0 and hi < w-1
            clipped += lo == 0 or hi == w-1
            if variant != "no_crosspiece":
                span = slice(max(0, lo-1), min(w, hi+2))
                path[r-1, span] |= ~wall[r-1, span]
            if variant != "no_split":
                for c in (lo-1, hi+1):
                    if variant == "wrap_edges":
                        c %= w
                    if 0 <= c < w and not wall[r, c]:
                        merges += c in following
                        following.add(c)
                        path[r, c] = True
        active = following
    total_runs = sum(int(row[0]) + int(np.sum(row[1:] & ~row[:-1])) for row in wall)
    stats = dict(collisions=events, incident_hits=hits_total, two_flank=split_count,
                 clipped=clipped, simultaneous=simultaneous, merges=merges,
                 unreached_runs=total_runs-len(reached), path_cells=int(path.sum()),
                 wall_cells=int(wall.sum()), entries=int(stubs[0].sum()))
    return np.rot90(path, -rotation).copy(), stats


def _render(parsed, variant=None):
    d = parsed
    h, w = d["arena"].shape
    base, walls = _fields(d["carrier"], d["stencil"], h, w, d["background"], variant)
    path, stats = _route(walls, d["arena"], d["guide"], d["rotation"], variant)
    if variant == "no_rays":
        path = d["arena"] == d["guide"]
    out = base.copy()
    out[path] = d["guide"]
    return out, path, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only the input; output the developed carrier with its branching route."""
    return _render(_parse(input_grid))[0]


_MUTANTS = ("no_rays", "ignore_walls", "all_opaque", "mismatches",
            "tile_carrier_rows", "tile_carrier_columns", "scale_stencil",
            "phase_rows", "phase_columns", "no_split", "no_crosspiece", "outline_all",
            "one_per_layer", "ignore_singletons", "wrap_edges")
_REQUIRED = frozenset(_MUTANTS + ("carrier_scale_changes_route", "stencil_scale_changes_route"))


def _evidence(d, output):
    _, correct_path, _ = _render(d)
    evidence = set()
    for variant in _MUTANTS:
        wrong, path, _ = _render(d, variant)
        if not np.array_equal(wrong, output):
            evidence.add(variant)
        if not np.array_equal(path, correct_path):
            if variant in ("tile_carrier_rows", "tile_carrier_columns"):
                evidence.add("carrier_scale_changes_route")
            if variant == "scale_stencil":
                evidence.add("stencil_scale_changes_route")
    return evidence


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalized(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(x), len(roles)) for x in a.flat)


def _recipe(rng, n, m, inks, bg):
    """Independent coarse image arrangements and sparse native stencil motifs."""
    k = int(rng.integers(2, min(4, len(inks))+1))
    colors = rng.choice(inks, k, replace=False)
    rr, cc = np.indices((n, m))
    style = int(rng.integers(6))
    if style == 0:
        labels = np.broadcast_to(rng.integers(k, size=(n, 1)), (n, m)).copy()
    elif style == 1:
        labels = np.broadcast_to(rng.integers(k, size=(1, m)), (n, m)).copy()
    elif style == 2:
        labels = (rr + cc + int(rng.integers(k))) % k
    elif style == 3:
        labels = rng.integers(k, size=(n, m))
    elif style == 4:
        labels = np.zeros((n, m), dtype=int)
        for _ in range(int(rng.integers(2, 6))):
            r0, r1 = sorted(map(int, rng.choice(n+1, 2, replace=False)))
            c0, c1 = sorted(map(int, rng.choice(m+1, 2, replace=False)))
            labels[r0:r1, c0:c1] = int(rng.integers(k))
    else:
        seeds = rng.choice(n*m, k, replace=False)
        distances = abs(rr[..., None]-seeds//m) + abs(cc[..., None]-seeds % m)
        labels = np.argmin(distances, axis=2)
    if len(np.unique(labels)) == 1:
        labels.ravel()[int(rng.integers(n*m))] = (int(labels[0, 0])+1) % k
    carrier = colors[labels].astype(np.int8)
    kind = int(rng.integers(6))
    if kind == 0:
        mask = rng.random((n, m)) < rng.uniform(.15, .60)
    elif kind == 1:
        mask = (rr == int(rng.integers(n))) & (cc >= int(rng.integers(m)))
    elif kind == 2:
        mask = (cc == int(rng.integers(m))) & (rr >= int(rng.integers(n)))
    elif kind == 3:
        mask = (rr+cc+int(rng.integers(2))) % 2 == 0
        mask &= rng.random((n, m)) < .80
    elif kind == 4:
        mask = (rr == 0) | (rr == n-1) | (cc == 0) | (cc == m-1)
        mask &= rng.random((n, m)) < .60
    else:
        mask = np.zeros((n, m), dtype=bool)
        r, c = int(rng.integers(n)), int(rng.integers(m))
        for _ in range(int(rng.integers(2, n*m+1))):
            mask[r, c] = True
            r = min(n-1, max(0, r+int(rng.integers(-1, 2))))
            c = min(m-1, max(0, c+int(rng.integers(-1, 2))))
    if not mask.any():
        mask.ravel()[int(rng.integers(n*m))] = True
    if mask.all():
        mask.ravel()[int(rng.integers(n*m))] = False
    stencil = np.full((n, m), bg, dtype=np.int8)
    used = np.unique(carrier)
    stamp_inks = rng.choice(used, int(rng.integers(1, len(used)+1)), replace=False)
    # An extra unmatched key is allowed, without tying ink count to density.
    extras = [c for c in inks if c not in used]
    if extras and rng.random() < .25:
        stamp_inks = np.append(stamp_inks, rng.choice(extras))
    stencil[mask] = rng.choice(stamp_inks, int(mask.sum()))
    return carrier, stencil


def _layout(rng, carrier, stencil, arena, bg, frame):
    n, m = carrier.shape
    h, w = arena.shape
    rows, cols = n+h+7, max(2*m+5, w+2)+2
    canvas = np.full((rows, cols), bg, dtype=np.int8)
    def put(panel, r, c):
        ph, pw = panel.shape
        canvas[r:r+ph+2, c:c+pw+2] = frame
        canvas[r+1:r+ph+1, c+1:c+pw+1] = panel
    offset = int(rng.integers(1, cols-(2*m+5)))
    cards = [carrier, stencil]
    if rng.random() < .5:
        cards.reverse()
    put(cards[0], 1, offset)
    put(cards[1], 1, offset+m+3)
    put(arena, n+4, int(rng.integers(1, cols-w-2)))
    # Spatial parsing is independent of card order and page orientation.
    canvas = np.rot90(canvas, int(rng.integers(4)))
    if rng.random() < .5:
        canvas = np.fliplr(canvas)
    return canvas.copy()


def _sample_pair(rng, palette, test=False, recipe=None):
    bg, frame, guide, *inks = palette
    for _ in range(96):
        if recipe is None:
            n, m = int(rng.integers(2, 6)), int(rng.integers(2, 7))
            sy = int(rng.integers(2, min(3, (23-n)//n)+1))
            sx = int(rng.integers(4, min(6, 26//m)+1)) if test else int(rng.integers(2, 4))
            carrier, stencil = _recipe(rng, n, m, inks, bg)
        else:
            carrier, stencil, sy, sx = recipe
            n, m = carrier.shape
        h, w = n*sy, m*sx
        base, wall = _fields(carrier, stencil, h, w, bg)
        rotation = int(rng.integers(4))
        cw = np.rot90(wall, rotation)
        depth, width = cw.shape
        length = int(rng.integers(2, min(4, depth)))
        eligible = [int(c) for c in np.flatnonzero(~cw[:length].any(axis=0))]
        rng.shuffle(eligible)
        columns = []
        desired = int(rng.integers(1, min(4, (width+1)//2)+1))
        for c in eligible:
            if all(abs(c-other) >= 2 for other in columns):
                columns.append(c)
                if len(columns) == desired:
                    break
        if not columns:
            continue
        canonical = np.full((depth, width), bg, dtype=np.int8)
        canonical[:length, columns] = guide
        arena = np.rot90(canonical, -rotation).copy()
        path, stats = _route(wall, arena, guide, rotation)
        if not stats["two_flank"] or not stats["unreached_runs"]:
            continue
        if stats["path_cells"] <= int(np.sum(arena == guide)) or path.mean() > .74:
            continue
        _, opaque = _fields(carrier, stencil, h, w, bg, "all_opaque")
        opaque_path, _ = _route(opaque, arena, guide, rotation)
        if np.array_equal(path, opaque_path):
            continue
        a = _layout(rng, carrier, stencil, arena, bg, frame)
        d = _parse(a)
        out, _, _ = _render(d)
        return {"input": a, "output": out}
    return None


def _fallback_pair(rng, palette, test, serial=0):
    """Finite recovery pool. Values are a recipe, never an output lookup table."""
    bg, frame, guide, a, b, c, *_ = palette
    recipes = (
        ([[a, b, c, a], [b, a, b, c], [c, c, a, b]],
         [[bg, a, bg, b], [b, bg, c, bg], [bg, c, bg, a]]),
        ([[a, a, b], [b, c, a], [c, b, c], [a, c, b]],
         [[b, bg, bg], [bg, c, a], [a, bg, c], [bg, b, bg]]),
    )
    # Local deterministic streams give a bounded, distinct geometric candidate
    # pool even if the exploratory sampler was unlucky. No disk/cache state.
    for i in range(64):
        first, second = recipes[(serial+i) % 2]
        local = np.random.default_rng(91271 + 997*serial + i)
        recipe = (np.array(first, dtype=np.int8), np.array(second, dtype=np.int8),
                  2 + (i % 2), 4 + (i % 2) if test else 2 + ((i//2) % 2))
        pair = _sample_pair(local, palette, test, recipe)
        if pair is not None:
            return pair
    raise RuntimeError("finite geometric recovery pool exhausted")


def _validate_args(seed, num_train, num_test):
    for name, value in (("seed", seed), ("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")


def _recover_training(rng, palette):
    recovery = []
    for serial in range(24):
        pair = _fallback_pair(rng, palette, False, serial)
        d = _parse(pair["input"])
        evidence = _evidence(d, pair["output"])
        for old, oe, shape in recovery:
            if _REQUIRED <= evidence | oe and shape != d["carrier"].shape:
                return [old, pair]
        recovery.append((pair, evidence, d["carrier"].shape))
    raise RuntimeError("bounded teaching recovery failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _validate_args(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, 8, replace=False)))
    # Retain at most 12 small candidates, and select two complementary witnesses.
    candidates = []
    core = None
    for attempt in range(96):
        pair = _sample_pair(rng, palette)
        if pair is None:
            continue
        d = _parse(pair["input"])
        evidence = _evidence(d, pair["output"])
        for previous, pe, pd in candidates:
            if (_REQUIRED <= evidence | pe
                    and _key(previous["input"]) != _key(pair["input"])
                    and d["carrier"].shape != pd["carrier"].shape
                    and _normalized(d["carrier"]) != _normalized(pd["carrier"])):
                core = [previous, pair]
                break
        if core is not None:
            break
        candidates.append((pair, evidence, d))
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        candidates = candidates[:12]
    if core is None:
        core = _recover_training(rng, palette)
    puzzle = {"train": core, "test": []}
    seen = {_key(p["input"]) for p in core}
    for split, count in (("train", num_train), ("test", num_test)):
        while len(puzzle[split]) < count:
            pair = None
            for attempt in range(24):
                candidate = _sample_pair(rng, palette, split == "test")
                if candidate is not None and _key(candidate["input"]) not in seen:
                    pair = candidate
                    break
            if pair is None:
                for serial in range(48):
                    candidate = _fallback_pair(rng, palette, split == "test", serial)
                    if _key(candidate["input"]) not in seen:
                        pair = candidate
                        break
            if pair is None:
                raise RuntimeError("bounded unique-input recovery failed")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
           and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen, roles, evidence, train_shapes = set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, o = _grid(pair["input"]), _grid(pair["output"])
            _check(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            d = _parse(a)
            expected, path, stats = _render(d)
            _check(np.array_equal(expected, o), "incorrect route or carrier rendering")
            _check(o.shape == d["arena"].shape, "output is only the arena interior")
            _check(np.all(o[d["blocked"]] == d["base"][d["blocked"]]), "walls were overwritten")
            _check(np.all(o[~path] == d["base"][~path]), "off-route carrier changed")
            _check(np.all(o[d["arena"] == d["guide"]] == d["guide"]), "original guide lost")
            _check(stats["two_flank"] and stats["unreached_runs"], "missing branching or reachability contrast")
            _, wrong_path, _ = _render(d, "all_opaque")
            _check(not np.array_equal(path, wrong_path), "color coincidence has no causal effect")
            roles.add((d["background"], d["frame"], d["guide"]))
            n, m = d["carrier"].shape
            scales = sorted((o.shape[0]//n, o.shape[1]//m))
            if split == "train":
                evidence |= _evidence(d, o)
                train_shapes.add(d["carrier"].shape)
                _check(max(scales) <= 3, "training exceeds its scale support")
            else:
                _check(max(scales) >= 4, "test must hold out a larger replication factor")
    _check(len(roles) == 1, "marker roles must be consistent within a puzzle")
    _check(len(train_shapes) >= 2, "training needs two recipe geometries")
    _check(_REQUIRED <= evidence, "training leaves shortcuts: " + str(sorted(_REQUIRED-evidence)))


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, finite mutation probes and hand-checkable local regressions.

    Runs in memory and returns only aggregate diagnostics; writes no files.
    This is generator-owned verification, not an independent ARC evaluation.
    """
    wall = np.zeros((7, 11), dtype=bool)
    wall[2, 2:5] = wall[2, 6:9] = True
    arena = np.zeros(wall.shape, dtype=np.int8)
    arena[:2, [3, 7]] = 1
    path, stats = _route(wall, arena, 1, 0)
    expected = np.zeros_like(path)
    expected[:2, [3, 7]] = True
    expected[1, 1:10] = True
    expected[2:, [1, 5, 9]] = True
    _check(np.array_equal(path, expected) and stats["simultaneous"] == 1
           and stats["merges"] == 1, "two runs and a shared flank regression")
    # Clipping and blocked upstream pixels: a crosspiece cannot erase a wall.
    wall = np.zeros((6, 7), dtype=bool)
    wall[2, :3] = True
    wall[3, 2:6] = True
    arena = np.zeros(wall.shape, dtype=np.int8)
    arena[:2, 1] = 1
    path, stats = _route(wall, arena, 1, 0)
    _check(not path[2, 2] and path[2, 1] == 0 and path[2, 6]
           and path[3, 1] and path[3, 6] and stats["clipped"] == 1,
           "clipping and interrupted crosspiece regression")
    c = np.array([[2, 3], [3, 2]], dtype=np.int8)
    s = np.array([[0, 2], [3, 0]], dtype=np.int8)
    base, wall = _fields(c, s, 4, 6, 0)
    _check(np.array_equal(base[0], [2, 2, 2, 3, 3, 3])
           and wall[0, 1] and not wall[0, 3] and wall[1, 4]
           and not wall[1, 0], "two-scale color-equality regression")
    hashes, shapes, carriers, masks, answers, directions = set(), set(), set(), set(), set(), set()
    roles = [set(), set(), set()]
    counts = dict(collisions=0, simultaneous=0, clipped=0, merges=0, pairs=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _parse(pair["input"])
                output, path, stats = _render(d)
                _check(np.array_equal(solve(pair["input"]), output), "input-only prediction mismatch")
                shapes.add(output.shape)
                carriers.add(_normalized(d["carrier"]))
                masks.add(_key(d["stencil"] != d["background"]))
                directions.add(d["rotation"])
                for values, role in zip(roles, ("background", "frame", "guide")):
                    values.add(d[role])
                for name in ("collisions", "simultaneous", "clipped", "merges"):
                    counts[name] += stats[name]
                counts["pairs"] += 1
                if split == "test":
                    answers.add(_normalized(output))
        if seed < 3:
            _check(_digest(puzzle) == _digest(generate_puzzle(seed)), "seed replay mismatch")
    _check(len(hashes) == 42 and len(answers) == 42, "seed/output diversity collapsed")
    _check(len(shapes) >= 30 and len(carriers) >= 80 and len(masks) >= 60, "structural variation collapsed")
    _check(directions == set(range(4)) and all(len(r) >= 8 for r in roles), "orientation/palette coverage")
    _check(all(counts[k] > 0 for k in ("simultaneous", "clipped", "merges")), "missing collision cases")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(8100+10*nt+nq, nt, nq)
            _check(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts ignored")
    # Force the clean recovery path, then exercise the input's symmetries.
    rng = np.random.default_rng(82019)
    palette = [0, 9, 1, 2, 3, 4, 5, 6]
    core = _recover_training(rng, palette)
    fallback_test = _fallback_pair(rng, palette, True)
    validate_generated_puzzle({"train": core, "test": [fallback_test]})
    for split in (False, True):
        recovered = {_key(_fallback_pair(rng, palette, split, i)["input"]) for i in range(24)}
        _check(len(recovered) >= 14, "not enough distinct fallback inputs")
    for pair in core + [fallback_test]:
        a, o = pair["input"], pair["output"]
        for k in range(4):
            transformed = np.rot90(a, k).copy()
            _check(np.array_equal(solve(transformed), np.rot90(o, k)), "rotation covariance failed")
            _check(np.array_equal(solve(np.fliplr(transformed)), np.fliplr(np.rot90(o, k))),
                   "reflection covariance failed")
        perm = rng.permutation(10)
        _check(np.array_equal(solve(perm[a]), perm[o]), "palette-role covariance failed")
    # A label mutation must be rejected; checks do not merely accept schema.
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in pairs]
              for s, pairs in {"train": core, "test": [fallback_test]}.items()}
    broken["test"][0]["output"][0, 0] = 9
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, distinct_puzzles=len(hashes), output_shapes=len(shapes),
                carrier_structures=len(carriers), stencil_masks=len(masks),
                distinct_test_answers=len(answers), directions=len(directions),
                marker_role_colors=[len(r) for r in roles], **counts)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
