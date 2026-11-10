"""Stained rays through hidden panes. No parent modules or external data needed."""
from __future__ import annotations

import numpy as np


SHORTCUTS = ("markers_only", "no_split", "no_dye", "reverse_layers",
             "transparent_windows", "union_runs", "last_priority", "keep_best_stain")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _corners(bounds, diagonal):
    a, b, z, t = bounds
    if diagonal == 0:
        return {(a, b), (a, b+1), (a+1, b), (z, t), (z, t-1), (z-1, t)}
    return {(a, t), (a, t-1), (a+1, t), (z, b), (z, b+1), (z-1, b)}


def _recognize(cells):
    if len(cells) != 6:
        raise ValueError("a pane requires two three-cell L corner clues")
    a, b = min(r for r, c in cells), min(c for r, c in cells)
    z, t = max(r for r, c in cells), max(c for r, c in cells)
    h, w = z-a+1, t-b+1
    if min(h, w) < 2 or (min(h, w) == 2 and max(h, w) < 4):
        raise ValueError("ambiguous thin pane")
    for diagonal in (0, 1):
        if cells == _corners((a, b, z, t), diagonal):
            return (a, b, z, t), diagonal
    raise ValueError("clues are not opposite inward-facing corners")


def _parse(value):
    grid = _grid(value)
    h, w = grid.shape
    if w < 7 or h < 4 or not np.all(grid[:, 2] == grid[0, 2]):
        raise ValueError("missing uniform separator column")
    bg = int(grid[0, 2])
    rows = np.flatnonzero(grid[:, 0] != bg)
    if not 2 <= len(rows) <= 7 or not np.array_equal(rows, np.arange(rows[0], rows[-1]+1)):
        raise ValueError("legend must have 2..7 consecutive rows")
    mask = np.zeros(h, dtype=bool)
    mask[rows] = True
    if np.any(grid[~mask, :2] != bg):
        raise ValueError("nonempty cells outside legend")
    cues = list(map(int, grid[rows, 0]))
    fills = list(map(int, grid[rows, 1]))
    if len(set(cues)) != len(cues):
        raise ValueError("cue colors must be distinct")
    scene = grid[:, 3:].copy()
    remaining = set(map(int, np.unique(scene))) - {bg, *cues}
    if len(remaining) != 1:
        raise ValueError("guide color must be uniquely identifiable in the scene")
    guide = remaining.pop()
    directions = []
    for k in range(4):
        a = np.rot90(scene, k)
        entries = list(map(int, np.flatnonzero(a[0] == guide)))
        if not entries or any(y-x < 2 for x, y in zip(entries, entries[1:])):
            continue
        expected = np.zeros(a.shape, dtype=bool)
        lengths = []
        for c in entries:
            rr = np.flatnonzero(a[:, c] == guide)
            if len(rr) < 2 or len(rr) >= a.shape[0] or not np.array_equal(rr, np.arange(len(rr))):
                break
            expected[:len(rr), c] = True
            lengths.append(len(rr))
        else:
            if np.array_equal(expected, a == guide):
                directions.append((k, a.copy(), entries, lengths))
    if len(directions) != 1:
        raise ValueError("entry direction is ambiguous")
    k, canonical, entries, lengths = directions[0]
    objects = []
    for rank, (cue, fill) in enumerate(zip(cues, fills)):
        cells = {tuple(map(int, rc)) for rc in np.argwhere(canonical == cue)}
        if not cells:  # An unused legend row is an explicitly harmless extension.
            continue
        bounds, diagonal = _recognize(cells)
        a, b, z, t = bounds
        if np.any(canonical[a:z+1, b:t+1] == guide):
            raise ValueError("a pane cannot cover an original entry stub")
        objects.append(dict(rank=rank, cue=cue, fill=fill, bounds=bounds,
                            cells=cells, diagonal=diagonal))
    if not objects:
        raise ValueError("empty pane scene")
    return dict(grid=grid, bg=bg, guide=guide, cues=cues, fills=fills,
                rows=rows, scene=canonical, rotation=k, entries=entries,
                lengths=lengths, objects=objects)


def _owners(p, shortcut=None):
    """First covering legend row owns a cell, even when it opens a window."""
    owners = np.full(p["scene"].shape, -1, dtype=np.int16)
    stack = p["objects"]
    if shortcut == "reverse_layers":
        stack = list(reversed(stack))
    for obj in reversed(stack):
        if shortcut == "transparent_windows" and obj["fill"] == p["bg"]:
            continue
        a, b, z, t = obj["bounds"]
        tag = -1 if obj["fill"] == p["bg"] else obj["rank"]
        if shortcut == "markers_only":
            for r, c in obj["cells"]:
                owners[r, c] = tag
        else:
            owners[a:z+1, b:t+1] = tag
    return owners


def _trace(p, shortcut=None):
    """Forward simulation on virtual ownership, with rank-tagged ray memory."""
    src = p["scene"]
    h, w = src.shape
    owner = _owners(p, shortcut)
    out = src.copy()
    neutral = len(p["cues"])
    weights = np.full(src.shape, -1, dtype=np.int16)
    active = {c: neutral for c in p["entries"]}
    events, mergers, lost, two_flanks, simultaneous = [], 0, 0, 0, 0
    preference = max if shortcut == "last_priority" else min

    def ink(tag):
        return p["guide"] if tag == neutral or shortcut == "no_dye" else p["fills"][tag]

    def draw(r, c, tag):
        if 0 <= r < h and 0 <= c < w and src[r, c] == p["bg"] and owner[r, c] == -1:
            previous = int(weights[r, c])
            winner = tag if previous == -1 else preference(previous, tag)
            weights[r, c] = winner
            out[r, c] = ink(winner)

    for r in range(h):
        hits, following = {}, {}

        def join(c, tag):
            nonlocal mergers
            if c in following:
                mergers += following[c] != tag
                following[c] = preference(following[c], tag)
            else:
                following[c] = tag

        for c, tag in sorted(active.items()):
            material = int(owner[r, c])
            if material < 0:
                draw(r, c, tag)
                join(c, tag)
                continue
            left = right = c
            if shortcut == "union_runs":
                while left > 0 and owner[r, left-1] >= 0:
                    left -= 1
                while right+1 < w and owner[r, right+1] >= 0:
                    right += 1
            else:
                while left > 0 and owner[r, left-1] == material:
                    left -= 1
                while right+1 < w and owner[r, right+1] == material:
                    right += 1
            hits.setdefault((left, right, material), []).append(c)
        simultaneous += len(hits) >= 2
        for (left, right, material), columns in sorted(hits.items()):
            events.append((r, left, right, material, tuple(columns)))
            if shortcut == "no_split":
                continue
            emitted = min([material] + [active[c] for c in columns]) if shortcut == "keep_best_stain" else material
            for c in range(max(0, left-1), min(w, right+2)):
                draw(r-1, c, emitted)
            clear_flanks = 0
            for c in (left-1, right+1):
                if 0 <= c < w and owner[r, c] == -1:
                    join(c, emitted)
                    draw(r, c, emitted)
                    clear_flanks += 1
                else:
                    lost += 1
            two_flanks += clear_flanks == 2
        # Drawing the surviving tag again resolves both mergers and collisions
        # with a previously drawn crosspiece consistently.
        for c, tag in following.items():
            draw(r, c, tag)
        active = following
    result = p["grid"].copy()
    result[:, 3:] = np.rot90(out, -p["rotation"])
    metrics = dict(events=events, mergers=mergers, lost_flanks=lost,
                   two_flanks=two_flanks, simultaneous=simultaneous,
                   reached=len({e[3] for e in events}),
                   panes=len(p["objects"]), windows=sum(o["fill"] == p["bg"] for o in p["objects"]),
                   unused=len(p["cues"])-len(p["objects"]),
                   sizes=[(o["bounds"][2]-o["bounds"][0]+1,
                           o["bounds"][3]-o["bounds"][1]+1) for o in p["objects"]])
    return result, metrics


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, geometry, priority and every output pixel from this input."""
    return _trace(_parse(input_grid))[0]


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return (a.shape, a.tobytes())


def _assemble(rng, h, w, bounds, window_ids, order, entries, rotation, unused=False):
    # All literal identities are sampled locally; only spatial conventions persist.
    count = len(bounds)
    colors = list(map(int, rng.permutation(10)))
    bg, guide = colors[:2]
    cues = colors[2:2+count+int(unused)]
    paints = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, guide)])))
    fills = [bg if i in window_ids else paints[i % len(paints)] for i in range(count)]
    if count > 3 and rng.random() < .35:
        solids = [i for i in range(count) if i not in window_ids]
        if len(solids) >= 2:
            fills[solids[-1]] = fills[solids[0]]
    scene = np.full((h, w), bg, dtype=np.int8)
    used = set()
    for i, box in enumerate(bounds):
        options = list(map(int, rng.permutation(2)))
        for diagonal in options:
            cells = _corners(box, diagonal)
            if not cells & used:
                break
        else:
            return None
        used.update(cells)
        for r, c in cells:
            scene[r, c] = cues[i]
    first = min(box[0] for box in bounds)
    for c in entries:
        length = int(rng.integers(2, min(4, first-1)+1))
        if any(a <= length-1 and b <= c <= t for a, b, z, t in bounds):
            return None
        scene[:length, c] = guide
    if rng.random() < .5:
        scene = np.fliplr(scene).copy()
    scene = np.rot90(scene, -rotation).copy()
    sh, sw = scene.shape
    grid = np.full((sh, sw+3), bg, dtype=np.int8)
    grid[:, 3:] = scene
    table = [(cues[i], fills[i]) for i in order]
    if unused:
        table.insert(int(rng.integers(len(table)+1)), (cues[-1], int(rng.integers(10))))
    start = int(rng.integers(sh-len(table)+1))
    grid[start:start+len(table), :2] = table
    return grid


def _window_witness(rng, rotation):
    h, w = int(rng.integers(21, 28)), int(rng.integers(17, 28))
    top = int(rng.integers(4, 7))
    bottom = min(h-8, top+int(rng.integers(5, 9)))
    left, right = int(rng.integers(1, 4)), w-int(rng.integers(2, 5))
    wl = int(rng.integers(left+3, right-3))
    wr = min(right-3, wl+int(rng.integers(1, 4)))
    wt, wb = top-int(rng.integers(2)), bottom-int(rng.integers(1, 3))
    lt = bottom+int(rng.integers(2, 4))
    bounds = [(top, left, bottom, right), (wt, wl, wb, wr),
              (lt, 0, min(h-1, lt+int(rng.integers(2, 6))), left+int(rng.integers(1, 4)))]
    order = list(map(int, rng.permutation(3)))
    if order.index(1) > order.index(0):
        x, y = order.index(1), order.index(0)
        order[x], order[y] = order[y], order[x]
    return _assemble(rng, h, w, bounds, {1}, order, [left+1, (wl+wr)//2], rotation,
                     unused=bool(rng.random() < .4))


def _meeting_witness(rng, rotation):
    h, w = int(rng.integers(21, 28)), int(rng.integers(17, 28))
    gap = int(rng.integers(8, w-6))
    top = int(rng.integers(4, 7))
    lo, hi = int(rng.integers(3, 5)), int(rng.integers(gap+4, w))
    lb = min(h-6, top+int(rng.integers(5, 9)))
    rb = min(h-5, top+int(rng.integers(6, 10)))
    bottom = min(h-1, max(lb, rb)+int(rng.integers(2, 5)))
    bounds = [(top, lo, lb, gap-1), (top, gap+1, rb, hi),
              (top+4, gap-int(rng.integers(2, 4)),
               bottom, gap+int(rng.integers(2, 4))),
              (top+1, 0, top+3, lo-1)]
    # This short edge pane removes the outside left branch before the middle
    # gap closes. A union-of-material shortcut would wrongly resurrect it.
    order = [int(i) for i in rng.permutation([0, 1, 3])] + [2]
    entries = [int(rng.integers(lo+1, gap-1)), int(rng.integers(gap+2, hi))]
    return _assemble(rng, h, w, bounds, set(), order, entries, rotation,
                     unused=bool(rng.random() < .4))


def _free_scene(rng, rotation, split, scale):
    lo = 23 if split == "test" and scale else 15
    h, w = int(rng.integers(lo, 28)), int(rng.integers(15, 28))
    n = int(rng.integers(4, 7) if split == "test" else rng.integers(2, 6))
    bounds, used = [], set()
    for i in range(n):
        for _ in range(90):
            shape = int(rng.integers(4))
            hh = int(rng.integers(2 if shape == 0 else 3, h-4))
            ww = int(rng.integers(2 if shape == 1 else 3, w+1))
            if min(hh, ww) == 2 and max(hh, ww) < 4:
                continue
            top = int(rng.integers(5, h-hh+1))
            left = int(rng.integers(w-ww+1))
            box = (top, left, top+hh-1, left+ww-1)
            if box in bounds:
                continue
            d = int(rng.integers(2))
            cells = _corners(box, d)
            if cells & used:
                continue
            bounds.append(box)
            used.update(cells)
            break
        else:
            return None
    windows = {i for i in range(n) if rng.random() < .22}
    if len(windows) == n:
        windows.remove(int(rng.integers(n)))
    slots = list(map(int, rng.permutation(w)))
    entries = []
    quota = int(rng.integers(1, 5))
    for c in slots:
        if all(abs(c-d) >= 2 for d in entries):
            entries.append(c)
            if len(entries) == quota:
                break
    return _assemble(rng, h, w, bounds, windows, list(map(int, rng.permutation(n))),
                     sorted(entries), rotation, unused=bool(rng.random() < .25))


def _make_pair(rng, kind, rotation, seen, split="train", scale=False):
    required = {"markers_only", "no_split", "no_dye", "reverse_layers", "transparent_windows"} if kind == "window" else (
        {"union_runs", "last_priority", "keep_best_stain"} if kind == "meeting" else set())
    for _ in range(180):
        grid = (_window_witness(rng, rotation) if kind == "window" else
                _meeting_witness(rng, rotation) if kind == "meeting" else
                _free_scene(rng, rotation, split, scale))
        if grid is None or _signature(grid) in seen:
            continue
        try:
            parsed = _parse(grid)
        except ValueError:
            continue
        output, metrics = _trace(parsed)
        if np.array_equal(output, grid):
            continue
        if kind == "free" and split == "test" and (len(metrics["events"]) < 2 or not metrics["two_flanks"]):
            continue
        if any(np.array_equal(_trace(parsed, s)[0], output) for s in required):
            continue
        return dict(input=grid, output=output)
    return _fallback(kind, rotation, seen, split, scale)


def _fallback(kind, rotation, seen, split, scale):
    """Finite clean recovery family: at least fifty inputs for fourteen slots.

    Only used after spatial rejection; no random loop can exhaust the API.
    The free-scene fallback recombines the four-pane meeting demonstration.
    """
    window = kind == "window"
    if window:
        boxes = [(4, 2, 10, 16), (3, 7, 8, 10), (13, 0, 17, 4)]
        cues, fills, order, entries = [2, 3, 4], [5, 0, 6], [1, 0, 2], [3, 8]
    else:
        boxes = [(4, 3, 10, 8), (4, 10, 11, 17), (8, 6, 15, 12), (5, 0, 7, 2)]
        cues, fills, order, entries = [2, 3, 4, 5], [6, 7, 8, 9], [0, 1, 3, 2], [5, 13]
    for h in range(23 if split == "test" and scale else 21, 28):
        scene = np.zeros((h, 19), dtype=np.int8)
        scene[:2, entries] = 1
        for cue, box in zip(cues, boxes):
            for r, c in _corners(box, 0):
                scene[r, c] = cue
        scene = np.rot90(scene, -rotation).copy()
        grid = np.zeros((scene.shape[0], scene.shape[1]+3), dtype=np.int8)
        grid[:, 3:] = scene
        grid[:len(order), :2] = [(cues[i], fills[i]) for i in order]
        for offset in range(10):
            candidate = (grid+offset) % 10
            if _signature(candidate) not in seen:
                return dict(input=candidate, output=solve(candidate))
    raise AssertionError("more exclusions than the legal maximum of fourteen pairs")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 training, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    direction = int(rng.integers(4))
    mixed = bool(rng.random() < .6)
    scale = bool(rng.random() < .3)
    seen, puzzle = set(), {"train": [], "test": []}
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            rotation = (direction+i+int(split == "test")) % 4 if mixed else direction
            kind = ("window" if i == 0 else "meeting") if split == "train" and i < 2 else "free"
            pair = _make_pair(rng, kind, rotation, seen, split, scale)
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Array-derived geometry, protected clues, causal and teaching invariants."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, rejected = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, o = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == o.shape and not np.array_equal(a, o)
            key = _signature(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            parsed = _parse(a)
            expected, metrics = _trace(parsed)
            assert np.array_equal(expected, o), "incorrect stained-ray output"
            assert np.array_equal(o[:, :3], a[:, :3]), "legend changed"
            assert np.all(o[a != parsed["bg"]] == a[a != parsed["bg"]]), "clues/stubs changed"
            oc = np.rot90(o[:, 3:], parsed["rotation"])
            owner = _owners(parsed)
            assert np.array_equal(oc[owner >= 0], parsed["scene"][owner >= 0]), "paint on solid pane"
            allowed = {parsed["guide"]} | {obj["fill"] for obj in parsed["objects"] if obj["fill"] != parsed["bg"]}
            assert set(map(int, np.unique(o[o != a]))) <= allowed, "unmapped ink"
            if split == "train":
                for shortcut in SHORTCUTS:
                    if not np.array_equal(_trace(parsed, shortcut)[0], o):
                        rejected.add(shortcut)
            else:
                assert metrics["panes"] >= 4 and len(metrics["events"]) >= 2 and metrics["two_flanks"]
    assert rejected == set(SHORTCUTS), "training leaves a tested shortcut unresolved"


def _boundary_regressions():
    """Hand-calculated labels for boundary behavior outside the sampler recipe."""
    # A width-spanning pane destroys both branches. Only the preceding row
    # becomes a crosspiece; the pane's entire interior remains unpainted.
    scene = np.zeros((6, 7), dtype=np.int8)
    scene[:2, 3] = 1
    for r, c in _corners((3, 0, 5, 6), 0):
        scene[r, c] = 2
    board = np.zeros((6, 10), dtype=np.int8)
    board[:, 3:] = scene
    board[:2, :2] = [(2, 3), (4, 5)]
    expected = board.copy()
    expected[2, 3:] = 3
    assert np.array_equal(solve(board), expected), "full-width clipping regression"

    # A higher window leaves just the last row of a lower pane exposed.
    # The final-layer event has flank pixels but no room for future tails.
    scene = np.zeros((6, 8), dtype=np.int8)
    scene[:2, 3] = 1
    for cue, box in ((2, (3, 2, 5, 5)), (3, (2, 0, 4, 7))):
        for r, c in _corners(box, 0):
            assert scene[r, c] == 0
            scene[r, c] = cue
    board = np.zeros((6, 11), dtype=np.int8)
    board[:, 3:] = scene
    board[:2, :2] = [(3, 0), (2, 4)]
    expected = board.copy()
    for r, c, ink in ((2, 3, 1), (4, 1, 4), (4, 3, 4), (4, 4, 4), (5, 1, 4), (5, 6, 4)):
        expected[r, c+3] = ink
    assert np.array_equal(solve(board), expected), "final-layer aperture regression"


def self_test():
    """Bounded 42-seed family checks, determinism, API limits and diversity."""
    shapes, structures, palettes, rotations, pane_counts, sizes = set(), set(), set(), set(), set(), set()
    windows = mergers = lost = thin = unused = 0
    signatures = set()
    _boundary_regressions()
    # Directly exercise the normally rare recovery path, including all teaching
    # obligations and uniqueness when its first candidates have been excluded.
    recovery = {"train": [], "test": []}
    excluded = set()
    for kind, split in (("window", "train"), ("meeting", "train"), ("free", "test")):
        pair = _fallback(kind, 0, excluded, split, True)
        recovery[split].append(pair)
        excluded.add(_signature(pair["input"]))
    validate_generated_puzzle(recovery)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_signature(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                parsed = _parse(pair["input"])
                _, m = _trace(parsed)
                shapes.add(pair["input"].shape)
                palettes.add((parsed["bg"], parsed["guide"]))
                rotations.add(parsed["rotation"])
                pane_counts.add(m["panes"])
                sizes.update(m["sizes"])
                windows += m["windows"] > 0
                mergers += m["mergers"] > 0
                lost += m["lost_flanks"] > 0
                thin += any(min(hw) == 2 for hw in m["sizes"])
                unused += m["unused"] > 0
                rolemap = {parsed["bg"]: 0, parsed["guide"]: 1}
                rolemap.update({c: i+2 for i, c in enumerate(parsed["cues"])})
                normalized = np.array([[rolemap[int(c)] for c in row] for row in parsed["scene"]], dtype=np.int8)
                structures.add(_signature(normalized))
        if seed in (0, 21, 41):
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in ("input", "output"))
            for pair in puzzle["train"] + puzzle["test"]:
                assert np.array_equal(solve(9-pair["input"]), 9-pair["output"]), "palette equivariance"
                reflected = pair["input"].copy()
                reflected[:, 3:] = np.fliplr(reflected[:, 3:])
                expected = pair["output"].copy()
                expected[:, 3:] = np.fliplr(expected[:, 3:])
                assert np.array_equal(solve(reflected), expected), "scene reflection equivariance"
    assert len(signatures) == 42 and len(shapes) >= 35 and len(structures) >= 180
    assert len(palettes) >= 45 and rotations == set(range(4))
    assert {2, 3, 4, 5, 6} <= pane_counts and len(sizes) >= 70
    assert min(windows, mergers, lost, thin, unused) > 0
    for nt, nq in ((2, 1), (10, 4), (6, 3)):
        p = generate_puzzle(173, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return dict(seeds=42, shapes=len(shapes), structures=len(structures), palettes=len(palettes),
                pane_counts=sorted(pane_counts), pane_sizes=len(sizes), windows=windows,
                mergers=mergers, clipped_or_blocked=lost, thin=thin, unused=unused)


if __name__ == "__main__":
    print(self_test())
