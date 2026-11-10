"""Cross surveyor: a terrain route clocks an ordered word at plus centers.

The small plus beside the swatches is a persistent key. All other cells of
its color replace traversable terrain. Detection uses the original input;
stamping uses route chronology, so later crosses own shared arms.
"""
from __future__ import annotations

import json

import numpy as np


CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
CROSS = ((0, 0),) + CARDINAL
REQUIRED = frozenset({"off_route", "selective", "cell_clock", "ink_clock",
                      "multiplicity", "bend_phase", "restart", "raster_order",
                      "overlap_order", "original_detection", "cycling", "near_miss"})


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _cross(p):
    return tuple((p[0] + dr, p[1] + dc) for dr, dc in CROSS)


def _components(grid):
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for r, c in np.ndindex(grid.shape):
        if seen[r, c]:
            continue
        color = int(grid[r, c])
        stack, points = [(r, c)], []
        seen[r, c] = True
        while stack:
            p = stack.pop()
            points.append(p)
            for dr, dc in CARDINAL:
                q = (p[0] + dr, p[1] + dc)
                if _inside(grid.shape, q) and not seen[q] and int(grid[q]) == color:
                    seen[q] = True
                    stack.append(q)
        result.append((color, frozenset(points)))
    return result


def _orient(grid, code):
    return np.rot90(grid[:, ::-1] if code >= 4 else grid, code % 4).copy()


def _unorient(grid, code):
    a = np.rot90(grid, -(code % 4))
    return (a[:, ::-1] if code >= 4 else a).copy()


def _centers(mask):
    result = np.zeros(mask.shape, dtype=bool)
    if min(mask.shape) >= 3:
        result[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                              & mask[2:, 1:-1] & mask[1:-1, :-2]
                              & mask[1:-1, 2:])
    return {tuple(map(int, p)) for p in np.argwhere(result)}


def _trace(mask, start):
    """Step walker: straight ahead, then switch between east and north."""
    p, axis = start, 0
    path, leg_ids, collisions = [], [], []
    leg = 0
    for _ in range(mask.size):
        if not _inside(mask.shape, p) or not mask[p] or p in path:
            raise ValueError("invalid monotone route")
        path.append(p)
        leg_ids.append(leg)
        dr, dc = ((0, 1), (-1, 0))[axis]
        q = (p[0] + dr, p[1] + dc)
        if not _inside(mask.shape, q):
            return dict(path=tuple(path), legs=tuple(leg_ids),
                        collisions=tuple(collisions), end="edge")
        if mask[q]:
            p = q
            continue
        collisions.append(q)
        axis = 1 - axis
        dr, dc = ((0, 1), (-1, 0))[axis]
        q = (p[0] + dr, p[1] + dc)
        if not _inside(mask.shape, q) or not mask[q]:
            return dict(path=tuple(path), legs=tuple(leg_ids),
                        collisions=tuple(collisions), end="blocked")
        p, leg = q, leg + 1
    raise ValueError("route did not terminate")


def _symbols(grid):
    swatches, headers = [], []
    for color, cells in _components(grid):
        if len(cells) not in (4, 5):
            continue
        r, c = min(p[0] for p in cells), min(p[1] for p in cells)
        if len(cells) == 4 and cells == frozenset((r+i, c+j) for i in (0, 1) for j in (0, 1)):
            swatches.append((r, c, color, cells))
        if len(cells) == 5 and cells == frozenset(_cross((r+1, c+1))):
            headers.append((r, c, color, cells))
    return swatches, headers


def _decode(input_grid):
    """Find the asymmetric plus-and-swatches key; no frequency or seed clues."""
    observed = _grid(input_grid)
    possibilities = []
    for code in range(8):
        g = _unorient(observed, code)
        swatches, headers = _symbols(g)
        for hr, hc, ink, header_cells in headers:
            entries = sorted((s for s in swatches if s[0] == hr and s[1] >= hc + 4),
                             key=lambda s: s[1])
            if not 2 <= len(entries) <= 5 or not 4 <= entries[0][1] - hc <= 6:
                continue
            if any(not 3 <= b[1] - a[1] <= 5 for a, b in zip(entries, entries[1:])):
                continue
            word = tuple(s[2] for s in entries)
            if ink in word or len(set(word)) < 2:
                continue
            annotations = set(header_cells).union(*(s[3] for s in entries))
            # Cardinal rings provide the field role even when it is a minority.
            ring = {int(g[q]) for p in annotations for dr, dc in CARDINAL
                    if _inside(g.shape, q := (p[0]+dr, p[1]+dc)) and q not in annotations}
            if len(ring) != 1:
                continue
            field = ring.pop()
            remaining = set(map(int, np.unique(g))) - {field, ink} - set(word)
            if len(remaining) != 1 or field in word or field == ink:
                continue
            terrain = remaining.pop()
            start_cells = [tuple(map(int, p)) for p in np.argwhere(np.isin(g, word))
                           if tuple(p) not in annotations]
            if not 1 <= len(start_cells) <= 3 or any(int(g[p]) != word[0] for p in start_cells):
                continue
            starts = tuple(sorted(start_cells))
            if any(abs(r-s)+abs(c-t) <= 1 for j, (r, c) in enumerate(starts)
                   for s, t in starts[j+1:]):
                continue
            mask = (g == terrain) | (g == ink)
            for p in header_cells:
                mask[p] = False
            for p in starts:
                mask[p] = True
            if any(r < hr+4 or c+1 >= g.shape[1] or not mask[r, c+1]
                   or (c and mask[r, c-1]) for r, c in starts):
                continue
            # Key is in an empty strip, with no scene marks hidden among entries.
            if np.any(mask[:hr+4]):
                continue
            marks = g == ink
            for p in header_cells:
                marks[p] = False
            centers = _centers(marks)
            records = [_trace(mask, p) for p in starts]
            occupied_paths, occupied_stamps = set(), set()
            valid = True
            for record in records:
                footprint = set(record["path"])
                events = tuple((p, i, leg) for i, (p, leg) in enumerate(
                    zip(record["path"], record["legs"])) if p in centers)
                stamps = {q for p, _, _ in events for q in _cross(p)}
                if occupied_paths & footprint or occupied_stamps & stamps:
                    valid = False
                    break
                occupied_paths |= footprint
                occupied_stamps |= stamps
                record["events"] = events
            if valid:
                possibilities.append(dict(grid=g, code=code, field=field, terrain=terrain,
                                          ink=ink, word=word, header=(hr, hc),
                                          entries=tuple(s[1] for s in entries),
                                          annotations=annotations, mask=mask, marks=marks,
                                          centers=centers, starts=starts, records=records))
    if len(possibilities) != 1:
        raise ValueError(f"expected one visible program interpretation, got {len(possibilities)}")
    return possibilities[0]


def _paint(info, alternative=None):
    out = info["grid"].copy()
    word = info["word"]
    if alternative == "multiplicity":
        word = tuple(dict.fromkeys(word))
    global_phase = 0
    owned = set()
    for record in info["records"]:
        events = record["events"]
        if alternative == "raster_order":
            events = tuple(sorted(events))
        phase, last_leg = (global_phase if alternative == "restart" else 0), None
        ink_index = {p: i for i, p in enumerate(p for p in record["path"] if info["marks"][p])}
        for p, path_index, leg in events:
            if alternative == "original_detection" and any(int(out[q]) != info["ink"] for q in _cross(p)):
                continue
            if alternative == "bend_phase" and leg != last_leg:
                phase = 0
            last_leg = leg
            index = (path_index if alternative == "cell_clock" else ink_index[p]
                     if alternative == "ink_clock" else phase)
            color = word[index % len(word)]
            if alternative != "cycling" or phase < len(word):
                for q in _cross(p):
                    if alternative != "overlap_order" or q not in owned:
                        out[q] = color
                    owned.add(q)
            phase += 1
        global_phase += len(events)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    return _orient(_paint(info), info["code"])


def _evidence(info):
    correct = _paint(info)
    found = {name for name in REQUIRED - {"off_route", "selective", "near_miss"}
             if not np.array_equal(correct, _paint(info, name))}
    changed = correct != info["grid"]
    all_paths = {p for record in info["records"] for p in record["path"]}
    if any(info["marks"][p] and not changed[p] and p not in info["centers"]
           and all(_inside(info["grid"].shape, q) for q in _cross(p))
           and sum(bool(info["marks"][q]) for q in _cross(p)[1:]) == 3
           for p in all_paths):
        found.add("near_miss")
    if any(p not in all_paths and all(not changed[q] for q in _cross(p)) for p in info["centers"]):
        found.add("off_route")
    if any(info["marks"][p] and not changed[p]
           and any(_inside(changed.shape, q := (p[0]+dr, p[1]+dc)) and changed[q]
                   for dr, dc in CARDINAL) for p in zip(*np.nonzero(info["marks"]))):
        found.add("selective")
    return found


def _eligible(mask, record, starts):
    return [(p, i, leg) for i, (p, leg) in enumerate(zip(record["path"], record["legs"]))
            if all(_inside(mask.shape, q) and mask[q] and q not in starts for q in _cross(p))]


def _terrain_scene(rng, h, w, top, guided):
    """Broad ledges or independently changing sections with local bays/islands."""
    mask = np.zeros((h, w), bool)
    if guided:
        turn = int(rng.integers(top+2, min(top+6, h-12)+1))
        start_row = int(rng.integers(turn+4, h-7))
        left = int(rng.integers(0, 3))
        elbow = int(rng.integers(max(left+7, w//3), min(w-7, w//2+3)+1))
        mask[turn:start_row+3, left:elbow+1] = True
        mask[turn-2:turn+3, elbow+1:] = True
        # A second, independently started broad lobe, with a different residue.
        low = h-3
        end = int(rng.integers(7, min(w-3, 14)))
        mask[low-1:, :end+1] = True
        # Sometimes connect lobes through unused terrain; it does not set order.
        if rng.random() < .5:
            mask[start_row+2:low, :2] = True
        # Small carvings/additions outside the protected three-leg trace.
        for _ in range(int(rng.integers(2, 7))):
            r = int(rng.integers(turn+2, start_row+3))
            c = int(rng.integers(left+1, max(left+2, elbow-2)))
            if r != start_row and abs(r-start_row) > 1:
                mask[r, c] = bool(rng.integers(2))
        return mask, ((start_row, left), (low, 0))
    kind = int(rng.integers(3))
    if kind < 2:
        r = top
        while r < h:
            depth = int(rng.integers(1, 6))
            left = int(rng.integers(0, max(1, w//2)))
            right = int(rng.integers(max(left+3, w//2), w+1))
            mask[r:r+depth, left:right] = True
            r += depth
        if kind == 1:
            for _ in range(int(rng.integers(2, 8))):
                r, c = int(rng.integers(top, h)), int(rng.integers(w))
                mask[r:r+int(rng.integers(1, 5)), c:c+int(rng.integers(1, 7))] = bool(rng.integers(2))
    else:
        for _ in range(int(rng.integers(3, 10))):
            r, c = int(rng.integers(top, h-2)), int(rng.integers(w-3))
            dh = int(rng.integers(3, max(4, h-r+1)))
            dw = int(rng.integers(4, max(5, w-c+1)))
            mask[r:r+dh, c:c+dw] = True
    mask[:top] = False
    return mask, None


def _choose_routes(rng, mask, top, count):
    candidates = []
    for r in range(top, mask.shape[0]):
        for c in range(mask.shape[1]-1):
            if mask[r, c] and mask[r, c+1] and (c == 0 or not mask[r, c-1]):
                rec = _trace(mask, (r, c))
                if len(_eligible(mask, rec, {(r, c)})) >= 1:
                    candidates.append(((r, c), rec))
    rng.shuffle(candidates)
    selected, occupied = [], set()
    for start, rec in candidates:
        # Separating complete cross footprints prevents inter-route stamp ties.
        expanded = {q for p in rec["path"] for q in _cross(p) if _inside(mask.shape, q)}
        if expanded & occupied:
            continue
        selected.append(start)
        occupied |= expanded
        if len(selected) == count:
            return tuple(sorted(selected))
    return None


def _word(rng, length, witness):
    if witness:
        patterns = ((0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 1, 1, 0),
                    (0, 0, 1, 1), (0, 1, 2, 1))
        return patterns[int(rng.integers(len(patterns)))]
    distinct = int(rng.integers(2, min(length, 4)+1))
    values = list(range(distinct)) + list(map(int, rng.integers(distinct, size=length-distinct)))
    rng.shuffle(values)
    return tuple(values)


def _fallback_pair(rng, band, orientation):
    """A fully constructive witness, preserving the selected scale band."""
    h = 22 if band == "compact" else 28 if band == "large" else int(rng.integers(22, 31))
    w = 22 if band == "compact" else 28 if band == "large" else int(rng.integers(22, 31))
    field, terrain, ink, a, b = map(int, rng.choice(10, 5, replace=False))
    inp = np.full((h, w), field, np.int8)
    inp[6:14, :11] = terrain
    inp[4:9, 11:] = terrain
    low = h-3
    inp[low-2:, :12] = terrain
    points = [(11, c) for c in (2, 3, 6, 7, 8)]
    points += [(6, c) for c in (13, 14, 17, 18, 20)]
    points += [(low, 3), (low, 4), (8, 5), (1, 1)]
    for p in points:
        for q in _cross(p):
            inp[q] = ink
    # A visited T at the first elbow stays ink, and its side touches a stamp.
    inp[10:13, 10] = ink
    for c, color in zip((4, 7, 10), (a, a, b)):
        inp[:2, c:c+2] = color
    inp[11, 0] = inp[low, 0] = a
    info = _decode(inp)
    assert REQUIRED <= _evidence(info)
    return {"input": _orient(inp, orientation), "output": _orient(_paint(info), orientation)}


def _make_pair(rng, band, orientation, witness=False, crowded=False):
    for attempt in range(160):
        guided = (attempt > 100) or rng.random() < (.70 if witness else .12)
        pattern = _word(rng, int(rng.integers(2, 6)), witness)
        palette = tuple(map(int, rng.permutation(10)))
        field, terrain, ink = palette[:3]
        word = tuple(palette[3+i] for i in pattern)
        hr, hc = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        first = hc + int(rng.integers(4, 7))
        cols = [first]
        for _ in word[1:]:
            cols.append(cols[-1] + int(rng.integers(3, 6)))
        min_width = max(14, cols[-1]+3)
        if min_width > 30:
            continue
        low, high = {"compact": (14, 22), "large": (28, 30), "any": (14, 30)}[band]
        if guided:
            low = max(low, 22)
            high = max(high, low)
            min_width = max(min_width, 22)
        h = int(rng.integers(low, high+1))
        w = int(rng.integers(max(low, min_width), max(high, min_width)+1))
        if not guided and rng.random() < .2 and h >= min_width:
            w = h
        mask, starts = _terrain_scene(rng, h, w, hr+4, guided)
        count = 2 if witness else int(rng.choice((1, 2, 3), p=(.55, .35, .10)))
        if starts is not None and not witness:
            starts = starts[:min(count, 2)]
        elif starts is None:
            starts = _choose_routes(rng, mask, hr+4, count)
        if not starts:
            continue
        starts = tuple(sorted(starts))
        records = [_trace(mask, p) for p in starts]
        mark_mask = np.zeros(mask.shape, bool)
        selected = []
        okay = True
        for rec in records:
            options = _eligible(mask, rec, starts)
            if not options:
                okay = False
                break
            if witness and rec is records[0]:
                # Many events on both sides of a bend, with a forced adjacent pair.
                if len(options) < 9 or len({leg for p, i, leg in options}) < 2:
                    okay = False
                    break
                n = int(rng.integers(8, min(13, len(options))+1))
            elif witness:
                if len(options) < 2:
                    okay = False
                    break
                n = int(rng.integers(2, min(6, len(options))+1))
            else:
                n = int(rng.integers(1, min(14 if crowded else 9, len(options))+1))
            chosen = [options[int(j)][0] for j in rng.choice(len(options), n, replace=False)]
            if witness or crowded or rng.random() < .4:
                adjacent = [(a[0], b[0]) for a, b in zip(options, options[1:])
                            if abs(a[0][0]-b[0][0])+abs(a[0][1]-b[0][1]) == 1]
                if adjacent:
                    chosen.extend(adjacent[int(rng.integers(len(adjacent)))])
            selected.extend(chosen)
            for p in chosen:
                for q in _cross(p):
                    mark_mask[q] = True
        if not okay:
            continue
        all_paths = {p for rec in records for p in rec["path"]}
        occupied = {q for p in selected for q in _cross(p)}
        # Unvisited complete crosses and incomplete route-crossing marks.
        off_options = [p for p in _centers(mask) if p not in all_paths
                       and all(q not in occupied and q not in starts for q in _cross(p))]
        rng.shuffle(off_options)
        for p in off_options[:int(rng.integers(1, 5))]:
            for q in _cross(p):
                mark_mask[q] = True
        for rec in records:
            near = [p for p in rec["path"] if p not in starts and not mark_mask[p]
                    and all(_inside(mask.shape, q) and mask[q] and q not in occupied
                            and q not in starts for q in _cross(p))]
            if near:
                p = near[int(rng.integers(len(near)))]
                missing = int(rng.integers(1, 5))
                for j, q in enumerate(_cross(p)):
                    if j != missing:
                        mark_mask[q] = True
        # Ink is a layer on terrain, never a way to grow the traversable mask.
        for _ in range(int(rng.integers(2, 12 if crowded else 7))):
            r, c = int(rng.integers(hr+4, h)), int(rng.integers(w))
            style = int(rng.integers(4))
            points = ((r, c),)
            if style == 1:
                points = tuple((r, c+j) for j in range(int(rng.integers(2, 7))))
            elif style == 2:
                points = tuple((r+i, c+j) for i in range(2) for j in range(int(rng.integers(2, 5))))
            elif style == 3:
                points = tuple((r+i, c+j) for i, j in ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)))
            for q in points:
                if _inside(mask.shape, q) and mask[q] and q not in starts:
                    mark_mask[q] = True
        inp = np.where(mask, terrain, field).astype(np.int8)
        inp[mark_mask] = ink
        for q in _cross((hr+1, hc+1)):
            inp[q] = ink
        for c, color in zip(cols, word):
            inp[hr:hr+2, c:c+2] = color
        for p in starts:
            inp[p] = word[0]
        try:
            info = _decode(inp)
        except ValueError:
            continue
        evidence = _evidence(info)
        if witness and not REQUIRED <= evidence:
            continue
        if not any(rec["events"] for rec in info["records"]):
            continue
        return {"input": _orient(inp, orientation), "output": _orient(_paint(info), orientation)}
    return _fallback_pair(rng, band, orientation)


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("need 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    shift = int(rng.integers(3))  # scale, crowded ink, or a coherent orientation shift
    base = int(rng.integers(8))
    shift_code = (base//4)*4 + (base + int(rng.integers(1, 4))) % 4
    puzzle = {"train": [], "test": []}
    seen = set()
    witness_slots = {int(rng.integers(num_train))}
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(int(count)):
            witness = split == "train" and i in witness_slots
            band = ("compact" if split == "train" else "large") if shift == 0 else "any"
            orientation = shift_code if shift == 2 and split == "test" else base
            if shift != 2 and rng.random() < .45:
                orientation = int(rng.integers(8))
            for _ in range(16):
                pair = _make_pair(rng, band, orientation, witness,
                                  crowded=shift == 1 and split == "test")
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("input uniqueness construction failed")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_output(info):
    """Direct original-neighborhood scanner and explicit last-owner table.

    This is an authored consistency check, not an independent reference solver.
    """
    g, fg = info["grid"], info["ink"]
    direct = set()
    for r in range(1, g.shape[0]-1):
        for c in range(1, g.shape[1]-1):
            p = (r, c)
            if p in info["annotations"]:
                continue
            if all(int(g[q]) == fg for q in _cross(p)):
                direct.add(p)
    assert direct == info["centers"]
    owner = {}
    for rec in info["records"]:
        events = [p for p in rec["path"] if p in direct]
        assert events == [p for p, _, _ in rec["events"]]
        for index, p in enumerate(events):
            for q in _cross(p):
                owner[q] = info["word"][index % len(info["word"])]
    expected = g.copy()
    for p, color in owner.items():
        expected[p] = color
    changed = np.zeros(g.shape, bool)
    for p in owner:
        changed[p] = True
    assert np.array_equal(expected != g, changed)
    return expected


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training_evidence = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            expected = _audit_output(info)
            canonical = _unorient(out, info["code"])
            assert np.array_equal(canonical, expected)
            assert not np.array_equal(inp, out)
            assert np.array_equal(canonical[~info["marks"]], info["grid"][~info["marks"]])
            assert all(int(canonical[p]) == int(info["grid"][p]) for p in info["annotations"])
            # Original ink fills real terrain: every eligible cross footprint is
            # walkable and every stamped cell is original ink, with no clipping.
            for rec in info["records"]:
                for p, _, _ in rec["events"]:
                    assert all(info["mask"][q] and info["marks"][q] for q in _cross(p))
            if split == "train":
                training_evidence |= _evidence(info)
    assert REQUIRED <= training_evidence, REQUIRED - training_evidence


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def _pattern(word):
    role = {}
    return tuple(role.setdefault(c, len(role)) for c in word)


def self_test():
    """Bounded 42-seed checks, legal counts, mutation checks, and edge fixtures."""
    for j, band in enumerate(("compact", "large", "any")):
        fallback = _fallback_pair(np.random.default_rng(987+j), band, j+4)
        assert np.array_equal(solve(fallback["input"]), fallback["output"])
        assert REQUIRED <= _evidence(_decode(fallback["input"]))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    infos = [_decode(p["input"]) for task in puzzles for split in ("train", "test") for p in task[split]]
    assert len({i["grid"].shape for i in infos}) >= 25
    assert {i["code"] for i in infos} == set(range(8))
    assert len({_pattern(i["word"]) for i in infos}) >= 12
    assert {len(i["word"]) for i in infos} >= {2, 3, 4, 5}
    assert {len(i["records"]) for i in infos} >= {1, 2, 3}
    assert len({len(i["centers"]) for i in infos}) >= 10
    assert len({(i["field"], i["terrain"], i["ink"]) for i in infos}) >= 100
    for role in ("field", "terrain", "ink"):
        assert {i[role] for i in infos} == set(range(10))
    assert any(r["end"] == "edge" for i in infos for r in i["records"])
    assert any(r["end"] == "blocked" for i in infos for r in i["records"])
    assert any(len(set(r["legs"])) == 1 for i in infos for r in i["records"])
    assert any(len(set(r["legs"])) >= 3 for i in infos for r in i["records"])
    for task in puzzles:
        assert len({_key(p["input"])[0] for s in task for p in task[s]}) >= 2
        for p in task["train"] + task["test"]:
            assert np.array_equal(solve(p["input"]), p["output"])
    # Exhaust every legal explicit count once. This is small and deterministic.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            task = generate_puzzle(314159+a*10+b, a, b)
            assert len(task["train"]) == a and len(task["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Complete versus clipped cardinal neighborhoods, including overlap.
    for bits in range(512):
        mask = np.array([(bits >> j) & 1 for j in range(9)], bool).reshape(3, 3)
        expected = bool(mask[1, 1] and mask[0, 1] and mask[2, 1] and mask[1, 0] and mask[1, 2])
        assert ((1, 1) in _centers(mask)) == expected
    assert len(_centers(np.ones((3, 4), bool))) == 2
    # Leaving either canvas edge stops even when terrain around the corner exists.
    horizontal = np.zeros((5, 8), bool)
    horizontal[3, :] = True
    horizontal[:4, 7] = True
    assert _trace(horizontal, (3, 0))["path"] == tuple((3, c) for c in range(8))
    vertical = np.zeros((7, 9), bool)
    vertical[5, :4] = True
    vertical[:6, 3] = True
    vertical[0, 3:] = True
    assert _trace(vertical, (5, 0))["path"][-1] == (0, 3)
    # A small, literal fixture pins the event clock and later ownership. It is
    # below the sampler's size range and uses no construction or solver helpers
    # to define the expected colors at its eight stamped cells.
    small = np.zeros((10, 13), np.int8)
    small[4:, :11] = 1
    for p in ((1, 1), (5, 3), (5, 4), (8, 3)):
        for q in _cross(p):
            small[q] = 2
    small[5, 6] = 2
    small[:2, 4:6] = 4
    small[:2, 7:9] = 5
    small[5, 0] = 4
    expected = small.copy()
    expected[4, 3] = expected[6, 3] = expected[5, 2] = 4
    expected[4, 4] = expected[6, 4] = 5
    expected[5, 3:6] = 5
    for code in range(8):
        assert np.array_equal(solve(_orient(small, code)), _orient(expected, code))
    assert expected[5, 6] == 2 and np.all(expected[7:10, 2:5] == small[7:10, 2:5])
    # Every shortcut mutation has an actual training witness, rejected as a label.
    task = puzzles[0]
    for alternative in REQUIRED - {"off_route", "selective", "near_miss"}:
        witness = next(p for p in task["train"] if alternative in _evidence(_decode(p["input"])))
        info = _decode(witness["input"])
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in task[s]] for s in task}
        index = next(j for j, p in enumerate(task["train"]) if p is witness)
        bad["train"][index]["output"] = _orient(_paint(info, alternative), info["code"])
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"wrong {alternative} labels accepted")
    return {"seeds": 42, "puzzles": 42, "shapes": len({i["grid"].shape for i in infos}),
            "program_patterns": len({_pattern(i["word"]) for i in infos}),
            "route_counts": sorted({len(i["records"]) for i in infos}),
            "orientations": sorted({i["code"] for i in infos})}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
