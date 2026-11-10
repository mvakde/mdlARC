"""Chromatic condensation window: complete, capture, split, and edge-stamp.

Only NumPy and the standard library are used. No file access, cached examples,
latent scene state, or seed participates in solve(). Generated labels are made
from complete scenes before their panes are painted over the wires.
"""
from __future__ import annotations

import hashlib
import numpy as np


_FOUR = ((-1, 0), (0, -1), (0, 1), (1, 0))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _need(bool(np.all((g >= 0) & (g <= 9))), "grid color range")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _box(points):
    rows, cols = zip(*points)
    return min(rows), max(rows), min(cols), max(cols)


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _perimeter(box):
    t, b, l, r = box
    return ({(t, c) for c in range(l, r + 1)} |
            {(b, c) for c in range(l, r + 1)} |
            {(y, l) for y in range(t + 1, b)} |
            {(y, r) for y in range(t + 1, b)})


def _halo(points):
    return {(r + dr, c + dc) for r, c in points for dr, dc in ((0, 0),) + _FOUR}


def _groups(points, diagonal=False):
    unseen, groups = set(points), []
    steps = _EIGHT if diagonal else _FOUR
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _panes(g):
    """The connected ink ring identifies a pane without trying output sizes."""
    bg = int(g[0, 0])
    _need(all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))),
          "corners must identify background")
    result = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        points = {tuple(map(int, p)) for p in np.argwhere(g == color)}
        for group in _groups(points):
            t, b, l, r = _box(group)
            if b - t < 4 or r - l < 4:
                continue
            corners = {(t + 1, l + 1), (t + 1, r - 1),
                       (b - 1, l + 1), (b - 1, r - 1)}
            if group != _perimeter((t, b, l, r)) | corners:
                continue
            edges = (g[t + 1, l + 2:r - 1], g[t + 2:b - 1, r - 1],
                     g[b - 1, l + 2:r - 1], g[t + 2:b - 1, l + 1])
            if any(not np.all(a == a[0]) for a in edges):
                continue
            keys = tuple(int(a[0]) for a in edges)
            inside = g[t + 2:b - 1, l + 2:r - 1]
            fill = int(inside[0, 0])
            if (len(set(keys)) != 4 or set(keys) & {bg, color, fill} or
                    fill == color or not np.all(inside == fill)):
                continue
            result.append(dict(box=(t, b, l, r), keys=keys, ink=color,
                               fill=fill, size=int(inside.size)))
    result.sort(key=lambda p: (-p["size"], p["box"]))
    _need(bool(result), "no identifiable opaque pane")
    _need(len(result) == 1 or result[0]["size"] > result[1]["size"],
          "largest pane area tie")
    return bg, result


def _period_fits(g, visible, hidden):
    """Return a witnessed model only if every valid fit agrees on hidden cells."""
    hidden = sorted(hidden)
    predictions, fits = set(), []
    for period in range(1, 7):
        for sign in (1, -1):
            word, counts = {}, [0] * period
            for r, c in sorted(visible):
                residue, value = (r + sign * c) % period, int(g[r, c])
                if residue in word and word[residue] != value:
                    break
                word[residue] = value
                counts[residue] += 1
            else:
                if min(counts) >= 2:
                    w = tuple(word[j] for j in range(period))
                    predictions.add(tuple(w[(r + sign * c) % period] for r, c in hidden))
                    fits.append((period, sign, w))
    if not fits or len(predictions) != 1:
        return None
    return fits[0], dict(zip(hidden, next(iter(predictions))))


def _read_loops(g, bg, panes):
    main = panes[0]
    masked = set()
    for pane in panes:
        cells = _area(pane["box"])
        _need(not cells & _halo(masked), "panes touch or overlap")
        _need(pane["ink"] == main["ink"] and pane["fill"] == main["fill"] and
              set(pane["keys"]) == set(main["keys"]), "inconsistent pane conventions")
        masked.update(cells)
    points = {tuple(map(int, p)) for p in np.argwhere(g != bg)} - masked
    _need(all(int(g[p]) in main["keys"] for p in points), "unexplained exterior color")
    parts = _groups(points)
    # A single rectangular mask cuts a rectangular perimeter into at most two arcs.
    candidates = []
    for i in range(len(parts)):
        for j in range(i, len(parts)):
            visible = parts[i] if i == j else parts[i] | parts[j]
            box = _box(visible)
            t, b, l, r = box
            if b - t < 2 or r - l < 2:
                continue
            full = _perimeter(box)
            if full - masked != visible:
                continue
            # Extra panes are distractors, never additional unknown wire masks.
            if any(full & _area(p["box"]) for p in panes[1:]):
                continue
            fit = _period_fits(g, visible, full & masked)
            if fit is None:
                continue
            model, assignments = fit
            candidates.append(dict(parts=frozenset((i, j)), box=box,
                                   model=model, hidden=assignments, full=full))
    covers = []

    def cover(remaining, chosen, occupied):
        if len(covers) >= 2:
            return
        if not remaining:
            covers.append(chosen)
            return
        first = min(remaining)
        for candidate in candidates:
            if (first in candidate["parts"] and candidate["parts"] <= remaining and
                    not candidate["full"] & _halo(occupied)):
                cover(remaining - candidate["parts"], chosen + [candidate],
                      occupied | candidate["full"])

    cover(set(range(len(parts))), [], set())
    _need(len(covers) == 1, "wire geometry/completion is not unique")
    return covers[0]


def _captured(main, loops, *, phase_shift=False, filled=False):
    t, b, l, r = main["box"]
    cargo = np.full((b - t - 3, r - l - 3), -1, dtype=np.int8)
    for loop in loops:
        period, sign, word = loop["model"]
        cells = _area(loop["box"]) if filled else loop["full"]
        for y, x in cells:
            if t + 1 < y < b - 1 and l + 1 < x < r - 1:
                v = word[(y + sign * x + int(phase_shift and period > 1)) % period]
                cargo[y - t - 2, x - l - 2] = v
    return cargo


def _pieces(cargo, *, diagonal=False, grouped=False):
    result = []
    for color in sorted(int(v) for v in np.unique(cargo) if v >= 0):
        points = {tuple(map(int, p)) for p in np.argwhere(cargo == color)}
        for cells in ([points] if grouped else _groups(points, diagonal)):
            result.append((color, cells))
    return result


def _stamp(cargo, keys, ink, fill, *, diagonal=False, grouped=False, down=False):
    h, w = cargo.shape
    out = np.full((h, w), fill, dtype=np.int8)
    lookup = {color: side for side, color in enumerate(keys)}
    for color, cells in _pieces(cargo, diagonal=diagonal, grouped=grouped):
        side = 2 if down else lookup[color]
        a, z, u, v = _box(cells)
        dr, dc = ((-a, 0), (0, w - 1 - v), (h - 1 - z, 0), (0, -u))[side]
        for r, c in cells:
            out[r + dr, c + dc] = ink
    return out


def _analyze(value):
    g = _grid(value)
    bg, panes = _panes(g)
    loops = _read_loops(g, bg, panes)
    main = panes[0]
    cargo = _captured(main, loops)
    out = _stamp(cargo, main["keys"], main["ink"], main["fill"])
    return dict(background=bg, main=main, panes=panes, loops=loops,
                cargo=cargo, output=out)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from the input pixels alone."""
    return _analyze(input_grid)["output"].copy()


def _different(a, b):
    return not np.array_equal(a, b)


def _witnesses(a):
    m, cargo, expected = a["main"], a["cargo"], a["output"]
    args = (m["keys"], m["ink"], m["fill"])
    return dict(independent=_different(expected, _stamp(cargo, *args, grouped=True)),
                four=_different(expected, _stamp(cargo, *args, diagonal=True)),
                keyed=_different(expected, _stamp(cargo, *args, down=True)),
                phase=_different(expected, _stamp(_captured(m, a["loops"], phase_shift=True), *args)),
                perimeter=_different(expected, _stamp(_captured(m, a["loops"], filled=True), *args)),
                movement=_different(expected, np.where(cargo >= 0, m["ink"], m["fill"]).astype(np.int8)))


def _rand(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _random_box(rng, h, w):
    t, b = sorted(map(int, rng.choice(np.arange(1, h - 1), 2, replace=False)))
    l, r = sorted(map(int, rng.choice(np.arange(1, w - 1), 2, replace=False)))
    return t, b, l, r


def _geometry(rng, count, nested, reserve=False):
    if reserve:
        h = w = 30
        main = (8, 23, 8, 23)
    else:
        h, w = _rand(rng, 21, 30), _rand(rng, 21, 30)
        margin = 4 if nested else 3
        ih, iw = _rand(rng, 6, min(14, h - 4 - 2 * margin)), _rand(rng, 6, min(14, w - 4 - 2 * margin))
        t, l = _rand(rng, margin, h - ih - 4 - margin), _rand(rng, margin, w - iw - 4 - margin)
        main = t, t + ih + 3, l, l + iw + 3
    t, b, l, r = main
    main_area = _area(main)
    interior = _area((t + 2, b - 2, l + 2, r - 2))
    boxes, occupied = [], set()

    def accept(box, crossing=True):
        y0, y1, x0, x1 = box
        if y1 - y0 < 2 or x1 - x0 < 2:
            return False
        cells = _perimeter(box)
        visible = cells - main_area
        hidden = cells & interior
        if (not visible or _box(visible) != box or
                (len(hidden) < 2 if crossing else bool(cells & main_area)) or
                cells & _halo(occupied)):
            return False
        boxes.append(box)
        occupied.update(cells)
        return True

    if reserve:
        safe = [(1, 17, 1, 17), (3, 14, 3, 14), (20, 28, 20, 28), (2, 11, 20, 27)]
        for box in safe[:count]:
            if not accept(box):
                return None
    elif nested and count >= 2:
        outer = (_rand(rng, 1, t - 3), _rand(rng, t + 5, b - 2),
                 _rand(rng, 1, l - 3), _rand(rng, l + 5, r - 2))
        inner = (_rand(rng, outer[0] + 2, t - 1), _rand(rng, t + 2, outer[1] - 2),
                 _rand(rng, outer[2] + 2, l - 1), _rand(rng, l + 2, outer[3] - 2))
        if not accept(outer) or not accept(inner):
            return None
    for _ in range(len(boxes), count):
        for attempt in range(500):
            if accept(_random_box(rng, h, w)):
                break
        else:
            return None
    return h, w, main, boxes, occupied, accept


def _paint_pane(g, box, keys, ink, fill):
    t, b, l, r = box
    g[t:b + 1, l:r + 1] = ink
    g[t + 1, l + 2:r - 1] = keys[0]
    g[t + 2:b - 1, r - 1] = keys[1]
    g[b - 1, l + 2:r - 1] = keys[2]
    g[t + 2:b - 1, l + 1] = keys[3]
    g[t + 2:b - 1, l + 2:r - 1] = fill


def _word(rng, routes, force_four=False, force_solid=False):
    if force_solid:
        return (int(rng.choice(routes)),)
    if force_four:
        return tuple(map(int, rng.permutation(routes)))
    if rng.random() < .35:
        return (int(rng.choice(routes)),)
    period = _rand(rng, 2, 6)
    n = _rand(rng, 2, min(period, 4))
    colors = list(map(int, rng.permutation(routes)))[:n]
    word = colors + [int(rng.choice(colors)) for _ in range(period - n)]
    rng.shuffle(word)
    return tuple(word)


def _attempt_pair(rng, lesson, test, bg, turn, reserve=False):
    count = 2 if lesson is not None else (_rand(rng, 3, 4) if test else _rand(rng, 1, 2))
    geom = _geometry(rng, count, nested=(rng.random() < .55), reserve=reserve)
    if geom is None:
        return None
    h, w, main_box, boxes, occupied, accept = geom
    crossing_boxes = list(boxes)
    # Fully exposed wire distractors use exactly the same geometry/color language.
    if lesson != 0 and rng.random() < .55:
        for _ in range(100):
            hh, ww = _rand(rng, 3, 6), _rand(rng, 3, 6)
            yy, xx = _rand(rng, 1, h - hh - 1), _rand(rng, 1, w - ww - 1)
            if accept((yy, yy + hh - 1, xx, xx + ww - 1), crossing=False):
                break
    palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    ink, distinct_fill = palette[0:2]
    routes = tuple(palette[2:6])
    keys = tuple(map(int, rng.permutation(routes)))
    fill = bg if lesson == 0 or (lesson != 1 and rng.random() < .4) else distinct_fill
    extra_count = 1 if lesson == 1 or test else int(lesson != 0 and rng.random() < .35)
    extra_boxes = []
    blocked = _area(main_box) | occupied
    for _ in range(extra_count):
        for attempt in range(180):
            hh, ww = _rand(rng, 5, 8), _rand(rng, 5, 8)
            yy, xx = _rand(rng, 1, h - hh - 1), _rand(rng, 1, w - ww - 1)
            box = yy, yy + hh - 1, xx, xx + ww - 1
            cells = _area(box)
            if not cells & _halo(blocked):
                extra_boxes.append(box)
                blocked.update(cells)
                break
        else:
            return None
    loops = []
    shared = int(rng.choice(routes))
    for index, box in enumerate(boxes):
        word = ((shared,) if lesson == 0 else
                _word(rng, routes, force_four=lesson == 1 and index == 0))
        phase, sign = _rand(rng, 0, len(word) - 1), int(rng.choice((-1, 1)))
        # Store an absolute-coordinate word; this is construction-only metadata.
        absolute = tuple(word[(i + phase) % len(word)] for i in range(len(word)))
        loops.append(dict(box=box, full=_perimeter(box), model=(len(word), sign, absolute)))
    # The complete scene is known before applying any opaque pane.
    g = np.full((h, w), bg, dtype=np.int8)
    for loop in loops:
        period, sign, word = loop["model"]
        for y, x in loop["full"]:
            g[y, x] = word[(y + sign * x) % period]
    main = dict(box=main_box, keys=keys, ink=ink, fill=fill)
    cargo = _captured(main, loops)
    truth = _stamp(cargo, keys, ink, fill)
    _paint_pane(g, main_box, keys, ink, fill)
    for box in extra_boxes:
        _paint_pane(g, box, tuple(map(int, rng.permutation(routes))), ink, fill)
    inp, out = np.rot90(g, turn).copy(), np.rot90(truth, turn).copy()
    try:
        a = _analyze(inp)
        if not np.array_equal(a["output"], out) or not np.any(out == ink):
            return None
        if sum(bool(set(loop["hidden"]) & _area((a["main"]["box"][0] + 2,
                                                 a["main"]["box"][1] - 2,
                                                 a["main"]["box"][2] + 2,
                                                 a["main"]["box"][3] - 2)))
               for loop in a["loops"]) != len(crossing_boxes):
            return None
        if lesson is not None:
            witnesses = _witnesses(a)
            required = ("independent", "movement", "perimeter") if lesson == 0 else ("four", "keyed", "phase", "movement")
            if not all(witnesses[k] for k in required):
                return None
            if lesson == 1 and len({v for v in a["cargo"].flat if v >= 0}) != 4:
                return None
    except ValueError:
        return None
    return {"input": inp, "output": out}


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} out of range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    _need(num_train > num_test, "num_train must exceed num_test")
    _need(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)),
          "seed must be an integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    backgrounds = list(map(int, rng.permutation(10)))
    # A seed chooses either a shared scene orientation or pair-local rotations.
    fixed_turn = _rand(rng, 0, 3) if rng.random() < .4 else None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else None
            bg = backgrounds[len(seen) % 10]
            turn = fixed_turn if fixed_turn is not None else _rand(rng, 0, 3)
            for attempt in range(100):
                pair = _attempt_pair(rng, lesson, split == "test", bg, turn, reserve=attempt >= 75)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded scene sampler exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _crossings(a):
    t, b, l, r = a["main"]["box"]
    interior = _area((t + 2, b - 2, l + 2, r - 2))
    return sum(bool(loop["full"] & interior) for loop in a["loops"])


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            k = _key(inp)
            _need(k not in seen, "duplicate input")
            seen.add(k)
            a = _analyze(inp)
            _need(np.array_equal(out, a["output"]), "incorrect edge-stamp output")
            _need(np.any(out == a["main"]["ink"]), "generated example must capture wire")
            _need(out.shape[0] + 4 <= inp.shape[0] and out.shape[1] + 4 <= inp.shape[1], "crop size")
            _need(1 <= _crossings(a) <= (4 if split == "test" else 2), "crossing count regime")
            if split == "test":
                _need(_crossings(a) >= 3 and len(a["panes"]) >= 2, "test shift/distractor")
            analyses[split].append(a)
    first, second = analyses["train"][:2]
    _need(first["main"]["fill"] == first["background"] and
          second["main"]["fill"] != second["background"], "fill demonstrations")
    _need(len(second["panes"]) >= 2, "largest-pane demonstration")
    w0, w1 = _witnesses(first), _witnesses(second)
    _need(all(w0[k] for k in ("independent", "movement", "perimeter")), "first teaching witnesses")
    _need(all(w1[k] for k in ("four", "keyed", "phase", "movement")), "second teaching witnesses")
    _need(len({int(v) for v in second["cargo"].flat if v >= 0}) == 4,
          "second lesson must capture all four route colors")
    _need(len({a["background"] for a in analyses["train"]}) >= 2, "pair-local palettes")


def _features(pair):
    a = _analyze(pair["input"])
    loops, m = a["loops"], a["main"]
    cargo = a["cargo"]
    pieces = _pieces(cargo)
    nested, spanning, asymmetric = 0, 0, 0
    t, b, l, r = m["box"]
    for loop in loops:
        y, z, x, v = loop["box"]
        spanning += int((y < t and z > b) or (x < l and v > r))
        for other in loops:
            yy, zz, xx, vv = other["box"]
            nested += int(y < yy < zz < z and x < xx < vv < v)
    signatures = set()
    for color, cells in pieces:
        y, z, x, v = _box(cells)
        shape = tuple(sorted((rr - y, cc - x) for rr, cc in cells))
        signatures.add(shape)
        asymmetric += int(set(shape) != {(z - y - rr, cc) for rr, cc in shape})
    return dict(input_shape=pair["input"].shape, output_shape=pair["output"].shape,
                background=a["background"], crossings=_crossings(a), loops=len(loops),
                panes=len(a["panes"]), periods=tuple(loop["model"][0] for loop in loops),
                nested=nested, spanning=spanning, pieces=len(pieces),
                directions=tuple(i for i, c in enumerate(m["keys"]) if np.any(cargo == c)),
                signatures=signatures, asymmetric=asymmetric,
                ink=m["ink"], fill_matches=m["fill"] == a["background"])


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                g = np.asarray(pair[name], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded streaming suite: 42 puzzles, small sets of features, no disk I/O."""
    inputs, outputs, masks, fingerprints = set(), set(), set(), set()
    periods, backgrounds, directions, inks, counts = set(), set(), set(), set(), set()
    nested = spanning = asymmetric = exterior = 0
    piece_counts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                inputs.add(f["input_shape"])
                outputs.add(f["output_shape"])
                masks.update(f["signatures"])
                periods.update(f["periods"])
                backgrounds.add(f["background"])
                inks.add(f["ink"])
                directions.update(f["directions"])
                counts.add(f["crossings"])
                piece_counts.add(f["pieces"])
                nested += f["nested"]
                spanning += f["spanning"]
                asymmetric += f["asymmetric"]
                exterior += f["loops"] > f["crossings"]
    _need(len(fingerprints) == 42, "cross-seed diversity")
    _need(len(inputs) >= 35 and len(outputs) >= 25 and len(masks) >= 35, "structural diversity")
    _need(backgrounds == set(range(10)) and len(inks) >= 8 and directions == set(range(4)), "role diversity")
    _need(counts == {1, 2, 3, 4} and len(piece_counts) >= 8, "count diversity")
    _need(periods >= {1, 2, 3, 4, 5, 6}, "period diversity")
    _need(nested and spanning and asymmetric and exterior, "geometry diversity")
    # Legal extremes and every count combination, keeping only one puzzle at once.
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            generate_puzzle(8700 + 10 * n + m, n, m)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(456, 2, 1)
    _need(_fingerprint(p) == _fingerprint(generate_puzzle(456, 2, 1)), "determinism")
    # Validate lists as well as arrays and ensure inputs are never mutated.
    inp = p["test"][0]["input"]
    snapshot = inp.copy()
    _need(np.array_equal(solve(inp.tolist()), p["test"][0]["output"]), "list input")
    _need(np.array_equal(inp, snapshot), "solve mutated input")
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, input_shapes=len(inputs), output_shapes=len(outputs),
                component_masks=len(masks), primitive_periods=sorted(periods),
                crossing_counts=sorted(counts), distinct_piece_counts=len(piece_counts),
                nested_pairs=nested, spanning_loops=spanning, exterior_examples=exterior)
