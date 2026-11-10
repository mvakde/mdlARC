"""Reference-band motif completion puzzles."""

import hashlib
import numpy as np


# Hashes of the four original inputs. Rejecting these also rejects exact pairs;
# no original output is stored or used by the generator.
_SOURCE_INPUT_HASHES = {
    "2a130acf515a93331c856c352ca0fe66ee38f444f744f2e7eec923291d9283d2",
    "1fad5b450a0ce4e8c798b62a410583041f0b2d9a230aeda430e1fa2015787972",
    "75428642e61d3e2d0ed7723465fbf0b827552874ddb15bc43bc46c9e450f8f74",
    "e065ebba1a5f480feaab234357de68c2261dcc74f6cd76864dc1b1a8066b9229",
}


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        todo, group = [(y, x)], []
        while todo:
            cy, cx = todo.pop()
            group.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = cy + dy, cx + dx
                    if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                            and mask[yy, xx] and not seen[yy, xx]):
                        seen[yy, xx] = True
                        todo.append((yy, xx))
        groups.append(group)
    return groups


def _connected(cells):
    if not cells:
        return False
    pending = [next(iter(cells))]
    reached = set()
    while pending:
        y, x = pending.pop()
        if (y, x) in reached:
            continue
        reached.add((y, x))
        pending.extend((yy, xx) for yy, xx in cells
                       if max(abs(y - yy), abs(x - xx)) <= 1
                       and (yy, xx) not in reached)
    return len(reached) == len(cells)


def _motif(kind, a, b, bg, rng):
    if kind == "frame":
        h = int(rng.choice([3, 3, 4, 5]))
        w = int(rng.choice([3, 3, 4, 5]))
        t = np.full((h, w), b, dtype=np.int8)
        t[0, :] = t[-1, :] = a
        t[:, 0] = t[:, -1] = a
    elif kind == "triangle":
        t = np.full((3, 3), a, dtype=np.int8)
        t[0, 2] = t[1, 1] = t[2, 0] = b
        t = np.rot90(t, int(rng.integers(4))).copy()
    elif kind == "checker":
        h = int(rng.choice([2, 2, 3, 4]))
        t = np.fromfunction(lambda y, x: np.where((y+x) % 2, b, a),
                            (h, h), dtype=int).astype(np.int8)
    elif kind == "bar":
        h, w = int(rng.integers(2, 5)), int(rng.integers(4, 7))
        t = np.full((h, w), b, dtype=np.int8)
        t[:, 0] = t[:, -1] = a
        if rng.integers(2):
            t = t.T.copy()
    elif kind == "stripe":
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        t = np.full((h, w), a, dtype=np.int8)
        t[h // 2, :] = b
        if rng.integers(2):
            t = t.T.copy()
    elif kind == "gap":
        h, w = int(rng.integers(4, 6)), int(rng.integers(4, 6))
        t = np.full((h, w), bg, dtype=np.int8)
        t[0, :] = t[-1, :] = a
        t[1:-1, 1:-1] = b
        if rng.integers(2):
            t = t.T.copy()
    elif kind == "diagonal":
        n = int(rng.integers(4, 7))
        t = np.full((n, n), bg, dtype=np.int8)
        for i in range(n):
            t[i, i] = a if i in (0, n-1) else b
            t[i, min(n-1, i+1)] = a if i in (0, n-1) else b
        t = np.rot90(t, int(rng.integers(4))).copy()
    elif kind == "plus":
        n = int(rng.choice([3, 5]))
        t = np.full((n, n), bg, dtype=np.int8)
        t[n//2, :] = a
        t[:, n//2] = a
        t[n//2, n//2] = b
        if n == 5:
            t[1, 1] = t[1, 3] = t[3, 1] = t[3, 3] = b
    else:
        raise ValueError(kind)
    return t


def _rectangular(cells):
    ys = [p[0] for p in cells]
    xs = [p[1] for p in cells]
    y0, x0, y1, x1 = min(ys), min(xs), max(ys)+1, max(xs)+1
    return len(cells) == (y1-y0)*(x1-x0), (y0, x0, y1, x1)


def _matches(cells, templates, bg, bounds):
    """Visible-fragment policy: solid footprint, enclosed core, or colored clue."""
    rectangle, (fy0, fx0, fy1, fx1) = _rectangular(cells)
    values = set(cells.values())
    hits = []
    if rectangle and len(values) == 1:
        c = next(iter(values))
        fh, fw = fy1-fy0, fx1-fx0
        for j, t in enumerate(templates):
            h, w = t.shape
            if (fh, fw) == (h, w) and np.all(t != bg):
                same = int(np.count_nonzero(t == c))
                if same >= (h*w+1)//2 and same < h*w:
                    hits.append((j, fy0, fx0, h*w-same))
            # A core must be enclosed in the reference's bounding box.
            for yy in range(1, h-fh):
                for xx in range(1, w-fw):
                    if np.all(t[yy:yy+fh, xx:xx+fw] == c):
                        if np.count_nonzero(t != bg) > fh*fw:
                            hits.append((j, fy0-yy, fx0-xx, 0))
    elif len(cells) >= 3:
        for j, t in enumerate(templates):
            for yy, xx in np.argwhere(t != bg):
                ay, ax = fy0-int(yy), fx0-int(xx)
                if not all(0 <= y-ay < t.shape[0] and 0 <= x-ax < t.shape[1]
                           and t[y-ay, x-ax] != bg for y, x in cells):
                    continue
                errors = sum(int(t[y-ay, x-ax]) != c
                             for (y, x), c in cells.items())
                if errors > min(2, (len(cells)-2)//2):
                    continue
                exact_colors = {c for (y, x), c in cells.items()
                                if int(t[y-ay, x-ax]) == c}
                if len(values) > 1 and len(exact_colors) < 2:
                    continue
                hits.append((j, ay, ax, errors))
    if bounds is not None:
        y0, y1, x0, x1 = bounds
        hits = [(j, ay, ax, e) for j, ay, ax, e in hits
                if y0 <= ay and x0 <= ax
                and ay+templates[j].shape[0] <= y1
                and ax+templates[j].shape[1] <= x1]
    if not hits:
        return None
    best = min(e for _, _, _, e in hits)
    winners = {(j, ay, ax) for j, ay, ax, e in hits if e == best}
    if len(winners) != 1:
        return None
    return (*next(iter(winners)), best)


def _read_key(inp):
    h, w = inp.shape
    colors, counts = np.unique(inp, return_counts=True)
    dominant = int(colors[np.argmax(counts)])
    edges = (("top", inp[0]), ("bottom", inp[-1]),
             ("left", inp[:, 0]), ("right", inp[:, -1]))
    sides = []
    for side, line in edges:
        edge_colors, edge_counts = np.unique(line, return_counts=True)
        index = int(np.argmax(edge_counts))
        color = int(edge_colors[index])
        if color != dominant and edge_counts[index] >= .6*len(line):
            sides.append((side, color))
    assert len(sides) == 1, "reference band must have one full edge"
    side, panel_color = sides[0]
    if side == "top":
        thickness = next((y for y in range(h) if
                          np.count_nonzero(inp[y] == dominant) > .45*w), h)
        panel = inp[:thickness]
        bounds = (thickness, h, 0, w)
    elif side == "bottom":
        thickness = next((y for y in range(h) if
                          np.count_nonzero(inp[h-1-y] == dominant) > .45*w), h)
        panel = inp[h-thickness:]
        bounds = (0, h-thickness, 0, w)
    elif side == "left":
        thickness = next((x for x in range(w) if
                          np.count_nonzero(inp[:, x] == dominant) > .45*h), w)
        panel = inp[:, :thickness]
        bounds = (0, h, thickness, w)
    else:
        thickness = next((x for x in range(w) if
                          np.count_nonzero(inp[:, w-1-x] == dominant) > .45*h), w)
        panel = inp[:, w-thickness:]
        bounds = (0, h, 0, w-thickness)
    y0, y1, x0, x1 = bounds
    workspace = inp[y0:y1, x0:x1]
    colors, counts = np.unique(workspace, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert bg != panel_color and 4 <= thickness <= 10
    templates = []
    for comp in _components(panel != panel_color):
        ys, xs = zip(*comp)
        ry, rx = min(ys), min(xs)
        t = np.full((max(ys)-ry+1, max(xs)-rx+1), bg, dtype=np.int8)
        for y, x in comp:
            t[y-ry, x-rx] = panel[y, x]
        templates.append(t)
    return side, thickness, panel_color, bg, bounds, templates


def _solve(inp):
    side, thick, panel, bg, bounds, templates = _read_key(inp)
    y0, y1, x0, x1 = bounds
    workspace = inp[y0:y1, x0:x1]
    out = inp.copy()
    matches, unchanged = [], []
    for comp in _components(workspace != bg):
        cells = {(y+y0, x+x0): int(inp[y+y0, x+x0]) for y, x in comp}
        hit = _matches(cells, templates, bg, bounds)
        if hit is None:
            unchanged.append(cells)
            continue
        j, ay, ax, errors = hit
        t = templates[j]
        for dy, dx in np.argwhere(t != bg):
            out[ay+dy, ax+dx] = t[dy, dx]
        matches.append((j, errors, cells))
    return out, (side, thick, panel, bg, bounds, templates, matches, unchanged)


def _place(rng, height, width, region, occupied):
    y0, y1, x0, x1 = region
    if height > y1-y0 or width > x1-x0:
        return None
    def free(y, x):
        return all(y+height+1 <= ey or ey+eh+1 <= y
                   or x+width+1 <= ex or ex+ew+1 <= x
                   for ey, ex, eh, ew in occupied)
    for _ in range(160):
        y = int(rng.integers(y0, y1-height+1))
        x = int(rng.integers(x0, x1-width+1))
        if free(y, x):
            occupied.append((y, x, height, width))
            return y, x
    locations = [(y, x) for y in range(y0, y1-height+1)
                 for x in range(x0, x1-width+1) if free(y, x)]
    if not locations:
        return None
    y, x = locations[int(rng.integers(len(locations)))]
    occupied.append((y, x, height, width))
    return y, x


def _fragment_options(t, bg, rng):
    h, w = t.shape
    full = {(int(y), int(x)): int(t[y, x]) for y, x in np.argwhere(t != bg)}
    options = []
    if np.all(t != bg):
        for c in set(full.values()):
            if sum(v == c for v in full.values()) >= (len(full)+1)//2:
                options.append(({p: c for p in full}, "recolor"))
    for yy in range(1, h-1):
        for xx in range(1, w-1):
            for hh in range(1, h-yy):
                for ww in range(1, w-xx):
                    block = t[yy:yy+hh, xx:xx+ww]
                    if np.all(block == block[0, 0]) and block[0, 0] != bg:
                        cells = {(y, x): int(block[0, 0])
                                 for y in range(yy, yy+hh)
                                 for x in range(xx, xx+ww)}
                        options.append((cells, "add"))
                        if len(cells) >= 3:
                            p = list(cells)[int(rng.integers(len(cells)))]
                            altered = cells.copy()
                            altered[p] = next((v for v in full.values()
                                               if v != cells[p]), cells[p])
                            if altered[p] != cells[p]:
                                options.append((altered, "both"))
    points = list(full)
    for _ in range(18):
        n = int(rng.integers(3, len(points)+1))
        start = points[int(rng.integers(len(points)))]
        selected = {start}
        while len(selected) < n:
            frontier = [p for p in points if p not in selected
                        and any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1
                                for q in selected)]
            if not frontier:
                break
            selected.add(frontier[int(rng.integers(len(frontier)))])
        if len(selected) == len(full):
            selected.remove(next(iter(selected)))
        cells = {p: full[p] for p in selected}
        if len(cells) >= 3 and len(set(cells.values())) >= 2 and _connected(cells):
            options.append((cells, "add"))
            p = list(cells)[int(rng.integers(len(cells)))]
            alt = next((v for v in set(full.values()) if v != cells[p]), cells[p])
            if alt != cells[p]:
                changed = cells.copy()
                changed[p] = alt
                options.append((changed, "both"))
    if len(set(full.values())) >= 2:
        cells = full.copy()
        p = points[int(rng.integers(len(points)))]
        cells[p] = next(v for v in set(full.values()) if v != full[p])
        options.append((cells, "recolor"))
    return options


def _choose_colors(rng, bg, panel, count):
    pool = [c for c in range(10) if c not in (bg, panel)]
    rng.shuffle(pool)
    pairs = []
    for j in range(count):
        if j and rng.random() < .42:
            if rng.random() < .5:
                a, b = pairs[int(rng.integers(j))][::-1]
            else:
                a = pairs[int(rng.integers(j))][int(rng.integers(2))]
                b = pool[(2*j+1) % len(pool)]
                if a == b:
                    b = pool[(2*j+2) % len(pool)]
        else:
            a, b = pool[(2*j) % len(pool)], pool[(2*j+1) % len(pool)]
        pairs.append((int(a), int(b)))
    return pairs


def _pair(rng, side, regime, preference):
    for attempt in range(32):
        large = rng.random() < .76
        if rng.random() < .19:
            h = w = 30
            large = True
        else:
            h = int(rng.integers(25, 31) if large else rng.integers(20, 25))
            w = int(rng.integers(25, 31) if large else rng.integers(20, 25))
        bg = 0 if rng.random() < .67 else int(rng.choice([5, 6, 9]))
        panel = 3 if bg != 3 and rng.random() < .55 else int(
            rng.choice([c for c in range(1, 10) if c != bg]))
        k = int(rng.integers(2, 5 if large else 4))
        kinds = list(rng.choice(
            ["frame", "triangle", "checker", "bar", "stripe",
             "gap", "diagonal", "plus"], size=k, replace=False))
        colors = _choose_colors(rng, bg, panel, k)
        original = [_motif(kind, a, b, bg, rng)
                    for kind, (a, b) in zip(kinds, colors)]
        thick = max(max(t.shape) for t in original) + 2
        if thick > 10 or min(h, w)-thick < 13:
            continue
        inp = np.full((h, w), bg, dtype=np.int8)
        if side == "top":
            inp[:thick] = panel
            key_region, area = (0, thick, 0, w), (thick, h, 0, w)
        elif side == "bottom":
            inp[h-thick:] = panel
            key_region, area = (h-thick, h, 0, w), (0, h-thick, 0, w)
        elif side == "left":
            inp[:, :thick] = panel
            key_region, area = (0, h, 0, thick), (0, h, thick, w)
        else:
            inp[:, w-thick:] = panel
            key_region, area = (0, h, w-thick, w), (0, h, 0, w-thick)
        key_region = (key_region[0]+1, key_region[1]-1,
                      key_region[2]+1, key_region[3]-1)
        reserved = []
        ok = True
        for t in original:
            location = _place(rng, *t.shape, key_region, reserved)
            if location is None:
                ok = False
                break
            y, x = location
            for dy, dx in np.argwhere(t != bg):
                inp[y+dy, x+dx] = t[dy, dx]
        if not ok:
            continue
        try:
            _, _, _, _, bounds, templates = _read_key(inp)
        except AssertionError:
            continue
        if len(templates) != k:
            continue
        available = {"add": [], "recolor": [], "both": []}
        for t in original:
            for cells, mode in _fragment_options(t, bg, rng):
                hit = _matches(cells, templates, bg, None)
                if hit is None:
                    continue
                j, ay, ax, _ = hit
                if (ay, ax) == (0, 0) and np.array_equal(templates[j], t):
                    if len(cells) < 2 and mode != "add":
                        continue
                    available[mode].append((t, cells))
        required = ["add", "recolor"] if regime == "train" and preference == 0 else (
            ["both"] if regime == "train" and preference == 1 else [])
        if any(not available[m] for m in required) or not available["add"]:
            continue
        # Higher loads occur mainly on canvases with room for separated objects.
        target = int(rng.integers(9, 17) if h == w == 30 and rng.random() < .55
                     else rng.integers(8, 14) if large and rng.random() < .38
                     else rng.integers(4, 9))
        if regime == "target_test" and rng.random() < .35:
            target = max(target, int(rng.integers(8, 13)))
        distractors = int(rng.integers(0, min(4, target-1)))
        if regime == "train" and preference == 0:
            distractors = max(1, distractors)
        positive = target-distractors
        if positive < len(required):
            continue
        out = inp.copy()
        occupied = []
        def choose(mode):
            options = available[mode]
            solids = [(t, c) for t, c in options
                      if _rectangular(c)[0] and len(set(c.values())) == 1]
            if solids and rng.random() < .76:
                options = solids
            if mode == "add" and rng.random() < .18:
                single = [(t, c) for t, c in options if len(c) == 1]
                if single:
                    options = single
            return options[int(rng.integers(len(options)))]
        selected = [(m, choose(m)) for m in required]
        modes = [m for m in ("add", "recolor", "both") if available[m]]
        while len(selected) < positive:
            m = modes[int(rng.integers(len(modes)))]
            selected.append((m, choose(m)))
        rng.shuffle(selected)
        placed = 0
        for mode, (t, cells) in selected:
            location = _place(rng, *t.shape, area, occupied)
            if location is None:
                break
            y, x = location
            for (dy, dx), c in cells.items():
                inp[y+dy, x+dx] = c
            for dy, dx in np.argwhere(t != bg):
                out[y+dy, x+dx] = t[dy, dx]
            placed += 1
        if placed < max(2, len(required)) or any(
                mode not in [m for m, _ in selected[:placed]] for mode in required):
            continue
        # Distractors include key colors, panel colors, and solid blocks.
        used_colors = [int(v) for t in original for v in np.unique(t) if v != bg]
        for _ in range(distractors):
            candidates = [(int(rng.choice(used_colors)), 1, 1)]
            for _ in range(5):
                color = panel if rng.random() < .2 else int(rng.choice(used_colors))
                size = ((2, 2), (2, 3), (3, 3))[int(rng.integers(3))]
                candidates.append((color, *size))
            if rng.random() < .55:
                candidates.reverse()
            for color, dh, dw in candidates:
                location = _place(rng, dh, dw, area, occupied)
                if location is None:
                    continue
                y, x = location
                cells = {(y+dy, x+dx): color for dy in range(dh) for dx in range(dw)}
                if _matches(cells, templates, bg, bounds) is not None:
                    occupied.pop()
                    continue
                for cy, cx in cells:
                    inp[cy, cx] = out[cy, cx] = color
                break
        try:
            solved, info = _solve(inp)
        except AssertionError:
            continue
        if np.array_equal(solved, out) and not np.array_equal(inp, out):
            return {"input": inp, "output": out}
    raise RuntimeError("bounded pair construction failed")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    for name, value, low, high in (
            ("num_train", num_train, 2, 10),
            ("num_test", num_test, 1, 4)):
        if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(name)
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10)-1)+1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test+1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    # The convention shared across the puzzle is visual: a solid edge band
    # contains complete motifs and the remaining canvas is the workspace.
    # Colors, motif vocabulary, dimensions, and band side are visible pair data.
    sides = ["top", "bottom", "left", "right"]
    first = sides[int(rng.integers(4))]
    second = sides[int(rng.integers(4))]
    puzzle = {"train": [], "test": []}
    for i in range(num_train):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
        side = first if i == 0 else second if i == 1 else sides[int(rng.integers(4))]
        puzzle["train"].append(_pair(pair_rng, side, "train", i))
    for i in range(num_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
        side = "bottom" if rng.random() < .47 else sides[int(rng.integers(4))]
        puzzle["test"].append(_pair(pair_rng, side, "target_test", i))
    return puzzle


def _predict_from_examples(train_pairs, test_input):
    """Use only visible demonstrations and an unlabeled test input."""
    assert len(train_pairs) >= 2
    for pair in train_pairs:
        predicted, _ = _solve(pair["input"])
        assert np.array_equal(predicted, pair["output"])
    return _solve(test_input)[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert all(type(puzzle[s]) is list for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    train_modes = set()
    train_has_unmatched = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            both = (key, out.tobytes())
            assert key not in inputs and both not in pairs
            if inp.shape == (30, 30):
                assert hashlib.sha256(inp.astype(np.int8).tobytes()).hexdigest() not in _SOURCE_INPUT_HASHES
            inputs.add(key)
            pairs.add(both)
            solved, info = _solve(inp)
            assert np.array_equal(solved, out)
            side, thick, panel, bg, bounds, templates, matches, unchanged = info
            assert 2 <= len(templates) <= 4
            assert len(matches) >= 2
            assert all(len(set(int(v) for v in np.unique(t) if v != bg)) >= 2
                       for t in templates)
            y0, y1, x0, x1 = bounds
            assert not np.any(inp[y0:y1, x0:x1] == panel) or panel != bg
            assert np.array_equal(inp[:y0], out[:y0])
            assert np.array_equal(inp[y1:], out[y1:])
            assert np.array_equal(inp[:, :x0], out[:, :x0])
            assert np.array_equal(inp[:, x1:], out[:, x1:])
            if split == "train":
                train_has_unmatched |= bool(unchanged)
                for _, _, cells in matches:
                    j, ay, ax, _ = _matches(cells, templates, bg, bounds)
                    t = templates[j]
                    added = recolored = False
                    for dy, dx in np.argwhere(t != bg):
                        old, new = inp[ay+dy, ax+dx], out[ay+dy, ax+dx]
                        added |= old == bg and new != bg
                        recolored |= old != bg and old != new
                    if added and recolored:
                        train_modes.add("both")
                    elif added:
                        train_modes.add("add")
                    elif recolored:
                        train_modes.add("recolor")
    assert train_modes == {"add", "recolor", "both"}
    assert train_has_unmatched
    copies = [{"input": p["input"].copy(), "output": p["output"].copy()}
              for p in puzzle["train"]]
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_examples(copies, pair["input"].copy()),
                              pair["output"])


def _equal(a, b):
    return all(np.array_equal(pa[k], pb[k])
               for s in ("train", "test")
               for pa, pb in zip(a[s], b[s])
               for k in ("input", "output")) and all(
                   len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sides, shapes, kinds, structures, palettes = set(), set(), set(), set(), set()
    backgrounds, bands, key_counts = set(), set(), set()
    stats = {name: 0 for name in (
        "multi", "middle", "solid", "singletons", "add", "recolor", "both",
        "overlap", "no_common", "unused", "repeat", "zero_noise",
        "many_noise", "panel_noise", "key_noise", "crowded", "simple",
        "bottom_train", "bottom_test", "right_train", "same_side",
        "thirty", "nonthirty", "within_shapes", "within_palettes",
        "overlap_crowded", "nonzero_bottom")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_sides = {"train": set(), "test": set()}
        puzzle_shapes, puzzle_palettes = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                solved, info = _solve(inp)
                assert np.array_equal(solved, out)
                side, thick, panel, bg, bounds, templates, matched, unmatched = info
                sides.add(side)
                shapes.add(inp.shape)
                backgrounds.add(bg)
                bands.add(panel)
                key_counts.add(len(templates))
                puzzle_sides[split].add(side)
                puzzle_shapes.add(inp.shape)
                puzzle_palettes.add((bg, panel))
                kinds.add(tuple(sorted((t.shape, int(np.count_nonzero(t != bg)))
                                       for t in templates)))
                palettes.add((bg, panel, tuple(sorted(np.unique(inp).tolist()))))
                structures.add(hashlib.sha256(inp.tobytes()).hexdigest())
                stats["thirty" if inp.shape == (30, 30) else "nonthirty"] += 1
                stats["crowded" if len(matched)+len(unmatched) >= 11
                      else "simple" if len(matched)+len(unmatched) <= 5
                      else "middle"] += 1
                if split == "train" and side == "bottom":
                    stats["bottom_train"] += 1
                if split == "train" and side == "right":
                    stats["right_train"] += 1
                if split == "test" and side == "bottom":
                    stats["bottom_test"] += 1
                if bg != 0 and side == "bottom":
                    stats["nonzero_bottom"] += 1
                color_sets = [set(int(v) for v in np.unique(t) if v != bg)
                              for t in templates]
                overlap = any(color_sets[a] & color_sets[b]
                              for a in range(len(templates))
                              for b in range(a+1, len(templates)))
                stats["overlap"] += overlap
                stats["no_common"] += not set.intersection(*color_sets)
                stats["overlap_crowded"] += overlap and len(matched)+len(unmatched) >= 11
                copies = [sum(j == n for j, _, _ in matched)
                          for n in range(len(templates))]
                stats["unused"] += any(n == 0 for n in copies)
                stats["repeat"] += any(n >= 3 for n in copies)
                stats["zero_noise"] += len(unmatched) == 0
                stats["many_noise"] += len(unmatched) >= 2
                key_colors = set.union(*color_sets)
                for cells in unmatched:
                    stats["panel_noise"] += panel in cells.values()
                    stats["key_noise"] += any(c in key_colors for c in cells.values())
                for _, _, cells in matched + [(None, None, c) for c in unmatched]:
                    rectangle, _ = _rectangular(cells)
                    if len(cells) >= 2:
                        stats["multi"] += 1
                        stats["solid"] += rectangle and len(set(cells.values())) == 1
                for _, _, cells in matched:
                    stats["singletons"] += len(cells) == 1
                    j, ay, ax, _ = _matches(cells, templates, bg, bounds)
                    t = templates[j]
                    add = recolor = False
                    for dy, dx in np.argwhere(t != bg):
                        old, new = inp[ay+dy, ax+dx], out[ay+dy, ax+dx]
                        add |= old == bg and new != bg
                        recolor |= old != bg and old != new
                    stats["both" if add and recolor else
                          "add" if add else "recolor"] += 1
        stats["same_side"] += bool(puzzle_sides["train"] & puzzle_sides["test"])
        stats["within_shapes"] += len(puzzle_shapes) >= 2
        stats["within_palettes"] += len(puzzle_palettes) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert sides == {"top", "bottom", "left", "right"}
    assert key_counts == {2, 3, 4}
    assert len(shapes) >= 20 and len(kinds) >= 30
    assert len(structures) >= 200 and len(palettes) >= 30
    assert len(backgrounds) >= 3 and len(bands) >= 6
    assert stats["solid"] >= .35*stats["multi"]
    for name, minimum in (
            ("singletons", 30), ("add", 60), ("recolor", 60),
            ("both", 60), ("overlap", 30), ("no_common", 30),
            ("unused", 10), ("repeat", 10), ("zero_noise", 10),
            ("many_noise", 10), ("panel_noise", 5), ("key_noise", 20),
            ("crowded", 10), ("simple", 10), ("bottom_train", 5),
            ("bottom_test", 5), ("right_train", 5), ("same_side", 10),
            ("thirty", 10), ("nonthirty", 100), ("within_shapes", 30),
            ("within_palettes", 20), ("overlap_crowded", 3),
            ("nonzero_bottom", 5)):
        assert stats[name] >= minimum, (name, stats[name])
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        p = generate_puzzle(1000, train, test)
        validate_generated_puzzle(p)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        assert _equal(p, generate_puzzle(1000, train, test))
    for train, test in ((1, None), (11, None), (None, 0),
                        (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(7, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
