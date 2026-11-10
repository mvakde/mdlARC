"""L-ray occupancy determines the unique largest empty 2-D rectangle.

Only stdlib and NumPy; no parent modules, files, seed state, or examples in solve.
Highlight is the fixed family color 6. Other colors are local visible roles.
Generation and self-test retain no files or execution traces.
"""
from __future__ import annotations

import hashlib
import numpy as np

HIGHLIGHT = 6
_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] * 2 > a.size, "background must be a strict majority")
    _require(HIGHLIGHT not in a, "highlight is output-only")
    h, w = a.shape
    seen = np.zeros(a.shape, dtype=bool)
    sources, mirrors = [], []
    for r, c in zip(*np.nonzero(a != bg)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        color = int(a[r, c])
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and a[yy, xx] == color:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        cells = tuple(sorted(cells))
        y0, y1 = min(y for y, x in cells), max(y for y, x in cells)
        x0, x1 = min(x for y, x in cells), max(x for y, x in cells)
        if len(cells) == 3 and y1-y0 == x1-x0 == 1:
            missing = next(p for p in ((y0, x0), (y0, x1), (y1, x0), (y1, x1)) if p not in cells)
            elbow = (y0+y1-missing[0], x0+x1-missing[1])
            direction = (elbow[0]-missing[0], elbow[1]-missing[1])
            sources.append((elbow, direction, color, cells))
        else:
            _require(len(cells) % 2 == 1 and (y0 == y1 or x0 == x1), "invalid object")
            _require(len(cells) == max(y1-y0, x1-x0)+1, "noncontiguous bar")
            axes = (0, 1) if len(cells) == 1 else (0,) if y0 == y1 else (1,)
            mirrors.append(((y0+y1)//2, (x0+x1)//2, color, axes, cells))
    _require(bool(sources), "at least one L is required")
    return bg, sources, mirrors


def _trace(a, *, reflect=True, source_indices=None):
    """Cell tracing; original objects are obstacles, not newly painted ray cells."""
    bg, sources, mirrors = _parse(a)
    h, w = a.shape
    out = a.copy()
    painted, routes = {}, []
    chosen = range(len(sources)) if source_indices is None else source_indices
    for source_index in chosen:
        elbow, direction, color, _ = sources[source_index]
        r, c = elbow
        dr, dc = direction
        visited, path, contacts = set(), [], []
        while True:
            r, c = r+dr, c+dc
            if not (0 <= r < h and 0 <= c < w):
                break
            state = (r, c, dr, dc)
            _require(state not in visited, "ray loop")
            visited.add(state)
            _require(a[r, c] == bg, "ray hits an original object")
            hits = []
            if reflect:
                for i, (y, x, shade, axes, _) in enumerate(mirrors):
                    if 0 in axes and c == x and y-r == dr:
                        hits.append((i, 0, shade))
                    if 1 in axes and r == y and x-c == dc:
                        hits.append((i, 1, shade))
            _require(len(hits) <= 1, "ambiguous contact")
            if hits:
                i, axis, shade = hits[0]
                contacts.append((i, axis, (r, c), color, shade))
                color = shade
                dr, dc = (-dr, dc) if axis == 0 else (dr, -dc)
            _require((r, c) not in painted or painted[r, c] == color, "conflicting rays")
            painted[r, c] = color
            out[r, c] = color
            path.append((r, c))
        routes.append({"source": source_index, "path": tuple(path), "contacts": tuple(contacts),
                       "tail_direction": (dr, dc)})
    return out, {"background": bg, "sources": sources, "mirrors": mirrors, "routes": routes}


def _rectangles(empty):
    """Maximal-width runs for each row interval suffice for a global maximum."""
    h, w = empty.shape
    bits = [sum(1 << int(c) for c in np.flatnonzero(row)) for row in empty]
    candidates = []
    for top in range(h-1):
        common = bits[top]
        for bottom in range(top+1, h):
            common &= bits[bottom]
            if not common:
                break
            rest = common
            while rest:
                left = (rest & -rest).bit_length()-1
                shifted = rest >> left
                width = (shifted ^ (shifted+1)).bit_length()-1
                rest &= ~(((1 << width)-1) << left)
                if width >= 2:
                    height = bottom-top+1
                    candidates.append((height*width, top, left, height, width))
    return candidates


def _best(empty):
    candidates = _rectangles(empty)
    if not candidates:
        return 0, []
    area = max(item[0] for item in candidates)
    return area, [item[1:] for item in candidates if item[0] == area]


def _mask(shape, rect):
    top, left, h, w = rect
    result = np.zeros(shape, dtype=bool)
    result[top:top+h, left:left+w] = True
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    out, info = _trace(a)
    _, winners = _best(out == info["background"])
    _require(len(winners) == 1, "largest eligible rectangle must be unique")
    out[_mask(a.shape, winners[0])] = HIGHLIGHT
    return out


def _key(a):
    return a.shape, a.tobytes()


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return ((r, c), (r-dr, c), (r, c-dc))


def _put(a, cells, color, bg):
    h, w = a.shape
    cells = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or a[r, c] != bg:
            return None
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (y, x) not in cells and 0 <= y < h and 0 <= x < w and a[y, x] == color:
                return None
    trial = a.copy()
    for p in cells:
        trial[p] = color
    return trial


def _add_source(a, rng, bg, palette):
    h, w = a.shape
    for _ in range(64):
        dr, dc = _DIRECTIONS[int(rng.integers(4))]
        r, c = int(rng.integers(h)), int(rng.integers(w))
        if rng.random() < .35:
            if rng.integers(2):
                r = 1 if dr > 0 else h-2
            else:
                c = 1 if dc > 0 else w-2
        trial = _put(a, _source_cells((r, c), (dr, dc)), int(rng.choice(palette)), bg)
        if trial is None:
            continue
        try:
            _, info = _trace(trial)
        except ValueError:
            continue
        index = next(i for i, s in enumerate(info["sources"]) if s[0] == (r, c))
        if len(info["routes"][index]["path"]) >= 3:
            return trial, (r, c)
    return None


def _extend(a, elbow, rng, palette, axis=None, length=None, recolor=None):
    _, info = _trace(a)
    index = next(i for i, s in enumerate(info["sources"]) if s[0] == elbow)
    route = info["routes"][index]
    anchor = route["contacts"][-1][2] if route["contacts"] else elbow
    incoming = route["contacts"][-1][4] if route["contacts"] else info["sources"][index][2]
    dr, dc = route["tail_direction"]
    h, w = a.shape
    distance = min(h-1-anchor[0] if dr > 0 else anchor[0], w-1-anchor[1] if dc > 0 else anchor[1])
    options = [(step, ax, n) for step in range(1, distance+1)
               for ax in ((0, 1) if axis is None else (axis,))
               for n in ((1, 3, 5, 7) if length is None else (length,))]
    rng.shuffle(options)
    for step, ax, n in options[:64]:
        contact = (anchor[0]+step*dr, anchor[1]+step*dc)
        nr, nc = (-dr, dc) if ax == 0 else (dr, -dc)
        after = (contact[0]+nr, contact[1]+nc)
        if not (0 <= after[0] < h and 0 <= after[1] < w):
            continue
        mid = (contact[0]+(dr if ax == 0 else 0), contact[1]+(dc if ax == 1 else 0))
        cells = [(mid[0]+(j if ax == 1 else 0), mid[1]+(j if ax == 0 else 0))
                 for j in range(-(n//2), n//2+1)]
        if recolor is True:
            alternatives = [c for c in palette if c != incoming]
            if not alternatives:
                return None
            color = int(rng.choice(alternatives))
        elif recolor is False:
            color = incoming
        else:
            color = incoming if rng.random() < .30 else int(rng.choice(palette))
        trial = _put(a, cells, color, info["background"])
        if trial is None:
            continue
        try:
            _, after_info = _trace(trial)
        except ValueError:
            continue
        j = next(i for i, s in enumerate(after_info["sources"]) if s[0] == elbow)
        new = after_info["routes"][j]["contacts"]
        def geometry(contacts, mirrors):
            return [(mirrors[i][:2], axis, pos) for i, axis, pos, _, _ in contacts]
        if geometry(new, after_info["mirrors"]) == geometry(route["contacts"], info["mirrors"]) + [(mid, ax, contact)]:
            return trial
    return None


def _palette(rng, minimum=1):
    bg = int(rng.choice([c for c in range(10) if c != HIGHLIGHT]))
    colors = [c for c in range(10) if c not in (bg, HIGHLIGHT)]
    rng.shuffle(colors)
    return bg, colors[:int(rng.integers(minimum, 6))]


def _transform(a, rng):
    a = np.rot90(a, int(rng.integers(4)))
    if rng.integers(2):
        a = a[:, ::-1]
    return a.copy()


def _add_unhit_mirrors(a, rng, palette):
    """Off-route reflectors still constrain empty rectangles, with ordinary semantics."""
    traced, before = _trace(a)
    paths = [r["path"] for r in before["routes"]]
    ray_cells = traced != a
    wanted = int(rng.integers(3, 13))
    placed = 0
    h, w = a.shape
    for _ in range(wanted*8):
        if placed == wanted:
            break
        y, x = int(rng.integers(h)), int(rng.integers(w))
        axis = int(rng.integers(2))
        length = int(rng.choice([1, 3, 5, 7, 9]))
        cells = [(y+(j if axis else 0), x+(j if not axis else 0))
                 for j in range(-(length//2), length//2+1)]
        if any(0 <= r < h and 0 <= c < w and ray_cells[r, c] for r, c in cells):
            continue
        trial = _put(a, cells, int(rng.choice(palette)), before["background"])
        if trial is None:
            continue
        try:
            new_trace, after = _trace(trial)
        except ValueError:
            continue
        if [r["path"] for r in after["routes"]] != paths or not np.array_equal(new_trace[ray_cells], traced[ray_cells]):
            continue
        a = trial
        placed += 1
    return a


def _longest(empty):
    longest = 0
    for rows in (empty, empty.T):
        for row in rows:
            run = 0
            for value in row:
                run = run+1 if value else 0
                longest = max(longest, run)
    return longest


def _features(a):
    traced, info = _trace(a)
    bg = info["background"]
    area, winners = _best(traced == bg)
    old_area, old_winners = _best(a == bg)
    _require(len(winners) == 1 and len(old_winners) == 1, "tied maximum")
    _require(old_area > area, "rays must strictly reduce maximum rectangle area")
    rect = winners[0]
    remaining = traced == bg
    remaining[_mask(a.shape, rect)] = False
    candidates = _rectangles(traced == bg)
    top, left, rh, rw = rect
    rivals = [item for item in candidates if not (top <= item[1] and left <= item[2]
              and item[1]+item[3] <= top+rh and item[2]+item[4] <= left+rw)]
    # A rectangle belongs to a larger region if some adjacent empty cell exists.
    target_mask = _mask(a.shape, rect)
    adjacent = np.zeros(a.shape, dtype=bool)
    adjacent[1:] |= target_mask[:-1]
    adjacent[:-1] |= target_mask[1:]
    adjacent[:, 1:] |= target_mask[:, :-1]
    adjacent[:, :-1] |= target_mask[:, 1:]
    return {"traced": traced, "info": info, "area": area, "rect": rect,
            "old_area": old_area, "old_rect": old_winners[0],
            "long_line": _longest(remaining) > area,
            "attachments": int(np.count_nonzero(adjacent & remaining)),
            "runner_up": max((v[0] for v in rivals), default=0),
            "wider_loser": any(v[4] > rw for v in rivals),
            "taller_loser": any(v[3] > rh for v in rivals)}


def _corridor_scene(rng):
    """A teaching regime, not a rule: long unilluminated 1-D corridors."""
    bg, palette = _palette(rng)
    width = int(rng.integers(24, 31))
    top_height = int(rng.integers(5, 9))
    a = np.full((top_height+4, width), bg, dtype=np.int8)
    length = width if width % 2 else width-1
    gap = int(rng.integers(width-length+1))
    for row in (top_height, top_height+2):
        a[row, gap:gap+length] = int(rng.choice(palette))
    n = int(rng.integers(5, 8))
    columns = np.linspace(1, width-2, n).astype(int)
    dc = int(rng.choice([-1, 1]))
    elbows = []
    for col in columns:
        elbow = (top_height-2, int(col))
        trial = _put(a, _source_cells(elbow, (-1, dc)), int(rng.choice(palette)), bg)
        if trial is None:
            return None
        a = trial
        elbows.append(elbow)
    try:
        _trace(a)
    except ValueError:
        return None
    rng.shuffle(elbows)
    for elbow in elbows:
        trial = _extend(a, elbow, rng, palette, axis=1, length=3, recolor=False)
        if trial is not None:
            return _transform(trial, rng)
    return None


def _general_scene(rng, regime):
    bg, palette = _palette(rng, minimum=2 if regime == "chain_demo" else 1)
    low, high = (12, 25) if regime == "chain_demo" else (18, 31) if regime == "test" else (10, 29)
    h, w = int(rng.integers(low, high)), int(rng.integers(low, high))
    a = np.full((h, w), bg, dtype=np.int8)
    n = 1 if regime == "chain_demo" else int(rng.integers(2, 6)) if regime == "test" else int(rng.integers(1, 6))
    first_axes = list(rng.permutation(2))
    for i in range(n):
        added = _add_source(a, rng, bg, palette)
        if added is None:
            return None
        a, elbow = added
        bounces = (2 if regime == "chain_demo" else int(rng.integers(2, 5)) if regime == "test" else int(rng.integers(0, 4))) if i == 0 else int(rng.integers(0, 3))
        for j in range(bounces):
            ax = int(first_axes[j]) if regime == "chain_demo" and j < 2 else None
            length = int(rng.choice([3, 5])) if regime == "chain_demo" and j == 0 else 1 if regime == "chain_demo" else None
            recolor = True if regime == "chain_demo" and j == 0 else None
            trial = _extend(a, elbow, rng, palette, axis=ax, length=length, recolor=recolor)
            if trial is None:
                if i == 0 and regime in ("chain_demo", "test"):
                    return None
                break
            a = trial
    if rng.random() < (.25 if regime == "chain_demo" else .60):
        a = _add_unhit_mirrors(a, rng, palette)
    return _transform(a, rng)


def _teaching_coverage(features):
    contacts = [(f, c) for f in features for r in f["info"]["routes"] for c in r["contacts"]]
    bar_axes = {c[1] for f, c in contacts if len(f["info"]["mirrors"][c[0]][4]) > 1}
    point = any(len(f["info"]["mirrors"][c[0]][4]) == 1 for f, c in contacts)
    directions = {s[1] for f in features for s in f["info"]["sources"]}
    return (bar_axes == {0, 1} and point and len(directions) >= 2
            and any(c[3] == c[4] for f, c in contacts)
            and any(c[3] != c[4] for f, c in contacts)
            and any(f["wider_loser"] for f in features)
            and any(f["taller_loser"] for f in features))


def _pair(rng, regime, seen, forbidden_shapes=(), teaching=()):
    for _ in range(192):
        a = _corridor_scene(rng) if regime == "line_demo" else _general_scene(rng, regime)
        if a is None or a.shape in forbidden_shapes or _key(a) in seen:
            continue
        try:
            f = _features(a)
        except ValueError:
            continue
        if regime == "line_demo" and not f["long_line"]:
            continue
        if regime == "test" and max(len(r["contacts"]) for r in f["info"]["routes"]) < 2:
            continue
        if regime == "chain_demo" and not (f["attachments"] and f["runner_up"] >= .60*f["area"]):
            continue
        if regime == "chain_demo" and not _teaching_coverage([*teaching, f]):
            continue
        out = f["traced"].copy()
        out[_mask(a.shape, f["rect"])] = HIGHLIGHT
        return {"input": a, "output": out}
    raise RuntimeError("bounded scene search exhausted: " + regime)


def _counts(num_train, num_test):
    for value, lower, upper in ((num_train, 2, 10), (num_test, 1, 4)):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and lower <= value <= upper, "illegal example count")
    _require(num_train > num_test, "num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            regime = ("line_demo" if i == 0 else "chain_demo" if i == 1 else "mixed") if split == "train" else "test"
            shapes = {p["input"].shape for p in puzzle["train"]} if split == "test" else ()
            teaching = [_features(puzzle["train"][0]["input"])] if regime == "chain_demo" else ()
            pair = _pair(rng, regime, seen, shapes, teaching)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape and _key(a) not in seen, "shape mismatch or duplicate input")
            seen.add(_key(a))
            f = _features(a)
            target = _mask(a.shape, f["rect"])
            _require(np.array_equal(out[target], np.full(f["area"], HIGHLIGHT)), "wrong rectangle fill")
            _require(np.array_equal(out[~target], f["traced"][~target]), "incorrect tracing or preservation")
            _require(np.array_equal(out[a != f["info"]["background"]], a[a != f["info"]["background"]]),
                     "original objects changed")
            _require(np.array_equal(solve(a.copy()), out), "solve disagrees")
            _require(np.any(f["traced"] != a), "no ray cells")
            _require(not np.any(target & (f["traced"] != a)), "highlight intersects a ray")
            if split == "train":
                train_features.append(f)
            else:
                _require(len(f["info"]["sources"]) >= 2, "test lacks source recombination")
                _require(max(len(r["contacts"]) for r in f["info"]["routes"]) >= 2, "test lacks chained reflections")
                _require(a.shape not in {v["traced"].shape for v in train_features}, "test shape was seen in training")
    _require(any(f["long_line"] for f in train_features), "training lacks a longer 1-D distractor")
    _require(any(f["attachments"] and f["runner_up"] >= .60*f["area"] for f in train_features),
             "training lacks attached candidates and area comparisons")
    _require(_teaching_coverage(train_features), "training lacks directions, mirror types, axes, or color evidence")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    # Stream 42 seeds: at most two puzzles live, no samples/logs written.
    shapes, targets, backgrounds, source_counts, mirror_lengths = set(), set(), set(), set(), set()
    directions, structures, digests = set(), set(), set()
    changed_contacts = unchanged_contacts = repeated_axes = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = _digest(p)
        _require(digest not in digests, "identical puzzles at different seeds")
        digests.add(digest)
        if seed < 2:
            _require(digest == _digest(generate_puzzle(seed)), "not deterministic")
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                f = _features(a)
                info = f["info"]
                shapes.add(a.shape)
                targets.add(f["rect"][2:])
                backgrounds.add(info["background"])
                source_counts.add(len(info["sources"]))
                directions.update(s[1] for s in info["sources"])
                mirror_lengths.update(len(m[4]) for m in info["mirrors"])
                mapping = {info["background"]: 0}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in a.flat)
                structures.add((a.shape, normalized))
                for route in info["routes"]:
                    axes = [c[1] for c in route["contacts"]]
                    repeated_axes += any(x == y for x, y in zip(axes, axes[1:]))
                    for _, _, _, incoming, outgoing in route["contacts"]:
                        changed_contacts += incoming != outgoing
                        unchanged_contacts += incoming == outgoing
    _require(len(shapes) >= 50 and len(targets) >= 20 and len(structures) >= 190, "insufficient geometric variety")
    _require(backgrounds == set(range(10)) - {HIGHLIGHT} and len(source_counts) >= 6, "insufficient role/count variation")
    _require(directions == set(_DIRECTIONS) and {1, 3, 5, 7} <= mirror_lengths, "insufficient trajectory variation")
    _require(changed_contacts > 20 and unchanged_contacts > 20 and repeated_axes > 0, "insufficient reflection variety")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "incorrect counts")
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Mutation checks are semantic, not snapshots of the construction recipe.
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    f = _features(p["test"][0]["input"])
    p["test"][0]["output"][f["rect"][0], f["rect"][1]] = f["info"]["background"]
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("missing highlight cell accepted")
    _rule_regressions()


def _rule_regressions():
    # Hand-enumerated route: first reflector-colored cell is the contact.
    a = np.zeros((8, 11), dtype=np.int8)
    a[0, 2] = a[1, 1] = a[1, 2] = 1
    a[3:6, 6] = 2
    expected = a.copy()
    for p in ((2, 3), (3, 4)):
        expected[p] = 1
    for p in ((4, 5), (5, 4), (6, 3), (7, 2)):
        expected[p] = 2
    _require(np.array_equal(_trace(a)[0], expected), "manual bar route failed")
    a[3, 6] = a[5, 6] = 0
    expected[3, 6] = expected[5, 6] = 0
    _require(np.array_equal(_trace(a)[0], expected), "manual point route failed")
    rows = ["000110000000", "001110000000", "011110000000", "011110000000",
            "000000000000", "111111111111"]
    empty = np.array([[c == "1" for c in row] for row in rows])
    _require(_best(empty) == (9, [(1, 2, 3, 3)]), "area/2-D comparison failed")
    empty = np.zeros((3, 7), dtype=bool)
    empty[:2, :2] = empty[:2, 5:7] = True
    _require(_best(empty)[0] == 4 and len(_best(empty)[1]) == 2, "ties missed")
    ambiguous = np.zeros((8, 8), dtype=np.int8)
    ambiguous[0, 1] = ambiguous[1, 0] = ambiguous[1, 1] = 1
    ambiguous[4, 3], ambiguous[3, 4] = 2, 3
    try:
        _trace(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous midpoint contact accepted")
    pair = generate_puzzle(73, 2, 1)["test"][0]
    for turns in range(4):
        for flip in (False, True):
            a, out = (np.rot90(pair[k], turns) for k in ("input", "output"))
            if flip:
                a, out = a[:, ::-1], out[:, ::-1]
            _require(np.array_equal(solve(a), out), "geometric equivariance failed")
    mapping = np.arange(10, dtype=np.int8)
    allowed = [c for c in range(10) if c != HIGHLIGHT]
    mapping[allowed] = allowed[::-1]
    _require(np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]]), "palette equivariance failed")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal count combinations)")
