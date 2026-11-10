"""L-anchored rectangular jigsaws followed by stable whole-column deduplication.

Only standard-library modules and NumPy are used. Searches, retries, and caches
are bounded; generation and self_test do not write files or execution traces.
"""
from functools import lru_cache
import hashlib
import numpy as np

CORNERS = ("TL", "TR", "BL", "BR")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any(g < 0) or np.any(g > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _column_keys(g):
    return [tuple(int(v) for v in col) for col in g.T]


def _compress(g):
    seen, keep = set(), []
    for x, vector in enumerate(_column_keys(g)):
        if vector not in seen:
            seen.add(vector)
            keep.append(x)
    return g[:, keep].copy()


def _marker_cells(point, corner):
    r, c = point
    dr = -1 if corner[0] == "T" else 1
    dc = -1 if corner[1] == "L" else 1
    return ((r + dr, c), (r, c + dc), (r + dr, c + dc))


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells = [p]
        for r, c in cells:
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
        result.append(cells)
    return result


def _extract(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("background must be the unique most frequent color")
    markers = []
    for color in range(10):
        q = np.argwhere(g == color)
        if color == bg or len(q) != 3:
            continue
        r, c = map(int, q.min(0))
        y, x = map(int, q.max(0))
        if (y - r, x - c) != (1, 1):
            continue
        missing = {(r, c), (r, x), (y, c), (y, x)} - set(map(tuple, q.tolist()))
        point = missing.pop()
        if int(g[point]) in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[
            (point[0] - r, point[1] - c)]
        markers.append((color, point, corner))
    if len(markers) != 1:
        raise ValueError("input needs one locally distinct external three-cell L")
    marker, point, corner = markers[0]
    pieces, anchor = [], None
    for cells in _components((g != bg) & (g != marker)):
        q = np.array(cells)
        r, c = map(int, q.min(0))
        y, x = map(int, q.max(0) + 1)
        if len(cells) != (y - r) * (x - c):
            raise ValueError("payload components must be filled rectangles")
        if point in cells:
            anchor = len(pieces)
        pieces.append((g[r:y, c:x].copy(), (r, c)))
    if anchor is None or not 3 <= len(pieces) <= 7:
        raise ValueError("expected 3..7 payload pieces including an anchor")
    m, (r, c) = pieces[anchor]
    expected = (r + (m.shape[0] - 1 if corner[0] == "B" else 0),
                c + (m.shape[1] - 1 if corner[1] == "R" else 0))
    if point != expected:
        raise ValueError("L does not point to the corresponding piece corner")
    return bg, marker, point, corner, pieces, anchor


def _assemblies(pieces, anchor, point, corner, canvas, match=True):
    """Complete rectangle exact cover, yielding pictures and translations.

    The first uncovered cell must be the top-left cell of the next rectangle.
    All four facing contacts are checked, including contacts with a preplaced
    bottom/right anchor. Non-anchor input coordinates never enter the search.
    Identical pieces are interchangeable, but distinct picture ties are not.
    """
    mats = [m for m, _ in pieces]
    area = sum(m.size for m in mats)
    nodes = 0
    for h in range(1, canvas[0] + 1):
        if area % h:
            continue
        w = area // h
        if w > canvas[1] or any(m.shape[0] > h or m.shape[1] > w for m in mats):
            continue
        oy = point[0] - (h - 1 if corner[0] == "B" else 0)
        ox = point[1] - (w - 1 if corner[1] == "R" else 0)
        if oy < 0 or ox < 0 or oy + h > canvas[0] or ox + w > canvas[1]:
            continue
        picture = np.full((h, w), -1, dtype=np.int8)
        positions = [None] * len(mats)
        am = mats[anchor]
        ah, aw = am.shape
        ar = h - ah if corner[0] == "B" else 0
        ac = w - aw if corner[1] == "R" else 0
        # A fixed piece, not merely a corner-colored pixel.
        if (oy + ar, ox + ac) != pieces[anchor][1]:
            continue
        picture[ar:ar + ah, ac:ac + aw] = am
        positions[anchor] = (oy + ar, ox + ac)

        def visit(remaining):
            nonlocal nodes
            nodes += 1
            if nodes > 100000:
                raise ValueError("exact-cover budget exceeded; uniqueness is unproved")
            if not remaining:
                yield picture.copy(), (oy, ox), tuple(positions)
                return
            free = np.flatnonzero(picture.ravel() < 0)
            r, c = divmod(int(free[0]), w)
            tried = set()
            for i in remaining:
                m = mats[i]
                ph, pw = m.shape
                key = _key(m)
                if key in tried:
                    continue
                tried.add(key)
                if r + ph > h or c + pw > w or np.any(picture[r:r + ph, c:c + pw] >= 0):
                    continue
                if match:
                    contacts = []
                    if r:
                        contacts.append((picture[r - 1, c:c + pw], m[0]))
                    if c:
                        contacts.append((picture[r:r + ph, c - 1], m[:, 0]))
                    if r + ph < h:
                        contacts.append((picture[r + ph, c:c + pw], m[-1]))
                    if c + pw < w:
                        contacts.append((picture[r:r + ph, c + pw], m[:, -1]))
                    if any(np.any((a >= 0) & (a != b)) for a, b in contacts):
                        continue
                picture[r:r + ph, c:c + pw] = m
                positions[i] = (oy + r, ox + c)
                yield from visit(tuple(j for j in remaining if j != i))
                picture[r:r + ph, c:c + pw] = -1
                positions[i] = None
        yield from visit(tuple(i for i in range(len(mats)) if i != anchor))


def _place_compressed(picture, point, corner, canvas, bg):
    compressed = _compress(picture)
    h, w = compressed.shape
    r = point[0] - (h - 1 if corner[0] == "B" else 0)
    c = point[1] - (w - 1 if corner[1] == "R" else 0)
    out = np.full(canvas, bg, dtype=np.int8)
    out[r:r + h, c:c + w] = compressed
    return out


@lru_cache(maxsize=64)
def _infer(shape, data):
    # This bounded memo is keyed only by visible input bytes, never by seed.
    g = np.frombuffer(data, dtype=np.int8).reshape(shape)
    bg, marker, point, corner, pieces, anchor = _extract(g)
    winner, winner_key, placements, count = None, None, None, 0
    for picture, origin, positions in _assemblies(pieces, anchor, point, corner, shape):
        count += 1
        key = (origin, _key(picture))
        if winner_key is not None and key != winner_key:
            raise ValueError("different full assemblies satisfy the visible clues")
        winner, winner_key, placements = picture, key, positions
    if winner is None:
        raise ValueError("no exact-edge rectangular assembly")
    return winner, placements, count


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the input: assemble uniquely first, then deduplicate columns."""
    g = _grid(input_grid)
    bg, _, point, corner, _, _ = _extract(g)
    picture, _, _ = _infer(*_key(g))
    return _place_compressed(picture, point, corner, g.shape, bg)


def _partition(rng, h, w, n, mode):
    rects = []
    if mode == "bands":
        bands = int(rng.integers(2, min(3, n - 1, w // 2) + 1))
        widths = [2] * bands
        for _ in range(w - 2 * bands):
            widths[int(rng.integers(bands))] += 1
        counts = [1] * bands
        for _ in range(n - bands):
            choices = [i for i in range(bands) if counts[i] < h // 2]
            if not choices:
                return None
            counts[int(rng.choice(choices))] += 1
        c = 0
        for pw, count in zip(widths, counts):
            heights = [2] * count
            for _ in range(h - 2 * count):
                heights[int(rng.integers(count))] += 1
            r = 0
            for ph in heights:
                rects.append((r, c, ph, pw))
                r += ph
            c += pw
    elif mode == "grid" and n % 2 == 0 and w >= n:
        widths = [2] * (n // 2)
        for _ in range(w - n):
            widths[int(rng.integers(n // 2))] += 1
        cut = int(rng.integers(2, h - 1))
        for r, ph in ((0, cut), (cut, h - cut)):
            c = 0
            for pw in widths:
                rects.append((r, c, ph, pw))
                c += pw
    else:
        rects = [(0, 0, h, w)]
        for _ in range(n - 1):
            options = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects)
                       for axis, size in enumerate((ph, pw)) if size >= 4]
            if not options:
                return None
            i, axis = options[int(rng.integers(len(options)))]
            r, c, ph, pw = rects.pop(i)
            k = int(rng.integers(2, (ph if axis == 0 else pw) - 1))
            rects.extend([(r, c, k, pw), (r + k, c, ph - k, pw)] if axis == 0
                         else [(r, c, ph, k), (r, c + k, ph, pw - k)])
    labels = np.empty((h, w), dtype=int)
    for i, (r, c, ph, pw) in enumerate(rects):
        labels[r:r + ph, c:c + pw] = i
    # Transposition changes partitions and source geometry; solving is always
    # left-to-right column filtering, irrespective of picture aspect ratio.
    if rng.random() < .25 and max(h, w) <= 14:
        labels = labels.T.copy()
    return labels


def _picture(rng, labels, palette):
    h, w = labels.shape
    n = int(rng.integers(3, min(8, w - 1) + 1))
    ids = list(range(n))
    ids.insert(int(rng.integers(1, n)), 0)
    ids.extend(int(v) for v in rng.integers(n, size=w - len(ids)))
    parent = list(range(h * n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[root(b)] = root(a)

    for r in range(h):
        for c in range(w):
            for y, x in ((r + 1, c), (r, c + 1)):
                if y < h and x < w and labels[r, c] != labels[y, x]:
                    union(r * n + ids[c], y * n + ids[x])
    # Some legitimate repeated rows distinguish column-only from two-axis dedupe.
    if rng.random() < .35:
        r = int(rng.integers(h - 1))
        for i in range(n):
            union(r * n + i, (r + 1) * n + i)
    values = {root(i): int(rng.choice(palette)) for i in range(h * n)}
    return np.array([[values[root(r * n + ids[c])] for c in range(w)]
                     for r in range(h)], dtype=np.int8)


def _dedupe_witnesses(pic):
    columns = _column_keys(pic)
    unique = list(dict.fromkeys(columns))
    separated = any(columns[i] != columns[i - 1] and columns[i] in columns[:i - 1]
                    for i in range(2, len(columns)))
    crop_fails = columns[:len(unique)] != unique
    last_order = list(dict.fromkeys(reversed(columns)))[::-1]
    return (3 <= len(unique) < len(columns) and separated and crop_fails
            and last_order != unique and unique != sorted(unique))


def _scatter(rng, pieces, anchor, final, markers, canvas):
    order = sorted((i for i in range(len(pieces)) if i != anchor),
                   key=lambda i: -pieces[i][0].size)
    for _ in range(10):
        blocked = np.zeros(canvas, dtype=bool)
        positions = [None] * len(pieces)
        positions[anchor] = final[anchor]

        def block(r, c, h, w):
            blocked[max(0, r - 1):min(canvas[0], r + h + 1),
                    max(0, c - 1):min(canvas[1], c + w + 1)] = True

        r, c = final[anchor]
        block(r, c, *pieces[anchor][0].shape)
        for r, c in markers:
            block(r, c, 1, 1)
        for i in order:
            ph, pw = pieces[i][0].shape
            choices = [(r, c) for r in range(canvas[0] - ph + 1)
                       for c in range(canvas[1] - pw + 1)]
            for k in rng.permutation(len(choices)):
                r, c = choices[int(k)]
                if not blocked[r:r + ph, c:c + pw].any():
                    positions[i] = (r, c)
                    block(r, c, ph, pw)
                    break
            else:
                break
        else:
            return positions
    return None


def _edge_necessary(g, expected):
    bg, _, point, corner, pieces, anchor = _extract(g)
    for pic, _, _ in _assemblies(pieces, anchor, point, corner, g.shape, match=False):
        alternative = _place_compressed(pic, point, corner, g.shape, bg)
        if not np.array_equal(alternative, expected):
            return True
    return False


def _partial_alias(picture, pieces, positions, point, corner):
    h, w = picture.shape
    ox = point[1] - (w - 1 if corner[1] == "R" else 0)
    # A partial column can look identical while completed columns differ.
    for (m, _), (_, c) in zip(pieces, positions):
        if m.shape[0] == h:
            continue
        c -= ox
        for a in range(m.shape[1]):
            for b in range(a):
                if np.array_equal(m[:, a], m[:, b]) and not np.array_equal(
                        picture[:, c + a], picture[:, c + b]):
                    return True
    return False


def _make_pair(rng, wide, corner):
    for _ in range(256):
        h = int(rng.integers(6, 13 if wide else 11))
        w = int(rng.integers(15, 21) if wide else rng.integers(8, 15))
        n = int(rng.integers(5, 8) if wide else rng.integers(3, 7))
        labels = _partition(rng, h, w, n, str(rng.choice(("recursive", "bands", "grid"))))
        if labels is None:
            continue
        h, w = labels.shape
        colors = rng.permutation(10)
        bg, marker = map(int, colors[:2])
        palette = colors[2:2 + int(rng.integers(2, 7))]
        pic = _picture(rng, labels, palette)
        if not _dedupe_witnesses(pic):
            continue
        pieces = []
        for i in range(n):
            q = np.argwhere(labels == i)
            r, c = map(int, q.min(0))
            y, x = map(int, q.max(0) + 1)
            pieces.append((pic[r:y, c:x].copy(), (r, c)))
        local = (h - 1 if corner[0] == "B" else 0, w - 1 if corner[1] == "R" else 0)
        anchor = int(labels[local])
        area = h * w
        base_h = max(h + 2, int(np.ceil(np.sqrt(area * 2.7))))
        base_w = max(w + 2, int(np.ceil(area * 2.7 / base_h)))
        canvas = (min(30, base_h + int(rng.integers(0, 5))),
                  min(30, base_w + int(rng.integers(0, 5))))
        oy = int(rng.integers(1, canvas[0] - h))
        ox = int(rng.integers(1, canvas[1] - w))
        point = (oy + local[0], ox + local[1])
        markers = _marker_cells(point, corner)
        final = [(oy + r, ox + c) for _, (r, c) in pieces]
        positions = _scatter(rng, pieces, anchor, final, markers, canvas)
        if positions is None:
            continue
        g = np.full(canvas, bg, dtype=np.int8)
        for (m, _), (r, c) in zip(pieces, positions):
            g[r:r + m.shape[0], c:c + m.shape[1]] = m
        for p in markers:
            g[p] = marker
        try:
            _, _, _, _, visible, visible_anchor = _extract(g)
            inferred, recovered, _ = _infer(*_key(g))
            if not np.array_equal(inferred, pic):
                continue
            out = _place_compressed(pic, point, corner, canvas, bg)
            if not _edge_necessary(g, out):
                continue
            if not _partial_alias(pic, visible, recovered, point, corner):
                continue
        except ValueError:
            continue
        return {"input": g, "output": out}
    raise RuntimeError("bounded candidate search exhausted; no ambiguous label was emitted")


def _count(value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError("illegal example count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # Two distinct corners already establish the geometric role at minimum counts.
    corners = [str(c) for c in rng.choice(CORNERS, 2, replace=False)]
    train_corners = corners + [str(rng.choice(CORNERS)) for _ in range(nt - 2)]
    rng.shuffle(train_corners)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plan in (("train", train_corners),
                        ("test", [str(rng.choice(CORNERS)) for _ in range(ne)])):
        for corner in plan:
            for _ in range(8):
                pair = _make_pair(rng, split == "test", corner)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(g.shape == out.shape, "canvas changed")
            key = _key(g)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            bg, marker, point, corner, pieces, anchor = _extract(g)
            pic, positions, _ = _infer(*key)
            _require(_dedupe_witnesses(pic), "missing separated/early/first-order repeat evidence")
            _require(_partial_alias(pic, pieces, positions, point, corner), "no partial/full distinction")
            _require(np.array_equal(solve(g), out), "input-only solution mismatch")
            _require(_edge_necessary(g, out), "shape alone suffices; edge evidence is unnecessary")
            _require(positions[anchor] == pieces[anchor][1], "assembly anchor moved")
            _require(not np.any(out == marker), "marker survived")
            _require(all(out[p] == bg for p in _marker_cells(point, corner)), "marker not erased")
            _require(pic.size == sum(m.size for m, _ in pieces), "payload area changed before dedupe")
            occupied = np.zeros(pic.shape, dtype=bool)
            oy = point[0] - (pic.shape[0] - 1 if corner[0] == "B" else 0)
            ox = point[1] - (pic.shape[1] - 1 if corner[1] == "R" else 0)
            for (m, _), (r, c) in zip(pieces, positions):
                r, c = r - oy, c - ox
                ph, pw = m.shape
                _require(not occupied[r:r + ph, c:c + pw].any(), "overlap")
                occupied[r:r + ph, c:c + pw] = True
                _require(np.array_equal(pic[r:r + ph, c:c + pw], m), "piece pixels changed")
            _require(occupied.all(), "assembly has gaps")
            compressed = _compress(pic)
            _require(len(set(_column_keys(compressed))) == compressed.shape[1], "duplicate survives")
            _require(compressed.shape[1] < pic.shape[1], "A omitted")
            infos[split].append((pic.shape, corner))
    _require(len({corner for _, corner in infos["train"]}) >= 2, "train needs two marker directions")
    _require(min(shape[1] for shape, _ in infos["test"]) >
             max(shape[1] for shape, _ in infos["train"]), "missing assembled-width shift")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain only small feature sets, never puzzle corpora."""
    fingerprints, canvas_shapes, picture_shapes, counts, palettes, corners, partitions = [set() for _ in range(7)]
    repeated_rows = near_matches = partial_contacts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        if seed == 0:
            _require(_fingerprint(generate_puzzle(seed)) == _fingerprint(puzzle), "nondeterministic")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g = pair["input"]
                _, _, point, corner, pieces, _ = _extract(g)
                pic, positions, _ = _infer(*_key(g))
                canvas_shapes.add(g.shape)
                picture_shapes.add(pic.shape)
                counts.add(len(pieces))
                palettes.add(tuple(int(c) for c in np.unique(pic)))
                corners.add(corner)
                oy = point[0] - (pic.shape[0] - 1 if corner[0] == "B" else 0)
                ox = point[1] - (pic.shape[1] - 1 if corner[1] == "R" else 0)
                partitions.add((pic.shape, tuple(sorted((r-oy, c-ox, *m.shape)
                                                       for (m, _), (r, c) in zip(pieces, positions)))))
                u = _compress(pic)
                if corner[1] == "R":
                    inward = np.array(list(dict.fromkeys(reversed(_column_keys(pic))))[::-1],
                                      dtype=np.int8).T
                    _require(not np.array_equal(u, inward), "right-corner scan witness")
                repeated_rows += len(set(map(tuple, u.tolist()))) < u.shape[0]
                near_matches += any(np.count_nonzero(u[:, i] != u[:, j]) == 1
                                    for i in range(u.shape[1]) for j in range(i))
                partial_contacts += any(m.shape[0] < pic.shape[0] and m.shape[1] < pic.shape[1]
                                        for m, _ in pieces)
    _require(len(fingerprints) == 42, "seed diversity")
    _require(len(canvas_shapes) >= 30 and len(picture_shapes) >= 30, "geometry diversity")
    _require(counts == set(range(3, 8)) and corners == set(CORNERS), "count/corner diversity")
    _require(len(palettes) >= 35 and len(partitions) >= 100, "palette/partition diversity")
    _require(repeated_rows >= 10 and near_matches >= 5 and partial_contacts >= 100, "structural witnesses")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(4200 + nt * 10 + ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A uniform four-square jigsaw has distinct valid full assemblies and must fail.
    pieces = [(np.full((2, 2), 2, dtype=np.int8), (2, 2)) for _ in range(4)]
    alternatives = set()
    for pic, origin, _ in _assemblies(pieces, 0, (2, 2), "TL", (15, 15)):
        alternatives.add((origin, _key(pic)))
    _require(len(alternatives) > 1, "ambiguity witness disappeared")
    ambiguous = np.zeros((15, 15), dtype=np.int8)
    for r, c in ((2, 2), (2, 7), (7, 2), (7, 7)):
        ambiguous[r:r + 2, c:c + 2] = 2
    for point in _marker_cells((2, 2), "TL"):
        ambiguous[point] = 1
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("input-only solver accepted ambiguous assembly")
    p = generate_puzzle(199, 2, 1)
    pair = p["test"][0]
    bg, marker, point, corner, pieces, anchor = _extract(pair["input"])
    relocated = _scatter(np.random.default_rng(2200), pieces, anchor,
                         [pos for _, pos in pieces], _marker_cells(point, corner),
                         pair["input"].shape)
    _require(relocated is not None, "relocation fixture failed")
    g = np.full(pair["input"].shape, bg, dtype=np.int8)
    for (m, _), (r, c) in zip(pieces, relocated):
        g[r:r + m.shape[0], c:c + m.shape[1]] = m
    for q in _marker_cells(point, corner):
        g[q] = marker
    _require(not np.array_equal(g, pair["input"]), "fixture did not relocate pieces")
    _require(np.array_equal(solve(g), pair["output"]), "scattered order affected solution")
    for damage in ("color", "marker", "duplicate", "shift"):
        broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = broken["test"][0]
        bg, marker, point, corner, _, _ = _extract(pair["input"])
        if damage == "color":
            r, c = np.argwhere(pair["output"] != bg)[0]
            pair["output"][r, c] = marker
        elif damage == "marker":
            pair["output"][_marker_cells(point, corner)[0]] = marker
        elif damage == "duplicate":
            broken["train"][1] = {k: v.copy() for k, v in broken["train"][0].items()}
        else:
            pair["output"] = np.roll(pair["output"], 1, axis=0)
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupt puzzle accepted: " + damage)
    return {"seeds": 42, "distinct_puzzles": len(fingerprints),
            "canvas_shapes": len(canvas_shapes), "assembled_shapes": len(picture_shapes),
            "piece_counts": sorted(counts), "marker_corners": sorted(corners),
            "payload_palettes": len(palettes), "partitions": len(partitions),
            "repeated_row_pairs": int(repeated_rows), "near_match_pairs": int(near_matches)}
