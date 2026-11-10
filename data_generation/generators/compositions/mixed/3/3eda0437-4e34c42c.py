"""Register exact overlapping views, crop their union, highlight its largest void.

Only NumPy and the standard library are needed. Sampling metadata never reaches
the solver. All JSON written by the optional command-line self-test is compact.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


HIGHLIGHT = 6


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _groups(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _tight(a, background):
    rr, cc = np.nonzero(a != background)
    _require(len(rr) > 0, "empty fragment")
    return a[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _parts(a, background):
    parts = []
    for group in _groups(zip(*np.nonzero(a != background))):
        top = min(r for r, c in group)
        left = min(c for r, c in group)
        h = max(r for r, c in group) - top + 1
        w = max(c for r, c in group) - left + 1
        part = np.full((h, w), background, dtype=np.int8)
        for r, c in group:
            part[r - top, c - left] = a[r, c]
        parts.append(part)
    return parts


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    _require(int(counts.max()) * 2 > a.size, "background must occupy more than half the input")
    background = int(colors[counts.argmax()])
    _require(background != HIGHLIGHT and HIGHLIGHT not in colors, "highlight is output-only")
    return background


def _matches(a, b, background):
    """Context-exact, same-orientation multicolor translations.

    The intersecting observed rectangles must agree, including empty cells.
    Three or more foreground cells of at least two colors support a match.
    Remove only matches whose footprints on BOTH pieces are strict/subsets of
    another match. This is partial-feature suppression, not overlap scoring.
    Incomparable supports survive and must be resolved globally or rejected.
    """
    ah, aw = a.shape
    bh, bw = b.shape
    candidates = []
    for dr in range(1 - bh, ah):
        r0, r1 = max(0, dr), min(ah, dr + bh)
        for dc in range(1 - bw, aw):
            c0, c1 = max(0, dc), min(aw, dc + bw)
            if (r1 - r0) * (c1 - c0) < 3:
                continue
            aa = a[r0:r1, c0:c1]
            bb = b[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            if not np.array_equal(aa, bb):
                continue
            rr, cc = np.nonzero(aa != background)
            if len(rr) < 3 or len(np.unique(aa[rr, cc])) < 2:
                continue
            fa = frozenset((int(r + r0), int(c + c0)) for r, c in zip(rr, cc))
            fb = frozenset((r - dr, c - dc) for r, c in fa)
            candidates.append(((dr, dc), fa, fb))
    return [x for x in candidates if not any(
        x[1] <= y[1] and x[2] <= y[2] and (x[1] != y[1] or x[2] != y[2])
        for y in candidates)]


def _compatible(a, apos, b, bpos):
    """Check agreement of all jointly observed cells, even without a key."""
    ar, ac = apos
    br, bc = bpos
    r0, r1 = max(ar, br), min(ar + a.shape[0], br + b.shape[0])
    c0, c1 = max(ac, bc), min(ac + a.shape[1], bc + b.shape[1])
    if r0 >= r1 or c0 >= c1:
        return True
    return np.array_equal(a[r0 - ar:r1 - ar, c0 - ac:c1 - ac],
                          b[r0 - br:r1 - br, c0 - bc:c1 - bc])


def _register(parts, background):
    """Recover a unique connected translation system, up to global translation.

    Every surviving repeated support constrains the system. There is no
    minimum-area or maximum-coincidence objective and no target-rectangle test
    in registration. Ambiguous placements fail before the void search starts.
    """
    n = len(parts)
    _require(2 <= n <= 8, "need two to eight separated fragments")
    adjacency = [[] for _ in parts]
    for i in range(n):
        for j in range(i + 1, n):
            matches = _matches(parts[i], parts[j], background)
            if matches:
                deltas = {x[0] for x in matches}
                adjacency[i].append((j, deltas))
                adjacency[j].append((i, {(-r, -c) for r, c in deltas}))
    offsets, solutions = {0: (0, 0)}, []
    visits = 0

    def search():
        nonlocal visits
        visits += 1
        _require(visits <= 4000, "registration search bound exceeded")
        if len(solutions) >= 2:
            return
        if len(offsets) == n:
            solutions.append(dict(offsets))
            return
        options = []
        for j in range(n):
            if j in offsets:
                continue
            domain = None
            for i, deltas in adjacency[j]:
                if i in offsets:
                    proposed = {(offsets[i][0] - r, offsets[i][1] - c) for r, c in deltas}
                    domain = proposed if domain is None else domain & proposed
            if domain is not None:
                if not domain:
                    return
                options.append((len(domain), j, domain))
        if not options:
            return
        _, j, domain = min(options, key=lambda item: (item[0], item[1]))
        for pos in sorted(domain):
            if not all(_compatible(parts[j], pos, parts[i], p) for i, p in offsets.items()):
                continue
            offsets[j] = pos
            rr = [p[0] for p in offsets.values()]
            cc = [p[1] for p in offsets.values()]
            h = max(p[0] + parts[i].shape[0] for i, p in offsets.items()) - min(rr)
            w = max(p[1] + parts[i].shape[1] for i, p in offsets.items()) - min(cc)
            if h <= 30 and w <= 30:
                search()
            del offsets[j]

    search()
    _require(len(solutions) == 1, "disconnected, inconsistent, or ambiguous registration")
    offsets = solutions[0]
    top = min(r for r, c in offsets.values())
    left = min(c for r, c in offsets.values())
    offsets = {i: (r - top, c - left) for i, (r, c) in offsets.items()}
    h = max(r + parts[i].shape[0] for i, (r, c) in offsets.items())
    w = max(c + parts[i].shape[1] for i, (r, c) in offsets.items())
    composite = np.full((h, w), background, dtype=np.int8)
    for i, (r, c) in offsets.items():
        part = parts[i]
        view = composite[r:r + part.shape[0], c:c + part.shape[1]]
        view[part != background] = part[part != background]
    # Each part is already tight, so the union's bounding box is tight too.
    _require(np.array_equal(_tight(composite, background), composite), "loose crop")
    return composite, offsets


def _rectangles(a, background):
    """All maximal-width empty runs for each contiguous row span.

    A global area winner is necessarily maximal in width, so this enumerates
    every possible winner without retaining every contained subrectangle.
    """
    h, w = a.shape
    result = []
    for top in range(h - 1):
        common = np.ones(w, dtype=bool)
        for bottom in range(top, h):
            common &= a[bottom] == background
            if bottom == top:
                continue
            edges = np.flatnonzero(np.diff(np.r_[False, common, False]))
            for left, right in zip(edges[::2], edges[1::2]):
                if right - left >= 2:
                    rh, rw = bottom - top + 1, int(right - left)
                    result.append((rh * rw, top, int(left), rh, rw))
    return result


def _winner(a, background):
    candidates = _rectangles(a, background)
    _require(bool(candidates), "no eligible background rectangle")
    area = max(x[0] for x in candidates)
    best = [x[1:] for x in candidates if x[0] == area]
    _require(len(best) == 1, "largest rectangle is tied")
    return best[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    background = _background(a)
    composite, _ = _register(_parts(a, background), background)
    r, c, h, w = _winner(composite, background)
    output = composite.copy()
    output[r:r + h, c:c + w] = HIGHLIGHT
    return output


def _features(composite, background, rect):
    r, c, h, w = rect
    area = h * w
    target = {(y, x) for y in range(r, r + h) for x in range(c, c + w)}
    component = next(g for g in _groups(zip(*np.nonzero(composite == background))) if (r, c) in g)
    rivals = [x for x in _rectangles(composite, background)
              if not (r <= x[1] and c <= x[2] and x[1] + x[3] <= r + h and x[2] + x[4] <= c + w)]
    unchanged = composite == background
    unchanged[r:r + h, c:c + w] = False
    longest = 0
    for direction in (unchanged, unchanged.T):
        for row in direction:
            edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
            longest = max(longest, max((int(b - a) for a, b in zip(edges[::2], edges[1::2])), default=0))
    return {"area": area, "target_shape": (h, w), "attached": len(component) > area,
            "boundary": r == 0 or c == 0 or r + h == composite.shape[0] or c + w == composite.shape[1],
            "rival_area": max((x[0] for x in rivals), default=0),
            "overlapping_rival": any(bool(target & {(y, z) for y in range(x[1], x[1] + x[3])
                                                    for z in range(x[2], x[2] + x[4])}) for x in rivals),
            "long_line": longest > area}


def _make_composite(rng, background, colors, profile, test):
    if profile == "competitive":
        rh, rw = ((3, 3), (2, 5), (3, 4), (4, 3))[int(rng.integers(4))]
        h, w = int(rng.integers(10, 15)), int(rng.integers(max(13, rh * rw + 3), 19))
    else:
        rh = int(rng.integers(2, 7 if test else 6))
        rw = int(rng.integers(3, 8 if test else 7))
        h = int(rng.integers(max(7, rh + 3), 17 if test else 15))
        w = int(rng.integers(max(9, rw + 3), 20 if test else 19))
    r = int(rng.integers(1, h - rh))
    c = int(rng.integers(1, w - rw))
    if profile == "free" and rng.random() < .40:
        if rng.random() < .5:
            r = 0 if rng.random() < .5 else h - rh
        else:
            c = 0 if rng.random() < .5 else w - rw
    mask = np.zeros((h, w), dtype=bool)
    mask[r:r + rh, c:c + rw] = True

    # Partial strips make the selected rectangle only part of an irregular void.
    if profile in ("competitive", "attached") or rng.random() < .55:
        sides = [side for side, valid in enumerate((r > 0, r + rh < h, c > 0, c + rw < w)) if valid]
        side = int(rng.choice(sides))
        if side < 2 and rw >= 3:
            y = r - 1 if side == 0 else r + rh
            length = int(rng.integers(1, rw))
            x = c + int(rng.integers(rw - length + 1))
            mask[y, x:x + length] = True
        elif rh >= 3:
            x = c - 1 if side == 2 else c + rw
            length = int(rng.integers(1, rh))
            y = r + int(rng.integers(rh - length + 1))
            mask[y:y + length, x] = True
    # Place a substantial separate rival when requested; otherwise optional.
    if profile == "competitive" or rng.random() < .70:
        shapes = [(a, b) for a in range(2, min(7, h - 2))
                  for b in range(2, min(9, w - 2)) if a * b < rh * rw]
        rng.shuffle(shapes)
        shapes.sort(key=lambda s: -s[0] * s[1])
        for a, b in shapes[:10]:
            positions = [(y, x) for y in range(1, h - a) for x in range(1, w - b)
                         if not mask[max(0, y - 1):y + a + 1, max(0, x - 1):x + b + 1].any()]
            if positions:
                y, x = positions[int(rng.integers(len(positions)))]
                mask[y:y + a, x:x + b] = True
                break
    if profile == "competitive" or rng.random() < .30:
        length = min(w - 2, rh * rw + int(rng.integers(1, 4)))
        positions = [(y, x) for y in range(1, h - 1) for x in range(1, w - length)
                     if not mask[y - 1:y + 2, x - 1:x + length + 1].any()]
        if positions:
            y, x = positions[int(rng.integers(len(positions)))]
            mask[y, x:x + length] = True
    # Edge notches, pinholes, and asymmetric material silhouettes are optional.
    noise = rng.random((h, w)) < float(rng.uniform(.015, .10))
    noise[max(0, r - 1):r + rh + 1, max(0, c - 1):c + rw + 1] = False
    mask |= noise
    for _ in range(int(rng.integers(0, 4))):
        side = int(rng.integers(4))
        if side < 2:
            y = 0 if side == 0 else h - 1
            x = int(rng.integers(1, w - 1))
            mask[y, x:min(w - 1, x + int(rng.integers(1, 4)))] = True
        else:
            x = 0 if side == 2 else w - 1
            y = int(rng.integers(1, h - 1))
            mask[y:min(h - 1, y + int(rng.integers(1, 4))), x] = True
    a = rng.choice(colors, size=(h, w)).astype(np.int8)
    # Patches and runs soften the texture, while literal multicolor keys remain
    # independently sampled in each example (there is no shared motif bank).
    for _ in range(int(rng.integers(2, 6))):
        y, x = int(rng.integers(h)), int(rng.integers(w))
        a[y:min(h, y + int(rng.integers(1, 4))), x:min(w, x + int(rng.integers(1, 5)))] = int(rng.choice(colors))
    a[mask] = background
    _require(len(_groups(zip(*np.nonzero(~mask)))) == 1, "scaffold disconnected")
    _require(_winner(a, background) == (r, c, rh, rw), "planting changed the maximum")
    features = _features(a, background, (r, c, rh, rw))
    if profile == "competitive":
        _require(features["rival_area"] >= 8 and features["long_line"], "missing comparison witnesses")
    if profile == "attached":
        _require(features["attached"] and features["overlapping_rival"], "missing irregular-void witness")
    return a, (r, c, rh, rw)


def _cut_views(a, rect, n, background, rng):
    r, c, rh, rw = rect
    # Cutting either axis is legal. Choosing the axis before cutting also lets
    # a boundary notch remain a connected observation on each side.
    if rh >= 3 and rng.random() < .5:
        a = a.T.copy()
        r, c, rh, rw = c, r, rw, rh
    overlap = int(rng.integers(1, min(2, rw - 2) + 1))
    cut = int(rng.integers(c + 1, c + rw - overlap))
    views = [a[:, :cut + overlap].copy(), a[:, cut:].copy()]
    if n >= 3:
        side = int(rng.integers(2))
        part = views[side]
        y = int(rng.integers(max(1, r - 1), min(a.shape[0] - 2, r + rh) + 1))
        ov = int(rng.integers(1, min(3, part.shape[0] - y)))
        views[side:side + 1] = [part[:y + ov].copy(), part[y:].copy()]
    if n == 4:
        # Split the remaining full-height view at an independently chosen row:
        # staggered cuts need not form a regular four-tile lattice.
        index = next(i for i, view in enumerate(views) if view.shape[0] == a.shape[0])
        part = views[index]
        y = int(rng.integers(max(1, r - 1), min(a.shape[0] - 2, r + rh) + 1))
        ov = int(rng.integers(1, min(3, part.shape[0] - y)))
        views[index:index + 1] = [part[:y + ov].copy(), part[y:].copy()]
    views = [_tight(v, background) for v in views]
    _require(all(len(_groups(zip(*np.nonzero(v != background)))) == 1 for v in views), "cut disconnected a view")
    # Transform the entire observation system together; individual fragments
    # are always preserved in their observed orientation at solve time.
    transpose, flip_r, flip_c = [bool(rng.integers(2)) for _ in range(3)]

    def transform(v):
        if transpose:
            v = v.T
        if flip_r:
            v = v[::-1]
        if flip_c:
            v = v[:, ::-1]
        return v.copy()

    return transform(a), [transform(v) for v in views]


def _pack(views, background, rng):
    """Random shelf partitions, order, gaps, and margins; no slot dictionary."""
    n = len(views)
    foreground = sum(int(np.count_nonzero(v != background)) for v in views)
    layouts = []
    gap = int(rng.integers(1, 4))
    for order in itertools.permutations(range(n)):
        for breaks in range(1 << (n - 1)):
            rows, row = [], []
            for k, i in enumerate(order):
                row.append(i)
                if k == n - 1 or (breaks >> k) & 1:
                    rows.append(row)
                    row = []
            heights = [max(views[i].shape[0] for i in row) for row in rows]
            widths = [sum(views[i].shape[1] for i in row) + gap * (len(row) - 1) for row in rows]
            h, w = sum(heights) + gap * (len(rows) - 1), max(widths)
            if h <= 30 and w <= 30:
                layouts.append((rows, heights, widths, h, w))
    _require(bool(layouts) and foreground * 2 < 900, "views cannot fit")
    rows, heights, widths, min_h, min_w = layouts[int(rng.integers(len(layouts)))]
    sizes = [(h, w) for h in range(min_h, 31) for w in range(min_w, 31) if h * w > foreground * 2]
    _require(bool(sizes), "cannot expose a dominant background")
    h, w = sizes[int(rng.integers(len(sizes)))]
    grid = np.full((h, w), background, dtype=np.int8)
    y = int(rng.integers(h - min_h + 1))
    for row, row_h, row_w in zip(rows, heights, widths):
        x = int(rng.integers(w - row_w + 1))
        for i in row:
            part = views[i]
            yy = y + int(rng.integers(row_h - part.shape[0] + 1))
            grid[yy:yy + part.shape[0], x:x + part.shape[1]] = part
            x += part.shape[1] + gap
        y += row_h + gap
    return grid


def _key(a):
    return tuple(a.shape), np.asarray(a, dtype=np.int8).tobytes()


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    _require(_integer(num_train) and _integer(num_test), "example counts must be integers")
    _require(2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test, "illegal example counts")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    background = int(rng.choice([c for c in range(10) if c != HIGHLIGHT]))
    palette = [int(x) for x in rng.permutation([c for c in range(10) if c not in (background, HIGHLIGHT)])]
    palette = palette[:int(rng.integers(3, 7))]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            test = split == "test"
            n = (4 if index == 0 else int(rng.integers(3, 5))) if test else (2 if index == 0 else 3 if index == 1 else int(rng.integers(2, 4)))
            profile = "competitive" if not test and index == 0 else "attached" if not test and index == 1 else "free"
            colors = palette if index == 0 and not test else list(rng.choice(palette, int(rng.integers(2, len(palette) + 1)), replace=False))
            for _ in range(160):
                try:
                    composite, rect = _make_composite(rng, background, colors, profile, test)
                    composite, views = _cut_views(composite, rect, n, background, rng)
                    inp = _pack(views, background, rng)
                    reconstructed, offsets = _register(_parts(inp, background), background)
                    _require(np.array_equal(reconstructed, composite), "registration differs from the scaffold")
                    rect = _winner(reconstructed, background)
                    rr, cc, rh, rw = rect
                    # Every fragment misses at least one row/column of the
                    # winning window; completing the answer needs registration.
                    parts = _parts(inp, background)
                    _require(not any(r <= rr and c <= cc and rr + rh <= r + parts[i].shape[0]
                                     and cc + rw <= c + parts[i].shape[1]
                                     for i, (r, c) in offsets.items()), "target wholly visible in one view")
                    out = reconstructed.copy()
                    out[rr:rr + rh, cc:cc + rw] = HIGHLIGHT
                    key = _key(inp)
                    _require(key not in seen, "duplicate input")
                    seen.add(key)
                    puzzle[split].append({"input": inp, "output": out})
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(f"bounded construction failed for {split}/{index}")
    return puzzle


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    inp, output = _grid(pair["input"]), _grid(pair["output"])
    background = _background(inp)
    parts = _parts(inp, background)
    composite, offsets = _register(parts, background)
    rect = _winner(composite, background)
    r, c, h, w = rect
    expected = composite.copy()
    expected[r:r + h, c:c + w] = HIGHLIGHT
    _require(np.array_equal(output, expected), "output is not registered union plus maximum rectangle")
    _require(set(np.unique(composite)) - {background} == set(np.unique(inp)) - {background}, "material lost")
    _require(not any(y <= r and x <= c and r + h <= y + parts[i].shape[0]
                     and c + w <= x + parts[i].shape[1]
                     for i, (y, x) in offsets.items()), "registration not needed to observe the whole window")
    info = _features(composite, background, rect)
    info.update({"background": background, "fragments": len(parts), "shape": tuple(composite.shape)})
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    _require(len({i["background"] for s in infos.values() for i in s}) == 1, "puzzle background changed")
    _require(any(i["rival_area"] >= 8 and i["long_line"] for i in infos["train"]), "training lacks area/1D comparisons")
    _require(any(i["attached"] and i["overlapping_rival"] for i in infos["train"]), "training lacks rectangle-versus-component evidence")
    _require(any(i["fragments"] > max(j["fragments"] for j in infos["train"]) for i in infos["test"]), "missing assembly-depth shift")


def self_test():
    """Stream 42 seeds; no grids, logs, or traces are saved by this function."""
    shapes, inputs, targets, palettes, backgrounds, signatures = set(), set(), set(), set(), set(), set()
    counts, attached, rivals, lines, boundary, pairs = set(), 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair)
                original = pair["input"].copy()
                _require(np.array_equal(solve(original), pair["output"]), "solve mismatch")
                _require(np.array_equal(original, pair["input"]), "solve mutated its input")
                shapes.add(info["shape"])
                inputs.add(tuple(pair["input"].shape))
                targets.add(info["target_shape"])
                palettes.add(len(np.unique(pair["input"])))
                backgrounds.add(info["background"])
                counts.add(info["fragments"])
                attached += int(info["attached"])
                rivals += int(info["rival_area"] > 0)
                lines += int(info["long_line"])
                boundary += int(info["boundary"])
                pairs += 1
                for role in ("input", "output"):
                    digest.update(repr(pair[role].shape).encode())
                    digest.update(pair[role].tobytes())
        signatures.add(digest.hexdigest())
        if seed == 0:
            again = generate_puzzle(seed)
            _require(all(np.array_equal(p[role], q[role]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                         for role in ("input", "output")), "nondeterminism")
            for pair in puzzle["train"]:
                _require(np.array_equal(solve(pair["input"].T), pair["output"].T), "transpose changed the rule")
                _require(np.array_equal(solve(pair["input"][::-1, ::-1]), pair["output"][::-1, ::-1]), "whole-image reflection changed the rule")
                bg = _background(pair["input"])
                repacked = _pack(_parts(pair["input"], bg), bg, np.random.default_rng(123))
                _require(np.array_equal(solve(repacked), pair["output"]), "input layout encodes adjacency")
    _require(len(signatures) == 42 and len(inputs) >= 30 and len(shapes) >= 25, "insufficient structural variation")
    _require(len(targets) >= 12 and len(palettes) >= 4 and len(backgrounds) >= 7 and counts == {2, 3, 4}, "insufficient family variation")
    _require(attached >= 42 and rivals >= 42 and lines >= 42, "missing shortcut witnesses")
    _require(boundary >= 10, "missing boundary winners")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Independent exhaustive small-mask checks for the area selector.
    rng = np.random.default_rng(71)
    for _ in range(24):
        a = rng.integers(0, 2, size=(5, 6))
        candidates = [(h * w, r, c, h, w) for r in range(4) for c in range(5)
                      for h in range(2, 6 - r) for w in range(2, 7 - c)
                      if np.all(a[r:r + h, c:c + w] == 0)]
        fast = _rectangles(a, 0)
        _require(max((x[0] for x in fast), default=0) == max((x[0] for x in candidates), default=0), "area enumeration failed")
    periodic = [np.array([[1, 2, 1, 2, 1, 2]], dtype=np.int8), np.array([[1, 2, 1, 2]], dtype=np.int8)]
    try:
        _register(periodic, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous repeated texture accepted")
    try:
        _winner(np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=np.int8), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("tied rectangles accepted")
    return {"seeds": 42, "pairs": pairs, "input_shapes": len(inputs), "output_shapes": len(shapes),
            "target_shapes": len(targets), "palette_sizes": sorted(palettes), "backgrounds": sorted(backgrounds),
            "fragment_counts": sorted(counts), "attached_voids": attached, "with_rivals": rivals,
            "long_lines": lines, "boundary_winners": boundary}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
