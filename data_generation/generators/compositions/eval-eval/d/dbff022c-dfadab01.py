"""Exclusive cavity imprints: an input-carried stencil dictionary and exposures.

Only NumPy and the standard library are used. No parent code is imported.
"""
from __future__ import annotations

import json
import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int64, copy=True)


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, cells = [(r, c)], []
        while todo:
            x, y = todo.pop()
            cells.append((x, y))
            for dr, dc in EIGHT if diagonal else FOUR:
                u, v = x + dr, y + dc
                if 0 <= u < h and 0 <= v < w and mask[u, v] and not seen[u, v]:
                    seen[u, v] = True
                    todo.append((u, v))
        result.append(cells)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def _tight(mask):
    r, c, b, d = _box(list(map(tuple, np.argwhere(mask))))
    return mask[r:b, c:d].copy()


def _holes(wall, diagonal=False):
    """Negative space enclosed by a single ink component, with exterior padding."""
    padded = np.pad(wall, 1)
    inside = np.zeros(padded.shape, dtype=bool)
    h, w = padded.shape
    for cells in _components(~padded, diagonal):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            for q in cells:
                inside[q] = True
    return inside[1:-1, 1:-1]


def _read(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("ambiguous background")
    bg = int(colors[np.argmax(counts)])
    if np.count_nonzero(a == bg) <= a.size // 2:
        raise ValueError("background must occupy more than half the input")
    components = _components(a != bg, True)
    candidates = []
    for cells in components:
        r, c, b, d = _box(cells)
        h, w = b - r, d - c
        if len(cells) != h * w or len({int(a[q]) for q in cells}) < 2:
            continue
        lanes = []
        if h == 2 and 2 <= w <= 4:
            if r == 0:
                lanes.append(("top", a[r, c:d], a[r + 1, c:d]))
            if b == a.shape[0]:
                lanes.append(("bottom", a[b - 1, c:d], a[b - 2, c:d]))
        if w == 2 and 2 <= h <= 4:
            if c == 0:
                lanes.append(("left", a[r:b, c], a[r:b, c + 1]))
            if d == a.shape[1]:
                lanes.append(("right", a[r:b, d - 1], a[r:b, d - 2]))
        for side, source, fill in lanes:
            s, f = tuple(map(int, source)), tuple(map(int, fill))
            if len(set(s)) == len(s) and set(s).isdisjoint(f):
                candidates.append((cells, side, dict(zip(s, f))))
    if len(candidates) != 1:
        raise ValueError("input must contain one unambiguous isolated edge key")
    key_cells, side, mapping = candidates[0]
    key_set, codes = set(key_cells), set(mapping)
    stencils, templates, live, rejected, material, open_objects = {}, [], [], [], [], []
    for cells in components:
        if set(cells) == key_set:
            continue
        vals = {int(a[q]) for q in cells}
        r, c, b, d = _box(cells)
        if len(vals) == 1 and next(iter(vals)) in codes and len(cells) > 1:
            code = next(iter(vals))
            wall = np.zeros((b - r, d - c), dtype=bool)
            for x, y in cells:
                wall[x - r, y - c] = True
            holes = _holes(wall)
            if holes.any():
                # Nested ink or commands inside a prototype would obscure its mask.
                if np.any(a[r:b, c:d][holes] != bg):
                    raise ValueError("a stencil cavity must contain only background")
                mask = _tight(holes)
                if code in stencils and not np.array_equal(stencils[code], mask):
                    raise ValueError("conflicting closed stencils of one source color")
                stencils[code] = mask
                templates.append((code, cells, holes))
            else:
                open_objects.append((code, cells))
            continue
        points = [(x, y, int(a[x, y])) for x, y in cells if int(a[x, y]) in codes]
        if len(points) == 1 and points[0][:2] == (r, c):
            live.append(points[0])
        elif points:
            rejected.extend(points)
        else:
            material.append(cells)
    if not stencils or any(code not in stencils for _, _, code in live):
        raise ValueError("every live code needs a closed stencil in this input")
    return {"grid": a, "background": bg, "side": side, "mapping": mapping,
            "key": key_cells, "stencils": stencils, "templates": templates,
            "live": live, "rejected": rejected, "material": material,
            "open": open_objects}


def _expose(shape, bg, mapping, masks, records):
    coverage = np.zeros(shape, dtype=np.int16)
    ink = np.full(shape, bg, dtype=np.int64)
    for r, c, code in records:
        mask = masks[code]
        h, w = min(mask.shape[0], shape[0] - r), min(mask.shape[1], shape[1] - c)
        visible = mask[:h, :w]
        coverage[r:r + h, c:c + w] += visible
        ink[r:r + h, c:c + w][visible] = mapping[code]
    out = np.full(shape, bg, dtype=np.int64)
    out[coverage == 1] = ink[coverage == 1]
    return out, coverage, ink


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all conventions and geometry from this input, then count exposures."""
    info = _read(input_grid)
    return _expose(info["grid"].shape, info["background"], info["mapping"],
                   info["stencils"], info["live"])[0]


def _mask(rng, kind):
    h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    m = np.zeros((h, w), dtype=bool)
    if kind == "rectangle":
        m[:] = True
    elif kind == "rounded":
        m[:] = True
        m[0, 0] = m[-1, -1] = False
        if rng.random() < .5:
            m[0, -1] = False
        if rng.random() < .5:
            m[-1, 0] = False
    elif kind == "diamond":
        n = int(rng.choice([3, 5]))
        r, c = np.indices((n, n))
        m = abs(r - n // 2) + abs(c - n // 2) <= n // 2
    elif kind == "multi":
        # Two or three separate cavities retain their relative offsets.
        w = int(rng.choice([5, 7]))
        m = np.zeros((h, w), dtype=bool)
        m[:, :2] = True
        m[int(rng.integers(0, 2)):, 3:5] = True
        if w == 7:
            m[:int(rng.integers(2, h + 1)), 6] = True
    elif kind == "profile":
        # Intersecting row intervals form a connected, hole-free polyomino.
        left, right = 0, w - 1
        for r in range(h):
            if r:
                anchor = int(rng.integers(left, right + 1))
                left = int(rng.integers(0, anchor + 1))
                right = int(rng.integers(anchor, w))
            m[r, left:right + 1] = True
        if m.all():
            m[-1, -1] = False
    else:
        m[:, 0] = True
        m[-1, :] = True
        m[int(rng.integers(h)), :int(rng.integers(2, w + 1))] = True
    if kind not in ("rectangle", "rounded"):
        m = np.rot90(m, int(rng.integers(4)))
        if rng.random() < .5:
            m = np.fliplr(m)
    return _tight(m)


def _contour(rng, mask, diagonal=False):
    thickness = 1 if max(mask.shape) > 5 else int(rng.choice([1, 1, 2]))
    inner = np.pad(mask, thickness)
    if diagonal and len(_components(mask)) == 1:
        wall = inner.copy()
        wall[1:] |= inner[:-1]
        wall[:-1] |= inner[1:]
        wall[:, 1:] |= inner[:, :-1]
        wall[:, :-1] |= inner[:, 1:]
        wall &= ~inner
        wall = _tight(wall)
    else:
        wall = ~inner
        if rng.random() < .5:
            wall[0, 0] = wall[-1, -1] = False
    if len(_components(wall, True)) != 1 or not np.array_equal(_tight(_holes(wall)), mask):
        # Dense outer material works for every supported mask, including multi-hole masks.
        wall = ~np.pad(mask, 1)
    return wall


def _key(grid, side, mapping, rng, fallback=False):
    items = list(mapping.items())
    rng.shuffle(items)
    n = len(items)
    axis = grid.shape[1] if side in ("top", "bottom") else grid.shape[0]
    start = (12 if side in ("top", "bottom") else 1) if fallback else int(rng.integers(1, axis - n))
    for i, (source, ink) in enumerate(items):
        p = start + i
        if side == "top":
            grid[0, p], grid[1, p] = source, ink
        elif side == "bottom":
            grid[-1, p], grid[-2, p] = source, ink
        elif side == "left":
            grid[p, 0], grid[p, 1] = source, ink
        else:
            grid[p, -1], grid[p, -2] = source, ink


def _reserve(blocked, r, c, h, w):
    H, W = blocked.shape
    blocked[max(0, r - 1):min(H, r + h + 1), max(0, c - 1):min(W, c + w + 1)] = True


def _patch(grid, blocked, mask, color, rng, where=None):
    h, w = mask.shape
    H, W = grid.shape
    if where is None:
        choices = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)
                   if not blocked[r:r + h, c:c + w].any()]
        if not choices:
            return False
        edges = [(r, c) for r, c in choices if r == 0 or c == 0 or r + h == H or c + w == W]
        if edges and rng.random() < .3:
            choices = edges
        r, c = choices[int(rng.integers(len(choices)))]
    else:
        r, c = where
        if r + h > H or c + w > W or blocked[r:r + h, c:c + w].any():
            return False
    grid[r:r + h, c:c + w][mask] = color
    _reserve(blocked, r, c, h, w)
    return True


def _pair(rng, codes, inks, bg, unlisted, side, lesson, index, mapping):
    """Build labels from the chosen masks/commands, independently of input parsing."""
    for attempt in range(61):
        fallback = attempt == 60
        if fallback:
            # A fixed small vocabulary makes the last packing attempt reliable;
            # it is an ordinary member of the same rule, not a semantic exception.
            masks = {codes[0]: np.ones((3, 3), dtype=bool),
                     codes[1]: np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool)}
            if len(codes) == 3:
                masks[codes[2]] = np.array([[1, 1, 0, 1, 1]] * 3, dtype=bool)
        else:
            masks = {codes[0]: _mask(rng, "rectangle")}
            masks[codes[1]] = _mask(rng, "rounded" if lesson == 1 else str(rng.choice(["profile", "diamond", "corner", "rounded"])))
            if len(codes) == 3:
                masks[codes[2]] = _mask(rng, str(rng.choice(["multi", "multi", "profile", "diamond"])))
        # Packing fallbacks limit silhouettes to a guaranteed 9x9 template box.
        walls = ({c: ~np.pad(m, 1) for c, m in masks.items()} if fallback else
                 {c: _contour(rng, m, diagonal=(c != codes[0] and rng.random() < .45)) for c, m in masks.items()})
        if lesson == 0:
            # This closes under four-neighbor background travel but leaks under
            # eight-neighbor travel: a concrete enclosure-connectivity lesson.
            walls[codes[0]] = ~np.pad(masks[codes[0]], 1)
            walls[codes[0]][0, 0] = walls[codes[0]][-1, -1] = False
        if fallback or attempt >= 15:
            H = W = 30
        elif lesson in (1, 2):
            H, W = int(rng.integers(25, 31)), int(rng.integers(25, 31))
        else:
            H, W = int(rng.integers(19, 31)), int(rng.integers(19, 31))
        grid = np.full((H, W), bg, dtype=np.int64)
        _key(grid, side, mapping, rng, fallback)
        blocked = np.zeros(grid.shape, dtype=bool)
        for r, c in np.argwhere(grid != bg):
            _reserve(blocked, int(r), int(c), 1, 1)
        ok = True
        for i, code in enumerate(codes):
            if not _patch(grid, blocked, walls[code], code, rng, (5, 10 * i) if fallback else None):
                ok = False
                break
        if not ok:
            continue
        # An opened mapped contour carries no closed negative space; an unlisted
        # closed contour carries no codebook entry. These are visibly distinct.
        if lesson == 0:
            wall = np.ones((5, 5), dtype=bool)
            wall[1:4, 1:4] = False
            wall[0, 2] = False
            if not _patch(grid, blocked, wall, codes[1], rng, (15, 23) if fallback else None):
                continue
            inert = np.ones((4, 5), dtype=bool)
            inert[1:3, 1:4] = False
            if not _patch(grid, blocked, inert, unlisted, rng, (15, 14) if fallback else None):
                continue
        rich = lesson in (1, 2) or (lesson == 3 and rng.random() < .5)
        if rich:
            bare = masks[codes[0]]
            if not _patch(grid, blocked, bare, mapping[codes[0]], rng, (23, 19) if fallback else None):
                continue
            # A copy of the rectangular imprint with a code attached beyond its
            # lower-right ink is a rejected component, not an extra request.
            h, w = bare.shape
            composite = np.zeros((h + 1, w + 1), dtype=bool)
            composite[:h, :w] = bare
            composite[h, w] = True
            free = [(r, c) for r in range(H - h) for c in range(W - w)
                    if not blocked[r:r + h + 1, c:c + w + 1].any()]
            if fallback:
                free = [(15, 20)] if not blocked[15:16 + h, 20:21 + w].any() else []
            if not free:
                continue
            r, c = free[int(rng.integers(len(free)))]
            grid[r:r + h, c:c + w][bare] = mapping[codes[0]]
            grid[r + h, c + w] = codes[0]
            _reserve(blocked, r, c, h + 1, w + 1)
        records = []
        # Only command pixels are protected from other command pixels; output
        # footprints may overlap. Other input objects are separated from prints.
        command_halo = np.zeros(grid.shape, dtype=bool)
        triple_witness = None
        def add(r, c, code, painted=False, no_overlap=False):
            m = masks[code]
            h, w = min(m.shape[0], H - r), min(m.shape[1], W - c)
            if h <= 0 or w <= 0 or blocked[r:r + h, c:c + w].any() or command_halo[r, c]:
                return False
            if triple_witness is not None:
                x, y = triple_witness
                if r <= x < r + h and c <= y < c + w and m[x - r, y - c]:
                    return False
            if no_overlap:
                current = _expose(grid.shape, bg, mapping, masks, records)[1]
                if np.any(current[r:r + h, c:c + w][m[:h, :w]]):
                    return False
            if painted:
                if m[0, 0] or r + m.shape[0] > H or c + m.shape[1] > W:
                    return False
                obj = m.copy()
                obj[0, 0] = True
                if len(_components(obj, True)) != 1:
                    return False
                # Keep painted input material away from all existing command pixels.
                if command_halo[r:r + h, c:c + w][obj].any():
                    return False
                grid[r:r + h, c:c + w][m] = mapping[code]
                _reserve(command_halo, r, c, h, w)
            else:
                _reserve(command_halo, r, c, 1, 1)
            grid[r, c] = code
            records.append((r, c, code))
            return True
        if lesson in (1, 2):
            # Three same-code requests and one different code meet. The rectangle
            # guarantees double and triple exposure; the fourth teaches color blindness.
            mh = max(masks[codes[0]].shape[0] + 2, masks[codes[1]].shape[0] + 2)
            mw = max(masks[codes[0]].shape[1] + 2, masks[codes[1]].shape[1] + 2)
            choices = [(r, c) for r in range(H - mh + 1) for c in range(W - mw + 1)
                       if not blocked[r:r + mh, c:c + mw].any()]
            if fallback:
                choices = [(15, 0)] if not blocked[15:15 + mh, :mw].any() else []
            if not choices:
                continue
            r, c = choices[int(rng.integers(len(choices)))]
            for dr, dc, code in ((0, 0, codes[0]), (2, 0, codes[0]), (0, 2, codes[0]), (2, 2, codes[1])):
                if not add(r + dr, c + dc, code):
                    ok = False
                    break
            if not ok:
                continue
            triple_cells = np.argwhere(_expose(grid.shape, bg, mapping, masks, records)[1] == 3)
            if not len(triple_cells):
                continue
            triple_witness = tuple(map(int, triple_cells[0]))
        # Full copies teach each local stencil. A correctly anchored preprint is
        # regenerated, while the bare and lower-right attached copies vanish.
        needed = codes if lesson not in (1, 2) else codes[1:]
        for code in needed:
            m = masks[code]
            painted = rich and code == codes[1] and not m[0, 0] and m[:2, :2].any()
            choices = [(r, c) for r in range(H - m.shape[0] + 1) for c in range(W - m.shape[1] + 1)]
            rng.shuffle(choices)
            if fallback:
                choices = ([(15, 10)] if code == codes[1] and rich else
                           [(23, 0)] if len(codes) == 3 and code == codes[2] else
                           [(15, 1), (15, 8), (23, 8)])
            if not any(add(r, c, code, painted, no_overlap=lesson == 0) for r, c in choices):
                ok = False
                break
        if not ok:
            continue
        if lesson in (1, 2):
            # Training supplies separate bottom/right truncations; tests combine them.
            edges = ("bottom", "right") if lesson == 1 else ("bottom", "right", "corner")
            for edge in edges:
                if fallback:
                    choices = {"bottom": [(H - 2, 5)], "right": [(22, W - 2)], "corner": [(H - 2, W - 2)]}[edge]
                elif edge == "bottom":
                    choices = [(H - 2, c) for c in range(W - masks[codes[0]].shape[1] + 1)]
                elif edge == "right":
                    choices = [(r, W - 2) for r in range(H - masks[codes[0]].shape[0] + 1)]
                else:
                    choices = [(H - 2, W - 2)]
                rng.shuffle(choices)
                if not any(add(r, c, codes[0]) for r, c in choices):
                    ok = False
                    break
        if not ok:
            continue
        target = (len(records) if lesson == 1 else int(rng.integers(9, 13)) if lesson == 2
                  else len(codes) + int(rng.integers(0, 3)) if lesson == 0 else int(rng.integers(3, 8)))
        positions = [(r, c) for r in range(H) for c in range(W)]
        rng.shuffle(positions)
        for r, c in positions:
            if len(records) >= target:
                break
            add(r, c, int(rng.choice(codes)), no_overlap=lesson == 0)
        if len(records) < target:
            continue
        out, coverage, _ = _expose(grid.shape, bg, mapping, masks, records)
        try:
            info = _read(grid)
        except ValueError:
            continue
        if sorted(info["live"]) != sorted(records) or any(not np.array_equal(info["stencils"].get(c), m) for c, m in masks.items()):
            continue
        if lesson in (1, 2):
            # Actual triple coverage (not merely four windows meeting) is needed
            # to disambiguate exact-one printing from XOR/parity.
            if not np.any(coverage == 3):
                continue
        if lesson == 0 and coverage.max() != 1:
            continue
        return {"input": grid, "output": out}
    raise RuntimeError("bounded scene packing failed")


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    n = int(rng.integers(2, 4))
    palette = list(map(int, rng.permutation(10)))
    codes, inks = palette[:n], palette[n:2 * n]
    backgrounds, unlisted = palette[2 * n:2 * n + 2], palette[2 * n + 2]
    sides = list(map(str, rng.permutation(["top", "left", "bottom", "right"])))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt + nq):
        split, lesson = ("train", i if i < 2 else 3) if i < nt else ("test", 2)
        bg = backgrounds[i % 2]
        # The first two inputs explicitly change color associations while keeping
        # the same source colors. Each pair's local edge key is authoritative.
        destinations = inks if i == 0 else inks[1:] + inks[:1] if i == 1 else list(map(int, rng.permutation(inks)))
        if i >= 2 and rng.random() < .25:
            destinations = [int(rng.choice(inks)) for _ in codes]
        mapping = dict(zip(codes, destinations))
        side = sides[i] if i < 2 else str(rng.choice(sides))
        for _ in range(8):
            pair = _pair(rng, codes, inks, bg, unlisted, side, lesson, i, mapping)
            signature = _signature(pair["input"])
            if signature not in seen:
                seen.add(signature)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("distinct-input construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _metrics(info):
    shape, masks, records = info["grid"].shape, info["stencils"], info["live"]
    _, cov, ink = _expose(shape, info["background"], info["mapping"], masks, records)
    individual = []
    for record in records:
        individual.append(_expose(shape, info["background"], info["mapping"], masks, [record])[1] > 0)
    mixed = np.zeros(shape, dtype=bool)
    windows_only = False
    for i in range(len(records)):
        r, c, code = records[i]
        h, w = masks[code].shape
        for j in range(i):
            rr, cc, other = records[j]
            hh, ww = masks[other].shape
            overlap = individual[i] & individual[j]
            if info["mapping"][code] != info["mapping"][other]:
                mixed |= overlap
            if max(r, rr) < min(r + h, rr + hh, shape[0]) and max(c, cc) < min(c + w, cc + ww, shape[1]) and not overlap.any():
                windows_only = True
    return {"live": len(records), "double": int(np.count_nonzero(cov == 2)),
            "triple": int(np.count_nonzero(cov == 3)), "mixed": int(mixed.sum()),
            "windows_only": windows_only,
            "bottom": any(r + masks[code].shape[0] > shape[0] for r, c, code in records),
            "right": any(c + masks[code].shape[1] > shape[1] for r, c, code in records),
            "corner": any(r + masks[code].shape[0] > shape[0] and c + masks[code].shape[1] > shape[1] for r, c, code in records),
            "painted": any(len(comp) > 1 and any((r, c) in comp for r, c, _ in records)
                           for comp in _components((info["grid"] != info["background"]), True)),
            "multi": any(len(_components(m)) > 1 for m in masks.values()),
            "irregular": any(not m.all() for m in masks.values()),
            "unique": int(np.count_nonzero(cov == 1)), "coverage": cov, "ink": ink}


def validate_generated_puzzle(puzzle):
    """Reject corrupt labels and reconstruct semantic witnesses from returned grids."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos, metrics = set(), {"train": [], "test": []}, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            sig = _signature(a)
            assert sig not in seen, "duplicate input"
            seen.add(sig)
            info = _read(a)
            m = _metrics(info)
            expected = _expose(a.shape, info["background"], info["mapping"], info["stencils"], info["live"])[0]
            assert np.array_equal(out, expected), "incorrect imprint output"
            assert info["live"] and m["unique"] > 0 and not np.array_equal(a, out)
            assert np.all(out[m["coverage"] != 1] == info["background"]), "overlap must be blank"
            assert set(info["stencils"]) == set(info["mapping"]), "missing input stencil"
            assert len({c for c, _, _ in info["templates"]}) == len(info["mapping"])
            assert np.all(np.isin(out, [info["background"], *info["mapping"].values()]))
            infos[split].append(info)
            metrics[split].append(m)
    train = metrics["train"]
    assert any(m["double"] and m["triple"] and m["mixed"] for m in train), "missing exact-one / mixed-color witness"
    assert any(m["coverage"].max() == 1 for m in train), "missing unobscured single exposure lesson"
    assert any(m["irregular"] for m in train), "missing negative-space geometry lesson"
    assert any(i["open"] for i in infos["train"]), "missing open contour contrast"
    diagonal_witness = False
    for info in infos["train"]:
        for _, cells, holes in info["templates"]:
            r, c, b, d = _box(cells)
            wall = np.zeros((b - r, d - c), dtype=bool)
            for x, y in cells:
                wall[x - r, y - c] = True
            diagonal_witness |= holes.any() and not _holes(wall, True).any()
    assert diagonal_witness, "four-neighbor enclosure not distinguished from eight"
    assert any(i["rejected"] and i["material"] for i in infos["train"]), "missing command membership contrast"
    assert any(m["painted"] for m in train), "missing correctly anchored preprint"
    assert any(m["bottom"] for m in train) and any(m["right"] for m in train), "clipping not demonstrated"
    assert len({i["side"] for i in infos["train"]}) >= 2
    assert len({i["background"] for i in infos["train"]}) >= 2
    assert infos["train"][0]["mapping"] != infos["train"][1]["mapping"], "local lookup not demonstrated"
    for code, mask in infos["train"][0]["stencils"].items():
        H, W = infos["train"][0]["grid"].shape
        assert any(k == code and r + mask.shape[0] <= H and c + mask.shape[1] <= W
                   for r, c, k in infos["train"][0]["live"]), "missing full local stencil demonstration"
    assert all(m["live"] >= 9 and m["corner"] for m in metrics["test"]), "missing composed test shift"
    # Explicit ablations: filling bounding boxes instead of negative space, and
    # retaining only one command per code instead of every symbolic anchor.
    for name in ("bbox", "one_anchor", "union", "parity", "all_points", "fixed_mapping"):
        defeated = False
        for info in infos["train"]:
            masks, records = info["stencils"], info["live"]
            if name == "bbox":
                masks = {c: np.ones(m.shape, dtype=bool) for c, m in masks.items()}
            if name == "one_anchor":
                used, records = set(), []
                for r, c, code in info["live"]:
                    if code not in used:
                        records.append((r, c, code))
                        used.add(code)
            if name == "all_points":
                records = records + info["rejected"]
            mapping = infos["train"][0]["mapping"] if name == "fixed_mapping" else info["mapping"]
            altered, cov, ink = _expose(info["grid"].shape, info["background"], mapping, masks, records)
            if name in ("union", "parity"):
                altered = np.full(info["grid"].shape, info["background"], dtype=int)
                chosen = cov > 0 if name == "union" else cov % 2 == 1
                altered[chosen] = ink[chosen]
            correct = solve(info["grid"])
            defeated |= not np.array_equal(altered, correct)
        assert defeated, "training admits shortcut: " + name


def self_test():
    """Bounded 42-seed diversity, API, equivariance and semantic mutation checks."""
    # Hand-authored fixture, with expected pixels specified without _expose.
    # Three rectangles have a blank triple junction; two L-shaped boxes meet
    # without sharing a true pixel. There is also a clipped corner and an open
    # source contour beside a lower-right attached, rejected code point.
    a = np.zeros((16, 20), dtype=int)
    a[:2, 5:7] = [[1, 2], [3, 4]]
    a[4:9, :5] = 1
    a[5:8, 1:4] = 0
    a[4:9, 10:15] = 2
    a[5, 11:14] = 0
    a[6:8, 11] = 0
    a[3:6, 16:19] = [[2, 0, 2], [2, 0, 2], [2, 2, 2]]
    a[4:6, 6:8] = 3
    a[6, 8] = 1
    for r, c, code in ((10, 0, 1), (10, 2, 1), (12, 0, 1),
                       (10, 10, 2), (12, 12, 2), (15, 19, 1)):
        a[r, c] = code
    expected = np.zeros_like(a)
    expected[10:15, :5] = [[3, 3, 0, 3, 3], [3, 3, 0, 3, 3],
                          [0, 0, 0, 3, 3], [3, 3, 3, 0, 0], [3, 3, 3, 0, 0]]
    expected[10, 10:13] = expected[12, 12:15] = 4
    expected[11:13, 10] = expected[13:15, 12] = 4
    expected[15, 19] = 3
    assert np.array_equal(solve(a), expected)
    info = _read(a)
    assert (6, 8, 1) in info["rejected"] and info["open"]
    assert _metrics(info)["windows_only"] and _metrics(info)["triple"] == 1
    puzzles, shapes, silhouettes, counts, sides, palettes = [], set(), set(), set(), set(), set()
    multi = windows_only = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                info = _read(pair["input"])
                m = _metrics(info)
                shapes.add(info["grid"].shape)
                counts.add(m["live"])
                sides.add(info["side"])
                palettes.add(tuple(sorted(info["mapping"].items())))
                multi += m["multi"]
                windows_only += m["windows_only"]
                silhouettes.update(_signature(mask) for mask in info["stencils"].values())
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                # Color renaming must not change the semantic roles or answer.
                perm = np.roll(np.arange(10), 3)
                assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    assert len({json.dumps({s: [{k: np.asarray(v).tolist() for k, v in pair.items()} for pair in p[s]] for s in p}, separators=(",", ":")) for p in puzzles}) == 42
    assert len(shapes) >= 30 and len(silhouettes) >= 40 and len(counts) >= 8
    assert sides == {"top", "left", "bottom", "right"} and len(palettes) >= 42
    assert multi >= 5 and windows_only >= 5
    p = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], puzzles[0][s]) for k in a)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(42, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    out = broken["test"][0]["output"]
    bg = _read(broken["test"][0]["input"])["background"]
    out[tuple(np.argwhere(out != bg)[0])] = bg
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing imprint pixel accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
