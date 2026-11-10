"""Stencil weaving: input-local keyed tiles print only rooted material domains.

The solver interprets visible swatches and eight-connected components. It never
uses generation records, training labels, a seed, or a persistent dictionary.
"""
from __future__ import annotations

import numpy as np


_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask, neighbors=_D8):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(sorted(cells))
    return result


def _bounds(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _tile_key(tile, bg):
    # Compare transparent geometry across examples with different backgrounds.
    return _key(np.where(tile == bg, -1, tile))


def _interpret(input_grid):
    """Recover background, delimited tile dictionary, and material objects."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if np.count_nonzero(a == bg) <= a.size // 2:
        raise ValueError("the exterior background must have a strict majority")
    swatches = []
    for color in colors:
        if color == bg:
            continue
        for cells in _components(a == color, _D4):
            y, x, h, w = _bounds(cells)
            if h < 4 or w < 4:
                continue
            perimeter = {(r, c) for r in range(y, y + h) for c in range(x, x + w)
                         if r in (y, y + h - 1) or c in (x, x + w - 1)}
            if set(cells) != perimeter:
                continue
            inner = a[y + 1:y + h - 1, x + 1:x + w - 1]
            code = int(inner[0, 0])
            if code in (bg, int(color)) or np.count_nonzero(inner == code) != 1:
                continue
            tile = inner.copy()
            tile[0, 0] = bg
            if np.count_nonzero(tile != bg) < 2 or np.any(tile == color):
                continue
            swatches.append((y, x, h, w, int(color), code, tile))
    if not swatches or len({s[4] for s in swatches}) != 1:
        raise ValueError("expected one common frame role for the tile swatches")
    excluded = np.zeros(a.shape, dtype=bool)
    book = {}
    for y, x, h, w, frame, code, tile in swatches:
        if excluded[y:y + h, x:x + w].any():
            raise ValueError("swatches cannot overlap")
        excluded[y:y + h, x:x + w] = True
        if code in book and not np.array_equal(book[code], tile):
            raise ValueError("conflicting swatches for one key")
        book[code] = tile
    codes = set(book)
    inks = {int(v) for tile in book.values() for v in np.unique(tile) if v != bg}
    frame = swatches[0][4]
    if (codes & inks or frame in codes | inks or bg in codes | inks):
        raise ValueError("key, ink, frame and background roles must be disjoint")
    objects = _components((a != bg) & ~excluded)
    material_colors = {int(a[p]) for cells in objects for p in cells} - codes
    if len(material_colors) != 1 or material_colors & (inks | {frame, bg}):
        raise ValueError("field objects must use one distinct material color plus keys")
    material = material_colors.pop()
    live, rejected, bare = [], [], []
    for cells in objects:
        y, x, h, w = _bounds(cells)
        markers = [(r, c, int(a[r, c])) for r, c in cells if int(a[r, c]) in codes]
        record = {"cells": cells, "box": (y, x, h, w), "markers": markers}
        if len(markers) == 1 and markers[0][:2] == (y, x):
            record["code"] = markers[0][2]
            live.append(record)
        elif markers:
            rejected.append(record)
        else:
            bare.append(record)
    return {"grid": a, "background": bg, "frame": frame, "material": material,
            "book": book, "swatches": swatches, "live": live,
            "rejected": rejected, "bare": bare}


def _print_domains(info, *, membership=True, global_phase=False, rectangles=False,
                   once=False, solid=False):
    """The flags are explicit shortcut counterfactuals used only by validation."""
    bg, a, book = info["background"], info["grid"], info["book"]
    out = np.full(a.shape, bg, dtype=np.int8)
    domains = info["live"]
    if not membership:
        domains = domains + info["rejected"]
    for obj in domains:
        y, x, h, w = obj["box"]
        code = obj.get("code", obj["markers"][0][2])
        tile = book[code]
        cells = obj["cells"]
        if rectangles:
            cells = [(r, c) for r in range(y, y + h) for c in range(x, x + w)]
        for r, c in cells:
            dr, dc = (r, c) if global_phase else (r - y, c - x)
            if once and (dr >= tile.shape[0] or dc >= tile.shape[1]):
                continue
            color = int(tile[dr % tile.shape[0], dc % tile.shape[1]])
            if solid:
                color = int(tile[tile != bg][0])
            if color != bg:
                out[r, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Print periodic, transparent tiles inside admitted input silhouettes."""
    return _print_domains(_interpret(input_grid))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _sample_tile(rng, bg, inks):
    h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    tile = np.full((h, w), bg, dtype=np.int8)
    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) != (0, 0)]
    # Both axes have visible ink, and the symbolic key is always transparent.
    chosen = {(h - 1, int(rng.integers(w))), (int(rng.integers(h)), w - 1)}
    chosen.discard((0, 0))
    count = int(rng.integers(max(3, len(chosen)), len(candidates) + 1))
    rng.shuffle(candidates)
    chosen.update(candidates[:count])
    assignments = list(inks) + [int(_pick(rng, inks)) for _ in range(len(chosen) - len(inks))]
    rng.shuffle(assignments)
    for (r, c), color in zip(sorted(chosen), assignments):
        tile[r, c] = color
    return tile


def _domain(rng, h, w, style):
    """Connected, tight silhouettes; root corner is always part of the body."""
    mask = np.zeros((h, w), dtype=bool)
    if style == "hole":
        mask[:] = True
        # Off-center holes with variable dimensions, plus optional edge notches.
        a = int(rng.integers(1, h - 2))
        b = int(rng.integers(1, w - 2))
        eh = int(rng.integers(a + 1, h))
        ew = int(rng.integers(b + 1, w))
        mask[a:eh, b:ew] = False
        if h > 6 and w > 6 and rng.random() < .5:
            mask[-2, w // 2:] = False
    elif style == "stair":
        lengths = np.sort(rng.integers(2, w + 1, size=h))
        lengths[-1] = w
        if rng.random() < .5:
            lengths = lengths[::-1]
        for r, length in enumerate(lengths):
            mask[r, :int(length)] = True
    elif style == "comb":
        mask[:, :int(rng.integers(1, 3))] = True
        for r in range(0, h, int(rng.integers(2, 4))):
            mask[r, :int(rng.integers(2, w + 1))] = True
        mask[0, :] = mask[-1, :] = True
    elif style == "notch":
        mask[:] = True
        mask[1:h - 1, int(rng.integers(2, w)): ] = False
        if w > 4:
            mask[2:h - 2, 1] = False
    elif style == "solid":
        mask[:] = True
    else:
        # A monotone skeleton guarantees full extents and connectedness.
        r = c = 0
        mask[r, c] = True
        while r < h - 1 or c < w - 1:
            if r == h - 1 or (c < w - 1 and rng.random() < .5):
                c += 1
            else:
                r += 1
            mask[r, c] = True
        target = int(rng.integers(max(h + w - 1, h * w // 3), h * w + 1))
        while np.count_nonzero(mask) < target:
            frontier = sorted({(int(r) + dr, int(c) + dc)
                               for r, c in np.argwhere(mask) for dr, dc in _D4
                               if 0 <= r + dr < h and 0 <= c + dc < w and not mask[r + dr, c + dc]})
            r, c = _pick(rng, frontier)
            mask[r, c] = True
    assert mask[0, 0] and len(_components(mask)) == 1
    return mask


def _place_boxes(rng, shapes, field):
    y0, x0, H, W = field
    for _ in range(20):
        occupied = np.zeros((H, W), dtype=bool)
        positions = [None] * len(shapes)
        order = sorted(range(len(shapes)), key=lambda i: -np.prod(shapes[i]))
        for i in order:
            h, w = shapes[i]
            if h > H or w > W:
                break
            candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
            rng.shuffle(candidates)
            for r, c in candidates:
                if not occupied[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1].any():
                    occupied[r:r + h, c:c + w] = True
                    positions[i] = (y0 + r, x0 + c)
                    break
            if positions[i] is None:
                break
        else:
            return positions
    # Bounded shelf fallback in both orientations. It has no semantic effect.
    for transpose in (False, True):
        A, B = (W, H) if transpose else (H, W)
        order = sorted(range(len(shapes)), key=lambda i: -(shapes[i][1] if transpose else shapes[i][0]))
        r = c = shelf = 0
        positions = [None] * len(shapes)
        for i in order:
            h, w = shapes[i][::-1] if transpose else shapes[i]
            if c + w > B:
                r, c, shelf = r + shelf + 1, 0, 0
            if r + h > A or w > B:
                break
            positions[i] = (y0 + c, x0 + r) if transpose else (y0 + r, x0 + c)
            c += w + 1
            shelf = max(shelf, h)
        else:
            maxr = max(p[0] + shapes[i][0] for i, p in enumerate(positions))
            maxc = max(p[1] + shapes[i][1] for i, p in enumerate(positions))
            dy, dx = int(rng.integers(y0 + H - maxr + 1)), int(rng.integers(x0 + W - maxc + 1))
            return [(r + dy, c + dx) for r, c in positions]
    return None


def _swatch_layout(rng, a, book, bg, frame, side):
    entries = list(book.items())
    rng.shuffle(entries)
    horizontal = side in ("top", "bottom")
    shapes = [(t.shape[0] + 2, t.shape[1] + 2) for _, t in entries]
    lengths = [s[1] if horizontal else s[0] for s in shapes]
    gaps = [int(rng.integers(1, 3)) for _ in lengths[:-1]]
    total = sum(lengths) + sum(gaps)
    extent = a.shape[1] if horizontal else a.shape[0]
    if total + 2 > extent:
        return None
    thick = max(s[0] if horizontal else s[1] for s in shapes)
    cursor = int(rng.integers(1, extent - total))
    for i, (code, tile) in enumerate(entries):
        h, w = shapes[i]
        if side == "top":
            y, x = 1 + int(rng.integers(thick - h + 1)), cursor
        elif side == "bottom":
            y, x = a.shape[0] - thick - 1 + int(rng.integers(thick - h + 1)), cursor
        elif side == "left":
            y, x = cursor, 1 + int(rng.integers(thick - w + 1))
        else:
            y, x = cursor, a.shape[1] - thick - 1 + int(rng.integers(thick - w + 1))
        a[y:y + h, x:x + w] = frame
        a[y + 1:y + h - 1, x + 1:x + w - 1] = tile
        a[y + 1, x + 1] = code
        cursor += lengths[i] + (gaps[i] if i < len(gaps) else 0)
    # One blank line separates the swatch rail from the unframed field.
    if side == "top":
        return thick + 3, 0, a.shape[0] - thick - 3, a.shape[1]
    if side == "bottom":
        return 0, 0, a.shape[0] - thick - 3, a.shape[1]
    if side == "left":
        return 0, thick + 3, a.shape[0], a.shape[1] - thick - 3
    return 0, 0, a.shape[0], a.shape[1] - thick - 3


def _make_pair(rng, config, index, split, previous):
    codes, inks = config["codes"], config["inks"]
    bg = config["backgrounds"][index % 2] if index < 2 else int(_pick(rng, config["backgrounds"]))
    frame, material = config["frame"], config["material"]
    if split == "train" and index == 0:
        requests, artifacts = [codes[0], codes[0]], ["attached", "bare"]
        styles = ["hole", "stair"]
    elif split == "train" and index == 1:
        requests, artifacts = list(codes), ["wrong", "double", "bare"]
        styles = ["comb", "growth", "hole"][:len(requests)]
    else:
        n = int(rng.integers(1, 4))
        requests = [int(_pick(rng, codes)) for _ in range(n)]
        artifacts = list(rng.choice(["attached", "bare", "wrong", "double"],
                                    size=int(rng.integers(0, 3)), replace=False))
        styles = [_pick(rng, ("hole", "stair", "comb", "growth", "notch", "solid")) for _ in requests]
    if split == "test":
        # Move the swatch rail to the perpendicular axis; tile orientation stays fixed.
        sides = ("left", "right") if config["rail_axis"] == 0 else ("top", "bottom")
    else:
        sides = ("top", "bottom") if config["rail_axis"] == 0 else ("left", "right")
    side = _pick(rng, sides)
    for attempt in range(60):
        H, W = (30, 30) if attempt >= 45 else (int(rng.integers(23, 31)), int(rng.integers(23, 31)))
        book = {code: _sample_tile(rng, bg, inks) for code in codes}
        if previous is not None and _tile_key(book[codes[0]], bg) == previous:
            continue
        a = np.full((H, W), bg, dtype=np.int8)
        field = _swatch_layout(rng, a, book, bg, frame, side)
        if field is None:
            continue
        descriptors = []
        for i, code in enumerate(requests):
            th, tw = book[code].shape
            # Every live domain spans more than two periods along both axes.
            h, w = 2 * th + 1, 2 * tw + 1
            if attempt < 20:
                h += int(rng.integers(3))
                w += int(rng.integers(3))
            mask = _domain(rng, h, w, styles[i])
            descriptors.append(("live", code, mask))
        for kind in artifacts:
            h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            mask = _domain(rng, h, w, _pick(rng, ("solid", "stair", "growth")))
            if kind == "attached":
                last_r = mask.shape[0] - 1
                last_c = int(np.flatnonzero(mask[last_r])[-1])
                enlarged = np.zeros((h + 1, max(w, last_c + 2)), dtype=bool)
                enlarged[:h, :w] = mask
                enlarged[h, last_c + 1] = True
                mask = enlarged
            descriptors.append((kind, int(_pick(rng, codes)), mask))
        positions = _place_boxes(rng, [d[2].shape for d in descriptors], field)
        if positions is None:
            continue
        expected = np.full(a.shape, bg, dtype=np.int8)
        for (kind, code, mask), (y, x) in zip(descriptors, positions):
            block = a[y:y + mask.shape[0], x:x + mask.shape[1]]
            block[mask] = material
            cells = list(map(tuple, np.argwhere(mask)))
            if kind == "live":
                block[0, 0] = code
                tile = book[code]
                # Direct construction from the selected domain, before parsing input.
                for r, c in cells:
                    value = int(tile[r % tile.shape[0], c % tile.shape[1]])
                    if value != bg:
                        expected[y + r, x + c] = value
            elif kind == "attached":
                r, c = cells[-1]
                block[r, c] = code
            elif kind in ("wrong", "double"):
                r, c = _pick(rng, cells[1:])
                block[r, c] = code
                if kind == "double":
                    block[0, 0] = int(_pick(rng, codes))
        if np.count_nonzero(a == bg) <= a.size // 2:
            continue
        info = _interpret(a)
        if len(info["live"]) != len(requests) or not np.array_equal(_print_domains(info), expected):
            continue
        if not _evidence(info, expected)["basic"]:
            continue
        if index == 0 and split == "train":
            f = _evidence(info, expected)
            if not all(f[k] for k in ("membership", "silhouette", "phase")):
                continue
        return {"input": a, "output": expected}
    raise RuntimeError("bounded material packing exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    nkeys, ninks = int(rng.integers(1, 4)), int(rng.integers(1, 3))
    config = {"backgrounds": palette[:2], "frame": palette[2], "material": palette[3],
              "codes": palette[4:4 + nkeys], "inks": palette[4 + nkeys:4 + nkeys + ninks],
              "rail_axis": int(rng.integers(2))}
    puzzle, seen, previous = {"train": [], "test": []}, set(), None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            for _ in range(12):
                pair = _make_pair(rng, config, i, split, previous)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not make a distinct input")
            seen.add(_key(pair["input"]))
            info = _interpret(pair["input"])
            previous = _tile_key(info["book"][config["codes"][0]], info["background"])
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(info, out):
    shortcut = lambda **kw: not np.array_equal(_print_domains(info, **kw), out)
    return {"basic": bool(info["live"]) and shortcut(once=True) and shortcut(solid=True),
            "repeat": shortcut(once=True), "transparent": shortcut(solid=True),
            "membership": shortcut(membership=False), "silhouette": shortcut(rectangles=True),
            "phase": shortcut(global_phase=True)}


def validate_generated_puzzle(puzzle):
    """Validate visible semantics and demonstration coverage, never latent records."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, train_evidence, categories = set(), [], {"train": set(), "test": set()}
    palettes, dictionaries = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            assert a.shape == out.shape and not np.array_equal(a, out)
            info = _interpret(a)
            assert np.array_equal(_print_domains(info), out), "incorrect woven output"
            assert all(2 <= dim <= 4 for tile in info["book"].values() for dim in tile.shape)
            assert len(info["book"]) <= 3
            bg = info["background"]
            assert all(tile[0, 0] == bg for tile in info["book"].values())
            assert all(np.count_nonzero(tile != bg) >= 3 and (tile[-1] != bg).any()
                       and (tile[:, -1] != bg).any() for tile in info["book"].values())
            boxes = [obj["box"] for kind in ("live", "rejected", "bare") for obj in info[kind]]
            for i, (y, x, h, w) in enumerate(boxes):
                assert all(y+h < v or v+k < y or x+w < u or u+l < x
                           for v, u, k, l in boxes[i+1:]), "material boxes must have a blank separator"
            for obj in info["live"]:
                y, x, h, w = obj["box"]
                th, tw = info["book"][obj["code"]].shape
                assert h > 2 * th and w > 2 * tw, "live body must expose tiling"
                # All printed pixels remain inside the exact original silhouette.
                cells = set(obj["cells"])
                assert all((int(r) + y, int(c) + x) in cells
                           for r, c in zip(*np.nonzero(out[y:y+h, x:x+w] != bg)))
            for y, x, h, w, _, _, _ in info["swatches"]:
                assert np.all(out[y:y+h, x:x+w] == bg), "legend must disappear"
            f = _evidence(info, out)
            assert f["basic"], "every example must need repetition and transparent printing"
            if info["bare"]:
                categories[split].add("bare")
            for obj in info["rejected"]:
                categories[split].add("multiple" if len(obj["markers"]) > 1 else "misrooted")
            if split == "train":
                train_evidence.append(f)
                dictionaries.append({k: _tile_key(t, bg) for k, t in info["book"].items()})
            palettes.append((info["frame"], info["material"], tuple(sorted(info["book"])),
                             tuple(sorted({int(v) for t in info["book"].values() for v in np.unique(t) if v != bg}))))
    assert len(set(palettes)) == 1, "palette roles must remain stable inside a puzzle"
    assert categories["test"] <= categories["train"], "test membership category not demonstrated"
    assert {"bare", "misrooted", "multiple"} <= categories["train"]
    assert all(any(f[k] for f in train_evidence) for k in ("membership", "silhouette", "phase"))
    assert any(dictionaries[0][k] != dictionaries[1][k] for k in dictionaries[0]), "teach input-local dictionaries"


def self_test():
    """Bounded semantic, mutation, equivariance, and 42-seed diversity checks."""
    puzzles, records, fingerprints, shapes, tiles, silhouettes = [], [], set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        fingerprints.add(tuple(_key(pair["input"]) for s in p.values() for pair in s))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _interpret(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                shapes.add(pair["input"].shape)
                records.append((len(info["book"]), len(info["live"]), _evidence(info, pair["output"])))
                for tile in info["book"].values():
                    tiles.add(_tile_key(tile, info["background"]))
                for obj in info["live"]:
                    y, x, h, w = obj["box"]
                    silhouettes.add((h, w, tuple((r-y, c-x) for r, c in obj["cells"])))
    assert len(fingerprints) == 42
    assert len(shapes) >= 25 and len(tiles) >= 100 and len(silhouettes) >= 100
    assert {r[0] for r in records} == {1, 2, 3}
    assert {r[1] for r in records} >= {1, 2, 3}
    for name in ("membership", "silhouette", "phase", "repeat", "transparent"):
        assert sum(f[name] for _, _, f in records) >= 42
    p = puzzles[0]
    again = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], again[s]) for k in a)
    # Arbitrary palette permutations must not change the geometric rule.
    recolor = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
    pair = p["test"][0]
    assert np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]])
    # Reference sensitivity: change an actually used tile pixel, keep field fixed.
    a = pair["input"].copy()
    info = _interpret(a)
    obj = info["live"][0]
    code = obj["code"]
    tile = info["book"][code]
    y, x, _, _ = obj["box"]
    r, c = next((r-y, c-x) for r, c in obj["cells"] if tile[(r-y) % tile.shape[0], (c-x) % tile.shape[1]] != info["background"])
    tr, tc = r % tile.shape[0], c % tile.shape[1]
    sy, sx, *_ = next(s for s in info["swatches"] if s[5] == code)
    a[sy+1+tr, sx+1+tc] = info["background"]
    # A tile has at least three ink pixels under current construction.
    assert not np.array_equal(solve(a), pair["output"]), "reference must control the answer"
    # Membership sensitivity: move a live key within the same unchanged body.
    moved = pair["input"].copy()
    oy, ox, _, _ = obj["box"]
    mr, mc = next(p for p in obj["cells"] if p != (oy, ox))
    moved[oy, ox], moved[mr, mc] = info["material"], code
    prediction = solve(moved)
    assert all(prediction[r, c] == info["background"] for r, c in obj["cells"])
    assert not np.array_equal(prediction, pair["output"]), "root admission must control the answer"
    # Deleting an entire live domain's output must be rejected with inputs fixed.
    broken = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in ps] for s, ps in p.items()}
    for r, c in obj["cells"]:
        broken["test"][0]["output"][r, c] = info["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing live texture accepted")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1) + 1):
            q = generate_puzzle(314159 + nt * 10 + nq, nt, nq)
            assert len(q["train"]) == nt and len(q["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(7, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, semantic counterfactuals, all legal counts")
