"""Cavity-driven self-stamping: exact shape fit controls local block expansion.

No parent module, files, seed, or demonstrations are consulted by solve().
"""

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIR4:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(frozenset(cells))
    return result


def _connected(mask):
    return len(_components(mask)) == 1


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _shape(cells):
    cells = tuple(cells)
    r0, _, c0, _ = _box(cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _mask_shape(mask):
    return _shape(map(tuple, np.argwhere(mask)))


def _orient(array, op):
    """0..3 are CCW rotations; 4..7 reflect a CCW rotation left/right."""
    a = np.rot90(array, op % 4)
    return (np.fliplr(a) if op >= 4 else a).copy()


def _orbit(mask):
    return {_mask_shape(_orient(mask, op)) for op in range(8)}


def _canon(mask):
    return min(_orbit(mask))


def _array_key(array):
    a = np.ascontiguousarray(array, dtype=np.int8)
    return a.shape, a.tobytes()


def _imprint(tile, op, background, policy=(False, True)):
    transformed = _orient(tile, op)
    colors = sorted(set(map(int, np.unique(tile))) - {int(background)})
    if len(colors) != 2:
        raise ValueError("a stamp has exactly two payload colors")
    if policy[int(op >= 4)]:
        a, b = colors
        original = transformed.copy()
        transformed[original == a] = b
        transformed[original == b] = a
    return transformed


def _matches(cavity, tiles, background):
    """Geometry first: never infer the plug from its color value or area."""
    wanted = _shape(cavity)
    candidates = []
    for ki, tile in enumerate(tiles):
        colors = sorted(set(map(int, np.unique(tile))) - {background})
        for op in range(8):
            oriented = _orient(tile, op)
            for color in colors:
                if _mask_shape(oriented == color) == wanted:
                    candidates.append({"key": ki, "color": color, "op": op})
    if not candidates:
        raise ValueError("a cavity matches no complete stamp color-region")
    imprints = {_array_key(_imprint(tiles[m["key"]], m["op"], background))
                for m in candidates}
    if len(imprints) != 1:
        raise ValueError("a cavity admits competing colored imprints")
    return candidates


def _read(input_grid):
    """Recover a monochrome board, a stamp library, and every local fit."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if max(counts) * 2 <= a.size:
        raise ValueError("the input background must be a strict majority")
    readings = []
    for signal in map(int, colors):
        if signal == bg:
            continue
        rb = _box(map(tuple, np.argwhere(a == signal)))
        r0, r1, c0, c1 = rb
        board = a[r0:r1 + 1, c0:c1 + 1]
        if not set(map(int, np.unique(board))) <= {bg, signal} or not np.any(board == bg):
            continue
        payload_mask = (a != bg) & (a != signal)
        records, bad = [], False
        for cells in _components(payload_mask):
            kr0, kr1, kc0, kc1 = _box(cells)
            # At least one wholly blank row or column separates bounding boxes.
            if not (kr1 + 1 < r0 or r1 + 1 < kr0 or kc1 + 1 < c0 or c1 + 1 < kc0):
                bad = True
                break
            tile = a[kr0:kr1 + 1, kc0:kc1 + 1].copy()
            payload = set(map(int, np.unique(tile))) - {bg}
            if (tile.shape[0] != tile.shape[1] or len(payload) != 2 or signal in payload
                    or set(map(tuple, np.argwhere(tile != bg)))
                    != {(r - kr0, c - kc0) for r, c in cells}
                    or not all(_connected(tile == color) for color in payload)):
                bad = True
                break
            records.append({"tile": tile, "box": (kr0, kr1, kc0, kc1)})
        if bad or not records:
            continue
        n = records[0]["tile"].shape[0]
        if (any(x["tile"].shape != (n, n) for x in records)
                or board.shape[0] * n > 30 or board.shape[1] * n > 30):
            continue
        tiles = [x["tile"] for x in records]
        cavities = _components(board == bg)
        try:
            matches = [_matches(cells, tiles, bg) for cells in cavities]
        except ValueError:
            continue
        used = {m["key"] for group in matches for m in group}
        if used != set(range(len(tiles))):
            continue
        readings.append({"background": bg, "signal": signal, "board": board.copy(),
                         "board_box": rb, "tiles": tiles, "key_boxes": [x["box"] for x in records],
                         "n": n, "cavities": cavities, "matches": matches})
    if len(readings) != 1:
        raise ValueError("expected one unambiguous board/library interpretation")
    return readings[0]


def _expand(info, policy=(False, True), mode="normal"):
    bg, n = info["background"], info["n"]
    h, w = info["board"].shape
    out = np.full((h * n, w * n), bg, dtype=np.int8)
    for cavity, candidates in zip(info["cavities"], info["matches"]):
        m = candidates[0]
        tile = info["tiles"][m["key"]]
        if mode == "ignore_fit":
            imprint = info["tiles"][0]
        elif mode == "ignore_orientation":
            imprint = _imprint(tile, 0, bg, (policy[1], policy[1])) if m["op"] >= 4 else tile
        elif mode == "inverse_fit":
            inverse = (-m["op"]) % 4 if m["op"] < 4 else m["op"]
            imprint = _imprint(tile, inverse, bg, policy)
        else:
            imprint = _imprint(tile, m["op"], bg, policy)
        cells = [min(cavity)] if mode == "one_per_cavity" else cavity
        for r, c in cells:
            out[r * n:(r + 1) * n, c * n:(c + 1) * n] = imprint
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible geometry only; the family fixes reflection swapping."""
    return _expand(_read(input_grid))


def _make_tile(rng, n, forbidden=(), small_area=None, force_holes=False):
    """Grow two connected color-regions, optionally eroding background gaps.

    A small growth window lets several cavities coexist at the coarser scale.
    It does not restrict which region the solver can fit.
    """
    for _ in range(800):
        a = np.zeros((n, n), dtype=bool)
        side = min(3, n)
        pr, pc = int(rng.integers(n - side + 1)), int(rng.integers(n - side + 1))
        area = small_area if small_area is not None else int(rng.integers(4, min(8, n*n-3)))
        r, c = pr + int(rng.integers(side)), pc + int(rng.integers(side))
        a[r, c] = True
        for _ in range(area - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in np.argwhere(a) for dr, dc in _DIR4
                               if pr <= r + dr < pr + side and pc <= c + dc < pc + side
                               and not a[r + dr, c + dc]})
            a[frontier[int(rng.integers(len(frontier)))]] = True
        b = ~a
        if not _connected(b) or len(_orbit(a)) != 8 or len(_orbit(b)) != 8:
            continue
        tile = np.where(a, 1, 2).astype(np.int8)
        # Size 3 supports the asymmetric L/P partition; removing a pixel would
        # destroy the unique fit. Larger tiles can preserve real background cells.
        hole_target = (int(rng.integers(1, max(2, n))) if force_holes else
                       int(rng.integers(0, max(2, n))) if n > 3 else 0)
        removed = 0
        for index in rng.permutation(n * n):
            if removed == hole_target:
                break
            r, c = divmod(int(index), n)
            previous = int(tile[r, c])
            if previous == 1 and small_area is not None:
                continue
            tile[r, c] = 0
            masks = (tile == 1, tile == 2)
            support = tile != 0
            if (min(int(x.sum()) for x in masks) < 4
                    or not all(_connected(x) and len(_orbit(x)) == 8 for x in masks)
                    or not _connected(support)
                    or not all(np.any(x) for x in (support[0], support[-1], support[:, 0], support[:, -1]))):
                tile[r, c] = previous
            else:
                removed += 1
        if force_holes and removed == 0:
            continue
        signatures = {_canon(tile == color) for color in (1, 2)}
        if len(signatures) != 2 or signatures & set(forbidden):
            continue
        return _orient(tile, int(rng.integers(8)))
    raise RuntimeError("bounded asymmetric stamp construction exhausted")


def _library(rng, n, count, witness=False):
    tiles, forbidden = [], set()
    for i in range(count):
        area = (5 if count == 2 else 4) if witness else None
        tile = _make_tile(rng, n, forbidden, area, force_holes=witness or (n > 3 and rng.random() < .6))
        tiles.append(tile)
        forbidden.update(_canon(tile == color) for color in (1, 2))
    return tiles


def _small_op(rng, tile, color, parity):
    options = list(range(parity * 4, parity * 4 + 4))
    heights = []
    for op in options:
        r0, r1, _, _ = _box(map(tuple, np.argwhere(_orient(tile, op) == color)))
        heights.append(r1 - r0 + 1)
    best = [op for op, h in zip(options, heights) if h == min(heights)]
    return best[int(rng.integers(len(best)))]


def _pack_cavities(rng, tiles, plans, board_shapes):
    shapes = []
    for ki, color, op in plans:
        shapes.append(_shape(map(tuple, np.argwhere(_orient(tiles[ki], op) == color))))
    order = sorted(range(len(shapes)), key=lambda i: -len(shapes[i]))
    for _ in range(120):
        h, w = board_shapes[int(rng.integers(len(board_shapes)))]
        reserved, placed = set(), [None] * len(shapes)
        for i in order:
            sh = shapes[i]
            ch, cw = max(r for r, c in sh) + 1, max(c for r, c in sh) + 1
            if ch > h or cw > w:
                break
            anchors = [(r, c) for r in range(h - ch + 1) for c in range(w - cw + 1)]
            rng.shuffle(anchors)
            if rng.random() < .35:
                anchors.sort(key=lambda p: p[0] not in (0, h-ch) and p[1] not in (0, w-cw))
            for ar, ac in anchors:
                cells = {(ar + r, ac + c) for r, c in sh}
                if not cells & reserved:
                    placed[i] = frozenset(cells)
                    reserved.update((r + dr, c + dc) for r, c in cells
                                    for dr, dc in ((0, 0),) + _DIR4)
                    break
            else:
                break
        else:
            board = np.ones((h, w), dtype=np.int8)
            for cells in placed:
                for p in cells:
                    board[p] = 0
            if (int(board.sum()) >= 4
                    and all(np.any(x) for x in (board[0], board[-1], board[:, 0], board[:, -1]))
                    and len(_components(board == 0)) == len(plans)):
                return board, placed
    raise ValueError("cavities did not pack with distinct components and tight board bounds")


def _separated(a, b):
    r, c, h, w = a
    s, t, j, k = b
    return r + h < s or s + j < r or c + w < t or t + k < c


def _display(rng, board, tiles, bg):
    objects = [board] + tiles
    foreground = sum(int(np.count_nonzero(x != bg)) for x in objects)
    shapes = [x.shape for x in objects]
    order = sorted(range(len(objects)), key=lambda i: -objects[i].size)
    for attempt in range(100):
        min_h, min_w = max(h for h, w in shapes), max(w for h, w in shapes)
        if attempt >= 80:
            h = w = 30
        else:
            h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
        if h*w <= 2 * foreground:
            continue
        boxes = [None] * len(objects)
        for i in order:
            sh, sw = shapes[i]
            anchors = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
            rng.shuffle(anchors)
            if rng.random() < .35:
                anchors.sort(key=lambda p: p[0] not in (0, h-sh) and p[1] not in (0, w-sw))
            for r, c in anchors:
                box = (r, c, sh, sw)
                if all(_separated(box, other) for other in boxes if other is not None):
                    boxes[i] = box
                    break
            else:
                break
        else:
            inp = np.full((h, w), bg, dtype=np.int8)
            for obj, (r, c, sh, sw) in zip(objects, boxes):
                inp[r:r+sh, c:c+sw] = obj
            return inp
    raise ValueError("bounded display packing exhausted")


def _pair(rng, params, kind, index=0):
    witness = kind == "witness"
    if witness:
        n, count, cavities = 4, (2 if params["library"] else 1), 2
    elif kind == "size_witness":
        n, count, cavities = 3, 1, 2
    elif kind == "train":
        n = int(rng.integers(3, 5))
        count = 2 if n == 4 and params["library"] and rng.random() < .6 else 1
        cavities = int(rng.integers(count, 3))
    elif params["regime"] == "repetition":
        n, count = 4, (2 if params["library"] else 1)
        cavities = int(rng.integers(3, 5))
    else:
        n, count = 5, 1
        cavities = int(rng.integers(1, 3))

    # Conditions are constructive sampling constraints, never hidden solve data.
    tiles = _library(rng, n, count, witness=witness)
    if witness:
        if count == 1:
            plans = [(0, 1, _small_op(rng, tiles[0], 1, 0)),
                     (0, 2, int(rng.integers(4, 8)))]
            board_shapes = [(7, w) for w in range(4, 8)]
        else:
            plans = [(i, 1, _small_op(rng, tiles[i], 1, i)) for i in range(2)]
            board_shapes = [(h, w) for h in range(5, 8) for w in range(5, 8)]
    else:
        small_only = count > 1 or cavities > 2 or (n == 5 and cavities == 2)
        plans = []
        for i in range(cavities):
            ki = i if i < count else int(rng.integers(count))
            color = 1 if small_only else 1 + int(rng.integers(2))
            if kind == "size_witness":
                color = i + 1
                op = int(rng.choice([1, 3])) if i == 0 else int(rng.integers(4, 8))
            else:
                op = int(rng.integers(8))
            plans.append((ki, color, op))
        limit = 30 // n
        lower = 5 if cavities == 3 else 6 if cavities == 4 else 4
        board_shapes = [(h, w) for h in range(lower, limit+1) for w in range(lower, limit+1)]
    board_codes, locations = _pack_cavities(rng, tiles, plans, board_shapes)
    if params["stable"]:
        palette = params["palette"]
    else:
        palette = [int(c) for c in rng.permutation(10)]
    bg, signal = palette[:2]
    payload = palette[2:]
    colored_tiles = []
    for i, tile in enumerate(tiles):
        # Shared, overlapping, or separate key palettes; shape decides association.
        pair = payload[:2] if i == 0 or rng.random() < .45 else payload[1:3] if rng.random() < .5 else payload[2:4]
        lut = np.array([bg, *pair], dtype=np.int8)
        colored_tiles.append(lut[tile])
    board = np.where(board_codes, signal, bg).astype(np.int8)
    inp = _display(rng, board, colored_tiles, bg)
    out = np.full((board.shape[0]*n, board.shape[1]*n), bg, dtype=np.int8)
    for cells, (ki, color, op) in zip(locations, plans):
        tile = _imprint(colored_tiles[ki], op, bg)
        for r, c in cells:
            out[r*n:(r+1)*n, c*n:(c+1)*n] = tile
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= int(num_train) <= 10 or not 1 <= int(num_test) <= 4
            or num_train <= num_test):
        raise ValueError("counts require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    params = {"regime": str(rng.choice(["repetition", "resolution"])),
              "stable": bool(rng.random() < .3),
              "palette": [int(c) for c in rng.permutation(10)]}
    params["library"] = params["regime"] == "repetition" and rng.random() < .65
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            kind = "witness" if split == "train" and i == 0 else "size_witness" if split == "train" and i == 1 else split
            for _ in range(40):
                try:
                    pair = _pair(rng, params, kind, i)
                    info = _read(pair["input"])
                    if not np.array_equal(_expand(info), pair["output"]):
                        continue
                    if kind == "witness" and params["library"] and not _wrong_nearest(info):
                        continue
                except (ValueError, RuntimeError):
                    continue
                signature = _array_key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded verified pair construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    bg = info["background"]
    selected = [group[0] for group in info["matches"]]
    return {"board_shape": info["board"].shape, "n": info["n"],
            "cavity_count": len(info["cavities"]), "key_count": len(info["tiles"]),
            "active_cells": sum(map(len, info["cavities"])),
            "operations": {m["op"] for m in selected},
            "reuse": len(selected) > len({m["key"] for m in selected}),
            "plug_relations": {int(np.sign(np.count_nonzero(info["tiles"][m["key"]] == m["color"])
                                 - np.count_nonzero((info["tiles"][m["key"]] != bg)
                                                   & (info["tiles"][m["key"]] != m["color"]))))
                               for m in selected},
            "holes": sum(int(np.count_nonzero(t == bg)) for t in info["tiles"]),
            "equal_area_decoy": _equal_area_decoy(info),
            "wrong_nearest": _wrong_nearest(info),
            "edge_cavity": any(any(r in (0, info["board"].shape[0]-1)
                                         or c in (0, info["board"].shape[1]-1) for r, c in s)
                               for s in info["cavities"])}


def _equal_area_decoy(info):
    selected = [m[0] for m in info["matches"]]
    return any(a["key"] != b["key"] and len(ca) == len(cb)
               for j, (ca, a) in enumerate(zip(info["cavities"], selected))
               for cb, b in list(zip(info["cavities"], selected))[j+1:])


def _wrong_nearest(info):
    if len(info["tiles"]) < 2:
        return False
    r0, _, c0, _ = info["board_box"]
    centers = [((a+b)/2, (c+d)/2) for a, b, c, d in info["key_boxes"]]
    for cells, group in zip(info["cavities"], info["matches"]):
        cr = r0 + sum(r for r, c in cells) / len(cells)
        cc = c0 + sum(c for r, c in cells) / len(cells)
        distances = [abs(r-cr) + abs(c-cc) for r, c in centers]
        actual = group[0]["key"]
        if any(d < distances[actual] for i, d in enumerate(distances) if i != actual):
            return True
    return False


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _array_key(inp)
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            info = _read(inp)
            assert np.array_equal(_expand(info), out), "incorrect cavity expansion"
            assert not np.array_equal(inp, out)
            assert info["signal"] not in out, "the stencil is an instruction, not payload"
            information[split].append(info)
            n, bg = info["n"], info["background"]
            assert out.shape == tuple(d*n for d in info["board"].shape)
            assert np.all(out.reshape(info["board"].shape[0], n,
                                      info["board"].shape[1], n).transpose(0, 2, 1, 3)
                          [info["board"] != bg] == bg), "inactive stencil cells must be blank blocks"
            for cells, group in zip(info["cavities"], info["matches"]):
                assert len(cells) >= 4 and len(_components(_cells_mask(cells))) == 1
                m = group[0]
                tile = _imprint(info["tiles"][m["key"]], m["op"], bg)
                for r, c in cells:
                    assert np.array_equal(out[r*n:(r+1)*n, c*n:(c+1)*n], tile)
    train, test = information["train"], information["test"]
    for policy in ((False, False), (True, False), (True, True)):
        assert any(not np.array_equal(_expand(i, policy), pair["output"])
                   for i, pair in zip(train, puzzle["train"])), "parity alternative survives training"
    for mode in ("ignore_fit", "ignore_orientation", "inverse_fit", "one_per_cavity"):
        assert any(not np.array_equal(_expand(i, mode=mode), p["output"])
                   for i, p in zip(train, puzzle["train"])), "a parent-omitting shortcut survives training"
    assert any(_features(i)["plug_relations"] == {-1, 1} for i in train), "demonstrate both region-size roles"
    assert any(_features(i)["holes"] > 0 for i in train), "demonstrate imprint background retention"
    assert len({i["n"] for i in train}) >= 2, "demonstrate visible, variable block size"
    assert max(len(i["tiles"]) for i in test) <= max(len(i["tiles"]) for i in train)
    if any(len(i["tiles"]) > 1 for i in test):
        assert any(_equal_area_decoy(i) for i in train), "demonstrate shape matching beyond area"
        assert any(_wrong_nearest(i) for i in train), "demonstrate matching beyond proximity"
    # The test shift is either more independent/reused cavities or finer stamps.
    assert all(len(i["cavities"]) > max(len(t["cavities"]) for t in train)
               or i["n"] > max(t["n"] for t in train) for i in test)


def _cells_mask(cells):
    shape = _shape(cells)
    a = np.zeros((max(r for r, c in shape)+1, max(c for r, c in shape)+1), dtype=bool)
    for p in shape:
        a[p] = True
    return a


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> dict:
    """Bounded, observable diversity and family-invariant audit over 42 seeds."""
    inputs, boards, tile_patterns, dimensions, operations = set(), set(), set(), set(), set()
    counts, relations, backgrounds, imprints = set(), set(), set(), set()
    flags = {k: 0 for k in ("reuse", "multiple_keys", "edge_cavity", "fine_holes", "equal_area_decoy",
                           "wrong_nearest", "repetition_shift", "resolution_shift")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _equal(puzzle, generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read(pair["input"])
                f = _features(info)
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                inputs.add(_array_key(pair["input"]))
                boards.add(_array_key(info["board"] == info["background"]))
                dimensions.add(pair["input"].shape)
                operations.update(f["operations"])
                counts.add((f["key_count"], f["cavity_count"]))
                relations.update(f["plug_relations"])
                backgrounds.add(info["background"])
                imprints.add(f["n"])
                for tile in info["tiles"]:
                    mapping, normalized = {info["background"]: 0}, []
                    for c in tile.flat:
                        normalized.append(mapping.setdefault(int(c), len(mapping)))
                    tile_patterns.add((tile.shape, tuple(normalized)))
                flags["reuse"] += f["reuse"]
                flags["multiple_keys"] += f["key_count"] > 1
                flags["edge_cavity"] += f["edge_cavity"]
                flags["fine_holes"] += f["holes"] > 0
                flags["equal_area_decoy"] += f["equal_area_decoy"]
                flags["wrong_nearest"] += f["wrong_nearest"]
                if split == "test":
                    flags["repetition_shift"] += f["cavity_count"] >= 3
                    flags["resolution_shift"] += f["n"] == 5
    assert len(inputs) == 210 and len(boards) >= 100 and len(tile_patterns) >= 70
    assert len(dimensions) >= 70 and operations == set(range(8))
    assert {-1, 1} <= relations and backgrounds == set(range(10)) and imprints == {3, 4, 5}
    assert all(v > 0 for v in flags.values())
    assert {n for k, n in counts} >= {1, 2, 3, 4} and {k for k, n in counts} == {1, 2}
    # Counts, mutation rejection, reordering, and input immutability.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(313, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    validate_generated_puzzle({"train": p["train"][::-1], "test": p["test"]})
    before = p["test"][0]["input"].copy()
    solve(p["test"][0]["input"])
    assert np.array_equal(before, p["test"][0]["input"])
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    # A symmetric plug genuinely fails the unique-imprint condition.
    ambiguous_tile = np.array([[1, 1, 1], [2, 2, 2], [2, 2, 2]], dtype=np.int8)
    try:
        _matches(frozenset((0, c) for c in range(3)), [ambiguous_tile], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("competing reflected imprints accepted")
    return {"seeds": 42, "pairs": 210, "distinct_inputs": len(inputs), "board_patterns": len(boards),
            "normalized_stamp_patterns": len(tile_patterns), "input_shapes": len(dimensions),
            "operations": sorted(operations), "key_cavity_counts": sorted(counts),
            "plug_size_relations": sorted(relations), "flags": flags,
            "semantic_verification": "generator-owned input-only reconstruction and counterfactual checks"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
