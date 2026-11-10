"""Contour-selected inward bands. Only the returned input determines its answer."""
from collections import deque
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            q = stack.pop()
            cells.append(q)
            for dr, dc in (_EIGHT if diagonal else _FOUR):
                rr, cc = q[0] + dr, q[1] + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(cells)
    return result


def _read_input(a):
    """Find the local multicolor edge table, then original background cavities."""
    values, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("Background must be the unique most frequent input color")
    bg = int(values[np.argmax(counts)])
    h, w = a.shape
    candidates = []
    for cells in _components(a != bg, diagonal=True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        if len(cells) != (r1-r0+1)*(c1-c0+1):
            continue
        if len({int(a[q]) for q in cells}) < 2:
            continue
        sides = [s for s, flag in (("top", r0 == 0), ("bottom", r1 == h-1),
                                  ("left", c0 == 0), ("right", c1 == w-1)) if flag]
        if len(sides) != 1:
            continue
        side = sides[0]
        patch = a[r0:r1+1, c0:c1+1]
        # Canonical table: first axis is boundary -> inward. Second axis is
        # merely a set of independently paired columns; its order is irrelevant.
        table = {"top": lambda: patch, "bottom": lambda: patch[::-1],
                 "left": lambda: patch.T, "right": lambda: patch[:, ::-1].T}[side]()
        length, width = table.shape[0] - 1, table.shape[1]
        if not (1 <= length <= 6 and 1 <= width <= 5):
            continue
        sources = list(map(int, table[0]))
        if len(set(sources)) != width:
            continue
        codes = {source: tuple(map(int, table[1:, j])) for j, source in enumerate(sources)}
        candidates.append({"bg": bg, "side": side, "cells": set(cells),
                           "sources": tuple(sources), "codes": codes, "length": length,
                           "table": table.copy()})
    if len(candidates) != 1:
        raise ValueError("Expected one isolated dense multicolor table on one edge")
    key = candidates[0]
    regions = []
    for cells in _components(a == bg):
        boundary, edge = set(), False
        for r, c in cells:
            edge |= r in (0, h-1) or c in (0, w-1)
            for dr, dc in _FOUR:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and a[rr, cc] != bg:
                    boundary.add((rr, cc))
        if edge:
            regions.append({"cells": cells, "edge": True, "owner": None})
            continue
        colors = {int(a[q]) for q in boundary}
        if boundary & key["cells"] or len(colors) != 1:
            raise ValueError("Every enclosed cavity must have one contour color")
        regions.append({"cells": cells, "edge": False, "owner": next(iter(colors))})
    return key, regions


def _depths(cells):
    """Distance zero is the first background layer adjacent to the wall."""
    members = set(cells)
    depths, queue = {}, deque()
    for r, c in cells:
        if any((r+dr, c+dc) not in members for dr, dc in _FOUR):
            depths[(r, c)] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in _FOUR:
            q = (r+dr, c+dc)
            if q in members and q not in depths:
                depths[q] = depths[(r, c)] + 1
                queue.append(q)
    return depths


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    key, regions = _read_input(a)
    out = a.copy()
    for region in regions:
        code = key["codes"].get(region["owner"])
        if region["edge"] or code is None:
            continue
        for q, d in _depths(region["cells"]).items():
            out[q] = code[min(d, len(code)-1)]
    return out


def _erode_expected(a, key, regions):
    """Validation construction: peel simultaneous shells, without distance BFS."""
    out = a.copy()
    for region in regions:
        code = key["codes"].get(region["owner"])
        if region["edge"] or code is None:
            continue
        remaining, layer = set(region["cells"]), 0
        while remaining:
            shell = {q for q in remaining if any((q[0]+dr, q[1]+dc) not in remaining
                                                 for dr, dc in _FOUR)}
            assert shell
            for q in shell:
                out[q] = code[min(layer, len(code)-1)]
            remaining -= shell
            layer += 1
    return out


def _fingerprint(a):
    return a.shape, a.tobytes()


def _features(a, out):
    key, regions = _read_input(a)
    active = [r for r in regions if not r["edge"] and r["owner"] in key["codes"]]
    result = {"side": key["side"], "length": key["length"], "sources": len(key["codes"]),
              "bg": key["bg"], "max_depth": 0, "active": len(active),
              "inert": sum(not r["edge"] and r["owner"] not in key["codes"] for r in regions),
              "nonflat": False, "repeat": False, "cap": False, "chain": False,
              "irregular": False, "bbox_witness": False, "diagonal": False,
              "selection": False, "multi": False, "open": False, "global_witness": False}
    global_code = key["codes"][key["sources"][0]]
    for region in regions:
        if not region["edge"]:
            result["global_witness"] |= any(int(out[q]) != global_code[min(d, len(global_code)-1)]
                                             for q, d in _depths(region["cells"]).items())
    active_first = set()
    for region in active:
        depths = _depths(region["cells"])
        code = key["codes"][region["owner"]]
        maxdepth = max(depths.values())
        result["max_depth"] = max(result["max_depth"], maxdepth)
        result["nonflat"] |= any(int(out[q]) != code[0] for q in depths)
        compressed = tuple(c for i, c in enumerate(code) if i == 0 or c != code[i-1])
        result["repeat"] |= any(code[min(d, len(code)-1)] != compressed[min(d, len(compressed)-1)]
                                 for d in depths.values())
        result["cap"] |= maxdepth >= len(code) and code[-1] != code[0]
        result["chain"] |= code[0] in key["codes"] and key["codes"][code[0]][0] != code[0]
        active_first.add(code[0])
        rr, cc = zip(*region["cells"])
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        result["irregular"] |= len(depths) != (r1-r0+1)*(c1-c0+1)
        result["bbox_witness"] |= any(code[min(d, len(code)-1)] != code[min(r-r0, r1-r, c-c0, c1-c, len(code)-1)]
                                       for (r, c), d in depths.items())
    result["selection"] = len(active_first) >= 2
    # Object grouping is only for diagnostics, never for cavity semantics.
    for color in np.unique(a):
        if color == key["bg"]:
            continue
        mask = a == color
        for q in key["cells"]:
            mask[q] = False
        for cells in _components(mask, diagonal=True):
            members = set(cells)
            owned = []
            for region in regions:
                if region["edge"] or region["owner"] != color:
                    continue
                if any((r+dr, c+dc) in members for r, c in region["cells"] for dr, dc in _FOUR):
                    owned.append(region)
            result["multi"] |= len(owned) >= 2
            result["open"] |= color in key["codes"] and not owned
            if owned and color in key["codes"]:
                local = np.zeros(a.shape, dtype=bool)
                for q in members:
                    local[q] = True
                result["diagonal"] |= len(_components(local)) > 1
    return result


def _dilate(mask, diagonal):
    out = mask.copy()
    h, w = mask.shape
    for dr, dc in (_EIGHT if diagonal else _FOUR):
        ra, rb = max(0, dr), min(h, h+dr)
        ca, cb = max(0, dc), min(w, w+dc)
        out[ra:rb, ca:cb] |= mask[ra-dr:rb-dr, ca-dc:cb-dc]
    return out


def _wall(cavity, thickness=1, diagonal=True):
    hole = np.pad(cavity, thickness)
    ink = hole.copy()
    for _ in range(thickness):
        ink = _dilate(ink, diagonal)
    ink &= ~hole
    rr, cc = np.nonzero(ink)
    return ink[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _shape(rng, kind, deep=False, level=None):
    """Procedural cavity masks rather than a fixed library of finished grids."""
    if kind == "diamond":
        radius = int(level if level is not None else rng.integers(2, 5))
        rr, cc = np.indices((2*radius+1, 2*radius+1))
        ink = abs(rr-radius) + abs(cc-radius) == radius
    elif kind == "multi":
        # One connected wall with 2..4 independently shaped cavities.
        h = int(rng.integers(5, 9))
        widths = [int(rng.integers(1, 4)) for _ in range(int(rng.integers(2, 5)))]
        ink = np.ones((h, sum(widths) + len(widths) + 1), dtype=bool)
        col = 1
        for width in widths:
            top = int(rng.integers(1, 3))
            bottom = int(rng.integers(max(top+1, h-3), h))
            ink[top:bottom, col:col+width] = False
            col += width+1
    else:
        if level is not None:
            h = 2*int(level)+1
            w = h + int(rng.integers(0, 3))
        else:
            h = int(rng.integers(7, 10) if deep else rng.integers(1, 8))
            w = int(rng.integers(8, 11) if deep else rng.integers(1, 9))
        if kind == "dumbbell":
            # Two lobes share a narrow passage; peeling can split the core.
            h = max(5, h)
            lobe = max(3, min(7, w//2+1))
            cavity = np.zeros((h, 2*lobe+3), dtype=bool)
            cavity[:, :lobe] = cavity[:, -lobe:] = True
            mid = h//2
            cavity[mid:mid+int(rng.integers(1, 3)), lobe:-lobe] = True
        else:
            cavity = np.ones((h, w), dtype=bool)
            if kind == "notch":
                cut_h = int(rng.integers(2, min(4, h)))
                cut_w = int(rng.integers(1, min(3, w)))
                cavity[:cut_h, :cut_w] = False
            elif kind == "profile":
                # Overlapping horizontal spans give connected, concave cavities.
                left, right = 0, w-1
                for r in range(h):
                    anchor = int(rng.integers(left, right+1))
                    left = int(rng.integers(0, anchor+1))
                    right = int(rng.integers(anchor, w))
                    cavity[r] = False
                    cavity[r, left:right+1] = True
        thickness = 1 if deep or level is not None else int(rng.choice([1, 1, 2]))
        ink = _wall(cavity, thickness, diagonal=kind in ("rectangle", "open") or rng.random() < .5)
        if kind == "open":
            # Rectangular wall; carve a full-width leak through its top thickness.
            col = thickness + w//2
            ink[:thickness, col] = False
    ink = np.rot90(ink, int(rng.integers(4)))
    if rng.random() < .5:
        ink = np.fliplr(ink)
    return ink.copy()


def _put_table(a, table, start):
    a[:table.shape[0], start:start+table.shape[1]] = table


def _pack(rng, table, shapes):
    """Variable canvas and bounded packing, ending with spacious 30x30 trials."""
    maxh = max(s.shape[0] for s in shapes)
    maxw = max(s.shape[1] for s in shapes)
    for attempt in range(35):
        h = int(rng.integers(max(12, maxh+2, table.shape[0]+3), 31))
        w = int(rng.integers(max(12, maxw+2, table.shape[1]+2), 31))
        if attempt >= 20:
            h = w = 30
        a = np.zeros((h, w), dtype=np.int8)
        start = int(rng.integers(1, w-table.shape[1]))
        _put_table(a, table, start)
        blocked = _dilate(a != 0, True)
        positions = [None]*len(shapes)
        for i in sorted(range(len(shapes)), key=lambda j: -shapes[j].size):
            sh, sw = shapes[i].shape
            options = [(r, c) for r in range(h-sh+1) for c in range(w-sw+1)
                       if not np.any(blocked[r:r+sh, c:c+sw])]
            if not options:
                break
            border = [(r, c) for r, c in options if r == 0 or c == 0 or r+sh == h or c+sw == w]
            pool = border if border and rng.random() < .35 else options
            r, c = pool[int(rng.integers(len(pool)))]
            positions[i] = (r, c)
            blocked[max(0, r-1):min(h, r+sh+1), max(0, c-1):min(w, c+sw+1)] = True
        else:
            return a, positions
    return None


def _make_pair(rng, lesson=None, shift=None, side="top", bg=0):
    for attempt in range(60):
        k = int(rng.integers(2, 5) if lesson is not None else rng.integers(1, 6))
        length = 3 if lesson == 0 else 2 if lesson == 1 else int(
            rng.integers(4, 7) if shift == "longer" else rng.integers(1, 4) if shift == "train_short" else rng.integers(1, 7))
        codes = rng.integers(1, 10, size=(length, k), dtype=np.int8)
        # A nontrivial panel and first-layer direction witness, even for K=L=1.
        if codes[0, 0] == 1:
            codes[0, 0] = 2
        inert = k+1
        if lesson == 0:
            codes[:, 0] = [2, 2, inert]
            codes[0, 1] = k+2  # An actual overlap/one-step witness.
            specs = [(1, "notch", True, None), (2, "rectangle", False, None),
                     (1, "open", False, None), (inert, "rectangle", False, None)]
        elif lesson == 1:
            codes[:, 0] = [inert, k+2]
            codes[0, 1] = k+3
            specs = [(1, "diamond", False, 4), (2, "multi", False, None),
                     (inert, "rectangle", False, None)]
        elif shift in ("deeper", "longer"):
            level = int(rng.integers(5, 7)) if shift == "deeper" else max(3, length-1)
            kind = str(rng.choice(["rectangle", "notch", "diamond"]))
            # Diamond radius gives depth radius-1, others give depth 'level'.
            specs = [(int(rng.integers(1, k+1)), kind, False, level+1 if kind == "diamond" else level)]
            if rng.random() < .7:
                specs.append((int(rng.integers(1, k+1)), str(rng.choice(["profile", "multi", "open"])), False, None))
            if rng.random() < .35:
                specs.append((inert, "rectangle", False, None))
        else:
            n = int(rng.integers(1, 6))
            kinds = ["rectangle", "profile", "diamond", "multi", "open", "dumbbell"]
            specs = [(int(rng.integers(1, k+1)), str(rng.choice(kinds[:-1])), False, None)]
            if shift == "recombine":
                specs[0] = (1, "dumbbell", False, None)
                n = max(2, n)
            for _ in range(n-1):
                color = int(rng.integers(1, k+1)) if rng.random() < .8 else inert
                specs.append((color, str(rng.choice(kinds)), False, None))
        shapes = [_shape(rng, kind, deep, level) for _, kind, deep, level in specs]
        order = rng.permutation(k)
        table = np.vstack([np.arange(1, k+1, dtype=np.int8), codes])[:, order]
        packed = _pack(rng, table, shapes)
        if packed is None:
            continue
        a, positions = packed
        for (color, _, _, _), ink, (r, c) in zip(specs, shapes, positions):
            a[r:r+ink.shape[0], c:c+ink.shape[1]][ink] = color
        if np.count_nonzero(a == 0) <= max(np.count_nonzero(a == c) for c in range(1, 10)):
            continue
        # Rotate the whole visible convention, then permute all literal roles.
        a = np.rot90(a, {"top": 0, "left": 1, "bottom": 2, "right": 3}[side]).copy()
        palette = np.array([bg] + list(map(int, rng.permutation([c for c in range(10) if c != bg]))), dtype=np.int8)
        a = palette[a]
        out = solve(a)
        f = _features(a, out)
        if f["active"] == 0 or not np.any(a != out):
            continue
        if lesson == 0 and not all(f[n] for n in ("nonflat", "repeat", "chain", "bbox_witness", "selection", "open", "inert")):
            continue
        if lesson == 1 and not all(f[n] for n in ("nonflat", "cap", "diagonal", "multi", "selection")):
            continue
        if shift == "deeper" and f["max_depth"] < 5:
            continue
        if shift == "longer" and f["max_depth"] < length-1:
            continue
        if shift == "train_short" and f["max_depth"] > 4:
            continue
        if shift in ("deeper", "longer", "recombine") and not (f["nonflat"] and f["global_witness"]):
            continue
        return {"input": a, "output": out}
    raise RuntimeError("Bounded scene construction failed")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside permitted range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    backgrounds = list(map(int, rng.permutation(10)))
    shift = str(rng.choice(["deeper", "longer", "recombine"]))
    sides = list(map(str, rng.permutation(["top", "left", "bottom", "right"])))
    train_sides = [sides[0], sides[1]] + [str(rng.choice(sides)) for _ in range(num_train-2)]
    familiar = set(train_sides)
    test_pool = [s for s in sides if s not in familiar] if rng.random() < .5 else list(sides[:2])
    if not test_pool:
        test_pool = sides
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            lesson = i if split == "train" and i < 2 else None
            scene_shift = ("train_short" if shift in ("longer", "deeper") else None) if split == "train" else shift
            side = train_sides[i] if split == "train" else str(rng.choice(test_pool))
            for _ in range(20):
                bg = backgrounds[(i if split == "train" else num_train+i) % 10]
                pair = _make_pair(rng, lesson, scene_shift, side, bg)
                signature = _fingerprint(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Unique scene construction failed")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _count(len(puzzle["train"]), "num_train", 2, 10)
    _count(len(puzzle["test"]), "num_test", 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            signature = _fingerprint(a)
            assert signature not in seen
            seen.add(signature)
            key, regions = _read_input(a)
            assert np.array_equal(out, _erode_expected(a, key, regions)), "Incorrect cavity bands"
            assert np.array_equal(out[a != key["bg"]], a[a != key["bg"]]), "Original ink changed"
            for region in regions:
                if region["edge"] or region["owner"] not in key["codes"]:
                    assert all(out[q] == a[q] for q in region["cells"]), "Inert/exterior region changed"
            f = _features(a, out)
            assert f["active"] and np.any(a != out)
            features[split].append(f)
    train = features["train"]
    # Returned grids, not hidden generation flags, must teach both contributions.
    for name in ("nonflat", "repeat", "cap", "chain", "irregular", "bbox_witness", "diagonal",
                 "multi", "selection", "open", "inert"):
        assert any(f[name] for f in train), "Missing training witness: " + name
    assert len({f["side"] for f in train}) >= 2
    assert len({f["length"] for f in train}) >= 2
    assert len({f["bg"] for f in train}) >= 2
    assert all(f["nonflat"] and f["global_witness"] for f in features["test"]), "Test must require both concepts"


def _puzzle_fingerprint(puzzle):
    return tuple((split, _fingerprint(np.asarray(p["input"], dtype=np.int8)),
                  _fingerprint(np.asarray(p["output"], dtype=np.int8)))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed coverage, semantic fixtures, mutations and API boundaries."""
    signatures, shapes, normalized, geometries = set(), set(), set(), set()
    lengths, source_counts, backgrounds, sides, depths = set(), set(), set(), set(), set()
    records = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_fingerprint(puzzle))
        assert _puzzle_fingerprint(puzzle) == _puzzle_fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                before = a.copy()
                assert np.array_equal(solve(a), out)
                assert np.array_equal(a, before)
                f = _features(a, out)
                records.append(f)
                shapes.add(a.shape)
                lengths.add(f["length"])
                source_counts.add(f["sources"])
                backgrounds.add(f["bg"])
                sides.add(f["side"])
                depths.add(f["max_depth"])
                roles = {}
                normalized.add((a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)))
                key, regions = _read_input(a)
                for r in regions:
                    if r["edge"]:
                        continue
                    rr, cc = zip(*r["cells"])
                    geometries.add(tuple(sorted((x-min(rr), y-min(cc)) for x, y in r["cells"])))
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(normalized) >= 180 and len(geometries) >= 70
    assert lengths == set(range(1, 7)) and source_counts == set(range(1, 6))
    assert backgrounds == set(range(10)) and sides == {"top", "bottom", "left", "right"}
    assert max(depths) >= 6 and len(depths) >= 5
    for name in ("nonflat", "repeat", "cap", "chain", "bbox_witness", "diagonal", "multi", "open"):
        assert sum(bool(f[name]) for f in records) >= 20

    # Hand-derived two-color code [4,5], 5x7 hole: shell 4, all deeper cells 5.
    a = np.zeros((13, 17), dtype=np.int8)
    a[:3, 2:4] = [[1, 2], [4, 6], [5, 7]]
    a[5:12, 1:10] = 1
    a[6:11, 2:9] = 0
    # Open listed ring and closed unlisted diamond must remain unchanged.
    a[5:9, 12:16] = 2
    a[6:8, 13:15] = 0
    a[5, 13] = 0
    for q in ((10, 13), (11, 12), (11, 14), (12, 13)):
        a[q] = 8
    expected = a.copy()
    expected[6:11, 2:9] = 4
    expected[7:10, 3:8] = 5
    assert np.array_equal(solve(a), expected)
    # Recoloring and rotation are semantic symmetries, not extra output rules.
    for turn in range(4):
        palette = np.roll(np.arange(10, dtype=np.int8), turn+2)
        assert np.array_equal(solve(palette[np.rot90(a, turn)]), palette[np.rot90(expected, turn)])
    # A diagonal diamond: four-neighbor cavity depth, not square-box depth.
    b = np.zeros((15, 16), dtype=np.int8)
    b[:4, 2:4] = [[1, 2], [3, 7], [4, 8], [5, 9]]
    rr, cc = np.indices((9, 9))
    b[5:14, 5:14][abs(rr-4)+abs(cc-4) == 4] = 1
    predicted = solve(b)
    for r in range(9):
        for c in range(9):
            distance = abs(r-4)+abs(c-4)
            if distance < 4:
                assert predicted[r+5, c+5] == (3, 4, 5)[min(3-distance, 2)]
    # Every legal explicit count combination and common noninteger mistakes.
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(1000+10*tr+te, tr, te)
            assert len(p["train"]) == tr and len(p["test"]) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(0)
    key, regions = _read_input(bad["test"][0]["input"])
    region = next(r for r in regions if not r["edge"] and r["owner"] in key["codes"])
    q = region["cells"][0]
    bad["test"][0]["output"][q] = key["bg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted band accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, API boundaries, topology and depth fixtures)")
