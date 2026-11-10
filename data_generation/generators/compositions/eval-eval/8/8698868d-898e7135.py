"""Counted hollow stamps: joint outline/count matching at a recovered scale.

No parent imports, file reads, caches of examples, or generation state in solve.
All diagnostic suites are bounded and keep only one puzzle at a time.
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _STEPS
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                other = (r + dr, c + dc)
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
        result.append(cells)
    return result


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _tight(mask):
    r, s, c, d = _box(np.argwhere(mask))
    return np.asarray(mask[r:s, c:d], dtype=bool).copy()


def _transforms(mask):
    return [np.rot90(mask, q).copy() for q in range(4)] + [
        np.rot90(np.fliplr(mask), q).copy() for q in range(4)]


def _mask_key(mask):
    return mask.shape, mask.tobytes()


def _canonical(mask):
    return min(_mask_key(m) for m in _transforms(mask))


def _cavities(material):
    h, w = material.shape
    result = []
    for cells in _components(~material):
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells):
            result.append(cells)
    return result


def _envelope(material):
    envelope = material.copy()
    holes = _cavities(material)
    for cells in holes:
        for cell in cells:
            envelope[cell] = True
    return envelope, holes


def _interior(mask):
    return [(r, c) for r, c in np.argwhere(mask)
            if all(0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1]
                   and mask[r + dr, c + dc] for dr, dc in _STEPS)]


def _compress(mask, scale):
    h, w = mask.shape
    if h % scale or w % scale:
        return None
    blocks = mask.reshape(h // scale, scale, w // scale, scale)
    low, high = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    return low.copy() if np.array_equal(low, high) else None


def _expand(grid, scale):
    return np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)


def _perimeter(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _check_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or not all(1 <= size <= 30 for size in grid.shape)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected an integer grid with sides 1..30 and colors 0..9")
    return grid.astype(np.int8, copy=False)


def _interpret_board(grid, background, wall, bounds):
    r0, r1, c0, c1 = bounds
    board = grid[r0:r1, c0:c1]
    if min(board.shape) < 5 or not np.all(_perimeter(board) == wall):
        return None
    others = set(map(int, np.unique(board))) - {background, wall}
    if len(others) > 1:
        return None
    marker = next(iter(others)) if others else None
    sockets = []
    for cells in _components(board != wall):
        a, b, c, d = _box(cells)
        mask = np.zeros((b - a, d - c), dtype=bool)
        for r, col in cells:
            mask[r - a, col - c] = True
        if (a == 0 or c == 0 or b == board.shape[0] or d == board.shape[1]
                or len(cells) < 6 or _cavities(mask)):
            return None
        dots = [(r - a, col - c) for r, col in cells
                if marker is not None and board[r, col] == marker]
        if len(dots) > 3 or not set(dots) <= set(_interior(mask)):
            return None
        if any(abs(r - x) + abs(c - y) == 1
               for i, (r, c) in enumerate(dots) for x, y in dots[i + 1:]):
            return None
        sockets.append({"mask": mask, "cells": cells, "origin": (a, c), "count": len(dots)})
    if not 2 <= len(sockets) <= 4:
        return None
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    # A full background moat separates object bounding boxes, including the board.
    if np.any(grid[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1]
              [outside[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1]] != background):
        return None
    loose = []
    for cells in _components((grid != background) & outside):
        colors = {int(grid[r, c]) for r, c in cells}
        if len(colors) != 1:
            return None
        color = next(iter(colors))
        if color == wall or color == marker:
            return None
        a, b, c, d = _box(cells)
        loose.append({"mask": grid[a:b, c:d] == color, "color": color, "bounds": (a, b, c, d)})
    if len(loose) != len(sockets):
        return None
    boxes = [bounds] + [p["bounds"] for p in loose]
    for i, (a, b, c, d) in enumerate(boxes):
        for e, f, g, h in boxes[i + 1:]:
            if not (b < e or f < a or d < g or h < c):
                return None
    return board, marker, sockets, loose


def _assemblies(board, wall, sockets, pieces):
    """Enumerate colored results, stopping as soon as two distinct answers exist."""
    options = []
    for socket in sockets:
        choices = []
        for j, piece in enumerate(pieces):
            if socket["count"] != piece["count"]:
                continue
            seen = set()
            for envelope, material in zip(_transforms(piece["envelope"]), _transforms(piece["material"])):
                if np.array_equal(envelope, socket["mask"]) and _mask_key(material) not in seen:
                    seen.add(_mask_key(material))
                    choices.append((j, material))
        if not choices:
            return []
        options.append(choices)
    results = {}
    order = sorted(range(len(sockets)), key=lambda i: len(options[i]))
    canvas = np.full(board.shape, wall, dtype=np.int8)
    selected = [None] * len(sockets)

    def visit(depth, used):
        if len(results) >= 2:
            return
        if depth == len(order):
            results.setdefault(canvas.tobytes(), (canvas.copy(), selected.copy()))
            return
        i = order[depth]
        a, c = sockets[i]["origin"]
        h, w = sockets[i]["mask"].shape
        for j, material in options[i]:
            if j in used:
                continue
            view = canvas[a:a + h, c:c + w]
            view[material] = pieces[j]["color"]
            selected[i] = (j, material)
            visit(depth + 1, used | {j})
            view[material] = wall
            if len(results) >= 2:
                return
    visit(0, set())
    return list(results.values())


def _recover(input_grid):
    grid = _check_grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= grid.size // 2:
        raise ValueError("canvas background must occupy a strict majority")
    solutions = []
    for wall in map(int, np.unique(grid)):
        if wall == background:
            continue
        for cells in _components(grid == wall):
            bounds = _box(cells)
            decoded = _interpret_board(grid, background, wall, bounds)
            if decoded is None:
                continue
            board, marker, sockets, loose = decoded
            for scale in (2, 3):
                if max(board.shape) * scale > 30:
                    continue
                pieces = []
                for item in loose:
                    material = _compress(item["mask"], scale)
                    if material is None or len(_components(material)) != 1:
                        break
                    envelope, holes = _envelope(material)
                    if len(holes) > 3:
                        break
                    pieces.append(dict(item, material=material, envelope=envelope,
                                       count=len(holes), hole_sizes=tuple(map(len, holes))))
                else:
                    assemblies = _assemblies(board, wall, sockets, pieces)
                    if len(assemblies) > 1:
                        raise ValueError("ambiguous stamp orientation or assignment")
                    if len(assemblies) == 1:
                        logical, selected = assemblies[0]
                        solutions.append((_expand(logical, scale), {
                            "background": background, "wall": wall, "marker": marker,
                            "board_bounds": bounds, "board_shape": board.shape,
                            "scale": scale, "sockets": sockets, "pieces": pieces,
                            "selected": selected, "logical": logical}))
    if len(solutions) != 1:
        raise ValueError("expected one board, scale and colored assembly")
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer solely from the current input."""
    return _recover(input_grid)[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _outline(rng, height, width, *, cut=None):
    """Boundary carving around a protected rectangular core, never a shape bank."""
    if height == width == 3:
        # A 3x3 outline with an interior pore cannot have a unique D4 frame.
        # Its only possible pore is central and respects every outline symmetry.
        mask = np.ones((3, 3), dtype=bool)
        if rng.random() < .7:
            mask[_pick(rng, [(0, 0), (0, 2), (2, 0), (2, 2)])] = False
        return mask
    for attempt in range(50):
        mask = np.ones((height, width), dtype=bool)
        if cut is not None:
            mask[cut] = False
        else:
            boundary = [(r, c) for r in range(height) for c in range(width)
                        if r in (0, height - 1) or c in (0, width - 1)]
            for j in rng.permutation(len(boundary))[:int(rng.integers(1, max(2, len(boundary) // 3)))]:
                mask[boundary[int(j)]] = False
        if (len(_components(mask)) == 1 and not _cavities(mask)
                and _tight(mask).shape == mask.shape and len(_interior(mask)) >= 1
                and len({_mask_key(m) for m in _transforms(mask)}) == 8):
            return mask
    # A corner removed from a nonsquare rectangle always has a unique orientation.
    mask = np.ones((height, width), dtype=bool)
    mask[0, 0] = False
    if height == width:
        mask[0, 1] = False
    return mask


def _make_pores(rng, outline, count, style=None):
    if count == 0:
        return np.zeros_like(outline)
    candidates = _interior(outline)
    for attempt in range(160):
        mask = np.zeros_like(outline)
        if count > len(candidates):
            return None
        for j in rng.choice(len(candidates), count, replace=False):
            mask[candidates[int(j)]] = True
        if len(_components(mask)) != count:
            continue
        if style == "diagonal" and len(_components(mask, True)) != 1:
            continue
        wanted = 3 if style == "large" else count if style == "diagonal" else int(rng.integers(count, len(candidates) + 1))
        for j in rng.permutation(len(candidates)):
            if int(mask.sum()) >= wanted:
                break
            p = candidates[int(j)]
            if mask[p]:
                continue
            trial = mask.copy()
            trial[p] = True
            if len(_components(trial)) == count and len(_components(outline & ~trial)) == 1:
                mask = trial
        if style == "diagonal" and int(mask.sum()) != count:
            # The connectivity witness consists of two diagonal single cells.
            continue
        if style == "large" and int(mask.sum()) < 3:
            continue
        material = outline & ~mask
        recovered, holes = _envelope(material)
        if (len(_components(material)) == 1 and len(holes) == count
                and np.array_equal(recovered, outline)):
            return mask
    return None


def _clues(rng, outline, count):
    candidates = _interior(outline)
    if count == 0:
        return []
    eligible = [chosen for chosen in itertools.combinations(candidates, count)
                if all(abs(r - x) + abs(c - y) != 1
                       for i, (r, c) in enumerate(chosen) for x, y in chosen[i + 1:])]
    return list(_pick(rng, eligible)) if eligible else None


def _place_sockets(rng, outlines, scale):
    limit = 30 // scale
    for attempt in range(150):
        # A single blueprint with independently displaced cavities, not tile cells.
        bh = int(rng.integers(max(7, limit - 4), limit + 1))
        bw = int(rng.integers(max(7, limit - 4), limit + 1))
        placements = []
        boxes = []
        for outline in outlines:
            transform = int(rng.integers(8))
            mask = _transforms(outline)[transform]
            h, w = mask.shape
            origins = [(r, c) for r in range(1, bh - h) for c in range(1, bw - w)
                       if all(r + h < a or b < r or c + w < x or y < c
                              for a, b, x, y in boxes)]
            if not origins:
                break
            r, c = _pick(rng, origins)
            boxes.append((r, r + h, c, c + w))
            placements.append((r, c, transform, mask))
        else:
            return (bh, bw), placements
    return None


def _free_origins(blocked, height, width):
    if height > len(blocked) or width > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = summed[height:, width:] - summed[:-height, width:] - summed[height:, :-width] + summed[:-height, :-width]
    return np.argwhere(totals == 0)


def _pack(rng, objects, background):
    for attempt in range(60):
        h = int(rng.integers(24, 31)) if attempt < 20 else 30
        w = int(rng.integers(24, 31)) if attempt < 20 else 30
        canvas = np.full((h, w), background, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        # Size-first order is randomized on ties; all two-dimensional origins are legal.
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(rng.random())))
        for i in order:
            obj = objects[i]
            oh, ow = obj.shape
            origins = _free_origins(blocked, oh, ow)
            if not len(origins):
                break
            r, c = map(int, origins[int(rng.integers(len(origins)))])
            canvas[r:r + oh, c:c + ow] = obj
            blocked[max(0, r - 1):r + oh + 1, max(0, c - 1):c + ow + 1] = True
        else:
            if np.count_nonzero(canvas == background) <= canvas.size // 2:
                continue
            rr, ss, cc, dd = _box(np.argwhere(canvas != background))
            rr = max(0, rr - int(rng.integers(4)))
            cc = max(0, cc - int(rng.integers(4)))
            ss = min(h, ss + int(rng.integers(4)))
            dd = min(w, dd + int(rng.integers(4)))
            crop = canvas[rr:ss, cc:dd]
            if rng.random() < .75 and np.count_nonzero(crop == background) > crop.size // 2:
                canvas = crop.copy()
            return canvas
    return None


def _make_pair(rng, scale, count, *, witness=False, novel_topology=False, teach_zero=False):
    for attempt in range(100):
        palette = list(map(int, rng.permutation(10)))
        background, wall, marker = palette[:3]
        if witness:
            common = _outline(rng, 4, 5, cut=(0, 0))
            other = _outline(rng, 4, 5, cut=(0, 1))
            outlines = [common, common.copy(), other]
            counts = [1, 2, 1]
        elif novel_topology:
            common = _outline(rng, 5, 5)
            other = _outline(rng, 4, 5)
            outlines = [common, common.copy(), other]
            counts = [1, 3, 1]
        else:
            if scale == 3:
                common = _outline(rng, 3, 4)
                other = _outline(rng, 3, 3)
                codes = [0, 1]
            else:
                hh, ww = (4, 5) if count == 4 else _pick(rng, [(4, 5), (4, 6), (5, 5)])
                common = _outline(rng, hh, ww)
                other = _outline(rng, 4, 4 if count == 4 else 5)
                codes = [0, 1] if teach_zero else list(map(int, rng.choice([0, 1, 2], 2, replace=False)))
            if _canonical(common) == _canonical(other):
                continue
            outlines = [common, common.copy()] + ([other] if count >= 3 else [])
            counts = codes + ([codes[0]] if count >= 3 else [])
            if count == 4:
                outlines.append(other.copy())
                counts.append(codes[1])
        pores = []
        for i, (outline, code) in enumerate(zip(outlines, counts)):
            style = "large" if witness and i == 0 else "diagonal" if witness and i == 1 else None
            mask = _make_pores(rng, outline, code, style)
            if mask is None:
                break
            pores.append(mask)
        if len(pores) != len(outlines):
            continue
        packed = _place_sockets(rng, outlines, scale)
        if packed is None:
            continue
        board_shape, placements = packed
        board = np.full(board_shape, wall, dtype=np.int8)
        logical = board.copy()
        loose = []
        colors = palette[3:3 + len(outlines)]
        # Distinct witness colors expose the two matching constraints. Other
        # scenes can share colors because their joint keys still distinguish stamps.
        if not witness and rng.random() < .4:
            size = int(rng.integers(1, len(outlines)))
            colors = [palette[3 + int(rng.integers(size))] for _ in outlines]
        for i, (outline, pore, code, color, placement) in enumerate(zip(outlines, pores, counts, colors, placements)):
            r, c, transform, target = placement
            h, w = target.shape
            view = board[r:r + h, c:c + w]
            view[target] = background
            dots = _clues(rng, target, code)
            if dots is None:
                break
            for dr, dc in dots:
                view[dr, dc] = marker
            material = outline & ~pore
            target_material = _transforms(material)[transform]
            logical[r:r + h, c:c + w][target_material] = color
            orientation = int(rng.integers(8))
            if witness and i == 0:
                # Opposite reflection parity guarantees a real chiral witness.
                orientation = _pick(rng, list(range(4, 8)) if transform < 4 else list(range(4)))
            elif witness and i == 1:
                orientation = (transform // 4) * 4 + (transform + 1) % 4
            source_material = _transforms(material)[orientation]
            obj = np.where(source_material, color, background).astype(np.int8)
            loose.append(_expand(obj, scale))
        if len(loose) != len(outlines):
            continue
        canvas = _pack(rng, [board] + loose, background)
        if canvas is None:
            continue
        output = _expand(logical, scale)
        try:
            reconstructed, info = _recover(canvas)
        except ValueError:
            continue
        if not np.array_equal(output, reconstructed):
            raise AssertionError("construction and input-derived assembly disagree")
        if witness and not all(_features(info)[key] for key in
                               ("count_witness", "shape_witness", "area_inversion", "diagonal_witness", "reflection", "rotation")):
            continue
        return {"input": canvas, "output": output}
    raise RuntimeError("bounded stamp construction exhausted")


def _features(info):
    sockets, pieces = info["sockets"], info["pieces"]
    count_witness = shape_witness = False
    for i, a in enumerate(sockets):
        for j, b in enumerate(sockets[i + 1:], i + 1):
            same_outline = _canonical(a["mask"]) == _canonical(b["mask"])
            pa = pieces[info["selected"][i][0]]
            pb = pieces[info["selected"][j][0]]
            count_witness |= same_outline and a["count"] != b["count"] and pa["color"] != pb["color"]
            shape_witness |= (not same_outline and a["count"] == b["count"]
                              and int(a["mask"].sum()) == int(b["mask"].sum())
                              and pa["color"] != pb["color"])
    reflection = rotation = identity = 0
    for socket, (j, _) in zip(sockets, info["selected"]):
        shape = pieces[j]["envelope"]
        if np.array_equal(shape, socket["mask"]):
            identity += 1
        elif any(np.array_equal(m, socket["mask"]) for m in _transforms(shape)[:4]):
            rotation += 1
        else:
            reflection += 1
    areas = [(p["count"], sum(p["hole_sizes"])) for p in pieces]
    return {
        "count_witness": bool(count_witness), "shape_witness": bool(shape_witness),
        "area_inversion": any(a < b and x > y for a, x in areas for b, y in areas),
        "diagonal_witness": any(len(_components(p["envelope"] & ~p["material"], True)) < p["count"] for p in pieces),
        "multicell": any(max(p["hole_sizes"], default=0) > 1 for p in pieces),
        "reflection": reflection, "rotation": rotation, "identity": identity,
        "piece_count": len(pieces), "scale": info["scale"],
        "counts": tuple(sorted(s["count"] for s in sockets)),
        "outline_keys": tuple(sorted({_canonical(s["mask"]) for s in sockets})),
    }


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid_key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))  # Unseen scale, larger assembly, or new cavity count.
    puzzle, seen, train_scales = {"train": [], "test": []}, set(), []
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            witness = split == "train" and index == 0
            teach_zero = split == "train" and index == 1
            scale = 2 if witness or shift in (0, 2) else int(rng.choice([2, 3]))
            count = 3 if witness or teach_zero else int(rng.choice([2, 3]))
            novel_topology = split == "test" and shift == 2
            if split == "test":
                if shift == 0:
                    scale, count = 3, 3
                elif shift == 1:
                    scale, count = int(rng.choice(train_scales)), 4
                else:
                    scale, count = 2, 3
            for attempt in range(12):
                pair = _make_pair(rng, scale, count, witness=witness,
                                  novel_topology=novel_topology, teach_zero=teach_zero)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_scales.append(scale)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            key = _grid_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            expected, info = _recover(inp)
            assert np.array_equal(out, expected), "wrong assembly, pore substitution, enlargement or crop"
            assert not ({info["background"], info["marker"]} & set(map(int, np.unique(out))))
            scale = info["scale"]
            logical = out[::scale, ::scale]
            assert np.array_equal(out, _expand(logical, scale)), "broken output macrocells"
            for socket, (j, material) in zip(info["sockets"], info["selected"]):
                assert socket["count"] == info["pieces"][j]["count"]
                assert np.array_equal(_envelope(material)[0], socket["mask"])
                a, c = socket["origin"]
                for r, col in np.argwhere(socket["mask"] & ~material):
                    assert logical[a + r, c + col] == info["wall"], "perforation must become board material"
            if split == "train":
                training.append(_features(info))
    for key in ("count_witness", "shape_witness", "area_inversion", "diagonal_witness", "multicell", "reflection", "rotation"):
        assert any(row[key] for row in training), f"training lacks {key} evidence"
    assert any(0 in row["counts"] for row in training), "zero-hole convention is not demonstrated"


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                array = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(array.shape))
                digest.update(array.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streaming seeds, genuine family features, ablations and corruptions."""
    digests, shapes, boards, outlines, schedules = set(), set(), set(), set(), set()
    scales, counts, cavity_sizes = set(), set(), set()
    traits = {k: 0 for k in ("count_witness", "shape_witness", "area_inversion", "diagonal_witness", "reflection", "rotation", "identity")}
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        per_split = []
        for split in ("train", "test"):
            local = set()
            for pair in puzzle[split]:
                expected, info = _recover(pair["input"])
                assert np.array_equal(expected, pair["output"])
                f = _features(info)
                total += 1
                local.add((f["scale"], f["piece_count"], f["counts"]))
                shapes.add(pair["input"].shape)
                boards.add(info["board_shape"])
                outlines.update(f["outline_keys"])
                scales.add(f["scale"])
                counts.add(f["piece_count"])
                cavity_sizes.update(n for p in info["pieces"] for n in p["hole_sizes"])
                for k in traits:
                    traits[k] += int(bool(f[k]))
            per_split.append(tuple(sorted(local)))
        schedules.add(tuple(per_split))
        if seed == 0:
            assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(0))
            sample = puzzle
    assert len(digests) == 42 and len(shapes) >= 20 and len(boards) >= 15
    assert len(outlines) >= 35 and scales == {2, 3} and counts == {2, 3, 4}
    assert len(schedules) >= 20 and max(cavity_sizes) >= 3
    assert all(traits[k] > 0 for k in traits)
    # Full range of explicit counts, and representative type/range failures.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(42 + nt * 5 + ne, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Removing either matching constraint leaves a demonstrably wrong color choice.
    witness = next(p for p in sample["train"] if _features(_recover(p["input"])[1])["shape_witness"])
    answer, info = _recover(witness["input"])
    for omit in ("count", "outline"):
        found = False
        for i, a in enumerate(info["sockets"]):
            for j, b in enumerate(info["sockets"]):
                if i >= j:
                    continue
                same = _canonical(a["mask"]) == _canonical(b["mask"])
                if (omit == "count" and same and a["count"] != b["count"]) or (
                        omit == "outline" and not same and a["count"] == b["count"]):
                    pa = info["pieces"][info["selected"][i][0]]
                    pb = info["pieces"][info["selected"][j][0]]
                    assert pa["color"] != pb["color"]
                    wrong = answer.copy()
                    ma, mb = wrong == pa["color"], wrong == pb["color"]
                    wrong[ma], wrong[mb] = pb["color"], pa["color"]
                    assert not np.array_equal(answer, wrong)
                    found = True
        assert found
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    # A single missing pixel breaks the uniform block encoding of a genuine stamp.
    damaged = witness["input"].copy()
    a, b, c, d = info["pieces"][0]["bounds"]
    pos = np.argwhere(damaged[a:b, c:d] == info["pieces"][0]["color"])[0]
    damaged[a + int(pos[0]), c + int(pos[1])] = info["background"]
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged macrocell accepted")
    damaged = witness["input"].copy()
    dot = np.argwhere(damaged == info["marker"])[0]
    damaged[tuple(dot)] = info["background"]
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong socket count accepted")
    # Symmetric envelopes are safe only when their material gives one result.
    rectangle = np.ones((4, 5), dtype=bool)
    off_center = rectangle.copy()
    off_center[1, 1] = False
    dummy = np.full((7, 8), 2, dtype=np.int8)
    socket = {"mask": rectangle, "count": 1, "origin": (1, 1)}
    piece = {"envelope": rectangle, "material": off_center, "count": 1, "color": 3}
    assert len(_assemblies(dummy, 2, [socket], [piece])) == 2
    centered = np.ones((3, 3), dtype=bool)
    material = centered.copy()
    material[1, 1] = False
    socket = {"mask": centered, "count": 1, "origin": (1, 1)}
    piece = {"envelope": centered, "material": material, "count": 1, "color": 3}
    assert len(_assemblies(dummy, 2, [socket], [piece])) == 1
    return {"seeds": 42, "pairs": total, "distinct_puzzles": len(digests),
            "input_shapes": len(shapes), "board_shapes": len(boards),
            "outline_classes": len(outlines), "schedules": len(schedules),
            "scales": sorted(scales), "piece_counts": sorted(counts),
            "cavity_sizes": sorted(cavity_sizes), "traits": traits,
            "semantic_check": "generator-owned; no independent or blind solver"}
