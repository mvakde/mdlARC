"""Fractured sheets settle into a shared shaft in source-compartment order.

Only NumPy and the standard library are used.  No parent code or data is loaded.
"""
from __future__ import annotations

from collections import Counter
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _parse(input_grid):
    """Infer every role and all geometry from this input alone."""
    a = _grid(input_grid)
    height, full_width = a.shape
    bg = int(a[0, 0])
    bars = [c for c in range(full_width)
            if int(a[0, c]) != bg and np.all(a[:, c] == a[0, c])]
    _require(bool(bars), "no compartment dividers")
    width = bars[0]
    count = len(bars) + 1
    _require(count >= 2 and width >= 5 and full_width == count * width + count - 1,
             "compartments must have a common width")
    _require(bars == [width + k * (width + 1) for k in range(count - 1)],
             "irregular dividers")
    divider = int(a[0, bars[0]])
    _require(all(np.all(a[:, c] == divider) for c in bars), "divider colors disagree")
    rectangles = []
    candidate_cuts = None
    for index in range(count):
        panel = a[:, index * (width + 1):index * (width + 1) + width]
        _require(not np.any(panel == divider), "divider used inside a compartment")
        points = np.argwhere(panel != bg)
        _require(len(points) > 0, "empty compartment")
        top, left = (int(v) for v in points.min(axis=0))
        bottom, right = (int(v) + 1 for v in points.max(axis=0))
        h, w = bottom - top, right - left
        _require(h >= 2 and w >= 3 and 1 <= left and right < width,
                 "sheet needs two sides and horizontal movement clearance")
        _require(len(points) == h * w, "sheet must fill its rectangle")
        block = panel[top:bottom, left:right]
        colors = {int(c) for c in np.unique(block)}
        _require(len(colors) == 2, "sheet needs one material and one cut color")
        choices = set()
        for color in colors:
            mask = block == color
            if np.all(mask.sum(axis=1) == 1):
                path = np.argmax(mask, axis=1)
                if np.all((path >= 1) & (path <= w - 2)) and np.all(np.abs(np.diff(path)) <= 1):
                    choices.add(color)
        _require(len(choices) == 1, "sheet cut is not uniquely identifiable")
        candidate_cuts = choices if candidate_cuts is None else candidate_cuts & choices
        cut = next(iter(choices))
        path = tuple(int(c) for c in np.argmax(block == cut, axis=1))
        rectangles.append(dict(top=top, left=left, h=h, w=w, path=path,
                               material=next(iter(colors - {cut}))))
    _require(candidate_cuts is not None and len(candidate_cuts) == 1, "cuts must share one color")
    cut = next(iter(candidate_cuts))
    _require(all(o["material"] not in (bg, divider, cut) for o in rectangles), "role collision")
    _require(sum(o["h"] for o in rectangles) <= height, "insufficient shaft height")
    return dict(height=height, width=width, count=count, bg=bg, divider=divider,
                cut=cut, rectangles=rectangles)


def _fragments(obj, distance=1, flip_rows=False):
    """Coordinates use the source band origin, with its source y-offset removed."""
    pieces = [[], []]
    for r, p in enumerate(obj["path"]):
        rr = obj["h"] - 1 - r if flip_rows else r
        for c in range(obj["w"]):
            if c == p:
                continue
            side = int(c > p)
            cc = obj["left"] + c + (distance if side else -distance)
            pieces[side].append((rr, cc, obj["material"]))
    return pieces


def _drop(pieces, height, width, occupied, output, start):
    """Synchronous rigid gravity; a stopped piece becomes an obstacle immediately.

    Moving pieces share a velocity, so they cannot collide with one another while
    both move.  If one stops this tick, any piece whose next step would hit its
    current footprint also stops.  This is simultaneous, with no left/right tie.
    """
    offsets = [start] * len(pieces)
    active = set(range(len(pieces)))
    for _ in range(height - start + 2):
        if not active:
            break
        current = {i: {(r + offsets[i], c) for r, c, _ in pieces[i]} for i in active}
        proposed = {i: {(r + 1, c) for r, c in current[i]} for i in active}
        blocked = {i for i in active if any(r >= height or not 0 <= c < width
                                          for r, c in proposed[i])
                   or bool(proposed[i] & occupied)}
        while True:
            stopped_cells = set().union(*(current[i] for i in blocked)) if blocked else set()
            more = {i for i in active - blocked if proposed[i] & stopped_cells}
            if not more:
                break
            blocked |= more
        # All footprints are computed before any board mutation.
        for i in sorted(blocked):
            for r, c, color in pieces[i]:
                rr = r + offsets[i]
                _require(0 <= rr < height and 0 <= c < width, "settled outside shaft")
                _require((rr, c) not in occupied, "settled pieces overlap")
                occupied.add((rr, c))
                output[rr, c] = color
        for i in active - blocked:
            offsets[i] += 1
        active -= blocked
    _require(not active, "gravity failed to terminate")
    return tuple(offsets)


def _assemble(info, *, distance=1, reverse_feed=False, linked=False, flip_rows=False):
    h, w, bg = info["height"], info["width"], info["bg"]
    output = np.full((h, w), bg, dtype=np.int8)
    occupied = set()
    trace = [None] * info["count"]
    indices = list(range(info["count"]))
    if reverse_feed:
        indices.reverse()
    for index in indices:
        obj = info["rectangles"][index]
        pieces = _fragments(obj, distance, flip_rows)
        if linked:
            pieces = [pieces[0] + pieces[1]]
        trace[index] = _drop(pieces, h, w, occupied, output, -obj["h"])
    return output, trace


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver for the entire specified family, independent of sampling."""
    return _assemble(_parse(input_grid))[0]


def _band_stack(info):
    """Rival: fracture, but then use A's ordinary full-band concatenation."""
    out = np.full((info["height"], info["width"]), info["bg"], dtype=np.int8)
    y = info["height"]
    for obj in info["rectangles"]:
        y -= obj["h"]
        for piece in _fragments(obj):
            for r, c, color in piece:
                out[y + r, c] = color
    return out


def _pixel_gravity(info):
    """Rival: all columns compact independently, destroying rigid shapes."""
    out = np.full((info["height"], info["width"]), info["bg"], dtype=np.int8)
    tops = [info["height"]] * info["width"]
    for obj in info["rectangles"]:
        by_col = {}
        for piece in _fragments(obj):
            for r, c, color in piece:
                by_col.setdefault(c, []).append((r, color))
        for c, cells in by_col.items():
            cells.sort()
            tops[c] -= len(cells)
            for offset, (_, color) in enumerate(cells):
                out[tops[c] + offset, c] = color
    return out


def _rivals(info):
    return {
        "reverse_feed": _assemble(info, reverse_feed=True)[0],
        "unbroken_bands": _band_stack(info),
        "linked_halves": _assemble(info, linked=True)[0],
        "no_outward_step": _assemble(info, distance=0)[0],
        "pixel_gravity": _pixel_gravity(info),
        "flip_internal_rows": _assemble(info, flip_rows=True)[0],
    }


def _path(rng, h, w, force_jagged=False):
    high = w - 2
    mode = int(rng.integers(4))
    pos = int(rng.integers(1, high + 1))
    direction = int(rng.choice([-1, 1]))
    path = [pos]
    for r in range(1, h):
        if mode == 0:
            step = 0
        elif mode == 1:
            step = direction if r % 2 else -direction
        elif mode == 2:
            step = int(rng.integers(-1, 2))
        else:
            if not 1 <= pos + direction <= high or rng.random() < 0.2:
                direction *= -1
            step = direction if rng.random() < 0.8 else 0
        pos = max(1, min(high, pos + step))
        path.append(pos)
    if force_jagged and high >= 2 and len(set(path)) == 1:
        path[-1] += 1 if path[-1] < high else -1
    return tuple(path)


def _sample_pair(rng, split, index, avoid_roles=None):
    witness = split == "train" and index == 0
    second = split == "train" and index == 1
    challenge = split == "test" and index == 0
    n = 2 if witness else (3 if second else int(rng.integers(2, 5 if challenge else 6)))
    max_width = (31 - n) // n
    min_width = 8 if witness else (6 if second or challenge else 5)
    width = int(rng.integers(min_width, max_width + 1))
    heights = [int(rng.integers(3 if witness else 2, 6)) for _ in range(n)]
    if split == "test":
        tall = int(rng.integers(n))
        heights[tall] = int(rng.integers(6, 10))
        for k in range(n):
            if k != tall:
                heights[k] = int(rng.integers(2, min(7, 30 - sum(heights) + heights[k]) + 1))
    total = sum(heights)
    padding = int(rng.integers(0, min(8, 30 - total) + 1))
    height = total + padding
    roles = [int(c) for c in rng.permutation(10)]
    if avoid_roles is not None:
        # Pair-local roles change visibly in the second demonstration.
        selected, available = [], list(range(10))
        for previous in avoid_roles:
            selected.append(int(rng.choice([c for c in available if c != previous])))
            available.remove(selected[-1])
        roles = selected + [int(c) for c in rng.permutation(available)]
    bg, divider, cut = roles[:3]
    if witness:
        materials = roles[3:3 + n]
    elif second:
        materials = [roles[3], roles[3], roles[4]]
    else:
        palette = roles[3:3 + int(rng.integers(1, min(n, 4) + 1))]
        materials = [int(rng.choice(palette)) for _ in range(n)]
    placement = int(rng.integers(4))
    shared_top = int(rng.integers(height - max(heights) + 1))
    inp = np.full((height, n * width + n - 1), bg, dtype=np.int8)
    for i, h in enumerate(heights):
        if i < n - 1:
            inp[:, i * (width + 1) + width] = divider
        w = int(rng.integers(4 if witness else 3, width - 1))
        left = int(rng.integers(1, width - w))
        top = (0, height - h, shared_top, int(rng.integers(height - h + 1)))[placement]
        path = _path(rng, h, w, force_jagged=witness or second)
        block = np.full((h, w), materials[i], dtype=np.int8)
        for r, p in enumerate(path):
            block[r, p] = cut
        col = i * (width + 1) + left
        inp[top:top + h, col:col + w] = block
    info = _parse(inp)
    out, trace = _assemble(info)
    if witness:
        # This single witness handles even the minimum 2/1 count request.
        if not any(t[0] != t[1] for t in trace):
            return None
        if any(np.array_equal(out, rival) for rival in _rivals(info).values()):
            return None
    if challenge:
        packed_height = height - int(np.flatnonzero(np.any(out != bg, axis=1))[0])
        if not (any(t[0] != t[1] for t in trace) and packed_height < total
                and any(len(set(o["path"])) > 1 for o in info["rectangles"])):
            return None
        if np.array_equal(out, _assemble(info, reverse_feed=True)[0]):
            return None
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(1000):
                pair = _sample_pair(rng, split, index, first_roles if split == "train" and index == 1 else None)
                if pair is not None and _signature(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not construct a distinct teaching example")
            puzzle[split].append(pair)
            seen.add(_signature(pair["input"]))
            if split == "train" and index == 0:
                info = _parse(pair["input"])
                first_roles = info["bg"], info["divider"], info["cut"]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Pixel-derived geometry, conservation, stability, witnesses, and shift checks."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "invalid counts")
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _parse(inp)
            expected, trace = _assemble(info)
            _require(out.shape == (info["height"], info["width"]) and np.array_equal(out, expected),
                     "incorrect rigid falling output")
            material_counts = Counter({})
            final_pieces = []
            for obj, offsets in zip(info["rectangles"], trace):
                material_counts[obj["material"]] += obj["h"] * (obj["w"] - 1)
                for piece, dy in zip(_fragments(obj), offsets):
                    cells = {(r + dy, c) for r, c, _ in piece}
                    _require(all(int(out[r, c]) == obj["material"] for r, c in cells), "fragment lost")
                    final_pieces.append(cells)
            _require(Counter(int(c) for c in out.flat if int(c) != info["bg"]) == material_counts,
                     "material colors or cell counts changed")
            occupied = set().union(*final_pieces)
            _require(sum(len(cells) for cells in final_pieces) == len(occupied), "fragments overlap")
            _require(any(r == out.shape[0] - 1 for r, _ in occupied), "nothing rests on the floor")
            for cells in final_pieces:
                down = {(r + 1, c) for r, c in cells}
                _require(any(r >= out.shape[0] for r, _ in down) or bool(down & (occupied - cells)),
                         "a final rigid fragment can still fall")
            _require(not np.any(out == info["cut"]) and not np.any(out == info["divider"]), "markers survived")
            hs = [o["h"] for o in info["rectangles"]]
            if split == "train":
                _require(max(hs) <= 5, "training height regime violated")
            else:
                _require(6 <= max(hs) <= 9, "test height shift absent")
            infos[split].append(info)
    first = infos["train"][0]
    first_out, first_trace = _assemble(first)
    _require(any(t[0] != t[1] for t in first_trace), "missing independently falling halves witness")
    for name, rival in _rivals(first).items():
        _require(not np.array_equal(first_out, rival), f"training ambiguity: {name}")
    second = infos["train"][1]
    _require(first["count"] == 2 and second["count"] == 3, "count variation witness absent")
    _require(all(first[k] != second[k] for k in ("bg", "divider", "cut")),
             "local color role witness absent")
    _require(len({o["material"] for o in second["rectangles"]}) < second["count"],
             "repeated material witness absent")
    challenge = infos["test"][0]
    challenge_out, challenge_trace = _assemble(challenge)
    occupied_rows = np.flatnonzero(np.any(challenge_out != challenge["bg"], axis=1))
    _require(any(t[0] != t[1] for t in challenge_trace) and
             challenge["height"] - int(occupied_rows[0]) < sum(o["h"] for o in challenge["rectangles"]),
             "test must combine height extrapolation with contour nesting")
    _require(any(len(set(o["path"])) > 1 for o in challenge["rectangles"]), "test lacks a jagged seam")
    _require(not np.array_equal(challenge_out, _assemble(challenge, reverse_feed=True)[0]),
             "test does not exercise feed order")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed structural variation and semantic checks; returns metrics."""
    # Hand-worked support example: the later left half stops at y=4, while
    # its right half stops at y=3.  The older left half has an empty cell below
    # its overhang, which gravity must preserve rather than compact by column.
    inp = np.array([[int(c) for c in row] for row in (
        "02322000900000000", "02232000900000000", "02322000900000000",
        "00000000900000000", "00000000900000000", "00000000900443400",
        "00000000900434400", "00000000900000000")], dtype=np.int8)
    expected = np.array([[int(c) for c in row] for row in (
        "00000000", "00000000", "00000000", "00000040",
        "04400440", "24002200", "22000200", "20002200")], dtype=np.int8)
    _require(np.array_equal(solve(inp), expected), "hand-worked support regression failed")
    _require(_assemble(_parse(inp))[1] == [(5, 5), (4, 3)], "incorrect settling translations")
    _require(np.array_equal(solve(9 - inp), 9 - expected), "color-role equivariance failed")
    relocated = np.zeros_like(inp)
    relocated[:, 8] = 9
    relocated[-3:, 1:5] = inp[:3, 1:5]
    relocated[:2, 11:15] = inp[5:7, 11:15]
    _require(np.array_equal(solve(relocated), expected), "source vertical offsets leaked into result")
    extended = np.concatenate([np.zeros((2, inp.shape[1]), dtype=np.int8), inp])
    extended[:2, 8] = 9
    _require(np.array_equal(solve(extended), np.concatenate([
        np.zeros((2, expected.shape[1]), dtype=np.int8), expected])), "floor alignment changed with padding")
    shapes, geometries, paths, widths, counts, backgrounds = set(), set(), set(), set(), set(), set()
    palettes, source_positions, puzzles = set(), set(), set()
    staggered = wandering = nested = repeated = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                out, trace = _assemble(info)
                _require(np.array_equal(solve(pair["input"].tolist()), out), "list input solve mismatch")
                objects = info["rectangles"]
                shapes.add(pair["input"].shape)
                widths.add(info["width"])
                counts.add(info["count"])
                backgrounds.add(info["bg"])
                palettes.add((info["bg"], info["divider"], info["cut"], tuple(o["material"] for o in objects)))
                geometries.add((info["height"], info["width"], tuple(
                    (o["left"], o["h"], o["w"], o["path"]) for o in objects)))
                source_positions.add(tuple((o["top"], o["left"]) for o in objects))
                paths.update(o["path"] for o in objects)
                staggered += any(t[0] != t[1] for t in trace)
                wandering += any(len(set(o["path"])) >= 3 for o in objects)
                repeated += len({o["material"] for o in objects}) < len(objects)
                active_rows = np.flatnonzero(np.any(out != info["bg"], axis=1))
                nested += info["height"] - int(active_rows[0]) < sum(o["h"] for o in objects)
    _require(len(puzzles) == 42 and len(geometries) >= 170, "insufficient geometric diversity")
    _require(len(shapes) >= 55 and len(paths) >= 100 and len(source_positions) >= 100, "narrow layout/path sampling")
    _require(counts == {2, 3, 4, 5} and len(widths) >= 8 and len(backgrounds) == 10, "missing role/count/width variation")
    _require(staggered >= 42 and nested >= 42 and repeated >= 42 and wandering >= 10, "missing fracture interactions")
    _require(_equal(generate_puzzle(123, 7, 3), generate_puzzle(123, 7, 3)), "nondeterministic generation")
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(1000 + tr * 10 + te, tr, te)
            _require(len(p["train"]) == tr and len(p["test"]) == te, "explicit counts ignored")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4, False), (4.0, 1), (None, 1), (4, None), ("4", 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {tr}/{te}")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(puzzles), input_shapes=len(shapes),
                geometric_layouts=len(geometries), cut_paths=len(paths), widths=sorted(widths),
                panel_counts=sorted(counts), palettes=len(palettes), staggered_pairs=staggered,
                nested_pairs=nested, repeated_material_pairs=repeated, wandering_pairs=wandering)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
