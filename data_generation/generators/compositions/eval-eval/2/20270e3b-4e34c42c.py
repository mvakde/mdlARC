"""Keyed zippers: repeated colored words register crossing temporary rails.

Only NumPy and the standard library are used. Nothing is read from disk, and
solve uses its one input only. All construction retries and diagnostics are bounded.
"""
from collections import Counter
import hashlib
import itertools

import numpy as np


MARKER = 5
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def _groups(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            p = stack.pop()
            for d in DIRECTIONS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    group.add(q)
        result.append(group)
    return result


def _bounds(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points), max(c for r, c in points))


def _normalise(cells):
    r, c, _, _ = _bounds(cells)
    return {(y-r, x-c): v for (y, x), v in cells.items()}


def _render(cells, background):
    top, left, bottom, right = _bounds(cells)
    result = np.full((bottom-top+1, right-left+1), background, dtype=np.int8)
    for (r, c), value in cells.items():
        result[r-top, c-left] = value
    return result


def _checked_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return grid


def _parse(value):
    """Recover components, ports, words and normals solely from input pixels."""
    grid = _checked_grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    if background == MARKER or int(counts.max()) * 2 <= grid.size:
        raise ValueError("background must be a strict non-gray majority")
    cells = {tuple(map(int, p)): int(grid[tuple(p)])
             for p in np.argwhere(grid != background)}
    components = _groups(cells)
    if len(components) < 2:
        raise ValueError("need separated fragments")
    owners = {p: i for i, part in enumerate(components) for p in part}
    ports = []
    for bar in _groups(p for p, v in cells.items() if v == MARKER):
        top, left, bottom, right = _bounds(bar)
        if len(bar) < 2 or (top != bottom and left != right):
            raise ValueError("gray ports must be straight non-singleton rails")
        tangent = (0, 1) if top == bottom else (1, 0)
        origin = (top, left)
        if len(bar) != (right-left+1 if top == bottom else bottom-top+1):
            raise ValueError("broken rail")
        choices = []
        normals = ((-1, 0), (1, 0)) if top == bottom else ((0, -1), (0, 1))
        for normal in normals:
            # normal points OUT of the body, from its word to its guide.
            backing, front = [], []
            for k in range(len(bar)):
                p = _add(origin, (k*tangent[0], k*tangent[1]))
                backing.append(cells.get(_sub(p, normal), background))
                front.append(cells.get(_add(p, normal), background))
            occupied = [k for k, v in enumerate(backing) if v != background]
            if not occupied or any(v != background for v in front):
                continue
            lo, hi = min(occupied), max(occupied)
            word = tuple(backing[lo:hi+1])
            if (occupied != list(range(lo, hi+1)) or len(word) < 2
                    or len(set(word)) < 2 or MARKER in word):
                continue
            start = _sub(_add(origin, (lo*tangent[0], lo*tangent[1])), normal)
            choices.append({"owner": owners[origin], "normal": normal,
                            "start": start, "rail": origin, "length": len(bar),
                            "offset": lo, "word": word})
        if len(choices) != 1:
            raise ValueError("rail needs exactly one visible, contiguous backing word")
        ports.append(choices[0])
    by_word = {}
    for port in ports:
        # Horizontal and vertical words are separate oriented features. Whole
        # scene rotations may reverse one axis's reading order but not the other.
        signature = (bool(port["normal"][0]), port["word"])
        by_word.setdefault(signature, []).append(port)
    edges = []
    for signature, pair in by_word.items():
        if len(pair) != 2:
            raise ValueError("each oriented word must occur on exactly two rails")
        a, b = pair
        if a["owner"] == b["owner"] or a["normal"] != tuple(-x for x in b["normal"]):
            raise ValueError("mates must belong to different fragments and face oppositely")
        edges.append((a, b))
    if not edges or {p["owner"] for p in ports} != set(range(len(components))):
        raise ValueError("every fragment needs a port")
    return {"background": background, "cells": cells, "parts": components,
            "ports": ports, "edges": edges}


def _assemble(parsed, mode="zip"):
    """Solve the relative coordinate graph; reject conflicts and unresolved pieces.

    The non-default modes are explicit counterfactuals for demonstration checks.
    They are never used by solve or to select an intended output.
    """
    adjacency = [[] for _ in parsed["parts"]]
    for a, b in parsed["edges"]:
        normal = a["normal"]
        if mode == "endpoints":
            delta = _sub(_sub(a["rail"], normal), b["rail"])
        else:
            step = 0 if mode == "coincide" else -1 if mode == "reverse" else 1
            delta = _sub(_add(a["start"], tuple(step*x for x in normal)), b["start"])
        i, j = a["owner"], b["owner"]
        adjacency[i].append((j, delta))
        adjacency[j].append((i, tuple(-x for x in delta)))
    offsets, stack = {0: (0, 0)}, [0]
    while stack:
        i = stack.pop()
        for j, delta in adjacency[i]:
            position = _add(offsets[i], delta)
            if j in offsets:
                if offsets[j] != position:
                    raise ValueError("inconsistent seam cycle")
            else:
                offsets[j] = position
                stack.append(j)
    if len(offsets) != len(adjacency):
        raise ValueError("disconnected seam graph")
    merged, ordinary_count = {}, 0
    for i, part in enumerate(parsed["parts"]):
        for p in sorted(part):
            color = parsed["cells"][p]
            if color == MARKER:
                continue
            ordinary_count += 1
            target = _add(p, offsets[i])
            if target in merged and merged[target] != color:
                raise ValueError("conflicting ordinary colors")
            merged[target] = color
    if mode == "guides":
        for i, part in enumerate(parsed["parts"]):
            for p in sorted(part):
                if parsed["cells"][p] == MARKER:
                    merged.setdefault(_add(p, offsets[i]), MARKER)
    result = _render(merged, parsed["background"])
    if max(result.shape) > 30:
        raise ValueError("assembly exceeds 30 cells")
    return result, offsets, ordinary_count - len(merged)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Match exact words, cross their guides, remove gray, and crop the union."""
    return _assemble(_parse(input_grid))[0]


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _differs(parsed, output, mode):
    try:
        other = _assemble(parsed, mode)[0]
    except ValueError:
        return True
    return not np.array_equal(other, output)


def _word_start(shape, direction, along):
    h, w = shape
    return ((-1, along), (along, w), (h, along), (along, -1))[direction]


def _put(cells, position, value):
    if position in cells and cells[position] != value:
        raise ValueError("local port collision")
    cells[position] = int(value)


def _path(a, b):
    r, c = a
    result = {a}
    while r != b[0]:
        r += 1 if r < b[0] else -1
        result.add((r, c))
    while c != b[1]:
        c += 1 if c < b[1] else -1
        result.add((r, c))
    return result


def _body(rng, shape, attachments, palette):
    h, w = shape
    kind = int(rng.integers(6))
    center = (h//2, w//2)
    if kind == 0:  # closed frame, with meaningful negative space
        mask = {(r, c) for r in range(h) for c in range(w)
                if r in (0, h-1) or c in (0, w-1)}
    elif kind == 1:  # filled mass with a variable corner notch
        cut = int(rng.integers(1, min(h, w)))
        mask = {(r, c) for r in range(h) for c in range(w) if r+c >= cut}
    elif kind == 2:  # comb with unequal teeth
        mask = {(r, w//2) for r in range(h)}
        for r in range(0, h, 2):
            lo, hi = int(rng.integers(w//2+1)), int(rng.integers(w//2, w))
            mask.update((r, c) for c in range(lo, hi+1))
    elif kind == 3:  # open U
        mask = {(r, c) for r in range(h) for c in range(w)
                if c in (0, w-1) or r == h-1}
    elif kind == 4:  # irregular bent strokes
        mask = _path((0, int(rng.integers(w))), center)
        mask |= _path(center, (h-1, int(rng.integers(w))))
        mask |= _path(center, (int(rng.integers(h)), w-1))
    else:
        mask = {(r, c) for r in range(h) for c in range(w)}
    for p in attachments:
        if p not in mask:
            nearest = min(mask, key=lambda q: (abs(q[0]-p[0])+abs(q[1]-p[1]), q))
            mask.update(_path(nearest, p))
    # At least one connected body; no repairing disconnected pieces after solving.
    if len(_groups(mask)) != 1:
        raise ValueError("disconnected body")
    base = int(rng.choice(palette))
    return {p: int(rng.choice(palette)) if rng.random() < .12 else base for p in sorted(mask)}


def _tree_parts(rng, count, background, teaching):
    """Grow an irregular self-avoiding tree with independently sampled ports."""
    palette = [int(x) for x in rng.choice([c for c in range(10) if c not in (background, MARKER)],
                                        int(rng.integers(3, 6)), replace=False)]
    shapes = [(int(rng.integers(3, 7)), int(rng.integers(3, 7))) for _ in range(count)]
    origins, used_sides, ports = [(0, 0)], [set() for _ in range(count)], [[] for _ in range(count)]
    words = set()
    # Two same-multiset words explicitly distinguish ordered patterns from color bags.
    a, b = [int(x) for x in rng.choice(palette, 2, replace=False)]
    taught_words = [(a, b), (b, a)]
    for j in range(1, count):
        placed = False
        for _ in range(90):
            if teaching == "branch":
                i = 0
            elif teaching == "chain":
                i = j-1
            else:
                i = int(rng.integers(j))
            available = [d for d in range(4) if d not in used_sides[i]]
            if not available:
                continue
            direction = (ports[0][0][0] if teaching == "chain" and j > 1
                         else int(rng.choice(available)))
            opposite = (direction+2) % 4
            normal = DIRECTIONS[direction]
            axis = 1 if direction in (0, 2) else 0
            maxq = min(4, shapes[i][axis], shapes[j][axis])
            if teaching and j <= 2:
                word = taught_words[j-1]
            else:
                q = int(rng.integers(2, maxq+1))
                word = tuple(int(x) for x in rng.choice(palette, q))
            if len(set(word)) < 2 or word in words:
                continue
            q = len(word)
            ai = int(rng.integers(shapes[i][axis]-q+1))
            aj = int(rng.integers(shapes[j][axis]-q+1))
            si, sj = _word_start(shapes[i], direction, ai), _word_start(shapes[j], opposite, aj)
            origin = _add(origins[i], _sub(_add(si, normal), sj))
            # Other cores keep a two-cell moat so joins are visibly keyed.
            top, left = origin
            h, w = shapes[j]
            if any(k != i and not (top+h < r-1 or r+shapes[k][0] < top-1
                                    or left+w < c-1 or c+shapes[k][1] < left-1)
                   for k, (r, c) in enumerate(origins)):
                continue
            boxes = [(r, c, r+shapes[k][0]-1, c+shapes[k][1]-1)
                     for k, (r, c) in enumerate(origins)] + [(top, left, top+h-1, left+w-1)]
            if (max(x[2] for x in boxes)-min(x[0] for x in boxes)+1 > 28
                    or max(x[3] for x in boxes)-min(x[1] for x in boxes)+1 > 28):
                continue
            # Rails have independent overhangs; their endpoints are not coordinates.
            if teaching:
                length = 5
                oi = 0 if j % 2 else 2
                oj = 2 if j % 2 else 0
                li, lj = length, length
            else:
                oi, oj = int(rng.integers(3)), int(rng.integers(3))
                li, lj = q+oi+int(rng.integers(3)), q+oj+int(rng.integers(3))
            ports[i].append((direction, si, word, oi, li))
            ports[j].append((opposite, sj, word, oj, lj))
            used_sides[i].add(direction)
            used_sides[j].add(opposite)
            words.add(word)
            origins.append(origin)
            placed = True
            break
        if not placed:
            raise ValueError("bounded tree growth failed")
    parts = []
    target = {}
    for i, records in enumerate(ports):
        attachments = []
        for direction, start, word, offset, length in records:
            normal = DIRECTIONS[direction]
            tangent = (0, 1) if direction in (0, 2) else (1, 0)
            attachments.extend(_sub(_add(start, (k*tangent[0], k*tangent[1])), normal)
                               for k in range(len(word)))
        cells = _body(rng, shapes[i], attachments, palette)
        for direction, start, word, offset, length in records:
            normal = DIRECTIONS[direction]
            tangent = (0, 1) if direction in (0, 2) else (1, 0)
            for k, color in enumerate(word):
                _put(cells, _add(start, (k*tangent[0], k*tangent[1])), color)
            guide = _sub(_add(start, normal), (offset*tangent[0], offset*tangent[1]))
            for k in range(length):
                _put(cells, _add(guide, (k*tangent[0], k*tangent[1])), MARKER)
        if len(_groups(cells)) != 1:
            raise ValueError("port detached from its body")
        for p, color in cells.items():
            if color != MARKER:
                _put(target, _add(p, origins[i]), color)
        parts.append(_normalise(cells))
    return parts, _render(target, background)


def _pack(rng, parts, background):
    shapes = [(max(r for r, c in p)+1, max(c for r, c in p)+1) for p in parts]
    maxh, maxw = max(h for h, w in shapes), max(w for h, w in shapes)
    # Random placement, not source order or output adjacency. At most 32 canvases.
    for attempt in range(32):
        h = int(rng.integers(maxh, 31)) if attempt < 22 else 30
        w = int(rng.integers(maxw, 31)) if attempt < 22 else 30
        if h*w <= 2*sum(map(len, parts)):
            continue
        grid = np.full((h, w), background, dtype=np.int8)
        halo = np.zeros((h, w), dtype=bool)
        order = sorted(range(len(parts)), key=lambda i: (-len(parts[i]), float(rng.random())))
        for i in order:
            ph, pw = shapes[i]
            points = np.array(sorted(parts[i]), dtype=int)
            candidates = rng.permutation((h-ph+1)*(w-pw+1))
            for v in candidates:
                r0, c0 = divmod(int(v), w-pw+1)
                rr, cc = points[:, 0]+r0, points[:, 1]+c0
                if np.any(halo[rr, cc]):
                    continue
                for (r, c), color in parts[i].items():
                    y, x = r+r0, c+c0
                    grid[y, x] = color
                    halo[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)] = True
                break
            else:
                break
        else:
            return grid
    raise ValueError("bounded fragment packing failed")


def _make_pair(rng, count, background, teaching, seen):
    for _ in range(160):
        try:
            parts, expected = _tree_parts(rng, count, background, teaching)
            inp = _pack(rng, parts, background)
            parsed = _parse(inp)
            output, _, overlap = _assemble(parsed)
            if len(parsed["parts"]) != count or not np.array_equal(output, expected) or overlap:
                continue
            if _key(inp) in seen:
                continue
            if not all(_differs(parsed, output, m) for m in ("coincide", "endpoints", "reverse")):
                continue
            # A visible unused part of a guide teaches erasure, not repainting.
            if teaching and not _differs(parsed, output, "guides"):
                continue
            return {"input": inp, "output": output}
        except ValueError:
            continue
    raise RuntimeError("bounded keyed-zipper construction exhausted")


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    background = int(rng.choice([c for c in range(10) if c != MARKER]))
    changing_background = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", nq)):
        for index in range(number):
            teaching = "chain" if split == "train" and index == 0 else "branch" if split == "train" and index == 1 else None
            count = 3 if teaching == "chain" else 4 if teaching == "branch" else int(rng.integers(2, 5)) if split == "train" else 5
            bg = int(rng.choice([c for c in range(10) if c != MARKER])) if changing_background else background
            pair = _make_pair(rng, count, bg, teaching, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check pixels, seam conservation, teaching counterfactuals and count shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, information = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, output = _checked_grid(pair["input"]), _checked_grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            parsed = _parse(inp)
            expected, positions, overlap = _assemble(parsed)
            assert np.array_equal(output, expected), "wrong seam, union or tight crop"
            assert not np.any(output == MARKER), "temporary guides retained"
            assert overlap == 0, "generated bodies should not overlap"
            assert np.count_nonzero(output != parsed["background"]) == sum(
                v != MARKER for v in parsed["cells"].values()), "ordinary material lost"
            assert len(parsed["edges"]) == len(parsed["parts"])-1, "generated graph must be a tree"
            assert len(_groups(zip(*np.nonzero(output != parsed["background"])))) == 1
            assert all(_differs(parsed, output, mode) for mode in ("coincide", "endpoints", "reverse"))
            # Directly verify both rows/columns of every stitched word.
            merged = {_add(p, positions[i]): parsed["cells"][p]
                      for i, part in enumerate(parsed["parts"]) for p in part
                      if parsed["cells"][p] != MARKER}
            for a, b in parsed["edges"]:
                sa = _add(a["start"], positions[a["owner"]])
                sb = _add(b["start"], positions[b["owner"]])
                assert sb == _add(sa, a["normal"])
                tangent = (0, 1) if a["normal"][0] else (1, 0)
                for k, color in enumerate(a["word"]):
                    d = (k*tangent[0], k*tangent[1])
                    assert merged[_add(sa, d)] == merged[_add(sb, d)] == color
            information[split].append(parsed)
    train = information["train"]
    assert max(len(p["parts"]) for p in train) < min(len(p["parts"]) for p in information["test"])
    assert any(_differs(p, np.asarray(pair["output"]), "guides") for p, pair in zip(train, puzzle["train"]))
    assert any(len({p["length"] for p in info["ports"]}) == 1
               and len({p["word"] for p in info["ports"]}) > 1 for info in train)
    assert any(any(a != b and sorted(a) == sorted(b) for a, b in itertools.combinations(
        {p["word"] for p in info["ports"]}, 2)) for info in train)


def _holes(grid, background):
    h, w = grid.shape
    return sum(all(0 < r < h-1 and 0 < c < w-1 for r, c in part)
               for part in _groups(zip(*np.nonzero(grid == background))))


def self_test():
    """42 streaming seeds, finite counterfactual and metamorphic checks; no files."""
    input_shapes, output_shapes, structures, signatures = set(), set(), set(), set()
    normals, word_lengths, rail_lengths, backgrounds, sizes, offsets = set(), set(), set(), set(), set(), set()
    holes = branching = unequal_rails = pairs = 0
    digest = hashlib.sha256()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        local = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, output = pair["input"], pair["output"]
                parsed = _parse(inp)
                assert np.array_equal(solve(inp.copy()), output)
                input_shapes.add(inp.shape)
                output_shapes.add(output.shape)
                backgrounds.add(parsed["background"])
                sizes.add(len(parsed["parts"]))
                degree = Counter(p["owner"] for p in parsed["ports"])
                branching += max(degree.values()) >= 3
                holes += _holes(output, parsed["background"]) > 0
                unequal_rails += any(a["length"] != b["length"] for a, b in parsed["edges"])
                for port in parsed["ports"]:
                    normals.add(port["normal"])
                    word_lengths.add(len(port["word"]))
                    rail_lengths.add(port["length"])
                    offsets.add(port["offset"])
                labels = {parsed["background"]: 0, MARKER: 1}
                norm = bytes(labels.setdefault(int(c), len(labels)) for c in inp.flat)
                structures.add(hashlib.sha256(str(inp.shape).encode()+norm).digest())
                for grid in (inp, output):
                    local.update(str(grid.shape).encode())
                    local.update(grid.tobytes())
                pairs += 1
        signatures.add(local.digest())
        digest.update(local.digest())
        if seed < 3:
            for pair in puzzle["train"][:2]:
                # Rotate/reflect the whole scene, never a fragment during assembly.
                assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
                assert np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"]))
                # Any permutation of non-gray colors preserves semantic roles.
                perm = np.arange(10, dtype=np.int8)
                free = [c for c in range(10) if c != MARKER]
                perm[free] = free[::-1]
                assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(input_shapes) >= 30 and len(output_shapes) >= 40
    assert normals == set(DIRECTIONS) and word_lengths == {2, 3, 4}
    assert rail_lengths >= {2, 3, 4, 5, 6, 7, 8} and offsets == {0, 1, 2}
    assert len(backgrounds) >= 8 and sizes == {2, 3, 4, 5}
    assert holes >= 10 and branching >= 42 and unequal_rails >= 20
    first, second = generate_puzzle(7), generate_puzzle(7)
    assert all(_key(a[role]) == _key(b[role]) for split in ("train", "test")
               for a, b in zip(first[split], second[split]) for role in ("input", "output"))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Reject semantically damaged outputs, rather than checking solve alone.
    damaged = generate_puzzle(0, 2, 1)
    output = damaged["test"][0]["output"].copy()
    bg = _parse(damaged["test"][0]["input"])["background"]
    p = tuple(np.argwhere(output != bg)[0])
    output[p] = bg
    damaged["test"][0]["output"] = output
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing material accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "normalized_inputs": len(structures), "backgrounds": len(backgrounds),
            "word_lengths": sorted(word_lengths), "rail_lengths": sorted(rail_lengths),
            "fragment_counts": sorted(sizes), "hole_pairs": holes,
            "branch_pairs": branching, "unequal_rail_pairs": unequal_rails,
            "sample_digest_sha256": digest.hexdigest()}
