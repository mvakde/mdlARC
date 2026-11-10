"""Facing sockets: translation-only frame assembly constrained by colored chords.

Only NumPy and the Python standard library are used. No parent code or saved
examples are imported. The public solver reads all conventions from its input.
"""
from __future__ import annotations

from collections import defaultdict
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(int, copy=False)


def _perimeter(side):
    return ([(0, c) for c in range(side)]
            + [(r, side - 1) for r in range(1, side)]
            + [(side - 1, c) for c in range(side - 2, -1, -1)]
            + [(r, 0) for r in range(side - 2, 0, -1)])


def _components(grid, background):
    unseen = set(map(tuple, np.argwhere(grid != background).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, found = [start], []
        while stack:
            r, c = stack.pop()
            found.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(found)))
    return result


def _piece(cells, values, inks, body):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    ordered = sorted(zip(cells, values))
    local = tuple((r - r0, c - c0) for (r, c), _ in ordered)
    colors = tuple(int(v) for _, v in ordered)
    return {"cells": local, "values": colors, "body": int(body),
            "ports": tuple((v, r, c) for (r, c), v in zip(local, colors)
                           if v in inks)}


def _decode(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    if int(counts.max()) * 2 <= grid.size:
        raise ValueError("background must occupy more than half the input")
    components = _components(grid, background)
    if not 3 <= len(components) <= 6:
        raise ValueError("expected three to six separated fragments")
    members = defaultdict(set)
    for i, cells in enumerate(components):
        for r, c in cells:
            members[int(grid[r, c])].add(i)
    inks = {v for v, indices in members.items() if len(indices) > 1}
    if not 2 <= len(inks) <= 4:
        raise ValueError("expected two to four shared socket colors")
    for ink in inks:
        if len(members[ink]) != 2 or np.count_nonzero(grid == ink) != 2:
            raise ValueError("each socket color must occur once on two fragments")
    pieces = []
    for cells in components:
        values = [int(grid[r, c]) for r, c in cells]
        bodies = set(values) - inks
        if len(bodies) != 1:
            raise ValueError("each fragment needs exactly one private body color")
        pieces.append(_piece(cells, values, inks, next(iter(bodies))))
    area = sum(len(p["cells"]) for p in pieces)
    if area % 4 or not 5 <= area // 4 + 1 <= 30:
        raise ValueError("fragment area must be a square perimeter of side >= 5")
    return grid, background, area // 4 + 1, pieces, tuple(sorted(inks))


def _opposite(a, b, side):
    r, c = a
    u, v = b
    return ((r == u and 0 < r < side - 1 and {c, v} == {0, side - 1})
            or (c == v and 0 < c < side - 1 and {r, u} == {0, side - 1}))


def _assemblies(pieces, side, *, paired=True, limit=2):
    """Enumerate translation covers; optionally enforce opposite socket pairs.

    Counting stops after `limit` complete solutions. No search truncation is used
    before that, so a returned singleton means uniqueness, not first-found choice.
    """
    ring = _perimeter(side)
    bit = {cell: 1 << i for i, cell in enumerate(ring)}
    target = (1 << len(ring)) - 1
    options = []
    for piece in pieces:
        cells = piece["cells"]
        height = max(r for r, c in cells) + 1
        width = max(c for r, c in cells) + 1
        candidates = []
        for dr in range(side - height + 1):
            for dc in range(side - width + 1):
                translated = [(r + dr, c + dc) for r, c in cells]
                if not all(p in bit for p in translated):
                    continue
                ports = [(ink, (r + dr, c + dc))
                         for ink, r, c in piece["ports"]]
                # Exactly one coordinate is on an edge: sockets cannot be corners.
                if paired and any((r in (0, side - 1)) == (c in (0, side - 1))
                                  for _, (r, c) in ports):
                    continue
                candidates.append((sum(bit[p] for p in translated), ports, (dr, dc)))
        if not candidates:
            return []
        options.append(candidates)
    order = sorted(range(len(pieces)),
                   key=lambda i: (len(options[i]), -len(pieces[i]["cells"]), i))
    solutions, placements, endpoints = [], {}, {}

    def search(depth, occupied):
        if len(solutions) >= limit:
            return
        if depth == len(order):
            if occupied == target:
                solutions.append(tuple(placements[i] for i in range(len(pieces))))
            return
        i = order[depth]
        for mask, ports, delta in options[i]:
            if mask & occupied:
                continue
            if paired and any(ink in endpoints and not _opposite(p, endpoints[ink], side)
                              for ink, p in ports):
                continue
            previous = endpoints.copy()
            endpoints.update(ports)
            placements[i] = delta
            search(depth + 1, occupied | mask)
            placements.pop(i)
            endpoints.clear()
            endpoints.update(previous)

    search(0, 0)
    return solutions


def _border_and_lines(pieces, placement, side, background):
    border = np.full((side, side), background, dtype=int)
    horizontal, vertical = {}, {}
    for piece, (dr, dc) in zip(pieces, placement):
        for (r, c), value in zip(piece["cells"], piece["values"]):
            border[r + dr, c + dc] = value
        for ink, r, c in piece["ports"]:
            r, c = r + dr, c + dc
            if r in (0, side - 1):
                vertical[c] = ink
            else:
                horizontal[r] = ink
    return border, horizontal, vertical


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Find the sole facing-socket assembly, then extend its lines."""
    _, background, side, pieces, _ = _decode(input_grid)
    solutions = _assemblies(pieces, side)
    if len(solutions) != 1:
        raise ValueError("input must have exactly one facing-socket frame assembly")
    output, horizontal, vertical = _border_and_lines(pieces, solutions[0], side, background)
    for col, ink in vertical.items():
        output[:, col] = ink
    for row, ink in horizontal.items():
        output[row, :] = ink
    return output


def _cuts(rng, side, count):
    length = 4 * side - 4
    style = int(rng.integers(6))
    required = set()
    if style == 1:
        start = int(rng.integers(length))
        required = {start, (start + 1) % length}
    elif style == 2:
        required = {int(rng.integers(4)) * (side - 1)}
    elif style == 3:
        start = int(rng.integers(4)) * (side - 1)
        required = {start, (start + side) % length}
    elif style == 4:
        start = int(rng.integers(length))
        span = int(rng.integers(side + 1, min(2 * side + 2, length)))
        required = {start, (start + span) % length}
    remaining = [j for j in range(length) if j not in required]
    extra = rng.choice(remaining, count - len(required), replace=False)
    return sorted(required | set(map(int, extra)))


def _design(rng, side, piece_count, nh, nv, ambiguity):
    """Build intended answers directly, retaining only uniquely solvable inputs."""
    if not (5 <= side <= 30 and 3 <= piece_count <= 6 and 1 <= nh < side - 2
            and 1 <= nv < side - 2 and 2 <= nh + nv <= 4
            and 1 + piece_count + nh + nv <= 10):
        raise ValueError("incompatible frame, fragment, or socket counts")
    ring = _perimeter(side)
    length = len(ring)
    palette = list(map(int, rng.permutation(10)))
    background = palette[0]
    bodies = palette[1:piece_count + 1]
    inks = palette[piece_count + 1:piece_count + 1 + nh + nv]
    for _ in range(1800):
        cuts = _cuts(rng, side, piece_count)
        arcs = []
        for i, start in enumerate(cuts):
            end = cuts[(i + 1) % piece_count]
            arcs.append([ring[j % length]
                         for j in range(start, end if end > start else end + length)])
        owner = {p: i for i, arc in enumerate(arcs) for p in arc}
        rows = list(map(int, rng.choice(np.arange(1, side - 1), nh, replace=False)))
        cols = list(map(int, rng.choice(np.arange(1, side - 1), nv, replace=False)))
        socket_pairs = [((r, 0), (r, side - 1)) for r in rows]
        socket_pairs += [((0, c), (side - 1, c)) for c in cols]
        if any(owner[a] == owner[b] for a, b in socket_pairs):
            continue
        portmap = {p: ink for ink, pair in zip(inks, socket_pairs) for p in pair}
        if any(all(p in portmap for p in arc) for arc in arcs):
            continue  # Every piece retains at least one private body pixel.
        pieces = [_piece(arc, [portmap.get(p, bodies[i]) for p in arc], set(inks), bodies[i])
                  for i, arc in enumerate(arcs)]
        geometry = _assemblies(pieces, side, paired=False)
        if ambiguity == "ambiguous" and len(geometry) < 2:
            continue
        if ambiguity == "unique" and len(geometry) != 1:
            continue
        solutions = _assemblies(pieces, side)
        if len(solutions) != 1:
            continue
        # The construction's own translations must be the inferred unique ones.
        intended = tuple((min(r for r, c in arc), min(c for r, c in arc)) for arc in arcs)
        if solutions[0] != intended:
            raise AssertionError("constructed frame failed its own constraints")
        border = np.full((side, side), background, dtype=int)
        for i, arc in enumerate(arcs):
            for r, c in arc:
                border[r, c] = portmap.get((r, c), bodies[i])
        output = border.copy()
        for col, ink in zip(cols, inks[nh:]):
            output[:, col] = ink
        for row, ink in zip(rows, inks[:nh]):
            output[row, :] = ink
        return pieces, background, output
    raise RuntimeError("could not sample a unique socket-constrained frame")


def _misaligned_socket_pair(grid, inks):
    for ink in inks:
        (r, c), (u, v) = np.argwhere(grid == ink).tolist()
        if r != u and c != v:
            return True
    return False


def _scatter(rng, pieces, side, background, *, small=False):
    inks = {ink for piece in pieces for ink, _, _ in piece["ports"]}
    upper = min(30, side + (7 if small else 13))
    height = int(rng.integers(side + 3, upper + 1))
    width = int(rng.integers(side + 3, upper + 1))
    for attempt in range(120):
        if attempt in (40, 80):
            height, width = min(30, height + 3), min(30, width + 3)
        occupied, placed = set(), {}
        # Random ties with larger pieces first; input order has no semantic role.
        order = list(map(int, rng.permutation(len(pieces))))
        order.sort(key=lambda i: -len(pieces[i]["cells"]))
        for i in order:
            cells = pieces[i]["cells"]
            ph, pw = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
            for _ in range(100):
                dr = int(rng.integers(height - ph + 1))
                dc = int(rng.integers(width - pw + 1))
                translated = [(r + dr, c + dc) for r, c in cells]
                if any(p in occupied or any(q in occupied for q in
                        ((p[0]-1, p[1]), (p[0]+1, p[1]),
                         (p[0], p[1]-1), (p[0], p[1]+1))) for p in translated):
                    continue
                placed[i] = translated
                occupied.update(translated)
                break
            else:
                break
        else:
            grid = np.full((height, width), background, dtype=int)
            for i, cells in placed.items():
                for (r, c), value in zip(cells, pieces[i]["values"]):
                    grid[r, c] = value
            if len(occupied) * 2 < grid.size and _misaligned_socket_pair(grid, inks):
                return grid
    raise RuntimeError("could not separate the scattered decorated fragments")


def _checked_integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _checked_integer(seed, "seed", 0)
    num_train = _checked_integer(num_train, "num_train", 2, 10)
    num_test = _checked_integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    repeated_horizontal = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for _ in range(20):
                if split == "train" and i == 0:
                    side, pc = int(rng.integers(5, 7)), int(rng.integers(3, 5))
                    nh, nv, ambiguity = 1, 1, "unique"
                elif split == "train":
                    side = int(rng.integers(6, 9)) if i == 1 else int(rng.integers(5, 11))
                    pc = int(rng.integers(4, 6)) if i == 1 else int(rng.integers(3, 7))
                    k = 3 if i == 1 else int(rng.integers(2, 4))
                    nh = 2 if k == 3 and repeated_horizontal else 1
                    nv = k - nh
                    ambiguity = "ambiguous" if i == 1 else "any"
                else:
                    side = int(rng.integers(6, 12))
                    k = 3 if rng.random() < 0.65 else 4
                    pc = int(rng.integers(4, 7 if k == 3 else 6))
                    nh = (1 if repeated_horizontal else 2) if k == 3 else 2
                    nv, ambiguity = k - nh, "ambiguous"
                try:
                    pieces, background, output = _design(rng, side, pc, nh, nv, ambiguity)
                    inp = _scatter(rng, pieces, side, background,
                                   small=split == "train" and i == 0)
                except RuntimeError:
                    continue
                signature = (inp.shape, inp.tobytes())
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle[split].append({"input": inp, "output": output})
                break
            else:
                raise RuntimeError("could not generate a distinct family example")
    return puzzle


def _analyze_pair(pair):
    inp, bg, side, pieces, inks = _decode(pair["input"])
    out = _grid(pair["output"])
    _require(out.shape == (side, side), "wrong tight output dimensions")
    solutions = _assemblies(pieces, side)
    _require(len(solutions) == 1, "socket-constrained assembly must be unique")
    border, horizontal, vertical = _border_and_lines(pieces, solutions[0], side, bg)
    _require(horizontal and vertical, "both line axes must contribute")
    _require(len(horizontal) + len(vertical) == len(inks), "sockets did not define distinct chords")
    _require(_misaligned_socket_pair(inp, inks), "input positions accidentally solve all alignments")
    _require(len(horizontal) < side - 2 and len(vertical) < side - 2,
             "need at least one unpainted interior background cell")
    # Check the full cell formula, separately from solve's drawing operations.
    for r in range(side):
        for c in range(side):
            if r in (0, side - 1) or c in (0, side - 1):
                expected = border[r, c]
            else:
                expected = horizontal.get(r, vertical.get(c, bg))
            _require(out[r, c] == expected, "incorrect border, chord, or crossing color")
    for piece, (dr, dc) in zip(pieces, solutions[0]):
        for (r, c), value in zip(piece["cells"], piece["values"]):
            _require(out[r + dr, c + dc] == value, "decorated fragment was changed")
        _require(np.count_nonzero(out == piece["body"])
                 == np.count_nonzero(inp == piece["body"]), "body color leaked into interior")
    _require(np.any(out[1:-1, 1:-1] == bg), "missing empty interior witness")
    _require(not np.array_equal(out, border), "line mechanism has no effect")
    _require(all(out[r, c] == horizontal[r] and out[r, c] != vertical[c]
                 for r in horizontal for c in vertical), "priority is not observable")
    geometry_count = len(_assemblies(pieces, side, paired=False))
    lengths = tuple(sorted(len(p["cells"]) for p in pieces))
    long_bend = any(len(p["cells"]) > side and
                    len({r for r, c in p["cells"]}) > 1 and
                    len({c for r, c in p["cells"]}) > 1 for p in pieces)
    return {"side": side, "pieces": len(pieces), "nh": len(horizontal),
            "nv": len(vertical), "geometry_count_capped": geometry_count,
            "background": bg, "lengths": lengths, "long_bend": long_bend,
            "shape_profile": tuple(sorted(p["cells"] for p in pieces)),
            "input_shape": inp.shape}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, metadata = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            m = _analyze_pair(pair)
            _require(np.array_equal(solve(inp), out), "public solver disagrees")
            metadata[split].append(m)
    train, test = metadata["train"], metadata["test"]
    _require(train[0]["geometry_count_capped"] == 1 and (train[0]["nh"], train[0]["nv"]) == (1, 1),
             "first demonstration must anchor simple assembly and crossing")
    _require(train[1]["geometry_count_capped"] == 2 and train[1]["nh"] + train[1]["nv"] == 3,
             "second demonstration must expose socket disambiguation and repetition")
    max_h, max_v = max(m["nh"] for m in train), max(m["nv"] for m in train)
    _require((max_h, max_v) in ((2, 1), (1, 2)), "training repetition convention is wrong")
    held_axis = "nv" if max_v == 1 else "nh"
    _require(all(m[held_axis] >= 2 and m["geometry_count_capped"] == 2 for m in test),
             "test must combine new-axis repetition with ambiguous geometry")
    return metadata


def _signature(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
                       for s in ("train", "test")}, separators=(",", ":"))


def self_test():
    """Bounded checks on 42 seeds, semantic ablations, and invariance to scatter."""
    puzzles, shapes, profiles, sides, fragment_counts = set(), set(), set(), set(), set()
    backgrounds, line_counts, ambiguous, singletons, long_bends = set(), set(), 0, 0, 0
    normalized_inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        metadata = validate_generated_puzzle(puzzle)
        puzzles.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair, m in zip(puzzle[split], metadata[split]):
                shapes.add(m["input_shape"])
                profiles.add(m["shape_profile"])
                sides.add(m["side"])
                fragment_counts.add(m["pieces"])
                backgrounds.add(m["background"])
                line_counts.add((m["nh"], m["nv"]))
                ambiguous += m["geometry_count_capped"] == 2
                singletons += 1 in m["lengths"]
                long_bends += m["long_bend"]
                inp = np.asarray(pair["input"])
                normalized_inputs.add((inp.shape, (inp != m["background"]).tobytes()))
        if seed in (0, 17, 41):
            _require(_signature(generate_puzzle(seed)) == _signature(puzzle), "nondeterminism")
            for pair in (puzzle["train"][0], puzzle["train"][1], puzzle["test"][0]):
                inp, bg, side, pieces, _ = _decode(pair["input"])
                relocated = _scatter(np.random.default_rng(seed + 700), pieces, side, bg)
                _require(np.array_equal(solve(relocated), pair["output"]), "scatter changed the answer")
                palette = np.random.default_rng(seed + 900).permutation(10)
                _require(np.array_equal(solve(palette[inp]), palette[pair["output"]]),
                         "literal colors acquired unintended meaning")
    _require(len(puzzles) == 42 and len(normalized_inputs) >= 180, "too little structural variation")
    _require(len(shapes) >= 40 and len(profiles) >= 50, "canvas or fragment geometry too narrow")
    _require(sides >= set(range(5, 11)) and fragment_counts == {3, 4, 5, 6}, "size/count coverage missing")
    _require(backgrounds == set(range(10)), "not all colors can be background")
    _require(line_counts >= {(1, 1), (2, 1), (1, 2), (2, 2)}, "line multiplicities missing")
    _require(ambiguous >= 84 and singletons > 0 and long_bends > 0, "fragment interactions too narrow")
    # Family members beyond the normal sampler's size/repetition bands. These
    # still use the same input-only solver and the same cellwise semantic audit.
    for side, nh, nv in ((12, 3, 1), (14, 1, 3)):
        rng = np.random.default_rng(1000 + side)
        pieces, bg, out = _design(rng, side, 5, nh, nv, "ambiguous")
        inp = _scatter(rng, pieces, side, bg)
        _analyze_pair({"input": inp, "output": out})
        _require(np.array_equal(solve(inp), out), "solver depends on sampler size bands")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(12345, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "explicit count ignored")
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (None, 1), (4, None), (True, 1), (4, 1.0), ("4", 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "input_shapes": len(shapes),
            "fragment_profiles": len(profiles), "normalized_inputs": len(normalized_inputs),
            "sides": sorted(sides), "fragment_counts": sorted(fragment_counts),
            "line_counts": sorted(line_counts), "ambiguous_geometry_pairs": ambiguous,
            "singleton_pairs": singletons, "long_bend_pairs": long_bends}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
