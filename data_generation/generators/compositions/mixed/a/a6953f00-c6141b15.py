"""Exchange arrangements, then read the parity-selected top 2x2 corner.

No parent module, saved sample, seed, or generation metadata is used by solve.
Generation retains no files. Run this module for a bounded 42-seed self-test.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 774056411, 337299806, 440692296, 115015236, 19057715,
              301034190, 757699908, 665098131, 614805614, 387941815,
              33287017, 888983581, 192187210, 637370024, 238145021,
              582148455, 87675128, 924434072, 182042314, 157940623,
              282003416, 638464174, 127131786, 971881590, 182417415,
              22964951, 632900100, 757071055, 476734536, 604754715,
              124381815, 134597112, 847890685, 647670363, 636177454,
              406505682, 894804462, 888759417, 407218780, 405459883,
              750676501)

# All are 8-connected, have tight odd bounding boxes and contain their anchor.
MOTIFS = {
    "vbar": frozenset({(-1, 0), (0, 0), (1, 0)}),
    "hbar": frozenset({(0, -1), (0, 0), (0, 1)}),
    "plus": frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)}),
    "x": frozenset({(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)}),
    "tee": frozenset({(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)}),
    "arrow": frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1)}),
    "hook": frozenset({(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 0)}),
    "tall_i": frozenset({(-2, -1), (-2, 0), (-2, 1), (-1, 0), (0, 0),
                          (1, 0), (2, -1), (2, 0), (2, 1)}),
    "wide_i": frozenset({(-1, -2), (0, -2), (1, -2), (0, -1), (0, 0),
                          (0, 1), (-1, 2), (0, 2), (1, 2)}),
    "dot": frozenset({(0, 0)}),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and 2 <= num_train <= 10,
             "num_train must be an integer in 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4,
             "num_test must be an integer in 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be in 0..9")
    return a


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require(dr == 0 or dc == 0 or abs(dr) == abs(dc), "unsupported line slope")
    length = max(abs(dr), abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return frozenset((a[0] + i * sr, a[1] + i * sc) for i in range(length + 1))


def _components(cells):
    unseen, result = set(cells), []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, part = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in unseen:
                        unseen.remove(q)
                        part.add(q)
                        todo.append(q)
        result.append(frozenset(part))
    return result


def _stamps(centers, motif):
    return frozenset((r + dr, c + dc) for r, c in centers for dr, dc in motif)


def _figure(centers):
    return frozenset(q for a, b in combinations(centers, 2) for q in _line(a, b))


def _fits(cells, n):
    return all(0 <= r < n and 0 <= c < n for r, c in cells)


def _analyze(input_grid):
    grid = _grid(input_grid)
    n = grid.shape[0]
    _require(grid.shape == (n, n) and n >= 2, "input must be square")
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 3, "input needs background, segment and glyph colors")
    bg_index = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[bg_index]) == 1, "ambiguous background")
    bg = int(colors[bg_index])
    foreground = [int(v) for v in colors if v != bg]
    candidates = []
    for edge, glyph in (foreground, foreground[::-1]):
        source = frozenset(map(tuple, np.argwhere(grid == edge).tolist()))
        # Lexicographic extremes are endpoints of any supported straight segment.
        endpoints = (min(source), max(source))
        try:
            if len(source) < 4 or source != _line(*endpoints):
                continue
        except ValueError:
            continue
        parts = _components(map(tuple, np.argwhere(grid == glyph).tolist()))
        if len(parts) not in (2, 3):
            continue
        centers, motifs = [], []
        for part in parts:
            rows, cols = zip(*part)
            if (min(rows) + max(rows)) % 2 or (min(cols) + max(cols)) % 2:
                break
            center = ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2)
            motif = frozenset((r - center[0], c - center[1]) for r, c in part)
            if (0, 0) not in motif:
                break
            centers.append(center)
            motifs.append(motif)
        if len(centers) != len(parts) or len(set(motifs)) != 1:
            continue
        centers = tuple(sorted(centers))
        if len(centers) == 3:
            a, b, c = centers
            if (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0]):
                continue
        try:
            figure = _figure(centers)
        except ValueError:
            continue
        stamps = _stamps(endpoints, motifs[0])
        if (not _fits(stamps, n) or not figure.isdisjoint(stamps)
                or len(_components(stamps)) != 2):
            continue
        candidates.append({"n": n, "background": bg, "edge": edge, "glyph": glyph,
                           "source": source, "endpoints": endpoints, "centers": centers,
                           "motif": motifs[0], "figure": figure, "stamps": stamps})
    _require(len(candidates) == 1, "input must have exactly one valid role assignment")
    return candidates[0]


def _paint(n, background, layers):
    result = np.full((n, n), background, dtype=np.int8)
    for cells, color in layers:
        for r, c in cells:
            result[r, c] = color
    return result


def _exchange(info):
    return _paint(info["n"], info["background"],
                  ((info["figure"], info["edge"]), (info["stamps"], info["glyph"])))


def _crop(grid):
    return (grid[:2, :2] if grid.shape[0] % 2 else grid[:2, -2:]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover visible objects, exchange their arrangements, then crop by parity."""
    return _crop(_exchange(_analyze(input_grid)))


def _fingerprint(grid, normalize=False):
    a = np.asarray(grid, dtype=np.uint8)
    if normalize:
        labels = {}
        a = np.array([labels.setdefault(int(v), len(labels)) for v in a.flat],
                     dtype=np.uint8).reshape(a.shape)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _near_corner(rng, n, motif, right=False):
    ry = max(abs(r) for r, _ in motif)
    rx = max(abs(c) for _, c in motif)
    row = ry + int(rng.integers(2))
    col = rx + int(rng.integers(2))
    return (row, n - 1 - col if right else col)


def _sample_centers(rng, anchor, count, n):
    distance = int(rng.integers(4, n - 2))
    if count == 2:
        direction = ((0, 1), (1, 0), (1, 1), (1, -1))[int(rng.integers(4))]
        relative = [(0, 0), (distance * direction[0], distance * direction[1])]
    elif rng.integers(2):
        relative = [(0, 0), (distance, 0), (0, distance)]
    else:
        relative = [(0, 0), (distance, distance), (distance, -distance)]
    # All dihedral orientations; select any vertex to place at the corner anchor.
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    transformed = []
    for r, c in relative:
        if reflect:
            c = -c
        for _ in range(turns):
            r, c = c, -r
        transformed.append((r, c))
    base = transformed[int(rng.integers(count))]
    return tuple((r - base[0] + anchor[0], c - base[1] + anchor[1])
                 for r, c in transformed)


def _sample_endpoints(rng, anchor, n, right=False):
    length = int(rng.integers(4, n - 2))
    dr, dc = ((0, 1), (1, 0), (1, 1))[int(rng.integers(3))]
    if right:
        dc = -dc
    return (anchor, (anchor[0] + length * dr, anchor[1] + length * dc))


def _dual_geometry(rng, n, count, motif_name):
    """Two exchange operations are interleaved in the canonical left crop."""
    if motif_name == "dot":
        # The left crop includes a newly drawn line-interior pixel, and so does
        # the right crop. Recoloring glyph centers alone cannot solve this case.
        return ((0, 1), (0, n - 1), (n - 2, 1)), ((1, 0), (int(rng.integers(5, n - 1)), 0))
    horizontal = motif_name == "hbar"
    centers = [(1, 1), (1, n - 2)]
    if count == 3:
        centers.append((n - 2, 1 if rng.integers(2) else n - 2))
    last = int(rng.integers(n - 3, n - 1) if horizontal else rng.integers(5, n - 1))
    endpoints = ((1, 0), (last, 0))
    if horizontal:
        centers = [(c, r) for r, c in centers]
        endpoints = tuple((c, r) for r, c in endpoints)
    return tuple(centers), endpoints


def _make_pair(rng, n, mode, motif_name, count, selected_dual=True):
    motif = MOTIFS[motif_name]
    palette = tuple(int(v) for v in rng.choice(10, 3, replace=False))
    bg, edge, glyph = palette
    for _ in range(1800):
        if mode == "dual":
            centers, endpoints = _dual_geometry(rng, n, count, motif_name)
        else:
            centers = _sample_centers(rng, _near_corner(rng, n, motif, mode == "stamp"),
                                      count, n)
            endpoints = _sample_endpoints(rng, _near_corner(rng, n, motif, mode == "figure"),
                                          n, mode == "figure")
        # Reflect positions, not the glyph: its orientation is an input-visible
        # appearance that is always copied literally, including for asymmetries.
        mirror = (n % 2 == 0) if mode != "dual" or selected_dual else (n % 2 == 1)
        if mirror:
            centers = tuple((r, n - 1 - c) for r, c in centers)
            endpoints = tuple((r, n - 1 - c) for r, c in endpoints)
        original = _stamps(centers, motif)
        stamps = _stamps(endpoints, motif)
        source = _line(*endpoints)
        figure = _figure(centers)
        if not all(_fits(cells, n) for cells in (original, stamps, source, figure)):
            continue
        if (not source.isdisjoint(original) or not figure.isdisjoint(stamps)
                or len(_components(original)) != count or len(_components(stamps)) != 2):
            continue
        grid = _paint(n, bg, ((source, edge), (original, glyph)))
        full = _paint(n, bg, ((figure, edge), (stamps, glyph)))
        output = _crop(full)
        other = full[:2, -2:] if n % 2 else full[:2, :2]
        if not np.any(output != bg) or np.array_equal(output, other):
            continue
        if np.array_equal(output, _crop(grid)):
            continue
        # Both candidate corners carry exchanged ink in all mandatory dual
        # witnesses. Larger motifs can legitimately leave the other one blank.
        if mode == "dual" and not np.any(other != bg):
            continue
        info = _analyze(grid)
        if not np.array_equal(_exchange(info), full):
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("bounded placement search exhausted")


def _sides(rng, count, test=False):
    options = list(range(21, 31) if test else range(11, 21))
    first = int(rng.integers(2))
    parities = [first, 1 - first]
    if count >= 4:
        parities += [first, 1 - first]
    parities += [int(rng.integers(2)) for _ in range(max(0, count - len(parities)))]
    used = {0: set(), 1: set()}
    result = []
    for parity in parities[:count]:
        choices = [n for n in options if n % 2 == parity and n not in used[parity]]
        if not choices:
            choices = [n for n in options if n % 2 == parity]
        n = int(rng.choice(choices))
        used[parity].add(n)
        result.append(n)
    return result


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    num_train, num_test = int(num_train), int(num_test)
    train_sides, test_sides = _sides(rng, num_train), _sides(rng, num_test, True)
    bar = str(rng.choice(("vbar", "hbar")))
    teaching = [bar, "dot"]
    # The bar proves copying off-center pixels and erasure; the dot triangle
    # proves line drawing beyond the centers and rejects "choose richer corner".
    specifications = [("dual", bar, 2, True), ("dual", "dot", 3, False)]
    rng.shuffle(specifications)
    for i in range(2, num_train):
        mode = "stamp" if i % 2 == 0 else "figure"
        names = list(MOTIFS)[2:9] if mode == "stamp" else list(MOTIFS)[2:7]
        name = str(rng.choice(names))
        teaching.append(name)
        specifications.append((mode, name, int(rng.integers(2, 4)), True))
    # Tests transfer to larger scales and recombine already demonstrated motifs.
    # With only two demonstrations, use the same well-exposed boundary layouts.
    for _ in range(num_test):
        mode = str(rng.choice(("dual", "stamp", "figure"), p=(.5, .25, .25))) if num_train >= 3 else "dual"
        names = [bar, "dot"] if mode == "dual" else teaching
        if mode == "figure":
            names = [name for name in names if max(abs(r) for r, _ in MOTIFS[name]) <= 1
                     and max(abs(c) for _, c in MOTIFS[name]) <= 1]
        name = str(rng.choice(names))
        count = 3 if mode == "dual" and name == "dot" else int(rng.integers(2, 4))
        specifications.append((mode, name, count, bool(rng.random() < .7)))
    puzzle, seen, emitted_sides = {"train": [], "test": []}, set(), set()
    for i, (n, spec) in enumerate(zip(train_sides + test_sides, specifications)):
        # Leave room for a three-glyph figure and a second complete arrangement.
        # Increasing by an even amount preserves the scheduled parity.
        motif = MOTIFS[spec[1]]
        if n < 15 and ((spec[2] == 3 and len(motif) > 3)
                       or max(max(abs(r), abs(c)) for r, c in motif) == 2):
            n += 2 * ((15 - n + 1) // 2)
        if i < min(4, num_train) and n in emitted_sides:
            minimum = 15 if (spec[2] == 3 and len(motif) > 3) or any(
                max(abs(r), abs(c)) == 2 for r, c in motif) else 11
            n = int(rng.choice([v for v in range(minimum, 21)
                                if v % 2 == n % 2 and v not in emitted_sides]))
        emitted_sides.add(n)
        for _ in range(16):
            pair = _make_pair(rng, n, *spec)
            key = _fingerprint(pair["input"])
            if i == 1:
                first = _analyze(puzzle["train"][0]["input"])
                current = _analyze(pair["input"])
                if all(first[role] == current[role] for role in ("background", "edge", "glyph")):
                    continue
            if key not in seen:
                seen.add(key)
                puzzle["train" if i < num_train else "test"].append(pair)
                break
        else:
            raise RuntimeError("bounded uniqueness search exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_checks(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    grid, output = _grid(pair["input"]), _grid(pair["output"])
    _require(output.shape == (2, 2), "output must be 2x2")
    info = _analyze(grid)
    full = _exchange(info)
    _require(np.array_equal(output, _crop(full)), "incorrect exchange/parity crop")
    _require(np.any(output != info["background"]), "uninformative blank crop")
    other = full[:2, -2:] if info["n"] % 2 else full[:2, :2]
    _require(not np.array_equal(output, other), "parity does not affect this answer")
    _require(not np.array_equal(output, _crop(grid)), "exchange does not affect this answer")
    _require(np.count_nonzero(full == info["glyph"]) == 2 * len(info["motif"]),
             "glyph copies clipped or merged")
    _require(np.count_nonzero(full == info["edge"]) == len(info["figure"]),
             "figure pixels lost")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _pair_checks(pair)
            key = _fingerprint(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                train.append((pair, info))
    _require({info["n"] % 2 for _, info in train} == {0, 1}, "both parity witnesses required")
    if len(train) >= 4:
        _require(all(len({i["n"] for _, i in train if i["n"] % 2 == p}) >= 2 for p in (0, 1)),
                 "show each parity at two sizes")
    _require({len(info["centers"]) for _, info in train} == {2, 3}, "show both figure counts")
    _require(len({(i["background"], i["edge"], i["glyph"]) for _, i in train}) >= 2,
             "training must vary literal color roles")
    _require(any(i["edge"] in pair["output"] and i["glyph"] in pair["output"]
                 for pair, i in train), "show both exchange operations in one crop")
    _require(any(len(np.unique(pair["output"])) == 2 for pair, _ in train),
             "include a crop that is not the richest corner")
    _require(any(any(r < 2 and (c < 2 if i["n"] % 2 else c >= i["n"] - 2)
                         for r, c in i["figure"] - set(i["centers"]))
                 for _, i in train), "show a line interior, not just recolored centers")
    _require(any(not np.array_equal(pair["output"], _crop(_paint(
                     i["n"], i["background"], ((i["figure"], i["edge"]),
                                               (frozenset(i["endpoints"]), i["glyph"])))))
                 for pair, i in train), "show full glyph copying, not just endpoint dots")
    erasure_witness = False
    for pair, i in train:
        retained = np.asarray(pair["input"]).copy()
        for cells, color in ((i["figure"], i["edge"]), (i["stamps"], i["glyph"])):
            for r, c in cells:
                retained[r, c] = color
        erasure_witness |= not np.array_equal(_crop(retained), pair["output"])
    _require(erasure_witness, "show that original objects must be erased")
    for role in ("edge", "glyph"):
        # Omitting either half of B must change an observed training answer.
        _require(any(i[role] in pair["output"] for pair, i in train),
                 "an exchange operation is never answer-relevant")
    motifs = {info["motif"] for _, info in train}
    _require(all(_analyze(pair["input"])["motif"] in motifs for pair in puzzle["test"]),
             "test motifs need a training demonstration")


def _puzzle_hash(puzzle):
    data = {split: [{role: np.asarray(value).tolist() for role, value in pair.items()}
                    for pair in puzzle[split]] for split in ("train", "test")}
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """42 fixed host seeds, all 30 legal count combinations, no disk writes."""
    puzzles, structures, sizes, palettes, motifs = set(), set(), set(), set(), set()
    directions, glyph_counts, crop_roles = set(), set(), set()
    pairs, dual_crops, triangle_pairs = 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = _puzzle_hash(puzzle)
        _require(digest == _puzzle_hash(generate_puzzle(seed)), "nondeterministic seed")
        puzzles.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _pair_checks(pair)
                grid, out = pair["input"], pair["output"]
                _require(np.array_equal(solve(grid.tolist()), out), "list solver mismatch")
                structures.add(_fingerprint(grid, True))
                sizes.add(i["n"])
                palettes.add((i["background"], i["edge"], i["glyph"]))
                motifs.add(i["motif"])
                a, b = i["endpoints"]
                directions.add("h" if a[0] == b[0] else "v" if a[1] == b[1] else "d")
                glyph_counts.add(len(i["centers"]))
                crop_roles.add((split, i["n"] % 2, i["edge"] in out, i["glyph"] in out))
                pairs += 1
                dual_crops += i["edge"] in out and i["glyph"] in out
                triangle_pairs += len(i["centers"]) == 3
                # Literal colors carry no hidden convention.
                permutation = np.roll(np.arange(10, dtype=np.int8), 3)
                _require(np.array_equal(solve(permutation[grid]), permutation[out]),
                         "color equivariance failed")
                _require(np.array_equal(solve(grid.astype(np.int64)), out), "dtype dependency")
    _require(len(puzzles) == 42 and len(structures) >= 150, "insufficient structural diversity")
    _require(len(sizes) >= 18 and len(palettes) >= 100 and len(motifs) >= 7,
             "insufficient size, color or motif variation")
    _require(directions == {"h", "v", "d"} and glyph_counts == {2, 3}, "geometry diversity")
    _require(all(any(s == split and p == parity and edge and glyph
                     for s, p, edge, glyph in crop_roles)
                 for split in ("train", "test") for parity in (0, 1)), "missing dual readout")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(10000 + a * 10 + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "wrong counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    _manual_checks()
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(puzzles),
            "normalized_inputs": len(structures), "sizes": sorted(sizes),
            "semantic_palettes": len(palettes), "motifs": len(motifs),
            "source_directions": sorted(directions), "dual_crops": int(dual_crops),
            "triangles": int(triangle_pairs), "legal_count_combinations": 30}


def _manual_checks():
    # Analytic fixture: two 3x1 marks, vertical source, two ink roles in crop.
    n, bg, edge, glyph = 11, 0, 2, 5
    centers, endpoints = ((1, 1), (1, 9)), ((1, 0), (6, 0))
    grid = _paint(n, bg, ((_line(*endpoints), edge), (_stamps(centers, MOTIFS["vbar"]), glyph)))
    _require(np.array_equal(solve(grid), [[5, 0], [5, 2]]), "analytic odd witness")
    # Same objects and top-left pixels, one extra background row/column: parity
    # switches the readout to the right and moves its window by one column.
    padded = np.full((12, 12), bg, dtype=np.int8)
    padded[:11, :11] = grid
    _require(np.array_equal(solve(padded), [[0, 0], [0, 0]]), "analytic even boundary")
    mirrored = np.fliplr(padded)
    _require(np.array_equal(solve(mirrored), [[0, 5], [2, 5]]), "analytic reflected witness")
    before = grid.copy()
    solve(grid)
    _require(np.array_equal(grid, before), "solver mutated input")
    # Solver is broader than the informative generation subset: this crop can
    # legitimately be empty. No nonblank or mode condition belongs in solve.
    singleton_centers, singleton_endpoints = ((4, 5), (4, 9)), ((7, 2), (11, 2))
    fixture = _paint(15, 7, ((_line(*singleton_endpoints), 1),
                            (_stamps(singleton_centers, {(0, 0)}), 8)))
    _require(np.array_equal(solve(fixture), np.full((2, 2), 7)), "general singleton support")
    # An asymmetric motif has an occupied bounding-box anchor, not an ink
    # centroid. Its top-left copied pixel distinguishes a 180-degree turn.
    asymmetric = _paint(13, 0, ((_line((1, 1), (7, 1)), 2),
                                (_stamps(((4, 7), (4, 11)), MOTIFS["tee"]), 5)))
    _require(np.array_equal(solve(asymmetric), [[5, 5], [0, 5]]), "asymmetric literal copying")
    triangle = _paint(13, 0, ((_line((1, 0), (7, 0)), 2),
                            (_stamps(((0, 1), (0, 12), (11, 1)), {(0, 0)}), 5)))
    _require(np.array_equal(solve(triangle), [[0, 2], [5, 2]]), "triangle interior witness")
    _require(np.array_equal(solve(np.fliplr(triangle)), [[2, 2], [0, 2]]),
             "monochrome triangle readout")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
