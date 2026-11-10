"""Canonical divider lattices, colored by region-graph parity.

No files, parent generators, random state, or demonstrations are read by solve.
Generation and self_test retain only one bounded puzzle at a time.
"""
from collections import deque
import hashlib
import json

import numpy as np


ANCHOR = 5
ALTERNATE = 3
SELF_TEST_SEEDS = (
    0, 216042270, 370801927, 44508692, 870498364, 547587433, 94463759,
    23830781, 352843166, 945504204, 503865825, 197795344, 754186899,
    256663963, 932347949, 320395280, 198768088, 824202500, 29202119,
    217975760, 539403056, 811699357, 988165945, 58674128, 450614142,
    230542917, 213593355, 261142250, 577672292, 982499905, 536741487,
    619003433, 975117305, 317670969, 313678181, 771874930, 44718187,
    735325739, 91065241, 789950276, 343675816, 465319752,
)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be two-dimensional with sides in 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(np.all((a >= 0) & (a <= 9)), "grid colors must be in 0..9")
    return a.astype(np.int8, copy=True)


def _runs(a, axis):
    vectors = a if axis == 0 else a.T
    starts = np.r_[0, np.flatnonzero(np.any(vectors[1:] != vectors[:-1], axis=1)) + 1]
    return starts, np.diff(np.r_[starts, len(vectors)])


def _compress(a):
    # Select independently on the original grid. Deleting duplicate vectors on
    # the other axis cannot remove a distinguishing value, so the orders agree.
    rows, _ = _runs(a, 0)
    cols, _ = _runs(a, 1)
    return a[np.ix_(rows, cols)].copy()


def _lattice(c):
    """Read a compressed lattice using only its visible field and line values."""
    height, width = c.shape
    bg = int(c[-1, 0])
    _require(bg not in (ANCHOR, ALTERNATE), "field cannot use a reserved fill")
    _require(height % 2 == 1 and width % 2 == 1, "lattice sides must be odd")
    h, v = (height - 1) // 2, (width - 1) // 2
    _require(h + v > 0, "at least one effective divider is required")
    _require(np.all(c[::2, ::2] == bg), "field cross-sections disagree")
    horizontal = tuple(int(c[r, 0]) for r in range(1, height, 2))
    vertical = tuple(int(c[0, col]) for col in range(1, width, 2))
    _require(all(x not in (bg, ANCHOR, ALTERNATE) for x in horizontal + vertical),
             "line, field, and fill roles must be distinct")
    for i, color in enumerate(horizontal):
        _require(np.all(c[2 * i + 1, ::2] == color), "broken horizontal divider")
    for j, color in enumerate(vertical):
        _require(np.all(c[::2, 2 * j + 1] == color), "broken vertical divider")
    for i in range(h):
        for j in range(v):
            _require(int(c[2 * i + 1, 2 * j + 1]) in (horizontal[i], vertical[j]),
                     "crossing must retain a participating line color")
    return bg, horizontal, vertical


def _region_graph(c, bg):
    """Four-connected field regions; edges cross one divider segment.

    A crossing joins four wall segments, not the diagonally opposite regions.
    Thus a wall connected component is deliberately NOT one graph edge.
    """
    height, width = c.shape
    labels = np.full(c.shape, -1, dtype=np.int16)
    count = 0
    for r in range(height):
        for col in range(width):
            if c[r, col] != bg or labels[r, col] >= 0:
                continue
            labels[r, col] = count
            stack = [(r, col)]
            while stack:
                row, column = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, column + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and c[nr, nc] == bg and labels[nr, nc] < 0):
                        labels[nr, nc] = count
                        stack.append((nr, nc))
            count += 1
    edges = set()
    for r, col in np.argwhere(c != bg):
        for dr, dc in ((1, 0), (0, 1)):
            a, b = (r - dr, col - dc), (r + dr, col + dc)
            if 0 <= a[0] < height and 0 <= b[0] < height and 0 <= a[1] < width and 0 <= b[1] < width:
                first, second = int(labels[a]), int(labels[b])
                if first >= 0 and second >= 0 and first != second:
                    edges.add(tuple(sorted((first, second))))
    neighbors = [set() for _ in range(count)]
    for first, second in sorted(edges):
        neighbors[first].add(second)
        neighbors[second].add(first)
    root = int(labels[-1, 0])
    _require(root >= 0, "lower-left anchor must remain field")
    parity = [-1] * count
    parity[root] = 0
    queue = deque([root])
    while queue:
        region = queue.popleft()
        for other in sorted(neighbors[region]):
            if parity[other] < 0:
                parity[other] = 1 - parity[region]
                queue.append(other)
            else:
                _require(parity[other] != parity[region], "non-bipartite region graph")
    _require(all(x >= 0 for x in parity), "disconnected region graph")
    return labels, parity, edges


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Collapse equal adjacent vectors, then fill the resulting region graph."""
    c = _compress(_grid(input_grid))
    bg, _, _ = _lattice(c)
    labels, parity, _ = _region_graph(c, bg)
    output = c.copy()
    for region, bit in enumerate(parity):
        output[labels == region] = ANCHOR if bit == 0 else ALTERNATE
    return output


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer)) and low <= value <= high,
             f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.tobytes()


def _canonical(rng, h, v, bg, palette_mode, crossing_mode):
    available = np.array([x for x in range(10) if x not in (bg, ANCHOR, ALTERNATE)])
    total = h + v
    if palette_mode == 0 and total <= len(available):
        colors = rng.choice(available, total, replace=False)
    else:
        colors = rng.choice(available, total, replace=True)
        if palette_mode == 1:  # repeated parallel identities
            if h >= 2:
                colors[1] = colors[0]
            elif v >= 2:
                colors[h + 1] = colors[h]
        elif palette_mode == 2 and h and v:
            colors[h] = colors[0]
    horizontal, vertical = colors[:h], colors[h:]
    c = np.full((2 * h + 1, 2 * v + 1), bg, dtype=np.int8)
    for j, color in enumerate(vertical):
        c[:, 2 * j + 1] = color
    for i, color in enumerate(horizontal):
        c[2 * i + 1, :] = color
    if crossing_mode == 0:
        bits = np.zeros((h, v), dtype=np.int8)
    elif crossing_mode == 1:
        bits = np.ones((h, v), dtype=np.int8)
    elif crossing_mode == 2:
        bits = np.repeat(rng.integers(0, 2, (h, 1)), v, axis=1)
    elif crossing_mode == 3:
        bits = np.repeat(rng.integers(0, 2, (1, v)), h, axis=0)
    elif crossing_mode == 4:
        bits = rng.integers(0, 2, (h, v))
    else:
        bits = (np.arange(h)[:, None] + np.arange(v)[None, :]) % 2
    for i in range(h):
        for j in range(v):
            c[2 * i + 1, 2 * j + 1] = vertical[j] if bits[i, j] else horizontal[i]
    return c


def _expansion(rng, length, style, teaching=False, no_op=False):
    runs = np.ones(length, dtype=int)
    if no_op:
        return runs
    # Teaching includes duplicate FIELD vectors and duplicate DIVIDER vectors.
    if teaching:
        runs[int(rng.choice(np.arange(0, length, 2)))] += 1
        teaching_divider = int(rng.choice(np.arange(1, length, 2)))
        runs[teaching_divider] += 1
    room = 30 - int(runs.sum())
    low, high = (0, min(8, room)) if style == 0 else (min(7, room), room)
    extra = int(rng.integers(low, high + 1))
    if extra:
        if style == 2:
            # Concentrate repeated vectors inside, rather than at the margins.
            position = int(rng.integers(1, length - 1)) if length > 2 else 0
            runs[position] += extra
        elif style == 1:
            runs[int(rng.choice((0, length - 1)))] += extra
        else:
            weights = rng.random(length) + 0.1
            runs += rng.multinomial(extra, weights / weights.sum())
    if teaching and runs[teaching_divider] % 2:
        # An even-width wall explicitly contradicts counting individual wall
        # pixels as separate crossings; keep its required thickness >=2.
        runs[teaching_divider] -= 1
    return runs


def _make_pair(rng, h, v, bg, palette_mode, crossing_mode, style, teaching, used):
    for _ in range(16):
        c = _canonical(rng, h, v, bg, palette_mode, crossing_mode)
        no_op = -1 if teaching or rng.random() >= 0.22 else int(rng.integers(2))
        row_runs = _expansion(rng, len(c), style, teaching, no_op == 0)
        col_runs = _expansion(rng, c.shape[1], style, teaching, no_op == 1)
        if row_runs.sum() == len(c) and col_runs.sum() == c.shape[1]:
            target = row_runs if no_op != 0 else col_runs
            target[0] += 1
        source = np.repeat(np.repeat(c, row_runs, axis=0), col_runs, axis=1)
        if _key(source) not in used:
            break
    else:
        # At most 13 previous inputs; canonical sides <=13. At least 17
        # different row expansions fit, so this fallback cannot be exhausted.
        for first_run in range(2, 32 - c.shape[0]):
            row_runs = np.ones(len(c), dtype=int)
            col_runs = np.ones(c.shape[1], dtype=int)
            row_runs[0], col_runs[0] = first_run, 2
            source = np.repeat(np.repeat(c, row_runs, axis=0), col_runs, axis=1)
            if _key(source) not in used:
                break
        else:
            raise RuntimeError("uniqueness fallback exhausted")
    used.add(_key(source))
    # Separate, explicit lattice-distance construction of generated ground truth.
    # solve uses connected components and graph propagation instead.
    output = c.copy()
    for i in range(h + 1):
        for j in range(v + 1):
            output[2 * i, 2 * j] = ANCHOR if (h - i + j) % 2 == 0 else ALTERNATE
    return {"input": source, "output": output}


def _test_inventory(rng, training, count, regime):
    pool = [(h, v) for h in range(7) for v in range(7) if h + v]
    if len(training) == 2:
        # Two rectangular demonstrations cannot identify all four corners with
        # arbitrary learned fill phases. Keep every corner/phase alternative
        # consistent with training prediction-equivalent on the requested test.
        pool = [(h, v) for h, v in pool if (h + v) % 2 == 1]
    order = [pool[int(i)] for i in rng.permutation(len(pool))]
    max_h, max_v = max(h for h, _ in training), max(v for _, v in training)
    if regime == 0:  # more separators on at least one axis
        preferred = [x for x in order if x[0] > max_h or x[1] > max_v]
    elif regime == 1:  # transfer to one orientation, not a larger canvas
        preferred = [x for x in order if 0 in x and sum(x) <= 4]
    elif regime == 2:  # familiar inventory, new dilation concentration
        preferred = [x for x in order if x in training]
    else:  # recombine familiar inventory ranges with different crossing ownership
        preferred = [x for x in order if 0 < x[0] <= max_h and 0 < x[1] <= max_v]
    chosen = preferred + [x for x in order if x not in preferred]
    return chosen[:count]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate distinct examples with shared, input-readable conventions."""
    _require(not isinstance(seed, (bool, np.bool_))
             and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(4))
    backgrounds = [int(x) for x in rng.permutation([x for x in range(10) if x not in (3, 5)])]
    varying_background = bool(rng.integers(2))
    training = [(int(rng.choice((1, 3))), int(rng.choice((2, 4)))),
                (int(rng.choice((2, 4))), int(rng.choice((1, 3))))]
    if num_train >= 3:
        training.append((int(rng.choice((1, 3))), int(rng.choice((1, 3)))))
    remaining = [(h, v) for h in range(5) for v in range(5)
                 if h + v and (h, v) not in training]
    rng.shuffle(remaining)
    training.extend(remaining[:num_train - len(training)])
    testing = _test_inventory(rng, training, num_test, regime)
    second_crossing_owner = int(rng.integers(2))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, inventories in (("train", training), ("test", testing)):
        for index, (h, v) in enumerate(inventories):
            serial = index if split == "train" else num_train + index
            bg = backgrounds[serial % len(backgrounds)] if varying_background else backgrounds[0]
            teaching = split == "train" and index < 2
            palette_mode = 0 if teaching else int(rng.integers(3))
            crossing_mode = (5 if index == 0 else second_crossing_owner) if teaching else int(rng.integers(5))
            if split == "test" and regime == 3:
                crossing_mode = 1 - second_crossing_owner
            style = 0 if split == "train" else (2 if regime == 2 else int(rng.integers(3)))
            pair = _make_pair(rng, h, v, bg, palette_mode, crossing_mode, style, teaching, used)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
    source, output = _grid(pair["input"]), _grid(pair["output"])
    c = _compress(source)
    bg, horizontal, vertical = _lattice(c)
    h, v = len(horizontal), len(vertical)
    _, rows = _runs(source, 0)
    _, cols = _runs(source, 1)
    _require(np.array_equal(np.repeat(np.repeat(c, rows, axis=0), cols, axis=1), source),
             "contraction must reconstruct all source values losslessly")
    _require(np.array_equal(_compress(c), c), "compression must be idempotent")
    _require(np.array_equal(_compress(source.T).T, c), "axis contraction must commute")
    _require(output.shape == c.shape and source.shape != output.shape, "wrong output dimensions")
    _require(np.array_equal(output[c != bg], c[c != bg]), "divider or crossing changed")
    labels, parity, edges = _region_graph(c, bg)
    _require(len(parity) == (h + 1) * (v + 1), "unexpected number of field regions")
    _require(len(edges) == h * (v + 1) + v * (h + 1), "unexpected region adjacency")
    _require(int(output[-1, 0]) == ANCHOR, "wrong lower-left phase")
    for region, bit in enumerate(parity):
        _require(np.all(output[labels == region] == (ANCHOR if bit == 0 else ALTERNATE)),
                 "incorrect graph-parity coloring")
    # Each omission is witnessed directly, without latent construction state.
    _require(not np.array_equal(c, output), "compression alone must give a different answer")
    _require(np.array_equal(solve(source), output), "input-only solver disagreement")
    crossing_owners = set()
    for i in range(h):
        for j in range(v):
            if horizontal[i] != vertical[j]:
                crossing_owners.add(int(c[2 * i + 1, 2 * j + 1] == vertical[j]))
    return {"h": h, "v": v, "bg": bg, "rows": tuple(int(x) for x in rows),
            "cols": tuple(int(x) for x in cols), "owners": crossing_owners,
            "horizontal": horizontal, "vertical": vertical,
            "corners": tuple(int(output[r, col]) for r, col in ((0, 0), (0, -1), (-1, -1)))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _count(a, "num_train", 2, 10)
    _count(b, "num_test", 1, 4)
    _require(a > b, "num_train must exceed num_test")
    seen, analyses = set(), {}
    for split in ("train", "test"):
        analyses[split] = []
        for pair in puzzle[split]:
            info = _analyze(pair)
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "inputs must be distinct within a puzzle")
            seen.add(key)
            analyses[split].append(info)
    first = analyses["train"][0]
    _require(first["owners"] == {0, 1}, "training must demonstrate both crossing values")
    for info in analyses["train"][:2]:
        for runs in (info["rows"], info["cols"]):
            _require(max(runs[::2]) > 1 and max(runs[1::2]) > 1,
                     "training must compress both fields and dividers on both axes")
            _require(any(x % 2 == 0 for x in runs[1::2]),
                     "training must contradict pixel-count crossing parity")
        _require(len(set(info["horizontal"] + info["vertical"])) == info["h"] + info["v"],
                 "teaching lines must have distinct identities")
    _require(len({(x["h"], x["v"]) for x in analyses["test"]}) == b,
             "multiple tests must have different inventories")
    # Test every corner AND both phase choices using training only. A surviving
    # alternative may remain, but must not produce a different requested test.
    for corner in range(3):
        for color in (ANCHOR, ALTERNATE):
            if all(x["corners"][corner] == color for x in analyses["train"]):
                _require(all(x["corners"][corner] == color for x in analyses["test"]),
                         "training leaves a test-relevant corner/phase ambiguity")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded checks; return small descriptive measurements, write no files."""
    fingerprints, shapes, output_shapes, layouts, inventories, backgrounds = set(), set(), set(), set(), set(), set()
    one_axis = single_orientation = both_axes = crossing_cycles = pairs = 0
    for index, seed in enumerate(SELF_TEST_SEEDS):
        ntrain = 2 + index % 9
        ntest = 1 + index % min(4, ntrain - 1)
        puzzle = generate_puzzle(seed, ntrain, ntest)
        signature = _fingerprint(puzzle)
        _require(signature == _fingerprint(generate_puzzle(seed, ntrain, ntest)), "nondeterminism")
        fingerprints.add(signature)
        for pair in puzzle["train"] + puzzle["test"]:
            info = _analyze(pair)
            h, v = info["h"], info["v"]
            shapes.add(pair["input"].shape)
            output_shapes.add(pair["output"].shape)
            inventories.add((h, v))
            backgrounds.add(info["bg"])
            layouts.add((h, v, info["rows"], info["cols"]))
            shrink_r, shrink_c = max(info["rows"]) > 1, max(info["cols"]) > 1
            one_axis += shrink_r != shrink_c
            both_axes += shrink_r and shrink_c
            single_orientation += h == 0 or v == 0
            crossing_cycles += h > 0 and v > 0
            pairs += 1
    _require(len(fingerprints) == 42 and len(shapes) >= 60 and len(inventories) >= 20,
             "insufficient seed, size, or structural diversity")
    _require(len(layouts) >= 150 and len(backgrounds) == 8 and len(output_shapes) >= 20,
             "insufficient spacing, palette, or output variation")
    _require(one_axis >= 10 and single_orientation >= 10 and crossing_cycles >= 80,
             "missing no-op-axis, parallel, or crossed-lattice regimes")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(991, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Hand-check an exact equal-vector contraction with mixed crossing owners.
    raw = np.array([[0, 0, 2, 0, 0, 4, 0], [0, 0, 2, 0, 0, 4, 0],
                    [1, 1, 2, 1, 1, 1, 1], [1, 1, 2, 1, 1, 1, 1],
                    [0, 0, 2, 0, 0, 4, 0]], dtype=np.int8)
    expected = np.array([[3, 2, 5, 4, 3], [1, 2, 1, 1, 1], [5, 2, 3, 4, 5]], dtype=np.int8)
    _require(np.array_equal(solve(raw), expected), "hand fixture failed")
    c = _compress(raw)
    _require(np.array_equal(solve(c), expected), "canonical no-op compression failed")
    # Demonstrate retained nonadjacent repeats: top and bottom vectors equal.
    _require(np.array_equal(c[0], c[-1]) and len(c) == 3, "global dedup regression")
    # The solver covers the specified family beyond the sampler's six-line cap.
    for h, v in ((14, 14), (0, 14), (14, 0)):
        maximum = _canonical(np.random.default_rng(21), h, v, 0, 1, 4)
        actual = solve(maximum)
        for i in range(h + 1):
            for j in range(v + 1):
                _require(actual[2 * i, 2 * j] == (ANCHOR if (h - i + j) % 2 == 0 else ALTERNATE),
                         "maximum family inventory failed")
        _require(np.array_equal(actual[maximum != 0], maximum[maximum != 0]),
                 "maximum family crossing preservation failed")
    # Force random uniqueness retries to fail, then exercise deterministic fallback.
    class CollisionSet(set):
        def __contains__(self, key):
            if getattr(self, "calls", 0) < 16:
                self.calls = getattr(self, "calls", 0) + 1
                return True
            return super().__contains__(key)
    fallback = _make_pair(np.random.default_rng(12), 6, 6, 0, 1, 4, 0, False, CollisionSet())
    _analyze(fallback)
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "inventories": len(inventories), "expansion_layouts": len(layouts),
            "field_colors": len(backgrounds), "one_axis_compression": one_axis,
            "two_axis_compression": both_axes, "single_orientation": single_orientation,
            "crossed_lattices": crossing_cycles, "independent_semantic_review": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
