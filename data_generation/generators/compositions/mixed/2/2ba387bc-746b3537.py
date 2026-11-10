"""Rank-pair hollow/solid squares, then collapse equal adjacent pixel bands.

Only standard-library modules and NumPy are used. No evidence files, saved
examples, seeds, or generator metadata are consulted by solve().
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"invalid {name}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _tile(size, color, bg, kind):
    tile = np.full((size, size), color, dtype=np.int8)
    if kind == 0:  # hollow left; solid right
        tile[1:-1, 1:-1] = bg
    return tile


def _parse(value):
    grid = _grid(value)
    bg = int(grid[0, 0])
    if not all(np.all(edge == bg) for edge in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError("input must have a complete background border")
    remaining = set(zip(*np.nonzero(grid != bg)))
    objects, size = [], None
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], [start]
        while todo:
            r, c = todo.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
                    component.append(p)
        r = int(min(p[0] for p in component))
        c = int(min(p[1] for p in component))
        bottom = int(max(p[0] for p in component)) + 1
        right = int(max(p[1] for p in component)) + 1
        s = bottom - r
        if s < 3 or right-c != s or (size is not None and size != s):
            raise ValueError("objects must be squares of one common size >= 3")
        size = s
        tile = grid[r:bottom, c:right]
        color = int(tile[0, 0])
        if color == bg:
            raise ValueError("malformed tile")
        if np.all(tile == color):
            kind = 1
        elif np.array_equal(tile, _tile(s, color, bg, 0)):
            kind = 0
        else:
            raise ValueError("only solid squares and one-cell hollow rims are allowed")
        neighborhood = grid[r-1:bottom+1, c-1:right+1].copy()
        neighborhood[1:-1, 1:-1] = bg
        if not np.all(neighborhood == bg):
            raise ValueError("tiles need a complete background gap")
        objects.append(dict(top=r, left=c, kind=kind, color=color))
    if not objects:
        raise ValueError("no tiles")
    objects.sort(key=lambda o: (o["top"], o["left"]))
    for kind in (0, 1):
        q = [o for o in objects if o["kind"] == kind]
        if not q or len({o["top"] for o in q}) != len(q):
            raise ValueError("both classes and distinct same-class top rows are required")
        if len(q)*size > 30:
            raise ValueError("uncompressed queue height exceeds 30")
    if 2*size > 30:
        raise ValueError("uncompressed queue width exceeds 30")
    return dict(grid=grid, bg=bg, size=size, objects=objects)


def _queues(a, order="vertical"):
    queues = [[o for o in a["objects"] if o["kind"] == k] for k in (0, 1)]
    if order in ("horizontal", "horizontal_reverse"):
        return [sorted(q, key=lambda o: (o["left"], o["top"]),
                       reverse=order == "horizontal_reverse") for q in queues]
    return queues


def _assemble(a, queues=None):
    queues = _queues(a) if queues is None else queues
    s = a["size"]
    result = np.full((s*max(map(len, queues)), 2*s), a["bg"], dtype=np.int8)
    for col, queue in enumerate(queues):
        for rank, o in enumerate(queue):
            result[rank*s:(rank+1)*s, col*s:(col+1)*s] = _tile(
                s, o["color"], a["bg"], o["kind"])
    return result


def _starts(grid, axis):
    """First index of every maximal run of equal full rows or columns."""
    vectors = grid if axis == 0 else grid.T
    return np.flatnonzero(np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)])


def _collapse(grid, rows=True, columns=True):
    # Both masks are computed on the same image, independently. The two
    # operations also commute: discarded equal copies cannot hide a difference.
    rr = _starts(grid, 0) if rows else np.arange(grid.shape[0])
    cc = _starts(grid, 1) if columns else np.arange(grid.shape[1])
    return grid[np.ix_(rr, cc)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Parse visible square instances, rank-pair them, then compress the image."""
    return _collapse(_assemble(_parse(input_grid)))


def _greedy(a):
    slots = [[None, None]]
    for o in a["objects"]:
        k = o["kind"]
        if slots[-1][k] is not None:
            slots.append([None, None])
        slots[-1][k] = o
    s = a["size"]
    result = np.full((len(slots)*s, 2*s), a["bg"], dtype=np.int8)
    for rank, row in enumerate(slots):
        for col, o in enumerate(row):
            if o is not None:
                result[rank*s:(rank+1)*s, col*s:(col+1)*s] = _tile(
                    s, o["color"], a["bg"], col)
    return result


def _distinct_vectors(grid, axis):
    vectors = grid if axis == 0 else grid.T
    seen, indices = set(), []
    for i, v in enumerate(vectors):
        key = v.tobytes()
        if key not in seen:
            seen.add(key)
            indices.append(i)
    return np.asarray(indices, dtype=int)


def _evidence(a):
    q = _assemble(a)
    out = _collapse(q)
    queues = _queues(a)
    s = a["size"]
    local = np.concatenate([_collapse(q[i:i+s], columns=False)
                            for i in range(0, len(q), s)], axis=0)
    local = _collapse(local, rows=False)
    local_columns = np.concatenate([_collapse(q[:, :s], rows=False),
                                    _collapse(q[:, s:], rows=False)], axis=1)
    local_columns = _collapse(local_columns, columns=False)
    unique = q[np.ix_(_distinct_vectors(q, 0), _distinct_vectors(q, 1))]
    return dict(
        greedy=not np.array_equal(out, _collapse(_greedy(a))),
        horizontal=not np.array_equal(out, _collapse(_assemble(a, _queues(a, "horizontal")))),
        horizontal_reverse=not np.array_equal(out, _collapse(_assemble(a, _queues(a, "horizontal_reverse")))),
        reverse_hollow=not np.array_equal(out, _collapse(_assemble(a, [queues[0][::-1], queues[1]]))),
        reverse_solid=not np.array_equal(out, _collapse(_assemble(a, [queues[0], queues[1][::-1]]))),
        color_order=not np.array_equal(out, _collapse(_assemble(a, [
            sorted(v, key=lambda o: o["color"]) for v in queues]))),
        color_reverse=not np.array_equal(out, _collapse(_assemble(a, [
            sorted(v, key=lambda o: o["color"], reverse=True) for v in queues]))),
        local_rows=not np.array_equal(out, local),
        local_columns=not np.array_equal(out, local_columns),
        global_unique=not np.array_equal(out, unique),
        no_assembly=not np.array_equal(out, _collapse(a["grid"])),
        row_reduction=q.shape[0]-out.shape[0],
        column_reduction=q.shape[1]-out.shape[1],
        seam_merge=out.shape[1] == 3,
        partial_match=(out.shape[1] == 4 and any(
            h["color"] == s["color"] for h, s in zip(*queues))),
        counts=tuple(len(v) for v in queues),
    )


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _layout(rng, size, count):
    """A constructive separated packing followed by bounded free relocation."""
    extent = 28  # reserve a one-cell background border on all sides
    capacity = (extent+1)//(size+1)
    feasible = [b for b in range(1, count+1)
                if (count+b-1)//b <= capacity and b*size+count-1 <= extent]
    bands = int(rng.choice(feasible))
    amounts = [count//bands + int(i < count % bands) for i in range(bands)]
    rng.shuffle(amounts)
    # Move capacity between shelves to avoid a fixed lattice.
    for _ in range(2*count):
        i, j = (int(v) for v in rng.integers(0, bands, size=2))
        if i != j and amounts[i] > 1 and amounts[j] < capacity:
            amounts[i] -= 1
            amounts[j] += 1
    min_h = bands*size+count-1
    min_w = max(amounts)*(size+1)-1
    mode = int(rng.integers(5))
    if mode == 0:
        h, w = min_h, min_w
    elif mode == 1:
        h = w = int(rng.integers(max(min_h, min_w), extent+1))
    elif mode == 2:
        h, w = extent, int(rng.integers(min_w, extent+1))
    elif mode == 3:
        h, w = int(rng.integers(min_h, extent+1)), extent
    else:
        h, w = int(rng.integers(min_h, extent+1)), int(rng.integers(min_w, extent+1))
    gaps = rng.multinomial(h-min_h, np.full(bands+1, 1/(bands+1)))
    top, positions = int(gaps[0]), []
    for b, n in enumerate(amounts):
        xg = rng.multinomial(w-(n*(size+1)-1), np.full(n+1, 1/(n+1)))
        left = int(xg[0])
        offsets = rng.permutation(n)
        for j in range(n):
            positions.append((top+int(offsets[j]), left))
            left += size+1+int(xg[j+1])
        top += size+n+int(gaps[b+1])
    rr, cc = np.indices((h-size+1, w-size+1))
    for _ in range(2):
        for i in rng.permutation(count):
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(positions):
                if j != i:
                    legal &= ((np.abs(rr-r) > size) | (np.abs(cc-c) > size))
                    legal &= rr != r
            sites = np.argwhere(legal)
            r, c = sites[int(rng.integers(len(sites)))]
            positions[int(i)] = int(r), int(c)
    return (h+2, w+2), sorted((r+1, c+1) for r, c in positions)


def _sequence(rng, counts, witness=False):
    h, s = counts
    mode = int(rng.integers(4))
    first = int(rng.integers(2))
    if witness or mode == 0:
        result = [first]*counts[first] + [1-first]*counts[1-first]
    elif mode == 1:
        result, remaining = [], [h, s]
        while sum(remaining):
            for k in (first, 1-first):
                if remaining[k]:
                    result.append(k)
                    remaining[k] -= 1
    else:
        result = [int(k) for k in rng.permutation([0]*h+[1]*s)]
    return result


def _color_sequences(rng, counts, bg, style):
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != bg])]
    if style == "boundary":
        sequences = []
        for k, n in enumerate(counts):
            # Early rank pairs share a color, but the last pair differs.
            # Local seam equality must not override full-column inequality.
            a, b = palette[0], palette[k+1]
            seq = [a, a, b] + [int(rng.choice((a, b))) for _ in range(n-3)]
            seq[-1] = b
            sequences.append(seq)
        return sequences
    if style == "aligned":
        n = counts[0]
        seq = palette[:2] + ([palette[0]] if n >= 3 else [])
        seq += [int(rng.choice(palette[:min(n, 5)])) for _ in range(n-len(seq))]
        return [seq[:], seq[:]]
    pool = palette[:int(rng.integers(1, min(9, sum(counts))+1))]
    mode = int(rng.integers(4))
    sequences = []
    for n in counts:
        if mode == 0:
            seq = [int(rng.choice(pool))]*n
        elif mode == 1:
            seq = [pool[i % len(pool)] for i in range(n)]
        else:
            seq = [int(rng.choice(pool)) for _ in range(n)]
        if style == "tail" and n == max(counts) and abs(counts[0]-counts[1]) >= 2:
            # Long solid tails may collapse across whole rank slots. Hollow
            # tails still retain separated rim/interior/rim rows.
            seq[min(counts):] = [seq[-1]]*(n-min(counts))
        sequences.append(seq)
    return sequences


def _make_pair(rng, size, counts, style, witness=False):
    bg = int(rng.integers(10))
    shape, positions = _layout(rng, size, sum(counts))
    kinds = _sequence(rng, counts, witness)
    colors = _color_sequences(rng, counts, bg, style)
    ranks = [0, 0]
    grid = np.full(shape, bg, dtype=np.int8)
    for (r, c), kind in zip(positions, kinds):
        color = colors[kind][ranks[kind]]
        ranks[kind] += 1
        grid[r:r+size, c:c+size] = _tile(size, color, bg, kind)
    return dict(input=grid, output=solve(grid))


def _fallback_demo(rng, sign):
    # Deterministic geometry with observable nonmonotone horizontal ordering.
    bg = int(rng.integers(10))
    p = [int(c) for c in rng.permutation([c for c in range(10) if c != bg])]
    kinds = ([0]*3+[1]*4) if sign > 0 else ([1]*3+[0]*4)
    colors = [p[0], p[0], p[1], p[0], p[0], p[3], p[4]]
    positions = [(1, 1), (2, 11), (5, 6), (7, 1), (9, 14), (12, 6), (14, 1)]
    grid = np.full((18, 19), bg, dtype=np.int8)
    for (r, c), k, color in zip(positions, kinds, colors):
        grid[r:r+3, c:c+3] = _tile(3, color, bg, k)
    return dict(input=grid, output=solve(grid))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    sign = 1 if rng.integers(2) else -1  # sign of solids minus hollows
    puzzle, seen, padding_signs = dict(train=[], test=[]), set(), set()
    for index in range(num_train+num_test):
        training = index < num_train
        witness = index == 0
        first_test = index == num_train
        if witness:
            size = int(rng.integers(3, 7))
            small = int(rng.integers(3, min(5, 30//size-1)+1))
            counts = (small, small+1) if sign > 0 else (small+1, small)
            style = "boundary"
        elif index == 1:
            size = int(rng.integers(3, 9))
            # A,B,A in both queues teaches recurrence and refutes either
            # numeric-color sort, while their equal signatures merge the seam.
            n = int(rng.integers(3, min(4, 30//size)+1))
            counts, style = (n, n), "aligned"
        elif first_test:
            size = int(rng.integers(3, 9))
            limit = 30//size
            small = int(rng.integers(1, limit-1))
            large = int(rng.integers(small+2, limit+1))
            counts = (small, large) if sign > 0 else (large, small)
            style = "tail"
        else:
            size = int(rng.integers(3, 9))
            limit = 30//size
            if not training and rng.random() < 0.3:
                n = int(rng.integers(1, limit+1))
                counts, style = (n, n), "aligned"
            else:
                n = int(rng.integers(1, limit+1))
                delta = int(rng.integers(-1, 2))
                if not training and delta and delta not in padding_signs:
                    delta = sign
                counts = (n, max(1, min(limit, n+delta)))
                style = "free"
        for _ in range(64):
            pair = _make_pair(rng, size, counts, style, witness)
            if _key(pair["input"]) in seen:
                continue
            if witness:
                evidence = _evidence(_parse(pair["input"]))
                if not all(evidence[k] for k in ("greedy", "horizontal", "horizontal_reverse",
                                                 "reverse_hollow", "reverse_solid", "local_rows")):
                    continue
            break
        else:
            if not witness:
                raise RuntimeError("bounded unique input construction exhausted")
            pair = _fallback_demo(rng, sign)
        seen.add(_key(pair["input"]))
        puzzle["train" if training else "test"].append(pair)
        if training:
            padding_signs.add(int(np.sign(counts[1]-counts[0])))
    # Shuffle demonstrations so evidence has no prescribed presentation order.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible geometry, exact compression, and demonstrative witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence = set(), []
    train_signs, test_counts, train_gaps = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            a = _parse(inp)
            q = _assemble(a)
            rr, cc = _starts(q, 0), _starts(q, 1)
            assert np.array_equal(out, q[np.ix_(rr, cc)]), "incorrect assembled-band compression"
            # Expanding each surviving vector run reconstructs every pixel of
            # the assembled image. These lengths are computed, never stored.
            row_lengths = np.diff(np.r_[rr, q.shape[0]])
            col_lengths = np.diff(np.r_[cc, q.shape[1]])
            reconstructed = np.repeat(np.repeat(out, row_lengths, axis=0), col_lengths, axis=1)
            assert np.array_equal(q, reconstructed), "compression changed colors or order"
            assert set(np.unique(out)) == set(np.unique(q)), "colors must be preserved"
            assert len(_starts(out, 0)) == out.shape[0]
            assert len(_starts(out, 1)) == out.shape[1]
            queues = _queues(a)
            counts = tuple(len(v) for v in queues)
            # With this square vocabulary the left edge and right edge are
            # separated equal vectors. Only the seam can merge across classes.
            same_queue = [o["color"] for o in queues[0]] == [o["color"] for o in queues[1]]
            assert out.shape[1] == (3 if same_queue else 4)
            assert np.array_equal(_collapse(q), _collapse(_collapse(q, columns=False), rows=False))
            assert np.array_equal(_collapse(q), _collapse(_collapse(q, rows=False), columns=False))
            assert not np.array_equal(inp, out)
            if split == "train":
                evidence.append(_evidence(a))
                train_signs.add(int(np.sign(counts[1]-counts[0])))
                train_gaps.append(abs(counts[1]-counts[0]))
            else:
                test_counts.append(counts)
    for name in ("greedy", "horizontal", "horizontal_reverse", "reverse_hollow", "reverse_solid",
                 "color_order", "color_reverse", "local_rows", "local_columns",
                 "global_unique", "no_assembly", "seam_merge", "partial_match"):
        assert any(e[name] for e in evidence), f"missing training evidence: {name}"
    assert any(e["row_reduction"] > 0 and e["column_reduction"] > 0 for e in evidence)
    assert any(not e["seam_merge"] for e in evidence), "teach unequal full-column signatures"
    assert max(train_gaps) == 1, "training should teach one-slot padding"
    assert any(abs(s-h) >= 2 and int(np.sign(s-h)) in train_signs for h, s in test_counts), \
        "missing longer-tail transfer with a demonstrated padding side"
    assert all(s == h or int(np.sign(s-h)) in train_signs for h, s in test_counts), \
        "test padding side must be demonstrated"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for key in ("input", "output"):
                shape, data = _key(pair[key])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """42 seeds, streamed in memory; no files, traces, workers, or saved runs."""
    shapes, layouts, count_pairs, sizes, backgrounds, signatures = set(), set(), set(), set(), set(), set()
    output_shapes, normal_outputs, indicators = set(), set(), Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _parse(pair["input"])
                q, out = _assemble(a), pair["output"]
                assert np.array_equal(solve(pair["input"].tolist()), out)
                shapes.add(pair["input"].shape)
                sizes.add(a["size"])
                backgrounds.add(a["bg"])
                queues = _queues(a)
                counts = tuple(map(len, queues))
                count_pairs.add(counts)
                layouts.add((a["size"], tuple((o["top"], o["left"], o["kind"]) for o in a["objects"])))
                output_shapes.add(out.shape)
                roles = {}
                norm = tuple(tuple(roles.setdefault(int(v), len(roles)) for v in row) for row in out)
                normal_outputs.add(norm)
                indicators["pairs"] += 1
                indicators["seam_merge"] += int(out.shape[1] == 3)
                indicators["no_seam_merge"] += int(out.shape[1] == 4)
                indicators["black_tiles"] += int(any(o["color"] == 0 for o in a["objects"]))
                indicators["repeated_tile_colors"] += int(len({o["color"] for o in a["objects"]}) < len(a["objects"]))
                indicators["solid_tail"] += int(counts[1] > counts[0])
                indicators["hollow_tail"] += int(counts[0] > counts[1])
                indicators["balanced"] += int(counts[0] == counts[1])
                # Color relabeling, changing scale, and both axis orders must
                # preserve the same semantics. Relabeling includes background.
                permutation = np.random.default_rng(seed+700).permutation(10)
                assert np.array_equal(solve(permutation[pair["input"]]), permutation[out])
                assert np.array_equal(_collapse(out), out)
                assert np.array_equal(_collapse(q), _collapse(_collapse(q, columns=False), rows=False))
    assert len(signatures) == 42
    assert sizes == set(range(3, 9)) and backgrounds == set(range(10))
    assert len(shapes) >= 50 and len(layouts) >= 180 and len(count_pairs) >= 25
    assert len(output_shapes) >= 15 and len(normal_outputs) >= 90
    assert all(indicators[k] >= 10 for k in ("seam_merge", "no_seam_merge", "black_tiles",
                                           "solid_tail", "hollow_tail", "balanced"))
    # Hand-computable witnesses separate adjacency from global uniqueness.
    q = np.array([[1, 1, 2, 1], [1, 1, 2, 1], [3, 3, 4, 3],
                  [1, 1, 2, 1]], dtype=np.int8)
    expected = np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]], dtype=np.int8)
    assert np.array_equal(_collapse(q), expected)
    # Exact queue witnesses used in task_spec.md: cross-rank row merging,
    # partial versus complete column equality, and separated color recurrence.
    for hollow, solid, rows in (
            ((2, 2, 5), (2, 2, 8),
             ("2222", "2022", "2222", "2022", "2222", "5558", "5058", "5558")),
            ((2, 5, 2), (2, 5, 2),
             ("222", "202", "222", "555", "505", "555", "222", "202", "222"))):
        q = np.concatenate([np.concatenate([_tile(4, c, 0, k) for c in queue], axis=0)
                            for k, queue in enumerate((hollow, solid))], axis=1)
        expected = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
        assert np.array_equal(_collapse(q), expected)
        assert not np.array_equal(_collapse(q, rows=False), expected)
        assert not np.array_equal(_collapse(q, columns=False), expected)
    # The solver accepts feasible larger squares beyond the sampler's 3..8.
    for size in (3, 4, 8, 9, 13):
        grid = np.full((size+2, 2*size+3), 7, dtype=np.int8)
        grid[1:1+size, 1:1+size] = _tile(size, 0, 7, 0)
        grid[1:1+size, size+2:2*size+2] = _tile(size, 0, 7, 1)
        assert np.array_equal(solve(grid), np.array([[0, 0, 0], [0, 7, 0], [0, 0, 0]], dtype=np.int8))
    # Every legal explicit count request, and type/range rejection.
    legal_counts = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1009+nt*5+ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
            legal_counts += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.0, 1), (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for bad_seed in (-1, 0.5, True):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    for sign in (-1, 1):
        fallback = _fallback_demo(np.random.default_rng(6), sign)
        e = _evidence(_parse(fallback["input"]))
        assert all(e[k] for k in ("greedy", "horizontal", "horizontal_reverse", "local_rows",
                                  "reverse_hollow", "reverse_solid"))
    damaged = generate_puzzle(0)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                layouts=len(layouts), count_pairs=len(count_pairs), sizes=sorted(sizes),
                backgrounds=sorted(backgrounds), output_shapes=len(output_shapes),
                color_normalized_outputs=len(normal_outputs), legal_count_requests=legal_counts,
                indicators=dict(indicators), independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
