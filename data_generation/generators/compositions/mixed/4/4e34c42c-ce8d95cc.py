"""Register stretched weave fragments, then collapse equal full-vector runs.

Registration keys are repeated, literal 2x2 patches with four distinct foreground
colors. They are ordinary retained picture cells. No seed or examples enter solve.
Only standard-library modules and NumPy are used; this module writes no files.
"""
from itertools import permutations

import numpy as np


HOST_SEEDS = (0, 231736685, 689593815, 918267934, 784167426, 892205501,
              983126505, 280509532, 440843987, 343839494, 686217438, 144319100,
              982072793, 344316925, 396168432, 330666100, 712002705, 780204992,
              181017095, 181832469, 162482135, 518514690, 644193628, 791089990,
              774395576, 130167595, 218233339, 871962133, 182733364, 706816251,
              749361107, 407416441, 674082149, 176738267, 447948411, 586054519,
              943501577, 510366454, 973075561, 303873258, 547555906, 891031504)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be 1..30 by 1..30 integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _background(a):
    values, counts = np.unique(a, return_counts=True)
    if counts.max() * 2 <= a.size:
        raise ValueError("background must occupy more than half of input")
    return int(values[counts.argmax()])


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _crop(a, bg):
    rr, cc = np.nonzero(a != bg)
    if not len(rr):
        raise ValueError("empty picture")
    return a[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _components(a, bg):
    result = []
    for group in _groups(zip(*np.nonzero(a != bg))):
        r0, c0 = min(r for r, c in group), min(c for r, c in group)
        h, w = max(r for r, c in group)-r0+1, max(c for r, c in group)-c0+1
        part = np.full((h, w), bg, dtype=np.int8)
        for r, c in group:
            part[r-r0, c-c0] = a[r, c]
        result.append(part)
    return result


def _signatures(a, bg):
    result = {}
    for r in range(a.shape[0]-1):
        for c in range(a.shape[1]-1):
            pattern = tuple(int(v) for v in a[r:r+2, c:c+2].flat)
            if bg not in pattern and len(set(pattern)) == 4:
                result.setdefault(pattern, []).append((r, c))
    return result


def _compress(a):
    # Compute BOTH equivalence partitions on the complete original rectangle.
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return a[np.ix_(rows, cols)].copy()


def _assemble(parts, bg):
    if not 2 <= len(parts) <= 6:
        raise ValueError("need two to six separated fragments")
    occurrences = {}
    for i, part in enumerate(parts):
        for pattern, places in _signatures(part, bg).items():
            if len(places) != 1:
                raise ValueError("a registration key recurs within a fragment")
            occurrences.setdefault(pattern, []).append((i, places[0]))
    adjacency = [[] for _ in parts]
    for copies in occurrences.values():
        for (i, (r, c)), (j, (u, v)) in zip(copies, copies[1:]):
            delta = r-u, c-v
            adjacency[i].append((j, delta))
            adjacency[j].append((i, (-delta[0], -delta[1])))
    offsets = {0: (0, 0)}
    stack = [0]
    while stack:
        i = stack.pop()
        for j, (dr, dc) in adjacency[i]:
            candidate = offsets[i][0]+dr, offsets[i][1]+dc
            if j in offsets:
                if offsets[j] != candidate:
                    raise ValueError("inconsistent registration keys")
            else:
                offsets[j] = candidate
                stack.append(j)
    if len(offsets) != len(parts):
        raise ValueError("registration graph is disconnected")
    r0 = min(r for r, c in offsets.values())
    c0 = min(c for r, c in offsets.values())
    offsets = [(offsets[i][0]-r0, offsets[i][1]-c0) for i in range(len(parts))]
    h = max(r+p.shape[0] for p, (r, c) in zip(parts, offsets))
    w = max(c+p.shape[1] for p, (r, c) in zip(parts, offsets))
    if max(h, w) > 30:
        raise ValueError("assembled rectangle exceeds 30")
    # Each component is a rectangular observation, so observed holes agree too.
    observed = np.full((h, w), -1, dtype=np.int8)
    union = np.full((h, w), bg, dtype=np.int8)
    for p, (r, c) in zip(parts, offsets):
        previous = observed[r:r+p.shape[0], c:c+p.shape[1]]
        if np.any((previous >= 0) & (previous != p)):
            raise ValueError("conflicting foreground or observed negative space")
        previous[:] = p
        view = union[r:r+p.shape[0], c:c+p.shape[1]]
        view[p != bg] = p[p != bg]
    for places in _signatures(union, bg).values():
        if len(places) != 1:
            raise ValueError("same registration key denotes two union positions")
    return _crop(union, bg), offsets


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate by repeated signatures; preserve holes; contract full vectors."""
    a = _grid(input_grid)
    bg = _background(a)
    union, _ = _assemble(_components(a, bg), bg)
    return _crop(_compress(union), bg)


def _axis_runs(a, axis):
    vectors = a if axis == 0 else a.T
    starts, lengths = [0], []
    for i in range(1, len(vectors)):
        if not np.array_equal(vectors[i], vectors[i-1]):
            lengths.append(i-starts[-1])
            starts.append(i)
    lengths.append(len(vectors)-starts[-1])
    return starts, lengths


def _recurs(a, axis):
    vectors = a if axis == 0 else a.T
    return len({v.tobytes() for v in vectors}) < len(vectors)


def _analyze(pair):
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        raise ValueError("pair must have exactly input and output")
    a, output = _grid(pair["input"]), _grid(pair["output"])
    bg = _background(a)
    parts = _components(a, bg)
    union, offsets = _assemble(parts, bg)
    expected = _crop(_compress(union), bg)
    if not np.array_equal(output, expected):
        raise ValueError("wrong registration, union, run collapse, or crop")
    row_starts, row_runs = _axis_runs(union, 0)
    col_starts, col_runs = _axis_runs(union, 1)
    # Run expansion is a lossless check including holes and every crossing.
    assert np.array_equal(np.repeat(np.repeat(expected, row_runs, 0), col_runs, 1), union)
    assert set(np.unique(a)) == set(np.unique(output)), "palette lost"
    assert np.any(output[0] != bg) and np.any(output[-1] != bg)
    assert np.any(output[:, 0] != bg) and np.any(output[:, -1] != bg)
    assert len(_groups(zip(*np.nonzero(output != bg)))) == 1
    assert np.array_equal(_compress(output), output)
    local_false_equalities = [0, 0]
    for p, (r, c) in zip(parts, offsets):
        for i in range(1, p.shape[0]):
            local_false_equalities[0] += int(np.array_equal(p[i], p[i-1])
                                           and not np.array_equal(union[r+i], union[r+i-1]))
        for j in range(1, p.shape[1]):
            local_false_equalities[1] += int(np.array_equal(p[:, j], p[:, j-1])
                                           and not np.array_equal(union[:, c+j], union[:, c+j-1]))
    try:
        premature, _ = _assemble([_compress(p) for p in parts], bg)
        premature_fails = not np.array_equal(_crop(_compress(premature), bg), output)
    except ValueError:
        premature_fails = True
    contributions = []
    for i, (p, (r, c)) in enumerate(zip(parts, offsets)):
        cover = np.zeros(union.shape, dtype=np.int8)
        for j, (q, (u, v)) in enumerate(zip(parts, offsets)):
            if j != i:
                cover[u:u+q.shape[0], v:v+q.shape[1]] |= (q != bg)
        contributions.append(int(np.count_nonzero((p != bg) &
                              (cover[r:r+p.shape[0], c:c+p.shape[1]] == 0))))
    # Several overlapping views may cover every cell more than once, while no
    # individual view contains the whole answer. Exclusive-cell counts alone
    # therefore cannot establish whether assembly is necessary.
    single_fragment_solves = any(np.array_equal(_crop(_compress(p), bg), output)
                                for p in parts)
    holes = _groups(zip(*np.nonzero(output == bg)))
    hole_count = sum(all(0 < r < output.shape[0]-1 and 0 < c < output.shape[1]-1
                         for r, c in group) for group in holes)
    return {"background": bg, "fragments": len(parts), "union_shape": union.shape,
            "output_shape": output.shape, "row_runs": row_runs, "column_runs": col_runs,
            "recurrences": [_recurs(output, 0), _recurs(output, 1)],
            "local_false_equalities": local_false_equalities,
            "premature_compression_fails": premature_fails, "holes": hole_count,
            "contributions": contributions, "offsets": offsets,
            "keys": len(_signatures(union, bg)),
            "single_fragment_solves": single_fragment_solves,
            "registered_not_collapsed": union.shape != output.shape,
            "unregistered_collapse_differs": not np.array_equal(_crop(_compress(a), bg), output)}


def _weave(rng, bg):
    """Stretched multicolor network with literal crossing values and key patches."""
    nh, nv = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    palette = rng.choice([x for x in range(10) if x != bg], int(rng.integers(4, 10)), replace=False)
    hc, vc = rng.choice(palette, nh), rng.choice(palette, nv)
    base = np.full((2*nh+1, 2*nv+1), bg, dtype=np.int8)
    for j, color in enumerate(vc):
        base[:, 2*j+1] = color
    for i, color in enumerate(hc):
        base[2*i+1, :] = color
    owner_mode = int(rng.integers(4))
    for i in range(nh):
        for j in range(nv):
            vertical = owner_mode == 1 or (owner_mode == 2 and i % 2) or (owner_mode == 3 and rng.integers(2))
            base[2*i+1, 2*j+1] = vc[j] if vertical else hc[i]
    row_runs = rng.integers(1, 4, len(base))
    col_runs = rng.integers(1, 4, base.shape[1])
    # Make several room-sized gaps, without imposing a constant scale factor.
    row_runs[::2] = rng.integers(2, 5, nh+1)
    col_runs[::2] = rng.integers(2, 5, nv+1)
    if max(sum(row_runs), sum(col_runs)) > 17:
        raise ValueError("oversized weave candidate")
    a = np.repeat(np.repeat(base, row_runs, 0), col_runs, 1)
    rs, cs = np.r_[0, np.cumsum(row_runs)], np.r_[0, np.cumsum(col_runs)]
    positions = [(int(rs[2*i] + (row_runs[2*i]-2 if i == 0 else 0)),
                  int(cs[2*j] + (col_runs[2*j]-2 if j == 0 else 0)))
                 for i in range(nh+1) for j in range(nv+1)]
    # In a margin, choose the corner adjacent to the first line. Interior keys
    # touch a preceding line. Thus the whole foreground remains connected.
    rng.shuffle(positions)
    count = int(rng.integers(2, min(5, len(positions))+1))
    for r, c in positions[:count]:
        a[r:r+2, c:c+2] = rng.choice(palette, 4, replace=False).reshape(2, 2)
    if any(len(v) != 1 for v in _signatures(a, bg).values()):
        raise ValueError("accidentally repeated key")
    assert len(_groups(zip(*np.nonzero(a != bg)))) == 1
    return a


def _cut(a, bg, rng, target, mixed_axes):
    parts = [(a, (0, 0))]
    axes_used = set()
    for step in range(target-1):
        proposals = []
        for i, (p, origin) in enumerate(parts):
            for axis in (0, 1):
                if mixed_axes and step == 1 and axis in axes_used:
                    continue
                for places in _signatures(p, bg).values():
                    k = places[0][axis]
                    for start in range(max(1, k-1), k+1):
                        for stop in range(k+2, min(p.shape[axis]-1, k+4)+1):
                            if not 0 < start < stop < p.shape[axis]:
                                continue
                            x = p[:stop] if axis == 0 else p[:, :stop]
                            y = p[start:] if axis == 0 else p[:, start:]
                            # Full rectangular views: no trimming off observed context.
                            if not np.array_equal(_crop(x, bg), x) or not np.array_equal(_crop(y, bg), y):
                                continue
                            if any(len(_groups(zip(*np.nonzero(q != bg)))) != 1 for q in (x, y)):
                                continue
                            if min(np.count_nonzero(x != bg), np.count_nonzero(y != bg)) < 8:
                                continue
                            new_origin = list(origin)
                            new_origin[axis] += start
                            proposals.append((i, axis, (x.copy(), origin), (y.copy(), tuple(new_origin))))
        if not proposals:
            raise ValueError("cannot cut connected overlapping views")
        i, axis, x, y = proposals[int(rng.integers(len(proposals)))]
        axes_used.add(axis)
        parts[i:i+1] = [x, y]
    result = [p for p, origin in parts]
    union, _ = _assemble(result, bg)
    if not np.array_equal(union, a):
        raise ValueError("cuts did not reconstruct the intended union")
    return result


def _pack(parts, bg, rng):
    # Exhaustive shelf layouts for <=4 rectangles: at most 24*8 candidates.
    n = len(parts)
    fg = sum(np.count_nonzero(p != bg) for p in parts)
    layouts = []
    for order in permutations(range(n)):
        for mask in range(1 << (n-1)):
            shelves, shelf = [], []
            for k, i in enumerate(order):
                shelf.append(i)
                if k == n-1 or mask & (1 << k):
                    shelves.append(shelf)
                    shelf = []
            width = max(sum(parts[i].shape[1] for i in s)+len(s)-1 for s in shelves)
            height = sum(max(parts[i].shape[0] for i in s) for s in shelves)+len(shelves)-1
            if max(width, height) <= 30 and 2*fg < 900:
                layouts.append((height, width, shelves))
    if not layouts:
        raise ValueError("fragments cannot fit a legal background-dominant canvas")
    h, w, shelves = layouts[int(rng.integers(len(layouts)))]
    choices = [(hh, ww) for hh in range(h, 31) for ww in range(w, 31) if hh*ww > 2*fg]
    h, w = choices[int(rng.integers(len(choices)))]
    a = np.full((h, w), bg, dtype=np.int8)
    used_height = sum(max(parts[i].shape[0] for i in s) for s in shelves)+len(shelves)-1
    r = int(rng.integers(h-used_height+1))
    for s in shelves:
        shelf_height = max(parts[i].shape[0] for i in s)
        used_width = sum(parts[i].shape[1] for i in s)+len(s)-1
        c = int(rng.integers(w-used_width+1))
        for i in s:
            p = parts[i]
            top = r+int(rng.integers(shelf_height-p.shape[0]+1))
            a[top:top+p.shape[0], c:c+p.shape[1]] = p
            c += p.shape[1]+1
        r += shelf_height+1
    assert len(_components(a, bg)) == n
    return a


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("illegal training/test count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    fixed_bg = int(rng.integers(10))
    vary_bg = bool(rng.integers(2))
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            target = (2 if index == 0 else 3) if split == "train" else 4
            for attempt in range(240):
                bg = int(rng.integers(10)) if vary_bg else fixed_bg
                try:
                    union = _weave(rng, bg)
                    parts = _cut(union, bg, rng, target, mixed_axes=target >= 3)
                    # Reflect/transpose the whole example, never individual pieces.
                    transpose, flip_r, flip_c = [bool(x) for x in rng.integers(2, size=3)]
                    def transform(a):
                        a = a.T if transpose else a
                        return a[::(-1 if flip_r else 1), ::(-1 if flip_c else 1)].copy()
                    parts, union = [transform(p) for p in parts], transform(union)
                    a = _pack(parts, bg, rng)
                    pair = {"input": a, "output": _crop(_compress(union), bg)}
                    info = _analyze(pair)
                    if (not info["registered_not_collapsed"] or not all(max(info[k]) > 1 for k in ("row_runs", "column_runs"))
                            or not info["unregistered_collapse_differs"] or info["holes"] == 0
                            or info["single_fragment_solves"]):
                        continue
                    if split == "train" and index == 0 and not all(info["recurrences"]):
                        continue
                    if split == "train" and index == 1 and not (sum(info["local_false_equalities"]) and info["premature_compression_fails"]):
                        continue
                    if _key(a) in seen:
                        continue
                    seen.add(_key(a))
                    puzzle[split].append(pair)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded generation exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze(pair)
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert info["registered_not_collapsed"] and info["unregistered_collapse_differs"]
            assert not info["single_fragment_solves"], "assembly can be skipped"
            assert max(info["row_runs"]) > 1 and max(info["column_runs"]) > 1
            assert info["holes"] > 0
            infos[split].append(info)
    assert any(all(i["recurrences"]) for i in infos["train"]), "separated recurrence not demonstrated"
    assert any(sum(i["local_false_equalities"]) and i["premature_compression_fails"]
               for i in infos["train"]), "global equality dependency not demonstrated"
    assert max(i["fragments"] for i in infos["train"]) < min(i["fragments"] for i in infos["test"])


def self_test():
    """42 seeds, streaming summaries only; no files, subprocesses, or traces."""
    shapes_in, shapes_out, structures, palettes, backgrounds, counts, puzzles = (set() for _ in range(7))
    local_traps = both_recur = holes = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = []
        for split in ("train", "test"):
            for pair in p[split]:
                a, b = pair["input"], pair["output"]
                signature.append((_key(a), _key(b)))
                assert np.array_equal(solve(a), b)
                info = _analyze(pair)
                shapes_in.add(a.shape)
                shapes_out.add(b.shape)
                roles = {}
                structures.add((b.shape, bytes(roles.setdefault(int(v), len(roles)) for v in b.flat)))
                backgrounds.add(info["background"])
                palettes.add(len(np.unique(a)))
                counts.add(info["fragments"])
                local_traps += int(info["premature_compression_fails"])
                both_recur += int(all(info["recurrences"]))
                holes += info["holes"]
        puzzles.add(tuple(signature))
    assert len(puzzles) == 42 and len(shapes_in) >= 35 and len(shapes_out) >= 25
    assert len(structures) >= 160 and len(backgrounds) >= 8 and len(palettes) >= 4
    assert counts == {2, 3, 4} and both_recur >= 42 and local_traps >= 42 and holes >= 210
    for nt, nq in ((2, 1), (10, 4), (10, 1)):
        validate_generated_puzzle(generate_puzzle(57, nt, nq))
    # Regression for the retained version's exclusive-coverage assertion.
    validate_generated_puzzle(generate_puzzle(3))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(99), generate_puzzle(99)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # Literal crossing preservation and independent, commuting axis partitions.
    g = np.array([[1, 2, 2, 1], [1, 2, 2, 1], [3, 4, 4, 3], [1, 2, 2, 1]], dtype=np.int8)
    assert np.array_equal(_compress(g), np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]]))
    assert np.array_equal(_compress(g).T, _compress(g.T))
    # A non-weave picture with six views exercises the full solver domain.
    # No construction coordinates or expected grid enter solve.
    base = np.random.default_rng(605).integers(1, 10, (5, 20), dtype=np.int8)
    base[2, 8] = 0
    expanded = np.repeat(np.repeat(base, [1, 2, 1, 1, 1], axis=0),
                         [1]*10+[2]+[1]*9, axis=1)
    canvas = np.zeros((25, 30), dtype=np.int8)
    for i, start in enumerate(range(0, 16, 3)):
        r, c = 7*(i//3), 7*(i % 3)
        canvas[r:r+6, c:c+6] = expanded[:, start:start+6]
    assert len(_components(canvas, 0)) == 6
    assert np.array_equal(solve(canvas), base)
    bad = {"train": [{k: v.copy() for k, v in pair.items()} for pair in a["train"]], "test": a["test"]}
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
