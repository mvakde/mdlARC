"""Restore a sparse drawing's column coordinates, then complete its links.

Only standard-library modules and NumPy are used. No parent imports, files,
seed-dependent solving conventions, or saved examples are needed by solve.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (
    0, 600850969, 524793283, 675552728, 214400921, 530815376, 49772515,
    585468224, 590541680, 701013473, 773564243, 847456513, 851753142,
    807823909, 647341206, 326025527, 71355265, 229827621, 450291183,
    466716453, 909950400, 620633580, 446781771, 817337025, 526865271,
    916821590, 832679554, 695390172, 134950473, 109085707, 50719104,
    471497581, 410298414, 472646678, 438673206, 413085246, 381232935,
    813859181, 531808109, 316353537, 964652056, 921678163,
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "color outside 0..9")
    return a.astype(np.int8, copy=True)


def _column_keys(a):
    return [tuple(map(int, a[:, c])) for c in range(a.shape[1])]


def _compact(a):
    seen, keep = set(), []
    for c, key in enumerate(_column_keys(a)):
        if key not in seen:
            keep.append(c)
            seen.add(key)
    return a[:, keep].copy()


def _sign(x):
    return int(x > 0) - int(x < 0)


def _aligned(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) > 0 and (dr == 0 or dc == 0 or dr == dc)


def _line(a, b):
    _need(_aligned(a, b), "unaligned endpoints")
    dr, dc = b[0] - a[0], b[1] - a[1]
    return {(a[0] + i * _sign(dr), a[1] + i * _sign(dc))
            for i in range(max(abs(dr), abs(dc)) + 1)}


def _turn(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _crossing(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k])
           or max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return _turn(a, b, c) * _turn(a, b, d) <= 0 and _turn(c, d, a) * _turn(c, d, b) <= 0


def _infer_chain(points, search_limit=None):
    """Unique noncrossing spanning cycle first, otherwise unique spanning path.

    Reversal and cycle starting point are the same solution. Edge intersections
    are tested continuously, including crossings between diagonal pixel centers.
    """
    n = len(points)
    _need(2 <= n <= 12, "source needs 2..12 landmarks")
    point_set = set(points)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if _aligned(points[i], points[j])
             and not ((_line(points[i], points[j]) - {points[i], points[j]}) & point_set)]
    neighbors = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        neighbors[i].append((j, k))
        neighbors[j].append((i, k))
    _need(all(neighbors), "disconnected landmark")
    conflicts = [set() for _ in edges]
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges[:i]):
            if not {a, b} & {c, d} and _crossing(points[a], points[b], points[c], points[d]):
                conflicts[i].add(j)
                conflicts[j].add(i)

    def search(closed):
        answers, visits = {}, 0

        def visit(order, used, chosen):
            nonlocal visits
            visits += 1
            if search_limit is not None:
                _need(visits <= search_limit, "topology search budget exceeded")
            if len(answers) >= 2:
                return
            if len(order) == n:
                selected = chosen
                if closed:
                    closing = [e for v, e in neighbors[order[-1]] if v == order[0]]
                    if not closing or conflicts[closing[0]] & chosen:
                        return
                    selected = chosen | {closing[0]}
                answers[tuple(sorted(selected))] = order
                return
            for v, e in neighbors[order[-1]]:
                if v not in used and not conflicts[e] & chosen:
                    visit(order + [v], used | {v}, chosen | {e})

        for start in ([0] if closed else range(n)):
            visit([start], {start}, set())
            if len(answers) >= 2:
                break
        return answers

    cycles = search(True) if n >= 3 else {}
    _need(len(cycles) <= 1, "ambiguous cycles")
    answers = cycles or search(False)
    _need(len(answers) == 1, "no unique spanning path")
    signature, order = next(iter(answers.items()))
    selected = [edges[k] for k in signature]
    pixels, long_pixels, lengths = set(), set(), []
    for i, j in selected:
        segment = _line(points[i], points[j])
        length = len(segment) - 1
        pixels.update(segment)
        lengths.append(length)
        if length > 1:
            long_pixels.update(segment)
    _need(pixels - point_set, "drawing already complete")
    all_pixels = set().union(*(_line(points[i], points[j]) for i, j in edges))
    turns = [_sign(_turn(points[order[i-1]], points[order[i]], points[order[(i+1) % n]]))
             for i in range(n)] if cycles else []
    return {"points": points, "edges": selected, "closed": bool(cycles),
            "pixels": pixels, "long_pixels": long_pixels, "lengths": lengths,
            "chords": len(edges) - len(selected), "rejected": len(all_pixels - pixels),
            "concave": -1 in turns and 1 in turns}


def _complete(board, search_limit=None):
    """Analyze and finish a board whose columns have already been compacted."""
    colors, counts = np.unique(board, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _need(int(max(counts)) * 2 > board.size, "background must be a strict majority")
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(board == c)]
              for c in colors if int(c) != bg}
    source_colors = [c for c, ps in groups.items() if len(ps) >= 2]
    _need(len(source_colors) == 1, "expected one source color")
    color = source_colors[0]
    chain = _infer_chain(groups[color], search_limit)
    marker_groups = [(c, ps) for c, ps in groups.items() if c != color]
    _need(len(marker_groups) <= 1, "at most one companion color")
    offset, marker, echo_color, echo = None, None, None, set()
    if marker_groups:
        echo_color, ps = marker_groups[0]
        _need(len(ps) == 1, "companion marker must be a singleton after compaction")
        marker = ps[0]
        anchors = [p for p in chain["points"]
                   if max(abs(marker[0]-p[0]), abs(marker[1]-p[1])) == 1]
        _need(len(anchors) == 1, "ambiguous marker anchor")
        anchor = anchors[0]
        _need(anchor in chain["long_pixels"], "marker must anchor a nontrivial link")
        offset = marker[0]-anchor[0], marker[1]-anchor[1]
        echo = {(r+offset[0], c+offset[1]) for r, c in chain["long_pixels"]}
        _need(all(0 <= r < board.shape[0] and 0 <= c < board.shape[1] for r, c in echo),
              "companion must fit without clipping")
        _need(marker not in chain["pixels"], "marker lies on source stroke")
    out = board.copy()
    for p in echo:
        if board[p] == bg and p not in chain["pixels"]:
            out[p] = echo_color
    for p in chain["pixels"]:
        if board[p] == bg:
            out[p] = color
    visible = int(np.count_nonzero((out == echo_color) & (board == bg))) if marker else 0
    overlap = sum(board[p] == bg for p in echo & chain["pixels"])
    unit_extra = set()
    if marker:
        for r, c in chain["pixels"] - chain["long_pixels"]:
            p = r + offset[0], c + offset[1]
            if (0 <= p[0] < board.shape[0] and 0 <= p[1] < board.shape[1]
                    and board[p] == bg and p not in chain["pixels"] and p not in echo):
                unit_extra.add(p)
    info = dict(chain, background=bg, color=color, offset=offset, marker=marker,
                echo_color=echo_color, echo=echo, visible=visible, overlap=overlap,
                unit_difference=len(unit_extra))
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from input: stable full-column dedup, then geometric inference."""
    board = _compact(_grid(input_grid))
    return _complete(board)[0]


def _transform(points, code):
    result = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    return result


def _sample_vertices(rng, family):
    if family == "cycle":
        if rng.random() < .55:
            w, h = int(rng.integers(2, 9)), int(rng.integers(2, 13))
            points = [(0, 0), (0, w)] + [(h+w-c, c) for c in range(w, -1, -1)]
            if w >= 3 and rng.random() < .65:
                points.pop(int(rng.integers(3, len(points)-1)))
        else:
            w = int(rng.integers(2, 6))
            top = np.cumsum(rng.integers(-1, 2, size=w+1))
            bottom = np.cumsum(rng.integers(-1, 2, size=w+1)) + int(rng.integers(5, 12))
            points = [(int(top[c]), c) for c in range(w+1)]
            points += [(int(bottom[c]), c) for c in range(w, -1, -1)]
        closed = True
    elif family == "unit":
        h = int(rng.integers(3, 14))
        points, closed = [(0, 0), (1, 0), (h, 0), (h, 2)], False
    elif family == "segment":
        points, closed = [(0, 0), (int(rng.integers(2, 18)), 0)], False
    elif rng.random() < .65:
        w = int(rng.integers(2, 9))
        points, r = [(0, 0)], 0
        tail_cols = set(map(int, rng.choice(w+1, size=int(rng.integers(1, 4)), replace=False)))
        for c in range(w+1):
            if c:
                r += int(rng.integers(-1, 2))
                points.append((r, c))
            if c in tail_cols:
                r += int(rng.choice((-1, 1))) * int(rng.integers(2, 8))
                points.append((r, c))
        closed = False
    else:
        points, closed = [(0, 0)], False
        for _ in range(int(rng.integers(2, 10))):
            dr, dc = OFFSETS[int(rng.integers(8))]
            length = int(rng.choice((1, 1, 1, 2, 2, 3, 4, 5)))
            points.append((points[-1][0]+dr*length, points[-1][1]+dc*length))
    return _transform(points, int(rng.integers(8))), closed


def _render_candidate(rng, points, closed, offsets, marked, palette):
    """Build labels from proposed ordered geometry, before input-only inference."""
    if len(points) != len(set(points)) or not 2 <= len(points) <= 12:
        return None
    edges = list(zip(points, points[1:])) + ([(points[-1], points[0])] if closed else [])
    if not all(_aligned(a, b) for a, b in edges):
        return None
    source, long_source = set(), set()
    for a, b in edges:
        segment = _line(a, b)
        source.update(segment)
        if len(segment) > 2:
            long_source.update(segment)
    if not source - set(points):
        return None
    options = []
    if marked:
        for dr, dc in offsets:
            echo = {(r+dr, c+dc) for r, c in long_source}
            for anchor in points:
                marker = anchor[0]+dr, anchor[1]+dc
                if anchor not in long_source or marker in source:
                    continue
                if sum(max(abs(marker[0]-r), abs(marker[1]-c)) == 1 for r, c in points) != 1:
                    continue
                options.append((marker, echo))
        rng.shuffle(options)
    else:
        options = [(None, set())]
    bg, color, echo_color = palette
    for marker, echo in options:
        footprint = source | echo
        r0, r1 = min(r for r, c in footprint), max(r for r, c in footprint)
        c0, c1 = min(c for r, c in footprint), max(c for r, c in footprint)
        # Row margins let unit-only copy alternatives be visibly distinguished.
        upper, lower = int(rng.integers(0, 6)), int(rng.integers(0, 6))
        h, w = r1-r0+1+upper+lower, c1-c0+1
        if h > 30 or w > 13:
            continue
        shift = lambda p: (p[0]-r0+upper, p[1]-c0)
        board = np.full((h, w), bg, dtype=np.int8)
        for p in points:
            board[shift(p)] = color
        if marker is not None:
            board[shift(marker)] = echo_color
        if len(set(_column_keys(board))) != w or np.count_nonzero(board == bg)*2 <= board.size:
            continue
        expected = board.copy()
        for p in echo:
            if board[shift(p)] == bg:
                expected[shift(p)] = echo_color
        for p in source:
            if board[shift(p)] == bg:
                expected[shift(p)] = color
        try:
            actual, info = _complete(board, search_limit=50000)
        except ValueError:
            continue
        intended = {frozenset((shift(a), shift(b))) for a, b in edges}
        inferred = {frozenset((info["points"][i], info["points"][j])) for i, j in info["edges"]}
        if inferred == intended and np.array_equal(expected, actual):
            return board, expected, info
    return None


def _board(rng, role, offsets, palette, marked):
    family = "cycle" if role == "contour_demo" else "unit" if role == "unit_demo" else str(
        rng.choice(("cycle", "walk", "segment"), p=(.50, .43, .07)))
    for _ in range(180):
        points, closed = _sample_vertices(rng, family)
        result = _render_candidate(rng, points, closed, offsets, marked, palette)
        if result is None:
            continue
        board, out, info = result
        if board.shape[1] < 3:
            continue
        if role == "contour_demo" and not (info["rejected"] and info["visible"] >= 2 and info["overlap"]):
            continue
        if role == "unit_demo" and not (info["unit_difference"] and info["visible"] >= 2):
            continue
        if marked and info["visible"] < 2:
            continue
        return board, out, info
    # Finite fallbacks preserve requested evidence; no fallback changes the rule.
    if role == "contour_demo":
        candidates = [([(0, 0), (0, 2), (2, 2), (2, 0)], True)]
    elif role == "unit_demo":
        candidates = [(_transform([(0, 0), (1, 0), (6, 0), (6, 2)], k), False)
                      for k in (0, 2, 4, 6)]
    else:
        candidates = [([(0, 0), (0, 2), (3, 2), (4, 1), (5, 0)], True)]
    for points, closed in candidates:
        result = _render_candidate(rng, points, closed, offsets, marked, palette)
        if result is None:
            continue
        info = result[2]
        if role == "contour_demo" and not (info["rejected"] and info["visible"] >= 2 and info["overlap"]):
            continue
        if role == "unit_demo" and not (info["unit_difference"] and info["visible"] >= 2):
            continue
        if marked and info["visible"] < 2:
            continue
        return result
    raise RuntimeError("bounded geometric construction exhausted")


def _expand(rng, board, info, deep, witness):
    w = board.shape[1]
    high = min(30-w, (2*w+5) if deep else (w+3))
    extra = int(rng.integers(max(2, w//2), high+1))
    counts = [1] * w
    eligible = [c for c in range(w-1) if np.any(board[:, c] == info["color"])]
    anchor = int(rng.choice(eligible))
    # A repeated source-bearing column precedes a new column in witnesses.
    counts[anchor] += 2 if witness else 1
    extra -= 2 if witness else 1
    for _ in range(extra):
        choices = [c for c in range(w) if counts[c] < (7 if deep else 5)]
        counts[int(rng.choice(choices))] += 1
    stream, remaining = [], counts.copy()
    for c in range(w):
        stream.append(c)
        remaining[c] -= 1
        if witness and c == anchor:
            stream.append(anchor)
            remaining[anchor] -= 1
        # Test first occurrences are delayed by more of the available repeats.
        if c < w-1 and rng.random() < (.85 if deep else .28):
            copies = [j for j in range(c+1)
                      for _ in range(remaining[j] - int(witness and j == anchor))]
            rng.shuffle(copies)
            take = len(copies) if deep else int(rng.integers(len(copies)+1))
            stream.extend(copies[:take])
            for j in copies[:take]:
                remaining[j] -= 1
    tail = [j for j in range(w)
            for _ in range(remaining[j] - int(witness and j == anchor))]
    if rng.random() < .5:
        rng.shuffle(tail)
    elif rng.random() < .5:
        tail.reverse()
    stream.extend(tail)
    # A final earlier source column is nonadjacent and defeats last-occurrence
    # retention: its last occurrence follows the last new column's occurrence.
    if witness:
        stream.append(anchor)
    return board[:, stream].copy()


def _fingerprint(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _check_count(value, low, high, name):
    _need(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
          and low <= int(value) <= high, f"invalid {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _need(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
          and seed >= 0, "seed must be a nonnegative integer")
    seed = int(seed)
    nt = _check_count(num_train, 2, 10, "num_train")
    nq = _check_count(num_test, 1, 4, "num_test")
    _need(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    held = OFFSETS[int(rng.integers(8))]
    training_offsets = tuple(d for d in OFFSETS if d != held)
    # Evidence roles are shuffled; they impose no input-dependent rule change.
    roles = ["contour_demo", "unit_demo"] + ["free"] * (nt-2)
    rng.shuffle(roles)
    puzzle, seen = {"train": [], "test": []}, set()
    first_bg = None
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            role = roles[i] if split == "train" else "free"
            offsets = training_offsets if split == "train" else (held,)
            marked = role != "free" or split == "test" or rng.random() < .65
            for _ in range(20):
                palette = tuple(map(int, rng.choice(10, 3, replace=False)))
                if split == "train" and i == 1 and palette[0] == first_bg:
                    continue
                board, out, info = _board(rng, role, offsets, palette, marked)
                inp = _expand(rng, board, info, split == "test", role != "free" or split == "test")
                key = _fingerprint(inp)
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append({"input": inp, "output": out})
                if split == "train" and i == 0:
                    first_bg = palette[0]
                break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    keys = _column_keys(inp)
    unique = list(dict.fromkeys(keys))
    board = np.array(unique, dtype=np.int8).T
    expected, info = _complete(board)
    _need(inp.shape[1] > board.shape[1], "input must contain repeated columns")
    _need(np.array_equal(out, expected), "wrong deduplication/completion label")
    _need(np.array_equal(board[board != info["background"]], out[board != info["background"]]),
          "a retained landmark/marker changed")
    _need(np.any(board != out), "completion must add cells")
    _need(np.count_nonzero(out == info["color"]) > len(info["points"]), "source completion absent")
    ids = [unique.index(k) for k in keys]
    last_order = sorted(range(len(unique)), key=lambda j: len(ids)-1-ids[::-1].index(j))
    info.update(shape=inp.shape, output_shape=out.shape, unique=board.shape[1],
                multiplicities=tuple(Counter(ids).values()),
                crop_fails=not np.array_equal(inp[:, :board.shape[1]], board),
                nonadjacent=any(ids[i] in ids[:i-1] and np.any(inp[:, i] == info["color"])
                                for i in range(2, len(ids))),
                last_fails=last_order != list(range(len(unique))),
                raw_landmarks=int(np.count_nonzero(inp == info["color"])),
                blank_columns=sum(np.all(board[:, c] == info["background"]) for c in range(board.shape[1])),
                directions={"V" if info["points"][i][1] == info["points"][j][1] else
                            "H" if info["points"][i][0] == info["points"][j][0] else "D"
                            for i, j in info["edges"]})
    return info


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _check_count(nt, 2, 10, "training length")
    _check_count(nq, 1, 4, "test length")
    _need(nt > nq, "illegal split counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            info = _features(pair)
            key = _fingerprint(_grid(pair["input"]))
            _need(key not in seen, "duplicate input")
            seen.add(key)
            _need(info["blank_columns"] <= 1, "identical empty columns survived")
            if info["marker"] is not None:
                _need(info["visible"] >= 2, "companion must visibly grow")
            infos[split].append(info)
    train = infos["train"]
    _need(any(x["closed"] and x["rejected"] and x["overlap"] and x["visible"] >= 2 for x in train),
          "training needs contour/chord/priority evidence")
    _need(any(not x["closed"] and x["unit_difference"] for x in train),
          "training needs open-chain and nontrivial-link-copy evidence")
    _need(any(x["crop_fails"] and x["nonadjacent"] and x["last_fails"]
              and x["raw_landmarks"] > len(x["points"]) for x in train),
          "training needs answer-relevant early and nonadjacent full-column duplicates")
    _need(len({x["background"] for x in train}) >= 2, "training backgrounds must vary")
    familiar = {x["offset"] for x in train if x["offset"] is not None}
    _need(all(x["offset"] is not None and x["offset"] not in familiar for x in infos["test"]),
          "test must introduce its visibly encoded held-out offset")


def _digest_puzzle(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                a = np.asarray(pair[key], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, streaming checks over the 42 requested seeds; writes no files."""
    # Hand-rasterized cases distinguish chords/priority and unit-only copying;
    # these expected grids do not come from the geometry label renderer.
    square = np.array([[4,0,0],[2,0,2],[0,0,0],[2,0,2],[0,0,0]], dtype=int)
    _need(np.array_equal(solve(square[:, [0,1,0,2,1,2,0]]),
                         [[4,4,4],[2,2,2],[2,4,2],[2,2,2],[0,0,0]]),
          "literal chord/priority regression")
    stub = np.array([[2,0,0,0],[2,0,0,0],[0,0,0,0],[0,0,0,0],
                     [2,0,2,4],[0,0,0,0],[0,0,0,0]], dtype=int)
    _need(np.array_equal(solve(stub[:, [0,0,1,2,0,3,1]]),
                         [[2,0,0,0],[2,4,0,0],[2,4,0,0],[2,4,0,0],
                          [2,2,2,4],[0,0,0,0],[0,0,0,0]]),
          "literal unit-only copy regression")
    signatures, input_shapes, output_shapes, geometry = set(), set(), set(), set()
    offsets, backgrounds, directions, kinds, widths = set(), set(), set(), set(), set()
    measures = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = _digest_puzzle(puzzle)
        _need(signature == _digest_puzzle(generate_puzzle(seed)), "nondeterminism")
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                _need(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                input_shapes.add(f["shape"])
                output_shapes.add(f["output_shape"])
                r0, c0 = min(r for r, c in f["points"]), min(c for r, c in f["points"])
                geometry.add((f["closed"], tuple(sorted((r-r0, c-c0) for r, c in f["points"]))))
                backgrounds.add(f["background"])
                if f["offset"] is not None:
                    offsets.add(f["offset"])
                directions.update(f["directions"])
                kinds.add(f["closed"])
                widths.add(f["unique"])
                for feature in ("concave", "rejected", "unit_difference", "overlap", "crop_fails", "nonadjacent", "last_fails"):
                    measures[feature] += bool(f[feature])
                measures["unmarked"] += f["offset"] is None
                measures["unequal"] += len(set(f["multiplicities"])) > 1
    _need(len(signatures) == 42, "duplicate puzzles across seeds")
    _need(len(input_shapes) >= 35 and len(output_shapes) >= 25, "insufficient dimension variation")
    _need(len(geometry) >= 50 and len(widths) >= 5, "insufficient structural variation")
    _need(offsets == set(OFFSETS) and backgrounds == set(range(10)), "palette/offset coverage missing")
    _need(directions == {"H", "V", "D"} and kinds == {False, True}, "link/topology coverage missing")
    _need(all(measures[k] > 0 for k in measures) and measures["unmarked"] > 0, "missing family feature")
    # Exercise every legal count without retaining a corpus.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(7100+10*nt+nq, nt, nq)
            _need((len(p["train"]), len(p["test"])) == (nt, nq), "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A wrong added cell must be rejected by the semantic validator.
    bad = generate_puzzle(0)
    pair = bad["test"][0]
    compact = _compact(pair["input"])
    p = tuple(map(int, np.argwhere(compact != pair["output"])[0]))
    pair["output"][p] = compact[p]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "pairs": 210, "puzzles": len(signatures), "literal_regressions": 2,
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "geometries": len(geometry), "widths": sorted(widths),
            "offsets": len(offsets), "backgrounds": len(backgrounds), "features": dict(measures)}
