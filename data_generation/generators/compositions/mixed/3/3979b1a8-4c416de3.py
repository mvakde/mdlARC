"""Complete a corner stencil, then recursively grow ornament-valued SE Ls.

Only Python/NumPy; no files, generation metadata, or examples enter solve().
The logical cell size is twice the square frame's inset plus one.
"""

from __future__ import annotations

import hashlib
import numpy as np


_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_PROFILES = ((3, 2), (3, 3), (3, 4), (3, 5), (5, 2), (5, 3))
_HOST_SEEDS = (0, 110109768, 903569390, 873678672, 562368419,
               261446075, 517055459, 541755693, 346417303, 261391502,
               401279077, 162784479, 617672625, 990215619, 647267187,
               900942134, 895317066, 8062964, 758582983, 564179827,
               231792341, 978230870, 513393626, 973486472, 554649289,
               702610319, 516769079, 650043328, 432331353, 607445103,
               878312153, 5646560, 44351042, 635145495, 810390380,
               821291261, 50459777, 398969793, 807316495, 440980598,
               879648232, 942884048)


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _need(not isinstance(value, (bool, np.bool_)) and
          isinstance(value, (int, np.integer)), name + " must be an integer")
    value = int(value)
    _need(value >= low and (high is None or value <= high), name + " out of range")
    return value


def _counts(train, test):
    train = _integer(train, "num_train", 2, 10)
    test = _integer(test, "num_test", 1, 4)
    _need(train > test, "num_train must exceed num_test")
    return train, test


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= a) & (a <= 9)), "bad grid color")
    return a.astype(np.int8, copy=True)


def _connected(pattern):
    if not pattern:
        return False
    seen = {min(pattern)}
    todo = list(seen)
    while todo:
        u, v = todo.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                p = (u + du, v + dv)
                if p in pattern and p not in seen:
                    seen.add(p)
                    todo.append(p)
    return len(seen) == len(pattern)


def _domain(q):
    r = q // 2
    # Keep every inward frame-arm pixel readable. Ornaments may replace the
    # corner itself, either outward axis, and inside/outside background cells.
    return frozenset((u, v) for u in range(-r, r + 1)
                     for v in range(-r, r + 1)
                     if not ((u == 0 and v > 0) or (v == 0 and u > 0)))


def _anchors(side, q):
    r = q // 2
    b = side - 1 - r
    return ((r, r), (r, b), (b, r), (b, b))


def _base(side, q, background, frame):
    a = np.full((side, side), background, dtype=np.int8)
    r = q // 2
    b = side - 1 - r
    a[r, r:b + 1] = frame
    a[b, r:b + 1] = frame
    a[r:b + 1, r] = frame
    a[r:b + 1, b] = frame
    return a


def _paint(a, anchor, orientation, pattern, color):
    dy, dx = _DIRS[orientation]
    y, x = anchor
    for u, v in pattern:
        a[y + dy * u, x + dx * v] = color


def _analyze(value):
    """Infer scale, structural colors, four local clues, stencil, and cycle.

    Enumerating at most two scales and four backgrounds is bounded. The
    entire input must equal the inferred frame plus its observed clues.
    No output or hidden sampled stencil is consulted.
    """
    a = _grid(value)
    side = a.shape[0]
    _need(a.shape == (side, side) and side <= 15, "seed must be a square of side <=15")
    colors = tuple(int(c) for c in np.unique(a))
    _need(len(colors) == 5, "need background, frame, and three accent roles")
    interpretations = []
    for q, n in _PROFILES:
        if side != q * n:
            continue
        r = q // 2
        frame = int(a[r, r + 1])  # A reserved, inward frame-arm pixel.
        for background in colors:
            if background == frame:
                continue
            try:
                base = _base(side, q, background, frame)
                observed = base.copy()
                anchors = _anchors(side, q)
                clues, accents = [], []
                for k, ((y, x), (dy, dx)) in enumerate(zip(anchors, _DIRS)):
                    clue = set()
                    accent = set()
                    for u in range(-r, r + 1):
                        for v in range(-r, r + 1):
                            color = int(a[y + dy * u, x + dx * v])
                            if color not in (background, frame):
                                clue.add((u, v))
                                accent.add(color)
                    _need(len(accent) == 1 and clue, "corner needs one nonempty accent clue")
                    _need(clue <= _domain(q), "ornament covers protected inward arm")
                    accent = accent.pop()
                    accents.append(accent)
                    clues.append(frozenset(clue))
                    _paint(observed, (y, x), k, clue, accent)
                _need(np.array_equal(observed, a), "unexplained seed pixels")
                _need(len(set(accents[:3])) == 3 and accents[3] in accents[:3],
                      "invalid geometric three-role cycle")
                maximal = {p for p in clues if all(c <= p for c in clues)}
                _need(len(maximal) == 1, "no single intact, compatible reference")
                pattern = maximal.pop()
                limit = 6 if q == 3 else 14
                _need(3 <= len(pattern) <= limit and _connected(pattern), "bad stencil")
                _need(clues[3] < pattern, "growth corner must need completion")
                completed = base.copy()
                for k, (anchor, color) in enumerate(zip(anchors, accents)):
                    _paint(completed, anchor, k, pattern, color)
                interpretations.append({"input": a, "side": side, "q": q, "n": n,
                    "background": background, "frame": frame, "base": base,
                    "anchors": anchors, "clues": tuple(clues), "pattern": pattern,
                    "roles": tuple(accents[:3]), "accents": tuple(accents),
                    "start": accents[:3].index(accents[3]), "completed": completed})
            except ValueError:
                continue
    _need(len(interpretations) == 1, "input must have one geometric interpretation")
    return interpretations[0]


def _grow(completed, q, roles, start):
    """Ground-truth construction: append Ls of whole ornament tiles."""
    side = completed.shape[0]
    n = side // q
    out = np.empty((2 * side, 2 * side), dtype=np.int8)
    out[:side, :side] = completed
    corner = completed[-q:, -q:].copy()
    state = roles[start]
    for k in range(n, 2 * n):
        edge = k * q
        out[:edge, edge:edge + q] = np.tile(corner, (k, 1))
        out[edge:edge + q, :edge] = np.tile(corner, (1, k))
        successor = roles[(roles.index(state) + 1) % 3]
        corner[corner == state] = successor
        out[edge:edge + q, edge:edge + q] = corner
        state = successor
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction using a direct logical-cell phase formula."""
    info = _analyze(input_grid)
    q, n, side = info["q"], info["n"], info["side"]
    completed = info["completed"]
    template = completed[-q:, -q:].copy()
    mask = template == info["accents"][3]
    out = np.empty((2 * side, 2 * side), dtype=np.int8)
    out[:side, :side] = completed
    for row in range(2 * n):
        for col in range(2 * n):
            if row < n and col < n:
                continue
            phase = (info["start"] + max(row, col) - n + int(row == col)) % 3
            tile = template.copy()
            tile[mask] = info["roles"][phase]
            out[row * q:(row + 1) * q, col * q:(col + 1) * q] = tile
    return out


def _sample_pattern(rng, q, asymmetric=False):
    domain = _domain(q)
    limit = 6 if q == 3 else 14
    for _ in range(100):
        size = int(rng.integers(3, limit + 1))
        points = [p for p in sorted(domain)]
        pattern = {points[int(rng.integers(len(points)))]}
        while len(pattern) < size:
            frontier = [p for p in sorted(domain - pattern)
                        if any(max(abs(p[0] - u), abs(p[1] - v)) <= 1 for u, v in pattern)]
            if not frontier:
                break
            pattern.add(frontier[int(rng.integers(len(frontier)))])
        if len(pattern) == size and (not asymmetric or pattern != {(v, u) for u, v in pattern}):
            return frozenset(pattern)
    # A connected asymmetric fallback, not a stencil dictionary for solving.
    return frozenset({(-1, 0), (0, 0), (1, 1)})


def _make_pair(rng, profile, palette, start, asymmetric=False, avoid=None, sparse=False):
    q, n = profile
    side = q * n
    for _ in range(100):
        pattern = _sample_pattern(rng, q, asymmetric)
        if pattern != avoid:
            break
    else:
        raise RuntimeError("bounded stencil sampling exhausted")
    background, frame, *roles = palette
    accents = roles + [roles[start]]
    anchors = _anchors(side, q)
    base = _base(side, q, background, frame)
    inp, completed = base.copy(), base.copy()
    reference = int(rng.integers(3))
    full = {reference}
    if rng.random() < .25:
        full.add(int(rng.integers(3)))
    for k, (anchor, color) in enumerate(zip(anchors, accents)):
        _paint(completed, anchor, k, pattern, color)
        if k in full:
            clue = pattern
        else:
            high = max(2, (len(pattern) + 1) // 2) if sparse else len(pattern)
            keep = int(rng.integers(1, high))
            order = rng.permutation(len(pattern))[:keep]
            points = sorted(pattern)
            clue = {points[int(i)] for i in order}
        _paint(inp, anchor, k, clue, color)
    output = _grow(completed, q, roles, start)
    # This also rejects any accidental ambiguous structural interpretation.
    info = _analyze(inp)
    _need(info["pattern"] == pattern and info["q"] == q, "construction is ambiguous")
    return {"input": inp, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one rule, with a held-out (tile size, seed depth) test profile."""
    seed = _integer(seed, "seed", 0)
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    held = _PROFILES[int(rng.integers(len(_PROFILES)))]
    available = [p for p in _PROFILES if p != held]
    # Two demonstrations vary BOTH macro depth and tile size. At least one
    # depth >=3 exposes the entire cycle, even with the minimum train count.
    choices = [(a, b) for a in available for b in available
               if a[0] != b[0] and a[1] != b[1] and max(a[1], b[1]) >= 3]
    first, second = choices[int(rng.integers(len(choices)))]
    train_profiles = [first, second]
    train_profiles += [available[int(rng.integers(len(available)))] for _ in range(nt - 2)]
    profiles = train_profiles + [held] * nq
    puzzle = {"train": [], "test": []}
    seen = set()
    first_palette = tuple(int(c) for c in rng.choice(10, 5, replace=False))
    first_start = int(rng.integers(3))
    first_pattern = None
    for index, profile in enumerate(profiles):
        for attempt in range(64):
            if index == 0:
                palette = first_palette
            elif index == 1:
                # Reuse literal colors in different roles: numeric ordering
                # cannot explain the cycle, and background/frame also swap.
                b, f, x, y, z = first_palette
                palette = (f, b, y, x, z)
            else:
                palette = tuple(int(c) for c in rng.choice(10, 5, replace=False))
            start = (first_start if index == 0 else (first_start + 1) % 3 if index == 1
                     else int(rng.integers(3)))
            try:
                pair = _make_pair(rng, profile, palette, start,
                                  asymmetric=index == 0, avoid=first_pattern if index == 1 else None,
                                  sparse=index >= nt)
                key = (pair["input"].shape, pair["input"].tobytes())
                _need(key not in seen, "duplicate input")
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("bounded pair generation exhausted")
        if index == 0:
            first_pattern = _analyze(pair["input"])["pattern"]
        seen.add(key)
        puzzle["train" if index < nt else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check observable semantics, demonstration witnesses, and the shift."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "bad splits")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            info = _analyze(inp)
            q, n, side = info["q"], info["n"], info["side"]
            _need(out.shape == (2 * side, 2 * side), "output must double side")
            key = (inp.shape, inp.tobytes())
            _need(key not in seen, "duplicate input")
            seen.add(key)
            _need(np.array_equal(out[:side, :side], info["completed"]), "wrong seed completion")
            changed = inp != info["completed"]
            _need(np.any(changed) and np.all(np.isin(inp[changed], [info["background"], info["frame"]])),
                  "completion must only replace structural cells")
            tile = info["completed"][-q:, -q:].copy()
            mask = tile == info["accents"][3]
            role_index = info["start"]
            for k in range(n, 2 * n):
                expected_arm = tile.copy()
                expected_arm[mask] = info["roles"][role_index]
                for j in range(k):
                    _need(np.array_equal(out[j*q:(j+1)*q, k*q:(k+1)*q], expected_arm), "right arm inheritance")
                    _need(np.array_equal(out[k*q:(k+1)*q, j*q:(j+1)*q], expected_arm), "bottom arm inheritance")
                role_index = (role_index + 1) % 3
                expected_corner = tile.copy()
                expected_corner[mask] = info["roles"][role_index]
                _need(np.array_equal(out[k*q:(k+1)*q, k*q:(k+1)*q], expected_corner), "corner successor")
            _need(np.array_equal(solve(inp), out), "input-only prediction disagrees")
            # Ablation witnesses: growing the raw partial corner loses actual
            # stencil pixels in the FIRST arm; doing only B has no new region.
            raw = inp[-q:, -q:]
            _need(not np.array_equal(raw, out[:q, side:side+q]), "B has no effect on growth")
            _need(out.shape != inp.shape, "A has no effect")
            infos[split].append(info)
    train = infos["train"]
    _need(train[0]["q"] != train[1]["q"] and train[0]["n"] != train[1]["n"],
          "demonstrations must distinguish tile size and depth")
    _need(max(x["n"] for x in train) >= 3, "training must expose full three-cycle")
    _need(train[0]["start"] != train[1]["start"], "starting role must vary visibly")
    first_successor = {train[0]["roles"][i]: train[0]["roles"][(i + 1) % 3] for i in range(3)}
    second_successor = {train[1]["roles"][i]: train[1]["roles"][(i + 1) % 3] for i in range(3)}
    _need(set(first_successor) == set(second_successor) and first_successor != second_successor,
          "first two demonstrations must reverse a reused literal palette cycle")
    _need(train[0]["pattern"] != train[1]["pattern"], "stencil must vary within puzzle")
    _need(train[0]["pattern"] != {(v, u) for u, v in train[0]["pattern"]},
          "need an adjacent-orientation asymmetric witness")
    training_profiles = {(x["q"], x["n"]) for x in train}
    _need(all((x["q"], x["n"]) not in training_profiles for x in infos["test"]),
          "test must hold out a structural combination")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                a = np.asarray(pair[key], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded: 42 puzzles streamed, legal counts, corruptions, and metrics.

    Retain only small feature sets/digests, never a corpus of grids or traces.
    Returned measurements are descriptive, not independent semantic approval.
    """
    profiles, patterns, sizes, starts, references = set(), set(), set(), set(), set()
    footprints, populations, normalized, digests = set(), set(), set(), set()
    absent_corner = offset_clues = symmetric = multiple_refs = 0
    pixels = 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                p = info["pattern"]
                patterns.add(tuple(sorted(p)))
                profiles.add((info["q"], info["n"]))
                sizes.add(len(p))
                starts.add(info["start"])
                footprints.add((max(u for u, _ in p)-min(u for u, _ in p)+1,
                                max(v for _, v in p)-min(v for _, v in p)+1))
                symmetric += int(p == {(v, u) for u, v in p})
                absent_corner += int((0, 0) not in p)
                offset_clues += sum((0, 0) not in c for c in info["clues"] if c != p)
                intact = [i for i, c in enumerate(info["clues"]) if c == p]
                references.update(intact)
                multiple_refs += int(len(intact) > 1)
                populations.add(tuple(len(c) for c in info["clues"]))
                a = pair["input"]
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in a.flat)
                normalized.add((a.shape, norm))
                pixels += pair["input"].size + pair["output"].size
    _need(len(digests) == 42 and profiles == set(_PROFILES), "42-seed diversity failed")
    _need(len(patterns) >= 60 and len(normalized) >= 150, "insufficient structural diversity")
    _need(starts == {0, 1, 2} and references == {0, 1, 2}, "roles/references lack coverage")
    _need(absent_corner > 0 and offset_clues > 0 and symmetric > 0 and multiple_refs > 0,
          "missing stencil/clue/reference modes")
    _need(len(sizes) >= 8 and len(footprints) >= 8 and len(populations) >= 40,
          "stencil and damage variation failed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(0, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a = generate_puzzle(0)
    _need(_digest(a) == _digest(generate_puzzle(0)), "seed not deterministic")
    _need(_digest(a) != _digest(generate_puzzle(1)), "seeds not distinct")
    original = a["train"][0]["output"].copy()
    info = _analyze(a["train"][0]["input"])
    # Three semantic corruptions: missing completion, wrong arm role, wrong
    # new-corner role. Each must be rejected by the returned-puzzle validator.
    tile = info["completed"][-info["q"]:, -info["q"]:]
    ty, tx = (int(v) for v in np.argwhere(tile == info["accents"][3])[0])
    targets = [tuple(int(v) for v in np.argwhere(info["input"] != info["completed"])[0]),
               (ty, info["side"] + tx), (info["side"] + ty, info["side"] + tx)]
    for y, x in targets:
        a["train"][0]["output"] = original.copy()
        a["train"][0]["output"][y, x] = (int(original[y, x]) + 1) % 10
        try:
            validate_generated_puzzle(a)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted semantics accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(digests),
            "profiles": sorted(profiles), "distinct_stencils": len(patterns),
            "normalized_inputs": len(normalized), "stencil_sizes": sorted(sizes),
            "footprints": len(footprints), "clue_count_profiles": len(populations),
            "stencils_without_corner": absent_corner, "offset_partial_clues": offset_clues,
            "symmetric_stencils": symmetric, "multiple_reference_pairs": multiple_refs,
            "total_grid_cells_streamed": pixels, "independent_semantic_approval": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
