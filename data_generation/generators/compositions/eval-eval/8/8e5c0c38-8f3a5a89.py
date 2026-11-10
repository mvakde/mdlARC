"""Contour-triggered mirror locks. Standard library + NumPy; no file I/O.

The frame identifies the inert wall color; the unique singleton is the seed.
Only colors exposed on the selected region's EXTERNAL boundary are cleaned.
Cleaning opens doors, so exposure is recomputed until no deletion is possible.
Vertical reflection is a fixed family convention, including half-column axes.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid")
    return a


def _roles(a):
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[counts.argmax()])
    corners = [int(a[r, c]) for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    singletons = colors[counts == 1]
    if (len(singletons) != 1 or len(set(corners)) != 1
            or counts.max() <= a.size / 2):
        raise ValueError("ambiguous background, frame, or seed")
    wall, seed = corners[0], int(singletons[0])
    if len({bg, wall, seed}) != 3:
        raise ValueError("roles must be distinct")
    if not (np.all(a[0] == wall) and np.all(a[-1] == wall)
            and np.all(a[:, 0] == wall) and np.all(a[:, -1] == wall)):
        raise ValueError("frame must be complete")
    return bg, wall, seed, sorted(set(map(int, colors)) - {bg, wall, seed})


def _flood(allowed, start):
    reached = np.zeros(allowed.shape, dtype=bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    queue = deque([start])
    h, w = allowed.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and allowed[nr, nc]
                    and not reached[nr, nc]):
                reached[nr, nc] = True
                queue.append((nr, nc))
    return reached


def _dilate(mask, diagonal=False):
    h, w = mask.shape
    p = np.pad(mask, 1)
    out = np.zeros_like(mask)
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
               if dr or dc] if diagonal else N4
    for dr, dc in offsets:
        out |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _regions(a, bg, seed):
    start = tuple(map(int, np.argwhere(a == seed)[0]))
    selected = _flood((a == bg) | (a == seed), start)
    outside = _flood(np.pad(~selected, 1, constant_values=True), (0, 0))
    return selected, outside


def _axis(cells, width):
    # Explicit support counting: integral and half-integral vertical axes.
    scores = [(sum((r, axis - c) in cells for r, c in cells), axis)
              for axis in range(2 * width - 1)]
    best = max(n for n, _ in scores)
    axes = [axis for n, axis in scores if n == best]
    if len(axes) != 1:
        raise ValueError("reflection support tie")
    axis = axes[0]
    return axis, {p for p in cells if (p[0], axis - p[1]) in cells}


def _symmetry(a, colors):
    result = {}
    for color in colors:
        cells = set(map(tuple, np.argwhere(a == color).tolist()))
        axis, keep = _axis(cells, a.shape[1])
        result[color] = (axis, cells, keep)
    return result


def _evolve(a, *, one_round=False):
    bg, wall, seed, colors = _roles(a)
    sym = _symmetry(a, colors)
    work = a.copy()
    opened, rounds = set(), []
    initial, _ = _regions(work, bg, seed)
    # Each productive round removes at least one previously uncleaned color's
    # entire unmatched set, so at most len(colors) productive rounds exist.
    for _ in range(len(colors) + 1):
        selected, outside = _regions(work, bg, seed)
        contact = _dilate(selected) & outside[1:-1, 1:-1]
        exposed = [color for color in colors if np.any(contact & (work == color))]
        changed = [color for color in exposed if color not in opened
                   and sym[color][1] != sym[color][2]]
        if not changed:
            break
        rounds.append(tuple(changed))
        for color in changed:
            for p in sym[color][1] - sym[color][2]:
                work[p] = bg
        opened.update(changed)
        if one_round:
            break
    else:
        raise AssertionError("monotone cleaning failed to terminate")
    selected, outside = _regions(work, bg, seed)
    rim = selected & _dilate(outside, True)[1:-1, 1:-1]
    output = work.copy()
    output[rim & (work == bg)] = seed
    return output, {"background": bg, "wall": wall, "seed": seed,
                    "colors": colors, "symmetry": sym, "clean": work,
                    "initial": initial, "selected": selected,
                    "outside": outside, "rim": rim, "rounds": rounds,
                    "opened": opened}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full fixed point and contour from this input alone."""
    return _evolve(_grid(input_grid))[0]


def _capacity(rect):
    r0, r1, c0, c1 = rect
    return ((r1 - r0 + 1) // 7) * ((c1 - c0 + 1) // 7)


def _floorplan(rng, h, w, count):
    rooms = [(1, h - 1, 1, w - 1)]
    while len(rooms) < count:
        capacity = sum(map(_capacity, rooms))
        options = []
        for i, rect in enumerate(rooms):
            r0, r1, c0, c1 = rect
            for cut in range(r0 + 6, r1 - 6):
                parts = [(r0, cut, c0, c1), (cut + 1, r1, c0, c1)]
                if capacity - _capacity(rect) + sum(map(_capacity, parts)) >= count:
                    options.append((i, parts))
            for cut in range(c0 + 6, c1 - 6):
                parts = [(r0, r1, c0, cut), (r0, r1, cut + 1, c1)]
                if capacity - _capacity(rect) + sum(map(_capacity, parts)) >= count:
                    options.append((i, parts))
        if not options:
            return None
        i, parts = options[int(rng.integers(len(options)))]
        rooms.pop(i)
        rooms.extend(parts)
    return rooms


def _adjacency(rooms):
    edges = {}
    for i, a in enumerate(rooms):
        for j, b in enumerate(rooms[:i]):
            r0, r1, c0, c1 = a
            s0, s1, d0, d1 = b
            positions = []
            if c1 + 1 == d0 or d1 + 1 == c0:
                col = c1 if c1 + 1 == d0 else d1
                positions = [(r, col) for r in range(max(r0, s0) + 1, min(r1, s1) - 1)]
            if r1 + 1 == s0 or s1 + 1 == r0:
                row = r1 if r1 + 1 == s0 else s1
                positions = [(row, c) for c in range(max(c0, d0) + 1, min(c1, d1) - 1)]
            if positions:
                edges[tuple(sorted((i, j)))] = positions
    return edges


def _route(rng, edges, room_count, locks, branch):
    neighbors = {i: [] for i in range(room_count)}
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    if branch:
        root = int(rng.integers(room_count))
        visited, route = {root}, []
        for _ in range(locks):
            choices = [(i, j) for i in sorted(visited) for j in neighbors[i] if j not in visited]
            if not choices:
                return None
            i, j = choices[int(rng.integers(len(choices)))]
            route.append((i, j))
            visited.add(j)
        return root, route
    # Tiny finite graph; a randomized DFS finds a path of the required length.
    def extend(path):
        if len(path) == locks + 1:
            return path
        for j in rng.permutation(neighbors[path[-1]]):
            j = int(j)
            if j not in path:
                found = extend(path + [j])
                if found:
                    return found
        return None
    for root in rng.permutation(room_count):
        path = extend([int(root)])
        if path:
            return path[0], list(zip(path, path[1:]))
    return None


def _motif(rng, max_h, max_w):
    h = int(rng.integers(2, min(5, max_h) + 1))
    w = int(rng.integers(3, min(5, max_w) + 1))
    kind = int(rng.integers(4))
    mask = np.zeros((h, w), bool)
    if kind == 0:
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
    elif kind == 1:
        for r in range(h):
            radius = int(rng.integers(1, w // 2 + 1))
            mask[r, w // 2 - radius:w - (w // 2 - radius)] = True
        mask |= np.fliplr(mask)
    elif kind == 2:
        mask[:, [0, -1]] = True
        mask[int(rng.integers(h)), :] = True
    else:
        half = rng.random((h, (w + 1) // 2)) < .65
        half[:, 0] = True
        mask[:, :half.shape[1]] = half
        mask |= np.fliplr(mask)
    return mask


def _place_figure(rng, a, room, color, bg, gate=None):
    """Return the exact additive corruption, or fail without altering a."""
    r0, r1, c0, c1 = room
    available = [(r, c) for r in range(r0 + 1, r1 - 1)
                 for c in range(c0 + 1, c1 - 1)]
    for _ in range(70):
        mask = _motif(rng, r1 - r0 - 2, c1 - c0 - 2)
        h, w = mask.shape
        top = int(rng.integers(r0 + 1, r1 - h))
        left = int(rng.integers(c0 + 1, c1 - w))
        core = {(top + int(r), left + int(c)) for r, c in np.argwhere(mask)}
        if any(a[p] != bg for p in core):
            continue
        # Different figures retain a one-cell free moat in their room.
        occupied = a != bg
        halo = _dilate(occupied, True)
        if any(halo[p] for p in core):
            continue
        axis = 2 * left + w - 1
        additions = set() if gate is None else {gate}
        if gate is not None and (2 * gate[1] == axis or (gate[0], axis - gate[1]) in core):
            continue
        try:
            winner, supported = _axis(core | additions, a.shape[1])
        except ValueError:
            continue
        if winner != axis or supported != core:
            continue
        candidates = [p for p in available if a[p] == bg and p not in core
                      and not halo[p]]
        rng.shuffle(candidates)
        requested = int(rng.integers(1, 4))
        for p in candidates:
            if len(additions) >= requested + (gate is not None):
                break
            if 2 * p[1] == axis or (p[0], axis - p[1]) in core | additions:
                continue
            try:
                winner, supported = _axis(core | additions | {p}, a.shape[1])
            except ValueError:
                continue
            if winner == axis and supported == core:
                additions.add(p)
        if gate is None and not additions:
            continue
        # Gate-only corruption is a legitimate simple active lock.
        for p in core | additions:
            a[p] = color
        return additions
    return None


def _label(clean, bg, seed):
    selected, outside = _regions(clean, bg, seed)
    rim = selected & _dilate(outside, True)[1:-1, 1:-1]
    out = clean.copy()
    out[rim & (clean == bg)] = seed
    return out


def _make_pair(rng, roles, locks, branch, small, seen, closed_parity=None):
    bg, wall, seed, *colors = roles
    for attempt in range(100):
        count = min(8, locks + int(rng.integers(2, 4)))
        if small:
            h, w = int(rng.integers(17, 23)), int(rng.integers(24, 31))
            if rng.integers(2):
                h, w = w, h
        else:
            h, w = int(rng.integers(24, 31)), int(rng.integers(24, 31))
        rooms = _floorplan(rng, h, w, count)
        if rooms is None:
            continue
        edges = _adjacency(rooms)
        route_info = _route(rng, edges, count, locks, branch)
        if route_info is None:
            continue
        root, route = route_info
        selected_rooms = {root} | {j for _, j in route}
        remote_rooms = [i for i in range(count) if i not in selected_rooms]
        a = np.full((h, w), wall, dtype=np.int8)
        for r0, r1, c0, c1 in rooms:
            a[r0:r1, c0:c1] = bg
        deletions = []
        good = True
        for color, (i, j) in zip(colors, route):
            positions = edges[tuple(sorted((i, j)))]
            gate = positions[int(rng.integers(len(positions)))]
            noise = _place_figure(rng, a, rooms[j], color, bg, gate)
            if noise is None:
                good = False
                break
            deletions.extend(noise)
        if not good:
            continue
        # One noisy but enclosed figure, plus one beyond permanent walls.
        for color, index in ((colors[locks], root),
                             (colors[locks + 1], int(rng.choice(remote_rooms)))):
            if _place_figure(rng, a, rooms[index], color, bg) is None:
                good = False
                break
        # A supported pair forms a contrasting CLOSED colored door on a
        # selected/remote boundary. One stone can be detached from its mate.
        # This defeats the shortcut "every colored doorway opens".
        boundary_edges = [(i, j) for i, j in edges
                          if (i in selected_rooms) != (j in selected_rooms)]
        rng.shuffle(boundary_edges)
        closed = False
        for edge in boundary_edges:
            positions = list(edges[edge])
            rng.shuffle(positions)
            for gate in positions:
                if a[gate] != wall:
                    continue
                mates = [(gate[0], c) for c in range(max(1, gate[1] - 7), min(w - 1, gate[1] + 8))
                         if abs(c - gate[1]) >= 2 and a[gate[0], c] in (bg, wall)
                         and (closed_parity is None or (c + gate[1]) % 2 == closed_parity)]
                if mates:
                    mate = mates[int(rng.integers(len(mates)))]
                    a[gate] = a[mate] = colors[locks + 2]
                    closed = True
                    break
            if closed:
                break
        # Reserve one background cell for the singleton seed as well.
        if not good or not closed or np.count_nonzero(a == bg) - 1 <= a.size / 2:
            continue
        r0, r1, c0, c1 = rooms[root]
        positions = [(r, c) for r in range(r0, r1) for c in range(c0, c1) if a[r, c] == bg]
        rng.shuffle(positions)
        # At most thirteen earlier inputs exist, and the initial room has >14
        # seed positions. Exhaustive placement avoids a random duplicate loop.
        for position in positions:
            a[position] = seed
            key = (a.shape, a.tobytes())
            if key not in seen:
                break
            a[position] = bg
        else:
            continue
        clean = a.copy()
        for p in deletions:
            clean[p] = bg
        expected = _label(clean, bg, seed)
        predicted, info = _evolve(a)
        if (len(info["opened"]) != locks or not np.array_equal(predicted, expected)
                or not np.any(info["selected"] & ~info["initial"])):
            continue
        # Path teaching examples must actually require all successive rounds.
        if not branch and len(info["rounds"]) != locks:
            continue
        if closed_parity is not None:
            # Do not let the teaching witnesses also fit an otherwise
            # identical horizontal-reflection interpretation.
            try:
                horizontal = _evolve(a.T)[0].T
            except ValueError:
                horizontal = None
            if horizontal is not None and np.array_equal(horizontal, expected):
                continue
        seen.add(key)
        return {"input": a, "output": expected}
    raise RuntimeError("bounded floorplan construction exhausted")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, "num_train", 2, 10)
    m = _count(num_test, "num_test", 1, 4)
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base = list(map(int, rng.permutation(10)))
    varying_palette = bool(rng.integers(2))
    seen, puzzle = set(), {"train": [], "test": []}
    for split, count in (("train", n), ("test", m)):
        for i in range(count):
            roles = list(map(int, rng.permutation(10))) if varying_palette else base.copy()
            # Gate colors are shuffled even in a stable role palette.
            roles[3:] = list(map(int, rng.permutation(roles[3:])))
            if split == "train":
                locks = i + 1 if i < 2 else int(rng.integers(1, 3))
                branch = i >= 2 and bool(rng.integers(2))
            else:
                locks = int(rng.integers(3, 5))
                branch = bool(rng.integers(2))
            puzzle[split].append(_make_pair(rng, roles, locks, branch,
                                            split == "train" and locks == 1, seen,
                                            i if split == "train" and i < 2 else None))
        rng.shuffle(puzzle[split])
    return puzzle


def _measure(a):
    output, info = _evolve(a)
    bg, wall, seed = info["background"], info["wall"], info["seed"]
    sym, opened = info["symmetry"], info["opened"]
    selected, outside = info["selected"], info["outside"][1:-1, 1:-1]
    untouched_island = untouched_remote = 0
    for color, (_, cells, keep) in sym.items():
        if cells != keep and color not in opened:
            if any(_dilate(selected)[p] and not outside[p] for p in cells):
                untouched_island += 1
            elif all(outside[p] for p in cells):
                untouched_remote += 1
    orth_rim = selected & _dilate(info["outside"])[1:-1, 1:-1]
    contact = _dilate(selected) & outside
    closed = sum(cells == keep and any(contact[p] for p in cells)
                 for _, cells, keep in sym.values())
    global_clean = a.copy()
    for _, cells, keep in sym.values():
        for p in cells - keep:
            global_clean[p] = bg
    return {"shape": tuple(a.shape), "roles": (bg, wall, seed),
            "figures": len(sym), "opened": len(opened), "rounds": len(info["rounds"]),
            "growth": int((selected & ~info["initial"]).sum()),
            "islands": untouched_island, "remote": untouched_remote, "closed": closed,
            "diagonal_rim": int((info["rim"] & ~orth_rim & (info["clean"] == bg)).sum()),
            "parities": tuple(sorted({axis % 2 for axis, _, _ in sym.values()})),
            "core_sizes": tuple(sorted(len(keep) for _, _, keep in sym.values())),
            "noise": sum(len(cells - keep) for _, cells, keep in sym.values()),
            "omit_a_differs": not np.array_equal(output, _label(a, bg, seed)),
            "global_differs": not np.array_equal(output, _label(global_clean, bg, seed)),
            "one_round_differs": not np.array_equal(output, _evolve(a, one_round=True)[0])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, train_stats, horizontal_fits = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            key = (a.shape, a.astype(np.int8).tobytes())
            assert key not in seen
            seen.add(key)
            predicted, info = _evolve(a)
            assert np.array_equal(out, predicted), "incorrect fixed point or external rim"
            bg, seed = info["background"], info["seed"]
            assert np.array_equal(a[[0, -1]], out[[0, -1]])
            assert np.array_equal(a[:, [0, -1]], out[:, [0, -1]])
            assert np.all(out[a == seed] == seed)
            assert np.all((out == a) | (out == bg) | (out == seed))
            for color, (_, cells, keep) in info["symmetry"].items():
                actual = set(map(tuple, np.argwhere(out == color).tolist()))
                assert actual == (keep if color in info["opened"] else cells)
            stats = _measure(a)
            assert stats["growth"] > 0 and stats["islands"] and stats["remote"] and stats["closed"]
            assert stats["omit_a_differs"] and stats["global_differs"]
            if split == "train":
                train_stats.append(stats)
                try:
                    horizontal_fits.append(np.array_equal(out, _evolve(a.T)[0].T))
                except ValueError:
                    horizontal_fits.append(False)
    assert any(s["rounds"] >= 2 and s["one_round_differs"] for s in train_stats)
    assert any(s["diagonal_rim"] > 0 for s in train_stats)
    assert {1, 2} <= {s["opened"] for s in train_stats}
    assert {p for s in train_stats for p in s["parities"]} == {0, 1}
    assert not all(horizontal_fits), "training does not distinguish vertical reflection"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _boundary_regressions():
    """Small hand-specified geometry, not labels returned by the constructor."""
    a = np.zeros((15, 23), dtype=np.int8)
    a[[0, -1]] = 5
    a[:, [0, -1, 7, 15]] = 5
    for r in (2, 3, 4):
        a[r, [10, 12]] = 2
    a[7, 7] = 2  # unmatched first door
    for r in (6, 7, 8):
        a[r, [2, 4]] = 3
    a[5, 2] = 3  # noisy, enclosed, and therefore protected
    for r in (3, 4, 5):
        a[r, [18, 20]] = 4
    a[6, 18] = 4  # unreachable noise remains
    a[10, [15, 19]] = 6  # matched second door stays shut
    a[2, 2] = 7
    expected = a.copy()
    expected[7, 7] = 0
    # Two rooms are explored. These explicit line coordinates include the
    # diagonal-only corner cells beside the aperture, but no island outlines.
    for r in range(1, 14):
        for c in range(1, 15):
            if expected[r, c] == 0 and (r in (1, 13) or c in (1, 6, 8, 14)
                                       or (r, c) == (7, 7)):
                expected[r, c] = 7
    assert np.array_equal(solve(a), expected)
    assert expected[5, 2] == 3 and expected[6, 18] == 4
    assert expected[7, 6] == 7 and expected[7, 8] == 7
    assert expected[7, 9] == 0 and expected[10, 15] == 6
    moved = a.copy()
    moved[2, 2], moved[11, 3] = 0, 7
    moved_expected = expected.copy()
    moved_expected[2, 2], moved_expected[11, 3] = 0, 7
    assert np.array_equal(solve(moved), moved_expected)
    across = a.copy()
    across[2, 2], across[11, 21] = 0, 7
    across_output, info = _evolve(across)
    assert not info["opened"] and across_output[7, 7] == 2
    assert across_output[1, 16] == 7 and across_output[1, 1] == 0
    # Attaching the same formerly enclosed noisy island to the inert wall
    # changes its eligibility. The symmetry calculation itself is unchanged.
    attached = a.copy()
    attached[6, 1] = 5
    _, info = _evolve(attached)
    assert 3 in info["opened"] and info["clean"][5, 2] == 0
    assert all(info["clean"][r, c] == 3 for r in (6, 7, 8) for c in (2, 4))
    return 4


def self_test():
    """Bounded, in-memory checks across 42 seeds; returns compact measurements."""
    boundary_checks = _boundary_regressions()
    fingerprints, shapes, cores, roles, parities, rounds = set(), set(), set(), set(), set(), set()
    pairs, growth, diagonal, max_noise = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _fingerprint(puzzle)
        assert signature not in fingerprints
        fingerprints.add(signature)
        assert signature == _fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                saved = a.copy()
                assert np.array_equal(solve(a), out) and np.array_equal(a, saved)
                # Horizontal reflection and vertical reflection preserve the
                # fixed vertical-axis convention; quarter-turning does not.
                if seed < 3:
                    for transform in (np.fliplr, np.flipud):
                        assert np.array_equal(solve(transform(a)), transform(out))
                    perm = np.roll(np.arange(10), seed + 1)
                    assert np.array_equal(solve(perm[a]), perm[out])
                stat = _measure(a)
                pairs += 1
                shapes.add(stat["shape"])
                cores.update(stat["core_sizes"])
                roles.add(stat["roles"])
                parities.update(stat["parities"])
                rounds.add(stat["rounds"])
                growth += stat["growth"]
                diagonal += stat["diagonal_rim"]
                max_noise = max(max_noise, stat["noise"])
                if split == "test":
                    assert stat["opened"] >= 3
        assert any(_measure(p["input"])["one_round_differs"] for p in puzzle["train"])
    assert len(shapes) >= 25 and len(cores) >= 8 and len(roles) >= 20
    assert parities == {0, 1} and {1, 2, 3} <= rounds
    # Exercise every legal count combination once; no grids are retained.
    count_calls = 0
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            puzzle = generate_puzzle(9000 + 10 * n + m, n, m)
            assert len(puzzle["train"]) == n and len(puzzle["test"]) == m
            validate_generated_puzzle(puzzle)
            count_calls += 1
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(91, 2, 1)
    cell = np.argwhere(bad["test"][0]["input"] != bad["test"][0]["output"])[0]
    r, c = map(int, cell)
    bad["test"][0]["output"][r, c] = bad["test"][0]["input"][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "pairs": pairs,
            "shapes": len(shapes), "core_sizes": sorted(cores), "role_palettes": len(roles),
            "axis_parities": sorted(parities), "productive_rounds": sorted(rounds),
            "total_newly_reached_cells": growth, "diagonal_rim_cells": diagonal,
            "max_noise_per_pair": max_noise, "legal_count_calls": count_calls,
            "hand_specified_boundary_cases": boundary_checks,
            "semantic_scope": "generator-owned invariant and metamorphic checks"}
