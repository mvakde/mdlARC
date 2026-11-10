"""Competing cross reservoirs: geometry defines capacity and geodesic catchments.

No files, metadata, examples, or random state enter solve(). All experiments and
the 42-seed self-test are bounded and in memory; this module writes no artifacts.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer 1..30 by 1..30 grid of colors 0..9")
    return a.astype(np.int8, copy=False)


def _centers(mask):
    result = np.zeros_like(mask, dtype=bool)
    result[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                          & mask[2:, 1:-1] & mask[1:-1, :-2]
                          & mask[1:-1, 2:])
    return result


def _live(mask):
    centers = _centers(mask)
    union = centers.copy()
    union[1:] |= centers[:-1]
    union[:-1] |= centers[1:]
    union[:, 1:] |= centers[:, :-1]
    union[:, :-1] |= centers[:, 1:]
    return union


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    components = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        cells = [(r, c)]
        seen[r, c] = True
        for a, b in cells:
            for dr, dc in STEPS:
                nr, nc = a + dr, b + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    cells.append((nr, nc))
        components.append(sorted(cells))
    return components


def _distance(wall, starts):
    h, w = wall.shape
    distance = np.full(wall.shape, -1, dtype=np.int16)
    queue = deque(starts)
    for p in starts:
        distance[p] = 0
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = distance[r, c] + 1
                queue.append((nr, nc))
    return distance


def _geometry(ink):
    live = _live(ink)
    reservoirs = _components(live)
    if not reservoirs:
        raise ValueError("The stroke color must contain a complete plus")
    wall = ink & ~live
    distances = np.stack([_distance(wall, cells) for cells in reservoirs])
    ranked = np.where(distances < 0, 3000, distances)
    best = ranked.min(axis=0)
    unique = (ranked == best).sum(axis=0) == 1
    owner = np.argmin(ranked, axis=0).astype(np.int16)
    owner[(best == 3000) | ~unique | ink] = -1
    ties = (best < 3000) & ~unique & ~ink
    return live, reservoirs, wall, distances, owner, ties


def _roles(a):
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3 or counts.max() <= a.size / 2:
        raise ValueError("Exactly three colors and a strict majority background are required")
    background = int(colors[np.argmax(counts)])
    other = [int(c) for c in colors if c != background]
    plus_colors = [c for c in other if _centers(a == c).any()]
    if len(plus_colors) != 1:
        raise ValueError("Exactly one nonbackground color must contain complete pluses")
    ink = plus_colors[0]
    payload = next(c for c in other if c != ink)
    return background, ink, payload


def _inspect(a):
    background, ink_color, payload = _roles(a)
    ink = a == ink_color
    live, reservoirs, wall, distances, owner, ties = _geometry(ink)
    source = a == payload
    selected = np.zeros_like(ink)
    loads = []
    for j, cells in enumerate(reservoirs):
        candidates = source & (owner == j)
        values = np.sort(distances[j][candidates])
        capacity = len(cells)
        count = min(capacity, len(values))
        if len(values) > capacity and values[capacity - 1] == values[capacity]:
            raise ValueError("An excess load needs a strict distance gap at capacity")
        if count:
            selected |= candidates & (distances[j] <= values[count - 1])
        loads.append((len(values), count))
    return dict(background=background, ink_color=ink_color, payload=payload,
                ink=ink, live=live, reservoirs=reservoirs, wall=wall,
                distances=distances, owner=owner, ties=ties, source=source,
                selected=selected, loads=loads)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible input alone, planning all reservoirs simultaneously."""
    a = _grid(input_grid)
    f = _inspect(a)
    out = a.copy()
    out[f["selected"]] = f["background"]
    for cells, (_, count) in zip(f["reservoirs"], f["loads"]):
        for p in cells[:count]:
            out[p] = f["payload"]
    return out


def _plus(ink, r, c):
    for dr, dc in ((0, 0),) + STEPS:
        ink[r + dr, c + dc] = True


def _ring(ink, r0, c0, r1, c1):
    ink[r0, c0:c1 + 1] = True
    ink[r1, c0:c1 + 1] = True
    ink[r0:r1 + 1, c0] = True
    ink[r0:r1 + 1, c1] = True


def _transform_scene(rng, ink):
    # Transform INPUT geometry only. The displayed prefix is solved afterward.
    ink = np.rot90(ink, int(rng.integers(4)))
    if rng.random() < .5:
        ink = np.fliplr(ink)
    return ink.copy()


def _shared_scene(rng):
    h, w = int(rng.integers(16, 25)), int(rng.integers(18, 30))
    ink = np.zeros((h, w), dtype=bool)
    r, left = int(rng.integers(8, h - 3)), int(rng.integers(2, 5))
    right = left + 2 * int(rng.integers(4, (w - left - 2) // 2 + 1))
    ink[r, left - 1:right + 2] = True
    _plus(ink, r, left)
    _plus(ink, r, right)
    _ring(ink, 1, 1, int(rng.integers(4, 7)), int(rng.integers(4, 8)))
    # T/long-arm camouflage is connected to a reservoir without adding a plus.
    if rng.random() < .6:
        ink[r + 1:r + 3, left] = True
    return _transform_scene(rng, ink)


def _detour_scene(rng):
    h, w = int(rng.integers(23, 31)), int(rng.integers(23, 31))
    ink = np.zeros((h, w), dtype=bool)
    r, c = int(rng.integers(4, 8)), int(rng.integers(4, 7))
    end = int(rng.integers(h - 5, h - 2))
    ink[r - 3, c - 3:c + 6] = True
    ink[r - 3:end + 1, c - 3] = True
    ink[r - 3:end + 1, c + 5] = True
    _plus(ink, r, c)
    if rng.random() < .5:
        _plus(ink, r, c + 1)  # adjacent pluses: capacity 8, not 10
    else:
        _plus(ink, r, c + 2)  # shared arm: capacity 9, not 10
    _plus(ink, h - 3, w - 4)
    return _transform_scene(rng, ink)


def _random_scene(rng, wanted):
    h, w = int(rng.integers(15, 31)), int(rng.integers(15, 31))
    ink = np.zeros((h, w), dtype=bool)
    anchors = []
    candidates = [(r, c) for r in range(1, h - 3) for c in range(1, w - 3)]
    for index in rng.permutation(len(candidates)):
        r, c = candidates[int(index)]
        if all(max(abs(r - a), abs(c - b)) >= 6 for a, b in anchors):
            anchors.append((r, c))
            kind = int(rng.integers(5))
            _plus(ink, r, c)
            if kind == 1:
                _plus(ink, r, c + 1)
            elif kind == 2:
                _plus(ink, r + 2, c)
            elif kind == 3:
                ink[r - 1:r + 2, c - 1:c + 3] = True
            elif kind == 4:
                _plus(ink, r, c + 1)
                _plus(ink, r + 1, c)
            if len(anchors) == wanted:
                break
    if len(anchors) != wanted:
        return None
    # Independent orthogonal walls, near misses, and extensions. New accidental
    # pluses are real reservoirs; the finished geometry is always rescanned.
    for _ in range(int(rng.integers(0, 7))):
        r, c = int(rng.integers(h - 3)), int(rng.integers(w - 3))
        a = min(h - 1, r + int(rng.integers(2, 9)))
        b = min(w - 1, c + int(rng.integers(2, 9)))
        kind = int(rng.integers(5))
        if kind == 0:
            _ring(ink, r, c, a, b)
            if rng.random() < .6:
                ink[r, c + 1] = False
        elif kind == 1:
            ink[r:a + 1, c] = True
            ink[a, c:b + 1] = True
        elif kind == 2:
            ink[r, c:b + 1] = True
        elif kind == 3:
            ink[r:a + 1, c] = True
        else:
            ink[r, c:c + 3] = True
            ink[r + 1, c + 1] = True
    return _transform_scene(rng, ink)


def _can_add(mask, p):
    if mask[p]:
        return False
    r, c = p
    h, w = mask.shape
    mask[p] = True
    # Only a center at p or next to p could have become a new plus.
    ok = True
    for dr, dc in ((0, 0),) + STEPS:
        a, b = r + dr, c + dc
        if 0 < a < h - 1 and 0 < b < w - 1:
            if mask[a, b] and all(mask[a + x, b + y] for x, y in STEPS):
                ok = False
                break
    mask[p] = False
    return ok


def _pick(rng, payload, pool, count, style, forced=()):
    """Pick cells without ever making a payload-colored plus; no retries saved."""
    chosen = []
    for p in forced:
        p = tuple(map(int, p))
        if not pool[p] or not _can_add(payload, p):
            return None
        payload[p] = True
        chosen.append(p)
    if len(chosen) > count:
        return None
    points = np.argwhere(pool & ~payload)
    if not len(points) and len(chosen) < count:
        return None
    if len(points):
        anchor = np.array(chosen[0]) if chosen else points[int(rng.integers(len(points)))]
        if style == "compact":
            score = ((points - anchor) ** 2).sum(axis=1) + rng.random(len(points)) * 3
        elif style == "line":
            axis = int(rng.integers(2))
            score = abs(points[:, axis] - anchor[axis]) * 40 + abs(points[:, 1 - axis] - anchor[1 - axis]) + rng.random(len(points))
        else:
            score = rng.random(len(points))
        for index in np.argsort(score):
            if len(chosen) == count:
                break
            p = tuple(map(int, points[int(index)]))
            if _can_add(payload, p):
                payload[p] = True
                chosen.append(p)
    return chosen if len(chosen) == count else None


def _manhattan(shape, cells):
    rr, cc = np.indices(shape)
    return np.minimum.reduce([abs(rr - r) + abs(cc - c) for r, c in cells])


def _excess_pools(rng, pool, distance, cells, count, require_reversal):
    capacity = len(cells)
    geometric = _manhattan(pool.shape, cells)
    levels = np.unique(distance[pool])
    for threshold in rng.permutation(levels):
        near = pool & (distance <= threshold)
        far = pool & (distance > threshold)
        if near.sum() < capacity + 2 or far.sum() < count - capacity + 2:
            continue
        if require_reversal:
            # Force a selected geometrically farther cell and retained nearer
            # adjacent cell: one connected payload cluster is split at capacity.
            # This rejects Manhattan ranking and whole-object collection.
            pairs = []
            h, w = pool.shape
            for r, c in np.argwhere(near & (distance == threshold)):
                r, c = int(r), int(c)
                for dr, dc in STEPS:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and far[nr, nc]
                            and geometric[nr, nc] < geometric[r, c]):
                        pairs.append(((r, c), (nr, nc)))
            if not pairs:
                continue
            p, q = pairs[int(rng.integers(len(pairs)))]
            return near, far, [p], [q]
        return near, far, [], []
    return None


def _populate(rng, ink, mode):
    # A later wall opening may remove the last planted plus. Such a scene has
    # no visible stroke-role witness and is a construction rejection, not a
    # valid input for the solver.
    if not _centers(ink).any():
        return None
    live, reservoirs, wall, distances, owner, ties = _geometry(ink)
    if len(reservoirs) > 6 or max(map(len, reservoirs)) > 28:
        return None
    payload = np.zeros_like(ink)
    overloaded = None
    shared_loads = None
    if mode == "shared":
        if len(reservoirs) != 2:
            return None
        shared_loads = rng.permutation([1, 4])
    if mode == "detour":
        eligible = [j for j, cells in enumerate(reservoirs)
                    if int(_centers(ink)[tuple(np.array(cells).T)].sum()) >= 2]
        if not eligible:
            return None
        overloaded = eligible[int(rng.integers(len(eligible)))]
    for j in rng.permutation(len(reservoirs)):
        j = int(j)
        cells = reservoirs[j]
        capacity = len(cells)
        pool = owner == j
        style = str(rng.choice(("dispersed", "compact", "line")))
        if mode == "shared":
            count = int(shared_loads[j])  # First slot and row-vs-column witness.
        elif j == overloaded:
            count = capacity + int(rng.integers(2, 8))
        elif mode == "detour":
            count = int(rng.integers(2, min(4, capacity) + 1))
        else:
            load_mode = rng.random()
            count = (capacity + int(rng.integers(1, 7)) if load_mode < .3
                     else capacity if load_mode < .45 else int(rng.integers(0, capacity)))
        if count > capacity:
            pools = _excess_pools(rng, pool, distances[j], cells, count, j == overloaded)
            if pools is None:
                return None
            near, far, p, q = pools
            if _pick(rng, payload, near, capacity, style, p) is None:
                return None
            if _pick(rng, payload, far, count - capacity, style, q) is None:
                return None
        elif _pick(rng, payload, pool, count, style) is None:
            return None
    unreachable = ~ink & (distances.max(axis=0) < 0)
    if mode == "shared" and (not unreachable.any() or not ties.any()):
        return None
    for pool, mandatory in ((ties, mode == "shared"), (unreachable, mode == "shared")):
        limit = min(5, int(pool.sum()))
        count = int(rng.integers(1 if mandatory else 0, limit + 1)) if limit else 0
        if _pick(rng, payload, pool, count, "dispersed") is None:
            return None
    if not payload.any() or (ink | payload).sum() * 2 >= ink.size:
        return None
    return payload


def _features(a):
    f = _inspect(a)
    selected, source = f["selected"], f["source"]
    shared = False
    for component in _components(f["ink"]):
        membership = set()
        points = set(component)
        for j, cells in enumerate(f["reservoirs"]):
            if points.intersection(cells):
                membership.add(j)
        shared |= len(membership) >= 2
    reversal = False
    overflow_with_spare = False
    split_cluster = False
    for component in _components(source):
        split_cluster |= any(selected[p] for p in component) and any(not selected[p] for p in component)
    for j, cells in enumerate(f["reservoirs"]):
        own = source & (f["owner"] == j)
        retained = own & ~selected
        chosen = own & selected
        if retained.any() and chosen.any():
            metric = _manhattan(a.shape, cells)
            reversal |= int(metric[chosen].max()) > int(metric[retained].min())
            for k, (_, count) in enumerate(f["loads"]):
                if k != j and count < len(f["reservoirs"][k]):
                    overflow_with_spare |= bool(np.any(retained & (f["distances"][k] >= 0)))
    centers = _centers(f["ink"])
    overlap = int(f["live"].sum()) < 5 * int(centers.sum())
    order_sensitive = any(set(cells[:count]) != set(sorted(cells, key=lambda p: (p[1], p[0]))[:count])
                          for cells, (_, count) in zip(f["reservoirs"], f["loads"]))
    result = dict(shape=a.shape, roles=(f["background"], f["ink_color"], f["payload"]),
                  reservoirs=len(f["reservoirs"]), capacities=tuple(map(len, f["reservoirs"])),
                  centers=int(centers.sum()), selected=int(selected.sum()),
                  tie_payload=int((source & f["ties"]).sum()),
                  unreachable=int((source & (f["distances"].max(axis=0) < 0)).sum()),
                  excess=sum(n - count for n, count in f["loads"]),
                  partial=any(2 <= count <= 4 and count < len(cells)
                              for cells, (_, count) in zip(f["reservoirs"], f["loads"])),
                  singleton=any(count == 1 for _, count in f["loads"]),
                  order_sensitive=order_sensitive,
                  overlap=overlap, shared=shared, reversal=reversal,
                  split_cluster=split_cluster,
                  overflow_with_spare=overflow_with_spare,
                  max_distance=int(f["distances"].max()),
                  walls=int(f["wall"].sum()))
    return result


def _make_pair(rng, mode, palette, seen, wanted=2):
    for attempt in range(100):
        if mode == "shared":
            ink = _shared_scene(rng)
        elif mode == "detour":
            ink = _detour_scene(rng)
        else:
            ink = _random_scene(rng, wanted)
            if ink is None:
                continue
        payload = _populate(rng, ink, mode)
        if payload is None:
            continue
        a = np.full(ink.shape, palette[0], dtype=np.int8)
        a[ink] = palette[1]
        a[payload] = palette[2]
        key = (a.shape, a.tobytes())
        if key in seen:
            continue
        features = _features(a)
        if mode == "shared" and not (features["shared"] and features["partial"]
                                      and features["tie_payload"] and features["unreachable"]
                                      and features["order_sensitive"]):
            continue
        if mode == "detour" and not (features["overlap"] and features["reversal"]
                                      and features["excess"] and features["overflow_with_spare"]
                                      and features["split_cluster"]):
            continue
        if mode == "random" and not wanted <= features["reservoirs"] <= wanted + 1:
            continue
        seen.add(key)
        return {"input": a, "output": solve(a)}
    raise RuntimeError(f"Bounded construction exhausted for {mode}")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("Need 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    vary_palette = bool(rng.random() < .7)
    puzzle = {"train": [], "test": []}
    seen = set()
    requests = [("shared", 2), ("detour", 2)]
    requests += [("random", int(rng.integers(1, 4))) for _ in range(num_train - 2)]
    requests = [requests[int(i)] for i in rng.permutation(len(requests))]
    for split, specs in (("train", requests),
                         ("test", [("random", int(rng.integers(3, 6))) for _ in range(num_test)])):
        for mode, wanted in specs:
            colors = tuple(map(int, rng.choice(10, 3, replace=False))) if vary_palette else palette
            puzzle[split].append(_make_pair(rng, mode, colors, seen, wanted))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            key = a.shape, a.tobytes()
            assert key not in seen, "Duplicate input"
            seen.add(key)
            f = _inspect(a)
            assert np.array_equal(b, solve(a)), "Incorrect output"
            # Audit the conservation and preservation laws explicitly.
            assert int((a == f["payload"]).sum()) == int((b == f["payload"]).sum())
            assert np.all(b[f["wall"]] == a[f["wall"]])
            assert np.all(b[f["source"] & ~f["selected"]] == f["payload"])
            changed = f["selected"] | f["live"]
            assert np.array_equal(b[~changed], a[~changed])
            for cells, (supply, count) in zip(f["reservoirs"], f["loads"]):
                assert count == min(len(cells), supply)
                assert all(b[p] == (f["payload"] if i < count else f["ink_color"])
                           for i, p in enumerate(cells))
            if split == "train":
                train_features.append(_features(a))
    # These obligations are read from returned inputs, not construction flags.
    for name in ("partial", "shared", "tie_payload", "unreachable", "overlap",
                 "reversal", "excess", "overflow_with_spare", "split_cluster",
                 "order_sensitive", "singleton"):
        assert any(f[name] for f in train_features), "Missing training evidence: " + name
    assert any(f["selected"] > 9 for f in train_features), "Training must reject a global nine-cell capacity"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _counterfactual(a, mode):
    """Deliberately wrong rules for ablation tests, never used for labels."""
    f = _inspect(a)
    out = a.copy()
    if mode == "no_transfer":
        out[f["live"]] = f["payload"]
        return out
    if mode == "whole_stroke":
        reservoirs = _components(f["ink"])
        # Omitting local recognition makes every material cell usable and
        # removes all dead-stroke barriers, including the closed pocket wall.
        distances = np.stack([_distance(np.zeros_like(a, dtype=bool), cells)
                              for cells in reservoirs])
        best = distances.min(axis=0)
        owner = np.argmin(distances, axis=0)
        owner[(distances == best).sum(axis=0) > 1] = -1
    elif mode == "manhattan":
        reservoirs, owner = f["reservoirs"], f["owner"]
        distances = np.stack([_manhattan(a.shape, cells) for cells in reservoirs])
    else:
        raise ValueError("Unknown ablation")
    for j, cells in enumerate(reservoirs):
        candidates = [tuple(map(int, p)) for p in np.argwhere(f["source"] & (owner == j))]
        candidates.sort(key=lambda p: (int(distances[j][p]), p))
        # Deterministic row/column tie breaking is only for this wrong-rule
        # counterfactual; the real family never breaks a capacity-frontier tie.
        selected = candidates[:len(cells)]
        for p in selected:
            out[p] = f["background"]
        for p in cells[:len(selected)]:
            out[p] = f["payload"]
    return out


def self_test():
    """42 streamed seeds, targeted local regressions, counts, and corruption.

    Returns small descriptive metrics for optional compact reporting. No grids,
    traces, logs, or per-seed files are retained by this function.
    """
    signatures, shapes, roles, capacities, center_counts, sink_counts = (set() for _ in range(6))
    selections, path_lengths, wall_counts, excesses = (set() for _ in range(4))
    feature_counts = {name: 0 for name in ("shared", "overlap", "reversal", "tie_payload", "unreachable", "split_cluster")}
    ablations = {mode: 0 for mode in ("whole_stroke", "no_transfer", "manhattan")}
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for mode in ablations:
            witnessed = any(not np.array_equal(_counterfactual(p["input"], mode), p["output"])
                            for p in puzzle["train"])
            assert witnessed, "Training did not reject ablation: " + mode
            ablations[mode] += witnessed
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair["input"])
                shapes.add(f["shape"])
                roles.add(f["roles"])
                capacities.update(f["capacities"])
                center_counts.add(f["centers"])
                sink_counts.add(f["reservoirs"])
                selections.add(f["selected"])
                path_lengths.add(f["max_distance"])
                wall_counts.add(f["walls"])
                excesses.add(f["excess"])
                for name in feature_counts:
                    feature_counts[name] += bool(f[name])
                pairs += 1
        if seed == 0:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(shapes) >= 60 and len(roles) >= 30
    assert {5, 8, 9} <= capacities and len(capacities) >= 5
    assert {1, 2, 3, 4, 5} <= sink_counts
    assert len(center_counts) >= 8 and len(selections) >= 15
    assert len(path_lengths) >= 15 and len(wall_counts) >= 20 and len(excesses) >= 8
    assert min(feature_counts.values()) >= 42
    # All 512 local binary neighborhoods: direct definition versus dilation.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        expected = np.zeros((3, 3), dtype=bool)
        if all(mask[p] for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2))):
            for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)):
                expected[p] = True
        assert np.array_equal(_live(mask), expected)
    # Two overlapping pluses have eight distinct slots, not ten or five.
    block = np.ones((3, 4), dtype=bool)
    assert _centers(block).sum() == 2 and _live(block).sum() == 8
    # A valid member outside the sampler's dimension bands: arms touch edges.
    tiny = np.zeros((3, 5), dtype=np.int8)
    tiny_ink = np.zeros_like(tiny, dtype=bool)
    _plus(tiny_ink, 1, 1)
    tiny[tiny_ink] = 1
    tiny[0, 4] = 2
    expected = tiny.copy()
    expected[0, 4], expected[0, 1] = 0, 2
    assert np.array_equal(solve(tiny), expected)
    clipped = np.zeros((4, 5), dtype=bool)
    clipped[0, 1:4] = True
    clipped[1, 2] = clipped[3, 2] = True
    assert not _live(clipped).any()
    # Explicit boundary tie: six distance-one tokens cannot be split into five.
    tied = np.zeros((9, 9), dtype=np.int8)
    _ink = np.zeros((9, 9), dtype=bool)
    _plus(_ink, 4, 4)
    tied[_ink] = 1
    for p in ((2, 4), (3, 3), (3, 5), (4, 2), (4, 6), (5, 3)):
        tied[p] = 2
    try:
        solve(tied)
    except ValueError as error:
        assert "strict distance gap" in str(error)
    else:
        raise AssertionError("Capacity tie was silently resolved")
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(9871, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    broken = generate_puzzle(17, 2, 1)
    broken["train"][0]["output"] = broken["train"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted corrupted labels")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                shapes=len(shapes), palettes=len(roles), capacities=sorted(capacities),
                reservoirs=sorted(sink_counts), center_counts=sorted(center_counts),
                selected_counts=sorted(selections), path_lengths=sorted(path_lengths),
                wall_counts=len(wall_counts), excess_counts=sorted(excesses),
                feature_pairs=feature_counts, ablation_witness_puzzles=ablations)
