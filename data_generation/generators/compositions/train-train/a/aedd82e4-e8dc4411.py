"""Fading diagonal echoes: local support determines the next active stencil.

Only the standard library and NumPy are used.  The input-only solver has no
access to generation parameters, a seed, examples, or stored artifacts.
"""
from __future__ import annotations

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CORNERS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_INF = 10000  # Larger than any finite lifetime possible on a 30x30 source.


def _coords(mask):
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _shift(cells, vector, multiple=1):
    dr, dc = vector
    return {(r + multiple * dr, c + multiple * dc) for r, c in cells}


def _overlap(cells, vector):
    dr, dc = vector
    return any((r + dr, c + dc) in cells for r, c in cells)


def _candidate_vectors(source, cue):
    vectors = set()
    for r, c in source:
        dr, dc = cue[0] - r, cue[1] - c
        if not dr or abs(dr) != abs(dc) or _overlap(source, (dr, dc)):
            continue
        sr, sc = (1 if dr > 0 else -1), (1 if dc > 0 else -1)
        if all(_overlap(source, (sr * k, sc * k))
               for k in range(1, abs(dr))):
            vectors.add((dr, dc))
    return sorted(vectors)


def _disjoint_repetitions(source, vector):
    # Beyond 30 shifts, the bounding boxes of any legal source cannot meet.
    return all(not _overlap(source, (k * vector[0], k * vector[1]))
               for k in range(1, 31))


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _N4:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _parse(input_grid):
    grid = np.asarray(input_grid)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 integer ARC grid")
    values, counts = np.unique(grid, return_counts=True)
    if len(values) != 3 or int(counts.max()) * 2 <= grid.size:
        raise ValueError("expected three colors and a strict majority background")
    background = int(values[int(np.argmax(counts))])
    singles = [int(v) for v, n in zip(values, counts)
               if int(v) != background and n == 1]
    if len(singles) != 1:
        raise ValueError("expected one phase color occurring exactly once")
    dry = singles[0]
    ink = next(int(v) for v in values if int(v) not in (background, dry))
    source = _coords(grid == ink)
    parts = _components(source)
    if len(source) < 4 or not any(len(p) == 1 for p in parts) or not any(
            len(p) > 1 for p in parts):
        raise ValueError("stencil must contain both loose and connected ink")
    cue = next(iter(_coords(grid == dry)))
    vectors = _candidate_vectors(source, cue)
    if len(vectors) != 1 or not _disjoint_repetitions(source, vectors[0]):
        raise ValueError("phase must specify a unique, safely repeating vector")
    return {"background": background, "ink": ink, "dry": dry,
            "source": source, "cue": cue, "vector": vectors[0]}


def _replay(grid, info, *, neighbors=_N4, old_support=True, freeze=False,
            dry_support=False):
    """Chronological simulation; alternatives are used only for ablation checks."""
    height, width = grid.shape
    output = grid.copy()
    active = set(info["source"])
    permanent = set(active)
    dried = set()
    trace = []
    for stage in range(1, 32):
        proposed = {p for p in _shift(active, info["vector"])
                    if 0 <= p[0] < height and 0 <= p[1] < width}
        if not proposed:
            break
        occupancy = proposed | (permanent if old_support else set())
        if dry_support:
            occupancy |= dried
        if freeze and stage > 1:
            dead = set()
        else:
            dead = {p for p in proposed if not any(
                (p[0] + dr, p[1] + dc) in occupancy for dr, dc in neighbors)}
        alive = proposed - dead
        trace.append({"stage": stage, "old": set(permanent),
                      "proposed": proposed, "dead": dead, "alive": alive,
                      "clipped": len(proposed) < len(active)})
        for p in alive:
            output[p] = info["ink"]
        for p in dead:
            output[p] = info["dry"]
        permanent.update(alive)
        dried.update(dead)
        active = alive
        if not active:
            break
    else:
        raise AssertionError("monotone diagonal echoes must leave a 30x30 grid")
    return output, trace


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles and displacement from the input, then replay fading echoes."""
    grid = np.asarray(input_grid)
    info = _parse(grid)
    output, trace = _replay(grid, info)
    if not trace or info["cue"] not in trace[0]["dead"]:
        raise ValueError("the phase dot must be a first-stage dry cell")
    return output


def _lifetimes(source, vector):
    """Unclipped live-echo lifetimes, computed from a forward support DAG.

    Connected source cells have permanent internal support.  A source singleton
    p can touch an older q at delay k when p+k*v is next to q.  If q lives L
    echoes, this supplies p at stages k..k+L, including q's original at stage 0.
    Only an unbroken prefix of supported stages can be reached by a fading p.
    Supports always lie strictly forward, so reverse diagonal order is acyclic.
    This is used to sample/validate relay depth, never by the input-only solve.
    """
    dr, dc = vector
    span = max(max(r for r, _ in source) - min(r for r, _ in source),
               max(c for _, c in source) - min(c for _, c in source)) + 2
    lives = {}
    for p in sorted(source, key=lambda q: (-(q[0] * dr + q[1] * dc), q)):
        if any((p[0] + a, p[1] + b) in source for a, b in _N4):
            lives[p] = _INF
            continue
        intervals = []
        for k in range(1, span + 1):
            r, c = p[0] + k * dr, p[1] + k * dc
            for a, b in _N4:
                q = (r + a, c + b)
                if q in source:
                    assert q in lives
                    intervals.append((k, min(_INF, k + lives[q])))
        last = 0
        for start, stop in sorted(intervals):
            if start > last + 1:
                break
            last = max(last, stop)
        lives[p] = last
    return lives


def _profile(source, vector):
    lives = _lifetimes(source, vector)
    finite = [n + 1 for n in lives.values() if n < _INF]
    return max(finite, default=0), lives


def _full_copy_inside(source, vector, stage, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1]
               for r, c in _shift(source, vector, stage))


def _features(grid, info, trace):
    source, vector = info["source"], info["vector"]
    diagonal_death = orphan = False
    delayed = set()
    for frame in trace:
        union = frame["old"] | frame["proposed"]
        n = frame["stage"]
        for r, c in frame["dead"]:
            diagonal_death |= any((r + a, c + b) in union for a, b in _CORNERS)
            label = (r - n * vector[0], c - n * vector[1])
            if any((label[0] + a, label[1] + b) in source for a, b in _N4):
                orphan = True
            if _full_copy_inside(source, vector, n, grid.shape):
                delayed.add(n)
    return {"diagonal_death": diagonal_death, "orphan": orphan,
            "full_death_stages": delayed,
            "stages": len(trace),
            "partial": any(f["clipped"] for f in trace)}


def _sample_mask(rng):
    # Rectangles, including elongated ones; the occupied bounding box is trimmed.
    if rng.random() < .23:
        h, w = int(rng.integers(2, 6)), int(rng.integers(7, 14))
        if rng.random() < .5:
            h, w = w, h
    else:
        h, w = (int(x) for x in rng.integers(3, 11, size=2))
    fill = float(rng.uniform(.18, .68))
    mask = rng.random((h, w)) < fill
    # Occasionally grow a bar/bend/branch in otherwise random occupied geometry.
    if rng.random() < .20:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(3, 2 * max(h, w) + 1))):
            mask[r, c] = True
            a, b = _N4[int(rng.integers(4))]
            r, c = max(0, min(h - 1, r + a)), max(0, min(w - 1, c + b))
    cells = _coords(mask)
    if not cells:
        return set()
    top, left = min(r for r, _ in cells), min(c for _, c in cells)
    return {(r - top, c - left) for r, c in cells}


def _add_carrier(rng, source, step, depth):
    """Add a varied clump in a separate diagonal lane, preserving the relay.

    The invariant r-c is unchanged by canonical (d,d) translations. A gap of
    three between lane ranges prevents even corner contacts between the added
    clump and the relay at every generation, rather than just in the source.
    """
    carrier = {(0, 0)}
    target_size = int(rng.integers(3, 15))
    while len(carrier) < target_size:
        frontier = sorted({(r + a, c + b) for r, c in carrier for a, b in _N4}
                          - carrier)
        valid = []
        for p in frontier:
            trial = carrier | {p}
            if (max(r for r, _ in trial) - min(r for r, _ in trial) > 7
                    or max(c for _, c in trial) - min(c for _, c in trial) > 7):
                continue
            if _disjoint_repetitions(trial, (step, step)):
                valid.append(p)
        if not valid:
            break
        carrier.add(valid[int(rng.integers(len(valid)))])
    top, left = min(r for r, _ in carrier), min(c for _, c in carrier)
    carrier = {(r - top, c - left) for r, c in carrier}
    low, high = min(r - c for r, c in source), max(r - c for r, c in source)
    clow, chigh = min(r - c for r, c in carrier), max(r - c for r, c in carrier)
    gap = int(rng.integers(3, 6))
    anchors = [(high + gap - clow, 0), (0, chigh - low + gap)]
    rng.shuffle(anchors)
    for anchor in anchors:
        trial = source | {(r + anchor[0], c + anchor[1]) for r, c in carrier}
        extent = max(max(r for r, _ in trial), max(c for _, c in trial)) + 1
        if extent + (depth + 1) * step <= 30:
            return trial
    return source


def _sample_template(rng, depth, diagonal_witness, forced_step=None):
    for _ in range(16000):
        source = _sample_mask(rng)
        if len(source) < 4 or not any(
                (r + a, c + b) in source for r, c in source for a, b in _N4):
            continue
        step = next(k for k in range(1, 15) if not _overlap(source, (k, k)))
        if step > 5 or (forced_step is not None and step != forced_step):
            continue
        vector = (step, step)
        if not _disjoint_repetitions(source, vector):
            continue
        profile, lives = _profile(source, vector)
        if profile != depth:
            continue
        height, width = max(r for r, _ in source) + 1, max(c for _, c in source) + 1
        if max(height, width) + (depth + 1) * step > 30:
            continue
        if rng.random() < (.65 if depth == 3 else .30):
            source = _add_carrier(rng, source, step, depth)
            profile, lives = _profile(source, vector)
            assert profile == depth and _disjoint_repetitions(source, vector)
        first = _shift(source, vector)
        dead = _shift({p for p, life in lives.items() if life == 0}, vector)
        if diagonal_witness and not any(
                (r + a, c + b) in source | first
                for r, c in dead for a, b in _CORNERS):
            continue
        # Sampling a phase cannot rely on an assumed bounding-box corner.
        eligible = sorted(p for p in dead if _candidate_vectors(source, p) == [vector])
        if eligible:
            return source, vector, eligible
    raise RuntimeError("could not sample the requested support relay")


def _transform(grid, rng):
    result = grid
    if rng.random() < .5:
        result = result.T
    if rng.random() < .5:
        result = result[::-1, :]
    if rng.random() < .5:
        result = result[:, ::-1]
    return result.copy()


def _make_pair(rng, colors, depth, teaching=False, forced_step=None):
    background, ink, dry = colors
    for _ in range(160):
        source, vector, eligible = _sample_template(rng, depth, teaching, forced_step)
        h, w = max(r for r, _ in source) + 1, max(c for _, c in source) + 1
        step = vector[0]
        for _ in range(64):
            required_h, required_w = h + (depth + 1) * step, w + (depth + 1) * step
            top = int(rng.integers(0, min(6, 30 - required_h) + 1))
            left = int(rng.integers(0, min(6, 30 - required_w) + 1))
            height = required_h + top + int(rng.integers(0, min(9, 30 - required_h - top) + 1))
            width = required_w + left + int(rng.integers(0, min(9, 30 - required_w - left) + 1))
            cue_local = eligible[int(rng.integers(len(eligible)))]
            grid = np.full((height, width), background, dtype=np.int8)
            for r, c in source:
                grid[r + top, c + left] = ink
            cue = (cue_local[0] + top, cue_local[1] + left)
            grid[cue] = dry
            info = _parse(grid)
            output, trace = _replay(grid, info)
            feature = _features(grid, info, trace)
            if (np.count_nonzero(output == dry) < 2
                    or not np.any((grid == background) & (output == ink))):
                continue
            if depth not in feature["full_death_stages"]:
                continue
            if teaching and not (feature["orphan"] and feature["diagonal_death"]):
                continue
            # Transform the input and derive the label again, independently of
            # the orientation used during construction.
            grid = _transform(grid, rng)
            return {"input": grid, "output": solve(grid)}
    raise RuntimeError("could not place a complete, instructive support relay")


def _checked_int(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid_key(grid):
    grid = np.asarray(grid)
    return grid.shape, tuple(int(v) for v in grid.flat)


def _puzzle_key(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _checked_int(seed, "seed", 0)
    num_train = _checked_int(num_train, "num_train", 2, 10)
    num_test = _checked_int(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    # Plan tests first so a small-step relay gets matching training evidence.
    result, seen = {"train": [], "test": []}, set()
    for _ in range(num_test):
        for _ in range(30):
            pair = _make_pair(rng, colors, 3)
            key = _grid_key(pair["input"])
            if key not in seen:
                seen.add(key)
                result["test"].append(pair)
                break
        else:
            raise RuntimeError("could not create distinct tests")
    needs_step_one = any(abs(_parse(p["input"])["vector"][0]) == 1
                         for p in result["test"])
    # Both witnesses remain present at the minimum legal training count.
    depths = [1, 2] + [int(rng.integers(1, 3)) for _ in range(num_train - 2)]
    rng.shuffle(depths)
    teaching_index = next(i for i, d in enumerate(depths) if d == 2)
    basic_index = next(i for i, d in enumerate(depths) if d == 1)
    for index, depth in enumerate(depths):
        for _ in range(30):
            forced_step = 1 if needs_step_one and index == basic_index else None
            pair = _make_pair(rng, colors, depth, teaching=(index == teaching_index),
                              forced_step=forced_step)
            key = _grid_key(pair["input"])
            if key not in seen:
                seen.add(key)
                result["train"].append(pair)
                break
        else:
            raise RuntimeError("could not create distinct inputs")
    validate_generated_puzzle(result)
    return result


def _unpruned(grid, info):
    """Ablation: replicate all source cells; preserve the given dry cue."""
    result = grid.copy()
    for n in range(1, 31):
        points = {p for p in _shift(info["source"], info["vector"], n)
                  if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]}
        if not points:
            break
        for p in points:
            result[p] = info["ink"]
    result[info["cue"]] = info["dry"]
    return result


def _classify_future_union(grid, info):
    """Ablation: complete all future full stamps before testing isolation."""
    occupied = set(info["source"])
    for n in range(1, 31):
        points = {p for p in _shift(info["source"], info["vector"], n)
                  if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]}
        if not points:
            break
        occupied.update(points)
    result = grid.copy()
    for r, c in occupied - info["source"]:
        supported = any((r + a, c + b) in occupied for a, b in _N4)
        result[r, c] = info["ink"] if supported else info["dry"]
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, palettes, train_profiles = set(), set(), set()
    diagonal_training = orphan_training = old_matters = reevaluation_matters = False
    dry_matters = False
    split_steps = {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            before, after = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (before, after):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert before.shape == after.shape
            key = _grid_key(before)
            assert key not in seen
            seen.add(key)
            info = _parse(before)
            colors = (info["background"], info["ink"], info["dry"])
            palettes.add(colors)
            split_steps[split].add(abs(info["vector"][0]))
            expected, trace = _replay(before, info)
            assert np.array_equal(expected, after)
            assert trace and info["cue"] in trace[0]["dead"]
            assert np.array_equal(solve(before), after)
            profile, lives = _profile(info["source"], info["vector"])
            assert profile in ((1, 2) if split == "train" else (3,))
            assert _full_copy_inside(info["source"], info["vector"], profile + 1, before.shape)
            feature = _features(before, info, trace)
            assert profile in feature["full_death_stages"]
            assert np.count_nonzero(after == info["dry"]) >= 2
            assert np.count_nonzero((before == info["background"]) & (after == info["ink"])) >= 2
            assert np.array_equal(before[before != info["background"]],
                                  after[before != info["background"]])
            assert set(int(x) for x in np.unique(after)) == set(colors)
            printed = set(info["source"])
            for frame in trace:
                proposed, dead, alive = frame["proposed"], frame["dead"], frame["alive"]
                assert not proposed & printed
                printed.update(proposed)
                # A second formulation checks local singleton decisions by
                # connected components of the complete occupancy at that time.
                singletons = set().union(*(p for p in _components(frame["old"] | proposed)
                                           if len(p) == 1))
                assert dead == proposed & singletons
                assert alive == proposed - dead
                n = frame["stage"]
                if _full_copy_inside(info["source"], info["vector"], n, before.shape):
                    labels = {(r - n * info["vector"][0], c - n * info["vector"][1])
                              for r, c in dead}
                    assert labels == {p for p, life in lives.items() if life + 1 == n}
            assert not np.array_equal(after, _unpruned(before, info))
            assert not np.array_equal(after, _classify_future_union(before, info))
            if split == "train":
                train_profiles.add(profile)
                diagonal_training |= feature["diagonal_death"] and not np.array_equal(
                    after, _replay(before, info, neighbors=_N4 + _CORNERS)[0])
                orphan_training |= feature["orphan"]
                old_matters |= not np.array_equal(after, _replay(before, info, old_support=False)[0])
                reevaluation_matters |= not np.array_equal(after, _replay(before, info, freeze=True)[0])
                dry_matters |= not np.array_equal(after, _replay(before, info, dry_support=True)[0])
    assert len(palettes) == 1
    assert train_profiles == {1, 2}
    assert diagonal_training and orphan_training and old_matters and reevaluation_matters and dry_matters
    if 1 in split_steps["test"]:
        assert 1 in split_steps["train"]


def self_test():
    """Bounded semantic, count, and measured diversity checks over 42 seeds."""
    puzzle_keys, palettes, shapes, motifs, profiles = set(), set(), set(), set(), set()
    directions = {"train": set(), "test": set()}
    steps = {"train": set(), "test": set()}
    counts, component_sizes, densities = set(), set(), []
    partials = {"train": 0, "test": 0}
    diagonal = orphans = smaller_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_keys.add(_puzzle_key(puzzle))
        max_train_area = max(np.asarray(p["input"]).size for p in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = np.asarray(pair["input"])
                info = _parse(before)
                source, vector = info["source"], info["vector"]
                predicted, trace = _replay(before, info)
                assert np.array_equal(predicted, pair["output"])
                f = _features(before, info, trace)
                palettes.add((info["background"], info["ink"], info["dry"]))
                shapes.add(before.shape)
                sr, sc = min(r for r, _ in source), min(c for _, c in source)
                motif = tuple(sorted((r - sr, c - sc) for r, c in source))
                motifs.add(motif)
                sizes = tuple(sorted(len(p) for p in _components(source)))
                profiles.add(sizes)
                component_sizes.update(sizes)
                counts.add(len(trace))
                area = ((max(r for r, _ in source) - sr + 1)
                        * (max(c for _, c in source) - sc + 1))
                densities.append(len(source) / area)
                steps[split].add(abs(vector[0]))
                directions[split].add((int(np.sign(vector[0])), int(np.sign(vector[1]))))
                partials[split] += f["partial"]
                diagonal += f["diagonal_death"]
                orphans += f["orphan"]
                if split == "test":
                    smaller_tests += before.size <= max_train_area
    assert len(puzzle_keys) == 42 and len(palettes) >= 30
    assert len(shapes) >= 50 and len(motifs) >= 150 and len(profiles) >= 50
    assert all(len(directions[s]) == 4 for s in directions)
    assert len(steps["train"]) >= 4 and len(steps["test"]) >= 2
    assert max(component_sizes) >= 10 and max(counts) >= 7 and len(counts) >= 5
    assert min(densities) < .25 and max(densities) > .6
    assert partials["train"] >= 25 and partials["test"] >= 5
    assert diagonal >= 50 and orphans >= 42 and smaller_tests >= 5
    # A fixed input outside the sampler's 1/2/3-depth split exercises a longer
    # relay. These seven cells have death counts 2,0,2,1 in the first four
    # fully visible echoes, with a connected pair continuing afterward.
    rows = ("001010", "010000", "100101", "100000")
    source = {(r, c) for r, row in enumerate(rows) for c, v in enumerate(row) if v == "1"}
    assert _profile(source, (1, 1))[0] == 4
    eligible = sorted(p for p in _shift(source, (1, 1))
                      if _candidate_vectors(source, p) == [(1, 1)]
                      and not any((p[0] + a, p[1] + b) in source | _shift(source, (1, 1))
                                  for a, b in _N4))
    assert eligible
    fixture = np.full((16, 18), 7, dtype=np.int8)
    for r, c in source:
        fixture[r + 3, c + 2] = 0
    fixture[eligible[0][0] + 3, eligible[0][1] + 2] = 4
    info = _parse(fixture)
    predicted, trace = _replay(fixture, info)
    assert [len(f["dead"]) for f in trace[:5]] == [2, 0, 2, 1, 0]
    assert np.array_equal(predicted, solve(fixture))
    assert np.array_equal(solve(fixture[::-1, ::-1]), predicted[::-1, ::-1])
    # All legal explicit count combinations, plus invalid types and limits.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(700 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for seed in (-1, True, 0.5, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    for seed, a, b in ((0, 4, 1), (912429611, 2, 1), (314159, 10, 4)):
        assert _puzzle_key(generate_puzzle(seed, a, b)) == _puzzle_key(generate_puzzle(seed, a, b))
    return {"seeds": 42, "distinct_puzzles": len(puzzle_keys),
            "palettes": len(palettes), "input_shapes": len(shapes),
            "stencil_patterns": len(motifs), "component_profiles": len(profiles),
            "train_steps": sorted(steps["train"]), "test_steps": sorted(steps["test"]),
            "echo_counts": sorted(counts), "partial_pairs": partials,
            "diagonal_death_pairs": diagonal, "orphan_pairs": orphans,
            "tests_no_larger_than_a_train": smaller_tests,
            "stencil_density_range": [min(densities), max(densities)]}
