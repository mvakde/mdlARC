"""Load-activated scaffolds: branch replacement and rigid-beam gravity interact.

Only NumPy and the standard library are used. No files, seeds, or demonstrations
are consulted by solve. self_test streams its 42 seeds and stores no artifacts.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


REPAIR = 2
SWITCH = 8
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _add(a, d):
    return a[0] + d[0], a[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p, cells):
    return [_add(p, d) for d in DIRS if _add(p, d) in cells]


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [(a[0] + k * dr, a[1] + k * dc)
            for k in range(max(abs(a[0] - b[0]), abs(a[1] - b[1])) + 1)]


def _polyline(points):
    walk = [points[0]]
    for a, b in zip(points, points[1:]):
        walk.extend(_line(a, b)[1:])
    return walk


def _ray(p, dc, shape):
    return {(p[0], c) for c in (range(p[1] + 1, shape[1]) if dc == 1
                               else range(p[1]))}


def _beam_cells(beam):
    r, c, n, _ = beam
    return {(r, x) for x in range(c, c + n)}


def _walk(start, previous, cells):
    result = []
    while True:
        assert start not in result, "scaffold branch cannot loop"
        result.append(start)
        following = [q for q in _neighbors(start, cells) if q != previous]
        assert len(following) <= 1, "initial scaffold must be a simple path"
        if not following:
            return result
        previous, start = start, following[0]


def _decode(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] * 2 > inp.size and bg not in (REPAIR, SWITCH)
    markers = [tuple(map(int, p)) for p in np.argwhere(inp == SWITCH)]
    assert markers, "a scene needs at least one switch"
    raw = []
    for q in markers:
        candidates = []
        for dc in (-1, 1):
            p = (q[0], q[1] - dc)
            sites = [p, (p[0] + 1, p[1]), (p[0] + 2, p[1]),
                     (p[0], p[1] - dc)]
            if not all(_inside(x, inp.shape) for x in sites):
                continue
            values = {int(inp[x]) for x in sites}
            colors = values - {REPAIR}
            if len(colors) == 1 and not colors & {bg, SWITCH}:
                candidates.append((p, dc, next(iter(colors))))
        assert len(candidates) == 1, "marker must have one witnessed pivot and down stem"
        raw.append(candidates[0])
    materials = {x[2] for x in raw}
    assert len(materials) == 1, "all wall scaffolds in a scene use one material"
    material = next(iter(materials))
    scaffold = {tuple(map(int, p)) for p in np.argwhere(np.isin(inp, [material, REPAIR]))}
    switches, covered = [], set()
    for p, dc, _ in raw:
        assert set(_neighbors(p, scaffold)) == {(p[0], p[1] - dc), (p[0] + 1, p[1])}
        old = _walk((p[0], p[1] - dc), p, scaffold)
        tail = _walk((p[0] + 1, p[1]), p, scaffold)
        assert len(tail) >= 2 and all(c == p[1] for r, c in tail)
        assert old[-1][1] == (0 if dc == 1 else inp.shape[1] - 1)
        # Every old route is a monotone staircase toward its side wall and up.
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip([p] + old, old)]
        assert all(d in ((-1, 0), (0, -dc)) for d in steps)
        component = {p} | set(old) | set(tail)
        assert not component & covered
        covered |= component
        switches.append({"pivot": p, "dc": dc, "old": set(old),
                         "walk": [p] + old, "new": _ray(p, dc, inp.shape)})
    assert covered == scaffold, "unassigned scaffold fragments"
    assert all(len(_neighbors(q, scaffold)) == 1 for q in markers)
    fixed = scaffold - set().union(*(s["old"] for s in switches))
    beams = []
    for r in range(inp.shape[0]):
        c = 0
        while c < inp.shape[1]:
            color = int(inp[r, c])
            end = c + 1
            while end < inp.shape[1] and int(inp[r, end]) == color:
                end += 1
            if color not in (bg, material, REPAIR, SWITCH):
                assert end - c >= 2, "cargo is a horizontal rigid beam"
                beams.append((r, c, end - c, color))
            c = end
    assert beams and not set().union(*map(_beam_cells, beams)) & scaffold
    return inp.shape, bg, material, scaffold, fixed, switches, beams


def _scaffold(fixed, switches, active, draw_rays):
    cells = set(fixed)
    for i, switch in enumerate(switches):
        if i not in active:
            cells.update(switch["old"])
        elif draw_rays:
            cells.update(switch["new"])
    return cells


def _movable(beams, scaffold, height):
    cells = [_beam_cells(b) for b in beams]
    targets = [{(r + 1, c) for r, c in s} for s in cells]
    moving = {i for i, target in enumerate(targets)
              if all(r < height for r, c in target) and not target & scaffold}
    while True:
        blocked = {i for i in moving if any(targets[i] & cells[j]
                                           for j in range(len(beams)) if j not in moving)}
        if not blocked:
            return moving
        moving -= blocked


def _simulate(decoded, *, draw_rays=True, gravity=True, fire=True):
    shape, bg, material, initial, fixed, switches, original = decoded
    beams, active = list(original), set()
    # Each nonterminal iteration activates a switch or increases total cargo row.
    limit = shape[0] * len(beams) + len(switches) + 2
    for _ in range(limit):
        pressure = {(r + 1, c) for b in beams for r, c in _beam_cells(b)}
        newly = {i for i, s in enumerate(switches)
                 if fire and i not in active and s["old"] & pressure}
        if newly:
            active.update(newly)
            scaffold = _scaffold(fixed, switches, active, draw_rays)
            assert not scaffold & set().union(*map(_beam_cells, beams)), "shelf grew through cargo"
            # Re-evaluate pressure after ALL simultaneous edits and before motion.
            continue
        scaffold = _scaffold(fixed, switches, active, draw_rays)
        moving = _movable(beams, scaffold, shape[0]) if gravity else set()
        if not moving:
            return beams, scaffold, active
        beams = [(r + (i in moving), c, n, color)
                 for i, (r, c, n, color) in enumerate(beams)]
    raise AssertionError("bounded monotone dynamics did not terminate")


def _render(shape, bg, material, scaffold, beams):
    out = np.full(shape, bg, dtype=np.int8)
    for p in scaffold:
        out[p] = material
    occupied = set(scaffold)
    for beam in beams:
        cells = _beam_cells(beam)
        assert not cells & occupied
        occupied.update(cells)
        for p in cells:
            out[p] = beam[3]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct and evolve the input only; no RNG or stored state."""
    decoded = _decode(input_grid)
    beams, scaffold, _ = _simulate(decoded)
    return _render(decoded[0], decoded[1], decoded[2], scaffold, beams)


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _geometry(rng, count, max_steps, teaching):
    w = int(rng.integers(14, 31))
    levels = [int(rng.integers(max_steps + 1)) for _ in range(count)]
    if teaching == 1:
        levels[0] = max_steps
    if teaching == 2:
        levels[0] = 0
    # Richer staircases share a finite 30-row budget with multiple switches.
    while sum(levels) > (4 if count == 3 else 5):
        i = next(i for i in reversed(range(count)) if levels[i])
        levels[i] -= 1
    first_dc = int(rng.choice([-1, 1]))
    directions = [first_dc]
    if count >= 2:
        directions.append(-first_dc if teaching == 2 else first_dc)
    if count == 3:
        directions.append(int(rng.choice([-1, 1])))
    scaffold, switches, tails = set(), [], []
    top = int(rng.integers(3, 6))
    for level, dc in zip(levels, directions):
        pc = int(rng.integers(2 * level + 3, w - 4))
        xs = sorted(map(int, rng.choice(np.arange(2, pc - 1), level, replace=False)), reverse=True)
        if any(a - b < 2 for a, b in zip([pc] + xs, xs)):
            return None
        rises = [int(rng.integers(2, 4)) for _ in range(level)]
        pr = top + sum(rises)
        p = (pr, pc)
        points = [p]
        r = pr
        for x, rise in zip(xs, rises):
            points.extend([(r, x), (r - rise, x)])
            r -= rise
        points.append((top, 0))
        old = _polyline(points)[1:]
        tail_length = int(rng.integers(2, 4))
        tail = [(pr + k, pc) for k in range(1, tail_length + 1)]
        if dc == -1:
            p = (pr, w - 1 - pc)
            old = [(r, w - 1 - c) for r, c in old]
            tail = [(r, w - 1 - c) for r, c in tail]
        component = {p} | set(old) | set(tail)
        if component & scaffold:
            return None
        scaffold |= component
        switches.append({"pivot": p, "dc": dc, "old": set(old), "walk": [p] + old})
        tails.append(tail)
        top = pr + tail_length + int(rng.integers(2, 4))
    h = max(12, top + int(rng.integers(1, 5)))
    if h > 30:
        return None
    for s in switches:
        s["new"] = _ray(s["pivot"], s["dc"], (h, w))
    return (h, w), scaffold, switches, tails


def _new_pair(rng, count, max_steps, teaching=0, query=False):
    for _ in range(180):
        geometry = _geometry(rng, count, max_steps, teaching)
        if geometry is None:
            continue
        shape, scaffold, switches, tails = geometry
        h, w = shape
        bg, material = map(int, rng.choice([c for c in range(10) if c not in (REPAIR, SWITCH)], 2, replace=False))
        palette = [c for c in range(10) if c not in (bg, material, REPAIR, SWITCH)]
        markers = {(s["pivot"][0], s["pivot"][1] + s["dc"]) for s in switches}
        occupied = scaffold | markers
        beams = []

        def insert(r, c, n):
            color = int(rng.choice(palette))
            beam = (r, c, n, color)
            cells = _beam_cells(beam)
            if not (0 <= r < h and 0 <= c and c + n <= w) or cells & occupied:
                return False
            # Preserve unique initial maximal-run decomposition when colors repeat.
            if any(b[0] == r and b[3] == color and
                   (b[1] + b[2] == c or c + n == b[1]) for b in beams):
                return False
            beams.append(beam)
            occupied.update(cells)
            return True

        left_limit = min(s["pivot"][1] for s in switches) - 1
        right_start = max(s["pivot"][1] for s in switches) + 2
        top = min(r for r, c in switches[0]["old"])
        # Two witness loads: one presses the old side; one needs the new shelf.
        for side in (-1, 1):
            if side == -1:
                n = int(rng.integers(2, min(5, left_limit) + 1))
                c = int(rng.integers(0, left_limit - n + 1))
            else:
                n = int(rng.integers(2, min(5, w - right_start) + 1))
                c = int(rng.integers(right_start, w - n + 1))
            insert(top - 2, c, n)
        if len(beams) != 2:
            continue
        # Teaching scene 2 also exposes cargo-on-cargo support, rather than
        # allowing a per-pixel gravity interpretation of both demonstrations.
        if teaching == 2:
            b = beams[int(rng.integers(2))]
            # Extend the upper beam beyond its support. Independent pixel drops
            # would bend this overhang; a rigid beam preserves its entire row.
            c = b[1] if b[1] + b[2] < w else b[1] - 1
            insert(top - 3, c, b[2] + 1)
        elif not teaching:
            desired = int(rng.integers(3 if query else 2, 8 if query else 7))
            for _ in range(35):
                if len(beams) >= desired:
                    break
                n = int(rng.integers(2, min(12, w) + 1))
                r = int(rng.integers(0, h - 1))
                c = int(rng.integers(0, w - n + 1))
                insert(r, c, n)
        inp = _render(shape, bg, material, scaffold, beams)
        for p in markers:
            inp[p] = SWITCH
        # At least one erased old-route waypoint and one retained stem endpoint
        # are annotated. Extra repairs include corners, pivots and straight cells.
        repairs = {switches[0]["walk"][int(rng.integers(1, len(switches[0]["walk"])))], tails[0][-1]}
        candidates = sorted(scaffold)
        rng.shuffle(candidates)
        repairs.update(candidates[:int(rng.integers(0, min(8, len(candidates) // 4) + 1))])
        for p in repairs:
            inp[p] = REPAIR
        try:
            decoded = _decode(inp)
            assert sorted(decoded[-1]) == sorted(beams)
            final, new_scaffold, active = _simulate(decoded)
            assert active
            if teaching == 2:
                assert len(active) == 1 and len(beams) >= 3
            if query:
                assert len(active) >= 2
            no_ray, _, _ = _simulate(decoded, draw_rays=False)
            assert any(a[0] != b[0] for a, b in zip(final, no_ray)), "new shelf must change cargo landing"
            assert any(a[0] > b[0] for a, b in zip(final, decoded[-1]))
            out = _render(shape, bg, material, new_scaffold, final)
        except AssertionError:
            continue
        return {"input": inp, "output": out}
    raise AssertionError("could not construct a witnessed, noninterpenetrating scaffold")


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """All counts are explicit integers; defaults are exactly four and one."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    max_steps = int(rng.integers(1, 3))
    train = [_new_pair(rng, 1, max_steps, teaching=1),
             _new_pair(rng, 2, max_steps, teaching=2)]
    seen = {_key(p["input"]) for p in train}
    for _ in range(num_train - 2):
        for _ in range(40):
            pair = _new_pair(rng, int(rng.integers(1, 3)), max_steps)
            if _key(pair["input"]) not in seen:
                train.append(pair)
                seen.add(_key(pair["input"]))
                break
        else:
            raise AssertionError("bounded distinct-input construction exhausted")
    test = []
    for _ in range(num_test):
        for _ in range(40):
            pair = _new_pair(rng, 3, max_steps, query=True)
            if _key(pair["input"]) not in seen:
                test.append(pair)
                seen.add(_key(pair["input"]))
                break
        else:
            raise AssertionError("bounded distinct-input construction exhausted")
    rng.shuffle(train)
    rng.shuffle(test)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _bends(walk):
    ds = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _features(inp):
    decoded = _decode(inp)
    final, scaffold, active = _simulate(decoded)
    no_ray, _, _ = _simulate(decoded, draw_rays=False)
    static_paths, _, _ = _simulate(decoded, fire=False)
    initial = decoded[-1]
    distances = [a[0] - b[0] for a, b in zip(final, initial)]
    cargo_supports = sum(bool({(r + 1, c) for r, c in _beam_cells(b)} & _beam_cells(q))
                         for i, b in enumerate(final) for j, q in enumerate(final) if i != j)
    occupied = scaffold | set().union(*map(_beam_cells, final))
    rigid_overhangs = sum(b[0] < inp.shape[0] - 1 and
                         0 < len({(r + 1, c) for r, c in _beam_cells(b)} & occupied) < b[2]
                         for b in final)
    return {"shape": list(inp.shape), "background": decoded[1], "material": decoded[2],
            "switches": len(decoded[-2]), "active": len(active), "inactive": len(decoded[-2]) - len(active),
            "beams": len(initial), "lengths": sorted({b[2] for b in initial}),
            "bends": [_bends(s["walk"]) for s in decoded[-2]],
            "directions": sorted({s["dc"] for s in decoded[-2]}),
            "fall_distances": sorted(set(distances)), "cargo_supports": cargo_supports,
            "rigid_overhangs": rigid_overhangs,
            "ray_changes_landing": any(a[0] != b[0] for a, b in zip(final, no_ray)),
            "static_paths_changes_landing": any(a[0] != b[0] for a, b in zip(final, static_paths)),
            "deleted_repairs": sum(int(inp[p]) == REPAIR for i in active for p in decoded[-2][i]["old"]),
            "retained_repairs": sum(int(inp[p]) == REPAIR for p in scaffold)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            decoded = _decode(inp)
            final, scaffold, active = _simulate(decoded)
            assert np.array_equal(out, _render(decoded[0], decoded[1], decoded[2], scaffold, final))
            assert not np.any(np.isin(out, [REPAIR, SWITCH]))
            assert not _movable(final, scaffold, inp.shape[0])
            assert all(a[0] >= b[0] and a[1:] == b[1:] for a, b in zip(final, decoded[-1]))
            assert any(a[0] > b[0] for a, b in zip(final, decoded[-1]))
            assert active
            if split == "train":
                train_features.append(_features(inp))
    assert any(f["inactive"] for f in train_features), "teach contact gating"
    assert any(f["ray_changes_landing"] for f in train_features), "teach new shelves as supports"
    assert any(f["static_paths_changes_landing"] for f in train_features), "teach physical branch replacement"
    assert any(f["cargo_supports"] for f in train_features), "teach rigid stacks"
    assert any(f["rigid_overhangs"] for f in train_features), "teach whole-beam rigidity"
    assert any(f["deleted_repairs"] for f in train_features)
    assert any(f["retained_repairs"] for f in train_features)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _mechanics_test():
    # A manually specified straight switch: the old-side load reaches the floor;
    # the opposite load rests one row above the new shelf.
    inp = np.zeros((12, 14), dtype=np.int8)
    inp[5, :7] = 5
    inp[6:8, 6] = 5
    inp[5, 7] = SWITCH
    inp[5, 2] = inp[7, 6] = REPAIR
    inp[2, 1:4] = 1
    inp[1, 9:12] = 3
    expected = np.zeros_like(inp)
    expected[5, 6:] = 5
    expected[6:8, 6] = 5
    expected[11, 1:4] = 1
    expected[4, 9:12] = 3
    assert np.array_equal(solve(inp), expected)
    # The lower opposite-facing old branch is never pressed: it must survive.
    inp = np.zeros((16, 14), dtype=np.int8)
    inp[4, :7] = inp[5:7, 6] = 5
    inp[4, 7] = SWITCH
    inp[10, 7:] = inp[11:13, 7] = 5
    inp[10, 6] = SWITCH
    inp[1, 1:4], inp[0, 10:13] = 1, 3
    final, scaffold, active = _simulate(_decode(inp))
    assert active == {0} and final == [(3, 10, 3, 3), (15, 1, 3, 1)]
    assert {(10, c) for c in range(7, 14)} <= scaffold


def self_test():
    """42 streamed seeds, hand calculations, ablations, invariance and mutations.

    No JSON, images, execution traces, or per-seed payloads are saved. Returned
    data are small descriptive counters; semantic checks are author-owned.
    """
    _mechanics_test()
    fingerprints, shapes, structures = set(), set(), set()
    backgrounds, materials, beam_counts, lengths, bends = set(), set(), set(), set(), set()
    counts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            before = inp.copy()
            assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
            assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
            # Other color identities carry no hidden action map.
            free = [c for c in range(10) if c not in (REPAIR, SWITCH)]
            mapping = np.arange(10, dtype=np.int8)
            mapping[free] = free[1:] + free[:1]
            assert np.array_equal(solve(mapping[inp]), mapping[out])
            f = _features(inp)
            shapes.add(inp.shape)
            # Color-normalized structure, rather than palette changes alone.
            roles = {}
            normal = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
            structures.add((inp.shape, hashlib.sha256(normal).hexdigest()))
            backgrounds.add(f["background"])
            materials.add(f["material"])
            beam_counts.add(f["beams"])
            lengths.update(f["lengths"])
            bends.update(f["bends"])
            counts["pairs"] += 1
            counts["active_switches"] += f["active"]
            counts["inactive_switches"] += f["inactive"]
            counts["ray_changes_landing"] += f["ray_changes_landing"]
            counts["static_paths_changes_landing"] += f["static_paths_changes_landing"]
            counts["cargo_support_pairs"] += f["cargo_supports"] > 0
            counts["rigid_overhang_pairs"] += f["rigid_overhangs"] > 0
            counts["multiple_fall_distances"] += len(set(f["fall_distances"]) - {0}) > 1
            counts["multi_activation_pairs"] += f["active"] >= 2
            counts["deleted_repairs"] += f["deleted_repairs"]
            counts["retained_repairs"] += f["retained_repairs"]
        # Reject a corrupted preserved pixel and a corrupted cargo pixel.
        query = puzzle["test"][0]
        for p in ((0, 0), tuple(map(int, np.argwhere(query["output"] != _decode(query["input"])[1])[0]))):
            bad = {"train": puzzle["train"], "test": [dict(query)]}
            bad["test"][0]["output"] = query["output"].copy()
            bad["test"][0]["output"][p] = (int(query["output"][p]) + 1) % 10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError("corrupt output accepted")
    assert len(fingerprints) == 42 and len(shapes) >= 40 and len(structures) >= 180
    assert len(backgrounds) == 8 and len(materials) == 8
    assert len(beam_counts) >= 5 and len(lengths) >= 8 and bends == {0, 2, 4}
    assert counts["inactive_switches"] >= 42 and counts["multi_activation_pairs"] >= 42
    assert counts["cargo_support_pairs"] >= 42 and counts["multiple_fall_distances"] >= 100
    assert counts["rigid_overhang_pairs"] >= 42
    assert counts["ray_changes_landing"] == counts["pairs"]
    assert counts["static_paths_changes_landing"] >= 42
    assert counts["deleted_repairs"] >= 42 and counts["retained_repairs"] >= 42
    assert _fingerprint(generate_puzzle(786, 7, 3)) == _fingerprint(generate_puzzle(786, 7, 3))
    # Exercise every legal count, streaming each puzzle rather than retaining runs.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(900 + nt * 4 + nq, nt, nq)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "shapes": len(shapes),
            "color_normalized_inputs": len(structures), "backgrounds": len(backgrounds),
            "materials": len(materials), "beam_counts": sorted(beam_counts),
            "beam_lengths": sorted(lengths), "old_branch_bends": sorted(bends),
            "measurements": dict(counts), "independent_semantic_verification": False}
