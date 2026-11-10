"""Corner-ray elections: geometric scope controls both a mode and a scale.

Only the standard library and NumPy are used. Generation constructs answers
from sampled geometry; solve reparses the supplied grid and has no puzzle state.
"""

from collections import Counter
from functools import lru_cache
from itertools import product

import numpy as np


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _edges(h, w, r, c, size):
    return tuple(name for name, yes in (
        ("top", r == 0), ("bottom", r + size == h),
        ("left", c == 0), ("right", c + size == w)) if yes)


def _rays(h, w, r, c, size):
    result = []
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr = r if dr == -1 else r + size - 1
        cc = c if dc == -1 else c + size - 1
        cells = []
        rr, cc = rr + dr, cc + dc
        while 0 <= rr < h and 0 <= cc < w:
            cells.append((rr, cc))
            rr, cc = rr + dr, cc + dc
        result.append((dr, dc, tuple(cells)))
    return tuple(result)


def _mode(hist):
    if not hist:
        raise ValueError("empty electorate")
    best = max(hist.values())
    winners = [color for color, count in hist.items() if count == best]
    if len(winners) != 1:
        raise ValueError("the electorate must have a unique mode")
    return int(winners[0])


def _analyze(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    candidates = []
    for color in np.unique(a):
        if color == 0:
            continue
        locations = np.argwhere(a == color)
        low = locations.min(axis=0)
        high = locations.max(axis=0) + 1
        height, width = high - low
        if (height == width and height in (2, 3)
                and len(locations) == height * width):
            candidates.append((int(color), int(low[0]), int(low[1]), int(height)))
    if len(candidates) != 1:
        raise ValueError("need one uniquely identifiable solid square color")
    marker, r, c, size = candidates[0]
    rays = _rays(h, w, r, c, size)
    ray_mask = np.zeros(a.shape, dtype=bool)
    hist, front_hist, first_hist = Counter(), Counter(), Counter()
    for _, _, cells in rays:
        first = True
        for index, (rr, cc) in enumerate(cells):
            ray_mask[rr, cc] = True
            color = int(a[rr, cc])
            if color != 0:
                hist[color] += 1
                if index == 0:
                    front_hist[color] += 1
                if first:
                    first_hist[color] += 1
                    first = False
    q = len(hist)
    if not 2 <= q <= 4:
        raise ValueError("need two to four distinct colors on the corner rays")
    winner = _mode(hist)
    scale = q + 1
    if max(h, w) * scale > 30:
        raise ValueError("enlarged grid exceeds 30")
    voters = Counter(int(x) for x in a.flat if int(x) not in (0, marker))
    exclusive = set(voters) - set(hist)
    return dict(grid=a, h=h, w=w, r=r, c=c, size=size, marker=marker,
                rays=rays, ray_mask=ray_mask, hist=hist, winner=winner,
                q=q, scale=scale, front_hist=front_hist, first_hist=first_hist,
                voters=voters, exclusive=exclusive,
                edges=_edges(h, w, r, c, size))


def _render(a, r, c, size, rays, scale, winner):
    """Construction path: work in individual enlarged source-cell blocks."""
    out = np.repeat(np.repeat(a, scale, axis=0), scale, axis=1)
    out[r * scale:(r + size) * scale, c * scale:(c + size) * scale] = winner
    for dr, dc, cells in rays:
        if not cells:
            continue
        rr, cc = cells[0]
        for k in range(scale):
            local_col = k if dr == dc else scale - 1 - k
            out[rr * scale + k, cc * scale + local_col] = winner
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input alone; never modify it."""
    info = _analyze(input_grid)
    return _render(info["grid"], info["r"], info["c"], info["size"],
                   info["rays"], info["scale"], info["winner"])


@lru_cache(maxsize=None)
def _geometries(q, size, kind):
    geometries = []
    for h in range(5, 30 // (q + 1) + 1):
        for w in range(5, 30 // (q + 1) + 1):
            for r in range(h - size + 1):
                for c in range(w - size + 1):
                    edge = _edges(h, w, r, c, size)
                    if kind == "interior" and edge:
                        continue
                    if kind in ("top", "bottom", "left", "right") and edge != (kind,):
                        continue
                    if kind == "corner" and len(edge) != 2:
                        continue
                    rays = _rays(h, w, r, c, size)
                    capacity = sum(len(cells) for _, _, cells in rays)
                    if capacity < q + 1:
                        continue
                    if kind == "contrast":
                        # A long ray can hold at least four distant votes,
                        # while two other rays each contribute a nearer rival.
                        if edge or max(len(cells) for _, _, cells in rays) < 5:
                            continue
                    geometries.append((h, w, r, c))
    return tuple(geometries)


@lru_cache(maxsize=None)
def _profiles(q, capacity, forbidden_winner_count):
    result = []
    for win_count in range(2, min(9, capacity - q + 1) + 1):
        if win_count == forbidden_winner_count:
            continue
        for other in product(range(1, win_count), repeat=q - 1):
            profile = (win_count,) + other
            if sum(profile) <= capacity:
                result.append(profile)
    return tuple(result)


def _make_pair(rng, q, size, kind, used_inputs, winner=None,
               forbidden_winner_count=-1):
    geometries = _geometries(q, size, kind)
    if not geometries:
        raise RuntimeError("infeasible geometry request")
    for _ in range(256):
        h, w, r, c = geometries[int(rng.integers(len(geometries)))]
        rays = _rays(h, w, r, c, size)
        ray_cells = [p for _, _, cells in rays for p in cells]
        capacity = len(ray_cells)
        if winner is None:
            win_color = int(rng.integers(1, 10))
        else:
            win_color = int(winner)
        marker = int(rng.choice([x for x in range(1, 10) if x != win_color]))
        remaining = [x for x in range(1, 10) if x not in (marker, win_color)]
        others = [int(x) for x in rng.choice(remaining, size=q - 1, replace=False)]
        active = [win_color] + others
        spare = [x for x in remaining if x not in others]
        exclusive = [int(x) for x in rng.choice(spare, size=int(rng.integers(1, 4)),
                                               replace=False)]
        a = np.zeros((h, w), dtype=np.int8)
        a[r:r + size, c:c + size] = marker
        if kind == "contrast":
            long_choices = [i for i, (_, _, cells) in enumerate(rays) if len(cells) >= 5]
            long_index = int(rng.choice(long_choices))
            far = rays[long_index][2][1:]
            win_count = int(rng.integers(4, min(7, len(far)) + 1))
            selected = rng.choice(len(far), size=win_count, replace=False)
            for index in selected:
                a[far[int(index)]] = win_color
            other_rays = [i for i in range(4) if i != long_index]
            for index in rng.choice(other_rays, size=2, replace=False):
                a[rays[int(index)][2][0]] = others[0]
            profile = (win_count, 2)
        else:
            profiles = _profiles(q, capacity, forbidden_winner_count)
            if not profiles:
                continue
            profile = profiles[int(rng.integers(len(profiles)))]
            values = [color for color, count in zip(active, profile) for _ in range(count)]
            positions = rng.choice(capacity, size=len(values), replace=False)
            rng.shuffle(values)
            for index, value in zip(positions, values):
                a[ray_cells[int(index)]] = value

        # An off-ray rival wins the GLOBAL vote, sometimes sharing a color
        # with an on-ray rival, sometimes having an exclusively off-ray color.
        ray_set = set(ray_cells)
        off = [(rr, cc) for rr in range(h) for cc in range(w)
               if (rr, cc) not in ray_set
               and not (r <= rr < r + size and c <= cc < c + size)]
        challenger = int(rng.choice(others if rng.random() < 0.7 else exclusive))
        target = profile[0] + int(rng.integers(2, 5))
        baseline = dict(zip(active, profile)).get(challenger, 0)
        need = target - baseline
        extras = [color for color in exclusive if color != challenger]
        extra_values = [color for color in extras for _ in range(int(rng.integers(1, 3)))]
        if need + len(extra_values) > len(off):
            continue
        rng.shuffle(off)
        if rng.random() < 0.5:
            center = off[int(rng.integers(len(off)))]
            off.sort(key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1]))
        for p in off[:need]:
            a[p] = challenger
        free = off[need:]
        rng.shuffle(free)
        for p, color in zip(free, extra_values):
            a[p] = color
        free = free[len(extra_values):]
        totals = Counter(int(x) for x in a.flat if int(x) not in (0, marker))
        for p in free[:int(rng.integers(5))]:
            choices = [color for color in active + exclusive
                       if color != challenger and totals[color] < target - 1]
            if choices:
                color = int(rng.choice(choices))
                a[p] = color
                totals[color] += 1
        try:
            info = _analyze(a)
        except ValueError:
            continue
        if (info["marker"] != marker or info["winner"] != win_color
                or info["q"] != q or not info["exclusive"]
                or _mode(info["voters"]) == win_color):
            continue
        if not any(cells and int(a[cells[0]]) != win_color for _, _, cells in rays):
            continue
        sig = _signature(a)
        if sig in used_inputs:
            continue
        used_inputs.add(sig)
        return {"input": a, "output": _render(a, r, c, size, rays, q + 1, win_color)}
    raise RuntimeError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    size_plan = int(rng.choice(3, p=[0.5, 0.2, 0.3]))
    if size_plan == 2:
        train_sizes = [2, 3] + [int(rng.choice([2, 3])) for _ in range(num_train - 2)]
    else:
        train_sizes = [2 if size_plan == 0 else 3] * num_train
    test_sizes = [int(rng.choice(sorted(set(train_sizes)))) for _ in range(num_test)]
    shift = str(rng.choice(["palette", "boundary"]))
    edge = str(rng.choice(["top", "bottom", "left", "right"]))
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[edge]
    initial_winners = [int(x) for x in rng.choice(np.arange(1, 10), size=2, replace=False)]
    used = set()
    train = [_make_pair(rng, 2, train_sizes[0], "contrast", used, initial_winners[0])]
    first_count = _analyze(train[0]["input"])["hist"][initial_winners[0]]
    train.append(_make_pair(rng, 3, train_sizes[1], edge, used, initial_winners[1], first_count))
    for index in range(2, num_train):
        q = int(rng.choice([2, 3] if shift == "palette" else [2, 3, 4]))
        kind = str(rng.choice(["interior", edge]))
        # Four-color electorates fit only certain compact interior geometries.
        if not _geometries(q, train_sizes[index], kind):
            kind = "interior"
        train.append(_make_pair(rng, q, train_sizes[index], kind, used))
    test = []
    for index in range(num_test):
        if shift == "palette" and index == 0:
            q, kind = 4, "interior"
        elif shift == "boundary":
            q = int(rng.choice([2, 3]))
            kind = str(rng.choice([opposite, "corner"]))
        else:
            q = int(rng.choice([2, 3, 4]))
            kind = str(rng.choice(["interior", opposite, "corner"]))
        if not _geometries(q, test_sizes[index], kind):
            kind = "interior"
        test.append(_make_pair(rng, q, test_sizes[index], kind, used))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pixels(a, out, info):
    """Validation path: start at enlarged square corners, walk s pixels out."""
    s, size, r, c = info["scale"], info["size"], info["r"], info["c"]
    assert out.shape == (a.shape[0] * s, a.shape[1] * s)
    untouched = np.kron(a, np.ones((s, s), dtype=np.int8))
    changed = np.zeros(out.shape, dtype=bool)
    changed[r * s:(r + size) * s, c * s:(c + size) * s] = True
    stroke_pixels = 0
    for vertical, horizontal in product((-1, 1), repeat=2):
        rr = r * s if vertical < 0 else (r + size) * s - 1
        cc = c * s if horizontal < 0 else (c + size) * s - 1
        for distance in range(1, s + 1):
            yy, xx = rr + vertical * distance, cc + horizontal * distance
            if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1]:
                changed[yy, xx] = True
                stroke_pixels += int(untouched[yy, xx] != info["winner"])
    assert stroke_pixels >= s, "at least one corner stroke must be visible"
    expected = np.where(changed, info["winner"], untouched)
    assert np.array_equal(out, expected), "incorrect seed fill, stroke, or replication"


def _different_mode(hist, winner):
    try:
        return _mode(hist) != winner
    except ValueError:
        return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(a)
            assert sig not in seen, "duplicate input"
            seen.add(sig)
            info = _analyze(a)
            assert info["exclusive"], "need a color that occurs only off the rays"
            assert _mode(info["voters"]) != info["winner"], "global voting shortcut survives"
            assert len(np.unique(a)) - 1 > info["scale"], "global palette shortcut survives"
            _check_pixels(a, out, info)
            assert np.array_equal(solve(a.copy()), out)
            infos[split].append(info)
    train, test = infos["train"], infos["test"]
    assert {2, 3} <= {i["q"] for i in train}
    assert len({i["winner"] for i in train}) >= 2
    assert len({i["hist"][i["winner"]] for i in train}) >= 2
    assert any(not i["edges"] for i in train) and any(i["edges"] for i in train)
    assert any(_different_mode(i["front_hist"], i["winner"])
               and _different_mode(i["first_hist"], i["winner"])
               and i["hist"][i["winner"]] != i["scale"] for i in train)
    assert {i["size"] for i in test} <= {i["size"] for i in train}
    assert any(i["q"] not in {j["q"] for j in train}
               or i["edges"] not in {j["edges"] for j in train} for i in test)


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(pair["input"]), _signature(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _normalized(a):
    mapping = {}
    return a.shape, tuple(mapping.setdefault(int(x), len(mapping)) for x in a.flat)


def self_test():
    """Bounded diversity, semantic invariants, equivariance and API checks."""
    shapes, outputs, structures, profiles, ray_lengths = set(), set(), set(), set(), set()
    palettes, markers, winners, sizes, qs, edges = set(), set(), set(), set(), set(), set()
    signatures = set()
    shifts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed))
        signatures.add(_puzzle_signature(puzzle))
        train_info = [_analyze(pair["input"]) for pair in puzzle["train"]]
        test_info = _analyze(puzzle["test"][0]["input"])
        shifts["palette" if test_info["q"] not in {i["q"] for i in train_info}
               else "boundary"] += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                before = a.copy()
                assert np.array_equal(solve(a), out) and np.array_equal(a, before)
                info = _analyze(a)
                shapes.add(a.shape)
                outputs.add(out.shape)
                structures.add(_normalized(a))
                profiles.add(tuple(sorted(info["hist"].values(), reverse=True)))
                ray_lengths.add(tuple(len(cells) for _, _, cells in info["rays"]))
                palettes.add(len(np.unique(a)))
                markers.add(info["marker"])
                winners.add(info["winner"])
                sizes.add(info["size"])
                qs.add(info["q"])
                edges.add(info["edges"])
        # Metamorphic checks use only grids, including previously unseen layouts.
        pair = puzzle["test"][0]
        a, out = pair["input"], pair["output"]
        assert np.array_equal(solve(np.rot90(a)), np.rot90(out))
        assert np.array_equal(solve(a.T), out.T)
        permutation = np.asarray([0] + list(range(9, 0, -1)))
        assert np.array_equal(solve(permutation[a]), permutation[out])
        # Removing every off-ray voter leaves the election and scale unchanged.
        info = _analyze(a)
        sparse = a.copy()
        keep = info["ray_mask"].copy()
        keep[info["r"]:info["r"] + info["size"],
             info["c"]:info["c"] + info["size"]] = True
        sparse[~keep] = 0
        sparse_info = _analyze(sparse)
        assert sparse_info["hist"] == info["hist"]
        assert sparse_info["scale"] == info["scale"]
        sparse_out = solve(sparse)
        expanded_keep = np.repeat(np.repeat(keep, info["scale"], axis=0), info["scale"], axis=1)
        assert np.array_equal(sparse_out[expanded_keep], out[expanded_keep])
        assert np.all(sparse_out[~expanded_keep] == 0)
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 15 and len(outputs) >= 15
    assert len(profiles) >= 12 and len(ray_lengths) >= 30
    assert qs == {2, 3, 4} and sizes == {2, 3}
    assert markers == set(range(1, 10)) and winners == set(range(1, 10))
    assert len(palettes) >= 4 and len(edges) >= 8
    assert shifts["palette"] >= 8 and shifts["boundary"] >= 8
    # Exercise every legal explicit count, not just selected boundaries.
    for nt in range(2, 11):
        for nv in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7300 + nt * 10 + nv, nt, nv)
            assert len(p["train"]) == nt and len(p["test"]) == nv
    for nt, nv in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {nt}/{nv}")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "output_shapes": len(outputs),
            "normalized_inputs": len(structures), "vote_profiles": len(profiles),
            "ray_length_patterns": len(ray_lengths), "palette_sizes": sorted(palettes),
            "seed_sizes": sorted(sizes), "active_color_counts": sorted(qs),
            "test_shifts": dict(shifts)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
