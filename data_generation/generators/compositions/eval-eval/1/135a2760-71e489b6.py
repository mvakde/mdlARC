"""Periodic phase quarantine: infer a weave, then broadcast asymmetric faults.

No files, generation metadata, examples, or random state are read by solve().
All experimentation is in memory; this module writes no artifacts or traces.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1, constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for y in range(3):
        for x in range(3):
            result |= p[y:y+h, x:x+w]
    return result


def _parse(input_grid):
    a = _grid(input_grid)
    if min(a.shape) < 7:
        raise ValueError("frame, margin, and repeated interior are required")
    alarm, field = int(a[0, 0]), int(a[1, 1])
    if alarm == field:
        raise ValueError("frame and field colors must differ")
    if not (np.all(a[[0, -1], :] == alarm) and np.all(a[:, [0, -1]] == alarm)):
        raise ValueError("outer frame must be intact")
    if not (np.all(a[[1, -2], 1:-1] == field)
            and np.all(a[1:-1, [1, -2]] == field)):
        raise ValueError("input margin must be intact")
    core = a[2:-2, 2:-2]
    colors = set(map(int, np.unique(core)))
    if len(colors) != 2 or field not in colors or alarm in colors:
        raise ValueError("core must use field and one distinct solid color")
    solid = next(iter(colors - {field}))
    return a, field, solid, alarm, core == solid


def _limit(size):
    return max(2, size // 24)


def _recover(observed):
    """Enumerate all sparsely damaged periodic explanations, using pixels only.

    A phase must have at least three observations on EACH axis. Every admitted
    phase has a strict majority. All explanations within the error budget must
    agree on the whole clean core; a smaller tile is not allowed to hide a tie.
    """
    h, w = observed.shape
    rr, cc = np.indices((h, w))
    chosen = None
    periods = []
    for p in range(1, h // 3 + 1):
        for q in range(1, w // 3 + 1):
            address = (rr % p) * q + cc % q
            totals = np.bincount(address.ravel(), minlength=p*q)
            ones = np.bincount(address.ravel(), weights=observed.ravel(), minlength=p*q)
            if np.any(2*ones == totals):
                continue
            errors = int(np.minimum(ones, totals-ones).sum())
            if errors > _limit(observed.size):
                continue
            tile = (2*ones > totals).reshape(p, q)
            clean = tile[rr % p, cc % q]
            if clean.all() or not clean.any():
                continue
            if chosen is None:
                chosen = clean
            elif not np.array_equal(chosen, clean):
                raise ValueError("competing periodic completions")
            periods.append((p, q))
    if chosen is None:
        raise ValueError("no supported periodic completion")
    # The unique clean core determines its primitive rectangular period.
    p, q = min(periods, key=lambda pq: (pq[0]*pq[1], pq))
    return chosen, p, q


def _decode(input_grid):
    a, field, solid, alarm, observed = _parse(input_grid)
    clean, p, q = _recover(observed)
    holes = clean & ~observed
    extras = ~clean & observed
    if not holes.any():
        raise ValueError("at least one missing solid cell is required")
    rr, cc = np.indices(clean.shape)
    bad = np.zeros((p, q), dtype=bool)
    yy, xx = np.nonzero(holes)
    bad[yy % p, xx % q] = True
    targets = clean & bad[rr % p, cc % q]
    full_targets = np.zeros_like(a, dtype=bool)
    full_targets[2:-2, 2:-2] = targets
    halo = _dilate(full_targets) & ~full_targets
    return {"grid": a, "field": field, "solid": solid, "alarm": alarm,
            "observed": observed, "clean": clean, "p": p, "q": q,
            "holes": holes, "extras": extras, "bad": bad,
            "targets": targets, "full_targets": full_targets, "halo": halo}


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    out = d["grid"].copy()
    out[2:-2, 2:-2] = np.where(d["clean"], d["solid"], d["field"])
    out[d["full_targets"]] = d["field"]
    out[d["halo"]] = d["alarm"]
    return out


def _tile(rng, p, q, style):
    if style == 0:
        return rng.random((p, q)) < rng.uniform(.35, .68)
    tile = np.zeros((p, q), dtype=bool)
    if style == 1:
        for _ in range(int(rng.integers(2, 5))):
            y, v = sorted(map(int, rng.choice(p+1, 2, replace=False)))
            x, u = sorted(map(int, rng.choice(q+1, 2, replace=False)))
            tile[y:v, x:u] = True
    else:
        y, x = int(rng.integers(p)), int(rng.integers(q))
        for _ in range(int(rng.integers(p+q, 2*(p+q)+1))):
            tile[y, x] = True
            dy, dx = ((0, 1), (0, -1), (1, 0), (-1, 0))[int(rng.integers(4))]
            y, x = (y+dy) % p, (x+dx) % q
    if rng.random() < .3:
        tile = ~tile
    return tile


def _primitive(tile):
    p, q = tile.shape
    return (all(not np.array_equal(tile, np.roll(tile, k, axis=0)) for k in range(1, p))
            and all(not np.array_equal(tile, np.roll(tile, k, axis=1)) for k in range(1, q)))


def _extent(rng, period, partial):
    if period == 1:
        return int(rng.integers(4, 10))
    tail = int(rng.integers(1, min(period, 27-4*period))) if partial else 0
    repeats = int(rng.integers(4, (26-tail)//period+1))
    return repeats*period+tail


def _paint(tile, height, width, bad, holes, extras, colors):
    """Construction renderer. Explicit tile/masks, separate from input decoding."""
    field, solid, alarm = colors
    p, q = tile.shape
    clean = np.tile(tile, ((height+p-1)//p, (width+q-1)//q))[:height, :width]
    infected = np.tile(bad, ((height+p-1)//p, (width+q-1)//q))[:height, :width]
    inp = np.full((height+4, width+4), field, dtype=np.int8)
    inp[[0, -1], :], inp[:, [0, -1]] = alarm, alarm
    inp[2:-2, 2:-2] = np.where(clean, solid, field)
    inp[2:-2, 2:-2][holes] = field
    inp[2:-2, 2:-2][extras] = solid
    out = inp.copy()
    out[2:-2, 2:-2] = np.where(clean & ~infected, solid, field)
    # Explicit translated 3x3 stamps; every target center is protected last.
    for y, x in np.argwhere(infected):
        out[y+1:y+4, x+1:x+4] = alarm
    out[2:-2, 2:-2][infected] = field
    return {"input": inp, "output": out}, clean


def _make_pair(rng, index, is_test, colors):
    one_axis = not is_test and index != 1 and rng.random() < .25
    for _ in range(160):
        # The first two training examples teach a single fault and interacting
        # faults. Other examples sometimes collapse one period to one (bands).
        p, q = map(int, rng.integers(4 if (is_test or index == 1) else 3, 7, size=2))
        if one_axis:
            if rng.integers(2): p = 1
            else: q = 1
        if is_test:
            partial_y = partial_x = True
        elif index == 0:
            partial_y = partial_x = False
        else:
            partial_y = bool(rng.integers(2))
            partial_x = not partial_y
        h, w = _extent(rng, p, partial_y), _extent(rng, q, partial_x)
        tile = _tile(rng, p, q, int(rng.integers(3)))
        if not tile.any() or tile.all() or not _primitive(tile):
            continue
        ones = list(map(tuple, np.argwhere(tile)))
        k = (int(rng.integers(2, 4)) if is_test else
             1 if index == 0 else 2 if index == 1 else int(rng.integers(1, 4)))
        if len(ones) < k:
            continue
        first = ones[int(rng.integers(len(ones)))]
        selected = [first]
        # Nearby addresses make their halos merge, including across tile seams.
        while len(selected) < k:
            nearby = [z for z in ones if z not in selected and any(
                max(min(abs(z[0]-s[0]), p-abs(z[0]-s[0])),
                    min(abs(z[1]-s[1]), q-abs(z[1]-s[1]))) <= 1 for s in selected)]
            if not nearby:
                break
            selected.append(nearby[int(rng.integers(len(nearby)))])
        if len(selected) != k:
            continue
        bad = np.zeros_like(tile)
        for y, x in selected:
            bad[y, x] = True
        rr, cc = np.indices((h, w))
        clean = tile[rr % p, cc % q]
        targets = bad[rr % p, cc % q]
        halo = _dilate(targets) & ~targets
        # Preserve visible unaffected structure, and leave a clear erasure site.
        far = np.argwhere(~clean & ~halo)
        near = np.argwhere(~clean & halo)
        if not len(far) or not len(near) or not (clean & ~targets & ~halo).any():
            continue
        holes = np.zeros_like(clean)
        for j, (y, x) in enumerate(selected):
            locations = np.argwhere((rr % p == y) & (cc % q == x))
            n = (int(rng.integers(1, 3)) if is_test else
                 2 if index == 1 and j == 0 else 1)
            for at in rng.choice(len(locations), n, replace=False):
                yy, xx = locations[int(at)]
                holes[yy, xx] = True
        extras = np.zeros_like(clean)
        yy, xx = far[int(rng.integers(len(far)))]
        extras[yy, xx] = True
        if index == 1 and not is_test:
            yy, xx = near[int(rng.integers(len(near)))]
            extras[yy, xx] = True
        budget = min(5, _limit(clean.size)-int(holes.sum()))
        if budget < int(extras.sum()):
            continue
        wanted = int(rng.integers(int(extras.sum()), budget+1))
        options = np.argwhere(~clean & ~extras)
        for at in rng.choice(len(options), wanted-int(extras.sum()), replace=False):
            yy, xx = options[int(at)]
            extras[yy, xx] = True
        pair, built_clean = _paint(tile, h, w, bad, holes, extras, colors)
        try:
            d = _decode(pair["input"])
        except ValueError:
            continue
        if (d["p"] != p or d["q"] != q or not np.array_equal(d["clean"], built_clean)
                or not np.array_equal(d["bad"], bad)):
            continue
        assert np.array_equal(solve(pair["input"]), pair["output"])
        return pair
    raise RuntimeError("bounded sampling exhausted while seeking an unambiguous weave")


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    # Geometrically identified colors may all change between examples.
    for split, count in (("train", a), ("test", b)):
        for index in range(count):
            for _ in range(16):
                colors = tuple(map(int, rng.choice(10, 3, replace=False)))
                pair = _make_pair(rng, index, split == "test", colors)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded duplicate avoidance exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(d):
    halo = d["halo"][2:-2, 2:-2]
    return {"shape": d["grid"].shape, "period": (d["p"], d["q"]),
            "bad_classes": int(d["bad"].sum()), "holes": int(d["holes"].sum()),
            "extras": int(d["extras"].sum()),
            "remote_targets": int((d["targets"] & ~d["holes"]).sum()),
            "far_extras": int((d["extras"] & ~halo).sum()),
            "near_extras": int((d["extras"] & halo).sum()),
            "partial": (d["clean"].shape[0] % d["p"], d["clean"].shape[1] % d["q"])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _decode(inp)
            f = _features(d)
            assert np.array_equal(solve(inp), out)
            assert np.all(out[d["full_targets"]] == d["field"])
            assert np.all(out[d["halo"]] == d["alarm"])
            untouched = ~(d["full_targets"] | d["halo"])
            base = inp.copy()
            base[2:-2, 2:-2] = np.where(d["clean"], d["solid"], d["field"])
            assert np.array_equal(out[untouched], base[untouched])
            assert np.array_equal(out[[0, -1], :], inp[[0, -1], :])
            assert np.array_equal(out[:, [0, -1]], inp[:, [0, -1]])
            assert f["remote_targets"] > 0 and f["far_extras"] > 0
            assert 1 <= f["bad_classes"] <= 3 and 1 <= f["extras"] <= 5
            assert np.count_nonzero(d["clean"] != d["observed"]) <= _limit(d["clean"].size)
            # Omitting A's address broadcast leaves intact remote centers solid.
            assert np.any(d["clean"] & ~d["holes"] & d["targets"])
            # Omitting B's direction distinction incorrectly activates extras.
            assert np.all(~d["targets"][d["extras"]])
            # Both binary roles survive outside the alarms, so outputs stay legible.
            assert (d["clean"] & ~d["targets"] & ~d["halo"][2:-2, 2:-2]).any()
            if split == "train":
                training.append(f)
            else:
                assert all(f["partial"]) and f["bad_classes"] >= 2
    assert training[0]["bad_classes"] == 1 and training[0]["holes"] == 1
    assert training[0]["partial"] == (0, 0)
    assert training[1]["bad_classes"] == 2 and training[1]["holes"] == 3
    assert training[1]["near_extras"] >= 1
    assert any(training[1]["partial"])


def self_test():
    """42 puzzles, streaming measurements, symmetries, counts, and ablations.

    Only SHA-256 digests and small descriptor sets are retained; no sample bank.
    """
    # A small explicit semantic regression, separate from _paint and generation.
    tile = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0],
                     [0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 1],
                     [0, 0, 0, 1, 1, 0]], dtype=np.int8)
    fixture = np.full((25, 29), 2, dtype=np.int8)
    fixture[[0, -1], :], fixture[:, [0, -1]] = 8, 8
    for y in range(21):
        for x in range(25):
            fixture[y+2, x+2] = 5 if tile[y % 5, x % 6] else 2
    expected = fixture.copy()
    centers = {(y+2, x+2) for y in range(21) for x in range(25)
               if y % 5 == 0 and x % 6 in (0, 1)}
    for y, x in centers:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (y+dy, x+dx) not in centers:
                    expected[y+dy, x+dx] = 8
    for y, x in centers:
        expected[y, x] = 2
    fixture[2, 2] = fixture[12, 15] = 2  # Two different missing addresses.
    fixture[4, 2] = fixture[3, 4] = 5    # One far extra and one halo-covered extra.
    assert len(centers) == 45
    assert expected[1, 1] == 8 and expected[2, 3] == 2
    assert np.array_equal(solve(fixture), expected)
    fixture[7, 8] = 2  # A second observed hole at an already bad address.
    assert np.array_equal(solve(fixture), expected)

    shapes, periods, palettes, structures, puzzles = set(), set(), set(), set(), set()
    counts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                d = _decode(inp)
                f = _features(d)
                shapes.add(inp.shape)
                periods.add(f["period"])
                palettes.add((d["field"], d["solid"], d["alarm"]))
                normalized = d["observed"].astype(np.uint8).tobytes()
                structures.add(hashlib.sha256(normalized).digest())
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                counts["pairs"] += 1
                counts["one_axis"] += int(1 in f["period"])
                counts["partial_both"] += int(all(f["partial"]))
                counts["bad_classes_"+str(f["bad_classes"])] += 1
                counts["duplicate_phase_observations"] += int(f["holes"] > f["bad_classes"])
                # Applying B only locally cannot create the remote quarantine.
                local = np.zeros_like(inp, dtype=bool)
                local[2:-2, 2:-2] = d["holes"]
                local_out = inp.copy()
                local_out[2:-2, 2:-2] = np.where(d["clean"], d["solid"], d["field"])
                local_out[local] = d["field"]
                local_out[_dilate(local) & ~local] = d["alarm"]
                assert not np.array_equal(local_out, out)
                # Symmetric error repair (A alone) also differs.
                repaired = inp.copy()
                repaired[2:-2, 2:-2] = np.where(d["clean"], d["solid"], d["field"])
                assert not np.array_equal(repaired, out)
                # A symmetric fault classifier would also alarm on extra specks.
                rr, cc = np.indices(d["clean"].shape)
                flags = d["bad"].copy()
                yy, xx = np.nonzero(d["extras"])
                flags[yy % d["p"], xx % d["q"]] = True
                wrong_targets = np.zeros_like(inp, dtype=bool)
                wrong_targets[2:-2, 2:-2] = flags[rr % d["p"], cc % d["q"]]
                symmetric = repaired.copy()
                symmetric[wrong_targets] = d["field"]
                symmetric[_dilate(wrong_targets) & ~wrong_targets] = d["alarm"]
                assert not np.array_equal(symmetric, out)
                if seed < 3:
                    for turn in range(4):
                        transformed = np.rot90(inp, turn).T.copy()
                        assert np.array_equal(solve(transformed), np.rot90(out, turn).T)
        puzzles.add(digest.digest())
        if seed == 0:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], repeat[s]) for k in p)
    assert len(puzzles) == 42 and len(structures) >= 180
    assert len(shapes) >= 45 and len(periods) >= 18 and len(palettes) >= 80
    assert counts["one_axis"] >= 10 and counts["partial_both"] == 42
    assert all(counts["bad_classes_"+str(k)] > 10 for k in (1, 2, 3))
    # All legal counts, plus type/range/order rejections. No persistent outputs.
    for a in range(2, 11):
        for b in range(1, min(a-1, 4)+1):
            validate_generated_puzzle(generate_puzzle(10000+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "shapes": len(shapes),
            "period_pairs": len(periods), "palettes": len(palettes),
            "normalized_structures": len(structures), "features": dict(counts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
