"""Pinwheel block printing: directed motifs and dye at geometric bends.

Standalone: standard library and NumPy only. No files, parent generators,
seed lookups, global RNG state, or generated-example state enter solve().
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _full_box(mask):
    return (mask.shape == (3, 3) and 2 <= int(mask.sum()) <= 8
            and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any())


def _orbit_size(mask):
    return len({np.rot90(m, k).tobytes()
                for m in (mask, mask[:, ::-1]) for k in range(4)})


_MASKS = tuple(m for code in range(1, 511)
               if _full_box(m := np.array([(code >> k) & 1 for k in range(9)],
                                          dtype=bool).reshape(3, 3)))
_DENSITIES = np.array([int(m.sum()) for m in _MASKS])
_ASYMMETRIC = np.array([_orbit_size(m) == 8 for m in _MASKS])


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return grid


def _key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _decode(value):
    grid = _grid(value)
    h, w = grid.shape
    if h % 3 or w % 3 or min(h, w) < 15:
        raise ValueError("the field must contain 5..10 whole blocks per axis")
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3 or int(counts.max()) <= grid.size // 2:
        raise ValueError("expected three colors and a majority background")
    bg = int(colors[np.argmax(counts)])
    singletons = colors[counts == 1]
    if len(singletons) != 1:
        raise ValueError("expected one singleton dye color")
    dye = int(singletons[0])
    ink = next(int(c) for c in colors if c not in (bg, dye))
    y, x = (int(v) for v in np.argwhere(grid == dye)[0])
    if y % 3 != 1 or x % 3 != 1:
        raise ValueError("dye must be at the center of its hub block")
    cy, cx = y // 3, x // 3
    shape = (h // 3, w // 3)
    active = (grid == ink).reshape(shape[0], 3, shape[1], 3).any(axis=(1, 3))
    n = int(active.sum())
    if n % 4 or not 1 <= n // 4 <= 3:
        raise ValueError("expected four equal spokes of length 1..3")
    length = n // 4
    if not (length <= cy < shape[0]-length and length <= cx < shape[1]-length):
        raise ValueError("all four seed spokes must be visible")
    mask = (grid[3*(cy-1):3*cy, 3*cx:3*cx+3] == ink)
    if not _full_box(mask):
        raise ValueError("motif must have a full 3x3 bounding box and gaps")
    info = {"bg": bg, "ink": ink, "dye": dye, "center": (cy, cx),
            "shape": shape, "length": length, "mask": mask.copy()}
    if not np.array_equal(grid, _seed_grid(info)):
        raise ValueError("spokes must be exact outward-rotated motif copies")
    return info


def _seed_grid(info):
    h, w = info["shape"]
    cy, cx = info["center"]
    result = np.full((3*h, 3*w), info["bg"], dtype=np.int8)
    for phase, (dy, dx) in enumerate(_DIRS):
        motif = np.rot90(info["mask"], -phase)
        for j in range(1, info["length"]+1):
            y, x = cy+j*dy, cx+j*dx
            result[3*y:3*y+3, 3*x:3*x+3][motif] = info["ink"]
    result[3*cy+1, 3*cx+1] = info["dye"]
    return result


def _routes(shape, center, length, *, hand=1, ordinary=False):
    """Visible cells mapped to (incoming heading, outgoing heading, thread).

    Walk on the infinite plane. Final endpoints are far outside the field;
    all visible points have a real next step, including at the boundary.
    No full-plane grid or saved execution trace is allocated.
    """
    h, w = shape
    cy, cx = center
    radius = max(2, length)
    distance = max(cy, cx, h-1-cy, w-1-cx)
    stages = 4 * (distance // 8 + 2)
    records = {}
    for phase in range(4):
        segments = [(phase, radius)]
        for k in range(stages):
            forward = (phase+hand*(k+1)) % 4
            if ordinary:
                segments.append((forward, 4*k+2+radius))
            else:
                segments.extend(((forward, 4*k+2),
                                 ((forward+hand) % 4, radius-2),
                                 (forward, radius)))
        y, x, incoming = cy, cx, phase
        for outgoing, steps in segments:
            dy, dx = _DIRS[outgoing]
            for _ in range(steps):
                if (y, x) != center and 0 <= y < h and 0 <= x < w:
                    assert (y, x) not in records, "threads must not intersect"
                    records[y, x] = incoming, outgoing, phase
                y, x = y+dy, x+dx
                incoming = outgoing
        assert not (0 <= y < h and 0 <= x < w)
    return records


def _closed_routes(shape, center, length):
    """Author-owned cross-check using independent closed-form stage origins.

    It does not advance endpoints from one stage to the next. This is an
    implementation cross-check, not an independent semantic oracle.
    """
    h, w = shape
    cy, cx = center
    radius = max(2, length)
    records = {}
    bound = max(cy, cx, h-1-cy, w-1-cx)

    def emit(y, x, incoming, outgoing, phase):
        for _ in range(phase):
            y, x = x, -y
        y, x = y+cy, x+cx
        if 0 <= y < h and 0 <= x < w:
            assert (y, x) not in records
            records[y, x] = ((incoming+phase) % 4,
                             (outgoing+phase) % 4, phase)

    for phase in range(4):
        for j in range(1, radius+1):
            emit(-j, 0, 0, 0 if j < radius else 1, phase)
        for q in range(bound // 8+3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for s, (y, x) in enumerate(starts):
                forward = (s+1) % 4
                right = (forward+1) % 4
                a, jog = 16*q+4*s+2, radius-2
                dy, dx = _DIRS[forward]
                ey, ex = _DIRS[right]
                for j in range(1, a+1):
                    emit(y+j*dy, x+j*dx, forward,
                         right if j == a and jog else forward, phase)
                for j in range(1, jog+1):
                    emit(y+a*dy+j*ey, x+a*dx+j*ex, right,
                         forward if j == jog else right, phase)
                for j in range(1, radius+1):
                    emit(y+(a+j)*dy+jog*ey, x+(a+j)*dx+jog*ex,
                         forward, right if j == radius else forward, phase)
    return records


def _print(info, records, variant="normal"):
    result = np.full(tuple(3*n for n in info["shape"]), info["bg"], dtype=np.int8)
    motifs = [np.rot90(info["mask"], -k) for k in range(4)]
    for (y, x), (incoming, outgoing, phase) in records.items():
        bend = incoming != outgoing
        direction = phase if variant == "no_turn_rotation" else (
            outgoing if variant == "outgoing_orientation" else incoming)
        motif = motifs[direction]
        if variant == "solid" or (variant == "solid_bends" and bend):
            motif = np.ones((3, 3), dtype=bool)
        colored = bend
        if variant == "no_dye":
            colored = False
        elif variant == "all_dye":
            colored = True
        result[3*y:3*y+3, 3*x:3*x+3][motif] = info["dye"] if colored else info["ink"]
    cy, cx = info["center"]
    result[3*cy+1, 3*cx+1] = info["dye"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible geometry, motif and hub dye; never modify the input."""
    info = _decode(input_grid)
    return _print(info, _routes(info["shape"], info["center"], info["length"]))


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _choose_mask(rng, allowed):
    eligible = np.flatnonzero(allowed)
    if not len(eligible):
        raise AssertionError("finite motif pool exhausted")
    density = int(rng.choice(np.unique(_DENSITIES[eligible])))
    return int(rng.choice(eligible[_DENSITIES[eligible] == density]))


def _configurations(length, edge=False):
    low = max(5, 2*length+1)
    return [(length, (h, w), (y, x))
            for h in range(low, 11) for w in range(low, 11)
            for y in range(length, h-length) for x in range(length, w-length)
            if not edge or min(y-length, x-length,
                               h-1-y-length, w-1-x-length) == 0]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    bg, ink, dye = (int(c) for c in rng.choice(10, 3, replace=False))
    regime = int(rng.integers(3))
    if regime == 0:
        train_allowed = np.array([bool(m[1, 1]) for m in _MASKS])
        test_allowed = ~train_allowed
    elif regime == 1:
        train_allowed = (_DENSITIES >= 3) & (_DENSITIES <= 5)
        test_allowed = _DENSITIES >= 6
    else:
        train_allowed = _DENSITIES >= 5
        test_allowed = _DENSITIES <= 4
    first = _choose_mask(rng, train_allowed & _ASYMMETRIC)
    second = _choose_mask(rng, train_allowed & _ASYMMETRIC
                          & (_DENSITIES != _DENSITIES[first]))
    ids = [first, second]
    for index in range(2, ntrain+ntest):
        allowed = (train_allowed if index < ntrain else test_allowed).copy()
        allowed[ids] = False
        ids.append(_choose_mask(rng, allowed))
    lengths = [int(n) for n in rng.permutation([2, 3])]
    if ntrain >= 3:
        lengths.append(1)
    lengths.extend(int(rng.integers(1, 4)) for _ in range(ntrain-len(lengths)))
    lengths.extend(int(rng.choice(sorted(set(lengths)))) for _ in range(ntest))
    puzzle = {"train": [], "test": []}
    used_configs = set()
    for index, (mask_id, length) in enumerate(zip(ids, lengths)):
        if index < 2:
            shape = tuple(int(n) for n in rng.integers(9, 11, size=2))
            center = tuple(int(rng.integers(4, n-4)) for n in shape)
            config = length, shape, center
        else:
            configs = _configurations(length, edge=index >= ntrain)
            if index >= ntrain:
                configs = [c for c in configs if c not in used_configs]
            config = configs[int(rng.integers(len(configs)))]
        used_configs.add(config)
        _, shape, center = config
        info = {"bg": bg, "ink": ink, "dye": dye, "shape": shape,
                "center": center, "length": length, "mask": _MASKS[mask_id]}
        inp = _seed_grid(info)
        pair = {"input": inp, "output": solve(inp)}
        puzzle["train" if index < ntrain else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _teaching_alternatives(info):
    records = _routes(info["shape"], info["center"], info["length"])
    alternatives = {v: _print(info, records, v) for v in (
        "no_turn_rotation", "outgoing_orientation", "no_dye", "all_dye", "solid_bends")}
    alternatives["counterclockwise"] = _print(info, _routes(
        info["shape"], info["center"], info["length"], hand=-1))
    alternatives["ordinary_spiral"] = _print(info, _routes(
        info["shape"], info["center"], info["length"], ordinary=True))
    return alternatives


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    infos = {"train": [], "test": []}
    inputs, masks, configs = set(), set(), set()
    palettes = set()
    rejected = set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            info = _decode(inp)
            infos[split].append(info)
            assert inp.shape == out.shape and _key(inp) not in inputs
            inputs.add(_key(inp))
            assert info["mask"].tobytes() not in masks
            masks.add(info["mask"].tobytes())
            palettes.add((info["bg"], info["ink"], info["dye"]))
            records = _closed_routes(info["shape"], info["center"], info["length"])
            assert np.array_equal(out, _print(info, records)), "wrong printed winding"
            # Explicit family invariants, beyond equality with a renderer.
            assert set(int(c) for c in np.unique(out)) == {info[k] for k in ("bg", "ink", "dye")}
            assert np.all(out[inp == info["ink"]] != info["bg"])
            assert not np.array_equal(inp, out), "A's continuation must change the answer"
            assert not np.array_equal(out, _print(info, records, "solid")), "B's gaps must matter"
            occupied = int(info["mask"].sum())
            bends = sum(a != b for a, b, _ in records.values())
            assert bends > 0 and len(records) > 4*info["length"]
            assert int((out == info["dye"]).sum()) == 1+bends*occupied
            assert int((out == info["ink"]).sum()) == (len(records)-bends)*occupied
            cy, cx = info["center"]
            assert out[3*cy+1, 3*cx+1] == info["dye"]
            hub = out[3*cy:3*cy+3, 3*cx:3*cx+3]
            assert int((hub == info["bg"]).sum()) == 8
            config = info["length"], info["shape"], info["center"]
            if split == "test":
                assert config not in configs
                h, w = info["shape"]
                l = info["length"]
                assert min(cy-l, cx-l, h-1-cy-l, w-1-cx-l) == 0
            configs.add(config)
            if split == "train" and index < 2:
                assert _orbit_size(info["mask"]) == 8
                for name, alternative in _teaching_alternatives(info).items():
                    if not np.array_equal(out, alternative):
                        rejected.add(name)
    assert len(palettes) == 1
    train, test = infos["train"], infos["test"]
    assert {i["length"] for i in train[:2]} == {2, 3}
    assert int(train[0]["mask"].sum()) != int(train[1]["mask"].sum())
    assert all(min(i["shape"]) >= 9 and min(i["center"]) >= 4
               and all(c <= n-5 for c, n in zip(i["center"], i["shape"])) for i in train[:2])
    assert {i["length"] for i in test} <= {i["length"] for i in train}
    assert rejected == {"no_turn_rotation", "outgoing_orientation", "no_dye", "all_dye",
                        "solid_bends", "counterclockwise", "ordinary_spiral"}
    train_n = [int(i["mask"].sum()) for i in train]
    test_n = [int(i["mask"].sum()) for i in test]
    assert ((all(i["mask"][1, 1] for i in train) and all(not i["mask"][1, 1] for i in test))
            or (min(train_n) >= 3 and max(train_n) <= 5 and min(test_n) >= 6)
            or (min(train_n) >= 5 and max(test_n) <= 4))


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, content = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(content)
    return digest.hexdigest()


def _components(mask):
    pending = {tuple(int(n) for n in p) for p in np.argwhere(mask)}
    count = 0
    while pending:
        count += 1
        stack = [pending.pop()]
        while stack:
            y, x = stack.pop()
            found = {(y+dy, x+dx) for dy, dx in _DIRS} & pending
            pending -= found
            stack.extend(found)
    return count


def self_test():
    """Bounded checks: stream 42 puzzles, retain feature sets and digests only."""
    assert len(_MASKS) == 321
    for length in (1, 2, 3):
        for shape, center in (((10, 10), (length, length)),
                              ((7, 10), (6-length, 9-length)), ((9, 9), (4, 4))):
            assert _routes(shape, center, length) == _closed_routes(shape, center, length)
    records = _routes((10, 10), (4, 4), 3)
    assert records[1, 4] == (0, 1, 0)  # North tip: incoming north, next east.
    assert records[1, 6] == (1, 2, 0)  # First inward jog begins.
    assert records[2, 6] == (2, 1, 0)  # Jog ends: incoming south, next east.
    assert _routes((10, 10), (4, 4), 1) == _routes((10, 10), (4, 4), 2)
    # Every admissible topology is decodable, including motifs with an empty
    # internal row/column and masks symmetric under some or all rotations.
    for length in (1, 2, 3):
        reference = _closed_routes((9, 10), (4, 4), length)
        for mask in _MASKS:
            info = {"bg": 0, "ink": 2, "dye": 8, "shape": (9, 10),
                    "center": (4, 4), "length": length, "mask": mask}
            inp = _seed_grid(info)
            assert np.array_equal(solve(inp), _print(info, reference))
    features = {name: set() for name in ("puzzles", "motifs", "densities", "components",
                "shapes", "layouts", "palettes", "train_lengths", "test_lengths",
                "bend_counts", "test_centers", "train_centers", "outputs")}
    roles = [set(), set(), set()]
    shifts = [0, 0, 0]
    pairs_checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        features["puzzles"].add(_fingerprint(puzzle))
        decoded = {s: [_decode(p["input"]) for p in puzzle[s]] for s in puzzle}
        train, test = decoded["train"], decoded["test"]
        shifts[0] += int(all(i["mask"][1, 1] for i in train) and all(not i["mask"][1, 1] for i in test))
        shifts[1] += int(max(int(i["mask"].sum()) for i in train) < min(int(i["mask"].sum()) for i in test))
        shifts[2] += int(min(int(i["mask"].sum()) for i in train) > max(int(i["mask"].sum()) for i in test))
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], decoded[split]):
                pairs_checked += 1
                features["motifs"].add(info["mask"].tobytes())
                features["densities"].add(int(info["mask"].sum()))
                features["components"].add(_components(info["mask"]))
                features["shapes"].add(info["shape"])
                features["layouts"].add((info["length"], info["shape"], info["center"]))
                features[split+"_lengths"].add(info["length"])
                features[split+"_centers"].add(bool(info["mask"][1, 1]))
                palette = tuple(info[k] for k in ("bg", "ink", "dye"))
                features["palettes"].add(palette)
                for role, color in zip(roles, palette):
                    role.add(color)
                records = _routes(info["shape"], info["center"], info["length"])
                assert records == _closed_routes(info["shape"], info["center"], info["length"])
                features["bend_counts"].add(sum(a != b for a, b, _ in records.values()))
                features["outputs"].add(hashlib.sha256(pair["output"].tobytes()).hexdigest())
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"])
                # Rotation carries both the canonical frame and dye hub.
                assert np.array_equal(solve(np.rot90(before)), np.rot90(pair["output"]))
                permutation = np.array([4, 8, 1, 6, 0, 9, 2, 5, 7, 3], dtype=np.int8)
                assert np.array_equal(solve(permutation[before]), permutation[pair["output"]])
    assert len(features["puzzles"]) == 42
    assert len(features["motifs"]) >= 80 and len(features["shapes"]) >= 20
    assert len(features["layouts"]) >= 100 and len(features["palettes"]) >= 35
    assert features["densities"] == set(range(2, 9))
    assert len(features["components"]) >= 4 and len(features["bend_counts"]) >= 5
    assert features["train_lengths"] == features["test_lengths"] == {1, 2, 3}
    assert features["train_centers"] == features["test_centers"] == {False, True}
    assert all(len(role) == 10 for role in roles) and min(shifts) > 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(1000+10*a+b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1), (4, False), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for variant in ("solid_bends", "outgoing_orientation", "counterclockwise", "pixel"):
        damaged = generate_puzzle(101, 2, 1)
        pair = damaged["test"][0]
        info = _decode(pair["input"])
        records = _routes(info["shape"], info["center"], info["length"])
        if variant == "pixel":
            pair["output"][0, 0] = (int(pair["output"][0, 0])+1) % 10
        elif variant == "counterclockwise":
            pair["output"] = _print(info, _routes(info["shape"], info["center"], info["length"], hand=-1))
        else:
            # Use an asymmetric training witness: symmetric tests can make an
            # orientation ablation equal to the correct answer legitimately.
            pair = damaged["train"][0]
            info = _decode(pair["input"])
            pair["output"] = _print(info, _routes(info["shape"], info["center"], info["length"]), variant)
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"damaged label accepted: {variant}")
    return {"seeds": 42, "pairs": pairs_checked,
            "distinct": {k: len(v) for k, v in features.items()},
            "densities": sorted(features["densities"]),
            "train_lengths": sorted(features["train_lengths"]),
            "test_lengths": sorted(features["test_lengths"]),
            "role_color_counts": [len(r) for r in roles], "shift_witness_counts": shifts,
            "semantic_verification": "author-owned checks; no independent semantic solver"}
