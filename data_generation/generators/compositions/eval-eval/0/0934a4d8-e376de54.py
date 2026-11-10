"""Mirror shutters: normalize opaque bars and reveal a reflected substrate.

Only Python's standard library and NumPy are used. No file access or generation
state is used by solve. Diagnostics are returned in memory; no traces are saved.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 831658220, 807741258, 690556637, 492574247, 400139550,
              619609005, 187701951, 983500870, 293071935, 123502943, 125504904,
              659140947, 643697834, 903592915, 232074523, 930541774, 103067074,
              732395997, 668463914, 573096303, 386159932, 249076843, 140255253,
              998597797, 436862196, 903789905, 841723611, 998422227, 478081510,
              806377452, 918377185, 923722113, 638789573, 919469631, 877804605,
              868005639, 172097855, 113480456, 368785742, 348919519, 222394567)
_UNITS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    return tuple(a.shape), np.asarray(a, dtype=np.int8).tobytes()


def _orbit_ids(shape):
    r, c = np.indices(shape)
    return np.minimum(r, shape[0] - 1 - r) * 32 + np.minimum(c, shape[1] - 1 - c)


def _wallpaper(a, color, require_support=True):
    ids = _orbit_ids(a.shape)
    visible = a != color
    n = int(ids.max()) + 1
    lo = np.full(n, 10, dtype=np.int8)
    hi = np.full(n, -1, dtype=np.int8)
    np.minimum.at(lo, ids[visible], a[visible])
    np.maximum.at(hi, ids[visible], a[visible])
    counts = np.bincount(ids[visible], minlength=n)
    if np.any((counts > 0) & (lo != hi)):
        return None
    if require_support and len(np.unique(a[visible])) < 2:
        return None
    if require_support and np.count_nonzero(visible & (counts[ids] >= 2)) < max(12, a.size // 4):
        return None
    return ids, hi, counts


def _target_mask(shape, anchors, direction, length):
    result = np.zeros(shape, dtype=bool)
    for r, c in anchors:
        for step in range(length):
            rr, cc = r + step * direction[0], c + step * direction[1]
            if not (0 <= rr < shape[0] and 0 <= cc < shape[1]) or result[rr, cc]:
                return None
            result[rr, cc] = True
    return result


def _geometry(shape, anchors, direction, lengths):
    mask = np.zeros(shape, dtype=bool)
    for anchor, length in zip(anchors, lengths):
        one = _target_mask(shape, [anchor], direction, int(length))
        if one is None or np.any(mask & one):
            return None
        mask |= one
    ref = int(lengths[len(lengths) // 2])
    target = _target_mask(shape, anchors, direction, ref)
    if target is None:
        return None
    revealed = mask & ~target
    ids = _orbit_ids(shape)
    counts = np.bincount(ids[~mask], minlength=int(ids.max()) + 1)
    # Lost orbits are legal only if none of their cells must be revealed.
    if np.any(counts[ids[revealed]] == 0):
        return None
    protected = mask & (counts[ids] == 0)
    return mask, target, revealed, protected


def _routes(mask, revealed):
    """Availability of the two single reflections and their composition."""
    return (~mask[::-1][revealed], ~mask[:, ::-1][revealed],
            ~mask[::-1, ::-1][revealed])


def _mixed_routes(mask, revealed):
    if not np.any(revealed):
        return False
    h, v, _ = _routes(mask, revealed)
    return not bool(np.all(h)) and not bool(np.all(v))


def _analyze(input_grid):
    a = _grid(input_grid)
    candidates = []
    for raw_color in np.unique(a):
        color = int(raw_color)
        field = _wallpaper(a, color)
        if field is None:
            continue
        rr, cc = np.nonzero(a == color)
        for orientation, unit in enumerate(_UNITS):
            if orientation == 0:
                keys, axial, stride = rr, cc, 1
            elif orientation == 1:
                keys, axial, stride = cc, rr, 1
            elif orientation == 2:
                keys, axial, stride = rr - cc, rr + cc, 2
            else:
                keys, axial, stride = rr + cc, rr - cc, 2
            order = np.unique(keys)
            count = len(order)
            if count not in (3, 5, 7, 9):
                continue
            gaps = np.diff(order)
            if np.min(gaps) < 2 * stride or (stride == 2 and np.any(gaps % 2)):
                continue
            groups = []
            for k in order:
                ix = np.flatnonzero(keys == k)
                ix = ix[np.argsort(axial[ix])]
                if np.any(np.diff(axial[ix]) != stride):
                    break
                groups.append(ix)
            if len(groups) != count:
                continue
            same_min = len({int(axial[g[0]]) for g in groups}) == 1
            same_max = len({int(axial[g[-1]]) for g in groups}) == 1
            if same_min == same_max:
                continue
            sign = 1 if same_min else -1
            direction = tuple(sign * x for x in unit)
            anchors = [(int(rr[g[0] if same_min else g[-1]]),
                        int(cc[g[0] if same_min else g[-1]])) for g in groups]
            lengths = tuple(len(g) for g in groups)
            ref = lengths[count // 2]
            if ref < 2:
                continue
            geom = _geometry(a.shape, anchors, direction, lengths)
            if geom is None:
                continue
            mask, target, revealed, protected = geom
            if not np.array_equal(mask, a == color):
                continue
            ids, lookup, counts = field
            candidates.append({"color": color, "anchors": anchors,
                "direction": direction, "lengths": lengths, "reference": ref,
                "count": count, "keys": tuple(int(x) for x in order),
                "gaps": tuple(int(x) for x in gaps), "mask": mask,
                "target": target, "revealed": revealed, "protected": protected,
                "ids": ids, "lookup": lookup, "visible_counts": counts})
    if len(candidates) != 1:
        raise ValueError("Input must determine one shutter array over a mirrored tapestry")
    return a, candidates[0]


def _render(a, info):
    out = a.copy()
    revealed = info["revealed"]
    out[revealed] = info["lookup"][info["ids"][revealed]]
    out[info["target"]] = info["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only input pixels; fully hidden texture that stays hidden is ignored."""
    a, info = _analyze(input_grid)
    return _render(a, info)


def _transform(a, code):
    b = np.rot90(a, code % 4)
    return np.fliplr(b).copy() if code >= 4 else b.copy()


def _lengths(rng, count, reference, capacity, mode, witness):
    result = [reference] * count
    others = [int(i) for i in rng.permutation(count) if i != count // 2]

    def short():
        return int(rng.integers(1, reference))

    def long():
        return int(rng.integers(reference + 1, capacity + 1))

    if witness:
        majority_short = bool(rng.integers(2))
        for i in others:
            result[i] = short() if majority_short else long()
        result[others[0]] = long() if majority_short else short()
    elif mode.startswith("one_"):
        result[others[0]] = short() if mode == "one_extend" else long()
    else:
        for i in others:
            if rng.random() < .22:
                continue
            result[i] = short() if mode == "extend" or (mode == "mixed" and rng.random() < .5) else long()
        if mode == "mixed":
            result[others[0]], result[others[1]] = short(), long()
        elif result == [reference] * count:
            result[others[0]] = short() if mode == "extend" else long()
    return result


def _sample_geometry(rng, count, witness, protected, avoid_ref, mixed_routes, mode):
    # At most 256 small in-memory geometry attempts; never write retry payloads.
    for _ in range(256):
        if protected:
            h = int(rng.choice(np.arange(13, 30, 2)))
            w = int(rng.integers(18, 31))
            refs = [r for r in range(5, 12) if r not in avoid_ref]
            ref = int(rng.choice(refs))
            center = h // 2
            rows = [center - 2, center, center + 3]
            if rng.random() < .5:
                rows = [center - 3, center, center + 2]
            left = (w - ref) // 2 + int(rng.integers(-1, 2))
            anchors = [(r, left) for r in rows]
            direction = (0, 1)
            lengths = [int(rng.integers(1, ref)), ref,
                       int(rng.integers(ref + 1, min(w - left, ref + 6) + 1))]
            if rng.random() < .5:
                lengths[0], lengths[2] = lengths[2], lengths[0]
            shape = (h, w)
        else:
            diagonal = bool(rng.integers(2))
            if witness:
                gaps = [2] * (count - 1)
                gaps[0 if rng.random() < .5 else -1] += int(rng.choice((5, 7)))
            elif rng.random() < .35:
                gaps = [int(rng.integers(2, 4))] * (count - 1)
            else:
                gaps = [2] * (count - 1)
                for _ in range(int(rng.integers(1, min(10, 27 - sum(gaps)) + 1))):
                    gaps[int(rng.integers(count - 1))] += 1
            span = sum(gaps)
            capacity = min(20, 29 - span) if diagonal else int(rng.integers(12, 24))
            if span > 28 or capacity < 4:
                continue
            refs = [r for r in range(2, min(11, capacity - 1) + 1) if r not in avoid_ref]
            if not refs:
                continue
            ref = int(rng.choice(refs))
            lengths = _lengths(rng, count, ref, capacity, mode, witness)
            positions = [0] + [int(x) for x in np.cumsum(gaps)]
            direction = (1, 1) if diagonal else (0, 1)
            anchors = [(p, -p if diagonal else 0) for p in positions]
            cells = [(r + k * direction[0], c + k * direction[1])
                     for (r, c), length in zip(anchors, lengths)
                     for k in range(max(length, ref))]
            low = tuple(min(p[d] for p in cells) for d in (0, 1))
            box = tuple(max(p[d] for p in cells) - low[d] + 1 for d in (0, 1))
            if max(box) > 30:
                continue
            shape = tuple(int(rng.integers(max(9, n), 31)) for n in box)
            shift = tuple(int(rng.integers(shape[d] - box[d] + 1)) - low[d] for d in (0, 1))
            anchors = [(r + shift[0], c + shift[1]) for r, c in anchors]
        geom = _geometry(shape, anchors, direction, lengths)
        if geom is None:
            continue
        mask, target, revealed, lost = geom
        if protected and np.count_nonzero(lost) < 2:
            continue
        if mode not in ("extend", "one_extend") or witness or protected:
            if len(np.unique(_orbit_ids(shape)[revealed])) < 2:
                continue
        if mixed_routes and not _mixed_routes(mask, revealed):
            continue
        return shape, mask, target, revealed, lengths
    raise RuntimeError("No supported shutter geometry within bounded attempts")


def _tapestry(rng, shape, palette, style, revealed):
    bh, bw = (int((n + 1) // 2) for n in shape)
    if style == 0:
        base = rng.choice(palette, size=(bh, bw))
    elif style == 1:
        weights = rng.dirichlet(np.ones(len(palette))) + .12
        base = rng.choice(palette, size=(bh, bw), p=weights / weights.sum())
    elif style == 2:
        scale = int(rng.integers(2, 4))
        small = rng.choice(palette, size=((bh + scale - 1) // scale, (bw + scale - 1) // scale))
        base = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)[:bh, :bw].copy()
    elif style == 3:
        mh, mw = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        motif = rng.choice(palette, size=(mh, mw))
        base = np.tile(motif, ((bh + mh - 1) // mh, (bw + mw - 1) // mw))[:bh, :bw].copy()
    else:
        rr, cc = np.indices((bh, bw))
        bands = rng.choice(palette, size=max(bh, bw) + 1)
        base = bands[np.minimum(rr, cc)]
    # Ensure two different revealed texture values, when revelation is required.
    # These are base-orbit assignments, so every reflection remains exact.
    selected = np.unique(_orbit_ids(shape)[revealed])
    if len(selected) >= 2:
        chosen = rng.choice(selected, size=2, replace=False)
        for orbit, color in zip(chosen, palette[:2]):
            base[int(orbit) // 32, int(orbit) % 32] = color
    rr = np.minimum(np.arange(shape[0]), shape[0] - 1 - np.arange(shape[0]))
    cc = np.minimum(np.arange(shape[1]), shape[1] - 1 - np.arange(shape[1]))
    return base[rr[:, None], cc[None, :]].astype(np.int8)


def _make_pair(rng, count, *, witness=False, protected=False, avoid_ref=(),
               avoid_color=(), mixed_routes=False, mode="mixed", preferred_style=0):
    # Geometry and texture retries have fixed caps and no retained histories.
    for _ in range(12):
        shape, mask, target, revealed, lengths = _sample_geometry(
            rng, count, witness, protected, avoid_ref, mixed_routes, mode)
        color = int(rng.choice([c for c in range(10) if c not in avoid_color]))
        palette = rng.choice([c for c in range(10) if c != color],
                             size=int(rng.integers(2, 7)), replace=False)
        style = preferred_style if rng.random() < .6 else int(rng.integers(5))
        full = _tapestry(rng, shape, palette, style, revealed)
        inp, out = full.copy(), full.copy()
        inp[mask], out[target] = color, color
        code = int(rng.integers(8))
        pair = {"input": _transform(inp, code), "output": _transform(out, code)}
        try:
            a, info = _analyze(pair["input"])
        except ValueError:
            continue
        if not np.array_equal(_render(a, info), pair["output"]):
            continue
        if witness and not (_nonmedian(info) and _geometric_witness(info)):
            continue
        return pair
    raise RuntimeError("No unambiguous layered example within bounded attempts")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    style = int(rng.integers(5))
    puzzle = {"train": [], "test": []}
    first = _make_pair(rng, 5, witness=True, preferred_style=style)
    _, info = _analyze(first["input"])
    second = _make_pair(rng, 3, protected=True, avoid_ref=(info["reference"],),
                        avoid_color=(info["color"],), preferred_style=style)
    puzzle["train"].extend((first, second))
    seen = {_key(p["input"]) for p in puzzle["train"]}
    test_first_count = int(rng.choice((7, 9)))
    modes = ("mixed", "extend", "trim", "one_extend", "one_trim")
    for split, number in (("train", nt - 2), ("test", nq)):
        for index in range(number):
            count = int(rng.choice((3, 5))) if split == "train" else (
                test_first_count if index % 2 == 0 else 16 - test_first_count)
            mode = str(rng.choice(modes)) if split == "train" else str(rng.choice(("mixed", "trim")))
            for _ in range(12):
                pair = _make_pair(rng, count, mode=mode, mixed_routes=split == "test",
                                  preferred_style=style)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct distinct input")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _nonmedian(info):
    return sorted(info["lengths"])[info["count"] // 2] != info["reference"]


def _geometric_witness(info):
    keys = np.array(info["keys"])
    distances = abs(2 * keys - keys[0] - keys[-1])
    closest = np.flatnonzero(distances == distances.min())
    return (len(closest) == 1 and int(closest[0]) != info["count"] // 2
            and info["lengths"][int(closest[0])] != info["reference"])


def features(input_grid):
    a, f = _analyze(input_grid)
    h, v, both = _routes(f["mask"], f["revealed"])
    delta = [f["reference"] - n for n in f["lengths"]]
    return {"shape": tuple(a.shape), "direction": f["direction"], "count": f["count"],
        "reference": f["reference"], "lengths": f["lengths"], "gaps": f["gaps"],
        "shutter_color": f["color"], "field_palette": tuple(int(c) for c in np.unique(a[~f["mask"]])),
        "unequal": len(set(f["gaps"])) > 1, "nonmedian": bool(_nonmedian(f)),
        "geometric_witness": bool(_geometric_witness(f)),
        "extension": max(delta), "truncation": -min(delta),
        "revealed_cells": int(f["revealed"].sum()),
        "revealed_colors": len(np.unique(f["lookup"][f["ids"][f["revealed"]]])),
        "protected_cells": int(f["protected"].sum()),
        "protected_orbits": len(np.unique(f["ids"][f["protected"]])),
        "mixed_routes": bool(_mixed_routes(f["mask"], f["revealed"])),
        "composed_only_cells": int(np.count_nonzero(~h & ~v & both))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[sp], list) for sp in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            assert _key(a) not in seen, "Duplicate input"
            seen.add(_key(a))
            _, f = _analyze(a)
            assert np.array_equal(b, _render(a, f)), "Wrong layered normalization"
            assert np.array_equal(b == f["color"], f["target"])
            assert np.count_nonzero(b == f["color"]) == f["count"] * f["reference"]
            assert np.array_equal(a != b, f["mask"] ^ f["target"]), "Only free tails may change"
            assert not np.array_equal(a, b)
            assert np.all(f["target"][f["protected"]]), "An unknowable cell was exposed"
            assert _wallpaper(b, f["color"], require_support=False) is not None, "Revealed tapestry conflicts"
            for anchor, length in zip(f["anchors"], f["lengths"]):
                for step in range(min(length, f["reference"])):
                    r = anchor[0] + step * f["direction"][0]
                    c = anchor[1] + step * f["direction"][1]
                    assert a[r, c] == b[r, c] == f["color"]
            if split == "train":
                train.append(features(a))
    assert any(f["nonmedian"] and f["geometric_witness"] for f in train)
    assert any(f["protected_cells"] >= 2 for f in train)
    assert len({f["count"] for f in train}) >= 2
    assert len({f["reference"] for f in train}) >= 2
    assert any(f["revealed_colors"] >= 2 for f in train)
    assert any(f["extension"] > 0 for f in train)
    assert any(f["truncation"] > 0 for f in train)


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed campaign, ablations, metamorphisms and count checks."""
    signatures, rows = set(), {"train": [], "test": []}
    rng = np.random.default_rng(778313)
    changed_reference_checks = 0
    changed_texture_checks = 0
    for index, seed in enumerate(HOST_SEEDS):
        nt = 2 + index % 9
        nq = 1 + (index // 9) % min(4, nt - 1)
        puzzle = generate_puzzle(seed, nt, nq)
        signatures.add(_signature(puzzle))
        if index in (0, 17, 41):
            assert _signature(puzzle) == _signature(generate_puzzle(seed, nt, nq))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, f = _analyze(pair["input"])
                b = pair["output"]
                assert np.array_equal(solve(a.copy()), b)
                rows[split].append(features(a))
                # Uniform-background erasure must fail on informative examples.
                if len(np.unique(b[f["revealed"]])) >= 2:
                    uniform = a.copy()
                    values, counts = np.unique(a[~f["mask"]], return_counts=True)
                    uniform[f["revealed"]] = values[np.argmax(counts)]
                    uniform[f["target"]] = f["color"]
                    assert not np.array_equal(uniform, b)
                # Restoration alone omits B: preserve no newly extended shutter.
                if np.any(f["target"] & ~f["mask"]):
                    restoration = a.copy()
                    known = f["mask"] & (f["lookup"][f["ids"]] >= 0)
                    restoration[known] = f["lookup"][f["ids"][known]]
                    assert not np.array_equal(restoration, b)
                if f["protected"].any():
                    assert np.all(f["lookup"][f["ids"][f["protected"]]] == -1)
                    assert np.all(b[f["protected"]] == f["color"])
        # Each seed has a genuinely bigger test array and source-route shift.
        assert {features(p["input"])["count"] for p in puzzle["train"]} <= {3, 5}
        assert {features(p["input"])["count"] for p in puzzle["test"]} <= {7, 9}
        assert all(features(p["input"])["mixed_routes"] for p in puzzle["test"])
        if index < 8:
            for pair in puzzle["train"][:2] + puzzle["test"][:1]:
                a, f = _analyze(pair["input"])
                if f["revealed"].any():
                    orbit = int(f["ids"][f["revealed"]][0])
                    alternate = next(c for c in range(10) if c not in np.unique(a))
                    changed = a.copy()
                    changed[(f["ids"] == orbit) & ~f["mask"]] = alternate
                    expected = pair["output"].copy()
                    expected[(f["ids"] == orbit) & ~f["target"]] = alternate
                    assert np.array_equal(solve(changed), expected)
                    assert not np.array_equal(expected, pair["output"])
                    changed_texture_checks += 1
                middle = f["anchors"][f["count"] // 2]
                grown = _target_mask(a.shape, [middle], f["direction"], f["reference"] + 1)
                if grown is not None:
                    changed = a.copy()
                    changed[grown] = f["color"]
                    try:
                        _, altered = _analyze(changed)
                    except ValueError:
                        pass
                    else:
                        assert altered["reference"] == f["reference"] + 1
                        assert not np.array_equal(solve(changed), pair["output"])
                        changed_reference_checks += 1
                for code in range(8):
                    assert np.array_equal(solve(_transform(pair["input"], code)),
                                          _transform(pair["output"], code))
                perm = rng.permutation(10).astype(np.int8)
                assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    assert len(signatures) == 42
    assert changed_reference_checks > 0 and changed_texture_checks > 0
    for split, fs in rows.items():
        assert len({f["shape"] for f in fs}) >= 20
        assert len({f["direction"] for f in fs}) == 8
        assert len({f["reference"] for f in fs}) >= 7
        assert len({f["shutter_color"] for f in fs}) == 10
        assert any(f["unequal"] for f in fs)
        assert any(not f["unequal"] for f in fs)
        assert any(f["composed_only_cells"] > 0 for f in fs)
        assert max(f["truncation"] for f in fs) >= 5
    assert any(f["protected_orbits"] > 1 for f in rows["train"])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1931 + nt * 10 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    # Corrupting a label must be detected.
    p = generate_puzzle(991, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return {"seeds": 42, "puzzles": len(signatures),
            "pairs": sum(len(v) for v in rows.values()),
            "shapes": {sp: len({f["shape"] for f in fs}) for sp, fs in rows.items()},
            "protected_examples": sum(f["protected_cells"] > 0 for fs in rows.values() for f in fs),
            "composed_only_examples": sum(f["composed_only_cells"] > 0 for fs in rows.values() for f in fs),
            "changed_reference_checks": changed_reference_checks,
            "changed_texture_checks": changed_texture_checks}
