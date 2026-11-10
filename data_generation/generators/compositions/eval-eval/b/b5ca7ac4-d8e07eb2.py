"""Glyph-linked cargo: exact colored keys bind separated tiles into rigid bodies.

The two opposite rails visibly specify both the movement axis and frame map.
No files, training labels, random state, or generator parameters are read by solve.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 290883984, 55288498, 18619750, 990709659, 423528326,
              530106186, 132630569, 543276825, 338650387, 738305894,
              599947224, 62739447, 593314122, 834136869, 716352152,
              860088132, 760471310, 269938776, 540584920, 66687307,
              425307875, 208428264, 178450606, 758081539, 736876256,
              321119021, 723382080, 895729821, 243783996, 892633683,
              570556116, 87520927, 795175478, 535629676, 90999960,
              418661736, 333648996, 982438049, 144824707, 504245838, 8988135)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return g.astype(np.int8, copy=False)


def _key(patch, frame):
    # Full glyph canvas is identity; its bounds are also required to be tight.
    interior = patch[1:-1, 1:-1]
    return int(frame), interior.shape, interior.tobytes()


def _rows_overlap(a, b):
    return max(a["r"], b["r"]) < min(a["r"] + a["h"], b["r"] + b["h"])


def _overlap(a, b):
    return (_rows_overlap(a, b) and
            max(a["c"], b["c"]) < min(a["c"] + a["w"], b["c"] + b["w"]))


def _separated(a, b):
    # At least one all-background cell between source rectangles.
    return (a["r"] + a["h"] < b["r"] or b["r"] + b["h"] < a["r"]
            or a["c"] + a["w"] < b["c"] or b["c"] + b["w"] < a["c"])


def _canonical(g):
    """Put the two uniform rails at left/right, using only visible pixels."""
    choices = []
    for transpose in (False, True):
        a = g.T if transpose else g
        if min(a.shape) < 6:
            continue
        low, high = int(a[0, 0]), int(a[0, -1])
        bg = int(a[0, 1])
        if (len({bg, low, high}) == 3 and np.all(a[:, 0] == low)
                and np.all(a[:, -1] == high)
                and np.all(a[0, 1:-1] == bg)
                and np.all(a[-1, 1:-1] == bg)):
            choices.append((a, transpose, bg, low, high))
    if len(choices) != 1:
        raise ValueError("expected one unambiguous pair of opposite colored rails")
    return choices[0]


def _decode(value):
    g = _grid(value)
    a, transpose, bg, low, high = _canonical(g)
    occupied = np.zeros(a.shape, bool)
    occupied[:, [0, -1]] = True
    objects = []
    for r, c in np.ndindex(a.shape):
        if occupied[r, c] or a[r, c] == bg:
            continue
        frame = int(a[r, c])
        if frame not in (low, high):
            raise ValueError("ink outside a complete frame")
        right, bottom = c, r
        while right < a.shape[1] - 1 and a[r, right] == frame:
            right += 1
        while bottom < a.shape[0] and a[bottom, c] == frame:
            bottom += 1
        h, w = bottom - r, right - c
        if not (4 <= h <= 5 and 4 <= w <= 5):
            raise ValueError("tile must have a 2..3 by 2..3 glyph canvas")
        patch = a[r:bottom, c:right].copy()
        if (np.any(occupied[r:bottom, c:right]) or r == 0 or bottom >= a.shape[0]
                or c < 1 or right > a.shape[1] - 1
                or not np.all(patch[0] == frame) or not np.all(patch[-1] == frame)
                or not np.all(patch[:, 0] == frame) or not np.all(patch[:, -1] == frame)):
            raise ValueError("incomplete, overlapping, or boundary-crossing tile")
        token = patch[1:-1, 1:-1]
        ink = token != bg
        if (np.any((token == low) | (token == high))
                or len(np.unique(token[ink])) != 1
                or not all((np.any(ink[0]), np.any(ink[-1]),
                            np.any(ink[:, 0]), np.any(ink[:, -1])))):
            raise ValueError("glyph must be tight, nonempty, and use one non-frame ink")
        obj = dict(r=r, c=c, h=h, w=w, frame=frame, patch=patch,
                   key=_key(patch, frame))
        if any(not _separated(obj, p) for p in objects):
            raise ValueError("source tiles need a one-cell moat")
        objects.append(obj)
        occupied[r:bottom, c:right] = True
    if not objects or np.any((a != bg) & ~occupied):
        raise ValueError("unsegmented source pixels")
    return a, transpose, bg, low, high, objects


def _settle(objects, width, low, high):
    """Greatest allowed group travel, constrained by every overlapping row span.

    For a follower h behind g, t[h] <= t[g] + their original pixel gap.
    Starting from edge limits and relaxing these upper bounds computes the
    unique maximum travel vector. No same-class pair can overtake.
    """
    keys = list(dict.fromkeys(o["key"] for o in objects))
    travel = {}
    for key in keys:
        group = [o for o in objects if o["key"] == key]
        travel[key] = min(o["c"] - 1 if key[0] == low
                          else width - 1 - o["c"] - o["w"] for o in group)
    edges = []
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            if a["frame"] != b["frame"] or a["key"] == b["key"] or not _rows_overlap(a, b):
                continue
            left, right = (a, b) if a["c"] < b["c"] else (b, a)
            gap = right["c"] - left["c"] - left["w"]
            if gap < 0:
                raise ValueError("source tiles overlap")
            front, back = (left, right) if a["frame"] == low else (right, left)
            edges.append((front["key"], back["key"], gap))
    for _ in range(len(keys)):
        changed = False
        for front, back, gap in edges:
            bound = travel[front] + gap
            if travel[back] > bound:
                travel[back] = bound
                changed = True
        if not changed:
            break
    settled = [dict(o, c=o["c"] + (-1 if o["frame"] == low else 1) * travel[o["key"]])
               for o in objects]
    if any(_overlap(a, b) for i, a in enumerate(settled) for b in settled[i + 1:]):
        raise ValueError("opposite packed classes have overlapping destinations")
    return settled


def _render(shape, bg, low, high, objects):
    result = np.full(shape, bg, np.int8)
    result[:, 0], result[:, -1] = low, high
    for o in objects:
        result[o["r"]:o["r"] + o["h"], o["c"]:o["c"] + o["w"]] = o["patch"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, transpose, bg, low, high, objects = _decode(input_grid)
    settled = _settle(objects, a.shape[1], low, high)
    result = _render(a.shape, bg, low, high, settled)
    return result.T.copy() if transpose else result


def _simulate(objects, width, low, high):
    """Separate validator: repeatedly move whole assemblies by one cell.

    This uses actual rectangles and collisions, not the bound-relaxation solver.
    Other-frame assemblies are transparent during travel, as in parent A.
    """
    current = [dict(o) for o in objects]
    keys = list(dict.fromkeys(o["key"] for o in current))
    for _ in range(width * len(keys) + 1):
        moved = False
        for key in keys:
            delta = -1 if key[0] == low else 1
            indices = [i for i, o in enumerate(current) if o["key"] == key]
            candidates = [dict(current[i], c=current[i]["c"] + delta) for i in indices]
            blockers = [o for o in current if o["frame"] == key[0] and o["key"] != key]
            if (all(1 <= o["c"] and o["c"] + o["w"] <= width - 1 for o in candidates)
                    and not any(_overlap(o, b) for o in candidates for b in blockers)):
                for i, o in zip(indices, candidates):
                    current[i] = o
                moved = True
        if not moved:
            break
    else:
        raise AssertionError("rigid sliding failed to terminate within the travel bound")
    assert not any(_overlap(a, b) for i, a in enumerate(current) for b in current[i + 1:])
    return current


def _edge_first(objects, width, low, high):
    """Deliberately wrong rival: teleport a body into the first available gap."""
    objects = [dict(o, index=i) for i, o in enumerate(objects)]
    settled = []
    for frame in (low, high):
        keys = list(dict.fromkeys(o["key"] for o in objects if o["frame"] == frame))
        keys.sort(key=lambda k: min(o["c"] if frame == low else -o["c"] - o["w"]
                                   for o in objects if o["key"] == k))
        for key in keys:
            group = [o for o in objects if o["key"] == key]
            cap = min(o["c"] - 1 if frame == low else width - 1 - o["c"] - o["w"]
                      for o in group)
            for travel in range(cap, -1, -1):
                candidates = [dict(o, c=o["c"] + (-travel if frame == low else travel)) for o in group]
                if not any(_overlap(o, b) for o in candidates for b in settled if b["frame"] == frame):
                    settled.extend(candidates)
                    break
    return {o["index"]: o["c"] for o in settled}


def _token(rng, h, w, bg, ink, different=None):
    # All nonempty masks with tight bounding boxes, including disconnected masks.
    masks = []
    for code in range(1, 1 << (h * w)):
        mask = np.array([(code >> i) & 1 for i in range(h * w)], bool).reshape(h, w)
        if all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any())):
            token = np.where(mask, ink, bg).astype(np.int8)
            if different is None or not np.array_equal(token, different):
                masks.append(token)
    return masks[int(rng.integers(len(masks)))].copy()


def _object(r, c, frame, token):
    h, w = token.shape
    patch = np.full((h + 2, w + 2), frame, np.int8)
    patch[1:-1, 1:-1] = token
    return dict(r=int(r), c=int(c), h=h + 2, w=w + 2, frame=int(frame),
                patch=patch, key=_key(patch, frame))


def _pair(shape, bg, low, high, objects, transpose):
    settled = _settle(objects, shape[1], low, high)
    inp = _render(shape, bg, low, high, objects)
    out = _render(shape, bg, low, high, settled)
    if transpose:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _lesson(rng, bg, low, high, inks, direction, transpose):
    """A nonlocal blocker chain, with shape-only and color-only decoys."""
    gh, gw = map(int, rng.integers(2, 4, size=2))
    h, w = gh + 2, gw + 2
    gap1 = h + int(rng.integers(2, 5))
    top = int(rng.integers(1, 4))
    middle = top + h + gap1
    height = int(rng.integers(middle + h + 1, 31))
    near = int(rng.integers(2, 5))
    far = near + w + int(rng.integers(3, 6))
    stagger = int(rng.integers(1, 3))
    rear = far + stagger + w + int(rng.integers(1, 4))
    width = int(rng.integers(max(22, rear + w + 1, 2 + 4 * w + stagger), 31))
    token = _token(rng, gh, gw, bg, inks[0])
    other_shape = _token(rng, gh, gw, bg, inks[0], different=token)
    other_color = np.where(token == bg, bg, inks[1]).astype(np.int8)
    objects = [_object(top, far + stagger, low, token),
               _object(middle, far, low, token),
               _object(middle, near, low, other_shape),
               _object(top + h + 1, far, low, other_shape),
               _object(top, rear, low, other_color),
               _object(top, near, high, token)]
    if direction:
        # Reflect positions and exchange frame roles, without reflecting glyphs.
        for o in objects:
            frame = high if o["frame"] == low else low
            o["c"] = width - o["c"] - o["w"]
            o["frame"] = frame
            o["patch"] = o["patch"].copy()
            o["patch"][[0, -1], :] = frame
            o["patch"][:, [0, -1]] = frame
            o["key"] = _key(o["patch"], frame)
    assert all(_separated(a, b) for i, a in enumerate(objects) for b in objects[i + 1:])
    return _pair((height, width), bg, low, high, objects, transpose)


def _inventory(rng, bg, low, high, inks, test):
    count = int(rng.integers(3, 6))
    groups, used = [], set()
    for j in range(count):
        frame = (low, high)[j % 2] if j < 2 else int(rng.choice([low, high]))
        for _ in range(128):
            gh, gw = map(int, rng.integers(2, 4, size=2))
            token = _token(rng, gh, gw, bg, int(rng.choice(inks)))
            key = (frame, token.shape, token.tobytes())
            if key not in used:
                used.add(key)
                break
        else:
            raise RuntimeError("bounded distinct-glyph budget exhausted")
        members = int(rng.integers(1, 3))
        if test and j == 0:
            members = int(rng.integers(3, 5))
        groups.append((frame, token, members))
    return groups


def _free_pair(rng, bg, low, high, inks, test, transpose):
    groups = _inventory(rng, bg, low, high, inks, test)
    # General free placement; all retries are in memory and have finite budgets.
    for _ in range(32):
        shape = (int(rng.integers(23, 31)), int(rng.integers(24, 31)))
        objects = []
        for frame, token, members in groups:
            h, w = token.shape[0] + 2, token.shape[1] + 2
            for _ in range(members):
                for _ in range(48):
                    r = int(rng.integers(1, shape[0] - h))
                    c = int(rng.integers(1, shape[1] - w))
                    candidate = _object(r, c, frame, token)
                    if all(_separated(candidate, o) for o in objects):
                        objects.append(candidate)
                        break
                else:
                    break
            else:
                continue
            break
        if len(objects) != sum(g[2] for g in groups):
            continue
        try:
            pair = _pair(shape, bg, low, high, objects, transpose)
        except ValueError:
            continue
        if _features(pair["input"])["coupled"]:
            return pair
    # Guaranteed feasible fallback: randomized column bands for assemblies.
    # Bands are a sampler fallback, not a solving convention.
    height, width = 30, 30
    widths = [token.shape[1] + 2 for _, token, _ in groups]
    extra = width - 2 - sum(widths) - (len(groups) - 1)
    assert extra >= 0
    gaps = rng.multinomial(extra, np.full(len(groups) + 1, 1 / (len(groups) + 1)))
    c, objects = 1 + int(gaps[0]), []
    for j, (frame, token, members) in enumerate(groups):
        h = token.shape[0] + 2
        available = height - 2 - members * h - (members - 1)
        row_gaps = rng.multinomial(available, np.full(members + 1, 1 / (members + 1)))
        r = 1 + int(row_gaps[0])
        for k in range(members):
            objects.append(_object(r, c, frame, token))
            r += h + 1 + int(row_gaps[k + 1])
        c += widths[j] + 1 + int(gaps[j + 1])
    return _pair((height, width), bg, low, high, objects, transpose)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError("counts must be integers in train=2..10, test=1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    low, high = palette[:2]
    background_pool = palette[2:4]
    inks = palette[4:int(rng.integers(6, 10))]
    fixed_axis = bool(rng.integers(2))
    mixed_axis = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(16):
                bg = int(rng.choice(background_pool))
                transpose = bool(rng.integers(2)) if mixed_axis else fixed_axis
                if split == "train" and index < 2:
                    pair = _lesson(rng, bg, low, high, inks, index, transpose)
                else:
                    pair = _free_pair(rng, bg, low, high, inks, split == "test", transpose)
                key = pair["input"].shape, pair["input"].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-input budget exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    puzzle["test"] = [puzzle["test"][int(i)] for i in rng.permutation(num_test)]
    return puzzle


def _features(value):
    a, transpose, bg, low, high, objects = _decode(value)
    settled = _settle(objects, a.shape[1], low, high)
    # _settle uses key[0] for the direction, so preserve the frame there.
    independent = [dict(o, key=(o["frame"], i)) for i, o in enumerate(objects)]
    independent = _settle(independent, a.shape[1], low, high)
    keys = list(dict.fromkeys(o["key"] for o in objects))
    positions = [o["c"] for o in settled]
    jumping = _edge_first(objects, a.shape[1], low, high)
    envelopes = []
    for key in keys:
        group = [o for o in objects if o["key"] == key]
        r, c = min(o["r"] for o in group), min(o["c"] for o in group)
        envelopes.append(dict(r=r, c=c, h=max(o["r"]+o["h"] for o in group)-r,
                              w=max(o["c"]+o["w"] for o in group)-c, frame=key[0]))
    def changed_by_alias(mode):
        aliased = []
        for o in objects:
            token = o["patch"][1:-1, 1:-1]
            identity = ((token != bg).tobytes() if mode == "shape"
                        else int(token[token != bg][0]))
            aliased.append(dict(o, key=(o["frame"], token.shape, identity)))
        try:
            other = _settle(aliased, a.shape[1], low, high)
        except ValueError:
            return True
        return positions != [o["c"] for o in other]
    return {"axis": int(transpose), "shape": tuple(np.asarray(value).shape),
            "tiles": len(objects), "groups": len(keys),
            "members": sorted(sum(o["key"] == key for o in objects) for key in keys),
            "footprints": sorted(set((o["h"], o["w"]) for o in objects)),
            "coupled": positions != [o["c"] for o in independent],
            "jumping_wrong": any(o["c"] != jumping[i] for i, o in enumerate(settled)),
            "interleaved_voids": any(a["frame"] == b["frame"] and _overlap(a, b)
                                     for i, a in enumerate(envelopes) for b in envelopes[i+1:]),
            "shape_only_wrong": changed_by_alias("shape"),
            "color_only_wrong": changed_by_alias("color"),
            "moved_frames": sorted({o["frame"] for o, p in zip(objects, settled)
                                    if o["c"] != p["c"]}),
            "offset_link": any(a["key"] == b["key"] and a["c"] != b["c"]
                               for i, a in enumerate(objects) for b in objects[i + 1:]),
            "stationary": sum(o["c"] == p["c"] for o, p in zip(objects, settled))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, rails, features = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = inp.shape, inp.tobytes()
            assert key not in seen, "duplicate input"
            seen.add(key)
            a, transpose, bg, low, high, objects = _decode(inp)
            rails.add((low, high))
            physical = _simulate(objects, a.shape[1], low, high)
            expected = _render(a.shape, bg, low, high, physical)
            actual = out.T if transpose else out
            assert np.array_equal(actual, expected), "rigid collision simulation disagrees"
            assert np.array_equal(solve(inp), out), "input-only solver disagrees"
            # Check full patches, within-assembly offsets, and row preservation.
            for old, new in zip(objects, physical):
                assert old["r"] == new["r"]
                r, c, h, w = (new[k] for k in ("r", "c", "h", "w"))
                assert np.array_equal(actual[r:r+h, c:c+w], old["patch"])
                for other, dest in zip(objects, physical):
                    if old["key"] == other["key"]:
                        assert new["c"] - old["c"] == dest["c"] - other["c"]
            features[split].append(_features(inp))
    assert len(rails) == 1, "the two rail/frame colors must stay consistent"
    # Teaching guarantees, including for the minimum 2/1 request.
    train = features["train"]
    assert sum(f["coupled"] and f["shape_only_wrong"] and f["color_only_wrong"]
               and f["jumping_wrong"] and f["offset_link"] for f in train) >= 2
    assert sum(f["interleaved_voids"] for f in train) >= 2
    low, high = next(iter(rails))
    assert {c for f in train for c in f["moved_frames"]} == {low, high}
    assert max(max(f["members"]) for f in train) == 2
    assert all(max(f["members"]) >= 3 for f in features["test"])


def self_test():
    """Stream the 42 host seeds; no corpus, logs, or traces are written."""
    # Hand-calculated nonlocal chain: B holds A in a different row, then A
    # holds C; a member of B occupies the open gap inside A's envelope.
    first = np.array([[3, 0], [0, 3]], np.int8)
    second = np.array([[3, 3], [3, 0]], np.int8)
    recolored = np.where(first == 3, 4, first).astype(np.int8)
    objects = [_object(1, 12, 1, first), _object(13, 10, 1, first),
               _object(7, 10, 1, second), _object(13, 2, 1, second),
               _object(1, 18, 1, recolored), _object(1, 2, 2, first)]
    hand = _render((23, 30), 0, 1, 2, objects)
    expected = _render((23, 30), 0, 1, 2,
                       [dict(o, c=c) for o, c in zip(objects, (7, 5, 9, 1, 11, 25))])
    assert np.array_equal(solve(hand), expected)
    # Same-row members preserve their gap rather than compacting internally.
    row_link = [_object(1, c, 1, first) for c in (8, 17)]
    assert np.array_equal(solve(_render((8, 26), 0, 1, 2, row_link)),
                          _render((8, 26), 0, 1, 2,
                                  [dict(o, c=c) for o, c in zip(row_link, (1, 10))]))
    # Opposite row orders need no arbitrary topological tie breaker.
    cycle = [_object(1, 3, 1, first), _object(7, 12, 1, first),
             _object(1, 12, 1, second), _object(7, 3, 1, second)]
    assert np.array_equal(solve(_render((13, 24), 0, 1, 2, cycle)),
                          _render((13, 24), 0, 1, 2, [dict(o, c=o["c"]-2) for o in cycle]))
    fingerprints, shapes, axes, footprints, counts, group_counts = set(), set(), set(), set(), set(), set()
    offsets = coupled_tests = mixed_sizes = stationary = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprint = _fingerprint(p)
        assert fingerprint == _fingerprint(generate_puzzle(seed))
        fingerprints.add(fingerprint)
        for split in ("train", "test"):
            for pair in p[split]:
                f = _features(pair["input"])
                shapes.add(f["shape"])
                axes.add(f["axis"])
                footprints.update(f["footprints"])
                counts.add(f["tiles"])
                group_counts.add(f["groups"])
                offsets += f["offset_link"]
                coupled_tests += split == "test" and f["coupled"]
                mixed_sizes += len(f["footprints"]) > 1
                stationary += f["stationary"] > 0
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and axes == {0, 1}
    assert len(footprints) == 4 and len(counts) >= 5 and len(group_counts) >= 3
    assert offsets >= 40 and coupled_tests >= 30 and mixed_sizes >= 40 and stationary > 0
    # Every legal count combination, plus strict invalid type/count handling.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(12345 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.0, 1), (True, 1), (4, 1.5), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    # Palette equivariance: arbitrary bijective recoloring preserves the rule.
    remap = np.array([7, 4, 0, 8, 1, 9, 3, 6, 2, 5], np.int8)
    for pair in p["train"] + p["test"]:
        inp, out = pair["input"], pair["output"]
        assert np.array_equal(solve(remap[inp]), remap[out])
        assert np.array_equal(solve(inp.T), out.T)
    for defect in ("singleton", "pixel", "duplicate"):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        if defect == "duplicate":
            damaged["test"][0] = damaged["train"][0]
        elif defect == "pixel":
            out = damaged["test"][0]["output"]
            out[0, 0] = (int(out[0, 0]) + 1) % 10
        else:
            pair = next(pair for pair in damaged["train"] if _features(pair["input"])["coupled"])
            a, transpose, bg, low, high, objects = _decode(pair["input"])
            separate = [dict(o, key=(o["frame"], i)) for i, o in enumerate(objects)]
            out = _render(a.shape, bg, low, high, _settle(separate, a.shape[1], low, high))
            pair["output"] = out.T.copy() if transpose else out
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("semantic corruption accepted: " + defect)
    return {"seeds": 42, "distinct_puzzles": len(fingerprints),
            "grid_shapes": len(shapes), "axes": sorted(axes),
            "footprints": sorted(footprints), "tile_counts": sorted(counts),
            "assembly_counts": sorted(group_counts), "offset_link_pairs": offsets,
            "coupled_tests": coupled_tests, "mixed_footprint_pairs": mixed_sizes,
            "pairs_with_stationary_tiles": stationary}
