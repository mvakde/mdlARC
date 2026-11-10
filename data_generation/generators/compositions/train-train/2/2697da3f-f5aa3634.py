"""Exact antipodal duplicates reveal an axle; complete their quarter-turn orbits.

Only the input is used by solve. All coordinates there are in the observed
canvas; doubled pivot coordinates allow both pixel and lattice-intersection
centers without floating-point rounding.
"""
from collections import defaultdict
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(a, bg):
    """Four-connected objects, with their exact tight colored matrices."""
    seen = np.zeros(a.shape, dtype=bool)
    objects = []
    for r, c in np.argwhere(a != bg):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        cells, stack = [], [(r, c)]
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if (0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]
                        and not seen[yy, xx] and a[yy, xx] != bg):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        y0 = min(y for y, _ in cells)
        x0 = min(x for _, x in cells)
        y1 = max(y for y, _ in cells) + 1
        x1 = max(x for _, x in cells) + 1
        crop = a[y0:y1, x0:x1].copy()
        objects.append({"cells": cells, "box": (y0, x0, y1-y0, x1-x0),
                        "crop": crop})
    return objects


def _background(a):
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    if not np.all(border == border[0]):
        raise ValueError("the input must have a uniform blank border")
    return int(border[0])


def _analyze(a):
    a = _grid(a)
    bg = _background(a)
    objects = _components(a, bg)
    groups = defaultdict(list)
    for obj in objects:
        if np.count_nonzero(obj["crop"] != bg) != len(obj["cells"]):
            raise ValueError("an object crop contains another object")
        groups[_key(obj["crop"])].append(obj)
    pairs = [v for v in groups.values() if len(v) > 1]
    if not pairs or any(len(v) != 2 for v in pairs):
        raise ValueError("need exact pairs, with no triple occurrences")
    centers = []
    for first, second in pairs:
        t = first["crop"]
        if not np.array_equal(t, t[::-1, ::-1]):
            raise ValueError("a paired tile must have half-turn symmetry")
        r, c, h, w = first["box"]
        s, d, _, _ = second["box"]
        centers.append((r+s+h-1, c+d+w-1))
    if len(set(centers)) != 1 or (centers[0][0] - centers[0][1]) % 2:
        raise ValueError("all pairs must reveal one lattice-compatible axle")
    return bg, objects, pairs, centers[0]


def _turn(y, x, center):
    """90 degrees counterclockwise, for integer pixel coordinates."""
    r2, c2 = center
    return (r2+c2)//2 - x, (c2-r2)//2 + y


def _complete(a, bg, pairs, center, *, turn_tiles=True):
    """Coordinate construction, also used for a translation-only ablation."""
    pixels = {}
    for group in pairs:
        obj = group[0]
        r, c, h, w = obj["box"]
        for y, x in obj["cells"]:
            yy, xx = y, x
            for _ in range(4):
                if not turn_tiles:
                    # Same motif orientation at rotated bounding-box centers.
                    # This intentionally wrong alternative uses doubled centers.
                    cy2, cx2 = 2*r+h-1, 2*c+w-1
                    for __ in range(_):
                        cy2, cx2 = center[0]+center[1]-cx2, center[1]-center[0]+cy2
                    dy2, dx2 = cy2-(2*r+h-1), cx2-(2*c+w-1)
                    # A nonintegral translation is itself a failed alternative.
                    if dy2 % 2 or dx2 % 2:
                        return None, None
                    pos = (y+dy2//2, x+dx2//2)
                else:
                    pos = (yy, xx)
                color = int(a[y, x])
                if pos in pixels and pixels[pos] != color:
                    if not turn_tiles:
                        return None, None
                    raise ValueError("completed orbits overlap with conflicting colors")
                pixels[pos] = color
                yy, xx = _turn(yy, xx, center)
    lo_r = min(y for y, _ in pixels)
    lo_c = min(x for _, x in pixels)
    hi_r = max(y for y, _ in pixels)
    hi_c = max(x for _, x in pixels)
    out = np.full((hi_r-lo_r+1, hi_c-lo_c+1), bg, dtype=np.int8)
    for (y, x), color in pixels.items():
        out[y-lo_r, x-lo_c] = color
    return out, (lo_r, lo_c)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the common axle from exact duplicate pairs and complete them."""
    a = _grid(input_grid)
    bg, _, pairs, center = _analyze(a)
    out, _ = _complete(a, bg, pairs, center)
    if out.shape[0] != out.shape[1] or max(out.shape) > 30:
        raise ValueError("the completed rotor must be a legal square")
    return out


def _apart(a, b, gap=1):
    r, c, h, w = a
    y, x, hh, ww = b
    return (r+h+gap <= y or y+hh+gap <= r
            or c+w+gap <= x or x+ww+gap <= c)


def _orbit_boxes(r, c, h, w, side):
    boxes = []
    for _ in range(4):
        boxes.append((r, c, h, w))
        r, c, h, w = side-c-w, r, w, h
    return boxes


def _tile(rng, shape, bg, palette, *, no_mirrors=False):
    """Rejection is bounded; no catalog of parent or previously solved tiles."""
    h, w = shape
    orbits = [(r, c) for r in range(h) for c in range(w)
              if (r, c) <= (h-1-r, w-1-c)]
    for _ in range(180):
        probability = float(rng.uniform(0.42, 0.94))
        t = np.full(shape, bg, dtype=np.int8)
        for r, c in orbits:
            if rng.random() < probability:
                t[r, c] = t[h-1-r, w-1-c] = int(rng.choice(palette))
        mask = t != bg
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        occupied = [(r, c) for r, c in orbits if mask[r, c]]
        if len(occupied) < len(palette):
            continue
        for color, index in zip(palette, rng.permutation(len(occupied))):
            r, c = occupied[int(index)]
            t[r, c] = t[h-1-r, w-1-c] = color
        if len(_components(t, bg)) != 1 or np.array_equal(t, np.rot90(t)):
            continue
        if no_mirrors and (np.array_equal(t, t[::-1]) or np.array_equal(t, t[:, ::-1])):
            continue
        return t
    raise RuntimeError("could not sample a directional half-turn tile")


def _variant(rng, t, bg, palette, used):
    """A nonmatching singleton with the same mask and half-turn symmetry."""
    h, w = t.shape
    orbits = [(r, c) for r, c in np.argwhere(t != bg)
              if (r, c) <= (h-1-r, w-1-c)]
    for _ in range(80):
        v = t.copy()
        r, c = orbits[int(rng.integers(len(orbits)))]
        options = [int(x) for x in palette if x != int(v[r, c])]
        if not options:
            options = [x for x in range(10) if x not in (bg, int(v[r, c]))]
        v[r, c] = v[h-1-r, w-1-c] = int(rng.choice(options))
        if _key(v) not in used:
            return v
    raise RuntimeError("could not make a distinct same-mask singleton")


def _place_singletons(rng, side, tiles, used_boxes, bg):
    placements = []
    boxes = list(used_boxes)
    for t in tiles:
        h, w = t.shape
        places = [(r, c) for r in range(side-h+1) for c in range(side-w+1)]
        for index in rng.permutation(len(places)):
            r, c = places[int(index)]
            box = (r, c, h, w)
            if all(_apart(box, old) for old in boxes):
                boxes.append(box)
                placements.append((t, box))
                break
        else:
            raise RuntimeError("singletons did not fit")
    return placements


def _make_pair(rng, count, bg, palette, *, evidence, side_low, side_high):
    side = int(rng.integers(side_low, side_high+1))
    full_boxes, source_boxes, placements, targets = [], [], [], []
    used = set()
    for j in range(count):
        mode = "oblique" if j == 0 and evidence == 1 else "cardinal" if j == 0 and evidence == 2 else "free"
        heights = [h for h in range(2, 6) if mode != "cardinal" or h % 2 == side % 2]
        h, w = int(rng.choice(heights)), int(rng.integers(2, 6))
        color_n = 2 if evidence == 1 and j == 0 else int(rng.choice([1, 2, 2, 3]))
        colors = tuple(int(x) for x in rng.choice(palette, color_n, replace=False))
        t = _tile(rng, (h, w), bg, colors, no_mirrors=(evidence == 1 and j == 0))
        if any(_key(np.rot90(t, k)) in used for k in range(4)):
            raise RuntimeError("two target types coincide under rotation")
        candidates = []
        for r in range(side-h+1):
            for c in range(side-w+1):
                dy2, dx2 = 2*r+h-side, 2*c+w-side
                if mode == "cardinal" and dy2 != 0:
                    continue
                if mode == "oblique" and (not dy2 or not dx2 or abs(dy2) == abs(dx2)):
                    continue
                orbit = _orbit_boxes(r, c, h, w, side)
                if not all(_apart(a, b) for i, a in enumerate(orbit) for b in orbit[i+1:]):
                    continue
                if not all(_apart(a, b) for a in orbit for b in full_boxes):
                    continue
                candidates.append(orbit)
        if not candidates:
            raise RuntimeError("rotational orbits did not fit")
        orbit = candidates[int(rng.integers(len(candidates)))]
        full_boxes.extend(orbit)
        for box in (orbit[0], orbit[2]):
            placements.append((t.copy(), box))
            source_boxes.append(box)
        targets.append(t)
        used.add(_key(t))
    noise = []
    if evidence == 1:
        # Four visually identical silhouettes, but only two full colored crops match.
        for _ in range(2):
            v = _variant(rng, targets[0], bg, palette, used)
            used.add(_key(v))
            noise.append(v)
    elif evidence == 2:
        # Rotational equivalence is insufficient for an exact translated match.
        v = np.rot90(targets[0]).copy()
        if _key(v) in used:
            raise RuntimeError("rotation decoy duplicates another tile")
        used.add(_key(v))
        noise.append(v)
    else:
        if rng.random() < 0.4:
            v = _variant(rng, targets[int(rng.integers(count))], bg, palette, used)
            used.add(_key(v))
            noise.append(v)
    for _ in range(int(rng.integers(1, 3))):
        t = _tile(rng, (int(rng.integers(2, 6)), int(rng.integers(2, 6))),
                  bg, tuple(int(x) for x in rng.choice(palette, int(rng.integers(1, 4)), replace=False)))
        if _key(t) in used:
            raise RuntimeError("singleton is an exact match")
        used.add(_key(t))
        noise.append(t)
    placements.extend(_place_singletons(rng, side, noise, source_boxes, bg))
    a = np.full((side, side), bg, dtype=np.int8)
    for t, (r, c, h, w) in placements:
        a[r:r+h, c:c+w] = t
    # Crop away the construction canvas; add independently chosen visible padding.
    cells = np.argwhere(a != bg)
    low, high = cells.min(axis=0), cells.max(axis=0)+1
    a = a[low[0]:high[0], low[1]:high[1]]
    pads = []
    for n in a.shape:
        total = int(rng.integers(2, min(30-n, 6)+1))
        first = int(rng.integers(1, total))
        pads.append((first, total-first))
    a = np.pad(a, pads, constant_values=bg)
    # Avoid unrelated accidental repeats introduced by decoys; no latent metadata.
    _, objects, found, center = _analyze(a)
    if len(found) != count:
        raise RuntimeError("unexpected duplicate group")
    out, origin = _complete(a, bg, found, center)
    if evidence == 1 and center == (a.shape[0]-1, a.shape[1]-1):
        raise RuntimeError("the first demonstration must distinguish the frame center")
    if evidence == 2:
        decoy = next(o for o in objects if _key(o["crop"]) == _key(noise[0]))
        if _preserved(a, out, origin, decoy["cells"]):
            raise RuntimeError("the rotated singleton must visibly disappear")
    return {"input": a, "output": out}


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    home = int(rng.integers(10))
    order = [int(x) for x in rng.permutation(10)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            evidence = i+1 if split == "train" and i < 2 else 0
            count = (i+1 if evidence else int(rng.integers(1, 3))) if split == "train" else (3 if i == 0 else int(rng.integers(1, 4)))
            bg = home if i == 0 or rng.random() < 0.65 else int(rng.integers(10))
            palette = tuple(x for x in order if x != bg)[:int(rng.integers(3, 6))]
            for _ in range(100):
                try:
                    pair = _make_pair(rng, count, bg, palette, evidence=evidence,
                                      side_low=16 if count < 3 else 23,
                                      side_high=26 if count < 3 else 28)
                    key = _key(pair["input"])
                    if key in seen:
                        continue
                    if evidence == 2 and pair["output"].shape[0] % 2 == puzzle["train"][0]["output"].shape[0] % 2:
                        continue
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError("could not construct a distinct rotor scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def _preserved(a, out, origin, cells):
    return all(0 <= y-origin[0] < out.shape[0] and 0 <= x-origin[1] < out.shape[1]
               and out[y-origin[0], x-origin[1]] == a[y, x] for y, x in cells)


def _pair_info(pair):
    a, out = _grid(pair["input"]), _grid(pair["output"])
    bg, objects, pairs, center = _analyze(a)
    expected, origin = _complete(a, bg, pairs, center)
    assert np.array_equal(expected, out)
    assert out.shape[0] == out.shape[1] and np.array_equal(out, np.rot90(out))
    # The crop is tight, is centered on the inferred pivot, and has a blank axle.
    assert (out[0] != bg).any() and (out[-1] != bg).any()
    assert (out[:, 0] != bg).any() and (out[:, -1] != bg).any()
    assert center == (2*origin[0]+out.shape[0]-1, 2*origin[1]+out.shape[1]-1)
    middle = [(out.shape[0]-1)//2, out.shape[0]//2]
    assert np.all(out[np.ix_(middle, middle)] == bg)
    boxes = [obj["box"] for obj in objects]
    assert all(_apart(a, b) for i, a in enumerate(boxes) for b in boxes[i+1:])
    # Exact count growth verifies that no completed orbit overlaps another one.
    kept_pixels = sum(len(g[0]["cells"]) for g in pairs)
    assert np.count_nonzero(out != bg) == 4*kept_pixels
    out_components = _components(out, bg)
    assert len(out_components) == 4*len(pairs)
    for group in pairs:
        t = group[0]["crop"]
        assert 2 <= min(t.shape) <= max(t.shape) <= 5
        assert not np.array_equal(t, np.rot90(t))
        assert np.array_equal(t, np.rot90(t, 2))
        for obj in group:
            for y, x in obj["cells"]:
                assert out[y-origin[0], x-origin[1]] == a[y, x]
    paired_ids = {id(o) for group in pairs for o in group}
    singles = [o for o in objects if id(o) not in paired_ids]
    assert singles
    assert all(np.array_equal(o["crop"], np.rot90(o["crop"], 2)) for o in objects)
    out_boxes = [o["box"] for o in out_components]
    assert all(_apart(a, b) for i, a in enumerate(out_boxes) for b in out_boxes[i+1:])
    assert any(not _preserved(a, out, origin, o["cells"]) for o in singles)
    mask_decoys = sum(any(s["crop"].shape == g[0]["crop"].shape and
                          np.array_equal(s["crop"] != bg, g[0]["crop"] != bg)
                          for g in pairs) for s in singles)
    rotation_decoy = any(not _preserved(a, out, origin, s["cells"]) and
                         any(_key(s["crop"]) == _key(np.rot90(g[0]["crop"]))
                             for g in pairs) for s in singles)
    oblique = False
    for g in pairs:
        r, c, h, w = g[0]["box"]
        oblique |= 2*r+h-1 != center[0] and 2*c+w-1 != center[1]
    wrong, _ = _complete(a, bg, pairs, center, turn_tiles=False)
    assert wrong is None or not np.array_equal(wrong, out)
    # Omitting B: rotate all objects about even the *correct* axle. Exact
    # matching remains necessary despite granting this shortcut the true pivot.
    try:
        unfiltered, _ = _complete(a, bg, [[o] for o in objects], center)
    except ValueError:
        pass  # singleton orbits conflict, so this rule cannot give the answer
    else:
        assert not np.array_equal(unfiltered, out)
    return {"groups": len(pairs), "parity": center[0] % 2, "shapes": tuple(g[0]["crop"].shape for g in pairs),
            "size": out.shape[0], "background": bg, "mask_decoys": mask_decoys,
            "rotation_decoy": rotation_decoy, "oblique": bool(oblique),
            "holes": any(np.any(g[0]["crop"] == bg) for g in pairs),
            "singleton_count": len(singles), "input_shape": a.shape,
            "frame_center": center == (a.shape[0]-1, a.shape[1]-1)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_infos, test_infos = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a = _grid(pair["input"])
            key = _key(a)
            assert key not in seen
            seen.add(key)
            info = _pair_info(pair)
            assert 1 <= info["groups"] <= 3
            (train_infos if split == "train" else test_infos).append(info)
    assert {x["groups"] for x in train_infos} >= {1, 2}
    assert max(x["groups"] for x in train_infos) == 2
    assert test_infos[0]["groups"] == 3
    assert len({x["size"] for x in train_infos}) >= 2
    assert train_infos[0]["parity"] != train_infos[1]["parity"]
    assert not train_infos[0]["frame_center"]
    assert train_infos[0]["mask_decoys"] >= 2 and train_infos[0]["oblique"]
    assert train_infos[1]["rotation_decoy"]


def _fingerprint(p):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in p[split])


def self_test():
    """Bounded 42-seed semantic, diversity, count and input-only checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    infos = []
    masks, palettes = set(), set()
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                infos.append(_pair_info(pair))
                bg, _, groups, _ = _analyze(pair["input"])
                for g in groups:
                    t = g[0]["crop"]
                    masks.add(_key((t != bg).astype(np.int8)))
                    palettes.add(tuple(int(x) for x in np.unique(t)))
    assert {x["groups"] for x in infos} == {1, 2, 3}
    assert {x["parity"] for x in infos} == {0, 1}
    assert len({x["size"] for x in infos}) >= 10
    assert len({s for x in infos for s in x["shapes"]}) >= 12
    assert len({x["input_shape"] for x in infos}) >= 30
    assert len({x["background"] for x in infos}) == 10
    assert len(masks) >= 35 and len(palettes) >= 40
    assert sum(x["holes"] for x in infos) >= 30
    assert sum(x["oblique"] for x in infos) >= 60
    # Input-only metamorphic checks: translation, palette permutation and
    # transposition preserve the rule, without supplying generation metadata.
    example = puzzles[0]["train"][0]
    a, out = example["input"], example["output"]
    bg = _background(a)
    assert np.array_equal(solve(np.pad(a, 1, constant_values=bg)), out)
    permutation = np.array([7, 3, 9, 0, 6, 2, 8, 4, 1, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[a]), permutation[out])
    assert np.array_equal(solve(a.T), out.T)
    # The same tile at different paired offsets must give different answers.
    t = _tile(np.random.default_rng(123), (3, 4), 0, (2, 7), no_mirrors=True)
    displaced = []
    for top in (2, 1):
        a = np.zeros((24, 24), dtype=np.int8)
        a[top:top+3, 5:9] = t
        a[21-top:24-top, 15:19] = t
        a[11:13, 11:13] = 4  # singleton, not part of either rotor
        displaced.append(solve(a))
    assert displaced[0].shape != displaced[1].shape
    assert _fingerprint(generate_puzzle(612994233)) == _fingerprint(generate_puzzle(612994233))
    # All count combinations are small enough to exercise, rather than sampling boundaries.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(2718+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
