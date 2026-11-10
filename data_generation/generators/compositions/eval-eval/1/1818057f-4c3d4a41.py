"""Stencil lift: plus-union linkage, silhouette contact, and roof clipping.

Only Python's standard library and NumPy are used. The input-only solver
reads geometry and lifts patches stepwise. Generation labels and validation
use a direct destination formula with a separate local-pattern scanner.
"""
from collections import deque
import json

import numpy as np


HOST_SEEDS = (0, 143124082, 852649907, 270621704, 744519744, 538330068,
              886354763, 422976113, 89641616, 443104833, 391721655, 30391212,
              263903524, 212125167, 360887758, 22738949, 888524937, 342016805,
              590033570, 611474, 184145134, 926588306, 205414547, 765592218,
              503803221, 268867558, 57929863, 713441184, 583750970, 565734675,
              54218957, 512534611, 74708734, 379661728, 774412168, 630877860,
              816852735, 985148205, 794450024, 719821667, 902262840, 889602554)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _active(mask):
    """Vectorized simultaneous detector used by solve."""
    centers = np.zeros_like(mask, dtype=bool)
    centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] &
                          mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
    union = centers.copy()
    union[1:] |= centers[:-1]
    union[:-1] |= centers[1:]
    union[:, 1:] |= centers[:, :-1]
    union[:, :-1] |= centers[:, 1:]
    return union


def _scan(mask, destructive=False):
    """Direct coordinate scanner for construction and diagnostic alternatives."""
    work = mask.copy()
    centers, union = [], np.zeros_like(mask, dtype=bool)
    for r in range(1, mask.shape[0] - 1):
        for c in range(1, mask.shape[1] - 1):
            points = [(r, c)] + [(r + dr, c + dc) for dr, dc in _DIRS]
            if all(work[p] for p in points):
                centers.append((r, c))
                for p in points:
                    union[p] = True
                    if destructive:
                        work[p] = False
    return centers, union


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, cells = deque([start]), []
        while queue:
            r, c = queue.popleft()
            cells.append((int(r), int(c)))
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        result.append(cells)
    return result


def _read(input_grid):
    a0 = _grid(input_grid)
    values, counts = np.unique(a0, return_counts=True)
    assert len(values) == 4 and counts.max() > a0.size / 2, "four visible roles required"
    bg = int(values[np.argmax(counts)])
    candidates = []
    for rotation in range(4):
        a = np.rot90(a0, rotation).copy()
        for frame in map(int, values):
            if frame == bg:
                continue
            rr, cc = np.nonzero(a == frame)
            t, b, l, r = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
            if b - t < 4 or r - l < 5:
                continue
            expected = np.zeros_like(a, dtype=bool)
            expected[t, l:r + 1] = True
            expected[t:b + 1, l] = expected[t:b + 1, r] = True
            if not np.array_equal(expected, a == frame):
                continue
            for key in map(int, values):
                if key in (bg, frame):
                    continue
                ink = next(int(x) for x in values if x not in (bg, frame, key))
                kr, kc = np.nonzero(a == key)
                base = int(kr.max())
                if base <= b or kc.min() != l + 1 or kc.max() != r - 1:
                    continue
                if not np.all(a[base, l + 1:r] == key):
                    continue
                heights = []
                shape = np.zeros_like(a, dtype=bool)
                for c in range(l + 1, r):
                    q = base - 1
                    while q >= 0 and a[q, c] == key:
                        q -= 1
                    h = base - q - 1
                    heights.append(h)
                    shape[base - h:base + 1, c] = True
                if (sum(h > 0 for h in heights) < 2 or
                        max(heights) > b - t - 2 or base - max(heights) <= b or
                        not np.array_equal(shape, a == key)):
                    continue
                inside = np.zeros_like(a, dtype=bool)
                inside[t + 1:b, l + 1:r] = True
                if np.any((a == ink) & ~inside):
                    continue
                candidates.append(dict(a=a, background=bg, frame=frame, key=key,
                                       ink=ink, top=t, bottom=b, left=l, right=r,
                                       source_base=base, heights=tuple(heights),
                                       rotation=rotation))
    assert len(candidates) == 1, "scene must have one frame/comb interpretation"
    return candidates[0]


def _insert(a, f, union):
    out = a.copy()
    out[a == f["key"]] = f["background"]
    out[union] = f["background"]
    for c, h in enumerate(f["heights"], f["left"] + 1):
        out[f["bottom"] - h:f["bottom"] + 1, c] = f["key"]
    return out


def _displacement(cells, f):
    return max(0, max(r - f["bottom"] + f["heights"][c - f["left"] - 1] + 1
                      for r, c in cells))


def _formula(f, mode="normal"):
    """Direct label construction, plus executable shortcut counterexamples."""
    a = f["a"]
    centers, union = _scan(a == f["ink"])
    if mode == "all_ink":
        union = a == f["ink"]
    if mode == "greedy":
        _, union = _scan(a == f["ink"], destructive=True)
    out = _insert(a, f, union)
    if mode == "individual":
        patches = [[(r, c)] + [(r + dr, c + dc) for dr, dc in _DIRS]
                   for r, c in centers]
    elif mode == "raw_linkage":
        patches = [[p for p in component if union[p]]
                   for component in _components(a == f["ink"])]
        patches = [p for p in patches if p]
    else:
        patches = _components(union)
    for cells in patches:
        d = 0 if mode == "no_lift" else _displacement(cells, f)
        if mode == "bbox":
            lo, hi = min(c for r, c in cells), max(c for r, c in cells)
            d = max(0, max(r for r, c in cells) - f["bottom"] + 1 +
                    max(f["heights"][lo - f["left"] - 1:hi - f["left"]]))
        for r, c in cells:
            if r - d > f["top"]:
                out[r - d, c] = f["key"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the input; no seed or examples are read."""
    f = _read(input_grid)
    a = f["a"]
    union = _active(a == f["ink"])
    out = _insert(a, f, union)
    for cells in _components(union):
        distance = 0
        # Test actual contact with the translated key, one lattice step at a time.
        while any(r - distance >= f["bottom"] - f["heights"][c - f["left"] - 1]
                  for r, c in cells):
            distance += 1
        for r, c in cells:
            if r - distance > f["top"]:
                out[r - distance, c] = f["key"]
    return np.rot90(out, -f["rotation"]).copy()


def _facts(f):
    a = f["a"]
    mask = a == f["ink"]
    centers, union = _scan(mask)
    components = _components(union)
    shifts = [_displacement(p, f) for p in components]
    lost = sum(r - d <= f["top"] for p, d in zip(components, shifts) for r, c in p)
    touching = any(0 <= r + dr < a.shape[0] and 0 <= c + dc < a.shape[1] and
                   mask[r + dr, c + dc] and not union[r + dr, c + dc]
                   for r, c in zip(*np.nonzero(union)) for dr, dc in _DIRS)
    competing = False
    for p, d in zip(components, shifts):
        bottoms = {}
        for r, c in p:
            bottoms[c] = max(bottoms.get(c, -1), r)
        requests = [r - f["bottom"] + f["heights"][c - f["left"] - 1] + 1
                    for c, r in bottoms.items()]
        competing |= d > 0 and len(set(max(0, v) for v in requests)) > 1
    normal = _formula(f)
    return dict(centers=len(centers), components=len(components), shifts=shifts,
                lost=int(lost), attached=bool(touching), competing=bool(competing),
                linked=any(len(p) > 5 for p in components),
                a_witness=not np.array_equal(normal, _formula(f, "all_ink")),
                b_witness=not np.array_equal(normal, _formula(f, "no_lift")),
                independent_witness=not np.array_equal(normal, _formula(f, "individual")),
                raw_linkage_witness=not np.array_equal(normal, _formula(f, "raw_linkage")),
                footprint_witness=not np.array_equal(normal, _formula(f, "bbox")),
                simultaneous_witness=not np.array_equal(normal, _formula(f, "greedy")),
                depth=f["bottom"] - f["top"] - 1,
                width=f["right"] - f["left"] - 1,
                patch_width=max((max(c for r, c in p) - min(c for r, c in p) + 1
                                 for p in components), default=0),
                max_tooth=max(f["heights"]), profile=f["heights"],
                palette=(f["background"], f["frame"], f["key"], f["ink"]))


def _stamp(mask, r, c, kind="plus"):
    offsets = {"plus": [(0, 0), *_DIRS],
               "T": [(0, -1), (0, 0), (0, 1), (1, 0)],
               "X": [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)],
               "ring": [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)
                        if i or j]}[kind]
    for dr, dc in offsets:
        if 0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1]:
            mask[r + dr, c + dc] = True


def _ink_scene(rng, depth, width, kind):
    m = np.zeros((depth, width), dtype=bool)
    anchors = {}
    if kind == "separated":
        c1 = int(rng.integers(1, max(2, width // 2 - 1)))
        c2 = int(rng.integers(width // 2 + 2, width - 1))
        r1, r2 = depth - 2, int(rng.integers(1, max(2, depth - 5)))
        _stamp(m, r1, c1)
        _stamp(m, r2, c2)
        # The raw ink is connected, but the active union has separate pieces.
        m[r1, c1 + 2:c2 + 1] = True
        m[r2 + 2:r1 + 1, c2] = True
        if rng.random() < .55:
            _stamp(m, 1, width // 2, "T" if rng.random() < .5 else "X")
        anchors = dict(moving=c1, stationary=(r2, c2))
    elif kind == "linked":
        c = int(rng.integers(2, width - 4))
        rows = sorted(set(range(1, depth - 1, int(rng.integers(1, 3)))) | {depth - 2})
        for r in rows:
            _stamp(m, r, c)
        _stamp(m, int(rng.choice(rows)), c + 2)
        m[1, c - 2] = True
        if width >= 13 and rng.random() < .6:
            _stamp(m, int(rng.integers(1, depth - 1)), width - 2)
        anchors = dict(spine=c)
    else:
        if kind == "network":
            # A guaranteed wide bent linkage, built from shared-arm pluses.
            r = 2
            for j, c in enumerate(range(1, width - 1, 2)):
                _stamp(m, r, c)
                if j % 2 == 0:
                    nr = min(depth - 2, r + 2) if r < depth // 2 else max(1, r - 2)
                    _stamp(m, nr, c)
                    r = nr
        count = int(rng.integers(2, 8 if kind == "network" else 6))
        for _ in range(count):
            r, c = int(rng.integers(1, depth - 1)), int(rng.integers(1, width - 1))
            layout = int(rng.integers(5))
            if layout == 0:
                m[max(0, r - 1):min(depth, r + 2),
                  max(0, c - 1):min(width, c + int(rng.integers(2, 5)))] = True
            else:
                for j in range(int(rng.integers(1, 6)) if kind == "network" else 1):
                    _stamp(m, r, c)
                    dr, dc = _DIRS[int(rng.integers(4))]
                    r = int(np.clip(r + 2 * dr, 1, depth - 2))
                    c = int(np.clip(c + 2 * dc, 1, width - 2))
        if rng.random() < .8:
            _stamp(m, int(rng.integers(1, depth - 1)), int(rng.integers(1, width - 1)),
                   str(rng.choice(("T", "X", "ring"))))
        for _ in range(int(rng.integers(0, width + 1))):
            m[int(rng.integers(depth)), int(rng.integers(width))] = True
    return m, anchors


def _profile(rng, width, cap):
    heights = np.zeros(width, dtype=int)
    spacing = int(rng.integers(1, 5))
    columns = list(range(int(rng.integers(min(spacing, width))), width, spacing))
    if len(columns) < 2:
        columns = [0, width - 1]
    mode = int(rng.integers(5))
    h = rng.integers(1, cap + 1, len(columns))
    if mode == 0:
        h[:] = int(rng.integers(1, cap + 1))
    elif mode in (1, 2):
        h.sort()
        if mode == 2:
            h = h[::-1]
    elif mode == 3:
        h = rng.choice([1, cap], len(columns))
    heights[columns] = h
    return heights


def _new_pair(rng, kind, depth_band, width_band, tooth_band, palette, orientation,
              require_motion=False):
    for _ in range(100):
        d = int(rng.integers(depth_band[0], depth_band[1] + 1))
        w = int(rng.integers(width_band[0], width_band[1] + 1))
        mask, anchors = _ink_scene(rng, d, w, kind)
        top = int(rng.integers(0, min(2, 24 - d) + 1))
        left = int(rng.integers(0, min(2, 28 - w) + 1))
        bottom, right = top + d + 1, left + w + 1
        cap = min(tooth_band[1], d - 1, 27 - bottom)
        if cap < max(3, tooth_band[0]):
            continue
        heights = _profile(rng, w, cap)
        if kind == "separated":
            heights[anchors["moving"]] = int(rng.integers(3, cap + 1))
            sr, sc = anchors["stationary"]
            heights[sc - 1:sc + 2] = np.minimum(heights[sc - 1:sc + 2], d - sr - 2)
        elif kind == "linked":
            c = anchors["spine"]
            heights[c], heights[c - 1] = 1, int(rng.integers(3, cap + 1))
        if max(heights) < tooth_band[0]:
            heights[int(rng.integers(w))] = tooth_band[0]
        if np.count_nonzero(heights) < 2:
            heights[(int(np.argmax(heights)) + 2) % w] = 1
        gap = int(rng.integers(1, 3))
        base = bottom + int(max(heights)) + gap
        bm = int(rng.integers(0, min(2, 29 - base) + 1))
        rm = int(rng.integers(0, min(2, 29 - right) + 1))
        bg, frame, key, ink = palette
        a = np.full((base + bm + 1, right + rm + 1), bg, dtype=np.int8)
        a[top, left:right + 1] = frame
        a[top:bottom + 1, left] = a[top:bottom + 1, right] = frame
        field = a[top + 1:bottom, left + 1:right]
        field[mask] = ink
        for c, h in enumerate(heights, left + 1):
            a[base - h:base + 1, c] = key
        if np.count_nonzero(a == bg) <= a.size / 2:
            continue
        f = dict(a=a, background=bg, frame=frame, key=key, ink=ink,
                 top=top, bottom=bottom, left=left, right=right,
                 heights=tuple(map(int, heights)), source_base=base)
        facts = _facts(f)
        if kind == "separated" and not (0 in facts["shifts"] and max(facts["shifts"]) > 0
                                         and facts["attached"] and facts["a_witness"]
                                         and facts["raw_linkage_witness"]):
            continue
        if kind == "linked" and not (facts["lost"] and facts["linked"] and
                                      facts["competing"] and facts["independent_witness"] and
                                      facts["simultaneous_witness"] and facts["footprint_witness"]):
            continue
        if not facts["centers"]:
            continue
        if require_motion and not facts["b_witness"]:
            continue
        out = _formula(f)
        if orientation >= 4:
            a, out = np.fliplr(a), np.fliplr(out)
        return {"input": np.rot90(a, orientation % 4).copy(),
                "output": np.rot90(out, orientation % 4).copy()}
    raise RuntimeError("bounded scene construction failed")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    transfer = str(rng.choice(("depth", "width", "teeth", "network")))
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    change_colors = rng.random() < .7
    orientation = int(rng.integers(8))
    mixed_orientation = rng.random() < .6
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind = ("separated", "linked")[i] if split == "train" and i < 2 else "mixed"
            depth_band, width_band, tooth_band = (9, 13), (9, 18), (1, 7)
            if transfer == "depth":
                depth_band = (8, 11) if split == "train" else (15, 18)
            elif transfer == "width":
                width_band = (9, 13) if split == "train" else (19, 26)
            elif transfer == "teeth":
                depth_band = (10, 14)
                tooth_band = (1, 4) if split == "train" else (6, 9)
            elif transfer == "network":
                if split == "test":
                    kind = "network"
                    width_band = (16, 24)
                elif i >= 2:
                    kind = str(rng.choice(("separated", "linked")))
            colors = tuple(map(int, rng.choice(10, 4, replace=False))) if change_colors else palette
            direction = int(rng.integers(8)) if mixed_orientation else orientation
            for _ in range(60):
                pair = _new_pair(rng, kind, depth_band, width_band, tooth_band, colors, direction,
                                 require_motion=(split == "test"))
                k = _key(pair["input"])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded input uniqueness construction failed")
        puzzle[split] = [puzzle[split][int(j)] for j in rng.permutation(count)]
    # If a test needs simultaneous recognition, the linked teaching witness
    # already demonstrates it. No test-dependent hidden output convention.
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            f = _read(a)
            canonical_out = np.rot90(out, f["rotation"])
            expected = _formula(f)
            assert np.array_equal(canonical_out, expected), "incorrect stencil lift"
            assert np.array_equal(solve(a), out), "input-only stepwise solver disagrees"
            assert np.all(canonical_out[f["a"] == f["frame"]] == f["frame"])
            assert np.all(canonical_out[f["a"] == f["key"]] == f["background"])
            union = _active(f["a"] == f["ink"])
            landed = np.zeros_like(union)
            for cells in _components(union):
                d = _displacement(cells, f)
                survivors = [(r - d, c) for r, c in cells if r - d > f["top"]]
                assert survivors, "every activated patch has surviving ink"
                if d > 0:
                    assert any(r == f["bottom"] - f["heights"][c - f["left"] - 1] - 1
                               for r, c in survivors), "moving patch must contact a tooth"
                for p in survivors:
                    landed[p] = True
            assert np.all(canonical_out[landed] == f["key"])
            # Preserve inactive ink except where print or inserted key covers it.
            unchanged = (f["a"] == f["ink"]) & ~union & ~landed
            key_destination = np.zeros_like(union)
            for c, h in enumerate(f["heights"], f["left"] + 1):
                key_destination[f["bottom"] - h:f["bottom"] + 1, c] = True
            assert np.all(canonical_out[unchanged & ~key_destination] == f["ink"])
            assert np.all(canonical_out[key_destination] == f["key"])
            facts[split].append(_facts(f))
    train = facts["train"]
    for name in ("a_witness", "b_witness", "attached", "linked", "competing",
                 "independent_witness", "raw_linkage_witness", "footprint_witness"):
        assert any(f[name] for f in train), f"missing training evidence: {name}"
    assert any(f["lost"] > 0 for f in train), "training must demonstrate clipping"
    assert any(0 in f["shifts"] and max(f["shifts"]) > 0 for f in train)
    if any(f["simultaneous_witness"] for f in facts["test"]):
        assert any(f["simultaneous_witness"] for f in train)
    return True


def _fingerprint(puzzle):
    return tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s]
                 for k in ("input", "output"))


def _contact_regressions():
    def literal(mask, heights):
        d, w = mask.shape
        bottom = d + 1
        base = bottom + max(heights) + 2
        a = np.zeros((base + 1, w + 2), dtype=int)
        a[0, :] = 5
        a[:bottom + 1, 0] = a[:bottom + 1, -1] = 5
        a[1:bottom, 1:-1][mask] = 8
        for c, h in enumerate(heights, 1):
            a[base - h:base + 1, c] = 2
        return a

    mask = np.zeros((11, 12), dtype=bool)
    for r in (1, 3, 5, 7, 9):
        _stamp(mask, r, 3)
    h = [0] * 12
    h[2], h[3], h[10] = 4, 1, 8
    a = literal(mask, h)
    f = _read(a)
    # Side-arm bottom is one cell above the spine bottom: shift 3, not 4.
    assert _facts(f)["shifts"] == [3] and _facts(f)["lost"] == 5
    predicted = solve(a)
    assert np.array_equal(predicted, _formula(f))
    h[10] = 6  # Changing a taller tooth outside the footprint has no effect.
    assert np.array_equal(solve(literal(mask, h))[:12, 1:7], predicted[:12, 1:7])
    mask[5, 2] = False  # One missing arm breaks linkage between remote cells.
    broken = _facts(_read(literal(mask, h)))
    assert broken["components"] == 2 and sorted(broken["shifts"]) == [0, 3]
    mask = np.zeros((8, 9), dtype=bool)
    _stamp(mask, 2, 3, "T")
    a = literal(mask, [1, 0, 2, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(solve(a) == 8, a == 8), "no-match ink stays put"


def self_test():
    """Bounded 42-seed semantic/diversity audit; returns compact measurements."""
    _contact_regressions()
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(HOST_SEEDS[0]))
    records, shapes, structures, orientations = [], set(), set(), set()
    transfers = {name: 0 for name in ("depth", "width", "max_tooth", "patch_width")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        groups = {s: [_facts(_read(p["input"])) for p in puzzle[s]] for s in puzzle}
        visible_shifts = [name for name in transfers
                          if min(f[name] for f in groups["test"]) >
                          max(f[name] for f in groups["train"])]
        assert visible_shifts, "test must extend a measured structural axis"
        for name in visible_shifts:
            transfers[name] += 1
        assert all(f["b_witness"] for f in groups["test"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                f = _read(a)
                records.append(_facts(f))
                shapes.add(a.shape)
                roles = {}
                structures.add((a.shape, tuple(roles.setdefault(int(v), len(roles)) for v in a.flat)))
                orientations.add(f["rotation"])
    assert len(shapes) >= 50 and len(structures) >= 150
    assert len({f["profile"] for f in records}) >= 100
    assert len({f["centers"] for f in records}) >= 15
    assert len({f["components"] for f in records}) >= 4
    assert len({f["depth"] for f in records}) >= 8
    assert len({f["width"] for f in records}) >= 12
    assert len({f["palette"] for f in records}) >= 60
    assert orientations == {0, 1, 2, 3}
    assert {v for f in records for v in f["palette"]} == set(range(10))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(701 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(7, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example count accepted")
    # Exhaustive local neighborhoods exercise diagonals, arms and missing centers.
    for bits in range(512):
        m = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        assert np.array_equal(_active(m), _scan(m)[1])
    block = np.ones((3, 4), dtype=bool)
    assert _active(block).sum() == 8 and _scan(block, True)[1].sum() == 5
    edge = np.zeros((5, 5), dtype=bool)
    edge[0, 1:4] = edge[1, 2] = edge[-1, 2] = True
    assert not _active(edge).any(), "no wraparound"
    # Pixel-derived transform is equivariant to all scene symmetries and palettes.
    for pair in puzzles[0]["train"]:
        for code in range(8):
            for mapping in (np.arange(10), (np.arange(10) * 3 + 1) % 10):
                a, b = mapping[pair["input"]], mapping[pair["output"]]
                if code >= 4:
                    a, b = np.fliplr(a), np.fliplr(b)
                assert np.array_equal(solve(np.rot90(a, code % 4)), np.rot90(b, code % 4))
    # Detect a label error without accepting the generator's own output on trust.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted damaged output")
    return dict(seeds=42, pairs=len(records), shapes=len(shapes),
                normalized_structures=len(structures), profiles=len({f["profile"] for f in records}),
                center_counts=sorted({f["centers"] for f in records}),
                component_counts=sorted({f["components"] for f in records}),
                palettes=len({f["palette"] for f in records}),
                visible_transfers=transfers,
                independent_review=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
