"""Blank-window unions transported by a complete-row, bottom-anchored shear.

Only NumPy and the standard library are used. Generation and tests write no files.
Color roles are fixed: neutral/padding=0, highlight=1, texture in 2..9.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return a


def _window_union(a):
    """Detect on the original array, simultaneously, including overlaps."""
    h, w = a.shape
    mask = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return [], mask
    zeros = a == 0
    counts = np.zeros((h - 2, w - 2), dtype=np.int8)
    for dr in range(3):
        for dc in range(3):
            counts += zeros[dr:dr + h - 2, dc:dc + w - 2]
    starts = [tuple(map(int, p)) for p in np.argwhere(counts == 9)]
    for r, c in starts:
        mask[r:r + 3, c:c + 3] = True
    return starts, mask


def _shear(a):
    h, w = a.shape
    out = np.zeros((h, h + w - 1), dtype=np.int8)
    for r in range(h):
        out[r, h - 1 - r:h - 1 - r + w] = a[r]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver; supports all specified shapes, including no matches."""
    a = _grid(input_grid)
    if sum(a.shape) - 1 > 30:
        raise ValueError("sheared output would exceed 30 columns")
    colors = set(map(int, np.unique(a)))
    if 1 in colors or len(colors - {0}) > 1:
        raise ValueError("expected neutral 0 and at most one texture color 2..9")
    _, mask = _window_union(a)
    marked = a.copy()
    marked[mask] = 1
    return _shear(marked)


def _orient(rng, patch):
    patch = np.rot90(patch, int(rng.integers(4)))
    if int(rng.integers(2)):
        patch = patch[:, ::-1]
    return patch.copy()


def _motif(rng, kind):
    """True cells are blank; false cells remain texture, including holes."""
    if kind == "fringe":
        p = np.zeros((3, 4), dtype=bool)
        p[:, :3] = True
        p[1, 3] = True
    elif kind == "diagonal":
        p = np.zeros((4, 4), dtype=bool)
        p[:3, :3] = True
        p[1:, 1:] = True
    elif kind == "stair":
        n = int(rng.integers(2, 5))
        p = np.zeros((n + 2, n + 2), dtype=bool)
        for k in range(n):
            p[k:k + 3, k:k + 3] = True
    elif kind == "elbow":
        p = np.zeros((5, 5), dtype=bool)
        p[:3, :] = True
        p[:, :3] = True
    elif kind == "rectangle":
        p = np.ones((int(rng.integers(3, 6)), int(rng.integers(3, 8))), dtype=bool)
    elif kind == "near":
        p = np.ones((3, 3), dtype=bool)
        p[int(rng.integers(3)), int(rng.integers(3))] = False
    elif kind == "strip":
        p = np.ones((2, int(rng.integers(6, 9))), dtype=bool)
    elif kind == "square":
        p = np.ones((3, 3), dtype=bool)
    else:
        raise ValueError("unknown motif")
    return _orient(rng, p)


def _pack(rng, shape, patches):
    """Randomly place motifs with one texture cell of separation; edges allowed."""
    h, w = shape
    blocked = np.zeros(shape, dtype=bool)
    blanks = np.zeros(shape, dtype=bool)
    for patch in sorted(patches, key=lambda p: p.size, reverse=True):
        ph, pw = patch.shape
        options = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)
                   if not np.any(blocked[r:r + ph, c:c + pw])]
        if not options:
            return None
        edge_options = [(r, c) for r, c in options
                        if r == 0 or c == 0 or r + ph == h or c + pw == w]
        pool = edge_options if edge_options and rng.random() < 0.40 else options
        r, c = pool[int(rng.integers(len(pool)))]
        blanks[r:r + ph, c:c + pw] |= patch
        blocked[max(0, r - 1):min(h, r + ph + 1),
                max(0, c - 1):min(w, c + pw + 1)] = True
    return blanks, blocked


def _texture(rng, shape, texture):
    h, w = shape
    rr, cc = np.indices(shape)
    style = int(rng.integers(4))
    probability = float(rng.uniform(0.22, 0.58))
    if style == 0:
        blank = rng.random(shape) < probability
    elif style == 1:
        blank = ((rr + cc + int(rng.integers(3))) % int(rng.integers(2, 5))) == 0
    elif style == 2:
        blank = np.broadcast_to(rng.random((1, w)) < probability, shape).copy()
        blank ^= rng.random(shape) < 0.16
    else:
        blank = np.broadcast_to(rng.random((h, 1)) < probability, shape).copy()
        blank ^= rng.random(shape) < 0.16
    return np.where(blank, 0, texture).astype(np.int8)


def _anchor_shape(rng):
    h = int(rng.integers(9, 14))
    w = int(rng.integers(10, min(16, 27 - h) + 1))
    return (w, h) if int(rng.integers(2)) else (h, w)


def _shape(rng, test):
    if test:
        total = int(rng.integers(29, 32))
        if rng.random() < 0.65:
            h = int(rng.integers(3, 7))
        else:
            h = int(rng.integers(10, total - 9))
        shape = (h, total - h)
    else:
        h = int(rng.integers(3, 22))
        w = int(rng.integers(3, min(21, 28 - h) + 1))
        shape = (h, w)
    return shape[::-1] if int(rng.integers(2)) else shape


def _make_pair(rng, texture, anchor, test, seen, other_height=None):
    # At most 80 candidates, each <= 240 input cells; no retained retries/logs.
    for _ in range(80):
        shape = _anchor_shape(rng) if anchor is not None else _shape(rng, test)
        if anchor == 1 and shape[0] == other_height:
            continue
        if anchor == 0:
            kinds = ["fringe", "strip", "near"]
        elif anchor == 1:
            kinds = ["diagonal", "square", "near"]
        else:
            choices = ("square", "fringe", "diagonal", "stair", "elbow", "rectangle")
            kinds = [choices[int(rng.integers(len(choices)))]]
            kinds += [choices[int(rng.integers(len(choices)))]
                      for _ in range(int(rng.integers(3)))]
            kinds += ["near"] if rng.random() < 0.65 else []
        patches = [_motif(rng, kind) for kind in kinds]
        # Thin canvases retain the rule but need appropriately oriented motifs.
        fitted = []
        for patch in patches:
            if patch.shape[0] > shape[0] or patch.shape[1] > shape[1]:
                patch = patch.T
            if patch.shape[0] <= shape[0] and patch.shape[1] <= shape[1]:
                fitted.append(patch)
            elif anchor is not None:
                break
        if anchor is not None and len(fitted) != len(patches):
            continue
        if not fitted:
            fitted = [np.ones((3, 3), dtype=bool)]
        packed = _pack(rng, shape, fitted)
        if packed is None:
            continue
        blanks, protected = packed
        base = np.full(shape, texture, dtype=np.int8)
        base[blanks] = 0
        intended, _ = _window_union(base)
        if not intended:
            continue
        a = _texture(rng, shape, texture)
        a[protected] = texture
        a[blanks] = 0
        # Remove only accidental noisy windows, without changing motif geometry.
        # These edits are monotone (0 -> texture), so no new blank window appears.
        starts, _ = _window_union(a)
        intended_set = set(intended)
        for r, c in starts:
            if (r, c) in intended_set or np.any(a[r:r + 3, c:c + 3]):
                continue
            free = [(r + dr, c + dc) for dr in range(3) for dc in range(3)
                    if not protected[r + dr, c + dc]]
            if not free:
                raise AssertionError("unprotected accidental window expected")
            a[free[int(rng.integers(len(free)))]] = texture
        if not np.any(a == texture):
            continue
        key = (a.shape, a.tobytes())
        if key in seen:
            continue
        if set(_window_union(a)[0]) != intended_set:
            raise AssertionError("motif windows changed")
        seen.add(key)
        return {"input": a, "output": solve(a)}
    raise RuntimeError("bounded unique-pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(np.arange(2, 10))))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            texture = palette[i] if split == "train" and i < 2 else int(rng.choice(palette))
            anchor = i if split == "train" and i < 2 else None
            other_height = puzzle["train"][0]["input"].shape[0] if anchor == 1 else None
            puzzle[split].append(_make_pair(rng, texture, anchor, split == "test", seen, other_height))
    validate_generated_puzzle(puzzle)
    return puzzle


def _literal_union(a):
    """Simple specification check, using no vectorized detector or solver."""
    h, w = a.shape
    starts = []
    mask = np.zeros(a.shape, dtype=bool)
    for r in range(h - 2):
        for c in range(w - 2):
            if all(int(a[r + dr, c + dc]) == 0 for dr in range(3) for dc in range(3)):
                starts.append((r, c))
                for dr in range(3):
                    for dc in range(3):
                        mask[r + dr, c + dc] = True
    return starts, mask


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    while remaining:
        component = {remaining.pop()}
        frontier = list(component)
        while frontier:
            r, c = frontier.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    frontier.append(point)
        components.append(component)
    return components


def _features(a):
    starts, mask = _literal_union(a)
    h, w = a.shape
    overlap = len(starts) * 9 > int(mask.sum())
    components = _components(a == 0)
    fringe = any(any(mask[p] for p in component) and any(not mask[p] for p in component)
                 for component in components)
    largest = max(components, key=len) if components else set()
    target_set = set(map(tuple, np.argwhere(mask)))
    largest_fails = largest != target_set
    near = any(sum(int(a[r + dr, c + dc]) == 0 for dr in range(3) for dc in range(3)) == 8
               and not np.any(mask[r:r + 3, c:c + 3])
               for r in range(h - 2) for c in range(w - 2))
    edge = any(r in (0, h - 3) or c in (0, w - 3) for r, c in starts)
    return {"windows": len(starts), "marked": int(mask.sum()), "overlap": overlap,
            "fringe": fringe, "largest_fails": largest_fails, "near": near, "edge": edge,
            "target_components": len(_components(mask))}


def validate_generated_puzzle(puzzle):
    """Validate the relation and the evidence needed in even two demonstrations."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, training_features, textures = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            h, w = a.shape
            colors = set(map(int, np.unique(a)))
            assert 0 in colors and len(colors) == 2 and 1 not in colors
            assert h + w - 1 <= 30 and out.shape == (h, h + w - 1)
            starts, mask = _literal_union(a)
            assert starts, "generated examples must exercise window detection"
            expected = np.zeros(out.shape, dtype=np.int8)
            for r in range(h):
                for c in range(w):
                    expected[r, c + h - 1 - r] = 1 if mask[r, c] else a[r, c]
            assert np.array_equal(out, expected), "source-coordinate membership or shear is wrong"
            assert int(np.count_nonzero(out == 1)) == int(mask.sum())
            texture = next(iter(colors - {0}))
            assert np.count_nonzero(out == texture) == np.count_nonzero(a == texture)
            # Exact inverse row slices and their exposed padding, including zeros.
            for r in range(h):
                start = h - 1 - r
                restored = out[r, start:start + w].copy()
                restored[restored == 1] = 0
                assert np.array_equal(restored, a[r])
                assert not np.any(out[r, :start]) and not np.any(out[r, start + w:])
            assert not np.array_equal(out, _shear(a)), "B must affect each output"
            key = (a.shape, a.astype(np.int8).tobytes())
            assert key not in seen
            seen.add(key)
            if split == "train":
                textures.add(texture)
                training_features.append(_features(a))
    assert len(textures) >= 2
    assert len({np.asarray(p["input"]).shape[0] for p in puzzle["train"]}) >= 2
    assert any(f["windows"] == 1 for f in training_features)
    assert any(f["overlap"] and f["target_components"] >= 2 for f in training_features)
    assert any(f["fringe"] for f in training_features)
    assert any(f["near"] for f in training_features)
    assert any(f["largest_fails"] for f in training_features)


def _digest(puzzle):
    serial = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
              for s in ("train", "test")}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """42 streamed seeds, literal relation checks, mutations, and count boundaries."""
    shapes, textures, layouts, signatures = set(), set(), set(), set()
    counts, marked_sizes = set(), set()
    totals = {"pairs": 0, "overlap": 0, "edge": 0, "fringe": 0, "near": 0,
              "width_30": 0, "thin": 0, "shifted_tests": 0}
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_digest(p))
        max_train_width = max(pair["output"].shape[1] for pair in p["train"])
        for split in ("train", "test"):
            for pair in p[split]:
                a, out = pair["input"], pair["output"]
                saved = a.copy()
                assert np.array_equal(solve(a), out) and np.array_equal(a, saved)
                f = _features(a)
                shapes.add(a.shape)
                textures.update(set(map(int, np.unique(a))) - {0})
                starts, _ = _literal_union(a)
                layouts.add((a.shape, tuple(starts)))
                counts.add(f["windows"])
                marked_sizes.add(f["marked"])
                totals["pairs"] += 1
                for key in ("overlap", "edge", "fringe", "near"):
                    totals[key] += int(f[key])
                totals["width_30"] += out.shape[1] == 30
                totals["thin"] += min(a.shape) <= 4
                if split == "test":
                    assert out.shape[1] > max_train_width
                    totals["shifted_tests"] += 1
        # Wrong order: detect axis-aligned windows on the already sheared image,
        # including padding. This must visibly disagree on anchor demonstrations.
        for pair in p["train"][:2]:
            wrong = _shear(pair["input"])
            wrong[_literal_union(wrong)[1]] = 1
            assert not np.array_equal(wrong, pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 40 and len(layouts) >= 180
    assert len(textures) == 8 and len(counts) >= 8 and len(marked_sizes) >= 10
    assert all(totals[key] > 0 for key in totals)
    assert _digest(generate_puzzle(17, 7, 3)) == _digest(generate_puzzle(17, 7, 3))
    legal_calls = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(42000 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
            legal_calls += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Solver extensions: no windows, wholly blank input, and every width boundary.
    for h in range(1, 31):
        w = 31 - h
        a = np.full((h, w), 8, dtype=np.int8)
        if h >= 3 and w >= 3:
            a[:3, :3] = 0
        else:
            a[0, 0] = 0
        starts, mask = _literal_union(a)
        expected = np.zeros((h, 30), dtype=np.int8)
        for r in range(h):
            for c in range(w):
                expected[r, c + h - 1 - r] = 1 if mask[r, c] else a[r, c]
        assert np.array_equal(solve(a), expected)
    assert np.all(solve(np.zeros((3, 3), dtype=int))[1, 1:4] == 1)
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "layouts": len(layouts), "texture_colors": len(textures),
            "window_counts": sorted(counts), "marked_sizes": len(marked_sizes),
            "legal_count_calls": legal_calls, **totals}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
