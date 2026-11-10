"""Mirror-socket looms: the holes of a motif request reflected microcopies.

Only standard Python and NumPy are used.  The sole chamber is the motif's
coordinate system, not a panel containing a separately encoded puzzle.
"""
from __future__ import annotations

import numpy as np


CORE_SHAPES = tuple((h, w) for h in range(2, 6) for w in range(2, 6) if h != w)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _corners(top, left, height, width):
    bottom, right = top + height - 1, left + width - 1
    return {(top, left), (top, left + 1), (top + 1, left),
            (top, right), (top, right - 1), (top + 1, right),
            (bottom, left), (bottom, left + 1), (bottom - 1, left),
            (bottom, right), (bottom, right - 1), (bottom - 1, right)}


def _decode(input_grid):
    """Infer every role and the entire padded motif from visible geometry."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3 or int(counts.max()) * 2 <= a.size:
        raise ValueError("input must have three colors and a majority background")
    background = int(colors[np.argmax(counts)])
    possibilities = []
    for marker in (int(c) for c in colors if c != background):
        points = np.argwhere(a == marker)
        if len(points) != 12:
            continue
        top, left = (int(v) for v in points.min(axis=0))
        bottom, right = (int(v) for v in points.max(axis=0))
        fh, fw = bottom - top + 1, right - left + 1
        h, w = fh - 4, fw - 4
        if (h, w) not in CORE_SHAPES:
            continue
        if set(map(tuple, points.tolist())) != _corners(top, left, fh, fw):
            continue
        primary = next(int(c) for c in colors if c not in (background, marker))
        chamber = a[top:bottom + 1, left:right + 1]
        permitted = np.zeros(chamber.shape, dtype=bool)
        permitted[2:-2, 2:-2] = True
        for r, c in _corners(0, 0, fh, fw):
            permitted[r, c] = True
        if np.any(chamber[~permitted] != background):
            continue
        core = a[top + 2:bottom - 1, left + 2:right - 1]
        if not np.all((core == primary) | (core == background)):
            continue
        mask = core == primary
        if not 0 < int(mask.sum()) < h * w:
            continue
        possibilities.append({"background": background, "primary": primary,
                              "marker": marker, "top": top, "left": left,
                              "mask": mask.copy()})
    if len(possibilities) != 1:
        raise ValueError("input must encode exactly one unambiguous chamber")
    return possibilities[0]


def _reflection(mask, *, short_axis=False):
    h, w = mask.shape
    axis = 1 if w > h else 0
    if short_axis:
        axis = 1 - axis
    return np.flip(mask, axis=axis)


def _render(mask, background, primary, marker, mode="rule"):
    """Block construction; ablations are used only by validation/self_test."""
    h, w = mask.shape
    result = np.full((h * h + 4, w * w + 4), background, dtype=np.int8)
    echo = _reflection(mask, short_axis=(mode == "short_axis"))
    if mode == "no_reflection":
        echo = mask
    for r in range(h):
        for c in range(w):
            if mode == "uniform_primary":
                block = np.where(mask, primary, background)
            elif mask[r, c]:
                block = np.where(mask, primary, background)
            else:
                block = np.where(echo, marker, background)
            result[2 + r*h:2 + (r+1)*h, 2 + c*w:2 + (c+1)*w] = block
    for r, c in _corners(0, 0, *result.shape):
        result[r, c] = marker
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    return _render(info["mask"], info["background"], info["primary"], info["marker"])


def _full_extent(mask):
    return all(np.any(edge) for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]))


def _two_dimensional(mask):
    return np.unique(mask, axis=0).shape[0] > 1 and np.unique(mask, axis=1).shape[1] > 1


def _sample_mask(rng, shape, teaching=None, extreme=False):
    h, w = shape
    for _ in range(400):
        mode = int(rng.integers(7))
        if teaching is not None:
            mode = 0
        mask = np.zeros(shape, dtype=bool)
        if mode == 0:
            n = int(rng.integers(2, h*w - 1))
            mask.ravel()[rng.choice(h*w, n, replace=False)] = True
        elif mode == 1:
            # Off-center singleton or a single hole.
            mask[:] = bool(rng.integers(2))
            mask[int(rng.integers(h)), int(rng.integers(w))] ^= True
        elif mode == 2:
            mask = (np.indices(shape).sum(axis=0) + int(rng.integers(2))) % 2 == 0
        elif mode == 3:
            # A compact rectangle, with its empty rows/columns still in the domain.
            r0, c0 = int(rng.integers(h)), int(rng.integers(w))
            mask[r0:int(rng.integers(r0 + 1, h + 1)),
                 c0:int(rng.integers(c0 + 1, w + 1))] = True
        elif mode == 4:
            axis = int(rng.integers(2))
            if axis == 0:
                mask[rng.choice(h, int(rng.integers(1, h)), replace=False), :] = True
            else:
                mask[:, rng.choice(w, int(rng.integers(1, w)), replace=False)] = True
        elif mode == 5:
            mask = rng.random(shape) < float(rng.uniform(.2, .8))
            mask |= _reflection(mask)
        else:
            mask = rng.random(shape) < float(rng.uniform(.2, .8))
        if extreme:
            mask[:] = bool(rng.integers(2))
            mask[int(rng.integers(h)), int(rng.integers(w))] ^= True
        if teaching == "blank_edge":
            if w > h:
                mask[:, 0 if rng.random() < .5 else -1] = False
            else:
                mask[0 if rng.random() < .5 else -1, :] = False
        if not 0 < int(mask.sum()) < h*w:
            continue
        if teaching is not None:
            if (not _two_dimensional(mask)
                    or np.array_equal(mask, np.flipud(mask))
                    or np.array_equal(mask, np.fliplr(mask))
                    or np.array_equal(mask, mask[::-1, ::-1])):
                continue
            if teaching == "full_extent" and not _full_extent(mask):
                continue
            if teaching == "blank_edge" and _full_extent(mask):
                continue
        return mask
    raise RuntimeError("could not sample a teaching motif")


def _make_input(rng, mask, palette, require_noise=False):
    background, primary, marker = palette
    h, w = mask.shape
    fh, fw = h + 4, w + 4
    # Free placement, including occasional contact with a canvas boundary.
    height = int(rng.integers(fh + 4, min(30, fh + 18) + 1))
    width = int(rng.integers(fw + 4, min(30, fw + 18) + 1))
    top = int(rng.integers(height - fh + 1))
    left = int(rng.integers(width - fw + 1))
    a = np.full((height, width), background, dtype=np.int8)
    a[top + 2:top + h + 2, left + 2:left + w + 2] = np.where(mask, primary, background)
    for r, c in _corners(top, left, fh, fw):
        a[r, c] = marker
    outside = np.ones(a.shape, dtype=bool)
    outside[top:top + fh, left:left + fw] = False
    free = {tuple(map(int, p)) for p in np.argwhere(outside)}
    noise_count = int(rng.integers(0, 19))
    if require_noise:
        noise_count = max(3, noise_count)
    selected = []
    for i in range(noise_count):
        # Mix a short connected cluster with dispersed cells.
        adjacent = sorted(p for p in free if any(
            (p[0] + dr, p[1] + dc) in selected
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        choices = adjacent if adjacent and (i < 3 or rng.random() < .4) else sorted(free)
        p = choices[int(rng.integers(len(choices)))]
        selected.append(p)
        free.remove(p)
        a[p] = primary
    return a


def _key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    palette_mode = int(rng.integers(3))  # fixed, swapping inks, or fully local roles
    long_side = int(rng.integers(3, 5))
    first_shapes = [(2, long_side), (long_side, 2)]
    if rng.random() < .5:
        first_shapes.reverse()
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = None
            if split == "train" and index < 2:
                shape = first_shapes[index]
                teaching = "full_extent" if index == 0 else "blank_edge"
            elif split == "test" and rng.random() < .7:
                choices = [s for s in CORE_SHAPES if max(s) > long_side]
                shape = choices[int(rng.integers(len(choices)))]
            else:
                shape = CORE_SHAPES[int(rng.integers(len(CORE_SHAPES)))]
            if palette_mode == 1 and (index + (split == "test")) % 2:
                colors = palette[0], palette[2], palette[1]
            elif palette_mode == 2 and not (split == "train" and index == 0):
                colors = tuple(int(c) for c in rng.choice(10, 3, replace=False))
            else:
                colors = palette
            for _ in range(80):
                mask = _sample_mask(rng, shape, teaching,
                                    extreme=(split == "test" and rng.random() < .3))
                a = _make_input(rng, mask, colors,
                                require_noise=(split == "train" and index == 0))
                try:
                    _decode(a)
                except ValueError:
                    continue
                if _key(a) in seen:
                    continue
                seen.add(_key(a))
                puzzle[split].append({"input": a, "output": _render(mask, *colors)})
                break
            else:
                raise RuntimeError("could not construct a distinct unambiguous input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _cell_prediction(info):
    """Direct coordinate equation, separately written from the block renderer."""
    m = info["mask"]
    h, w = m.shape
    b, p, s = (info[k] for k in ("background", "primary", "marker"))
    a = np.full((h*h + 4, w*w + 4), b, dtype=np.int8)
    for y in range(h*h):
        for x in range(w*w):
            r, c = y // h, x // w
            u, v = y % h, x % w
            if m[r, c]:
                if m[u, v]:
                    a[y + 2, x + 2] = p
            else:
                occupied = m[u, w - 1 - v] if w > h else m[h - 1 - u, v]
                if occupied:
                    a[y + 2, x + 2] = s
    for r, c in _corners(0, 0, *a.shape):
        a[r, c] = s
    return a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _decode(a)
            m = info["mask"]
            h, w = m.shape
            p, s, b = (info[k] for k in ("primary", "marker", "background"))
            n = int(m.sum())
            assert out.shape == (h*h + 4, w*w + 4)
            assert np.array_equal(out, _cell_prediction(info))
            # Every macrocell carries n ink cells; original and echo inks disjoint.
            assert np.count_nonzero(out == p) == n*n
            assert np.count_nonzero(out == s) == (h*w - n)*n + 12
            assert set(map(int, np.unique(out))) == {b, p, s}
            assert np.all(out[1, 1:-1] == b) and np.all(out[-2, 1:-1] == b)
            assert np.all(out[1:-1, 1] == b) and np.all(out[1:-1, -2] == b)
            if split == "train":
                training.append((a, out, info))
    assert {"H" if i["mask"].shape[1] > i["mask"].shape[0] else "V"
            for _, _, i in training} == {"H", "V"}
    assert any(_full_extent(i["mask"]) for _, _, i in training)
    assert any(not _full_extent(i["mask"]) for _, _, i in training)
    assert any(np.count_nonzero(a == i["primary"]) > i["mask"].sum()
               for a, _, i in training)
    # Both directions have a teaching witness separating the main alternatives.
    for horizontal in (False, True):
        witnesses = [(o, i) for _, o, i in training
                     if (i["mask"].shape[1] > i["mask"].shape[0]) == horizontal]
        assert any(_two_dimensional(i["mask"])
                   and all(not np.array_equal(o, _render(i["mask"], i["background"],
                               i["primary"], i["marker"], mode))
                           for mode in ("no_reflection", "short_axis", "uniform_primary"))
                   for o, i in witnesses)


def self_test():
    """Bounded 42-seed audit, count boundaries, and metamorphic checks."""
    shapes, canvases, motifs, positions, palettes, densities = (set() for _ in range(6))
    puzzles = set()
    symmetric = exterior_free = blank_edges = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        signature = tuple(_key(p[k]) for split in ("train", "test")
                          for p in puzzle[split] for k in ("input", "output"))
        assert signature == tuple(_key(p[k]) for split in ("train", "test")
                                  for p in again[split] for k in ("input", "output"))
        puzzles.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                info = _decode(a)
                m = info["mask"]
                h, w = m.shape
                shapes.add(m.shape)
                canvases.add(a.shape)
                motifs.add((m.shape, m.tobytes()))
                positions.add((info["top"], info["left"]))
                palettes.add(tuple(info[k] for k in ("background", "primary", "marker")))
                densities.add((int(m.sum()), m.size))
                blank_edges += not _full_extent(m)
                symmetric += np.array_equal(m, _reflection(m))
                assert np.array_equal(solve(a.tolist()), out)
                # Exterior payload is irrelevant even when all of it is removed.
                clean = a.copy()
                outside = np.ones(a.shape, dtype=bool)
                t, l = info["top"], info["left"]
                outside[t:t+h+4, l:l+w+4] = False
                exterior_free += not np.any((a == info["primary"]) & outside)
                clean[outside] = info["background"]
                assert np.array_equal(solve(clean), out)
                # Arbitrary palette permutations preserve all geometric decisions.
                permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
                assert np.array_equal(solve(permutation[a]), permutation[out])
    assert len(puzzles) == 42 and shapes == set(CORE_SHAPES)
    assert len(canvases) >= 70 and len(motifs) >= 100 and len(positions) >= 50
    assert len(palettes) >= 50 and len(densities) >= 35
    assert symmetric >= 4 and exterior_free >= 3 and blank_edges >= 42
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(7100 + 10*train + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, reflection/scale/clutter invariants")
