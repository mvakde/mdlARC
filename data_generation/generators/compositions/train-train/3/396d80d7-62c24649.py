"""Corner prints from a stencil superposed by two perpendicular paper folds.

No parent module, seed, training example, or generator state is used by solve.
Only Python's standard library and NumPy are required.
"""

from __future__ import annotations

import json
import numpy as np


_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_STYLES = ("guides", "strokes", "walk", "ring", "stagger", "hooks", "mixture")
_SHORTCUTS = ("local", "prepaint", "diagonal", "translate", "horizontal_only",
              "vertical_only", "first", "overlap_only", "majority", "intersection",
              "no_shadow")
_CORE_SHORTCUTS = _SHORTCUTS[:-1]


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _regions(grid):
    h, w = grid.shape[0] // 2, grid.shape[1] // 2
    return [grid[:h, :w], grid[:h, w+1:],
            grid[h+1:, :w], grid[h+1:, w+1:]]


def _flip(region, layer):
    return region[::(-1 if layer // 2 else 1), ::(-1 if layer % 2 else 1)]


def _decode(input_grid):
    x = np.asarray(input_grid)
    if (x.ndim != 2 or not np.issubdtype(x.dtype, np.integer)
            or not all(3 <= n <= 29 and n % 2 == 1 for n in x.shape)
            or not np.all((0 <= x) & (x <= 9))):
        raise ValueError("expected an odd rectangular integer sheet, dimensions 3..29")
    h, w = x.shape[0] // 2, x.shape[1] // 2
    crease, ink = int(x[h, 0]), int(x[h, w])
    if (crease == ink or np.any(np.delete(x[h], w) != crease)
            or np.any(np.delete(x[:, w], h) != crease)):
        raise ValueError("invalid crease cross or center swatch")
    cells = np.concatenate([q.ravel() for q in _regions(x)])
    if np.any(cells == crease):
        raise ValueError("crease color is reserved for the cross")
    colors, counts = np.unique(cells, return_counts=True)
    remaining = [int(c) for c in colors if c != ink]
    if len(remaining) != 2:
        raise ValueError("expected background and scaffold in the regions")
    background = int(colors[int(np.argmax(counts))])
    if background == ink or int(counts.max()) * 2 <= cells.size:
        raise ValueError("background must occupy a strict majority of region cells")
    scaffold = next(c for c in remaining if c != background)
    return x, (background, scaffold, ink, crease)


def _neighbors(mask):
    p = np.pad(np.asarray(mask, dtype=np.int8), 1)
    d = p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
    o = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return d, o


def _eligible(mask):
    d, o = _neighbors(mask)
    return ~mask & (d > 0) & (o == 0)


def _layers(x, scaffold):
    return [_flip(q == scaffold, i) for i, q in enumerate(_regions(x))]


def _print(x, printable, background, ink, reflect=True):
    result = x.copy()
    for i, q in enumerate(_regions(result)):
        mask = _flip(printable, i) if reflect else printable
        q[(q == background) & mask] = ink
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles and print locations from the visible sheet alone."""
    x, (background, scaffold, ink, _) = _decode(input_grid)
    union = np.logical_or.reduce(_layers(x, scaffold))
    return _print(x, _eligible(union), background, ink)


def _manual_result(x, roles):
    """Separate scalar coordinate/set formulation for validation, not generation."""
    bg, sc, ink, _ = roles
    H, W = x.shape
    h, w = H // 2, W // 2
    stencil = set()
    for r in range(H):
        for c in range(W):
            if r != h and c != w and int(x[r, c]) == sc:
                stencil.add((min(r, H-1-r), min(c, W-1-c)))
    expected = x.copy()
    for r in range(H):
        for c in range(W):
            if r == h or c == w or int(x[r, c]) != bg:
                continue
            a, b = min(r, H-1-r), min(c, W-1-c)
            diagonal = any((a+dr, b+dc) in stencil for dr, dc in _DIAG)
            edge = any((a+dr, b+dc) in stencil for dr, dc in _ORTH)
            if (a, b) not in stencil and diagonal and not edge:
                expected[r, c] = ink
    return expected


def _alternative(x, mode):
    x, (bg, sc, ink, _) = _decode(x)
    layers = _layers(x, sc)
    union = np.logical_or.reduce(layers)
    if mode == "local":
        y = x.copy()
        for q in _regions(y):
            q[(q == bg) & _eligible(q == sc)] = ink
        return y
    if mode == "prepaint":
        mask = ~union & np.logical_or.reduce([_eligible(a) for a in layers])
    elif mode == "diagonal":
        mask = ~union & (_neighbors(union)[0] > 0)
    elif mode == "translate":
        raw_union = np.logical_or.reduce([q == sc for q in _regions(x)])
        return _print(x, _eligible(raw_union), bg, ink, reflect=False)
    elif mode in ("horizontal_only", "vertical_only"):
        partial = []
        for i, q in enumerate(_regions(x)):
            transform = (i % 2) if mode == "horizontal_only" else (2*(i // 2))
            partial.append(_flip(q == sc, transform))
        mask = _eligible(np.logical_or.reduce(partial))
        y = x.copy()
        for i, q in enumerate(_regions(y)):
            transform = (i % 2) if mode == "horizontal_only" else (2*(i // 2))
            q[(q == bg) & _flip(mask, transform)] = ink
        return y
    elif mode == "first":
        mask = _eligible(layers[0])
    elif mode == "overlap_only":
        mask = _eligible(np.sum(layers, axis=0) >= 2)
    elif mode == "majority":
        mask = _eligible(np.sum(layers, axis=0) >= 3)
    elif mode == "intersection":
        mask = _eligible(np.sum(layers, axis=0) == 4)
    elif mode == "no_shadow":
        d, o = _neighbors(union)
        mask = (d > 0) & (o == 0)
    else:
        raise ValueError(mode)
    return _print(x, mask, bg, ink)


def _features(x, output=None):
    x, roles = _decode(x)
    bg, sc, ink, _ = roles
    layers = _layers(x, sc)
    union = np.logical_or.reduce(layers)
    d, o = _neighbors(union)
    mask = _eligible(union)
    y = solve(x) if output is None else output
    remote = cross_veto = existing_qualifying = existing_unrelated = 0
    for i, (q, layer) in enumerate(zip(_regions(x), layers)):
        local_d, local_o = _neighbors(layer)
        available = _flip(q == bg, i)
        remote += int(np.count_nonzero(mask & available & (local_d == 0)))
        cross_veto += int(np.count_nonzero(
            ~union & available & (local_d > 0) & (local_o == 0) & (o > 0)))
        folded_ink = _flip(q == ink, i)
        existing_qualifying += int(np.count_nonzero(folded_ink & mask))
        existing_unrelated += int(np.count_nonzero(folded_ink & ~union & (d == 0) & (o == 0)))
    rr, cc = np.nonzero(union)
    interior = mask.copy()
    interior[:int(rr.min())+1] = False
    interior[int(rr.max()):] = False
    interior[:, :int(cc.min())+1] = False
    interior[:, int(cc.max()):] = False
    return {
        "shape": list(union.shape), "roles": list(roles),
        "active_layers": sum(bool(a.any()) for a in layers),
        "scaffold_sites": int(union.sum()),
        "scaffold_cells": sum(int(a.sum()) for a in layers),
        "overlap_sites": int(np.count_nonzero(np.sum(layers, axis=0) >= 2)),
        "eligible": int(mask.sum()),
        "veto": int(np.count_nonzero(~union & (d > 0) & (o > 0))),
        "double": int(np.count_nonzero(mask & (d >= 2))),
        "interior": int(interior.sum()),
        "remote": remote, "cross_veto": cross_veto,
        "existing_ink": sum(int((q == ink).sum()) for q in _regions(x)),
        "existing_qualifying": existing_qualifying,
        "existing_unrelated": existing_unrelated,
        "boundary_scaffold": bool(union[0].any() or union[-1].any()
                                  or union[:, 0].any() or union[:, -1].any()),
        "boundary_print": bool(mask[0].any() or mask[-1].any()
                               or mask[:, 0].any() or mask[:, -1].any()),
        "shortcut_differences": {m: int(np.count_nonzero(y != _alternative(x, m)))
                                 for m in _SHORTCUTS},
    }


def _motif(rng, h, w, style):
    """Stochastic structured geometry; no stored puzzle or witness template."""
    cells = set()

    def add(r, c):
        if 0 <= r < h and 0 <= c < w:
            cells.add((int(r), int(c)))

    def point():
        return int(rng.integers(h)), int(rng.integers(w))

    if style == "guides":
        for _ in range(int(rng.integers(4, max(6, h*w//7)))):
            r, c = point()
            add(r, c)
            if rng.random() < .35:
                dr, dc = _ORTH[int(rng.integers(4))]
                add(r+dr, c+dc)
    elif style == "strokes":
        for _ in range(int(rng.integers(2, 6))):
            r, c = point()
            dr, dc = (_ORTH + _DIAG)[int(rng.integers(8))]
            for k in range(int(rng.integers(2, min(6, max(h, w))))):
                add(r+k*dr, c+k*dc)
    elif style == "walk":
        for _ in range(int(rng.integers(1, 4))):
            r, c = point()
            for _ in range(int(rng.integers(4, h+w))):
                add(r, c)
                dr, dc = (_DIAG + _ORTH)[int(rng.integers(8))]
                r, c = int(np.clip(r+dr, 0, h-1)), int(np.clip(c+dc, 0, w-1))
    elif style == "ring":
        r, c = point()
        radius = int(rng.integers(2, max(3, min(h, w)//2+1)))
        for dr in range(-radius, radius+1):
            dc = radius-abs(dr)
            for sign in (-1, 1):
                if rng.random() < .92:
                    add(r+dr, c+sign*dc)
        # Occasional short spur changes both interior and exterior neighborhoods.
        if rng.random() < .7:
            for k in range(int(rng.integers(1, 4))):
                add(r+k, c)
    elif style == "stagger":
        period = int(rng.integers(2, 5))
        step = int(rng.integers(2, 4))
        for j, r in enumerate(range(int(rng.integers(2)), h, step)):
            for c in range(j % 2, w, period):
                if rng.random() < .8:
                    add(r, c)
                    if rng.random() < .35:
                        add(r, c+1)
    elif style == "hooks":
        for _ in range(int(rng.integers(2, 5))):
            r, c = point()
            dr, dc = _ORTH[int(rng.integers(4))]
            length = int(rng.integers(2, 5))
            for k in range(length):
                add(r+k*dr, c+k*dc)
            for k in range(1, int(rng.integers(2, 5))):
                add(r+(length-1)*dr-k*dc, c+(length-1)*dc+k*dr)
    elif style == "mixture":
        styles = rng.choice(_STYLES[:-1], size=2, replace=False)
        for child in styles:
            a = _motif(rng, h, w, str(child))
            cells.update(zip(*np.nonzero(a)))
    else:
        raise ValueError(style)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    # Geometry orientation is independent of palette and region assignments.
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    return mask


def _sheet(layers, roles, rng=None):
    bg, sc, ink, crease = roles
    h, w = layers[0].shape
    x = np.full((2*h+1, 2*w+1), bg, dtype=np.int8)
    x[h, :] = crease
    x[:, w] = crease
    x[h, w] = ink
    for i, q in enumerate(_regions(x)):
        q[_flip(layers[i], i)] = sc
    if rng is not None:
        # Existing ink can be on a future print, in a stencil shadow, or elsewhere.
        for _ in range(int(rng.integers(0, 6))):
            options = [(i, int(r), int(c)) for i, q in enumerate(_regions(x))
                       for r, c in zip(*np.nonzero(q == bg))]
            if options:
                i, r, c = options[int(rng.integers(len(options)))]
                _regions(x)[i][r, c] = ink
    return x


def _make_pair(rng, split, index, preferred, roles):
    if split == "test":
        low, high, active_count = 8, 14, 4
    elif index == 0:
        low, high, active_count = 4, 7, 2
    elif index == 1:
        low, high, active_count = 5, 9, 3
    else:
        low, high, active_count = 4, 10, int(rng.integers(2, 4))
    witness = split == "train" and index < 2
    for _ in range(600):
        h, w = (int(v) for v in rng.integers(low, high+1, size=2))
        style = preferred if rng.random() < .55 else str(rng.choice(_STYLES))
        # Occupied extent and blank margins vary independently of panel size.
        mh, mw = h, w
        roomy = rng.random() < .5
        margin_h, margin_w = int(roomy and h >= 6), int(roomy and w >= 6)
        if roomy:
            mh = int(rng.integers(4, h-2*margin_h+1))
            mw = int(rng.integers(4, w-2*margin_w+1))
        top = int(rng.integers(margin_h, h-mh-margin_h+1))
        left = int(rng.integers(margin_w, w-mw-margin_w+1))
        union = np.zeros((h, w), dtype=bool)
        union[top:top+mh, left:left+mw] = _motif(rng, mh, mw, style)
        d, o = _neighbors(union)
        eligible = _eligible(union)
        if not (4 <= int(union.sum()) <= h*w*.38 and int(eligible.sum()) >= 3
                and np.any(~union & (d > 0) & (o > 0))):
            continue
        if witness and not np.any(eligible & (d >= 2)):
            continue
        active = rng.choice(4, size=active_count, replace=False)
        layers = [np.zeros((h, w), dtype=bool) for _ in range(4)]
        duplicate_probability = float(rng.uniform(.08, .4))
        for r, c in zip(*np.nonzero(union)):
            chosen = [int(rng.choice(active))]
            if rng.random() < duplicate_probability:
                chosen = list(rng.choice(active, size=int(rng.integers(2, active_count+1)),
                                         replace=False))
            for layer in chosen:
                layers[int(layer)][r, c] = True
        if sum(bool(a.any()) for a in layers) != active_count:
            continue
        x = _sheet(layers, roles, rng)
        if split == "train" and index == 1:
            unrelated = np.argwhere(~union & (d == 0) & (o == 0))
            if len(unrelated) == 0:
                continue
            # Explicitly demonstrate preservation both on and off the print mask.
            for options in (unrelated, np.argwhere(eligible)):
                r, c = options[int(rng.integers(len(options)))]
                layer = int(rng.integers(4))
                _flip(_regions(x)[layer], layer)[r, c] = roles[2]
        y = solve(x)
        f = _features(x, y)
        if (f["remote"] == 0 or f["cross_veto"] == 0
                or not all(f["shortcut_differences"][m] for m in _CORE_SHORTCUTS)):
            continue
        if witness and (f["interior"] == 0 or f["double"] == 0
                        or f["shortcut_differences"]["no_shadow"] == 0):
            continue
        return {"input": x, "output": y}
    raise RuntimeError("bounded construction exhausted; no hidden fallback rule")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    preferred = str(rng.choice(_STYLES))
    stable = bool(rng.integers(2))
    shared_roles = tuple(int(v) for v in rng.choice(10, size=4, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            roles = shared_roles if stable else tuple(
                int(v) for v in rng.choice(10, size=4, replace=False))
            for _ in range(40):
                pair = _make_pair(rng, split, index, preferred, roles)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen = set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            x, y = (np.asarray(pair[k]) for k in ("input", "output"))
            for grid in (x, y):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            x, roles = _decode(x)
            assert x.shape == y.shape and _key(x) not in seen
            seen.add(_key(x))
            assert np.array_equal(y, _manual_result(x, roles)), "scalar rule check"
            assert np.array_equal(y[x != roles[0]], x[x != roles[0]])
            changed = x != y
            assert changed.any() and np.all(y[changed] == roles[2])
            f = _features(x, y)
            assert f["eligible"] >= 3 and f["remote"] > 0 and f["cross_veto"] > 0
            assert all(f["shortcut_differences"][m] for m in _CORE_SHORTCUTS), "shortcut fits an example"
            if split == "train" and index < 2:
                assert f["interior"] > 0 and f["double"] > 0
                assert f["shortcut_differences"]["no_shadow"] > 0
                assert f["active_layers"] == index+2
                if index == 1:
                    assert f["existing_qualifying"] > 0 and f["existing_unrelated"] > 0
            elif split == "test":
                assert f["active_layers"] == 4
                assert all(8 <= v <= 14 for v in f["shape"])


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def _targeted_tests():
    # A corner supplied on layer 0 is vetoed by a distant edge on layer 3.
    layers = [np.zeros((6, 7), dtype=bool) for _ in range(4)]
    layers[0][1, 1] = True
    roles = (0, 5, 2, 8)
    before = _sheet(layers, roles)
    folded_before = _flip(_regions(solve(before))[0], 0)
    assert folded_before[2, 2] == 2
    layers[3][2, 3] = True
    after = _sheet(layers, roles)
    for i, q in enumerate(_regions(solve(after))):
        assert _flip(q, i)[2, 2] == 0, "remote orthogonal veto"
    assert np.array_equal(_regions(after)[0], _regions(before)[0])
    # Duplicate occupancy cannot cancel or multiply an influence.
    layers[1][1, 1] = True
    duplicate = _sheet(layers, roles)
    a, b = solve(after), solve(duplicate)
    differences = after != duplicate
    assert np.array_equal(a[~differences], b[~differences])
    # Ink neither emits scaffold influence nor replaces the stencil.
    inked = after.copy()
    inked[5, 6] = roles[2]
    expected = solve(after)
    expected[5, 6] = roles[2]
    assert np.array_equal(solve(inked), expected)
    # A shadow stays blank even when it has otherwise qualifying corner contact.
    shadow_layers = [np.zeros((6, 7), dtype=bool) for _ in range(4)]
    shadow_layers[0][1, 1] = True
    shadow_layers[3][2, 2] = True
    shadow = _sheet(shadow_layers, roles)
    assert _flip(_regions(solve(shadow))[1], 1)[1, 1] == roles[0]
    assert _flip(_regions(_alternative(shadow, "no_shadow"))[1], 1)[1, 1] == roles[2]
    # Degenerate folded axes are supported by solve even though not generated.
    for shape in ((1, 4), (4, 1), (1, 1)):
        tiny = [np.zeros(shape, dtype=bool) for _ in range(4)]
        tiny[0][0, 0] = True
        x = _sheet(tiny, roles)
        assert np.array_equal(solve(x), _manual_result(x, roles))


def self_test():
    """Bounded deterministic suite; return compact descriptive measurements."""
    _targeted_tests()
    shapes, stencils, assignments, palettes, site_counts, overlaps = (set() for _ in range(6))
    signatures = set()
    boundary_scaffold = boundary_print = inked = mixed_shape = varied_roles = 0
    total_pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        assert _same(p, generate_puzzle(seed)), "seed determinism"
        signatures.add(tuple(_key(q["input"]) for s in p for q in p[s]))
        local_shapes, local_roles = set(), set()
        for split in ("train", "test"):
            for pair in p[split]:
                x, y = pair["input"], pair["output"]
                _, roles = _decode(x)
                ls = _layers(x, roles[1])
                union = np.logical_or.reduce(ls)
                f = _features(x, y)
                shapes.add(union.shape)
                local_shapes.add(union.shape)
                stencils.add(_key(union))
                assignments.add(tuple(_key(layer) for layer in ls))
                palettes.add(roles)
                local_roles.add(roles)
                site_counts.add(f["scaffold_sites"])
                overlaps.add(f["overlap_sites"])
                boundary_scaffold += f["boundary_scaffold"]
                boundary_print += f["boundary_print"]
                inked += f["existing_ink"] > 0
                total_pairs += 1
                # Arbitrary color relabeling must commute with the transformation.
                permutation = np.random.default_rng(seed+711).permutation(10)
                assert np.array_equal(solve(permutation[x]), permutation[y])
                # Swapping sheet axes or reflecting the entire sheet changes no rule.
                for transform in (lambda z: z.T, lambda z: z[:, ::-1], lambda z: z[::-1]):
                    assert np.array_equal(solve(transform(x)), transform(y))
        mixed_shape += len(local_shapes) > 1
        varied_roles += len(local_roles) > 1
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(stencils) >= 180 and len(assignments) >= 190
    assert len(palettes) >= 60 and len(site_counts) >= 20 and len(overlaps) >= 8
    assert boundary_scaffold >= 80 and boundary_print >= 80 and inked >= 100
    assert mixed_shape == 42 and varied_roles >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(9011, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    for seed in (-1, True, 1.0, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    return {"seeds": 42, "pairs": total_pairs, "folded_shapes": len(shapes),
            "distinct_stencils": len(stencils), "layer_assignments": len(assignments),
            "role_palettes": len(palettes), "site_counts": len(site_counts),
            "overlap_counts": len(overlaps), "boundary_scaffold_pairs": boundary_scaffold,
            "boundary_print_pairs": boundary_print, "existing_ink_pairs": inked,
            "mixed_shape_puzzles": mixed_shape, "varied_role_puzzles": varied_roles}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
