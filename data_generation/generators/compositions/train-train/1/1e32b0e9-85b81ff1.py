"""Area-prioritized, color-provenance completion of registered stencil fragments."""

from __future__ import annotations

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return grid.astype(np.int8, copy=False)


def _parse(input_grid):
    grid = _grid(input_grid)
    frame = int(grid[0, 0])
    if not (np.all(grid[[0, -1], :] == frame)
            and np.all(grid[:, [0, -1]] == frame)):
        raise ValueError("the scene needs an enclosing frame")
    lines_r = np.flatnonzero(np.all(grid == frame, axis=1)).tolist()
    lines_c = np.flatnonzero(np.all(grid == frame, axis=0)).tolist()
    if len(lines_r) < 2 or len(lines_c) < 2:
        raise ValueError("missing tile boundaries")
    heights = np.diff(lines_r) - 1
    widths = np.diff(lines_c) - 1
    if (min(heights) < 3 or min(widths) < 3
            or len(set(heights)) != 1 or len(set(widths)) != 1):
        raise ValueError("tiles must have congruent interiors of at least 3x3")
    nr, nc = len(heights), len(widths)
    if not 3 <= nr * nc <= 6:
        raise ValueError("expected three through six sources")
    bounds, blocks = [], []
    for r in range(nr):
        for c in range(nc):
            box = (lines_r[r] + 1, lines_r[r + 1],
                   lines_c[c] + 1, lines_c[c + 1])
            a, b, x, y = box
            bounds.append(box)
            blocks.append(grid[a:b, x:y])
    common = set.intersection(*(set(map(int, np.unique(b))) for b in blocks))
    if len(common) != 1:
        raise ValueError("background must be the unique common interior color")
    background = common.pop()
    if frame == background:
        raise ValueError("frame and background must differ")
    colors, masks = [], []
    for block in blocks:
        palette = set(map(int, np.unique(block)))
        if frame in palette or len(palette) != 2 or background not in palette:
            raise ValueError("each tile needs one source color and background")
        color = (palette - {background}).pop()
        colors.append(color)
        masks.append(block == color)
    if len(set(colors)) != len(colors):
        raise ValueError("source colors must be distinct")
    masks = np.array(masks, dtype=bool)
    areas = masks.sum(axis=(1, 2)).astype(int)
    if len(set(map(int, areas))) != len(areas):
        raise ValueError("source areas must be distinct")
    union = np.any(masks, axis=0)
    if union.all():
        raise ValueError("the stencil must leave some non-stencil background")
    return dict(grid=grid, frame=frame, background=background, colors=tuple(colors),
                masks=masks, areas=areas, union=union, bounds=bounds,
                geometry=(nr, nc, int(heights[0]), int(widths[0])))


def _owners(info, order):
    owner = np.full(info["union"].shape, -1, dtype=int)
    for i in order:
        owner[info["masks"][i] & (owner == -1)] = int(i)
    return owner


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer registration, source roles, union, and repair priority from input only."""
    info = _parse(input_grid)
    owner = _owners(info, np.argsort(info["areas"]))
    output = info["grid"].copy()
    for i, (a, b, x, y) in enumerate(info["bounds"]):
        missing = info["union"] & ~info["masks"][i]
        output[a:b, x:y][missing] = np.asarray(info["colors"])[owner[missing]]
    return output


def _render(masks, colors, geometry, background, frame):
    """Construction renderer; does not call solve or its input parser."""
    nr, nc, h, w = geometry
    inp = np.full((nr * (h + 1) + 1, nc * (w + 1) + 1), background, dtype=np.int8)
    inp[::h + 1, :] = frame
    inp[:, ::w + 1] = frame
    out = inp.copy()
    sizes = [int(mask.sum()) for mask in masks]
    for receiver, mask in enumerate(masks):
        r, c = divmod(receiver, nc)
        a, x = 1 + r * (h + 1), 1 + c * (w + 1)
        inp[a:a + h, x:x + w][mask] = colors[receiver]
        block = out[a:a + h, x:x + w]
        for row in range(h):
            for col in range(w):
                if mask[row, col]:
                    block[row, col] = colors[receiver]
                else:
                    donors = [i for i, source in enumerate(masks) if source[row, col]]
                    if donors:
                        donor = min(donors, key=lambda i: sizes[i])
                        block[row, col] = colors[donor]
    return {"input": inp, "output": out}


def _geometry(rng, n, square=False):
    factors = [(r, n // r) for r in range(1, n + 1) if n % r == 0]
    nr, nc = (2, 2) if square else factors[int(rng.integers(len(factors)))]
    h = int(rng.integers(3, min(9, 29 // nr - 1) + 1))
    w = int(rng.integers(3, min(9, 29 // nc - 1) + 1))
    return nr, nc, h, w


def _stencil(rng, h, w, n):
    kinds = ("rectangle", "frame", "walk", "growth", "scatter", "punctured")
    for _ in range(100):
        kind = kinds[int(rng.integers(len(kinds)))]
        sh, sw = int(rng.integers(2, h + 1)), int(rng.integers(2, w + 1))
        if sh * sw < n + 2:
            continue
        local = np.zeros((sh, sw), dtype=bool)
        if kind == "rectangle":
            local[:] = True
        elif kind == "frame":
            if min(sh, sw) < 3:
                continue
            local[[0, -1], :] = True
            local[:, [0, -1]] = True
        elif kind == "walk":
            r, c = int(rng.integers(sh)), int(rng.integers(sw))
            local[r, c] = True
            for _ in range(int(rng.integers(3, 7))):
                tr, tc = int(rng.integers(sh)), int(rng.integers(sw))
                if rng.random() < 0.5:
                    local[min(r, tr):max(r, tr) + 1, c] = True
                    local[tr, min(c, tc):max(c, tc) + 1] = True
                else:
                    local[r, min(c, tc):max(c, tc) + 1] = True
                    local[min(r, tr):max(r, tr) + 1, tc] = True
                r, c = tr, tc
        elif kind == "growth":
            local[int(rng.integers(sh)), int(rng.integers(sw))] = True
            target = int(rng.integers(n + 2, sh * sw + 1))
            while int(local.sum()) < target:
                frontier = set()
                for r, c in np.argwhere(local):
                    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        rr, cc = int(r + dr), int(c + dc)
                        if 0 <= rr < sh and 0 <= cc < sw and not local[rr, cc]:
                            frontier.add((rr, cc))
                candidates = sorted(frontier)
                local[candidates[int(rng.integers(len(candidates)))]] = True
        elif kind == "scatter":
            count = int(rng.integers(n + 2, sh * sw + 1))
            local.flat[rng.choice(sh * sw, count, replace=False)] = True
        else:
            local[:] = True
            interior = [(r, c) for r in range(1, sh - 1) for c in range(1, sw - 1)]
            if not interior:
                continue
            holes = int(rng.integers(1, len(interior) + 1))
            for j in rng.choice(len(interior), holes, replace=False):
                local[interior[int(j)]] = False
        if not n + 2 <= int(local.sum()) < h * w:
            continue
        # Crop before placement, so offsets do not depend on invisible padding.
        points = np.argwhere(local)
        top, left = points.min(axis=0)
        bottom, right = points.max(axis=0)
        local = local[top:bottom + 1, left:right + 1]
        lh, lw = local.shape
        placement = int(rng.integers(4))
        if placement == 0:
            a, x = 0, 0
        elif placement == 1:
            a, x = (h - lh) // 2, (w - lw) // 2
        else:
            a, x = int(rng.integers(h - lh + 1)), int(rng.integers(w - lw + 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[a:a + lh, x:x + lw] = local
        return mask
    # Bounded fallback: still an arbitrary registered stencil, never a stored pair.
    mask = np.zeros((h, w), dtype=bool)
    mask.flat[rng.choice(h * w, n + 2, replace=False)] = True
    return mask


def _ranked_fragments(rng, stencil, n, full):
    points = np.argwhere(stencil)
    points = points[rng.permutation(len(points))]
    m = len(points)
    # Every rank owns a position not seen in any earlier rank. Extra positions
    # have random owners; later ranks may borrow earlier-owned positions.
    labels = np.empty(m, dtype=int)
    labels[:n] = np.arange(n)
    labels[n:] = rng.integers(0, n, size=m - n)
    # In complementary scenes the first rank must leave room for n-1 strictly
    # larger, still incomplete sources. There is no absolute small-size window.
    cap = m - n + (1 if full else 0)
    extra_zeroes = np.flatnonzero(labels[n:] == 0) + n
    excess = int(np.count_nonzero(labels == 0)) - cap
    if excess > 0:
        labels[extra_zeroes[:excess]] = rng.integers(1, n, size=excess)
    fragments, previous = [], 0
    for rank in range(n):
        own = np.flatnonzero(labels == rank)
        older = np.flatnonzero(labels < rank)
        low = max(len(own), previous + 1)
        high = min(len(own) + len(older), m - (n - 1 - rank) - (not full))
        # The ownership construction guarantees a nonempty feasible interval.
        size = m if full and rank == n - 1 else int(rng.integers(low, high + 1))
        selected = np.concatenate((own, rng.choice(older, size - len(own), replace=False)))
        mask = np.zeros(stencil.shape, dtype=bool)
        mask[tuple(points[selected].T)] = True
        fragments.append(mask)
        previous = size
    return np.array(fragments)


def _alternative(info, mode):
    """Deliberately wrong rules, used only for evidence checks."""
    n = len(info["colors"])
    if mode == "largest":
        order = np.argsort(-info["areas"])
    elif mode == "reading":
        order = list(range(n))
    elif mode == "numeric":
        order = sorted(range(n), key=lambda i: info["colors"][i])
    elif mode == "bbox":
        def bbox(i):
            p = np.argwhere(info["masks"][i])
            return int(np.prod(p.max(axis=0) - p.min(axis=0) + 1))
        order = sorted(range(n), key=bbox)
    else:
        order = np.argsort(info["areas"])
    output = info["grid"].copy()
    for receiver, (a, b, x, y) in enumerate(info["bounds"]):
        missing = info["union"] & ~info["masks"][receiver]
        if mode == "frame":
            output[a:b, x:y][missing] = info["frame"]
            continue
        if mode == "nearest":
            nr, nc, _, _ = info["geometry"]
            r, c = divmod(receiver, nc)
            order = sorted(range(n), key=lambda i: (abs(i // nc - r) + abs(i % nc - c), i))
        owner = _owners(info, order)
        output[a:b, x:y][missing] = np.asarray(info["colors"])[owner[missing]]
    return output


def _evidence(info, output, pedagogy=False):
    owner = _owners(info, np.argsort(info["areas"]))
    used = set()
    competing = False
    for mask in info["masks"]:
        missing = info["union"] & ~mask
        used.update(map(int, np.unique(owner[missing])))
        competing |= bool(np.any(missing & (info["masks"].sum(axis=0) >= 2)))
    if used != set(range(len(info["colors"]))) or not competing:
        return False
    modes = ["largest", "reading", "numeric", "frame"]
    if pedagogy:
        modes += ["bbox", "nearest"]
    return all(not np.array_equal(output, _alternative(info, mode)) for mode in modes)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    backgrounds = list(map(int, rng.permutation(10)))
    square_first = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(num_train + num_test):
        training = index < num_train
        if index < 2:
            square = (index == 0) == square_first
            n = 4 if square else int(rng.choice([3, 5]))
            full = index == 0
        else:
            square = False
            n = int(rng.integers(3, 6)) if training else int(rng.integers(5, 7))
            full = training and rng.random() < 0.5
        background = backgrounds[index % 10]
        for _ in range(256):
            geometry = _geometry(rng, n, square)
            stencil = _stencil(rng, geometry[2], geometry[3], n)
            ranked = _ranked_fragments(rng, stencil, n, full)
            masks = ranked[rng.permutation(n)]
            palette = list(map(int, rng.choice([c for c in range(10) if c != background],
                                                n + 1, replace=False)))
            frame, colors = palette[0], palette[1:]
            pair = _render(masks, colors, geometry, background, frame)
            signature = (pair["input"].shape, pair["input"].tobytes())
            info = _parse(pair["input"])
            if signature not in seen and _evidence(info, pair["output"], index < 2):
                seen.add(signature)
                puzzle["train" if training else "test"].append(pair)
                break
        else:
            raise RuntimeError(f"could not construct informative example {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema and semantic invariants by reparsing the returned grids."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, infos = set(), []
    for index, pair in enumerate(train + test):
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        try:
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            info = _parse(inp)
        except (ValueError, TypeError, IndexError) as error:
            raise AssertionError("invalid stencil-repair scene") from error
        assert inp.shape == out.shape
        signature = (inp.shape, inp.tobytes())
        assert signature not in seen, "duplicate input"
        seen.add(signature)
        assert np.array_equal(solve(inp), out), "incorrect repair output"
        assert _evidence(info, out, index < 2), "missing disambiguating evidence"
        interior = np.zeros(inp.shape, dtype=bool)
        for i, (a, b, x, y) in enumerate(info["bounds"]):
            interior[a:b, x:y] = True
            before, after = inp[a:b, x:y], out[a:b, x:y]
            assert np.array_equal(after != info["background"], info["union"])
            assert np.array_equal(after[info["masks"][i]], before[info["masks"][i]])
            assert np.all(after[~info["union"]] == info["background"])
            # Direct per-cell check, independent of the sorted-stamp algorithm.
            for row, col in np.argwhere(info["union"] & ~info["masks"][i]):
                donors = np.flatnonzero(info["masks"][:, row, col])
                donor = min(donors, key=lambda j: int(info["areas"][j]))
                assert int(after[row, col]) == info["colors"][donor]
        assert np.array_equal(out[~interior], inp[~interior]), "frame changed"
        assert set(map(int, np.unique(out))) == set(map(int, np.unique(inp)))
        assert np.any(inp != out), "repair must be nontrivial"
        infos.append(info)
    sizes = [int(i["union"].sum()) for i in infos]
    assert max(infos[0]["areas"]) == sizes[0], "first demo needs an exemplar"
    assert max(infos[1]["areas"]) < sizes[1], "second demo needs distributed evidence"
    assert {i["geometry"][:2] for i in infos[:2]} >= {(2, 2)}
    assert len({i["geometry"] for i in infos[:2]}) == 2
    assert infos[0]["background"] != infos[1]["background"]
    for info, size in zip(infos[len(train):], sizes[len(train):]):
        assert len(info["colors"]) in (5, 6) and max(info["areas"]) < size


def _signature(puzzle):
    return tuple((split, np.asarray(p["input"]).shape, np.asarray(p["input"]).tobytes(),
                  np.asarray(p["output"]).tobytes())
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded semantic, corruption, count, reproducibility, and diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(generate_puzzle(0)) == _signature(puzzles[0])
    geometries, stencils, area_vectors, role_palettes, bgs, source_counts, smallest = [set() for _ in range(7)]
    corners = disconnected = full = complementary = 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                geometries.add(info["geometry"])
                stencils.add((info["union"].shape, info["union"].tobytes()))
                area_vectors.add(tuple(sorted(map(int, info["areas"]))))
                role_palettes.add((info["background"], info["frame"], info["colors"]))
                bgs.add(info["background"])
                source_counts.add(len(info["colors"]))
                smallest.add(int(min(info["areas"])))
                corners += bool(info["union"][0, 0])
                full += max(info["areas"]) == int(info["union"].sum())
                complementary += max(info["areas"]) < int(info["union"].sum())
                points = set(map(tuple, np.argwhere(info["union"])))
                todo, visited = [min(points)], set()
                while todo:
                    r, c = todo.pop()
                    if (r, c) in visited:
                        continue
                    visited.add((r, c))
                    todo.extend(p for p in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
                                if p in points and p not in visited)
                disconnected += len(visited) != len(points)
    assert len(geometries) >= 50 and len(stencils) >= 100
    assert len(area_vectors) >= 80 and len(role_palettes) >= 150
    assert bgs == set(range(10)) and source_counts == {3, 4, 5, 6}
    assert len(smallest) >= 8 and max(smallest) >= 10
    assert corners >= 10 and disconnected >= 3 and full >= 42 and complementary >= 84
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (10, 5),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-built overlap witness: area-2, area-3, area-5 sources. The middle
    # source lacks a, so it borrows that color from the smallest donor.
    masks = np.zeros((3, 3, 3), dtype=bool)
    for i, cells in enumerate(((0, 1), (1, 3, 4), (0, 1, 3, 4, 8))):
        masks[i].flat[list(cells)] = True
    witness = _render(masks, [7, 2, 8], (1, 3, 3, 3), 0, 6)
    actual = solve(witness["input"])
    assert np.array_equal(actual, witness["output"])
    assert actual[1, 5] == 7 and actual[1, 6] == 2  # borrowed a; original b
    assert actual[2, 2] == 2 and actual[3, 3] == 8  # Q owns d; R owns e
    # Real outputs must be sensitive to damage in both repair and preserved cells.
    original = puzzles[0]
    info = _parse(original["train"][0]["input"])
    for damage in ("repair", "original", "frame"):
        broken = {s: [{k: np.array(v, copy=True) for k, v in pair.items()}
                      for pair in original[s]] for s in ("train", "test")}
        target = broken["train"][0]["output"]
        if damage == "frame":
            target[0, 0] = info["background"]
        else:
            for i, (a, b, x, y) in enumerate(info["bounds"]):
                cells = info["masks"][i] if damage == "original" else info["union"] & ~info["masks"][i]
                if cells.any():
                    r, c = np.argwhere(cells)[0]
                    target[a + r, x + c] = info["background"]
                    break
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator missed {damage} corruption")


if __name__ == "__main__":
    self_test()
    print("self_test passed across 42 seeds and all legal count combinations")
