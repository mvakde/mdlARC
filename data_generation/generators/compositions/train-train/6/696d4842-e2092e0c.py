"""Tethered stamp: a measured opposite-end prefix is imprinted at an exact twin.

Only NumPy and the standard library are used. All solving roles are input-visible.
"""
from __future__ import annotations

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    array = np.asarray(value)
    assert array.ndim == 2 and all(1 <= n <= 30 for n in array.shape)
    assert np.issubdtype(array.dtype, np.integer)
    assert np.all((0 <= array) & (array <= 9))
    return array.astype(np.int8, copy=False)


def _neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))


def _inside(cell, shape):
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _bends(route):
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(route, route[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _l_route(h, w, attachment):
    # Canonical source is top-left. The route always starts at its FREE end.
    if attachment == "top":
        return [(h, c) for c in range(w + 1)] + [(r, w) for r in range(h - 1, -1, -1)]
    return [(r, w) for r in range(h + 1)] + [(h, c) for c in range(w - 1, -1, -1)]


def _source_candidates(grid, background):
    candidates = []
    for rotation in range(4):
        view = np.rot90(grid, rotation)
        for h in range(2, 5):
            for w in range(2, 5):
                if h >= view.shape[0] or w >= view.shape[1]:
                    continue
                marker = int(view[h, w])
                template = view[:h, :w]
                if marker == background or np.any(template == marker) or np.any(template == background):
                    continue
                if np.all(view[:h, w] == marker) and np.all(view[h, :w + 1] == marker):
                    candidates.append((rotation, h, w, marker))
    return candidates


def _matches(grid, template):
    windows = np.lib.stride_tricks.sliding_window_view(grid, template.shape)
    return [tuple(map(int, pos)) for pos in np.argwhere(np.all(windows == template, axis=(-2, -1)))]


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        stack = [start]
        while stack:
            for cell in _neighbors(stack.pop()):
                if cell in remaining:
                    remaining.remove(cell)
                    component.add(cell)
                    stack.append(cell)
        result.append(component)
    return result


def _analyze(input_grid):
    """Parse the whole family without access to a seed or construction metadata."""
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > grid.size / 2, "background must have a strict majority"
    sources = _source_candidates(grid, background)
    assert len(sources) == 1, "source L must be unique"
    rotation, h, w, marker = sources[0]
    view = np.rot90(grid, rotation)
    template = view[:h, :w]
    content = set(map(int, np.unique(template)))
    assert 2 <= len(content) <= 4
    singleton_colors = [color for color, count in enumerate(counts)
                        if count == 1 and color not in content and color not in (background, marker)]
    assert len(singleton_colors) == 1, "one singleton beacon color is required"
    ink = singleton_colors[0]
    beacon = tuple(map(int, np.argwhere(view == ink)[0]))
    assert set(map(int, np.unique(view))) == content | {background, marker, ink}

    cells = {tuple(map(int, cell)) for cell in np.argwhere(view == marker)}
    degree = {cell: sum(n in cells for n in _neighbors(cell)) for cell in cells}
    ends = [cell for cell in sorted(cells) if degree[cell] == 1]
    assert len(ends) == 2 and all(d in (1, 2) for d in degree.values())
    free = [cell for cell in ends if cell in ((h, 0), (0, w))]
    assert len(free) == 1
    attachment = "top" if free[0] == (h, 0) else "left"
    route, previous = [free[0]], None
    while len(route) < len(cells):
        choices = [n for n in _neighbors(route[-1]) if n in cells and n != previous]
        assert len(choices) == 1, "the cable must be one connected simple path"
        previous, current = route[-1], choices[0]
        assert current not in route
        route.append(current)
    assert route[-1] in ends and route[-1] != route[0]
    l_route = _l_route(h, w, attachment)
    assert route[:len(l_route)] == l_route and len(route) > len(l_route)
    tip = route[-1]
    outward = (tip[0] - route[-2][0], tip[1] - route[-2][1])
    delta = (beacon[0] - tip[0], beacon[1] - tip[1])
    distance = delta[0] * outward[0] + delta[1] * outward[1]
    assert distance >= 2 and delta == (distance * outward[0], distance * outward[1])
    gap = [(tip[0] + i * outward[0], tip[1] + i * outward[1]) for i in range(1, distance)]
    g = len(gap)
    assert 1 <= g <= len(l_route)
    assert all(view[cell] == background for cell in gap)
    assert all(view[n] == background for n in _neighbors(beacon) if _inside(n, view.shape))

    matches = _matches(view, template)
    assert len(matches) == 2 and (0, 0) in matches, "exactly one duplicate is required"
    target = next(cell for cell in matches if cell != (0, 0))
    r, c = target
    assert 1 <= r < view.shape[0] - h and 1 <= c < view.shape[1] - w
    frame = ({(r - 1, col) for col in range(c - 1, c + w + 1)}
             | {(r + h, col) for col in range(c - 1, c + w + 1)}
             | {(row, c - 1) for row in range(r, r + h)}
             | {(row, c + w) for row in range(r, r + h)})
    assert all(view[cell] == background for cell in frame)
    assert frame.isdisjoint(gap)
    imprint = [(a + r, b + c) for a, b in route[:g]]
    assert set(imprint) <= frame and len(set(imprint)) == g
    assert not set(imprint) & set(gap)

    # Full tiles are parsed independently of cable connectivity and exact matching.
    tile_components = _components(tuple(map(int, cell)) for cell in np.argwhere(np.isin(view, list(content))))
    tiles, decoys, histogram_decoys = [], [], 0
    for component in tile_components:
        row = min(a for a, _ in component)
        col = min(b for _, b in component)
        assert component == {(row + a, col + b) for a in range(h) for b in range(w)}
        tiles.append((row, col))
        if (row, col) in ((0, 0), target):
            continue
        patch = view[row:row + h, col:col + w]
        mismatch = int(np.count_nonzero(patch != template))
        assert 1 <= mismatch <= 2
        decoys.append((row, col))
        histogram_decoys += np.array_equal(np.sort(patch.ravel()), np.sort(template.ravel()))
        # Non-source tiles have a blank moat separating them from cable and others.
        for a in range(row - 1, row + h + 1):
            for b in range(col - 1, col + w + 1):
                if _inside((a, b), view.shape) and (a, b) not in component:
                    assert view[a, b] == background
    assert 1 <= len(decoys) <= 3 and (0, 0) in tiles and target in tiles
    assert len(tiles) == len(decoys) + 2
    return dict(grid=view, rotation=rotation, background=background, marker=marker, ink=ink,
                template=template, shape=(h, w), target=target, beacon=beacon,
                route=route, l_route=l_route, attachment=attachment, gap=gap, g=g,
                imprint=imprint, frame=frame, decoys=decoys, histogram_decoys=histogram_decoys,
                crossing=_bends(route[:g]) > 0, tail_bends=_bends(route[len(l_route) - 1:]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    result = info["grid"].copy()
    for cell in info["gap"]:
        result[cell] = info["marker"]
    for cell in info["imprint"]:
        result[cell] = info["ink"]
    return np.rot90(result, -info["rotation"]).copy()


def _make_geometry(rng, shape, h, w, attachment, bends, g):
    route = _l_route(h, w, attachment)
    cells = set(route)
    source = {(r, c) for r in range(h) for c in range(w)}
    direction = 1 if attachment == "top" else 2
    directions = [direction]
    for index in range(bends):
        if index == 0:
            direction = 2 if attachment == "top" else 1
        else:
            direction = (direction + int(rng.choice((-1, 1)))) % 4
        directions.append(direction)
    for index, direction in enumerate(directions):
        length = int(rng.integers(2 if index == 0 else 1, 8))
        dr, dc = DIRECTIONS[direction]
        for _ in range(length):
            new = (route[-1][0] + dr, route[-1][1] + dc)
            if not _inside(new, shape) or new in source or new in cells:
                return None
            if sum(n in cells for n in _neighbors(new)) != 1:
                return None
            # Except at the preexisting L, keep the cable away from tile content.
            if any(n in source for n in _neighbors(new)):
                return None
            route.append(new)
            cells.add(new)
    dr, dc = DIRECTIONS[directions[-1]]
    tip = route[-1]
    gap = [(tip[0] + i * dr, tip[1] + i * dc) for i in range(1, g + 1)]
    beacon = (tip[0] + (g + 1) * dr, tip[1] + (g + 1) * dc)
    if not _inside(beacon, shape):
        return None
    for index, cell in enumerate(gap + [beacon]):
        if cell in cells or cell in source:
            return None
        if any(n in source for n in _neighbors(cell)):
            return None
        adjacent = {n for n in _neighbors(cell) if n in cells}
        if adjacent != ({tip} if index == 0 else set()):
            return None
    return route, gap, beacon


def _tile_sites(rng, shape, h, w, blocked, number):
    occupied = np.zeros(shape, dtype=bool)
    for cell in blocked:
        occupied[cell] = True
    sites = []
    for _ in range(number):
        choices = [(r, c) for r in range(1, shape[0] - h)
                   for c in range(1, shape[1] - w)
                   if not np.any(occupied[r - 1:r + h + 1, c - 1:c + w + 1])]
        if not choices:
            return None
        edges = [p for p in choices if p[0] in (1, shape[0] - h - 1)
                 or p[1] in (1, shape[1] - w - 1)]
        pool = edges if edges and rng.random() < .25 else choices
        r, c = pool[int(rng.integers(len(pool)))]
        sites.append((r, c))
        occupied[r - 1:r + h + 1, c - 1:c + w + 1] = True
    return sites


def _colors(rng, previous=None):
    roles = []
    for index in range(3):
        choices = [c for c in range(10) if c not in roles
                   and (previous is None or c != previous[index])]
        roles.append(int(rng.choice(choices)))
    content_count = int(rng.integers(2, 5))
    content = rng.choice([c for c in range(10) if c not in roles], size=content_count, replace=False)
    return tuple(roles), content


def _decoy(rng, template, content, histogram):
    patch = template.copy()
    if histogram:
        first = int(rng.integers(patch.size))
        different = np.flatnonzero(patch.ravel() != patch.flat[first])
        second = int(rng.choice(different))
        patch.flat[first], patch.flat[second] = int(patch.flat[second]), int(patch.flat[first])
    else:
        number = int(rng.integers(1, 3))
        for index in rng.choice(patch.size, size=number, replace=False):
            patch.flat[index] = int(rng.choice(content[content != patch.flat[index]]))
    return patch


def _generate_pair(rng, plan, previous_colors=None):
    h, w = plan["shape"]
    attachment = plan["attachment"]
    l_route = _l_route(h, w, attachment)
    first_leg = w + 1 if attachment == "top" else h + 1
    roles, content = _colors(rng, previous_colors)
    background, marker, ink = roles
    for _ in range(100):
        template = rng.choice(content, size=(h, w)).astype(np.int8)
        # Every selected content color appears; at least two permit a swap decoy.
        for index, color in zip(rng.permutation(h * w), content):
            template.flat[index] = color
        # A content-colored nested L could otherwise impersonate a smaller source.
        # Reject that motif before trying hundreds of unrelated cable layouts.
        probe = np.full((12, 12), background, dtype=np.int8)
        probe[:h, :w] = template
        for cell in l_route:
            probe[cell] = marker
        if _source_candidates(probe, background) == [(0, h, w, marker)]:
            break
    else:
        raise RuntimeError("could not construct an unambiguous source motif")
    for _ in range(450):
        size = (int(rng.integers(18, 31)), int(rng.integers(18, 31)))
        if rng.random() < .12:
            size = (int(rng.integers(14, 22)), int(rng.integers(22, 31)))
        if rng.random() < .15:
            size = (size[1], size[1])
        if plan["crossing"]:
            g = int(rng.integers(first_leg + 1, len(l_route) + 1))
        else:
            g = int(rng.integers(1, first_leg))
        bends = int(rng.integers(3, 5)) if plan["test"] else int(rng.integers(0, 3))
        geometry = _make_geometry(rng, size, h, w, attachment, bends, g)
        if geometry is None:
            continue
        route, gap, beacon = geometry
        decoy_count = int(rng.integers(2, 4)) if plan["test"] else int(rng.integers(1, 3))
        blocked = set(route) | set(gap) | {beacon} | {(r, c) for r in range(h) for c in range(w)}
        sites = _tile_sites(rng, size, h, w, blocked, decoy_count + 1)
        if sites is None:
            continue
        eligible = list(range(len(sites)))
        if plan["anti_rank"]:
            eligible = [i for i, site in enumerate(sites)
                        if any(sum(other) < sum(site) for other in sites)
                        and any(other < site for other in sites)]
            if not eligible:
                continue
        target_index = int(rng.choice(eligible))
        target = sites[target_index]
        grid = np.full(size, background, dtype=np.int8)
        grid[:h, :w] = template
        for cell in route:
            grid[cell] = marker
        grid[beacon] = ink
        decoy_index = 0
        for index, (r, c) in enumerate(sites):
            if index == target_index:
                patch = template
            else:
                patch = _decoy(rng, template, content, decoy_index == 0 or rng.random() < .4)
                decoy_index += 1
            grid[r:r + h, c:c + w] = patch
        output = grid.copy()
        for cell in gap:
            output[cell] = marker
        for r, c in route[:g]:
            output[r + target[0], c + target[1]] = ink
        rotation = plan["rotation"]
        grid = np.rot90(grid, rotation).copy()
        output = np.rot90(output, rotation).copy()
        try:
            info = _analyze(grid)
        except AssertionError:
            continue
        assert info["g"] == g and info["crossing"] == plan["crossing"]
        assert info["tail_bends"] == bends
        return {"input": grid, "output": output}, roles
    raise RuntimeError("could not construct a separated tethered stamp")


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= int(num_train) <= 10:
        raise ValueError("num_train must be an integer from 2 to 10")
    if not _integer(num_test) or not 1 <= int(num_test) <= 4:
        raise ValueError("num_test must be an integer from 1 to 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    shapes = [(h, w) for h in range(2, 5) for w in range(2, 5)]
    shape_indices = rng.choice(len(shapes), size=2, replace=False)
    attachment = ("top", "left")[int(rng.integers(2))]
    rotations = list(map(int, rng.choice(4, size=2, replace=False)))
    train_plans = [dict(shape=shapes[int(shape_indices[i])],
                        attachment=attachment if i == 0 else ("left" if attachment == "top" else "top"),
                        rotation=rotations[i], crossing=bool(i), test=False, anti_rank=bool(i))
                   for i in range(2)]
    for _ in range(int(num_train) - 2):
        train_plans.append(dict(shape=shapes[int(rng.integers(len(shapes)))],
                                attachment=("top", "left")[int(rng.integers(2))],
                                rotation=int(rng.integers(4)), crossing=bool(rng.integers(2)),
                                test=False, anti_rank=False))
    # The first two jointly provide the essential contrast, in variable display order.
    if rng.random() < .5:
        train_plans[0], train_plans[1] = train_plans[1], train_plans[0]
    test_plans = [dict(shape=train_plans[int(rng.integers(len(train_plans)))]["shape"],
                       attachment=("top", "left")[int(rng.integers(2))],
                       rotation=train_plans[int(rng.integers(len(train_plans)))]["rotation"],
                       crossing=True, test=True, anti_rank=False)
                  for _ in range(int(num_test))]
    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = None
    for split, plans in (("train", train_plans), ("test", test_plans)):
        for index, plan in enumerate(plans):
            for _ in range(10):
                pair, roles = _generate_pair(rng, plan, first_roles if split == "train" and index == 1 else None)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
            if split == "train" and index == 0:
                first_roles = roles
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _analyze(inp)
            actual = np.rot90(out, info["rotation"])
            view = info["grid"]
            changed = {tuple(map(int, cell)) for cell in np.argwhere(actual != view)}
            assert changed == set(info["gap"]) | set(info["imprint"])
            assert len(changed) == 2 * info["g"]
            assert all(actual[cell] == info["marker"] for cell in info["gap"])
            assert all(actual[cell] == info["ink"] for cell in info["imprint"])
            assert all(actual[cell] == info["marker"] for cell in info["route"])
            h, w = info["shape"]
            r, c = info["target"]
            assert np.array_equal(actual[:h, :w], info["template"])
            assert np.array_equal(actual[r:r + h, c:c + w], info["template"])
            assert actual[info["beacon"]] == info["ink"]
            assert set(map(int, np.unique(inp))) == set(map(int, np.unique(out)))
            assert np.array_equal(solve(inp), out)
            assert info["histogram_decoys"] >= 1
            infos[split].append(info)
    first = infos["train"][:2]
    assert {info["crossing"] for info in first} == {False, True}
    assert {info["attachment"] for info in first} == {"top", "left"}
    assert len({info["rotation"] for info in first}) == 2
    assert len({info["shape"] for info in first}) == 2
    for role in ("background", "marker", "ink"):
        assert len({info[role] for info in first}) == 2
    assert all(info["tail_bends"] <= 2 and 1 <= len(info["decoys"]) <= 2 for info in infos["train"])
    crossing = next(info for info in first if info["crossing"])
    assert any(sum(decoy) < sum(crossing["target"]) for decoy in crossing["decoys"])
    assert any(decoy < crossing["target"] for decoy in crossing["decoys"])
    for info in infos["test"]:
        assert info["crossing"] and 3 <= info["tail_bends"] <= 4
        assert 2 <= len(info["decoys"]) <= 3
        assert info["shape"] in {t["shape"] for t in infos["train"]}
        assert info["rotation"] in {t["rotation"] for t in infos["train"]}


def self_test():
    """Bounded 42-seed diversity, ablation, count-contract, and replay checks."""
    shapes, tile_shapes, corners, backgrounds, markers, inks = (set() for _ in range(6))
    gaps, tail_bends, decoy_counts, attachments, structures, signatures = (set() for _ in range(6))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                tile_shapes.add(info["shape"])
                corners.add(info["rotation"])
                backgrounds.add(info["background"])
                markers.add(info["marker"])
                inks.add(info["ink"])
                gaps.add(info["g"])
                tail_bends.add(info["tail_bends"])
                decoy_counts.add(len(info["decoys"]))
                attachments.add(info["attachment"])
                structures.add((pair["input"].shape, info["shape"], tuple(info["route"]),
                                info["target"], info["g"], tuple(sorted(info["decoys"]))))
                # A omitted: full L instead of a measured prefix must fail on a short demo.
                if not info["crossing"]:
                    alternative = info["grid"].copy()
                    for cell in info["gap"]:
                        alternative[cell] = info["marker"]
                    for r, c in info["l_route"]:
                        alternative[r + info["target"][0], c + info["target"][1]] = info["ink"]
                    assert not np.array_equal(alternative, np.rot90(pair["output"], info["rotation"]))
                # B omitted: local opposite-end recoloring differs on every pair.
                local = info["grid"].copy()
                for cell in info["gap"]:
                    local[cell] = info["marker"]
                for cell in info["route"][:info["g"]]:
                    local[cell] = info["ink"]
                assert not np.array_equal(local, np.rot90(pair["output"], info["rotation"]))
                # Solver does not mutate its argument.
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"])
    assert len(signatures) == 42 and len(structures) == 210
    assert len(shapes) >= 60 and tile_shapes == {(h, w) for h in range(2, 5) for w in range(2, 5)}
    assert corners == {0, 1, 2, 3} and attachments == {"top", "left"}
    assert len(backgrounds) == len(markers) == len(inks) == 10
    assert {1, 2, 3, 4, 5, 6, 7, 8, 9} <= gaps
    assert tail_bends == {0, 1, 2, 3, 4} and decoy_counts == {1, 2, 3}
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(10000 + train * 10 + test, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for bad_seed in (-1, True, 1.5, None):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, both concept ablations, and all legal counts")
