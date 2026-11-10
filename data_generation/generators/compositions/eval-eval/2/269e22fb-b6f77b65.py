"""Boundary-cut scaffolds: register a silhouette crop, cut whole rods, settle.

Only the visible input is used by solve. All random choices are made through
the local default_rng in generate_puzzle. No parent code or stored bitmap is used.
"""
from __future__ import annotations

import numpy as np


# A rod is (orientation, top row, left column, length, literal color).
def _cells(rod):
    orientation, row, col, length, _ = rod
    return frozenset((row + k * (orientation == "V"),
                      col + k * (orientation == "H")) for k in range(length))


def _translate(rod, distance):
    return (rod[0], rod[1] + distance, *rod[2:])


def _dihedral(array, index):
    """Indices 0..3 are rotations; 4..7 are reflected rotations."""
    source = np.fliplr(array) if index >= 4 else array
    return np.rot90(source, index % 4).copy()


def _settle(rods, height):
    """Simultaneously move the greatest unblocked set one cell downward."""
    result = list(rods)
    for _ in range(height * max(1, len(rods)) + 1):
        occupied = [_cells(p) for p in result]
        targets = [frozenset((r + 1, c) for r, c in cells) for cells in occupied]
        falling = {i for i, cells in enumerate(targets)
                   if all(r < height for r, c in cells)}
        while falling:
            blocked = {i for i in falling if any(
                targets[i] & occupied[j] for j in range(len(result)) if j not in falling)}
            if not blocked:
                break
            falling -= blocked
        if not falling:
            return result
        result = [_translate(p, 1) if i in falling else p for i, p in enumerate(result)]
    raise AssertionError("Gravity failed to converge")


def _draw_scene(rods, shape, background):
    result = np.full(shape, background, dtype=np.int8)
    occupied = set()
    for rod in rods:
        footprint = _cells(rod)
        assert not occupied & footprint, "Rods must not overlap"
        occupied.update(footprint)
        for r, c in footprint:
            assert 0 <= r < shape[0] and 0 <= c < shape[1]
            result[r, c] = rod[4]
    return result


def _silhouette(rods, shape):
    mask = np.zeros(shape, dtype=np.int8)
    for rod in rods:
        for cell in _cells(rod):
            mask[cell] = 1
    return mask


def _parse_rods(scene, background):
    remaining = {(r, c) for r, c in zip(*np.nonzero(scene != background))}
    rods = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = int(scene[start])
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt in remaining and int(scene[nxt]) == color:
                    remaining.remove(nxt)
                    component.add(nxt)
                    todo.append(nxt)
        rows = {r for r, c in component}
        cols = {c for r, c in component}
        if len(component) < 2 or (len(rows) != 1 and len(cols) != 1):
            raise ValueError("Every monochrome component must be a straight rod of length >=2")
        rod = ("H" if len(rows) == 1 else "V", min(rows), min(cols), len(component), color)
        if _cells(rod) != component:
            raise ValueError("Rod is not a contiguous straight segment")
        rods.append(rod)
    if not rods:
        raise ValueError("A scaffold is required")
    return rods


def _grid(value):
    result = np.asarray(value)
    if (result.ndim != 2 or not all(1 <= n <= 30 for n in result.shape)
            or not np.issubdtype(result.dtype, np.integer)
            or not np.all((0 <= result) & (result <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return result.astype(np.int8, copy=True)


def _read_input(value):
    grid = _grid(value)
    background = int(grid[0, 0])
    if not np.all(grid[0] == background):
        raise ValueError("The first row must encode the background")
    separators = [r for r in range(1, grid.shape[0])
                  if grid[r, 0] != background and np.all(grid[r] == grid[r, 0])]
    if not separators:
        raise ValueError("Missing full-width separator")
    divider = separators[0]
    marker = int(grid[divider, 0])
    header = grid[:divider]
    rr, cc = np.nonzero(header == marker)
    if not len(rr):
        raise ValueError("Missing crop frame")
    top, bottom, left, right = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    if top != 1 or bottom != divider - 2 or left < 1 or right >= grid.shape[1] - 1:
        raise ValueError("Frame must have a blank margin and a blank row above the divider")
    if bottom - top < 2 or right - left < 2:
        raise ValueError("Empty frame")
    crop = grid[top + 1:bottom, left + 1:right].copy()
    if len(np.unique(crop)) != 2 or marker in crop:
        raise ValueError("The crop must have exactly two colors distinct from its frame")
    expected = np.full(header.shape, background, dtype=np.int8)
    expected[top:bottom + 1, left] = marker
    expected[top:bottom + 1, right] = marker
    expected[top, left:right + 1] = marker
    expected[bottom, left:right + 1] = marker
    expected[top + 1:bottom, left + 1:right] = crop
    if not np.array_equal(expected, header):
        raise ValueError("Unexpected content in the control band")
    scene = grid[divider + 1:].copy()
    if not scene.size or np.any(scene == marker):
        raise ValueError("The divider color is reserved for the frame and divider")
    rods = _parse_rods(scene, background)
    return grid, background, marker, divider, crop, scene, rods


def _matches(mask, binary_crop, poses=range(8)):
    matches = []
    for pose in poses:
        patch = _dihedral(binary_crop, pose)
        h, w = patch.shape
        if h > mask.shape[0] or w > mask.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(mask, (h, w))
        for exchange in (0, 1):
            candidate = patch if exchange == 0 else 1 - patch
            rr, cc = np.nonzero(np.all(windows == candidate, axis=(-2, -1)))
            matches.extend((pose, exchange, int(r), int(c), h, w) for r, c in zip(rr, cc))
    return matches


def _locate(scene, background, crop):
    # Sorting colors introduces no semantic convention: both role assignments
    # are tried, including the possibility that the crop background is bright.
    binary = (crop == np.unique(crop)[1]).astype(np.int8)
    matches = _matches((scene != background).astype(np.int8), binary)
    if len(matches) != 1:
        raise ValueError("Crop must uniquely determine pose, color-role map, and location")
    return matches[0]


def _classify(rods, window):
    r, c, h, w = window
    inside, straddling, outside = [], [], []
    for i, rod in enumerate(rods):
        footprint = _cells(rod)
        contained = sum(r <= y < r + h and c <= x < c + w for y, x in footprint)
        (outside if contained == 0 else inside if contained == len(footprint)
         else straddling).append(i)
    return inside, straddling, outside


def _analysis(value):
    grid, bg, marker, divider, crop, scene, rods = _read_input(value)
    pose, exchange, r, c, h, w = _locate(scene, bg, crop)
    inside, cut, outside = _classify(rods, (r, c, h, w))
    kept_ids = [i for i in range(len(rods)) if i not in cut]
    kept = [rods[i] for i in kept_ids]
    settled = _settle(kept, scene.shape[0])
    return dict(grid=grid, bg=bg, marker=marker, divider=divider, crop=crop,
                scene=scene, rods=rods, pose=pose, exchange=exchange,
                window=(r, c, h, w), inside=inside, cut=cut, outside=outside,
                kept_ids=kept_ids, kept=kept, settled=settled)


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _analysis(input_grid)
    result = data["grid"].copy()
    result[data["divider"] + 1:] = _draw_scene(
        data["settled"], data["scene"].shape, data["bg"])
    return result


def _support_edges(rods):
    footprints = [_cells(p) for p in rods]
    return [(i, j) for i, p in enumerate(rods) for j in range(len(rods))
            if i != j and _cells(_translate(p, 1)) & footprints[j]]


def _depth(rods):
    edges = _support_edges(rods)
    def visit(i, path):
        if i in path:
            raise AssertionError("Cyclic support graph")
        return 1 + max((visit(j, path | {i}) for a, j in edges if a == i), default=0)
    return max((visit(i, set()) for i in range(len(rods))), default=0)


def _properties(data):
    distances = [q[1] - p[1] for p, q in zip(data["kept"], data["settled"])]
    initial_edges = _support_edges(data["rods"])
    final_edges = _support_edges(data["settled"])
    cut, kept_ids = set(data["cut"]), set(data["kept_ids"])
    redundant_cut = any(i in kept_ids and j in cut
                        and sum(a == i for a, b in initial_edges) >= 2
                        for i, j in initial_edges)
    landing = any(distances[i] > 0 and data["settled"][i][0] == "H"
                  for i, j in final_edges)
    repeated_cut_color = bool({data["rods"][i][4] for i in cut}
                              & {data["rods"][i][4] for i in kept_ids})
    contact_separation = any(distances[i] != distances[j]
                             and _touch(_cells(data["kept"][i]), _cells(data["kept"][j]))
                             for i in range(len(distances)) for j in range(i))
    return dict(distances=distances, moved=sum(d > 0 for d in distances),
                stationary=sum(d == 0 for d in distances), landing=landing,
                redundant_cut=redundant_cut, repeated_cut_color=repeated_cut_color,
                contact_separation=contact_separation, depth=_depth(data["rods"]))


def _drop_into(rods, height, orientation, col, length):
    candidate = (orientation, 0, col, length, 0)
    occupied = set().union(*(_cells(p) for p in rods)) if rods else set()
    for row in range(height + 1):
        footprint = _cells(_translate(candidate, row))
        if any(r >= height for r, c in footprint) or footprint & occupied:
            if row == 0:
                return False
            rods.append(_translate(candidate, row - 1))
            return True
    raise AssertionError("No floor found")


def _geometry(rng, target, large=False, bridge=False, teaching=False):
    height = int(rng.integers(14 if large else 12, 20 if large else 19))
    width = int(rng.integers(18 if large else 13, 31 if large else 29))
    rods = []
    if bridge:
        span = int(rng.integers(5, 10))
        left = int(rng.integers(1, width - span + 1)) if teaching else int(rng.integers(0, width - span + 1))
        spaced = [(a, b, c) for a in range(left, left + span)
                  for b in range(a + 2, left + span) for c in range(b + 2, left + span)]
        xs = spaced[int(rng.integers(len(spaced)))]
        tall = int(rng.integers(5, min(10, height - 3)))
        short = int(rng.integers(2, tall - 1))
        rods = [("V", height - tall, xs[0], tall, 0),
                ("V", height - tall, xs[2], tall, 0),
                ("V", height - short, xs[1], short, 0),
                ("H", height - tall - 1, left, span, 0)]
        beam_row = height - tall - 1
        upper_length = int(rng.integers(2, min(4, beam_row)))
        upper_x = int(rng.integers(left, left + span))
        rods.append(("V", beam_row - upper_length, upper_x, upper_length, 0))
        cap_length = int(rng.integers(2, min(7, span + 1)))
        cap_left = max(0, min(width - cap_length, upper_x - int(rng.integers(cap_length))))
        rods.append(("H", beam_row - upper_length - 1, cap_left, cap_length, 0))
        if teaching:
            # A floor-supported neighbor touches the beam at its side without
            # supporting it from below. It stays fixed as the beam falls. The
            # binary crop hides their color seam: complete-rod reasoning is
            # essential even where an occupied run continues outside the crop.
            rods.append(("V", beam_row, left - 1, height - beam_row, 0))
    for _ in range(target * 25):
        if len(rods) == target:
            break
        orientation = "H" if rng.random() < .58 else "V"
        length = int(rng.integers(2, min(width, 15) + 1)) if orientation == "H" else int(rng.integers(2, min(height - 2, 10) + 1))
        col = int(rng.integers(width - length + 1)) if orientation == "H" else int(rng.integers(width))
        _drop_into(rods, height, orientation, col, length)
    if len(rods) != target:
        return None
    assert _settle(rods, height) == rods
    return (height, width), rods


def _window(rng, shape, rods, pose, effect=None, smaller=False):
    mask = _silhouette(rods, shape)
    height, width = shape
    footprints = [_cells(p) for p in rods]
    edges = _support_edges(rods)
    visited = set()
    for _ in range(850):
        if effect == "release":
            # Propose from the geometric constraints of the diagnostic command,
            # rather than hoping a uniform rectangle happens to contain the
            # full beam while excluding its adjacent neighbor.
            h = int(rng.integers(2, min(9, 25 - height) + 1))
            c = rods[3][2]
            w = int(rng.integers(rods[3][3], min(12, width - c, width - 4) + 1))
            r = int(rng.integers(max(0, rods[3][1] - h + 2), min(rods[3][1], height - h) + 1))
        else:
            h = int(rng.integers(2, min(height, 7 if smaller else 9) + 1))
            w = int(rng.integers(2, min(width, 7 if smaller else 12) + 1))
            r, c = int(rng.integers(height - h + 1)), int(rng.integers(width - w + 1))
        query_height = w if pose % 2 else h
        query_width = h if pose % 2 else w
        if query_height + height + 5 > 30 or query_width + 4 > width:
            continue
        window = (r, c, h, w)
        if window in visited:
            continue
        visited.add(window)
        inside, cut, outside = _classify(rods, window)
        if not inside or not cut or not outside or len(cut) > len(rods) - 2:
            continue
        if effect == "release" and (0 not in cut or 1 not in cut or 3 not in inside or 6 not in outside):
            continue
        kept_ids = [i for i in range(len(rods)) if i not in cut]
        kept = [rods[i] for i in kept_ids]
        settled = _settle(kept, height)
        distances = [q[1] - p[1] for p, q in zip(kept, settled)]
        if effect == "release":
            # The known teaching bridge must lose BOTH tall legs while its
            # complete main beam survives. The other anchor loses only one leg.
            # A complete beam must survive, fall, and land on another survivor.
            if not any(distances) or not any(d == 0 for d in distances):
                continue
            landings = _support_edges(settled)
            if not any(distances[i] > 0 and kept_ids[i] in inside and kept[i][0] == "H"
                       for i, j in landings):
                continue
        elif effect == "redundant":
            if any(distances):
                continue
            if not any(i in kept_ids and j in cut and sum(a == i for a, b in edges) >= 2
                       for i, j in edges):
                continue
        # Need at least one possible same-color retained/removed pair, so color
        # deletion is distinguishable without fusing input components.
        choices = [(i, j) for i in cut for j in kept_ids if not _touch(footprints[i], footprints[j])]
        if not choices:
            continue
        patch = mask[r:r + h, c:c + w]
        if np.unique(patch).size != 2 or len(_matches(mask, patch)) != 1:
            continue
        return window, patch, choices
    return None


def _touch(a, b):
    return bool(a & b or any((r + dr, c + dc) in b for r, c in a
                            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))))


def _color_rods(rng, rods, palette, choices):
    footprints = [_cells(p) for p in rods]
    adjacency = {i: {j for j in range(len(rods)) if i != j and _touch(footprints[i], footprints[j])}
                 for i in range(len(rods))}
    for _ in range(40):
        a, b = choices[int(rng.integers(len(choices)))]
        repeated = int(rng.choice(palette))
        assignments = {a: repeated, b: repeated}
        order = sorted((i for i in range(len(rods)) if i not in assignments),
                       key=lambda i: (-len(adjacency[i]), i))
        for i in order:
            forbidden = {assignments[j] for j in adjacency[i] if j in assignments}
            allowed = [c for c in palette if c not in forbidden]
            if not allowed:
                break
            assignments[i] = int(rng.choice(allowed))
        if len(assignments) == len(rods):
            return [(*p[:4], assignments[i]) for i, p in enumerate(rods)]
    return None


def _package(rng, shape, rods, window, patch, pose, query_colors, exchange, background, marker):
    query = _dihedral(patch, pose)
    if exchange:
        query = 1 - query
    query = np.where(query == 0, query_colors[0], query_colors[1]).astype(np.int8)
    qh, qw = query.shape
    width = max(shape[1], qw + 4)
    # Scene width already exceeds every query width under either orientation.
    assert width == shape[1]
    divider = qh + 4
    grid = np.full((divider + 1 + shape[0], width), background, dtype=np.int8)
    left = int(rng.integers(1, width - qw - 2))
    grid[1, left:left + qw + 2] = marker
    grid[qh + 2, left:left + qw + 2] = marker
    grid[1:qh + 3, left] = marker
    grid[1:qh + 3, left + qw + 1] = marker
    grid[2:qh + 2, left + 1:left + qw + 1] = query
    grid[divider] = marker
    grid[divider + 1:] = _draw_scene(rods, shape, background)
    cut = set(_classify(rods, window)[1])
    survivors = [p for i, p in enumerate(rods) if i not in cut]
    output = grid.copy()
    output[divider + 1:] = _draw_scene(_settle(survivors, shape[0]), shape, background)
    # Ground truth uses the constructed rods/window. Check that visible input
    # reconstruction recovers them, rather than making labels by calling solve.
    decoded = _analysis(grid)
    assert set(decoded["rods"]) == set(rods)
    assert decoded["window"] == window
    assert np.array_equal(solve(grid), output)
    return {"input": grid, "output": output}


def _check_count(value, minimum, maximum):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not minimum <= int(value) <= maximum):
        raise ValueError("Invalid example count")
    return int(value)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _check_count(num_train, 2, 10), _check_count(num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.permutation(10)))
    background, marker = colors[:2]
    palette = colors[2:2 + int(rng.integers(4, 8))]
    query_colors = tuple(int(x) for x in rng.choice([c for c in range(10) if c != marker], 2, replace=False))
    # The first two examples use the SAME scene and opposite crop role maps.
    # They contrast a release with deletion of one redundant load-bearing rod.
    anchors = None
    for _ in range(80):
        geometry = _geometry(rng, 7, bridge=True, teaching=True)
        if geometry is None:
            continue
        shape, rods = geometry
        release = _window(rng, shape, rods, 0, "release")
        redundant = _window(rng, shape, rods, 1, "redundant")
        if release is None or redundant is None:
            continue
        colored = None
        redundant_cut = set(_classify(rods, redundant[0])[1])
        for _ in range(20):
            attempt = _color_rods(rng, rods, palette, release[2])
            if attempt is not None and ({attempt[i][4] for i in redundant_cut}
                    & {attempt[i][4] for i in range(len(rods)) if i not in redundant_cut}):
                colored = attempt
                break
        if colored is None:
            continue
        anchors = [_package(rng, shape, colored, release[0], release[1], 0, query_colors, 0, background, marker),
                   _package(rng, shape, colored, redundant[0], redundant[1], 1, query_colors, 1, background, marker)]
        break
    if anchors is None:
        raise RuntimeError("Could not construct diagnostic demonstrations within the bound")
    puzzle = {"train": anchors, "test": []}
    seen = {_key(p["input"]) for p in anchors}
    # With >=3 demonstrations a reflection is also taught. Two demonstrations
    # teach C4 through a quarter turn; tests never require an untaught reflection.
    allowed_poses = list(range(8 if nt >= 3 else 4))
    for split, count in (("train", nt - 2), ("test", nq)):
        for j in range(count):
            pose = 4 if split == "train" and j == 0 else int(rng.choice(allowed_poses))
            accepted = None
            for _ in range(75):
                target = int(rng.integers(9, 14)) if split == "test" else int(rng.integers(5, 9))
                geometry = _geometry(rng, target, large=split == "test",
                                     bridge=target >= 6 and rng.random() < .55)
                if geometry is None:
                    continue
                shape, rods = geometry
                candidate = _window(rng, shape, rods, pose, smaller=split == "test")
                if candidate is None:
                    continue
                colored = _color_rods(rng, rods, palette, candidate[2])
                if colored is None:
                    continue
                # Query colors vary independently of the physical colors.
                pair_colors = tuple(int(x) for x in rng.choice([c for c in range(10) if c != marker], 2, replace=False))
                pair = _package(rng, shape, colored, candidate[0], candidate[1], pose, pair_colors,
                                int(rng.integers(2)), background, marker)
                if _key(pair["input"]) in seen:
                    continue
                accepted = pair
                break
            if accepted is None:
                raise RuntimeError("No distinctive scaffold/crop within the construction bound")
            seen.add(_key(accepted["input"]))
            puzzle[split].append(accepted)
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records, conventions = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "Duplicate input"
            seen.add(_key(inp))
            data = _analysis(inp)
            conventions.add((data["bg"], data["marker"]))
            assert inp.shape == out.shape
            assert np.array_equal(inp[:data["divider"] + 1], out[:data["divider"] + 1])
            assert _settle(data["rods"], data["scene"].shape[0]) == data["rods"], "Input must be stable"
            assert data["inside"] and data["cut"] and data["outside"]
            assert len(data["kept"]) >= 2
            assert _settle(data["settled"], data["scene"].shape[0]) == data["settled"]
            for p, q in zip(data["kept"], data["settled"]):
                assert p[0] == q[0] and p[2:] == q[2:] and q[1] >= p[1], "Rigidity/column invariance"
            expected = inp.copy()
            expected[data["divider"] + 1:] = _draw_scene(data["settled"], data["scene"].shape, data["bg"])
            assert np.array_equal(expected, out), "Incorrect boundary selection or settlement"
            # Visible per-color material is conserved exactly for the surviving
            # rods: there are no hidden intersections in this new family.
            for color in range(10):
                if color != data["bg"]:
                    assert np.sum(out[data["divider"] + 1:] == color) == sum(p[3] for p in data["kept"] if p[4] == color)
            props = _properties(data)
            assert props["repeated_cut_color"], "A global color-delete shortcut must differ"
            assert (9 <= len(data["rods"]) <= 13) if split == "test" else (5 <= len(data["rods"]) <= 8)
            records[split].append((data, props))
    assert len(conventions) == 1
    release = [(d, f) for d, f in records["train"]
               if f["moved"] and f["stationary"] and f["landing"] and f["contact_separation"]]
    harmless = [(d, f) for d, f in records["train"] if not f["moved"] and f["redundant_cut"]]
    assert release and harmless, "Demonstrate both actual gravity and redundant support"
    assert any(np.array_equal(a["scene"], b["scene"])
               and np.array_equal(np.unique(a["crop"]), np.unique(b["crop"]))
               and a["exchange"] != b["exchange"]
               for a, _ in release for b, _ in harmless), "Matched scene counterfactuals must teach color-role exchange"
    # Check an actual common horizontal survivor whose redundant supports are
    # jointly removed in the release and only partly removed in the harmless
    # command. Do not count an unrelated cap fall as the teaching contrast.
    contrast = False
    for a, _ in release:
        for b, _ in harmless:
            if not np.array_equal(a["scene"], b["scene"]):
                continue
            edges = _support_edges(a["rods"])
            for i in a["inside"]:
                supports = {j for k, j in edges if k == i}
                if (a["rods"][i][0] == "H" and len(supports) >= 2
                        and supports <= set(a["cut"]) and i in b["kept_ids"]
                        and supports & set(b["cut"]) and supports - set(b["cut"])):
                    contrast = True
    assert contrast, "Teach single-support redundancy versus joint support release"
    assert any(d["pose"] % 2 for d, f in records["train"]), "Teach axis exchange"
    if nt >= 3:
        assert any(d["pose"] >= 4 for d, f in records["train"]), "Teach reflection"
    else:
        assert all(d["pose"] < 4 for d, f in records["test"])


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _mechanics_test():
    # Two equal legs: deleting one cannot move the complete bridge. Deleting
    # both releases it three cells onto the shorter middle post.
    rods = [("V", 4, 1, 6, 1), ("V", 4, 7, 6, 2),
            ("V", 7, 4, 3, 3), ("H", 3, 0, 9, 4)]
    assert _settle(rods, 10) == rods
    assert _settle(rods[1:], 10) == rods[1:]
    assert _settle(rods[2:], 10) == [rods[2], _translate(rods[3], 3)]
    # Different stacks fall unequal distances. A pixel-gravity calculation
    # would destroy these horizontal bars rather than preserve their spans.
    survivors = [("H", 4, 0, 3, 2), ("H", 6, 5, 3, 3)]
    assert _settle(survivors, 9) == [_translate(survivors[0], 4), _translate(survivors[1], 2)]
    # Equal occupancy can have different colored seams and different cuts:
    # a long rod straddles, while its complete lower replacement segment survives.
    window = (4, 0, 3, 3)
    long = [("V", 2, 1, 5, 1)]
    split = [("V", 2, 1, 2, 1), ("V", 4, 1, 3, 2)]
    assert np.array_equal(_silhouette(long, (8, 4)), _silhouette(split, (8, 4)))
    assert _classify(long, window) == ([], [0], [])
    assert _classify(split, window) == ([1], [], [0])
    # Inclusion of an endpoint is still complete; a stick need not have a blank
    # margin to be protected inside a window.
    assert _classify([("H", 4, 0, 3, 1)], window) == ([0], [], [])


def self_test():
    """Bounded semantic diagnostics and actual-grid diversity over 42 seeds."""
    _mechanics_test()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    masks, shapes, windows, poses, falls, depths, counts, cut_counts, backgrounds = set(), set(), set(), set(), set(), set(), set(), set(), set()
    multilevel_falls, repeat_counterfactuals = 0, 0
    for puzzle in puzzles:
        scenes = []
        identity_fails = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _analysis(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                props = _properties(data)
                mask = (data["scene"] != data["bg"]).astype(np.int8)
                binary_crop = (data["crop"] == np.unique(data["crop"])[1]).astype(np.int8)
                if split == "train" and not _matches(mask, binary_crop, poses=[0]):
                    identity_fails = True
                masks.add(min(_key(_dihedral(mask, k)) for k in range(8)))
                shapes.add(data["scene"].shape)
                windows.add(data["window"])
                poses.add(data["pose"])
                falls.update(props["distances"])
                depths.add(props["depth"])
                counts.add(len(data["rods"]))
                cut_counts.add(len(data["cut"]))
                backgrounds.add(data["bg"])
                multilevel_falls += len(set(props["distances"]) - {0}) >= 2
                if split == "train":
                    scenes.append(_key(data["scene"]))
                # Both parents matter: no cut differs because removed rods
                # have extensions outside the window; no gravity differs on
                # every release. Keeping all rods touching the window also
                # differs, because every sample preserves an entire inner rod.
                assert data["cut"] and data["inside"] and data["outside"]
                # Ablate spatial membership: deleting all of the identified
                # colors also deletes an intact survivor in every example.
                selected_colors = {data["rods"][i][4] for i in data["cut"]}
                color_survivors = [p for p in data["rods"] if p[4] not in selected_colors]
                assert not np.array_equal(_draw_scene(_settle(color_survivors, data["scene"].shape[0]),
                                                     data["scene"].shape, data["bg"]),
                                          np.asarray(pair["output"])[data["divider"] + 1:])
                # Ablate whole-piece boundary reasoning: removing every
                # intersecting rod wrongly removes protected inner material.
                only_outside = [data["rods"][i] for i in data["outside"]]
                assert not np.array_equal(_draw_scene(_settle(only_outside, data["scene"].shape[0]),
                                                     data["scene"].shape, data["bg"]),
                                          np.asarray(pair["output"])[data["divider"] + 1:])
                no_fall = data["grid"].copy()
                no_fall[data["divider"] + 1:] = _draw_scene(data["kept"], data["scene"].shape, data["bg"])
                if props["moved"]:
                    assert not np.array_equal(no_fall, pair["output"])
        repeat_counterfactuals += len(set(scenes)) < len(scenes)
        assert identity_fails, "At least one demonstration must require crop reorientation"
        assert any(_properties(_analysis(p["input"]))["contact_separation"] for p in puzzle["train"]), "Bonded-component gravity must fail in training"
    assert len(masks) >= 140, "Silhouette diversity after dihedral normalization"
    assert len(shapes) >= 45 and len(windows) >= 140
    assert poses == set(range(8))
    assert len(counts) >= 8 and len(cut_counts) >= 5 and len(depths) >= 5
    assert len(falls) >= 7 and multilevel_falls >= 5
    assert len(backgrounds) >= 8 and repeat_counterfactuals == 42
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(123456, nt, nq)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Validators must reject label edits, independently of stored metadata.
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    out = bad["test"][0]["output"]
    out[-1, -1] = (int(out[-1, -1]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupted test output was accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal count combinations)")
