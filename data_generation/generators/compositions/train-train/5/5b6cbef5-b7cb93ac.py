"""Self-similar jigsaw sockets: exact tile packing constrained at two scales.

Only the empty socket's dimensions are given.  Its occupied mask must be M x M
(Boolean Kronecker product), for some unknown rectangular Boolean motif M.
The unique largest loose tile cannot turn; all other tiles may quarter-turn.
All legal assemblies must agree on one colored output.  There is no cost or
color-based tie break.  No parent generator is imported.
"""

from functools import lru_cache
from itertools import combinations
from math import isqrt

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SHAPES = ((2, 3), (3, 2), (3, 3), (2, 4), (4, 2))


def _normalize(cells):
    cells = tuple(cells)
    top = min(r for r, c in cells)
    left = min(c for r, c in cells)
    return tuple(sorted((int(r - top), int(c - left)) for r, c in cells))


def _dims(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _rotate(shape, turns):
    for _ in range(turns % 4):
        shape = _normalize((c, -r) for r, c in shape)
    return shape


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        reached, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                neighbor = r + dr, c + dc
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    reached.add(neighbor)
                    stack.append(neighbor)
        result.append(tuple(sorted(reached)))
    return sorted(result, key=lambda cells: (-len(cells), cells))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid with axes 1..30 and colors 0..9")
    return grid.astype(np.int8, copy=False)


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    if not (np.all(grid[0] == bg) and np.all(grid[-1] == bg)
            and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)):
        raise ValueError("the background must form an uninterrupted outer border")
    records, sockets = [], []
    for cells in _components(map(tuple, np.argwhere(grid != bg))):
        colors = {int(grid[r, c]) for r, c in cells}
        if len(colors) != 1:
            raise ValueError("loose tiles and socket must be separated and monochromatic")
        color = colors.pop()
        shape = _normalize(cells)
        height, width = _dims(shape)
        ring = {(r, c) for r in range(height) for c in range(width)
                if r in (0, height - 1) or c in (0, width - 1)}
        inner = height - 2, width - 2
        roots = tuple(isqrt(max(0, n)) for n in inner)
        if (set(shape) == ring and roots in _SHAPES
                and tuple(n * n for n in roots) == inner):
            top, left = min(r for r, c in cells), min(c for r, c in cells)
            if not np.all(grid[top + 1:top + height - 1, left + 1:left + width - 1] == bg):
                raise ValueError("socket interior must be empty")
            sockets.append((color, roots))
        else:
            records.append({"color": color, "shape": shape})
    if len(sockets) != 1:
        raise ValueError("expected exactly one empty square-axis socket")
    marker, shape = sockets[0]
    if any(tile["color"] == marker for tile in records):
        raise ValueError("socket color cannot also label a tile")
    if not 3 <= len(records) <= 5:
        raise ValueError("expected three to five loose tiles")
    areas = [len(tile["shape"]) for tile in records]
    if areas.count(max(areas)) != 1:
        raise ValueError("expected one strictly largest orientation anchor")
    total = sum(areas)
    m = isqrt(total)
    if m * m != total or not 3 <= m < shape[0] * shape[1]:
        raise ValueError("tile area must be a nontrivial motif's squared cell count")
    return bg, marker, shape, records


@lru_cache(maxsize=32)
def _templates(h, w, count):
    """Pure geometry cache, indexed only by dimensions and area, not examples."""
    templates = []
    for indices in combinations(range(h * w), count):
        motif = tuple((i // w, i % w) for i in indices)
        support = tuple(sorted((r * h + a, c * w + b)
                               for r, c in motif for a, b in motif))
        bits = sum(1 << (r * w * w + c) for r, c in support)
        templates.append((bits, motif))
    return tuple(templates)


def _placements(shape, height, width, allow_rotation):
    result = []
    orientations = {_rotate(shape, k) for k in (range(4) if allow_rotation else (0,))}
    for oriented in sorted(orientations):
        sh, sw = _dims(oriented)
        for top in range(height - sh + 1):
            for left in range(width - sw + 1):
                cells = tuple((r + top) * width + c + left for r, c in oriented)
                result.append((sum(1 << p for p in cells), cells))
    return result


def _assemblies(tiles, shape, rotations="all", limit=2):
    """Find up to two distinct colored answers, over ALL possible motifs.

    Rotation modes 'none' and 'bars' are used only for counterfactual checks.
    Placement witnesses retain component identity even when colors repeat.
    """
    h, w = shape
    height, width = h * h, w * w
    total = sum(len(tile["shape"]) for tile in tiles)
    count = isqrt(total)
    if count * count != total:
        return [], 0
    anchor = max(range(len(tiles)), key=lambda i: len(tiles[i]["shape"]))
    options = []
    for i, tile in enumerate(tiles):
        bar = min(_dims(tile["shape"])) == 1
        allowed = i != anchor and (rotations == "all" or (rotations == "bars" and bar))
        options.append(_placements(tile["shape"], height, width, allowed))
    solutions = {}
    anchor_templates = 0
    for target, motif in _templates(h, w, count):
        anchor_options = [p for p in options[anchor] if p[0] & target == p[0]]
        if not anchor_options:
            continue
        anchor_templates += 1
        fitting = [([p for p in choices if p[0] & target == p[0]]
                    if i != anchor else anchor_options)
                   for i, choices in enumerate(options)]
        if any(not choices for choices in fitting):
            continue
        board = [-1] * (height * width)
        chosen = {}

        def search(remaining, occupied):
            if not remaining:
                key = tuple(board)
                solutions[key] = {"pixels": key, "motif": motif, "placements": dict(chosen)}
                return
            feasible = {i: [p for p in fitting[i] if not p[0] & occupied] for i in remaining}
            index = (anchor if anchor in remaining else
                     min(remaining, key=lambda i: (len(feasible[i]), i)))
            for bits, cells in feasible[index]:
                chosen[index] = cells
                for p in cells:
                    board[p] = tiles[index]["color"]
                search(tuple(i for i in remaining if i != index), occupied | bits)
                for p in cells:
                    board[p] = -1
                del chosen[index]
                if len(solutions) >= limit:
                    return

        search(tuple(range(len(tiles))), 0)
        if len(solutions) >= limit:
            break
    return list(solutions.values()), anchor_templates


def _paint(solution, shape, background):
    values = np.array(solution["pixels"], dtype=np.int8)
    values[values == -1] = background
    return values.reshape(shape[0] ** 2, shape[1] ** 2)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the sole legal assembly entirely from the input."""
    bg, marker, shape, tiles = _parse(input_grid)
    solutions, _ = _assemblies(tiles, shape)
    if len(solutions) != 1:
        raise ValueError("socket and tiles do not specify one colored assembly")
    return _paint(solutions[0], shape, bg)


def _grow(rng, allowed, wanted):
    allowed = set(allowed)
    start = sorted(allowed)[int(rng.integers(len(allowed)))]
    selected = {start}
    while len(selected) < wanted:
        frontier = sorted(({(r + dr, c + dc) for r, c in selected for dr, dc in _DIRS}
                           & allowed) - selected)
        if not frontier:
            return None
        selected.add(frontier[int(rng.integers(len(frontier)))])
    return selected


def _make_tiles(rng, shape, count, palette, *, quiet=False, multiple=False, nonbar=False):
    h, w = shape
    choices = _templates(h, w, count)
    for _ in range(4000):
        bits, motif = choices[int(rng.integers(len(choices)))]
        support = {(r * h + a, c * w + b) for r, c in motif for a, b in motif}
        wanted = int(rng.integers(count * count // 3 + 1, 3 * count * count // 4 + 1))
        anchor = _grow(rng, support, wanted)
        if anchor is None:
            continue
        rest = _components(support - anchor)
        if not 1 <= len(rest) <= 4:
            continue
        # Connected cuts also produce adjacent, differently colored fillers.
        if len(rest) < 4:
            j = max(range(len(rest)), key=lambda i: len(rest[i]))
            if len(rest[j]) >= 3 and (len(rest) == 1 or rng.random() < 0.65):
                region = set(rest[j])
                first = _grow(rng, region, int(rng.integers(1, len(region))))
                if first and len(_components(region - first)) == 1:
                    rest = rest[:j] + [tuple(sorted(first)), tuple(sorted(region - first))] + rest[j + 1:]
        if not 2 <= len(rest) <= 4:
            continue
        if len(anchor) <= max(map(len, rest)):
            continue
        if nonbar and not any(len(cells) > 2 and min(_dims(_normalize(cells))) > 1 for cells in rest):
            continue
        colors = [int(c) for c in rng.choice(palette, size=len(rest) + 1, replace=False)]
        if len(rest) >= 2 and rng.random() < 0.35:
            colors[-1] = colors[1]
        tiles = [{"color": colors[0], "shape": _normalize(anchor)}]
        for i, cells in enumerate(rest):
            turns = 0 if quiet else int(rng.integers(4))
            tiles.append({"color": colors[i + 1], "shape": _rotate(_normalize(cells), turns)})
        solutions, anchor_templates = _assemblies(tiles, shape)
        if len(solutions) != 1 or (multiple and anchor_templates < 2):
            continue
        if nonbar and _assemblies(tiles, shape, rotations="bars", limit=1)[0]:
            continue
        if not _relocation_witness(tiles, shape, solutions[0]):
            continue
        if not _pixel_swap_witness(tiles, shape, solutions[0]):
            continue
        return tiles, solutions[0]
    raise RuntimeError(f"could not find a unique self-similar jigsaw for {shape}, {count}")


def _scatter(rng, tiles, shape, background, marker):
    fh, fw = shape[0] ** 2 + 2, shape[1] ** 2 + 2
    for attempt in range(100):
        height = 30 if attempt >= 75 else int(rng.integers(23, 31))
        width = 30 if attempt >= 75 else int(rng.integers(23, 31))
        top = int(rng.integers(1, height - fh))
        left = int(rng.integers(1, width - fw))
        # Reserve the ENTIRE socket, so tiles cannot be hidden inside it.
        reserved = {(top + r, left + c) for r in range(fh) for c in range(fw)}
        grid = np.full((height, width), background, dtype=np.int8)
        grid[top, left:left + fw] = grid[top + fh - 1, left:left + fw] = marker
        grid[top:top + fh, left] = grid[top:top + fh, left + fw - 1] = marker
        success = True
        for index in sorted(range(len(tiles)), key=lambda i: (-np.prod(_dims(tiles[i]["shape"])), i)):
            tile = tiles[index]
            sh, sw = _dims(tile["shape"])
            positions = [(r, c) for r in range(1, height - sh) for c in range(1, width - sw)]
            for pos in rng.permutation(len(positions)):
                r0, c0 = positions[int(pos)]
                cells = {(r0 + r, c0 + c) for r, c in tile["shape"]}
                if any((r, c) in reserved or any((r + dr, c + dc) in reserved for dr, dc in _DIRS)
                       for r, c in cells):
                    continue
                for r, c in cells:
                    grid[r, c] = tile["color"]
                reserved.update(cells)
                break
            else:
                success = False
                break
        if success:
            # This also excludes an accidental second socket-shaped loose tile.
            try:
                _parse(grid)
            except ValueError:
                continue
            return grid
    raise RuntimeError("unable to scatter socket and tiles with separation")


def _checked_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    background, marker = [int(c) for c in rng.choice(10, size=2, replace=False)]
    palette = [c for c in range(10) if c not in (background, marker)]
    held_out = _SHAPES[int(rng.integers(len(_SHAPES)))]
    train_shapes = [shape for shape in _SHAPES if shape != held_out]
    rectangles = [shape for shape in train_shapes if shape[0] != shape[1]]
    first = rectangles[int(rng.integers(len(rectangles)))]
    second_options = [shape for shape in train_shapes if shape != first]
    second = second_options[int(rng.integers(len(second_options)))]
    jobs = [("train", first, True, True, False), ("train", second, False, False, True)]
    for _ in range(num_train - 2):
        shape = train_shapes[int(rng.integers(len(train_shapes)))]
        jobs.append(("train", shape, False, False, False))
    rng.shuffle(jobs)
    jobs.extend(("test", held_out, False, False, False) for _ in range(num_test))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, shape, quiet, multiple, nonbar in jobs:
        area = shape[0] * shape[1]
        minimum = 5 if shape == (3, 3) and split == "test" else max(4, area - 3)
        count = area - 1 if quiet or nonbar else int(rng.integers(minimum, area))
        tiles, solution = _make_tiles(rng, shape, count, palette,
                                     quiet=quiet, multiple=multiple, nonbar=nonbar)
        for _ in range(20):
            input_grid = _scatter(rng, tiles, shape, background, marker)
            key = _key(input_grid)
            if key not in seen:
                seen.add(key)
                puzzle[split].append({"input": input_grid, "output": _paint(solution, shape, background)})
                break
        else:
            raise RuntimeError("could not make distinct inputs")
    return puzzle


def _recursive_mask(mask, shape):
    h, w = shape
    coarse = np.array([[mask[r * h:(r + 1) * h, c * w:(c + 1) * w].any()
                        for c in range(w)] for r in range(h)], dtype=bool)
    return np.array_equal(mask, np.kron(coarse, coarse))


def _relocation_witness(tiles, shape, solution):
    """A legal rigid-tile placement that fails only the cross-scale constraint."""
    height, width = shape[0] ** 2, shape[1] ** 2
    occupied = sum(1 << p for p, color in enumerate(solution["pixels"]) if color != -1)
    anchor = max(range(len(tiles)), key=lambda i: len(tiles[i]["shape"]))
    for i, tile in enumerate(tiles):
        if i == anchor:
            continue
        old = sum(1 << p for p in solution["placements"][i])
        rest = occupied ^ old
        for bits, cells in _placements(tile["shape"], height, width, True):
            if bits == old or bits & rest:
                continue
            alternative = np.array([bool((rest | bits) & (1 << p))
                                    for p in range(height * width)]).reshape(height, width)
            if not _recursive_mask(alternative, shape):
                return True
    return False


def _pixel_swap_witness(tiles, shape, solution):
    """Preserve self-similarity and color counts, but break the rigid anchor."""
    anchor = max(tiles, key=lambda tile: len(tile["shape"]))
    if sum(tile["color"] == anchor["color"] for tile in tiles) != 1:
        return False
    width = shape[1] ** 2
    pixels = solution["pixels"]
    original = {p for p, color in enumerate(pixels) if color == anchor["color"]}
    others = [p for p, color in enumerate(pixels) if color not in (-1, anchor["color"])]
    for old in sorted(original):
        for new in others:
            altered = (original - {old}) | {new}
            if _normalize((p // width, p % width) for p in altered) != anchor["shape"]:
                # Exchanging these two colors leaves the Boolean support intact.
                return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _checked_count(len(puzzle["train"]), "num_train", 2, 10)
    _checked_count(len(puzzle["test"]), "num_test", 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, conventions = set(), set()
    shapes = {"train": set(), "test": set()}
    quiet_example = joint_example = nonbar_example = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            bg, marker, shape, tiles = _parse(inp)
            conventions.add((bg, marker))
            shapes[split].add(shape)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert out.shape == (shape[0] ** 2, shape[1] ** 2)
            assert marker not in out
            mask = out != bg
            assert _recursive_mask(mask, shape)
            assert int(mask.sum()) == sum(len(t["shape"]) for t in tiles)
            for color in range(10):
                if color not in (bg, marker):
                    assert int(np.count_nonzero(inp == color)) == int(np.count_nonzero(out == color))
            solutions, anchor_templates = _assemblies(tiles, shape)
            assert len(solutions) == 1, "ambiguous packing or no packing"
            assert np.array_equal(_paint(solutions[0], shape, bg), out)
            # Inspect the actual witness: preserve all pixels, orientation and color.
            anchor = max(range(len(tiles)), key=lambda i: len(tiles[i]["shape"]))
            assert all(len(tiles[anchor]["shape"]) > len(tile["shape"])
                       for i, tile in enumerate(tiles) if i != anchor)
            for i, tile in enumerate(tiles):
                cells = solutions[0]["placements"][i]
                packed = _normalize((p // out.shape[1], p % out.shape[1]) for p in cells)
                variants = {tile["shape"]} if i == anchor else {_rotate(tile["shape"], k) for k in range(4)}
                assert packed in variants
                assert all(int(out[p // out.shape[1], p % out.shape[1]]) == tile["color"] for p in cells)
            assert _relocation_witness(tiles, shape, solutions[0]), "A must constrain the placement"
            assert _pixel_swap_witness(tiles, shape, solutions[0]), "B must constrain the answer"
            if split == "train":
                joint_example |= anchor_templates >= 2
                quiet_example |= bool(_assemblies(tiles, shape, rotations="none", limit=1)[0])
                nonbar_example |= not _assemblies(tiles, shape, rotations="bars", limit=1)[0]
    assert len(conventions) == 1
    assert len(shapes["train"]) >= 2 and not shapes["train"] & shapes["test"]
    assert quiet_example and joint_example and nonbar_example


def self_test():
    """Bounded 42-seed suite, with structural coverage and rule ablations."""
    signatures, motifs, output_shapes, input_shapes = set(), set(), set(), set()
    profiles, colors, backgrounds, markers, counts = set(), set(), set(), set(), set()
    repeated = touching_fillers = disconnected = blank_band = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
                bg, marker, shape, tiles = _parse(inp)
                signature.append((_key(inp), _key(out)))
                backgrounds.add(bg)
                markers.add(marker)
                colors.update(t["color"] for t in tiles)
                profiles.add(tuple(sorted((len(t["shape"]) for t in tiles), reverse=True)))
                counts.add(len(tiles))
                input_shapes.add(inp.shape)
                output_shapes.add(out.shape)
                mask = out != bg
                motifs.add((shape, mask.tobytes()))
                repeated += len({t["color"] for t in tiles}) < len(tiles)
                disconnected += len(_components(map(tuple, np.argwhere(mask)))) > 1
                blank_band += bool((~mask.any(axis=1)).any() or (~mask.any(axis=0)).any())
                anchor_color = max(tiles, key=lambda t: len(t["shape"]))["color"]
                filler = (out != bg) & (out != anchor_color)
                touching_fillers += bool(np.any(filler[:, :-1] & filler[:, 1:] & (out[:, :-1] != out[:, 1:]))
                                         or np.any(filler[:-1, :] & filler[1:, :] & (out[:-1, :] != out[1:, :])))
        signatures.add(tuple(signature))
        if seed == 0:
            # Whole-scene rotation changes the visible anchor orientation and
            # socket axes together. A bijective palette map changes only roles.
            pair = puzzle["train"][0]
            assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
            permutation = np.array([7, 2, 9, 1, 8, 0, 4, 6, 3, 5], dtype=np.int8)
            assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert len(signatures) == 42
    assert len(output_shapes) == 5 and len(input_shapes) >= 20
    assert len(motifs) >= 60 and len(profiles) >= 60
    assert counts == {3, 4, 5} and len(colors) == 10
    assert len(backgrounds) >= 8 and len(markers) >= 8
    assert repeated >= 20 and touching_fillers >= 20
    for a, b in ((2, 1), (10, 4), (7, 3)):
        first = generate_puzzle(796681291, a, b)
        second = generate_puzzle(796681291, a, b)
        validate_generated_puzzle(first)
        assert all(_key(p[role]) == _key(q[role])
                   for split in ("train", "test") for p, q in zip(first[split], second[split])
                   for role in ("input", "output"))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {a}/{b}")
    return {"seeds": 42, "default_pairs": 210, "distinct_puzzles": len(signatures),
            "recursive_masks": len(motifs), "tile_area_profiles": len(profiles),
            "input_shapes": len(input_shapes), "output_shapes": sorted(output_shapes),
            "tile_counts": sorted(counts), "tile_colors": len(colors),
            "background_colors": len(backgrounds), "frame_colors": len(markers),
            "repeated_filler_color_pairs": repeated, "touching_different_fillers": touching_fillers,
            "disconnected_supports": disconnected, "blank_band_supports": blank_band,
            "independent_semantic_verification": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
