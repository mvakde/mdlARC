"""Diagonal cutting molds: construct a boundary, then identify its cavity pieces.

Only the standard library and NumPy are used. No parent generator is imported.
"""

import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape),
             "Grids must be two-dimensional with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "Cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, component = [start], {start}
        while todo:
            for cell in _neighbors(todo.pop()):
                if cell in unseen:
                    unseen.remove(cell)
                    component.add(cell)
                    todo.append(cell)
        result.append(component)
    return result


def _cells(mask):
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(mask))}


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _silhouette(cells):
    r0, r1, c0, c1 = _bounds(cells)
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _shape_key(cells):
    mask = _silhouette(cells)
    return (mask.shape, mask.tobytes())


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _canonical(array, edge, reflected):
    result = np.rot90(array, edge)
    return np.flipud(result).copy() if reflected else result.copy()


def _screen(array, edge, reflected):
    result = np.flipud(array) if reflected else array
    return np.rot90(result, -edge).copy()


def _parse_input(input_grid):
    """Read all conventions and the exact color correspondence from one input."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "The scene needs a uniform background margin")
    objects = _components(_cells(grid != bg))
    molds, exemplars = [], []
    for obj in objects:
        colors = {int(grid[p]) for p in obj}
        if len(colors) > 1:
            molds.append(obj)
        else:
            exemplars.append((obj, next(iter(colors))))
    _require(len(molds) == 1, "There must be one two-color mold")
    r0, r1, c0, c1 = _bounds(molds[0])
    tile_screen = grid[r0:r1, c0:c1]
    n = r1 - r0
    _require(n == c1 - c0 and n >= 5, "The mold must be square")
    _require(_cells(tile_screen != bg) == {(r - r0, c - c0) for r, c in molds[0]},
             "Another object overlaps the mold bounds")
    candidates = []
    for edge in range(4):
        for reflected in (False, True):
            tile = _canonical(tile_screen, edge, reflected)
            rail = int(tile[0, 0])
            rail_mask = np.zeros((n, n), dtype=bool)
            rail_mask[:, 0] = True
            rail_mask[-1, 1] = True
            if rail != bg and np.array_equal(tile == rail, rail_mask):
                colors = set(int(x) for x in np.unique(tile)) - {bg, rail}
                if len(colors) == 1:
                    material = next(iter(colors))
                    hole = tile == bg
                    if (not np.any(hole[0]) and not np.any(hole[-1]) and
                            not np.any(hole[:, 0]) and not np.any(hole[:, -1])):
                        candidates.append((edge, reflected, tile.copy(), rail, material, hole))
    _require(len(candidates) == 1, "The rail and its base starter must orient the mold uniquely")
    edge, reflected, tile, rail, material, hole = candidates[0]
    _require(len(_components(_cells(hole))) == 1, "The uncut cavity must be connected")
    diagonal = np.fromfunction(lambda r, c: r + c == n - 1, (n, n), dtype=int)
    diagonal[-1, 0] = False  # The inherited rail always owns this corner.
    _require(bool(np.any(hole & diagonal)), "The diagonal must cut through the cavity")
    pieces = _components(_cells(hole & ~diagonal))
    _require(2 <= len(pieces) <= 7 and all(len(p) >= 3 for p in pieces),
             "The cut must expose 2..7 nontrivial pieces")
    sides = [int(sum(next(iter(p))) > n - 1) for p in pieces]
    _require(set(sides) == {0, 1}, "Pieces must occur on both sides of the diagonal")
    keys = [_shape_key(p) for p in pieces]
    _require(len(set(keys)) == len(keys), "Piece shapes must be distinct under translation")
    _require(len(exemplars) == len(pieces), "Each cut piece needs exactly one exemplar")
    lookup, exemplar_order, exemplar_colors = {}, [], []
    for obj, color in exemplars:
        _require(color not in (bg, material, rail), "Exemplar colors must have separate roles")
        # Normalize the entire scene's orientation, never an individual match.
        mask = _canonical(_silhouette(obj), edge, reflected)
        key = _shape_key(_cells(mask))
        _require(key in keys and key not in lookup, "Exact silhouette matching must be bijective")
        lookup[key] = color
        exemplar_order.append(keys.index(key))
        exemplar_colors.append(color)
    _require(len(set(exemplar_colors)) == len(pieces), "Exemplars need distinct colors")
    screen_positions = []
    for i, p in enumerate(pieces):
        mask = np.zeros((n, n), dtype=bool)
        for cell in p:
            mask[cell] = True
        screen_positions.append((min(_cells(_screen(mask, edge, reflected))), i))
    screen_order = tuple(i for pos, i in sorted(screen_positions))
    return {"grid": grid, "tile": tile, "size": n, "edge": edge, "reflected": reflected,
            "background": bg, "material": material, "rail": rail,
            "hole": hole, "diagonal": diagonal, "pieces": pieces,
            "colors": [lookup[key] for key in keys], "keys": keys,
            "areas": tuple(len(p) for p in pieces), "sides": tuple(sides),
            "order": tuple(exemplar_order),
            "order_aligned": tuple(exemplar_order) == screen_order,
            "mold_position": (r0, c0)}


def _finish(facts):
    out = facts["tile"].copy()
    for piece, color in zip(facts["pieces"], facts["colors"]):
        for cell in piece:
            out[cell] = color
    out[facts["diagonal"]] = facts["rail"]
    out[-1, 1:] = facts["rail"]
    return _screen(out, facts["edge"], facts["reflected"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete and return the mold, using only its input scene."""
    return _finish(_parse_input(input_grid))


def _sample_cavity(rng, n, count, equal_area=False):
    """Grow pieces from alternating branches of one connected diagonal channel.

    The channel is generation scaffolding, not a requirement of the solver.
    Growth is constrained only to keep the eventual pieces separate, the
    original cavity connected, and the surrounding material connected.
    """
    inner = {(r, c) for r in range(1, n - 1) for c in range(1, n - 1)}
    whole = {(r, c) for r in range(n) for c in range(n)}
    for _ in range(700):
        length = int(rng.integers(count + 1, n - 1))
        start = int(rng.integers(1, n - length))
        cuts = sorted(int(x) for x in rng.choice(np.arange(1, length - 1),
                                                size=count - 1, replace=False))
        ends = [0] + cuts + [length - 1]
        channel = {(r, n - 1 - r) for r in range(start, start + length)}
        first_side = int(rng.integers(0, 2))
        pieces = []
        for i in range(count):
            lo, hi = start + ends[i], start + ends[i + 1]
            if (i + first_side) % 2 == 0:
                part = {(r, n - 2 - r) for r in range(lo, hi)}
                part.update((r, n - 3 - r) for r in range(lo, hi - 1))
            else:
                part = {(r + 1, n - 1 - r) for r in range(lo, hi)}
                part.update((r + 2, n - 1 - r) for r in range(lo, hi - 1))
            pieces.append(part)
        areas = [int(rng.integers(max(3, len(p)), max(4, len(p) + 1, n + 3)))
                 for p in pieces]
        if equal_area:
            areas = [max(4, max(areas))] * count
        possible = True
        for i, part in enumerate(pieces):
            upper = sum(next(iter(part))) < n - 1
            other = set().union(*(p for j, p in enumerate(pieces) if i != j))
            while len(part) < areas[i]:
                frontier = set()
                for cell in part:
                    for candidate in _neighbors(cell):
                        if (candidate not in inner or candidate in part or
                                sum(candidate) == n - 1 or
                                (sum(candidate) < n - 1) != upper):
                            continue
                        if candidate in other or any(p in other for p in _neighbors(candidate)):
                            continue
                        frontier.add(candidate)
                if not frontier:
                    possible = False
                    break
                options = sorted(frontier)
                part.add(options[int(rng.integers(len(options)))])
            if not possible:
                break
        if not possible:
            continue
        cavity = channel | set().union(*pieces)
        if len(_components(whole - cavity)) != 1:
            continue
        actual = _components({p for p in cavity if sum(p) != n - 1})
        if len(actual) != count or len({_shape_key(p) for p in actual}) != count:
            continue
        return cavity, actual
    raise RuntimeError("Could not sample a connected mold cavity")


def _pack_scene(rng, items, background):
    """Scatter a square mold and loose silhouettes, with empty space between."""
    for attempt in range(180):
        if attempt < 30:
            height, width = (int(x) for x in rng.integers(21, 31, size=2))
        else:
            height = width = 30
        occupied = np.zeros((height, width), dtype=bool)
        positions = {}
        # Place large rectangles first, but choose their positions freely.
        order = sorted(range(len(items)),
                       key=lambda i: (-items[i].size, -max(items[i].shape), i))
        for i in order:
            h, w = items[i].shape
            if h + 2 > height or w + 2 > width:
                break
            for _ in range(140):
                r = int(rng.integers(1, height - h))
                c = int(rng.integers(1, width - w))
                if np.any(occupied[r - 1:r + h + 1, c - 1:c + w + 1]):
                    continue
                occupied[r:r + h, c:c + w] = True
                positions[i] = (r, c)
                break
            else:
                break
        if len(positions) != len(items):
            continue
        scene = np.full((height, width), background, dtype=np.int8)
        for i, item in enumerate(items):
            r, c = positions[i]
            h, w = item.shape
            scene[r:r + h, c:c + w] = item
        r0, r1, c0, c1 = _bounds(_cells(scene != background))
        r0 = max(0, r0 - int(rng.integers(1, 3)))
        r1 = min(height, r1 + int(rng.integers(1, 3)))
        c0 = max(0, c0 - int(rng.integers(1, 3)))
        c1 = min(width, c1 + int(rng.integers(1, 3)))
        return scene[r0:r1, c0:c1].copy()
    raise RuntimeError("Could not pack the scene within 30 by 30")


def _make_pair(rng, n, count, edge, reflected, background, equal_area=False, displaced=False):
    for _ in range(30):
        cavity, pieces = _sample_cavity(rng, n, count, equal_area)
        palette = [int(x) for x in rng.permutation([c for c in range(10) if c != background])]
        material, rail = palette[:2]
        colors = palette[2:2 + count]
        tile = np.full((n, n), material, dtype=np.int8)
        for cell in cavity:
            tile[cell] = background
        tile[:, 0] = rail
        tile[-1, 1] = rail
        expected = tile.copy()
        for part, color in zip(pieces, colors):
            for cell in part:
                expected[cell] = color
        for r in range(n - 1):
            expected[r, n - 1 - r] = rail
        expected[-1, 1:] = rail
        items = [_screen(tile, edge, reflected)]
        for part, color in zip(pieces, colors):
            mask = _screen(_silhouette(part), edge, reflected)
            exemplar = np.full(mask.shape, background, dtype=np.int8)
            exemplar[mask] = color
            items.append(exemplar)
        for _ in range(20 if displaced else 1):
            scene = _pack_scene(rng, items, background)
            facts = _parse_input(scene)
            if displaced and facts["order_aligned"]:
                continue
            return {"input": scene, "output": _screen(expected, edge, reflected)}
    raise RuntimeError("Could not construct a displaced demonstration")


def _integer(value, name, low, high=None):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), name + " is outside its allowed range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(0, 3))
    if shift == 0:  # New number of pieces; dimensions can overlap.
        train_sizes, test_sizes = list(range(9, 14)), list(range(9, 15))
    elif shift == 1:  # Larger unseen molds, with familiar and novel pieces.
        train_sizes, test_sizes = [8, 9, 10], [11, 12, 13, 14]
    else:  # Smaller unseen molds; extrapolation need not mean enlargement.
        train_sizes, test_sizes = [11, 12, 13, 14], [8, 9, 10]
    first_edge = int(rng.integers(0, 4))
    second_edge = int(rng.choice([e for e in range(4) if e != first_edge]))
    first_reflected = bool(rng.integers(0, 2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_size = first_bg = None
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            for _ in range(80):
                if split == "train" and i < 2:
                    count = 2 + i
                    edge = first_edge if i == 0 else second_edge
                    reflected = first_reflected if i == 0 else not first_reflected
                else:
                    count = 4 if split == "test" and shift == 0 else int(rng.integers(2, 4 if shift == 0 else 5))
                    edge = int(rng.integers(0, 4))
                    reflected = bool(rng.integers(0, 2))
                size_pool = train_sizes if split == "train" else test_sizes
                size_pool = [n for n in size_pool if n >= (9 if count == 4 else 8)]
                if split == "train" and i == 1:
                    size_pool = [n for n in size_pool if n != first_size]
                n = int(rng.choice(size_pool))
                bg_pool = [c for c in range(10) if not (split == "train" and i == 1 and c == first_bg)]
                background = int(rng.choice(bg_pool))
                pair = _make_pair(rng, n, count, edge, reflected, background,
                                  equal_area=(split == "train" and i == 0),
                                  displaced=(split == "train" and i == 0))
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise RuntimeError("Could not generate a distinct example")
            if split == "train" and i == 0:
                first_size, first_bg = n, background
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Bad pair schema")
    facts = _parse_input(pair["input"])
    out = _grid(pair["output"])
    _require(np.array_equal(out, _finish(facts)), "Incorrect completed mold")
    canonical = _canonical(out, facts["edge"], facts["reflected"])
    before = facts["tile"]
    _require(np.array_equal(canonical[:, 0], before[:, 0]), "The rail must be preserved")
    _require(np.all(canonical[facts["diagonal"]] == facts["rail"]), "The diagonal is incomplete")
    _require(np.all(canonical[-1] == facts["rail"]), "The base is incomplete")
    untouched = ~(facts["hole"] | facts["diagonal"])
    untouched[-1, 1:] = False
    _require(np.array_equal(canonical[untouched], before[untouched]), "Uncut material changed")
    _require(facts["background"] not in canonical, "Some cavity cells remain unfilled")
    for piece, color in zip(facts["pieces"], facts["colors"]):
        _require(all(canonical[p] == color for p in piece), "Piece color was not transferred exactly")
    # A is necessary for segmentation, not merely for the visible added border.
    _require(len(_components(_cells(facts["hole"]))) == 1 < len(facts["pieces"]),
             "The geometric cut must change the objects being matched")
    return facts


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal example counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _grid_key(pair["input"])
            _require(key not in seen, "Repeated input")
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    _require(any(len(set(f["areas"])) < len(f["areas"]) for f in train),
             "Training must disambiguate equal-area silhouettes")
    _require(any(len(f["pieces"]) >= 3 for f in train),
             "Training must distinguish multiple pieces on one side")
    _require(any(not f["order_aligned"] for f in train),
             "Training must reject positional color correspondence")
    for name in ("size", "edge", "reflected", "background"):
        _require(len({f[name] for f in train}) >= 2, "Training lacks variation in " + name)
    sizes = {f["size"] for f in train}
    counts = {len(f["pieces"]) for f in train}
    _require(all(f["size"] not in sizes or len(f["pieces"]) not in counts for f in infos["test"]),
             "Each test must shift mold size or piece count")


def self_test():
    """Bounded structural, ablation, equivariance, count and 42-seed checks."""
    metrics = {name: set() for name in ("sizes", "edges", "reflections", "counts", "backgrounds", "materials",
                                       "rails", "dimensions", "structures", "puzzles")}
    equal_area = same_box = aligned = larger = smaller = new_count = 0
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_grid_key(pair[k]) for s in ("train", "test")
                          for pair in puzzle[s] for k in ("input", "output"))
        metrics["puzzles"].add(signature)
        train = [_parse_input(p["input"]) for p in puzzle["train"]]
        for s in ("train", "test"):
            for pair in puzzle[s]:
                f = _check_pair(pair)
                total_pairs += 1
                for field, name in (("size", "sizes"), ("edge", "edges"), ("reflected", "reflections"),
                                    ("background", "backgrounds"), ("material", "materials"), ("rail", "rails")):
                    metrics[name].add(f[field])
                metrics["counts"].add(len(f["pieces"]))
                metrics["dimensions"].add(np.asarray(pair["input"]).shape)
                metrics["structures"].add((f["size"], f["edge"], f["reflected"], tuple(sorted(f["keys"]))))
                equal_area += len(set(f["areas"])) < len(f["areas"])
                same_box += len({key[0] for key in f["keys"]}) < len(f["pieces"])
                aligned += f["order_aligned"]
                # A meaningful omission of B: one color for each triangle.
                if len(f["pieces"]) >= 3:
                    wrong = f["tile"].copy()
                    side_colors = {}
                    for p, side, color in zip(f["pieces"], f["sides"], f["colors"]):
                        side_colors.setdefault(side, color)
                        for cell in p:
                            wrong[cell] = side_colors[side]
                    wrong[f["diagonal"]] = f["rail"]
                    wrong[-1, 1:] = f["rail"]
                    _require(not np.array_equal(_screen(wrong, f["edge"], f["reflected"]), pair["output"]),
                             "Ignoring shape identity did not affect the answer")
                # A meaningful omission of A: leave the connected cavity uncut.
                wrong = f["tile"].copy()
                wrong[f["hole"]] = f["colors"][0]
                right = _canonical(np.asarray(pair["output"]), f["edge"], f["reflected"])
                _require(np.any(wrong[f["hole"]] != right[f["hole"]]),
                         "Omitting the geometric cut did not affect cavity colors")
                if s == "test":
                    larger += f["size"] > max(t["size"] for t in train)
                    smaller += f["size"] < min(t["size"] for t in train)
                    new_count += len(f["pieces"]) not in {len(t["pieces"]) for t in train}
        if seed < 4:
            pair = puzzle["train"][0]
            before = pair["input"].copy()
            _require(np.array_equal(solve(before), pair["output"]) and
                     np.array_equal(before, pair["input"]), "solve mutates its input")
            for k in range(4):
                _require(np.array_equal(solve(np.rot90(before, k)), np.rot90(pair["output"], k)),
                         "Whole-scene rotation changed the rule")
            _require(np.array_equal(solve(np.flipud(before)), np.flipud(pair["output"])),
                     "Whole-scene reflection changed the rule")
            permutation = np.random.default_rng(1000 + seed).permutation(10)
            _require(np.array_equal(solve(permutation[before]), permutation[pair["output"]]),
                     "Literal color identity changed the rule")
    _require(len(metrics["puzzles"]) == 42, "Seeds repeat whole puzzles")
    _require(metrics["sizes"] == set(range(8, 15)), "Mold scale coverage is incomplete")
    _require(metrics["edges"] == set(range(4)) and metrics["reflections"] == {False, True} and
             metrics["counts"] == {2, 3, 4}, "Geometry lacks variation")
    for name in ("backgrounds", "materials", "rails"):
        _require(len(metrics[name]) == 10, "Palette roles lack variation")
    _require(len(metrics["dimensions"]) >= 30 and len(metrics["structures"]) >= 190,
             "The family repeats too few structures")
    _require(equal_area >= 42 and same_box >= 10 and aligned > 0, "Shape and permutation coverage is weak")
    _require(larger > 0 and smaller > 0 and new_count > 0, "Train/test shifts are incomplete")
    # Every legal explicit count combination, including the 2/1 minimum.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(700 + 10 * a + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "Counts were ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, "1"), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    for seed in (-1, True, 0.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal seed accepted")
    first, second = generate_puzzle(123456, 6, 3), generate_puzzle(123456, 6, 3)
    _require(all(np.array_equal(p[k], q[k]) for s in ("train", "test")
                 for p, q in zip(first[s], second[s]) for k in ("input", "output")),
             "Generation is nondeterministic")
    return {"seeds": 42, "pairs": total_pairs,
            "distinct_puzzles": len(metrics["puzzles"]), "mold_sizes": sorted(metrics["sizes"]),
            "rail_orientations": sorted(metrics["edges"]), "piece_counts": sorted(metrics["counts"]),
            "reflected_conventions": sorted(metrics["reflections"]),
            "scene_dimensions": len(metrics["dimensions"]), "mold_structures": len(metrics["structures"]),
            "equal_area_pairs": equal_area, "same_box_pairs": same_box,
            "aligned_pairs": aligned, "larger_tests": larger, "smaller_tests": smaller,
            "novel_count_tests": new_count}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
