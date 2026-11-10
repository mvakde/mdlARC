"""Keyed jigsaw assembly: joint contour matching and local color containment.

No parent generators, files, seeds, or construction records are used by solve.
The solver searches the specified full-width, column-convex chain family.
"""
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _runs(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return list(zip(map(int, starts), map(int, ends)))


def _components(mask, grid):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        color = int(grid[first])
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and int(grid[p]) == color:
                    remaining.remove(p)
                    stack.append(p)
        result.append((np.array(sorted(cells), dtype=int), color))
    return result


def _dilate(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[r:r+h, c:c+w]
                                 for r in range(3) for c in range(3)])


def _scene(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("Background must be the unique most frequent color")
    bg = int(counts.argmax())
    candidates = []
    for axis, view in enumerate((grid, grid.T)):
        for band in sorted(set(map(int, grid.ravel())) - {bg}):
            full = np.all(view == band, axis=1)
            for center in range(2, view.shape[0]-2):
                if not (full[center-2] and full[center] and full[center+2]):
                    continue
                keys, keymask = [], np.zeros(view.shape, dtype=bool)
                valid = True
                for side in (-1, 1):
                    row = center + side
                    for start, end in _runs(view[row] != band):
                        values = view[row, start:end]
                        query = values[values != bg]
                        if (start == 0 or end == view.shape[1]
                                or not 3 <= end-start <= 7
                                or len(query) != 3 or len(set(map(int, query))) != 3):
                            valid = False
                            break
                        keys.append({"side": side, "start": start, "width": end-start,
                                     "query": frozenset(map(int, query)), "row": row})
                        keymask[row, start:end] = True
                if valid and keys:
                    candidates.append((axis, view, bg, band, center, keys, keymask))
    if len(candidates) != 1:
        raise ValueError("Expected one five-cell band with recognizable local keys")
    axis, view, bg, band, center, keys, keymask = candidates[0]
    structure = (view == band) | keymask
    if len(_components(view == band, view)) != 1:
        raise ValueError("Structural band must be connected")
    edge = np.array([(int(np.flatnonzero(view[:, c] == band)[0]),
                      int(np.flatnonzero(view[:, c] == band)[-1])+1)
                     for c in range(view.shape[1])])
    # Outside the keyed five-row core, each column is a solid projection.
    for c, (lo, hi) in enumerate(edge):
        if (lo > center-2 or hi < center+3
                or np.any(view[lo:center-2, c] != band)
                or np.any(view[center+3:hi, c] != band)):
            raise ValueError("Band projections must have no holes")
    mask = (view != bg) & ~structure
    pieces = _components(mask, view)
    if not pieces or len(pieces) > 14:
        raise ValueError("Expected 1..14 detached monochrome pieces")
    occupied = structure.copy()
    for cells, color in pieces:
        if color == band or np.any(_dilate(occupied)[cells[:, 0], cells[:, 1]]):
            raise ValueError("Input pieces need one-cell clearance")
        occupied[cells[:, 0], cells[:, 1]] = True
    return {"view": view, "axis": axis, "background": bg, "band": band,
            "center": center, "keys": keys, "structure": structure,
            "keymask": keymask, "edges": edge, "pieces": pieces}


def _description(cells, side):
    cross = cells[:, 1] - cells[:, 1].min()
    outward = side*cells[:, 0]
    lower, upper = [], []
    for c in range(int(cross.max())+1):
        values = np.sort(outward[cross == c])
        if not len(values) or not np.array_equal(values, np.arange(values[0], values[-1]+1)):
            raise ValueError("Pieces must be full-width column-convex layers")
        lower.append(int(values[0]))
        upper.append(int(values[-1])+1)
    offset = min(lower)
    return (tuple(v-offset for v in lower), tuple(v-offset for v in upper),
            np.column_stack((outward-offset, cross)))


def _chains(info, key):
    """All geometrically completed chains, including nonqualifying alternatives."""
    side, start, width = (key[k] for k in ("side", "start", "width"))
    descriptions = [_description(cells, side) for cells, _ in info["pieces"]]
    lookup = {}
    for i, (lo, hi, local) in enumerate(descriptions):
        if len(lo) == width:
            lookup.setdefault(lo, []).append(i)
    initial = (1-info["edges"][start:start+width, 0] if side == -1
               else info["edges"][start:start+width, 1])
    limit = 1 if side == -1 else info["view"].shape[0]
    result = []

    def visit(profile, bits, placements, colors):
        if bits and len(set(profile)) == 1:
            cells = frozenset((int(r), int(c)) for _, a in placements for r, c in a)
            result.append({"bits": bits, "placements": placements, "cells": cells,
                           "colors": colors, "positive": key["query"] <= colors,
                           "terminal": profile[0], "initial": tuple(map(int, initial))})
        offset = min(profile)
        normalized = tuple(v-offset for v in profile)
        for i in lookup.get(normalized, ()):
            if bits & (1 << i):
                continue
            lo, hi, local = descriptions[i]
            next_profile = tuple(offset+v for v in hi)
            if max(next_profile) > limit:
                continue
            cells = np.column_stack((side*(local[:, 0]+offset), local[:, 1]+start))
            if (np.any(cells < 0) or np.any(cells[:, 0] >= info["view"].shape[0])
                    or np.any(info["structure"][cells[:, 0], cells[:, 1]])):
                continue
            if set(map(tuple, cells)) == set(map(tuple, info["pieces"][i][0])):
                continue
            visit(next_profile, bits | (1 << i), placements+[(i, cells)],
                  colors | {info["pieces"][i][1]})

    visit(tuple(map(int, initial)), 0, [], frozenset())
    return result


def _answers(info, require_query=True):
    choices = [[c for c in _chains(info, key) if c["positive"] or not require_query]
               for key in info["keys"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    answers = {}

    def visit(depth, used, occupied, chosen):
        if len(answers) >= 2:
            return
        if depth == len(order):
            # Non-selected pieces remain at their source coordinates, and
            # cannot be overwritten by a selected assembly.
            for i, (cells, _) in enumerate(info["pieces"]):
                if not used & (1 << i) and occupied & set(map(tuple, cells)):
                    return
            out = info["view"].copy()
            for i, (cells, _) in enumerate(info["pieces"]):
                if used & (1 << i):
                    out[cells[:, 0], cells[:, 1]] = info["background"]
            for chain in chosen.values():
                for i, cells in chain["placements"]:
                    out[cells[:, 0], cells[:, 1]] = info["pieces"][i][1]
            answers.setdefault(out.tobytes(), (out, chosen.copy()))
            return
        k = order[depth]
        for chain in choices[k]:
            if chain["bits"] & used or chain["cells"] & occupied:
                continue
            chosen[k] = chain
            visit(depth+1, used | chain["bits"], occupied | chain["cells"], chosen)
            del chosen[k]

    visit(0, 0, frozenset(), {})
    return list(answers.values())


def _solve_info(input_grid):
    info = _scene(input_grid)
    answers = _answers(info)
    if len(answers) != 1:
        raise ValueError("Keys and geometry must determine exactly one colored output")
    answer, chosen = answers[0]
    info["chosen"] = chosen
    return (answer.T if info["axis"] else answer).copy(), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _solve_info(input_grid)[0]


def _profiles(rng, width, count, amplitude):
    cuts, seen = [], set()
    for _ in range(count):
        for attempt in range(100):
            p = rng.integers(0, amplitude+1, size=width)
            p -= p.min()
            key = tuple(map(int, p))
            if p.max() and key not in seen:
                break
        else:
            raise ValueError("Profile pool exhausted")
        seen.add(key)
        if cuts:
            lo = cuts[-1]
            # Neighboring column intervals overlap, even at narrow necks.
            offset = max(int(np.max(lo+1-p)),
                         int(np.max(np.maximum(lo[:-1], lo[1:])+1
                                    - np.minimum(p[:-1], p[1:]))))
            p = p + offset + int(rng.integers(0, 2))
        cuts.append(p)
    cuts.append(np.full(width, int(cuts[-1].max())+int(rng.integers(1, 3))))
    return cuts


def _scatter(rng, base, tiles, bg, forbidden_for_decoys, reserve_destinations=False):
    h, w = base.shape
    for attempt in range(32):
        result = base.copy()
        occupied = base != bg
        locations = [None]*len(tiles)
        order = sorted(range(len(tiles)), key=lambda i: -len(tiles[i][0])+rng.random()*3)
        for i in order:
            cells, color, selected = tiles[i]
            local = cells-cells.min(axis=0)
            ph, pw = local.max(axis=0)+1
            forbidden = _dilate(occupied)
            if not selected or reserve_destinations:
                forbidden = forbidden | forbidden_for_decoys
            valid = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in local:
                valid &= ~forbidden[r:r+h-ph+1, c:c+w-pw+1]
            if selected:
                origin = cells.min(axis=0)
                if np.all(origin >= 0) and origin[0] < valid.shape[0] and origin[1] < valid.shape[1]:
                    valid[tuple(origin)] = False
            positions = np.argwhere(valid)
            if not len(positions):
                break
            target = local+positions[int(rng.integers(len(positions)))]
            result[target[:, 0], target[:, 1]] = color
            occupied[target[:, 0], target[:, 1]] = True
            locations[i] = target
        else:
            return result, locations
    raise ValueError("Scatter did not fit")


def _make_pair(rng, package, palette, transpose):
    bg, band = palette[:2]
    labels = list(palette[2:])
    amplitude = int(rng.integers(1, 3))
    if package == "identity":
        count = 2
        width = int(rng.integers(3, 6))
        cuts = _profiles(rng, width, 3, amplitude)
        widths, profiles, sides = [width]*2, [cuts, [p.copy() for p in cuts]], [1, 1]
        q0 = labels[:3]
        shared = int(rng.integers(3))
        q1 = q0[:shared]+labels[3:6-shared]
        colors = [q0, q1]
        queries = [frozenset(q0), frozenset(q1)]
    elif package == "extras":
        count = 1
        widths, sides = [int(rng.integers(3, 7))], [1]
        profiles = [_profiles(rng, widths[0], 5, amplitude)]
        colors = [labels[:3]+[labels[3], labels[3]]]
        queries = [frozenset(labels[:3])]
    else:
        count = int(rng.choice([1, 2, 3], p=[.2, .6, .2]))
        widths = [int(rng.integers(3, 7)) for _ in range(count)]
        sides = [int(rng.choice([-1, 1])) for _ in range(count)]
        profiles, colors, queries = [], [], []
        for width in widths:
            n = int(rng.integers(3, 6))
            profiles.append(_profiles(rng, width, n, amplitude))
            query = [int(x) for x in rng.choice(labels, 3, replace=False)]
            extras = [c for c in labels if c not in query]
            chain_colors = query + [int(rng.choice(labels if rng.random() < .5 else extras))
                                    for _ in range(n-3)]
            rng.shuffle(chain_colors)
            colors.append(chain_colors)
            queries.append(frozenset(query))
    w = int(rng.integers(max(20, sum(widths)+2*count+2), 31))
    starts, reserved = [], {-1: set(), 1: set()}
    for width, side in zip(widths, sides):
        options = [c for c in range(1, w-width)
                   if not set(range(c-1, c+width+1)) & reserved[side]]
        if not options:
            raise ValueError("Key spacing did not fit")
        start = int(rng.choice(options))
        starts.append(start)
        reserved[side].update(range(start, start+width))
    # Shallow unused projections form structural context, never extra keys.
    edge = {s: np.zeros(w, dtype=int) for s in (-1, 1)}
    for side in (-1, 1):
        for _ in range(int(rng.integers(1, 5))):
            a = int(rng.integers(1, w-1))
            edge[side][a:min(w-1, a+int(rng.integers(1, 4))) ] = int(rng.integers(1, 3))
    for start, width, side, cuts in zip(starts, widths, sides, profiles):
        edge[side][start:start+width] = cuts[0]
    depth = {s: max([int(edge[s].max())]+[int(p[-1][0]) for p, side in zip(profiles, sides) if side == s])
             for s in (-1, 1)}
    if 5+depth[-1]+depth[1] > 29:
        raise ValueError("Tabs did not fit")
    h = int(rng.integers(max(24, 5+depth[-1]+depth[1]+1), 31))
    center = int(rng.integers(depth[-1]+2, h-depth[1]-2))
    base = np.full((h, w), bg, dtype=np.int8)
    base[center-2:center+3, :] = band
    for c in range(w):
        base[center-2-edge[-1][c]:center-2, c] = band
        base[center+3:center+3+edge[1][c], c] = band
    tiles, targetmask = [], np.zeros((h, w), dtype=bool)
    for start, width, side, cuts, chain_colors, query in zip(starts, widths, sides, profiles, colors, queries):
        base[center+side, start:start+width] = bg
        positions = rng.choice(width, 3, replace=False)
        for position, color in zip(positions, rng.permutation(sorted(query))):
            base[center+side, start+int(position)] = int(color)
        origin = center+3 if side == 1 else center-3
        for lo, hi, color in zip(cuts, cuts[1:], chain_colors):
            cells = np.array([(origin+side*u, start+c) for c in range(width)
                              for u in range(int(lo[c]), int(hi[c]))], dtype=int)
            tiles.append((cells, int(color), True))
            targetmask[cells[:, 0], cells[:, 1]] = True
    selected_count = len(tiles)
    if package == "identity":
        # A shape-compatible terminal cap with an irrelevant identity.
        tiles.append((tiles[2][0].copy(), labels[-1], False))
    elif package == "extras":
        # Each query identity can be removed by a geometric twin; none of
        # these richer incomplete chains satisfies the key.
        for i in range(3):
            # The first decoy shares the selected extra's color: components,
            # rather than color classes, are the objects being allocated.
            tiles.append((tiles[i][0].copy(), labels[3] if i == 0 else labels[4], False))
    else:
        for _ in range(int(rng.integers(3))):
            i = int(rng.integers(selected_count))
            tiles.append((tiles[i][0].copy(), int(rng.choice(labels)), False))
    if len(tiles) > 14:
        raise ValueError("Too many parts")
    inp, sources = _scatter(rng, base, tiles, bg, targetmask,
                            reserve_destinations=package in ("identity", "extras"))
    out = inp.copy()
    for i, (cells, color, selected) in enumerate(tiles):
        if selected:
            source = sources[i]
            out[source[:, 0], source[:, 1]] = bg
    for cells, color, selected in tiles:
        if selected:
            out[cells[:, 0], cells[:, 1]] = color
    if rng.random() < .5:
        inp, out = inp[::-1], out[::-1]
    if rng.random() < .5:
        inp, out = inp[:, ::-1], out[:, ::-1]
    if transpose:
        inp, out = inp.T, out.T
    pair = {"input": inp.copy(), "output": out.copy()}
    expected, info = _solve_info(inp)
    if not np.array_equal(expected, out):
        raise ValueError("Construction disagrees with joint search")
    if package == "identity" and len(_answers(info, require_query=False)) != 2:
        raise ValueError("Teaching scene must need its keys")
    return pair


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _structural_signature(info):
    """Ignore palette, scatter coordinates, margins, and whole-scene orientation."""
    result = []
    for k, chain in info["chosen"].items():
        key = info["keys"][k]
        root = chain["initial"]
        root = tuple(v-min(root) for v in root)
        layers = tuple(_description(info["pieces"][i][0], key["side"])[:2]
                       for i, _ in chain["placements"])
        record = (key["width"], root, layers)
        reflected = (key["width"], root[::-1], tuple((lo[::-1], hi[::-1]) for lo, hi in layers))
        result.append(min(record, reflected))
    return tuple(sorted(result))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))  # stable horizontal, stable vertical, or both taught
    stable_palette = rng.random() < .3
    palette = list(map(int, rng.permutation(10)))
    puzzle, seen, training_structures = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            package = ("identity" if index == 0 else "extras" if index == 1 else "free") if split == "train" else "free"
            transpose = mode == 1 or (mode == 2 and (index % 2 == 1 if split == "train" else bool(rng.integers(2))))
            local_palette = palette if stable_palette else list(map(int, rng.permutation(10)))
            for attempt in range(180):
                try:
                    pair = _make_pair(rng, package, local_palette, transpose)
                except ValueError:
                    continue
                _, info = _solve_info(pair["input"])
                signature = _structural_signature(info)
                if split == "test" and signature in training_structures:
                    continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError(f"Bounded construction exhausted for {package}")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                training_structures.add(signature)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    expected, info = _solve_info(inp)
    assert np.array_equal(expected, out), "Incorrect keyed assembly"
    view = out.T if info["axis"] else out
    assert np.array_equal(view[info["structure"]], info["view"][info["structure"]])
    assert np.array_equal(np.bincount(inp.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10))
    used, target = set(), set()
    for k, chain in info["chosen"].items():
        key = info["keys"][k]
        assert key["query"] <= chain["colors"]
        assert isinstance(chain["terminal"], int)
        for i, cells in chain["placements"]:
            assert i not in used
            used.add(i)
            source, color = info["pieces"][i]
            shifted = source+cells.min(axis=0)-source.min(axis=0)
            assert set(map(tuple, shifted)) == set(map(tuple, cells)), "Not a translation"
            assert not target & set(map(tuple, cells))
            target.update(map(tuple, cells))
            assert np.all(view[cells[:, 0], cells[:, 1]] == color)
        # Every outward column is filled without a gap from the actual band
        # edge to the same terminal; this checks the rectangle, not just labels.
        side, start, width = key["side"], key["start"], key["width"]
        required = {(side*u, start+c) for c in range(width)
                    for u in range(chain["initial"][c], chain["terminal"])}
        assert required == chain["cells"], "Incomplete rectangle or excess cells"
    for i, (cells, color) in enumerate(info["pieces"]):
        if i not in used:
            assert np.array_equal(view[cells[:, 0], cells[:, 1]], info["view"][cells[:, 0], cells[:, 1]])
    return info


def _unblocked(info, chain):
    """A geometric alternative must not be rejected for a survivor collision."""
    return all(not chain["cells"] & set(map(tuple, cells))
               for i, (cells, _) in enumerate(info["pieces"])
               if not chain["bits"] & (1 << i))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_infos, test_infos = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            assert _key(pair["input"]) not in seen
            seen.add(_key(pair["input"]))
            if split == "train":
                train_infos.append(info)
            else:
                test_infos.append(info)
    # Two witness packages survive even at the two-example minimum.
    assert any(len(_answers(i, require_query=False)) == 2 for i in train_infos), "Keys must affect allocation"
    records = [(i, k, chain) for i in train_infos for k, key in enumerate(i["keys"])
               for chain in _chains(i, key)]
    assert any(c["positive"] and len(c["placements"]) == 3 and len(c["colors"]) == 3 for _, _, c in records)
    assert any(c["positive"] and len(c["colors"]) > 3 and len(c["placements"]) > len(c["colors"])
               for _, _, c in records), "Extras and repetition must be taught"
    assert any(not c["positive"] and len(c["colors"]) > 3 and _unblocked(i, c)
               for i, _, c in records), "Color-count shortcut"
    assert any(all(any(not c["positive"] and c["colors"] & key["query"] == key["query"]-{q}
                           and _unblocked(i, c)
                           for c in _chains(i, key)) for q in key["query"])
               for i in train_infos for key in i["keys"]), "Each key identity must matter"
    assert any(any(info["pieces"][i][1] == info["pieces"][j][1]
                   for i in range(len(info["pieces"])) if used & (1 << i)
                   for j in range(len(info["pieces"])) if not used & (1 << j))
               for info in train_infos
               for used in [sum(c["bits"] for c in info["chosen"].values())]), "Color classes are not objects"
    training_structures = {_structural_signature(i) for i in train_infos}
    assert all(_structural_signature(i) not in training_structures for i in test_infos), "Structural test shift"


def self_test():
    shapes, structures, roles, widths, counts, sides, sockets = set(), set(), set(), set(), set(), set(), set()
    puzzles = []
    query_swaps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for pair in puzzle["train"]+puzzle["test"]:
            out, info = _solve_info(pair["input"])
            assert np.array_equal(out, pair["output"])
            shapes.add(out.shape)
            mapping, normalized = {}, []
            for value in pair["input"].flat:
                normalized.append(mapping.setdefault(int(value), len(mapping)))
            structures.add((out.shape, tuple(normalized)))
            roles.add((info["background"], info["band"]))
            sockets.add(len(info["keys"]))
            for k, chain in info["chosen"].items():
                widths.add(info["keys"][k]["width"])
                counts.add(len(chain["placements"]))
                sides.add((info["axis"], info["keys"][k]["side"]))
            # Query positions and order carry no spatial meaning.
            moved = info["view"].copy()
            for key in info["keys"]:
                row, start, width = key["row"], key["start"], key["width"]
                moved[row, start:start+width] = np.roll(moved[row, start:start+width], 1)
            predicted = solve(moved.T if info["axis"] else moved)
            expected = pair["output"].T.copy() if info["axis"] else pair["output"].copy()
            expected[info["keymask"]] = moved[info["keymask"]]
            assert np.array_equal(predicted, expected.T if info["axis"] else expected)
            if len(info["keys"]) == 2:
                a, b = info["keys"]
                if a["side"] == b["side"] and a["width"] == b["width"] and a["query"] != b["query"]:
                    changed = info["view"].copy()
                    ar, ac, aw = a["row"], a["start"], a["width"]
                    br, bc = b["row"], b["start"]
                    changed[ar, ac:ac+aw], changed[br, bc:bc+aw] = (changed[br, bc:bc+aw].copy(), changed[ar, ac:ac+aw].copy())
                    try:
                        swapped = solve(changed.T if info["axis"] else changed)
                    except ValueError:
                        pass  # Arbitrary free-scene keys need not be exchangeable.
                    else:
                        canonical = swapped.T if info["axis"] else swapped
                        assert not np.array_equal(canonical[~info["keymask"]], expected[~info["keymask"]])
                        query_swaps += 1
        if seed < 3:
            other = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in puzzle for a, b in zip(puzzle[s], other[s]) for k in a)
    assert len(shapes) >= 25 and len(structures) >= 200
    assert len(roles) >= 35 and widths >= {3, 4, 5, 6}
    assert counts >= {3, 4, 5} and sockets >= {1, 2, 3}
    assert sides == {(0, -1), (0, 1), (1, -1), (1, 1)} and query_swaps >= 42
    # An explicit member outside the sampler's width/count range establishes
    # that solve does not depend on those construction schedules.
    fixture_rng = np.random.default_rng(61017)
    for attempt in range(20):
        cuts = _profiles(fixture_rng, 7, 6, 1)
        if cuts[-1][0] <= 13:
            break
    else:
        raise AssertionError("Extended fixture did not fit")
    base = np.zeros((30, 30), dtype=np.int8)
    base[12:17] = 8
    base[15, 18:25] = 0
    base[15, [18, 21, 24]] = [1, 2, 3]
    for c in range(7):
        base[17:17+cuts[0][c], 18+c] = 8
    tiles, targetmask = [], np.zeros_like(base, dtype=bool)
    for lo, hi, color in zip(cuts, cuts[1:], [1, 2, 3, 4, 4, 1]):
        cells = np.array([(17+u, 18+c) for c in range(7)
                          for u in range(int(lo[c]), int(hi[c]))])
        tiles.append((cells, color, True))
        targetmask[cells[:, 0], cells[:, 1]] = True
    tiles.append((tiles[1][0].copy(), 5, False))
    inp, locations = _scatter(fixture_rng, base, tiles, 0, targetmask)
    expected = inp.copy()
    for source in locations[:6]:
        expected[source[:, 0], source[:, 1]] = 0
    for cells, color, selected in tiles:
        if selected:
            expected[cells[:, 0], cells[:, 1]] = color
    for reflected in (False, True):
        for transposed in (False, True):
            a, b = (inp[::-1], expected[::-1]) if reflected else (inp, expected)
            a, b = (a.T, b.T) if transposed else (a, b)
            assert np.array_equal(solve(a.tolist()), b)
            _validate_pair({"input": a, "output": b})
    for a in range(2, 11):
        for b in range(1, min(a-1, 4)+1):
            generate_puzzle(123, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # A cell mutation tests the semantic validator, rather than API execution.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["train"][0]["output"] = bad["train"][0]["input"].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Identity output accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "structures": len(structures),
            "role_assignments": len(roles), "key_swap_witnesses": query_swaps}


if __name__ == "__main__":
    print(self_test())
