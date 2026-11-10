from __future__ import annotations

import numpy as np


_SOURCE_INPUTS = (
    ("000000000000", "001111001110", "001001001010", "111001111011",
     "000000000000", "000000000000", "000000000000", "000000005000",
     "000500555050", "550550555050", "550050005050", "000000000000"),
    ("000000000000", "000111100000", "000100100000", "111100111111",
     "000000000000", "000000000000", "000000000000", "000000000000",
     "000000005000", "055505505500", "055005500500", "000000000000"),
    ("000000000000", "000000000000", "001110011100", "001011110110",
     "111001100011", "000000000000", "000000000000", "000000000000",
     "000005005005", "050055505005", "555005005055", "000000000000"),
    ("000000000000", "000000000000", "000000000000", "000000000000",
     "000001111000", "000001001000", "111111001111", "000000000000",
     "000050000000", "055050000000", "055050000000", "000000000000"),
    ("000000000000", "000000000000", "011100011100", "010110010100",
     "110011110111", "000000000000", "000000000000", "000005000000",
     "050055000050", "055050055050", "005050055050", "000000000000"),
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    n = _checked_count("num_train", num_train, 2, 10)
    m = _checked_count("num_test", num_test, 1, 4)
    if n is not None and m is not None and n <= m:
        raise ValueError("num_train must exceed num_test")
    if m is None and n is None:
        m = int(rng.integers(1, 5))
    if n is None:
        n = int(rng.integers(max(2, m + 1), 11))
    if m is None:
        m = int(rng.integers(1, min(4, n - 1) + 1))
    return n, m


def _components(cells):
    remaining = set(cells)
    found = []
    while remaining:
        first = remaining.pop()
        comp = {first}
        todo = [first]
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    comp.add(q)
                    todo.append(q)
        found.append(comp)
    return found


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _rotate(cells, turns):
    result = set(cells)
    for _ in range(turns % 4):
        result = {(c, -r) for r, c in result}
    return _normalize(result)


def _fits(piece, cavity, baseline):
    # A full socket is covered at and above its mouth. All extra material is
    # below that mouth. The object can be translated and quarter-turned.
    top = min(r for r, _ in cavity)
    left = min(c for _, c in cavity)
    right = max(c for _, c in cavity)
    for turns in range(4):
        oriented = _rotate(piece, turns)
        pleft = min(c for _, c in oriented)
        pright = max(c for _, c in oriented)
        for shift in range(left - pright, right - pleft + 1):
            upper = {(r + top, c + shift) for r, c in oriented if r + top <= baseline}
            if upper == cavity:
                return True
    return False


def _fit_sockets(piece, cavities, baseline):
    return tuple(i for i, cavity in enumerate(cavities) if _fits(piece, cavity, baseline))


def _profile(rng, irregular=False):
    width = int(rng.integers(2, 5))
    if not irregular and rng.random() < 0.43:
        depth = int(rng.integers(2, 5))
        return (depth,) * width
    for _ in range(12):
        depth = int(rng.integers(2, 5))
        heights = [depth]
        for _ in range(width - 1):
            heights.append(max(1, min(4, heights[-1] + int(rng.integers(-1, 2)))))
        if len(set(heights)) > 1:
            return tuple(heights)
    return (2, 3) if width == 2 else (2, 3, 2, 3)[:width]


def _cavity(profile, baseline, start):
    return frozenset(
        (r, start + x)
        for x, depth in enumerate(profile)
        for r in range(baseline - depth + 1, baseline + 1)
    )


def _boundary(width, baseline, cavities):
    interior = set().union(*cavities)
    boundary = {(baseline, c) for c in range(width) if (baseline, c) not in interior}
    for r, c in interior:
        for q in ((r - 1, c), (r, c - 1), (r, c + 1)):
            if q not in interior:
                boundary.add(q)
    for cavity in cavities:
        columns = sorted({c for _, c in cavity})
        tops = {c: min(r for r, x in cavity if x == c) for c in columns}
        for c in (columns[0], columns[-1]):
            side = c - 1 if c == columns[0] else c + 1
            boundary.add((tops[c] - 1, side))
        for left, right in zip(columns, columns[1:]):
            shallow = left if tops[left] > tops[right] else right
            for r in range(min(tops[left], tops[right]) - 1,
                           max(tops[left], tops[right])):
                boundary.add((r, shallow))
    return boundary


def _extension(rng, cavity, baseline, kind):
    shape = set(cavity)
    if kind == "exact":
        return _normalize(shape)
    mouth = sorted(c for r, c in cavity if r == baseline)
    if kind == "stem":
        width = int(rng.integers(1, min(3, len(mouth)) + 1))
        first = int(rng.integers(0, len(mouth) - width + 1))
        lo = mouth[first]
        hi = mouth[first + width - 1]
    elif kind == "base":
        lo, hi = mouth[0], mouth[-1]
    else:
        lo = int(rng.choice(mouth))
        hi = min(mouth[-1], lo + int(rng.integers(0, 2)))
    length = int(rng.integers(1, 4))
    for j in range(1, length + 1):
        if kind == "base":
            lo -= int(rng.random() < 0.55)
            hi += int(rng.random() < 0.55)
        elif kind == "irregular" and j > 1:
            step = int(rng.integers(-1, 2))
            old_lo, old_hi = lo, hi
            lo += step
            hi += step
            if lo > old_hi:
                lo = old_hi
            if hi < old_lo:
                hi = old_lo
        shape.update((baseline + j, c) for c in range(lo, hi + 1))
    return _normalize(shape)


def _positive(rng, cavity, baseline, prefer_extension=False):
    kinds = ("stem", "base", "irregular") if prefer_extension else (
        "exact", "exact", "stem", "base", "irregular")
    kind = kinds[int(rng.integers(len(kinds)))]
    return _extension(rng, cavity, baseline, kind)


def _negative(rng, cavities, baseline):
    for _ in range(32):
        cavity = cavities[int(rng.integers(len(cavities)))]
        piece = set(_extension(rng, cavity, baseline,
                               ("exact", "stem", "base")[int(rng.integers(3))]))
        if rng.random() < 0.55:
            order = list(piece)
            rng.shuffle(order)
            for cell in order:
                trial = piece - {cell}
                if len(trial) >= 2 and len(_components(trial)) == 1:
                    piece = trial
                    break
        else:
            edge = list(piece)
            rng.shuffle(edge)
            for r, c in edge:
                neighbors = ((r - 1, c), (r, c - 1), (r, c + 1))
                q = neighbors[int(rng.integers(3))]
                if q not in piece:
                    piece.add(q)
                    break
        piece = _normalize(piece)
        if not _fit_sockets(piece, cavities, baseline):
            return piece
    # A long thin piece exceeds every generated socket in either direction.
    return frozenset((0, c) for c in range(6))


def _allocate(rng, extra, slots):
    return tuple(int(x) for x in rng.multinomial(extra, np.ones(slots) / slots))


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _pair_palette(rng, params):
    if params["stable"]:
        return params["palette"]
    colors = [int(x) for x in rng.permutation([x for x in range(10) if x != params["highlight"]])]
    return colors[0], colors[1], tuple(colors[2:5])


def _make_pair(rng, regime, index, params):
    for _ in range(70):
        n_sockets = int(rng.integers(1, 4))
        n_candidates = int(rng.integers(2, 5 if regime == "source_train" else 6))
        profiles = [_profile(rng, irregular=(regime == "source_train" and index == 1 and i == 0))
                    for i in range(n_sockets)]
        if n_sockets >= 2 and rng.random() < 0.27:
            profiles[-1] = profiles[0]
        baseline = max(max(p) for p in profiles) + int(rng.integers(1, 4))
        # Relative cavities suffice for constructing and checking pieces.
        local = [_cavity(p, baseline, 0) for p in profiles]
        if regime == "source_train" and index < 2:
            mode = "mixed"
        else:
            mode = rng.choice(("mixed", "all", "none"), p=(0.76, 0.12, 0.12))
        labels = ([True, False] if mode == "mixed" else
                  [mode == "all"] * n_candidates)
        if mode == "mixed":
            labels += [bool(rng.integers(2)) for _ in range(n_candidates - 2)]
            rng.shuffle(labels)
        shapes = []
        for j, label in enumerate(labels):
            repeated = [s for s, same in zip(shapes, labels[:j]) if same == label]
            if repeated and rng.random() < 0.3:
                shape = repeated[int(rng.integers(len(repeated)))]
            elif label:
                source = local[int(rng.integers(n_sockets))]
                shape = _positive(rng, source, baseline,
                                  prefer_extension=(regime == "source_train" and index == 1 and j == 0))
            else:
                shape = _negative(rng, local, baseline)
            shape = _rotate(shape, int(rng.integers(4)))
            shapes.append(shape)
        if any(len(_components(s)) != 1 for s in shapes):
            continue
        # Check labels against every socket, including a duplicate or larger one.
        if any(bool(_fit_sockets(s, local, baseline)) != label
               for s, label in zip(shapes, labels)):
            continue
        socket_min = 2 + sum(len(p) for p in profiles) + 2 * (n_sockets - 1)
        candidate_min = 2 + sum(max(c for _, c in s) + 1 for s in shapes) + n_candidates - 1
        minimum = max(socket_min, candidate_min)
        if minimum > 30:
            continue
        width = minimum + int(rng.integers(0, min(5, 31 - minimum)))
        so = _allocate(rng, width - socket_min, n_sockets + 1)
        x = 1 + so[0]
        sockets = []
        for i, profile in enumerate(profiles):
            sockets.append(_cavity(profile, baseline, x))
            x += len(profile) + 2 + so[i + 1]
        boundary = _boundary(width, baseline, sockets)
        co = _allocate(rng, width - candidate_min, n_candidates + 1)
        x = 1 + co[0]
        placed = []
        for i, shape in enumerate(shapes):
            top = baseline + 3 + int(rng.integers(0, 4))
            cells = {(top + r, x + c) for r, c in shape}
            placed.append(cells)
            x += max(c for _, c in shape) + 2 + co[i + 1]
        height = max(r for cells in placed for r, _ in cells) + 2 + int(rng.integers(0, 3))
        if height > 30:
            continue
        background, reference, candidate_colors = _pair_palette(rng, params)
        inp = np.full((height, width), background, dtype=np.int8)
        out = inp.copy()
        for r, c in boundary:
            inp[r, c] = reference
            out[r, c] = reference
        rich = not params["stable"] and rng.random() < 0.4
        for i, cells in enumerate(placed):
            color = candidate_colors[int(rng.integers(min(3, len(candidate_colors))))] if rich else candidate_colors[0]
            for r, c in cells:
                inp[r, c] = color
                out[r, c] = params["highlight"] if labels[i] else color
        turns = 0
        if params["rotation"] and (index == 1 or (regime == "target_test" and index == 0)
                                   or rng.random() < 0.23):
            turns = params["rotation"]
        if turns:
            inp = np.rot90(inp, turns).copy()
            out = np.rot90(out, turns).copy()
        return {"input": inp, "output": out}
    raise AssertionError("bounded scene construction failed")


def _analyze_upright(inp, out, background):
    height, width = inp.shape
    foreground = {(r, c) for r in range(height) for c in range(width) if inp[r, c] != background}
    components = _components(foreground)
    _require(len(components) >= 3, "need one boundary and at least two objects")
    boundary = max(components, key=lambda comp: max(c for _, c in comp) - min(c for _, c in comp))
    _require(min(c for _, c in boundary) == 0 and max(c for _, c in boundary) == width - 1,
             "boundary does not span canvas")
    reference = int(inp[next(iter(boundary))])
    _require(all(inp[r, c] == reference for r, c in boundary), "boundary has mixed colors")
    baseline = max(r for r, _ in boundary)
    _require(baseline >= 2, "socket band is too shallow")
    mouths = []
    run = []
    for c in range(width):
        if (baseline, c) not in boundary:
            run.append(c)
        elif run:
            mouths.append(tuple(run))
            run = []
    _require(not run and 1 <= len(mouths) <= 3, "invalid socket mouths")
    _require(all(len(m) >= 2 and m[0] > 0 and m[-1] < width - 1 for m in mouths),
             "unbounded or narrow socket mouth")
    cavities = []
    for mouth in mouths:
        seen = set()
        todo = [(baseline, c) for c in mouth]
        while todo:
            r, c = todo.pop()
            if (r, c) in seen or not (0 <= r <= baseline and 0 <= c < width) or (r, c) in boundary:
                continue
            seen.add((r, c))
            todo.extend(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        _require(len(seen) >= len(mouth) and max(r for r, _ in seen) == baseline,
                 "socket is not enclosed")
        cavities.append(frozenset(seen))
    _require(not any(cavities[i] & cavities[j] for i in range(len(cavities))
                     for j in range(i + 1, len(cavities))), "sockets join")
    _require(boundary == _boundary(width, baseline, cavities), "boundary is not a readable socket outline")
    candidates = [comp for comp in components if comp is not boundary]
    _require(2 <= len(candidates) <= 5, "invalid candidate count")
    _require(all(min(r for r, _ in comp) >= baseline + 3 for comp in candidates),
             "candidate overlaps socket band")
    infos = []
    for comp in candidates:
        colors = {int(inp[r, c]) for r, c in comp}
        _require(len(colors) == 1 and reference not in colors, "candidate colors are ambiguous")
        shape = _normalize(comp)
        fits = _fit_sockets(shape, cavities, baseline)
        changed = {int(out[r, c]) for r, c in comp if out[r, c] != inp[r, c]}
        _require(all(out[r, c] == inp[r, c] for r, c in comp) or
                 all(out[r, c] != inp[r, c] for r, c in comp), "partial recoloring")
        _require(bool(changed) == bool(fits), "candidate decision differs from socket fit")
        infos.append({"shape": shape, "area": len(comp), "fits": fits,
                      "accepted": bool(fits), "changed": changed,
                      "top": min(r for r, _ in comp), "left": min(c for _, c in comp)})
    _require(all(out[r, c] == inp[r, c] for r, c in foreground - set().union(*candidates)),
             "boundary changed")
    _require(all(out[r, c] == background for r in range(height) for c in range(width)
                 if inp[r, c] == background), "background changed")
    return {"background": background, "reference": reference,
            "baseline": baseline, "cavities": cavities, "objects": infos}


def _analyze_pair(inp, out):
    _require(inp.shape == out.shape, "input/output dimensions differ")
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    for turns in range(4):
        a, b = np.rot90(inp, turns), np.rot90(out, turns)
        try:
            info = _analyze_upright(a, b, background)
            info["orientation"] = turns
            return info
        except AssertionError:
            continue
    raise AssertionError("pair has no valid downward-open socket interpretation")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "pair sections must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid is not a 2D array")
                _require(all(1 <= dim <= 30 for dim in grid.shape), "grid dimensions invalid")
                _require(np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9)),
                         "grid colors or dtype invalid")
            inp, out = pair["input"], pair["output"]
            key = _grid_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(_source_rows(inp) not in _SOURCE_INPUTS, "source pair copied")
            info = _analyze_pair(inp, out)
            _require(len(info["objects"]) <= (4 if section == "train" else 5),
                     "regime candidate count exceeded")
            analyses[section].append(info)
    changes = [next(iter(obj["changed"])) for section in analyses.values()
               for info in section for obj in info["objects"] if obj["changed"]]
    _require(changes and len(set(changes)) == 1, "highlight color inconsistent")
    highlight = changes[0]
    _require(any(obj["accepted"] for info in analyses["train"] for obj in info["objects"]),
             "training never demonstrates highlight")
    _require(any(not obj["accepted"] for info in analyses["train"] for obj in info["objects"]),
             "training never demonstrates rejection")
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(highlight not in pair["input"], "highlight appears in input")
    for section in analyses.values():
        for info in section:
            decisions = {}
            for obj in info["objects"]:
                shape = obj["shape"]
                if shape in decisions:
                    _require(decisions[shape] == obj["accepted"],
                             "identical candidates have different decisions")
                decisions[shape] = obj["accepted"]
    # Training examples expose the complete, palette-independent fit rule.
    _require(any(any(obj["accepted"] for obj in a["objects"]) and
                 any(not obj["accepted"] for obj in a["objects"])
                 for a in analyses["train"]), "no mixed training example")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    highlight = int(rng.integers(10))
    palette_pool = [int(x) for x in rng.permutation([x for x in range(10) if x != highlight])]
    params = {"highlight": highlight, "stable": bool(rng.random() < 0.4),
              "palette": (palette_pool[0], palette_pool[1], tuple(palette_pool[2:5])),
              "rotation": int(rng.choice((0, 0, 0, 1, 2, 3)))}
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if section == "train" else "target_test"
        for index in range(count):
            for _ in range(12):
                pair = _make_pair(rng, regime, index, params)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise AssertionError("could not create a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                 for section in ("train", "test") for p in puzzle[section])


def _socket_distance(piece, cavity, baseline):
    top = min(r for r, _ in cavity)
    left, right = min(c for _, c in cavity), max(c for _, c in cavity)
    best = len(cavity) + len(piece)
    for turns in range(4):
        rotated = _rotate(piece, turns)
        for shift in range(left - max(c for _, c in rotated),
                           right - min(c for _, c in rotated) + 1):
            upper = {(r + top, c + shift) for r, c in rotated if r + top <= baseline}
            best = min(best, len(upper ^ cavity))
    return best


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack range")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack range")
    _require(len({_signature(p) for p in puzzles}) >= 40, "puzzles repeat")
    shapes, sizes, socket_counts, candidate_counts = set(), set(), set(), set()
    accepted_counts, profiles, orientations, palettes = set(), set(), set(), set()
    multiple_fit = repeated_shape = rectangular = stepped = 0
    exact = extensions = near_misses = all_fit = no_fit = 0
    one_socket_many = more_sockets = multicolor = rotated = 0
    stable_palettes = variable_palettes = test_larger = test_more = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_palettes = set()
        train_areas, train_counts = [], []
        test_areas, test_counts = [], []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_pair(pair["input"], pair["output"])
                puzzle_palettes.add(tuple(int(x) for x in np.unique(pair["input"])))
                sizes.add(pair["input"].shape)
                socket_counts.add(len(info["cavities"]))
                candidate_counts.add(len(info["objects"]))
                accepted = sum(x["accepted"] for x in info["objects"])
                accepted_counts.add(accepted)
                orientations.add(info["orientation"])
                palettes.add((info["background"], info["reference"]))
                all_fit += accepted == len(info["objects"])
                no_fit += accepted == 0
                one_socket_many += len(info["cavities"]) == 1 and accepted >= 2
                more_sockets += len(info["cavities"]) > len(info["objects"])
                multicolor += len(np.unique(pair["input"])) > 3
                rotated += info["orientation"] != 0
                if section == "train":
                    train_areas.append(max(len(c) for c in info["cavities"]))
                    train_counts.append(len(info["objects"]))
                else:
                    test_areas.append(max(len(c) for c in info["cavities"]))
                    test_counts.append(len(info["objects"]))
                for cavity in info["cavities"]:
                    profile = tuple(sum((r, c) in cavity for r in range(info["baseline"] + 1))
                                    for c in range(min(c for _, c in cavity), max(c for _, c in cavity) + 1))
                    profiles.add(profile)
                    rectangular += len(set(profile)) == 1
                    stepped += len(set(profile)) > 1
                object_shapes = [x["shape"] for x in info["objects"]]
                shapes.update(object_shapes)
                repeated_shape += len(object_shapes) != len(set(object_shapes))
                multiple_fit += any(len(x["fits"]) > 1 for x in info["objects"])
                for obj in info["objects"]:
                    rotated_shape = _rotate(obj["shape"], 1)
                    moved_shape = {(r + 7, c + 9) for r, c in obj["shape"]}
                    _require(bool(_fit_sockets(rotated_shape, info["cavities"], info["baseline"]))
                             == obj["accepted"], "quarter-turn changed a fit decision")
                    _require(bool(_fit_sockets(moved_shape, info["cavities"], info["baseline"]))
                             == obj["accepted"], "translation changed a fit decision")
                    if obj["accepted"]:
                        if obj["area"] == min(len(info["cavities"][j]) for j in obj["fits"]):
                            exact += 1
                        else:
                            extensions += 1
                    elif min(_socket_distance(obj["shape"], c, info["baseline"])
                             for c in info["cavities"]) == 1:
                        near_misses += 1
        stable_palettes += len(puzzle_palettes) == 1
        variable_palettes += len(puzzle_palettes) > 1
        test_larger += max(test_areas) > max(train_areas)
        test_more += max(test_counts) > max(train_counts)
    _require(len(sizes) >= 20 and len(shapes) >= 30 and len(profiles) >= 12,
             "geometry lacks diversity")
    _require(socket_counts == {1, 2, 3} and {2, 3, 4}.issubset(candidate_counts),
             "count diversity missing")
    _require(len(accepted_counts) >= 4 and len(orientations) >= 2 and len(palettes) >= 15,
             "selection, orientation, or palettes lack diversity")
    _require(multiple_fit > 0 and repeated_shape > 0, "shared sockets or repeated shapes absent")
    _require(min(rectangular, stepped, exact, extensions, near_misses) > 0,
             "socket, plug, extension, or near-miss variety absent")
    _require(min(all_fit, no_fit, one_socket_many, more_sockets, multicolor, rotated) > 0,
             "pair composition variety absent")
    _require(min(stable_palettes, variable_palettes, test_larger, test_more) > 0,
             "palette or test-regime variety absent")
    for n, m in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(700 + n + m, n, m)
        _require((len(p["train"]), len(p["test"])) == (n, m), "explicit counts ignored")
    for n, m in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(800 + (n or 0) + (m or 0), n, m)
        _require((n is None or len(p["train"]) == n) and
                 (m is None or len(p["test"]) == m), "partial counts ignored")
    a, b = generate_puzzle(123456, 7, 3), generate_puzzle(123456, 7, 3)
    _require(_signature(a) == _signature(b), "generator is not deterministic")
    _require(_signature(a) != _signature(generate_puzzle(123457, 7, 3)),
             "adjacent seeds coincide")
    for options in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(9, **options)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {options}")
