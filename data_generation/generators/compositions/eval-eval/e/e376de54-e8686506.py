"""Assemble a keyed rack; the destination's middle key calibrates all stems.

Only visible geometry is used by solve. The sampler builds labeled examples
directly and then checks their recoverability by the input parser and exact fit.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, comp = [start], {start}
        while stack:
            r, c = stack.pop()
            for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if cell in remaining:
                    remaining.remove(cell)
                    comp.add(cell)
                    stack.append(cell)
        result.append(comp)
    return result


def _key_type(key):
    return (key["head"], key["color"], key["length"])


def _fit_heads(rack, background, ports, keys):
    """Exact use of each oriented head, registered at exactly one collar.

    Answers are ordered visible key types, including their raw stem lengths.
    Permutations of completely identical keys do not create distinct answers.
    There are at most seven keys, hence a bounded, small permutation search.
    """
    holes = {tuple(map(int, cell)) for cell in np.argwhere(rack == background)}
    if len(holes) != sum(len(key["head"]) for key in keys):
        return [], []
    positions = {cell: i for i, cell in enumerate(sorted(holes))}
    choices = []
    for col in ports:
        moves = []
        for i, key in enumerate(keys):
            cells = {(rack.shape[0] - 1 + dr, col + dc) for dr, dc in key["head"]}
            if cells <= holes:
                moves.append((i, sum(1 << positions[cell] for cell in cells)))
        choices.append(moves)
    answers = {}
    assignment = [-1] * len(ports)
    full_mask = (1 << len(holes)) - 1

    def visit(used_keys, used_cells):
        if len(answers) >= 2:
            return
        unfilled = [p for p, i in enumerate(assignment) if i < 0]
        if not unfilled:
            if used_cells == full_mask:
                signature = tuple(_key_type(keys[i]) for i in assignment)
                answers.setdefault(signature, tuple(assignment))
            return
        possible = {}
        for p in unfilled:
            possible[p] = [(i, mask) for i, mask in choices[p]
                           if not (used_keys >> i & 1) and not (used_cells & mask)]
        port = min(unfilled, key=lambda p: (len(possible[p]), p))
        seen_types = set()
        for i, mask in possible[port]:
            signature = _key_type(keys[i])
            if signature in seen_types:
                continue
            seen_types.add(signature)
            assignment[port] = i
            visit(used_keys | (1 << i), used_cells | mask)
            assignment[port] = -1

    visit(0, 0)
    return list(answers.values()), choices


def _canonical_candidates(grid, background):
    """Canonical view: a framed rack with its collar ports on the bottom face."""
    for scaffold in sorted(set(map(int, np.unique(grid))) - {background}):
        rr, cc = np.where(grid == scaffold)
        r, c = int(rr.min()), int(cc.min())
        h, w = int(rr.max()) - r + 1, int(cc.max()) - c + 1
        if h < 4 or w < 7:
            continue
        rack = grid[r:r + h, c:c + w]
        if not (np.all(rack[0] == scaffold) and np.all(rack[:, 0] == scaffold)
                and np.all(rack[:, -1] == scaffold)):
            continue
        ports = [int(x) for x in np.flatnonzero(rack[-1] != scaffold)]
        if len(ports) not in (3, 5, 7) or min(np.diff(ports)) < 2:
            continue
        collar_values = set(map(int, rack[-1, ports]))
        if len(collar_values) != 1 or background in collar_values:
            continue
        collar = collar_values.pop()
        if set(map(int, np.unique(rack))) != {background, scaffold, collar}:
            continue
        if np.count_nonzero(rack == collar) != len(ports):
            continue
        if any(rack[-2, p] != background for p in ports):
            continue
        outside = grid != background
        outside[r:r + h, c:c + w] = False
        comps = _components(tuple(map(int, cell)) for cell in np.argwhere(outside))
        if len(comps) != len(ports):
            continue
        keys, valid = [], True
        for comp in comps:
            dots = [cell for cell in sorted(comp) if grid[cell] == collar]
            if len(dots) != 1:
                valid = False
                break
            ar, ac = dots[0]
            payload = comp - {dots[0]}
            colors = {int(grid[cell]) for cell in payload}
            if len(colors) != 1 or colors & {background, scaffold, collar}:
                valid = False
                break
            relative = {(x - ar, y - ac) for x, y in payload}
            head = {cell for cell in relative if cell[0] < 0}
            stem = {cell for cell in relative if cell[0] > 0}
            length = len(stem)
            if (relative != head | stem or length < 1
                    or stem != {(step, 0) for step in range(1, length + 1)}
                    or (-1, 0) not in head or len(_components(head)) != 1
                    or len({y for _, y in head}) < 2 or min(x for x, _ in head) > -2):
                valid = False
                break
            keys.append({"head": tuple(sorted(head)), "color": colors.pop(),
                         "length": length, "source_collar": (ar, ac)})
        if not valid:
            continue
        assignments, choices = _fit_heads(rack, background, ports, keys)
        if len(assignments) != 1:
            continue
        order = assignments[0]
        reference = keys[order[len(ports) // 2]]["length"]
        if reference < 2 or h + reference > 30:
            continue
        yield {"rack": rack.copy(), "bbox": (r, c, h, w), "ports": ports,
               "keys": keys, "order": order, "reference": reference,
               "background": background, "scaffold": scaffold, "collar": collar,
               "choices": choices}


def _analyze(input_grid):
    inp = _grid(input_grid)
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[np.argmax(counts)])
    _require(np.count_nonzero(inp == background) > inp.size / 2,
             "background must occupy more than half the input")
    interpretations = []
    for turns in range(4):
        for candidate in _canonical_candidates(np.rot90(inp, turns), background):
            candidate["turns"] = turns
            interpretations.append(candidate)
    _require(len(interpretations) == 1, "expected one framed rack and one key assignment")
    return interpretations[0]


def _answer(a, length=None):
    rack, ports, keys, order = a["rack"], a["ports"], a["keys"], a["order"]
    reference = a["reference"] if length is None else int(length)
    h, w = rack.shape
    _require(1 <= reference and h + reference <= 30, "invalid target extent")
    out = np.full((h + reference, w), a["background"], dtype=np.int8)
    out[:h] = rack
    for port, i in zip(ports, order):
        key = keys[i]
        for dr, dc in key["head"]:
            out[h - 1 + dr, port + dc] = key["color"]
        out[h:, port] = key["color"]
    return np.rot90(out, -a["turns"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover head placement, then read the middle destination's visible length."""
    return _answer(_analyze(input_grid))


def _random_head(rng, depth, half_width):
    for _ in range(80):
        cells = {(-1, 0)}
        size = int(rng.integers(3, min(14, depth * (2 * half_width + 1)) + 1))
        while len(cells) < size:
            neighbors = sorted({(r + dr, c + dc) for r, c in cells
                                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                                if -depth <= r + dr <= -1
                                and -half_width <= c + dc <= half_width} - cells)
            cells.add(neighbors[int(rng.integers(len(neighbors)))])
        if min(r for r, _ in cells) <= -2 and len({c for _, c in cells}) >= 2:
            return tuple(sorted(cells))
    raise RuntimeError("head construction failed")


def _layout(rng, count, depth, witness, rich):
    for _ in range(120):
        gaps = [2 if rich or witness else int(rng.integers(3, 5)) for _ in range(count - 1)]
        if witness:
            gaps[0 if rng.integers(2) else -1] += 5 if count == 5 else 7
        elif rich and rng.random() < .75:
            gaps[int(rng.integers(count - 1))] += int(rng.integers(1, 4))
        half_width = 2 if count == 3 else 1
        ports = [half_width + 1]
        for gap in gaps:
            ports.append(ports[-1] + gap)
        w = ports[-1] + half_width + 2
        if w > 30:
            continue
        h = depth + 2
        heads = [None] * count
        if witness:
            eligible = [i for i, gap in enumerate(gaps) if gap == 2]
            p = int(rng.choice(eligible))
            sign = 1 if rng.integers(2) else -1
            small = ((-2, 0), (-2, sign), (-1, 0))
            large = small + ((-1, sign),)
            heads[p], heads[p + 1] = tuple(sorted(small)), tuple(sorted(large))
        occupied = set()
        for i, head in enumerate(heads):
            if head is not None:
                occupied.update((h - 1 + dr, ports[i] + dc) for dr, dc in head)
        success = True
        for i in rng.permutation(count):
            i = int(i)
            if heads[i] is not None:
                continue
            for _ in range(100):
                head = _random_head(rng, depth, half_width)
                cells = {(h - 1 + dr, ports[i] + dc) for dr, dc in head}
                if head not in heads and not cells & occupied:
                    heads[i] = head
                    occupied.update(cells)
                    break
            else:
                success = False
                break
        if success:
            return h, w, ports, heads
    raise RuntimeError("could not construct head layout")


def _lengths(rng, count, reference, mode):
    result = [reference] * count
    other = [int(i) for i in rng.permutation(count) if i != count // 2]
    short = lambda: int(rng.integers(1, reference))
    long = lambda: int(rng.integers(reference + 1, 15))
    if mode == "witness":
        # A majority of shorter keys makes the numerical median wrong.
        for i in other:
            result[i] = short()
        result[other[0]] = long()
    elif mode in ("one_extend", "one_trim"):
        result[other[0]] = short() if mode == "one_extend" else long()
    else:
        for i in other:
            if rng.random() < .22:
                continue
            result[i] = short() if mode == "extend" or (mode == "mixed" and rng.random() < .5) else long()
        if mode == "mixed":
            result[other[0]], result[other[1]] = short(), long()
        elif all(x == reference for x in result):
            result[other[0]] = short() if mode == "extend" else long()
    return result


def _scatter(rng, rack, keys, background, collar, reference, source_witness):
    """Separate connected keys in a loose band, with independent vertical offsets.

    The band may lie above or below the rack. Its order is an unrelated random
    permutation, and no common source baseline is required.
    """
    h, w = rack.shape
    for _ in range(40):
        order = [int(x) for x in rng.permutation(len(keys))]
        if source_witness and keys[order[len(keys) // 2]]["length"] == reference:
            candidates = [p for p, i in enumerate(order) if keys[i]["length"] != reference]
            swap = int(rng.choice(candidates))
            mid = len(keys) // 2
            order[mid], order[swap] = order[swap], order[mid]
        widths = [max(c for _, c in keys[i]["head"]) - min(c for _, c in keys[i]["head"]) + 1 for i in order]
        heights = [-min(r for r, _ in keys[i]["head"]) + 1 + keys[i]["length"] for i in order]
        band_h = max(heights) + int(rng.integers(0, 4))
        minimum_w = max(w, sum(widths) + len(keys) - 1)
        below = bool(rng.random() < .72 or band_h + 1 + h + reference > 30)
        minimum_h = band_h + h + 1 + (0 if below else reference)
        if minimum_h > 30:
            continue
        ch = int(rng.integers(max(18, minimum_h), 31))
        cw = int(rng.integers(minimum_w, 31))
        extra_h = ch - minimum_h
        before = int(rng.integers(extra_h + 1))
        if below:
            br, band_top = before, before + h + 1
        else:
            band_top, br = before, before + band_h + 1
        bc = int(rng.integers(cw - w + 1))
        canvas = np.full((ch, cw), background, dtype=np.int8)
        canvas[br:br + h, bc:bc + w] = rack
        spare = cw - sum(widths) - (len(keys) - 1)
        spaces = [0] * (len(keys) + 1)
        for _ in range(int(rng.integers(spare + 1))):
            spaces[int(rng.integers(len(spaces)))] += 1
        left = spaces[0] + int(rng.integers(spare - sum(spaces) + 1))
        for p, i in enumerate(order):
            key = keys[i]
            depth = -min(r for r, _ in key["head"])
            ar = band_top + int(rng.integers(band_h - heights[p] + 1)) + depth
            ac = left - min(c for _, c in key["head"])
            canvas[ar, ac] = collar
            for dr, dc in key["head"]:
                canvas[ar + dr, ac + dc] = key["color"]
            canvas[ar + 1:ar + 1 + key["length"], ac] = key["color"]
            left += widths[p] + 1 + spaces[p + 1]
        if np.count_nonzero(canvas == background) > canvas.size / 2:
            return canvas
    raise RuntimeError("could not scatter keys")


def _generate_pair(rng, count, reference, depth, turns, palette, witness=False,
                   monochrome=False, rich=False, mode="mixed"):
    background, scaffold, collar = map(int, palette[:3])
    for _ in range(160):
        h, w, ports, heads = _layout(rng, count, depth, witness, rich)
        lengths = _lengths(rng, count, reference, "witness" if witness else mode)
        if monochrome:
            colors = [int(palette[3])] * count
        else:
            size = int(rng.integers(2, count + 1))
            colors = [int(palette[3 + i]) for i in range(size)]
            colors += [int(rng.choice(palette[3:3 + size])) for _ in range(count - size)]
            rng.shuffle(colors)
        # Repeated head inventory is legitimate when its color AND raw length
        # agree. Otherwise an identical-head exchange could change the ruler.
        if not witness and count >= 5 and rng.random() < .35:
            others = [i for i in range(count) if i != count // 2]
            i, j = map(int, rng.choice(others, 2, replace=False))
            proposed = list(heads)
            proposed[j] = heads[i]
            cells = [(h - 1 + dr, port + dc)
                     for port, head in zip(ports, proposed) for dr, dc in head]
            if len(cells) == len(set(cells)):
                heads = proposed
                colors[j], lengths[j] = colors[i], lengths[i]
        if all(length == reference for length in lengths):
            continue
        keys = [{"head": head, "color": color, "length": length}
                for head, color, length in zip(heads, colors, lengths)]
        rack = np.full((h, w), scaffold, dtype=np.int8)
        for port, key in zip(ports, keys):
            rack[-1, port] = collar
            for dr, dc in key["head"]:
                rack[h - 1 + dr, port + dc] = background
        assignments, choices = _fit_heads(rack, background, ports, keys)
        if (len(assignments) != 1 or tuple(_key_type(keys[i]) for i in assignments[0])
                != tuple(_key_type(key) for key in keys)):
            continue
        if witness and not any(len(moves) > 1 for moves in choices):
            continue
        inp = _scatter(rng, rack, keys, background, collar, reference, witness)
        # Ground truth is drawn from the sampled placements, not from solve.
        out = np.full((h + reference, w), background, dtype=np.int8)
        out[:h] = rack
        for port, key in zip(ports, keys):
            for dr, dc in key["head"]:
                out[h - 1 + dr, port + dc] = key["color"]
            out[h:, port] = key["color"]
        pair = {"input": np.rot90(inp, turns).copy(), "output": np.rot90(out, turns).copy()}
        try:
            recovered = _analyze(pair["input"])
        except ValueError:
            continue
        if np.array_equal(_answer(recovered), pair["output"]):
            return pair
    raise RuntimeError("could not generate a uniquely keyed rack")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _array_key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(map(int, a.ravel()))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    exchange = bool(rng.random() < .75)
    style, base = int(rng.integers(3)), int(rng.integers(4))
    first_count = int(rng.choice((5, 7)))
    first_reference = int(rng.integers(3, 9))
    second_reference = int(rng.choice([x for x in range(2, 10) if x != first_reference]))
    puzzle, seen = {"train": [], "test": []}, set()
    modes = ("mixed", "extend", "trim", "one_extend", "one_trim")
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            witness = split == "train" and index == 0
            mono = split == "train" and index == 1
            if witness:
                count = first_count
            elif mono:
                count = 3
            elif split == "test" and index == 0:
                count = int(rng.choice((first_count, 3), p=(.65, .35)))
            else:
                count = int(rng.choice((3, 5, 7)))
            if witness:
                reference = first_reference
            elif mono:
                reference = second_reference
            elif split == "test" and index == 0:
                # Cross the two demonstrations' count/length factors, whichever
                # familiar count is reused. The rule itself is unchanged.
                reference = first_reference if count == 3 else second_reference
            else:
                reference = int(rng.integers(2, 10))
            depth = int(rng.integers(2, 4)) if split == "train" else int(rng.integers(3, 5))
            rich = witness or (split == "test" and index == 0) or bool(rng.random() < .4)
            if style == 0:
                turns = base
            elif style == 1:
                turns = (base + (index % 2 if split == "train" else 2 + index % 2)) % 4
            else:
                turns = (base + index) % 4 if split == "train" and index < 2 else int(rng.integers(4))
            pair_palette = rng.permutation(10) if exchange else palette
            for _ in range(8):
                pair = _generate_pair(rng, count, reference, depth, turns, pair_palette,
                                      witness, mono, rich, "mixed" if mono else str(rng.choice(modes)))
                key = _array_key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
            puzzle[split].append(pair)
            seen.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    keys = [a["keys"][i] for i in a["order"]]
    lengths = tuple(k["length"] for k in keys)
    colors = tuple(k["color"] for k in keys)
    ports = a["ports"]
    distances = [abs(2 * p - ports[0] - ports[-1]) for p in ports]
    nearest = [i for i, d in enumerate(distances) if d == min(distances)]
    source = sorted(a["keys"], key=lambda k: k["source_collar"][1])
    owners = {(a["rack"].shape[0] - 1 + dr, port + dc): i
              for i, (port, key) in enumerate(zip(ports, keys)) for dr, dc in key["head"]}
    cavities = _components(owners)
    shared_cavities = sum(len({owners[cell] for cell in comp}) > 1 for comp in cavities)
    return {"count": len(keys), "reference": a["reference"], "lengths": lengths,
            "colors": colors, "gaps": tuple(np.diff(ports).tolist()),
            "heads": tuple(k["head"] for k in keys), "rack_shape": a["rack"].shape,
            "direction": a["turns"], "background": a["background"],
            "nonmedian": sorted(lengths)[len(keys) // 2] != a["reference"],
            "geometric_witness": len(nearest) == 1 and nearest[0] != len(keys) // 2
            and lengths[nearest[0]] != a["reference"],
            "source_witness": source[len(source) // 2]["length"] != a["reference"],
            "local_ambiguity": any(len(moves) > 1 for moves in a["choices"]),
            "shared_cavities": shared_cavities,
            "repeated_key": len({_key_type(key) for key in keys}) < len(keys),
            "extension": max(a["reference"] - n for n in lengths),
            "truncation": max(n - a["reference"] for n in lengths)}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _count(len(puzzle["train"]), "num_train", 2, 10)
    _count(len(puzzle["test"]), "num_test", 1, 4)
    _require(len(puzzle["train"]) > len(puzzle["test"]), "illegal split sizes")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _array_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyze(inp)
            _require(np.array_equal(out, _answer(a)), "wrong assembled/calibrated output")
            canonical = np.rot90(out, a["turns"])
            h, w = a["rack"].shape
            fixed = a["rack"] != a["background"]
            _require(np.array_equal(canonical[:h][fixed], a["rack"][fixed]), "changed rack scaffold or collars")
            _require(not np.any(canonical[:h] == a["background"]), "unfilled head cavity")
            _require(np.count_nonzero(canonical == a["collar"]) == len(a["ports"]), "incorrect collar inventory")
            head_counts = Counter()
            for item in a["keys"]:
                head_counts[item["color"]] += len(item["head"])
            for color, cells in head_counts.items():
                _require(np.count_nonzero(canonical[:h] == color) == cells, "head inventory changed")
            f = _features(a)
            _require(f["extension"] > 0 or f["truncation"] > 0, "no stem correction")
            if split == "train":
                training.append(f)
    _require(len({f["reference"] for f in training}) >= 2, "training must vary target length")
    _require(len({f["count"] for f in training}) >= 2, "training must vary port count")
    _require(any(f["nonmedian"] and f["geometric_witness"] and f["source_witness"]
                 and f["extension"] > 0 and f["truncation"] > 0 and f["local_ambiguity"]
                 and f["shared_cavities"] > 0
                 for f in training), "missing integrated training witness")
    _require(any(len(set(f["colors"])) == 1 for f in training)
             and any(len(set(f["colors"])) > 1 for f in training), "missing color-role demonstrations")


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p["input"]), _array_key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> dict:
    """Bounded 42-seed diversity and semantic checks; no files or external state."""
    signatures, shapes, heads, geometry, records = set(), set(), set(), set(), []
    rng = np.random.default_rng(20261002)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        if seed < 3:
            generate_puzzle(seed + 999)
            _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                f = _features(a)
                records.append((split, f))
                shapes.add(pair["input"].shape)
                heads.update(f["heads"])
                geometry.add((f["count"], f["reference"], f["heads"], f["gaps"], f["lengths"]))
        # Literal palettes and all four face directions must commute with solving.
        pair = puzzle["test"][0]
        perm = rng.permutation(10).astype(np.int8)
        _require(np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]]), "palette dependence")
        for turns in range(4):
            _require(np.array_equal(solve(np.rot90(pair["input"], turns)), np.rot90(pair["output"], turns)), "rotation dependence")
        _require(np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"])), "reflection dependence")
        # Ignoring A: median, geometric center, source order, and unresized rods fail.
        pair = puzzle["train"][0]
        a = _analyze(pair["input"])
        f = _features(a)
        median = sorted(f["lengths"])[f["count"] // 2]
        _require(not np.array_equal(_answer(a, median), pair["output"]), "median shortcut survived")
        source = sorted(a["keys"], key=lambda k: k["source_collar"][1])
        _require(not np.array_equal(_answer(a, source[len(source) // 2]["length"]), pair["output"]), "source-order shortcut survived")
        nearest = min(range(f["count"]), key=lambda i: abs(2 * a["ports"][i] - a["ports"][0] - a["ports"][-1]))
        _require(not np.array_equal(_answer(a, f["lengths"][nearest]), pair["output"]), "geometric-center shortcut survived")
        # Shorten just the stem on the key assigned to the middle port. Its
        # head and the entire rack stay unchanged; all output stems must change.
        canonical = np.rot90(pair["input"], a["turns"]).copy()
        central = a["keys"][a["order"][f["count"] // 2]]
        ar, ac = central["source_collar"]
        canonical[ar + central["length"], ac] = a["background"]
        altered = np.rot90(canonical, -a["turns"])
        _require(_analyze(altered)["reference"] == a["reference"] - 1,
                 "visible central length was not used")
        _require(not np.array_equal(solve(altered), pair["output"]), "central intervention had no effect")
        # Ignoring B: exchange which keyed head goes to the middle port. The reference
        # changes when the middle exchanges with any visibly different length.
        middle = len(a["ports"]) // 2
        other = next(p for p, i in enumerate(a["order"]) if a["keys"][i]["length"] != a["reference"])
        wrong_order = list(a["order"])
        wrong_order[middle], wrong_order[other] = wrong_order[other], wrong_order[middle]
        wrong_reference = a["keys"][wrong_order[middle]]["length"]
        _require(not np.array_equal(_answer(a, wrong_reference), pair["output"]), "placement did not affect calibration")
    _require(len(signatures) == 42 and len(shapes) >= 35 and len(heads) >= 70
             and len(geometry) >= 180, "insufficient structural variation")
    _require(any(len(set(f["heads"])) < f["count"] for _, f in records),
             "repeated heads were never exercised")
    for split in ("train", "test"):
        rows = [f for s, f in records if s == split]
        _require({f["count"] for f in rows} == {3, 5, 7}, "port counts missing")
        _require({f["direction"] for f in rows} == {0, 1, 2, 3}, "face direction missing")
        _require(len({f["reference"] for f in rows}) >= 7, "reference variation missing")
        _require({f["background"] for f in rows} == set(range(10)), "background variation missing")
        _require(any(f["extension"] >= 5 for f in rows) and any(f["truncation"] >= 5 for f in rows), "large corrections missing")
        _require(any(f["local_ambiguity"] for f in rows), "inventory interaction missing")
    for n in range(2, 11):
        for t in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(9000 + n * 10 + t, n, t)
            _require(len(p["train"]) == n and len(p["test"]) == t, "ignored counts")
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    # Identical-piece permutations are one assignment; a visible color change
    # in that inventory produces genuinely different assignments.
    head = ((-2, 0), (-1, 0), (-1, 1))
    rack = np.full((4, 11), 7, dtype=np.int8)
    ports = [2, 5, 8]
    for port in ports:
        rack[-1, port] = 9
        for dr, dc in head:
            rack[3 + dr, port + dc] = 0
    items = [{"head": head, "color": 1, "length": 3} for _ in ports]
    _require(len(_fit_heads(rack, 0, ports, items)[0]) == 1, "identical swaps are not equivalent")
    items[0] = dict(items[0], color=2)
    _require(len(_fit_heads(rack, 0, ports, items)[0]) == 2, "color-swap ambiguity was not detected")
    # A broken fixed cell and an erased head cell must both be detected.
    p = generate_puzzle(0)
    broken = {s: [{k: np.asarray(g).copy() for k, g in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (_grid(broken["test"][0]["output"])[0, 0] + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted corrupt output")
    a = _analyze(p["test"][0]["input"])
    canonical = np.rot90(p["test"][0]["input"], a["turns"]).copy()
    key = a["keys"][0]
    ar, ac = key["source_collar"]
    dr, dc = key["head"][0]
    canonical[ar + dr, ac + dc] = a["background"]
    try:
        solve(np.rot90(canonical, -a["turns"]))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted missing head inventory")
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "head_shapes": len(heads),
            "structural_combinations": len(geometry)}
