"""Register stamped seed fragments, retain their union, and grow southeast Ls.

Only standard library and NumPy are used. All solving conventions are pixel
properties; generation coordinates, seeds and demonstrations are never queried
by solve(). See task_spec.md for the deliberately bounded stamp vocabulary.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must have dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(np.all((a >= 0) & (a <= 9)), "grid colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _background(a):
    values, counts = np.unique(a, return_counts=True)
    _require(counts.max() * 2 > a.size, "background must occupy a strict majority")
    return int(values[int(np.argmax(counts))])


def _components(a, background):
    remaining = {(int(r), int(c)) for r, c in zip(*np.nonzero(a != background))}
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, group = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    group.append(q)
        r0, c0 = min(r for r, c in group), min(c for r, c in group)
        h = max(r for r, c in group) - r0 + 1
        w = max(c for r, c in group) - c0 + 1
        _require(len(group) == h * w, "each fragment is a solid rectangular view")
        _require(min(h, w) >= 3, "fragment is too small to contain a stamp")
        result.append(a[r0:r0 + h, c0:c0 + w].copy())
    _require(2 <= len(result) <= 5, "need two to five separated fragments")
    return result


def _stamps(a):
    """An exact same-orientation 3x3 window containing all three seed inks."""
    result = {}
    for r in range(a.shape[0] - 2):
        for c in range(a.shape[1] - 2):
            patch = a[r:r + 3, c:c + 3]
            if len(set(int(x) for x in patch.flat)) != 3:
                continue
            signature = patch.tobytes()
            _require(signature not in result, "one fragment repeats a stamp at two positions")
            result[signature] = (r, c)
    return result


def _roles(seed):
    n, w = seed.shape
    _require(n == w and n in (5, 7, 9, 11, 13, 15), "assembled seed must be an odd square, 5..15")
    colors = {int(x) for x in np.unique(seed)}
    _require(len(colors) == 3, "seed must have exactly three role colors")
    corner, cross = int(seed[0, 0]), int(seed[n // 2, n // 2])
    _require(corner != cross, "corner and central cross roles must differ")
    _require(all(int(seed[r, c]) == corner for r in (0, n - 1) for c in (0, n - 1)),
             "the four corners must agree")
    m = n // 2
    _require(all(int(seed[r, c]) == cross for r, c in
                 ((m, m), (m - 1, m), (m + 1, m), (m, m - 1), (m, m + 1))),
             "central orthogonal cross must be visible")
    filler = next(iter(colors - {corner, cross}))
    return corner, cross, filler


def _register(parts):
    """A connected signature graph fixes every offset, up to global translation.

    Every repeated stamp denotes one coordinate. No maximum-overlap or
    minimum-area tie-break, orientation search, or growth-pattern fitting occurs.
    Conflicting offset evidence is rejected rather than silently discarded.
    """
    occurrences = {}
    for i, part in enumerate(parts):
        for signature, position in _stamps(part).items():
            occurrences.setdefault(signature, []).append((i, position))
    adjacency = [[] for _ in parts]
    edge_pairs = set()
    repeated = 0
    for copies in occurrences.values():
        if len(copies) < 2:
            continue
        repeated += 1
        for (i, (r, c)), (j, (u, v)) in itertools.combinations(copies, 2):
            delta = (r - u, c - v)
            adjacency[i].append((j, delta))
            adjacency[j].append((i, (-delta[0], -delta[1])))
            edge_pairs.add((i, j))
    offsets = {0: (0, 0)}
    stack = [0]
    while stack:
        i = stack.pop()
        for j, delta in adjacency[i]:
            candidate = (offsets[i][0] + delta[0], offsets[i][1] + delta[1])
            if j in offsets:
                _require(offsets[j] == candidate, "inconsistent stamp translations")
            else:
                offsets[j] = candidate
                stack.append(j)
    _require(len(offsets) == len(parts), "stamp graph is disconnected")
    cells, placed = {}, []
    for i, part in enumerate(parts):
        dr, dc = offsets[i]
        points = set()
        for r in range(part.shape[0]):
            for c in range(part.shape[1]):
                p, color = (r + dr, c + dc), int(part[r, c])
                _require(p not in cells or cells[p] == color, "unequal colors overlap")
                cells[p] = color
                points.add(p)
        placed.append(points)
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    h = max(r for r, c in cells) - r0 + 1
    w = max(c for r, c in cells) - c0 + 1
    _require(h == w and h <= 15 and len(cells) == h * w,
             "registered views must cover one complete square seed")
    seed = np.empty((h, w), dtype=np.int8)
    for (r, c), color in cells.items():
        seed[r - r0, c - c0] = color
    roles = _roles(seed)
    _stamps(seed)  # No identical stamp denotes two different seed coordinates.
    _require(all(part.size < seed.size for part in parts), "a fragment already contains the complete seed")
    contributions = [len(points - set().union(*(q for j, q in enumerate(placed) if j != i)))
                     for i, points in enumerate(placed)]
    # At least two independent fragments must supply irreplaceable seed cells.
    _require(sum(x > 0 for x in contributions) >= 2, "registration contributes no shared seed")
    normalized_offsets = [(offsets[i][0] - r0, offsets[i][1] - c0) for i in range(len(parts))]
    return seed, roles, {"offsets": normalized_offsets,
                         "contributions": contributions, "edges": sorted(edge_pairs),
                         "repeated_stamps": repeated}


def _decode(input_grid):
    a = _grid(input_grid)
    background = _background(a)
    _require(len(np.unique(a)) == 4, "input must use background and three seed inks")
    parts = _components(a, background)
    seed, roles, info = _register(parts)
    info.update(background=background, fragment_count=len(parts),
                fragment_shapes=[part.shape for part in parts], seed_side=seed.shape[0])
    return seed, roles, parts, info


def _grow(seed, roles):
    n = seed.shape[0]
    output = np.empty((2 * n, 2 * n), dtype=np.int8)
    output[:n, :n] = seed
    successor = {roles[i]: roles[(i + 1) % 3] for i in range(3)}
    state = int(seed[-1, -1])
    for k in range(n, 2 * n):
        output[:k, k] = state
        output[k, :k] = state
        state = successor[state]
        output[k, k] = state
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    seed, roles, _, _ = _decode(input_grid)
    return _grow(seed, roles)


def _sample_seed(rng, n, roles, rich):
    """Textured stamps connect registration to the retained corner/cross seed.

    The texture has no learned glyph dictionary. Three-ink windows are unique
    coordinate codes; all other windows are simply retained seed material.
    """
    for _ in range(80):
        seed = np.asarray(roles, dtype=np.int8)[rng.integers(3, size=(n, n))]
        patch = int(rng.integers(1, min(3, (n - 1) // 2) + 1)) if rich else 1
        reach = int(rng.integers(1, n // 2)) if rich else 1
        half = int(rng.integers(2)) if rich and reach > 1 else 0
        m = n // 2
        corner_mask = np.zeros((n, n), dtype=bool)
        for rs in (slice(0, patch), slice(n - patch, n)):
            for cs in (slice(0, patch), slice(n - patch, n)):
                corner_mask[rs, cs] = True
        cross_mask = np.zeros((n, n), dtype=bool)
        cross_mask[m - half:m + half + 1, m - reach:m + reach + 1] = True
        cross_mask[m - reach:m + reach + 1, m - half:m + half + 1] = True
        if np.any(corner_mask & cross_mask):
            continue
        seed[corner_mask] = roles[0]
        seed[cross_mask] = roles[1]
        try:
            _roles(seed)
            stamps = _stamps(seed)
        except AssertionError:
            continue
        if stamps:
            return seed
    raise AssertionError("could not sample unique seed stamps")


def _cut(rng, seed, target):
    n = seed.shape[0]
    rectangles = [(0, n, 0, n)]
    for _ in range(target - 1):
        choices = []
        for i, (r0, r1, c0, c1) in enumerate(rectangles):
            for axis, lo, hi in ((0, r0, r1), (1, c0, c1)):
                # Both views are proper; their observed overlap is three cells wide.
                for start in range(lo + 1, hi - 3):
                    choices.append((i, axis, start))
        _require(choices, "no further proper overlapping cut")
        i, axis, start = choices[int(rng.integers(len(choices)))]
        r0, r1, c0, c1 = rectangles[i]
        children = ([(r0, start + 3, c0, c1), (start, r1, c0, c1)] if axis == 0
                    else [(r0, r1, c0, start + 3), (r0, r1, start, c1)])
        rectangles[i:i + 1] = children
    return [seed[r0:r1, c0:c1].copy() for r0, r1, c0, c1 in rectangles]


def _shelves(order, mask):
    rows = [[order[0]]]
    for k, i in enumerate(order[1:]):
        if mask & (1 << k):
            rows.append([])
        rows[-1].append(i)
    return rows


def _pack(parts, background, rng):
    """Bounded, shuffled shelf packing with jitter and either canvas axis.

    Shelf order is unrelated to solved adjacency. A deterministic exhaustive
    fallback searches at most 2*5!*16 layouts, all in memory, without retries
    or logs on disk. No component is rotated or reflected.
    """
    count = len(parts)
    foreground = sum(p.size for p in parts)

    def feasible(order, mask, swap):
        sizes = [p.shape[::-1] if swap else p.shape for p in parts]
        rows = _shelves(order, mask)
        heights = [max(sizes[i][0] for i in row) for row in rows]
        widths = [sum(sizes[i][1] for i in row) + len(row) - 1 for row in rows]
        bh, bw = sum(heights) + len(rows) - 1, max(widths)
        if bh > 30 or bw > 30:
            return None
        canvases = [(h, w) for h in range(bh, 31) for w in range(bw, 31)
                    if h * w > 2 * foreground]
        return (sizes, rows, heights, widths, canvases, swap) if canvases else None

    selected = None
    for _ in range(60):
        order = tuple(int(x) for x in rng.permutation(count))
        selected = feasible(order, int(rng.integers(1 << (count - 1))), bool(rng.integers(2)))
        if selected is not None:
            break
    if selected is None:
        for order in itertools.permutations(range(count)):
            for mask in range(1 << (count - 1)):
                for swap in (False, True):
                    selected = feasible(order, mask, swap)
                    if selected is not None:
                        break
                if selected is not None:
                    break
            if selected is not None:
                break
    _require(selected is not None, "fragments cannot be packed within 30x30")
    sizes, rows, heights, widths, canvases, swap = selected
    h, w = canvases[int(rng.integers(len(canvases)))]
    canvas = np.full((w, h) if swap else (h, w), background, dtype=np.int8)
    slack = h - sum(heights) - len(rows) + 1
    gaps = rng.multinomial(slack, np.full(len(rows) + 1, 1 / (len(rows) + 1)))
    top = int(gaps[0])
    for k, row in enumerate(rows):
        spare = w - widths[k]
        horizontal = rng.multinomial(spare, np.full(len(row) + 1, 1 / (len(row) + 1)))
        left = int(horizontal[0])
        for j, i in enumerate(row):
            ph, pw = sizes[i]
            r = top + int(rng.integers(heights[k] - ph + 1))
            c = left
            rr, cc = (c, r) if swap else (r, c)
            ah, aw = parts[i].shape
            canvas[rr:rr + ah, cc:cc + aw] = parts[i]
            left += pw + 1 + int(horizontal[j + 1])
        top += heights[k] + 1 + int(gaps[k + 1])
    return canvas


def _make_pair(rng, n, roles, background, target, absorbed, rich, both_axes=False):
    for _ in range(60):
        try:
            seed = _sample_seed(rng, n, roles, rich)
            parts = _cut(rng, seed, target)
            reconstructed, _, info = _register(parts)
            if not all(x > 0 for x in info["contributions"]):
                continue
            if both_axes and not (len({r for r, c in info["offsets"]}) > 1
                                  and len({c for r, c in info["offsets"]}) > 1):
                continue
            if not np.array_equal(seed, reconstructed):
                continue
            if absorbed:
                # A full registration stamp becomes a wholly redundant view.
                positions = sorted(_stamps(seed).values())
                r, c = positions[int(rng.integers(len(positions)))]
                parts.append(seed[r:r + 3, c:c + 3].copy())
            inp = _pack(parts, background, rng)
            decoded, got_roles, _, final_info = _decode(inp)
            # Adding a duplicate must not accidentally make a primary view
            # redundant by covering all of its formerly exclusive cells.
            if sum(x > 0 for x in final_info["contributions"]) != target:
                continue
            if np.array_equal(decoded, seed) and got_roles == roles:
                return {"input": inp, "output": _grow(seed, roles)}
        except AssertionError:
            continue
    raise RuntimeError("bounded seed/cut/packing attempts exhausted")


def _integer(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    result, used = {"train": [], "test": []}, set()
    first_palette = tuple(int(x) for x in rng.choice(10, 4, replace=False))
    first_side = int(rng.choice((5, 7)))
    second_side = int(rng.choice((9, 11)))
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            child = np.random.default_rng(int(rng.integers(0, 2**63)))
            if split == "train" and index < 2:
                background = first_palette[0]
                base = first_palette[1:]
                roles = base if index == 0 else (base[1], base[2], base[0])
                n = first_side if index == 0 else second_side
                target, absorbed, rich = (2, False, False) if index == 0 else (4, True, True)
            else:
                palette = tuple(int(x) for x in child.choice(10, 4, replace=False))
                background, roles = palette[0], palette[1:]
                n = int(child.choice((5, 7, 9, 11) if split == "train" else (13, 15)))
                target = int(child.integers(2, 5))
                absorbed, rich = bool(child.integers(2)), bool(child.integers(2))
            for _ in range(8):
                pair = _make_pair(child, n, roles, background, target, absorbed, rich,
                                  both_axes=(split == "train" and index == 1))
                signature = _key(pair["input"])
                if signature not in used:
                    used.add(signature)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("could not create a distinct input")
    return result


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys must be input/output")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    seed, roles, parts, info = _decode(inp)
    n = seed.shape[0]
    _require(out.shape == (2 * n, 2 * n), "output must double the assembled seed side")
    _require(np.array_equal(out[:n, :n], seed), "assembled seed was not preserved")
    # Validate every new cell with a direct formula, rather than _grow().
    for r in range(2 * n):
        for c in range(2 * n):
            if r < n and c < n:
                continue
            phase = (r - n + 1) % 3 if r == c else (max(r, c) - n) % 3
            _require(int(out[r, c]) == roles[phase], "incorrect inherited arm or advanced corner")
    _require(info["background"] not in set(int(x) for x in out.flat), "canvas background entered the union")
    _require(np.array_equal(solve(inp), out), "input-only solve disagrees")
    _require(sum(p.size for p in parts) > seed.size, "there is no genuine overlap")
    # Grant the true side and role order to a no-registration alternative.
    # It still cannot recover the seed texture by co-locating fragment origins.
    naive = np.full_like(seed, roles[2])
    for part in parts:
        h, w = part.shape
        naive[:h, :w] = part
    _require(not np.array_equal(_grow(naive, roles), out), "origin coalescing bypassed registration")
    info["roles"] = roles
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys must be train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            signature = _key(pair["input"])
            _require(signature not in seen, "input repeated within puzzle")
            seen.add(signature)
            infos[split].append(info)
    # Demonstration sufficiency for generated puzzles, separate from solving.
    train = infos["train"]
    _require(len({i["seed_side"] for i in train}) >= 2, "demonstrations must distinguish doubling from fixed growth")
    _require(any(0 in i["contributions"] for i in train), "demonstrations must show duplicate absorption")
    _require(any(len({r for (r, c), amount in zip(i["offsets"], i["contributions"]) if amount > 0}) > 1
                 and len({c for (r, c), amount in zip(i["offsets"], i["contributions"]) if amount > 0}) > 1
                 for i in train), "demonstrations must show contributing views offset on both axes")
    _require(any(x in b["roles"] and b["roles"].index(x) != k
                 for a, b in itertools.combinations(train, 2) for k, x in enumerate(a["roles"])),
             "demonstrations must change literal color roles")
    _require(max(i["seed_side"] for i in train) < min(i["seed_side"] for i in infos["test"]),
             "generated tests must extrapolate seed size")


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, streamed metrics, malformed evidence and semantic ablations.

    No artifacts are written. At most one puzzle and one repacked pair are
    retained; metrics store short hashes or small integers, never grid copies.
    """
    digests, sizes, counts, shapes, backgrounds, geometries = set(), set(), set(), set(), set(), set()
    patterns, both_axes, absorbed, noncentral = set(), 0, 0, 0
    example = None
    for seed_value in range(42):
        puzzle = generate_puzzle(seed_value)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                tile, roles, parts, info = _decode(pair["input"])
                n = tile.shape[0]
                sizes.add(n)
                counts.add(len(parts))
                backgrounds.add(info["background"])
                shapes.add(pair["input"].shape)
                geometries.add(tuple(sorted((p.shape, offset) for p, offset in zip(parts, info["offsets"]))))
                normalized = np.empty_like(tile)
                for i, color in enumerate(roles):
                    normalized[tile == color] = i
                patterns.add(hashlib.sha256(normalized.tobytes()).hexdigest())
                both_axes += int(len({r for r, c in info["offsets"]}) > 1
                                 and len({c for r, c in info["offsets"]}) > 1)
                absorbed += int(0 in info["contributions"])
                _require(info["contributions"].count(0) <= 1,
                         "an absorbed stamp erased a primary view's contribution")
                noncentral += int(any(x not in (0, (n - 3) // 2) for p in info["offsets"] for x in p))
                # Omitting A returns only the registered seed and fails doubling.
                _require(_key(tile) != _key(pair["output"]), "A ablation did not change answer")
                # Omitting B cannot preserve a full seed from any one view.
                _require(all(p.size < n * n for p in parts), "B ablation found a complete seed")
                # Wrong A variants: static corners, reversed cycle, uniform Ls.
                out = pair["output"]
                _require(out[n, n] != roles[0] and out[n + 1, n + 1] != roles[0], "corner cycle invisible")
                _require(out[0, n] != out[n, n], "corner and arm colors did not differ")
                _require(out[0, n + 1] != out[0, n], "arms failed to inherit the changing corner")
        if seed_value == 0:
            example = puzzle
            _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(0)), "nondeterminism")
    _require(len(digests) == 42, "42 seeds did not produce distinct puzzles")
    _require(sizes == {5, 7, 9, 11, 13, 15} and counts == {2, 3, 4, 5}, "size/count coverage missing")
    _require(len(shapes) >= 25 and len(geometries) >= 60, "geometry variation too narrow")
    _require(len(patterns) >= 190 and backgrounds == set(range(10)), "texture or palette variation missing")
    _require(both_axes >= 42 and absorbed >= 42 and noncentral >= 20, "two-axis, absorption or uneven cuts missing")
    # Repacking and arbitrary bijective recoloring must preserve the same rule.
    pair = example["train"][1]
    tile, roles, parts, info = _decode(pair["input"])
    repacked = _pack(parts, info["background"], np.random.default_rng(19001))
    _require(np.array_equal(solve(repacked), pair["output"]), "canvas location affected answer")
    permutation = np.random.default_rng(19002).permutation(10).astype(np.int8)
    _require(np.array_equal(solve(permutation[repacked]), permutation[pair["output"]]), "literal color affected answer")
    # Lost material, wrong cycle, wrong arm, and bad dimensions are rejected.
    n = tile.shape[0]
    for point in ((0, 0), (n, n), (0, n), (0, n + 1)):
        wrong = pair["output"].copy()
        wrong[point] = (int(wrong[point]) + 1) % 10
        try:
            _check_pair({"input": pair["input"], "output": wrong})
        except AssertionError:
            pass
        else:
            raise AssertionError("damaged output accepted")
    try:
        _check_pair({"input": pair["input"], "output": pair["output"][:-1]})
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output dimensions accepted")
    # A constant/repeated stamp has unresolved coordinates, so is rejected.
    try:
        _stamps(np.tile(np.array([[1, 2, 3]], dtype=np.int8), (5, 2)))
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous repeated stamp accepted")
    # Every explicit legal count is checked by verify.py; exercise boundaries here.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(19003, nt, nq)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, nq), "counts ignored")
        validate_generated_puzzle(puzzle)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(digests),
            "seed_sides": sorted(sizes), "fragment_counts": sorted(counts),
            "input_shapes": len(shapes), "registration_geometries": len(geometries),
            "normalized_seeds": len(patterns), "background_colors": sorted(backgrounds),
            "both_axis_pairs": both_axes, "absorbed_pairs": absorbed}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
