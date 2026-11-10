"""Lost mosaic window: jointly pack a fragment and register it to its source.

The reference is visible in each input. Nothing is learned or cached by solve.
Only standard-library modules and NumPy are used; generation writes no files.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _d4(a, k):
    return np.rot90(np.fliplr(a) if k >= 4 else a, k % 4).copy()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _scene(value):
    a = _grid(value)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("the outer border must be background")
    refs, pieces = [], []
    occupied = a != bg
    for cells in _components(occupied):
        lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
        colors = np.unique(a[cells[:, 0], cells[:, 1]])
        if len(colors) == 2 and len(cells) == int(np.prod(hi - lo)):
            refs.append(a[lo[0]:hi[0], lo[1]:hi[1]].copy())
        elif len(colors) == 1:
            pieces.append((cells - lo, int(colors[0])))
        else:
            raise ValueError("objects must be a solid binary reference or monochrome tiles")
        # Separate objects even at diagonal contacts.
        own = set(map(tuple, cells))
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]
                            and occupied[p] and p not in own):
                        raise ValueError("objects need background clearance")
    if len(refs) != 1 or not 4 <= len(pieces) <= 6:
        raise ValueError("expected one reference and four to six fragments")
    rc = set(map(int, np.unique(refs[0])))
    pc = {color for _, color in pieces}
    if len(pc) != 2 or rc & pc or bg in rc | pc:
        raise ValueError("background, reference and fragment palettes must be disjoint")
    return refs[0], pieces, bg


def _assemblies(pieces):
    """All full-width, column-convex ribbon packings, in either axis.

    Pieces are never rotated. Transposing the coordinate system only lets the
    same search handle horizontal and vertical ribbons. Equal colored grids,
    including permutations of indistinguishable pieces, are deduplicated.
    """
    answers = {}
    n = len(pieces)
    area = sum(len(cells) for cells, _ in pieces)
    for axis in (0, 1):
        descriptions = []
        for cells, color in pieces:
            xy = cells if axis == 0 else cells[:, ::-1]
            width = int(xy[:, 1].max()) + 1
            lower, upper = [], []
            for c in range(width):
                rows = np.sort(xy[xy[:, 1] == c, 0])
                if not len(rows) or not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
                    break
                lower.append(int(rows[0]))
                upper.append(int(rows[-1]) + 1)
            else:
                descriptions.append((tuple(lower), tuple(upper), xy, color))
                continue
            break
        if len(descriptions) != n or len({len(d[0]) for d in descriptions}) != 1:
            continue
        width = len(descriptions[0][0])
        if area % width or not 1 <= area // width <= 30:
            continue
        height = area // width

        def visit(profile, used, order):
            if len(order) == n:
                if profile != (height,) * width:
                    return
                mosaic = np.full((height, width), -1, dtype=np.int8)
                placements = []
                for i, shift in order:
                    _, _, xy, color = descriptions[i]
                    target = xy + np.array([shift, 0])
                    assert np.all(mosaic[target[:, 0], target[:, 1]] == -1)
                    mosaic[target[:, 0], target[:, 1]] = color
                    placements.append((i, target if axis == 0 else target[:, ::-1]))
                assert np.all(mosaic >= 0)
                if axis:
                    mosaic = mosaic.T.copy()
                answers.setdefault(_key(mosaic), (mosaic, placements))
                return
            offset = min(profile)
            normalized = tuple(v - offset for v in profile)
            equivalent = set()
            for i, (lower, upper, xy, color) in enumerate(descriptions):
                if used & (1 << i) or lower != normalized:
                    continue
                identity = (lower, upper, color)
                if identity in equivalent:
                    continue
                equivalent.add(identity)
                nxt = tuple(offset + v for v in upper)
                if max(nxt) <= height:
                    visit(nxt, used | (1 << i), order + [(i, offset)])

        visit((0,) * width, 0, [])
    return list(answers.values())


def _locations(full, patch):
    h, w = patch.shape
    H, W = full.shape
    if h > H or w > W:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    rr, cc = np.nonzero(np.all(windows[:, :, 0, :] == patch[0], axis=-1))
    for r in range(1, h):
        if not len(rr):
            break
        keep = np.all(full[rr[:, None] + r, cc[:, None] + np.arange(w)] == patch[r], axis=1)
        rr, cc = rr[keep], cc[keep]
    return [(int(r), int(c)) for r, c in zip(rr, cc)]


def _candidates(reference, pieces, assemblies=None):
    assemblies = _assemblies(pieces) if assemblies is None else assemblies
    reference_colors = np.unique(reference)
    fragment_colors = sorted({color for _, color in pieces})
    binary = (reference == reference_colors[1]).astype(np.int8)
    result = {}
    for k in range(8):
        full = _d4(binary, k)
        for swap in (0, 1):
            colored = np.where(full ^ swap, fragment_colors[1], fragment_colors[0]).astype(np.int8)
            witnesses = []
            for index, (mosaic, _) in enumerate(assemblies):
                if mosaic.size >= colored.size:
                    continue
                witnesses.extend((index, r, c) for r, c in _locations(colored, mosaic))
            if witnesses:
                result.setdefault(_key(colored), (colored, witnesses))
    return list(result.values())


def _independent_candidates(reference, pieces):
    """A deliberately relaxed ablation: each tile matches somewhere on its own.

    It ignores common crop, complementary boundaries, overlap and joint use.
    Generated examples require multiple different completions under this rule.
    """
    rc = np.unique(reference)
    pc = sorted({color for _, color in pieces})
    binary = (reference == rc[1]).astype(np.int8)
    answers = set()
    for k in range(8):
        for swap in (0, 1):
            full = np.where(_d4(binary, k) ^ swap, pc[1], pc[0]).astype(np.int8)
            for cells, color in pieces:
                h, w = cells.max(axis=0) + 1
                if h > full.shape[0] or w > full.shape[1]:
                    break
                valid = np.ones((full.shape[0] - h + 1, full.shape[1] - w + 1), dtype=bool)
                for r, c in cells:
                    valid &= full[r:r + valid.shape[0], c:c + valid.shape[1]] == color
                if not valid.any():
                    break
            else:
                answers.add(_key(full))
    return answers


def _relations(reference, output):
    rc, oc = np.unique(reference), np.unique(output)
    binary = (reference == rc[1]).astype(np.int8)
    return [(k, swap) for k in range(8) for swap in (0, 1)
            if np.array_equal(np.where(_d4(binary, k) ^ swap, oc[1], oc[0]), output)]


def solve(input_grid: np.ndarray) -> np.ndarray:
    reference, pieces, _ = _scene(input_grid)
    answers = _candidates(reference, pieces)
    if len(answers) != 1:
        raise ValueError("joint packing and reference matching must determine one completed image")
    return answers[0][0].copy()


def _profile(rng, width):
    # Structured teeth, ramps, steps and random walks, not random picture noise.
    mode = int(rng.integers(4))
    if mode == 0:
        p = rng.integers(0, 3, width)
    elif mode == 1:
        p = np.clip(np.cumsum(rng.integers(-1, 2, width)), -1, 1)
    elif mode == 2:
        p = np.floor(np.linspace(0, 2, width)).astype(int)
    else:
        p = np.zeros(width, dtype=int)
        a, b = sorted(map(int, rng.choice(width, 2, replace=False)))
        p[a:b + 1] = int(rng.integers(1, 3))
    p -= p.min()
    if not p.any():
        p[int(rng.integers(width))] = 1
    return p[::-1].copy() if rng.random() < .5 else p


def _mosaic(rng, count):
    width = int(rng.integers(4, 8))
    q, p, s = (_profile(rng, width) for _ in range(3))
    # Repeated interface states produce reorderable excursions. Reference
    # evidence, rather than an arbitrary tie-break, must resolve their order.
    profiles = {4: [q, q, q], 5: [q, p, q, q], 6: [q, p, q, s, q]}[count]
    if rng.random() < .5:
        profiles = profiles[::-1]
    cuts = [np.zeros(width, dtype=int)]
    for shape in profiles:
        lo = cuts[-1]
        offset = max(int(np.max(lo + 1 - shape)),
                     int(np.max(np.maximum(lo[:-1], lo[1:]) + 1
                                - np.minimum(shape[:-1], shape[1:]))))
        cuts.append(shape + offset + int(rng.integers(2)))
    cuts.append(np.full(width, int(cuts[-1].max()) + int(rng.integers(1, 3))))
    h = int(cuts[-1][0])
    if h > 17:
        raise ValueError("mosaic exceeds the size preference")
    patch = np.zeros((h, width), dtype=np.int8)
    pieces = []
    first = int(rng.integers(2))
    for i, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
        cells = np.array([(r, c) for c in range(width) for r in range(int(lo[c]), int(hi[c]))], dtype=int)
        color = (first + i) % 2
        patch[cells[:, 0], cells[:, 1]] = color
        pieces.append((cells - cells.min(axis=0), color))
    return patch, pieces


def _reference(rng, patch):
    h, w = patch.shape
    H = h + int(rng.integers(2, 6))
    W = w + int(rng.integers(5, 10))
    full = np.zeros((H, W), dtype=np.int8)
    # A composition of thick bars, elbows, blocks and combs. Broad uniform
    # regions deliberately permit misleading individual-fragment matches.
    for _ in range(int(rng.integers(5, 11))):
        r, c = int(rng.integers(H)), int(rng.integers(W))
        rh = int(rng.integers(1, min(7, H - r) + 1))
        cw = int(rng.integers(1, min(8, W - c) + 1))
        full[r:r + rh, c:c + cw] = int(rng.integers(2))
        if rng.random() < .5:
            full[r:r + rh, c] = 1
            full[r + rh - 1, c:c + cw] = 1
    # At a random location, supply a broad ink region as well as open space.
    rh, cw = min(7, H), min(7, W)
    r, c = int(rng.integers(H - rh + 1)), int(rng.integers(W - cw + 1))
    full[r:r + rh, c:c + cw] = 1
    r, c = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
    full[r:r + h, c:c + w] = patch
    return full


def _scatter(rng, reference, pieces, bg):
    objects = [(np.argwhere(np.ones(reference.shape, dtype=bool)), reference)]
    for cells, color in pieces:
        h, w = cells.max(axis=0) + 1
        tile = np.full((h, w), bg, dtype=np.int8)
        tile[cells[:, 0], cells[:, 1]] = color
        objects.append((cells, tile))
    # A compact canvas when possible, with bounded retries at full size.
    H = max(reference.shape[0] + 4, int(rng.integers(24, 31)))
    W = max(reference.shape[1] + 4, int(rng.integers(24, 31)))
    H, W = min(H, 30), min(W, 30)
    for attempt in range(24):
        if attempt >= 12:
            H = W = 30
        scene = np.full((H, W), bg, dtype=np.int8)
        forbidden = np.zeros((H, W), dtype=bool)
        forbidden[[0, -1], :] = True
        forbidden[:, [0, -1]] = True
        order = [0] + list(rng.permutation(np.arange(1, len(objects))))
        for i in order:
            cells, tile = objects[int(i)]
            h, w = tile.shape
            valid = np.ones((H - h + 1, W - w + 1), dtype=bool)
            for r, c in cells:
                valid &= ~forbidden[r:r + valid.shape[0], c:c + valid.shape[1]]
            locations = np.argwhere(valid)
            if not len(locations):
                break
            origin = locations[int(rng.integers(len(locations)))]
            target = cells + origin
            scene[target[:, 0], target[:, 1]] = tile[cells[:, 0], cells[:, 1]]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    forbidden[target[:, 0] + dr, target[:, 1] + dc] = True
        else:
            return scene
    raise ValueError("insufficient scattering clearance")


def _make_pair(rng, count, relative, swap):
    for _ in range(160):
        try:
            patch, tiles = _mosaic(rng, count)
        except ValueError:
            continue
        ref = _reference(rng, patch)
        if len({_key(_d4(ref, k) ^ s) for k in range(8) for s in (0, 1)}) != 16:
            continue
        display_orientation = int(rng.integers(8))
        display = _d4(ref, display_orientation)
        # First orient the displayed source, then apply the demonstrated
        # relative isometry. A tile's position is discarded, its mask is not.
        transformed = []
        for cells, color in tiles:
            mask = np.zeros(patch.shape, dtype=np.int8)
            mask[cells[:, 0], cells[:, 1]] = 1
            xy = np.argwhere(_d4(_d4(mask, display_orientation), relative))
            transformed.append((xy - xy.min(axis=0), color))
        assemblies = _assemblies(transformed)
        if len(assemblies) < 2:
            continue
        palette = rng.choice(10, 5, replace=False).astype(np.int8)
        palette[1:3] = np.sort(palette[1:3])
        palette[3:5] = np.sort(palette[3:5])
        bg = int(palette[0])
        source = np.where(display, palette[2], palette[1]).astype(np.int8)
        payload = [(cells, int(palette[3 + (color ^ swap)])) for cells, color in transformed]
        actual = np.where(_d4(display, relative), palette[3 + (1 ^ swap)], palette[3 + swap]).astype(np.int8)
        packed = _assemblies(payload)
        candidates = _candidates(source, payload, packed)
        if len(candidates) != 1 or not np.array_equal(candidates[0][0], actual):
            continue
        if len({i for i, _, _ in candidates[0][1]}) == len(packed):
            continue
        if len(_independent_candidates(source, payload)) < 2:
            continue
        try:
            inp = _scatter(rng, source, payload, bg)
        except ValueError:
            continue
        return {"input": inp, "output": actual}
    raise RuntimeError("bounded construction could not find an identifying mosaic")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is outside the legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    train = [(4, 0, 0), (5, 1, 1)]
    if nt >= 3:
        train.append((int(rng.integers(4, 6)), 4, int(rng.integers(2))))
    train += [(int(rng.integers(4, 7)), int(rng.integers(8)), int(rng.integers(2)))
              for _ in range(nt - len(train))]
    rng.shuffle(train)
    test = [(6 if j == 0 else int(rng.integers(5, 7)),
             int(rng.integers(4 if nt == 2 else 8)), int(rng.integers(2))) for j in range(nq)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, schedule in (("train", train), ("test", test)):
        for n, orientation, swap in schedule:
            for _ in range(8):
                pair = _make_pair(rng, n, orientation, swap)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    reference, pieces, bg = _scene(inp)
    assemblies = _assemblies(pieces)
    assert len(assemblies) >= 2, "geometry alone must leave a real colored ambiguity"
    answers = _candidates(reference, pieces, assemblies)
    assert len(answers) == 1 and np.array_equal(answers[0][0], out)
    assert len({i for i, _, _ in answers[0][1]}) < len(assemblies), "reference must rule out a packing"
    relaxed = _independent_candidates(reference, pieces)
    assert _key(out) in relaxed and len(relaxed) >= 2, "individual matches must be insufficient"
    assert sorted(out.shape) == sorted(reference.shape)
    relations = _relations(reference, out)
    assert len(relations) == 1, "source must distinguish orientation and role assignment"
    assert set(map(int, np.unique(out))) == {color for _, color in pieces}
    assert out.size > sum(len(cells) for cells, _ in pieces)
    # Explicitly audit exact cover and translation, independently of label equality.
    index, r, c = answers[0][1][0]
    mosaic, placements = assemblies[index]
    assert np.array_equal(out[r:r + mosaic.shape[0], c:c + mosaic.shape[1]], mosaic)
    cover = np.zeros(mosaic.shape, dtype=int)
    used = set()
    for i, target in placements:
        assert i not in used
        used.add(i)
        cells, color = pieces[i]
        assert np.array_equal(target - target.min(axis=0), cells)
        assert np.all(mosaic[target[:, 0], target[:, 1]] == color)
        cover[target[:, 0], target[:, 1]] += 1
    assert used == set(range(len(pieces))) and np.all(cover == 1)
    return {"reference_shape": reference.shape, "crop_shape": mosaic.shape,
            "pieces": len(pieces), "assemblies": len(assemblies),
            "orientation": relations[0][0], "swap": relations[0][1],
            "relaxed_answers": len(relaxed), "palette": (bg, *sorted(np.unique(reference)), *sorted(np.unique(out))),
            "reference_class": min(_key(_d4((reference == np.max(reference)).astype(np.int8), k) ^ s)
                                   for k in range(8) for s in (0, 1))}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen = set()
    counts = {"train": set(), "test": set()}
    relations = {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            features = _validate_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            counts[split].add(features["pieces"])
            relations[split].add((features["orientation"], features["swap"]))
    assert {4, 5} <= counts["train"] and 6 in counts["test"]
    assert {(0, 0), (1, 1)} <= relations["train"]
    if nt == 2:
        assert all(k < 4 for k, _ in relations["test"])
    else:
        assert any(k == 4 for k, _ in relations["train"])


def self_test():
    """42 seeds, streamed; no files, traces, or retained puzzle corpus."""
    structures, references, shapes, crops, palettes, counts, ambiguity = (set() for _ in range(7))
    pairs = 0
    digest = hashlib.sha256()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _validate_pair(pair)
                inp = pair["input"]
                roles, norm = {}, []
                for value in inp.flat:
                    norm.append(roles.setdefault(int(value), len(roles)))
                structures.add((inp.shape, bytes(norm)))
                references.add(f["reference_class"])
                shapes.add(inp.shape)
                crops.add(f["crop_shape"])
                palettes.add(tuple(map(int, f["palette"])))
                counts.add(f["pieces"])
                ambiguity.add((f["assemblies"], f["relaxed_answers"]))
                digest.update(inp.tobytes())
                digest.update(pair["output"].tobytes())
                assert np.array_equal(solve(inp), pair["output"])
                pairs += 1
    assert len(structures) == pairs
    assert len(references) >= 190 and len(shapes) >= 20 and len(crops) >= 25
    assert len(palettes) >= 190 and counts == {4, 5, 6} and len(ambiguity) >= 4
    first, again = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], again[s]) for k in a)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1200 + 10 * nt + nq, nt, nq)
            validate_generated_puzzle(p)
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    out = broken["test"][0]["output"]
    colors = np.unique(out)
    out[0, 0] = colors[int(out[0, 0] == colors[0])]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    return {"seeds": 42, "pairs": pairs, "input_structures": len(structures),
            "reference_classes": len(references), "input_shapes": len(shapes), "crop_shapes": len(crops),
            "role_palettes": len(palettes), "piece_counts": sorted(counts),
            "ambiguity_profiles": sorted(ambiguity), "corpus_sha256": digest.hexdigest()}
