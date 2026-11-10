"""Registered dye behind folded dams: input-only mirror registration and flood fill.

Only the standard library and NumPy are used. Background and wall colors are
read from visible anchors, never from puzzle-wide or generation state.
"""
from collections import deque
import numpy as np


HOST_SEEDS = (0, 532508957, 993387850, 741822369, 214908982, 976322743,
              938857680, 866441012, 180130776, 229931361, 408675895, 282247014,
              817868726, 714899159, 636383000, 426669056, 391721271, 379372058,
              376093783, 285091536, 349226660, 764318793, 991722541, 142695409,
              495892478, 379508941, 768429291, 697853586, 567529690, 457214142,
              878687713, 724778602, 614812168, 681768954, 474193358, 442774845,
              749368542, 314613765, 914043912, 27951486, 430192825, 331082097)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(open_mask):
    """Four-neighbor passage, including when two wall cells meet diagonally."""
    h, w = open_mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    regions = []
    for r, c in np.argwhere(open_mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        label = len(regions)
        labels[r, c] = label
        queue = deque([(r, c)])
        cells = []
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < h and 0 <= xx < w and open_mask[yy, xx]
                        and labels[yy, xx] < 0):
                    labels[yy, xx] = label
                    queue.append((yy, xx))
        regions.append(cells)
    return labels, regions


def _decode(input_grid):
    a = _grid(input_grid)
    h, full_w = a.shape
    if full_w < 3 or full_w % 2 != 1:
        raise ValueError("two equal flanks and one central crease are required")
    w = full_w // 2
    bg, wall_color = int(a[0, 0]), int(a[0, w])
    if bg == wall_color or not np.all(a[:, w] == wall_color):
        raise ValueError("crease must be complete, uniform, and non-background")
    left, right = a[:, :w].copy(), a[:, w + 1:][:, ::-1].copy()
    for flank in (left, right):
        if any(int(flank[r, c]) != bg for r, c in ((0, 0), (0, w - 1),
                                                   (h - 1, 0), (h - 1, w - 1))):
            raise ValueError("flank corners must show the background")
    conflict = (left != bg) & (right != bg) & (left != right)
    if np.any(conflict):
        raise ValueError("unlike non-background cells may not align")
    wall = (left == wall_color) | (right == wall_color)
    sources = (left == right) & (left != bg) & (left != wall_color)
    labels, regions = _components(~wall)
    colors = []
    for i in range(len(regions)):
        dyes = np.unique(left[sources & (labels == i)])
        if len(dyes) > 1:
            raise ValueError("a connected region cannot have conflicting registered dyes")
        colors.append(int(dyes[0]) if len(dyes) else bg)
    return {"input": a, "left": left, "right": right, "background": bg,
            "wall_color": wall_color, "wall": wall, "sources": sources,
            "labels": labels, "regions": regions, "colors": colors}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fold walls by union; flood only from same-color bilateral dye matches."""
    info = _decode(input_grid)
    out = np.full(info["wall"].shape, info["wall_color"], dtype=np.int8)
    for i, color in enumerate(info["colors"]):
        out[info["labels"] == i] = color
    return out


def _count(value, name, low, high):
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    a, b = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if a is None:
        b = int(rng.integers(1, 5)) if b is None else b
        a = int(rng.integers(b + 1, 11))
    elif b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _profile(rng, height, width, style):
    low, high = 2, height - 3
    base = int(rng.integers(low, high + 1))
    if style == 0:
        return [base] * width
    if style == 1:
        base = min(base, high - 1)
        phase = int(rng.integers(4))
        return [base + (0, 1, 1, 0)[(i + phase) % 4] for i in range(width)]
    if style == 2:
        end = int(rng.integers(low, high + 1))
        return [int(round(base + (end - base) * i / (width - 1))) for i in range(width)]
    values = [base]
    for _ in range(width - 1):
        values.append(int(np.clip(values[-1] + rng.choice([-1, 0, 1]), low, high)))
    return values


def _split(rng, h, w):
    transpose = bool(rng.integers(2))
    hh, ww = (w, h) if transpose else (h, w)
    wall = np.zeros((hh, ww), dtype=bool)
    profile = _profile(rng, hh, ww, int(rng.integers(4)))
    for c, r in enumerate(profile):
        wall[r, c] = True
        if c and rng.random() < 0.45:
            lo, hi = sorted((r, profile[c - 1]))
            wall[lo:hi + 1, c] = True
    return wall.T.copy() if transpose else wall


def _room(rng, h, w, top, bottom, left, right):
    """Outline of a rectangular, chamfered, or concave connected room.

    Arguments bound the interior; an eight-neighbor rim surrounds it.
    """
    interior = np.zeros((h, w), dtype=bool)
    interior[top:bottom, left:right] = True
    rh, cw = bottom - top, right - left
    kind = int(rng.integers(3))
    if rh >= 4 and cw >= 4:
        dy, dx = int(rng.integers(1, rh - 1)), int(rng.integers(1, cw - 1))
        if kind == 1:
            interior[top:top + dy, left:left + dx] = False
        elif kind == 2:
            interior[top, left] = False
            interior[bottom - 1, right - 1] = False
    dilation = np.zeros_like(interior)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys = slice(max(0, dy), min(h, h + dy))
            xs = slice(max(0, dx), min(w, w + dx))
            dilation[ys, xs] |= interior[slice(max(0, -dy), min(h, h - dy)),
                                           slice(max(0, -dx), min(w, w - dx))]
    return dilation & ~interior


def _geometry(rng, kind):
    w = int(rng.integers(7, 15))
    h = int(rng.integers(10, 31))
    wall = np.zeros((h, w), dtype=bool)
    if kind == "split":
        wall = _split(rng, h, w)
    elif kind == "loop":
        top = int(rng.integers(2, max(3, h // 3)))
        bottom = int(rng.integers(max(top + 3, 2 * h // 3), h - 1))
        left = int(rng.integers(2, max(3, w // 3)))
        right = int(rng.integers(max(left + 3, 2 * w // 3), w - 1))
        wall = _room(rng, h, w, top, bottom, left, right)
    elif kind in ("junction", "cross"):
        middle = int(rng.integers(3, h - 3))
        profile = np.array(_profile(rng, 7, w, int(rng.integers(2)))) - 3 + middle
        profile = np.clip(profile, 2, h - 3)
        for c, r in enumerate(profile):
            wall[r, c] = True
        col = int(rng.integers(2, w - 2))
        wall[:int(profile[col]) + 1, col] = True
        if kind == "cross":
            col2 = int(rng.integers(2, w - 2))
            wall[int(profile[col2]):, col2] = True
        if rng.random() < 0.5:
            wall = wall[::-1].copy()
    elif kind in ("rooms", "compound"):
        h = max(h, 18)
        wall = np.zeros((h, w), dtype=bool)
        middle = int(rng.integers(8, h - 8 + 1))
        for top, bottom in ((2, middle - 1), (middle + 2, h - 2)):
            left = int(rng.integers(2, max(3, w // 3)))
            right = int(rng.integers(max(left + 3, 2 * w // 3), w - 1))
            wall |= _room(rng, h, w, top, bottom, left, right)
        if kind == "compound":
            wall[middle, :] = True
    elif kind == "nested":
        w, h = max(w, 11), max(h, 16)
        wall = np.zeros((h, w), dtype=bool)
        # Nested rings deliberately use simple rims so the annulus stays open.
        wall[1, 1:-1] = wall[-2, 1:-1] = True
        wall[1:-1, 1] = wall[1:-1, -2] = True
        wall[4, 4:-4] = wall[-5, 4:-4] = True
        wall[4:-4, 4] = wall[4:-4, -5] = True
    elif kind == "lattice":
        h = max(h, 15)
        wall = np.zeros((h, w), dtype=bool)
        wall[h // 3, :] = wall[2 * h // 3, :] = True
        wall[:, w // 2] = True
    else:
        raise ValueError("unknown geometry")
    labels, regions = _components(~wall)
    if len(regions) < 2 or min(map(len, regions)) < 3:
        return None
    return wall, labels, regions


def _partial_leak(wall, sources, partial_wall):
    """Can a source reach an originally unseeded open region without this half?"""
    final_labels, _ = _components(~wall)
    seeded = set(int(x) for x in final_labels[sources])
    unseeded = (~wall) & ~np.isin(final_labels, list(seeded))
    labels, _ = _components(~partial_wall)
    reached = np.isin(labels, list(set(int(x) for x in labels[sources])))
    return int(np.count_nonzero(reached & unseeded))


def _ownership(rng, wall, sources, fine):
    """Assign coherent or interleaved pieces, and require topological cooperation."""
    h, w = wall.shape
    yy, xx = np.indices((h, w))
    for _ in range(80):
        if fine:
            style = int(rng.integers(3))
            owner = ((yy + xx) % 2 if style == 0 else
                     rng.integers(0, 2, size=wall.shape) if style == 1 else
                     (yy // 2 + xx) % 2)
        else:
            style = int(rng.integers(3))
            span = int(rng.integers(2, 6))
            phase = int(rng.integers(span))
            owner = ((yy + phase) // span % 2 if style == 0 else
                     (xx + phase) // span % 2 if style == 1 else
                     (yy + xx + phase) // span % 2)
        both = rng.random(wall.shape) < float(rng.uniform(0.0, 0.35))
        a = wall & ((owner == 0) | both)
        b = wall & ((owner == 1) | both)
        if (np.any(a & ~b) and np.any(b & ~a)
                and _partial_leak(wall, sources, a) >= 3
                and _partial_leak(wall, sources, b) >= 3):
            return a, b
    return None


def _make_pair(rng, kind, seeded_count, fine, redundant):
    geom = _geometry(rng, kind)
    if geom is None:
        return None
    wall, labels, regions = geom
    h, w = wall.shape
    if len(regions) <= seeded_count:
        return None
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    candidates = [[cell for cell in cells if cell not in corners] for cells in regions]
    if min(map(len, candidates)) < 2:
        return None
    palette = [int(x) for x in rng.permutation(10)]
    bg, wc = palette[:2]
    dyes = palette[2:]
    chosen = [int(x) for x in rng.choice(len(regions), seeded_count, replace=False)]
    source_colors = {label: dyes[i] for i, label in enumerate(chosen)}
    sources = np.zeros_like(wall)
    source_values = np.full(wall.shape, bg, dtype=np.int8)
    for i, label in enumerate(chosen):
        cells = candidates[label]
        # The first source uses a non-nearest distance, exposing true reflection.
        options = [cell for cell in cells if cell[1] < w - 2
                   and 2 * cell[1] != w - 1] if i == 0 else cells
        if not options:
            return None
        first = options[int(rng.integers(len(options)))]
        source_cells = [first]
        if redundant and i == 0:
            extra = [cell for cell in cells if cell != first]
            rng.shuffle(extra)
            # Leave an eligible open cell for misleading one-sided noise even
            # in a very small seeded component.
            if len(extra) >= 2:
                source_cells += extra[:int(rng.integers(1, min(3, len(extra) - 1) + 1))]
        for r, c in source_cells:
            sources[r, c] = True
            source_values[r, c] = source_colors[label]
    ownership = _ownership(rng, wall, sources, fine)
    if ownership is None:
        return None
    aw, bw = ownership
    left = np.full(wall.shape, bg, dtype=np.int8)
    right = left.copy()
    left[aw], right[bw] = wc, wc
    left[sources] = right[sources] = source_values[sources]
    expected = np.full(wall.shape, bg, dtype=np.int8)
    expected[wall] = wc
    for label, color in source_colors.items():
        expected[labels == label] = color

    # Noise is deliberately one-sided. Include a source-colored false signal in
    # an unseeded region, and several wrong-color dots in a seeded region.
    free = [tuple(map(int, cell)) for cell in np.argwhere(~wall & ~sources)
            if tuple(map(int, cell)) not in corners]
    rng.shuffle(free)
    forced = []
    unseeded_cells = [cell for cell in free if int(labels[cell]) not in source_colors]
    if not unseeded_cells:
        return None
    forced.append((unseeded_cells[0], dyes[0]))
    seeded_cells = [cell for cell in free if int(labels[cell]) in source_colors]
    for cell in seeded_cells[:3]:
        wrong = next(c for c in dyes if c != source_colors[int(labels[cell])])
        forced.append((cell, wrong))
    forced_dict = dict(forced)
    count = min(len(free), max(len(forced), int(len(free) * rng.uniform(0.06, 0.23))))
    selected = list(forced_dict) + [cell for cell in free if cell not in forced_dict][:max(0, count - len(forced_dict))]
    for r, c in selected:
        color = forced_dict.get((r, c), int(rng.choice(dyes[:max(3, seeded_count)])))
        side = left if rng.integers(2) == 0 else right
        side[r, c] = color
    inp = np.concatenate((left, np.full((h, 1), wc, dtype=np.int8), right[:, ::-1]), axis=1)
    return {"input": inp, "output": expected}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    shift = int(rng.integers(3))
    basic = ["split", "loop", "junction", "rooms", "cross"]
    # Two anchor examples establish one and two independently registered dyes.
    plans = [(str(rng.choice(["split", "loop"])), 1, False, bool(rng.integers(2))),
             (str(rng.choice(["junction", "rooms", "cross"])), 2, False, bool(rng.integers(2)))]
    for _ in range(train_count - 2):
        kind = str(rng.choice(basic if shift == 1 else basic + ["nested", "compound", "lattice"]))
        nseeded = 1 if kind in ("split", "loop") else int(rng.integers(1, 3))
        plans.append((kind, nseeded, bool(shift != 0 and rng.integers(2)), bool(rng.integers(2))))
    rng.shuffle(plans)
    test_plans = []
    for _ in range(test_count):
        kind = str(rng.choice(["compound", "lattice"] if shift == 2 else
                              ["nested", "compound", "lattice"] if shift == 1 else
                              basic + ["nested", "compound", "lattice"]))
        nseeded = 3 if shift == 2 else 1 if kind in ("split", "loop") else int(rng.integers(1, 3))
        test_plans.append((kind, nseeded,
                           True if shift == 0 else bool(rng.integers(2)), bool(rng.integers(2))))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, split_plans in (("train", plans), ("test", test_plans)):
        for kind, nseeded, fine, redundant in split_plans:
            for _ in range(160):
                pair = _make_pair(rng, kind, nseeded, fine, redundant)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError(f"could not construct {kind} with {nseeded} seeded regions")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    info = _decode(pair["input"])
    out = _grid(pair["output"])
    wall, src = info["wall"], info["sources"]
    bg, wc = info["background"], info["wall_color"]
    left, right = info["left"], info["right"]
    labels = info["labels"]
    _require(out.shape == wall.shape, "fold dimensions")
    _require(len(info["regions"]) >= 2 and min(map(len, info["regions"])) >= 3,
             "need meaningful separated regions")
    _require(np.array_equal(out == wc, wall), "folded walls must be preserved exactly")
    seeded = [i for i, color in enumerate(info["colors"]) if color != bg]
    _require(1 <= len(seeded) < len(info["regions"]), "need seeded and unseeded regions")
    _require(np.any(src[:, :-2]), "no reflected source away from crease")
    _require(any(src[r, c] and c < src.shape[1] - 2
                 and 2 * c != src.shape[1] - 1 for r, c in np.argwhere(src)),
             "registration could be explained without horizontal reversal")
    for i, color in enumerate(info["colors"]):
        _require(np.all(out[labels == i] == color), "region label is not its registered dye")
    _require(np.array_equal(solve(info["input"]), out), "solve disagreement")
    dye_union = ((left != bg) & (left != wc)) | ((right != bg) & (right != wc))
    noise = dye_union & ~src
    unseeded = (~wall) & ~np.isin(labels, seeded)
    _require(np.any(noise & unseeded), "missing false dye in an unseeded region")
    folded = np.where(left != bg, left, right)
    _require(np.any(noise & unseeded & np.isin(folded, [info["colors"][i] for i in seeded])),
             "unseeded region needs a false copy of an authenticated dye")
    wrong_noise = noise & ~unseeded & (folded != out)
    _require(np.any(wrong_noise), "missing misleading dye inside a seeded region")
    _require(np.count_nonzero(folded != out) >= 3, "plain folding shortcut not distinguished")
    _require(_partial_leak(wall, src, left == wc) >= 3,
             "right wall pieces have no topological effect")
    _require(_partial_leak(wall, src, right == wc) >= 3,
             "left wall pieces have no topological effect")
    info["seeded_count"] = len(seeded)
    info["noise_count"] = int(noise.sum())
    info["source_count"] = int(src.sum())
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "split schema")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, train_infos = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(info["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                train_infos.append(info)
    _require(any(i["seeded_count"] == 1 for i in train_infos), "missing single-dye demonstration")
    _require(any(i["seeded_count"] >= 2 for i in train_infos), "missing independent region dyes")
    _require(len({_key(i["wall"].astype(np.int8)) for i in train_infos}) >= 2,
             "training geometry did not vary")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test():
    """Bounded diversity, ablation, counterfactual, and API checks on 42 seeds."""
    shapes, walls, palettes, signatures, seeded_counts, region_counts = set(), set(), set(), set(), set(), set()
    redundant = singleton = enclosed = edge_regions = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
        for pair in p["train"] + p["test"]:
            info = _inspect_pair(pair)
            shapes.add(info["wall"].shape)
            walls.add(_key(info["wall"].astype(np.int8)))
            palettes.add((info["background"], info["wall_color"]))
            seeded_counts.add(info["seeded_count"])
            region_counts.add(len(info["regions"]))
            for i, cells in enumerate(info["regions"]):
                n = int(np.sum(info["sources"] & (info["labels"] == i)))
                redundant += n >= 2
                singleton += n == 1
                touches = any(r in (0, info["wall"].shape[0] - 1)
                              or c in (0, info["wall"].shape[1] - 1) for r, c in cells)
                enclosed += n > 0 and not touches
                edge_regions += n > 0 and touches
            # Same union, different registration: remove only the right-hand
            # copy of every genuine dye. The union is identical, but all
            # components must now clean to background.
            a, b = info["left"].copy(), info["right"].copy()
            before_union = np.where(a != info["background"], a, b)
            b[info["sources"]] = info["background"]
            after_union = np.where(a != info["background"], a, b)
            _require(np.array_equal(before_union, after_union), "counterfactual union changed")
            changed = np.concatenate((a, np.full((a.shape[0], 1), info["wall_color"], dtype=np.int8), b[:, ::-1]), axis=1)
            predicted = solve(changed)
            _require(np.all(predicted[~info["wall"]] == info["background"]), "unregistered noise survived")
            _require(not np.array_equal(predicted, pair["output"]), "registration is unnecessary")
    _require(len(signatures) == 42 and len(walls) >= 100 and len(shapes) >= 40,
             "structural variation collapsed")
    _require(len(palettes) >= 35 and {1, 2, 3}.issubset(seeded_counts)
             and {2, 3, 4, 6}.issubset(region_counts), "role/count variation missing")
    _require(min(redundant, singleton, enclosed, edge_regions) >= 10,
             "source-placement variation missing")
    _require(_equal(generate_puzzle(873, 5, 2), generate_puzzle(873, 5, 2)), "nondeterminism")
    _require(not _equal(generate_puzzle(873), generate_puzzle(874)), "seeds ignored")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1200 + 10 * a + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "counts ignored")
    for a, b in ((None, 4), (2, None), (None, None)):
        validate_generated_puzzle(generate_puzzle(42, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, "1")):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(19)
    i = _decode(bad["train"][0]["input"])
    r, c = map(int, np.argwhere(~i["wall"])[0])
    bad["train"][0]["output"][r, c] = i["wall_color"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "output_shapes": len(shapes),
            "wall_structures": len(walls), "background_wall_roles": len(palettes),
            "seeded_region_counts": sorted(seeded_counts), "region_counts": sorted(region_counts),
            "redundant_source_regions": redundant, "single_source_regions": singleton,
            "enclosed_seeded_regions": enclosed, "edge_seeded_regions": edge_regions}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
