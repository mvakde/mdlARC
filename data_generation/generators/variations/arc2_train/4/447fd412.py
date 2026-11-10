"""Pair-local multiscale stamp completion puzzles."""

import hashlib
import numpy as np

_SOURCE_DIGESTS = {
    "47d61fa5ed8c4b1e6dc0e61be3105e19e7cd8e32dda3fcbab91145cebe1034f5",
    "2e5eef8dd9073f00beee4bfc6117fc6f233f7ce85ae99022a89e6ae00f202c9c",
    "888998ba4dcde1ae94b5942ac085fde693c33bf8932a083997d5328a933f6dc4",
    "99e7bf4ea26d661e768a445842e8612e913b340e09065978099f758319737410",
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _key(grid):
    return (grid.shape, grid.tobytes())


def _source_digest(inp, out):
    return hashlib.sha256(
        str(inp.shape).encode() + inp.astype(np.int8).tobytes()
        + out.astype(np.int8).tobytes()
    ).hexdigest()


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        result.append(frozenset(component))
    return result


def _scaled(cells, scale, origin, height, width):
    result = set()
    for r, c in cells:
        top, left = origin[0] + scale * r, origin[1] + scale * c
        for rr in range(max(0, top), min(height, top + scale)):
            for cc in range(max(0, left), min(width, left + scale)):
                result.add((rr, cc))
    return frozenset(result)


def _touches(a, b):
    return any(
        (r + dr, c + dc) in b
        for r, c in a
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    )


def _infer_input(inp):
    """Reconstruct solely from the input, including local colors and guide grouping."""
    height, width = inp.shape
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 3, "input needs three distinct color roles")
    background = int(colors[np.argmax(counts)])
    foreground = {int(x) for x in colors if int(x) != background}
    occupied = {(int(r), int(c)) for r, c in np.argwhere(inp != background)}
    components = _components(occupied)
    mixed = [comp for comp in components if {int(inp[p]) for p in comp} == foreground]
    _require(len(mixed) == 1, "one connected complete two-color prototype is required")
    prototype = mixed[0]
    external = set(occupied - prototype)
    _require(external, "at least one guide-only target is required")
    external_colors = {int(inp[p]) for p in external}
    _require(len(external_colors) == 1, "external regions must all have the guide color")
    guide = next(iter(external_colors))
    payload = next(iter(foreground - {guide}))
    guide_abs = {p for p in prototype if int(inp[p]) == guide}
    payload_abs = {p for p in prototype if int(inp[p]) == payload}
    _require(guide_abs and len(payload_abs) >= 2, "prototype lacks a useful stamp")
    top = min(r for r, _ in prototype)
    left = min(c for _, c in prototype)
    guides = frozenset((r - top, c - left) for r, c in guide_abs)
    payloads = frozenset((r - top, c - left) for r, c in payload_abs)
    guide_components = _components(guides)
    external_components = _components(external)
    lookup = {comp: index for index, comp in enumerate(external_components)}
    candidates = {}
    for ext in external_components:
        er, ec = min(r for r, _ in ext), min(c for _, c in ext)
        for unit in guide_components:
            ur, uc = min(r for r, _ in unit), min(c for _, c in unit)
            for scale in range(1, 6):
                origin = (er - scale * ur, ec - scale * uc)
                if _scaled(unit, scale, origin, height, width) != ext:
                    continue
                full_guides = _scaled(guides, scale, origin, height, width)
                if len(full_guides) != len(guides) * scale * scale:
                    continue
                indices = []
                for part in guide_components:
                    shape = _scaled(part, scale, origin, height, width)
                    if shape not in lookup:
                        break
                    indices.append(lookup[shape])
                if len(indices) != len(guide_components):
                    continue
                new_payload = _scaled(payloads, scale, origin, height, width)
                if not new_payload or any(int(inp[p]) != background for p in new_payload):
                    continue
                signature = (frozenset(indices), origin, scale)
                candidates[signature] = new_payload
    _require(candidates, "no complete scaled guide mask exists")
    by_part = [[] for _ in external_components]
    for signature in candidates:
        for index in signature[0]:
            by_part[index].append(signature)
    solutions = []
    def cover(covered, chosen):
        if len(solutions) >= 2:
            return
        if len(covered) == len(external_components):
            solutions.append(chosen)
            return
        uncovered = [i for i in range(len(external_components)) if i not in covered]
        first = min(uncovered, key=lambda i: sum(covered.isdisjoint(s[0]) for s in by_part[i]))
        for signature in by_part[first]:
            if covered.isdisjoint(signature[0]):
                cover(covered | signature[0], chosen + [signature])
    cover(frozenset(), [])
    _require(len(solutions) == 1, "guide masks have ambiguous scales or grouping")
    expected = inp.copy()
    placed = set()
    records = []
    for signature in solutions[0]:
        _, origin, scale = signature
        new_payload = candidates[signature]
        _require(placed.isdisjoint(new_payload), "target payloads overlap")
        for p in new_payload:
            expected[p] = payload
        placed.update(new_payload)
        records.append((scale, len(new_payload) < len(payloads) * scale * scale,
                        len(new_payload), origin))
    return expected, {
        "roles": (background, guide, payload),
        "guides": guides, "payloads": payloads,
        "motif_shape": (max(r for r, _ in prototype) - top + 1,
                        max(c for _, c in prototype) - left + 1),
        "records": tuple(sorted(records)),
        "guide_count": len(guides),
    }


def _analyze_pair(inp, out):
    expected, info = _infer_input(inp)
    _require(np.array_equal(expected, out), "output differs from input-derived stamping")
    return info


def _sample_motif(rng, max_height, max_width, guide_count):
    """Construct a connected stamp; payload islands may remain disconnected."""
    max_height, max_width = min(8, max_height), min(8, max_width)
    for _ in range(100):
        style = int(rng.integers(5))
        if style == 0 and max(max_height, max_width) >= 4:
            length = int(rng.integers(4, min(9, max(max_height, max_width)) + 1))
            cells = {(0, c) for c in range(length)}
            if bool(rng.integers(2)) and length <= max_height:
                cells = {(c, 0) for c in range(length)}
        elif style == 1 and max_height >= 2 and max_width >= 2:
            h = int(rng.integers(2, min(5, max_height) + 1))
            w = int(rng.integers(2, min(5, max_width) + 1))
            cells = {(r, c) for r in range(h) for c in range(w)}
        elif style == 2 and max_height >= 3 and max_width >= 3:
            h = int(rng.integers(3, min(6, max_height) + 1))
            w = int(rng.integers(3, min(6, max_width) + 1))
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h - 1) or c in (0, w - 1)}
            if bool(rng.integers(2)):
                cells.add((h // 2, w // 2))
        else:
            h = int(rng.integers(2, max_height + 1))
            w = int(rng.integers(2, max_width + 1))
            if h * w < guide_count + 2:
                continue
            goal = int(rng.integers(max(4, guide_count + 2), min(h * w, 18) + 1))
            cells = {(int(rng.integers(h)), int(rng.integers(w)))}
            while len(cells) < goal:
                frontier = sorted({(r + dr, c + dc) for r, c in cells
                                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                                   if 0 <= r + dr < h and 0 <= c + dc < w
                                   and (r + dr, c + dc) not in cells})
                if not frontier:
                    break
                cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) < guide_count + 2:
            continue
        top, left = min(r for r, _ in cells), min(c for _, c in cells)
        cells = {(r - top, c - left) for r, c in cells}
        ordered = sorted(cells)
        chosen = rng.choice(len(ordered), size=guide_count, replace=False)
        guides = frozenset(ordered[int(i)] for i in chosen)
        payloads = frozenset(cells - guides)
        return guides, payloads
    raise ValueError("motif sampling exhausted")


def _candidate_targets(rng, guides, payloads, scale, clipped, height, width):
    gminr, gmaxr = min(r for r, _ in guides), max(r for r, _ in guides)
    gminc, gmaxc = min(c for _, c in guides), max(c for _, c in guides)
    pminr, pmaxr = min(r for r, _ in payloads), max(r for r, _ in payloads)
    pminc, pmaxc = min(c for _, c in payloads), max(c for _, c in payloads)
    origins = [(r, c)
               for r in range(-scale * gminr, height - scale * (gmaxr + 1) + 1)
               for c in range(-scale * gminc, width - scale * (gmaxc + 1) + 1)
               if ((r + scale * pminr < 0 or r + scale * (pmaxr + 1) > height
                    or c + scale * pminc < 0 or c + scale * (pmaxc + 1) > width) == clipped)]
    rng.shuffle(origins)
    result = []
    for origin in origins:
        guide_cells = _scaled(guides, scale, origin, height, width)
        payload_cells = _scaled(payloads, scale, origin, height, width)
        full = len(payloads) * scale * scale
        if len(payload_cells) < max(scale * scale, int(full * 0.28)):
            continue
        result.append((guide_cells, payload_cells, scale, origin))
        if len(result) >= 150:
            break
    return result


def _construct_pair(rng, roles, shape, scales, clips, guide_count):
    height, width = shape
    max_scale = max(scales)
    max_h = min(8, max(2, height // max_scale + 1))
    max_w = min(8, max(2, width // max_scale + 1))
    for _ in range(70):
        guides, payloads = _sample_motif(rng, max_h, max_w, guide_count)
        motif_h = max(r for r, _ in guides | payloads) + 1
        motif_w = max(c for _, c in guides | payloads) + 1
        if motif_h > height or motif_w > width:
            continue
        specs = list(zip(scales, clips))
        specs.sort(key=lambda x: (-x[0], not x[1]))
        options = [_candidate_targets(rng, guides, payloads, s, clip, height, width)
                   for s, clip in specs]
        if any(not group for group in options):
            continue
        origins = [(r, c) for r in range(height - motif_h + 1)
                   for c in range(width - motif_w + 1)]
        rng.shuffle(origins)
        for origin in origins[:35]:
            pg = _scaled(guides, 1, origin, height, width)
            pp = _scaled(payloads, 1, origin, height, width)
            initial = set(pg | pp)
            node_count = [0]
            def place(index, input_cells, output_cells, selected):
                if index == len(options):
                    return selected
                for candidate in options[index]:
                    node_count[0] += 1
                    if node_count[0] > 1500:
                        return None
                    cg, cp, _, _ = candidate
                    if not (cg | cp).isdisjoint(output_cells):
                        continue
                    if _touches(cg, input_cells):
                        continue
                    found = place(index + 1, input_cells | cg,
                                  output_cells | cg | cp, selected + [candidate])
                    if found is not None:
                        return found
                return None
            chosen = place(0, initial, initial, [])
            if chosen is None:
                continue
            background, guide, payload = roles
            inp = np.full(shape, background, dtype=np.int8)
            for p in pg:
                inp[p] = guide
            for p in pp:
                inp[p] = payload
            for cg, _, _, _ in chosen:
                for p in cg:
                    inp[p] = guide
            out = inp.copy()
            for _, cp, _, _ in chosen:
                for p in cp:
                    out[p] = payload
            try:
                info = _analyze_pair(inp, out)
            except AssertionError:
                continue
            if sorted((s, clip) for s, clip, _, _ in info["records"]) == sorted(specs):
                return inp, out
    raise ValueError("pair construction exhausted")


def _valid_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    train = _valid_count("num_train", num_train, 2, 10)
    test = _valid_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + (seed // 7 + seed) % (10 - test)
    elif test is None:
        test = 1 + (seed // 5 + seed) % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _shape(rng, scales, target_count):
    max_scale = max(scales)
    for _ in range(50):
        mode = int(rng.integers(4))
        if mode == 0:
            h, w = int(rng.integers(10, 19)), int(rng.integers(10, 19))
        elif mode == 1:
            h, w = int(rng.integers(9, 17)), int(rng.integers(20, 31))
            if bool(rng.integers(2)):
                h, w = w, h
        elif mode == 2:
            h, w = int(rng.integers(19, 31)), int(rng.integers(19, 31))
        else:
            h, w = int(rng.integers(12, 31)), int(rng.integers(12, 31))
        if min(h, w) >= max(6, max_scale * 2) and h * w >= 5 * (1 + sum(s * s for s in scales)):
            return h, w
    return 30, 30


def _sample_roles(rng):
    if rng.random() < 0.08:
        return tuple(int(x) for x in rng.permutation((0, 1, 2)))
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_key = int(seed) % (1 << 64)
    train_count, test_count = _counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(seed_key)
    palette_mode = int(rng.integers(3))  # one fixed, two pair-local modes
    source_like_schedule = bool(rng.integers(4) == 0)
    no_clipping = bool(rng.integers(5) == 0)
    fixed_roles = _sample_roles(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_roles = []
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            child = np.random.default_rng(int(rng.integers(0, 2**63)))
            if palette_mode == 0:
                roles = fixed_roles
            else:
                roles = _sample_roles(child)
                if split == "train" and index == 1 and roles == train_roles[0]:
                    roles = (roles[1], roles[2], roles[0])
            if split == "train":
                train_roles.append(roles)
            if split == "train" and index == 0:
                scales = [1] * int(child.integers(1, 3))
            elif split == "train" and index == 1:
                scales = [int(child.integers(2, 4))]
                if not source_like_schedule and bool(child.integers(2)):
                    scales.append(int(child.integers(1, 4)))
            elif split == "train" and source_like_schedule:
                scales = [int(child.integers(1, 4))] * int(child.integers(1, 3))
            elif split == "test" and source_like_schedule:
                target_count = int(child.integers(2, 5))
                scales = [int(child.integers(1, 4)) for _ in range(target_count)]
                if len(set(scales)) == 1:
                    scales[-1] = 1 + scales[-1] % 3
            else:
                target_count = int(child.integers(1, 5))
                scales = [int(child.choice([1, 1, 2, 2, 3, 3, 4, 5]))
                          for _ in range(target_count)]
                if max(scales) >= 4 and target_count > 2:
                    scales = scales[:2]
            clips = [bool(not no_clipping and child.random() <
                          (0.20 if split == "train" else 0.30))
                     for _ in scales]
            guide_count = int(child.choice([1, 1, 2, 2, 3, 4, 5]))
            for attempt in range(12):
                shape = _shape(child, scales, len(scales))
                try:
                    inp, out = _construct_pair(child, roles, shape, scales, clips, guide_count)
                except ValueError:
                    continue
                if _key(inp) in seen or _source_digest(inp, out) in _SOURCE_DIGESTS:
                    continue
                seen.add(_key(inp))
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise ValueError(f"could not generate {split} pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have only train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    input_keys, pair_keys = set(), set()
    train_infos, test_infos = [], []
    for split, infos in (("train", train_infos), ("test", test_infos)):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have only input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D NumPy array")
                _require(all(1 <= d <= 30 for d in grid.shape), "grid size out of range")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "illegal ARC color")
            _require(inp.shape == out.shape and not np.array_equal(inp, out),
                     "stamp must add color without resizing")
            ik, pk = _key(inp), (_key(inp), _key(out))
            _require(ik not in input_keys and pk not in pair_keys, "duplicate pair")
            input_keys.add(ik)
            pair_keys.add(pk)
            _require(_source_digest(inp, out) not in _SOURCE_DIGESTS,
                     "exact source pair copied")
            infos.append(_analyze_pair(inp, out))
    _require(any(s == 1 for info in train_infos for s, _, _, _ in info["records"]),
             "training must show unit stamping")
    _require(any(s >= 2 for info in train_infos for s, _, _, _ in info["records"]),
             "training must show uniform enlargement")
    train_roles = {info["roles"] for info in train_infos}
    test_roles = {info["roles"] for info in test_infos}
    _require(len(train_roles) > 1 or len(test_roles) == 1 and test_roles == train_roles,
             "test palette changes must be shown in training")
    # The mask, palette, orientation, and scale are observable in each input.
    for pair in puzzle["test"]:
        predicted, _ = _infer_input(pair["input"].copy())
        _require(np.array_equal(predicted, pair["output"]),
                 "test cannot be reconstructed from its input")


def _puzzle_key(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    shapes, role_sets, motif_shapes, scale_sets, target_counts = set(), set(), set(), set(), set()
    line_count = connected_guide_count = clipped_count = unclipped_test_count = 0
    coverage = {name: 0 for name in (
        "no_gap", "long_motif", "many_guides", "wide_guides", "enclosed_payload",
        "mixed_train", "high_scale", "multi_clip", "small_clip", "single_test",
        "equal_test", "mixed_test", "combined", "varying_shape_puzzle",
        "varying_palette_puzzle", "varying_guide_puzzle", "unclipped_puzzle",
        "unclipped_test_puzzle", "smaller_test_puzzle", "fixed_palette_puzzle",
        "source_palette", "payload_islands", "hollow", "mild_clip",
        "small_clip_large_fits", "isolated_train_mixed_test",
        "train_clip_test_unclip",
    )}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)),
                 "generation is nondeterministic")
        puzzle_keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {split: [_analyze_pair(p["input"], p["output"]) for p in puzzle[split]]
                 for split in ("train", "test")}
        all_pairs = [(split, p, info) for split in ("train", "test")
                     for p, info in zip(puzzle[split], infos[split])]
        coverage["varying_shape_puzzle"] += len({p["input"].shape for _, p, _ in all_pairs}) > 1
        coverage["varying_palette_puzzle"] += len({info["roles"] for _, _, info in all_pairs}) > 1
        coverage["fixed_palette_puzzle"] += len({info["roles"] for _, _, info in all_pairs}) == 1
        coverage["varying_guide_puzzle"] += len({info["guide_count"] for _, _, info in all_pairs}) > 1
        coverage["unclipped_puzzle"] += all(not any(c for _, c, _, _ in info["records"])
                                             for _, _, info in all_pairs)
        coverage["unclipped_test_puzzle"] += all(not any(c for _, c, _, _ in info["records"])
                                                  for info in infos["test"])
        minimum_train_area = min(np.prod(p["input"].shape) for p in puzzle["train"])
        coverage["smaller_test_puzzle"] += any(np.prod(p["input"].shape) < minimum_train_area
                                                for p in puzzle["test"])
        coverage["isolated_train_mixed_test"] += (
            all(len({s for s, _, _, _ in info["records"]}) == 1 for info in infos["train"])
            and any(len({s for s, _, _, _ in info["records"]}) > 1 for info in infos["test"]))
        coverage["train_clip_test_unclip"] += (
            any(any(c for _, c, _, _ in info["records"]) for info in infos["train"])
            and any(not any(c for _, c, _, _ in info["records"]) for info in infos["test"]))
        role_sets.add(tuple(info["roles"] for split in ("train", "test")
                            for info in infos[split]))
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                shapes.add(pair["input"].shape)
                motif_shapes.add(info["motif_shape"])
                scale_sets.add(tuple(sorted(s for s, _, _, _ in info["records"])))
                target_counts.add(len(info["records"]))
                line_count += min(info["motif_shape"]) == 1
                connected_guide_count += any(len(comp) > 1 for comp in _components(info["guides"]))
                clipped_count += any(clip for _, clip, _, _ in info["records"])
                guides, payloads = info["guides"], info["payloads"]
                recs = info["records"]
                coverage["source_palette"] += set(info["roles"]) == {0, 1, 2}
                coverage["payload_islands"] += len(_components(payloads)) > 1
                h, w = info["motif_shape"]
                coverage["hollow"] += any(0 < r < h - 1 and 0 < c < w - 1
                                          and (r, c) not in guides | payloads
                                          for r in range(h) for c in range(w))
                coverage["mild_clip"] += any(clip and visible >= 0.7 * len(payloads) * scale * scale
                                              for scale, clip, visible, _ in recs)
                coverage["small_clip_large_fits"] += any(
                    scale < other and clip and not other_clip
                    for scale, clip, _, _ in recs
                    for other, other_clip, _, _ in recs)
                coverage["no_gap"] += np.prod(info["motif_shape"]) == len(guides) + len(payloads)
                coverage["long_motif"] += max(info["motif_shape"]) >= 5
                coverage["many_guides"] += len(guides) >= 4
                coverage["wide_guides"] += len(_components(guides)) > 1 and (
                    max(r for r, _ in guides) - min(r for r, _ in guides) >= 4 or
                    max(c for _, c in guides) - min(c for _, c in guides) >= 4)
                coverage["enclosed_payload"] += (
                    min(r for r, _ in payloads) >= min(r for r, _ in guides) and
                    max(r for r, _ in payloads) <= max(r for r, _ in guides) and
                    min(c for _, c in payloads) >= min(c for _, c in guides) and
                    max(c for _, c in payloads) <= max(c for _, c in guides))
                coverage["high_scale"] += any(s >= 4 for s, _, _, _ in recs)
                coverage["multi_clip"] += sum(c for _, c, _, _ in recs) >= 2
                coverage["small_clip"] += any(c and s < max(q for q, _, _, _ in recs)
                                               for s, c, _, _ in recs)
                coverage["combined"] += (any(len(comp) > 1 for comp in _components(guides))
                                          and any(s > 1 for s, _, _, _ in recs)
                                          and any(c for _, c, _, _ in recs))
                if split == "train":
                    coverage["mixed_train"] += len({s for s, _, _, _ in recs}) > 1
                else:
                    coverage["single_test"] += len(recs) == 1
                    coverage["equal_test"] += len({s for s, _, _, _ in recs}) == 1
                    coverage["mixed_test"] += len({s for s, _, _, _ in recs}) > 1
            if split == "test":
                unclipped_test_count += sum(not any(clip for _, clip, _, _ in info["records"])
                                            for info in infos[split])
    _require(len(puzzle_keys) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3,
             "insufficient whole-puzzle/count diversity")
    _require(len(shapes) >= 35 and len(motif_shapes) >= 12 and len(scale_sets) >= 8,
             "insufficient geometric diversity")
    _require(len(target_counts) >= 3 and line_count > 0 and connected_guide_count > 0,
             "missing line, count, or connected-guide cases")
    _require(clipped_count > 0 and unclipped_test_count > 0 and len(role_sets) >= 20,
             "missing clipping or palette variation")
    minimums = {"no_gap": 25, "long_motif": 25, "many_guides": 15,
                "wide_guides": 5, "enclosed_payload": 5, "mixed_train": 20,
                "high_scale": 10, "multi_clip": 3, "small_clip": 5,
                "single_test": 3, "equal_test": 5, "mixed_test": 5,
                "combined": 5, "varying_shape_puzzle": 35,
                "varying_palette_puzzle": 15, "varying_guide_puzzle": 25,
                "unclipped_puzzle": 1, "unclipped_test_puzzle": 5,
                "smaller_test_puzzle": 5, "fixed_palette_puzzle": 5,
                "source_palette": 1, "payload_islands": 20, "hollow": 10,
                "mild_clip": 5, "small_clip_large_fits": 5,
                "isolated_train_mixed_test": 3, "train_clip_test_unclip": 5}
    for name, minimum in minimums.items():
        _require(coverage[name] >= minimum,
                 f"review coverage {name}: {coverage[name]} < {minimum}")
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(1901 + (train or 0) + (test or 0), train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4)):
        try:
            generate_puzzle(17, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    # This original source line and clipped scale-three case was formerly rejected.
    source_in = np.zeros((14, 12), dtype=np.int8)
    source_in[2, 1:4] = 1
    source_in[2, 4] = 2
    source_in[7:10, 7:10] = 2
    source_out, source_info = _infer_input(source_in)
    _require(source_info["motif_shape"] == (1, 4), "source line not accepted")
    _require(source_info["records"][0][0:2] == (3, True),
             "source scale-three clipping not accepted")
    _require(np.array_equal(source_out[7:10, :7], np.ones((3, 7), dtype=np.int8)),
             "source clipped strip is wrong")
