import numpy as np


_SOURCE_PAIR_TEXTS = frozenset({
    (
        "/".join((
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000008000000",
            "0000088800000",
            "0002803082000",
            "0000088800000",
            "0000008000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        )),
        "/".join((
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0008000008000",
            "0088800088800",
            "0800080800080",
            "0088800088800",
            "0008000008000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        )),
    ),
    (
        "/".join((
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000282000000",
            "0000838000000",
            "0000282000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        )),
        "/".join((
            "0000000000000",
            "0000000000000",
            "0000808000000",
            "0008080800000",
            "0000808000000",
            "0008080800000",
            "0000808000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        )),
    ),
    (
        "/".join((
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000200000000",
            "0000800000000",
            "0088888000000",
            "0088088000000",
            "0088388000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        )),
        "/".join((
            "0000000000000",
            "0000000000000",
            "0000800000000",
            "0088888000000",
            "0088088000000",
            "0088088000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        )),
    ),
    (
        "/".join((
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0020000020000",
            "0000808000000",
            "0008808800000",
            "0000838000000",
            "0000080000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        )),
        "/".join((
            "0000000000000",
            "0808000808000",
            "8808808808800",
            "0808000808000",
            "0080000080000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        )),
    ),
})


def _grid_text(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _check_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not minimum <= int(value) <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_optional_count("num_train", num_train, 2, 10)
    _check_optional_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        resolved_test = 1 + seed % 4
        minimum_train = max(2, resolved_test + 1)
        train_span = 11 - minimum_train
        resolved_train = minimum_train + (seed // 4) % train_span
    elif num_train is None:
        resolved_test = int(num_test)
        minimum_train = max(2, resolved_test + 1)
        resolved_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        resolved_train = int(num_train)
        maximum_test = min(4, resolved_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for num_train")
        resolved_test = int(rng.integers(1, maximum_test + 1))
    else:
        resolved_train = int(num_train)
        resolved_test = int(num_test)

    if resolved_train <= resolved_test:
        raise ValueError("num_train must be greater than num_test")

    return resolved_train, resolved_test


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = [
        int(value)
        for value in rng.choice(np.arange(10), size=4, replace=False)
    ]
    return {
        "background": colors[0],
        "payload": colors[1],
        "source": colors[2],
        "destination": colors[3],
    }


def _symmetry(offsets: set[tuple[int, int]]) -> tuple[bool, bool]:
    return (
        {(r, -c) for r, c in offsets} == offsets,
        {(-r, c) for r, c in offsets} == offsets,
    )


def _connected_to_anchor(offsets: set[tuple[int, int]]) -> bool:
    remaining = set(offsets)
    reached = {(0, 0)}
    frontier = [(0, 0)]
    while frontier:
        r, c = frontier.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cell = (r + dr, c + dc)
                if cell in remaining:
                    remaining.remove(cell)
                    reached.add(cell)
                    frontier.append(cell)
    return not remaining


def _sample_motif(rng: np.random.Generator, style: str) -> tuple[tuple[int, int], ...]:
    # Every shape is a set of payload offsets; the marked origin is excluded.
    for _ in range(100):
        cells: set[tuple[int, int]] = set()
        if style in ("mirror_lr", "mirror_ud", "mirror_both"):
            target = int(rng.integers(4, 14))
            radius = int(rng.integers(2, 5))
            for _ in range(80):
                candidates = []
                for r in range(-radius, radius + 1):
                    for c in range(-radius, radius + 1):
                        if (r, c) == (0, 0):
                            continue
                        orbit = {(r, c)}
                        if style in ("mirror_lr", "mirror_both"):
                            orbit |= {(rr, -cc) for rr, cc in tuple(orbit)}
                        if style in ("mirror_ud", "mirror_both"):
                            orbit |= {(-rr, cc) for rr, cc in tuple(orbit)}
                        if orbit.issubset(cells) or len(cells | orbit) > target:
                            continue
                        if any(max(abs(a - b), abs(d - e)) <= 1
                               for a, d in orbit for b, e in cells | {(0, 0)}):
                            candidates.append(orbit)
                if not candidates:
                    break
                orbit = candidates[int(rng.integers(len(candidates)))]
                cells |= orbit
                if len(cells) >= target - 1:
                    break
        elif style == "bar":
            vertical = bool(rng.integers(2))
            before = int(rng.integers(1, 4))
            after = int(rng.integers(1, 4))
            for k in list(range(-before, 0)) + list(range(1, after + 1)):
                cells.add((k, 0) if vertical else (0, k))
            if bool(rng.integers(2)):
                side = -1 if rng.integers(2) else 1
                cells.add((side, 1) if vertical else (1, side))
        elif style == "cross":
            arms = [int(rng.integers(1, 4)) for _ in range(4)]
            if bool(rng.integers(2)):
                arms[1] = arms[0]
            if bool(rng.integers(2)):
                arms[3] = arms[2]
            for direction, length in zip(((-1, 0), (1, 0), (0, -1), (0, 1)), arms):
                for distance in range(1, length + 1):
                    cells.add((direction[0] * distance, direction[1] * distance))
        elif style == "ring":
            radius = int(rng.integers(1, 3))
            cells = {(r, c) for r in range(-radius, radius + 1)
                     for c in range(-radius, radius + 1)
                     if max(abs(r), abs(c)) == radius}
            if radius == 2:
                cells |= {(r, c) for r, c in ((-1, 0), (1, 0), (0, -1), (0, 1))}
            if bool(rng.integers(2)):
                for cell in sorted(cells):
                    if len(cells) <= 6:
                        break
                    if rng.random() < 0.25 and _connected_to_anchor(cells - {cell}):
                        cells.remove(cell)
        else:
            target = int(rng.integers(4, 11)) if style == "compact" else int(rng.integers(10, 19))
            radius = 2 if style == "compact" else 4
            cells = set()
            for _ in range(250):
                if len(cells) >= target:
                    break
                base = sorted(cells | {(0, 0)})[int(rng.integers(len(cells) + 1))]
                dr, dc = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                          if (a, b) != (0, 0)][int(rng.integers(8))]
                cell = (base[0] + dr, base[1] + dc)
                if cell != (0, 0) and max(abs(cell[0]), abs(cell[1])) <= radius:
                    cells.add(cell)
        if not 4 <= len(cells) <= 20 or (0, 0) in cells:
            continue
        if not _connected_to_anchor(cells):
            continue
        horizontal, vertical = _symmetry(cells)
        if style == "compact" and (horizontal or vertical):
            continue
        if style == "mirror_lr" and not horizontal:
            continue
        if style == "mirror_ud" and not vertical:
            continue
        if style == "mirror_both" and not (horizontal and vertical):
            continue
        return tuple(sorted(cells))
    # A deterministic, connected fallback for any unlucky random stream.
    return ((-1, 0), (0, -1), (0, 1), (1, 0)) if style != "compact" else ((-1, 0), (0, -1), (0, 1), (1, 1))


def _overlap_count(motif: tuple[tuple[int, int], ...], destinations: tuple[tuple[int, int], ...]) -> int:
    placed = {(r + dr, c + dc) for dr, dc in destinations for r, c in motif}
    return len(motif) * len(destinations) - len(placed)


def _layout_label(destinations: tuple[tuple[int, int], ...]) -> str:
    if len(destinations) == 1:
        r, c = destinations[0]
        if r == 0:
            return "side_single"
        if c == 0:
            return "vertical_single"
        return "diagonal_single"
    if len(destinations) == 2:
        (r1, c1), (r2, c2) = destinations
        if r1 == r2:
            return "horizontal_pair"
        if c1 == c2:
            return "vertical_pair"
        if r1 < 0 and r2 < 0 and c1 * c2 < 0:
            return "upper_branch"
        if r1 * r2 > 0 or c1 * c2 > 0:
            return "same_side_pair"
        return "mixed_pair"
    if len(destinations) == 3:
        return "triple"
    quadrants = {(r > 0, c > 0) for r, c in destinations if r and c}
    return "quadrant_four" if len(quadrants) == 4 else "irregular_four"


def _sample_layout(rng: np.random.Generator, motif: tuple[tuple[int, int], ...],
                   count: int, interaction: str) -> tuple[tuple[int, int], ...]:
    motif_set = set(motif)
    pool = [(r, c) for r in range(-6, 7) for c in range(-6, 7)
            if (r, c) != (0, 0) and (r, c) not in motif_set
            and max(abs(r), abs(c)) >= 2]
    motif_differences = sorted({(a - b, c - d) for a, c in motif for b, d in motif
                                if (a, c) != (b, d)})
    best = None
    best_score = -1000
    for _ in range(180):
        layout_style = int(rng.choice((0, 1, 2, 2, 2, 3, 4, 5, 6, 7)))
        chosen: list[tuple[int, int]] = []
        if count >= 2 and interaction in ("overlap", "light", "strong") and rng.random() < 0.45:
            d1 = pool[int(rng.integers(len(pool)))]
            delta = motif_differences[int(rng.integers(len(motif_differences)))]
            d2 = (d1[0] + delta[0], d1[1] + delta[1])
            if d2 in pool:
                chosen = [d1, d2]
        if not chosen:
            if count == 1:
                dr, dc = ((0, -1), (0, 1), (-1, 0), (1, 0),
                          (-1, -1), (-1, 1), (1, -1), (1, 1))[int(rng.integers(8))]
                chosen = [(dr * int(rng.integers(2, 7)),
                           dc * int(rng.integers(2, 7)))]
            elif count == 2 and layout_style < 5:
                a = int(rng.integers(2, 7))
                b = int(rng.integers(2, 7))
                if layout_style == 0:
                    chosen = [(0, -a), (0, b)]
                elif layout_style == 1:
                    chosen = [(-a, 0), (b, 0)]
                elif layout_style == 2:
                    chosen = [(-a, -b), (-int(rng.integers(2, 7)), int(rng.integers(2, 7)))]
                elif layout_style == 3:
                    chosen = [(0, -a), (0, -a - b)]
                else:
                    chosen = [(-a, -b), (a, b)]
            elif count == 4 and layout_style < 3:
                chosen = [(-int(rng.integers(2, 5)), -int(rng.integers(2, 5))),
                          (-int(rng.integers(2, 5)), int(rng.integers(2, 5))),
                          (int(rng.integers(2, 5)), -int(rng.integers(2, 5))),
                          (int(rng.integers(2, 5)), int(rng.integers(2, 5)))]
        while len(chosen) < count:
            cell = pool[int(rng.integers(len(pool)))]
            if cell not in chosen:
                chosen.append(cell)
        if len(chosen) != count or any(cell not in pool for cell in chosen):
            continue
        result = tuple(sorted(chosen))
        overlap = _overlap_count(motif, result)
        if interaction == "none" and overlap == 0:
            return result
        if interaction == "overlap" and overlap > 0:
            return result
        if interaction == "light" and 0 < overlap <= max(2, len(motif) // 4):
            return result
        if interaction == "strong" and overlap >= max(2, len(motif) // 3):
            return result
        if interaction == "any":
            return result
        score = overlap if interaction in ("overlap", "strong") else -overlap
        if score > best_score:
            best, best_score = result, score
    if best is not None:
        return best
    raise RuntimeError("could not create a destination layout")


def _generate_pair(rng: np.random.Generator, parameters: dict,
                   motif_style: str, marker_count: int,
                   interaction: str) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        motif = _sample_motif(rng, motif_style)
        destinations = _sample_layout(rng, motif, marker_count, interaction)
        if any(cell in motif for cell in destinations):
            continue
        input_cells = {(0, 0), *motif, *destinations}
        output_cells = {(r + dr, c + dc) for dr, dc in destinations for r, c in motif}
        all_cells = input_cells | output_cells
        min_r = min(r for r, _ in all_cells)
        max_r = max(r for r, _ in all_cells)
        min_c = min(c for _, c in all_cells)
        max_c = max(c for _, c in all_cells)
        # Zero margins create true border contacts; independent margins preserve off-center placement.
        pads = [int(rng.choice((0, 0, 1, 2, 3, 5, 7))) for _ in range(4)]
        if rng.random() < 0.25:
            pads[int(rng.integers(4))] = 0
        height = max_r - min_r + 1 + pads[0] + pads[1]
        width = max_c - min_c + 1 + pads[2] + pads[3]
        if height > 30 or width > 30:
            continue
        if len(output_cells) * 2 >= height * width:
            # Keep the background role obvious while still leaving one tight side.
            pads[1] += 3
            pads[3] += 3
            height += 3
            width += 3
        if height > 30 or width > 30 or len(output_cells) * 2 >= height * width:
            continue
        sr, sc = pads[0] - min_r, pads[2] - min_c
        background, payload = parameters["background"], parameters["payload"]
        source, destination = parameters["source"], parameters["destination"]
        inp = np.full((height, width), background, dtype=np.int8)
        out = np.full((height, width), background, dtype=np.int8)
        for r, c in motif:
            inp[sr + r, sc + c] = payload
        inp[sr, sc] = source
        for r, c in destinations:
            inp[sr + r, sc + c] = destination
        for r, c in output_cells:
            out[sr + r, sc + c] = payload
        return inp, out
    raise RuntimeError("could not fit motif and full stamps in grid")


def _inspect_generated_puzzle(puzzle: dict) -> dict:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_text(inp)
            assert (inp.shape, key) not in seen_inputs
            seen_inputs.add((inp.shape, key))
            assert (key, _grid_text(out)) not in _SOURCE_PAIR_TEXTS
            records.append((split, inp, out))
    train = records[:len(puzzle["train"])]
    output_sets = [{int(x) for x in np.unique(out)} for _, _, out in records]
    assert all(len(s) == 2 and s == output_sets[0] for s in output_sets)
    a, b = tuple(output_sets[0])
    backgrounds = [x for x, y in ((a, b), (b, a)) if all(
        int(np.count_nonzero(out == x)) > int(np.count_nonzero(out == y))
        for _, _, out in records)]
    assert len(backgrounds) == 1
    background = backgrounds[0]
    payload = next(x for x in output_sets[0] if x != background)
    marker_sets = [{int(x) for x in np.unique(inp)} - output_sets[0]
                   for _, inp, _ in records]
    assert all(len(s) == 2 and s == marker_sets[0] for s in marker_sets)
    markers = tuple(marker_sets[0])
    source_candidates = [x for x in markers if all(
        int(np.count_nonzero(inp == x)) == 1 for _, inp, _ in train)]
    assert len(source_candidates) == 1
    source = source_candidates[0]
    destination = next(x for x in markers if x != source)
    assert any(int(np.count_nonzero(inp == destination)) > 1 for _, inp, _ in train)
    features = []
    for split, inp, out in records:
        assert {int(x) for x in np.unique(inp)} == {background, payload, source, destination}
        src = np.argwhere(inp == source)
        dst = np.argwhere(inp == destination)
        motif_cells = np.argwhere(inp == payload)
        assert src.shape == (1, 2) and 1 <= len(dst) <= 4
        assert 4 <= len(motif_cells) <= 20
        sr, sc = map(int, src[0])
        motif = tuple(sorted((int(r) - sr, int(c) - sc) for r, c in motif_cells))
        destinations = tuple(sorted((int(r) - sr, int(c) - sc) for r, c in dst))
        assert (0, 0) not in motif and not set(motif) & set(destinations)
        assert _connected_to_anchor(set(motif))
        union = {(sr + r + dr, sc + c + dc)
                 for dr, dc in destinations for r, c in motif}
        assert all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] for r, c in union)
        actual = {(int(r), int(c)) for r, c in np.argwhere(out == payload)}
        assert actual == union
        assert int(np.count_nonzero(out == background)) > len(union)
        horizontal, vertical = _symmetry(set(motif))
        overlap = len(motif) * len(destinations) - len(union)
        border = any(r in (0, inp.shape[0] - 1) or c in (0, inp.shape[1] - 1)
                     for r, c in set((sr + r, sc + c) for r, c in motif) |
                     {(sr, sc)} | {(sr + r, sc + c) for r, c in destinations} | union)
        all_foreground = (set((sr + r, sc + c) for r, c in motif) |
                          {(sr, sc)} |
                          {(sr + r, sc + c) for r, c in destinations} | union)
        rows = [r for r, _ in all_foreground]
        cols = [c for _, c in all_foreground]
        margins = (min(rows), inp.shape[0] - 1 - max(rows),
                   min(cols), inp.shape[1] - 1 - max(cols))
        output_border = any(r in (0, inp.shape[0] - 1) or c in (0, inp.shape[1] - 1)
                            for r, c in union)
        features.append({"split": split, "shape": inp.shape, "motif_size": len(motif),
                         "motif": motif, "symmetric": horizontal or vertical,
                         "both_symmetries": horizontal and vertical,
                         "marker_count": len(destinations), "layout": _layout_label(destinations),
                         "destinations": destinations, "overlap": overlap,
                         "border": border, "output_border": output_border,
                         "unequal_margins": margins[0] != margins[1] or margins[2] != margins[3],
                         "source_offcenter": abs(2 * sr - inp.shape[0] + 1) > 1 or
                                             abs(2 * sc - inp.shape[1] + 1) > 1,
                         "covers_old_input": bool(union & {(sr + r, sc + c) for r, c in motif}),
                         "covers_marker": bool(union & ({(sr, sc)} | {(sr + r, sc + c) for r, c in destinations}))})
    training_features = features[:len(train)]
    assert any(f["marker_count"] > 1 for f in training_features)
    assert any(f["overlap"] > 0 for f in training_features)
    assert any(not f["symmetric"] for f in training_features)
    assert any(f["symmetric"] for f in training_features)
    return {"colors": (background, payload, source, destination), "pairs": features}


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _inspect_generated_puzzle(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(int(seed), rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    # The first two demonstrations establish role multiplicity, overlap union,
    # exact copying of an asymmetric motif, and a reflected compact motif.
    styles = ["compact", str(rng.choice(("mirror_lr", "mirror_ud", "mirror_both")))]
    styles += [str(rng.choice(("compact", "large", "mirror_lr", "mirror_ud", "mirror_both", "bar", "cross", "ring")))
               for _ in range(train_count - 2)]
    specs = []
    specs.append((styles[0], int(rng.integers(2, 5)), "overlap"))
    specs.append((styles[1], 1, "none"))
    for style in styles[2:]:
        count = int(rng.choice((1, 2, 2, 3, 4)))
        interaction = str(rng.choice(("none", "light", "strong", "any"))) if count > 1 else "none"
        specs.append((style, count, interaction))
    rng.shuffle(specs)
    test_specs = []
    for _ in range(test_count):
        style = str(rng.choice(("compact", "large", "mirror_lr", "mirror_ud", "mirror_both", "bar", "cross", "ring")))
        count = int(rng.choice((1, 2, 2, 3, 4)))
        interaction = str(rng.choice(("none", "light", "strong", "any"))) if count > 1 else "none"
        test_specs.append((style, count, interaction))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, items in (("train", specs), ("test", test_specs)):
        for style, count, interaction in items:
            pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            pair_rng = np.random.default_rng(pair_seed)
            for _ in range(100):
                inp, out = _generate_pair(pair_rng, parameters, style, count, interaction)
                pair_key = (_grid_text(inp), _grid_text(out))
                if _grid_text(inp) in seen or pair_key in _SOURCE_PAIR_TEXTS:
                    continue
                seen.add(_grid_text(inp))
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("could not generate unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_text(p["input"]), _grid_text(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(17)
    assert _puzzle_fingerprint(automatic) == _puzzle_fingerprint(generate_puzzle(17))
    explicit = generate_puzzle(12345, 7, 3)
    assert _puzzle_fingerprint(explicit) == _puzzle_fingerprint(generate_puzzle(12345, 7, 3))
    assert _puzzle_fingerprint(explicit) != _puzzle_fingerprint(generate_puzzle(12346, 7, 3))
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(20000 + (a or 0) * 10 + (b or 0), a, b)
        validate_generated_puzzle(puzzle)
        if a is not None:
            assert len(puzzle["train"]) == a
        if b is not None:
            assert len(puzzle["test"]) == b
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 3}, {"num_train": 2, "num_test": 4},
                   {"num_train": 2.5}, {"num_test": True}):
        try:
            generate_puzzle(22000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
    all_features = []
    colors, shapes, motifs, layouts, fingerprints = set(), set(), set(), set(), set()
    train_counts, test_counts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        metadata = _inspect_generated_puzzle(puzzle)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        colors.add(metadata["colors"])
        fingerprints.add(_puzzle_fingerprint(puzzle))
        for f in metadata["pairs"]:
            all_features.append(f)
            shapes.add(f["shape"])
            motifs.add(f["motif"])
            layouts.add((f["marker_count"], f["layout"], f["destinations"]))
        assert len({f["motif"] for f in metadata["pairs"]}) >= 2
        assert len({f["layout"] for f in metadata["pairs"]}) >= 2
        assert len({f["shape"] for f in metadata["pairs"]}) >= 2
        assert len({f["marker_count"] for f in metadata["pairs"]}) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(colors) >= 30 and len(fingerprints) == 42
    assert len(shapes) >= 50 and len(motifs) >= 150 and len(layouts) >= 100
    assert sum(4 <= f["motif_size"] <= 6 for f in all_features) >= 50
    assert sum(f["symmetric"] for f in all_features) >= 80
    assert sum(f["both_symmetries"] for f in all_features) >= 25
    assert sum(not f["symmetric"] for f in all_features) >= 50
    assert sum(f["marker_count"] == 3 for f in all_features) >= 10
    assert sum(f["output_border"] for f in all_features) >= 30
    assert sum(f["unequal_margins"] for f in all_features) >= 100
    assert sum(f["source_offcenter"] for f in all_features) >= 100
    assert sum(f["covers_old_input"] for f in all_features) >= 20
    assert sum(f["covers_marker"] for f in all_features) >= 10
    assert sum(f["overlap"] == 0 for f in all_features) >= 30
    assert sum(0 < f["overlap"] <= 2 for f in all_features) >= 15
    assert sum(f["overlap"] > 2 for f in all_features) >= 10
    assert sum(f["split"] == "test" and f["marker_count"] in (2, 3)
               and f["overlap"] > 0 for f in all_features) >= 5
    for layout in ("horizontal_pair", "vertical_pair", "same_side_pair",
                   "upper_branch", "triple", "irregular_four", "quadrant_four",
                   "side_single", "vertical_single"):
        assert any(f["layout"] == layout for f in all_features)
