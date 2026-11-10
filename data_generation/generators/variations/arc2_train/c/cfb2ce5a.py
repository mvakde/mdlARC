"""Four reflected, independently recolored tiles with sparse color keys."""

import numpy as np


_SOURCE_INPUTS = {
    "0000000000033339000003388700000383800000038830000000004000000000152000000000000000000000000000000000",
    "0000000000021210008001121000000222100000011113000000007000000000000000000000000004000000500000000000",
    "0000000000082880000002222600000828800000082880100000540300000000000000000000010000000000000000000000",
    "0000000000044118000004111000000111400000011445000000060000000700003000000000000000000000000000000000",
}


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _layout(rng):
    th = int(rng.integers(3, 11))
    tw = int(rng.integers(3, 11))
    top = int(rng.integers(1, min(7, 30 - 2 * th)))
    left = int(rng.integers(1, min(7, 30 - 2 * tw)))
    bottom = int(rng.integers(1, min(9, 31 - 2 * th - top)))
    right = int(rng.integers(1, min(9, 31 - 2 * tw - left)))
    corner = (int(rng.integers(0, 2)), int(rng.integers(0, 2)))
    return (top + 2 * th + bottom, left + 2 * tw + right,
            th, tw, top, left, corner)


def _motif(rng, height, width, symmetric, style):
    for _ in range(200):
        if style == 0:  # irregular, with varied role balance
            p = float(rng.uniform(.24, .76))
            motif = (rng.random((height, width)) < p).astype(np.int8)
        elif style == 1:  # bands and repeated rows
            period = int(rng.integers(2, min(height, 4) + 1))
            rows = rng.integers(0, 2, size=(period, width), dtype=np.int8)
            motif = rows[np.arange(height) % period].copy()
            motif[rng.random((height, width)) < .08] ^= 1
        elif style == 2:  # repeated columns
            period = int(rng.integers(2, min(width, 4) + 1))
            columns = rng.integers(0, 2, size=(height, period), dtype=np.int8)
            motif = columns[:, np.arange(width) % period].copy()
            motif[rng.random((height, width)) < .08] ^= 1
        elif style == 3:  # blocks
            motif = np.zeros((height, width), dtype=np.int8)
            rh = int(rng.integers(2, height + 1))
            rw = int(rng.integers(2, width + 1))
            r = int(rng.integers(0, height - rh + 1))
            c = int(rng.integers(0, width - rw + 1))
            motif[r:r + rh, c:c + rw] = 1
            motif[rng.random((height, width)) < .07] ^= 1
        elif style == 4:  # connected growing shape
            motif = np.zeros((height, width), dtype=np.int8)
            r, c = int(rng.integers(height)), int(rng.integers(width))
            target = int(rng.integers(3, height * width - 2))
            for _step in range(target * 8):
                motif[r, c] = 1
                if int(motif.sum()) >= target:
                    break
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                r = min(height - 1, max(0, r + dr))
                c = min(width - 1, max(0, c + dc))
        else:  # deliberate stripes with local interruptions
            axis = int(rng.integers(2))
            stripe = rng.integers(0, 2, size=height if axis == 0 else width)
            motif = np.repeat(stripe[:, None], width, axis=1) if axis == 0 else np.repeat(stripe[None, :], height, axis=0)
            motif = motif.astype(np.int8)
            motif[rng.random((height, width)) < .13] ^= 1
        if symmetric:
            for r in range(height):
                for c in range(width):
                    rr, cc = height - 1 - r, width - 1 - c
                    if (r, c) < (rr, cc):
                        motif[rr, cc] = motif[r, c]
        n = int(motif.sum())
        if not 3 <= n <= height * width - 3:
            continue
        if symmetric != np.array_equal(motif, motif[::-1, ::-1]):
            continue
        if np.array_equal(motif, motif[::-1, :]) or np.array_equal(motif, motif[:, ::-1]):
            continue
        return motif
    # Deterministic fallback; the rejection above almost always succeeds.
    motif = np.zeros((height, width), dtype=np.int8)
    motif[0, :2] = 1
    motif[1, 0] = 1
    if symmetric:
        motif[-1, -2:] = 1
        motif[-2, -1] = 1
    else:
        motif[-1, -1] = 1
    return motif


def _orient(motif, source, destination):
    result = motif
    if source[0] != destination[0]:
        result = result[::-1, :]
    if source[1] != destination[1]:
        result = result[:, ::-1]
    return result


def _maps_for_corner(rng, background, template_map, corner, background_count):
    destinations = [q for q in ((0, 0), (0, 1), (1, 0), (1, 1)) if q != corner]
    rng.shuffle(destinations)
    background_tiles = set(destinations[:background_count])
    visible = [v for v in range(10) if v != background]
    maps = {corner: template_map}
    for q in destinations:
        if q in background_tiles:
            role = int(rng.integers(2))
            color = int(rng.choice(visible))
            maps[q] = (background, color) if role == 0 else (color, background)
            continue
        mode = int(rng.integers(0, 6))
        if mode == 0:
            mapping = template_map
        elif mode == 1:
            mapping = template_map[::-1]
        elif mode == 2 and len(maps) > 1:
            candidates = [m for m in maps.values() if background not in m]
            mapping = candidates[int(rng.integers(len(candidates)))]
        else:
            sample = rng.choice(visible, size=2, replace=False)
            mapping = (int(sample[0]), int(sample[1]))
        maps[q] = mapping
    return maps


def _make_pair(rng, layout, background, symmetric, style, background_count):
    h, w, th, tw, top, left, corner = layout
    motif = _motif(rng, th, tw, symmetric, style)
    colors = rng.choice([v for v in range(10) if v != background], size=2, replace=False)
    template_map = tuple(sorted(int(v) for v in colors))
    maps = _maps_for_corner(rng, background, template_map, corner, background_count)
    inp = np.full((h, w), background, dtype=np.int8)
    out = np.full((h, w), background, dtype=np.int8)
    area = th * tw
    for qr in range(2):
        for qc in range(2):
            q = (qr, qc)
            rs = slice(top + qr * th, top + (qr + 1) * th)
            cs = slice(left + qc * tw, left + (qc + 1) * tw)
            roles = _orient(motif, corner, q)
            mapping = maps[q]
            out[rs, cs] = np.where(roles == 0, mapping[0], mapping[1])
            if q == corner:
                inp[rs, cs] = out[rs, cs]
                continue
            tile = inp[rs, cs]
            budget = max(2, area // 3)
            visible_roles = [role for role in (0, 1) if mapping[role] != background]
            for index, role in enumerate(visible_roles):
                positions = np.argwhere(roles == role)
                maximum = min(6, len(positions) - 1, budget - (len(visible_roles) - index - 1))
                count = min(maximum, int(rng.choice((1, 2, 3, 4, 5, 6), p=(.42, .25, .15, .09, .06, .03))))
                chosen = rng.choice(len(positions), size=count, replace=False)
                for loc in np.atleast_1d(chosen):
                    row, col = positions[int(loc)]
                    tile[row, col] = mapping[role]
                budget -= count
    return {"input": inp, "output": out}


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _source_copy(grid):
    return grid.shape == (10, 10) and "".join(str(int(v)) for v in grid.flat) in _SOURCE_INPUTS


def _interpret(inp):
    """Find a unique mosaic and predict its output using input cells alone."""
    h, w = inp.shape
    border = np.concatenate((inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]))
    if len(set(int(v) for v in border)) != 1:
        return []
    background = int(border[0])
    occupied = inp != background
    points = np.argwhere(occupied)
    if len(points) == 0:
        return []
    rmin, cmin = points.min(axis=0)
    rmax, cmax = points.max(axis=0)
    results = []
    for th in range(3, min(10, (h - 2) // 2) + 1):
        for tw in range(3, min(10, (w - 2) // 2) + 1):
            for top in range(max(1, int(rmax) - 2 * th + 1), min(int(rmin), h - 2 * th - 1) + 1):
                for left in range(max(1, int(cmax) - 2 * tw + 1), min(int(cmin), w - 2 * tw - 1) + 1):
                    inside = occupied[top:top + 2 * th, left:left + 2 * tw]
                    if int(inside.sum()) != len(points):
                        continue
                    for qr in range(2):
                        for qc in range(2):
                            source = (qr, qc)
                            rs = slice(top + qr * th, top + (qr + 1) * th)
                            cs = slice(left + qc * tw, left + (qc + 1) * tw)
                            template = inp[rs, cs]
                            if np.any(template == background):
                                continue
                            colors = np.unique(template)
                            if len(colors) != 2:
                                continue
                            motif = (template == colors[1]).astype(np.int8)
                            if min(int(motif.sum()), th * tw - int(motif.sum())) < 3:
                                continue
                            prediction = np.full_like(inp, background)
                            prediction[rs, cs] = template
                            mappings = {source: (int(colors[0]), int(colors[1]))}
                            clues = {}
                            valid = True
                            for tr in range(2):
                                for tc in range(2):
                                    q = (tr, tc)
                                    if q == source:
                                        continue
                                    rsl = slice(top + tr * th, top + (tr + 1) * th)
                                    csl = slice(left + tc * tw, left + (tc + 1) * tw)
                                    tile = inp[rsl, csl]
                                    count = int(np.count_nonzero(tile != background))
                                    if not 1 <= count <= max(2, th * tw // 3):
                                        valid = False
                                        break
                                    roles = _orient(motif, source, q)
                                    mapping = []
                                    role_counts = []
                                    for role in (0, 1):
                                        values = tile[roles == role]
                                        visible = values[values != background]
                                        uniques = np.unique(visible)
                                        if len(uniques) > 1:
                                            valid = False
                                            break
                                        mapping.append(int(uniques[0]) if len(uniques) else background)
                                        role_counts.append(len(visible))
                                    if not valid or mapping[0] == mapping[1] or np.all(tile != background):
                                        valid = False
                                        break
                                    mappings[q] = tuple(mapping)
                                    clues[q] = tuple(role_counts)
                                    prediction[rsl, csl] = np.where(roles == 0, mapping[0], mapping[1])
                                if not valid:
                                    break
                            if valid and not np.array_equal(prediction, inp):
                                results.append({"layout": (h, w, th, tw, top, left, source),
                                                "background": background, "motif": motif,
                                                "maps": mappings, "clues": clues,
                                                "prediction": prediction})
    return results


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    ntrain, ntest = _resolve_counts(rng, num_train, num_test)
    base_background = int(rng.integers(10))
    steady_layout = bool(rng.random() < .14)
    shared_layout = _layout(rng) if steady_layout else None
    puzzle = {"train": [], "test": []}
    seen = set()
    first_layout = None
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            for _attempt in range(400):
                layout = shared_layout if shared_layout is not None else _layout(rng)
                if first_layout is not None and not steady_layout and split == "train" and index == 1:
                    if layout[2:4] == first_layout[2:4] or layout[-1] == first_layout[-1]:
                        continue
                background = base_background if rng.random() < .55 else int(rng.integers(10))
                symmetric = bool(rng.random() < (.22 if split == "test" else .16))
                if split == "train" and index < 2:
                    symmetric = False
                style = int(rng.integers(6))
                if split == "train" and index == 0:
                    background_count = 1
                elif split == "train" and index == 1:
                    background_count = 0
                else:
                    background_count = int(rng.choice(4, p=[.40, .38, .18, .04]))
                pair = _make_pair(rng, layout, background, symmetric, style, background_count)
                key = _key(pair["input"])
                if key in seen or _source_copy(pair["input"]):
                    continue
                analyses = _interpret(pair["input"])
                if len(analyses) != 1 or not np.array_equal(analyses[0]["prediction"], pair["output"]):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if first_layout is None:
                    first_layout = layout
                break
            else:
                raise RuntimeError("could not construct an unambiguous pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert not _source_copy(inp)
            ikey, okey = _key(inp), _key(out)
            assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            candidates = _interpret(inp)
            assert len(candidates) == 1
            analysis = candidates[0]
            assert np.array_equal(analysis["prediction"], out)
            analyses[split].append(analysis)
    # The first two demonstrations expose both the one-key background rule
    # and fully keyed recoloring using motifs that reveal both reflections.
    first, second = analyses["train"][:2]
    assert any(first["background"] in m for q, m in first["maps"].items() if q != first["layout"][-1])
    assert all(second["background"] not in m for q, m in second["maps"].items() if q != second["layout"][-1])
    for analysis in (first, second):
        motif = analysis["motif"]
        assert not np.array_equal(motif, motif[::-1, ::-1])
        assert not np.array_equal(motif, motif[::-1, :])
        assert not np.array_equal(motif, motif[:, ::-1])


def self_test() -> None:
    train_counts, test_counts, shapes, sizes, corners = set(), set(), set(), set(), set()
    backgrounds, motif_types, bg_positions, clue_counts = set(), set(), set(), set()
    mixed = steady = reused = adjacent_bg = multi_bg = zero_key = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        layouts = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _interpret(pair["input"])[0]
                layout = a["layout"]
                layouts.add(layout)
                shapes.add(layout[:2])
                sizes.add(layout[2:4])
                corners.add(layout[-1])
                backgrounds.add(a["background"])
                motif = a["motif"]
                motif_types.add((bool(np.array_equal(motif, motif[::-1, ::-1])), int(motif.sum()) * 4 // motif.size))
                maps = [m for q, m in a["maps"].items() if q != layout[-1]]
                nb = sum(a["background"] in m for m in maps)
                multi_bg += nb >= 2
                adjacent_bg += any(a["background"] in a["maps"][q] for q in ((layout[-1][0], 1 - layout[-1][1]), (1 - layout[-1][0], layout[-1][1])))
                bg_positions.add(nb)
                reused += len(set(maps + [a["maps"][layout[-1]]])) < 4
                zero_key += a["background"] != 0 and np.any(pair["input"] == 0)
                clue_counts.update(sum(v) for v in a["clues"].values())
        mixed += len(layouts) > 1
        steady += len(layouts) == 1
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(shapes) >= 30 and len(sizes) >= 25
    assert len(corners) == 4 and len(backgrounds) >= 8
    assert mixed >= 25 and steady >= 2
    assert adjacent_bg >= 20 and multi_bg >= 3 and reused >= 20 and zero_key >= 20
    assert bg_positions == {0, 1, 2, 3} and max(clue_counts) >= 5
    assert any(s for s, _ in motif_types) and any(not s for s, _ in motif_types)
    for seed, nt, ns in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        p = generate_puzzle(seed, nt, ns)
        assert len(p["train"]) == nt and len(p["test"]) == ns
    for args in ((None, 1), (None, 4), (2, None), (10, None)):
        generate_puzzle(1234, *args)
    first = generate_puzzle(2001, 6, 3)
    again = generate_puzzle(2001, 6, 3)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test") for x, y in zip(first[s], again[s]) for k in ("input", "output"))
    assert _key(first["train"][0]["input"]) != _key(generate_puzzle(2002, 6, 3)["train"][0]["input"])
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 2.5}, {"num_test": True}):
        try:
            generate_puzzle(3001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
