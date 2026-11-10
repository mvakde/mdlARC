"""Generate four-tile ARC odd-one-out puzzles."""
import numpy as np

_SOURCE_PAIRS = {
    ("10010\n01001\n00000\n10010\n11001", "10\n11"),
    ("02002\n22022\n00000\n02022\n22020", "22\n20"),
    ("88008\n80080\n00000\n88088\n80080", "08\n80"),
    ("55050\n05005\n00000\n55055\n05005", "50\n05"),
}


def _key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _text(grid):
    return "\n".join("".join(str(int(v)) for v in row) for row in grid)


def _source_copy(inp, out):
    return (_text(inp), _text(out)) in _SOURCE_PAIRS


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed + seed // 4) % (11 - minimum)
    elif train is None:
        minimum = max(2, test + 1)
        train = int(rng.integers(minimum, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _sample_geometry(rng, quiet):
    if quiet and rng.random() < .85:
        return (2, 2, 1, 1)
    scale = rng.random()
    if scale < .27:
        h = w = 2
    elif scale < .83:
        h = int(rng.integers(2, 9))
        w = h if rng.random() < .32 else int(rng.integers(2, 9))
    else:
        h = int(rng.integers(9, 15))
        w = h if rng.random() < .30 else int(rng.integers(2, 15))
    sr = int(rng.integers(1, min(4, 30 - 2 * h) + 1))
    sc = int(rng.integers(1, min(4, 30 - 2 * w) + 1))
    return h, w, sr, sc


def _repeat_ok(mask):
    return (bool(mask.any()) and bool((~mask).any())
            and bool(np.all(mask.any(axis=0))) and bool(np.all(mask.any(axis=1))))


def _sample_repeated(rng, h, w):
    n = h * w
    for _ in range(80):
        kind = int(rng.integers(0, 6))
        mask = np.zeros((h, w), dtype=bool)
        if kind == 0:  # irregular, variable density
            mask = rng.random((h, w)) < rng.uniform(.28, .76)
        elif kind == 1:  # sparse with row and column coverage
            mask[np.arange(h), rng.integers(0, w, h)] = True
            mask[rng.integers(0, h, w), np.arange(w)] = True
            if n > 4:
                mask |= rng.random((h, w)) < rng.uniform(0, .12)
        elif kind == 2:  # dense
            mask[:] = True
            count = int(rng.integers(1, max(2, n // 3 + 1)))
            mask.flat[rng.choice(n, size=count, replace=False)] = False
        elif kind == 3:  # checker or diagonal bands
            rows, cols = np.indices((h, w))
            period = 2 if rng.random() < .5 else 3
            mask = (rows + cols + int(rng.integers(0, period))) % period != 0
            if h > 2 and w > 2 and rng.random() < .5:
                mask ^= rng.random((h, w)) < .12
        elif kind == 4:  # edges and corners
            mask[0, :] = mask[-1, :] = True
            mask[:, 0] = mask[:, -1] = True
            if n > 4:
                mask ^= rng.random((h, w)) < .13
        else:  # asymmetric diagonal walk
            mask[np.arange(h), (np.arange(h) + int(rng.integers(w))) % w] = True
            mask[rng.integers(0, h, w), np.arange(w)] = True
            mask |= rng.random((h, w)) < rng.uniform(.08, .32)
        if _repeat_ok(mask):
            return mask
    mask = np.zeros((h, w), dtype=bool)
    mask[np.arange(h), np.arange(h) % w] = True
    mask[np.arange(w) % h, np.arange(w)] = True
    if mask.all():
        mask[0, 0] = False
    return mask


def _odd_mask(rng, repeated, requested):
    h, w = repeated.shape
    on, off = np.flatnonzero(repeated), np.flatnonzero(~repeated)
    for kind in (requested, "swap", "add", "delete", "rotate",
                 "reflect_h", "reflect_v", "distant"):
        odd = repeated.copy()
        if kind == "add" and len(off) > 1:
            odd.flat[int(rng.choice(off))] = True
        elif kind == "delete" and len(on) > 1:
            odd.flat[int(rng.choice(on))] = False
        elif kind == "swap" and len(on) and len(off):
            odd.flat[int(rng.choice(on))] = False
            odd.flat[int(rng.choice(off))] = True
        elif kind == "rotate":
            odd = repeated[::-1, ::-1].copy()
        elif kind == "reflect_h":
            odd = repeated[:, ::-1].copy()
        elif kind == "reflect_v":
            odd = repeated[::-1, :].copy()
        elif kind == "distant":
            flips = min(h * w - 1, max(3, int(rng.uniform(.30, .75) * h * w)))
            odd.flat[rng.choice(h * w, size=flips, replace=False)] ^= True
        else:
            continue
        if odd.any() and (~odd).any() and not np.array_equal(odd, repeated):
            return odd
    raise AssertionError("valid repeated tile must admit a distinct odd tile")


def _assemble(repeated, odd, geometry, palette, position):
    h, w, sr, sc = geometry
    bg, fg = palette
    base, unique = np.where(repeated, fg, bg), np.where(odd, fg, bg)
    inp = np.full((2 * h + sr, 2 * w + sc), bg, dtype=np.int64)
    starts = ((0, 0), (0, w + sc), (h + sr, 0), (h + sr, w + sc))
    for i, (r, c) in enumerate(starts):
        inp[r:r + h, c:c + w] = unique if i == position else base
    return inp, unique.astype(np.int64, copy=True)


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    colors = [int(v) for v in np.unique(inp)]
    assert len(colors) == 2
    candidates = []
    for bg in colors:
        rows = np.flatnonzero(np.all(inp == bg, axis=1))
        cols = np.flatnonzero(np.all(inp == bg, axis=0))
        if len(rows) == 0 or len(cols) == 0:
            continue
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            continue
        if not np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)):
            continue
        h, w, sr, sc = int(rows[0]), int(cols[0]), len(rows), len(cols)
        if h < 2 or w < 2 or 2 * h + sr != inp.shape[0] or 2 * w + sc != inp.shape[1]:
            continue
        starts = ((0, 0), (0, w + sc), (h + sr, 0), (h + sr, w + sc))
        tiles = [inp[r:r + h, c:c + w] for r, c in starts]
        if any(tile.shape != (h, w) for tile in tiles):
            continue
        keys = [_key(tile) for tile in tiles]
        frequencies = [keys.count(key) for key in keys]
        if sorted(frequencies) != [1, 3, 3, 3]:
            continue
        pos = frequencies.index(1)
        repeated, odd = tiles[frequencies.index(3)], tiles[pos]
        fg = next(color for color in colors if color != bg)
        repeated_mask, odd_mask = repeated == fg, odd == fg
        if not _repeat_ok(repeated_mask) or not odd_mask.any() or odd_mask.all():
            continue
        if out.shape != (h, w) or not np.array_equal(out, odd):
            continue
        candidates.append({"background": bg, "foreground": fg,
                           "geometry": (h, w, sr, sc), "position": pos,
                           "repeated": repeated_mask.copy(), "odd": odd_mask.copy()})
    assert len(candidates) == 1
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert not np.array_equal(pair["input"], pair["output"])
            assert not _source_copy(pair["input"], pair["output"])
            analyses[split].append(_analyze(pair))
    train = analyses["train"]
    assert len({a["position"] for a in train}) >= 2
    assert any(np.count_nonzero(a["repeated"] != a["odd"]) <= 2
               or int(a["repeated"].sum()) == int(a["odd"].sum()) for a in train)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    nt, ns = _resolve_counts(seed, rng, num_train, num_test)
    # Puzzle style changes sampling breadth, never the visible 3-to-1 rule.
    quiet = bool(rng.random() < .30)
    base_geometry = _sample_geometry(rng, quiet)
    first_bg = 0 if quiet and rng.random() < .42 else int(rng.integers(0, 10))
    first_fg = int(rng.choice([v for v in range(10) if v != first_bg]))
    palettes = [(first_bg, first_fg)]
    positions = rng.permutation(4).tolist()
    train_positions = positions[:int(rng.integers(2, min(4, nt) + 1))]
    schedule = [train_positions[i % len(train_positions)] for i in range(nt)]
    rng.shuffle(schedule)
    pair_seeds = np.random.SeedSequence(
        [int(v) for v in rng.integers(0, 2**32, size=4, dtype=np.uint32)]
    ).spawn(nt + ns)
    puzzle = {"train": [], "test": []}
    history, seen = [], set()
    for index in range(nt + ns):
        split = "train" if index < nt else "test"
        local = np.random.default_rng(pair_seeds[index])
        if index == 0:
            geom, palette = base_geometry, palettes[0]
        else:
            match_geometry = local.random() < (.93 if quiet else (.48 if split == "test" else .30))
            if match_geometry:
                geom = history[int(local.integers(len(history)))]["geometry"]
            else:
                geom = _sample_geometry(local, quiet and local.random() < .45)
                if not quiet and split == "train" and index == 1:
                    for _ in range(10):
                        if geom != base_geometry:
                            break
                        geom = _sample_geometry(local, False)
            if quiet and local.random() < .88:
                bg = first_bg
                fg = int(local.choice([v for v in range(10) if v != bg]))
                palette = (bg, fg)
            else:
                p = local.random()
                previous = palettes[int(local.integers(len(palettes)))]
                if p < .17:
                    palette = (previous[1], previous[0])
                elif p < .42:
                    palette = previous
                else:
                    bg = int(local.integers(0, 10))
                    fg = int(local.choice([v for v in range(10) if v != bg]))
                    palette = (bg, fg)
            palettes.append(palette)
        h, w = geom[:2]
        eligible = [item["repeated"] for item in history if item["geometry"][:2] == (h, w)]
        use_old = bool(eligible) and local.random() < (.25 if split == "train" else .43)
        repeated = eligible[int(local.integers(len(eligible)))].copy() if use_old else _sample_repeated(local, h, w)
        if split == "train":
            pos = int(schedule[index])
            if index == 0:
                relation = "swap" if local.random() < .5 else "add"
            elif quiet:
                relation = str(local.choice(["add", "delete", "swap", "rotate"],
                                            p=[.31, .30, .30, .09]))
            else:
                relation = str(local.choice(
                    ["add", "delete", "swap", "rotate", "reflect_h", "reflect_v", "distant"],
                    p=[.17, .16, .25, .12, .10, .08, .12]))
        else:
            familiar = [v for v in range(4) if v in train_positions]
            novel = [v for v in range(4) if v not in train_positions]
            pool = familiar if not novel or local.random() < .50 else novel
            pos = int(local.choice(pool))
            if quiet:
                relation = str(local.choice(["add", "delete", "swap", "rotate"],
                                            p=[.28, .28, .34, .10]))
            else:
                relation = str(local.choice(
                    ["add", "delete", "swap", "rotate", "reflect_h", "reflect_v", "distant"],
                    p=[.13, .13, .24, .14, .11, .10, .15]))
        for _ in range(30):
            odd = _odd_mask(local, repeated, relation)
            inp, out = _assemble(repeated, odd, geom, palette, pos)
            if _key(inp) not in seen and not _source_copy(inp, out):
                break
            repeated = _sample_repeated(local, h, w)
        else:
            raise ValueError("could not construct a distinct pair")
        pair = {"input": inp, "output": out}
        puzzle[split].append(pair)
        seen.add(_key(inp))
        history.append({"geometry": geom, "repeated": repeated.copy()})
    validate_generated_puzzle(puzzle)
    return puzzle


def _same(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[name], b[name])
        for a, b in zip(first[split], second[split])
        for name in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (104, None, 1), (105, None, 4),
                              (106, 2, None), (107, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(50, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {args}")
    assert _same(generate_puzzle(1234, 8, 3), generate_puzzle(1234, 8, 3))
    assert not _same(generate_puzzle(1234, 8, 3), generate_puzzle(1235, 8, 3))
    train_counts, test_counts, geometries, positions = set(), set(), set(), set()
    train_positions_seen, test_positions_seen = set(), set()
    mixed_geometry = mixed_background = swaps = close = distant = 0
    blank_odd_edges = asym_crosses = large_inputs = 0
    familiar_test = new_test = shared_palette = reversed_palette = 0
    repeated_reuse = new_motifs = 0
    additions = deletions = rotations = reflections = 0
    rich_motif_training = quiet_puzzles = combined_puzzles = 0
    larger_tests = smaller_tests = source_like_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train = [_analyze(p) for p in puzzle["train"]]
        test = [_analyze(p) for p in puzzle["test"]]
        all_pairs = train + test
        mixed_geometry += len({a["geometry"] for a in all_pairs}) > 1
        mixed_background += len({a["background"] for a in all_pairs}) > 1
        rich_motif_training += len({(a["geometry"][:2], _key(a["repeated"])) for a in train}) >= 3
        quiet_puzzles += all(a["geometry"] == (2, 2, 1, 1) for a in all_pairs)
        combined_puzzles += (
            len({a["geometry"] for a in all_pairs}) > 1
            and len({(a["background"], a["foreground"]) for a in all_pairs}) > 1
            and any(a["position"] == 0 for a in all_pairs)
            and any(np.count_nonzero(a["repeated"] != a["odd"]) <= 2 for a in all_pairs)
        )
        train_pos = {a["position"] for a in train}
        train_positions_seen.update(train_pos)
        test_positions_seen.update(a["position"] for a in test)
        train_motifs = {(a["geometry"][:2], _key(a["repeated"])) for a in train}
        train_palettes = {(a["background"], a["foreground"]) for a in train}
        for a in test:
            familiar_test += a["position"] in train_pos
            new_test += a["position"] not in train_pos
            shared_palette += (a["background"], a["foreground"]) in train_palettes
            reversed_palette += (a["foreground"], a["background"]) in train_palettes
            repeated_reuse += (a["geometry"][:2], _key(a["repeated"])) in train_motifs
            new_motifs += (a["geometry"][:2], _key(a["repeated"])) not in train_motifs
            area = a["geometry"][0] * a["geometry"][1]
            train_areas = [t["geometry"][0] * t["geometry"][1] for t in train]
            larger_tests += area > max(train_areas)
            smaller_tests += area < min(train_areas)
        for a in all_pairs:
            geometries.add(a["geometry"])
            positions.add(a["position"])
            h, w, sr, sc = a["geometry"]
            source_like_pairs += a["geometry"] == (2, 2, 1, 1)
            delta = int(np.count_nonzero(a["repeated"] != a["odd"]))
            close += delta <= 2
            distant += delta >= 3
            swaps += delta == 2 and int(a["repeated"].sum()) == int(a["odd"].sum())
            additions += delta == 1 and int(a["odd"].sum()) > int(a["repeated"].sum())
            deletions += delta == 1 and int(a["odd"].sum()) < int(a["repeated"].sum())
            rotations += np.array_equal(a["odd"], a["repeated"][::-1, ::-1]) and delta > 0
            reflections += (
                np.array_equal(a["odd"], a["repeated"][::-1, :])
                or np.array_equal(a["odd"], a["repeated"][:, ::-1])
            ) and delta > 0
            blank_odd_edges += not np.all(a["odd"].any(axis=0)) or not np.all(a["odd"].any(axis=1))
            asym_crosses += sr != sc
            large_inputs += 2 * h + sr > 15 or 2 * w + sc > 15
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(geometries) >= 25 and positions == {0, 1, 2, 3}
    assert train_positions_seen == test_positions_seen == {0, 1, 2, 3}
    assert mixed_geometry >= 15 and mixed_background >= 15
    assert close >= 30 and distant >= 15 and swaps >= 10
    assert blank_odd_edges >= 5 and asym_crosses >= 20 and large_inputs >= 10
    assert familiar_test >= 10 and new_test >= 10
    assert shared_palette >= 3 and reversed_palette >= 2
    assert repeated_reuse >= 3 and new_motifs >= 10
    assert additions >= 10 and deletions >= 10
    assert rotations >= 5 and reflections >= 5
    assert rich_motif_training >= 10 and quiet_puzzles >= 2
    assert combined_puzzles >= 10 and source_like_pairs >= 20
    assert larger_tests >= 2 and smaller_tests >= 2
