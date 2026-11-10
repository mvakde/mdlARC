"""Independent four-block ARC panel generator."""

import hashlib
import numpy as np

SCALES = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (4, 4))
SOURCE_HASHES = {
    (9, 9, "30e32a2e0f0e972f299f69ecb6bd97444659083865c583709a0163da847f8ae5"),
    (19, 9, "4f302a6c29f95aa63336a0942e2f2734a2a15fe6539605e374e2202f40c24ca9"),
    (9, 9, "a86a417d1da492b15602bc97cf2562515c1d18c8ce8b150f36f4890bf0178cda"),
    (19, 19, "69c263db39d30ce9cd745dba06b070c441fc1d15d9132cfbc625981167777675"),
    (19, 29, "431d634a0041f40208646c007ad1e5d6408c2770dcc9ff4119886717add1e2c9"),
    (29, 29, "b4ca8f7c5f4ba9a029c91c55c6e61be8c2f162c011e4b6e04bad38e2b9c6dcb0"),
}


def _valid_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, seed, train, test):
    train = _valid_count("num_train", train, 2, 10)
    test = _valid_count("num_test", test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _options(scale):
    bh, bw = scale
    ph, pw = 2 * bh + 3, 2 * bw + 3
    return [(r, c, bh, bw) for r in range(1, 4) for c in range(1, 4)
            if r * (ph + 1) - 1 <= 30 and c * (pw + 1) - 1 <= 30]


def _geometry(rng, scale, previous, minimum, target, classic=False):
    choices = [g for g in _options(scale) if g[0] * g[1] >= minimum]
    if classic:
        familiar = {(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)}
        choices = [g for g in choices if g[:2] in familiar]
    if previous is not None and rng.random() < 0.75:
        different = [g for g in choices if g[:2] != previous[:2]]
        if different:
            choices = different
    if target and rng.random() < 0.55:
        large = [g for g in choices if g[0] * g[1] >= 4]
        if large:
            choices = large
    return choices[int(rng.integers(len(choices)))]


def _types(rng, count, forced):
    mode = int(rng.integers(5))
    if mode == 0:
        kind = ("v", "h", "u")[int(rng.integers(3))]
        result = [kind] * count
    elif mode == 1:
        result = list(rng.choice(np.array(["u", "u", "u", "v", "h"]), size=count))
    elif mode == 2:
        result = list(rng.choice(np.array(["v", "v", "h", "h", "u"]), size=count))
    elif mode == 3:
        result = ["u"] * count
        result[int(rng.integers(count))] = ("v", "h")[int(rng.integers(2))]
    else:
        result = list(rng.choice(np.array(["v", "h", "u"]), size=count))
    for i, kind in forced.items():
        result[i] = kind
    return result


def _matrix(rng, kind, palette, witness):
    if kind == "u":
        a, b = rng.choice(palette, size=2, replace=False)
        result = np.full((2, 2), a, dtype=np.int8)
        result.flat[int(rng.integers(4))] = b
        return result
    if witness:
        return rng.permutation(palette)[:4].reshape((2, 2)).astype(np.int8)
    if rng.random() < 0.13:
        a, b = rng.choice(palette, size=2, replace=False)
        if kind == "v":
            return np.array([[a, a], [b, b]], dtype=np.int8)
        return np.array([[a, b], [a, b]], dtype=np.int8)
    for _ in range(12):
        result = rng.choice(palette, size=(2, 2), replace=True).astype(np.int8)
        reflected = result[:, ::-1] if kind == "v" else result[::-1, :]
        if not np.array_equal(result, reflected):
            return result
    return np.array([[palette[0], palette[1]], [palette[1], palette[0]]], dtype=np.int8)


def _paint(panel, matrix, bh, bw):
    for r in range(2):
        for c in range(2):
            y, x = 1 + r * (bh + 1), 1 + c * (bw + 1)
            panel[y:y + bh, x:x + bw] = matrix[r, c]


def _make_pair(rng, geom, forced, settings, target=False, require_new=False):
    rows, cols, bh, bw = geom
    ph, pw = 2 * bh + 3, 2 * bw + 3
    background = settings["background"]
    if background is None:
        background = int(rng.choice(np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
    available = np.array([x for x in range(10) if x != background], dtype=np.int8)
    pool = settings["palette"] if target or settings["train_palette"] is None else settings["train_palette"]
    pool = available if pool is None else np.array([x for x in pool if x != background], dtype=np.int8)
    lower = 4 if any(k in ("v", "h") for k in forced.values()) else 2
    palette = rng.choice(pool, size=int(rng.integers(lower, len(pool) + 1)), replace=False)
    novel = None
    if require_new:
        candidates = [int(x) for x in available if x not in settings["train_palette"]]
        novel = int(rng.choice(np.array(candidates, dtype=np.int8)))
        if novel not in palette:
            palette[0] = novel
    kinds = _types(rng, rows * cols, forced)
    novel_panel = int(rng.integers(rows * cols)) if novel is not None else None
    divider = int(rng.choice(available))
    shared_control = int(rng.choice(available)) if rng.random() < 0.45 else None
    height, width = rows * (ph + 1) - 1, cols * (pw + 1) - 1
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for r in range(ph, height, ph + 1):
        inp[r, :] = divider
        out[r, :] = divider
    for c in range(pw, width, pw + 1):
        inp[:, c] = divider
        out[:, c] = divider
    for i, kind in enumerate(kinds):
        r, c = divmod(i, cols)
        y, x = r * (ph + 1), c * (pw + 1)
        panel = inp[y:y + ph, x:x + pw]
        target = out[y:y + ph, x:x + pw]
        matrix = _matrix(rng, kind, palette, i in forced and kind != "u")
        if i == novel_panel:
            if kind == "u":
                other = int(rng.choice(np.array([x for x in palette if x != novel], dtype=np.int8)))
                matrix[:, :] = other
                matrix.flat[int(rng.integers(4))] = novel
            else:
                position = int(rng.integers(4))
                matrix.flat[position] = novel
                mirror = position ^ (1 if kind == "v" else 2)
                if matrix.flat[mirror] == novel:
                    matrix.flat[mirror] = int(rng.choice(np.array([x for x in palette if x != novel], dtype=np.int8)))
        if kind == "v":
            color = shared_control if shared_control is not None else int(rng.choice(available))
            panel[:, bw + 1] = color
            target[:, bw + 1] = color
            answer = matrix[:, ::-1]
        elif kind == "h":
            color = shared_control if shared_control is not None else int(rng.choice(available))
            panel[bh + 1, :] = color
            target[bh + 1, :] = color
            answer = matrix[::-1, :]
        else:
            values, counts = np.unique(matrix, return_counts=True)
            answer = np.full((2, 2), values[np.argmin(counts)], dtype=np.int8)
        _paint(panel, matrix, bh, bw)
        _paint(target, answer, bh, bw)
    if np.array_equal(inp, out):
        i = next(i for i, k in enumerate(kinds) if k in ("v", "h"))
        r, c = divmod(i, cols)
        y, x = r * (ph + 1), c * (pw + 1)
        panel = inp[y:y + ph, x:x + pw]
        target = out[y:y + ph, x:x + pw]
        matrix = np.array([[palette[0], palette[1]], [palette[1], palette[0]]], dtype=np.int8)
        _paint(panel, matrix, bh, bw)
        _paint(target, matrix[:, ::-1] if kinds[i] == "v" else matrix[::-1, :], bh, bw)
    return inp, out


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _source_copy(grid):
    digest = hashlib.sha256(grid.astype(np.int8, copy=False).tobytes()).hexdigest()
    return (grid.shape[0], grid.shape[1], digest) in SOURCE_HASHES


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, seed, num_train, num_test)
    classic = seed % 7 == 0
    fixed_scale = (3, 3) if classic else (
        SCALES[int(rng.integers(len(SCALES)))] if rng.random() < 0.25 else None)
    background = None
    if classic:
        background = 0 if rng.random() < 0.6 else int(rng.integers(1, 10))
    elif rng.random() < 0.36:
        background = int(rng.choice(np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
    shared_palette = None
    if classic or rng.random() < 0.38:
        available = [x for x in range(10) if x != background]
        lower = 4 if background is not None else 6
        high = min(7, len(available)) + 1 if classic else len(available) + 1
        size = int(rng.integers(lower, high))
        shared_palette = tuple(int(x) for x in rng.choice(available, size=size, replace=False))
    train_palette = None
    if not classic and shared_palette is None and rng.random() < 0.42:
        available = [x for x in range(10) if x != background]
        lower = 5 if background is not None else 6
        size = int(rng.integers(lower, min(7, len(available)) + 1))
        train_palette = tuple(int(x) for x in rng.choice(available, size=size, replace=False))
    settings = {"background": background, "palette": shared_palette,
                "train_palette": train_palette}
    geometries = []
    for i in range(ntrain + ntest):
        scale = fixed_scale or SCALES[int(rng.integers(len(SCALES)))]
        minimum = max(1, 3 - geometries[0][0] * geometries[0][1]) if i == 1 else 1
        if i == 0 and ntrain == 2:
            minimum = 2
        geometries.append(_geometry(rng, scale, geometries[-1] if geometries else None,
                                    minimum, i >= ntrain, classic))
    slots = [(i, j) for i in range(2) for j in range(geometries[i][0] * geometries[i][1])]
    forced = [dict() for _ in geometries]
    for kind, selected in zip(("v", "h", "u"), rng.choice(len(slots), size=3, replace=False)):
        i, j = slots[int(selected)]
        forced[i][j] = kind
    puzzle = {"train": [], "test": []}
    seen = set()
    for i, geom in enumerate(geometries):
        section = "train" if i < ntrain else "test"
        for _ in range(24):
            inp, out = _make_pair(rng, geom, forced[i], settings, section == "test",
                                  i == ntrain and train_palette is not None)
            if _key(inp) not in seen and not _source_copy(inp):
                break
        else:
            raise RuntimeError("could not construct a distinct pair")
        seen.add(_key(inp))
        puzzle[section].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(grid, geom):
    rows, cols, bh, bw = geom
    ph, pw = 2 * bh + 3, 2 * bw + 3
    assert grid.shape == (rows * (ph + 1) - 1, cols * (pw + 1) - 1)
    background = int(grid[0, 0])
    lines = [grid[r, :] for r in range(ph, grid.shape[0], ph + 1)]
    lines += [grid[:, c] for c in range(pw, grid.shape[1], pw + 1)]
    divider = None
    if lines:
        divider = int(lines[0][0])
        assert divider != background and all(np.all(line == divider) for line in lines)
    expected = grid.copy()
    kinds, witnesses, controls, matrices = [], [], [], []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        y, x = r * (ph + 1), c * (pw + 1)
        panel = grid[y:y + ph, x:x + pw]
        target = expected[y:y + ph, x:x + pw]
        matrix = np.empty((2, 2), dtype=np.int8)
        mask = np.zeros((ph, pw), dtype=bool)
        for br in range(2):
            for bc in range(2):
                yy, xx = 1 + br * (bh + 1), 1 + bc * (bw + 1)
                block = panel[yy:yy + bh, xx:xx + bw]
                assert np.all(block == block[0, 0]) and int(block[0, 0]) != background
                matrix[br, bc] = block[0, 0]
                mask[yy:yy + bh, xx:xx + bw] = True
        vertical = int(panel[0, bw + 1]) != background
        horizontal = int(panel[bh + 1, 0]) != background
        assert not (vertical and horizontal)
        gutter = np.full((ph, pw), background, dtype=np.int8)
        if vertical:
            kind = "v"
            color = int(panel[0, bw + 1])
            gutter[:, bw + 1] = color
            answer = matrix[:, ::-1]
            controls.append((kind, color))
        elif horizontal:
            kind = "h"
            color = int(panel[bh + 1, 0])
            gutter[bh + 1, :] = color
            answer = matrix[::-1, :]
            controls.append((kind, color))
        else:
            kind = "u"
            values, counts = np.unique(matrix, return_counts=True)
            assert len(values) == 2 and sorted(counts.tolist()) == [1, 3]
            answer = np.full((2, 2), values[np.argmin(counts)], dtype=np.int8)
        assert np.array_equal(panel[~mask], gutter[~mask])
        kinds.append(kind)
        matrices.append(tuple(int(z) for z in matrix.flat))
        if kind in ("v", "h") and len(set(matrix.flatten().tolist())) == 4:
            witnesses.append(kind)
        _paint(target, answer, bh, bw)
    return expected, {"layout": (rows, cols), "scale": (bh, bw),
                      "background": background, "divider": divider,
                      "kinds": tuple(kinds), "witnesses": tuple(witnesses),
                      "controls": tuple(controls), "matrices": tuple(matrices)}


def _infer(grid):
    candidates = []
    for bh in range(1, 5):
        for bw in range(1, 5):
            ph, pw = 2 * bh + 3, 2 * bw + 3
            for rows in range(1, 4):
                if rows * (ph + 1) - 1 != grid.shape[0]:
                    continue
                for cols in range(1, 4):
                    if cols * (pw + 1) - 1 != grid.shape[1]:
                        continue
                    try:
                        candidates.append(_inspect(grid, (rows, cols, bh, bw)))
                    except (AssertionError, IndexError):
                        pass
    assert len(candidates) == 1, "panel geometry must be recoverable from the input"
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, metadata = set(), {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert not _source_copy(inp)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            answer, info = _infer(inp)
            assert np.array_equal(out, answer)
            metadata[section].append(info)
    train_kinds = {kind for info in metadata["train"] for kind in info["kinds"]}
    train_witnesses = {kind for info in metadata["train"] for kind in info["witnesses"]}
    assert train_kinds == {"v", "h", "u"}
    assert train_witnesses == {"v", "h"}


def _same(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    layouts, scales, backgrounds, mixtures = set(), set(), set(), set()
    exceptions, placeholders = set(), set()
    observed = {name: 0 for name in (
        "quiet", "shared_axis_color", "different_axis_color", "same_axis_recolored",
        "large_train_small_test", "train_rich", "test_compact", "combined",
        "classic", "first_marked_repeated", "background_changes", "divider_changes")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        info_by_section = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                answer, info = _infer(pair["input"])
                assert np.array_equal(answer, pair["output"])
                info_by_section[section].append(info)
                layouts.add(info["layout"])
                scales.add(info["scale"])
                backgrounds.add(info["background"])
                mixtures.add(tuple(info["kinds"].count(k) for k in ("v", "h", "u")))
                controls = info["controls"]
                if any(a != b and ca == cb for a, ca in controls for b, cb in controls):
                    observed["shared_axis_color"] += 1
                if any(a != b and ca != cb for a, ca in controls for b, cb in controls):
                    observed["different_axis_color"] += 1
                if any(a == b and ca != cb for a, ca in controls for b, cb in controls):
                    observed["same_axis_recolored"] += 1
                for kind, matrix in zip(info["kinds"], info["matrices"]):
                    if kind == "u":
                        counts = {v: matrix.count(v) for v in set(matrix)}
                        exceptions.add(matrix.index(next(v for v, n in counts.items() if n == 1)))
                        placeholders.add(next(v for v, n in counts.items() if n == 3))
                    elif (kind == "v" and matrix == (matrix[1], matrix[0], matrix[3], matrix[2])) or (
                            kind == "h" and matrix == (matrix[2], matrix[3], matrix[0], matrix[1])):
                        observed["quiet"] += 1
                if section == "train" and len({v for matrix in info["matrices"] for v in matrix}) >= 6:
                    observed["train_rich"] += 1
                if section == "test":
                    if len({v for matrix in info["matrices"] for v in matrix}) <= 4:
                        observed["test_compact"] += 1
                    first = next((m for k, m in zip(info["kinds"], info["matrices"]) if k != "u"), None)
                    if first is not None and len(set(first)) < 4:
                        observed["first_marked_repeated"] += 1
                if info["scale"] != (3, 3) and info["layout"][0] * info["layout"][1] >= 4 and len(set(info["kinds"])) < 3:
                    observed["combined"] += 1
        all_info = info_by_section["train"] + info_by_section["test"]
        if max(i["layout"][0] * i["layout"][1] for i in info_by_section["train"]) > min(
                i["layout"][0] * i["layout"][1] for i in info_by_section["test"]):
            observed["large_train_small_test"] += 1
        if len({i["background"] for i in all_info}) > 1:
            observed["background_changes"] += 1
        if len({i["divider"] for i in all_info if i["divider"] is not None}) > 1:
            observed["divider_changes"] += 1
        if all(i["scale"] == (3, 3) and i["layout"] in {(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)} for i in all_info):
            observed["classic"] += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(layouts) >= 8
    assert len(scales) >= 7 and len(backgrounds) >= 8 and len(mixtures) >= 12
    assert exceptions == {0, 1, 2, 3} and len(placeholders) >= 6
    assert all(value > 0 for value in observed.values()), observed
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        puzzle = generate_puzzle(1000 + i, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + i, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    assert _same(generate_puzzle(314159, 8, 3), generate_puzzle(314159, 8, 3))
    assert not _same(generate_puzzle(314159, 8, 3), generate_puzzle(314160, 8, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count accepted: {kwargs}")
