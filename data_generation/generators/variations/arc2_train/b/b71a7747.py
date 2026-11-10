"""Lossless deinterleaving of aligned image tiles."""
import hashlib
import itertools
import numpy as np

_SOURCE_DIGESTS = frozenset((
    "be0b1ed1cc20a2de23e0bfc2d947174d869454a21bd2cb51ae637717d775f9cd",
    "cb644567afd0bb07f02566801a2ae33ea0ee5b94f2c421d6a666ee3e216dfbe3",
    "f1f8b6d8626b0074b712cd9622a04be7acbc2e8467496b3658dbd88c1f0cc99d",
))


def _require(value, message):
    if not bool(value):
        raise AssertionError(message)


def _counts(rng, n, m):
    for value, low, high in ((n, 2, 10), (m, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError("count is outside its legal integer range")
    if n is None and m is None:
        n = int(rng.choice((2, 2, 2, 2, 2, 3, 3, 3, 3,
                            4, 4, 5, 6, 7, 8, 9, 10)))
        m = int(rng.integers(1, min(4, n - 1) + 1))
    elif n is None:
        n = int(rng.integers(max(2, m + 1), 11))
    elif m is None:
        m = int(rng.integers(1, min(4, n - 1) + 1))
    if n <= m:
        raise ValueError("train count must exceed test count")
    return int(n), int(m)


def _axis(rng, size=None, count=None, wide=False):
    if size is None:
        size = int(rng.integers(3, 7))
    if count is None:
        count = int(rng.choice([k for k in range(2, 6)
                                if k * size + k + 1 <= 30]))
    if count * size + count + 1 > 30:
        raise ValueError("axis does not fit")
    values = [1] * (count + 1)  # leading margin, gaps, trailing margin
    caps = [5] + [4] * (count - 1) + [5]
    available = min(30 - count * size - sum(values),
                    sum(cap - 1 for cap in caps))
    spent = int(rng.integers(available // 2 if wide else 0, available + 1))
    for _ in range(spent):
        options = [i for i in range(len(values)) if values[i] < caps[i]]
        if not options:
            break
        values[int(rng.choice(options))] += 1
    return {"count": count, "size": size, "lead": values[0],
            "gaps": values[1:-1], "trail": values[-1]}


def _layout(rng, profile, anchor):
    sizes = {"first3": (3, 3), "second3": (3, 3), "first5": (5, 5),
             "grow": (5, 5), "shrink": (3, 3),
             "same": anchor, "gap": anchor}.get(profile)
    if sizes is None:
        sizes = (None, None)
    if profile in ("first3", "second3"):
        counts = (int(rng.integers(3, 5)), int(rng.integers(3, 5)))
    elif profile == "first5":
        counts = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
    else:
        counts = (None, None)
    return (_axis(rng, sizes[0], counts[0], profile == "gap"),
            _axis(rng, sizes[1], counts[1], profile == "gap"))


def _band_mask(axis):
    length = axis["lead"] + axis["count"] * axis["size"] + sum(axis["gaps"]) + axis["trail"]
    mask = np.zeros(length, dtype=bool)
    pos = axis["lead"]
    for i in range(axis["count"]):
        mask[pos:pos + axis["size"]] = True
        pos += axis["size"] + (axis["gaps"][i] if i < axis["count"] - 1 else 0)
    return mask


def _line(mask, a, b, vertical_first):
    r, c = a
    er, ec = b
    if vertical_first:
        mask[min(r, er):max(r, er) + 1, c] = True
        mask[er, min(c, ec):max(c, ec) + 1] = True
    else:
        mask[r, min(c, ec):max(c, ec) + 1] = True
        mask[min(r, er):max(r, er) + 1, ec] = True


def _drawing(rng, height, width, tile_h, tile_w, connected):
    mask = np.zeros((height, width), dtype=bool)
    if connected:
        points = []
        for tile_r in range(height // tile_h):
            order = list(range(width // tile_w))
            if tile_r % 2:
                order.reverse()
            for tile_c in order:
                points.append((tile_r * tile_h + int(rng.integers(0, tile_h)),
                               tile_c * tile_w + int(rng.integers(0, tile_w))))
        mask[points[0]] = True
        for a, b in zip(points, points[1:]):
            _line(mask, a, b, bool(rng.integers(0, 2)))
        cap = int(height * width * .47)
        for _ in range(int(rng.integers(0, 5))):
            occupied = np.argwhere(mask)
            r, c = occupied[int(rng.integers(0, len(occupied)))]
            target = (min(height - 1, max(0, int(r) + int(rng.integers(-4, 5)))),
                      min(width - 1, max(0, int(c) + int(rng.integers(-4, 5)))))
            proposed = mask.copy()
            _line(proposed, (int(r), int(c)), target, bool(rng.integers(0, 2)))
            if int(proposed.sum()) <= cap:
                mask = proposed
        if int(rng.integers(0, 5)) == 0:
            for r, c in np.argwhere(mask)[::max(3, int(mask.sum()) // 8)]:
                proposed = mask.copy()
                proposed[int(r):min(height, int(r) + 2), int(c)] = True
                if int(proposed.sum()) <= cap:
                    mask = proposed
    else:
        for _ in range(int(rng.integers(2, 9))):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            if bool(rng.integers(2)):
                end = (r, min(width - 1, max(0, c + int(rng.integers(-width // 2, width // 2 + 1)))))
            else:
                end = (min(height - 1, max(0, r + int(rng.integers(-height // 2, height // 2 + 1)))), c)
            _line(mask, (r, c), end, True)
        if not mask.any():
            mask[height // 2, width // 2] = True
    return mask


def _palette(rng, previous, reverse, scaffold_count):
    image_count = 3 if int(rng.integers(7)) == 0 else 2
    if reverse and previous is not None:
        image = [int(rng.choice(previous["scaffold"]))]
        scaffold = [int(rng.choice(previous["image"]))]
        pool = [c for c in range(10) if c not in image + scaffold]
        rng.shuffle(pool)
        image += pool[:image_count - 1]
        scaffold += pool[image_count - 1:image_count + scaffold_count - 2]
    else:
        colors = [int(x) for x in rng.choice(10, image_count + scaffold_count, replace=False)]
        image, scaffold = colors[:image_count], colors[image_count:]
    return {"image": image, "scaffold": scaffold}


def _scaffold(rng, height, width, colors, style, row_mask, col_mask):
    grid = np.full((height, width), colors[0], dtype=np.int8)
    if len(colors) == 1:
        return grid
    removed_rows = np.flatnonzero(~row_mask)
    removed_cols = np.flatnonzero(~col_mask)
    if style == "stripes":
        grid[removed_rows[::max(2, len(removed_rows) // 4)], :] = colors[1]
        grid[:, removed_cols[::max(2, len(removed_cols) // 4)]] = colors[-1]
    elif style == "markers":
        for r in removed_rows:
            for c in removed_cols:
                if int(rng.integers(3)) == 0:
                    grid[r, c] = colors[1]
        grid[removed_rows[len(removed_rows) // 2],
             removed_cols[len(removed_cols) // 2]] = colors[1]
        if len(colors) == 3:
            grid[removed_rows[0], removed_cols[-1]] = colors[2]
    elif style == "texture":
        rr, cc = np.indices((height, width))
        period = int(rng.integers(3 if len(colors) == 3 else 2, 5))
        phase = int(rng.integers(period))
        grid[(rr + cc + phase) % period == 0] = colors[1]
        if len(colors) == 3:
            grid[(rr + cc + phase) % period == 1] = colors[2]
    return grid


def _make_pair(rng, profile, anchor, previous, reverse, style, connected):
    rows, cols = _layout(rng, profile, anchor)
    scaffold_count = 1 if style == "flat" else int(rng.choice((2, 2, 3)))
    palette = _palette(rng, previous, reverse, scaffold_count)
    height, width = rows["count"] * rows["size"], cols["count"] * cols["size"]
    mask = _drawing(rng, height, width, rows["size"], cols["size"], connected)
    output = np.full((height, width), palette["image"][0], dtype=np.int8)
    output[mask] = palette["image"][1]
    if len(palette["image"]) == 3:
        positions = np.argwhere(mask)
        chosen = positions[rng.choice(len(positions), size=max(1, len(positions) // 7), replace=False)]
        output[chosen[:, 0], chosen[:, 1]] = palette["image"][2]
    row_mask, col_mask = _band_mask(rows), _band_mask(cols)
    input_grid = _scaffold(rng, len(row_mask), len(col_mask), palette["scaffold"],
                           style, row_mask, col_mask)
    input_grid[np.ix_(np.flatnonzero(row_mask), np.flatnonzero(col_mask))] = output
    return {"input": input_grid, "output": output}, palette


def _groups(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    return [(int(a[0]), int(a[-1]) + 1) for a in np.split(indices, cuts)]


def _extractions(grid):
    colors = [int(c) for c in np.unique(grid)]
    found = []
    for size in (2, 3):
        for palette in itertools.combinations(colors, size):
            cells = np.isin(grid, palette)
            rows, cols = cells.any(axis=1), cells.any(axis=0)
            if np.all(rows) or np.all(cols):
                continue
            if not np.array_equal(cells, rows[:, None] & cols[None, :]):
                continue
            row_groups, col_groups = _groups(rows), _groups(cols)
            if not (2 <= len(row_groups) <= 5 and 2 <= len(col_groups) <= 5):
                continue
            row_sizes = {b - a for a, b in row_groups}
            col_sizes = {b - a for a, b in col_groups}
            if len(row_sizes) != 1 or len(col_sizes) != 1:
                continue
            if min(row_sizes) < 3 or min(col_sizes) < 3:
                continue
            row_gaps = [row_groups[i + 1][0] - row_groups[i][1]
                        for i in range(len(row_groups) - 1)]
            col_gaps = [col_groups[i + 1][0] - col_groups[i][1]
                        for i in range(len(col_groups) - 1)]
            margins = (row_groups[0][0], grid.shape[0] - row_groups[-1][1],
                       col_groups[0][0], grid.shape[1] - col_groups[-1][1])
            if not (all(1 <= g <= 4 for g in row_gaps + col_gaps)
                    and all(1 <= m <= 5 for m in margins)):
                continue
            output = grid[np.ix_(np.flatnonzero(rows), np.flatnonzero(cols))].copy()
            found.append({"palette": frozenset(palette), "rows": rows, "cols": cols,
                          "row_groups": row_groups, "col_groups": col_groups,
                          "gaps": (row_gaps, col_gaps), "margins": margins,
                          "output": output,
                          "tile_shape": (next(iter(row_sizes)), next(iter(col_sizes)))})
    return found


def _digest(pair):
    h = hashlib.sha256()
    for name in ("input", "output"):
        a = np.asarray(pair[name], dtype=np.uint8)
        h.update(bytes(a.shape))
        h.update(a.tobytes())
    return h.hexdigest()


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _analyze(pair):
    found = _extractions(pair["input"])
    _require(len(found) == 1, "input has ambiguous tile bands")
    a = found[0]
    _require(np.array_equal(a["output"], pair["output"]), "incorrect extracted output")
    values, counts = np.unique(pair["output"], return_counts=True)
    _require(2 <= len(values) <= 3 and int(counts.max()) * 2 > pair["output"].size,
             "image must have a dominant base and drawing")
    scaffold_cells = pair["input"][~(a["rows"][:, None] & a["cols"][None, :])]
    scaffold = frozenset(int(x) for x in np.unique(scaffold_cells))
    _require(1 <= len(scaffold) <= 3 and a["palette"].isdisjoint(scaffold),
             "image and scaffold palettes overlap")
    a["scaffold_palette"] = scaffold
    return a


def _scaffold_kind(grid, analysis):
    rows, cols = analysis["rows"], analysis["cols"]
    if len(analysis["scaffold_palette"]) == 1:
        return "flat"
    outside = ~(rows[:, None] & cols[None, :])
    crossing = ~rows[:, None] & ~cols[None, :]
    for base in analysis["scaffold_palette"]:
        if not np.any((grid != base) & outside & ~crossing):
            return "markers"
    for base in analysis["scaffold_palette"]:
        row_colors = {}
        col_colors = {}
        for r in np.flatnonzero(~rows):
            values = grid[r, cols]
            if not np.all(values == values[0]):
                break
            row_colors[int(r)] = int(values[0])
        else:
            for c in np.flatnonzero(~cols):
                values = grid[rows, c]
                if not np.all(values == values[0]):
                    break
                col_colors[int(c)] = int(values[0])
            else:
                if all(grid[r, c] == (col_colors[int(c)] if col_colors[int(c)] != base
                                      else row_colors[int(r)])
                       for r in np.flatnonzero(~rows)
                       for c in np.flatnonzero(~cols)):
                    return "stripes"
    return "texture"


def _predict(train, test_input):
    for pair in train:
        possibilities = _extractions(pair["input"])
        _require(len(possibilities) == 1 and
                 np.array_equal(possibilities[0]["output"], pair["output"]),
                 "training does not reveal extraction")
    possibilities = _extractions(test_input)
    _require(len(possibilities) == 1, "test extraction is ambiguous")
    return possibilities[0]["output"]


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle schema is invalid")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts are invalid")
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair schema is invalid")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D NumPy array")
                _require(all(1 <= d <= 30 for d in grid.shape), "grid dimensions invalid")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype invalid")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "grid colors invalid")
            key = _grid_key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(_digest(pair) not in _SOURCE_DIGESTS, "source pair copied")
            analyses[split].append(_analyze(pair))
    train_shapes = {(a["tile_shape"], len(a["row_groups"]), len(a["col_groups"]))
                    for a in analyses["train"]}
    _require(len(train_shapes) >= 2, "training layouts do not vary")
    _require(len({a["palette"] for a in analyses["train"]}) >= 2,
             "training palettes do not vary")
    _require(len({_scaffold_kind(pair["input"], a)
                  for pair, a in zip(puzzle["train"], analyses["train"])}) >= 2,
             "training scaffold styles do not vary")
    _require(bool(analyses["train"][0]["palette"] &
                  analyses["train"][1]["scaffold_palette"]) and
             bool(analyses["train"][0]["scaffold_palette"] &
                  analyses["train"][1]["palette"]),
             "training must show that color roles can reverse")
    for pair in puzzle["test"]:
        _require(np.array_equal(_predict(puzzle["train"], pair["input"]),
                                pair["output"]), "test output is not inferable")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _counts(rng, num_train, num_test)
    mode = str(rng.choice(("grow", "grow", "same", "gap", "shrink", "free")))
    first_profile = "first5" if mode == "shrink" else "first3"
    anchor = (5, 5) if first_profile == "first5" else (3, 3)
    first_style = str(rng.choice(("flat", "stripes", "markers")))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous = None
    for split, total in (("train", n_train), ("test", n_test)):
        for index in range(total):
            if split == "train" and index == 0:
                profile, style, connected = first_profile, first_style, True
            elif split == "train" and index == 1:
                profile = "second3" if mode == "grow" else "free"
                style = str(rng.choice([s for s in ("flat", "stripes", "markers", "texture")
                                        if s != first_style]))
                connected = bool(rng.integers(0, 4))
            else:
                profile = mode if split == "test" else "free"
                style = str(rng.choice(("flat", "stripes", "stripes",
                                        "markers", "texture", "texture")))
                connected = bool(rng.integers(0, 4) != 0)
            for _ in range(40):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair, _ = _make_pair(child, profile, anchor, previous,
                                           split == "train" and index == 1,
                                           style, connected)
                key = _grid_key(pair["input"])
                if key in seen:
                    continue
                try:
                    analysis = _analyze(pair)
                except AssertionError:
                    continue
                if split == "train" and index == 1:
                    first = _analyze(puzzle["train"][0])
                    if (first["tile_shape"], len(first["row_groups"]), len(first["col_groups"])) == (
                        analysis["tile_shape"], len(analysis["row_groups"]),
                        len(analysis["col_groups"])):
                        continue
                seen.add(key)
                puzzle[split].append(pair)
                previous = {"image": sorted(analysis["palette"]),
                            "scaffold": sorted(analysis["scaffold_palette"])}
                break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    _require(_puzzles_equal(generate_puzzle(123), generate_puzzle(123)),
             "generation is not deterministic")
    train_counts, test_counts = set(), set()
    all_inputs, layouts, scaffold_counts = set(), set(), set()
    connected = fragmented = rectangular = uneven = 0
    all_seams = reversals = three_color_images = asymmetric = 0
    styles = set()
    compact = same_size = growth = shrink = smaller_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        compact += len(puzzle["train"]) <= 3 and len(puzzle["test"]) == 1
        first_train = _analyze(puzzle["train"][0])
        first_test = _analyze(puzzle["test"][0])
        same_size += first_train["tile_shape"] == first_test["tile_shape"]
        growth += all(y > x for x, y in zip(first_train["tile_shape"],
                                             first_test["tile_shape"]))
        shrink += all(y < x for x, y in zip(first_train["tile_shape"],
                                             first_test["tile_shape"]))
        smaller_test += any(p["output"].size < max(t["output"].size
                                 for t in puzzle["train"]) for p in puzzle["test"])
        previous = None
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                styles.add(_scaffold_kind(pair["input"], a))
                three_color_images += len(a["palette"]) == 3
                asymmetric += (a["margins"][0] != a["margins"][1] or
                               a["margins"][2] != a["margins"][3])
                if previous is not None:
                    reversals += (bool(a["palette"] & previous[1]) and
                                  bool(a["scaffold_palette"] & previous[0]))
                previous = (a["palette"], a["scaffold_palette"])
                all_inputs.add(_grid_key(pair["input"]))
                layouts.add((len(a["row_groups"]), len(a["col_groups"]),
                             a["tile_shape"], tuple(map(tuple, a["gaps"])), a["margins"]))
                scaffold_counts.add(len(a["scaffold_palette"]))
                rectangular += a["tile_shape"][0] != a["tile_shape"][1]
                uneven += (len(set(a["gaps"][0])) > 1 or len(set(a["gaps"][1])) > 1)
                output = pair["output"]
                values, frequencies = np.unique(output, return_counts=True)
                ink = output != values[int(np.argmax(frequencies))]
                remaining = set(map(tuple, np.argwhere(ink)))
                components = 0
                while remaining:
                    components += 1
                    stack = [remaining.pop()]
                    while stack:
                        r, c = stack.pop()
                        for near in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                            if near in remaining:
                                remaining.remove(near)
                                stack.append(near)
                connected += components == 1
                fragmented += components > 1
                all_seams += (all(np.any(ink[b - 1, :] & ink[b, :])
                                  for b in range(a["tile_shape"][0], output.shape[0],
                                                 a["tile_shape"][0])) and
                              all(np.any(ink[:, b - 1] & ink[:, b])
                                  for b in range(a["tile_shape"][1], output.shape[1],
                                                 a["tile_shape"][1])))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic counts lack range")
    _require(len(all_inputs) >= 100 and len(layouts) >= 70,
             "structural diversity lacks range")
    _require(scaffold_counts == {1, 2, 3}, "scaffold palette sizes lack range")
    _require(connected >= 50 and fragmented >= 15,
             "image connectivity lacks range")
    _require(rectangular >= 10 and uneven >= 10,
             "tile or separator geometry lacks range")
    _require(styles == {"flat", "stripes", "markers", "texture"},
             "scaffold geometry lacks range")
    _require(compact >= 5 and same_size >= 3 and growth >= 3 and
             shrink >= 3 and smaller_test >= 3,
             "train/test schedule lacks range")
    _require(reversals >= 30 and three_color_images >= 10 and
             asymmetric >= 30 and all_seams >= 50,
             "color roles, margins, or image continuity lack range")
    for i, (n, m) in enumerate(((2, 1), (10, 1), (10, 4),
                                (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(5000 + i, n, m)
        if n is not None:
            _require(len(puzzle["train"]) == n, "train count ignored")
        if m is not None:
            _require(len(puzzle["test"]) == m, "test count ignored")
    for n, m in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(999, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
