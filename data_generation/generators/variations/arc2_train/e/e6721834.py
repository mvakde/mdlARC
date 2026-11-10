"""Translation-only completion of rectangular templates from sparse locators."""

import hashlib

import numpy as np


_SOURCE_INPUT_DIGESTS = {
    "7dbefdafed324e8ff9feff877cc0b8e56bf725feb6914e418c191d2e9da04fb5",
    "7237cf46e8cbf3a4922203698aea1d3fc8e6d162a357c30079817219a2ad8a98",
    "f68d44ac6f16328a3896e4309310680d709398b7462a8d2b20af5103199d12ff",
    "6e64576e6a8952a77cd5101db90f4258ec9877ee1ca6b129278c2d80a5de6848",
}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " is outside its legal range")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
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


def _background(panel):
    colors, counts = np.unique(panel, return_counts=True)
    winner = int(np.argmax(counts))
    assert int(counts[winner]) > panel.size // 2
    assert np.count_nonzero(counts == counts[winner]) == 1
    return int(colors[winner])


def _components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(cells)
    return result


def _too_close(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (ar + ah < br or br + bh < ar or ac + aw < bc or bc + bw < ac)


def _parse_library(panel):
    background = _background(panel)
    components = _components(panel != background)
    assert 2 <= len(components) <= 4
    boxes = []
    matrices = []
    for cells in components:
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells) + 1
        c1 = max(c for _, c in cells) + 1
        matrix = panel[r0:r1, c0:c1].copy()
        assert matrix.shape[0] >= 2 and matrix.shape[1] >= 2
        assert matrix.size >= 6 and len(cells) == matrix.size
        box = (r0, c0, r1 - r0, c1 - c0)
        assert all(not _too_close(box, old) for old in boxes)
        boxes.append(box)
        matrices.append(matrix)
    colors, counts = np.unique(np.concatenate([m.ravel() for m in matrices]), return_counts=True)
    body = int(colors[int(np.argmax(counts))])
    assert body != background
    templates = []
    for box, matrix in sorted(zip(boxes, matrices), key=lambda item: item[0][:2]):
        markers = tuple(
            (r, c, int(matrix[r, c]))
            for r in range(matrix.shape[0])
            for c in range(matrix.shape[1])
            if int(matrix[r, c]) != body
        )
        assert 1 <= len(markers) <= 4
        assert np.count_nonzero(matrix == body) > len(markers)
        templates.append({
            "matrix": matrix, "h": matrix.shape[0], "w": matrix.shape[1],
            "markers": markers, "box": box,
        })
    assert len({body, background}) == 2
    return background, body, templates


def _solve_destination(destination, library_background, body, templates):
    background = _background(destination)
    marker_colors = {color for t in templates for _, _, color in t["markers"]}
    assert background not in {library_background, body}
    assert body not in marker_colors and background not in marker_colors
    assert library_background not in marker_colors
    keys = {
        (r, c): int(destination[r, c])
        for r in range(destination.shape[0])
        for c in range(destination.shape[1])
        if int(destination[r, c]) != background
    }
    assert keys and set(keys.values()) <= marker_colors
    candidates = []
    by_key = {key: [] for key in keys}
    for index, template in enumerate(templates):
        first_r, first_c, first_color = template["markers"][0]
        for kr, kc in keys:
            if keys[kr, kc] != first_color:
                continue
            top, left = kr - first_r, kc - first_c
            h, w = template["h"], template["w"]
            if top < 0 or left < 0 or top + h > destination.shape[0] or left + w > destination.shape[1]:
                continue
            expected = {
                (top + r, left + c): color
                for r, c, color in template["markers"]
            }
            if any(keys.get(position) != color for position, color in expected.items()):
                continue
            inside = {position for position in keys if top <= position[0] < top + h and left <= position[1] < left + w}
            if inside != set(expected):
                continue
            candidate = (index, top, left, frozenset(expected), (top, left, h, w))
            candidates.append(candidate)
            for position in expected:
                by_key[position].append(candidate)
    assert all(by_key.values())
    solutions = []

    def search(covered, used, chosen):
        if len(solutions) >= 2:
            return
        if len(covered) == len(keys):
            solutions.append(chosen)
            return
        remaining = set(keys) - covered
        pivot = min(remaining, key=lambda key: len(by_key[key]))
        for candidate in by_key[pivot]:
            index, _, _, occupied, box = candidate
            if index in used or occupied & covered:
                continue
            if any(_too_close(box, other[4]) for other in chosen):
                continue
            search(covered | occupied, used | {index}, chosen + [candidate])

    search(frozenset(), set(), [])
    assert len(solutions) == 1
    output = destination.copy()
    for index, top, left, _, _ in solutions[0]:
        template = templates[index]
        output[top:top + template["h"], left:left + template["w"]] = template["matrix"]
    return background, solutions[0], output


def _predict(input_grid):
    height, width = input_grid.shape
    options = []
    if width % 2 == 0:
        half = width // 2
        options.append(("vertical", (input_grid[:, :half], input_grid[:, half:])))
    if height % 2 == 0:
        half = height // 2
        options.append(("horizontal", (input_grid[:half], input_grid[half:])))
    results = []
    for orientation, panels in options:
        for library_index in (0, 1):
            library, destination = panels[library_index], panels[1 - library_index]
            try:
                library_bg, body, templates = _parse_library(library)
                destination_bg, placements, output = _solve_destination(
                    destination, library_bg, body, templates
                )
                results.append((output, {
                    "orientation": orientation, "library_index": library_index,
                    "panel_shape": destination.shape,
                    "colors": (library_bg, destination_bg, body,
                               tuple(sorted({color for t in templates for _, _, color in t["markers"]}))),
                    "num_templates": len(templates), "num_selected": len(placements),
                    "selected_templates": tuple(p[0] for p in placements),
                    "marker_counts": tuple(len(t["markers"]) for t in templates),
                    "template_sizes": tuple((t["h"], t["w"]) for t in templates),
                    "boxes": tuple(t["box"] for t in templates),
                    "placements": tuple(p[4] for p in placements),
                    "markers": tuple(t["markers"] for t in templates),
                }))
            except (AssertionError, IndexError, ValueError):
                continue
    assert len(results) == 1
    return results[0]


def _pack_boxes(rng, height, width, sizes):
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0] * sizes[i][1])
    for _ in range(18):
        boxes = [None] * len(sizes)
        for index in order:
            h, w = sizes[index]
            candidates = [
                (r, c, h, w)
                for r in range(height - h + 1)
                for c in range(width - w + 1)
                if all(old is None or not _too_close((r, c, h, w), old) for old in boxes)
            ]
            if not candidates:
                break
            boxes[index] = candidates[int(rng.integers(len(candidates)))]
        if all(box is not None for box in boxes):
            return boxes
    return None


def _sample_shape(rng, orientation):
    if orientation == "vertical":
        h, w = int(rng.integers(8, 19)), int(rng.integers(8, 16))
    else:
        h, w = int(rng.integers(8, 16)), int(rng.integers(8, 19))
    if rng.random() < 0.3:
        side = int(rng.integers(8, 16))
        h = w = side
    return h, w


def _sample_sizes(rng, count):
    sizes = []
    cap = 6 if count == 2 else 5 if count == 3 else 4
    for _ in range(count):
        if rng.random() < 0.28:
            h, w = 2, int(rng.integers(3, 9))
            if rng.random() < 0.5:
                h, w = w, h
        else:
            h, w = int(rng.integers(2, cap + 1)), int(rng.integers(2, cap + 1))
            if h == w == 2:
                h = 3
        sizes.append((h, w))
    if count > 1 and rng.random() < 0.27:
        sizes[1] = sizes[0]
    return sizes


def _sample_template(rng, h, w, body, locator_colors, used):
    cells = [(r, c) for r in range(h) for c in range(w)]
    maximum = min(4, (h * w - 1) // 2)
    for _ in range(80):
        count = int(rng.choice(np.arange(1, maximum + 1),
                               p=np.array([0.18, 0.31, 0.31, 0.20][:maximum]) /
                               sum([0.18, 0.31, 0.31, 0.20][:maximum])))
        mode = int(rng.integers(4))
        if mode == 1 and count <= w:
            row = int(rng.integers(h))
            pool = [(row, c) for c in range(w)]
        elif mode == 2 and count <= h:
            col = int(rng.integers(w))
            pool = [(r, col) for r in range(h)]
        else:
            pool = cells
        indices = rng.choice(len(pool), size=count, replace=False)
        positions = [pool[int(i)] for i in indices]
        if mode == 3:
            corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
            positions[0] = corners[int(rng.integers(4))]
            if len(set(positions)) != count:
                continue
        colors = [int(locator_colors[int(rng.integers(len(locator_colors)))]) for _ in positions]
        markers = tuple(sorted((r, c, color) for (r, c), color in zip(positions, colors)))
        signature = (h, w, markers)
        if signature in used:
            continue
        used.add(signature)
        matrix = np.full((h, w), body, dtype=np.int8)
        for r, c, color in markers:
            matrix[r, c] = color
        return {"matrix": matrix, "h": h, "w": w, "markers": markers}
    return None


def _similar_template(rng, original, body, used):
    h, w = original["h"], original["w"]
    markers = original["markers"]
    if len(markers) < 2:
        return None
    occupied = {(r, c) for r, c, _ in markers}
    alternatives = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied]
    for _ in range(20):
        index = int(rng.integers(len(markers)))
        new_r, new_c = alternatives[int(rng.integers(len(alternatives)))]
        changed = list(markers)
        changed[index] = (new_r, new_c, changed[index][2])
        changed = tuple(sorted(changed))
        signature = (h, w, changed)
        if signature in used:
            continue
        used.add(signature)
        matrix = np.full((h, w), body, dtype=np.int8)
        for r, c, color in changed:
            matrix[r, c] = color
        return {"matrix": matrix, "h": h, "w": w, "markers": changed}
    return None


def _generate_pair(rng, orientation, library_index, palette):
    library_bg, destination_bg, body = palette[:3]
    locator_colors = palette[3:4] if rng.random() < 0.45 else palette[3:5]
    for _ in range(350):
        height, width = _sample_shape(rng, orientation)
        count = int(rng.integers(2, 5))
        selected_count = int(rng.integers(1, count + 1))
        sizes = _sample_sizes(rng, count)
        library_boxes = _pack_boxes(rng, height, width, sizes)
        if library_boxes is None:
            continue
        used = set()
        templates = []
        for index, (h, w) in enumerate(sizes):
            template = None
            if index == 1 and templates[0] is not None and sizes[1] == sizes[0] and rng.random() < 0.78:
                template = _similar_template(rng, templates[0], body, used)
            if template is None:
                template = _sample_template(rng, h, w, body, locator_colors, used)
            templates.append(template)
        if any(template is None for template in templates):
            continue
        selected = sorted(int(i) for i in rng.choice(count, size=selected_count, replace=False))
        destination_boxes = _pack_boxes(rng, height, width, [sizes[i] for i in selected])
        if destination_boxes is None:
            continue
        library = np.full((height, width), library_bg, dtype=np.int8)
        destination = np.full((height, width), destination_bg, dtype=np.int8)
        output = destination.copy()
        for template, (top, left, h, w) in zip(templates, library_boxes):
            library[top:top + h, left:left + w] = template["matrix"]
        for index, (top, left, h, w) in zip(selected, destination_boxes):
            template = templates[index]
            for r, c, color in template["markers"]:
                destination[top + r, left + c] = color
            output[top:top + h, left:left + w] = template["matrix"]
        panels = [None, None]
        panels[library_index] = library
        panels[1 - library_index] = destination
        input_grid = np.concatenate(panels, axis=1 if orientation == "vertical" else 0)
        try:
            predicted, info = _predict(input_grid)
            assert np.array_equal(predicted, output)
            assert info["orientation"] == orientation and info["library_index"] == library_index
            assert info["num_templates"] == count and info["num_selected"] == selected_count
            return {"input": input_grid, "output": output}
        except AssertionError:
            continue
    raise RuntimeError("failed to construct an unambiguous pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # Translation matching is the puzzle convention. Colors and geometry are
    # visible pair-local roles, as in the source task.
    layouts = [("vertical", int(rng.integers(2))),
               ("horizontal", int(rng.integers(2)))]
    rng.shuffle(layouts)
    for _ in range(num_train - 2):
        layouts.append(("vertical" if rng.random() < 0.5 else "horizontal",
                        int(rng.integers(2))))
    test_layouts = [
        ("vertical" if rng.random() < 0.5 else "horizontal", int(rng.integers(2)))
        for _ in range(num_test)
    ]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=num_train + num_test, dtype=np.int64)
    # A shared color pool makes role exchanges common, while fresh draws
    # permit colors unseen in earlier pairs.
    pool = [int(x) for x in rng.choice(10, size=7, replace=False)]
    pairs = []
    seen_roles = set()
    for index, layout in enumerate(layouts + test_layouts):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        for _ in range(30):
            source = pool if pair_rng.random() < 0.72 else list(range(10))
            palette = tuple(int(x) for x in pair_rng.choice(source, size=5, replace=False))
            pair = _generate_pair(pair_rng, layout[0], layout[1], palette)
            roles = _predict(pair["input"])[1]["colors"]
            if roles not in seen_roles:
                seen_roles.add(roles)
                pairs.append(pair)
                break
        else:
            raise RuntimeError("failed to vary pair color roles")
    puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _input_digest(grid):
    raw = bytes([grid.shape[0], grid.shape[1]]) + grid.astype(np.uint8).tobytes()
    return hashlib.sha256(raw).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    train_infos = []
    all_infos = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _input_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS
            predicted, info = _predict(pair["input"])
            assert np.array_equal(predicted, pair["output"])
            assert not np.array_equal(pair["input"], pair["output"])
            all_infos.append(info)
            if split == "train":
                train_infos.append(info)
    # Both panel orientations are demonstrated even at the minimum train count.
    assert {info["orientation"] for info in train_infos} == {"horizontal", "vertical"}
    assert len({info["colors"] for info in all_infos}) == len(all_infos)


def _puzzles_equal(first, second):
    return all(
        np.array_equal(a[field], b[field])
        for split in ("train", "test")
        for a, b in zip(first[split], second[split])
        for field in ("input", "output")
    ) and all(len(first[s]) == len(second[s]) for s in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert _puzzles_equal(generate_puzzle(1037, 6, 3), generate_puzzle(1037, 6, 3))
    assert not _puzzles_equal(generate_puzzle(1037, 6, 3), generate_puzzle(1038, 6, 3))
    for seed, train, test in (
        (2001, 2, 1), (2002, 10, 1), (2003, 10, 4),
        (2004, None, 1), (2005, None, 4), (2006, 2, None), (2007, 10, None),
    ):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": 2.0, "num_test": 1},
    ):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            continue
        raise AssertionError("illegal pair count accepted")
    infos = []
    train_infos = []
    test_infos = []
    role_exchanges = 0
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        puzzle_infos = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _predict(pair["input"])[1]
                infos.append(info)
                puzzle_infos.append(info)
                (train_infos if split == "train" else test_infos).append(info)
        if any(
            color in other["colors"][:3]
            for info in puzzle_infos for color in info["colors"][3]
            for other in puzzle_infos if other is not info
        ):
            role_exchanges += 1
        assert len({info["colors"] for info in puzzle_infos}) == len(puzzle_infos)
    assert len({i["panel_shape"] for i in infos}) >= 25
    assert len({(i["orientation"], i["library_index"]) for i in infos}) == 4
    assert {i["num_templates"] for i in infos} == {2, 3, 4}
    assert {i["num_selected"] for i in infos} >= {1, 2, 3}
    assert {n for i in infos for n in i["marker_counts"]} == {1, 2, 3, 4}
    assert any(i["num_selected"] == i["num_templates"] for i in infos)
    assert any(i["num_templates"] - i["num_selected"] >= 2 for i in infos)
    assert any(h == w for i in infos for h, w in [i["panel_shape"]])
    assert any(h == 2 and w >= 6 or w == 2 and h >= 6
               for i in infos for h, w in i["template_sizes"])
    assert any(r == 0 or c == 0 or r + h == i["panel_shape"][0] or c + w == i["panel_shape"][1]
               for i in infos for r, c, h, w in i["placements"])
    assert len({i["colors"] for i in infos}) >= 60
    assert role_exchanges >= 35
    for split_infos in (train_infos, test_infos):
        assert {i["orientation"] for i in split_infos} == {"horizontal", "vertical"}
        assert {i["library_index"] for i in split_infos} == {0, 1}
        assert any(h == w for i in split_infos for h, w in [i["panel_shape"]])
        assert any(h != w for i in split_infos for h, w in [i["panel_shape"]])
        assert any(i["num_selected"] == i["num_templates"] for i in split_infos)
        assert any(i["num_templates"] - i["num_selected"] >= 2 for i in split_infos)
        assert any(len(i["colors"][3]) == 1 and i["num_selected"] < i["num_templates"]
                   for i in split_infos)
        assert any(1 in [i["marker_counts"][j] for j in i["selected_templates"]]
                   for i in split_infos)
    assert any(len({r for r, _, _ in markers}) == 1 and len(markers) >= 2
               for i in infos for markers in i["markers"])
    assert any(len({c for _, c, _ in markers}) == 1 and len(markers) >= 2
               for i in infos for markers in i["markers"])
    similar = 0
    for info in infos:
        for a in range(info["num_templates"]):
            for b in range(a + 1, info["num_templates"]):
                first, second = set(info["markers"][a]), set(info["markers"][b])
                if (info["template_sizes"][a] == info["template_sizes"][b]
                    and len(first) >= 2 and len(first) == len(second)
                    and {color for _, _, color in first} == {color for _, _, color in second}
                    and len(first - second) == len(second - first) == 1):
                    similar += 1
    assert similar >= 8
