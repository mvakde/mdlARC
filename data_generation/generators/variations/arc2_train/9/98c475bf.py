import hashlib

import numpy as np


_SOURCE_PAIR_HASHES = frozenset({
    "8a24f86a817778a65b4c6233805c9f3be9e8134f038111f7e876ea7d8761162f",
    "21d5b6bd044f88f8eeff0115121e5ea2b1b3d0908e7fed9b7dbb5d2bfeb8a3d0",
    "4d5190595a5b128d631c2931fa614e643a48df6e8a223696aed7c180ac178c58",
    "2ab14226fd70ae4ff318f19b898cf5e8530e88f0f764e9535e7ce1c85e219e56",
    "2bd86aa747865b0a2511c7959f49f43c41bbdfda0cb7cc94e8f580b3eb6239a4",
})


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif num_train is None:
        test = int(num_test)
        train = max(2, test + 1) + seed % (11 - max(2, test + 1))
    elif num_test is None:
        train = int(num_train)
        test = 1 + seed % min(4, train - 1)
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _sample_mask(rng: np.random.Generator, width: int) -> frozenset[tuple[int, int]]:
    span = _randint(rng, 3, min(6, width - 6))
    anchor = _randint(rng, 2, width - 2 - span)
    above = int(rng.choice((1, 1, 1, 2, 2, 3)))
    below = int(rng.choice((1, 1, 1, 2, 2, 3)))
    family = _randint(rng, 0, 4)
    offsets: set[tuple[int, int]] = set()
    for dr in range(-above, below + 1):
        if dr == 0:
            continue
        if family == 0:  # outline, including the twelve-cell source-sized type
            columns = range(span) if dr in (-above, below) else (0, span - 1)
        elif family == 1:  # filled block
            columns = range(span)
        elif family == 2:  # parallel posts, sometimes three
            columns = (0, span - 1) if span < 4 else (0, span // 2, span - 1)
        elif family == 3:  # tapered or asymmetric diamond
            reach = min(abs(dr), span - 2)
            columns = range(reach, span) if dr < 0 else range(0, span - reach)
        else:  # disconnected columns with a cap on one side
            columns = range(span) if dr == -above or dr == below else (0, span - 1)
            if dr > 0 and span > 3 and dr != below:
                columns = (0, span - 2)
        offsets.update((dr, anchor + dc) for dc in columns)
    carrier = set(range(1, width - 1))
    mode = int(rng.choice((0, 0, 0, 1, 1, 1, 1, 2)))
    if mode == 1:
        length = _randint(rng, 1, min(4, span))
        start = _randint(rng, anchor, anchor + span - length)
        carrier.difference_update(range(start, start + length))
    elif mode == 2:
        length = _randint(rng, 1, min(4, width - 6))
        start = _randint(rng, 2, width - 2 - length)
        carrier.difference_update(range(start, start + length))
    offsets.update((0, column) for column in carrier)
    return frozenset(offsets)


def _bounds(mask: frozenset[tuple[int, int]]) -> tuple[int, int]:
    drs = [dr for dr, _ in mask]
    return min(drs), max(drs)


def _sample_conventions(rng: np.random.Generator, train_count: int) -> dict:
    width = _randint(rng, 14, 29)
    vocabulary_size = _randint(rng, 2, min(7, train_count + 1))
    labels = tuple(int(x) for x in rng.permutation(10)[:vocabulary_size])
    masks: dict[int, frozenset[tuple[int, int]]] = {}
    for label in labels:
        for _ in range(100):
            mask = _sample_mask(rng, width)
            if mask not in masks.values():
                masks[label] = mask
                break
        else:
            raise RuntimeError("could not sample distinct motifs")
    return {"width": width, "labels": labels, "masks": masks}


def _training_schedule(rng: np.random.Generator, labels: tuple[int, ...],
                       train_count: int) -> list[tuple[int, int]]:
    order = [int(x) for x in rng.permutation(np.asarray(labels))]
    schedule = [(order[0], order[1])]
    if len(order) == 2:
        schedule.append((order[1], order[0]))
    else:
        schedule.extend((order[i], order[i + 1]) for i in range(1, len(order) - 1))
    while len(schedule) < train_count:
        complete, request = rng.choice(np.asarray(labels), size=2, replace=False)
        schedule.append((int(complete), int(request)))
    rng.shuffle(schedule)
    return schedule


def _choose_specs(rng: np.random.Generator, regime: str,
                  labels: tuple[int, ...], core: tuple[int, int] | None,
                  masks: dict[int, frozenset[tuple[int, int]]]) -> list[tuple[str, int]]:
    if regime == "source_train":
        assert core is not None
        full = [core[0]]
        requests = [core[1]]
        if float(rng.random()) < 0.42:
            requests.append(int(rng.choice([x for x in labels if x not in full])))
        if len(requests) <= 2 and float(rng.random()) < 0.35:
            choices = [x for x in labels if x not in full and x not in requests]
            if choices:
                full.append(int(rng.choice(choices)))
    elif regime == "target_test":
        full = [int(rng.choice(np.asarray(labels)))]
        request_count = int(rng.choice((1, 2, 3), p=(0.18, 0.52, 0.30)))
        available = [x for x in labels if x not in full]
        requests = [int(rng.choice(available))]
        while len(requests) < request_count:
            pool = available if float(rng.random()) < 0.30 else [
                x for x in available if x not in requests
            ]
            requests.append(int(rng.choice(pool if pool else available)))
        if len(requests) <= 2 and float(rng.random()) < 0.42:
            choices = [x for x in labels if x not in full and x not in requests]
            if choices:
                full.append(int(rng.choice(choices)))
    else:
        raise ValueError("unknown regime")
    specs = [("full", label) for label in full] + [("request", label) for label in requests]
    while sum(_bounds(masks[label])[1] - _bounds(masks[label])[0] + 1
              for _, label in specs) + len(specs) - 1 > 30:
        if len(requests) > 1:
            requests.pop()
        elif len(full) > 1:
            full.pop()
        else:
            raise RuntimeError("two motifs cannot fit")
        specs = [("full", label) for label in full] + [("request", label) for label in requests]
    return specs


def _layout(rng: np.random.Generator, specs: list[tuple[str, int]],
            masks: dict[int, frozenset[tuple[int, int]]]) -> tuple[int, list[int]]:
    indices = [int(x) for x in rng.permutation(len(specs))]
    arranged = [specs[i] for i in indices]
    minimum = sum(_bounds(masks[label])[1] - _bounds(masks[label])[0] + 1
                  for _, label in arranged) + len(arranged) - 1
    extra = _randint(rng, 0, min(9, 30 - minimum))
    if float(rng.random()) < 0.12:
        extra = 30 - minimum
    height = minimum + extra
    gap_extra = [0] * (len(specs) - 1)
    for _ in range(_randint(rng, 0, min(extra, len(gap_extra) * 2))):
        if not gap_extra:
            break
        gap_extra[_randint(rng, 0, len(gap_extra) - 1)] += 1
    remaining = extra - sum(gap_extra)
    top = int(rng.choice((0, remaining, _randint(rng, 0, remaining))))
    cursor = top
    rows = [0] * len(specs)
    for position, original_index in enumerate(indices):
        _, label = arranged[position]
        low, high = _bounds(masks[label])
        rows[original_index] = cursor - low
        cursor += high - low + 1
        if position < len(gap_extra):
            cursor += 1 + gap_extra[position]
    return height, rows


def _colors(rng: np.random.Generator, labels: tuple[int, ...],
            active: set[int]) -> tuple[int, int]:
    background = int(rng.choice([x for x in range(10) if x not in active]))
    reused = [x for x in labels if x not in active and x != background]
    fresh = [x for x in range(10) if x not in labels and x != background]
    if reused and (not fresh or float(rng.random()) < 0.5):
        rail = int(rng.choice(reused))
    else:
        rail = int(rng.choice(fresh))
    return background, rail


def _blank(height: int, width: int, background: int, rail: int) -> np.ndarray:
    grid = np.full((height, width), background, dtype=np.int8)
    grid[:, 0] = rail
    grid[:, -1] = rail
    return grid


def _stamp(grid: np.ndarray, cells: frozenset[tuple[int, int]], row: int,
           label: int, background: int) -> None:
    for dr, column in cells:
        y = row + dr
        if not (0 <= y < grid.shape[0] and 1 <= column < grid.shape[1] - 1):
            raise RuntimeError("motif out of bounds")
        if int(grid[y, column]) != background:
            raise RuntimeError("motifs overlap")
        grid[y, column] = label


def _pair(rng: np.random.Generator, regime: str, conventions: dict,
          core: tuple[int, int] | None) -> dict:
    labels = conventions["labels"]
    masks = conventions["masks"]
    width = conventions["width"]
    specs = _choose_specs(rng, regime, labels, core, masks)
    height, rows = _layout(rng, specs, masks)
    background, rail = _colors(rng, labels, {label for _, label in specs})
    before = _blank(height, width, background, rail)
    after = before.copy()
    endpoints = frozenset({(0, 1), (0, width - 2)})
    for (kind, label), row in zip(specs, rows):
        if kind == "full":
            _stamp(before, masks[label], row, label, background)
        else:
            _stamp(before, endpoints, row, label, background)
            _stamp(after, masks[label], row, label, background)
    return {"input": before, "output": after}


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    conventions = _sample_conventions(rng, train_count)
    schedule = _training_schedule(rng, conventions["labels"], train_count)
    puzzle: dict = {"train": [], "test": []}
    seen: set[tuple] = set()
    for split, cores in (("train", schedule), ("test", [None] * test_count)):
        regime = "source_train" if split == "train" else "target_test"
        for core in cores:
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(40):
                pair = _pair(pair_rng, regime, conventions, core)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray) -> None:
    _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid is not a 2D array")
    _check(np.issubdtype(grid.dtype, np.integer), "grid has noninteger dtype")
    _check(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
           "grid dimensions out of range")
    _check(0 <= int(grid.min()) <= int(grid.max()) <= 9, "invalid ARC color")


def _analyze(grid: np.ndarray) -> dict:
    height, width = grid.shape
    _check(width >= 14, "rail-bounded canvas too narrow")
    rail = int(grid[0, 0])
    _check(np.all(grid[:, 0] == rail) and np.all(grid[:, -1] == rail),
           "rails are not uniform")
    interior = grid[:, 1:-1]
    colors, counts = np.unique(interior, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    _check(background != rail, "rail and background coincide")
    occupied = np.any(interior != background, axis=1)
    objects = []
    start = None
    for row in range(height + 1):
        nonempty = row < height and bool(occupied[row])
        if nonempty and start is None:
            start = row
        if not nonempty and start is not None:
            band = grid[start:row, 1:-1]
            nonbackground = {int(v) for v in np.unique(band) if int(v) != background}
            _check(len(nonbackground) == 1, "different motifs share a row band")
            label = next(iter(nonbackground))
            _check(label != rail, "active motif has rail color")
            local = np.argwhere(band == label)
            coordinates = {(start + int(y), 1 + int(x)) for y, x in local}
            carriers = [y for y in range(start, row)
                        if (y, 1) in coordinates and (y, width - 2) in coordinates]
            _check(len(carriers) == 1, "motif lacks one clear carrier")
            carrier = carriers[0]
            endpoints = {(carrier, 1), (carrier, width - 2)}
            kind = "request" if coordinates == endpoints else "full"
            mask = frozenset((y - carrier, x) for y, x in coordinates)
            if kind == "full":
                _check(sum(y == carrier for y, _ in coordinates) >= 4,
                       "full carrier too sparse")
                _check(sum(y != carrier for y, _ in coordinates) >= 4,
                       "full ornament too sparse")
                _check(all(2 <= x <= width - 3 for y, x in coordinates if y != carrier),
                       "ornament touches endpoint column")
            objects.append({"kind": kind, "label": label,
                            "carrier": carrier, "mask": mask,
                            "cells": coordinates})
            start = None
    _check(len(objects) > 0, "grid contains no motifs")
    return {"background": background, "rail": rail, "width": width,
            "height": height, "objects": objects}


def _expected(info: dict, codebook: dict[int, frozenset[tuple[int, int]]]) -> np.ndarray:
    grid = _blank(info["height"], info["width"], info["background"], info["rail"])
    for obj in info["objects"]:
        if obj["kind"] == "request":
            _check(obj["label"] in codebook, "request lacks a training template")
            for dr, x in codebook[obj["label"]]:
                y = obj["carrier"] + dr
                _check(0 <= y < info["height"] and 1 <= x < info["width"] - 1,
                       "completed mask crosses canvas")
                _check(int(grid[y, x]) == info["background"], "completed masks overlap")
                grid[y, x] = obj["label"]
    return grid


def _source_hash(pair: dict) -> str:
    return hashlib.sha256(pair["input"].astype(np.int8).tobytes() +
                          pair["output"].astype(np.int8).tobytes()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle needs train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "splits must be lists")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
           and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    records: dict = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    widths: set[int] = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair needs input and output")
            before, after = pair["input"], pair["output"]
            _validate_grid(before)
            _validate_grid(after)
            _check(before.shape == after.shape and before.dtype == after.dtype,
                   "shape or dtype changed")
            _check(not np.array_equal(before, after), "unchanged pair")
            input_key = _grid_key(before)
            pair_key = (input_key, _grid_key(after))
            _check(input_key not in seen_inputs and pair_key not in seen_pairs,
                   "duplicate puzzle pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _check(_source_hash(pair) not in _SOURCE_PAIR_HASHES, "source pair copied")
            input_info, output_info = _analyze(before), _analyze(after)
            _check(input_info["background"] == output_info["background"] and
                   input_info["rail"] == output_info["rail"], "local colors changed")
            widths.add(input_info["width"])
            records[split].append((pair, input_info, output_info))
    _check(len(widths) == 1, "rail-relative templates need one puzzle width")
    codebook: dict[int, frozenset[tuple[int, int]]] = {}
    full_input_labels: set[int] = set()
    request_input_labels: set[int] = set()
    for pair, before, after in records["train"]:
        full = [o for o in before["objects"] if o["kind"] == "full"]
        requests = [o for o in before["objects"] if o["kind"] == "request"]
        _check(1 <= len(full) <= 2 and 1 <= len(requests) <= 2,
               "training object regime invalid")
        _check(len({o["label"] for o in full}) == len(full),
               "complete labels repeat within a pair")
        _check(not ({o["label"] for o in full} & {o["label"] for o in requests}),
               "same label is complete and requested locally")
        _check(len([o for o in after["objects"] if o["kind"] == "full"]) == len(requests)
               and not any(o["kind"] == "request" for o in after["objects"]),
               "training output has wrong completion count")
        full_input_labels.update(o["label"] for o in full)
        request_input_labels.update(o["label"] for o in requests)
        for obj in full + after["objects"]:
            if obj["label"] in codebook:
                _check(codebook[obj["label"]] == obj["mask"],
                       "contradictory label template")
            else:
                codebook[obj["label"]] = obj["mask"]
    _check(bool(full_input_labels & request_input_labels),
           "training lacks a cross-pair complete-to-request link")
    _check(len(codebook) >= 2 and len(set(codebook.values())) == len(codebook),
           "codebook is too small or ambiguous")
    for split in ("train", "test"):
        for pair, before, after in records[split]:
            full = [o for o in before["objects"] if o["kind"] == "full"]
            requests = [o for o in before["objects"] if o["kind"] == "request"]
            if split == "test":
                _check(1 <= len(full) <= 2 and 1 <= len(requests) <= 3,
                       "test object regime invalid")
                _check(not ({o["label"] for o in full} &
                            {o["label"] for o in requests}), "local roles conflict")
            _check(all(o["label"] in codebook and
                       o["mask"] == codebook[o["label"]] for o in full),
                   "input complete motif contradicts codebook")
            _check(np.array_equal(_expected(before, codebook), pair["output"]),
                   "output disagrees with learned deletion and completion")
            _check(all(o["kind"] == "full" for o in after["objects"]) and
                   len(after["objects"]) == len(requests),
                   "output has incorrect motifs")


def _fingerprint(puzzle: dict) -> tuple:
    return tuple((split, _grid_key(p["input"]), _grid_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    for i, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4), (4, 2),
                                         (7, 3), (None, 1), (None, 4),
                                         (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + i, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            _check(len(puzzle["train"]) == ntrain, "train count not honored")
        if ntest is not None:
            _check(len(puzzle["test"]) == ntest, "test count not honored")
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(12, *args)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    _check(_fingerprint(generate_puzzle(314159, 6, 3)) ==
           _fingerprint(generate_puzzle(314159, 6, 3)), "not deterministic")
    _check(_fingerprint(generate_puzzle(271828)) ==
           _fingerprint(generate_puzzle(271828)), "automatic not deterministic")
    train_counts, test_counts, widths, heights = set(), set(), set(), set()
    masks, layouts, fingerprints = set(), set(), set()
    regimes, backgrounds, rails = set(), set(), set()
    vocabulary_sizes: set[int] = set()
    cross_links = 0
    edge_motifs = 0
    varied_backgrounds = 0
    repeated_requests = 0
    large_ornaments = 0
    tall_ornaments = 0
    full_carriers = 0
    coherent_gaps = 0
    reused_rails = 0
    fresh_rails = 0
    combined_regimes = 0
    shared_train_test_regimes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_full = set()
        train_requests = set()
        pair_backgrounds = set()
        pair_rails = []
        puzzle_codebook = {}
        pair_regimes = {"train": set(), "test": set()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                widths.add(info["width"])
                heights.add(info["height"])
                backgrounds.add(info["background"])
                pair_backgrounds.add(info["background"])
                rails.add(info["rail"])
                pair_rails.append(info["rail"])
                full = [o for o in info["objects"] if o["kind"] == "full"]
                requests = [o for o in info["objects"] if o["kind"] == "request"]
                pair_regimes[split].add((len(full), len(requests)))
                regimes.add((split, len(full), len(requests),
                             len({o["label"] for o in requests})))
                repeated_requests += len(requests) > len({o["label"] for o in requests})
                combined_regimes += len(full) > 1 and len(requests) > 1
                layouts.add((info["height"], tuple((o["kind"], o["carrier"])
                                                     for o in info["objects"])))
                if split == "train":
                    train_full.update(o["label"] for o in full)
                    train_requests.update(o["label"] for o in requests)
                    for obj in full:
                        puzzle_codebook[obj["label"]] = obj["mask"]
                output_objects = _analyze(pair["output"])["objects"]
                if split == "train":
                    for obj in output_objects:
                        puzzle_codebook[obj["label"]] = obj["mask"]
                for obj in output_objects:
                    masks.add((info["width"], tuple(sorted(obj["mask"]))))
                    if min(y for y, _ in obj["cells"]) == 0 or \
                       max(y for y, _ in obj["cells"]) == info["height"] - 1:
                        edge_motifs += 1
        cross_links += bool(train_full & train_requests)
        varied_backgrounds += len(pair_backgrounds) > 1
        shared_train_test_regimes += bool(pair_regimes["train"] & pair_regimes["test"])
        vocabulary_sizes.add(len(puzzle_codebook))
        reused_rails += sum(rail in puzzle_codebook for rail in pair_rails)
        fresh_rails += sum(rail not in puzzle_codebook for rail in pair_rails)
        for mask in puzzle_codebook.values():
            off = [(dr, col) for dr, col in mask if dr]
            large_ornaments += len(off) >= 12
            tall_ornaments += len({dr for dr, _ in off}) >= 5
            gaps = [col for col in range(2, puzzle["train"][0]["input"].shape[1] - 2)
                    if (0, col) not in mask]
            full_carriers += not gaps
            coherent_gaps += bool(gaps) and any(
                min(col for _, col in off) <= gap <= max(col for _, col in off)
                for gap in gaps)
    _check(len(train_counts) >= 4 and len(test_counts) >= 3,
           "automatic count diversity insufficient")
    _check(len(fingerprints) >= 40 and len(widths) >= 8 and len(heights) >= 12,
           "puzzle or dimension diversity insufficient")
    _check(len(masks) >= 50 and len(layouts) >= 80,
           "motif or layout diversity insufficient")
    _check(len(backgrounds) >= 7 and len(rails) >= 7,
           "color diversity insufficient")
    _check(cross_links == 42 and edge_motifs > 0,
           "cross-pair links or edge placements absent")
    _check(len(vocabulary_sizes) >= 5 and 20 in widths and
           varied_backgrounds >= 35 and reused_rails > 20 and fresh_rails > 20,
           "vocabulary, canvas, or local-color diversity insufficient")
    _check(repeated_requests >= 20 and combined_regimes >= 10 and
           shared_train_test_regimes >= 20,
           "within-pair or train/test regime diversity insufficient")
    _check(large_ornaments >= 20 and tall_ornaments >= 10 and
           full_carriers >= 20 and coherent_gaps >= 20,
           "ornament and carrier diversity insufficient")
    _check(any(s[0] == "train" and s[2] == 2 for s in regimes) and
           any(s[0] == "test" and s[2] == 1 for s in regimes) and
           any(s[0] == "test" and s[2] == 3 for s in regimes) and
           any(s[1] == 2 for s in regimes) and
           any(s[2] > s[3] for s in regimes), "requested regimes absent")
