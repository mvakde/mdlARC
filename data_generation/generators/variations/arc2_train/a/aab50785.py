"""Generate horizontal two-row bracket extraction puzzles."""

import hashlib
import numpy as np


# SHA-256 of (height, width) as bytes followed by int8 input cells.
_SOURCE_INPUT_DIGESTS = {
    "1653abff94554a2b8a39fe840943175ce4e1b0aa451b3cf4af1f82b4c5e451c9",
    "3e99af334e328ef4d21c6e1b37973e239cee7173618c6285ecede31f5608e3d0",
    "9e9f89d32142f3e901091f20e9556e6c120787a50a866ab15a9b75a2f15d7746",
    "fb6b63723bafe7ed1e3be292d2cf411a339a334a90cdd47d4078cc4ae5c05ab5",
    "4935ebdf8681a8f116dae319005e97f204af4cab07b8435ae44fc8b77f5345fa",
    "f6a905c93a86e8685b68793555cbc4a6afc0a75118a79e867a0a0aa6cb3f71eb",
}


def _count(value, low, high, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _blocks(grid, color):
    cells = grid == color
    hits = cells[:-1, :-1] & cells[1:, :-1] & cells[:-1, 1:] & cells[1:, 1:]
    return [(int(r), int(c)) for r, c in np.argwhere(hits)]


def _packets(grid, marker):
    """Read complete bracket pairs; a lone solid block is a distractor."""
    by_row = {}
    for row, col in _blocks(grid, marker):
        by_row.setdefault(row, []).append(col)
    packets = []
    for row in sorted(by_row):
        cols = sorted(by_row[row])
        if len(cols) == 1:
            continue
        if len(cols) != 2 or cols[1] - cols[0] < 3:
            return None
        left, right = cols
        packets.append((row, left, right, grid[row:row + 2, left + 2:right]))
    if not packets or len({p[3].shape[1] for p in packets}) != 1:
        return None
    return packets


def _prediction(grid, marker):
    packets = _packets(grid, marker)
    return None if packets is None else np.vstack([p[3] for p in packets])


def _train_markers(train):
    candidates = []
    for color in range(10):
        if all((prediction := _prediction(pair["input"], color)) is not None
               and np.array_equal(prediction, pair["output"]) for pair in train):
            candidates.append(color)
    return candidates


def _width(rng):
    u = rng.random()
    if u < 0.17:
        return int(rng.integers(1, 3))
    if u < 0.60:
        return int(rng.integers(3, 7))
    if u < 0.86:
        return int(rng.integers(7, 13))
    return int(rng.integers(13, 27))


def _extra(rng, maximum):
    if maximum == 0:
        return 0
    u = rng.random()
    if u < 0.20:
        return 0
    if u < 0.45:
        return int(rng.integers(1, min(3, maximum) + 1))
    return int(rng.integers(0, maximum + 1))


def _payload(rng, width, marker, background, mode, force_background):
    colors = np.array([c for c in range(10) if c != marker], dtype=np.int8)
    if mode == "plain":
        payload = np.full((2, width), int(rng.choice(colors)), dtype=np.int8)
    elif mode == "sparse":
        payload = np.full((2, width), background, dtype=np.int8)
        count = int(rng.integers(1, max(2, width + 1)))
        places = rng.choice(payload.size, size=min(count, payload.size), replace=False)
        alternatives = colors[colors != background]
        payload.flat[places] = rng.choice(alternatives, size=len(places))
    elif mode == "repeat":
        motif = rng.choice(colors, size=int(rng.integers(1, min(4, width) + 1)))
        payload = np.tile(motif, (2, (width + len(motif) - 1) // len(motif)))[:, :width].astype(np.int8)
        if rng.random() < 0.5:
            payload[1] = np.roll(payload[1], int(rng.integers(0, width)))
    else:
        palette_size = int(rng.integers(2, min(9, max(3, width + 2)) + 1))
        palette = rng.choice(colors, size=palette_size, replace=False)
        payload = rng.choice(palette, size=(2, width)).astype(np.int8)
    if force_background:
        payload.flat[int(rng.integers(payload.size))] = background
    return payload


def _make_pair(rng, marker, packet_count, width, background, clutter, force_mixed=False):
    packet_width = width + 4
    minimum_height = 3 * packet_count - 1
    # Strong clutter reserves a free two-row band for visible complete decoys.
    reserved_tail = 4 if clutter >= 2 else 0
    height = minimum_height + reserved_tail + _extra(rng, 30 - minimum_height - reserved_tail)
    canvas_width = packet_width + _extra(rng, 30 - packet_width)
    slack = height - minimum_height - reserved_tail
    spacing = rng.multinomial(slack, np.ones(packet_count + 1) / (packet_count + 1))
    rows = [int(spacing[0])]
    for index in range(1, packet_count):
        rows.append(rows[-1] + 3 + int(spacing[index]))
    max_left = canvas_width - packet_width
    lefts = []
    for _ in rows:
        choice = rng.random()
        if choice < 0.28:
            lefts.append(0)
        elif choice < 0.56:
            lefts.append(max_left)
        else:
            lefts.append(int(rng.integers(0, max_left + 1)))

    protected = np.zeros((height, canvas_width), dtype=bool)
    grid = np.full((height, canvas_width), background, dtype=np.int8)
    for row, left in zip(rows, lefts):
        protected[row:row + 2, left:left + packet_width] = True
    available = np.array([c for c in range(10) if c != marker and c != background], dtype=np.int8)
    palette_size = int(rng.integers(1, len(available) + 1))
    palette = rng.choice(available, size=palette_size, replace=False)
    density = float(rng.uniform(0.02, 0.35)) if clutter >= 2 else (
        float(rng.uniform(0.02, 0.47)) if clutter else float(rng.uniform(0.0, 0.26))
    )
    noise = (rng.random(grid.shape) < density) & ~protected
    grid[noise] = rng.choice(palette, size=int(np.count_nonzero(noise)))

    payloads = []
    for index, (row, left) in enumerate(zip(rows, lefts)):
        mode = "mixed" if force_mixed and index == 0 else ("plain", "sparse", "repeat", "mixed")[int(rng.integers(0, 4))]
        payload = _payload(rng, width, marker, background, mode, force_mixed and index == 0)
        if force_mixed and index == 0 and len(np.unique(payload)) < 2:
            payload.flat[(int(np.argmax(payload == background)) + 1) % payload.size] = int(rng.choice(available))
        payloads.append(payload)
        grid[row:row + 2, left:left + 2] = marker
        grid[row:row + 2, left + 2:left + 2 + width] = payload
        grid[row:row + 2, left + 2 + width:left + 4 + width] = marker

    intended = {(row, col) for row, left in zip(rows, lefts) for col in (left, left + 2 + width)}
    if clutter >= 2:
        # A different-color square and an unpaired marker square are shown in
        # training before either can appear in a test. The free band keeps
        # both separate from packet brackets even on a compact canvas.
        grid[height - 4:height - 2, 0:2] = background
        decoy_row = height - 2
        grid[decoy_row:height, 0:2] = int(rng.choice(available))
        lone_col = canvas_width - 2
        grid[decoy_row:height, lone_col:canvas_width] = marker
        intended.add((decoy_row, lone_col))
    if clutter:
        for _ in range(int(rng.integers(1, 9))):
            row = int(rng.integers(height))
            col = int(rng.integers(canvas_width))
            if protected[row, col] or grid[row, col] == marker or (clutter >= 2 and row >= height - 4):
                continue
            previous = int(grid[row, col])
            grid[row, col] = marker
            if set(_blocks(grid, marker)) != intended:
                grid[row, col] = previous

    output = np.vstack(payloads).astype(np.int8)
    if set(_blocks(grid, marker)) != intended or not np.array_equal(_prediction(grid, marker), output):
        return None
    return {"input": grid, "output": output}


def _input_digest(grid):
    a = grid.astype(np.int8, copy=False)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    marker = int(rng.integers(1, 10))
    count_ceiling = 3 if rng.random() < 0.30 else 6
    train_counts = [int(rng.integers(2, min(5, count_ceiling) + 1)),
                    int(rng.integers(1, min(4, count_ceiling) + 1))]
    if train_counts[1] == train_counts[0]:
        train_counts[1] = 1
    train_counts.extend(int(x) for x in rng.integers(1, count_ceiling + 1, size=num_train - 2))
    train_widths = [_width(rng), _width(rng)]
    if train_widths[1] == train_widths[0]:
        train_widths[1] = 1 + train_widths[1] % 26
    train_widths.extend(_width(rng) for _ in range(num_train - 2))
    train_backgrounds = [0, int(rng.choice([c for c in range(1, 10) if c != marker]))]
    train_backgrounds.extend(0 if rng.random() < 0.55 else int(rng.choice([c for c in range(1, 10) if c != marker]))
                             for _ in range(num_train - 2))
    train_clutter = [0, 2]
    train_clutter.extend(int(rng.integers(0, 3)) for _ in range(num_train - 2))
    train = []
    test = []
    for index in range(num_train):
        for _ in range(30):
            pair = _make_pair(rng, marker, train_counts[index], train_widths[index],
                              train_backgrounds[index], train_clutter[index], index == 0)
            if pair is not None and _input_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS:
                train.append(pair)
                break
        else:
            raise RuntimeError("could not construct training pair")
    for _ in range(num_test):
        u = rng.random()
        if u < 0.35:
            packet_count = int(rng.integers(1, max(train_counts) + 1))
        elif u < 0.80:
            packet_count = int(rng.integers(min(max(train_counts) + 1, 8), 9))
        else:
            packet_count = int(rng.integers(1, 9))
        width = _width(rng)
        background = 0 if rng.random() < 0.55 else int(rng.choice([c for c in range(1, 10) if c != marker]))
        clutter = int(rng.integers(0, 3))
        for _ in range(30):
            pair = _make_pair(rng, marker, packet_count, width, background, clutter)
            if pair is not None and _input_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS:
                test.append(pair)
                break
        else:
            raise RuntimeError("could not construct test pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _array_key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    pairs = train + test
    keys = set()
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= n <= 30 for n in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
        key = _array_key(pair["input"])
        assert key not in keys
        keys.add(key)
        assert _input_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS
        assert not np.array_equal(pair["input"], pair["output"])
    markers = _train_markers(train)
    assert len(markers) == 1
    marker = markers[0]
    train_packets = []
    train_widths = []
    for index, pair in enumerate(pairs):
        packets = _packets(pair["input"], marker)
        assert packets is not None
        predicted = np.vstack([p[3] for p in packets])
        assert np.array_equal(predicted, pair["output"])
        assert marker not in pair["output"]
        assert pair["output"].shape == (2 * len(packets), packets[0][3].shape[1])
        assert all(packets[i + 1][0] - packets[i][0] >= 3 for i in range(len(packets) - 1))
        if index < len(train):
            train_packets.append(len(packets))
            train_widths.append(packets[0][3].shape[1])
    assert max(train_packets) >= 2 and len(set(train_packets)) >= 2 and len(set(train_widths)) >= 2
    assert any(len(np.unique(pair["output"])) >= 2 for pair in train)
    assert any(int(np.count_nonzero(pair["input"] == marker)) == 8 * len(_packets(pair["input"], marker))
               for pair in train)
    assert any(len(_blocks(pair["input"], marker)) > 2 * len(_packets(pair["input"], marker))
               and sum(bool(_blocks(pair["input"], color)) for color in range(10) if color != marker) >= 2
               for pair in train)
    for pair in test:
        assert np.array_equal(_prediction(pair["input"], marker), pair["output"])


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, markers = set(), set(), set()
    widths, shapes, layouts, palettes, packet_counts = set(), set(), set(), set(), set()
    offsets, gaps = set(), set()
    clean = cluttered = other_blocks = flush = small = large = easy_tests = harder_tests = 0
    unpaired = nonbackground_blocks = sparse_payloads = colorful_payloads = 0
    output_without_zero = few_color_outputs = retained_background = background_changes = 0
    low_count_puzzles = high_count_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        marker = _train_markers(puzzle["train"])[0]
        markers.add(marker)
        max_train = max(len(_packets(p["input"], marker)) for p in puzzle["train"])
        low_count_puzzles += max_train <= 3
        high_count_puzzles += max_train >= 4
        train_backgrounds = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                packets = _packets(grid, marker)
                assert np.array_equal(np.vstack([grid[r:r + 2, l + 2:rr] for r, l, rr, _ in packets]), pair["output"])
                count = len(packets)
                packet_counts.add(count)
                widths.add(pair["output"].shape[1])
                shapes.add(grid.shape)
                layouts.add((grid.shape, tuple((r, l, rr) for r, l, rr, _ in packets)))
                palettes.add(len(np.unique(grid)))
                marker_cells = int(np.count_nonzero(grid == marker))
                if marker_cells == 8 * count:
                    clean += 1
                else:
                    cluttered += 1
                if any(_blocks(grid, c) for c in range(10) if c != marker):
                    other_blocks += 1
                unpaired += len(_blocks(grid, marker)) > 2 * count
                if any(l == 0 and rr + 2 == grid.shape[1] for _, l, rr, _ in packets):
                    flush += 1
                small += grid.size <= 120
                large += grid.size >= 500
                output_without_zero += not np.any(pair["output"] == 0)
                few_color_outputs += len(np.unique(pair["output"])) <= 2
                outside = np.ones(grid.shape, dtype=bool)
                for row, left, right, payload in packets:
                    outside[row:row + 2, left:right + 2] = False
                    offsets.add(left)
                    sparse_payloads += np.unique(payload, return_counts=True)[1].max() / payload.size >= 0.75
                    colorful_payloads += len(np.unique(payload)) >= 4
                gaps.update(packets[i + 1][0] - packets[i][0] - 2 for i in range(count - 1))
                if np.any(outside):
                    background = int(np.bincount(grid[outside].ravel(), minlength=10).argmax())
                    nonbackground_blocks += any(_blocks(grid, color) for color in range(10)
                                                if color not in (background, marker))
                    retained_background += np.any(pair["output"] == background)
                    if split == "train":
                        train_backgrounds.append(background)
                if split == "test":
                    easy_tests += count <= max_train
                    harder_tests += count > max_train
        background_changes += len(set(train_backgrounds)) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(markers) >= 6 and len(widths) >= 10 and len(shapes) >= 40
    assert len(layouts) >= 100 and len(palettes) >= 5 and len(packet_counts) >= 7
    assert min(widths) <= 2 and max(widths) >= 13
    assert all(x >= 3 for x in (clean, cluttered, other_blocks, flush, small, large, easy_tests, harder_tests))
    assert all(x >= 3 for x in (unpaired, nonbackground_blocks, sparse_payloads, colorful_payloads,
                                output_without_zero, few_color_outputs, retained_background))
    assert background_changes >= 30 and len(offsets) >= 10 and len(gaps) >= 5
    assert min(palettes) <= 3 and max(palettes) >= 9
    assert low_count_puzzles >= 3 and high_count_puzzles >= 3
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(7000 + (ntrain or 0) * 10 + (ntest or 0), ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    a = generate_puzzle(9001, 7, 3)
    assert _puzzles_equal(a, generate_puzzle(9001, 7, 3))
    assert not _puzzles_equal(a, generate_puzzle(9002, 7, 3))
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(10001, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts were accepted")
