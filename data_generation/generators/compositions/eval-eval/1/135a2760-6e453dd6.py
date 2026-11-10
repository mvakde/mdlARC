"""Sealed-window waveguides: repair geometry, dock it, transmit its phase.

No parent code, examples, files, generation state, or RNG is used by solve.
Generation and tests retain no files. All randomized searches have finite caps.
"""

import hashlib
import json
from collections import deque

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue, cells = deque([(r, c)]), []
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        result.append(cells)
    return result


def _enclosed(card, bg):
    """Four-connected background pockets; boundary-connected pockets are open."""
    result = np.zeros(card.shape, dtype=bool)
    h, w = card.shape
    for cells in _components(card == bg):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            rr, cc = np.array(cells).T
            result[rr, cc] = True
    return result


def _gates(card, bg):
    enclosed = _enclosed(card, bg)
    return enclosed[:, -2] & (card[:, -1] != bg)


def _scene(grid):
    """Return the unique rotation with cards left of a rail and blank right side."""
    candidates = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        h, w = a.shape
        for col in range(1, w - 1):
            rail = int(a[0, col])
            if not np.all(a[:, col] == rail) or np.count_nonzero(a == rail) != h:
                continue
            bg = int(a[0, -1])
            if bg == rail or not np.all(a[:, col + 1:] == bg):
                continue
            if np.any(a[:, :col] != bg):
                candidates.append((a, turn, bg, rail, col))
    _require(len(candidates) == 1, "rail and projection side must be unique")
    return candidates[0]


def _recover(motif, bg, ink):
    """All supported strict-majority periodic explanations must agree.

    Each phase has at least three observations. Up to max(1, area//12)
    disagreements are allowed, with no ties at any phase/row coordinate.
    The smallest period is selected only after agreement on the whole repair.
    """
    h, length = motif.shape
    candidates = []
    cap = max(1, motif.size // 12)
    for period in range(1, length // 3 + 1):
        template = np.empty((h, period), dtype=np.int8)
        errors = 0
        for phase in range(period):
            block = motif[:, phase::period]
            counts = np.count_nonzero(block == ink, axis=1)
            if np.any(2 * counts == block.shape[1]):
                break
            template[:, phase] = np.where(2 * counts > block.shape[1], ink, bg)
            errors += int(np.minimum(counts, block.shape[1] - counts).sum())
        else:
            if errors <= cap and len(np.unique(template)) == 2:
                clean = template[:, np.arange(length) % period]
                candidates.append((period, clean, template, errors))
    _require(bool(candidates), "no supported periodic repair")
    first = candidates[0]
    _require(all(np.array_equal(first[1], c[1]) for c in candidates), "ambiguous periodic repair")
    return first[1].copy(), first[0], first[2].copy(), first[3]


def _objects(a, bg, col):
    objects = []
    previous_end = -2
    for cells in _components(a[:, :col] != bg):
        rr, cc = np.array(cells).T
        top, bottom, left, right = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
        _require(top > previous_end + 1, "cards need separated row bands")
        previous_end = bottom
        card = a[top:bottom + 1, left:right + 1].copy()
        h, w = card.shape
        _require(h >= 3 and w >= 8, "card too small")
        frame = int(card[0, 0])
        _require(frame != bg and np.all(card[np.ix_([0, h - 1], [0, w - 1])] == frame),
                 "card corners must establish its frame")
        boundary = np.ones(card.shape, dtype=bool)
        boundary[1:-1, 1:-1] = False
        _require(set(map(int, np.unique(card[boundary]))) <= {bg, frame}, "invalid shell")
        motif = card[1:-1, 1:-1]
        inks = set(map(int, np.unique(motif))) - {bg}
        _require(len(inks) == 1 and frame not in inks, "interior must use a distinct ink and background")
        ink = inks.pop()
        repaired, period, template, errors = _recover(motif, bg, ink)
        clean = card.copy()
        clean[1:-1, 1:-1] = repaired
        _require(len(_components(clean != bg)) == 1, "repaired card disconnected")
        objects.append(dict(top=top, left=left, right=right, card=card, clean=clean,
                            frame=frame, ink=ink, period=period, template=template,
                            errors=errors, shift=col - 1 - right, length=w - 2))
    return objects


def _transform(grid, variant="rule"):
    a, turn, bg, rail, col = _scene(grid)
    objects = _objects(a, bg, col)
    out = a.copy()
    out[:, :col] = bg
    for obj in objects:
        card, clean = obj["card"], obj["clean"]
        h, w = clean.shape
        top = obj["top"]
        out[top:top + h, col - w:col] = clean
        gates = _gates(clean, bg)
        if variant == "raw_topology":
            gates = _gates(card, bg)
        elif variant == "one_column":
            gates = clean[:, -2] == bg
            gates[[0, -1]] = False
        elif variant == "all_holes":
            gates = np.any(_enclosed(clean, bg), axis=1)
        elif variant == "no_topology":
            gates = np.zeros(h, dtype=bool)
            gates[1:-1] = np.any(clean[1:-1, 1:-1] == bg, axis=1)
        phase = obj["length"] + 2  # cap and rail occlude two motif positions
        if variant == "reset_phase":
            phase = 0
        elif variant == "unmoved_phase":
            phase += obj["shift"]
        positions = (phase + np.arange(a.shape[1] - col - 1)) % obj["period"]
        for row in np.flatnonzero(gates):
            _require(0 < row < h - 1, "shell row cannot emit")
            out[top + row, col + 1:] = (obj["ink"] if variant == "solid_rays" else
                                         obj["template"][row - 1, positions])
    return np.rot90(out, -turn).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _transform(_grid(input_grid))


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                    (num_test, 1, 4, "num_test")):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
                 name + " must be an integer")
        _require(low <= value <= high, name + " out of range")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _partition(rng, height, count):
    # Positive chamber heights, separated by one full ink row.
    spare = height - (2 * count - 1)
    heights = np.ones(count, dtype=int) + rng.multinomial(spare, [1 / count] * count)
    bands, row = [], 0
    for value in heights:
        bands.append((row, row + int(value)))
        row += int(value) + 1
    return bands


def _unit(rng, height, length, period, mode):
    bands = _partition(rng, height, int(rng.integers(1, min(3, (height + 1) // 2) + 1)))
    bits = np.ones((height, period), dtype=np.int8)
    q = (length - 1) % period
    for i, (start, stop) in enumerate(bands):
        active = mode != "recessed" and (i == 0 or rng.random() < .65)
        end = q if active else (q - 1) % period
        width = int(rng.integers(1, period))
        for row in range(start, stop):
            extent = width if rng.random() < .6 else int(rng.integers(1, period))
            bits[row, (end - np.arange(extent)) % period] = 0
    return bits, bands


def _shell(motif, bg, frame, mode, rng, bands):
    card = np.full((motif.shape[0] + 2, motif.shape[1] + 2), frame, dtype=np.int8)
    card[1:-1, 1:-1] = motif
    if mode == "open":
        # A deliberate shell doorway, outside the repairable binary interior.
        start, stop = bands[0]
        if start == 0 and int(rng.integers(2)) == 0:
            card[0, -2] = bg
        else:
            card[1 + int(rng.integers(start, stop)), -1] = bg
    elif mode == "mixed_open":
        # Some chambers are open while another leading chamber can still emit.
        start, stop = bands[0]
        card[1 + int(rng.integers(start, stop)), -1] = bg
    return card


def _damage(rng, clean, bg, ink, period, force_leading=False):
    interior = clean[1:-1, 1:-1]
    cap = min(3, max(1, interior.size // 12))
    for attempt in range(48):
        noisy = clean.copy()
        target = noisy[1:-1, 1:-1]
        count = 1 if attempt >= 32 else int(rng.integers(1, cap + 1))
        chosen, groups = [], set()
        if force_leading:
            rows = np.flatnonzero(interior[:, -1] == bg)
            row = int(rng.choice(rows))
            chosen.append((row, interior.shape[1] - 1))
            groups.add((row, (interior.shape[1] - 1) % period))
        for index in rng.permutation(interior.size):
            if len(chosen) >= count:
                break
            r, c = divmod(int(index), interior.shape[1])
            if (r, c % period) in groups:
                continue
            trial = noisy.copy()
            trial[r + 1, c + 1] = ink if target[r, c] == bg else bg
            if len(_components(trial != bg)) != 1:
                continue
            chosen.append((r, c))
            groups.add((r, c % period))
        for r, c in chosen:
            target[r, c] = ink if interior[r, c] == bg else bg
        if len(_components(noisy != bg)) != 1:
            continue
        try:
            repaired, got_period, _, _ = _recover(target, bg, ink)
        except ValueError:
            continue
        if got_period == period and np.array_equal(repaired, interior):
            if not force_leading or not np.array_equal(_gates(clean, bg), _gates(noisy, bg)):
                return noisy
    raise ValueError("bounded damage search exhausted")


def _make_card(rng, height, length, period, bg, frame, ink, mode, force=False):
    for _ in range(48):
        bits, bands = _unit(rng, height, length, period, mode)
        motif = np.where(bits[:, np.arange(length) % period], ink, bg).astype(np.int8)
        clean = _shell(motif, bg, frame, mode, rng, bands)
        if len(_components(clean != bg)) != 1:
            continue
        try:
            noisy = _damage(rng, clean, bg, ink, period, force)
        except ValueError:
            continue
        return noisy, clean, bits
    raise ValueError("bounded card search exhausted")


def _witness_cards(rng, bg, frame, ink, period, max_length):
    # Same chambers: closed, opened through the shell, and one column longer.
    lengths = [n for n in range(4 * period, min(5 * period - 1, max_length - 1) + 1)
               if (n + 2) % period != 0]
    length = int(rng.choice(lengths))
    height = int(rng.integers(1, 4))
    q = (length - 1) % period
    bits = np.ones((height, period), dtype=np.int8)
    bits[:, q] = 0
    cards = []
    for mode, size in (("closed", length), ("open", length), ("recessed", length + 1)):
        motif = np.where(bits[:, np.arange(size) % period], ink, bg).astype(np.int8)
        clean = _shell(motif, bg, frame, mode, rng, [(0, height)])
        noisy = _damage(rng, clean, bg, ink, period, force_leading=(mode == "closed"))
        cards.append((noisy, clean, bits.copy()))
    rng.shuffle(cards)
    return cards


def _render_pair(rng, cards, bg, rail, band, turn, force_motion=False):
    count = len(cards)
    minimum_h = sum(c[0].shape[0] for c in cards) + count + 1
    height = minimum_h + int(rng.integers(0, min(4, 30 - minimum_h) + 1))
    widest = max(c[0].shape[1] for c in cards)
    slack = 30 - widest - 1 - band
    extra = int(rng.integers(0 if not force_motion else 1, slack + 1))
    col = widest + extra
    a = np.full((height, col + 1 + band), bg, dtype=np.int8)
    out = a.copy()
    a[:, col] = out[:, col] = rail
    gaps = rng.multinomial(height - minimum_h, [1 / (count + 1)] * (count + 1))
    row = 1 + int(gaps[0])
    forced_emitter = False
    for i, (noisy, clean, bits) in enumerate(cards):
        h, w = noisy.shape
        limit = col - w
        shift = int(rng.integers(limit + 1)) if rng.random() > .25 else 0
        if force_motion and not forced_emitter and _gates(clean, bg).any():
            # Docking must visibly affect beam phase, not just a silent card.
            options = [n for n in range(1, limit + 1) if n % bits.shape[1]]
            shift = int(rng.choice(options))
            forced_emitter = True
        start = col - w - shift
        a[row:row + h, start:start + w] = noisy
        out[row:row + h, col - w:col] = clean
        # Construction uses the clean symbolic unit; solve instead recovers it.
        phase = w  # L+2: count both the leading cap and the rail as occluders.
        for y in np.flatnonzero(_gates(clean, bg)):
            out[row + y, col + 1:] = np.where(
                bits[y - 1, (phase + np.arange(band)) % bits.shape[1]],
                int(clean[1:-1, 1:-1][clean[1:-1, 1:-1] != bg][0]), bg)
        row += h + 1 + int(gaps[i + 1])
    return {"input": np.rot90(a, turn).copy(), "output": np.rot90(out, turn).copy()}


def _random_pair(rng, palette, turn, witness=False, multi=False, is_test=False, rich=True):
    # Sample geometry before placement; no category-to-travel schedule exists.
    bg, rail, common_frame, common_ink = palette[:4]
    if witness:
        period = int(rng.integers(2, 5))
        # Leave a positive displacement available for the matched contrast.
        band = period + 1
        max_length = 30 - band - 4
        cards = _witness_cards(rng, bg, common_frame, common_ink, period, max_length)
        return _render_pair(rng, cards, bg, rail, band, turn, force_motion=True)
    count = int(rng.integers(1, 5))
    periods = list(map(int, rng.integers(2, 6, size=count)))
    maximum = max(periods)
    band = int(rng.integers(maximum + 1, min(8, 27 - 4 * maximum) + 1))
    max_length = 30 - band - 3
    budget = 30 - 3 * count - 1
    heights = np.ones(count, dtype=int)
    if multi:
        heights[0] = 4  # two chambers, including a multirow one
    spare = min(budget - int(heights.sum()), int(rng.integers(0, 4 * count + 1)))
    for _ in range(spare):
        choices = np.flatnonzero(heights < 7)
        if not len(choices):
            break
        heights[int(rng.choice(choices))] += 1
    cards = []
    for i, (period, height) in enumerate(zip(periods, heights)):
        low = min(5 * period, max_length) if is_test else 4 * period
        high = min(max_length, (7 if is_test else 5) * period - 1)
        length = int(rng.integers(low, high + 1))
        frame, ink = common_frame, common_ink
        if rich and rng.random() < .5:
            frame, ink = map(int, rng.choice(palette[2:], 2, replace=False))
        mode = str(rng.choice(["closed", "open", "recessed", "mixed_open"], p=[.5, .2, .2, .1]))
        if multi and i == 0:
            # Explicit positive, multirow training; unrestricted examples vary.
            bits = np.ones((int(height), period), dtype=np.int8)
            bits[[0, 1, 3], (length - 1) % period] = 0
            motif = np.where(bits[:, np.arange(length) % period], ink, bg).astype(np.int8)
            clean = _shell(motif, bg, frame, "closed", rng, [(0, 2), (3, 4)])
            cards.append((_damage(rng, clean, bg, ink, period, True), clean, bits))
        else:
            cards.append(_make_card(rng, int(height), length, period, bg, frame, ink, mode))
    rng.shuffle(cards)
    return _render_pair(rng, cards, bg, rail, band, turn)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _unique(pair, seen):
    # Twenty distinct dihedral color relabelings; at most thirteen inputs blocked.
    # A nonidentity reflection fixes at most two colors; each input has >=4.
    for sign in (1, -1):
        for offset in range(10):
            candidate = {k: ((sign * v.astype(int) + offset) % 10).astype(np.int8)
                         for k, v in pair.items()}
            if _signature(candidate["input"]) not in seen:
                return candidate
    raise ValueError("finite uniqueness orbit exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    vary_palette = bool(rng.integers(2))
    rich = bool(rng.integers(2))
    base = int(rng.integers(4))
    train_turns = (base, (base + 1) % 4)
    test_turns = ((base + 2) % 4, (base + 3) % 4)
    order = list(map(int, rng.permutation(num_train)))
    witness_index, multi_index = order[:2]
    turns = [int(rng.choice(train_turns)) for _ in range(num_train)]
    turns[witness_index], turns[multi_index] = train_turns
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            role_colors = list(map(int, rng.permutation(10))) if vary_palette else palette
            turn = turns[i] if split == "train" else int(rng.choice(test_turns))
            pair = _random_pair(rng, role_colors, turn,
                                witness=split == "train" and i == witness_index,
                                multi=split == "train" and i == multi_index,
                                is_test=split == "test", rich=rich)
            pair = _unique(pair, seen)
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _band_difference(inp, output, variant):
    expected, turn, _, _, col = _scene(inp)
    actual = np.rot90(output, turn)
    alternative = np.rot90(_transform(inp, variant), turn)
    return bool(np.any(actual[:, col + 1:] != alternative[:, col + 1:]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, directions = set(), {"train": set(), "test": set()}
    witnesses = {v: False for v in ("raw_topology", "one_column", "all_holes", "no_topology",
                                    "solid_rays", "reset_phase", "unmoved_phase")}
    has_multirow = has_multi_cavity = has_zero = has_motion = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a, turn, bg, rail, col = _scene(inp)
            directions[split].add(turn)
            objects = _objects(a, bg, col)
            result = np.rot90(out, turn)
            _require(np.all(result[:, col] == rail), "rail changed")
            _require(np.array_equal(solve(inp), out), "incorrect output")
            _require(1 <= len(objects) <= 4, "unsupported card count")
            for obj in objects:
                _require(obj["errors"] >= 1, "generated card must demonstrate repair")
                _require(obj["length"] >= 4 * obj["period"], "insufficient repetition")
                _require(a.shape[1] - col - 1 >= obj["period"] + 1, "transmission band too short")
                # Full-row contact avoids a whole-cavity versus row-local ambiguity.
                for hole in _components(_enclosed(obj["clean"], bg)):
                    rows = {r for r, c in hole}
                    contact = {r for r, c in hole if c == obj["clean"].shape[1] - 2}
                    _require(not contact or contact == rows, "partial leading contact")
                if split == "train":
                    gates = _gates(obj["clean"], bg)
                    has_multirow |= bool(np.any(gates[1:] & gates[:-1]))
                    has_multi_cavity |= len(_components(gates[:, None])) >= 2
                    has_zero |= obj["shift"] == 0
                    has_motion |= obj["shift"] > 0
            if split == "train":
                for variant in witnesses:
                    witnesses[variant] |= _band_difference(inp, out, variant)
    _require(len(directions["train"]) == 2 and not directions["train"] & directions["test"],
             "orientation holdout missing")
    _require(has_motion and has_multirow and has_multi_cavity, "training lacks geometric evidence")
    _require(all(witnesses.values()), "training does not separate alternatives: " + str(witnesses))


def _puzzle_digest(puzzle):
    hasher = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                hasher.update(bytes(a.shape))
                hasher.update(a.tobytes())
    return hasher.hexdigest()


def self_test():
    """42 seeds, streamed measurements, count boundaries and semantic mutations.

    At most one puzzle and two determinism copies live at once. No files or
    traces are saved; hashes replace duplicate reconstructible payloads.
    """
    digests, shapes, normalized, periods, heights, counts, bands, shifts = (set() for _ in range(8))
    directions, palettes, signatures = set(), set(), set()
    pairs = cards = open_cards = silent_cards = stepped_cards = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                a, turn, bg, rail, col = _scene(pair["input"])
                objs = _objects(a, bg, col)
                counts.add(len(objs))
                bands.add(a.shape[1] - col - 1)
                directions.add(turn)
                shapes.add(pair["input"].shape)
                palettes.add((bg, rail))
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)
                normalized.add(hashlib.sha256(norm).hexdigest())
                for obj in objs:
                    cards += 1
                    periods.add(obj["period"])
                    heights.add(obj["clean"].shape[0] - 2)
                    shifts.add(obj["shift"])
                    mask = obj["clean"] != bg
                    signatures.add((mask.shape, mask.tobytes()))
                    open_cards += bool(np.any((obj["clean"][:, -2] == bg) & ~_gates(obj["clean"], bg)))
                    silent_cards += not _gates(obj["clean"], bg).any()
                    for hole in _components(_enclosed(obj["clean"], bg)):
                        rr, cc = np.array(hole).T
                        stepped_cards += len(hole) != (int(rr.max() - rr.min()) + 1) * (int(cc.max() - cc.min()) + 1)
    _require(len(digests) == 42 and len(normalized) >= 150, "structural diversity collapsed")
    _require(len(shapes) >= 45 and len(signatures) >= 100, "geometry collapsed")
    _require(periods == {2, 3, 4, 5} and counts == {1, 2, 3, 4}, "missing periods/counts")
    _require(len(heights) >= 6 and len(bands) >= 4 and len(shifts) >= 5, "size/placement collapsed")
    _require(directions == {0, 1, 2, 3} and len(palettes) >= 20, "role variation collapsed")
    _require(open_cards >= 42 and silent_cards >= 84 and stepped_cards >= 10, "topology variation collapsed")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(1000 + 10 * train + test, train, test)
            _require((len(p["train"]), len(p["test"])) == (train, test), "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(997, 2, 1), generate_puzzle(997, 2, 1)
    _require(_puzzle_digest(p) == _puzzle_digest(q), "nondeterminism")
    for turn in range(4):
        for pair in p["train"] + p["test"]:
            _require(np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn)),
                     "rotation equivariance failed")
    q["test"][0]["output"][0, 0] = (int(q["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(q)
    except ValueError:
        pass
    else:
        raise AssertionError("bad output accepted")
    pair, seen = p["train"][0], set()
    for _ in range(14):
        candidate = _unique(pair, seen)
        seen.add(_signature(candidate["input"]))
        _require(np.array_equal(solve(candidate["input"]), candidate["output"]), "color equivariance failed")
    return dict(seeds=42, pairs=pairs, cards=cards, distinct_puzzles=len(digests),
                shapes=len(shapes), normalized_inputs=len(normalized), object_geometries=len(signatures),
                periods=sorted(periods), card_counts=sorted(counts), interior_heights=sorted(heights),
                band_widths=sorted(bands), shifts=sorted(shifts), open_cards=open_cards,
                silent_cards=silent_cards, nonrectangular_holes=stepped_cards)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
