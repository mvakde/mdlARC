"""Selective excavation: phase-recover hidden colors, then compact selected units.

Only Python's standard library and NumPy are used. No parent code is imported.
"""

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or a.dtype.kind not in "iu" or
            not all(1 <= d <= 30 for d in a.shape) or
            np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer color grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _stripe(a):
    """Return canonical vertical coordinates; reject competing full-span lines."""
    candidates = []
    for transpose in (False, True):
        work = a.T if transpose else a
        for color in np.unique(work):
            columns = np.flatnonzero(np.all(work == color, axis=0))
            if not len(columns):
                continue
            start, end = int(columns[0]), int(columns[-1])
            if (start == 0 or end == work.shape[1] - 1 or
                    not np.array_equal(columns, np.arange(start, end + 1))):
                raise ValueError("competing or boundary full-span line")
            candidates.append((transpose, int(color), start, end))
    if len(candidates) != 1:
        raise ValueError("the full-span stripe must be unique")
    return candidates[0]


def _period(a):
    for p in range(1, a.shape[0] // 2 + 1):
        if np.array_equal(a[:-p], a[p:]):
            return p
    return None


def _recover(a, stripe):
    """Infer both missing color and completion, from all supported explanations."""
    answers = {}
    h, w = a.shape
    for missing in (int(x) for x in np.unique(a) if int(x) != stripe):
        mask = a == missing
        for p in range(2, h // 2 + 1):
            visible = (~mask[:-p]) & (~mask[p:])
            if np.any(a[:-p][visible] != a[p:][visible]):
                continue
            block = np.empty((p, w), dtype=np.int8)
            for phase in range(p):
                group = a[phase::p]
                known = group != missing
                low = np.where(known, group, 10).min(axis=0)
                high = np.where(known, group, -1).max(axis=0)
                if np.any(low != high):  # includes the no-witness case
                    break
                block[phase] = low
            else:
                repaired = block[np.arange(h) % p]
                if _period(repaired) == 1:
                    continue
                answers[(missing, _key(repaired))] = repaired
    if len(answers) != 1:
        raise ValueError("damage color or periodic completion is ambiguous")
    (missing, _), repaired = next(iter(answers.items()))
    return missing, repaired.copy()


def _counts(selected, start, end):
    return np.stack((selected[:, :start].sum(axis=1),
                     selected[:, end + 1:].sum(axis=1)), axis=1).astype(int)


def _paint(clean, color, start, end, counts):
    out = clean.copy()
    for row, (left, right) in enumerate(counts):
        out[row, start - int(left):start] = color
        out[row, end + 1:end + 1 + int(right)] = color
    return out


def _analyze(input_grid):
    a = _grid(input_grid)
    transpose, color, start, end = _stripe(a)
    work = a.T.copy() if transpose else a.copy()
    missing, clean = _recover(work, color)
    mask = work == missing
    selected = mask & (clean == color)
    counts = _counts(selected, start, end)
    return dict(work=work, clean=clean, mask=mask, selected=selected,
                counts=counts, transpose=transpose, stripe=color,
                missing=missing, start=start, end=end)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the current input only; never mutates its argument."""
    info = _analyze(input_grid)
    out = _paint(info["clean"], info["stripe"], info["start"], info["end"],
                 info["counts"])
    return out.T.copy() if info["transpose"] else out


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or
            not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _palette(rng):
    size = int(rng.integers(2, 6))
    colors = tuple(int(x) for x in rng.choice(10, size=size + 1, replace=False))
    return colors[:-1], colors[-1]


def _block(rng, p, width, content):
    """Independent word width, interleaved rotations, or repeated separators."""
    word_width = int(rng.choice([2, 3, 4, 5, width, width, width]))
    style = int(rng.integers(0, 3))
    probabilities = np.full(len(content), 1.0)
    probabilities[0] = float(rng.uniform(1.2, 2.8))
    probabilities /= probabilities.sum()

    def word():
        return rng.choice(content, size=word_width, p=probabilities)

    if style == 0:
        rows = np.stack([word() for _ in range(p)])
    else:
        families = int(rng.integers(1, min(3, p) + 1))
        bases = [word() for _ in range(families)]
        steps = rng.integers(1, word_width + 1, size=families)
        rows = np.stack([np.roll(bases[r % families],
                                 int((r // families) * steps[r % families]))
                         for r in range(p)])
        if style == 2 and p >= 4:
            separator = word()
            rows[1::2] = separator
    offset = int(rng.integers(0, word_width))
    return rows[:, (np.arange(width) + offset) % word_width].astype(np.int8)


def _position(start, end, side, distance):
    return start - distance if side == 0 else end + distance


def _regions(rng, h, w, p, test):
    coords = []
    for _ in range(int(rng.integers(1, 6 if test else 5))):
        kind = int(rng.integers(0, 5))
        rh = int(rng.integers(1, min(h, p + (5 if test else 2)) + 1))
        rw = int(rng.integers(1, min(w, 9) + 1))
        if kind == 0:
            rh = 1
        if kind == 1:
            rw = 1
        top = int(rng.integers(0, h - rh + 1))
        left = int(rng.integers(0, w - rw + 1))
        for r in range(rh):
            inset = min(rw - 1, r // 2) if kind == 2 else 0
            coords.extend((top + r, left + c) for c in range(inset, rw))
    if rng.random() < 0.6:
        coords.extend((int(r), int(c)) for r, c in
                      zip(rng.integers(0, h, size=12), rng.integers(0, w, size=12)))
    return coords


def _rich_evidence(info):
    """A particular lane demonstrates multiplicity, both sides and cell counts."""
    count, chosen = info["counts"], info["selected"]
    for row in range(chosen.shape[0]):
        if (min(count[row]) < 1 or max(count[row]) < 2 or
                count[row, 0] == count[row, 1]):
            continue
        for side, segment in enumerate((chosen[row, :info["start"]],
                                        chosen[row, info["end"] + 1:])):
            n = int(count[row, side])
            if n >= 2 and np.any(segment[:-1] & segment[1:]):
                # Counting qualifying connected runs must visibly differ.
                components = int(np.count_nonzero(segment & ~np.r_[False, segment[:-1]]))
                alternative = count.copy()
                alternative[row, side] = components
                if not np.array_equal(
                        _paint(info["clean"], info["stripe"], info["start"],
                               info["end"], count),
                        _paint(info["clean"], info["stripe"], info["start"],
                               info["end"], alternative)):
                    return True
    return False


def _make_pair(rng, p, transpose, fixed_palette, mode, test):
    for _attempt in range(160):
        content, missing = fixed_palette if fixed_palette is not None else _palette(rng)
        color, other = content[:2]
        h = int(rng.integers(3 * p, 31))
        if rng.random() < 0.3:
            h = int(rng.integers(3, 30 // p + 1)) * p
        width = int(rng.integers(11 if mode == "rich" else 8, 31))
        thickness = int(rng.integers(1, min(4, width - 7) + 1))
        options = [s for s in range(1, width - thickness)
                   if max(s, width - thickness - s) >= (5 if mode == "rich" else 3)
                   and (mode != "rich" or min(s, width - thickness - s) >= 2)]
        start = int(rng.choice(options))
        end = start + thickness - 1
        capacities = (start, width - end - 1)
        block = _block(rng, p, width, content)
        block[:, start:end + 1] = color
        anchor = int(rng.integers(0, h)) if mode != "random" else None
        anchor_cells = []
        if anchor is not None:
            side = 0 if capacities[0] >= capacities[1] else 1
            phase = anchor % p

            def set_at(s, d, value, damaged=False):
                c = _position(start, end, s, d)
                block[phase, c] = value
                if damaged:
                    anchor_cells.append((anchor, c))

            if mode == "simple":
                set_at(side, 1, other)
                set_at(side, 2, other, True)
                set_at(side, 3, color, True)
            else:
                for d in (1, 2, 3):
                    set_at(side, d, other, d == 3)
                for d in (4, 5):
                    set_at(side, d, color, True)
                set_at(1 - side, 1, other)
                set_at(1 - side, 2, color, True)
        # A short/binary cycle must not accidentally create additional seams.
        # Change another phase, preserving the anchor's hidden-color evidence.
        partner = ((anchor % p) + 1) % p if anchor is not None else int(rng.integers(0, p))
        for c in list(range(start)) + list(range(end + 1, width)):
            if np.all(block[:, c] == block[0, c]):
                block[partner, c] = other if block[0, c] == color else color
        clean = block[np.arange(h) % p].copy()
        try:
            if _period(clean) != p:
                continue
            if _stripe(clean) != (False, color, start, end):
                continue
        except ValueError:
            continue
        mask = np.zeros(clean.shape, dtype=bool)
        remaining = np.broadcast_to(np.bincount(np.arange(h) % p, minlength=p)[:, None],
                                    (p, width)).copy()
        minimum_witnesses = 1 if test or rng.random() < 0.4 else 2
        budget = max(5, int(h * (width - thickness) * rng.uniform(0.04, 0.27)))

        def add(r, c, force=False):
            if start <= c <= end or mask[r, c]:
                return
            if not force and (r == anchor or int(mask.sum()) >= budget):
                return
            if remaining[r % p, c] <= minimum_witnesses:
                return
            mask[r, c] = True
            remaining[r % p, c] -= 1

        for r, c in anchor_cells:
            add(r, c, True)
        for r, c in _regions(rng, h, width, p, test):
            add(r, c)
        if test:
            # Make a qualifying damaged cell depend on exactly one witness.
            locations = np.argwhere(mask & (clean == color))
            if not len(locations):
                continue
            r, c = (int(x) for x in locations[int(rng.integers(0, len(locations)))])
            copies = list(range(r % p, h, p))
            survivors = [rr for rr in copies if not mask[rr, c]]
            survivor = int(rng.choice(survivors))
            for rr in copies:
                mask[rr, c] = rr != survivor
        inp = clean.copy()
        inp[mask] = missing
        selected = mask & (clean == color)
        counts = _counts(selected, start, end)
        out = _paint(clean, color, start, end, counts)
        all_damage = _paint(clean, color, start, end, _counts(mask, start, end))
        if np.array_equal(out, clean) or np.array_equal(out, all_damage):
            continue
        display_input = inp.T.copy() if transpose else inp
        display_output = out.T.copy() if transpose else out
        try:
            info = _analyze(display_input)
        except ValueError:
            continue
        if not np.array_equal(info["clean"], clean):
            continue
        if mode == "rich" and not _rich_evidence(info):
            continue
        return {"input": display_input, "output": display_output}
    raise RuntimeError(f"bounded construction failed: period={p}, mode={mode}, test={test}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    fixed_palette = _palette(rng) if rng.random() < 0.35 else None
    orientation_mode = int(rng.integers(0, 3))
    common_axis = bool(rng.integers(0, 2))
    anchors = [int(x) for x in rng.choice(np.arange(2, 7), size=2, replace=False)]
    modes = ["simple", "rich"] + ["random"] * (num_train - 2)
    rng.shuffle(modes)
    puzzle = {"train": [], "test": []}
    seen = set()
    anchor_index = 0
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            mode = modes[index] if split == "train" else "random"
            if mode != "random":
                p = anchors[anchor_index]
                anchor_index += 1
            elif split == "test" and rng.random() < 0.4:
                p = int(rng.choice(anchors))
            else:
                p = int(rng.integers(2, 10))
            transpose = common_axis if orientation_mode < 2 else bool(rng.integers(0, 2))
            for _ in range(12):
                pair = _make_pair(rng, p, transpose, fixed_palette, mode, split == "test")
                k = _key(pair["input"])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, training_periods = set(), set()
    rich = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            clean, mask, selected = info["clean"], info["mask"], info["selected"]
            start, end, color = info["start"], info["end"], info["stripe"]
            actual = out.T if info["transpose"] else out
            counts = info["counts"]
            assert selected.any() and np.any(mask & ~selected)
            assert np.all(counts[:, 0] <= start)
            assert np.all(counts[:, 1] <= clean.shape[1] - end - 1)
            assert np.array_equal(actual, _paint(clean, color, start, end, counts))
            assert not np.array_equal(actual, clean), "A ablation is invisible"
            assert not np.array_equal(actual, _paint(clean, color, start, end,
                                                     _counts(mask, start, end))), "B ablation is invisible"
            assert not np.any(actual == info["missing"])
            assert np.all(actual[:, start:end + 1] == color)
            footprint = np.zeros(clean.shape, dtype=bool)
            for row, (left, right) in enumerate(counts):
                footprint[row, start - left:start] = True
                footprint[row, end + 1:end + 1 + right] = True
                assert np.all(actual[row, start - left:start] == color)
                assert np.all(actual[row, end + 1:end + 1 + right] == color)
            assert np.array_equal(actual[~footprint], clean[~footprint])
            assert np.array_equal(info["work"][~mask], clean[~mask])
            p = _period(clean)
            assert p is not None and p >= 2
            one_witness = False
            for phase in range(p):
                group = info["work"][phase::p]
                known = group != info["missing"]
                assert np.all(known.sum(axis=0) >= 1)
                assert np.all(np.where(known, group == clean[phase], True))
                one_witness |= bool(np.any((known.sum(axis=0) == 1) &
                                           selected[phase::p].any(axis=0)))
            if split == "train":
                training_periods.add(p)
                rich |= _rich_evidence(info)
            else:
                assert one_witness
    assert len(training_periods) >= 2
    assert rich, "training must demonstrate selected cell multiplicity and separate sides"
    return None


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded seed sweep, structural metrics, metamorphic and count checks."""
    shapes, periods, widths, totals, palettes, structures = set(), set(), set(), set(), set(), set()
    axes, partial, signatures = set(), set(), set()
    repeated_rows = saturated = one_side = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                copy = inp.copy()
                assert np.array_equal(solve(inp), out)
                assert np.array_equal(inp, copy)
                info = _analyze(inp)
                clean, count = info["clean"], info["counts"]
                p = _period(clean)
                shapes.add(inp.shape)
                periods.add(p)
                widths.add(info["end"] - info["start"] + 1)
                totals.add(int(count.sum()))
                palettes.add(len(np.unique(clean)))
                axes.add(info["transpose"])
                partial.add(clean.shape[0] % p != 0)
                repeated_rows += len({row.tobytes() for row in clean[:p]}) < p
                saturated += bool(np.any(count[:, 0] == info["start"]) or
                                  np.any(count[:, 1] == clean.shape[1] - info["end"] - 1))
                one_side += bool(count[:, 0].sum() == 0 or count[:, 1].sum() == 0)
                roles = {}
                structures.add((inp.shape, tuple(roles.setdefault(int(x), len(roles)) for x in inp.flat)))
        # Exact coordinate and palette symmetries should require no latent state.
        pair = puzzle["test"][0]
        assert np.array_equal(solve(pair["input"].T), pair["output"].T)
        permutation = np.random.default_rng(seed + 700).permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert len(signatures) == 42 and len(structures) >= 150
    assert len(shapes) >= 60 and periods.issuperset(range(2, 10))
    assert widths == {1, 2, 3, 4} and axes == {False, True}
    assert partial == {False, True} and len(totals) >= 15 and len(palettes) >= 4
    assert repeated_rows > 0 and saturated > 0 and one_side > 0
    assert _fingerprint(generate_puzzle(123)) == _fingerprint(generate_puzzle(123))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), shapes=len(shapes),
                normalized_inputs=len(structures), periods=sorted(periods),
                thicknesses=sorted(widths), selected_totals=len(totals),
                repeated_phase_rows=repeated_rows, saturated_pairs=saturated,
                one_side_pairs=one_side)
