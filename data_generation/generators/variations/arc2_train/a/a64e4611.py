"""Generate negative-space puzzles with coherent clear regions."""
import numpy as np


def _check_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} out of range")
    return value


def _counts(seed, rng, num_train, num_test):
    tr = _check_count("num_train", num_train, 2, 10)
    te = _check_count("num_test", num_test, 1, 4)
    if tr is None and te is None:
        te = seed % 4 + 1
        minimum = max(2, te + 1)
        tr = minimum + (seed // 4) % (11 - minimum)
    elif tr is None:
        tr = int(rng.integers(max(2, te + 1), 11))
    elif te is None:
        te = int(rng.integers(1, min(4, tr - 1) + 1))
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    return tr, te


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    groups = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            todo = [(r, c)]
            seen[r, c] = True
            group = []
            while todo:
                y, x = todo.pop()
                group.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        todo.append((yy, xx))
            groups.append(group)
    return groups


def _candidates(inp, bg):
    h, w = inp.shape
    padded = np.pad(inp != bg, 1, constant_values=False)
    near = np.zeros((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            near |= padded[dy:dy + h, dx:dx + w]
    return ~near


def _solve(inp, bg, highlight):
    out = inp.copy()
    for group in _components(_candidates(inp, bg)):
        if len(group) >= 5:
            for r, c in group:
                out[r, c] = highlight
    return out


def _dimensions(rng, minimum):
    h = 30 if rng.random() < .22 else int(rng.integers(minimum, 31))
    w = 30 if rng.random() < .22 else int(rng.integers(minimum, 31))
    return h, w


def _stripe(mask, rng, vertical):
    h, w = mask.shape
    long, short = (h, w) if vertical else (w, h)
    breadth = int(rng.integers(3, min(9, short // 2 + 1)))
    length = int(rng.integers(max(9, long // 2), long + 1))
    start = int(rng.integers(0, long - length + 1))
    if rng.random() < .35:
        start = 0 if rng.random() < .5 else long - length
    center = int(rng.integers(breadth // 2, short - (breadth + 1) // 2 + 1))
    for t in range(start, start + length):
        if t > start and rng.random() < .10:
            center = int(np.clip(center + int(rng.choice([-1, 1])), 3, short - 4))
        if t > start and rng.random() < .07:
            breadth = int(np.clip(breadth + int(rng.choice([-1, 1])), 3, 9))
        left = max(0, center - breadth // 2)
        right = min(short, left + breadth)
        if vertical:
            mask[t, left:right] = True
        else:
            mask[left:right, t] = True
    return start, start + length, center


def _arms(mask, rng, vertical, span, count):
    h, w = mask.shape
    short = w if vertical else h
    lo, hi, center = span
    for _ in range(count):
        thick = int(rng.integers(3, 7))
        along = int(rng.integers(lo, max(lo + 1, hi - thick + 1)))
        direction = int(rng.choice([-1, 1]))
        reach = int(rng.integers(4, max(5, short // 2 + 2)))
        endpoint = 0 if direction < 0 else short
        if rng.random() >= .3:
            endpoint = int(np.clip(center + direction * reach, 0, short))
        a, b = sorted((center, endpoint))
        if b - a < 3:
            continue
        if vertical:
            mask[along:along + thick, a:b] = True
        else:
            mask[a:b, along:along + thick] = True


def _patch(mask, rng, height, width, attempts=35):
    h, w = mask.shape
    if h < height + 2 or w < width + 2:
        return False
    for _ in range(attempts):
        r = int(rng.integers(1, h - height))
        c = int(rng.integers(1, w - width))
        if np.any(mask[max(0, r - 2):min(h, r + height + 2),
                       max(0, c - 2):min(w, c + width + 2)]):
            continue
        mask[r:r + height, c:c + width] = True
        return True
    return False


def _geometry(rng, h, w, vertical, complexity, force_pocket, simple=False):
    opening = np.zeros((h, w), dtype=bool)
    span = _stripe(opening, rng, vertical)
    _arms(opening, rng, vertical, span,
          0 if simple else int(rng.integers(0, complexity + 2)))
    if not simple and rng.random() < .25 + .13 * complexity:
        _stripe(opening, rng, not vertical)
    for _ in range(0 if simple else int(rng.integers(0, complexity + 1))):
        _patch(opening, rng, int(rng.integers(5, 8)), int(rng.integers(5, 8)))
    if force_pocket:
        _patch(opening, rng, 5, 5, 90)
    return opening


def _make_input(rng, opening, bg, fg, diagnostic, boundary_example=None):
    h, w = opening.shape
    reserved = opening.copy()
    foreground = rng.random((h, w)) < float(rng.uniform(.26, .49))
    for _ in range(int(rng.integers(2, 9))):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        length = int(rng.integers(2, 7))
        if rng.random() < .5:
            foreground[r, c:min(w, c + length)] = True
        else:
            foreground[r:min(h, r + length), c] = True
    foreground[opening] = False
    if rng.random() < .34:
        locations = np.argwhere(opening)
        if len(locations):
            r, c = locations[int(rng.integers(len(locations)))]
            foreground[r, c] = True
    if diagnostic:
        # Isolated 3x3 clear patch: its one candidate center stays background.
        for _ in range(80):
            r = int(rng.integers(1, h - 4))
            c = int(rng.integers(1, w - 4))
            if np.any(opening[r - 1:r + 4, c - 1:c + 4]):
                continue
            foreground[r - 1:r + 4, c - 1:c + 4] = True
            foreground[r:r + 3, c:c + 3] = False
            reserved[r - 1:r + 4, c - 1:c + 4] = True
            break
    if boundary_example is not None:
        ph, pw = (4, 4) if boundary_example == "negative" else (3, 7)
        placed = False
        for _ in range(150):
            r = int(rng.integers(1, h - ph))
            c = int(rng.integers(1, w - pw))
            if np.any(reserved[r - 1:r + ph + 1, c - 1:c + pw + 1]):
                continue
            foreground[r - 1:r + ph + 1, c - 1:c + pw + 1] = True
            foreground[r:r + ph, c:c + pw] = False
            reserved[r - 1:r + ph + 1, c - 1:c + pw + 1] = True
            placed = True
            break
        if not placed:
            return None
    if rng.random() < .48:
        ph, pw = (int(rng.integers(6, 10)), 2)
        if rng.random() < .5:
            ph, pw = pw, ph
        for _ in range(100):
            r = int(rng.integers(1, h - ph))
            c = int(rng.integers(1, w - pw))
            if np.any(reserved[r - 1:r + ph + 1, c - 1:c + pw + 1]):
                continue
            foreground[r - 1:r + ph + 1, c - 1:c + pw + 1] = True
            foreground[r:r + ph, c:c + pw] = False
            break
    inp = np.full((h, w), bg, dtype=np.int8)
    inp[foreground] = fg
    return inp


def _palette(rng):
    colors = list(range(1, 10))
    highlight = int(rng.choice(colors))
    colors.remove(highlight)
    other_bg = int(rng.choice(colors))
    colors.remove(other_bg)
    rng.shuffle(colors)
    return highlight, other_bg, colors


def _orientations(rng, ntrain, ntest):
    mode = int(rng.integers(0, 4))
    if mode == 0:
        train, test = [True] * ntrain, [False] * ntest
    elif mode == 1:
        train, test = [True] * ntrain, [True] * ntest
    elif mode == 2:
        train, test = [False] * ntrain, [False] * ntest
    else:
        train = [i % 2 == 0 for i in range(ntrain)]
        test = [bool(rng.integers(0, 2)) for _ in range(ntest)]
    if mode == 3:
        for plan in (train, test):
            if rng.random() < .35:
                index = int(rng.integers(0, len(plan)))
                plan[index] = not plan[index]
    return train, test, mode


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(seed, rng, num_train, num_test)
    highlight, other_bg, colors = _palette(rng)
    train_ori, test_ori, mode = _orientations(rng, ntrain, ntest)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, orientations in (("train", train_ori), ("test", test_ori)):
        for index, vertical in enumerate(orientations):
            for attempt in range(24):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                bg = (0 if split == "train" and index == 0 else
                      other_bg if split == "train" and index == 1 else
                      int(pair_rng.choice([0, other_bg])))
                if split == "train":
                    fg = colors[index % 3]
                else:
                    fg = (colors[int(pair_rng.integers(0, 3))]
                          if pair_rng.random() < .55 else
                          colors[int(pair_rng.integers(3, len(colors)))])
                h, w = _dimensions(pair_rng, 26 if split == "train" and index < 2 else 18)
                opening = _geometry(pair_rng, h, w, vertical,
                                    1 if split == "train" else 2,
                                    split == "train" and index == 1,
                                    simple=mode in (1, 2) or (mode == 0 and seed % 2 == 0))
                inp = _make_input(pair_rng, opening, bg, fg,
                                  (split == "train" and index == 0) or
                                  pair_rng.random() < .38,
                                  ("negative" if index == 0 else "positive")
                                  if split == "train" and index < 2 else None)
                if inp is None:
                    continue
                out = _solve(inp, bg, highlight)
                candidate = _candidates(inp, bg)
                largest = max((len(group) for group in _components(out == highlight)),
                              default=0)
                if (_key(inp) in used or not np.any(out == highlight) or
                    largest < (20 if split == "train" else 12) or
                    (split == "train" and index == 0 and
                     not np.any(candidate & (out == bg))) or
                    np.sum(inp == bg) <= inp.size // 2):
                    continue
                used.add(_key(inp))
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("could not construct a valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen = set()
    highlights = set()
    train_bgs = set()
    train_fgs = set()
    clear_unmarked = False
    has_four = False
    has_five = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            values, counts = np.unique(inp, return_counts=True)
            assert len(values) == 2 and counts.max() > inp.size // 2
            bg = int(values[np.argmax(counts)])
            fg = int(values[np.argmin(counts)])
            changed = inp != out
            assert np.any(changed) and np.all(inp[changed] == bg)
            novel = set(int(c) for c in np.unique(out)) - set(int(c) for c in values)
            assert len(novel) == 1
            highlight = novel.pop()
            highlights.add(highlight)
            assert np.array_equal(out, _solve(inp, bg, highlight))
            assert np.any(out == bg) and np.any(out == fg)
            if split == "train":
                train_bgs.add(bg)
                train_fgs.add(fg)
                clear_unmarked |= bool(np.any(_candidates(inp, bg) & (out == bg)))
                for group in _components(_candidates(inp, bg)):
                    has_four |= len(group) == 4 and all(out[r, c] == bg for r, c in group)
                    has_five |= len(group) == 5 and all(out[r, c] == highlight for r, c in group)
    assert len(highlights) == 1
    assert len(train_bgs) >= 2 and len(train_fgs) >= 2
    assert clear_unmarked
    assert has_four and has_five


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    signatures, shapes, highlights = set(), set(), set()
    full_size = multicomponent = small_clear = same_fg = novel_fg = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(q["input"]) for s in ("train", "test") for q in puzzle[s]))
        train_fgs = set()
        for pair in puzzle["train"]:
            vals, counts = np.unique(pair["input"], return_counts=True)
            train_fgs.add(int(vals[np.argmin(counts)]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                shapes.add(inp.shape)
                full_size += inp.shape == (30, 30)
                vals, counts = np.unique(inp, return_counts=True)
                bg = int(vals[np.argmax(counts)])
                fg = int(vals[np.argmin(counts)])
                highlight = int((set(np.unique(out)) - set(vals)).pop())
                highlights.add(highlight)
                assert np.array_equal(out, _solve(inp, bg, highlight))
                multicomponent += len(_components(out == highlight)) >= 2
                small_clear += bool(np.any(_candidates(inp, bg) & (out == bg)))
                if split == "test":
                    same_fg += fg in train_fgs
                    novel_fg += fg not in train_fgs
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(highlights) >= 5 and full_size >= 2
    assert multicomponent >= 5 and small_clear >= 100
    assert same_fg and novel_fg
    for seed, a, b in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                       (1004, None, 1), (1005, None, 4),
                       (1006, 2, None), (1007, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        if a is not None:
            assert len(puzzle["train"]) == a
        if b is not None:
            assert len(puzzle["test"]) == b
    def signature(p):
        return tuple((_key(q["input"]), _key(q["output"]))
                     for s in ("train", "test") for q in p[s])
    assert signature(generate_puzzle(31415, 8, 3)) == signature(generate_puzzle(31415, 8, 3))
    assert signature(generate_puzzle(31415)) != signature(generate_puzzle(31416))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {kwargs}")
