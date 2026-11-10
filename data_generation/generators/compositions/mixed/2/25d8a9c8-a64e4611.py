"""Whole-row equality creates obstacles for a clipped 3x3 empty-window test.

Output roles are fixed family conventions: uniform=5, fringe=0, clear=3.
Only standard-library modules and NumPy are used. No files are read or written.
"""

from hashlib import sha256

import numpy as np


POSITIVE = 5
BACKGROUND = 0
HIGHLIGHT = 3


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(num_train, num_test):
    tr = _integer(num_train, "num_train", 2, 10)
    te = _integer(num_test, "num_test", 1, 4)
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    return tr, te


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or grid.dtype.kind not in "iu" or
            not all(1 <= d <= 30 for d in grid.shape)):
        raise ValueError("grid must be an integer rectangle with dimensions 1..30")
    if np.any((grid < 0) | (grid > 9)):
        raise ValueError("grid colors must be 0..9")
    return grid


def _uniform(grid):
    return np.all(grid == grid[:, :1], axis=1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Build A's mask, then highlight B's empty clipped-neighborhood centers."""
    grid = _grid(input_grid)
    h, w = grid.shape
    obstacles = np.broadcast_to(_uniform(grid)[:, None], (h, w))
    padded = np.pad(obstacles, 1, constant_values=False)
    near = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + h, dc:dc + w]
    result = np.where(obstacles, POSITIVE, BACKGROUND).astype(np.int8)
    result[~near] = HIGHLIGHT
    return result


def _row_labels(mask):
    """Distance-form specification, also used to check the 2D implementation."""
    walls = np.flatnonzero(mask).tolist()
    return np.array([
        POSITIVE if value else
        BACKGROUND if any(abs(r - wall) <= 1 for wall in walls) else HIGHLIGHT
        for r, value in enumerate(mask)
    ], dtype=np.int8)


def _runs(mask, value):
    runs = []
    start = None
    for i in range(len(mask) + 1):
        if i < len(mask) and bool(mask[i]) == value:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    return runs


def _blocks(lengths, first=False):
    return np.concatenate([
        np.full(length, first if i % 2 == 0 else not first, dtype=bool)
        for i, length in enumerate(lengths)
    ])


def _diagnostic_mask(rng, index):
    if index == 0:
        # Mixed edge, a one-row gap, and a three-row interior gap.
        lengths = [int(rng.integers(2, 4)), int(rng.integers(1, 3)),
                   1, int(rng.integers(1, 3)), 3,
                   int(rng.integers(1, 3)), int(rng.integers(0, 3))]
    else:
        # A two-row interior gap and a four-row interior gap.
        lengths = [int(rng.integers(0, 3)), int(rng.integers(1, 4)),
                   2, int(rng.integers(1, 3)), 4,
                   int(rng.integers(1, 3)), int(rng.integers(2, 4))]
    mask = _blocks(lengths)
    if rng.random() < 0.5:
        mask = mask[::-1].copy()
    return mask


def _held_out_mask(rng):
    # The first test recombines known gap types with a longer interior gap.
    lengths = [int(rng.integers(0, 4)), int(rng.integers(1, 4)),
               int(rng.integers(5, 8)), int(rng.integers(1, 4)),
               int(rng.integers(1, 5)), int(rng.integers(1, 4)),
               int(rng.integers(0, 4))]
    mask = _blocks(lengths)
    return mask if rng.random() < 0.5 else mask[::-1].copy()


def _varied_mask(rng):
    mode = int(rng.integers(0, 8))
    h = int(rng.integers(1 if mode in (0, 1) else 4, 31))
    if mode == 0:
        return np.ones(h, dtype=bool)
    if mode == 1:
        return np.zeros(h, dtype=bool)
    if mode == 2:
        mask = np.zeros(h, dtype=bool)
        mask[int(rng.integers(h))] = True
        return mask
    if mode == 3:
        return (np.arange(h) + int(rng.integers(2))) % 2 == 0
    if mode == 4:
        # Thin gaps, including cases with no highlighted cells at all.
        mask = np.ones(h, dtype=bool)
        r = int(rng.integers(2))
        while r < h:
            length = int(rng.integers(1, 3))
            mask[r:min(r + length, h)] = False
            r += length + int(rng.integers(1, 4))
        return mask
    if mode == 5:
        mask = np.empty(h, dtype=bool)
        r, value = 0, bool(rng.integers(2))
        while r < h:
            length = int(rng.integers(1, 7 if not value else 5))
            mask[r:min(r + length, h)] = value
            r, value = r + length, not value
        return mask
    return rng.random(h) < float(rng.uniform(0.12, 0.68))


def _palette(rng, mandatory=()):
    size = int(rng.integers(max(2, len(mandatory)), 11))
    rest = [c for c in range(10) if c not in mandatory]
    extra = rng.choice(rest, size=size - len(mandatory), replace=False)
    palette = np.array([*mandatory, *extra], dtype=np.int8)
    rng.shuffle(palette)
    return palette


def _mixed_row(rng, palette, width, style=None):
    a, b = (int(x) for x in rng.choice(palette, 2, replace=False))
    style = int(rng.integers(7)) if style is None else style
    if style < 3:
        row = np.full(width, a, dtype=np.int8)
        pos = (0 if style == 0 else width - 1 if style == 2 else
               int(rng.integers(1, width - 1)) if width > 2 else 1)
        row[pos] = b
    elif style == 3:
        row = np.where(np.arange(width) % 2, a, b).astype(np.int8)
    elif style == 4:
        row = np.full(width, a, dtype=np.int8)
        row[int(rng.integers(1, width)):] = b
    elif style == 5:
        row = np.where(np.arange(width) < width // 2, a, b).astype(np.int8)
        rng.shuffle(row)
    else:
        row = np.asarray(rng.choice(palette, width), dtype=np.int8)
        if np.all(row == row[0]):
            row[-1] = b if row[0] == a else a
    return row


def _render(rng, mask, width, palette, diagnostic=None):
    inp = np.empty((len(mask), width), dtype=np.int8)
    forced = {}
    if diagnostic is not None:
        gap = next((a, b) for a, b in _runs(mask, False)
                   if a > 0 and b < len(mask) and b - a == (3 if diagnostic == 0 else 4))
        forced = {gap[0] + i: i for i in range(3)}
    uniform_colors, mixed_rows = [], []
    for r, solid in enumerate(mask):
        if solid:
            color = int(rng.choice(uniform_colors if uniform_colors and rng.random() < .3
                                   else palette))
            inp[r] = color
            uniform_colors.append(color)
        elif r in forced:
            inp[r] = _mixed_row(rng, palette, width, forced[r])
        elif mixed_rows and rng.random() < .15:
            inp[r] = mixed_rows[int(rng.integers(len(mixed_rows)))]
        else:
            inp[r] = _mixed_row(rng, palette, width)
        if not solid:
            mixed_rows.append(inp[r].copy())
    if diagnostic is not None:
        positions = np.flatnonzero(mask)
        # Black and output-role colors are ordinary input colors too.
        inp[positions[0]] = 0 if diagnostic == 0 else 3
        inp[positions[-1]] = int(next(c for c in palette if c != 0)) if diagnostic == 0 else 5
    return inp


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    tr, te = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # Width is semantic-neutral, but the wide demonstration covers test widths.
    cap = int(rng.integers(6, 17)) if rng.random() < .7 else int(rng.integers(17, 31))
    if rng.random() < .18:
        cap = 30
    puzzle = {"train": [], "test": []}
    seen, test_answers, palettes = set(), set(), []
    for split, count in (("train", tr), ("test", te)):
        for index in range(count):
            diagnostic = index if split == "train" and index < 2 else None
            for _ in range(64):
                if diagnostic is not None:
                    mask = _diagnostic_mask(rng, diagnostic)
                    width = int(rng.integers(3, 5)) if diagnostic == 0 else cap
                    palette = _palette(rng, (0,) if diagnostic == 0 else (3, 5))
                else:
                    mask = _held_out_mask(rng) if split == "test" and index == 0 else _varied_mask(rng)
                    width = int(rng.integers(2, cap + 1))
                    if np.all(mask) and rng.random() < .25:
                        width = 1
                    palette = (palettes[int(rng.integers(len(palettes)))].copy()
                               if palettes and rng.random() < .3 else _palette(rng))
                inp = _render(rng, mask, width, palette, diagnostic)
                out = solve(inp)
                answer = _key(out)
                if _key(inp) in seen or (split == "test" and answer in test_answers):
                    continue
                seen.add(_key(inp))
                if split == "test":
                    test_answers.add(answer)
                palettes.append(palette)
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("bounded pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _exception_style(row):
    values, counts = np.unique(row, return_counts=True)
    if len(values) != 2 or sorted(counts.tolist()) != [1, len(row) - 1]:
        return None
    pos = int(np.flatnonzero(row == values[np.argmin(counts)])[0])
    return "first" if pos == 0 else "last" if pos == len(row) - 1 else "interior"


def validate_generated_puzzle(puzzle):
    """Validate semantics by row distances and check diagnostic generation promises."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, answers, styles, gap_lengths = set(), set(), set(), set()
    has_edge = has_small_core = False
    zeros_are_obstacles = role_colors_are_obstacles = False
    train_width = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            mask = _uniform(inp)
            labels = _row_labels(mask)
            expected = np.repeat(labels[:, None], inp.shape[1], axis=1)
            _require(np.array_equal(out, expected), "row-distance rule failed")
            _require(np.all(out[mask] == POSITIVE), "uniform rows not obstacles")
            if split == "test":
                _require(_key(out) not in answers, "duplicate test answer")
                answers.add(_key(out))
            else:
                train_width = max(train_width, inp.shape[1])
                for row, solid in zip(inp, mask):
                    if not solid and inp.shape[1] >= 3:
                        styles.add(_exception_style(row))
                    if solid:
                        zeros_are_obstacles |= bool(row[0] == 0)
                        role_colors_are_obstacles |= bool(row[0] in (3, 5))
                for a, b in _runs(mask, False):
                    if a > 0 and b < len(mask):
                        gap_lengths.add(b - a)
                has_edge |= bool(labels[0] == HIGHLIGHT or labels[-1] == HIGHLIGHT)
                for a, b in _runs(labels == HIGHLIGHT, True):
                    has_small_core |= (b - a) * inp.shape[1] < 5
                # Both inherited concepts affect training answers.
                if np.any(labels == HIGHLIGHT) and np.any(mask):
                    no_b = np.repeat(np.where(mask, 5, 0)[:, None], inp.shape[1], axis=1)
                    no_a = np.full(inp.shape, HIGHLIGHT, dtype=np.int8)
                    _require(not np.array_equal(out, no_b), "B ablation changes nothing")
                    _require(not np.array_equal(out, no_a), "A ablation changes nothing")
    _require({"first", "interior", "last"} <= styles, "missing whole-row equality contrasts")
    _require({1, 2, 3, 4} <= gap_lengths, "missing narrow/wide gap contrasts")
    _require(has_edge and has_small_core, "missing clipped edge or small highlighted core")
    _require(zeros_are_obstacles and role_colors_are_obstacles, "input colors mistaken for roles")
    _require(train_width >= max(_grid(p["input"]).shape[1] for p in puzzle["test"]),
             "test width exceeds diagnostic demonstration width")
    first_test = _uniform(_grid(puzzle["test"][0]["input"]))
    _require(any(a > 0 and b < len(first_test) and b - a >= 5
                 for a, b in _runs(first_test, False)), "missing held-out longer gap")


def _fingerprint(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded tests: 42 varied seeds, all counts, exhaustive short row masks."""
    fingerprints, shapes, palettes, decisions, counts = set(), set(), set(), set(), set()
    all_uniform = all_mixed = no_highlight = 0
    for seed in range(42):
        tr = 2 + seed % 9
        te = 1 + (seed // 9) % min(4, tr - 1)
        puzzle = generate_puzzle(seed, tr, te)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        counts.add((tr, te))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                mask = _uniform(inp)
                shapes.add(inp.shape)
                palettes.add(tuple(int(c) for c in np.unique(inp)))
                decisions.add(tuple(bool(x) for x in mask))
                all_uniform += bool(np.all(mask))
                all_mixed += bool(not np.any(mask))
                no_highlight += bool(not np.any(out == HIGHLIGHT))
        # Equality is invariant under every bijective recoloring and column order.
        rng = np.random.default_rng(seed + 10000)
        pair = puzzle["train"][1]
        permutation = rng.permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]]), pair["output"]), "color dependence")
        _require(np.array_equal(solve(pair["input"][:, ::-1]), pair["output"]), "column-order dependence")
        _require(np.array_equal(solve(pair["input"][::-1]), pair["output"][::-1]), "vertical asymmetry")
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed, tr, te)), "nondeterminism")
    _require(len(fingerprints) == 42 and len(shapes) >= 50, "insufficient shape variation")
    _require(len(decisions) >= 100 and len(palettes) >= 50, "insufficient structural/palette variation")
    _require(all_uniform and all_mixed and no_highlight and len(counts) >= 20, "missing limits or counts")
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            generate_puzzle(9000 + 10 * tr + te, tr, te)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # All realizable short row masks, including exact boundary cases. The
    # distance formulation is deliberately separate from solve's 2D padding.
    for h in range(1, 9):
        for bits in range(1 << h):
            mask = np.array([bool(bits & (1 << r)) for r in range(h)])
            for w in (2, 3, 4):
                inp = np.full((h, w), 7, dtype=np.int8)
                inp[~mask, -1] = 2
                expected = np.repeat(_row_labels(mask)[:, None], w, axis=1)
                _require(np.array_equal(solve(inp), expected), "exhaustive row-mask failure")
    _require(np.all(solve(np.arange(30, dtype=np.int8).reshape(30, 1) % 10) == 5), "width-one failure")
    for grid in (np.zeros((0, 2), dtype=int), np.full((2, 2), 10),
                 np.zeros((31, 2), dtype=int), np.zeros((2, 2), dtype=float)):
        try:
            solve(grid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid grid accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
