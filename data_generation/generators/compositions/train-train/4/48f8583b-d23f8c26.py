"""Threaded self-portraits: rare-color posts joined by center-column ribbons.

All solving conventions are visible in a single input.  Generation uses no
parent code, and solve does not consult examples, files, or random state.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or grid.dtype.kind not in "iu"
            or not all(1 <= n <= 30 for n in grid.shape)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=True)


def _roles(tile):
    colors, counts = np.unique(tile, return_counts=True)
    if (len(colors) < 2 or np.count_nonzero(counts == counts.min()) != 1
            or np.count_nonzero(counts == counts.max()) != 1):
        raise ValueError("input needs unique least- and most-frequent colors")
    return int(colors[np.argmin(counts)]), int(colors[np.argmax(counts)])


def _bridges(mask):
    """Empty macro-cells strictly between two selected cells in one row."""
    result = np.zeros(mask.shape, dtype=bool)
    for r, row in enumerate(mask):
        columns = np.flatnonzero(row)
        if len(columns) >= 2:
            result[r, columns[0]:columns[-1] + 1] = ~row[columns[0]:columns[-1] + 1]
    return result


def _render(tile, selector, background, profile=None, whole_bridges=False):
    h, w = tile.shape
    mask = tile == selector
    ribbon = (tile if whole_bridges else np.repeat(
        (tile[:, w // 2] if profile is None else profile)[:, None], w, axis=1))
    bridges = _bridges(mask)
    result = np.full((h * h, w * w), background, dtype=np.int8)
    for r in range(h):
        for c in range(w):
            if mask[r, c] or bridges[r, c]:
                result[r*h:(r+1)*h, c*w:(c+1)*w] = tile if mask[r, c] else ribbon
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any member of the specified family from its input alone."""
    tile = _grid(input_grid)
    h, w = tile.shape
    if not (2 <= h <= 5 and w in (3, 5)):
        raise ValueError("family tiles have height 2..5 and width 3 or 5")
    selector, background = _roles(tile)
    return _render(tile, selector, background)


def _key(grid):
    return grid.shape, grid.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def _max_palette(n, k):
    # With >=3 colors, reserve one extra cell to make the maximum unique.
    return min(8, 1 + (n - k - 1) // (k + 1))


def _mask(rng, shape, k, witness=False, bridge=False, multirow=False, vertical=False):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    protected = set()
    if witness:
        # A single one-block gap, an exterior blank, and an asymmetric mask.
        rows = [r for r in range(h) if h % 2 == 0 or r != h // 2]
        row = int(rng.choice(rows))
        left = int(rng.choice([0, 2]))
        mask[row, left] = mask[row, left + 2] = True
        protected.update((row, c) for c in range(w) if not mask[row, c])
        if k == 4:
            other_row = int(rng.choice([r for r in range(h) if r != row]))
            other_left = int(rng.choice([0, 2]))
            mask[other_row, other_left] = mask[other_row, other_left + 2] = True
            protected.update((other_row, c) for c in range(w) if not mask[other_row, c])
        # Keep the other center cells free for colored and field-profile pins.
        protected.update((r, w // 2) for r in range(h) if r != row)
    elif vertical:
        col = int(rng.choice([0, w-1]))
        mask[0, col] = mask[h-1, col] = True
        if k == 3:
            row = int(rng.integers(1, h-1))
            other = int(rng.choice([c for c in range(w) if c != col]))
            mask[row, other] = True
    elif bridge:
        for row_value in rng.choice(h, size=2 if multirow else 1, replace=False):
            row = int(row_value)
            distance = int(rng.integers(2, w))
            left = int(rng.integers(w - distance))
            right = left + distance
            mask[row, left] = mask[row, right] = True
            protected.add((row, int(rng.integers(left + 1, right))))

    remaining = k - int(mask.sum())
    candidates = [(r, c) for r in range(h) for c in range(w)
                  if not mask[r, c] and (r, c) not in protected]
    if remaining < 0 or remaining > len(candidates):
        return None
    # General masks mix uniform, compact-growth, and vacant-band geometries.
    if not witness and not bridge and not vertical and rng.random() < 0.38:
        if rng.random() < 0.5 and k <= (h - 1) * w:
            vacant = int(rng.integers(h))
            candidates = [p for p in candidates if p[0] != vacant]
        elif k <= h * (w - 1):
            vacant = int(rng.integers(w))
            candidates = [p for p in candidates if p[1] != vacant]
    if not witness and not bridge and not vertical and rng.random() < 0.35:
        chosen = {candidates[int(rng.integers(len(candidates)))]}
        while len(chosen) < k:
            frontier = [p for p in candidates if p not in chosen and any(
                abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1 for q in chosen)]
            if not frontier:
                frontier = [p for p in candidates if p not in chosen]
            chosen.add(frontier[int(rng.integers(len(frontier)))])
        for p in chosen:
            mask[p] = True
    else:
        for index in rng.choice(len(candidates), size=remaining, replace=False):
            mask[candidates[int(index)]] = True
    return mask


def _tile(rng, shape, k, palette_size, selector, background, thread=None,
          thread_mode=None, witness=False, bridge=False, multirow=False, vertical=False, required=()):
    """Construct frequencies first; spatial constraints never change counts."""
    h, w = shape
    mandatory = list(dict.fromkeys(c for c in (thread, *required)
                                   if c is not None and c not in (selector, background)))
    available = [c for c in range(10) if c not in (selector, background, *mandatory)]
    needed = palette_size - 2 - len(mandatory)
    if needed < 0:
        raise ValueError("palette cannot contain required colors")
    ordinary = [background, *mandatory,
                *(int(c) for c in rng.choice(available, size=needed, replace=False))]
    for _ in range(256):
        mask = _mask(rng, shape, k, witness, bridge, multirow, vertical)
        if mask is None:
            continue
        if witness and any(np.array_equal(mask, np.fliplr(np.rot90(mask, code % 4))
                                         if code >= 4 else np.rot90(mask, code % 4))
                           for code in range(1, 8)):
            continue
        # Every ordinary color is strictly more frequent than the selector.
        counts = np.full(len(ordinary), k + 1, dtype=int)
        extra = h*w - k - int(counts.sum())
        if extra < 0:
            raise ValueError("infeasible frequencies")
        counts += rng.multinomial(extra, np.full(len(counts), 1 / len(counts)))
        maxima = np.flatnonzero(counts == counts.max())
        if len(maxima) > 1:
            if counts[maxima[1]] <= k + 1:
                continue
            counts[maxima[0]] += 1
            counts[maxima[1]] -= 1
        # Assign the uniquely largest ordinary count to the visible field role.
        largest = int(np.argmax(counts))
        counts[0], counts[largest] = counts[largest], counts[0]
        supply = dict(zip(ordinary, (int(c) for c in counts)))
        tile = np.full(shape, -1, dtype=np.int8)
        tile[mask] = selector
        pins = {}
        free_center = [r for r in range(h) if not mask[r, w // 2]]
        if witness:
            if len(free_center) < 2:
                continue
            rng.shuffle(free_center)
            choices = [c for c in ordinary if c not in (background, thread)]
            pins[(free_center[0], w // 2)] = int(rng.choice(choices))
            pins[(free_center[1], w // 2)] = background
        elif thread_mode == "on":
            if not free_center:
                continue
            pins[(int(rng.choice(free_center)), w // 2)] = thread
        for p, color in pins.items():
            tile[p] = color
            supply[color] -= 1
        if any(count < 0 for count in supply.values()):
            continue
        if thread_mode == "off":
            positions = [(r, c) for r in range(h) for c in range(w)
                         if c != w // 2 and tile[r, c] == -1]
            if len(positions) < supply[thread]:
                continue
            for i in rng.choice(len(positions), size=supply[thread], replace=False):
                tile[positions[int(i)]] = thread
            supply[thread] = 0
        values = np.asarray([c for c, count in supply.items() for _ in range(count)], dtype=np.int8)
        rng.shuffle(values)
        tile[tile == -1] = values
        if witness:
            profile = tile[:, w // 2]
            if any(np.array_equal(tile[:, c], profile) for c in range(w) if c != w // 2):
                continue
        _require(_roles(tile) == (selector, background) and np.count_nonzero(tile == selector) == k,
                 "spatial construction changed prescribed frequency roles")
        return tile
    raise RuntimeError("could not realize constrained tile")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    order = [int(c) for c in rng.permutation(10)]
    thread = order[0]
    selectors = order[1:]
    fixed_background = order[-1] if seed % 3 == 0 else None
    used = set()
    puzzle = {"train": [], "test": []}
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            first = index == 0
            witness = split == "train" and first
            transfer = split == "test" and first
            trained_multirow = any(np.count_nonzero(_bridges(p["input"] == _roles(p["input"])[0]).any(axis=1)) > 1
                                   for p in puzzle["train"])
            for _ in range(128):
                multirow = split == "test" and trained_multirow and bool(rng.random() < 0.55)
                if witness:
                    shape = (int(rng.integers(3, 6)), 5)
                    k = (2 if shape[0] == 3 else
                         int(rng.integers(2, 5 if shape[0] == 5 else 4)))
                elif split == "train" and index == 1:
                    shape = (int(rng.integers(3, 6)), int(rng.choice([3, 5])))
                    first_count = int(np.count_nonzero(puzzle["train"][0]["input"] == selectors[0]))
                    k = 3 if first_count == 2 else 2
                elif multirow:
                    shape = (int(rng.choice([4, 5])), 5)
                    maximum = ((shape[0]*shape[1] - 3) // 3 if transfer
                               else (shape[0]*shape[1] - 1) // 2)
                    k = int(rng.integers(4, maximum + 1))
                else:
                    shape = (int(rng.integers(3 if transfer else 2, 6)),
                             int(rng.choice([3, 5])))
                    maximum = ((shape[0]*shape[1] - 3) // 3 if transfer or split == "train"
                               else (shape[0]*shape[1] - 1) // 2)
                    k = int(rng.integers(2 if transfer else 1, maximum + 1))
                if split == "train":
                    selector = selectors[index % (8 if fixed_background is not None else 9)]
                elif transfer:
                    ordinary_train = sorted({int(c) for p in puzzle["train"]
                                             for c in np.unique(p["input"])
                                             if c != _roles(p["input"])[0]
                                             and c != thread and c != fixed_background})
                    selector = int(rng.choice(ordinary_train))
                else:
                    selector = int(rng.choice([c for c in range(10)
                                              if c not in (thread, fixed_background)]))
                candidates = [c for c in range(10) if c not in (thread, selector)]
                background = (fixed_background if fixed_background is not None
                              else int(rng.choice(candidates)))
                required = (selectors[0],) if split == "train" and index == 1 else ()
                required = tuple(c for c in required if c not in (selector, background, thread))
                include_thread = split == "train" or transfer or bool(rng.random() < 0.45)
                minimum = 2 + int(include_thread) + len(required)
                if witness:
                    minimum = max(4, minimum)
                maximum = _max_palette(shape[0]*shape[1], k)
                if minimum > maximum and minimum > 2:
                    continue
                palette_size = (int(rng.integers(minimum, maximum + 1))
                                if minimum >= 3 else int(rng.integers(2, max(2, maximum) + 1)))
                tile = _tile(rng, shape, k, palette_size, selector, background,
                             thread if include_thread else None,
                             "off" if split == "train" else "on" if transfer else None,
                             witness=witness, bridge=transfer or multirow,
                             multirow=multirow, vertical=split == "train" and index == 1,
                             required=required)
                if split == "test" and np.count_nonzero(_bridges(tile == selector).any(axis=1)) > 1:
                    if not trained_multirow:
                        continue
                if _key(tile) not in used:
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
            used.add(_key(tile))
            puzzle[split].append({"input": tile, "output": solve(tile)})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check the rule block by block and verify the teaching witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid counts")
    seen, train_records, test_records = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            tile, out = (_grid(pair[k]) for k in ("input", "output"))
            h, w = tile.shape
            _require(2 <= h <= 5 and w in (3, 5), "invalid tile shape")
            _require(out.shape == (h*h, w*w), "invalid macro geometry")
            selector, background = _roles(tile)
            mask = tile == selector
            bridges = _bridges(mask)
            for r in range(h):
                for c in range(w):
                    block = out[r*h:(r+1)*h, c*w:(c+1)*w]
                    if mask[r, c]:
                        _require(np.array_equal(block, tile), "post lost full tile information")
                    elif bridges[r, c]:
                        for y in range(h):
                            _require(np.all(block[y] == tile[y, w//2]), "incorrect center-column ribbon")
                    else:
                        _require(np.all(block == background), "unbounded ribbon or wrong field")
            _require(set(np.unique(out)) == set(np.unique(tile)), "palette not preserved")
            _require(_key(tile) not in seen, "duplicate input")
            seen.add(_key(tile))
            record = tile, selector, background, mask, bridges
            (train_records if split == "train" else test_records).append(record)

    _require(len({r[1] for r in train_records}) >= 2, "selector identity not varied")
    _require(len({int(r[3].sum()) for r in train_records}) >= 2, "copy counts not varied")
    joined = [r for r in train_records if np.any(r[4])]
    _require(joined, "training never exposes ribbons")
    _require(any(len(np.unique(t[:, t.shape[1]//2])) >= 3 and
                 bg in t[:, t.shape[1]//2] for t, _, bg, _, _ in joined),
             "training lacks colored threads and a field gap")
    _require(any(np.any(~(mask | bridges)[r]) for _, _, _, mask, bridges in joined
                 for r in range(mask.shape[0]) if np.any(bridges[r])),
             "training does not distinguish bounded extension")
    _require(any(np.any(_bridges(mask.T).T & ~(mask | bridges))
                 for _, _, _, mask, bridges in train_records),
             "training lacks an unjoined vertical interval")
    _require(any(not np.array_equal(t, np.repeat(t[:, t.shape[1]//2, None], t.shape[1], axis=1))
                 for t, _, _, _, _ in joined), "whole-tile gap shortcut unresolved")
    # Every fixed alternative column must be rejected by a training ribbon.
    for side in ("left", "right", "inner_left", "inner_right"):
        _require(any(not np.array_equal(t[:, t.shape[1]//2], t[:, {
            "left": 0, "right": t.shape[1]-1,
            "inner_left": t.shape[1]//2-1, "inner_right": t.shape[1]//2+1}[side]])
            for t, _, _, _, _ in joined), f"{side} profile shortcut unresolved")
    all_centers = set().union(*(set(t[:, t.shape[1]//2].tolist()) for t, *_ in train_records))
    common_colors = set.intersection(*(set(np.unique(t).tolist()) for t, *_ in train_records))
    off_only = common_colors - all_centers
    _require(any(np.any(bridges) and bool(set(t[:, t.shape[1]//2].tolist()) & off_only)
                 for t, _, _, _, bridges in test_records), "missing off-axis-to-thread test shift")
    ordinary_train = set().union(*(set(np.unique(t).tolist()) - {s}
                                  for t, s, _, _, _ in train_records))
    _require(test_records[0][1] in ordinary_train,
             "first test selector was not ordinary content in training")
    if any(np.count_nonzero(r[4].any(axis=1)) > 1 for r in test_records):
        _require(any(np.count_nonzero(r[4].any(axis=1)) > 1 for r in train_records),
                 "multiple joined rows need a training witness")
    for code in range(1, 8):
        def transform(mask):
            rotated = np.rot90(mask, code % 4)
            return np.fliplr(rotated) if code >= 4 else rotated
        if all(np.array_equal(mask, transform(mask)) for _, _, _, mask, _ in train_records):
            _require(all(np.array_equal(mask, transform(mask)) for _, _, _, mask, _ in test_records),
                     "unresolved macro-mask orientation")


def self_test():
    """Bounded checks over 42 seeds, count boundaries, ablations, and diversity."""
    puzzles, shapes, masks, palettes, centers = set(), set(), set(), set(), set()
    backgrounds, selectors, counts, lengths = set(), set(), set(), set()
    bridged = unbridged = full_center = foreground_majority = multi_train = multi_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterministic puzzle")
        puzzles.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                tile = pair["input"]
                s, bg = _roles(tile)
                mask = tile == s
                gaps = _bridges(mask)
                shapes.add(tile.shape)
                masks.add((tile.shape, mask.tobytes()))
                palettes.add(tuple(np.unique(tile).tolist()))
                centers.add(tuple(tile[:, tile.shape[1]//2].tolist()))
                backgrounds.add(bg)
                selectors.add(s)
                counts.add(int(mask.sum()))
                bridged += bool(np.any(gaps))
                unbridged += not np.any(gaps)
                full_center += bool(np.all(tile[:, tile.shape[1]//2] != bg))
                foreground_majority += np.count_nonzero(tile != bg) > tile.size / 2
                if np.count_nonzero(gaps.any(axis=1)) > 1:
                    if split == "train":
                        multi_train += 1
                    else:
                        multi_test += 1
                for r in range(tile.shape[0]):
                    columns = np.flatnonzero(mask[r])
                    lengths.update(int(y-x-1) for x, y in zip(columns, columns[1:]) if y > x+1)
        tile = puzzle["train"][0]["input"]
        s, bg = _roles(tile)
        answer = puzzle["train"][0]["output"]
        _require(not np.array_equal(answer, _render(tile, bg, bg)), "omitting rarest selection has no effect")
        _require(not np.array_equal(answer, _render(tile, s, bg, whole_bridges=True)),
                 "omitting center extraction has no effect")
        for c in range(tile.shape[1]):
            if c != tile.shape[1] // 2:
                _require(not np.array_equal(answer, _render(tile, s, bg, profile=tile[:, c])),
                         "wrong-column profile fits the primary witness")

    _require(len(puzzles) == 42 and len(shapes) == 8, "insufficient puzzle/shape diversity")
    _require(len(masks) >= 90 and len(palettes) >= 70 and len(centers) >= 100,
             "insufficient structural/palette/profile variation")
    _require(backgrounds == selectors == set(range(10)), "not all frequency color roles exercised")
    _require({1, 2, 3, 4, 5} <= counts and {1, 2, 3} <= lengths, "count/span variation missing")
    _require(min(bridged, unbridged, full_center, foreground_majority, multi_train, multi_test) > 0,
             "structural regime missing")
    for train in range(2, 11):
        for test in range(1, min(4, train-1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + train*10 + test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {train}/{test}")


if __name__ == "__main__":
    self_test()
    print("42-seed self-test passed")
