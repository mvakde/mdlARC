"""Bottom-anchored scaffold wave, then stable full-height column compression.

Only NumPy and the standard library are used. No parent code or files are
loaded at runtime. Tests and generation retain no execution traces or files.
"""
from __future__ import annotations

from itertools import product
import hashlib
import json

import numpy as np


_CYCLE = (0, -1, 0, 1)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.tobytes()


def _vectors(grid):
    return [tuple(map(int, grid[:, c])) for c in range(grid.shape[1])]


def _indices(grid):
    seen, keep = set(), []
    for c, vector in enumerate(_vectors(grid)):
        if vector not in seen:
            keep.append(c)
            seen.add(vector)
    return keep


def _compress(grid):
    return grid[:, _indices(grid)].copy()


def _geometry(grid):
    """Parse only the visible input, including thick rails and partial bars."""
    bg = int(grid[0, 0])
    colors = np.unique(grid)
    if len(colors) != 2:
        raise ValueError("A uniform background and one scaffold color are required")
    fg = int(colors[colors != bg][0])
    mask = grid != bg
    rows, cols = np.where(mask)
    t, b, l, r = map(int, (rows.min(), rows.max(), cols.min(), cols.max()))
    if b - t + 1 < 5 or l < 1 or r >= grid.shape[1] - 1:
        raise ValueError("Scaffold needs five rows and a lateral safety margin")
    if not np.all(mask[t:b + 1].any(axis=1)):
        raise ValueError("Scaffold rows must be contiguous")
    if not np.all(mask[[t, b], l:r + 1]):
        raise ValueError("The top and bottom bars must span the scaffold")
    core = mask[t + 1:b, l:r + 1]
    rails = np.all(core, axis=0)
    starts = np.flatnonzero(rails & ~np.r_[False, rails[:-1]])
    ends = np.flatnonzero(rails & ~np.r_[rails[1:], False])
    if len(starts) < 2 or not rails[0] or not rails[-1]:
        raise ValueError("At least two separated continuous supports are required")
    if sum(np.array_equal(row, rails) for row in core) < 2:
        raise ValueError("At least two rows must expose the bare support pattern")
    gaps = [(int(a + 1), int(z)) for a, z in zip(ends[:-1], starts[1:])]
    bars, partial, full = [], [], []
    for y, row in enumerate(core, t + 1):
        filled = []
        for a, z in gaps:
            segment = row[a:z]
            if not (np.all(segment) or not np.any(segment)):
                raise ValueError("A crossbar must fill a complete inter-support gap")
            filled.append(bool(np.all(segment)))
        if any(filled):
            bars.append(y)
            (full if all(filled) else partial).append(y)
    return dict(bg=bg, fg=fg, top=t, bottom=b, left=l, right=r,
                height=b - t + 1, width=r - l + 1,
                supports=tuple((int(a), int(z)) for a, z in zip(starts, ends)),
                gaps=tuple(z - a for a, z in gaps), bars=tuple(bars),
                partial=tuple(partial), full=tuple(full),
                bottom_padding=grid.shape[0] - b - 1)


def _wave(grid, info, table=_CYCLE, anchor="bottom"):
    result = np.full_like(grid, info["bg"])
    for row in range(info["top"], info["bottom"] + 1):
        if anchor == "top":
            distance = row - info["top"]
        elif anchor == "canvas":
            distance = grid.shape[0] - 1 - row
        elif anchor == "bars":
            below = [b for b in info["full"] + (info["top"], info["bottom"]) if b >= row]
            distance = min(below) - row
        else:
            distance = info["bottom"] - row
        columns = np.flatnonzero(grid[row] != info["bg"])
        shifted = columns + table[distance % 4]
        if shifted.min() < 0 or shifted.max() >= grid.shape[1]:
            raise ValueError("The wave would clip the scaffold")
        result[row, shifted] = grid[row, columns]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute A before B; colors and the bottom anchor come from this input."""
    source = _grid(input_grid)
    info = _geometry(source)
    return _compress(_wave(source, info))


def _reference(source):
    """Cellwise check, independent of the production wave/dedup helpers."""
    bg = int(source[0, 0])
    bottom = max(r for r in range(len(source)) if any(v != bg for v in source[r]))
    moved = [[bg] * source.shape[1] for _ in range(source.shape[0])]
    for r in range(source.shape[0]):
        d = bottom - r
        delta = 0 if d % 2 == 0 else (-1 if d % 4 == 1 else 1)
        for c in range(source.shape[1]):
            if source[r, c] != bg:
                _require(0 <= c + delta < source.shape[1], "Reference wave clips")
                moved[r][c + delta] = int(source[r, c])
    columns = list(zip(*moved))
    unique = []
    for column in columns:
        if column not in unique:
            unique.append(column)
    return np.array(moved, dtype=np.int8), np.array(unique, dtype=np.int8).T.copy()


def _features(source, moved, info):
    before, after = _vectors(source), _vectors(moved)
    blank = (info["bg"],) * source.shape[0]
    new_merge = split = nonadj = False
    for j in range(len(after)):
        for k in range(j + 1, len(after)):
            if after[j] == after[k] and after[j] != blank and k > j + 1:
                nonadj = True
                new_merge |= before[j] != before[k]
            split |= before[j] == before[k] and after[j] != after[k]
    return dict(new_merge=new_merge, split=split, nonadj=nonadj,
                types=len(set(after)), blank=blank in after,
                removed=source.shape[1] - len(set(after)))


def _alternatives(source, moved, info):
    """Concrete shortcuts, compared against demonstrations after compaction."""
    keys = _vectors(moved)
    unique = _indices(moved)
    adjacent = [c for c in range(len(keys)) if c == 0 or keys[c] != keys[c - 1]]
    last = [c for c in range(len(keys)) if keys[c] not in keys[c + 1:]]
    without_blank = [c for c in unique if any(v != info["bg"] for v in keys[c])]
    result = {
        "omit_A": _compress(source),
        "omit_B": moved,
        "choose_columns_before_wave": moved[:, _indices(source)],
        "adjacent_only": moved[:, adjacent],
        "keep_last": moved[:, last],
        "sort_vectors": np.array(sorted(set(keys)), dtype=np.int8).T,
        "crop_prefix": moved[:, :len(unique)],
        "discard_blank": moved[:, without_blank],
        "dedup_rows_too": _compress(_compress(moved).T).T,
        "top_anchor": _compress(_wave(source, info, anchor="top")),
        "canvas_anchor": _compress(_wave(source, info, anchor="canvas")),
        "restart_at_full_bar": _compress(_wave(source, info, anchor="bars")),
        "opposite_cycle": _compress(_wave(source, info, (0, 1, 0, -1))),
    }
    for phase in (1, 2, 3):
        table = _CYCLE[phase:] + _CYCLE[:phase]
        result[f"phase_{phase}"] = _compress(_wave(source, info, table))
    return result


def _training_evidence(train):
    rejected = set()
    features, infos = [], []
    alternatives = set()
    for pair in train:
        source, target = _grid(pair["input"]), _grid(pair["output"])
        info = _geometry(source)
        moved, _ = _reference(source)
        for name, prediction in _alternatives(source, moved, info).items():
            alternatives.add(name)
            if not np.array_equal(prediction, target):
                rejected.add(name)
        features.append(_features(source, moved, info))
        infos.append(info)
    _require(rejected == alternatives, "Training admits a documented shortcut: "
             + ",".join(sorted(alternatives - rejected)))
    _require(any(f["new_merge"] for f in features), "No wave-created nonadjacent equality")
    _require(any(f["split"] for f in features), "Wave never splits a source equality")
    _require(len({i["height"] % 4 for i in infos}) == 2,
             "Training must expose two different height phases")
    _require(len({i["bottom_padding"] % 4 for i in infos}) >= 2,
             "Training does not distinguish the occupied bottom from the canvas")
    return rejected


def _sample_candidate(rng, height, bottom_residue, style, force_full=False):
    strong = style == "merge"
    count = int(rng.integers(3 if strong else 2, 7))
    if strong:
        widths = [int(rng.integers(1, 4))] * count
        gaps = [int(rng.integers(2, 6))] * (count - 1)
    else:
        widths = rng.integers(1, 4, size=count).tolist()
        gaps = rng.integers(1, 7, size=count - 1).tolist()
    width = sum(widths) + sum(gaps)
    if width > (27 if strong else 28):
        return None
    bottom_options = list(range(bottom_residue, 31 - height, 4))
    if not bottom_options:
        return None
    bottom_pad = int(rng.choice(bottom_options))
    top_pad = int(rng.integers(0, min(6, 30 - height - bottom_pad) + 1))
    margin_sum = int(rng.integers(3 if strong else 2, 31 - width))
    left_pad = int(rng.integers(1, margin_sum))
    bg = 0 if rng.random() < 0.5 else int(rng.integers(1, 10))
    fg = int(rng.choice([c for c in range(10) if c != bg]))
    grid = np.full((top_pad + height + bottom_pad, width + margin_sum), bg, dtype=np.int8)
    gap_spans, cursor = [], left_pad
    for i, thickness in enumerate(widths):
        grid[top_pad:top_pad + height, cursor:cursor + thickness] = fg
        cursor += thickness
        if i < len(gaps):
            gap_spans.append((cursor, cursor + gaps[i]))
            cursor += gaps[i]
    grid[top_pad, left_pad:left_pad + width] = fg
    grid[top_pad + height - 1, left_pad:left_pad + width] = fg
    interior = list(range(1, height - 1))
    bars = {}
    if strong:
        # A partial bridge on a moving row erases a source-column distinction:
        # both trailing (or leading) gap cells sample a support after translation.
        d = int(rng.choice([v for v in interior if v % 2]))
        first = int(rng.integers(0, count - 1))
        last = int(rng.integers(first, count - 1))
        if first == 0 and last == count - 2:
            last -= 1
        bars[d] = (first, last)
        for distance in interior:
            if distance != d and rng.random() < 0.18:
                bars[distance] = (0, count - 2)
    elif style != "hollow":
        density = float(rng.uniform(0.12, 0.46))
        for d in interior:
            if rng.random() < density:
                first = int(rng.integers(0, count - 1))
                last = int(rng.integers(first, count - 1))
                bars[d] = (0, count - 2) if style == "full" or rng.random() < 0.5 else (first, last)
    protected = set()
    if strong:
        protected.add(d)
    if force_full:
        full_d = int(rng.choice([v for v in interior if v % 2 and v not in protected]))
        bars[full_d] = (0, count - 2)
        protected.add(full_d)
    # Expose at least two bare support rows, even on short rich scaffolds.
    while len(bars) > height - 4:
        removable = [v for v in bars if v not in protected]
        del bars[int(rng.choice(removable))]
    for distance, (first, last) in bars.items():
        row = top_pad + height - 1 - distance
        grid[row, gap_spans[first][0]:gap_spans[last][1]] = fg
    return grid


def _new_pair(rng, heights, bottom_residue, style, seen, force_full=False):
    for _ in range(256):
        height = int(rng.choice(heights))
        source = _sample_candidate(rng, height, bottom_residue, style, force_full)
        if source is None or _key(source) in seen:
            continue
        info = _geometry(source)
        moved = _wave(source, info)
        target = _compress(moved)
        f = _features(source, moved, info)
        if not f["nonadj"] or f["removed"] < 1:
            continue
        if style == "merge" and not f["new_merge"]:
            continue
        if (np.array_equal(_compress(source), target)
                or np.array_equal(moved[:, _indices(source)], target)
                or np.array_equal(moved[:, :target.shape[1]], target)):
            continue
        return {"input": source, "output": target}
    raise RuntimeError("Bounded scaffold sampler exhausted its 256 attempts")


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"Illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    residues = rng.permutation(4).tolist()
    train_residues, test_residues = residues[:2], residues[2:]
    pads = rng.choice(4, size=2, replace=False).tolist()
    seen, train = set(), []
    # Both required mechanisms remain visible even at the minimum 2/1 counts.
    for i in range(2):
        heights = [h for h in range(10 if i == 0 else 7, 26) if h % 4 == train_residues[i]]
        for _ in range(64):
            pair = _new_pair(rng, heights, pads[i], "merge" if i == 0 else "partial", seen,
                             force_full=i == 1)
            if i == 1:
                try:
                    _training_evidence(train + [pair])
                except AssertionError:
                    continue
            train.append(pair)
            seen.add(_key(pair["input"]))
            break
        else:
            raise RuntimeError("Could not establish the required training evidence")
    styles = ("hollow", "full", "partial", "merge")
    for _ in range(num_train - 2):
        heights = [h for h in range(7, 26) if h % 4 in train_residues]
        pair = _new_pair(rng, heights, int(rng.integers(4)), str(rng.choice(styles)), seen)
        train.append(pair)
        seen.add(_key(pair["input"]))
    rng.shuffle(train)
    test = []
    for _ in range(num_test):
        heights = [h for h in range(7, 29) if h % 4 in test_residues]
        # At height 28 only padding residues 0..2 fit; filter visibly feasible heights.
        pad = int(rng.integers(4))
        heights = [h for h in heights if h + pad <= 30]
        pair = _new_pair(rng, heights, pad, str(rng.choice(styles)), seen)
        test.append(pair)
        seen.add(_key(pair["input"]))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal pair counts")
    seen, residues = set(), {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(source) not in seen, "Duplicate input")
            seen.add(_key(source))
            info = _geometry(source)
            residues[split].add(info["height"] % 4)
            moved, expected = _reference(source)
            _require(np.array_equal(target, expected), "Wrong cellwise wave/first-occurrence output")
            _require(np.array_equal(solve(source), target), "Production solve disagrees")
            _require(target.shape[0] == source.shape[0] and target.shape[1] < source.shape[1],
                     "Only width may shrink")
            _require(len(set(_vectors(target))) == target.shape[1], "Output retains duplicate columns")
            _require(set(map(int, source.flat)) == set(map(int, target.flat)), "Palette changed")
            _require(np.array_equal((source != info["bg"]).sum(axis=1),
                                    (moved != info["bg"]).sum(axis=1)), "Wave lost cells")
            for row in range(source.shape[0]):
                x = np.flatnonzero(source[row] != info["bg"])
                y = np.flatnonzero(moved[row] != info["bg"])
                if len(x):
                    _require(np.all(y - x == _CYCLE[(info["bottom"] - row) % 4]),
                             "Wave is not a rigid bottom-phased row translation")
            f = _features(source, moved, info)
            _require(f["nonadj"], "No informative separated duplicate columns")
            _require(not np.array_equal(_compress(source), target), "A has no causal effect")
            _require(not np.array_equal(moved[:, _indices(source)], target), "Pre-wave selection suffices")
            _require(not np.array_equal(moved[:, :target.shape[1]], target), "Prefix crop suffices")
    _require(residues["train"].isdisjoint(residues["test"]), "Missing held-out height-phase shift")
    _training_evidence(puzzle["train"])


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = _grid(pair[role])
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def _phase_candidates(train):
    """A bounded hypothesis check, not an independent ARC/model solver."""
    candidates = []
    parsed = [(_grid(p["input"]), _geometry(_grid(p["input"])), _grid(p["output"])) for p in train]
    for table in product((-1, 0, 1), repeat=4):
        if all(np.array_equal(_compress(_wave(source, info, table)), target)
               for source, info, target in parsed):
            candidates.append(table)
    return candidates


def self_test():
    """42 streamed seeds, count boundaries, determinism, semantics and diversity."""
    fingerprints, shapes, masks, heights, widths, supports, bars, palettes = (set() for _ in range(8))
    new_merges = partials = thick = irregular = zero_fg = nonzero_bg = 0
    test_residues, pair_count = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = _fingerprint(puzzle)
        _require(digest == _fingerprint(generate_puzzle(seed)), "Nondeterministic seed")
        fingerprints.add(digest)
        _require(_phase_candidates(puzzle["train"]) == [_CYCLE], "Training leaves a four-phase ambiguity")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source = _grid(pair["input"])
                info = _geometry(source)
                moved, target = _reference(source)
                _require(np.array_equal(target, pair["output"]), "Reference check failed")
                shapes.add(source.shape)
                crop = source[info["top"]:info["bottom"] + 1, info["left"]:info["right"] + 1]
                masks.add((crop.shape, (crop != info["bg"]).tobytes()))
                heights.add(info["height"])
                widths.add(info["width"])
                supports.add(len(info["supports"]))
                bars.add(len(info["bars"]))
                palettes.add((info["bg"], info["fg"]))
                new_merges += _features(source, moved, info)["new_merge"]
                partials += bool(info["partial"])
                thick += any(z > a for a, z in info["supports"])
                irregular += len(set(info["gaps"])) > 1
                zero_fg += info["fg"] == 0
                nonzero_bg += info["bg"] != 0
                pair_count += 1
                if split == "test":
                    test_residues.add(info["height"] % 4)
    _require(len(fingerprints) == 42 and len(shapes) >= 40 and len(masks) >= 100,
             "Insufficient structural diversity across 42 seeds")
    _require(len(heights) >= 17 and len(widths) >= 15 and len(supports) >= 4 and len(bars) >= 7,
             "Geometry coverage is too narrow")
    _require(new_merges >= 42 and partials >= 42 and thick >= 50 and irregular >= 30,
             "Compatible structure combinations are missing")
    _require(len(palettes) >= 30 and zero_fg and nonzero_bg and test_residues == {0, 1, 2, 3},
             "Color roles or held-out phases are missing")
    legal_counts = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(42 + a * 10 + b, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "Counts ignored")
            legal_counts += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None), ("4", 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Corruption must fail the semantic validator, even with a legal ARC schema.
    bad = generate_puzzle(0)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a wrong answer")
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(fingerprints),
                input_shapes=len(shapes), cropped_masks=len(masks), heights=sorted(heights),
                widths=sorted(widths), support_counts=sorted(supports), bar_counts=sorted(bars),
                palettes=len(palettes), wave_created_merges=new_merges, partial_bar_pairs=partials,
                thick_support_pairs=thick, unequal_gap_pairs=irregular, zero_foreground_pairs=zero_fg,
                nonzero_background_pairs=nonzero_bg, unique_phase_table_puzzles=42,
                legal_count_combinations=legal_counts, independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
