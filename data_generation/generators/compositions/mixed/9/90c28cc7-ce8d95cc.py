"""Framed band mosaics: contract adjacent equal full vectors, preserving values.

The saved proposal fixes this operation.  The two parents overlap mathematically;
neither this module nor its validation claims independent parent necessity.
All generation is bounded, in memory, and uses only the standard library/NumPy.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


VERIFICATION_SEEDS = (
    0, 232898125, 986272970, 458417299, 226495402, 842059728,
    266138697, 170297305, 941687107, 404302189, 610405010, 360675370,
    715733679, 947250977, 965553011, 234303728, 306591026, 506896739,
    519759566, 507144873, 594582754, 611351337, 910056855, 367955535,
    215775143, 948720291, 987134890, 905056728, 690761489, 750604804,
    576735774, 24476471, 781009395, 733821660, 817026826, 267603327,
    978741813, 979616165, 278592029, 627724622, 280687904, 773431210,
)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    array = np.asarray(value)
    if array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape):
        raise ValueError("grid dimensions must be in 1..30")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("grid cells must be integers")
    if not np.all((0 <= array) & (array <= 9)):
        raise ValueError("grid colors must be in 0..9")
    return array


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the input; keep the first vector of each maximal equal run.

    The indices on both axes are computed from the ORIGINAL array. Keeping one
    copy of each row cannot remove a column difference, or conversely. There is
    no color priority, background removal, inferred scale, or line repainting.
    This also works for run-compressible inputs outside the sampled mesh layout.
    """
    grid = _grid(input_grid)
    rows = np.r_[True, np.any(grid[1:] != grid[:-1], axis=1)]
    columns = np.r_[True, np.any(grid[:, 1:] != grid[:, :-1], axis=0)]
    return grid[np.ix_(rows, columns)].astype(np.int8, copy=True)


def _partition(grid, axis):
    """Separate loop implementation for validation; return starts and lengths."""
    vectors = grid if axis == 0 else grid.T
    starts = [0]
    for i in range(1, len(vectors)):
        if tuple(map(int, vectors[i])) != tuple(map(int, vectors[i - 1])):
            starts.append(i)
    lengths = tuple(b - a for a, b in zip(starts, starts[1:] + [len(vectors)]))
    return starts, lengths


def _key(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def _separated_repeat(grid, axis):
    vectors = grid if axis == 0 else grid.T
    return any(np.array_equal(vectors[i], vectors[j])
               for i in range(len(vectors)) for j in range(i + 2, len(vectors)))


def _runs(rng, length, anchor=False, no_op=False):
    """Positive replication lengths with total <=30, without rejection sampling."""
    result = np.ones(length, dtype=int)
    if no_op:
        return result
    if anchor:
        # Actual interior bands, not just the margins, show unequal enlargement.
        indices = rng.choice(np.arange(1, length - 1), size=2, replace=False)
        result[indices] = (2, 3)
        extra = int(rng.integers(0, min(30 - int(result.sum()), 12) + 1))
        probabilities = np.r_[0., np.full(length - 1, 1 / (length - 1))]
        result += rng.multinomial(extra, probabilities)
        return result
    mode = int(rng.integers(3))
    if mode == 0:
        factor = int(rng.integers(2, 30 // length + 1))
        result[:] = factor
    else:
        extra = int(rng.integers(1, (min(8, 30 - length) if mode == 1 else 30 - length) + 1))
        if mode == 2:
            bulk = int(rng.integers((extra + 1) // 2, extra + 1))
            result[int(rng.integers(length))] += bulk
            extra -= bulk
        result += rng.multinomial(extra, np.full(length, 1 / length))
    return result


def _panels(rng, h, v, palette, anchor):
    style = int(rng.integers(5))
    if style == 0:
        result = rng.choice(palette, size=(h + 1, v + 1))
    elif style == 1:
        result = np.repeat(rng.choice(palette, size=(h + 1, 1)), v + 1, axis=1)
    elif style == 2:
        result = np.repeat(rng.choice(palette, size=(1, v + 1)), h + 1, axis=0)
    elif style == 3:
        i, j = np.indices((h + 1, v + 1))
        result = palette[(i + j + int(rng.integers(len(palette)))) % len(palette)]
    else:
        result = rng.choice(palette, size=(h + 1, v + 1))
        result[-1] = result[0]
        result[:, -1] = result[:, 0]
    if anchor:
        # Repeated full panel cross-sections, separated by real line bands.
        result[0, 0] = palette[0]
        result[-1] = result[0]
        result[:, -1] = result[:, 0]
        result[1, 1] = palette[1]
    elif np.unique(result).size == 1:
        result[-1, -1] = palette[1] if result[0, 0] == palette[0] else palette[0]
    return result.astype(np.int8)


def _crossings(rng, h, v, kind):
    if kind == 0:
        return np.zeros((h, v), dtype=np.int8)
    if kind == 1:
        return np.ones((h, v), dtype=np.int8)
    if kind == 2:
        return np.repeat(rng.integers(0, 2, size=(h, 1)), v, axis=1)
    if kind == 3:
        return np.repeat(rng.integers(0, 2, size=(1, v)), h, axis=0)
    return rng.integers(0, 2, size=(h, v), dtype=np.int8)


def _make_pair(rng, h, v, background, role):
    # Panel colors and line colors are disjoint, so adjacent compact vectors
    # always differ. Literal digits have no fixed role across examples.
    anchor = role == "mixed_control"
    content = rng.permutation([c for c in range(10) if c != background])
    panel_count = int(rng.integers(2, 4 if anchor else 5))
    panel_palette, line_palette = content[:panel_count], content[panel_count:]
    panels = _panels(rng, h, v, panel_palette, anchor)
    color_mode = 0 if anchor else int(rng.integers(4))
    colors = rng.choice(line_palette, size=h + v,
                        replace=color_mode != 0 or h + v > len(line_palette))
    horizontal, vertical = colors[:h].copy(), colors[h:].copy()
    if color_mode in (1, 3):
        if h >= 2:
            horizontal[1] = horizontal[0]
        if v >= 2:
            vertical[1] = vertical[0]
    if color_mode in (2, 3):
        vertical[0] = horizontal[0]
    if role == "single_owner_control":
        # Keep ownership observable, even for a single line on either axis.
        pivot = len(line_palette) // 2
        horizontal = rng.choice(line_palette[:pivot], size=h)
        vertical = rng.choice(line_palette[pivot:], size=v)
    kind = int(rng.integers(2)) if role == "single_owner_control" else int(rng.integers(5))
    bits = _crossings(rng, h, v, kind)
    if anchor:
        # A 2x2 checkerboard has both crossing owners along each participating line.
        bits[:2, :2] = np.array([[0, 1], [1, 0]], dtype=np.int8)
    mosaic = np.empty((2 * h + 1, 2 * v + 1), dtype=np.int8)
    mosaic[::2, ::2] = panels
    mosaic[1::2, ::2] = horizontal[:, None]
    mosaic[::2, 1::2] = vertical[None, :]
    mosaic[1::2, 1::2] = np.where(bits, vertical[None, :], horizontal[:, None])
    output = np.full((2 * h + 3, 2 * v + 3), background, dtype=np.int8)
    output[1:-1, 1:-1] = mosaic
    no_op = int(rng.integers(2)) if role == "ordinary" and rng.random() < .18 else None
    rows = _runs(rng, output.shape[0], anchor=role != "ordinary", no_op=no_op == 0)
    columns = _runs(rng, output.shape[1], anchor=role != "ordinary", no_op=no_op == 1)
    source = np.repeat(np.repeat(output, rows, axis=0), columns, axis=1)
    # Ground truth is the constructed compact canvas, not a call to solve().
    return {"input": source, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    base_background = int(rng.integers(10))
    varying_background = bool(rng.integers(2))
    anchor_inventory = (int(rng.integers(2, 4)), int(rng.integers(2, 4)))
    alternatives = [(h, v) for h in range(1, 4) for v in range(1, 4)
                    if (h, v) != anchor_inventory]
    second_inventory = alternatives[int(rng.integers(len(alternatives)))]
    train_inventory = [anchor_inventory, second_inventory]
    train_inventory += [(int(rng.integers(1, 4)), int(rng.integers(1, 4)))
                        for _ in range(num_train - 2)]
    max_h, max_v = map(max, zip(*train_inventory))
    if rng.random() < .5:
        shifted = (max_h + 1, int(rng.integers(1, max_v + 1)))
    else:
        shifted = (int(rng.integers(1, max_h + 1)), max_v + 1)
    other_tests = [(h, v) for h in range(1, 5) for v in range(1, 5) if (h, v) != shifted]
    rng.shuffle(other_tests)
    test_inventory = [shifted] + other_tests[:num_test - 1]
    puzzle = {"train": [], "test": []}
    used = set()
    for split, inventory in (("train", train_inventory), ("test", test_inventory)):
        for index, (h, v) in enumerate(inventory):
            if not varying_background or split == "train" and index == 0:
                background = base_background
            elif split == "train" and index == 1:
                background = (base_background + int(rng.integers(1, 10))) % 10
            else:
                background = int(rng.integers(10))
            role = ("mixed_control" if split == "train" and index == 0 else
                    "single_owner_control" if split == "train" and index == 1 else "ordinary")
            for _ in range(32):
                pair = _make_pair(rng, h, v, background, role)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("32 distinct-input attempts exhausted")
    # The controls have no fixed demonstration position.
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    source, output = _grid(pair["input"]), _grid(pair["output"])
    starts_r, runs_r = _partition(source, 0)
    starts_c, runs_c = _partition(source, 1)
    compact = source[np.ix_(starts_r, starts_c)]
    assert np.array_equal(compact, output), "output is not exact maximal-run contraction"
    assert source.shape != output.shape, "every sampled pair must compress at least one axis"
    assert np.array_equal(np.repeat(np.repeat(output, runs_r, axis=0), runs_c, axis=1), source)
    assert set(map(int, np.unique(source))) == set(map(int, np.unique(output)))
    height, width = compact.shape
    assert 5 <= height <= 11 and 5 <= width <= 11 and height % 2 == width % 2 == 1
    h, v = (height - 3) // 2, (width - 3) // 2
    bg = int(compact[0, 0])
    assert np.all(compact[[0, -1], :] == bg) and np.all(compact[:, [0, -1]] == bg)
    mosaic = compact[1:-1, 1:-1]
    assert np.all(mosaic != bg), "mosaic is a filled rectangle distinct from its exterior"
    yy, xx = np.nonzero(source != bg)
    rectangle = source[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    assert np.all(rectangle != bg) and np.count_nonzero(source != bg) == rectangle.size
    panels = mosaic[::2, ::2]
    assert 2 <= np.unique(panels).size <= 4, "panel-color preservation must be nontrivial"
    horizontal, vertical = mosaic[1::2, 0], mosaic[0, 1::2]
    assert np.all(mosaic[1::2, ::2] == horizontal[:, None])
    assert np.all(mosaic[::2, 1::2] == vertical[None, :])
    assert not set(map(int, panels.flat)) & set(map(int, np.r_[horizontal, vertical]))
    crossings = mosaic[1::2, 1::2]
    assert np.all((crossings == horizontal[:, None]) | (crossings == vertical[None, :]))
    bits = {int(crossings[i, j] == vertical[j]) for i in range(h) for j in range(v)
            if horizontal[i] != vertical[j]}
    # Check full rows/columns rather than individual monochromatic components.
    assert all(len(partition) == side for partition, side in
               ((_partition(compact, 0)[0], height), (_partition(compact, 1)[0], width)))
    return {"h": h, "v": v, "background": bg, "rows": runs_r, "columns": runs_c,
            "panel_colors": int(np.unique(panels).size), "ownership": bits,
            "repeat_rows": _separated_repeat(mosaic, 0),
            "repeat_columns": _separated_repeat(mosaic, 1),
            "distinct_lines": len(set(map(int, np.r_[horizontal, vertical]))) == h + v,
            "parallel_reuse": len(set(map(int, horizontal))) < h or len(set(map(int, vertical))) < v,
            "cross_reuse": bool(set(map(int, horizontal)) & set(map(int, vertical)))}


def validate_generated_puzzle(puzzle):
    """Check returned pixels and demonstration witnesses; do not use generation state."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _analyze_pair(pair)
            analyses[split].append(analysis)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen, "duplicate input in puzzle"
            seen.add(key)
            assert np.array_equal(solve(np.asarray(pair["input"])), pair["output"])
    train, test = analyses["train"], analyses["test"]
    assert len({(a["h"], a["v"]) for a in train}) >= 2
    assert any(a["repeat_rows"] and a["repeat_columns"] and a["ownership"] == {0, 1}
               and a["distinct_lines"] and max(a["rows"][1:-1]) > 1
               and max(a["columns"][1:-1]) > 1
               and len(set(a["rows"])) > 1 and len(set(a["columns"])) > 1 for a in train), \
        "missing unequal-expansion, separated-repeat, mixed-crossing training control"
    assert any(len(a["ownership"]) == 1 for a in train), "missing single-owner crossing control"
    assert len({(a["h"], a["v"]) for a in test}) == len(test)
    assert any(a["h"] > max(b["h"] for b in train) or
               a["v"] > max(b["v"] for b in train) for a in test), "missing inventory shift"
    train_backgrounds = {a["background"] for a in train}
    if not {a["background"] for a in test} <= train_backgrounds:
        assert len(train_backgrounds) >= 2


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for name in ("input", "output"):
                grid = np.asarray(pair[name], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _structure_key(grid):
    roles = {}
    normalized = bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)
    return grid.shape, hashlib.sha256(normalized).digest()


def self_test():
    """Bounded semantic checks and 42-seed structural measurements; writes no files."""
    hand = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 5, 1, 6, 1, 0],
                     [0, 3, 3, 3, 6, 3, 0],
                     [0, 1, 5, 2, 6, 1, 0],
                     [0, 4, 5, 4, 4, 4, 0],
                     [0, 1, 5, 1, 6, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    expanded = np.repeat(np.repeat(hand, [1, 3, 2, 1, 4, 2, 1], axis=0),
                         [2, 1, 4, 2, 3, 1, 2], axis=1)
    assert np.array_equal(solve(expanded), hand)
    assert np.array_equal(solve(expanded.T), hand.T)
    # Panel data and crossing data each affect the answer without changing the
    # other. These literal-data probes do not establish separate parent necessity.
    for position, replacement in (((3, 3), 8), ((2, 4), 3)):
        changed = hand.copy()
        changed[position] = replacement
        changed_input = np.repeat(np.repeat(changed, [1, 3, 2, 1, 4, 2, 1], axis=0),
                                  [2, 1, 4, 2, 3, 1, 2], axis=1)
        assert np.array_equal(solve(changed_input), changed)
        assert not np.array_equal(changed, hand)
        try:
            _analyze_pair({"input": expanded, "output": changed})
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted altered retained data")
    literal = np.array([[1, 1, 2], [1, 1, 2], [3, 3, 4], [1, 1, 2]])
    assert np.array_equal(solve(literal), [[1, 2], [3, 4], [1, 2]])
    assert np.array_equal(solve(np.full((30, 30), 9)), [[9]])
    assert np.array_equal(solve([[7]]), [[7]])
    assert np.array_equal(solve([[2, 2, 4, 2, 2]]), [[2, 4, 2]])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(1234567 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1),
                 (4, None), ("4", 1), (np.nan, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 0.5, "0", None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    fingerprints, input_shapes, output_shapes, structures, layouts, backgrounds = (set() for _ in range(6))
    stats = Counter()
    for seed in VERIFICATION_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        local_backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, output = pair["input"], pair["output"]
                analysis = _analyze_pair(pair)
                stats["pairs"] += 1
                input_shapes.add(source.shape)
                output_shapes.add(output.shape)
                structures.add(_structure_key(output))
                layouts.add((output.shape, analysis["rows"], analysis["columns"]))
                backgrounds.add(analysis["background"])
                local_backgrounds.add(analysis["background"])
                stats["mixed_crossings"] += analysis["ownership"] == {0, 1}
                stats["all_horizontal"] += analysis["ownership"] == {0}
                stats["all_vertical"] += analysis["ownership"] == {1}
                stats["parallel_reuse"] += analysis["parallel_reuse"]
                stats["cross_reuse"] += analysis["cross_reuse"]
                stats["separated_repeats_both_axes"] += analysis["repeat_rows"] and analysis["repeat_columns"]
                stats["one_no_op_axis"] += (source.shape[0] == output.shape[0]) != (source.shape[1] == output.shape[1])
                stats["large_canvas"] += max(source.shape) >= 24
                stats["rectangular_canvas"] += max(source.shape) >= 2 * min(source.shape)
                stats["large_band"] += max(analysis["rows"] + analysis["columns"]) >= 8
                assert np.array_equal(solve(output), output), "contraction is not idempotent"
                assert np.array_equal(solve(source.T), output.T), "axis order changed result"
                # A bijection of literal colors cannot affect the retained positions.
                color_map = np.array([9, 4, 7, 2, 6, 0, 8, 1, 5, 3], dtype=np.int8)
                assert np.array_equal(solve(color_map[source]), color_map[output])
                # These are shortcut counterexamples, NOT proofs of A/B necessity.
                stats["panel_discard_counterexamples"] += analysis["panel_colors"] > 1
                unique_rows = len({tuple(map(int, row)) for row in output[1:-1, 1:-1]})
                unique_cols = len({tuple(map(int, col)) for col in output[1:-1, 1:-1].T})
                stats["global_dedup_counterexamples"] += (unique_rows < output.shape[0] - 2 or
                                                         unique_cols < output.shape[1] - 2)
        stats["varying_background_puzzles" if len(local_backgrounds) > 1 else "shared_background_puzzles"] += 1
    assert len(fingerprints) == 42
    assert len(input_shapes) >= 60 and len(output_shapes) >= 12
    assert len(structures) >= 120 and len(layouts) >= 150 and len(backgrounds) >= 8
    for name in ("all_horizontal", "all_vertical", "parallel_reuse", "cross_reuse",
                 "one_no_op_axis", "large_canvas", "rectangular_canvas", "large_band",
                 "shared_background_puzzles", "varying_background_puzzles"):
        assert stats[name] >= 3, (name, stats[name])
    assert stats["separated_repeats_both_axes"] >= 42 and stats["mixed_crossings"] >= 42
    assert stats["panel_discard_counterexamples"] == 210
    assert stats["global_dedup_counterexamples"] >= 42
    return {"seeds": len(VERIFICATION_SEEDS), "distinct_puzzles": len(fingerprints),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "normalized_output_structures": len(structures), "expansion_layouts": len(layouts),
            "background_colors": len(backgrounds), "counts": dict(sorted(stats.items())),
            "semantic_verification": "generator-owned invariants and metamorphic checks only"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
