"""Four aligned, transparent ARC panels with reading-order priority."""

from itertools import permutations
import numpy as np


_SOURCE = {
    ("770014404/707014000/700714440/707714044/111111111/008010000/008016600/008010666/080810660", "7784/7670/7447/7877"),
    ("777010400/707014044/070714044/000710000/111111111/008016006/000016000/000016606/888016066", "7776/7074/4747/8887"),
    ("007710440/000710044/777710004/070010440/111111111/008810666/000010060/000816060/800016600", "0477/0047/7777/8740"),
    ("770010004/700014444/707014000/077014440/111111111/808016666/008810060/000010606/888810006", "7784/7444/7676/4778"),
    ("077710404/777014400/000010004/700010000/111111111/000016660/008810000/808016006/000810000", "6777/7778/8084/7008"),
    ("700714440/077714404/777014404/777010400/111111111/880816666/088810006/080810060/880810600", "7447/4777/7774/7778"),
    ("777010040/077014404/777710404/700014040/111111111/000810606/800816006/808016666/080810600", "7778/4774/7777/7848"),
}


def _checked_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if name == "num_train" and not 2 <= value <= 10:
        raise ValueError("num_train must be 2..10")
    if name == "num_test" and not 1 <= value <= 4:
        raise ValueError("num_test must be 1..4")
    return value


def _counts(rng, num_train, num_test):
    nt = _checked_count(num_train, "num_train")
    ns = _checked_count(num_test, "num_test")
    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
    if nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    if ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _shape(rng):
    mode = int(rng.integers(0, 5))
    if mode == 0:
        return 4, 4
    if mode == 1:
        side = int(rng.integers(5, 15))
        return side, side
    if mode == 2:
        return int(rng.integers(4, 9)), int(rng.integers(9, 15))
    if mode == 3:
        return int(rng.integers(9, 15)), int(rng.integers(4, 9))
    return int(rng.integers(4, 15)), int(rng.integers(4, 15))


def _primitive(rng, h, w):
    mode = int(rng.integers(0, 8))
    area = h * w
    mask = np.zeros((h, w), dtype=bool)
    if mode == 0:
        mask = rng.random((h, w)) < rng.uniform(0.12, 0.75)
    elif mode == 1:
        for _ in range(int(rng.integers(1, 5))):
            a = int(rng.integers(h))
            b = int(rng.integers(w))
            mask[a:int(rng.integers(a + 1, h + 1)),
                 b:int(rng.integers(b + 1, w + 1))] = True
    elif mode == 2:
        target = int(rng.integers(1, max(3, area // 2 + 1)))
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(area * 30):
            mask[r, c] = True
            if int(mask.sum()) >= target:
                break
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            r = min(h - 1, max(0, r + dr))
            c = min(w - 1, max(0, c + dc))
    elif mode == 3:
        for _ in range(int(rng.integers(1, 7))):
            if int(rng.integers(2)):
                r = int(rng.integers(h))
                a = int(rng.integers(w))
                mask[r, a:int(rng.integers(a + 1, w + 1))] = True
            else:
                c = int(rng.integers(w))
                a = int(rng.integers(h))
                mask[a:int(rng.integers(a + 1, h + 1)), c] = True
    elif mode == 4:
        a, b = int(rng.integers(h)), int(rng.integers(w))
        d, e = int(rng.integers(a + 1, h + 1)), int(rng.integers(b + 1, w + 1))
        mask[a:d, b] = mask[a:d, e - 1] = True
        mask[a, b:e] = mask[d - 1, b:e] = True
    elif mode == 5:
        for _ in range(int(rng.integers(2, 6))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            mask[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
    elif mode == 6:
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
    else:
        mask[:] = True
        mask[int(rng.integers(h)), int(rng.integers(w))] = False
    return mask


def _base_masks(rng, h, w):
    masks = np.stack([_primitive(rng, h, w) for _ in range(4)])
    relation = int(rng.integers(0, 8))
    if relation:
        source, target = rng.choice(4, size=2, replace=False)
        source, target = int(source), int(target)
        src = masks[source]
        if relation == 1:  # shifted
            masks[target] = np.roll(src, (int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))), axis=(0, 1))
        elif relation == 2:  # nested
            masks[target] = src & (rng.random((h, w)) < rng.uniform(0.35, 0.85))
        elif relation == 3:  # complementary
            masks[target] = ~src
        elif relation == 4:  # partly identical
            masks[target] = src.copy()
            for i in rng.choice(h * w, size=max(1, (h * w) // 9), replace=False):
                masks[target].flat[int(i)] ^= True
        elif relation == 5:  # nearly disjoint
            masks[target] &= ~src
        elif relation == 6:  # deep occlusion
            masks[target] = src | (rng.random((h, w)) < 0.04)
        else:  # narrow exposed sliver
            masks[target] = src.copy()
            masks[target].flat[int(rng.integers(h * w))] = True
    for layer in range(4):
        count = int(masks[layer].sum())
        if count == 0:
            masks[layer].flat[int(rng.integers(h * w))] = True
        elif count == h * w:
            masks[layer].flat[int(rng.integers(h * w))] = False
    return masks


def _place(rng, masks, patterns):
    cells = rng.choice(masks.shape[1] * masks.shape[2], size=len(patterns), replace=False)
    for cell, pattern in zip(cells, patterns):
        masks[:, int(cell) // masks.shape[2], int(cell) % masks.shape[2]] = pattern
    return set(int(c) for c in cells)


def _repair_case(rng, masks, reserved, background_visible, lowest_visible, hidden):
    h, w = masks.shape[1:]
    area = h * w
    free = [i for i in range(area) if i not in reserved]
    if hidden is not None:
        for r, c in zip(*np.nonzero(masks[hidden])):
            masks[int(rng.integers(hidden)), r, c] = True
        cell = int(rng.choice(free))
        r, c = divmod(cell, w)
        masks[hidden, r, c] = True
        masks[int(rng.integers(hidden)), r, c] = True
    if not lowest_visible:
        for r, c in zip(*np.nonzero(masks[3])):
            if not np.any(masks[:3, r, c]):
                choices = [i for i in range(3) if i != hidden]
                masks[int(rng.choice(choices)), r, c] = True
    else:
        if not np.any(masks[3] & ~np.any(masks[:3], axis=0)):
            cell = int(rng.choice(free))
            r, c = divmod(cell, w)
            masks[:, r, c] = (False, False, False, True)
    if not background_visible:
        for r, c in zip(*np.nonzero(~np.any(masks, axis=0))):
            choices = [i for i in range(4) if i != hidden and (lowest_visible or i != 3)]
            masks[int(rng.choice(choices)), r, c] = True
    elif not np.any(~np.any(masks, axis=0)):
        cell = int(rng.choice(free))
        masks[:, cell // w, cell % w] = False
    # A related or extremely sparse construction can leave an empty panel.
    for layer in range(4):
        if not np.any(masks[layer]):
            cell = int(rng.choice(free))
            r, c = divmod(cell, w)
            masks[layer, r, c] = True
            if layer == 3 and not lowest_visible:
                masks[int(rng.integers(3)), r, c] = True
            if layer == hidden:
                masks[int(rng.integers(layer)), r, c] = True
        if np.all(masks[layer]):
            cell = int(rng.choice(free))
            masks[layer, cell // w, cell % w] = False
    return (
        np.any(~np.any(masks, axis=0)) == background_visible
        and np.any(masks[3] & ~np.any(masks[:3], axis=0)) == lowest_visible
        and (hidden is None or np.all(~masks[hidden] | np.any(masks[:hidden], axis=0)))
        and all(0 < int(m.sum()) < area for m in masks)
    )


def _compose(masks, background, payloads):
    out = np.full(masks.shape[1:], background, dtype=np.int8)
    for layer in range(3, -1, -1):
        out[masks[layer]] = payloads[layer]
    return out


def _assemble(masks, palette):
    h, w = masks.shape[1:]
    background, divider, *payloads = palette
    inp = np.full((2 * h + 1, 2 * w + 1), background, dtype=np.int8)
    inp[h, :] = divider
    inp[:, w] = divider
    panels = (inp[:h, :w], inp[:h, w + 1:], inp[h + 1:, :w], inp[h + 1:, w + 1:])
    for layer, panel in enumerate(panels):
        panel[masks[layer]] = payloads[layer]
    return inp


def _rows(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _key(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    base_palette = tuple(int(v) for v in rng.choice(10, size=6, replace=False))
    recolor = bool(rng.integers(2))
    comparison_pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    rng.shuffle(comparison_pairs)
    pair_seeds = rng.integers(0, 2**63, size=nt + ns, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, pair_seed in enumerate(pair_seeds):
        prng = np.random.default_rng(int(pair_seed))
        h, w = _shape(prng)
        if recolor and index:
            # A deterministic role cycle guarantees reused digits in changed roles.
            palette = tuple(base_palette[(i + index) % 6] for i in range(6))
        else:
            palette = base_palette
        if index == 0:
            bg_visible, low_visible = True, True
        else:
            bg_visible = bool(prng.integers(2))
            low_visible = bool(prng.integers(2))
        hidden = None
        if index >= nt and prng.random() < 0.48:
            hidden = int(prng.integers(1, 4))
            if hidden == 3:
                low_visible = False
        patterns = []
        if index == 0:
            patterns = [(False,) * 4]
            patterns += [tuple(i == layer for i in range(4)) for layer in range(4)]
            pairs = comparison_pairs[:3]
        elif index == 1:
            pairs = comparison_pairs[3:]
        else:
            pairs = []
        patterns += [tuple(i in pair for i in range(4)) for pair in pairs]
        for _ in range(24):
            masks = _base_masks(prng, h, w)
            reserved = _place(prng, masks, patterns)
            if not _repair_case(prng, masks, reserved, bg_visible, low_visible, hidden):
                continue
            inp = _assemble(masks, palette)
            out = _compose(masks, palette[0], palette[2:])
            if _key(inp) in seen or (_rows(inp), _rows(out)) in _SOURCE:
                continue
            break
        else:
            raise ValueError("could not construct a valid distinct pair")
        seen.add(_key(inp))
        puzzle["train" if index < nt else "test"].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _decode(pair):
    inp, out = pair["input"], pair["output"]
    ih, iw = inp.shape
    assert ih % 2 == iw % 2 == 1
    h, w = ih // 2, iw // 2
    assert 4 <= h <= 14 and 4 <= w <= 14
    divider = int(inp[h, w])
    assert np.all(inp[h, :] == divider) and np.all(inp[:, w] == divider)
    panels = (inp[:h, :w], inp[:h, w + 1:], inp[h + 1:, :w], inp[h + 1:, w + 1:])
    sets = [set(int(v) for v in np.unique(panel)) for panel in panels]
    assert all(len(s) == 2 for s in sets)
    common = set.intersection(*sets)
    assert len(common) == 1
    bg = next(iter(common))
    payloads = tuple(next(iter(s - {bg})) for s in sets)
    assert len({bg, divider, *payloads}) == 6
    assert out.shape == (h, w)
    masks = np.stack([p == color for p, color in zip(panels, payloads)])
    return masks, bg, divider, payloads


def _orders_supported_by_training(training):
    supported = []
    for order in permutations(range(4)):
        agrees = True
        for pair in training:
            masks, bg, _, payloads = _decode(pair)
            predicted = np.full(masks.shape[1:], bg, dtype=np.int8)
            for layer in reversed(order):
                predicted[masks[layer]] = payloads[layer]
            if not np.array_equal(predicted, pair["output"]):
                agrees = False
                break
        if agrees:
            supported.append(order)
    return supported


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    patterns = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert (_rows(inp), _rows(out)) not in _SOURCE
            masks, bg, divider, payloads = _decode(pair)
            assert np.array_equal(out, _compose(masks, bg, payloads))
            if split == "train":
                patterns.update(tuple(bool(v) for v in bits)
                                for bits in masks.transpose(1, 2, 0).reshape(-1, 4))
    assert (False,) * 4 in patterns
    for layer in range(4):
        assert tuple(i == layer for i in range(4)) in patterns
    for i in range(4):
        for j in range(i + 1, 4):
            assert tuple(k in (i, j) for k in range(4)) in patterns
    assert _orders_supported_by_training(puzzle["train"]) == [(0, 1, 2, 3)]


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes = set()
    palettes = set()
    outcomes = set()
    train_changes = 0
    color_changes = 0
    structures = set()
    one_cell = 0
    one_hole = 0
    four_way = 0
    middle_hidden = 0
    near_disjoint = 0
    near_identical = 0
    for puzzle in puzzles:
        pair_shapes = []
        pair_palettes = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                masks, bg, divider, payloads = _decode(pair)
                pair_shapes.append(tuple(masks.shape[1:]))
                pair_palettes.append((bg, divider, *payloads))
                shapes.add(pair_shapes[-1])
                palettes.add(pair_palettes[-1])
                outcomes.add((bool(np.any(pair["output"] == bg)),
                              bool(np.any(pair["output"] == payloads[3]))))
                structures.add((masks.shape, masks.tobytes()))
                area = masks.shape[1] * masks.shape[2]
                one_cell += sum(int(m.sum()) == 1 for m in masks)
                one_hole += sum(int(m.sum()) == area - 1 for m in masks)
                four_way += bool(np.any(np.all(masks, axis=0)))
                middle_hidden += any(
                    np.any(masks[layer]) and not np.any(pair["output"] == payloads[layer])
                    for layer in (1, 2)
                )
                for first in range(4):
                    for second in range(first + 1, 4):
                        overlap = int(np.sum(masks[first] & masks[second]))
                        union = int(np.sum(masks[first] | masks[second]))
                        near_disjoint += overlap == 0
                        near_identical += overlap / union >= 0.8
        train_changes += len(set(pair_shapes)) > 1
        color_changes += len(set(pair_palettes)) > 1
    assert len(shapes) >= 25 and len(palettes) >= 30
    assert len(structures) >= 200
    assert train_changes >= 25 and color_changes >= 10
    assert len(outcomes) == 4
    assert one_cell >= 20 and one_hole >= 20
    assert four_way >= 15 and middle_hidden >= 10
    assert near_disjoint >= 20 and near_identical >= 20
    a = generate_puzzle(314159, 7, 3)
    b = generate_puzzle(314159, 7, 3)
    assert all(np.array_equal(x[k], y[k])
               for split in ("train", "test")
               for x, y in zip(a[split], b[split])
               for k in ("input", "output"))
    for seed, nt, ns in [(100, 2, 1), (101, 10, 1), (102, 10, 4),
                         (103, None, 1), (104, None, 4), (105, 2, None),
                         (106, 10, None)]:
        p = generate_puzzle(seed, nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for nt, ns in [(1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4), (True, 1), (3.0, 1)]:
        try:
            generate_puzzle(999, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal pair count accepted")
