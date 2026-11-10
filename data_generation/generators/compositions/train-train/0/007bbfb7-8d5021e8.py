"""Damaged mirror quilt: self-similar erasure repair with reflected tiles.

Only Python's standard library and NumPy are used. No parent module is imported.
The solver treats visible pixels as constraints, not as generation metadata.
"""

from collections import Counter
from math import isqrt

import numpy as np


ERASE = 5
SHAPES = tuple((h, w) for h in range(2, 6) for w in range(2, 6)
               if (h, w) != (2, 2))


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value, label="grid"):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= x <= 30 for x in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError(f"{label} must be an integer ARC grid")
    return array.astype(np.int8, copy=False)


def _key(array):
    return array.shape, array.tobytes()


def _address_maps(h, w, vertical=1, horizontal=1):
    """For each quilt pixel, return its coarse and unfolded fine motif indices."""
    rows, cols = np.indices((h * h, w * w))
    br, bc = rows // h, cols // w
    fr, fc = rows % h, cols % w
    if vertical:
        fr = np.where(br % 2 == (0 if vertical == 1 else 1), h - 1 - fr, fr)
    if horizontal:
        fc = np.where(bc % 2 == (0 if horizontal == 1 else 1), w - 1 - fc, fc)
    return br * w + bc, fr * w + fc


def _render(mask, background, ink, vertical=1, horizontal=1):
    h, w = mask.shape
    a, b = _address_maps(h, w, vertical, horizontal)
    flat = mask.ravel()
    core = np.where(flat[a] & flat[b], ink, background).astype(np.int8)
    result = np.full((h * h + 2, w * w + 2), background, dtype=np.int8)
    result[1:-1, 1:-1] = core
    return result


def _parse(input_grid):
    grid = _grid(input_grid, "input")
    H, W = grid.shape
    h, w = isqrt(max(0, H - 2)), isqrt(max(0, W - 2))
    if (h, w) not in SHAPES or h * h != H - 2 or w * w != W - 2:
        raise ValueError("interior dimensions must be h squared by w squared, h,w in 2..5")
    background = int(grid[0, 0])
    if background == ERASE or not (np.all(grid[0] == background)
            and np.all(grid[-1] == background) and np.all(grid[:, 0] == background)
            and np.all(grid[:, -1] == background)):
        raise ValueError("the intact outer frame must identify background")
    colors = set(map(int, np.unique(grid))) - {background, ERASE}
    if len(colors) != 1:
        raise ValueError("input must contain exactly one visible ink color")
    ink = colors.pop()
    return grid, h, w, background, ink


def _recover_mask(grid, h, w, background, ink, vertical=1, horizontal=1):
    """Unique solution of observed products M[a] AND M[b].

    Positive observations force both endpoints true. A zero on a self-edge,
    or incident to a forced-true endpoint, forces the other endpoint false.
    Every remaining variable could independently be set true in the all-false
    remainder, so any remainder proves ambiguity rather than permission to guess.
    """
    core = grid[1:-1, 1:-1]
    a, b = _address_maps(h, w, vertical, horizontal)
    positive, negative = core == ink, core == background
    true = np.zeros(h * w, dtype=bool)
    true[a[positive]] = True
    true[b[positive]] = True
    false = np.zeros(h * w, dtype=bool)
    na, nb = a[negative], b[negative]
    false[na[na == nb]] = True
    false[nb[true[na]]] = True
    false[na[true[nb]]] = True
    if np.any(true & false):
        raise ValueError("surviving pixels contradict the mirror-quilt rule")
    if not np.all(true | false):
        raise ValueError("erased quilt does not uniquely determine its motif")
    mask = true.reshape(h, w)
    if not 1 <= int(mask.sum()) < h * w:
        raise ValueError("motif must contain occupied and empty cells")
    return mask


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, h, w, background, ink = _parse(input_grid)
    mask = _recover_mask(grid, h, w, background, ink)
    answer = _render(mask, background, ink)
    visible = grid != ERASE
    if not np.array_equal(answer[visible], grid[visible]):
        raise ValueError("reconstruction changed surviving evidence")
    return answer


def _asymmetric(mask):
    variants = [np.rot90(mask, k) for k in range(4)] + [
        np.rot90(np.fliplr(mask), k) for k in range(4)]
    return all(not np.array_equal(mask, other) for other in variants[1:])


def _parities(mask, excluded=()):
    return {(r % 2, c % 2) for r, c in zip(*np.nonzero(mask))
            if r * mask.shape[1] + c not in excluded}


def _sample_mask(rng, shape, minimum, anchor=False):
    h, w = shape
    n = h * w
    for _ in range(300):
        k = int(rng.integers(max(minimum, 5 if anchor else minimum), n - 1))
        mode = int(rng.integers(6))
        if mode == 0:  # A growing connected polyomino, at many different densities.
            mask = np.zeros(shape, dtype=bool)
            start = int(rng.integers(n))
            mask.flat[start] = True
            while int(mask.sum()) < k:
                boundary = []
                for r, c in zip(*np.nonzero(mask)):
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                            boundary.append((nr, nc))
                r, c = boundary[int(rng.integers(len(boundary)))]
                mask[r, c] = True
        elif mode in (1, 2) and not anchor:
            # Genuine axis-symmetric motifs coexist with diagnostic asymmetric ones.
            mask = rng.random(shape) < float(rng.uniform(0.25, 0.8))
            if mode == 1:
                for r in range(h // 2):
                    mask[-1-r] = mask[r]
            else:
                for c in range(w // 2):
                    mask[:, -1-c] = mask[:, c]
        elif mode == 3:
            # Biased random arrangements give bands, holes, and corner concentrations.
            row_bias = rng.random((h, 1))
            col_bias = rng.random((1, w))
            scores = row_bias + col_bias + rng.random(shape) * 0.6
            mask = np.zeros(shape, dtype=bool)
            mask.flat[np.argsort(scores.ravel())[-k:]] = True
        else:
            mask = np.zeros(shape, dtype=bool)
            mask.flat[rng.choice(n, size=k, replace=False)] = True
        count = int(mask.sum())
        if not minimum <= count <= n - 2:
            continue
        if anchor and (not _asymmetric(mask) or len(_parities(mask)) != 4):
            continue
        # One whole occupied block will be hidden in training. Keep all four
        # mirror classes visible in the diagnostic anchor after that deletion.
        if anchor and not any(len(_parities(mask, (int(i),))) == 4
                              for i in np.flatnonzero(mask)):
            continue
        return mask
    raise RuntimeError("bounded motif sampling failed")


def _choose(rng, values):
    return int(values[int(rng.integers(len(values)))])


def _damage(rng, mask, background, ink, hard=False, blind_background=False, anchor=False):
    h, w = mask.shape
    n = h * w
    flat = mask.ravel()
    active = list(map(int, np.flatnonzero(flat)))
    empty = list(map(int, np.flatnonzero(~flat)))
    # Keep two occupied blocks available for complementary fragments.
    missing_active = min(len(active) - 2, int(rng.integers(2, 4)) if hard else 1)
    if anchor:
        candidates = [i for i in active if len(_parities(mask, (i,))) == 4]
        hidden_active = [_choose(rng, candidates)]
    else:
        hidden_active = list(map(int, rng.choice(active, size=missing_active, replace=False)))
    missing_empty = min(len(empty) - 1, int(rng.integers(1, 3)) if hard else 1)
    hidden_empty = list(map(int, rng.choice(empty, size=missing_empty, replace=False)))
    survivors = [i for i in active if i not in hidden_active]
    visible_empty = [i for i in empty if i not in hidden_empty]
    coarse_only_ink = _choose(rng, survivors)
    blind = {coarse_only_ink}
    if blind_background:
        blind.add(_choose(rng, visible_empty))

    # These are pixel erasures, not a semantic mask supplied to solve().
    probability = float(rng.uniform(0.25, 0.43) if not hard else rng.uniform(0.48, 0.68))
    erased = rng.random((h * h, w * w)) < probability
    for _ in range(int(rng.integers(1, 4 if hard else 3))):
        top, left = int(rng.integers(h * h)), int(rng.integers(w * w))
        rh = int(rng.integers(1, h + 2))
        rw = int(rng.integers(1, w + 2))
        erased[top:min(top + rh, h*h), left:min(left + rw, w*w)] = True
    a, b = _address_maps(h, w)
    for index in hidden_active + hidden_empty:
        erased[a == index] = True
    erased[np.isin(b, list(blind))] = True

    # Reserve lost fine cells in every surviving active block. No such tile can
    # reveal the complete motif; the remaining witnesses cover it jointly.
    reserved = {}
    nonblind_active = [i for i in active if i not in blind]
    nonblind_empty = [i for i in empty if i not in blind]
    for position, index in enumerate(survivors):
        reserved[index] = {nonblind_active[position % len(nonblind_active)],
                           nonblind_empty[position % len(nonblind_empty)]}
    # Cycling lets each reserved active coordinate be seen in another tile.
    # One remaining nonblind background coordinate needs a relaxed reservation.
    if set.intersection(*(reserved[i] for i in survivors)):
        # With only one nonblind empty coordinate, allow it to be observed.
        # The globally blind coordinate still prevents any complete tile.
        for index in survivors:
            reserved[index] &= set(active)
    for index in survivors:
        for fine in reserved[index]:
            erased[(a == index) & (b == fine)] = True

    def reveal(coarse, fine):
        erased[(a == coarse) & (b == fine)] = False

    # Positive macro evidence for every surviving occupied block, and at least
    # one visible negative fine cell in it. Never reveal a reserved/blind cell.
    for index in survivors:
        reveal(index, _choose(rng, [j for j in active if j not in blind and j not in reserved[index]]))
        reveal(index, _choose(rng, [j for j in empty if j not in blind and j not in reserved[index]]))
    # All nonblind fine variables are visible somewhere in an occupied block.
    for fine in range(n):
        if fine not in blind:
            eligible = [index for index in survivors if fine not in reserved[index]]
            reveal(_choose(rng, eligible), fine)
    # Each nonhidden empty block has a coarse negative witness against known
    # ink. In particular this proves the deliberately fine-blind empty cell.
    # It also prevents random damage from creating extra whole empty erasures.
    for index in visible_empty:
        reveal(index, _choose(rng, nonblind_active))

    output = _render(mask, background, ink)
    source = output.copy()
    source[1:-1, 1:-1][erased] = ERASE
    return {"input": source, "output": output}


def _plans(rng, mixed, num_train, num_test):
    first = (3, 3)
    if mixed:
        rectangles = ((2, 3), (3, 2), (3, 4), (4, 3))
        second = rectangles[int(rng.integers(len(rectangles)))]
        train = [first, second] + [SHAPES[int(rng.integers(len(SHAPES)))]
                                   for _ in range(num_train - 2)]
        # Tests may have a held-out size, after both axes and variable block size
        # have been demonstrated. They always also have harder block erasure.
        test = [SHAPES[int(rng.integers(len(SHAPES)))] for _ in range(num_test)]
    else:
        train, test = [first] * num_train, [first] * num_test
    return train, test


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    num_train = _integer(num_train, "num_train")
    num_test = _integer(num_test, "num_test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    background = _choose(rng, [i for i in range(10) if i != ERASE])
    colors = [i for i in range(10) if i not in (background, ERASE)]
    rng.shuffle(colors)
    palette_size = int(rng.integers(1, min(4, num_train) + 1))
    palette = colors[:palette_size]
    mixed = bool(rng.random() < 0.7)
    shapes = _plans(rng, mixed, num_train, num_test)
    puzzle = {"train": [], "test": []}
    used = set()
    first_train_mask = None
    for split, shape_plan in zip(("train", "test"), shapes):
        for position, shape in enumerate(shape_plan):
            anchor = split == "train" and position == 0
            hard = split == "test" or (position >= 2 and rng.random() < 0.25)
            blind_background = position == 1 or split == "test" or bool(rng.random() < 0.35)
            ink = (palette[position % palette_size] if split == "train" else
                   _choose(rng, colors if rng.random() < 0.35 else palette))
            for _ in range(100):
                mask = _sample_mask(rng, shape, minimum=4 if hard else 3, anchor=anchor)
                if split == "train" and position == 1 and first_train_mask is not None:
                    if np.array_equal(mask, first_train_mask):
                        continue
                pair = _damage(rng, mask, background, ink, hard, blind_background, anchor)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    if anchor:
                        first_train_mask = mask.copy()
                    break
            else:
                raise RuntimeError("could not sample a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    inp, h, w, background, ink = _parse(pair["input"])
    out = _grid(pair["output"], "output")
    _check(out.shape == inp.shape, "restoration must preserve dimensions")
    _check(set(map(int, np.unique(out))) == {background, ink}, "output palette")
    _check(np.any(inp == ERASE), "missing erasure markers")
    visible = inp != ERASE
    _check(np.array_equal(inp[visible], out[visible]), "surviving pixels changed")
    _check(np.all(out[0] == background) and np.all(out[-1] == background)
           and np.all(out[:, 0] == background) and np.all(out[:, -1] == background), "frame changed")
    core, target = inp[1:-1, 1:-1], out[1:-1, 1:-1]
    # Recover the motif from the OUTPUT's coarse occupancy, independent of the
    # input constraint solver; check every reflected tile directly with slices.
    mask = np.zeros((h, w), dtype=bool)
    hidden_active = hidden_empty = complete_active = 0
    macro_seen, fine_seen = set(), set()
    for r in range(h):
        for c in range(w):
            tile = target[r*h:(r+1)*h, c*w:(c+1)*w]
            mask[r, c] = np.any(tile == ink)
    k = int(mask.sum())
    _check(3 <= k <= h*w - 2, "motif must leave useful evidence at both scales")
    for r in range(h):
        for c in range(w):
            block = target[r*h:(r+1)*h, c*w:(c+1)*w]
            observed = core[r*h:(r+1)*h, c*w:(c+1)*w]
            if mask[r, c]:
                tile_mask = mask[::-1] if r % 2 == 0 else mask
                tile_mask = tile_mask[:, ::-1] if c % 2 == 0 else tile_mask
                expected = np.where(tile_mask, ink, background)
                hidden_active += bool(np.all(observed == ERASE))
                complete_active += bool(np.all(observed != ERASE))
            else:
                expected = np.full((h, w), background)
                hidden_empty += bool(np.all(observed == ERASE))
            _check(np.array_equal(block, expected), "incorrect self-similar mirror tile")
            if np.any(observed == ink):
                macro_seen.add(r*w + c)
            unfolded = observed[::-1] if r % 2 == 0 else observed
            unfolded = unfolded[:, ::-1] if c % 2 == 0 else unfolded
            fine_seen.update(map(int, np.flatnonzero(unfolded == ink)))
    _check(hidden_active >= 1 and hidden_empty >= 1, "both kinds of whole missing block are required")
    _check(complete_active == 0, "a complete active tile bypasses fragment pooling")
    _check(macro_seen - fine_seen, "missing coarse-only occupied-cell evidence")
    _check(fine_seen - macro_seen, "missing fine-only occupied-cell evidence")
    _check(np.count_nonzero(target == ink) == k*k, "self-substitution foreground count")
    _check(np.array_equal(solve(inp), out), "input-only solver disagrees")
    _, fine_addresses = _address_maps(h, w)
    fine_blind_empty = sum(not np.any((core != ERASE)[fine_addresses == j])
                           for j in np.flatnonzero(~mask.ravel()))
    # Omitting A and filling every block with the recovered tile is wrong.
    all_tiles = np.empty_like(target)
    for r in range(h):
        for c in range(w):
            tile = mask[::-1] if r % 2 == 0 else mask
            tile = tile[:, ::-1] if c % 2 == 0 else tile
            all_tiles[r*h:(r+1)*h, c*w:(c+1)*w] = np.where(tile, ink, background)
    _check(not np.array_equal(target, all_tiles), "omitting A did not change the answer")
    return {"mask": mask, "background": background, "ink": ink,
            "hidden_active": hidden_active, "hidden_empty": hidden_empty,
            "coarse_only": len(macro_seen - fine_seen), "fine_only": len(fine_seen - macro_seen),
            "visible_orientations": len({((j // w) % 2, (j % w) % 2) for j in macro_seen}),
            "fine_blind_empty": fine_blind_empty,
            "erased": int(np.count_nonzero(inp == ERASE))}


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid pair counts")
    seen, backgrounds = set(), set()
    data = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(_grid(pair["input"]))
            _check(key not in seen, "duplicate input")
            seen.add(key)
            backgrounds.add(info["background"])
            data[split].append(info)
    _check(len(backgrounds) == 1, "background must be consistent within a puzzle")
    _check(any(_asymmetric(info["mask"]) and info["visible_orientations"] == 4
               for info in data["train"]), "training must expose all four mirror orientations")
    _check(not np.array_equal(data["train"][0]["mask"], data["train"][1]["mask"]),
           "first two demonstrations must use different motifs")
    train_shapes = {info["mask"].shape for info in data["train"]}
    test_shapes = {info["mask"].shape for info in data["test"]}
    if not test_shapes <= train_shapes:
        _check(len(train_shapes) >= 2 and any(h != w for h, w in train_shapes),
               "new test sizes need multiple sizes and a rectangle in training")
    _check(all(info["hidden_active"] >= 2 for info in data["test"]), "test must hide more whole occupied blocks")
    _check(all(info["hidden_active"] == 1 for info in data["train"][:2]), "first demonstrations need the simpler damage regime")
    _check(all(info["hidden_empty"] == 1 for info in data["train"][:2]),
           "first demonstrations must hide exactly one empty block")
    _check(data["train"][1]["fine_blind_empty"] >= 1 and
           all(info["fine_blind_empty"] >= 1 for info in data["test"]),
           "negative cross-scale recovery needs training coverage")
    # Full training answers distinguish no reflection, the two single-axis rules,
    # and either reversed phase. They cannot all coincide on symmetric motifs.
    survivors = []
    for vertical in range(3):
        for horizontal in range(3):
            if all(np.array_equal(_render(info["mask"], info["background"], info["ink"], vertical, horizontal),
                                  np.asarray(pair["output"]))
                   for pair, info in zip(puzzle["train"], data["train"])):
                survivors.append((vertical, horizontal))
    _check(survivors == [(1, 1)], "training does not distinguish reflection conventions")
    return None


def _signature(puzzle):
    return tuple((split, tuple((_key(pair["input"]), _key(pair["output"]))
                              for pair in puzzle[split])) for split in ("train", "test"))


def _components(mask):
    unseen = set(zip(*np.nonzero(mask)))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            r, c = stack.pop()
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in unseen:
                    unseen.remove(point)
                    stack.append(point)
    return count


def self_test():
    """Bounded structural, evidence, variation, determinism, and API checks."""
    metrics = Counter()
    signatures, masks, shapes, backgrounds, inks = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        _check(_signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterministic generation")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                mask = info["mask"]
                masks.add(_key(mask))
                shapes.add(mask.shape)
                backgrounds.add(info["background"])
                inks.add(info["ink"])
                metrics["pairs"] += 1
                metrics["connected" if _components(mask) == 1 else "disconnected"] += 1
                metrics["axis_symmetric"] += bool(np.array_equal(mask, mask[::-1]) or np.array_equal(mask, mask[:, ::-1]))
                metrics["zero_ink"] += info["ink"] == 0
                metrics["square" if mask.shape[0] == mask.shape[1] else "rectangle"] += 1
                metrics["hidden_active_blocks"] += info["hidden_active"]
                metrics["hidden_empty_blocks"] += info["hidden_empty"]
                metrics["coarse_only_positive_bits"] += info["coarse_only"]
                metrics["fine_only_positive_bits"] += info["fine_only"]
                train_shapes = {_parse(p["input"])[1:3] for p in puzzle["train"]}
                metrics["heldout_test_size"] += split == "test" and mask.shape not in train_shapes
                # B ablation is necessary somewhere in every puzzle, including
                # every diagnostic training anchor, not just in aggregate.
                if split == "train" and pair is puzzle["train"][0]:
                    _check(not np.array_equal(pair["output"], _render(mask, info["background"], info["ink"], 0, 0)),
                           "omitting B did not change the diagnostic answer")
    _check(len(signatures) == 42, "repeated puzzles across seeds")
    _check(len(masks) >= 120 and len(shapes) >= 12, "insufficient geometric variation")
    _check(len(backgrounds) >= 7 and len(inks) >= 7 and metrics["zero_ink"] > 0, "palette variation missing")
    for name in ("connected", "disconnected", "axis_symmetric", "square", "rectangle", "heldout_test_size"):
        _check(metrics[name] > 0, f"missing variation: {name}")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1234, a, b)
            _check((len(p["train"]), len(p["test"])) == (a, b), "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count request accepted: {a}, {b}")
    # A legal-looking but information-free erasure must not be guessed.
    ambiguous = generate_puzzle(0)["train"][0]["input"].copy()
    full = solve(ambiguous)
    background = int(ambiguous[0, 0])
    positive = tuple(np.argwhere((full != background) & (full != ERASE))[0])
    ambiguous[1:-1, 1:-1] = ERASE
    ambiguous[positive] = full[positive]
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous input was guessed")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "distinct_motifs": len(masks),
            "motif_shapes": len(shapes), "backgrounds": len(backgrounds), "ink_colors": len(inks),
            **dict(metrics)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
