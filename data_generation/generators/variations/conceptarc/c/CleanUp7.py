"""Repeated-motif repair puzzles with independently varied occlusions."""
import numpy as np


_SOURCE_INPUTS = {
    ("000000000", "333000000", "030033300", "030088800", "000088800", "000088833", "000000030", "033300030", "003000000"),
    ("404000000", "444040400", "404044400", "000088888", "000088888", "404088888", "444000444", "404000404", "000000000"),
    ("060000060", "666000666", "060060888", "000666888", "000060888", "000006888", "006066888", "066606888", "006000000"),
    ("099900000", "090900000", "090900000", "000888899", "000888809", "009888809", "009090000", "009090000", "000000000"),
    ("000000000", "000006600", "066066000", "660088866", "060088860", "000088860", "000660000", "000060000"),
    ("888000888", "888000888", "010100101", "001000010", "000000000", "000100010", "001010101", "000100010", "000000000"),
}
_SIZES = ((2, 3), (3, 3), (3, 4), (4, 4))
_CANVASES = ((8, 9), (9, 8), (9, 9), (9, 12), (12, 9),
             (11, 11), (12, 14), (14, 12), (15, 16), (16, 15))


def _components(points, diagonal=True):
    left = set(points)
    answer = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while left:
        root = left.pop()
        part = {root}
        stack = [root]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    part.add(q)
                    stack.append(q)
        answer.append(part)
    return answer


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _motif(rng, size, diagonal):
    h, w = size
    for _ in range(300):
        density = float(rng.uniform(0.37, 0.68))
        motif = rng.random((h, w)) < density
        if not (motif[0].any() and motif[-1].any() and
                motif[:, 0].any() and motif[:, -1].any()):
            continue
        cells = {tuple(x) for x in np.argwhere(motif)}
        if not (max(4, h + w - 2) <= len(cells) <= h * w - 2):
            continue
        if len(_components(cells)) != 1:
            continue
        if (len(_components(cells, False)) > 1) != diagonal:
            continue
        return motif
    raise RuntimeError("motif sampling failed")


def _copy(motif, origin, shape):
    h, w = shape
    r0, c0 = origin
    return {(r0 + r, c0 + c) for r, c in np.argwhere(motif)
            if 0 <= r0 + r < h and 0 <= c0 + c < w}


def _footprint(origin, size, shape):
    h, w = shape
    mh, mw = size
    r0, c0 = origin
    return {(r, c) for r in range(max(0, r0), min(h, r0 + mh))
            for c in range(max(0, c0), min(w, c0 + mw))}


def _near(a, b):
    return any((r + dr, c + dc) in b for r, c in a
               for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _recover(grid):
    """Infer the motif from visible repeated components, then exact-cover visible ink.

    No output, seed, or construction metadata is used.  None means ambiguous.
    """
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3:
        return None
    bg = int(colors[np.argmax(counts)])
    other = [int(c) for c in colors if c != bg]
    answers = []
    for fg in other:
        mask = next(c for c in other if c != fg)
        visible = {tuple(x) for x in np.argwhere(grid == fg)}
        obscured = {tuple(x) for x in np.argwhere(grid == mask)}
        comps = _components(visible)
        groups = {}
        for comp in comps:
            if len(comp) < 4:
                continue
            signature = _normalize(comp)
            hh = max(r for r, _ in signature) + 1
            ww = max(c for _, c in signature) + 1
            if (hh, ww) in _SIZES:
                groups[signature] = groups.get(signature, 0) + 1
        for signature, multiplicity in groups.items():
            if multiplicity < 2:
                continue
            mh = max(r for r, _ in signature) + 1
            mw = max(c for _, c in signature) + 1
            motif = np.zeros((mh, mw), dtype=bool)
            for r, c in signature:
                motif[r, c] = True
            origins = {(r - mr, c - mc) for r, c in visible for mr, mc in signature}
            options = []
            h, w = grid.shape
            for origin in origins:
                full = _copy(motif, origin, grid.shape)
                if len(full) < 2:
                    continue
                footprint = _footprint(origin, (mh, mw), grid.shape)
                seen = full - obscured
                if len(seen) < 2 or not seen <= visible:
                    continue
                if (footprint & visible) != seen:
                    continue
                options.append((seen, full, full & obscured))
            by_cell = {cell: [] for cell in visible}
            for option in options:
                for cell in option[0]:
                    by_cell[cell].append(option)
            if any(not value for value in by_cell.values()):
                continue
            outcomes = set()
            visits = [0]
            def search(remaining, occupied, hidden, n):
                visits[0] += 1
                if visits[0] > 3500 or len(outcomes) > 1:
                    return
                if not remaining:
                    if 2 <= n <= 7:
                        outcomes.add(frozenset(hidden))
                    return
                if n >= 7:
                    return
                point = min(remaining, key=lambda x: len(by_cell[x]))
                for seen, full, masked in by_cell[point]:
                    if seen <= remaining and not full & occupied:
                        search(remaining - seen, occupied | full, hidden | masked, n + 1)
            search(visible, set(), set(), 0)
            if visits[0] <= 3500:
                answers.extend((bg, fg, mask, signature, hidden) for hidden in outcomes)
    unique = {(bg, fg, mask, hidden) for bg, fg, mask, _, hidden in answers}
    if len(unique) != 1:
        return None
    bg, fg, mask, hidden = next(iter(unique))
    output = grid.copy()
    output[grid == mask] = bg
    for r, c in hidden:
        output[r, c] = fg
    signatures = {a[3] for a in answers if (a[0], a[1], a[2], a[4]) == (bg, fg, mask, hidden)}
    return output, bg, fg, mask, next(iter(signatures))


def _positions(rng, motif, shape, count, want_close, want_clip):
    h, w = shape
    mh, mw = motif.shape
    full = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    clipped = [(r, c) for r in range(-mh + 1, h)
               for c in range(-mw + 1, w)
               if (r, c) not in full and len(_copy(motif, (r, c), shape)) >= 3]
    cells_at = {p: _copy(motif, p, shape) for p in full + clipped}
    for _ in range(18):
        chosen = []
        chosen_cells = []
        for slot in range(count):
            pool = clipped if slot == count - 1 and want_clip and clipped else full
            if want_close and slot == 3 and len(chosen) >= 3:
                target = chosen[2]
                near = [p for p in pool if abs(p[0]-target[0]) <= mh+1 and
                        abs(p[1]-target[1]) <= mw+1]
                if near:
                    pool = near
            order = rng.permutation(len(pool))
            placed = False
            for ix in order[:min(65, len(order))]:
                p = pool[int(ix)]
                cells = cells_at[p]
                if not cells or any(cells & prior for prior in chosen_cells):
                    continue
                if slot >= 2 and any(_near(cells, prior) for prior in chosen_cells[:2]):
                    continue
                if slot == 1 and _near(cells, chosen_cells[0]):
                    continue
                chosen.append(p)
                chosen_cells.append(cells)
                placed = True
                break
            if not placed:
                break
        if len(chosen) == count:
            return chosen
    return None


def _rect(r0, c0, rh, rw):
    return {(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)}


def _patch_candidates(rng, shape, copies, anchors, large):
    h, w = shape
    sizes = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3),
             (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4), (6, 3), (3, 6)]
    if large:
        broad, compact = sizes[5:], sizes[:5]
        rng.shuffle(broad)
        rng.shuffle(compact)
        sizes = broad + compact
    else:
        rng.shuffle(sizes)
    targets = set().union(*copies[2:])
    starts = list(targets)
    rng.shuffle(starts)
    for source in starts[:25]:
        for rh, rw in sizes:
            if rh > h or rw > w:
                continue
            offsets = [(dr, dc) for dr in range(rh) for dc in range(rw)]
            rng.shuffle(offsets)
            for dr, dc in offsets[:min(7, len(offsets))]:
                r0, c0 = source[0] - dr, source[1] - dc
                if not (0 <= r0 <= h-rh and 0 <= c0 <= w-rw):
                    continue
                patch = _rect(r0, c0, rh, rw)
                if patch & anchors:
                    continue
                touched = sum(bool(patch & cells) for cells in copies[2:])
                if not touched:
                    continue
                if large and len(copies) >= 4 and touched < 2:
                    continue
                yield patch


def _build_pair(rng, regime, bg, mask, fg, motif, shape, mask_count, large, close, clip):
    h, w = shape
    for _ in range(16):
        capacity = max(3, min(6, int(h*w/(motif.size*2.2))))
        count = int(rng.integers(4 if large and capacity >= 4 else 3, capacity + 1))
        positions = _positions(rng, motif, shape, count, close, clip)
        if positions is None:
            continue
        copies = [_copy(motif, p, shape) for p in positions]
        anchors = _footprint(positions[0], motif.shape, shape) | _footprint(positions[1], motif.shape, shape)
        output = np.full(shape, bg, dtype=np.int8)
        for cells in copies:
            for r, c in cells:
                output[r, c] = fg
        patches = []
        for patch in _patch_candidates(rng, shape, copies, anchors, large):
            remaining = [len(cells - patch) for cells in copies[2:]]
            if any(n < 2 for n in remaining):
                continue
            if len(patch & set().union(*copies)) < (3 if large else 1):
                continue
            if not any(output[r, c] == bg for r, c in patch):
                continue
            patches = [patch]
            break
        if not patches:
            continue
        if mask_count == 2:
            for second in _patch_candidates(rng, shape, copies, anchors, False):
                if any(_near(second, p) for p in patches):
                    continue
                if not any(output[r, c] == fg for r, c in second):
                    continue
                if not any(output[r, c] == bg for r, c in second):
                    continue
                union = patches[0] | second
                if any(len(cells - union) < 2 for cells in copies[2:]):
                    continue
                patches.append(second)
                break
            if len(patches) != 2:
                continue
        grid = output.copy()
        for patch in patches:
            for r, c in patch:
                grid[r, c] = mask
        recovered = _recover(grid)
        if recovered is not None and np.array_equal(recovered[0], output):
            return {"input": grid, "output": output}
    raise RuntimeError("could not construct recoverable pair")


def _resolve_counts(rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not lo <= value <= hi):
            raise ValueError("invalid pair count")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("train count must exceed test count")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _resolve_counts(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    used = set()
    # Color-role meaning and the rule are fixed; numeric colors and geometry are local.
    repeat_color = int(rng.integers(10))
    repeat_size = _SIZES[int(rng.integers(len(_SIZES)))]
    diagonal_test = bool(rng.integers(2))
    for split, amount in (("train", nt), ("test", ns)):
        for index in range(amount):
            for attempt in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
                shape = _CANVASES[int(pair_rng.integers(len(_CANVASES)))]
                sizes = [s for s in _SIZES if s[0] <= shape[0]//2 and
                         s[1] <= shape[1]//2 and s[0]*s[1]*3 <= shape[0]*shape[1]*0.42]
                if pair_rng.random() < 0.30 and repeat_size in sizes:
                    size = repeat_size
                else:
                    size = sizes[int(pair_rng.integers(len(sizes)))]
                diag = bool(pair_rng.random() < (0.25 if split == "train" else
                                                  0.65 if diagonal_test else 0.25))
                try:
                    motif = _motif(pair_rng, size, diag)
                except RuntimeError:
                    continue
                colors = list(range(10))
                pair_rng.shuffle(colors)
                bg, mask, fg = colors[:3]
                if index and pair_rng.random() < 0.40:
                    fg = repeat_color if repeat_color not in (bg, mask) else fg
                multiple = bool(pair_rng.random() < (0.20 if split == "train" else 0.40))
                large = bool(pair_rng.random() < (0.28 if split == "train" else 0.45))
                close = bool(pair_rng.random() < 0.55)
                clip = bool(pair_rng.random() < 0.40)
                try:
                    pair = _build_pair(pair_rng, split, bg, mask, fg, motif, shape,
                                       2 if multiple else 1, large, close, clip)
                except RuntimeError:
                    continue
                key = (pair["input"].shape, pair["input"].tobytes())
                if key in used:
                    continue
                used.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("distinct puzzle pair construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    train_diagonal = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert np.issubdtype(array.dtype, np.integer)
                assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
                assert bool(np.all((0 <= array) & (array <= 9)))
            assert grid.shape == output.shape
            key = (grid.shape, grid.tobytes())
            assert key not in seen
            seen.add(key)
            if grid.shape == (9, 9):
                assert tuple("".join(str(int(v)) for v in row) for row in grid) not in _SOURCE_INPUTS
            recovered = _recover(grid)
            assert recovered is not None and np.array_equal(recovered[0], output)
            _, bg, fg, mask, signature = recovered
            assert set(int(c) for c in np.unique(grid)) == {bg, fg, mask}
            assert set(int(c) for c in np.unique(output)) == {bg, fg}
            assert np.array_equal(grid[grid != mask], output[grid != mask])
            assert np.any(output[grid == mask] == bg)
            assert np.any(output[grid == mask] == fg)
            cells = set(signature)
            diagonal = len(_components(cells, False)) > 1
            if split == "train":
                train_diagonal.append(diagonal)
            patches = _components({tuple(x) for x in np.argwhere(grid == mask)}, False)
            assert 1 <= len(patches) <= 2
            for patch in patches:
                rs, cs = [r for r, _ in patch], [c for _, c in patch]
                assert patch == _rect(min(rs), min(cs), max(rs)-min(rs)+1, max(cs)-min(cs)+1)
    assert train_diagonal


def _same_puzzle(a, b):
    return all(len(a[k]) == len(b[k]) and all(
        np.array_equal(x[z], y[z]) for x, y in zip(a[k], b[k]) for z in ("input", "output"))
        for k in ("train", "test"))


def _output_copies(output, fg, signature):
    mh = max(r for r, _ in signature) + 1
    mw = max(c for _, c in signature) + 1
    motif = np.zeros((mh, mw), dtype=bool)
    for r, c in signature:
        motif[r, c] = True
    ink = {tuple(x) for x in np.argwhere(output == fg)}
    options = []
    h, w = output.shape
    for r in range(-mh + 1, h):
        for c in range(-mw + 1, w):
            cells = _copy(motif, (r, c), output.shape)
            if len(cells) >= 2 and cells <= ink:
                options.append(((r, c), cells))
    by_cell = {p: [] for p in ink}
    for option in options:
        for p in option[1]:
            by_cell[p].append(option)

    def search(left, chosen):
        if not left:
            return chosen if 3 <= len(chosen) <= 6 else None
        if len(chosen) >= 6:
            return None
        point = min(left, key=lambda p: len(by_cell[p]))
        for option in by_cell[point]:
            if option[1] <= left:
                answer = search(left - option[1], chosen + [option])
                if answer is not None:
                    return answer
        return None

    return search(ink, [])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _same_puzzle(generate_puzzle(19), generate_puzzle(19))
    assert _same_puzzle(generate_puzzle(19, 6, 3), generate_puzzle(19, 6, 3))
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, motifs, layouts, patch_shapes, colors = set(), set(), set(), set(), set()
    footprint_sizes, diagonal_modes = set(), {"train": set(), "test": set()}
    multi = {"train": 0, "test": 0}
    covered_many = touching = clipping = broad = combined = 0
    varied_roles = repeated_fg = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        roles_here, foregrounds_here = set(), []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                recovered = _recover(grid)
                assert recovered is not None and np.array_equal(recovered[0], pair["output"])
                _, bg, fg, mask, motif = recovered
                shapes.add(grid.shape)
                motifs.add(motif)
                colors.add((bg, fg, mask))
                roles_here.add((bg, mask))
                foregrounds_here.append(fg)
                footprint_sizes.add((max(r for r, _ in motif)+1,
                                     max(c for _, c in motif)+1))
                diagonal_modes[split].add(len(_components(set(motif), False)) > 1)
                layouts.add((grid.shape, tuple(map(tuple, np.argwhere(grid == fg)))))
                masked = {tuple(x) for x in np.argwhere(grid == mask)}
                patches = _components(masked, False)
                multi[split] += len(patches) == 2
                dimensions = tuple(sorted((max(r for r, _ in p)-min(r for r, _ in p)+1,
                                           max(c for _, c in p)-min(c for _, c in p)+1) for p in patches))
                patch_shapes.add(dimensions)
                broad += any(a >= 3 and b >= 3 for a, b in dimensions)
                copies = _output_copies(pair["output"], fg, motif)
                assert copies is not None
                affected = sum(bool(cells & masked) for _, cells in copies)
                covered_many += affected >= 2
                joined = any(_near(a[1], b[1]) for i, a in enumerate(copies)
                             for b in copies[i+1:])
                touching += joined
                clipped = any(r < 0 or c < 0 or r + max(x for x, _ in motif)+1 > grid.shape[0]
                              or c + max(y for _, y in motif)+1 > grid.shape[1]
                              for (r, c), _ in copies)
                clipping += clipped
                combined += affected >= 2 and joined and len(patches) == 2
        varied_roles += len(roles_here) > 1
        repeated_fg += len(set(foregrounds_here)) < len(foregrounds_here)
    assert len(shapes) >= 8 and len(motifs) >= 30 and len(layouts) >= 100
    assert len(colors) >= 40 and len(patch_shapes) >= 8
    assert footprint_sizes == set(_SIZES)
    assert diagonal_modes == {"train": {False, True}, "test": {False, True}}
    assert multi["train"] > 0 and multi["test"] > 0
    assert covered_many >= 30 and touching >= 20 and clipping >= 20
    assert broad >= 20 and combined > 0
    assert varied_roles >= 30 and repeated_fg >= 20
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(700 + (tr or 0)*10 + (te or 0), tr, te)
        validate_generated_puzzle(p)
        assert tr is None or len(p["train"]) == tr
        assert te is None or len(p["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(13, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
