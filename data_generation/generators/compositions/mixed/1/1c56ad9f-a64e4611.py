"""Bottom-phased scaffold wave followed by clipped empty-neighborhood marking.

Only NumPy and the standard library are used. No files or random state are used
by solve. Highlight color 3 is a fixed family convention; other roles are local.
"""

from __future__ import annotations

import hashlib
import numpy as np


HIGHLIGHT = 3
OFFSETS = (0, -1, 0, 1)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((array >= 0) & (array <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return array.astype(np.int8, copy=False)


def _roles(grid):
    # Every legal scaffold has blank columns on both sides, including at (0,0).
    bg = int(grid[0, 0])
    values = np.unique(grid)
    if len(values) != 2 or HIGHLIGHT in values:
        raise ValueError("input needs two colors, neither of them highlight 3")
    fg = int(values[values != bg][0])
    return bg, fg


def _shift(mask, anchor="bottom", offsets=OFFSETS):
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        raise ValueError("empty scaffold")
    top, bottom = int(rows[0]), int(rows[-1])
    shifted = np.zeros_like(mask, dtype=bool)
    for r in rows:
        distance = (bottom - r if anchor == "bottom" else
                    r - top if anchor == "top" else mask.shape[0] - 1 - r)
        columns = np.flatnonzero(mask[r]) + offsets[int(distance) % 4]
        if columns.min() < 0 or columns.max() >= mask.shape[1]:
            raise ValueError("row shift would leave the canvas")
        # Set union of translated occupancy, independent of stroke drawing order.
        shifted[r, columns] = True
    return shifted


def _clear(mask, diagonal=True):
    """Centers with no occupied in-grid neighbor; outside is never an obstacle."""
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    near = np.zeros_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            if diagonal or dy == 1 or dx == 1:
                near |= padded[dy:dy + h, dx:dx + w]
    return ~near


def _paint(mask, bg, fg):
    out = np.full(mask.shape, bg, dtype=np.int8)
    out[mask] = fg
    out[_clear(mask)] = HIGHLIGHT
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive colors, bottom phase, translated occupancy and highlights from input."""
    inp = _grid(input_grid)
    bg, fg = _roles(inp)
    shifted = _shift(inp == fg)
    return _paint(shifted, bg, fg)


def _components(mask):
    remaining = set(zip(*np.where(mask)))
    sizes = []
    while remaining:
        todo = [remaining.pop()]
        size = 0
        while todo:
            r, c = todo.pop()
            size += 1
            for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if cell in remaining:
                    remaining.remove(cell)
                    todo.append(cell)
        sizes.append(size)
    return sizes


def _analyze(inp):
    bg, fg = _roles(inp)
    mask = inp == fg
    rr, cc = np.where(mask)
    t, b, l, r = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    _require(l >= 1 and r < inp.shape[1] - 1, "missing lateral wave margin")
    _require(b - t + 1 >= 5, "scaffold too short")
    _require(np.all(mask[t:b + 1].any(axis=1)), "interrupted vertical extent")
    core = mask[t:b + 1, l:r + 1]
    _require(np.all(core[0]) and np.all(core[-1]), "incomplete end bars")
    rails = np.all(core, axis=0)
    starts = np.flatnonzero(rails & ~np.r_[False, rails[:-1]])
    ends = np.flatnonzero(rails & ~np.r_[rails[1:], False])
    _require(2 <= len(starts) <= 6 and rails[0] and rails[-1], "bad supports")
    _require(np.count_nonzero(np.all(core == rails, axis=1)) >= 2,
             "need two open support rows")
    bars, partial = [], 0
    gaps = []
    for i in range(len(starts) - 1):
        gaps.append(int(starts[i + 1] - ends[i] - 1))
    for y, row in enumerate(core[1:-1], 1):
        filled = []
        for i in range(len(starts) - 1):
            segment = row[ends[i] + 1:starts[i + 1]]
            _require(np.all(segment) or not np.any(segment), "broken crossbar")
            filled.append(bool(np.all(segment)))
        if any(filled):
            bars.append(y)
            partial += int(not all(filled))
    _require(len(_components(mask)) == 1, "scaffold must be connected")
    return dict(bg=bg, fg=fg, top=t, bottom=b, left=l, right=r,
                height=b - t + 1, width=r - l + 1,
                bottom_pad=inp.shape[0] - b - 1,
                widths=tuple(int(z - a + 1) for a, z in zip(starts, ends)),
                gaps=tuple(gaps), bars=tuple(bars), partial=partial)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _crop_key(inp):
    info = _analyze(inp)
    crop = inp[info["top"]:info["bottom"] + 1, info["left"]:info["right"] + 1]
    return crop.shape, (crop == info["fg"]).tobytes()


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _geometry(rng, height, kind):
    """Sample connected rails and gap-spanning bars, rather than parent puzzles."""
    if kind == "hollow":
        count = 2
        widths = [int(rng.integers(1, 4)) for _ in range(count)]
        gaps = [int(rng.integers(6, 11))]
    elif kind == "diagnostic":
        count = int(rng.integers(3, 5))
        widths = [int(rng.integers(1, 3)) for _ in range(count)]
        gaps = [4, int(rng.integers(6, 9))]
        if count == 4:
            gaps.append(int(rng.integers(1, 5)))
        rng.shuffle(gaps)
    elif kind == "rich":
        count = int(rng.integers(3, 6))
        widths = [int(rng.integers(1, 4)) for _ in range(count)]
        widths[0], widths[-1] = 1, 2
        gaps = [int(rng.integers(1, 7)) for _ in range(count - 1)]
        gaps[0], gaps[-1] = 2, 6
    else:
        count = int(rng.integers(2, 6))
        widths = ([1] * count if rng.random() < .35 else
                  [int(rng.integers(1, 4)) for _ in range(count)])
        gaps = [int(rng.integers(1, 10)) for _ in range(count - 1)]
    # Budget width before constructing any grid, always keeping wave margins.
    while sum(widths) + sum(gaps) > 28:
        candidates = [i for i, gap in enumerate(gaps) if gap > 1]
        i = int(rng.choice(candidates))
        gaps[i] -= 1
    starts, ends, cursor = [], [], 0
    for i, width in enumerate(widths):
        starts.append(cursor)
        ends.append(cursor + width - 1)
        cursor += width + (gaps[i] if i < len(gaps) else 0)
    mask = np.zeros((height, cursor), dtype=bool)
    for a, z in zip(starts, ends):
        mask[:, a:z + 1] = True
    mask[[0, -1], :] = True
    if kind == "diagnostic":
        # Five open rows above the bottom preserve isolated candidate centers in
        # the four-cell gap after the wave. Upper bars demonstrate non-resetting.
        candidates = [y for y in range(1, height - 6) if (height - 1 - y) % 4 != 0]
        chosen = [int(rng.choice(candidates))]
        chosen += [y for y in range(1, height - 6) if rng.random() < .18]
    elif kind == "hollow":
        chosen = []
    else:
        density = float(rng.uniform(.10, .40))
        chosen = [y for y in range(1, height - 1) if rng.random() < density]
        if kind == "rich":
            chosen += [2, 3]
        chosen = sorted(set(chosen))[:height - 4]
    for y in chosen:
        if kind in ("diagnostic", "full") or count == 2 or rng.random() < .48:
            mask[y, :] = True
        else:
            first = int(rng.integers(0, count - 1))
            last = int(rng.integers(first + 1, count))
            mask[y, ends[first]:starts[last] + 1] = True
    if kind == "rich":
        # Guaranteed partial bar, plus a full bar of the opposite row parity.
        mask[2, :] = False
        for a, z in zip(starts, ends):
            mask[2, a:z + 1] = True
        mask[2, ends[0]:starts[1] + 1] = True
        mask[3, :] = True
    return mask


def _make_pair(rng, height, kind, bg=None, fg=None, top=None, bottom_pad=None,
               edge_demo=False):
    shape = _geometry(rng, height, kind)
    spare = 30 - height
    if top is None:
        top = int(rng.integers(0, min(5, spare) + 1))
    if bottom_pad is None:
        bottom_pad = int(rng.integers(0, min(6, spare - top) + 1))
    left_base = 3 if edge_demo else 1
    spare_w = 30 - shape.shape[1] - left_base - 1
    left = left_base + int(rng.integers(0, min(6, spare_w) + 1))
    right = 1 + int(rng.integers(0, min(6, spare_w - left + left_base) + 1))
    if bg is None:
        bg = 0 if rng.random() < .45 else int(rng.choice([1, 2, 4, 5, 6, 7, 8, 9]))
    if fg is None or fg == bg:
        fg = int(rng.choice([x for x in range(10) if x not in (bg, HIGHLIGHT)]))
    inp = np.full((height + top + bottom_pad, shape.shape[1] + left + right),
                  bg, dtype=np.int8)
    inp[top:top + height, left:left + shape.shape[1]][shape] = fg
    return {"input": inp, "output": solve(inp)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one deterministic puzzle; explicit counts always take priority."""
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(0, 3))
    heldout = int(rng.integers(0, 4))
    train_heights = list(range(9, 17 if mode == 1 else 27))
    if mode == 0:
        train_heights = [h for h in train_heights if h % 4 != heldout]
    first_h = int(rng.choice([h for h in train_heights if h % 2 == 0]))
    second_h = int(rng.choice([h for h in train_heights if h % 4 != first_h % 4]))
    shared_top = int(rng.integers(0, min(3, 29 - max(first_h, second_h)) + 1))
    bg1 = int(rng.choice([0, 1, 2, 4, 5, 6, 7, 8, 9]))
    bg2 = int(rng.choice([c for c in range(10) if c not in (bg1, HIGHLIGHT)]))
    puzzle = {"train": [], "test": []}
    used, crops = set(), set()

    def add(split, height, kind, **kwargs):
        for _ in range(48):
            pair = _make_pair(rng, height, kind, **kwargs)
            inp = pair["input"]
            info = _analyze(inp)
            original = inp == info["fg"]
            shifted = _shift(original)
            clear = _clear(shifted)
            if (_key(inp) in used or _crop_key(inp) in crops
                    or not clear.any() or np.array_equal(clear, _clear(original))):
                continue
            if kind == "diagnostic" and not any(1 <= n <= 4 for n in _components(clear)):
                continue
            used.add(_key(inp))
            crops.add(_crop_key(inp))
            puzzle[split].append(pair)
            return
        raise RuntimeError("bounded construction exhausted")

    add("train", first_h, "hollow", bg=bg1, top=shared_top, bottom_pad=0, edge_demo=True)
    add("train", second_h, "diagnostic", bg=bg2, top=shared_top, bottom_pad=1)
    for _ in range(ntrain - 2):
        height = int(rng.choice(train_heights if mode == 0 else
                                list(range(7, 17 if mode == 1 else 27))))
        kind = str(rng.choice(["hollow", "full", "general"]))
        if mode == 2:
            kind = str(rng.choice(["hollow", "full"]))
        add("train", height, kind)
    rng.shuffle(puzzle["train"])
    train_colors = [_roles(p["input"])[1] for p in puzzle["train"]]
    for i in range(ntest):
        if mode == 0:
            choices = [h for h in range(7, 27) if h % 4 == heldout]
            kind = str(rng.choice(["hollow", "full", "general", "rich"]))
        elif mode == 1:
            choices = list(range(18, 27))
            kind = str(rng.choice(["hollow", "full", "general", "rich"]))
        else:
            choices, kind = list(range(9, 27)), "rich"
        height = int(rng.choice(choices))
        fg = train_colors[i % len(train_colors)] if rng.random() < .50 else None
        add("test", height, kind, fg=fg)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check geometry, exact occupancy, local equivalence and training witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "bad splits")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "bad counts")
    seen, crops, infos, learned = set(), set(), [], {}
    witnesses = dict(small=False, edge=False, diagonal=False, order=False,
                     whole_window=False, crossbar_restart=False)
    alternative_fits = {"top": True, "canvas": True}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            info = _analyze(inp)
            bg, fg, b = info["bg"], info["fg"], info["bottom"]
            _require(_key(inp) not in seen, "duplicate input")
            _require(_crop_key(inp) not in crops, "repeated scaffold geometry")
            seen.add(_key(inp))
            crops.add(_crop_key(inp))
            _require(set(np.unique(out)) == {bg, fg, HIGHLIGHT}, "output role loss")
            occupied = out == fg
            # Recover observed rigid offsets from output, without calling solve.
            for r in range(inp.shape[0]):
                a, z = np.flatnonzero(inp[r] == fg), np.flatnonzero(occupied[r])
                _require(len(a) == len(z), "row foreground count changed")
                if len(a):
                    delta = z - a
                    _require(np.all(delta == delta[0]), "row not rigid")
                    phase, shift = (b - r) % 4, int(delta[0])
                    _require(shift == OFFSETS[phase], "wrong wave phase")
                    if split == "train":
                        _require(phase not in learned or learned[phase] == shift, "phase conflict")
                        learned[phase] = shift
            # Exhaustive scalar clipped-window equivalence is separate from
            # solve's padded Boolean dilation implementation.
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    window = occupied[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                    expected = fg if occupied[r, c] else bg if window.any() else HIGHLIGHT
                    _require(int(out[r, c]) == expected, "wrong local marking/fringe")
            original = inp == fg
            clear = out == HIGHLIGHT
            _require(not np.array_equal(clear, _clear(original)), "A does not change highlight set")
            _require(clear.any(), "B has no effect")
            if split == "train":
                infos.append(info)
                witnesses["small"] |= any(1 <= n <= 4 for n in _components(clear))
                witnesses["edge"] |= bool(clear[0].any() or clear[-1].any()
                                           or clear[:, 0].any() or clear[:, -1].any())
                witnesses["diagonal"] |= bool(np.any(_clear(occupied, False) & ~clear))
                # Reversing the operations would shift an already-computed
                # highlight field. Test locations inside the scaffold's extent.
                naive = _clear(original)
                shifted_naive = np.zeros_like(naive)
                for r in range(inp.shape[0]):
                    shift = OFFSETS[(b - r) % 4] if original[r].any() else 0
                    cs = np.flatnonzero(naive[r]) + shift
                    cs = cs[(cs >= 0) & (cs < inp.shape[1])]
                    shifted_naive[r, cs] = True
                region = (slice(info["top"], b + 1), slice(1, inp.shape[1] - 1))
                witnesses["order"] |= bool(np.any((clear ^ shifted_naive)[region]))
                witnesses["whole_window"] |= bool(np.any(
                    ~_clear(clear) & ~occupied & ~clear))
                full_bars = [y for y in range(info["top"] + 1, b + 1)
                             if original[y, info["left"]:info["right"] + 1].all()]
                restart = np.zeros_like(original)
                for r in range(info["top"], b + 1):
                    anchor = next(y for y in full_bars if y >= r)
                    cs = np.flatnonzero(original[r]) + OFFSETS[(anchor - r) % 4]
                    restart[r, cs] = True
                witnesses["crossbar_restart"] |= not np.array_equal(restart, occupied)
                for anchor in alternative_fits:
                    alternative_fits[anchor] &= np.array_equal(
                        _shift(original, anchor), occupied)
    _require(learned == dict(enumerate(OFFSETS)), "training omits cycle phases")
    _require(all(witnesses.values()), "training misses a local-rule witness")
    _require(not any(alternative_fits.values()), "training permits wrong phase anchor")
    _require(len({i["height"] % 4 for i in infos}) >= 2, "top phases lack contrast")
    _require(len({i["bottom_pad"] % 4 for i in infos}) >= 2, "padding lacks contrast")
    _require(len({i["bottom"] % 4 for i in infos}) >= 2, "absolute anchors lack contrast")
    _require(any(not i["bars"] and len(i["widths"]) == 2 for i in infos), "no hollow demo")
    _require(any(i["bars"] and len(i["widths"]) >= 3 for i in infos), "no lattice demo")
    _require(len({i["bg"] for i in infos}) >= 2, "background roles lack contrast")
    # At least one intended shift: held-out height residue, taller scale, or a
    # test combining partial bars with mixed support widths and unequal gaps.
    ti = [_analyze(_grid(p["input"])) for p in puzzle["test"]]
    residues = {i["height"] % 4 for i in infos}
    _require(all(i["height"] % 4 not in residues for i in ti)
             or min(i["height"] for i in ti) > max(i["height"] for i in infos)
             or all(i["partial"] and len(set(i["widths"])) > 1
                    and len(set(i["gaps"])) > 1 for i in ti), "missing train/test shift")


def self_test():
    """Stream 42 seeds; retain only small signature/feature sets, never traces."""
    signatures, structures, shapes, colors, heights = set(), set(), set(), set(), set()
    features = dict(partial=False, thick=False, thin=False, unequal=False,
                    small_canvas=False, full_axis=False, zero_fg=False, nonzero_bg=False)
    test_residues = set()
    for seed in range(42):
        nt = 2 + seed % 9
        ne = 1 + (seed // 9) % min(4, nt - 1)
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze(inp)
                digest.update(bytes(inp.shape))
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                structures.add(hashlib.sha256(_crop_key(inp)[1]).digest())
                shapes.add(inp.shape)
                heights.add(info["height"])
                colors.add((info["bg"], info["fg"]))
                features["partial"] |= bool(info["partial"])
                features["thick"] |= max(info["widths"]) > 1
                features["thin"] |= max(info["widths"]) == 1
                features["unequal"] |= len(set(info["gaps"])) > 1
                features["small_canvas"] |= max(inp.shape) <= 15
                features["full_axis"] |= 30 in inp.shape
                features["zero_fg"] |= info["fg"] == 0
                features["nonzero_bg"] |= info["bg"] != 0
                _require(np.array_equal(solve(inp), out), "solver disagreement")
                if split == "test":
                    test_residues.add(info["height"] % 4)
        signatures.add(digest.digest())
    _require(len(signatures) == 42 and len(structures) >= 150, "structure diversity too low")
    _require(len(shapes) >= 50 and len(heights) >= 15 and len(colors) >= 25, "range too narrow")
    _require(all(features.values()) and test_residues == {0, 1, 2, 3}, "missing family modes")
    a, b = generate_puzzle(991, 10, 4), generate_puzzle(991, 10, 4)
    _require(all(np.array_equal(x[k], y[k]) for s in ("train", "test")
                 for x, y in zip(a[s], b[s]) for k in ("input", "output")), "nondeterministic")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(731, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "wrong count")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 streamed seeds and all legal count combinations)")
