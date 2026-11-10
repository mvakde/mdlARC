"""Mirror-registered, partially covered cards with a shared elastic design.

Generation renders a sampled logical design directly. solve() reads only pixels;
there is no seed, catalog, generator state, or stored-example lookup in solving.
"""
from __future__ import annotations

from collections import deque
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid cells must be integer colors 0..9")
    return a


def _axis(n):
    """Logical indices of the n-2 interior coordinates; two fixed cells per end."""
    _require(n >= 7, "card dimension must be at least seven")
    return np.array([0, 1] + [2] * (n - 6) + [3, 4], dtype=int)


def _expand(logical, shape):
    return logical[np.ix_(_axis(shape[0]), _axis(shape[1]))]


def _render(logical, shape, paper, tab, ink):
    card = np.full(shape, paper, dtype=np.int8)
    card[0, 0] = tab
    card[1:-1, 1:-1][_expand(logical, shape)] = ink
    return card


def _rectangles(grid):
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "exterior boundary must be uniform")
    remaining = grid != bg
    rectangles = []
    height, width = grid.shape
    for row in range(height):
        for col in range(width):
            if not remaining[row, col]:
                continue
            remaining[row, col] = False
            queue = deque([(row, col)])
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < height and 0 <= cc < width and remaining[rr, cc]:
                        remaining[rr, cc] = False
                        queue.append((rr, cc))
            rr, cc = zip(*cells)
            r0, r1, c0, c1 = min(rr), max(rr) + 1, min(cc), max(cc) + 1
            _require(len(cells) == (r1 - r0) * (c1 - c0),
                     "each non-exterior component must be a full rectangle")
            _require(r1 - r0 >= 7 and c1 - c0 >= 7, "card is too small")
            rectangles.append((r0, r1, c0, c1))
    _require(3 <= len(rectangles) <= 5, "expected three through five cards")
    return bg, sorted(rectangles)


def _parse(input_grid):
    grid = _grid(input_grid)
    bg, rectangles = _rectangles(grid)
    cards = []
    for rect in rectangles:
        r0, r1, c0, c1 = rect
        raw = grid[r0:r1, c0:c1]
        paper = int(raw[-1, 0])
        _require(np.all(raw[-1] == paper) and np.all(raw[1:, 0] == paper)
                 and np.all(raw[1:, -1] == paper), "card rim must be paper")
        left, right = int(raw[0, 0]), int(raw[0, -1])
        _require((left != paper) != (right != paper), "need one top-corner tab")
        _require(np.all(raw[0, 1:-1] == paper), "top rim has extra marks")
        flipped = right != paper
        tab = right if flipped else left
        norm = raw[:, ::-1] if flipped else raw
        others = set(map(int, np.unique(norm))) - {paper, tab}
        _require(len(others) <= 1, "too many card colors")
        cover = norm[1:-1, 1:-1] == tab
        if others:
            _require(np.any(cover), "reference must have an opaque cover")
            rr, cc = np.nonzero(cover)
            _require(int(cover.sum()) == (int(rr.max()) - int(rr.min()) + 1)
                     * (int(cc.max()) - int(cc.min()) + 1),
                     "opaque cover must be rectangular")
        else:
            _require(not np.any(cover), "unprinted card must be uncovered")
        cards.append({"rect": rect, "raw": raw, "norm": norm,
                      "paper": paper, "tab": tab, "flipped": flipped,
                      "ink": next(iter(others)) if others else None})
    tabs = {c["tab"] for c in cards}
    inks = {c["ink"] for c in cards if c["ink"] is not None}
    _require(len(tabs) == len(inks) == 1, "need one common tab and ink role")
    tab, ink = next(iter(tabs)), next(iter(inks))
    _require(bg not in (tab, ink) and tab != ink
             and all(c["paper"] not in (bg, tab, ink) for c in cards),
             "exterior, ink, tab and paper roles must be distinct")
    targets = [i for i, c in enumerate(cards) if c["ink"] is None]
    _require(len(targets) == 1, "need exactly one unprinted target")
    target = targets[0]
    refs = [c for i, c in enumerate(cards) if i != target]
    return {"grid": grid, "background": bg, "tab": tab, "ink": ink,
            "cards": cards, "target": cards[target], "references": refs}


def _observations(references, tab, ink, normalize=True):
    known = np.full((5, 5), -1, dtype=np.int8)
    visibility = []
    consistent = True
    for card in references:
        view = card["norm"] if normalize else card["raw"]
        interior = view[1:-1, 1:-1]
        rows, cols = _axis(view.shape[0]), _axis(view.shape[1])
        visible = np.zeros((5, 5), dtype=bool)
        for a in range(5):
            for b in range(5):
                pixels = interior[np.ix_(rows == a, cols == b)]
                values = set(map(int, pixels[pixels != tab]))
                if not values:
                    continue
                visible[a, b] = True
                bits = {int(v == ink) for v in values}
                if len(bits) != 1:
                    consistent = False
                    continue
                bit = next(iter(bits))
                if known[a, b] != -1 and known[a, b] != bit:
                    consistent = False
                else:
                    known[a, b] = bit
        visibility.append(visible)
    return known, visibility, consistent


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct from visible, registered observations in this input only."""
    info = _parse(input_grid)
    logical, _, consistent = _observations(info["references"], info["tab"], info["ink"])
    _require(consistent and np.all(logical >= 0),
             "registered references must determine a complete consistent design")
    target = info["target"]
    return _render(logical.astype(bool), target["raw"].shape, target["paper"],
                   info["tab"], info["ink"])


def _counts(num_train, num_test):
    for name, count, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(count) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _design(rng):
    # Local corner glyphs plus independently chosen bridging ribbons.
    glyphs = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 0]],
                       [[1, 0], [1, 0]], [[1, 1], [1, 0]],
                       [[1, 1], [1, 1]]], dtype=bool)
    for _ in range(100):
        m = np.zeros((5, 5), dtype=bool)
        active_count = int(rng.integers(2, 5))
        active = set(map(int, rng.choice(4, active_count, replace=False)))
        if active_count == 2 and active not in ({0, 3}, {1, 2}):
            continue
        for k, (rr, cc) in enumerate(((0, 0), (0, 3), (3, 0), (3, 3))):
            if k not in active:
                continue
            glyph = glyphs[int(rng.integers(len(glyphs)))].copy()
            if rr:
                glyph = glyph[::-1]
            if cc:
                glyph = glyph[:, ::-1]
            m[rr:rr + 2, cc:cc + 2] = glyph
        m[2, [0, 1, 3, 4]] = rng.random(4) < 0.42
        m[[0, 1, 3, 4], 2] = rng.random(4) < 0.42
        m[2, 2] = rng.random() < 0.15
        ribbons = int(m[2].sum() + m[:, 2].sum() - m[2, 2])
        if 6 <= int(m.sum()) <= 18 and ribbons >= 2 and all(
                not np.array_equal(m[:, ::-1], other) for other in
                (m, m[::-1], m[::-1, ::-1])):
            return m
    raise RuntimeError("could not sample an asymmetric elastic design")


def _layout(rng, target_shape, count):
    """Staggered shelves, then optionally transpose only the empty layout."""
    choices = {3: ((1, 2), (2, 1)),
               4: ((2, 2), (1, 3), (3, 1)),
               5: ((2, 3), (3, 2), (1, 2, 2), (2, 1, 2), (2, 2, 1))}
    groups = choices[count][int(rng.integers(len(choices[count])))]
    eligible = [k for k, n in enumerate(groups) if n <= 2]
    target_band = int(rng.choice(eligible))
    target_slot = int(rng.integers(groups[target_band]))
    top_margin, bottom_margin, left_margin, right_margin = map(int, rng.integers(1, 3, 4))
    row_gaps = list(map(int, rng.integers(1, 3, len(groups) - 1)))
    band_heights = [target_shape[0] if k == target_band else 7
                    for k in range(len(groups))]
    minimum_h = sum(band_heights) + sum(row_gaps) + top_margin + bottom_margin
    if minimum_h > 30:
        raise ValueError("layout is too tall")
    height = int(rng.integers(minimum_h, 31))
    band_heights = [h + int(e) for h, e in zip(
        band_heights, rng.multinomial(height - minimum_h, np.ones(len(groups)) / len(groups)))]
    col_gaps = [list(map(int, rng.integers(1, 3, n - 1))) for n in groups]
    minimums = []
    for k, n in enumerate(groups):
        minimums.append([target_shape[1] if (k, j) == (target_band, target_slot) else 7
                         for j in range(n)])
    minimum_w = max(sum(v) + sum(g) for v, g in zip(minimums, col_gaps))
    minimum_w += left_margin + right_margin
    if minimum_w > 30:
        raise ValueError("layout is too wide")
    width = int(rng.integers(minimum_w, 31))
    rectangles, target_index = [], None
    row = top_margin
    for k, n in enumerate(groups):
        extra = width - left_margin - right_margin - sum(col_gaps[k]) - sum(minimums[k])
        slots = [v + int(e) for v, e in zip(minimums[k], rng.multinomial(extra, np.ones(n) / n))]
        col = left_margin
        for j, slot in enumerate(slots):
            is_target = (k, j) == (target_band, target_slot)
            if is_target:
                ph, pw = target_shape
                target_index = len(rectangles)
            else:
                ph = int(rng.integers(7, band_heights[k] + 1))
                pw = int(rng.integers(7, slot + 1))
            r0 = row + int(rng.integers(band_heights[k] - ph + 1))
            c0 = col + int(rng.integers(slot - pw + 1))
            rectangles.append((r0, r0 + ph, c0, c0 + pw))
            col += slot + (col_gaps[k][j] if j < n - 1 else 0)
        row += band_heights[k] + (row_gaps[k] if k < len(groups) - 1 else 0)
    if rng.random() < 0.5:
        height, width = width, height
        rectangles = [(c0, c1, r0, r1) for r0, r1, c0, c1 in rectangles]
    return (height, width), rectangles, target_index


def _cover_masks():
    result = []
    for axis in (0, 1):
        for start in (0, 3):
            m = np.zeros((5, 5), dtype=bool)
            if axis == 0:
                m[start:start + 2, :] = True
            else:
                m[:, start:start + 2] = True
            result.append(m)
    for r in (0, 3):
        for c in (0, 3):
            m = np.zeros((5, 5), dtype=bool)
            m[r:r + 2, c:c + 2] = True
            result.append(m)
    return result


_COVERS = _cover_masks()


def _nearest_card(logical, shape, paper, tab, ink):
    rr = np.arange(shape[0] - 2) * 5 // (shape[0] - 2)
    cc = np.arange(shape[1] - 2) * 5 // (shape[1] - 2)
    result = np.full(shape, paper, dtype=np.int8)
    result[0, 0] = tab
    result[1:-1, 1:-1][logical[np.ix_(rr, cc)]] = ink
    return result


def _make_pair(rng, split, index, target_flipped):
    refs_count = 2 if split == "train" and index < 2 else int(rng.integers(2, 5))
    target_shape = ((int(rng.integers(8, 12)), int(rng.integers(8, 12)))
                    if split == "train" else (7, int(rng.integers(12, 18))))
    shape, rectangles, target_index = _layout(rng, target_shape, refs_count + 1)
    ref_shapes = [(r1 - r0, c1 - c0) for k, (r0, r1, c0, c1) in enumerate(rectangles)
                  if k != target_index]
    if len({h for h, _ in ref_shapes}) < 2 or len({w for _, w in ref_shapes}) < 2:
        raise ValueError("references need varied heights and widths")
    logical = _design(rng)
    colors = list(map(int, rng.permutation(10)))
    bg, tab, ink = colors[:3]
    papers = colors[3:3 + len(rectangles)]
    if rng.random() < 0.38:
        a, b = map(int, rng.choice(len(papers), 2, replace=False))
        papers[b] = papers[a]
    axis = (1 if index == 0 else 0) if split == "train" and index < 2 else int(rng.integers(2))
    # indices 0/1 are top/bottom; 2/3 are left/right logical strips.
    first_mask = 2 * axis + int(rng.integers(2))
    complementary = first_mask ^ 1
    masks = [_COVERS[first_mask], _COVERS[complementary]]
    for _ in range(refs_count - 2):
        candidates = [m for m in _COVERS if np.any(logical & m)]
        masks.append(candidates[int(rng.integers(len(candidates)))])
    # Guarantee each cover conceals actual ink, not just already blank paper.
    if any(not np.any(logical & cover) for cover in masks):
        raise ValueError("cover hides no design ink")
    refs_flipped = [False, True] + [bool(rng.integers(2)) for _ in range(refs_count - 2)]
    if rng.random() < 0.5:
        refs_flipped = [not v for v in refs_flipped]
    canvas = np.full(shape, bg, dtype=np.int8)
    ri = 0
    output = None
    for k, (r0, r1, c0, c1) in enumerate(rectangles):
        card_shape = (r1 - r0, c1 - c0)
        if k == target_index:
            card = np.full(card_shape, papers[k], dtype=np.int8)
            card[0, -1 if target_flipped else 0] = tab
            output = _render(logical, card_shape, papers[k], tab, ink)
        else:
            card = _render(logical, card_shape, papers[k], tab, ink)
            card[1:-1, 1:-1][_expand(masks[ri], card_shape)] = tab
            if refs_flipped[ri]:
                card = card[:, ::-1]
            ri += 1
        canvas[r0:r1, c0:c1] = card
    if split == "train" and index == 0 and np.array_equal(
            output, _nearest_card(logical, output.shape, papers[target_index], tab, ink)):
        raise ValueError("first demonstration must distinguish proportional scaling")
    # Keep a construction with hidden ink in every reference after covering.
    info = _parse(canvas)
    if len(info["references"]) != refs_count:
        raise ValueError("a reference lost all its visible ink")
    return {"input": canvas, "output": output}, logical


def _signature(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count, test_count = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    seen, train_designs = set(), []
    first_face = bool(rng.integers(2))
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(180):
                face = (first_face if index == 0 else not first_face) if split == "train" and index < 2 else bool(rng.integers(2))
                try:
                    pair, logical = _make_pair(rng, split, index, face)
                except (ValueError, AssertionError):
                    continue
                if _signature(pair["input"]) in seen:
                    continue
                if split == "train" and index == 1 and (
                        np.array_equal(logical, train_designs[0])
                        or pair["output"].shape == puzzle["train"][0]["output"].shape):
                    continue
                seen.add(_signature(pair["input"]))
                puzzle[split].append(pair)
                if split == "train":
                    train_designs.append(logical)
                break
            else:
                raise RuntimeError("could not construct a distinct inferable pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    info = _parse(pair["input"])
    logical, visibility, consistent = _observations(info["references"], info["tab"], info["ink"])
    _require(consistent and np.all(logical >= 0), "incomplete or conflicting registered evidence")
    _require(all(0 < int(v.sum()) < 25 for v in visibility),
             "each reference must leave some whole logical cells unknown")
    refs = info["references"]
    _require(len({c["raw"].shape[0] for c in refs}) >= 2
             and len({c["raw"].shape[1] for c in refs}) >= 2,
             "reference sizes must expose stretching on both axes")
    _require({c["flipped"] for c in refs} == {False, True}, "need both reference faces")
    tile = logical.astype(bool)
    for c in refs:
        cover = c["norm"][1:-1, 1:-1] == info["tab"]
        _require(np.any(cover & _expand(tile, c["raw"].shape)), "cover must conceal true ink")
    target = info["target"]
    expected = _render(tile, target["raw"].shape, target["paper"], info["tab"], info["ink"])
    _require(np.array_equal(pair["output"], expected), "incorrect completion, paper, tab or crop")
    _require(np.array_equal(solve(pair["input"]), expected), "input-only solve disagrees")
    # Omitting A: registering and cropping the unprinted target cannot add ink.
    _require(not np.array_equal(target["norm"], expected), "stencil inference has no effect")
    info["logical"], info["visibility"] = tile, visibility
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            for value in pair.values():
                _grid(value)
            sig = _signature(pair["input"])
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            info = _analyze(pair)
            analyses[split].append(info)
            h, w = info["target"]["raw"].shape
            if split == "train":
                _require(8 <= h <= 11 and 8 <= w <= 11, "training target range")
            else:
                _require(min(h, w) == 7 and 12 <= max(h, w) <= 17, "missing aspect/scale shift")
    first, second = analyses["train"][:2]
    raw, _, consistent = _observations(first["references"], first["tab"], first["ink"], False)
    _require(not consistent or np.any(raw < 0), "first pair does not require B's registration")
    _require(not np.array_equal(first["logical"], second["logical"]), "designs must be pair-local")
    _require(first["target"]["flipped"] != second["target"]["flipped"], "target faces need evidence")
    _require(first["target"]["raw"].shape != second["target"]["raw"].shape, "training target shapes repeat")
    _require(any(not np.array_equal(pair["output"], _nearest_card(
                    info["logical"], info["target"]["raw"].shape,
                    info["target"]["paper"], info["tab"], info["ink"]))
                 for pair, info in zip(puzzle["train"], analyses["train"])),
             "demonstrations do not distinguish fixed ends from proportional scaling")


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _metamorphic(pair):
    info = _parse(pair["input"])
    # Independently changing the face of every card leaves the registered answer invariant.
    reflected = np.array(pair["input"], copy=True)
    for card in info["cards"]:
        r0, r1, c0, c1 = card["rect"]
        reflected[r0:r1, c0:c1] = card["raw"][:, ::-1]
    _require(np.array_equal(solve(reflected), pair["output"]), "whole-card face invariance failed")
    permutation = np.array([7, 3, 9, 0, 4, 8, 1, 6, 2, 5])
    _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
             "semantic-color equivariance failed")
    logical, _, _ = _observations(info["references"], info["tab"], info["ink"])
    a, b = map(int, np.argwhere(logical == 0)[0])
    edited = np.array(pair["input"], copy=True)
    for card in info["references"]:
        norm = card["norm"].copy()
        region = (_axis(norm.shape[0])[:, None] == a) & (_axis(norm.shape[1])[None, :] == b)
        inner = norm[1:-1, 1:-1]
        inner[region & (inner != info["tab"])] = info["ink"]
        r0, r1, c0, c1 = card["rect"]
        edited[r0:r1, c0:c1] = norm[:, ::-1] if card["flipped"] else norm
    logical[a, b] = 1
    target = info["target"]
    changed = _render(logical.astype(bool), target["raw"].shape, target["paper"], info["tab"], info["ink"])
    _require(not np.array_equal(changed, pair["output"]) and np.array_equal(solve(edited), changed),
             "visible evidence does not control target design")


def self_test():
    """Bounded checks, with descriptive coverage across exactly 42 seed puzzles."""
    observed = {k: set() for k in ("designs", "canvas_shapes", "target_shapes", "reference_shapes",
                                    "layouts", "palettes", "cover_rectangles", "reference_counts")}
    reused_paper = portrait = landscape = 0
    puzzle_signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed in (0, 17, 41):
            _require(_equal_puzzles(puzzle, generate_puzzle(seed)), "seed is not deterministic")
        puzzle_signatures.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair)
                observed["designs"].add(info["logical"].tobytes())
                observed["canvas_shapes"].add(pair["input"].shape)
                target_shape = info["target"]["raw"].shape
                observed["target_shapes"].add(target_shape)
                portrait += target_shape[0] > target_shape[1]
                landscape += target_shape[0] < target_shape[1]
                observed["reference_shapes"].update(c["raw"].shape for c in info["references"])
                observed["layouts"].add((pair["input"].shape, tuple(c["rect"] for c in info["cards"])))
                observed["palettes"].add((info["background"], info["tab"], info["ink"],
                                          tuple(c["paper"] for c in info["cards"])))
                observed["reference_counts"].add(len(info["references"]))
                papers = [c["paper"] for c in info["cards"]]
                reused_paper += len(set(papers)) < len(papers)
                for c in info["references"]:
                    rr, cc = np.nonzero(c["norm"][1:-1, 1:-1] == info["tab"])
                    observed["cover_rectangles"].add((int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())))
        if seed < 3:
            _metamorphic(puzzle["train"][0])
            _metamorphic(puzzle["test"][0])
    _require(len(puzzle_signatures) == 42, "42 seeds did not make distinct puzzles")
    limits = {"designs": 160, "canvas_shapes": 30, "target_shapes": 24, "reference_shapes": 40,
              "layouts": 180, "palettes": 180, "cover_rectangles": 25}
    for name, minimum in limits.items():
        _require(len(observed[name]) >= minimum, f"insufficient variation in {name}")
    _require(observed["reference_counts"] == {2, 3, 4}, "reference counts do not vary")
    _require(reused_paper >= 30 and portrait >= 40 and landscape >= 40, "geometry/color reuse lacks variation")
    # All legal explicit counts, with a separate seed for each count combination.
    count_calls = 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * train + test, train, test)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "counts ignored")
            count_calls += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 0.5, "0"):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # A design outside the corner/ribbon sampler, including a dense elastic center.
    tile = np.array([[1, 0, 0, 1, 0], [0, 1, 1, 0, 1], [1, 0, 1, 1, 0],
                     [1, 1, 0, 0, 1], [0, 1, 1, 1, 0]], dtype=bool)
    canvas = np.zeros((24, 30), dtype=np.int8)
    for k, (r0, r1, c0, c1) in enumerate(((1, 9, 1, 10), (11, 22, 1, 13), (1, 8, 14, 28))):
        shape = (r1 - r0, c1 - c0)
        if k < 2:
            card = _render(tile, shape, 3 + k, 1, 2)
            card[1:-1, 1:-1][_expand(_COVERS[2 + k], shape)] = 1
            if k:
                card = card[:, ::-1]
        else:
            card = np.full(shape, 5, dtype=np.int8)
            card[0, -1] = 1
        canvas[r0:r1, c0:c1] = card
    _require(np.array_equal(solve(canvas), _render(tile, (7, 14), 5, 1, 2)),
             "solver does not generalize beyond sampled designs")
    return {"seeds_checked": 42, "pairs_checked": 210, "legal_count_calls": count_calls,
            "distinct_puzzles": len(puzzle_signatures),
            "variation": {k: len(v) for k, v in observed.items()},
            "paper_reuse_pairs": reused_paper, "portrait_targets": portrait,
            "landscape_targets": landscape, "metamorphic_pairs": 6}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
