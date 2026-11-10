"""Sample layered-rectangle puzzles with a recoverable lowest layer."""

import numpy as np


_SOURCE_PAIRS = (
    ("003300/003220/011221/011111/011111/003300", "33/33/33/33/33/33"),
    ("004440/064446/064446/064777/004777/000777", "66666/66666/66666"),
    ("000000/077777/078887/222887/222880/222000", "77777/77777/77777"),
    ("0000/0044/0884/0884/0884/0884", "44/44/44/44/44"),
    ("000000/000500/033530/034544/004544/000500", "3333/3333"),
    ("002000/052500/111111/052500/002000/002000", "555/555/555"),
)


def _parse_grid(text):
    return np.array([[int(v) for v in row] for row in text.split("/")], dtype=np.int8)


_SOURCE = tuple((_parse_grid(a), _parse_grid(b)) for a, b in _SOURCE_PAIRS)


def _source_copy(inp, out):
    return any(np.array_equal(inp, a) and np.array_equal(out, b) for a, b in _SOURCE)


def _bbox(grid, color):
    cells = np.argwhere(grid == color)
    if len(cells) == 0:
        return None
    return (int(cells[:, 0].min()), int(cells[:, 1].min()),
            int(cells[:, 0].max()) + 1, int(cells[:, 1].max()) + 1)


def _overlap(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _paint(shape, colors, rects):
    grid = np.zeros(shape, dtype=np.int8)
    for color, (r0, c0, r1, c1) in zip(colors, rects):
        grid[r0:r1, c0:c1] = color
    return grid


def _analyze(grid):
    """Recover visible extents and the unique possible backmost color.

    A visible cell must be above every other rectangle whose recovered extent
    contains it. These precedence edges allow several front-layer orders.
    """
    if not np.any(grid == 0):
        return None
    colors = tuple(int(v) for v in np.unique(grid) if v != 0)
    if not 2 <= len(colors) <= 5:
        return None
    rects = {color: _bbox(grid, color) for color in colors}
    cover = np.zeros((len(colors), *grid.shape), dtype=bool)
    for index, color in enumerate(colors):
        r0, c0, r1, c1 = rects[color]
        cover[index, r0:r1, c0:c1] = True
    if np.any((grid == 0) & np.any(cover, axis=0)):
        return None
    edges = {color: set() for color in colors}
    for index, color in enumerate(colors):
        visible = grid == color
        for lower_index, lower in enumerate(colors):
            if lower != color and np.any(visible & cover[lower_index]):
                edges[lower].add(color)
    indegree = {color: 0 for color in colors}
    for successors in edges.values():
        for color in successors:
            indegree[color] += 1
    roots = [color for color in colors if indegree[color] == 0]
    if len(roots) != 1:
        return None
    available = roots[:]
    visited = 0
    while available:
        color = available.pop()
        visited += 1
        for successor in edges[color]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                available.append(successor)
    if visited != len(colors):
        return None
    return rects, roots[0], edges


def _canvas(rng, split):
    draw = rng.random()
    if draw < 0.16:
        return 6, 6
    if draw < 0.28:
        return 6, 4
    if draw < 0.40:
        return 4, 6
    if split == "test" and draw < 0.52:
        return (int(rng.integers(7, 11)), int(rng.integers(4, 7)))
    return int(rng.integers(4, 11)), int(rng.integers(4, 11))


def _back_rect(rng, shape, interior, thin):
    height, width = shape
    if thin:
        if rng.random() < 0.5:
            h, w = int(rng.integers(3, height + 1)), 1
        else:
            h, w = 1, int(rng.integers(3, width + 1))
    else:
        max_h = height - 2 if interior else height
        max_w = width - 2 if interior else width
        if max_h < 2 or max_w < 2:
            return None
        h = int(rng.integers(2, max_h + 1))
        w = int(rng.integers(2, max_w + 1))
    places = [(r, c) for r in range(height - h + 1)
              for c in range(width - w + 1)
              if ((r > 0 and c > 0 and r + h < height and c + w < width)
                  if interior else
                  (r == 0 or c == 0 or r + h == height or c + w == width))]
    if not places:
        return None
    r, c = places[int(rng.integers(len(places)))]
    return r, c, r + h, c + w


def _front_rect(rng, shape, back, bar):
    height, width = shape
    for _ in range(70):
        if bar:
            if rng.random() < 0.5:
                h, w = 1, int(rng.integers(3, width + 1))
            else:
                h, w = int(rng.integers(3, height + 1)), 1
        else:
            h = int(rng.integers(2, height + 1))
            w = int(rng.integers(2, width + 1))
        r = int(rng.integers(height - h + 1))
        c = int(rng.integers(width - w + 1))
        rect = r, c, r + h, c + w
        if _overlap(rect, back) and rect != back:
            return rect
    return None


def _special_rectangles(rng, mode):
    if mode == "cross":
        shape = int(rng.integers(5, 11)), int(rng.integers(5, 11))
        height, width = shape
        h = int(rng.integers(3, height - 1))
        w = int(rng.integers(3, width - 1))
        r = int(rng.integers(1, height - h))
        c = int(rng.integers(1, width - w))
        back = (r, c, r + h, c + w)
        mid_r = int(rng.integers(r + 1, r + h - 1))
        mid_c = int(rng.integers(c + 1, c + w - 1))
        fronts = [(0, mid_c, height, mid_c + 1),
                  (mid_r, 0, mid_r + 1, width)]
        if rng.random() < 0.5:
            fronts.reverse()
        return shape, [back] + fronts
    if mode == "many":
        shape = int(rng.integers(8, 11)), int(rng.integers(8, 11))
        height, width = shape
        h = int(rng.integers(6, height - 1))
        w = int(rng.integers(6, width - 1))
        r = int(rng.integers(1, height - h))
        c = int(rng.integers(1, width - w))
        back = (r, c, r + h, c + w)
        fronts = [(r + 1, c + 1, r + 3, c + 3),
                  (r + 1, c + w - 3, r + 3, c + w - 1),
                  (r + h - 3, c + 1, r + h - 1, c + 3),
                  (r + h - 3, c + w - 3, r + h - 1, c + w - 1)]
        rng.shuffle(fronts)
        return shape, [back] + fronts
    shape = int(rng.integers(7, 11)), int(rng.integers(6, 11))
    height, width = shape
    h = int(rng.integers(5, height - 1))
    w = int(rng.integers(4, width - 1))
    r = int(rng.integers(1, height - h))
    c = int(rng.integers(1, width - w))
    back = (r, c, r + h, c + w)
    fronts = [(r + 1, c, r + 2, c + w),
              (r + h - 2, c, r + h - 1, c + w)]
    if rng.random() < 0.5:
        fronts.reverse()
    return shape, [back] + fronts


def _make_pair(rng, split, allowed_colors, back_color=None,
               required_color=None, thin_back=False, mode=None):
    for _ in range(1300):
        if mode in ("cross", "disjoint", "many"):
            shape, rects = _special_rectangles(rng, mode)
        else:
            shape = _canvas(rng, split)
            count = int(rng.choice((2, 3, 4, 5), p=(0.27, 0.43, 0.23, 0.07)))
            interior = rng.random() < 0.46
            back = _back_rect(rng, shape, interior, thin_back)
            if back is None:
                continue
            rects = [back]
            for _front in range(count - 1):
                rect = _front_rect(rng, shape, back, rng.random() < 0.34)
                if rect is None:
                    break
                rects.append(rect)
            if len(rects) != count:
                continue
        count = len(rects)
        colors = [int(v) for v in rng.choice(allowed_colors, size=count, replace=False)]
        if back_color is not None:
            if back_color in colors:
                colors.remove(back_color)
            else:
                colors.pop(0)
            colors.insert(0, back_color)
        if required_color is not None and required_color not in colors:
            replacement = int(rng.integers(1, count))
            colors[replacement] = required_color
        inp = _paint(shape, colors, rects)
        analysis = _analyze(inp)
        if analysis is None:
            continue
        recovered, lowest, _ = analysis
        if lowest != colors[0] or any(color not in recovered or recovered[color] != rect
                                     for color, rect in zip(colors, rects)):
            continue
        r0, c0, r1, c1 = rects[0]
        out = np.full((r1 - r0, c1 - c0), colors[0], dtype=np.int8)
        if np.array_equal(inp, out) or _source_copy(inp, out):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("Could not construct a recoverable rectangle pair")


def _count(name, value, low, high):
    if value is not None and (isinstance(value, bool) or
                              not isinstance(value, (int, np.integer)) or
                              not low <= value <= high):
        raise ValueError(name)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    _count("num_train", num_train, 2, 10)
    _count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))

    # Black is the shared background. The omitted color is deliberately novel
    # in at least one test input; the colored rectangles have no fixed roles.
    held_out = int(rng.integers(1, 10))
    train_colors = [c for c in range(1, 10) if c != held_out]
    all_colors = list(range(1, 10))
    puzzle = {"train": [], "test": []}
    thin_training = rng.random() < 0.36
    thin_index = int(rng.integers(num_train)) if thin_training else -1
    special_train = int(rng.integers(num_train))
    special_mode = str(rng.choice(("cross", "disjoint", "many"), p=(0.4, 0.4, 0.2)))

    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(120):
                if split == "train":
                    mode = special_mode if index == special_train and not (thin_training and index == thin_index) else None
                    if mode is None and rng.random() < 0.18 and not (thin_training and index == thin_index):
                        mode = str(rng.choice(("cross", "disjoint", "many"), p=(0.4, 0.4, 0.2)))
                    previous = {int(p["output"][0, 0]) for p in puzzle["train"]}
                    choices = [c for c in train_colors if c not in previous] or train_colors
                    back_color = int(rng.choice(choices))
                    allowed = train_colors
                    required = None
                else:
                    test_thin = thin_training and rng.random() < 0.28
                    mode = (str(rng.choice(("cross", "disjoint", "many"), p=(0.4, 0.4, 0.2)))
                            if not test_thin and rng.random() < 0.34 else None)
                    back_color = None
                    allowed = all_colors
                    required = held_out if index == 0 else None
                pair = _make_pair(np.random.default_rng(int(rng.integers(0, 2**63 - 1))),
                                  split, allowed, back_color, required,
                                  (split == "train" and index == thin_index)
                                  or (split == "test" and test_thin),
                                  mode)
                if all(not np.array_equal(pair["input"], old["input"])
                       for group in puzzle.values() for old in group):
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_colors = set()
    train_back_colors = set()
    train_has_thin_back = False
    test_has_thin_back = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            assert not _source_copy(inp, out)
            analysis = _analyze(inp)
            assert analysis is not None
            rects, back, edges = analysis
            assert all(_overlap(rects[back], rects[c]) for c in rects if c != back)
            r0, c0, r1, c1 = rects[back]
            assert out.shape == (r1 - r0, c1 - c0)
            assert np.all(out == back)
            assert not np.array_equal(inp, out)
            thin_back = r1 - r0 == 1 or c1 - c0 == 1
            if split == "train":
                train_colors.update(rects)
                train_back_colors.add(back)
                train_has_thin_back |= thin_back
            else:
                test_has_thin_back |= thin_back
    assert len(train_back_colors) >= 2
    assert len({frozenset(int(v) for v in np.unique(p["input"]) if v != 0)
                for p in puzzle["train"]}) >= 2
    assert any(set(int(v) for v in np.unique(p["input"]) if v != 0) - train_colors
               for p in puzzle["test"])
    assert not test_has_thin_back or train_has_thin_back


def _fingerprint(puzzle):
    return tuple((split, tuple((p["input"].shape, p["input"].tobytes(),
                                p["output"].shape, p["output"].tobytes())
                               for p in puzzle[split])) for split in ("train", "test"))


def _components(mask):
    cells = set(zip(*np.nonzero(mask)))
    count = 0
    while cells:
        count += 1
        stack = [cells.pop()]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
    return count


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, outputs, palettes, counts = set(), set(), set(), set()
    interior = thin_back = disjoint = four_fragments = 0
    mixed_shapes = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        pairs = puzzle["train"] + puzzle["test"]
        mixed_shapes += len({p["input"].shape for p in pairs}) > 1
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            rects, back, edges = _analyze(inp)
            assert out.shape == (rects[back][2] - rects[back][0],
                                 rects[back][3] - rects[back][1])
            assert np.all(out == back)
            shapes.add(inp.shape)
            outputs.add(out.shape)
            palettes.add(frozenset(rects))
            counts.add(len(rects))
            r0, c0, r1, c1 = rects[back]
            interior += r0 > 0 and c0 > 0 and r1 < inp.shape[0] and c1 < inp.shape[1]
            thin_back += r1 - r0 == 1 or c1 - c0 == 1
            four_fragments += _components(inp == back) >= 4
            fronts = [c for c in rects if c != back]
            disjoint += any(not _overlap(rects[a], rects[b])
                            for i, a in enumerate(fronts) for b in fronts[i + 1:])
    assert len(shapes) >= 20 and len(outputs) >= 15
    assert len(palettes) >= 30 and counts == {2, 3, 4, 5}
    assert mixed_shapes >= 35
    assert interior >= 40 and thin_back >= 8
    assert disjoint >= 8 and four_fragments >= 8
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1),
                 (None, 4), (2, None), (10, None)):
        p = generate_puzzle(12345, a, b)
        validate_generated_puzzle(p)
        assert (a is None or len(p["train"]) == a)
        assert (b is None or len(p["test"]) == b)
        assert _fingerprint(p) == _fingerprint(generate_puzzle(12345, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 3), (3, 4), (4, 4), (2, 4)):
        try:
            generate_puzzle(12345, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count request accepted")
