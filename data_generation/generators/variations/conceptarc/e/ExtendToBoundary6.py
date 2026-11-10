import numpy as np


_SOURCE_INPUTS = {
    ("222222", "222222", "000000", "000000", "000000", "010000"),
    ("000000000", "000000000", "222222222", "222222222", "000000000",
     "000000000", "000000000", "000000000", "000000000", "010000000"),
    ("000000000", "000000000", "222222000", "222222000", "000000000",
     "000000000", "000000000", "000000000", "000000000", "010000000"),
    ("0000000000", "0000000000", "0000000000", "0000000000",
     "0000002200", "0000002200", "0000002200", "0000002200",
     "0000002200", "1001002200"),
    ("000000220", "000000220", "000000220", "000000220",
     "000000220", "000000220", "000000220", "010000220"),
    ("000000000", "000000000", "222222200", "222222200", "000000000",
     "000000000", "000000000", "000000000", "001000000"),
    ("000000000", "000000000", "000222220", "000222220", "000000220",
     "000000220", "000000220", "000000000", "010100010"),
}
_KINDS = ("horizontal", "vertical", "bent", "gap", "double")


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("Illegal pair count")
    if num_test is None:
        upper = 4 if num_train is None else min(4, int(num_train) - 1)
        num_test = int(rng.integers(1, upper + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _dimensions(rng):
    scale = int(rng.integers(0, 4))
    if scale == 0:
        return int(rng.integers(6, 9)), int(rng.integers(6, 9))
    if scale == 1:
        return int(rng.integers(9, 16)), int(rng.integers(9, 16))
    if scale == 2:
        return int(rng.integers(13, 22)), int(rng.integers(6, 12))
    return int(rng.integers(6, 12)), int(rng.integers(13, 22))


def _make_mask(rng, kind, h, w):
    mask = np.zeros((h, w), dtype=bool)
    if kind == "horizontal":
        thick = int(rng.integers(1, min(3, h - 2) + 1))
        length = int(rng.integers(4, w + 1))
        y = int(rng.integers(0, h - thick))
        x = int(rng.integers(0, w - length + 1))
        mask[y:y + thick, x:x + length] = True
    elif kind == "vertical":
        thick = int(rng.integers(1, min(3, w - 2) + 1))
        length = h if rng.random() < 0.22 else int(rng.integers(4, h + 1))
        y = int(rng.integers(0, h - length + 1))
        x = int(rng.integers(1, w - thick + 1))
        mask[y:y + length, x:x + thick] = True
    elif kind == "bent":
        bh = int(rng.integers(4, h + 1))
        bw = int(rng.integers(4, w + 1))
        y = int(rng.integers(0, h - bh + 1))
        x = int(rng.integers(0, w - bw + 1))
        th = int(rng.integers(1, min(3, bh - 2) + 1))
        tw = int(rng.integers(1, min(3, bw - 2) + 1))
        upper = bool(rng.integers(0, 2))
        left = bool(rng.integers(0, 2))
        yr = slice(y, y + th) if upper else slice(y + bh - th, y + bh)
        xr = slice(x, x + tw) if left else slice(x + bw - tw, x + bw)
        mask[yr, x:x + bw] = True
        mask[y:y + bh, xr] = True
    elif kind == "gap":
        thick = int(rng.integers(1, min(3, h - 2) + 1))
        span = int(rng.integers(6, w + 1))
        gap = int(rng.integers(1, min(3, span - 4) + 1))
        left_len = int(rng.integers(2, span - gap - 1))
        y = int(rng.integers(0, h - thick))
        x = int(rng.integers(0, w - span + 1))
        mask[y:y + thick, x:x + left_len] = True
        mask[y:y + thick, x + left_len + gap:x + span] = True
    else:
        # Two separate blockers lie on one potential diagonal. Other
        # diagonals may meet only the second, or pass both.
        x0 = int(rng.integers(0, min(4, w - 4)))
        maximum = min(h - 1, w - 1 - x0)
        if maximum < 4:
            return mask
        d1 = int(rng.integers(1, maximum - 2))
        d2 = int(rng.integers(d1 + 2, maximum + 1))
        r1, c1 = h - 1 - d1, x0 + d1
        r2, c2 = h - 1 - d2, x0 + d2
        mask[r1, c1:c1 + int(rng.integers(1, 3))] = True
        mask[r2, c2:c2 + int(rng.integers(1, 3))] = True
    return mask


def _ray(mask, x):
    h, w = mask.shape
    cells = []
    obstacles = []
    for d in range(1, min(h, w - x)):
        r, c = h - 1 - d, x + d
        if mask[r, c]:
            obstacles.append((r, c))
        elif not obstacles:
            cells.append((r, c))
    return {"start": x, "cells": cells,
            "hit": obstacles[0] if obstacles else None,
            "obstacles": obstacles}


def _gap_bypass(mask, path):
    if path["hit"] is not None:
        return False
    yy, xx = np.where(mask)
    y0, y1, x0, x1 = yy.min(), yy.max(), xx.min(), xx.max()
    return any(y0 <= r <= y1 and x0 < c < x1 for r, c in path["cells"])


def _select_starts(rng, mask, kind, mode, count, allow_short=True):
    paths = [_ray(mask, x) for x in range(mask.shape[1] - 1)
             if not mask[-1, x]]
    if not allow_short:
        paths = [p for p in paths if p["hit"] is None
                 or len(p["cells"]) >= 2]
    hit = [p for p in paths if p["hit"] is not None]
    all_hit = hit
    clear = [p for p in paths if p["hit"] is None]
    long = [p for p in paths if len(p["cells"]) >= 3]
    if kind == "gap" and mode == "mixed":
        clear = [p for p in clear if _gap_bypass(mask, p)]
    if kind == "double" and mode in ("mixed", "hit"):
        hit = [p for p in hit if len(p["obstacles"]) >= 2]
    chosen = []
    def add(pool):
        options = [p for p in pool if p["start"] not in chosen]
        if not options:
            return False
        chosen.append(int(options[int(rng.integers(len(options)))]["start"]))
        return True
    if mode == "mixed":
        if not add(hit) or not add(clear):
            return None
    elif mode == "hit":
        if not add(hit):
            return None
    elif mode == "clear":
        if not add(clear):
            return None
    elif mode == "short":
        short = [p for p in hit if len(p["cells"]) <= 1]
        if not add(short) or not add(long):
            return None
    else:
        raise AssertionError("Unknown outcome")
    allowed = (all_hit if mode == "hit" else clear
               if mode == "clear" else paths)
    while len(chosen) < count:
        if not add(allowed):
            return None
    selected = [p for p in paths if p["start"] in chosen]
    if not any(len(p["cells"]) >= 2 for p in selected):
        return None
    return sorted(chosen)


def _draw_pair(mask, starts, colors):
    bg, ray, block = colors
    inp = np.full(mask.shape, bg, dtype=np.int8)
    inp[mask] = block
    inp[-1, starts] = ray
    out = inp.copy()
    h, w = mask.shape
    for x in starts:
        for d in range(1, min(h, w - x)):
            r, c = h - 1 - d, x + d
            if mask[r, c]:
                break
            out[r, c] = ray
    return {"input": inp, "output": out}


def _pair(rng, kind, mode, colors, allow_short, count=None):
    for _ in range(500):
        h, w = _dimensions(rng)
        mask = _make_mask(rng, kind, h, w)
        if not np.any(mask[:-1]) or np.mean(mask) > 0.38:
            continue
        minimum = 2 if mode in ("mixed", "short") else 1
        maximum = min(1 if kind == "double" and mode == "hit" else 4,
                      w - 1)
        n = (int(rng.integers(minimum, maximum + 1))
             if count is None else count)
        starts = _select_starts(rng, mask, kind, mode, n, allow_short)
        if starts is None:
            continue
        return _draw_pair(mask, starts, colors)
    raise AssertionError(f"Cannot construct {kind}/{mode} pair")


def _key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    nt, nq = _counts(rng, num_train, num_test)
    # The up-right direction and the bottom-row marker convention are
    # shared. The three visible color roles are intentionally local to a pair.
    straight = str(rng.choice(("horizontal", "vertical")))
    remainder = [k for k in _KINDS if k != straight]
    rng.shuffle(remainder)
    train_kinds = [straight, remainder[0]]
    for i in range(2, nt):
        train_kinds.append(str(rng.choice(_KINDS)))
    rng.shuffle(train_kinds)
    test_kinds = [str(rng.choice(list(set(train_kinds)))) for _ in range(nq)]
    if nq >= 2 and len(set(train_kinds)) >= 2:
        test_kinds[:2] = list(rng.choice(
            sorted(set(train_kinds)), size=2, replace=False
        ))
    modes = ["mixed"] + [str(rng.choice(("hit", "clear", "mixed")))
                          for _ in range(nt - 1)]
    rng.shuffle(modes)
    # Geometry and intended outcome are sampled separately. The first
    # witness forces both contact and bypass into every training set.
    modes[0] = "mixed"
    if nt > 2 and "vertical" in train_kinds and rng.random() < 0.45:
        j = train_kinds.index("vertical")
        if j != 0:
            modes[j] = "short"
    planned_counts = [None] * nt
    planned_counts[0] = int(rng.integers(2, 5))
    minimum_second = 2 if modes[1] in ("mixed", "short") else 1
    second_maximum = (1 if train_kinds[1] == "double"
                      and modes[1] == "hit" else 4)
    second_choices = [n for n in range(minimum_second, second_maximum + 1)
                      if n != planned_counts[0]]
    planned_counts[1] = int(rng.choice(second_choices))
    palette = []
    for _ in range(nt + nq):
        while True:
            colors = tuple(int(v) for v in rng.choice(10, 3, replace=False))
            if colors not in palette:
                palette.append(colors)
                break
    puzzle = {"train": [], "test": []}
    used = set()
    train_short = False
    for i, kind in enumerate(train_kinds + test_kinds):
        test = i >= nt
        mode = modes[i] if not test else str(
            rng.choice(("hit", "clear", "mixed"))
        )
        if kind in ("gap", "double") and mode == "clear" and not test:
            mode = "mixed"
        for _ in range(25):
            local = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _pair(local, kind, mode, palette[i],
                         allow_short=(not test or train_short),
                         count=(planned_counts[i] if not test else None))
            key = _key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle["test" if test else "train"].append(pair)
                if not test:
                    analysis = _inspect(pair)
                    train_short |= any(
                        p["hit"] is not None and len(p["cells"]) <= 1
                        for p in analysis["paths"]
                    )
                break
        else:
            raise AssertionError("Could not create a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    unseen = set(zip(*np.where(mask)))
    components = []
    while unseen:
        todo = [unseen.pop()]
        comp = set(todo)
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    todo.append(q)
        components.append(comp)
    return components


def _kind(mask):
    yy, xx = np.where(mask)
    assert len(yy) >= 2
    comps = _components(mask)
    if len(comps) == 2:
        boxes = []
        for comp in comps:
            rr = [p[0] for p in comp]
            cc = [p[1] for p in comp]
            y0, y1, x0, x1 = min(rr), max(rr), min(cc), max(cc)
            assert len(comp) == (y1 - y0 + 1) * (x1 - x0 + 1)
            boxes.append((y0, y1, x0, x1))
        if boxes[0][:2] == boxes[1][:2]:
            assert all(b[3] - b[2] + 1 >= 2 for b in boxes)
            assert boxes[0][1] - boxes[0][0] + 1 <= 3
            return "gap"
        assert boxes[0][:2] != boxes[1][:2]
        return "double"
    assert len(comps) == 1
    h = int(yy.max() - yy.min() + 1)
    w = int(xx.max() - xx.min() + 1)
    if len(yy) == h * w:
        assert min(h, w) <= 3 and max(h, w) >= 4
        return "horizontal" if w > h else "vertical"
    assert h >= 4 and w >= 4
    y0, y1, x0, x1 = int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max())
    for thick_y in range(1, min(3, h - 2) + 1):
        for thick_x in range(1, min(3, w - 2) + 1):
            for top in (True, False):
                for left in (True, False):
                    candidate = np.zeros_like(mask)
                    yr = (slice(y0, y0 + thick_y) if top
                          else slice(y1 - thick_y + 1, y1 + 1))
                    xr = (slice(x0, x0 + thick_x) if left
                          else slice(x1 - thick_x + 1, x1 + 1))
                    candidate[yr, x0:x1 + 1] = True
                    candidate[y0:y1 + 1, xr] = True
                    if np.array_equal(candidate, mask):
                        return "bent"
    raise AssertionError("Invalid bend")


def _inspect(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    assert inp.shape == out.shape
    assert not np.array_equal(inp, out)
    vals, counts = np.unique(inp, return_counts=True)
    assert len(vals) == 3
    bg = int(vals[np.argmax(counts)])
    other = [int(v) for v in vals if v != bg]
    candidates = []
    for ray, block in ((other[0], other[1]), (other[1], other[0])):
        if np.any(inp[:-1] == ray):
            continue
        starts = np.where(inp[-1] == ray)[0].tolist()
        if not 1 <= len(starts) <= 4:
            continue
        mask = inp == block
        if not np.any(mask[:-1]):
            continue
        expected = _draw_pair(mask, starts, (bg, ray, block))["output"]
        if np.array_equal(expected, out):
            candidates.append((ray, block, starts, mask))
    assert len(candidates) == 1
    ray, block, starts, mask = candidates[0]
    assert np.mean(mask) <= 0.38
    kind = _kind(mask)
    paths = [_ray(mask, x) for x in starts]
    assert any(len(p["cells"]) >= 2 for p in paths)
    source_form = tuple("".join(str(int(v)) for v in row) for row in inp)
    assert source_form not in _SOURCE_INPUTS
    return {"kind": kind, "colors": (bg, ray, block),
            "paths": paths, "shape": inp.shape,
            "mask": mask, "starts": starts}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    used = set()
    train, test = [], []
    for split, analyses in (("train", train), ("test", test)):
        for pair in puzzle[split]:
            a = _inspect(pair)
            key = _key(pair["input"])
            assert key not in used
            used.add(key)
            analyses.append(a)
    assert len({a["colors"] for a in train}) >= 2
    assert len({a["kind"] for a in train}) >= 2
    assert any(p["hit"] is not None for a in train for p in a["paths"])
    assert any(p["hit"] is None for a in train for p in a["paths"])
    train_kinds = {a["kind"] for a in train}
    assert all(a["kind"] in train_kinds for a in test)
    train_has_short = any(
        p["hit"] is not None and len(p["cells"]) <= 1
        for a in train for p in a["paths"]
    )
    if any(p["hit"] is not None and len(p["cells"]) <= 1
           for a in test for p in a["paths"]):
        assert train_has_short
    for kind in ("gap", "double"):
        if kind in {a["kind"] for a in test}:
            assert any(a["kind"] == kind and
                       any(p["hit"] is not None for p in a["paths"])
                       for a in train)


def _same(a, b):
    for split in ("train", "test"):
        if len(a[split]) != len(b[split]):
            return False
        for p, q in zip(a[split], b[split]):
            for key in ("input", "output"):
                if p[key].dtype != q[key].dtype or not np.array_equal(
                    p[key], q[key]
                ):
                    return False
    return True


def self_test() -> None:
    train_counts, test_counts = set(), set()
    kinds, shapes, signatures = set(), set(), set()
    outcomes, bend_orientations, start_counts = set(), set(), set()
    h_thickness, v_thickness = set(), set()
    geometry = set()
    edge_exits = set()
    color_changes = short_count = mixed_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        analyses = [_inspect(p) for p in puzzle["train"] + puzzle["test"]]
        kinds.update(a["kind"] for a in analyses)
        shapes.update(a["shape"] for a in analyses)
        assert len({a["shape"] for a in analyses}) >= 2
        assert len({len(a["paths"]) for a in analyses}) >= 2
        color_changes += len({a["colors"] for a in analyses}) > 1
        short_count += any(p["hit"] is not None and len(p["cells"]) <= 1
                           for a in analyses for p in a["paths"])
        mixed_count += any(
            any(p["hit"] is not None for p in a["paths"])
            and any(p["hit"] is None for p in a["paths"])
            for a in analyses
        )
        for a in analyses:
            kind, mask, paths = a["kind"], a["mask"], a["paths"]
            h, w = mask.shape
            yy, xx = np.where(mask)
            y0, y1 = int(yy.min()), int(yy.max())
            x0, x1 = int(xx.min()), int(xx.max())
            start_counts.add(len(paths))
            hits = sum(p["hit"] is not None for p in paths)
            mode = "mixed" if 0 < hits < len(paths) else (
                "hit" if hits else "clear"
            )
            outcomes.add((kind, mode))
            geometry.add("small" if h <= 8 and w <= 8 else
                         "large" if max(h, w) >= 15 else "medium")
            if kind == "horizontal":
                h_thickness.add(y1 - y0 + 1)
                geometry.add("horizontal_left" if x0 == 0 else
                             "horizontal_interior")
            elif kind == "vertical":
                v_thickness.add(x1 - x0 + 1)
                geometry.add("vertical_bottom" if y1 == h - 1 else
                             "vertical_nonbottom")
                if y0 == 0 and y1 == h - 1:
                    geometry.add("full_vertical")
            elif kind == "bent":
                arm = "top" if np.all(mask[y0, x0:x1 + 1]) else "bottom"
                leg = "left" if np.all(mask[y0:y1 + 1, x0]) else "right"
                bend_orientations.add((arm, leg))
            elif kind == "gap":
                if any(_gap_bypass(mask, p) for p in paths):
                    geometry.add("gap_bypass")
            elif any(len(p["obstacles"]) >= 2 for p in paths):
                geometry.add("double_first_hit")
            for p in paths:
                if p["hit"] is None:
                    edge_exits.add("top" if p["start"] + h <= w - 1
                                   else "right")
        signatures.add(tuple(_key(p["input"]) for p in
                             puzzle["train"] + puzzle["test"]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert kinds == set(_KINDS) and len(shapes) >= 30
    assert len(signatures) == 42 and color_changes == 42
    assert short_count >= 3 and mixed_count >= 30
    assert start_counts == {1, 2, 3, 4}
    assert outcomes == {(kind, mode) for kind in _KINDS
                        for mode in ("hit", "clear", "mixed")}
    assert bend_orientations == {(arm, leg)
                                 for arm in ("top", "bottom")
                                 for leg in ("left", "right")}
    assert h_thickness == v_thickness == {1, 2, 3}
    assert {"small", "medium", "large", "horizontal_left",
            "horizontal_interior", "vertical_bottom",
            "vertical_nonbottom", "full_vertical", "gap_bypass",
            "double_first_hit"} <= geometry
    assert edge_exits == {"top", "right"}
    for seed, nt, nq in (
        (100, 2, 1), (101, 10, 1), (102, 10, 4),
        (103, None, 1), (104, None, 4),
        (105, 2, None), (106, 10, None),
    ):
        puzzle = generate_puzzle(seed, nt, nq)
        assert _same(puzzle, generate_puzzle(seed, nt, nq))
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if nq is not None:
            assert len(puzzle["test"]) == nq
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(9, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
