"""Translate a source's tight crop into the matching hollow frame."""

import hashlib
import numpy as np

_SOURCE_HASHES = frozenset({
    "804002ed0a791e0e57a3eba7d973acb84b816315cf140649d9619bb4417abbd5",
    "e5817c8bbe77eb0f681f9fb666ce38ec88e82fbdaa6325b96ce8e65d471653db",
    "402d95f6815ba742214293c28f68ad4ce34a5dc7d1e1186e7d3f9c8ca7e463af",
    "4e0473a1a987eab045c99430e4d66d7d1e149f2c1334266ee0612181b5416199",
    "830f94cf25102d0f2c5a0657007c993eb7efa9e3ae916",
})


def _counts(rng, num_train, num_test):
    for n, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if n is not None and (isinstance(n, (bool, np.bool_)) or
                              not isinstance(n, (int, np.integer)) or not lo <= n <= hi):
            raise ValueError("Illegal pair count")
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("Training count must exceed test count")
    return int(num_train), int(num_test)


def _pattern(rng, h, w, bg, colors, motif):
    rr, cc = np.indices((h, w))
    if motif == "solid":
        mask = np.ones((h, w), bool)
    elif motif == "texture":
        mask = rng.random((h, w)) < rng.uniform(.3, .75)
    elif motif == "diamond":
        distance = abs(rr - (h - 1) / 2) / ((h - 1) / 2) + abs(cc - (w - 1) / 2) / ((w - 1) / 2)
        mask = distance <= rng.uniform(.9, 1.35)
        if rng.random() < .5:
            mask &= distance >= rng.uniform(.12, .55)
    elif motif == "cross":
        mask = (abs(rr - int(rng.integers(0, h))) <= int(rng.integers(0, 2))) | (abs(cc - int(rng.integers(0, w))) <= int(rng.integers(0, 2)))
    elif motif == "diagonals":
        diagonal = np.rint(rr * (w - 1) / (h - 1)).astype(int)
        mask = (cc == diagonal) | (cc == w - 1 - diagonal)
    elif motif == "steps":
        mask = cc <= np.rint(rr * (w - 1) / (h - 1))
        if rng.random() < .5:
            mask = np.fliplr(mask)
        if rng.random() < .5:
            mask = np.flipud(mask)
    elif motif == "broken_ring":
        mask = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
        side = int(rng.integers(0, 4))
        if side == 0:
            mask[0, int(rng.integers(1, w - 1))] = False
        elif side == 1:
            mask[-1, int(rng.integers(1, w - 1))] = False
        elif side == 2:
            mask[int(rng.integers(1, h - 1)), 0] = False
        else:
            mask[int(rng.integers(1, h - 1)), -1] = False
        if rng.random() < .6:
            mask[1:-1, 1:-1] = rng.random((h - 2, w - 2)) < .3
    else:  # islands and short strokes
        mask = np.zeros((h, w), bool)
        for _ in range(int(rng.integers(3, 7))):
            r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
            if rng.random() < .5:
                mask[r, c:min(w, c + int(rng.integers(1, 4)))] = True
            else:
                mask[r:min(h, r + int(rng.integers(1, 4))), c] = True
    # Extremal points vary; no prescribed corner or center is always occupied.
    if not mask[0].any():
        mask[0, int(rng.integers(0, w))] = True
    if not mask[-1].any():
        mask[-1, int(rng.integers(0, w))] = True
    if not mask[:, 0].any():
        mask[int(rng.integers(0, h)), 0] = True
    if not mask[:, -1].any():
        mask[int(rng.integers(0, h)), -1] = True
    positions = np.argwhere(mask)
    pattern = np.full((h, w), bg, dtype=np.int8)
    style = int(rng.integers(0, 5))
    if style == 0:
        index = rng.integers(0, len(colors), size=(h, w))
    elif style == 1:
        index = rr % len(colors)
    elif style == 2:
        index = cc % len(colors)
    elif style == 3:
        index = np.minimum.reduce((rr, h - 1 - rr, cc, w - 1 - cc)) % len(colors)
    else:
        index = ((rr >= h // 2).astype(int) + 2 * (cc >= w // 2)) % len(colors)
    for r, c in positions:
        pattern[r, c] = colors[int(index[r, c])]
    for (r, c), color in zip(positions[rng.permutation(len(positions))[:len(colors)]], colors):
        pattern[r, c] = color
    return pattern


def _colors(rng, bg, frame, overlap, new=()):
    count = int(rng.choice((1, 2, 3, 4), p=(.22, .35, .30, .13)))
    pool = [c for c in range(10) if c not in (bg, frame)]
    colors = [frame] if overlap else []
    new_pool = [c for c in new if c in pool]
    if new_pool:
        chosen = int(rng.choice(new_pool))
        colors.append(chosen)
        pool.remove(chosen)
    count = max(count, len(colors))
    colors += [int(c) for c in rng.choice(pool, size=count - len(colors), replace=False)]
    rng.shuffle(colors)
    return tuple(colors)


def _make_pair(rng, bg, frame, regime, plan, new=(), force_frame_only=False):
    aspect, density = plan
    h = int(rng.integers(3, 10))
    w = h if aspect == "square" else int(rng.integers(3, 11))
    if aspect == "rectangle" and h == w:
        w += 1 if w < 10 else -1
    overlap = force_frame_only or rng.random() < (.35 if regime == "source_train" else .55)
    colors = (frame,) if force_frame_only else _colors(rng, bg, frame, overlap, new)
    motifs = ("texture", "diamond", "cross", "diagonals", "steps", "broken_ring", "islands")
    motif = "solid" if density == "dense" or (density == "any" and rng.random() < .35) else str(rng.choice(motifs))
    pat = _pattern(rng, h, w, bg, colors, motif)
    if density == "sparse" and np.all(pat != bg):
        pat[int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))] = bg
    directions = ("above", "below", "left", "right")
    probabilities = (.35, .17, .30, .18) if regime == "source_train" else (.26, .22, .29, .23)
    direction = str(rng.choice(directions, p=probabilities))
    gap = int(rng.choice((0, 1, 2, 3), p=(.24, .34, .28, .14)))
    if overlap and gap == 0:
        gap = int(rng.integers(1, 4))
    top, left = int(rng.integers(0, 4)), int(rng.integers(0, 4))
    if direction in ("above", "below"):
        offset = int(rng.integers(-w // 2, w + 1))
        sc, fc = left - min(0, offset), left + offset - min(0, offset)
        sr, fr = (top, top + h + gap) if direction == "above" else (top + h + 2 + gap, top)
    else:
        offset = int(rng.integers(-h // 2, h + 1))
        sr, fr = top - min(0, offset), top + offset - min(0, offset)
        sc, fc = (left, left + w + gap) if direction == "left" else (left + w + 2 + gap, left)
    gh = max(sr + h, fr + h + 2) + int(rng.integers(0, 4))
    gw = max(sc + w, fc + w + 2) + int(rng.integers(0, 4))
    assert gh <= 30 and gw <= 30
    inp = np.full((gh, gw), bg, np.int8)
    inp[sr:sr + h, sc:sc + w] = pat
    inp[fr, fc:fc + w + 2] = frame
    inp[fr + h + 1, fc:fc + w + 2] = frame
    inp[fr:fr + h + 2, fc] = frame
    inp[fr:fr + h + 2, fc + w + 1] = frame
    out = inp.copy()
    out[fr + 1:fr + h + 1, fc + 1:fc + w + 1] = pat
    return inp, out


def _frames(grid, bg):
    height, width = grid.shape
    result = []
    for r in range(height - 4):
        for c in range(width - 4):
            color = int(grid[r, c])
            if color == bg:
                continue
            for right in range(c + 4, min(width, c + 12)):
                if not np.all(grid[r, c:right + 1] == color):
                    break
                for bottom in range(r + 4, min(height, r + 12)):
                    if (grid[bottom, c] == color and
                        np.all(grid[r:bottom + 1, c] == color) and
                        np.all(grid[r:bottom + 1, right] == color) and
                        np.all(grid[bottom, c:right + 1] == color) and
                        np.all(grid[r + 1:bottom, c + 1:right] == bg)):
                        result.append((color, (r, c, bottom, right)))
    return result


def _inspect(inp, out):
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    frames = _frames(inp, bg)
    assert len(frames) == 1
    frame, (fr0, fc0, fr1, fc1) = frames[0]
    outside = inp.copy()
    outside[fr0:fr1 + 1, fc0:fc1 + 1] = bg
    points = np.argwhere(outside != bg)
    assert len(points) > 0
    (sr0, sc0), (sr1, sc1) = points.min(0), points.max(0)
    sr0, sc0, sr1, sc1 = map(int, (sr0, sc0, sr1, sc1))
    assert (sr1 - sr0 + 1, sc1 - sc0 + 1) == (fr1 - fr0 - 1, fc1 - fc0 - 1)
    assert sr1 < fr0 or sr0 > fr1 or sc1 < fc0 or sc0 > fc1
    if sr1 + 1 == fr0:
        assert all(inp[sr1, c] != frame for c in range(max(sc0, fc0), min(sc1, fc1) + 1))
    if fr1 + 1 == sr0:
        assert all(inp[sr0, c] != frame for c in range(max(sc0, fc0), min(sc1, fc1) + 1))
    if sc1 + 1 == fc0:
        assert all(inp[r, sc1] != frame for r in range(max(sr0, fr0), min(sr1, fr1) + 1))
    if fc1 + 1 == sc0:
        assert all(inp[r, sc0] != frame for r in range(max(sr0, fr0), min(sr1, fr1) + 1))
    pat = inp[sr0:sr1 + 1, sc0:sc1 + 1]
    expected = inp.copy()
    expected[fr0 + 1:fr1, fc0 + 1:fc1] = pat
    assert np.array_equal(out, expected) and not np.array_equal(inp, out)
    colors = frozenset(int(c) for c in np.unique(pat) if c != bg)
    assert colors
    if sr1 < fr0:
        direction, gap = "above", fr0 - sr1 - 1
    elif sr0 > fr1:
        direction, gap = "below", sr0 - fr1 - 1
    elif sc1 < fc0:
        direction, gap = "left", fc0 - sc1 - 1
    else:
        direction, gap = "right", sc0 - fc1 - 1
    return {"background": bg, "frame": frame, "colors": colors, "shape": pat.shape,
            "direction": direction, "gap": gap, "dense": bool(np.all(pat != bg)),
            "pattern": pat, "source_box": (sr0, sc0, sr1, sc1),
            "frame_box": (fr0, fc0, fr1, fc1)}


def _hash(inp, out):
    return hashlib.sha256(str(inp.shape).encode() + inp.tobytes() + out.tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    bg = int(rng.integers(0, 10))
    frame = int(rng.choice([c for c in range(10) if c != bg]))
    first_square, first_dense = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
    plans = []
    for i in range(num_train):
        if i < 2:
            aspect = "square" if first_square == (i == 0) else "rectangle"
            density = "dense" if first_dense == (i == 0) else "sparse"
        else:
            aspect, density = str(rng.choice(("square", "rectangle"))), "any"
        plans.append((aspect, density))
    test_plans = [(str(rng.choice(("square", "rectangle"), p=(.38, .62))), "any")
                  for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    seen, train_colors = set(), set()
    for split, pair_plans in (("train", plans), ("test", test_plans)):
        regime = "source_train" if split == "train" else "target_test"
        for plan in pair_plans:
            for _ in range(160):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                new = tuple(c for c in range(10) if c not in train_colors and c not in (bg, frame))
                if split == "train" or pair_rng.random() >= .55:
                    new = ()
                try:
                    frame_only = split == "test" and pair_rng.random() < .17
                    inp, out = _make_pair(pair_rng, bg, frame, regime, plan, new,
                                          frame_only)
                    info = _inspect(inp, out)
                except AssertionError:
                    continue
                key = (inp.shape, inp.tobytes())
                if key in seen or _hash(inp, out) in _SOURCE_HASHES:
                    continue
                seen.add(key)
                puzzle[split].append({"input": inp, "output": out})
                if split == "train":
                    train_colors.update(info["colors"])
                break
            else:
                raise RuntimeError("Could not construct a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            key = (inp.shape, inp.tobytes())
            pair_key = (key, out.shape, out.tobytes())
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert _hash(inp, out) not in _SOURCE_HASHES
            infos[split].append(_inspect(inp, out))
    all_infos = infos["train"] + infos["test"]
    assert len({item["background"] for item in all_infos}) == 1
    assert len({item["frame"] for item in all_infos}) == 1
    assert {h == w for h, w in (item["shape"] for item in infos["train"])} == {True, False}
    assert {item["dense"] for item in infos["train"]} == {True, False}


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p[k], q[k]) for p, q in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    tc, sc, frames, backgrounds, shapes, canvases = (set() for _ in range(6))
    motifs, layouts, palettes, first_inputs = (set() for _ in range(4))
    dense = sparse = overlap = nonoverlap = touch = new_test = 0
    within_shapes = within_palettes = within_layouts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        tc.add(len(puzzle["train"]))
        sc.add(len(puzzle["test"]))
        first = puzzle["train"][0]["input"]
        first_inputs.add((first.shape, first.tobytes()))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect(pair["input"], pair["output"])
                local.append(info)
                frames.add(info["frame"])
                backgrounds.add(info["background"])
                shapes.add(info["shape"])
                canvases.add(pair["input"].shape)
                motifs.add((info["shape"], info["pattern"].tobytes()))
                layouts.add((info["direction"], info["gap"], info["source_box"], info["frame_box"]))
                palettes.add(info["colors"])
                dense += info["dense"]
                sparse += not info["dense"]
                overlap += info["frame"] in info["colors"]
                nonoverlap += info["frame"] not in info["colors"]
                touch += info["gap"] == 0
        within_shapes += len({x["shape"] for x in local}) > 1
        within_palettes += len({x["colors"] for x in local}) > 1
        within_layouts += len({x["direction"] for x in local}) > 1
        trained = set().union(*(x["colors"] for x in local[:len(puzzle["train"])]))
        new_test += any(x["colors"] - trained for x in local[len(puzzle["train"]):])
    assert len(tc) >= 4 and len(sc) >= 3 and len(first_inputs) >= 40
    assert len(frames) >= 7 and len(backgrounds) >= 7
    assert len(shapes) >= 25 and len(canvases) >= 40
    assert len(motifs) >= 100 and len(layouts) >= 100 and len(palettes) >= 30
    assert dense >= 30 and sparse >= 100 and overlap >= 50 and nonoverlap >= 100
    assert touch >= 15 and new_test >= 10
    assert within_shapes >= 40 and within_palettes >= 35 and within_layouts >= 30
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(1000 + (nt or 0) * 10 + (ns or 0), nt, ns)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
