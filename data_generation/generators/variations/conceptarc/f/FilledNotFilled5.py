import numpy as np


_FAMILIES = ("rectangle", "diamond", "trapezoid", "jagged")
_SOURCE_INPUTS = {
    tuple("""00000000000
03333333330
03000000030
03000000030
03000000030
03000000030
03000000030
03000000030
03333333330
00000000000
00000000000""".split()),
    tuple("""000000000000
000002000000
000022200000
000222220000
002222222000
000222220000
000022200000
000002000000
000000000000
000000000000""".split()),
    tuple("""000444000
004000400
040000040
400000004
444444444
000000000""".split()),
    tuple("""00000000000
06666666660
06666666660
06666666660
06666666660
06666666660
06666666660
06666666660
06666666660
06666666660
00000000000""".split()),
    tuple("""000000000000
001111111100
001000000100
001000000100
001100000100
000100000100
011100000100
010000000100
011000000100
001100000100
000111111100
000000000000""".split()),
}


def _choose_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError("Pair counts must be integers")
            if not low <= value <= high:
                raise ValueError("Pair count out of range")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("Training count must exceed test count")
    return int(num_train), int(num_test)


def _family(rng, test=False):
    weights = (0.28, 0.28, 0.25, 0.19) if not test else (
        0.13, 0.17, 0.34, 0.36
    )
    return str(rng.choice(_FAMILIES, p=weights))


def _sample_spans(rng, family):
    if family == "rectangle":
        h = int(rng.integers(4, 16))
        w = int(rng.integers(4, 19))
        if rng.random() < 0.28:
            w = h
        return [(0, w - 1)] * h

    if family == "diamond":
        radius = int(rng.integers(2, 7))
        tip = int(rng.choice((1, 1, 1, 3)))
        return [
            (abs(y - radius), 2 * radius - abs(y - radius) + tip - 1)
            for y in range(2 * radius + 1)
        ]

    if family == "trapezoid":
        h = int(rng.integers(4, 13))
        top_w = int(rng.integers(3, 12))
        left_growth = int(rng.integers(0, min(5, h - 1) + 1))
        right_growth = int(rng.integers(0, min(5, h - 1) + 1))
        if left_growth + right_growth < 2:
            right_growth = 2
        spans = [
            (
                left_growth - (left_growth * y) // (h - 1),
                left_growth + top_w - 1
                + (right_growth * y) // (h - 1),
            )
            for y in range(h)
        ]
        if rng.random() < 0.5:
            spans.reverse()
        if rng.random() < 0.5:
            width = max(r for _, r in spans) + 1
            spans = [(width - 1 - r, width - 1 - l) for l, r in spans]
        return spans

    if family == "jagged":
        h = int(rng.integers(7, 16))
        width = int(rng.integers(9, 17))
        left = [int(rng.integers(1, 5))]
        right = [left[0] + width - 1]
        variable_right = rng.random() < 0.58
        for _ in range(h - 1):
            l = int(np.clip(left[-1] + rng.integers(-2, 3), 0, 6))
            if variable_right:
                r = int(np.clip(right[-1] + rng.integers(-2, 3),
                                width + 2, width + 8))
            else:
                r = right[-1]
            right.append(max(r, l + 7))
            left.append(l)
        if max(left) - min(left) < 2:
            left[h // 2] = min(6, left[h // 2] + 2)
        spans = list(zip(left, right))
        if rng.random() < 0.5:
            width = max(right) + 1
            spans = [(width - 1 - r, width - 1 - l) for l, r in spans]
        return spans

    raise ValueError("Unknown family")


def _shape_masks(rng, family):
    spans = _sample_spans(rng, family)
    offset = min(l for l, _ in spans)
    width = max(r for _, r in spans) - offset + 1
    h = len(spans)
    filled = np.zeros((h, width), dtype=bool)
    outline = np.zeros_like(filled)
    for y, (left, right) in enumerate(spans):
        l, r = left - offset, right - offset
        filled[y, l:r + 1] = True
        if y == 0 or y == h - 1:
            outline[y, l:r + 1] = True
        elif family == "jagged":
            # The neighboring endpoints extend the contour over each jog.
            outer_left = max(spans[y - 1][0], left, spans[y + 1][0]) - offset
            outer_right = min(spans[y - 1][1], right, spans[y + 1][1]) - offset
            outline[y, l:outer_left + 1] = True
            outline[y, outer_right:r + 1] = True
        else:
            outline[y, l] = True
            outline[y, r] = True
    if family in ("rectangle", "diamond", "trapezoid"):
        turns = int(rng.choice((0, 0, 0, 1, 2, 3)))
        filled = np.rot90(filled, turns)
        outline = np.rot90(outline, turns)
    return filled, outline


def _connected(mask, diagonal=False):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(int(v) for v in points[0])
    seen = {start}
    queue = [start]
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        moves += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for y, x in queue:
        for dy, dx in moves:
            yy, xx = y + dy, x + dx
            if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                    and mask[yy, xx] and (yy, xx) not in seen):
                seen.add((yy, xx))
                queue.append((yy, xx))
    return len(seen) == len(points)


def _outside(outline):
    h, w = outline.shape
    seen = np.zeros((h, w), dtype=bool)
    queue = []
    for y in range(h):
        for x in (0, w - 1):
            if not outline[y, x] and not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    for x in range(w):
        for y in (0, h - 1):
            if not outline[y, x] and not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    for y, x in queue:
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if (0 <= yy < h and 0 <= xx < w and not outline[yy, xx]
                    and not seen[yy, xx]):
                seen[yy, xx] = True
                queue.append((yy, xx))
    return seen


def _grid_rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _turns(side):
    changes = np.diff(side)
    signs = np.sign(changes[changes != 0])
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _sample_padding(rng, length):
    room = 30 - length
    before = int(rng.integers(0, min(5, room) + 1))
    after = int(rng.integers(0, min(5, room - before) + 1))
    return before, after


def _make_pair(rng, family, state, background, foreground):
    for _ in range(64):
        filled, outline = _shape_masks(rng, family)
        if filled.shape[0] > 26 or filled.shape[1] > 26:
            continue
        top, bottom = _sample_padding(rng, filled.shape[0])
        left, right = _sample_padding(rng, filled.shape[1])
        if family == "trapezoid" and rng.random() < 0.22:
            left = right = 0
            if top == bottom == 0:
                top = 1
        h = top + filled.shape[0] + bottom
        w = left + filled.shape[1] + right
        full = np.zeros((h, w), dtype=bool)
        edge = np.zeros_like(full)
        full[top:top + filled.shape[0], left:left + filled.shape[1]] = filled
        edge[top:top + outline.shape[0], left:left + outline.shape[1]] = outline
        if full[0, 0] or full[0, -1] or full[-1, 0] or full[-1, -1]:
            continue
        if not np.any(full & ~edge):
            continue
        inp_mask, out_mask = (edge, full) if state == "hollow" else (
            full, edge
        )
        inp = np.where(inp_mask, foreground, background).astype(np.int8)
        out = np.where(out_mask, foreground, background).astype(np.int8)
        try:
            info = _analyze_pair(inp, out)
        except AssertionError:
            continue
        if family == "jagged":
            jumps = [abs(b - a) for side in (info["lefts"], info["rights"])
                     for a, b in zip(side, side[1:])]
            if (max(jumps) < 2 or not info["has_ledge"]
                    or max(_turns(info["lefts"]),
                           _turns(info["rights"])) < 2):
                continue
        return {"input": inp, "output": out}
    raise RuntimeError("Could not construct a valid shape")


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    h, w = inp.shape
    bg = int(inp[0, 0])
    assert all(int(grid[y, x]) == bg for grid in (inp, out)
               for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    colors = np.unique(inp)
    assert len(colors) == 2 and bg in colors
    assert np.array_equal(np.unique(out), colors)
    fg = int(colors[0] if int(colors[1]) == bg else colors[1])
    a, b = inp == fg, out == fg
    gained, lost = b & ~a, a & ~b
    assert np.any(gained) != np.any(lost)
    state = "hollow" if np.any(gained) else "solid"
    full = a | b
    edge = a & b
    interior = full & ~edge
    assert np.any(interior) and _connected(interior)
    assert _connected(full) and _connected(edge, diagonal=True)
    assert not np.any(interior & _outside(edge))
    rows = np.flatnonzero(np.any(full, axis=1))
    assert len(rows) >= 4
    assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))
    lefts, rights = [], []
    for y in rows:
        xs = np.flatnonzero(full[y])
        assert np.all(full[y, xs[0]:xs[-1] + 1])
        lefts.append(int(xs[0]))
        rights.append(int(xs[-1]))
    # Every cell exposed to the exterior along a cardinal direction stays.
    core = full.copy()
    core[1:, :] &= full[:-1, :]
    core[:-1, :] &= full[1:, :]
    core[:, 1:] &= full[:, :-1]
    core[:, :-1] &= full[:, 1:]
    core[0, :] = False
    core[-1, :] = False
    core[:, 0] = False
    core[:, -1] = False
    assert np.all(edge | ~full | core)
    # Extra ledge cells must still be immediately next to the contour.
    contour = full & ~core
    near = contour.copy()
    near[1:, :] |= contour[:-1, :]
    near[:-1, :] |= contour[1:, :]
    near[:, 1:] |= contour[:, :-1]
    near[:, :-1] |= contour[:, 1:]
    assert np.all(~edge | near)
    return {
        "state": state,
        "background": bg,
        "foreground": fg,
        "full": full,
        "edge": edge,
        "interior": interior,
        "has_ledge": bool(np.any(edge & core)),
        "lefts": tuple(lefts),
        "rights": tuple(rights),
        "margins": (
            int(rows[0]), int(h - 1 - rows[-1]),
            min(lefts), int(w - 1 - max(rights)),
        ),
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)

    states = ["hollow", "solid"]
    rng.shuffle(states)
    states += [str(rng.choice(("hollow", "solid")))
               for _ in range(num_train - 2)]
    if rng.random() < 0.22:
        kinds = ["rectangle", "diamond"]
        states[:2] = ["hollow", "solid"]
    else:
        first = _family(rng)
        second = first if rng.random() < 0.28 else _family(rng)
        if second == first and rng.random() < 0.7:
            second = str(rng.choice([k for k in _FAMILIES if k != first]))
        kinds = [first, second]
    for i in range(2, num_train):
        if i == 2 and rng.random() < 0.45:
            kinds.append(kinds[0])
            states[i] = "solid" if states[0] == "hollow" else "hollow"
        else:
            kinds.append(_family(rng))

    test_kinds = [_family(rng, test=True) for _ in range(num_test)]
    test_states = [str(rng.choice(("hollow", "solid")))
                   for _ in range(num_test)]
    if num_test >= 2 and all(k in ("rectangle", "diamond") for k in test_kinds):
        test_kinds[0] = str(rng.choice(("trapezoid", "jagged")))
    if rng.random() < 0.38:
        index = int(rng.integers(0, num_train))
        test_kinds[-1] = kinds[index]
        test_states[-1] = "solid" if states[index] == "hollow" else "hollow"
    if any(k == "jagged" and s == "solid"
           for k, s in zip(test_kinds, test_states)):
        if not any(k == "jagged" and s == "solid"
                   for k, s in zip(kinds, states)):
            solid_index = states.index("solid")
            kinds[solid_index] = "jagged"

    preferred_background = int(rng.integers(0, 10))
    reused_foreground = int(rng.integers(0, 10))
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, family_list, state_list in (
        ("train", kinds, states), ("test", test_kinds, test_states)
    ):
        for i, (family, state) in enumerate(zip(family_list, state_list)):
            for _ in range(128):
                bg = (preferred_background if rng.random() < 0.42
                      else int(rng.integers(0, 10)))
                if (regime == "train" and i < 2
                        and rng.random() < 0.58
                        and reused_foreground != bg):
                    fg = reused_foreground
                else:
                    fg = int(rng.integers(0, 9))
                    if fg >= bg:
                        fg += 1
                try:
                    pair = _make_pair(rng, family, state, bg, fg)
                except RuntimeError:
                    continue
                inp = pair["input"]
                key = (inp.shape, inp.tobytes())
                if key not in seen and _grid_rows(inp) not in _SOURCE_INPUTS:
                    seen.add(key)
                    puzzle[regime].append(pair)
                    break
            else:
                raise RuntimeError("Could not sample a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    train_states = set()
    solid_ledge_train = False
    solid_ledge_test = False
    for regime, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = (inp.shape, inp.astype(np.uint8).tobytes())
            pair_key = (key, out.shape, out.astype(np.uint8).tobytes())
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _grid_rows(inp) not in _SOURCE_INPUTS
            info = _analyze_pair(inp, out)
            if regime == "train":
                train_states.add(info["state"])
                solid_ledge_train |= (info["state"] == "solid"
                                      and info["has_ledge"])
            else:
                solid_ledge_test |= (info["state"] == "solid"
                                     and info["has_ledge"])
    assert train_states == {"hollow", "solid"}
    assert not solid_ledge_test or solid_ledge_train


def _fingerprint(puzzle):
    return tuple(
        (name, pair["input"].shape, pair["input"].tobytes(),
         pair["output"].shape, pair["output"].tobytes())
        for name in ("train", "test") for pair in puzzle[name]
    )


def self_test() -> None:
    train_counts, test_counts, fingerprints = set(), set(), set()
    dimensions, silhouettes, colors, margin_values = set(), set(), set(), set()
    states_by_profile = {}
    nonzero_backgrounds = 0
    zero_foregrounds = 0
    changed_background_puzzles = 0
    edge_pairs = 0
    both_side_contact = 0
    narrow_margin_pairs = 0
    wide_margin_pairs = 0
    ledge_pairs = 0
    ledge_connected_pairs = 0
    up_tapers = 0
    down_tapers = 0
    horizontal_tapers = 0
    left_only = 0
    right_only = 0
    both_sides = 0
    rough_nonzero_edge = 0
    zero_foreground_edge = 0
    same_profile_puzzles = 0
    familiar_reversed_tests = 0
    opposite_color_puzzles = 0
    difficult_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_fingerprint(puzzle))
        infos = {regime: [_analyze_pair(p["input"], p["output"])
                          for p in puzzle[regime]]
                 for regime in ("train", "test")}
        assert {x["state"] for x in infos["train"]} == {"hollow", "solid"}
        if len({x["background"] for x in infos["train"] + infos["test"]}) > 1:
            changed_background_puzzles += 1
        if any(a["foreground"] == b["foreground"] and
               a["state"] != b["state"]
               for a in infos["train"] for b in infos["train"]):
            opposite_color_puzzles += 1
        local_profiles = {"train": [], "test": []}
        for regime in ("train", "test"):
            for pair, info in zip(puzzle[regime], infos[regime]):
                inp = pair["input"]
                full, edge = info["full"], info["edge"]
                ys, xs = np.where(full)
                crop = full[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                silhouettes.add((crop.shape, crop.tobytes()))
                dimensions.add(inp.shape)
                colors.add((info["background"], info["foreground"]))
                margin_values.update(info["margins"])
                nonzero_backgrounds += info["background"] != 0
                zero_foregrounds += info["foreground"] == 0
                edge_pairs += 0 in info["margins"]
                both_side_contact += (info["margins"][2] == 0
                                      and info["margins"][3] == 0)
                narrow_margin_pairs += 1 in info["margins"]
                wide_margin_pairs += any(v >= 4 for v in info["margins"])
                zero_foreground_edge += (info["foreground"] == 0
                                         and 0 in info["margins"])
                lefts, rights = info["lefts"], info["rights"]
                large_jump = any(abs(a - b) >= 2 for seq in (lefts, rights)
                                 for a, b in zip(seq, seq[1:]))
                rough = (large_jump and
                         max(_turns(lefts), _turns(rights)) >= 2)
                ledge_pairs += info["has_ledge"]
                ledge_connected_pairs += (info["has_ledge"]
                                          and _connected(edge))
                rough_nonzero_edge += (rough and info["background"] != 0
                                       and 0 in info["margins"])
                widths = [r - l + 1 for l, r in zip(lefts, rights)]
                up_tapers += widths[0] > widths[-1]
                down_tapers += widths[-1] > widths[0]
                horizontal_tapers += (widths[0] == widths[-1]
                                      and len(set(lefts)) > 1
                                      and len(set(rights)) > 1 and not rough)
                left_only += len(set(lefts)) > 1 and len(set(rights)) == 1
                right_only += len(set(lefts)) == 1 and len(set(rights)) > 1
                both_sides += len(set(lefts)) > 1 and len(set(rights)) > 1
                profile = ("rect" if len(set(lefts)) == len(set(rights)) == 1
                           else "rough" if rough else "smooth")
                states_by_profile.setdefault(profile, set()).add(info["state"])
                local_profiles[regime].append((profile, info["state"]))
                if regime == "test" and len(set(widths)) >= 3:
                    difficult_tests += 1
        same_profile_puzzles += any(
            pa == pb and sa != sb
            for pa, sa in local_profiles["train"]
            for pb, sb in local_profiles["train"])
        familiar_reversed_tests += any(
            pa == pb and sa != sb
            for pa, sa in local_profiles["train"]
            for pb, sb in local_profiles["test"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42
    assert len(dimensions) >= 35 and len(silhouettes) >= 100
    assert len(colors) >= 40 and len(margin_values) >= 5
    assert states_by_profile["rect"] == {"hollow", "solid"}
    assert states_by_profile["smooth"] == {"hollow", "solid"}
    assert states_by_profile["rough"] == {"hollow", "solid"}
    assert nonzero_backgrounds >= 50 and zero_foregrounds >= 10
    assert changed_background_puzzles >= 25
    assert edge_pairs >= 35 and ledge_pairs >= 10
    assert ledge_connected_pairs == ledge_pairs
    assert both_side_contact >= 5
    assert narrow_margin_pairs >= 30 and wide_margin_pairs >= 30
    assert up_tapers >= 10 and down_tapers >= 10
    assert horizontal_tapers >= 10
    assert min(left_only, right_only, both_sides) >= 10
    assert rough_nonzero_edge >= 5 and zero_foreground_edge >= 5
    assert same_profile_puzzles >= 10 and familiar_reversed_tests >= 10
    assert opposite_color_puzzles >= 5 and difficult_tests >= 20

    cases = ((2, 1), (10, 1), (10, 4), (None, 1),
             (None, 4), (2, None), (10, None))
    for i, (ntrain, ntest) in enumerate(cases):
        puzzle = generate_puzzle(1000 + i, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(
            generate_puzzle(1000 + i, ntrain, ntest))
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for ntrain, ntest in ((1, None), (11, None), (None, 0),
                          (None, 5), (2, 2), (3, 4),
                          (2, 3), (True, None)):
        try:
            generate_puzzle(99, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
