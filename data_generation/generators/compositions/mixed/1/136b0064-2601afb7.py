"""Visible glyph dictionaries followed by opposite cycles of decoded attributes.

Only the standard library and NumPy are used. Generation never reads evidence,
writes files, or supplies hidden parameters to solve().
"""
from collections import Counter
import hashlib

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _runs(present):
    result = []
    for x in np.flatnonzero(present):
        x = int(x)
        if result and result[-1][1] == x:
            result[-1][1] += 1
        else:
            result.append([x, x + 1])
    return result


def _mask_key(mask):
    return tuple(int(x) for x in np.asarray(mask).flat)


def _connected(mask):
    cells = set(zip(*np.nonzero(mask)))
    if not cells:
        return False
    seen = {min(cells)}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = r + dr, c + dc
                if q in cells and q not in seen:
                    seen.add(q)
                    todo.append(q)
    return seen == cells


def _vocabulary(rng, gh, gw, count):
    """Finite sampling, with exhaustive completion if needed; <= 2**16 masks."""
    pool = {}

    def consider(bits):
        mask = np.array([(bits >> i) & 1 for i in range(gh * gw)],
                        dtype=np.int8).reshape(gh, gw)
        if (2 <= int(mask.sum()) < gh * gw and mask.any(axis=0).all()
                and mask.any(axis=1).all() and _connected(mask)):
            pool.setdefault(_mask_key(mask), mask)

    def enough():
        sizes = Counter(sum(k) for k in pool)
        return len(pool) >= count and any(n >= 2 for n in sizes.values())

    for _ in range(256):
        consider(int(rng.integers(1, 1 << (gh * gw))))
        if enough():
            break
    if not enough():
        for bits in range(1, 1 << (gh * gw)):
            consider(bits)
            if enough():
                break
    keys = list(pool)
    # Equal-area glyphs with different meanings defeat area-only decoding.
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
             if sum(a) == sum(b)]
    a, b = pairs[int(rng.integers(len(pairs)))]
    selected = [a, b]
    rest = [k for k in keys if k not in selected]
    selected += [rest[int(i)] for i in rng.permutation(len(rest))[:count - 2]]
    return tuple(pool[selected[int(i)]].copy() for i in rng.permutation(count))


def _read_input(input_grid):
    """Read the visible key, program lanes and destination ticks, in that order."""
    g = _grid(input_grid)
    # Glyph height is 2..4; the header has glyphs, one blank row, reference bars.
    candidates = []
    for gh in (2, 3, 4):
        y = gh + 2
        if y + 1 >= g.shape[0] or not np.all(g[y] == g[y, 0]):
            continue
        frame = int(g[y, 0])
        if np.any(g[:y] == frame) or not np.all(g[gh] == g[gh, 0]):
            continue
        bg = int(g[gh, 0])
        colors = set(int(x) for x in np.unique(g[:y])) - {bg}
        if frame == bg or len(colors) != 1:
            continue
        lower = g[y + 1:]
        divs = np.flatnonzero(np.all(lower == frame, axis=0))
        if len(divs) == 1 and 0 < divs[0] < g.shape[1] - 1:
            candidates.append((gh, bg, frame, colors.pop(), int(divs[0])))
    if len(candidates) != 1:
        raise ValueError("missing or ambiguous header/dividers")
    gh, bg, frame, marker, divider = candidates[0]
    y = gh + 2
    glyph_runs = _runs(np.any(g[:gh] != bg, axis=0))
    if not 3 <= len(glyph_runs) <= 5:
        raise ValueError("the key needs three to five entries")
    gw = glyph_runs[0][1] - glyph_runs[0][0]
    if not 2 <= gw <= 4 or any(b - a != gw for a, b in glyph_runs):
        raise ValueError("unequal or invalid key glyph widths")
    dictionary = {}
    expected_header = np.full((y, g.shape[1]), bg, dtype=np.int8)
    for j, (left, right) in enumerate(glyph_runs):
        block = g[:gh, left:right]
        mask = block == marker
        if (not np.all((block == bg) | mask) or not mask.any(axis=0).all()
                or not mask.any(axis=1).all() or not _connected(mask)
                or not 2 <= int(mask.sum()) < gh * gw):
            raise ValueError("invalid key mask")
        stop = glyph_runs[j + 1][0] if j + 1 < len(glyph_runs) else g.shape[1]
        row = g[gh + 1, left:stop]
        length = 0
        while length < len(row) and row[length] == marker:
            length += 1
        if not 1 <= length <= 6:
            raise ValueError("invalid reference stroke")
        key = _mask_key(mask)
        if key in dictionary or length in dictionary.values():
            raise ValueError("key masks and lengths must be distinct")
        dictionary[key] = length
        expected_header[:gh, left:right] = block
        expected_header[gh + 1, left:left + length] = marker
    if not np.array_equal(g[:y], expected_header):
        raise ValueError("unexpected header ink")
    lower = g[y + 1:]
    if np.any(lower[:, :divider] == frame) or np.any(lower[:, divider + 1:] == frame):
        raise ValueError("extra frame pixels")
    panel = lower[:, :divider]
    rows = _runs(np.any(panel != bg, axis=1))
    lanes = _runs(np.any(panel != bg, axis=0))
    if (not rows or not lanes or any(b - a != gh for a, b in rows)
            or any(b - a != gw for a, b in lanes)):
        raise ValueError("program is not a rectangular glyph array")
    tokens, colors, lengths = [], [], []
    expected_panel = np.full_like(panel, bg)
    for left, right in lanes:
        for top, bottom in rows:
            block = panel[top:bottom, left:right]
            ink = np.unique(block[block != bg])
            key = _mask_key(block != bg)
            if (len(ink) != 1 or int(ink[0]) in (frame, marker)
                    or key not in dictionary):
                raise ValueError("unrecognized or nonmonochrome program glyph")
            tokens.append(key)
            colors.append(int(ink[0]))
            lengths.append(dictionary[key])
            expected_panel[top:bottom, left:right] = block
    if not np.array_equal(panel, expected_panel):
        raise ValueError("unexpected program ink")
    pane = lower[:, divider + 1:].copy()
    marks = np.argwhere(pane != bg)
    n = len(tokens)
    if (not 3 <= n <= 10 or len(marks) != n or len(set(marks[:, 0])) != 1
            or not np.all(pane[tuple(marks.T)] == marker)):
        raise ValueError("one common-baseline tick is required for each glyph")
    baseline = int(marks[0, 0])
    slots = tuple(int(c) for c in marks[:, 1])
    if len(set(np.diff(slots))) != 1 or slots[1] - slots[0] < 2:
        raise ValueError("destination slots must be regularly spaced and separate")
    if max(dictionary.values()) > baseline:
        raise ValueError("reference strokes do not fit above the ticks")
    return {"dictionary": dictionary, "tokens": tuple(tokens), "colors": tuple(colors),
            "lengths": tuple(lengths), "pane": pane, "baseline": baseline,
            "slots": slots, "background": bg, "frame": frame, "marker": marker,
            "glyph_size": (gh, gw), "rows": len(rows), "lanes": len(lanes)}


def _draw(pane, baseline, slots, colors, lengths):
    out = pane.copy()
    for col, color, length in zip(slots, colors, lengths):
        out[baseline - int(length):baseline, col] = int(color)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    return _draw(info["pane"], info["baseline"], info["slots"],
                 np.roll(info["colors"], -1), np.roll(info["lengths"], 1))


def _program(rng, n, count, informative, cover):
    # At most 128 proposals, then a constructive asymmetric fallback.
    for _ in range(128):
        program = [int(x) for x in rng.integers(count, size=n)]
        if cover:
            program[:count] = [int(x) for x in rng.permutation(count)]
            program = [program[int(i)] for i in rng.permutation(n)]
        if informative and np.array_equal(np.roll(program, 1), np.roll(program, -1)):
            continue
        return tuple(program)
    return tuple(list(range(count)) + [0] * (n - count))


def _make_pair(rng, masks, meanings, roles, layout, anchor, cover, first_colors):
    bg, frame, marker = roles
    gh, gw = masks[0].shape
    lanes, bands = layout
    n = lanes * bands
    program = _program(rng, n, len(masks), anchor, cover)
    if anchor and not first_colors:
        lengths = tuple(meanings[k] for k in program)
        shifted = lengths[-1:] + lengths[:-1]
        # Do not accidentally teach sorting in the minimum two-example puzzle.
        if shifted in (tuple(sorted(lengths)), tuple(sorted(lengths, reverse=True))):
            return None
    available = [c for c in range(10) if c not in roles]
    if anchor and not first_colors:
        colors = [int(x) for x in rng.choice(available, n, replace=False)]
    elif anchor:
        colors = [int(rng.choice([c for c in available if c not in first_colors[k]]))
                  for k in program]
        # Demonstrate that different masks can carry the same color.
        a, b = next((a, b) for a in range(n) for b in range(a + 1, n)
                    if program[a] != program[b])
        shared = [c for c in available
                  if c not in first_colors[program[a]] | first_colors[program[b]]]
        colors[a] = colors[b] = int(rng.choice(shared))
    else:
        pool = rng.choice(available, int(rng.integers(1, 8)), replace=False)
        colors = [int(x) for x in rng.choice(pool, n)]
    xgap, ygap = (int(x) for x in rng.integers(1, 3, size=2))
    panel_width = lanes * gw + (lanes - 1) * xgap
    panel_height = bands * gh + (bands - 1) * ygap
    key_gap = 1
    tile_width = max(gw, max(meanings))
    key_width = len(masks) * tile_width + (len(masks) - 1) * key_gap
    feasible_pitch = [p for p in (2, 3, 4)
                      if panel_width + 2 + (n - 1) * p <= 30]
    if not feasible_pitch:
        return None
    pitch = int(rng.choice(feasible_pitch))
    occupied = (n - 1) * pitch + 1
    before = int(rng.integers(0, min(2, 30 - panel_width - 1 - occupied) + 1))
    minimum_width = max(key_width, panel_width + 1 + before + occupied)
    width = int(rng.integers(minimum_width, min(30, minimum_width + 3) + 1))
    inset = int(rng.integers(0, 3))
    lower_height = max(panel_height, max(meanings) + 1 + inset)
    if lower_height + gh + 3 > 30:
        return None
    lower_height += int(rng.integers(0, min(2, 30 - lower_height - gh - 3) + 1))
    baseline = lower_height - 1 - inset
    height = gh + 3 + lower_height
    inp = np.full((height, width), bg, dtype=np.int8)
    # Key order has no relationship to the order of the program or meanings.
    for tile, k in enumerate(rng.permutation(len(masks))):
        k = int(k)
        left = tile * (tile_width + key_gap)
        block = inp[:gh, left:left + gw]
        block[masks[k].astype(bool)] = marker
        inp[gh + 1, left:left + meanings[k]] = marker
    inp[gh + 2, :] = frame
    lower = inp[gh + 3:]
    lower[:, panel_width] = frame
    for j, (k, color) in enumerate(zip(program, colors)):
        lane, band = divmod(j, bands)
        top, left = band * (gh + ygap), lane * (gw + xgap)
        block = lower[top:top + gh, left:left + gw]
        block[masks[k].astype(bool)] = color
    slots = tuple(before + i * pitch for i in range(n))
    pane = lower[:, panel_width + 1:]
    pane[baseline, list(slots)] = marker
    # Construct labels directly from program attributes, independently of parsing.
    out = pane.copy()
    for i, col in enumerate(slots):
        length = meanings[program[(i - 1) % n]]
        out[baseline - length:baseline, col] = colors[(i + 1) % n]
    return {"input": inp, "output": out}, program, colors


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    gh, gw = (int(x) for x in rng.integers(2, 5, size=2))
    count = int(rng.integers(3, 6))
    masks = _vocabulary(rng, gh, gw, count)
    cap = 5 if count == 5 else 6
    meanings = tuple(int(x) for x in rng.choice(np.arange(1, cap + 1), count, replace=False))
    roles = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_colors = {}
    for split, total in (("train", nt), ("test", nq)):
        for index in range(total):
            anchor = split == "train" and index < 2
            if anchor:
                layouts = [(2, 3)] if index == 0 else [(3, 2)]
            elif split == "train":
                layouts = [(2, 2), (2, 3), (3, 2)]
            else:
                layouts = [(2, 4), (3, 3), (2, 5)]
            for _ in range(128):
                layout = layouts[int(rng.integers(len(layouts)))]
                # Four-glyph nonanchors need not cover a five-symbol dictionary.
                cover = anchor or split == "test"
                built = _make_pair(rng, masks, meanings, roles, layout, anchor,
                                   cover, first_colors)
                if built is None:
                    continue
                pair, program, colors = built
                key = pair["input"].shape, pair["input"].tobytes()
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if anchor and index == 0:
                    for k, color in zip(program, colors):
                        first_colors.setdefault(k, set()).add(color)
                break
            else:
                raise RuntimeError("bounded construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _output_attributes(output, info):
    out = _grid(output)
    assert out.shape == info["pane"].shape
    expected = info["pane"].copy()
    colors, lengths = [], []
    bg, base = info["background"], info["baseline"]
    for col in info["slots"]:
        active = np.flatnonzero(out[:base, col] != bg)
        assert len(active) and np.array_equal(active, np.arange(base - len(active), base))
        color = int(out[base - 1, col])
        assert color not in (bg, info["frame"], info["marker"])
        assert np.all(out[active, col] == color)
        colors.append(color)
        lengths.append(len(active))
        expected[active, col] = color
    assert np.array_equal(out, expected), "extra ink, moved tick, or changed margin"
    return tuple(colors), tuple(lengths)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in infos:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            key = inp.shape, inp.tobytes()
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _read_input(inp)
            colors, lengths = _output_attributes(pair["output"], info)
            n = len(colors)
            # Direct index checks plus conserved multisets; no call to solve().
            assert colors == tuple(info["colors"][(i + 1) % n] for i in range(n))
            assert lengths == tuple(info["lengths"][(i - 1) % n] for i in range(n))
            assert Counter(colors) == Counter(info["colors"])
            assert Counter(lengths) == Counter(info["lengths"])
            infos[split].append(info)
    together = infos["train"] + infos["test"]
    first = together[0]
    assert all(i["dictionary"] == first["dictionary"] and i["glyph_size"] == first["glyph_size"]
               and (i["background"], i["frame"], i["marker"])
               == (first["background"], first["frame"], first["marker"]) for i in together)
    assert min(len(i["tokens"]) for i in infos["test"]) > max(
        len(i["tokens"]) for i in infos["train"])
    # The two mandatory anchors expose every glyph under different colors.
    a, b = infos["train"][:2]
    assert (a["lanes"], a["rows"]) == (2, 3) and (b["lanes"], b["rows"]) == (3, 2)
    for anchor in (a, b):
        assert set(anchor["tokens"]) == set(first["dictionary"])
        assert not np.array_equal(np.roll(anchor["lengths"], 1), np.roll(anchor["lengths"], -1))
    assert len(set(a["colors"])) == len(a["colors"])
    assert any(sum(x) == sum(y) and x != y for x in first["dictionary"] for y in first["dictionary"])
    for glyph in first["dictionary"]:
        ca = {c for k, c in zip(a["tokens"], a["colors"]) if k == glyph}
        cb = {c for k, c in zip(b["tokens"], b["colors"]) if k == glyph}
        assert ca.isdisjoint(cb)
    assert any(c == d and k != m for k, c in zip(b["tokens"], b["colors"])
               for m, d in zip(b["tokens"], b["colors"]))
    # Check joint traversal/shift alternatives, not just isolated ablations.
    truth = (tuple(np.roll(a["colors"], -1)), tuple(np.roll(a["lengths"], 1)))
    n, bands = len(a["tokens"]), a["rows"]
    for row_major in (False, True):
        for reverse_lanes in (False, True):
            for reverse_rows in (False, True):
                lanes = list(range(a["lanes"]))[::(-1 if reverse_lanes else 1)]
                rows = list(range(bands))[::(-1 if reverse_rows else 1)]
                indices = ([c * bands + r for r in rows for c in lanes] if row_major
                           else [c * bands + r for c in lanes for r in rows])
                for cs in (-1, 0, 1):
                    colors = tuple(a["colors"][indices[(i - cs) % n]] for i in range(n))
                    if colors != truth[0]:
                        continue
                    for ls in (-1, 0, 1):
                        if not any((row_major, reverse_lanes, reverse_rows)) and (cs, ls) == (-1, 1):
                            continue
                        lengths = tuple(a["lengths"][indices[(i - ls) % n]] for i in range(n))
                        assert lengths != truth[1], "ambiguous traversal or cycle direction"
    assert truth[1] not in (tuple(sorted(a["lengths"])),
                            tuple(sorted(a["lengths"], reverse=True)))


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = _grid(pair[role])
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, count boundaries, ablations and mutations; no disk writes."""
    fingerprints, shapes, sizes, backgrounds = set(), set(), set(), set()
    lane_counts, counts, meanings, spacings, insets, vocabularies = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        fingerprints.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"])
                info = _read_input(original)
                shapes.add(original.shape)
                sizes.add(info["glyph_size"])
                backgrounds.add(info["background"])
                lane_counts.add(info["lanes"])
                counts.add(len(info["tokens"]))
                meanings.update(info["lengths"])
                spacings.add(info["slots"][1] - info["slots"][0])
                insets.add(info["pane"].shape[0] - 1 - info["baseline"])
                vocabularies.add((info["glyph_size"], tuple(sorted(info["dictionary"].items()))))
        # Parent A necessity: swap meanings of equal-area masks in the visible key.
        pair = puzzle["train"][0]
        info = _read_input(pair["input"])
        x, y = next((x, y) for x in info["dictionary"] for y in info["dictionary"]
                    if x != y and sum(x) == sum(y))
        changed = pair["input"].copy()
        gh, gw = info["glyph_size"]
        runs = _runs(np.any(changed[:gh] != info["background"], axis=0))
        for left, right in runs:
            key = _mask_key(changed[:gh, left:right] != info["background"])
            if key in (x, y):
                replacement = y if key == x else x
                changed[:gh, left:right] = np.where(
                    np.array(replacement).reshape(gh, gw), info["marker"], info["background"])
        assert not np.array_equal(solve(changed), pair["output"])
        # Parent B necessity: decode without either cycle is visibly different.
        uncycled = _draw(info["pane"], info["baseline"], info["slots"], info["colors"], info["lengths"])
        assert not np.array_equal(uncycled, pair["output"])
    assert len(fingerprints) == 42 and len(vocabularies) >= 35
    assert len(shapes) >= 50 and sizes == {(h, w) for h in range(2, 5) for w in range(2, 5)}
    assert backgrounds == set(range(10)) and lane_counts == {2, 3}
    assert {4, 6, 8, 9, 10} <= counts and meanings == set(range(1, 7))
    assert spacings == {2, 3, 4} and insets == {0, 1, 2}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject changed outputs, ticks and missing dictionary evidence.
    for damage in ("output", "tick", "key"):
        p = generate_puzzle(37, 2, 1)
        pair = p["test"][0]
        info = _read_input(pair["input"])
        if damage == "output":
            pair["output"][info["baseline"] - 1, info["slots"][0]] = info["background"]
        elif damage == "tick":
            lower_start = info["glyph_size"][0] + 3
            divider = int(np.flatnonzero(np.all(pair["input"][lower_start:] == info["frame"], axis=0))[0])
            pair["input"][lower_start + info["baseline"], divider + 1 + info["slots"][0]] = info["background"]
        else:
            pair["input"][info["glyph_size"][0] + 1, :] = info["background"]
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("semantic mutation accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds; no artifacts written)")
