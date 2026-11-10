"""Anchor-selected unary printers. Only NumPy and the standard library are used.

Cards have a frame, a one-row unary key header, a frame-colored divider,
and a body. Completed cards are the visible dictionary; blank cards are
requests. The unique blank card with a dictionary key is the answer card.
Nothing is read from disk, and solve never calls the sampling machinery.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _length(n):
    return n * n + 2 * n - 2


def _body(n, width, ink, separator):
    runs = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    stream = []
    for i, run in enumerate(runs):
        if i:
            stream.append(separator)
        stream.extend([ink] * run)
    height = (len(stream) + width - 1) // width
    result = np.full(height * width, separator, dtype=np.int8)
    result[:len(stream)] = stream
    return result.reshape(height, width)


def _check_body(body, n, ink, separator):
    """Check cells directly, separately from the output constructor."""
    h, w = body.shape
    assert h == (_length(n) + w - 1) // w, "nonminimal writing height"
    flat = body.ravel()
    position = 0
    for run_index in range(2 * n - 1):
        run = min(run_index + 1, 2 * n - 1 - run_index)
        assert np.all(flat[position:position + run] == ink), "wrong unary run"
        position += run
        if run_index < 2 * n - 2:
            assert flat[position] == separator, "wrong separator"
            position += 1
    assert position == _length(n)
    assert np.all(flat[position:] == separator), "wrong final-row padding"
    assert np.count_nonzero(body == ink) == n * n


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= size <= 30 for size in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return a


def _components(mask):
    unseen = set(zip(*np.nonzero(mask)))
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((int(r), int(c)))
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in unseen:
                    unseen.remove(point)
                    stack.append(point)
        yield component


def _interpret(input_grid):
    """Recover every role, reference, and selected request from input cells."""
    grid = _grid(input_grid)
    blank = int(grid[0, 0])
    assert (np.all(grid[0] == blank) and np.all(grid[-1] == blank)
            and np.all(grid[:, 0] == blank) and np.all(grid[:, -1] == blank)), "canvas rim"
    cards = []
    for color in (int(c) for c in np.unique(grid) if c != blank):
        for component in _components(grid == color):
            rows, cols = zip(*component)
            y, x = min(rows), min(cols)
            h, w = max(rows) - y + 1, max(cols) - x + 1
            if h < 5 or w < 4:
                continue
            crop = grid[y:y + h, x:x + w]
            frame_mask = np.zeros((h, w), dtype=bool)
            frame_mask[[0, 2, h - 1], :] = True
            frame_mask[:, [0, w - 1]] = True
            if not np.array_equal(crop == color, frame_mask):
                continue
            header = crop[1, 1:-1]
            key = int(header[0])
            assert key not in (blank, color), "invisible key"
            n = 0
            while n < len(header) and header[n] == key:
                n += 1
            assert n >= 2 and np.all(header[n:] == blank), "non-unary header"
            body = crop[3:-1, 1:-1]
            assert body.shape[0] == (_length(n) + body.shape[1] - 1) // body.shape[1]
            style = None
            if not np.all(body == blank):
                colors = set(int(c) for c in np.unique(body))
                assert blank not in colors and len(colors) == 2, "incomplete reference"
                ink = int(body[0, 0])
                separator = next(iter(colors - {ink}))
                _check_body(body, n, ink, separator)
                style = (ink, separator)
            cards.append({"box": (y, x, h, w), "key": key, "n": n,
                          "width": w - 2, "frame": color, "style": style})
    assert 2 <= len(cards) <= 5, "expected 2..5 cards"
    cards.sort(key=lambda card: card["box"][:2])
    covered = np.zeros(grid.shape, dtype=bool)
    for card in cards:
        y, x, h, w = card["box"]
        assert y > 0 and x > 0 and y + h < grid.shape[0] and x + w < grid.shape[1]
        assert not covered[y:y + h, x:x + w].any(), "overlapping cards"
        covered[y:y + h, x:x + w] = True
    assert np.all(grid[~covered] == blank), "unparsed content outside cards"
    frames = {card["frame"] for card in cards}
    assert len(frames) == 1, "one structural frame color per input"
    keys = {card["key"] for card in cards}
    templates = {}
    for card in cards:
        if card["style"] is not None:
            assert not (set(card["style"]) & (keys | frames | {blank})), "ambiguous color roles"
            key, style = card["key"], card["style"]
            assert key not in templates or templates[key] == style, "conflicting references"
            templates[key] = style
    assert templates, "no completed reference"
    empty = [card for card in cards if card["style"] is None]
    matches = [card for card in empty if card["key"] in templates]
    assert len(matches) == 1, "exactly one blank card must match a reference key"
    assert len(empty) <= 2, "at most one unmatched blank card"
    return grid, blank, cards, templates, matches[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select the visible key's printer and expand its own bar length."""
    grid, _, _, templates, target = _interpret(input_grid)
    y, x, h, w = target["box"]
    answer = grid[y:y + h, x:x + w].copy()
    ink, separator = templates[target["key"]]
    answer[3:-1, 1:-1] = _body(target["n"], target["width"], ink, separator)
    return answer


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _pack(rng, sizes):
    """Bounded shelf search in either direction; no files or execution traces."""
    count = len(sizes)
    orders = [tuple(sorted(range(count), key=lambda i: -sizes[i][axis])) for axis in (0, 1)]
    orders += [tuple(int(i) for i in rng.permutation(count)) for _ in range(8)]
    candidates = []
    for transpose in (False, True):
        oriented = [(w, h) if transpose else (h, w) for h, w in sizes]
        for order in orders:
            for limit in range(max(w for h, w in oriented), 29):
                y = x = shelf_h = used_w = 0
                boxes = [None] * count
                for i in order:
                    h, w = oriented[i]
                    if x and x + w > limit:
                        y += shelf_h + 1
                        x, shelf_h = 0, 0
                    boxes[i] = (y, x, h, w)
                    used_w = max(used_w, x + w)
                    x += w + 1
                    shelf_h = max(shelf_h, h)
                height = y + shelf_h
                if height <= 28:
                    if transpose:
                        boxes = [(x, y, w, h) for y, x, h, w in boxes]
                        candidates.append((used_w, height, boxes))
                    else:
                        candidates.append((height, used_w, boxes))
    if not candidates:
        return None
    if rng.random() < 0.5:
        candidates.sort(key=lambda item: item[0] * item[1])
        candidates = candidates[:max(1, len(candidates) // 3)]
    height, width, boxes = _pick(rng, candidates)
    if rng.random() < 0.5:
        boxes = [(height - y - h, x, h, w) for y, x, h, w in boxes]
    if rng.random() < 0.5:
        boxes = [(y, width - x - w, h, w) for y, x, h, w in boxes]
    return height, width, boxes


def _card(n, width, key, style):
    return {"n": n, "width": width, "key": key, "style": style}


def _draw_card(n, width, key, style, blank, frame):
    h = (_length(n) + width - 1) // width
    a = np.full((h + 4, width + 2), frame, dtype=np.int8)
    a[1, 1:-1] = blank
    a[1, 1:n + 1] = key
    a[3:-1, 1:-1] = blank if style is None else _body(n, width, *style)
    return a


def _make_pair(rng, stage, regime, train_widths, base_palette, stable):
    first, second = stage == "first", stage == "second"
    is_test = stage == "test"
    nkeys = 1 if first else 2 if second else int(rng.integers(1 if not is_test else 2, 4))
    decoy = first or second or bool(rng.random() < 0.7)
    palette = base_palette.copy() if stable else [int(c) for c in rng.permutation(10)]
    blank, frame = palette[:2]
    keys = palette[2:2 + nkeys]
    unknown = palette[2 + nkeys] if decoy else None
    style_colors = palette[2 + nkeys + int(decoy):]
    inks = [int(c) for c in rng.choice(style_colors, size=nkeys, replace=False)]
    styles = [(ink, _pick(rng, [c for c in style_colors if c != ink])) for ink in inks]
    selected = int(rng.integers(nkeys))
    if first:
        n = 3
        refs = [(0, 3), (0, 3)]
    else:
        if second:
            n = 5
        elif is_test:
            n = int(_pick(rng, (7, 7, 8, 9, 10))) if regime == "longer" else 3 if regime == "rewrap" else 4
        else:
            n = int(_pick(rng, (3, 5, 6) if regime == "interpolate" else (3, 4, 5, 6)))
        ref_pool = (2, 3) if regime == "interpolate" else (2, 3, 4)
        refs = [(k, 3 if second else int(_pick(rng, ref_pool))) for k in range(nkeys)]
        if not second and nkeys < 3 and rng.random() < 0.4:
            k = int(rng.integers(nkeys))
            refs.append((k, int(_pick(rng, ref_pool))))
    # At most three ordinary reference cards: combined with one decoy and
    # the target they admit a small constructive shelf fallback.
    assert len(refs) <= 3
    forbidden = train_widths.get(n, set()) if is_test and regime == "rewrap" else set()
    if first:
        target_options = list(range(14, 25))
    elif is_test and regime == "rewrap":
        target_options = [w for w in range(3, 13) if w not in forbidden]
    elif second:
        target_options = list(range(6, 15))
    else:
        target_options = list(range(n, 25))
    assert target_options
    target_index = len(refs)
    for attempt in range(17):
        fallback = attempt == 16
        descriptors = []
        for j, (k, rn) in enumerate(refs):
            if first:
                rw = (13 if j == 0 else 6) if fallback else int(rng.integers(13, 19) if j == 0 else rng.integers(5, 10))
            else:
                rw = 6 if fallback else int(rng.integers(max(rn, 4), 16))
            descriptors.append(_card(rn, rw, keys[k], styles[k]))
        options = [w for w in target_options if not first or w not in {d["width"] for d in descriptors}]
        if fallback:
            bounded = [w for w in options if (_length(n) + w - 1) // w <= 8]
            tw = int(_pick(rng, bounded or options))
        else:
            tw = int(_pick(rng, options))
        descriptors.append(_card(n, tw, keys[selected], None))
        if decoy:
            dn = 2 if fallback else int(rng.integers(2, 7))
            dw = 6 if fallback else int(rng.integers(dn, 17))
            descriptors.append(_card(dn, dw, unknown, None))
        cards = [_draw_card(d["n"], d["width"], d["key"], d["style"], blank, frame) for d in descriptors]
        packed = _pack(rng, [card.shape for card in cards])
        if packed is not None:
            break
    else:
        raise RuntimeError("constructive card packing failed")
    height, width, boxes = packed
    extra_h = int(rng.integers(min(4, 28 - height) + 1))
    extra_w = int(rng.integers(min(4, 28 - width) + 1))
    dy, dx = int(rng.integers(extra_h + 1)) + 1, int(rng.integers(extra_w + 1)) + 1
    grid = np.full((height + extra_h + 2, width + extra_w + 2), blank, dtype=np.int8)
    for card, (y, x, h, w) in zip(cards, boxes):
        grid[y + dy:y + dy + h, x + dx:x + dx + w] = card
    answer = cards[target_index].copy()
    answer[3:-1, 1:-1] = _body(n, tw, *styles[selected])
    return {"input": grid, "output": answer}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                shape, payload = _key(pair[name])
                digest.update(bytes(shape))
                digest.update(payload)
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    regime = _pick(rng, ("longer", "rewrap", "interpolate"))
    stable = bool(rng.random() < 0.3)
    base_palette = [int(c) for c in rng.permutation(10)]
    result, used, train_widths = {"train": [], "test": []}, set(), {}
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            stage = "test" if split == "test" else "first" if i == 0 else "second" if i == 1 else "extra"
            for _ in range(8):
                pair = _make_pair(rng, stage, regime, train_widths, base_palette, stable)
                fingerprint = _key(pair["input"])
                if fingerprint not in used:
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
            used.add(fingerprint)
            result[split].append(pair)
            if split == "train":
                _, _, _, _, target = _interpret(pair["input"])
                train_widths.setdefault(target["n"], set()).add(target["width"])
    validate_generated_puzzle(result)
    return result


def _features(pair):
    grid, blank, cards, templates, target = _interpret(pair["input"])
    out = _grid(pair["output"])
    y, x, h, w = target["box"]
    assert out.shape == (h, w), "answer must be precisely the selected framed card"
    source_crop = grid[y:y + h, x:x + w]
    mask = np.ones(out.shape, dtype=bool)
    mask[3:-1, 1:-1] = False
    assert np.array_equal(out[mask], source_crop[mask]), "changed frame, divider or header"
    ink, separator = templates[target["key"]]
    _check_body(out[3:-1, 1:-1], target["n"], ink, separator)
    assert np.all(source_crop[3:-1, 1:-1] == blank)
    refs = [c for c in cards if c["style"] is not None]
    matched = [c for c in refs if c["key"] == target["key"]]
    empty = [c for c in cards if c["style"] is None]
    n, width = target["n"], target["width"]
    binary = _body(n, width, 1, 0).ravel()[:_length(n)]
    cuts = list(range(width, _length(n), width))
    peak_start = n * (n - 1) // 2 + n - 1
    return {"n": n, "width": width, "body_height": h - 4, "pad": (-_length(n)) % width,
            "input_shape": tuple(grid.shape), "output_shape": tuple(out.shape),
            "reference_count": len(refs), "key_count": len(templates), "card_count": len(cards),
            "decoy": len(empty) == 2, "target_first": cards[0] is target,
            "length_change": all(c["n"] != n for c in matched),
            "width_contrast": any(a["key"] == b["key"] and a["n"] == b["n"] and a["width"] != b["width"]
                                  for i, a in enumerate(refs) for b in refs[i + 1:]),
            "inside_cut": any(binary[p - 1] == binary[p] == 1 for p in cuts),
            "separator_cut": any(binary[p - 1] == 0 or binary[p] == 0 for p in cuts),
            "peak_cut": any(peak_start < p < peak_start + n for p in cuts),
            "roles": (blank, target["frame"], target["key"], ink, separator),
            "reference_lengths": tuple(sorted(c["n"] for c in refs)),
            "same_length_style_contrast": any(a["n"] == b["n"] and a["style"] != b["style"]
                                              for i, a in enumerate(refs) for b in refs[i + 1:])}


def validate_generated_puzzle(puzzle):
    """Check schema, visible dictionary consistency, run arithmetic and evidence."""
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
        seen, evidence = set(), []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for value in pair.values():
                    _grid(value)
                fingerprint = _key(pair["input"])
                assert fingerprint not in seen, "duplicate input"
                seen.add(fingerprint)
                f = _features(pair)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                if split == "train":
                    evidence.append(f)
        assert any(f["width_contrast"] for f in evidence), "show rewrapping of one reference stream"
        assert any(f["length_change"] for f in evidence), "show parameterization beyond literal template copying"
        assert any(f["same_length_style_contrast"] for f in evidence), "show key-specific palettes"
        assert any(f["decoy"] for f in evidence), "show dictionary-based request selection"
        assert any(f["body_height"] == 1 for f in evidence)
        assert any(f["body_height"] > 1 for f in evidence)
        assert len({f["n"] for f in evidence}) >= 2
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError(f"invalid puzzle: {error}") from error


def self_test():
    """42 sequential seeds, streaming measurements, count boundaries and mutations.

    Retains at most two puzzles at a time. No files are written. The returned
    small report is useful for reproducible author evidence, not certification
    by an independent solver.
    """
    signatures, shapes, body_shapes, roles = set(), set(), set(), set()
    lengths, widths, key_counts, card_counts, reference_counts = set(), set(), set(), set(), set()
    totals = dict.fromkeys(("exact", "one_pad", "long_pad", "inside_cut", "separator_cut", "peak_cut",
                           "decoy", "target_first", "length_change", "width_contrast"), 0)
    shifts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _digest(puzzle)
        assert signature == _digest(generate_puzzle(seed)), "seed reproducibility"
        signatures.add(signature)
        train, test = [[_features(pair) for pair in puzzle[split]] for split in ("train", "test")]
        if any(f["n"] > max(t["n"] for t in train) for f in test):
            shifts.add("longer")
        if any(f["n"] == 4 and all(t["n"] != 4 for t in train) for f in test):
            shifts.add("interpolate")
        if any(f["n"] == 3 and all(t["width"] != f["width"] for t in train if t["n"] == 3) for f in test):
            shifts.add("rewrap")
        for f in train + test:
            shapes.add(f["input_shape"])
            body_shapes.add((f["body_height"], f["width"]))
            roles.add(f["roles"])
            lengths.add(f["n"]); widths.add(f["width"])
            key_counts.add(f["key_count"]); card_counts.add(f["card_count"])
            reference_counts.add(f["reference_count"])
            totals["exact"] += f["pad"] == 0
            totals["one_pad"] += f["pad"] == 1
            totals["long_pad"] += f["pad"] >= 5
            for name in ("inside_cut", "separator_cut", "peak_cut", "decoy", "target_first", "length_change", "width_contrast"):
                totals[name] += bool(f[name])
    assert len(signatures) == 42 and len(shapes) >= 30 and len(body_shapes) >= 30
    assert {3, 4, 5, 6, 7, 8, 9, 10} <= lengths and len(widths) >= 15
    assert key_counts == {1, 2, 3} and reference_counts == {1, 2, 3}
    assert len(card_counts) >= 3 and len(roles) >= 100
    assert shifts == {"longer", "interpolate", "rewrap"}
    assert all(totals[name] > 0 for name in totals), totals
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Both inherited mechanisms causally affect visible answers.
    p = generate_puzzle(17, 2, 1)
    inp = p["train"][1]["input"]
    grid, _, cards, templates, target = _interpret(inp)
    selected_style = templates[target["key"]]
    mutated = grid.copy()
    for card in cards:
        if card["key"] == target["key"] and card["style"] is not None:
            y, x, h, w = card["box"]
            section = mutated[y + 3:y + h - 1, x + 1:x + w - 1]
            old = section.copy()
            section[old == selected_style[0]] = selected_style[1]
            section[old == selected_style[1]] = selected_style[0]
    assert not np.array_equal(solve(mutated), solve(inp)), "B palette intervention must change the answer"
    y, x, h, w = target["box"]
    incorrect_copy = grid[y:y + h, x:x + w].copy()
    incorrect_copy[3:-1, 1:-1] = selected_style[0]
    assert not np.array_equal(incorrect_copy, solve(inp)), "A separators and changing run lengths are necessary"
    # Changing reference identity changes which spatial request is selected.
    # Changing only the selected bar length changes its squared ink count.
    fixture = np.zeros((13, 30), dtype=np.int8)
    for yy, xx, card in ((1, 1, _draw_card(3, 5, 2, (4, 5), 0, 1)),
                         (1, 10, _draw_card(4, 6, 2, None, 0, 1)),
                         (1, 20, _draw_card(5, 7, 3, None, 0, 1))):
        hh, ww = card.shape
        fixture[yy:yy + hh, xx:xx + ww] = card
    initial = solve(fixture)
    assert np.count_nonzero(initial[3:-1, 1:-1] == 4) == 16
    longer = fixture.copy()
    new_request = _draw_card(5, 6, 2, None, 0, 1)
    longer[1:1 + new_request.shape[0], 10:10 + new_request.shape[1]] = new_request
    expanded = solve(longer)
    assert np.count_nonzero(expanded[3:-1, 1:-1] == 4) == 25
    assert expanded.shape != initial.shape
    moved = fixture.copy()
    moved[2, 2:5] = 3
    selected_elsewhere = solve(moved)
    assert selected_elsewhere.shape != initial.shape
    assert np.count_nonzero(selected_elsewhere[3:-1, 1:-1] == 4) == 25
    # The unknown request is deliberately large: choosing the largest blank
    # card would disagree with the original fixture's answer.
    ambiguous = fixture.copy()
    ambiguous[2, 21:26] = 2
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous requests accepted")
    bad = {split: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[split]] for split in p}
    bad["test"][0]["output"][3, 1] = int(bad["test"][0]["output"][0, 0])
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt run accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "body_shapes": len(body_shapes), "role_profiles": len(roles),
            "target_lengths": sorted(lengths), "target_widths": sorted(widths),
            "key_counts": sorted(key_counts), "card_counts": sorted(card_counts),
            "reference_counts": sorted(reference_counts), "observed_shifts": sorted(shifts),
            "features": totals, "checks": "generator-owned; no independent semantic solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
