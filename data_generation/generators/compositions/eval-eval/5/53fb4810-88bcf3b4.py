"""Keyed cord: whole-word phase selects contour routes; loose cells set length.

Only standard library and NumPy. No parent imports, file access, latent solver
state, saved traces, caches, network access, or unbounded construction loops.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import numbers

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)
SEEDS = (0, 51860344, 765178949, 752012645, 673564787, 34575231,
         406634688, 855617838, 5839509, 696518570, 402133885, 442426104,
         190326614, 424502316, 306048599, 140422994, 270591076, 467536096,
         345781323, 429110108, 796748621, 788796276, 290246010, 995831806,
         152374727, 974373202, 554481805, 405249213, 930071452, 1992958,
         656361322, 279855410, 335644629, 251643639, 877287810, 676198367,
         520008139, 391495179, 316488947, 334443880, 128572936, 498800203)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= s <= 30 for s in g.shape), "grid size"
    assert np.issubdtype(g.dtype, np.integer), "integer cells required"
    assert np.all((0 <= g) & (g <= 9)), "color range"
    return g.astype(np.int8, copy=True)


def _components(cells, diagonal=False):
    todo = set(cells)
    result = []
    directions = DIR8 if diagonal else DIR4
    while todo:
        first = min(todo)
        todo.remove(first)
        group, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _unimodal(values):
    descending = False
    for a, b in zip(values, values[1:]):
        if abs(a - b) > 1:
            return False
        if b < a:
            descending = True
        elif b > a and descending:
            return False
    return True


def _detour(post, side):
    """Ordered orthogonal exterior contour, excluding entry (a-1,0).

    Outward rises happen before a horizontal step; inward falls happen after.
    Unimodal height profiles make this a simple, non-backtracking contour.
    """
    a, b = post["a"], post["b"]
    heights = post["upper"] if side == -1 else post["lower"]
    path = []
    x, y = a - 1, 0

    def vertical(target):
        nonlocal y
        while y != target:
            y += 1 if target > y else -1
            path.append((x, y))

    vertical(side * (heights[0] + 1))
    x = a
    path.append((x, y))
    for j in range(1, len(heights)):
        target = side * (heights[j] + 1)
        if heights[j] > heights[j - 1]:
            vertical(target)
        x += 1
        path.append((x, y))
        if heights[j] <= heights[j - 1]:
            vertical(target)
    x = b + 1
    path.append((x, y))
    vertical(0)
    assert len(path) == len(set(path))
    return path


def _route(word, posts, forced=None, shortest=False):
    """Build the conceptual path, and expose gate/phase decisions for checks."""
    p = len(word)
    path = [(x, 0) for x in range(1, p + 1)]
    decisions = []
    for i, post in enumerate(posts):
        path.extend((x, 0) for x in range(path[-1][0] + 1, post["a"]))
        choices = []
        for side in (-1, 1):
            branch = _detour(post, side)
            pin = post["pins"][side]
            checkpoint = (pin[0], pin[1] + side)
            index = len(path) + branch.index(checkpoint)
            choices.append((side, branch, index, word[index % p] == pin[2]))
        if forced is not None:
            chosen = next(c for c in choices if c[0] == forced[i])
        elif shortest:
            chosen = min(choices, key=lambda c: (len(c[1]), c[0]))
        else:
            matching = [c for c in choices if c[3]]
            assert len(matching) == 1, "each post needs exactly one matching pin"
            chosen = matching[0]
        side, branch, index, match = chosen
        decisions.append({"side": side, "index": index, "entry_index": len(path) - 1,
                          "end_index": len(path) + len(branch) - 1,
                          "longer": len(branch) > min(len(c[1]) for c in choices),
                          "tie": len(choices[0][1]) == len(choices[1][1]),
                          "checkpoint": branch[index - len(path)],
                          "match": match})
        path.extend(branch)
    assert len(path) == len(set(path))
    return path, decisions


def _finish(path, length):
    result = list(path)
    last_x = result[-1][0]
    result.extend((last_x + k, 0) for k in range(1, max(0, length - len(result)) + 1))
    return result[:length]


def _parse_with_hub(g, bg, hub, forward, groups_by_cell):
    rows, cols = zip(*hub)
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    if forward == (0, 1):
        origin = ((r0 + r1) // 2, c1)
    elif forward == (0, -1):
        origin = ((r0 + r1) // 2, c0)
    elif forward == (1, 0):
        origin = (r1, (c0 + c1) // 2)
    else:
        origin = (r0, (c0 + c1) // 2)
    lateral = (forward[1], -forward[0])

    def global_cell(p):
        x, y = p
        return (origin[0] + x * forward[0] + y * lateral[0],
                origin[1] + x * forward[1] + y * lateral[1])

    def local_cell(p):
        dr, dc = p[0] - origin[0], p[1] - origin[1]
        return dr * forward[0] + dc * forward[1], dr * lateral[0] + dc * lateral[1]

    def value(p):
        q = global_cell(p)
        return int(g[q]) if _inside(q, g.shape) else bg

    hub_color = int(g[min(hub)])
    hub_width = len({local_cell(q)[0] for q in hub})
    word = []
    for x in range(1, 31):
        color = value((x, 0))
        if color == bg:
            break
        word.append(color)
    assert 2 <= len(word) <= 5 and len(set(word)) >= 2
    assert hub_color not in word
    prefix = {global_cell((x, 0)) for x in range(1, len(word) + 1)}
    back = global_cell((-hub_width, 0))
    assert back in groups_by_cell
    slack_color, slack = groups_by_cell[back]
    assert slack_color not in (bg, hub_color, *word)
    local_slack = {local_cell(q) for q in slack}
    assert all(x <= -hub_width for x, y in local_slack)
    assert {q for q in local_slack if q[0] == -hub_width} == {(-hub_width, 0)}
    foreground = {tuple(map(int, q)) for q in np.argwhere(g != bg)}
    remaining = foreground - hub - prefix - slack
    raw_posts = _components(remaining)
    assert 1 <= len(raw_posts) <= 3
    posts, post_colors = [], set()
    for cells in raw_posts:
        local = {local_cell(q) for q in cells}
        xs = sorted({x for x, y in local})
        a, b = xs[0], xs[-1]
        assert a >= len(word) + 2 and xs == list(range(a, b + 1))
        assert 1 <= len(xs) <= 5
        upper, lower = [], []
        for x in xs:
            ys = sorted(y for u, y in local if u == x)
            assert ys == list(range(ys[0], ys[-1] + 1)) and ys[0] <= -1 and ys[-1] >= 1
            upper.append(-ys[0])
            lower.append(ys[-1])
        assert max(upper + lower) <= 5
        assert _unimodal(upper) and _unimodal(lower)
        body = value((a, 0))
        assert body not in (bg, hub_color, slack_color, *word)
        post_colors.add(body)
        markers = [(x, y, value((x, y))) for x, y in sorted(local) if value((x, y)) != body]
        assert len(markers) == 2 and all(v in word for x, y, v in markers)
        pins = {}
        for x, y, color in markers:
            if y == -upper[x - a] and -y == max(upper):
                assert -1 not in pins
                pins[-1] = (x, y, color)
            elif y == lower[x - a] and y == max(lower):
                assert 1 not in pins
                pins[1] = (x, y, color)
            else:
                raise AssertionError("pins must be on the two maximal outer faces")
        assert set(pins) == {-1, 1}
        posts.append({"a": a, "b": b, "upper": upper, "lower": lower,
                      "pins": pins, "cells": cells, "color": body})
    assert len(post_colors) == 1
    posts.sort(key=lambda item: item["a"])
    assert all(a["b"] + 3 <= b["a"] for a, b in zip(posts, posts[1:]))
    route, decisions = _route(word, posts)
    path = _finish(route, len(word) + len(slack))
    protected = hub | prefix | set().union(*(p["cells"] for p in posts))
    out = g.copy()
    for q in slack:
        out[q] = bg
    visible = []
    for index, q in enumerate(path[len(word):], len(word)):
        target = global_cell(q)
        if _inside(target, g.shape):
            assert target not in protected and out[target] == bg, "cord collides with fixed material"
            out[target] = word[index % len(word)]
            visible.append(target)
    return {"output": out, "background": bg, "hub": hub, "hub_color": hub_color,
            "word": word, "prefix": prefix, "slack": slack, "slack_color": slack_color,
            "local_slack": local_slack, "posts": posts, "decisions": decisions,
            "route": route, "path": path, "visible": visible, "global": global_cell,
            "forward": forward, "protected": protected}


def _analyze_input(input_grid):
    g = _grid(input_grid)
    frequencies = np.bincount(g.ravel(), minlength=10)
    bg = int(frequencies.argmax())
    assert np.count_nonzero(frequencies == frequencies[bg]) == 1
    groups_by_cell, candidates = {}, []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        cells = {tuple(map(int, q)) for q in np.argwhere(g == color)}
        for group in _components(cells, diagonal=True):
            for q in group:
                groups_by_cell[q] = (color, group)
            rows, cols = zip(*group)
            h, w = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
            if h in (3, 5) and w in (3, 5) and len(group) == h * w:
                candidates.append(group)
    answers = []
    for hub in candidates:
        for forward in DIR4:
            try:
                answers.append(_parse_with_hub(g, bg, hub, forward, groups_by_cell))
            except (AssertionError, KeyError, ValueError):
                continue
    assert len(answers) == 1, "input must have one geometric interpretation"
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the whole word, material count, keyed routes and phase from input."""
    return _analyze_input(input_grid)["output"]


def _profile(rng, width, limit=3, require_ramp=False):
    base = int(rng.integers(1, limit + 1))
    if width >= 2 and (require_ramp or rng.random() < .6):
        peak = int(rng.integers(width))
        height = min(limit, max(2, base))
        values = [max(1, height - abs(j - peak)) for j in range(width)]
        if len(set(values)) == 1:
            values[0] = max(1, values[0] - 1)
        return values
    return [base] * width


def _word(rng, p, palette):
    word = list(map(int, rng.choice(palette, p)))
    if len(set(word)) == 1:
        word[-1] = next(c for c in palette if c != word[0])
    if p >= 3 and rng.random() < .4:
        word[-1] = word[0]
        if len(set(word)) == 1:
            word[1] = next(c for c in palette if c != word[0])
    return word


def _reserve(rng, count, hub_width, diagonal, style):
    radius = int(rng.integers(4, 9))
    depth = max(2, (count + 2 + 2 * radius) // (2 * radius + 1))
    depth = min(4, depth)
    if depth * (2 * radius + 1) - 2 < count - 1:
        radius = 9
    allowed = {(x, y) for x in range(-hub_width - depth, -hub_width)
               for y in range(-radius, radius + 1)}
    anchor = (-hub_width, 0)
    cells = {anchor}
    if diagonal:
        allowed -= {(-hub_width - 1, y) for y in (-1, 0)}
        cells.add((-hub_width - 1, 1))
    directions = DIR8 if diagonal or style in (1, 2) else DIR4
    while len(cells) < count:
        frontier = sorted({(x + dx, y + dy) for x, y in cells for dx, dy in directions}
                          & allowed - cells)
        assert frontier, "reserve capacity"
        weights = []
        for x, y in frontier:
            neighbors = sum((x + dx, y + dy) in cells for dx, dy in DIR8)
            if style == 0:
                score = (neighbors + 1) ** 2
            elif style == 1:
                score = 1 / (neighbors + 1) ** 2
            elif style == 2:
                score = 1 + abs(y)
            else:
                score = 1
            weights.append(score)
        weights = np.asarray(weights, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return cells


def _build_pair(rng, lesson, regime, bg, orientation=None, compact=False):
    colors = [int(c) for c in rng.permutation([c for c in range(10) if c != bg])]
    hc, sc, oc = colors[:3]
    palette = colors[3:3 + int(rng.integers(2, 6))]
    if lesson == "whole":
        word = [palette[0], palette[1], palette[0]]
        count = 1
    elif lesson == "relay":
        word = colors[3:6]
        count = 2
    elif lesson == "test" and regime == 0:
        word = _word(rng, int(rng.integers(2, 4)), palette)
        count = 3
    elif lesson == "test" and regime == 1:
        word = _word(rng, 5, palette)
        count = int(rng.integers(1, 3))
    else:
        word = _word(rng, int(rng.integers(2, 5)), palette)
        count = 2 if lesson == "test" and regime == 2 else int(rng.integers(1, 3))
    p = len(word)
    side0 = int(rng.choice((-1, 1)))
    sides = [side0 if j % 2 == 0 else -side0 for j in range(count)]
    if lesson not in ("whole", "relay"):
        sides = list(map(int, rng.choice((-1, 1), count)))
    posts = []
    cursor = p
    for j in range(count):
        a = cursor + int(rng.integers(2 if j == 0 else 3, 4))
        width = (3 if lesson == "relay" and j == 0 else 1 if compact else
                 int(rng.integers(1, 3 if count == 3 else 5)))
        if lesson == "whole":
            selected, other = [3] * width, [1] * width
        elif lesson == "relay":
            selected = [2, 3, 2] if j == 0 else [1] * width
            other = [1] * width if j == 0 else [2] * width
        elif lesson == "test" and regime == 3:
            selected = [int(rng.integers(4, 6))] * width
            other = _profile(rng, width, 3)
        else:
            selected, other = _profile(rng, width), _profile(rng, width)
        upper, lower = (selected, other) if sides[j] == -1 else (other, selected)
        pins = {}
        for side, profile in ((-1, upper), (1, lower)):
            options = [i for i, h in enumerate(profile) if h == max(profile)]
            k = int(rng.choice(options))
            pins[side] = (a + k, side * profile[k], word[0])
        posts.append({"a": a, "b": a + width - 1, "upper": upper, "lower": lower, "pins": pins})
        cursor = a + width - 1
    # Set visible pins using the sampled true winding, never a hidden convention.
    partial = [(x, 0) for x in range(1, p + 1)]
    for post, side in zip(posts, sides):
        partial.extend((x, 0) for x in range(partial[-1][0] + 1, post["a"]))
        for candidate_side in (-1, 1):
            branch = _detour(post, candidate_side)
            x, y, _ = post["pins"][candidate_side]
            index = len(partial) + branch.index((x, y + candidate_side))
            expected = word[index % p]
            color = expected if candidate_side == side else int(rng.choice([c for c in sorted(set(word)) if c != expected]))
            post["pins"][candidate_side] = (x, y, color)
        partial.extend(_detour(post, side))
    route, decisions = _route(word, posts)
    assert partial == route
    clipped = lesson == "relay" or (lesson in ("extra", "test") and rng.random() < .25)
    if lesson == "test" and regime == 2:
        # Cut the material budget inside the last winding, not at a color cycle.
        last = decisions[-1]
        total_length = int(rng.integers(last["entry_index"] + 2, last["end_index"] + 1))
        clipped = False
    elif lesson == "extra" and rng.random() < .25:
        total_length = int(rng.integers(p + 3, len(route) + 1))
        clipped = False
    else:
        total_length = len(route) + int(rng.integers(6, 9) if clipped else rng.integers(1, 4))
    n = total_length - p
    hub_width, hub_height = int(rng.choice((3, 5))), int(rng.choice((3, 5)))
    if count == 3 or compact:
        hub_width = 3
    if compact:
        hub_height = 3
    slack = _reserve(rng, n, hub_width, lesson in ("whole", "relay") or rng.random() < .4,
                     int(rng.integers(4)))
    cells = {(x, y): hc for x in range(-hub_width + 1, 1)
             for y in range(-hub_height // 2 + 1, hub_height // 2 + 1)}
    cells.update({(x, 0): word[x - 1] for x in range(1, p + 1)})
    cells.update({q: sc for q in slack})
    for post in posts:
        for k, x in enumerate(range(post["a"], post["b"] + 1)):
            for y in range(-post["upper"][k], post["lower"][k] + 1):
                assert (x, y) not in cells
                cells[x, y] = oc
        for x, y, color in post["pins"].values():
            cells[x, y] = color
    path = _finish(route, total_length)
    all_points = set(cells) | set(route) | (set() if clipped else set(path))
    xmin = min(x for x, y in all_points) - (0 if compact else int(rng.integers(0, 3)))
    xmax = (route[-1][0] + int(rng.integers(1, 3)) if clipped else
            max(x for x, y in all_points) + (2 if compact else int(rng.integers(2 if lesson == "whole" else 0, 4))))
    ymin = min(y for x, y in all_points) - int(rng.integers(0, 3))
    ymax = max(y for x, y in all_points) + int(rng.integers(0, 3))
    width = xmax - xmin + 1
    assert width <= 30
    # Keep the background visibly dominant even for dense count encodings.
    min_height = (2 * len(cells) + width) // width
    missing = max(0, min_height - (ymax - ymin + 1))
    ymin -= missing // 2
    ymax += missing - missing // 2
    assert ymax - ymin + 1 <= 30
    inp = np.full((ymax - ymin + 1, width), bg, dtype=np.int8)
    for (x, y), color in cells.items():
        inp[y - ymin, x - xmin] = color
    out = inp.copy()
    for x, y in slack:
        out[y - ymin, x - xmin] = bg
    for index, (x, y) in enumerate(path[p:], p):
        if xmin <= x <= xmax and ymin <= y <= ymax:
            assert out[y - ymin, x - xmin] == bg
            out[y - ymin, x - xmin] = word[index % p]
    turns = int(rng.integers(4)) if orientation is None else int(orientation)
    mirror = bool(rng.integers(2))
    def transform(g):
        v = np.rot90(g, turns)
        return np.ascontiguousarray(np.fliplr(v) if mirror else v)
    pair = {"input": transform(inp), "output": transform(out)}
    analysis = _analyze_input(pair["input"])
    assert np.array_equal(analysis["output"], pair["output"])
    return pair


def _unique(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # At most thirteen earlier inputs versus at least 5! injective foreground
    # recolorings. No geometric or rule change, no repeated payload storage.
    bg = int(np.bincount(pair["input"].ravel(), minlength=10).argmax())
    colors = [int(c) for c in np.unique(pair["input"]) if c != bg]
    for order in itertools.islice(itertools.permutations(colors), 14):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = order
        candidate = {role: mapping[g] for role, g in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    raise AssertionError("finite palette repair exhausted")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not lo <= value <= hi:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(4))
    backgrounds = list(map(int, rng.permutation(10)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", nq)):
        for i in range(number):
            lesson = ("whole" if i == 0 else "relay" if i == 1 else "extra") if split == "train" else "test"
            bg = backgrounds[(i + (nt if split == "test" else 0)) % 10]
            for attempt in range(160):
                try:
                    pair = _build_pair(rng, lesson, regime, bg, compact=attempt >= 120)
                    pair = _unique(pair, seen)
                    key = _key(pair["input"])
                    assert key not in seen
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(a, mode):
    word, p = a["word"], len(a["word"])
    n = len(a["slack"])
    out = a["output"].copy()
    # Restore a canvas containing exactly the fixed material before re-drawing.
    for q in a["visible"]:
        out[q] = a["background"]
    if mode == "shortest":
        path, _ = _route(word, a["posts"], shortest=True)
        path = _finish(path, p + n)
    elif mode == "straight":
        path = [(x, 0) for x in range(1, p + n + 1)]
    elif mode == "boundary":
        path = _finish(a["route"], len(a["route"]) + 31)
    elif mode == "prefix":
        word = word[:next(k for k in range(1, p + 1) if all(word[i] == word[i % k] for i in range(p)))]
        path = a["path"]
    else:
        raise ValueError(mode)
    for index, q in enumerate(path[p:], p):
        target = a["global"](q)
        if _inside(target, out.shape):
            out[target] = word[index % len(word)]
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            a = _analyze_input(inp)
            assert np.array_equal(a["output"], out), "wrong route, phase, length, or preserved material"
            assert not np.array_equal(inp, out)
            assert all(out[q] == inp[q] for q in a["protected"])
            assert all(out[q] == a["background"] for q in a["slack"])
            assert len(a["path"]) == len(a["word"]) + len(a["slack"])
            assert len(a["visible"]) == len(set(a["visible"]))
            added = int(np.count_nonzero((inp == a["background"]) & (out != a["background"])))
            erased = int(np.count_nonzero((inp != a["background"]) & (out == a["background"])))
            assert erased == len(a["slack"]) and added == len(a["visible"])
            assert added <= erased
            assert all(abs(x - u) + abs(y - v) == 1 for (x, y), (u, v) in zip(a["path"], a["path"][1:]))
            # All contour geometry is visible; clipping, when present, is tail-only.
            assert all(_inside(a["global"](q), inp.shape) for q in a["route"])
            assert all(d["match"] for d in a["decisions"])
            if split == "train":
                training.append(a)
    assert any(any(d["longer"] for d in a["decisions"]) for a in training), "teach keyed choice over shortest route"
    assert any(len(a["posts"]) >= 2 and len(set(a["word"])) >= 3
               and any(d["index"] % len(a["word"]) != (d["checkpoint"][0] - 1) % len(a["word"])
                       for d in a["decisions"][1:]) for a in training), "teach phase relay"
    assert any(len(a["visible"]) == len(a["slack"]) and len(_components(a["local_slack"])) > 1
               for a in training), "teach eight-connected material count"
    assert any(len(a["visible"]) < len(a["slack"]) for a in training), "teach tail clipping"
    assert any(len(a["visible"]) == len(a["slack"]) and
               not np.array_equal(_alternative(a, "boundary"), a["output"]) for a in training), "teach finite budget"
    assert any(not np.array_equal(_alternative(a, "prefix"), a["output"]) for a in training), "teach whole word"
    assert any(not np.array_equal(_alternative(a, "shortest"), a["output"]) for a in training)
    assert all(not np.array_equal(_alternative(a, "straight"), a["output"]) for a in training)
    assert any(any(len(set(post["upper"] if d["side"] == -1 else post["lower"])) > 1
                   and d["end_index"] < len(a["path"])
                   for post, d in zip(a["posts"], a["decisions"])) for a in training), "teach actual silhouette"


def measure_pair(pair):
    a = _analyze_input(pair["input"])
    return {"shape": list(np.asarray(pair["input"]).shape), "background": a["background"],
            "word": a["word"], "posts": len(a["posts"]), "reserve": len(a["slack"]),
            "visible_extension": len(a["visible"]), "direction": list(a["forward"]),
            "profiles": [[post["upper"], post["lower"]] for post in a["posts"]],
            "sides": [d["side"] for d in a["decisions"]],
            "gate_phases": [d["index"] % len(a["word"]) for d in a["decisions"]],
            "longer_choices": sum(d["longer"] for d in a["decisions"]),
            "ties": sum(d["tie"] for d in a["decisions"]),
            "diagonal_count": len(_components(a["local_slack"])) > 1,
            "partial_last_post": len(a["path"]) <= a["decisions"][-1]["end_index"],
            "clipped": len(a["visible"]) < len(a["slack"])}


def self_test():
    """42 fixed seeds streamed in memory; no diagnostic files or trace payloads."""
    shapes, profiles, directions, backgrounds, lengths, post_counts = set(), set(), set(), set(), set(), set()
    signatures, normalized = set(), set()
    flags = {"clipped": 0, "partial_last_post": 0, "diagonal_count": 0, "longer_choices": 0, "ties": 0}
    pairs = 0
    for seed in SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                inp = pair["input"]
                digest.update(bytes(inp.shape) + inp.tobytes() + pair["output"].tobytes())
                m = measure_pair(pair)
                shapes.add(tuple(m["shape"]))
                directions.add(tuple(m["direction"]))
                backgrounds.add(m["background"])
                lengths.add(m["reserve"])
                post_counts.add(m["posts"])
                for profile in m["profiles"]:
                    profiles.add(tuple(tuple(v) for v in profile))
                for flag in flags:
                    flags[flag] += int(bool(m[flag]))
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.ravel())
                normalized.add(hashlib.sha256(bytes(inp.shape) + norm).hexdigest())
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(normalized) >= 180
    assert len(shapes) >= 45 and len(profiles) >= 40 and len(lengths) >= 20
    assert post_counts == {1, 2, 3} and len(directions) == 4 and len(backgrounds) == 10
    assert all(value > 0 for value in flags.values())
    # Input-only geometric/palette equivariance and purity.
    p = generate_puzzle(0)
    q = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    permutation = np.random.default_rng(923).permutation(10)
    for pair in p["train"] + p["test"]:
        original = pair["input"].copy()
        for turns in range(4):
            inp = np.ascontiguousarray(np.rot90(permutation[pair["input"]], turns))
            out = np.ascontiguousarray(np.rot90(permutation[pair["output"]], turns))
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
        assert np.array_equal(original, pair["input"])
    # Same geometry and reserve, changed visible pins: pattern constraints must
    # change the route itself. Conversely an extra material cell extends it.
    whole = next(pair for pair in p["train"] if len(_analyze_input(pair["input"])["posts"]) == 1
                 and _analyze_input(pair["input"])["word"][0] == _analyze_input(pair["input"])["word"][-1]
                 and len(_analyze_input(pair["input"])["visible"]) == len(_analyze_input(pair["input"])["slack"]))
    a = _analyze_input(whole["input"])
    old_side = a["decisions"][0]["side"]
    switched = whole["input"].copy()
    for side in (-1, 1):
        _, decision = _route(a["word"], a["posts"], forced=[side])
        expected = a["word"][decision[0]["index"] % len(a["word"])]
        color = expected if side != old_side else next(c for c in a["word"] if c != expected)
        x, y, _ = a["posts"][0]["pins"][side]
        switched[a["global"]((x, y))] = color
    assert not np.array_equal(solve(switched), whole["output"])
    larger = whole["input"].copy()
    edge = max(x for x, y in a["local_slack"])
    candidates = sorted({(x + dx, y + dy) for x, y in a["local_slack"] for dx, dy in DIR8}
                        - a["local_slack"])
    cell = next(q for q in candidates if q[0] < edge and _inside(a["global"](q), larger.shape)
                and larger[a["global"](q)] == a["background"])
    larger[a["global"](cell)] = a["slack_color"]
    new = _analyze_input(larger)
    assert len(new["visible"]) == len(a["visible"]) + 1
    assert not np.array_equal(new["output"], whole["output"])
    for nt, nq in ((2, 1), (10, 1), (10, 4), (6, 3)):
        generate_puzzle(7729, nt, nq)
    for regime in range(4):
        for lesson in ("whole", "relay", "extra", "test"):
            _build_pair(np.random.default_rng(17 + regime), lesson, regime, regime, compact=True)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    a = _analyze_input(damaged["test"][0]["input"])
    cell = min(a["hub"])
    damaged["test"][0]["output"][cell] = a["background"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged fixed material accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "normalized_inputs": len(normalized), "shapes": len(shapes),
            "profile_pairs": len(profiles), "reserve_counts": len(lengths),
            "directions": len(directions), "backgrounds": len(backgrounds), "features": flags}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
