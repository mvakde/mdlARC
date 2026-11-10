"""Registered crop queries recolor whole four-connected atlas components.

All semantic information is visible in each input. No parent code, stored
reference image, seed, or training examples are used by solve.
"""
from itertools import combinations, permutations

import numpy as np


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _d4(a, t):
    """0..3 rotations; 4..7 a left-right reflection followed by rotation."""
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _components(mask, diagonal=False):
    labels = np.zeros(mask.shape, dtype=np.int16)
    sizes = [0]
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c]:
            continue
        number = len(sizes)
        labels[r, c] = number
        todo, size = [(r, c)], 0
        while todo:
            y, x = todo.pop()
            size += 1
            steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
            if diagonal:
                steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
            for dy, dx in steps:
                yy, xx = y+dy, x+dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not labels[yy, xx]):
                    labels[yy, xx] = number
                    todo.append((yy, xx))
        sizes.append(size)
    return labels, sizes


def _matches(reference, patch):
    """Exhaust all orientations, role assignments, and hidden offsets."""
    matches = []
    h, w = patch.shape
    ids = np.arange(reference.size).reshape(reference.shape)
    for t in range(8):
        full = _d4(reference, t)
        if h > full.shape[0] or w > full.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
        for swap in (0, 1):
            rr, cc = np.nonzero(np.all(windows == (patch ^ swap), axis=(-2, -1)))
            for r, c in zip(rr, cc):
                mapping = _d4(ids, t)[r:r+h, c:c+w]
                matches.append((t, swap, int(r), int(c), mapping))
    return matches


def _rectangles(g, frame, paper):
    """Find panels by corners, binary interiors, and disjoint border colors."""
    H, W = g.shape
    result = []
    for top in range(1, H-3):
        for bottom in range(top+3, H-1):
            common = np.flatnonzero((g[top] == frame) & (g[bottom] == frame))
            for left, right in combinations(common, 2):
                left, right = int(left), int(right)
                if left < 1 or right >= W-1 or right-left < 3:
                    continue
                inside = g[top+1:bottom, left+1:right]
                colors = set(map(int, np.unique(inside)))
                if len(colors) != 2 or frame in colors:
                    continue
                edge = np.concatenate((g[top, left:right+1], g[bottom, left:right+1],
                                       g[top+1:bottom, left], g[top+1:bottom, right]))
                if paper in edge or colors.intersection(map(int, edge)):
                    continue
                result.append((top, left, bottom, right))
    return result


def _parse(g):
    paper = int(g[0, 0])
    if not (np.all(g[0] == paper) and np.all(g[-1] == paper)
            and np.all(g[:, 0] == paper) and np.all(g[:, -1] == paper)):
        raise ValueError("a paper margin must surround all panels")
    interpretations = []
    for frame in map(int, np.unique(g)):
        if frame == paper:
            continue
        rects = _rectangles(g, frame, paper)
        refs = []
        for rect in rects:
            t, l, b, r = rect
            interior = g[t+1:b, l+1:r]
            if (paper in interior and np.all(g[t, l:r+1] == frame)
                    and np.all(g[b, l:r+1] == frame)
                    and np.all(g[t:b+1, l] == frame)
                    and np.all(g[t:b+1, r] == frame)):
                refs.append(rect)
        if len(refs) != 1 or len(rects) < 2:
            continue
        covered = np.zeros(g.shape, dtype=bool)
        valid = True
        for t, l, b, r in rects:
            if covered[t:b+1, l:r+1].any():
                valid = False
                break
            covered[t:b+1, l:r+1] = True
        if not valid or np.any((g != paper) & ~covered):
            continue
        t, l, b, r = refs[0]
        atlas = g[t+1:b, l+1:r].copy()
        ink = next(c for c in map(int, np.unique(atlas)) if c != paper)
        reference = (atlas == ink).astype(np.int8)
        labels, sizes = _components(reference)
        cards, role_palette = [], None
        for rect in rects:
            if rect == refs[0]:
                continue
            t, l, b, r = rect
            patch = g[t+1:b, l+1:r]
            colors = tuple(map(int, np.unique(patch)))
            if set(colors) & {paper, ink, frame}:
                valid = False
                break
            if role_palette is None:
                role_palette = colors
            if colors != role_palette:
                valid = False
                break
            if (not np.array_equal(g[t, l+1:r], g[b, l+1:r])
                    or not np.array_equal(g[t+1:b, l], g[t+1:b, r])):
                valid = False
                break
            rows, cols = {}, {}
            for y, color in enumerate(g[t+1:b, l]):
                if color != frame:
                    rows.setdefault(int(color), []).append(y)
            for x, color in enumerate(g[t, l+1:r]):
                if color != frame:
                    cols.setdefault(int(color), []).append(x)
            requests = set(rows) | set(cols)
            if (not requests or requests & {paper, ink, frame, *colors}
                    or not (set(rows) & set(cols))):
                valid = False
                break
            binary = (patch == colors[1]).astype(np.int8)
            matches = _matches(reference, binary)
            if len(matches) != 1:
                valid = False
                break
            orientation, swap, y, x, mapping = matches[0]
            if mapping.size >= reference.size:
                valid = False
                break
            component_view = labels.flat[mapping]
            cards.append(dict(rect=rect, patch=patch.copy(), rows=rows, cols=cols,
                              orientation=orientation, swap=swap, offset=(y, x),
                              mapping=mapping, components=component_view))
        if valid:
            interpretations.append(dict(paper=paper, ink=ink, frame=frame,
                                        atlas=atlas, reference=reference,
                                        labels=labels, sizes=sizes, cards=cards))
    if len(interpretations) != 1:
        raise ValueError("panels and crop registrations must be unambiguous")
    return interpretations[0]


def _owners(parsed, mode="correct"):
    owners = {}
    for card in parsed["cards"]:
        for color in sorted(set(card["rows"]) & set(card["cols"])):
            rows, cols = card["rows"][color], card["cols"][color]
            if mode == "all_crop":
                hits = set(map(int, card["components"].flat)) - {0}
            else:
                points = (list(zip(rows, cols)) if mode == "zip"
                          else [(r, c) for r in rows for c in cols])
                view = parsed["labels"] if mode == "local" else card["components"]
                hits = {int(view[r, c]) for r, c in points} - {0}
            for component in hits:
                if component in owners and owners[component] != color:
                    if mode == "correct":
                        raise ValueError("different colors request the same component")
                    continue  # Only a deterministic diagnostic for wrong rules.
                owners[component] = color
    return owners


def _render(parsed, mode="correct"):
    out = parsed["atlas"].copy()
    if mode == "pixels":
        for card in parsed["cards"]:
            for color in set(card["rows"]) & set(card["cols"]):
                for r in card["rows"][color]:
                    for c in card["cols"][color]:
                        index = int(card["mapping"][r, c])
                        if parsed["reference"].flat[index]:
                            out.flat[index] = color
        return out
    owners = _owners(parsed, "correct" if mode == "eight" else mode)
    if mode == "eight":
        labels, _ = _components(parsed["reference"], diagonal=True)
        for component, color in sorted(owners.items()):
            extended = int(labels[parsed["labels"] == component][0])
            out[labels == extended] = color
    else:
        for component, color in owners.items():
            out[parsed["labels"] == component] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_parse(_grid(input_grid)))


def _shape(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    kind = int(rng.integers(5))
    if kind == 0:  # loop, with occasional inner bars or an opening
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
        if h > 3 and w > 3 and rng.random() < .5:
            mask[int(rng.integers(1, h-1)), :int(rng.integers(2, w))] = True
    elif kind == 1:  # branched orthogonal path
        x = int(rng.integers(w))
        mask[:, x] = True
        for y in rng.choice(h, min(h, 3), replace=False):
            endpoint = int(rng.choice([0, w-1]))
            mask[y, min(x, endpoint):max(x, endpoint)+1] = True
    else:  # connected growth or a thin, turning walk
        y, x = int(rng.integers(h)), int(rng.integers(w))
        mask[y, x] = True
        if kind == 2:
            for _ in range(h+w+int(rng.integers(h*w))):
                dy, dx = [(0, 1), (0, -1), (1, 0), (-1, 0)][int(rng.integers(4))]
                y, x = min(h-1, max(0, y+dy)), min(w-1, max(0, x+dx))
                mask[y, x] = True
        target = int(rng.uniform(.35, .85) * h*w)
        while mask.sum() < target:
            frontier = []
            for y, x in np.argwhere(~mask):
                if any(0 <= yy < h and 0 <= xx < w and mask[yy, xx]
                       for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))):
                    frontier.append((y, x))
            y, x = frontier[int(rng.integers(len(frontier)))]
            mask[y, x] = True
    rr, cc = np.nonzero(mask)
    return _d4(mask[rr.min():rr.max()+1, cc.min():cc.max()+1], int(rng.integers(8)))


def _banks(rng, reference, labels, sizes):
    banks = [[], []]  # smaller test crops, larger teaching crops
    seen = set()
    for category in (0, 1):
        for _ in range(180):
            low, high = (3, 6) if category == 0 else (5, 8)
            h, w = int(rng.integers(low, high)), int(rng.integers(low, high))
            if category == 0 and h*w >= 25:
                continue
            r = int(rng.integers(reference.shape[0]-h+1))
            c = int(rng.integers(reference.shape[1]-w+1))
            patch = reference[r:r+h, c:c+w]
            if patch.min() == patch.max() or (r, c, h, w) in seen:
                continue
            if len(_matches(reference, patch)) != 1:
                continue
            view = labels[r:r+h, c:c+w]
            # At least one selectable object continues outside this view.
            if not any(0 < np.count_nonzero(view == v) < sizes[v]
                       for v in set(map(int, view.flat)) - {0}):
                continue
            banks[category].append((r, c, h, w))
            seen.add((r, c, h, w))
            if len(banks[category]) >= 32:
                break
    return banks


def _cartesian_options(view, sizes, require_subset=False):
    result = []
    all_hits = set(map(int, view.flat)) - {0}
    for rows in combinations(range(view.shape[0]), 2):
        for cols in combinations(range(view.shape[1]), 2):
            values = [int(view[r, c]) for r in rows for c in cols]
            hits = set(values) - {0}
            zipped = {int(view[rows[0], cols[0]]), int(view[rows[1], cols[1]])} - {0}
            if (hits and hits != zipped and 0 in values
                    and any(np.count_nonzero(view == v) < sizes[v] for v in hits)
                    and (not require_subset or hits != all_hits)):
                result.append((list(rows), list(cols), hits))
    return result


def _dual_options(view):
    cells = list(map(tuple, np.argwhere(view > 0)))
    return [(a, b) for a, b in combinations(cells, 2)
            if a[0] != b[0] and a[1] != b[1] and view[a] != view[b]]


def _reference(rng):
    for _ in range(80):
        H, W = map(int, rng.integers(10, 17, size=2))
        reference = np.zeros((H, W), dtype=np.int8)
        wanted = int(rng.integers(3, 7))
        placed = 0
        for _ in range(130):
            h, w = map(int, rng.integers(3, 7, size=2))
            mask = _shape(rng, h, w)
            h, w = mask.shape
            if mask.sum() < 5:
                continue
            r, c = int(rng.integers(H-h+1)), int(rng.integers(W-w+1))
            forbidden = reference.astype(bool).copy()
            forbidden[1:] |= reference[:-1].astype(bool)
            forbidden[:-1] |= reference[1:].astype(bool)
            forbidden[:, 1:] |= reference[:, :-1].astype(bool)
            forbidden[:, :-1] |= reference[:, 1:].astype(bool)
            if np.any(forbidden[r:r+h, c:c+w] & mask):
                continue
            reference[r:r+h, c:c+w] |= mask
            placed += 1
            if placed == wanted:
                break
        labels, sizes = _components(reference)
        if len(sizes) < 4:
            continue
        if len({_key(_d4(reference, t) ^ s) for t in range(8) for s in (0, 1)}) != 16:
            continue
        banks = _banks(rng, reference, labels, sizes)
        if min(map(len, banks)) < 8:
            continue
        cart, dual = [], []
        for crop in banks[1]:
            r, c, h, w = crop
            view = labels[r:r+h, c:c+w]
            if _cartesian_options(view, sizes, True):
                cart.append(crop)
            if _dual_options(view):
                dual.append(crop)
        if cart and dual:
            return reference, labels, sizes, banks, cart, dual
    raise RuntimeError("bounded search failed to find a rich identifying atlas")


def _request_card(rng, reference, labels, sizes, bank, t, swap, palette,
                  owner, kind="random", incomplete=False):
    paper, ink, frame, a, b, *queries = palette
    for _ in range(120):
        r, c, h, w = bank[int(rng.integers(len(bank)))]
        bits = _d4(reference[r:r+h, c:c+w], t)
        view = _d4(labels[r:r+h, c:c+w], t)
        rows, cols = {}, {}
        new_owner = dict(owner)
        if kind == "cartesian":
            options = _cartesian_options(view, sizes, True)
            if not options:
                continue
            rr, cc, hits = options[int(rng.integers(len(options)))]
            proposals = [(rr, cc, hits)]
        elif kind == "dual":
            options = _dual_options(view)
            if not options:
                continue
            u, v = options[int(rng.integers(len(options)))]
            proposals = [([int(u[0])], [int(u[1])], {int(view[u])}),
                         ([int(v[0])], [int(v[1])], {int(view[v])})]
        else:
            cells = [(int(y), int(x)) for y, x in np.argwhere(view > 0)
                     if np.count_nonzero(view == view[y, x]) < sizes[int(view[y, x])]]
            if not cells:
                continue
            y, x = cells[int(rng.integers(len(cells)))]
            rr, cc = [y], [x]
            if rng.random() < .6:
                axis = int(rng.integers(2))
                length = view.shape[axis]
                values = sorted(set([y if axis == 0 else x,
                                     int(rng.integers(length)), int(rng.integers(length))]))
                if axis == 0:
                    rr = values
                else:
                    cc = values
            hits = {int(view[y, x]) for y in rr for x in cc} - {0}
            proposals = [(rr, cc, hits)]
        used_colors = set()
        valid = True
        for rr, cc, hits in proposals:
            previous = {new_owner[v] for v in hits if v in new_owner}
            available = [q for q in queries[:3] if q not in used_colors]
            if len(previous) > 1 or (previous and next(iter(previous)) not in available):
                valid = False
                break
            color = next(iter(previous)) if previous else int(rng.choice(available))
            used_colors.add(color)
            rows[color], cols[color] = rr, cc
            new_owner.update({v: color for v in hits})
        if not valid:
            continue
        if incomplete:
            spare_rows = [y for y in range(view.shape[0]) if all(y not in rr for rr in rows.values())]
            spare_cols = [x for x in range(view.shape[1]) if all(x not in cc for cc in cols.values())]
            if not spare_rows or not spare_cols:
                continue
            # One row-only and one column-only color explicitly establish inertia.
            available = [q for q in queries if q not in used_colors]
            if len(available) < 2:
                continue
            rows[available[0]] = [int(rng.choice(spare_rows))]
            cols[available[1]] = [int(rng.choice(spare_cols))]
        crop_colors = (a, b) if not swap else (b, a)
        panel = np.full((bits.shape[0]+2, bits.shape[1]+2), frame, dtype=np.int8)
        panel[1:-1, 1:-1] = np.where(bits, crop_colors[1], crop_colors[0])
        for color, rr in rows.items():
            for y in rr:
                panel[y+1, 0] = panel[y+1, -1] = color
        for color, cc in cols.items():
            for x in cc:
                panel[0, x+1] = panel[-1, x+1] = color
        return panel, new_owner
    raise RuntimeError("bounded request search failed")


def _layout(rng, panels, paper):
    choices = []
    n = len(panels)
    for order in permutations(range(n)):
        for breaks in range(1 << (n-1)):
            shelves = [[order[0]]]
            for j in range(n-1):
                if breaks & (1 << j):
                    shelves.append([])
                shelves[-1].append(order[j+1])
            widths = [sum(panels[i].shape[1] for i in s)+len(s)-1 for s in shelves]
            heights = [max(panels[i].shape[0] for i in s) for s in shelves]
            H, W = sum(heights)+len(shelves)-1, max(widths)
            if H <= 28 and W <= 28:
                choices.append((shelves, heights, widths, H, W))
    if not choices:
        raise ValueError("panels do not fit the ARC canvas")
    shelves, heights, widths, H, W = choices[int(rng.integers(len(choices)))]
    extra_h = int(rng.integers(min(2, 28-H)+1))
    extra_w = int(rng.integers(min(2, 28-W)+1))
    canvas = np.full((H+2+extra_h, W+2+extra_w), paper, dtype=np.int8)
    y = 1+int(rng.integers(extra_h+1))
    for shelf, height, width in zip(shelves, heights, widths):
        x = 1+int(rng.integers(W+extra_w-width+1))
        for i in shelf:
            panel = panels[i]
            yy = y+int(rng.integers(height-panel.shape[0]+1))
            canvas[yy:yy+panel.shape[0], x:x+panel.shape[1]] = panel
            x += panel.shape[1]+1
        y += height+1
    return canvas


def _pair(rng, model, split, index, seen):
    reference, labels, sizes, banks, cart, dual = model
    for _ in range(60):
        palette = list(map(int, rng.permutation(10)[:9]))
        paper, ink, frame = palette[:3]
        atlas = np.full((reference.shape[0]+2, reference.shape[1]+2), frame, dtype=np.int8)
        atlas[1:-1, 1:-1] = np.where(reference, ink, paper)
        owner, panels = {}, [atlas]
        if split == "train":
            count = 2 if index < 2 else int(rng.integers(1, 3))
            orientations = ([0, 1] if index == 0 else [4, int(rng.integers(8))]
                            if index == 1 else list(map(int, rng.integers(8, size=count))))
        else:
            count = int(rng.integers(1, 4))
            orientations = list(map(int, rng.integers(8, size=count)))
        try:
            for j, t in enumerate(orientations):
                kind = ("cartesian" if split == "train" and index == j == 0
                        else "dual" if split == "train" and index == 1 and j == 0
                        else "random")
                bank = cart if kind == "cartesian" else dual if kind == "dual" else banks[split == "train"]
                swap = j % 2 if split == "train" and index == 0 else int(rng.integers(2))
                panel, owner = _request_card(rng, reference, labels, sizes, bank, t, swap,
                                             palette, owner, kind,
                                             incomplete=split == "train" and index == j == 0)
                panels.append(panel)
            inp = _layout(rng, panels, paper)
        except (RuntimeError, ValueError):
            continue
        if _key(inp) in seen:
            continue
        # Ground truth paints construction-time component masks, without parsing
        # the composed input or calling solve.
        out = np.where(reference, ink, paper).astype(np.int8)
        for v, color in owner.items():
            out[labels == v] = color
        parsed = _parse(inp)
        if not np.array_equal(_render(parsed), out):
            raise AssertionError("construction disagrees with visible registration")
        if split == "train" and index == 0:
            if any(np.array_equal(out, _render(parsed, mode))
                   for mode in ("zip", "all_crop", "local", "pixels")):
                continue
        if split == "test" and np.array_equal(out, _render(parsed, "local")):
            continue
        seen.add(_key(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("bounded example construction failed")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    model = _reference(rng)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            puzzle[split].append(_pair(rng, model, split, index, seen))
    # If diagonal contact matters in a test, demonstrate four-neighbor ownership
    # before testing it. Keep the identity/quarter-turn/cartesian witness intact.
    def distinguishes_eight(pair):
        p = _parse(pair["input"])
        return not np.array_equal(_render(p), _render(p, "eight"))
    if any(distinguishes_eight(p) for p in puzzle["test"]):
        for _ in range(40):
            if any(distinguishes_eight(p) for p in puzzle["train"]):
                break
            seen.remove(_key(puzzle["train"][1]["input"]))
            puzzle["train"][1] = _pair(rng, model, "train", 1, seen)
        else:
            raise RuntimeError("could not demonstrate four-neighbor selection")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, refs, records = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            p = _parse(inp)
            refs.add(_key(p["reference"]))
            assert 4 <= len(p["sizes"]) <= 7
            assert all(10 <= n <= 16 for n in p["reference"].shape)
            assert min(p["sizes"][1:]) >= 5
            assert 1 <= len(p["cards"]) <= 3
            assert out.shape == p["reference"].shape
            assert np.array_equal(_render(p), out), "incorrect registered component selection"
            owners = _owners(p)
            assert owners
            # Whole regions, unchanged paper, and unchanged unselected objects.
            assert np.all(out[p["reference"] == 0] == p["paper"])
            for v in range(1, len(p["sizes"])):
                assert np.all(out[p["labels"] == v] == owners.get(v, p["ink"]))
            assert any(0 < np.count_nonzero(card["components"] == v) < p["sizes"][v]
                       for card in p["cards"] for v in owners)
            records[split].append(p)
    assert len(refs) == 1, "reference geometry must stay fixed within the puzzle"
    train = records["train"]
    for mode in ("zip", "all_crop", "local", "pixels"):
        assert any(not np.array_equal(_render(p), _render(p, mode)) for p in train), mode
    orientations = {card["orientation"] for p in train for card in p["cards"]}
    assert {0, 1, 4} <= orientations, "training must expose rotation and reflection"
    assert any(set(card["rows"]) != set(card["cols"]) for p in train for card in p["cards"])
    assert any(len(set(card["rows"]) & set(card["cols"])) >= 2
               for p in train for card in p["cards"])
    assert all(card["patch"].size >= 25 for p in train for card in p["cards"])
    assert all(card["patch"].size < 25 for p in records["test"] for card in p["cards"])
    if any(not np.array_equal(_render(p), _render(p, "eight")) for p in records["test"]):
        assert any(not np.array_equal(_render(p), _render(p, "eight")) for p in train)


def self_test():
    corpus = [generate_puzzle(seed) for seed in range(42)]
    signatures, refs, dimensions, crop_shapes = set(), set(), set(), set()
    orientations, component_counts, palettes, layouts, products = set(), set(), set(), set(), set()
    for puzzle in corpus:
        signatures.add(tuple((_key(pair["input"]), _key(pair["output"]))
                             for s in ("train", "test") for pair in puzzle[s]))
        for s in ("train", "test"):
            for pair in puzzle[s]:
                p = _parse(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                refs.add(min(_key(_d4(p["reference"], t) ^ sw)
                             for t in range(8) for sw in (0, 1)))
                dimensions.add(p["reference"].shape)
                component_counts.add(len(p["sizes"])-1)
                palettes.add((p["paper"], p["ink"], p["frame"]))
                layouts.add(pair["input"].shape)
                for card in p["cards"]:
                    crop_shapes.add(card["patch"].shape)
                    orientations.add(card["orientation"])
                    products.update((len(card["rows"][q]), len(card["cols"][q]))
                                    for q in set(card["rows"]) & set(card["cols"]))
    assert len(signatures) == len(refs) == 42
    assert len(dimensions) >= 20 and len(layouts) >= 40
    assert len(crop_shapes) >= 16 and len(palettes) >= 100
    assert orientations == set(range(8))
    assert {3, 4, 5} <= component_counts
    assert {(1, 1), (2, 2), (1, 2), (2, 1)} <= products
    first = corpus[0]
    again = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in ("train", "test")
               for a, b in zip(first[s], again[s]) for k in ("input", "output"))
    # The host checks every legal count combination. Keep this local test bounded
    # while also exercising the smallest and largest evidence budgets.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(100+nt*4+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Mutations use actual wrong transformations, rather than mirrored code tests.
    for mode in ("zip", "all_crop", "local", "pixels"):
        bad = {s: [{k: np.asarray(v).copy() for k, v in a.items()} for a in first[s]]
               for s in ("train", "test")}
        bad["train"][0]["output"] = _render(_parse(bad["train"][0]["input"]), mode)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted shortcut: {mode}")
    diagonal_witnesses = 0
    for puzzle in corpus:
        for index, pair in enumerate(puzzle["train"]):
            p = _parse(pair["input"])
            wrong = _render(p, "eight")
            if np.array_equal(wrong, pair["output"]):
                continue
            diagonal_witnesses += 1
            if diagonal_witnesses == 1:
                bad = {s: [{k: np.asarray(v).copy() for k, v in a.items()} for a in puzzle[s]]
                       for s in ("train", "test")}
                bad["train"][index]["output"] = wrong
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("accepted diagonal merging")
    assert diagonal_witnesses >= 5
    return {"seeds": 42, "reference_classes": len(refs), "atlas_shapes": len(dimensions),
            "input_shapes": len(layouts), "crop_shapes": len(crop_shapes),
            "component_counts": sorted(component_counts), "orientations": sorted(orientations),
            "coordinate_products": sorted(products), "diagonal_witnesses": diagonal_witnesses}
