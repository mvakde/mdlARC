"""Anti-picture jigsaws: color-matched exclusions select a rigid square cover.

Only Python's standard library and NumPy are used. Generation and solving have
no shared cache or latent parameters. All searches and diagnostics stay in RAM.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Invalid color")
    return a.astype(np.int16, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role])
                h.update(bytes(a.shape))
                h.update(a.astype(np.uint8).tobytes())
    return h.hexdigest()


def _tight(a, mask):
    rr, cc = np.nonzero(mask)
    _require(len(rr) > 0, "Empty piece")
    p = np.full((int(rr.max()-rr.min()+1), int(cc.max()-cc.min()+1)), -1, dtype=np.int16)
    p[rr-rr.min(), cc-cc.min()] = a[rr, cc]
    return p


def _connected(mask, neighbors=NEIGHBORS):
    remaining = set(map(tuple, np.argwhere(mask)))
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            q = (r+dr, c+dc)
            if q in remaining:
                remaining.remove(q)
                stack.append(q)
    return not remaining


def _components(a, background):
    remaining = set(map(tuple, np.argwhere(a != background)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBORS:
                q = (r+dr, c+dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        mask = np.zeros(a.shape, dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        result.append(_tight(a, mask))
    return result


def _orientations(p):
    seen, result = set(), []
    for k in range(4):
        q = np.rot90(p, k).copy()
        key = q.shape, q.tobytes()
        if key not in seen:
            seen.add(key)
            result.append(q)
    return result


def _carrier(p, marker):
    values = set(map(int, np.unique(p[p >= 0]))) - {marker}
    _require(len(values) == 1, "A rigid piece must have one carrier")
    return next(iter(values))


def _pack(pieces, side, marker, chart=None, limit=2):
    """Exhaustive colored exact cover; stop only after `limit` distinct answers.

    A placement's carrier owns its marker cell too. Masks may be symmetric;
    deduplicating identical masks loses no colored answer for these pieces.
    """
    if sum(int(np.sum(p >= 0)) for p in pieces) != side*side:
        return []
    marked = [i for i, p in enumerate(pieces) if np.any(p == marker)]
    if len(marked) != 1:
        return []
    anchor = marked[0]
    carriers = [_carrier(p, marker) for p in pieces]
    choices = []
    for i, p in enumerate(pieces):
        forbidden = 0 if chart is None else sum(
            1 << int(j) for j in np.flatnonzero(chart == carriers[i]))
        masks = set()
        for q in _orientations(p):
            h, w = q.shape
            if h > side or w > side or (i == anchor and q[0, 0] != marker):
                continue
            base = sum(1 << (int(r)*side+int(c)) for r, c in np.argwhere(q >= 0))
            offsets = [0] if i == anchor else [
                r*side+c for r in range(side-h+1) for c in range(side-w+1)]
            for offset in offsets:
                mask = base << offset
                if not mask & forbidden:
                    masks.add(mask)
        if not masks:
            return []
        choices.append(sorted(masks))
    full = (1 << (side*side))-1
    placed, answers = [0]*len(pieces), []

    def search(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            out = np.empty((side, side), dtype=np.int16)
            for i, mask in enumerate(placed):
                while mask:
                    bit = mask & -mask
                    out.flat[bit.bit_length()-1] = carriers[i]
                    mask ^= bit
            out[0, 0] = marker
            answers.append(out)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        # Every exact cover fills this cell with precisely one remaining piece.
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for mask in choices[i]:
                if mask & first and not mask & occupied:
                    placed[i] = mask
                    search(occupied | mask, rest)
                    if len(answers) >= limit:
                        return

    rest = tuple(i for i in range(len(pieces)) if i != anchor)
    for mask in choices[anchor]:
        placed[anchor] = mask
        search(mask, rest)
        if len(answers) >= limit:
            break
    return answers


def _read_input(value):
    a = _grid(value)
    bg = int(a[0, 0])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "Input must have a uniform background border")
    candidates = []
    for frame_color in map(int, np.unique(a)):
        if frame_color == bg:
            continue
        rr, cc = np.nonzero(a == frame_color)
        y, x, bottom, right = int(rr.min()), int(cc.min()), int(rr.max()+1), int(cc.max()+1)
        length = bottom-y
        if length < 6 or right-x != length or len(rr) != 4*length-5:
            continue
        card = a[y:bottom, x:right]
        perimeter = np.ones(card.shape, dtype=bool)
        perimeter[1:-1, 1:-1] = False
        missing = np.argwhere(perimeter & (card != frame_color))
        if len(missing) != 1:
            continue
        my, mx = map(int, missing[0])
        if my not in (0, length-1) or mx not in (0, length-1):
            continue
        marker = int(card[my, mx])
        if marker in (bg, frame_color) or np.count_nonzero(a == marker) != 2:
            continue
        if np.any(card[1:-1, 1:-1] == frame_color) or np.any(card[1:-1, 1:-1] == marker):
            continue
        for turns in range(4):
            normalized = np.rot90(card, turns)
            if normalized[0, 0] == marker:
                candidates.append((frame_color, marker, y, x, bottom, right, turns, normalized.copy()))
    _require(len(candidates) == 1, "Exactly one registered atlas is required")
    frame, marker, y, x, bottom, right, turns, card = candidates[0]
    side = card.shape[0]-2
    _require(4 <= side <= 8, "Supported square sides are 4..8")
    chart = card[1:-1, 1:-1].copy()
    chart[chart == bg] = -1
    outside = a.copy()
    outside[y:bottom, x:right] = bg
    _require(np.all(outside[max(0, y-1):min(a.shape[0], bottom+1),
                            max(0, x-1):min(a.shape[1], right+1)] == bg),
             "Pieces must be separated from the atlas by background")
    pieces = _components(outside, bg)
    _require(3 <= len(pieces) <= 5, "Require three to five separated pieces")
    _require(sum(int(np.sum(p >= 0)) for p in pieces) == side*side, "Piece area must equal map area")
    _require(np.count_nonzero(outside == marker) == 1, "Require one carried marker")
    carriers = [_carrier(p, marker) for p in pieces]
    _require(len(set(carriers)) == len(pieces), "Carrier colors must be distinct")
    _require(all(np.sum(p >= 0) >= 3 for p in pieces), "Pieces must have at least three cells")
    _require(all(_connected(p >= 0) for p in pieces), "Disconnected rigid object")
    ink = set(map(int, np.unique(chart[chart >= 0])))
    _require(set(carriers) <= ink, "Each carrier needs a matching negative stencil")
    _require(len(ink-set(carriers)) <= 2, "At most two unmatched stencil colors")
    _require(int(np.sum(chart >= 0))*100 <= 65*side*side, "Map must remain partial")
    return dict(background=bg, frame=frame, marker=marker, side=side, chart=chart,
                pieces=pieces, carriers=carriers, atlas_turns=turns,
                atlas_box=(y, x, bottom, right), distractors=len(ink-set(carriers)))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive a unique answer from the input, without examples or generation."""
    f = _read_input(input_grid)
    answers = _pack(f["pieces"], f["side"], f["marker"], f["chart"])
    _require(len(answers) == 1, "Require exactly one exclusion-constrained assembly")
    return answers[0].copy()


def _partition(rng, side, count, style, diagonal=False):
    labels = np.full((side, side), -1, dtype=np.int16)
    if style == "cuts":
        rects = [(0, 0, side, side)]
        while len(rects) < count:
            options = [(i, axis, k) for i, (r, c, h, w) in enumerate(rects)
                       for axis, length, breadth in ((0, h, w), (1, w, h))
                       for k in range(1, length) if min(k, length-k)*breadth >= 3]
            if not options:
                return None
            i, axis, k = options[int(rng.integers(len(options)))]
            r, c, h, w = rects.pop(i)
            rects.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                         [(r, c, h, k), (r, c+k, h, w-k)])
        for i, (r, c, h, w) in enumerate(rects):
            labels[r:r+h, c:c+w] = i
        for _ in range(int(rng.integers(side, side*4+1))):
            r, c = map(int, rng.integers(side, size=2))
            neighbors = NEIGHBORS if diagonal and rng.random() < .45 else EDGE
            dr, dc = neighbors[int(rng.integers(len(neighbors)))]
            nr, nc = r+dr, c+dc
            if not (0 <= nr < side and 0 <= nc < side):
                continue
            old, new = int(labels[r, c]), int(labels[nr, nc])
            if old == new or np.sum(labels == old) <= 3 or rng.random() < .35:
                continue
            labels[r, c] = new
            # Some corner-only attachments are valid, as in B's source.
            if not _connected(labels == old, NEIGHBORS if diagonal else EDGE):
                labels[r, c] = old
    else:
        starts = list(map(int, rng.choice(side*side, count, replace=False)))
        adjacent = [[rr*side+cc for dr, dc in EDGE
                     if 0 <= (rr := r+dr) < side and 0 <= (cc := c+dc) < side]
                    for r in range(side) for c in range(side)]
        frontiers = [set() for _ in range(count)]
        sizes = [1]*count
        weights = np.exp(rng.uniform(-.8, .8, count))
        for i, pos in enumerate(starts):
            labels.flat[pos] = i
        for i, pos in enumerate(starts):
            frontiers[i].update(p for p in adjacent[pos] if labels.flat[p] < 0)
        for _ in range(side*side-count):
            eligible = [i for i in range(count) if frontiers[i]]
            i = min(eligible, key=lambda j: sizes[j]/weights[j]+float(rng.random())*.5)
            choices = sorted(frontiers[i])
            pos = choices[int(rng.integers(len(choices)))]
            labels.flat[pos] = i
            sizes[i] += 1
            for frontier in frontiers:
                frontier.discard(pos)
            frontiers[i].update(p for p in adjacent[pos] if labels.flat[p] < 0)
    if min(int(np.sum(labels == i)) for i in range(count)) < 3:
        return None
    return labels


def _objects(labels, carriers, marker):
    anchor = int(labels[0, 0])
    order = [anchor]+[i for i in range(len(carriers)) if i != anchor]
    target = np.empty(labels.shape, dtype=np.int16)
    for i, label in enumerate(order):
        target[labels == label] = carriers[i]
    owners = target.copy()
    target[0, 0] = marker
    pieces = [_tight(target, labels == label) for label in order]
    return target, owners, pieces


def _twin_partition(rng, side):
    """Two congruent corner-connected inserts in an independently shaped rest.

    Sample masks, orientations and positions rather than a stored silhouette.
    Exchanging their carrier identities guarantees geometric ambiguity.
    """
    for _ in range(64):
        h = 2 if side == 4 else int(rng.integers(2, min(4, side)+1))
        w = 3 if side == 4 else int(rng.integers(2, min(4, side)+1))
        mask = rng.random((h, w)) < float(rng.uniform(.3, .65))
        if not 3 <= mask.sum() <= min(h*w-1, (side*side-3)//2):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _connected(mask) or _connected(mask, EDGE):
            continue
        positions = []
        seen = set()
        for q in _orientations(mask.astype(np.int16)):
            hh, ww = q.shape
            for r in range(side-hh+1):
                for c in range(side-ww+1):
                    full = np.zeros((side, side), dtype=bool)
                    full[r:r+hh, c:c+ww] = q.astype(bool)
                    if full[0, 0] or full.tobytes() in seen:
                        continue
                    seen.add(full.tobytes())
                    positions.append(full)
        pairs = [(i, j) for i in range(len(positions)) for j in range(i)
                 if not np.any(positions[i] & positions[j])]
        for index in rng.permutation(len(pairs)):
            i, j = pairs[int(index)]
            rest = ~(positions[i] | positions[j])
            if not _connected(rest):
                continue
            labels = np.zeros((side, side), dtype=np.int16)
            labels[positions[i]] = 1
            labels[positions[j]] = 2
            return labels
    return None


def _negative_map(rng, pieces, target, owners, carriers, marker):
    side = target.shape[0]
    chart = np.full(target.shape, -1, dtype=np.int16)
    for color in rng.permutation(carriers):
        positions = np.flatnonzero((owners != color) & (chart < 0))
        if len(positions) == 0:
            return None
        chart.flat[int(rng.choice(positions))] = int(color)
    for _ in range(side*side):
        answers = _pack(pieces, side, marker, chart)
        _require(answers, "Constructed target must remain feasible")
        if len(answers) == 1:
            _require(np.array_equal(answers[0], target), "Unexpected target")
            return chart
        alternative = next(a for a in answers if not np.array_equal(a, target))
        positions = np.flatnonzero((alternative != target) & (chart < 0) & (alternative != marker))
        if not len(positions):
            return None
        pos = int(rng.choice(positions))
        chart.flat[pos] = alternative.flat[pos]
        if np.count_nonzero(chart >= 0)*100 > side*side*55:
            return None
    return None


def _swap_witness(chart, out, marker):
    positions = list(map(int, np.flatnonzero((chart < 0) & (out != marker))))
    if not positions:
        return None
    first = positions[0]
    other = next((p for p in positions if out.flat[p] != out.flat[first]), None)
    return None if other is None else (first, other)


def _add_ink(rng, chart, owners, carriers, distractors):
    chart = chart.copy()
    area = chart.size
    # Extra valid exclusions vary clue density; the uniqueness proof is retained.
    desired = min(int(area*.65)-len(distractors),
                  max(int(np.count_nonzero(chart >= 0)), int(rng.uniform(.12, .43)*area)))
    for pos in rng.permutation(np.flatnonzero(chart < 0)):
        if np.count_nonzero(chart >= 0) >= desired:
            break
        options = [c for c in carriers if c != owners.flat[int(pos)]]
        chart.flat[int(pos)] = int(rng.choice(options))
    for index, color in enumerate(distractors):
        room = int(area*.65)-int(np.count_nonzero(chart >= 0))
        positions = np.flatnonzero(chart < 0)
        available = room-(len(distractors)-index-1)
        if available < 1:
            return None
        amount = int(rng.integers(1, min(3, available)+1))
        for pos in rng.choice(positions, amount, replace=False):
            chart.flat[int(pos)] = color
    return chart


def _scatter(rng, pieces, chart, background, frame, marker, atlas_rotation, anchor_rotation):
    side = chart.shape[0]
    atlas = np.full((side+2, side+2), frame, dtype=np.int16)
    atlas[1:-1, 1:-1] = np.where(chart < 0, background, chart)
    atlas[0, 0] = marker
    oriented = [np.rot90(atlas, atlas_rotation).copy()]
    for i, piece in enumerate(pieces):
        rotation = anchor_rotation if i == 0 else int(rng.integers(4))
        oriented.append(np.rot90(piece, rotation).copy())
    width = 28  # two guaranteed outer background cells fit within ARC's limit
    for _ in range(8):
        canvas = np.full((width, width), background, dtype=np.int16)
        blocked = 0
        lowr = lowc = width
        highr = highc = 0
        order = [0]+list(map(int, rng.permutation(np.arange(1, len(oriented)))))
        okay = True
        jitter = float(rng.choice([5, 18, 55]))
        for index in order:
            p = oriented[index]
            h, w = p.shape
            coords = [(int(r), int(c)) for r, c in np.argwhere(p >= 0)]
            base = sum(1 << (r*width+c) for r, c in coords)
            options = []
            for r in range(width-h+1):
                for c in range(width-w+1):
                    mask = base << (r*width+c)
                    if mask & blocked:
                        continue
                    if index == 0:
                        score = float(rng.random())
                    else:
                        bh = max(highr, r+h)-min(lowr, r)
                        bw = max(highc, c+w)-min(lowc, c)
                        score = bh*bw + float(rng.random())*jitter
                    options.append((score, r, c))
            if not options:
                okay = False
                break
            _, r, c = min(options)
            view = canvas[r:r+h, c:c+w]
            view[p >= 0] = p[p >= 0]
            for pr, pc in coords:
                rr, cc = r+pr, c+pc
                for nr in range(max(0, rr-1), min(width, rr+2)):
                    for nc in range(max(0, cc-1), min(width, cc+2)):
                        blocked |= 1 << (nr*width+nc)
            lowr, lowc = min(lowr, r), min(lowc, c)
            highr, highc = max(highr, r+h), max(highc, c+w)
        if okay:
            canvas = canvas[lowr:highr, lowc:highc]
            padding = []
            for length in canvas.shape:
                extra = int(rng.integers(min(4, 28-length)+1))
                first = int(rng.integers(extra+1))
                padding.append((1+first, 1+extra-first))
            return np.pad(canvas, padding, constant_values=background)
    return None


def _fallback(side, count, carriers, marker):
    """Guaranteed ambiguous geometry plus a certified sparse unique anti-map.

    A marked L frames (s-1)-wide horizontal slabs. Each target row forbids
    every other insert carrier. The first interior column contains all insert
    carriers as exclusions, ruling out an all-vertical slab arrangement.
    """
    labels = np.zeros((side, side), dtype=np.int16)
    k = count-1
    edges = [1]+[1+i*(side-1)//k for i in range(1, k)]+[side]
    for i in range(k):
        labels[edges[i]:edges[i+1], 1:] = i+1
    target, owners, pieces = _objects(labels, carriers, marker)
    chart = np.full((side, side), -1, dtype=np.int16)
    inserts = carriers[1:]
    for i in range(k):
        for j in range(1, k):
            chart[edges[i]:edges[i+1], j] = inserts[(i+j) % k]
    chart[1, -1] = carriers[0]
    _require(len(_pack(pieces, side, marker)) == 2, "Fallback needs geometric ambiguity")
    answers = _pack(pieces, side, marker, chart)
    _require(len(answers) == 1 and np.array_equal(answers[0], target), "Fallback must be unique")
    return target, owners, pieces, chart


def _make_pair(rng, side, count, background, frame, marker, carriers, distractors,
               atlas_rotation, anchor_rotation, demonstration, diagonal=False,
               require_diagonal=False):
    for attempt in range(128):
        style = "cuts" if rng.random() < .75 else "growth"
        if count == 3 and require_diagonal and rng.random() < .85:
            labels = _twin_partition(rng, side)
        else:
            labels = _partition(rng, side, count, style, diagonal)
        if labels is None:
            continue
        target, owners, pieces = _objects(labels, carriers, marker)
        has_diagonal = any(not _connected(p >= 0, EDGE) for p in pieces)
        if require_diagonal and not has_diagonal:
            continue
        if demonstration and min(pieces[0].shape) < 2:
            continue
        if len(_pack(pieces, side, marker)) != 2:
            continue
        chart = _negative_map(rng, pieces, target, owners, carriers, marker)
        if chart is None:
            continue
        chart = _add_ink(rng, chart, owners, carriers, distractors)
        if chart is None or _swap_witness(chart, target, marker) is None:
            continue
        inp = _scatter(rng, pieces, chart, background, frame, marker, atlas_rotation, anchor_rotation)
        if inp is not None:
            return {"input": inp, "output": target}, dict(proposals=attempt+1, fallback=0,
                                                         diagonal=has_diagonal)
    target, owners, pieces, chart = _fallback(side, count, carriers, marker)
    # The safety map already supplies the proof; add only unmatched ink while
    # retaining blank positions of different target colors.
    for _ in range(32):
        augmented = _add_ink(rng, chart, owners, carriers, distractors)
        if augmented is None or _swap_witness(augmented, target, marker) is None:
            continue
        inp = _scatter(rng, pieces, augmented, background, frame, marker, atlas_rotation, anchor_rotation)
        if inp is not None:
            return {"input": inp, "output": target}, dict(proposals=128, fallback=1, diagonal=False)
    raise AssertionError("Bounded construction failed")


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _build_puzzle(seed, num_train, num_test):
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    frame, marker = map(int, rng.choice(10, 2, replace=False))
    backgrounds = [c for c in range(10) if c not in (frame, marker)]
    fixed_bg = int(rng.choice(backgrounds))
    mixed_bg = rng.random() < .8
    scale_shift = rng.random() < .6
    diagonal_mode = bool(rng.random() < .6)
    diagonal_seen = False
    atlas_base = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    stats = dict(proposals=0, fallbacks=0)
    first_bg = None
    for i in range(nt+ne):
        split = "train" if i < nt else "test"
        if i == 0:
            side, count, nd = 4, 3, 1
        elif i == 1:
            side, count, nd = 5, 4, 0
        elif split == "train":
            count = int(rng.integers(3, 6 if scale_shift else 5))
            side, nd = max(count, int(rng.integers(4, 7))), int(rng.integers(3))
        else:
            count = int(rng.integers(3, 6)) if scale_shift else 5
            side = int(rng.integers(7, 9)) if scale_shift else int(rng.integers(5, 9))
            nd = int(rng.integers(3))
        bg_choices = [c for c in backgrounds if i != 1 or c != first_bg]
        bg = int(rng.choice(bg_choices)) if mixed_bg else fixed_bg
        if i == 0:
            first_bg = bg
        available = [c for c in range(10) if c not in (bg, frame, marker)]
        colors = list(map(int, rng.choice(available, count+nd, replace=False)))
        carriers, distractors = colors[:count], colors[count:]
        atlas_rotation = (atlas_base+i) % 4 if i < 2 else int(rng.integers(4))
        anchor_rotation = (1 if i == 0 else 3) if i < 2 else int(rng.integers(4))
        for _ in range(16):
            pair, info = _make_pair(rng, side, count, bg, frame, marker, carriers, distractors,
                                    atlas_rotation, anchor_rotation, i < 2,
                                    diagonal_mode and (i == 0 or diagonal_seen),
                                    diagonal_mode and i == 0)
            stats["proposals"] += info["proposals"]
            stats["fallbacks"] += info["fallback"]
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    diagonal_seen |= info["diagonal"]
                break
        else:
            raise AssertionError("Could not construct distinct inputs")
    return puzzle, stats


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return exactly train/test lists; the entire bundle is seed-reproducible."""
    return _build_puzzle(seed, num_train, num_test)[0]


def _features(pair, facts=None):
    f = _read_input(pair["input"]) if facts is None else facts
    out = np.asarray(pair["output"])
    anchor = next(_carrier(p, f["marker"]) for p in f["pieces"] if np.any(p == f["marker"]))
    rotations = 0
    diagonal = symmetric = bars = blocks = 0
    for p, color in zip(f["pieces"], f["carriers"]):
        mask = (out == color) | ((out == f["marker"]) if color == anchor else False)
        rotations += not np.array_equal(p, _tight(out, mask))
        diagonal += not _connected(p >= 0, EDGE)
        symmetric += len(_orientations((p >= 0).astype(np.int16))) < 4
        bars += min(p.shape) == 1
        blocks += bool(np.all(p >= 0))
    areas = [int(np.sum(p >= 0)) for p in f["pieces"]]
    return dict(side=f["side"], count=len(f["pieces"]), background=f["background"],
                frame=f["frame"], marker=f["marker"], atlas_turns=f["atlas_turns"],
                rotation_required=rotations, diagonal=diagonal, symmetric=symmetric,
                bars=bars, blocks=blocks, clues=int(np.sum(f["chart"] >= 0)),
                distractors=f["distractors"], area_ratio=max(areas)/min(areas))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    nt, ne = _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            f = _read_input(inp)
            s, marker, bg = f["side"], f["marker"], f["background"]
            _require(out.shape == (s, s) and out[0, 0] == marker, "Incorrect square or anchor")
            _require(not np.any((out == bg) | (out == f["frame"])), "Output must consist only of pieces")
            answers = _pack(f["pieces"], s, marker, f["chart"])
            _require(len(answers) == 1 and np.array_equal(answers[0], out), "Incorrect or ambiguous constrained cover")
            _require(len(_pack(f["pieces"], s, marker)) == 2, "Ablating A must leave multiple colored answers")
            _require(_swap_witness(f["chart"], out, marker) is not None, "Need a map-compatible nonrigid swap")
            anchor = next(_carrier(p, marker) for p in f["pieces"] if np.any(p == marker))
            owners = out.copy()
            owners[owners == marker] = anchor
            _require(not np.any((f["chart"] >= 0) & (f["chart"] == owners)), "Positive map ink is forbidden")
            for color in f["carriers"]+[marker]:
                count = sum(int(np.sum(p == color)) for p in f["pieces"])
                _require(np.count_nonzero(out == color) == count, "Piece color count changed")
            facts[split].append(_features(pair, f))
    train, test = facts["train"], facts["test"]
    _require(len({f["frame"] for f in train+test}) == 1 and len({f["marker"] for f in train+test}) == 1,
             "Frame and marker colors must be consistent within a puzzle")
    _require(all(f["rotation_required"] for f in train[:2]), "First demonstrations must establish rigid rotation")
    _require(len({f["atlas_turns"] for f in train[:2]}) == 2, "Demonstrate independent atlas orientation")
    _require(any(f["distractors"] for f in train) and any(not f["distractors"] for f in train),
             "Demonstrate both matched-only and unmatched stencil maps")
    _require(not any(f["diagonal"] for f in test) or any(f["diagonal"] for f in train),
             "Demonstrate corner-inclusive rigid grouping before testing it")
    scale_shift = min(f["side"] for f in test) > max(f["side"] for f in train)
    count_shift = min(f["count"] for f in test) > max(f["count"] for f in train)
    _require(scale_shift or count_shift, "Require a visible scale or cardinality shift")
    return None


def self_test():
    """Bounded streaming audit over 42 seeds; returns scalar metrics, no grids."""
    started = time.monotonic()
    signatures, inputs, outputs, normalized = set(), set(), set(), set()
    sides, counts, backgrounds, atlas_turns, distractors = set(), set(), set(), set(), set()
    totals = dict(pairs=0, fallbacks=0, proposals=0, rotated_pairs=0, diagonal_pieces=0,
                  symmetric_pieces=0, bars=0, blocks=0, map_orientation_matters=0,
                  scale_shift_puzzles=0, count_shift_puzzles=0)
    max_ratio, min_density, max_density = 0., 1., 0.
    for seed in range(42):
        puzzle, generation = _build_puzzle(seed, 4, 1)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        totals["fallbacks"] += generation["fallbacks"]
        totals["proposals"] += generation["proposals"]
        train_sides, train_counts, test_sides, test_counts = [], [], [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _read_input(pair["input"])
                feature = _features(pair, f)
                before = pair["input"].copy()
                prediction = solve(pair["input"])
                _require(np.array_equal(prediction, pair["output"]), "Input-only prediction disagrees")
                _require(np.array_equal(before, pair["input"]), "Solver mutated input")
                totals["pairs"] += 1
                totals["rotated_pairs"] += bool(feature["rotation_required"])
                for field, name in (("diagonal", "diagonal_pieces"), ("symmetric", "symmetric_pieces"),
                                    ("bars", "bars"), ("blocks", "blocks")):
                    totals[name] += feature[field]
                inputs.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                sides.add(f["side"])
                counts.add(len(f["pieces"]))
                backgrounds.add(f["background"])
                atlas_turns.add(f["atlas_turns"])
                distractors.add(f["distractors"])
                roles = {}
                normalized.add((pair["output"].shape, bytes(
                    roles.setdefault(int(c), len(roles)) for c in pair["output"].flat)))
                density = feature["clues"]/(f["side"]**2)
                min_density, max_density = min(min_density, density), max(max_density, density)
                max_ratio = max(max_ratio, feature["area_ratio"])
                if split == "train":
                    train_sides.append(f["side"])
                    train_counts.append(len(f["pieces"]))
                else:
                    test_sides.append(f["side"])
                    test_counts.append(len(f["pieces"]))
                # Counterfactual: read the rotated atlas literally instead of
                # normalizing its marker. Compare its admissible answers.
                if f["atlas_turns"]:
                    raw = np.rot90(f["chart"], -f["atlas_turns"])
                    wrong = _pack(f["pieces"], f["side"], f["marker"], raw)
                    totals["map_orientation_matters"] += not (
                        len(wrong) == 1 and np.array_equal(wrong[0], pair["output"]))
        totals["scale_shift_puzzles"] += min(test_sides) > max(train_sides)
        totals["count_shift_puzzles"] += min(test_counts) > max(train_counts)
        if seed in (0, 1, 41):
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "Nondeterministic generation")
    _require(len(signatures) == 42, "Repeated puzzles across seeds")
    _require(sides == set(range(4, 9)) and counts == {3, 4, 5}, "Missing scale/count variation")
    _require(atlas_turns == {0, 1, 2, 3} and distractors == {0, 1, 2}, "Missing map variation")
    _require(len(backgrounds) >= 8 and len(inputs) >= 30, "Insufficient palette/canvas variation")
    _require(len(normalized) >= 100, "Insufficient assembled geometry variation")
    _require(totals["map_orientation_matters"] >= 10, "Map normalization must affect answers")
    _require(totals["bars"] > 0 and totals["symmetric_pieces"] > 0 and totals["diagonal_pieces"] > 0,
             "Exercise simple, symmetric and diagonal pieces")
    _require(totals["scale_shift_puzzles"] > 0 and totals["count_shift_puzzles"] > 0,
             "Exercise both curricula")
    # All explicit count combinations, plus malformed count types and values.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+nt*10+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("Illegal counts were accepted")
    # Certify the safety construction throughout its supported parameter range.
    for side in range(4, 9):
        for count in range(3, min(side, 5)+1):
            target, owners, pieces, chart = _fallback(side, count, list(range(1, count+1)), 9)
            _require(_swap_witness(chart, target, 9) is not None, "Fallback lacks B ablation witness")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][0, 0] = int(damaged["train"][0]["input"][0, 0])
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a damaged anchor")
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(inputs),
                output_shapes=sorted([list(s) for s in outputs]),
                normalized_output_structures=len(normalized), sides=sorted(sides),
                piece_counts=sorted(counts), backgrounds=sorted(backgrounds),
                atlas_turns=sorted(atlas_turns), distractor_counts=sorted(distractors),
                clue_density=[min_density, max_density], maximum_area_ratio=max_ratio,
                totals=totals, seconds=round(time.monotonic()-started, 3),
                semantic_verification="generator-owned exhaustive checks; no independent solver")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
