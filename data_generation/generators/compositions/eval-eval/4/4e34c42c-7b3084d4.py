"""Stitched panels: shared patches register rigid panels in a filled square.

Only NumPy and the standard library are used. The solver reads visible color
ownership, glyphs, and geometry; generation coordinates are never solver inputs.
"""
from __future__ import annotations

from collections import defaultdict
from math import isqrt
import hashlib
import json

import numpy as np


EDGE = ((-1, 0), (0, -1), (0, 1), (1, 0))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "grid colors")
    return g.astype(np.int8, copy=True)


def _groups(points, neighbors=EDGE):
    remaining = set(points)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _norm(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return {(r-r0, c-c0): int(v) for (r, c), v in cells.items()}


def _shape(cells):
    return max(r for r, c in cells) + 1, max(c for r, c in cells) + 1


def _turn(cells):
    return _norm({(-c, r): v for (r, c), v in cells.items()})


def _key(cells):
    return tuple((r, c, v) for (r, c), v in sorted(cells.items()))


def _canonical(cells):
    current, keys = _norm(cells), []
    for _ in range(4):
        keys.append(_key(current))
        current = _turn(current)
    return min(keys)


def _glyphs(cells, shared):
    return [{p: cells[p] for p in group}
            for group in _groups(p for p, v in cells.items() if v in shared)]


def _decode(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    _require(counts.max() * 2 > g.size, "background must occupy over half the canvas")
    background = int(colors[counts.argmax()])
    singleton = [int(c) for c, n in zip(colors, counts) if n == 1 and c != background]
    _require(len(singleton) == 1, "one singleton anchor color required")
    marker = singleton[0]
    points = [tuple(map(int, p)) for p in np.argwhere(g != background)]
    parts = [_norm({p: int(g[p]) for p in group}) for group in _groups(points, EIGHT)]
    _require(2 <= len(parts) <= 6, "need two to six separated panels")
    owners = defaultdict(set)
    for i, part in enumerate(parts):
        for color in set(part.values()) - {marker}:
            owners[color].add(i)
    shared = {c for c, indices in owners.items() if len(indices) > 1}
    _require(len(shared) == 2, "two shared stitch colors required")
    carriers = []
    occurrences, motifs, part_ids = defaultdict(list), {}, []
    private_area = 1  # the anchor is private material, counted once
    for i, part in enumerate(parts):
        private = set(part.values()) - shared - {marker}
        _require(len(private) == 1, "one private carrier per panel")
        carrier = next(iter(private))
        carriers.append(carrier)
        area = sum(v == carrier for v in part.values())
        _require(area >= 2, "private carriers must not be singleton colors")
        private_area += area
        ids = {}
        for glyph in _glyphs(part, shared):
            _require(set(glyph.values()) == shared, "a stitch contains both shared colors")
            _require(2 <= len(glyph) <= 8, "stitch area must be two to eight")
            identity = _canonical(glyph)
            _require(identity not in ids, "a panel cannot contain two copies of the same stitch")
            ids[identity] = glyph
            occurrences[identity].append(i)
            motifs[identity] = len(glyph)
        part_ids.append(ids)
    _require(len(set(carriers)) == len(parts), "private carrier identities must differ")
    _require(occurrences and all(len(o) >= 2 for o in occurrences.values()),
             "every stitch must recur on at least two panels")
    area = private_area + sum(motifs.values())
    side = isqrt(area)
    _require(side * side == area and 4 <= side <= 12, "deduplicated area must be a square of side 4..12")
    adjacency = [set() for _ in parts]
    for indices in occurrences.values():
        for i in indices:
            adjacency[i].update(set(indices) - {i})
    clusters, remaining = [], set(range(len(parts)))
    while remaining:
        root = min(remaining)
        group, stack = {root}, [root]
        remaining.remove(root)
        while stack:
            for j in adjacency[stack.pop()] & remaining:
                remaining.remove(j)
                group.add(j)
                stack.append(j)
        clusters.append(group)
    symmetric = 0
    for identity in motifs:
        glyph = {(r, c): v for r, c, v in identity}
        orientations = set()
        for _ in range(4):
            orientations.add(_key(glyph))
            glyph = _turn(glyph)
        symmetric += len(orientations) < 4
    return dict(grid=g, background=background, marker=marker, shared=shared,
                parts=parts, part_ids=part_ids, motifs=motifs, occurrences=occurrences,
                side=side, raw_area=len(points), area=area, clusters=clusters,
                symmetric=symmetric, carriers=carriers)


def _candidates(facts):
    """All rigid quarter-turn placements, retaining the full colored footprint."""
    side, marker = facts['side'], facts['marker']
    lists = []
    for part in facts['parts']:
        marked = marker in part.values()
        current, seen, candidates = dict(part), set(), []
        for turn in range(4):
            fingerprint = _key(current)
            if fingerprint in seen:
                current = _turn(current)
                continue
            seen.add(fingerprint)
            h, w = _shape(current)
            if h <= side and w <= side:
                glyphs = {_canonical(m): m for m in _glyphs(current, facts['shared'])}
                if marked:
                    positions = [(0, 0)] if current.get((0, 0)) == marker else []
                else:
                    positions = [(r, c) for r in range(side-h+1) for c in range(side-w+1)]
                for r0, c0 in positions:
                    cells = tuple(sorted(((r+r0)*side+c+c0, v)
                                         for (r, c), v in current.items()))
                    mask = sum(1 << p for p, _ in cells)
                    keys, glyph_masks = {}, {}
                    for identity, glyph in glyphs.items():
                        key = tuple(sorted(((r+r0)*side+c+c0, v)
                                           for (r, c), v in glyph.items()))
                        keys[identity] = key
                        glyph_masks[identity] = sum(1 << p for p, _ in key)
                    candidates.append((mask, keys, glyph_masks, cells, turn, (r0, c0)))
            current = _turn(current)
        _require(candidates, "a panel cannot fit the deduplicated square")
        lists.append(candidates)
    return lists


def _assemble(facts, limit=2):
    """Joint registration/coverage search; no solved subassembly is assumed."""
    lists = _candidates(facts)
    side = facts['side']
    full = (1 << (side*side)) - 1
    anchor = next(i for i, p in enumerate(facts['parts']) if facts['marker'] in p.values())
    answers, signatures = [], set()
    placed = {}

    def compatible(candidate, occupied, registered):
        mask, keys, glyph_masks = candidate[:3]
        allowed = 0
        for identity, key in keys.items():
            if identity in registered:
                if registered[identity] != key:
                    return False
                allowed |= glyph_masks[identity]
        # Only corresponding COMPLETE stitch copies may overlap.
        return not ((mask & occupied) & ~allowed)

    def search(occupied, remaining, registered):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            answer = np.full((side, side), -1, dtype=np.int8)
            for candidate in placed.values():
                for p, v in candidate[3]:
                    answer.flat[p] = v
            signature = answer.tobytes()
            if signature not in signatures:
                signatures.add(signature)
                answers.append((answer, dict(placed)))
            return
        linked = [i for i in remaining if set(facts['part_ids'][i]) & registered.keys()]
        if linked:
            domains = [(i, [p for p in lists[i] if compatible(p, occupied, registered)]) for i in linked]
            i, options = min(domains, key=lambda item: (len(item[1]), item[0]))
            choices = [(i, p) for p in options]
        else:
            # A new stitch cluster must eventually cover the first empty cell.
            # Try whichever remaining panel supplies it; square coverage, not
            # input location or an arbitrary ordering, starts the cluster.
            first = (full ^ occupied) & -(full ^ occupied)
            choices = [(i, p) for i in remaining for p in lists[i]
                       if p[0] & first and compatible(p, occupied, registered)]
        for i, candidate in choices:
            placed[i] = candidate
            updated = dict(registered)
            updated.update(candidate[1])
            search(occupied | candidate[0], tuple(j for j in remaining if j != i), updated)
            del placed[i]
            if len(answers) >= limit:
                return

    for candidate in lists[anchor]:
        placed[anchor] = candidate
        search(candidate[0], tuple(i for i in range(len(lists)) if i != anchor), dict(candidate[1]))
        del placed[anchor]
        if len(answers) >= limit:
            break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _decode(input_grid)
    answers = _assemble(facts)
    _require(len(answers) == 1, "require exactly one colored anchored square assembly")
    return answers[0][0].copy()


def _partition(rng, side, count, style):
    """Original connected-region construction, unrelated to parent generators."""
    labels = np.full((side, side), -1, dtype=np.int8)
    if style == 'cuts':
        boxes = [(0, 0, side, side)]
        while len(boxes) < count:
            options = [(i, axis, cut) for i, (r, c, h, w) in enumerate(boxes)
                       for axis, length, breadth in ((0, h, w), (1, w, h))
                       for cut in range(1, length) if min(cut, length-cut)*breadth >= 4]
            if not options:
                return None
            i, axis, cut = options[int(rng.integers(len(options)))]
            r, c, h, w = boxes.pop(i)
            boxes.extend([(r, c, cut, w), (r+cut, c, h-cut, w)] if axis == 0 else
                         [(r, c, h, cut), (r, c+cut, h, w-cut)])
        for i, (r, c, h, w) in enumerate(boxes):
            labels[r:r+h, c:c+w] = i
        # Uneven stepped interfaces, preserving each original region's link.
        for _ in range(side*3):
            r, c = map(int, rng.integers(side, size=2))
            dr, dc = EDGE[int(rng.integers(4))]
            rr, cc = r+dr, c+dc
            if not (0 <= rr < side and 0 <= cc < side):
                continue
            a, b = int(labels[r, c]), int(labels[rr, cc])
            if a != b and np.sum(labels == a) > 4:
                labels[r, c] = b
                if len(_groups(map(tuple, np.argwhere(labels == a)))) != 1:
                    labels[r, c] = a
        return labels
    if style == 'pockets':
        labels.fill(0)
        for i in range(1, count):
            candidates = []
            for h, w in ((1, 3), (2, 2), (2, 3), (3, 2)):
                for r in range(1, side-h):
                    for c in range(1, side-w):
                        if not np.any(labels[r-1:r+h+1, c-1:c+w+1]):
                            candidates.append((r, c, h, w))
            if not candidates:
                return None
            r, c, h, w = candidates[int(rng.integers(len(candidates)))]
            labels[r:r+h, c:c+w] = i
        return labels
    # Biased multi-source growth gives branches, unequal areas and arbitrary
    # corner ownership. Sorted frontiers make seeds reproducible across runs.
    seeds = list(map(int, rng.choice(side*side, count, replace=False)))
    weights = np.exp(rng.uniform(-.8, .8, count))
    frontiers = [set() for _ in range(count)]
    sizes = [1]*count
    adjacent = {}
    for r in range(side):
        for c in range(side):
            adjacent[r*side+c] = [rr*side+cc for dr, dc in EDGE
                                 if 0 <= (rr := r+dr) < side and 0 <= (cc := c+dc) < side]
    for i, p in enumerate(seeds):
        labels.flat[p] = i
    for i, p in enumerate(seeds):
        frontiers[i].update(q for q in adjacent[p] if labels.flat[q] < 0)
    for _ in range(side*side-count):
        eligible = [i for i in range(count) if frontiers[i]]
        i = min(eligible, key=lambda j: sizes[j]/weights[j]+float(rng.random())*.5)
        choices = sorted(frontiers[i])
        p = choices[int(rng.integers(len(choices)))]
        labels.flat[p] = i
        sizes[i] += 1
        for frontier in frontiers:
            frontier.discard(p)
        frontiers[i].update(q for q in adjacent[p] if labels.flat[q] < 0)
    return labels


def _symbol(rng, colors, symmetric=False):
    a, b = colors
    if symmetric:
        kind = int(rng.integers(3))
        if kind == 0:
            return {(0, 0): a, (0, 1): b, (0, 2): a}
        if kind == 1:
            return {(0, 0): a, (0, 1): b, (1, 0): b, (1, 1): a}
        return {(0, 1): a, (1, 0): a, (1, 1): b, (1, 2): a, (2, 1): a}
    points = {(0, 0)}
    target = int(rng.integers(2, 8))
    while len(points) < target:
        frontier = sorted({(r+dr, c+dc) for r, c in points for dr, dc in EDGE} - points)
        p = frontier[int(rng.integers(len(frontier)))]
        proposed = points | {p}
        if (max(r for r, c in proposed)-min(r for r, c in proposed) < 4
                and max(c for r, c in proposed)-min(c for r, c in proposed) < 4):
            points = proposed
    cells = _norm({p: int(rng.choice(colors)) for p in sorted(points)})
    if len(set(cells.values())) == 1:
        cells[max(cells)] = b if cells[min(cells)] == a else a
    return cells


def _label_edges(labels):
    edges = set()
    side = labels.shape[0]
    for r in range(side):
        for c in range(side):
            for dr, dc in ((0, 1), (1, 0)):
                if r+dr < side and c+dc < side:
                    a, b = int(labels[r, c]), int(labels[r+dr, c+dc])
                    if a != b:
                        edges.add(tuple(sorted((a, b))))
    return sorted(edges)


def _stitch_plan(rng, labels, count, mode):
    edges = _label_edges(labels)
    if mode == 'two_clusters':
        matchings = [(a, b) for i, a in enumerate(edges) for b in edges[i+1:]
                     if len(set(a+b)) == 4]
        if not matchings:
            return None
        return list(matchings[int(rng.integers(len(matchings)))])
    if mode == 'connected':
        linked, plan = {int(labels[0, 0])}, []
        while len(linked) < count:
            options = [e for e in edges if len(set(e) & linked) == 1]
            e = options[int(rng.integers(len(options)))]
            plan.append(e)
            linked.update(e)
    else:
        n = int(rng.integers(1, min(count, len(edges))+1))
        plan = [edges[int(i)] for i in rng.choice(len(edges), n, replace=False)]
    # A stitch may bind three panels at a junction. Coalesce adjacent planned
    # edges; the actual patch still has to straddle all three original regions.
    junctions = [(i, j, tuple(sorted(set(a+b)))) for i, a in enumerate(plan)
                 for j, b in enumerate(plan[i+1:], i+1) if len(set(a+b)) == 3]
    if junctions and rng.random() < .45:
        i, j, triple = junctions[int(rng.integers(len(junctions)))]
        plan = [e for k, e in enumerate(plan) if k not in (i, j)] + [triple]
    return plan


def _panels(rng, labels, colors, shared, marker, plan):
    side, count = labels.shape[0], len(colors)
    output = np.asarray(colors, dtype=np.int8)[labels].copy()
    output[0, 0] = marker
    masks = [labels == i for i in range(count)]
    used, identities = set(), set()
    for index, edge in enumerate(plan):
        successful = False
        for _ in range(25):
            glyph = _symbol(rng, shared, symmetric=index == 0)
            identity = _canonical(glyph)
            if identity in identities:
                continue
            locations = []
            for _ in range(4):
                h, w = _shape(glyph)
                for r0 in range(side-h+1):
                    for c0 in range(side-w+1):
                        points = {(r+r0, c+c0) for r, c in glyph}
                        if (0, 0) in points or any((r+dr, c+dc) in used for r, c in points for dr, dc in EIGHT+((0, 0),)):
                            continue
                        if {int(labels[p]) for p in points} == set(edge):
                            locations.append(({(r+r0, c+c0): v for (r, c), v in glyph.items()}, points))
                glyph = _turn(glyph)
            if not locations:
                continue
            patch, points = locations[int(rng.integers(len(locations)))]
            for p, v in patch.items():
                output[p] = v
                for i in edge:
                    masks[i][p] = True
            used.update(points)
            identities.add(identity)
            successful = True
            break
        if not successful:
            return None
    panels = [_norm({tuple(map(int, p)): int(output[tuple(p)]) for p in np.argwhere(mask)})
              for mask in masks]
    if any(sum(v == color for v in part.values()) < 2 for color, part in zip(colors, panels)):
        return None
    return output, panels


def _scatter(rng, parts, background, anchor_rotation, marker):
    oriented = []
    for part in parts:
        k = anchor_rotation if marker in part.values() else int(rng.integers(4))
        for _ in range(k):
            part = _turn(part)
        oriented.append(part)
    compact = bool(rng.random() < .75)
    for attempt in range(20):
        canvas = np.full((30, 30), background, dtype=np.int8)
        blocked = np.zeros((30, 30), dtype=bool)
        order = sorted(range(len(parts)), key=lambda i: (-len(oriented[i]), i)) if attempt == 0 else list(map(int, rng.permutation(len(parts))))
        maxr = maxc = 0
        okay = True
        for i in order:
            part = oriented[i]
            h, w = _shape(part)
            rr = np.array([p[0] for p in part])
            cc = np.array([p[1] for p in part])
            options = []
            for r in range(31-h):
                for c in range(31-w):
                    if np.any(blocked[rr+r, cc+c]):
                        continue
                    score = max(maxr, r+h)*max(maxc, c+w) + float(rng.uniform(0, 35)) if compact else float(rng.random())
                    options.append((score, r, c))
            if not options:
                okay = False
                break
            _, r0, c0 = min(options)
            for (r, c), v in part.items():
                r, c = r+r0, c+c0
                canvas[r, c] = v
                blocked[max(0, r-1):min(30, r+2), max(0, c-1):min(30, c+2)] = True
            maxr, maxc = max(maxr, r0+h), max(maxc, c0+w)
        if not okay:
            continue
        rr, cc = np.nonzero(canvas != background)
        crop = canvas[rr.min():rr.max()+1, cc.min():cc.max()+1]
        h, w = crop.shape
        H = h + int(rng.integers(min(6, 30-h)+1))
        W = w + int(rng.integers(min(6, 30-w)+1))
        area = len(rr)
        while H*W <= 2*area:
            if H < 30 and (H <= W or W == 30):
                H += 1
            elif W < 30:
                W += 1
            else:
                return None
        result = np.full((H, W), background, dtype=np.int8)
        r0, c0 = int(rng.integers(H-h+1)), int(rng.integers(W-w+1))
        result[r0:r0+h, c0:c0+w] = crop
        return result
    return None


def _make_pair(rng, side, count, mode, rotation, marker):
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != marker])]
    background, shared, colors = palette[0], palette[1:3], palette[3:3+count]
    for attempt in range(350):
        style = str(rng.choice(['growth', 'cuts', 'pockets'], p=[.5, .35, .15]))
        if mode == 'two_clusters' and style == 'pockets':
            style = 'growth'
        labels = _partition(rng, side, count, style)
        if labels is None:
            continue
        plan = _stitch_plan(rng, labels, count, mode)
        if plan is None:
            continue
        built = _panels(rng, labels, colors, shared, marker, plan)
        if built is None:
            continue
        output, panels = built
        inp = _scatter(rng, panels, background, rotation, marker)
        if inp is None:
            continue
        facts = _decode(inp)
        answers = _assemble(facts)
        if len(answers) == 1 and np.array_equal(answers[0][0], output):
            return {'input': inp, 'output': output}
    raise RuntimeError('bounded stitched-panel construction exhausted')


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    _require(num_train > num_test, 'num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if split == 'train':
                side = int(rng.integers(6, 9)) if index == 0 else int(rng.integers(5, 9))
                panels = 4 if index == 0 else 3 if index == 1 else int(rng.integers(2, 6))
                mode = 'two_clusters' if index == 0 else 'connected' if index == 1 else str(rng.choice(['connected', 'mixed']))
                # Separate shared patches need physical breathing room. This
                # is a sampling constraint, never an extra solving rule.
                if mode == 'connected':
                    side = max(side, panels+2)
                rotation = 1 if index == 0 else 3 if index == 1 else int(rng.choice([0, 1, 3]))
            else:
                side, panels = int(rng.integers(9, 12)), int(rng.integers(3, 7))
                mode, rotation = str(rng.choice(['connected', 'mixed'])), 2
            for _ in range(8):
                pair = _make_pair(rng, side, panels, mode, rotation, marker)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError('could not create distinct inputs')
    return result


def _pair_facts(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
    inp, out = _grid(pair['input']), _grid(pair['output'])
    facts = _decode(inp)
    answers = _assemble(facts)
    _require(len(answers) == 1, 'ambiguous or impossible assembly')
    predicted, placed = answers[0]
    _require(np.array_equal(out, predicted), 'output does not match rigid stitched square')
    _require(out.shape == (facts['side'], facts['side']) and out[0, 0] == facts['marker'], 'square anchor')
    _require(not np.any(out == facts['background']), 'square must have no gaps')
    # Private ink is conserved; shared ink loses exactly the redundant copies.
    for c in facts['carriers'] + [facts['marker']]:
        _require(np.sum(inp == c) == np.sum(out == c), 'private cells not conserved')
    for c in facts['shared']:
        expected = sum(sum(v == c for r, col, v in identity) for identity in facts['motifs'])
        _require(int(np.sum(out == c)) == expected < int(np.sum(inp == c)), 'stitch multiplicity')
    multiplicity = np.zeros(out.shape, dtype=int)
    for candidate in placed.values():
        for p, v in candidate[3]:
            _require(out.flat[p] == v, 'rigid panel/color conservation')
            multiplicity.flat[p] += 1
    _require(np.all(multiplicity > 0), 'incomplete coverage')
    _require(np.all((multiplicity > 1) == np.isin(out, list(facts['shared']))), 'overlap must be exactly the stitches')
    _require(int(np.sum(multiplicity-1)) == facts['raw_area'] - facts['area'] > 0, 'deduplicated area')
    facts['rotation_required'] = any(p[4] != 0 for p in placed.values())
    return facts


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            f = _pair_facts(pair)
            key = (f['grid'].shape, f['grid'].tobytes())
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            facts[split].append(f)
    train, test = facts['train'], facts['test']
    _require(len({f['marker'] for f in train+test}) == 1, 'one puzzle-wide marker color')
    _require(any(len(f['clusters']) > 1 for f in train), 'demonstrate square placement between stitch clusters')
    _require(any(f['symmetric'] for f in train), 'demonstrate symmetric registration keys')
    _require(any(f['rotation_required'] for f in train), 'demonstrate rigid rotation')
    _require(max(f['side'] for f in train) < min(f['side'] for f in test), 'generated size shift')
    return True


def _digest(puzzle):
    payload = {s: [{k: np.asarray(v).tolist() for k, v in pair.items()} for pair in puzzle[s]] for s in ('train', 'test')}
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def _without_square_witness(facts, placed):
    """A legal stitch registration that becomes wrong when B is omitted.

    Move an entire non-anchor stitch cluster beyond the right edge. All
    quarter-turn poses and all shared-patch agreements remain exactly intact.
    The result exposes why registration alone cannot determine the answer.
    """
    anchor = next(i for i, part in enumerate(facts['parts']) if facts['marker'] in part.values())
    moving = next(cluster for cluster in facts['clusters'] if anchor not in cluster)
    cells, bindings = {}, {}
    side = facts['side']
    for i, candidate in placed.items():
        dc = side+1 if i in moving else 0
        for p, color in candidate[3]:
            r, c = divmod(p, side)
            point = r, c+dc
            assert point not in cells or cells[point] == color
            cells[point] = color
        for identity, key in candidate[1].items():
            moved = tuple((p//side, p % side+dc, v) for p, v in key)
            assert identity not in bindings or bindings[identity] == moved
            bindings[identity] = moved
    result = np.full(_shape(cells), facts['background'], dtype=np.int8)
    for point, color in cells.items():
        result[point] = color
    assert result[0, 0] == facts['marker'] and result.shape[1] > side
    return result


def self_test():
    """Bounded 42-seed audit, semantic mutations and geometric invariance checks."""
    signatures, shapes, structures, panel_counts, sizes, backgrounds = set(), set(), set(), set(), set(), set()
    motif_shapes, cluster_counts, duplicate_counts, multiplicities = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _decode(pair['input'])
                shapes.add(pair['input'].shape)
                sizes.add(f['side'])
                backgrounds.add(f['background'])
                panel_counts.add(len(f['parts']))
                cluster_counts.add(len(f['clusters']))
                duplicate_counts.add(f['raw_area']-f['area'])
                multiplicities.update(len(indices) for indices in f['occurrences'].values())
                for identity in f['motifs']:
                    cells = {(r, c): v for r, c, v in identity}
                    motif_shapes.add((_shape(cells), len(cells)))
                roles = {}
                structures.add((pair['output'].shape, tuple(roles.setdefault(int(c), len(roles)) for c in pair['output'].flat)))
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(structures) >= 180
    assert panel_counts == {2, 3, 4, 5, 6} and sizes == set(range(5, 12))
    assert len(backgrounds) >= 8 and len(motif_shapes) >= 12
    assert len(cluster_counts) >= 3 and len(duplicate_counts) >= 10
    assert {2, 3} <= multiplicities
    p = generate_puzzle(0, 2, 1)
    assert _digest(p) == _digest(generate_puzzle(0, 2, 1))
    # Input scattering, whole-canvas rotation and a bijective color permutation
    # do not change registration or the upper-left convention.
    pair = p['train'][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair['input'], k)), pair['output'])
    mapping = np.array([7, 2, 8, 0, 6, 9, 4, 1, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
    f = _decode(pair['input'])
    assert f['raw_area'] > f['side']**2  # dropping A cannot conserve raw area
    assert len(f['clusters']) == 2  # dropping B leaves free cluster translations
    assert _pair_facts(pair)['rotation_required']
    placed = _assemble(f)[0][1]
    alternative = _without_square_witness(f, placed)
    assert not np.array_equal(alternative, pair['output'])
    assert int(np.sum(alternative != f['background'])) == f['area']
    # Reflecting output is a meaningful negative check, as is losing a stitch.
    missing_stitch = pair['output'].copy()
    point = tuple(np.argwhere(np.isin(missing_stitch, list(f['shared'])))[0])
    missing_stitch[point] = f['background']
    for changed in (np.fliplr(pair['output']).copy(), missing_stitch):
        try:
            _pair_facts({'input': pair['input'], 'output': changed})
        except ValueError:
            pass
        else:
            raise AssertionError('corrupted output accepted')
    # Semantic boundary cases outside default sampling: sides four/twelve and
    # an eight-cell complete stitch. These use the same parser and joint search.
    for side in (4, 12):
        output = np.full((side, side), 2, dtype=np.int8)
        output[side//2:] = 3
        output[0, 0] = 9
        patch = {(side//2-1+r, side//2-2+c): v
                 for r, row in enumerate(((5, 6, 6, 5), (6, 5, 5, 6)))
                 for c, v in enumerate(row)}
        for point, v in patch.items():
            output[point] = v
        panels = []
        for top in (True, False):
            points = {(r, c) for r in range(side) for c in range(side)
                      if (r < side//2) == top} | set(patch)
            panels.append(_norm({p: int(output[p]) for p in points}))
        inp = _scatter(np.random.default_rng(600+side), panels, 0, 1, 9)
        assert inp is not None and np.array_equal(solve(inp), output)
        _pair_facts({'input': inp, 'output': output})
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            _counts(a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'pairs': 210, 'input_shapes': len(shapes),
            'output_structures': len(structures), 'panel_counts': sorted(panel_counts),
            'square_sides': sorted(sizes), 'motif_shapes': len(motif_shapes),
            'cluster_counts': sorted(cluster_counts), 'backgrounds': len(backgrounds),
            'stitch_multiplicities': sorted(multiplicities)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
