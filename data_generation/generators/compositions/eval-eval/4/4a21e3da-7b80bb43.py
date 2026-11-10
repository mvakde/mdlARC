"""Repair-gated corner dispatch: geometry-only cleanup, contact, and transport.

No parent imports, stored pairs, or generation state are used by solve().
"""
from __future__ import annotations

import numpy as np


_SIDES = ("top", "right", "bottom", "left")
_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer)
            or not all(1 <= d <= 30 for d in g.shape)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("a grid must be a 1..30 integer rectangle with colors 0..9")
    return g


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _ORTH:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        groups.append(cells)
    return groups


def _strong(mask):
    """All cells supported by a horizontal or vertical run of at least three."""
    supported = np.zeros_like(mask)
    for m, s in ((mask, supported), (mask.T, supported.T)):
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for k in range(3):
            s[:, k:k + triples.shape[1]] |= triples
    return supported


def _gaps(mask):
    result = []
    for m, direction in ((mask, (0, 1)), (mask.T, (1, 0))):
        bracket = (m[:, :-4] & m[:, 4:]
                   & ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1]))
        for y, x in np.argwhere(bracket):
            p = (int(y), int(x)) if direction[1] else (int(x), int(y))
            result.append((*p, *direction))
    return result


def _repairs(mask):
    """A bracket, continuation, and unique two-cell deviation are visible evidence."""
    supported = _strong(mask)
    repairs = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r - dr, c - dc) or _at(mask, r + 5*dr, c + 5*dc)):
            continue
        matches = []
        for endpoint in (0, 4):
            er, ec = r + endpoint*dr, c + endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for bend in (-1, 1):
                sr, sc = toward*dr + bend*dc, toward*dc + bend*dr
                spur = tuple((er + k*sr, ec + k*sc) for k in (1, 2))
                if (all(_at(mask, *p) and not supported[p] for p in spur)
                        and not _at(mask, er + 3*sr, ec + 3*sc)):
                    matches.append(spur)
        if len(matches) > 1:
            raise ValueError("multiple deviations claim the same gap")
        if matches:
            repairs.append(((r, c, dr, dc), matches[0]))
    used = set()
    for (r, c, dr, dc), spur in repairs:
        edits = {(r + k*dr, c + k*dc) for k in (1, 2, 3)} | set(spur)
        if edits & used:
            raise ValueError("overlapping repairs or reused deviation")
        used.update(edits)
    return repairs


def _cleanup(mask, repair=True):
    evidence = _repairs(mask)
    additions, anchors = np.zeros_like(mask), np.zeros_like(mask)
    for (r, c, dr, dc), _ in evidence:
        anchors[r, c] = anchors[r + 4*dr, c + 4*dc] = True
        if repair:
            for k in (1, 2, 3):
                additions[r + k*dr, c + k*dc] = True
    # Only unsupported diagonal cells disappear. Retained brackets and valid
    # straight runs are protected, including cells on the far side of a break.
    weak = mask & ~_strong(mask | additions) & ~anchors
    noise = np.zeros_like(mask)
    for a, b, sa, sb in (
        (weak[:-1, :-1], weak[1:, 1:], (slice(None, -1), slice(None, -1)),
         (slice(1, None), slice(1, None))),
        (weak[:-1, 1:], weak[1:, :-1], (slice(None, -1), slice(1, None)),
         (slice(1, None), slice(None, -1))),
    ):
        links = a & b
        noise[sa] |= links
        noise[sb] |= links
    return (mask & ~noise) | additions


def _read_input(value):
    grid = _grid(value)
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3:
        raise ValueError("inputs have exactly background, line, and marker colors")
    bg = int(colors[np.argmax(counts)])
    if np.count_nonzero(grid == bg) <= grid.size // 2:
        raise ValueError("background must occupy more than half the input")
    rim = np.zeros_like(grid, dtype=bool)
    rim[0] = rim[-1] = True
    rim[:, 0] = rim[:, -1] = True
    points = np.argwhere(rim & (grid != bg))
    if len(points) not in (1, 2):
        raise ValueError("one or two non-corner border contacts are required")
    mark = int(grid[tuple(points[0])])
    if (any(int(grid[tuple(p)]) != mark for p in points)
            or np.count_nonzero(grid == mark) != len(points)):
        raise ValueError("the marker color appears only at the contacts")
    line = next(int(c) for c in colors if int(c) not in (bg, mark))
    markers = {}
    for r, c in points.tolist():
        sides = [s for s, yes in (("top", r == 0), ("bottom", r == h-1),
                                  ("left", c == 0), ("right", c == w-1)) if yes]
        if len(sides) != 1 or sides[0] in markers:
            raise ValueError("contacts must be on different sides, away from corners")
        markers[sides[0]] = (r, c)
    if len(markers) == 2 and not (
        len(set(markers) & {"top", "bottom"}) == 1
        and len(set(markers) & {"left", "right"}) == 1
    ):
        raise ValueError("two contacts must be on adjacent sides")
    return grid, bg, line, mark, markers


def _ray(side, point, shape):
    h, w = shape
    r, c = point
    dr, dc = {"top": (1, 0), "right": (0, -1),
              "bottom": (-1, 0), "left": (0, 1)}[side]
    result = []
    while 0 <= r < h and 0 <= c < w:
        result.append((r, c))
        r, c = r + dr, c + dc
    return result


def _activate(clean, markers, all_aligned=False):
    contacts = set()
    for side, point in markers.items():
        hits = [p for p in _ray(side, point, clean.shape) if clean[p]]
        if not hits:
            raise ValueError("every marker must contact cleaned linework")
        contacts.update(hits if all_aligned else hits[:1])
    powered = np.zeros_like(clean)
    for group in _components(clean):
        if group & contacts:
            for p in group:
                powered[p] = True
    return powered


def _regions(powered, markers):
    ar = next((r for s, (r, c) in markers.items() if s in ("left", "right")), None)
    ac = next((c for s, (r, c) in markers.items() if s in ("top", "bottom")), None)
    fixed = np.zeros_like(powered)
    if ar is not None:
        fixed[ar] = powered[ar]
    if ac is not None:
        fixed[:, ac] |= powered[:, ac]
    groups = {}
    for r, c in np.argwhere(powered & ~fixed).tolist():
        v = ("top" if r < ar else "bottom") if ar is not None else next(iter(markers))
        u = ("left" if c < ac else "right") if ac is not None else next(iter(markers))
        groups.setdefault((v, u), set()).add((r, c))
    return fixed, groups


def _translated(cells, corner, shape):
    rr, cc = zip(*cells)
    dr = -min(rr) if corner[0] == "top" else shape[0] - 1 - max(rr)
    dc = -min(cc) if corner[1] == "left" else shape[1] - 1 - max(cc)
    return {(r + dr, c + dc) for r, c in cells}


def _dispatch(clean, markers, bg, line, mark, *, powered=None,
              nearest=False, separate=False, collisions=False):
    if powered is None:
        powered = _activate(clean, markers)
    fixed, groups = _regions(powered, markers)
    output = np.full(clean.shape, bg, dtype=np.int8)
    output[(clean & ~powered) | fixed] = line
    for side, point in markers.items():
        ray = _ray(side, point, clean.shape)
        hits = [i for i, p in enumerate(ray) if powered[p]]
        if not hits:
            raise ValueError("a guide needs a powered hit")
        stop = min(hits) if nearest else max(hits)
        for p in ray[:stop + 1]:
            if output[p] == bg:
                output[p] = mark
    for corner, cells in groups.items():
        if not (set(corner) & set(markers)):
            continue
        partitions = [cells]
        if separate:
            mask = np.zeros_like(clean)
            for p in cells:
                mask[p] = True
            partitions = _components(mask)
        for group in partitions:
            targets = _translated(group, corner, clean.shape)
            if not collisions and any(output[p] != bg for p in targets):
                raise ValueError("translated material collides with a retained cell")
            for p in targets:
                output[p] = line
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the drawing, select first-contact networks, and dispatch their cuts."""
    grid, bg, line, mark, markers = _read_input(input_grid)
    clean = _cleanup(grid == line)
    return _dispatch(clean, markers, bg, line, mark)


def _thick(mask):
    return bool((mask[:-1, :-1] & mask[1:, :-1]
                 & mask[:-1, 1:] & mask[1:, 1:]).any())


def _key(grid):
    g = np.asarray(grid)
    return g.shape, g.astype(np.int8).tobytes()


def _diagonal_chains(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    chains = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if (dr or dc) and q in remaining:
                        remaining.remove(q)
                        stack.append(q)
                        group.add(q)
        chains.append(sorted(group))
    return chains


def _valid_noise(mask):
    for cells in _diagonal_chains(mask):
        if len(cells) not in (2, 3):
            return False
        steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(cells, cells[1:])]
        if len(set(steps)) != 1 or steps[0][0] != 1 or abs(steps[0][1]) != 1:
            return False
    return True


def _mutant_diff(clean, markers, roles, expected, **kwargs):
    try:
        return not np.array_equal(_dispatch(clean, markers, *roles, **kwargs), expected)
    except ValueError:
        return True


def _features(inp, expected=None):
    grid, bg, line, mark, markers = _read_input(inp)
    raw = grid == line
    clean, unfilled = _cleanup(raw), _cleanup(raw, repair=False)
    active, unreached = _activate(clean, markers), _activate(unfilled, markers)
    output = _dispatch(clean, markers, bg, line, mark)
    if expected is not None:
        assert np.array_equal(output, expected), "the full expected output differs"
    fixed, groups = _regions(active, markers)
    retained = [cells for corner, cells in groups.items() if set(corner) & set(markers)]
    component_counts = []
    for group in retained:
        mask = np.zeros_like(clean)
        for p in group:
            mask[p] = True
        component_counts.append(len(_components(mask)))
    all_closed = clean.copy()
    for r, c, dr, dc in _gaps(clean):
        for k in (1, 2, 3):
            all_closed[r + k*dr, c + k*dc] = True
    all_active = _activate(clean, markers, all_aligned=True)
    _, inactive_regions = _regions(clean & ~active, markers)
    inactive_opposite = sum(len(cells) for corner, cells in inactive_regions.items()
                            if not set(corner) & set(markers))
    try:
        all_gap_output = _dispatch(all_closed, markers, bg, line, mark)
        all_gap_legal = True
        all_gap_diff = not np.array_equal(all_gap_output, output)
    except ValueError:
        all_gap_legal, all_gap_diff = False, True
    nodes = int(clean.sum())
    edges = int((clean[:-1] & clean[1:]).sum() + (clean[:, :-1] & clean[:, 1:]).sum())
    return {
        "shape": grid.shape, "roles": (bg, line, mark),
        "markers": tuple(sorted(markers)), "arity": len(markers),
        "repairs": len(_repairs(raw)), "remaining_gaps": len(_gaps(clean)),
        "components": len(_components(clean)), "cycles": edges - nodes + len(_components(clean)),
        "active_components": len(_components(active)),
        "active_cells": int(active.sum()), "inactive_cells": int((clean & ~active).sum()),
        "bridge_gain": int((active & unfilled & ~unreached).sum()),
        "occluded_cells": int((all_active & ~active).sum()),
        "inactive_opposite": inactive_opposite,
        "partition_components": tuple(component_counts),
        "discarded": sum(len(cells) for corner, cells in groups.items()
                         if not set(corner) & set(markers)),
        "region_count": len(groups),
        "valid_noise": _valid_noise(raw & ~clean),
        "nearest_diff": _mutant_diff(clean, markers, (bg, line, mark), output, nearest=True),
        "separate_diff": _mutant_diff(clean, markers, (bg, line, mark), output,
                                      separate=True, collisions=True),
        "all_gap_diff": all_gap_diff, "all_gap_legal": all_gap_legal,
        "all_contact_diff": _mutant_diff(clean, markers, (bg, line, mark), output, powered=all_active),
        "all_line_diff": _mutant_diff(clean, markers, (bg, line, mark), output, powered=clean),
        "no_B_diff": _mutant_diff(unfilled, markers, (bg, line, mark), output),
        "no_A_diff": not np.array_equal(np.where(clean, line, np.where(grid == mark, mark, bg)), output),
    }


def _coordinates(rng, size, count):
    # Irregularly spaced nodes ensure long thin corridors, not solid blocks.
    spare = size - 5 - 5*(count - 1)
    increments = rng.multinomial(spare, np.full(count + 1, 1/(count + 1)))
    values = [2 + int(increments[0])]
    for i in range(1, count):
        values.append(values[-1] + 5 + int(increments[i]))
    return values


def _network(rng, h, w, richness):
    nr = int(rng.integers(3, min(4, (h-5)//5 + 1) + 1))
    nc = int(rng.integers(3, min(4, (w-5)//5 + 1) + 1))
    rows, cols = _coordinates(rng, h, nr), _coordinates(rng, w, nc)
    root = int(rng.integers(nr)), int(rng.integers(nc))
    nodes, edges = {root}, []
    wanted = int(rng.integers(6, nr*nc + 1))
    while len(nodes) < wanted:
        frontier = [(a, (a[0] + dr, a[1] + dc)) for a in sorted(nodes) for dr, dc in _ORTH
                    if 0 <= a[0] + dr < nr and 0 <= a[1] + dc < nc
                    and (a[0] + dr, a[1] + dc) not in nodes]
        a, b = frontier[int(rng.integers(len(frontier)))]
        nodes.add(b)
        edges.append((a, b))
    possible = [(a, (a[0] + dr, a[1] + dc)) for a in sorted(nodes) for dr, dc in ((1, 0), (0, 1))
                if (a[0] + dr, a[1] + dc) in nodes
                and (a, (a[0] + dr, a[1] + dc)) not in edges
                and ((a[0] + dr, a[1] + dc), a) not in edges]
    rng.shuffle(possible)
    if richness and rng.random() < .5:
        edges.extend(possible[:int(rng.integers(1, min(3, len(possible)) + 1))] if possible else [])
    mask = np.zeros((h, w), bool)
    for a, b in edges:
        ra, ca, rb, cb = rows[a[0]], cols[a[1]], rows[b[0]], cols[b[1]]
        mask[min(ra, rb):max(ra, rb) + 1, min(ca, cb):max(ca, cb) + 1] = True
    # Short side stubs and secondary arms add asymmetric local geometry.
    for _ in range(int(rng.integers(richness + 2))):
        p = np.argwhere(mask)[int(rng.integers(mask.sum()))]
        r, c = map(int, p)
        dr, dc = _ORTH[int(rng.integers(4))]
        length = int(rng.integers(1, 5))
        path = [(r + k*dr, c + k*dc) for k in range(1, length + 1)]
        if not all(2 <= rr < h-2 and 2 <= cc < w-2 and not mask[rr, cc] for rr, cc in path):
            continue
        trial = mask.copy()
        for q in path:
            trial[q] = True
        if not _thick(trial) and np.array_equal(_cleanup(trial), trial):
            mask = trial
    return mask


def _windows(clean):
    windows = []
    for r, c in np.argwhere(clean).tolist():
        for dr, dc in ((0, 1), (1, 0)):
            if not all(_at(clean, r + k*dr, c + k*dc) for k in range(5)):
                continue
            if not (_at(clean, r-dr, c-dc) or _at(clean, r+5*dr, c+5*dc)):
                continue
            gap = tuple((r + k*dr, c + k*dc) for k in (1, 2, 3))
            if any(_at(clean, rr+dc, cc+dr) or _at(clean, rr-dc, cc-dr) for rr, cc in gap):
                continue
            windows.append(((r, c, dr, dc), gap))
    return windows


def _intentionally_cut(rng, original, count):
    clean = original.copy()
    for _ in range(count):
        options = _windows(clean)
        rng.shuffle(options)
        old_count = len(_components(clean))
        for _, gap in options:
            trial = clean.copy()
            for p in gap:
                trial[p] = False
            groups = _components(trial)
            if (len(groups) > old_count and min(map(len, groups)) >= 3
                    and np.array_equal(_cleanup(trial), trial)):
                clean = trial
                break
    return clean


def _damage(rng, clean, target):
    raw = clean.copy()
    options = _windows(clean)
    rng.shuffle(options)
    changed = set()
    count = 0
    for (r, c, dr, dc), gap in options:
        if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for p in gap for q in changed):
            continue
        deviations = []
        for endpoint in (0, 4):
            er, ec = r + endpoint*dr, c + endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for bend in (-1, 1):
                sr, sc = toward*dr + bend*dc, toward*dc + bend*dr
                spur = tuple((er + k*sr, ec + k*sc) for k in (1, 2))
                if all(1 <= rr < clean.shape[0]-1 and 1 <= cc < clean.shape[1]-1
                       and not clean[rr, cc] and not raw[rr, cc] for rr, cc in spur):
                    deviations.append(spur)
        rng.shuffle(deviations)
        for spur in deviations:
            trial = raw.copy()
            for p in gap:
                trial[p] = False
            for p in spur:
                trial[p] = True
            try:
                recoverable = (np.array_equal(_cleanup(trial), clean)
                               and len(_repairs(trial)) == count + 1)
            except ValueError:
                recoverable = False
            if recoverable:
                raw = trial
                changed.update(gap + spur)
                count += 1
                break
        if count >= target:
            break
    if count != target:
        return None
    # Detached diagonal decoys, with no fixed location or obligation to add one.
    for _ in range(int(rng.integers(4))):
        for _ in range(15):
            r, c = int(rng.integers(1, clean.shape[0]-1)), int(rng.integers(1, clean.shape[1]-1))
            dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
            path = [(r + k*dr, c + k*dc) for k in range(int(rng.integers(2, 4)))]
            if not all(1 <= rr < clean.shape[0]-1 and 1 <= cc < clean.shape[1]-1
                       and not clean[rr, cc] and not raw[rr, cc] for rr, cc in path):
                continue
            trial = raw.copy()
            for p in path:
                trial[p] = True
            try:
                good = np.array_equal(_cleanup(trial), clean) and len(_repairs(trial)) == target
            except ValueError:
                good = False
            if good:
                raw = trial
                break
    return raw


def _fallback_pair(dimensions, roles, arity, target, variant=0):
    """An open branched network with a powered bridge and an occluded broken tail.

    This is a procedural construction, not a saved example. The parameters give
    at least 48 distinct geometries even on the smallest 18 by 18 canvas.
    """
    h, w = dimensions
    top = 2 + variant % 2
    right = w - 4 + (variant // 2) % 2
    middle = top + 6
    axis_choices = right - 11
    axis = 8 + (variant // 4) % axis_choices
    bottom_choices = h - middle - 4
    bottom = middle + 3 + (variant // (4*axis_choices)) % bottom_choices
    clean = np.zeros((h, w), bool)
    clean[top, 3:right + 1] = True
    clean[top:bottom + 1, 3] = True
    clean[middle, 3:right + 1] = True
    clean[top:top + 5, right] = True
    clean[middle:h - 1, axis] = True
    clean[middle + 2:middle + 5, axis] = False
    # A stationary branch in the unselected quadrant distinguishes erasing the
    # powered quadrant from indiscriminately erasing that entire spatial area.
    clean[middle + 5, axis - 3:axis] = True
    raw = clean.copy()
    gaps = [((top, c) for c in range(4, 7)),
            ((r, 3) for r in range(top + 3, top + 6)),
            ((middle, c) for c in range(axis + 1, axis + 4))]
    spurs = [((top + 1, 6), (top + 2, 5)),
             ((top + 3, 2), (top + 4, 1)),
             ((middle + 1, axis + 3), (middle + 2, axis + 2))]
    for gap, spur in zip(gaps[:target], spurs[:target]):
        for p in gap:
            raw[p] = False
        for p in spur:
            raw[p] = True
    assert np.array_equal(_cleanup(raw), clean)
    bg, line, mark = roles
    grid = np.where(raw, line, bg).astype(np.int8)
    grid[0, axis] = mark
    if arity == 2:
        grid[top + 3, -1] = mark
    return {'input': grid, 'output': _dispatch(clean, _read_input(grid)[-1], *roles)}


def _acceptable(f, witness):
    if (f['bridge_gain'] < 3 or f['inactive_cells'] < 3 or f['active_cells'] < 10
            or not f['valid_noise'] or not all(f[k] for k in ('no_A_diff', 'no_B_diff', 'all_line_diff'))
            or f['region_count'] != (2 if f['arity'] == 1 else 4)
            or (f['arity'] == 2 and f['discarded'] < 3)):
        return False
    if witness == 'contact' and not (f['occluded_cells'] >= 3 and f['all_contact_diff']
            and f['nearest_diff'] and f['all_gap_diff'] and f['all_gap_legal']):
        return False
    return witness != 'collective' or (f['separate_diff'] and f['inactive_opposite'] >= 3)


def _component_labels(mask):
    """Label a fixed drawing once while trying different marker positions."""
    labels = np.zeros(mask.shape, dtype=np.int16)
    for index, group in enumerate(_components(mask), 1):
        for p in group:
            labels[p] = index
    return labels


def _candidate_context(raw, clean):
    # These arrays belong only to this construction attempt. Neither solve nor
    # public validation uses cached construction data.
    unfilled = _cleanup(raw, repair=False)
    labels, raw_labels = _component_labels(clean), _component_labels(unfilled)
    rays = []
    for mask, lab in ((clean, labels), (unfilled, raw_labels)):
        first = np.argmax(mask, axis=0)
        last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
        top_hits = lab[first, np.arange(mask.shape[1])]
        right_hits = lab[np.arange(mask.shape[0]), last]
        rays.append((top_hits, right_hits))
    return clean, unfilled, labels, raw_labels, rays, _valid_noise(raw & ~clean)


def _candidate_can_pass(context, ar, ac, witness):
    """Reject only candidates that necessarily fail existing feature thresholds.

    The constructor's unrotated markers are top and optionally right. Component
    labels give exactly the same first-contact activation as _activate without
    reflooding the unchanged drawing for each candidate. All expensive raster
    counterfactuals still run through _features for candidates that survive.
    """
    clean, unfilled, labels, raw_labels, rays, valid_noise = context
    if not valid_noise:
        return False
    powered_masks = []
    for lab, (top_hits, right_hits) in zip((labels, raw_labels), rays):
        ids = [int(top_hits[ac])]
        if ar is not None:
            ids.append(int(right_hits[ar]))
        if 0 in ids:
            return False  # _features would reject a ray with no cleaned hit.
        selected = np.zeros(int(lab.max()) + 1, dtype=bool)
        selected[ids] = True
        powered_masks.append(selected[lab])
    active, unreached = powered_masks
    inactive = clean & ~active
    if (active.sum() < 10 or inactive.sum() < 3
            or np.count_nonzero(active & unfilled & ~unreached) < 3):
        return False
    if ar is None:
        if not active[:, :ac].any() or not active[:, ac + 1:].any():
            return False
    else:
        regions = (active[:ar, :ac], active[:ar, ac + 1:],
                   active[ar + 1:, :ac], active[ar + 1:, ac + 1:])
        if not all(region.any() for region in regions) or regions[2].sum() < 3:
            return False
        if witness == 'collective' and inactive[ar + 1:, :ac].sum() < 3:
            return False
    if witness == 'contact':
        aligned_ids = np.unique(labels[:, ac])
        selected = np.zeros(int(labels.max()) + 1, dtype=bool)
        selected[aligned_ids] = True
        selected[0] = False
        if np.count_nonzero(selected[labels] & ~active) < 3:
            return False
    return True


def _make_pair(rng, *, arity, turn, roles, richness, target_repairs, witness, dimensions, seen):
    h, w = dimensions
    for _ in range(180):
        original = _network(rng, h, w, richness)
        clean = _intentionally_cut(rng, original, int(rng.integers(1, 3)))
        if len(_components(clean)) < 2:
            continue
        raw = _damage(rng, clean, target_repairs)
        if raw is None:
            continue
        rr, cc = np.argwhere(clean).T
        choices = [(None, c) for c in range(int(cc.min())+1, int(cc.max()))] if arity == 1 else [
            (r, c) for r in range(int(rr.min())+1, int(rr.max()))
            for c in range(int(cc.min())+1, int(cc.max()))]
        rng.shuffle(choices)
        context = _candidate_context(raw, clean)
        for ar, ac in choices[:70]:
            if not _candidate_can_pass(context, ar, ac, witness):
                continue
            bg, line, mark = roles
            grid = np.where(raw, line, bg).astype(np.int8)
            grid[0, ac] = mark
            if arity == 2:
                grid[ar, -1] = mark
            try:
                f = _features(grid)
                if not _acceptable(f, witness):
                    continue
                out = solve(grid)
                if _thick(out == line):
                    continue
            except (ValueError, AssertionError):
                continue
            pair = {"input": np.rot90(grid, -turn).copy(),
                    "output": np.rot90(out, -turn).copy()}
            if _key(pair['input']) not in seen:
                return pair
    # There are more procedural fallback variants than the maximum 14 inputs.
    # Enumerate them, so uniqueness does not depend on another random retry.
    for variant in range(256):
        pair = _fallback_pair(dimensions, roles, arity, target_repairs, variant)
        if not _acceptable(_features(pair['input'], pair['output']), witness):
            continue
        pair = {k: np.rot90(v, -turn).copy() for k, v in pair.items()}
        if _key(pair['input']) not in seen:
            return pair
    raise AssertionError("the constructive fallback must have a free geometry")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Sample visible conventions and a presentation shift, never a hidden rule."""
    n = _count(num_train, 'num_train', 2, 10)
    t = _count(num_test, 'num_test', 1, 4)
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    vary_palette = bool(rng.integers(2))
    base_turn = int(rng.integers(4))
    shift = int(rng.integers(3))  # held-out direction, repair count, topology/size
    richness = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for index in range(count):
            training = split == 'train'
            arity = (1 if index == 0 else 2) if training and index < 2 else int(rng.integers(1, 3))
            witness = ('contact' if index == 0 else 'collective') if training and index < 2 else ''
            turn = base_turn if training and shift == 0 else int(rng.integers(4))
            if not training and shift == 0:
                turn = (base_turn + int(rng.integers(1, 4))) % 4
            target = int(rng.integers(1, 3)) if not (not training and shift == 1) else 3
            if training and shift == 1 and index < 2:
                target = index + 1
            low, high = ((18, 23) if training else (24, 30)) if shift == 2 else (18, 30)
            dimensions = tuple(map(int, rng.integers(low, high + 1, size=2)))
            if rng.random() < .2:
                dimensions = (dimensions[0], dimensions[0])
            if vary_palette:
                roles = tuple(map(int, rng.choice(10, 3, replace=False)))
            pair = _make_pair(rng, arity=arity, turn=turn, roles=roles,
                              richness=richness if training or shift != 2 else 2,
                              target_repairs=target, witness=witness, dimensions=dimensions, seen=seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _features(inp, out)
            clean = _cleanup(inp == f['roles'][1])
            assert f['repairs'] >= 1 and f['bridge_gain'] >= 3
            assert f['inactive_cells'] >= 3 and f['active_cells'] >= 10
            assert f['region_count'] == (2 if f['arity'] == 1 else 4)
            assert f['arity'] == 1 or f['discarded'] >= 3
            assert f['no_A_diff'] and f['no_B_diff'] and f['all_line_diff']
            assert f['valid_noise'], 'erased fragments must be separate diagonal pairs or triples'
            assert not _thick(clean) and not _thick(out == f['roles'][1])
            assert set(np.unique(out)) == set(f['roles'])
            assert np.count_nonzero(out == f['roles'][1]) == int(clean.sum()) - f['discarded']
            assert np.count_nonzero(clean & ~(inp == f['roles'][1])) == 3*f['repairs']
            infos[split].append(f)
    train = infos['train']
    assert {f['arity'] for f in train} == {1, 2}
    assert any(f['occluded_cells'] >= 3 and f['all_contact_diff'] and f['nearest_diff']
               and f['all_gap_diff'] and f['all_gap_legal'] for f in train)
    assert any(f['separate_diff'] and f['inactive_opposite'] >= 3 for f in train)


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features, shapes, normalized = [], set(), set()
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair['input'], pair['output'])
                features.append(f)
                shapes.add(f['shape'])
                g = pair['input']
                bg, line, mark = f['roles']
                normalized.add((g.shape, np.where(g == bg, 0, np.where(g == line, 1, 2)).tobytes()))
                # The transformation is equivariant to the entire scene's D4 symmetry.
                for turns in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(g, turns)), np.rot90(pair['output'], turns))
                assert np.array_equal(solve(g[:, ::-1]), pair['output'][:, ::-1])
    assert len(normalized) >= 180 and len(shapes) >= 40
    assert {f['arity'] for f in features} == {1, 2}
    assert len({f['markers'] for f in features}) == 8
    assert len({f['roles'] for f in features}) >= 30
    assert len({f['components'] for f in features}) >= 2
    assert {1, 2, 3} <= {f['repairs'] for f in features}
    assert any(f['cycles'] for f in features)
    assert any(f['active_components'] == 2 for f in features)
    assert max(f['bridge_gain'] for f in features) >= 10
    assert max(max(f['partition_components']) for f in features) >= 3
    assert _puzzle_key(generate_puzzle(17)) == _puzzle_key(puzzles[17])
    for n, t in ((2, 1), (10, 4), (6, 3)):
        validate_generated_puzzle(generate_puzzle(123 + n*10 + t, n, t))
    for shape in ((18, 18), (18, 30), (30, 18), (30, 30)):
        for arity in (1, 2):
            for target in (1, 2, 3):
                keys = set()
                for variant in range(16):
                    pair = _fallback_pair(shape, (0, 7, 2), arity, target, variant)
                    f = _features(pair['input'], pair['output'])
                    assert _acceptable(f, 'contact' if arity == 1 else 'collective')
                    assert not _thick(pair['output'] == 7)
                    keys.add(_key(pair['input']))
                assert len(keys) >= 14
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
                 for s in ('train', 'test')}
    pair = corrupted['train'][0]
    f = _features(pair['input'])
    r, c = np.argwhere(pair['output'] == f['roles'][1])[0]
    pair['output'][r, c] = f['roles'][0]
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing output cell accepted")
    return {"seeds": 42, "pairs": len(features), "shapes": len(shapes),
            "normalized_inputs": len(normalized)}


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))
