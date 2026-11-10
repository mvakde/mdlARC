"""Rank rigid keys within plug-shape classes along a marked rail.

Only NumPy and the standard library are used. Construction paints labels from
its explicit geometric scene; solving reconstructs stations and keys from cells.
"""
from collections import Counter, defaultdict
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
RAIL, START = 8, 9


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _cells(mask):
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        point = min(remaining)
        remaining.remove(point)
        stack, part = [point], {point}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(frozenset(part))
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _rectangle(box):
    a, b, c, d = box
    return frozenset((r, x) for r in range(a, b + 1) for x in range(c, d + 1))


def _rectangular(cells):
    return bool(cells) and cells == _rectangle(_bbox(cells))


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _halo(cells):
    return set(cells) | {(r + dr, c + dc) for r, c in cells for dr, dc in DIR4}


def _raw(point, op):
    r, c = point
    # 0..3 have determinant +1; 4..7 have determinant -1.
    return ((r, c), (c, -r), (-r, -c), (-c, r),
            (r, -c), (-c, -r), (-r, c), (c, r))[op]


def _transform(regions, op):
    raw = [frozenset(_raw(p, op) for p in region) for region in regions]
    box = _bbox(frozenset().union(*raw))
    return tuple(_shift(region, -box[0], -box[2]) for region in raw)


def _signature(cells):
    a, _, c, _ = _bbox(cells)
    return tuple(sorted((r - a, x - c) for r, x in cells))


def _orbit(cells):
    return {_signature(_transform((cells,), op)[0]) for op in range(8)}


def _canonical(cells):
    return min(_orbit(cells))


def _array_key(grid):
    return grid.shape, grid.tobytes()


def _rail_order(inp):
    marker = _cells(inp == START)
    track = _cells((inp == RAIL) | (inp == START))
    if len(marker) != 1 or len(track) < 12:
        raise ValueError("one marked endpoint and a rail of at least 12 cells are required")
    neighbors = {p: sorted(q for q in _halo({p}) - {p} if q in track) for p in track}
    endpoints = [p for p in track if len(neighbors[p]) == 1]
    if len(endpoints) != 2 or not marker <= set(endpoints):
        raise ValueError("the marker must be an endpoint of a simple rail")
    if any(len(v) not in (1, 2) for v in neighbors.values()):
        raise ValueError("the rail must not branch or touch itself")
    previous, current, order = None, next(iter(marker)), []
    while current is not None:
        order.append(current)
        choices = [q for q in neighbors[current] if q != previous]
        previous, current = current, choices[0] if choices else None
        if len(order) > len(track):
            raise ValueError("cyclic rail")
    if len(order) != len(track):
        raise ValueError("disconnected rail")
    return order


def _handle_depth(handle, box):
    r0, r1, c0, c1 = box
    a, b, c, d = _bbox(handle)
    if c > c1 and r0 <= a <= b <= r1:
        return d - c1, b - a + 1
    if d < c0 and r0 <= a <= b <= r1:
        return c0 - c, b - a + 1
    if a > r1 and c0 <= c <= d <= c1:
        return b - r1, d - c + 1
    if b < r0 and c0 <= c <= d <= c1:
        return r0 - a, d - c + 1
    raise ValueError("handle must extend normally from one rectangle side")


def _inspect(inp):
    hist = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(hist))
    if bg in (RAIL, START) or np.count_nonzero(hist == hist[bg]) != 1:
        raise ValueError("background must be the unique most frequent unreserved color")
    path = _rail_order(inp)
    track = frozenset(path)
    index = {p: i for i, p in enumerate(path)}
    objects = _components(_cells((inp != bg) & (inp != RAIL) & (inp != START)))
    mono, keys = [], []
    for obj in objects:
        colors = sorted({int(inp[p]) for p in obj})
        if len(colors) == 1:
            mono.append({"cells": obj, "color": colors[0]})
        elif len(colors) == 2:
            regions = [frozenset(p for p in obj if inp[p] == color) for color in colors]
            if any(len(_components(region)) != 1 for region in regions):
                raise ValueError("key regions must be connected")
            rect = [_rectangular(region) for region in regions]
            if sum(rect) != 1:
                raise ValueError("one key region must be a nonrectangular plug, the other a rectangle")
            pi = 1 if rect[0] else 0
            if len(_orbit(regions[pi])) != 8:
                raise ValueError("plugs must have eight distinct rigid orientations")
            keys.append({"cells": obj, "regions": tuple(regions), "colors": tuple(colors),
                         "plug": pi, "class": _canonical(regions[pi])})
        else:
            raise ValueError("unrecognized foreground component")
    if not 2 <= len(keys) <= 5 or len(mono) != 2 * len(keys):
        raise ValueError("need 2..5 keys and two fragments per socket")
    if len({m['color'] for m in mono}) != 1:
        raise ValueError("all fixed fragments in a scene share a color")
    tc = mono[0]['color']
    if any(tc in k['colors'] or _halo(k['cells']) & track for k in keys):
        raise ValueError("keys must be separate from the rail and use local non-target colors")

    shapes = {key['class'] for key in keys}
    targets = []
    for i, left in enumerate(mono):
        for j in range(i + 1, len(mono)):
            union = left['cells'] | mono[j]['cells']
            box = _bbox(union)
            rectangle = _rectangle(box)
            if frozenset(p for p in rectangle if inp[p] != bg) != union:
                continue
            slot = rectangle - union
            if not slot or len(_components(slot)) != 1 or _rectangular(slot):
                continue
            cls = _canonical(slot)
            if cls not in shapes or len(_orbit(slot)) != 8:
                continue
            contacts = sorted(index[p] for p in _halo(union) & track)
            if not contacts or contacts != list(range(contacts[0], contacts[-1] + 1)):
                continue
            targets.append({"parts": (i, j), "box": box, "rectangle": rectangle,
                            "slot": slot, "class": cls, "order": contacts[0],
                            "contacts": tuple(contacts), "cells": union})
    if not targets:
        raise ValueError("no readable sockets")
    moves = {}
    for ki, key in enumerate(keys):
        pi = key['plug']
        for ti, target in enumerate(targets):
            if key['class'] != target['class']:
                continue
            fits = []
            for op in range(8):
                regions = _transform(key['regions'], op)
                pr0, _, pc0, _ = _bbox(regions[pi])
                sr0, _, sc0, _ = _bbox(target['slot'])
                absolute = tuple(_shift(region, sr0 - pr0, sc0 - pc0) for region in regions)
                if absolute[pi] != target['slot']:
                    continue
                handle = absolute[1 - pi]
                if (handle & target['rectangle'] or any(not (0 <= r < inp.shape[0]
                        and 0 <= c < inp.shape[1]) for r, c in handle)):
                    continue
                if any(inp[p] != bg for p in handle):
                    continue
                try:
                    depth, width = _handle_depth(handle, target['box'])
                except ValueError:
                    continue
                colors = key['colors'][::-1] if op >= 4 else key['colors']
                paint = {p: color for region, color in zip(absolute, colors) for p in region}
                fits.append({"key": ki, "target": ti, "op": op, "depth": depth,
                             "width": width, "regions": absolute, "paint": paint,
                             "full": frozenset(paint)})
            if len(fits) > 1:
                raise ValueError("ambiguous rigid fit")
            if fits:
                moves[ki, ti] = fits[0]
    return {"background": bg, "target_color": tc, "path": path, "mono": mono,
            "keys": keys, "targets": targets, "moves": moves}


def _plan(inp):
    """Find a socket partition, then solve rank allocation within shape classes."""
    a = _inspect(inp)
    targets, keys, fits = a['targets'], a['keys'], a['moves']
    by_part = defaultdict(list)
    for ti, target in enumerate(targets):
        if any((ki, ti) in fits for ki in range(len(keys))):
            for part in target['parts']:
                by_part[part].append(ti)
    outputs = {}
    key_groups = defaultdict(list)
    for ki, key in enumerate(keys):
        key_groups[key['class']].append(ki)

    def finish(selected):
        target_groups = defaultdict(list)
        for ti in selected:
            target_groups[targets[ti]['class']].append(ti)
        if {c: len(v) for c, v in target_groups.items()} != {c: len(v) for c, v in key_groups.items()}:
            return
        if len({targets[ti]['order'] for ti in selected}) != len(selected):
            return
        contact_visits = [i for ti in selected for i in targets[ti]['contacts']]
        if len(set(contact_visits)) != len(contact_visits):
            return
        chosen = []
        for cls, kis in key_groups.items():
            tis = sorted(target_groups[cls], key=lambda ti: targets[ti]['order'])
            if any((ki, ti) not in fits for ki in kis for ti in tis):
                return  # Every competitor must fit every socket in its class.
            depths = {ki: fits[ki, tis[0]]['depth'] for ki in kis}
            if len(set(depths.values())) != len(kis):
                return  # No answer-bearing stable-order/color tie convention.
            if any(fits[ki, ti]['depth'] != depths[ki] for ki in kis for ti in tis):
                return
            for ki, ti in zip(sorted(kis, key=lambda ki: depths[ki]), tis):
                chosen.append(fits[ki, ti])
        out = np.full_like(inp, a['background'])
        for p in a['path']:
            out[p] = inp[p]
        for m in a['mono']:
            for p in m['cells']:
                out[p] = m['color']
        for move in chosen:
            for p, color in move['paint'].items():
                if out[p] != a['background']:
                    return
                out[p] = color
        outputs[_array_key(out)] = (out, chosen, selected)

    def search(unused, selected, occupied):
        if len(outputs) > 1:
            return
        if not unused:
            finish(selected)
            return
        first = min(unused)
        for ti in by_part[first]:
            t = targets[ti]
            parts = set(t['parts'])
            if parts <= unused and not t['rectangle'] & occupied:
                search(unused - parts, selected + [ti], occupied | t['rectangle'])

    search(set(range(len(a['mono']))), [], frozenset())
    if len(outputs) != 1:
        raise ValueError("input must have exactly one complete ranked docking answer")
    out, chosen, selected = next(iter(outputs.values()))
    a['chosen'], a['selected'] = chosen, selected
    return out, a


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _plan(_grid(input_grid))[0]


def _make_shape(rng, forbidden):
    for _ in range(300):
        h, w = int(rng.integers(4, 7)), int(rng.integers(3, 6))
        y = int(rng.integers(1, h - 1))
        slot = {(y, 0)}
        for c in range(1, w):
            new_y = int(np.clip(y + rng.integers(-1, 2), 1, h - 2))
            slot.update((r, c) for r in range(min(y, new_y), max(y, new_y) + 1))
            y = new_y
        slot = frozenset(slot)
        target = _rectangle((0, h - 1, 0, w - 1)) - slot
        if (len(_components(target)) == 2 and len(_orbit(slot)) == 8
                and _canonical(slot) not in forbidden):
            return {'h': h, 'w': w, 'slot': slot, 'target': target, 'tip': y}
    raise RuntimeError("could not draw an asymmetric separator")


def _polyline(vertices):
    path = [vertices[0]]
    for end in vertices[1:]:
        r, c = path[-1]
        dr, dc = int(np.sign(end[0] - r)), int(np.sign(end[1] - c))
        if dr and dc:
            raise ValueError("non-axis-aligned rail segment")
        while (r, c) != end:
            r, c = r + dr, c + dc
            path.append((r, c))
    return path


def _random_path(rng, h, w, mode=None):
    kind = int(rng.integers(5)) if mode is None else mode
    top, bottom = int(rng.integers(1, 5)), h - int(rng.integers(2, 6))
    left, right = int(rng.integers(1, 5)), w - int(rng.integers(2, 6))
    midr, midc = (top + bottom) // 2, (left + right) // 2
    vertices = [
        [(midr, left), (midr, right)],
        [(top, left), (bottom, left), (bottom, right)],
        [(top, left), (top, right), (bottom, right), (bottom, left)],
        [(top, left), (top, midc), (midr, midc), (midr, right), (bottom, right)],
        [(top, left), (top, right), (bottom, right), (bottom, left), (midr, left), (midr, midc)],
    ][kind]
    path = _polyline(vertices)
    # Reflecting/transposing a path changes its spatial order, not its marked order.
    if rng.random() < .5:
        path = [(h - 1 - r, c) for r, c in path]
    if rng.random() < .5:
        path = [(r, w - 1 - c) for r, c in path]
    if rng.random() < .5:
        path.reverse()
    if rng.random() < .5:
        path = [(c, r) for r, c in path]
        h, w = w, h
    return path, h, w


def _place_targets(rng, shapes, groups, handles, path, h, w):
    track, index = frozenset(path), {p: i for i, p in enumerate(path)}
    blocked, stations, used_contacts = set(), [], set()
    classes = list(groups)
    rng.shuffle(classes)
    for cls in classes:
        shape = shapes[cls]
        envelope = _rectangle((0, shape['h'] - 1, 0, shape['w'] - 1))
        for handle in handles[cls]:
            envelope |= handle['bar']
        possibilities = []
        ops = list(range(8))
        rng.shuffle(ops)
        for op in ops:
            target, slot, env = _transform((shape['target'], shape['slot'], envelope), op)
            _, r1, _, c1 = _bbox(env)
            # Contact forces an origin near the rail; inspect the full feasible domain.
            for r in range(h - r1):
                for c in range(w - c1):
                    placed_env = _shift(env, r, c)
                    if placed_env & (track | blocked):
                        continue
                    placed_target = _shift(target, r, c)
                    contacts = sorted(index[p] for p in _halo(placed_target) & track)
                    if (contacts and not set(contacts) & used_contacts
                            and contacts == list(range(contacts[0], contacts[-1] + 1))):
                        possibilities.append((target, slot, env, r, c, op, contacts))
        if not possibilities:
            raise RuntimeError("no station packing")
        # Favor varied orientations, rather than selecting the first screen position.
        target, slot, env, r, c, op, contacts = possibilities[int(rng.integers(len(possibilities)))]
        placed_env = _shift(env, r, c)
        blocked |= _halo(placed_env)
        used_contacts.update(contacts)
        stations.append({'class': cls, 'target': _shift(target, r, c),
                         'slot': _shift(slot, r, c), 'env': placed_env, 'origin': (r, c),
                         'base_env': envelope, 'op': op, 'order': contacts[0]})
    return sorted(stations, key=lambda s: s['order']), blocked


def _pack_keys(rng, keys, blocked, h, w):
    occupied = set(blocked)
    order = sorted(range(len(keys)), key=lambda i: -len(keys[i]['regions'][0] | keys[i]['regions'][1]))
    for i in order:
        key = keys[i]
        cells = key['regions'][0] | key['regions'][1]
        _, r1, _, c1 = _bbox(cells)
        positions = [(r, c) for r in range(h - r1) for c in range(w - c1)]
        rng.shuffle(positions)
        for r, c in positions:
            placed = _shift(cells, r, c)
            if not placed & occupied:
                key['absolute'] = tuple(_shift(region, r, c) for region in key['regions'])
                occupied |= _halo(placed)
                break
        else:
            raise RuntimeError("no loose-key packing")


def _build_scene(rng, palette, witness=0, test=False):
    if witness == 1:
        group_sizes = [3]
    elif witness == 2:
        group_sizes = [2, 2]
    elif test:
        group_sizes = [3, 1] if rng.random() < .4 else [3, 2] if rng.random() < .5 else [4, 1]
    else:
        group_sizes = list(([2], [3], [2, 1], [2, 2], [3, 1])[int(rng.integers(5))])
    n = sum(group_sizes)
    shapes, forbidden, handles = [], set(), []
    for cls, size in enumerate(group_sizes):
        shape = _make_shape(rng, forbidden)
        forbidden.add(_canonical(shape['slot']))
        shapes.append(shape)
        if witness == 1:
            depths, widths = [2, 3, 5], [3, 2, 1]
        else:
            depths = sorted(int(x) for x in rng.choice(np.arange(1, 9 if test else 7), size, replace=False))
            if test and cls == 0 and max(depths) < 7:
                depths[-1] = int(rng.integers(7, 9))
            widths = [int(rng.integers(1, 4)) for _ in depths]
        local = []
        for depth, width in zip(depths, widths):
            lo = max(0, shape['tip'] - width + 1)
            hi = min(shape['tip'], shape['h'] - width)
            start = int(rng.integers(lo, hi + 1))
            bar = _rectangle((start, start + width - 1, shape['w'], shape['w'] + depth - 1))
            local.append({'depth': depth, 'width': width, 'bar': bar})
        handles.append(local)
    bg, tc = palette[:2]
    key_pool = palette[2:]
    h, w = [int(x) for x in rng.integers(26 if n >= 4 else 22, 31, 2)]
    mode = (int(rng.integers(2)) if witness == 1 else 2 if witness == 2 else
            int(rng.integers(2, 5)) if test else None)
    path, h, w = _random_path(rng, h, w, mode)
    groups = [cls for cls, size in enumerate(group_sizes) for _ in range(size)]
    stations, blocked = _place_targets(rng, shapes, groups, handles, path, h, w)
    if witness == 2:
        sequence = [s['class'] for s in stations]
        if sum(x != y for x, y in zip(sequence, sequence[1:])) < 2:
            raise RuntimeError("need interleaved shape classes")
        screen = sorted(stations, key=lambda s: min(s['target']))
        if all([s['order'] for s in stations if s['class'] == cls]
               == [s['order'] for s in screen if s['class'] == cls]
               for cls in range(len(group_sizes))):
            raise RuntimeError("need rail order to disagree with screen order")
    blocked |= _halo(path)
    keys, assigned = [], Counter()
    for si, station in enumerate(stations):
        cls = station['class']
        handle = handles[cls][assigned[cls]]
        assigned[cls] += 1
        # The first two training scenes show both color behaviors in each scene.
        parity = si % 2 if witness else int(rng.integers(2))
        source_op = int(rng.integers(4)) + 4 * (int(station['op'] >= 4) ^ parity)
        colors = tuple(int(x) for x in rng.choice(key_pool, 2, replace=False))
        regions = _transform((shapes[cls]['slot'], handle['bar']), source_op)
        keys.append({'regions': regions, 'colors': colors, 'station': station,
                     'class': cls, 'depth': handle['depth'], 'width': handle['width'],
                     'bar': handle['bar'], 'parity': parity})
    if witness == 2 and [key['depth'] for key in keys] == sorted(key['depth'] for key in keys):
        raise RuntimeError("need class-local sorting to disagree with global sorting")
    _pack_keys(rng, keys, blocked, h, w)
    if witness == 1:
        spatial = sorted(keys, key=lambda k: min(k['absolute'][0] | k['absolute'][1]))
        depths = [k['depth'] for k in spatial]
        if depths == sorted(depths) or depths[::-1] == sorted(depths):
            raise RuntimeError("need a non-reversal sorting witness")
    inp = np.full((h, w), bg, dtype=np.int8)
    out = inp.copy()
    for i, p in enumerate(path):
        inp[p] = out[p] = START if i == 0 else RAIL
    for station in stations:
        for p in station['target']:
            inp[p] = out[p] = tc
    for key in keys:
        for region, color in zip(key['absolute'], key['colors']):
            for p in region:
                inp[p] = color
        station = key['station']
        shape = shapes[key['class']]
        _, slot, bar, _ = _transform((shape['target'], shape['slot'], key['bar'], station['base_env']), station['op'])
        colors = key['colors'][::-1] if key['parity'] else key['colors']
        for region, color in zip((slot, bar), colors):
            for p in _shift(region, *station['origin']):
                out[p] = color
    # Preserve the complete input/output footprint; optional padding is uninformative.
    used = _cells((inp != bg) | (out != bg))
    a, b, c, d = _bbox(used)
    margins = [int(x) for x in rng.integers(0, 3, 4)]
    a, b = max(0, a - margins[0]), min(h - 1, b + margins[1])
    c, d = max(0, c - margins[2]), min(w - 1, d + margins[3])
    inp, out = inp[a:b + 1, c:d + 1].copy(), out[a:b + 1, c:d + 1].copy()
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .3)
    shared_palette = tuple(int(x) for x in rng.permutation(8))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            witness = i + 1 if split == 'train' and i < 2 else 0
            for _ in range(180):
                palette = shared_palette if stable else tuple(int(x) for x in rng.permutation(8))
                try:
                    pair = _build_scene(rng, palette, witness, test=split == 'test')
                    if _array_key(pair['input']) in seen:
                        continue
                    solved, _ = _plan(pair['input'])
                    if not np.array_equal(solved, pair['output']):
                        continue
                except (RuntimeError, ValueError):
                    continue
                puzzle[split].append(pair)
                seen.add(_array_key(pair['input']))
                break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, analysis=None):
    a = analysis if analysis is not None else _plan(inp)[1]
    selected = sorted(a['selected'], key=lambda ti: a['targets'][ti]['order'])
    classes = [a['targets'][ti]['class'] for ti in selected]
    groups = defaultdict(list)
    for move in a['chosen']:
        groups[a['keys'][move['key']]['class']].append(move)
    area_bad = any([m['key'] for m in sorted(v, key=lambda m: (m['depth'], m['key']))]
                   != [m['key'] for m in sorted(v, key=lambda m: (m['depth'] * m['width'], m['key']))]
                   for v in groups.values())
    spatial_keys = sorted(range(len(a['keys'])), key=lambda ki: min(a['keys'][ki]['cells']))
    sequence = {m['key']: m['depth'] for m in a['chosen']}
    non_reversal = any((ds := [sequence[ki] for ki in spatial_keys if a['keys'][ki]['class'] == cls])
                       != sorted(ds) and ds[::-1] != sorted(ds) for cls in groups)
    station_screen = sorted(selected, key=lambda ti: min(a['targets'][ti]['cells']))
    screen_bad = any([ti for ti in selected if a['targets'][ti]['class'] == cls]
                     != [ti for ti in station_screen if a['targets'][ti]['class'] == cls] for cls in groups)
    path = a['path']
    directions = [(b[0] - x[0], b[1] - x[1]) for x, b in zip(path, path[1:])]
    depth_by_target = {m['target']: m['depth'] for m in a['chosen']}
    path_depths = [depth_by_target[ti] for ti in selected]
    return {'count': len(a['keys']), 'class_count': len(groups),
            'classes': tuple(classes), 'depths': tuple(sorted(m['depth'] for m in a['chosen'])),
            'widths': tuple(m['width'] for m in a['chosen']),
            'operations': tuple(m['op'] for m in a['chosen']), 'area_bad': area_bad,
            'non_reversal': non_reversal, 'screen_bad': screen_bad,
            'global_bad': path_depths != sorted(path_depths),
            'turns': sum(x != y for x, y in zip(directions, directions[1:])),
            'background': a['background'], 'target_color': a['target_color'],
            'shape': inp.shape, 'plug_shapes': tuple(groups)}


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("puzzle must have exactly train/test keys")
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise ValueError("splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError("pairs must have exactly input/output keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            if inp.shape != out.shape or _array_key(inp) in seen:
                raise ValueError("dimensions must be preserved and inputs must be distinct")
            seen.add(_array_key(inp))
            expected, a = _plan(inp)
            if not np.array_equal(out, expected):
                raise ValueError("output disagrees with ranked rigid docking")
            # Relational checks: fixed pixels, erased sources, full rigid regions,
            # conserved depth multiset, and ascending ranks independently of labels.
            for p in a['path']:
                assert inp[p] == out[p]
            for obj in a['mono']:
                assert all(inp[p] == out[p] for p in obj['cells'])
            assert all(out[p] == a['background'] for key in a['keys'] for p in key['cells'])
            groups = defaultdict(list)
            for m in a['chosen']:
                key, target = a['keys'][m['key']], a['targets'][m['target']]
                pi = key['plug']
                assert m['regions'][pi] == target['slot']
                assert m['regions'][1 - pi].isdisjoint(target['rectangle'])
                assert all(len(src) == len(dst) for src, dst in zip(key['regions'], m['regions']))
                assert all(out[p] == color for p, color in m['paint'].items())
                groups[key['class']].append((target['order'], m['depth']))
            assert any(len(v) > 1 for v in groups.values())
            for values in groups.values():
                depths = [depth for _, depth in sorted(values)]
                assert depths == sorted(set(depths))
            features[split].append(_features(inp, a))
    train = features['train']
    assert any(f['area_bad'] for f in train), "demonstrate depth rather than area"
    assert any(f['non_reversal'] for f in train), "demonstrate genuine sorting"
    assert any(f['class_count'] > 1 for f in train), "demonstrate shape-scoped ranks"
    assert any(f['global_bad'] for f in train), "rule out global length sorting"
    assert any(f['screen_bad'] for f in train), "rule out screen-order docking"
    assert {int(op >= 4) for f in train for op in f['operations']} == {0, 1}
    assert any(f['turns'] > 0 for f in train), "demonstrate bent-rail traversal"


def self_test():
    """A bounded 42-seed geometry sweep, counterfactuals, and API checks."""
    signatures, shapes, plugs, operations, depths, widths, counts, turns = set(), set(), set(), set(), set(), set(), set(), set()
    backgrounds, targets, classes = set(), set(), set()
    screen_bad = area_bad = non_reversal = False
    mutation_checked = False
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = tuple(_array_key(pair['input']) for s in ('train', 'test') for pair in p[s])
        signatures.add(signature)
        for s in ('train', 'test'):
            for pair in p[s]:
                inp, out = pair['input'], pair['output']
                predicted, a = _plan(inp.copy())
                assert np.array_equal(predicted, out)
                f = _features(inp, a)
                shapes.add(f['shape']); plugs.update(f['plug_shapes']); operations.update(f['operations'])
                depths.update(f['depths']); widths.update(f['widths']); counts.add(f['count']); turns.add(f['turns'])
                backgrounds.add(f['background']); targets.add(f['target_color']); classes.add(f['class_count'])
                screen_bad |= f['screen_bad']; area_bad |= f['area_bad']; non_reversal |= f['non_reversal']
                # Omitting A: reverse the two smallest members of a repeated class.
                group = defaultdict(list)
                for m in a['chosen']:
                    group[a['keys'][m['key']]['class']].append(m)
                competing = next(v for v in group.values() if len(v) > 1)
                x, y = sorted(competing, key=lambda m: m['depth'])[:2]
                mx, my = a['moves'][y['key'], x['target']], a['moves'][x['key'], y['target']]
                assert mx['paint'] != x['paint'] or my['paint'] != y['paint']
                # Omitting B's parity: every reflected move has different colors.
                for m in a['chosen']:
                    if m['op'] >= 4:
                        key = a['keys'][m['key']]
                        assert all(out[next(iter(region))] != color
                                   for region, color in zip(m['regions'], key['colors']))
                if not mutation_checked:
                    damaged = {s0: [{'input': q['input'].copy(), 'output': q['output'].copy()}
                                   for q in p[s0]] for s0 in ('train', 'test')}
                    damaged['test'][0]['output'] = damaged['test'][0]['input'].copy()
                    try:
                        validate_generated_puzzle(damaged)
                    except ValueError:
                        mutation_checked = True
                    else:
                        raise AssertionError("accepted an unchanged loose-key scene")
    assert len(signatures) == 42 and len(shapes) >= 20 and len(plugs) >= 25
    assert operations == set(range(8)) and depths >= set(range(1, 9)) and widths == {1, 2, 3}
    assert counts == {2, 3, 4, 5} and classes == {1, 2}
    assert {0, 1, 2, 3, 4} <= turns and backgrounds == targets == set(range(8))
    assert screen_bad and area_bad and non_reversal and mutation_checked
    first = generate_puzzle(9001, 2, 1)
    again = generate_puzzle(9001, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(first[s], again[s]) for k in ('input', 'output'))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'grid_shapes': len(shapes),
            'plug_shapes': len(plugs), 'counts': sorted(counts), 'turns': sorted(turns),
            'operations': sorted(operations)}
