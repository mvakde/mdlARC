"""Mirror molds: restore reflected material, dock, and cast leading cavities.

Standard library + NumPy only. No files are read by generation or solving.
Diagnostics are small aggregates; self_test never writes grids or traces.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


DAMAGE = 8


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            result.append(cells)
    return result


def _cavities(mask):
    h, w = mask.shape
    return [cells for cells in _components(~mask)
            if not any(r in (0, h-1) or c in (0, w-1) for r, c in cells)]


def _casting(mask, mode="leading"):
    """Return (row, distance beyond rail) for full reflected cavity shapes.

    mode='all' is an intentional counterfactual that ignores the leading wall.
    mode='open' also permits exterior background, to test enclosure shortcuts.
    """
    if not mask.any():
        return set()
    right = int(np.nonzero(mask)[1].max())
    regions = _components(~mask) if mode == "open" else _cavities(mask)
    points = set()
    for cells in regions:
        contact = any(c == right-1 and mask[r, right] for r, c in cells)
        if mode == "all" or contact:
            points.update((r, 1+right-c) for r, c in cells)
    return points


def _restore(tile, bg):
    """Complete each four-member reflection orbit using visible pixels only."""
    tile = tile.copy()
    h, w = tile.shape
    for r in range((h+1)//2):
        for c in range((w+1)//2):
            orbit = {(r, c), (h-1-r, c), (r, w-1-c), (h-1-r, w-1-c)}
            visible = {int(tile[y, x]) for y, x in orbit if tile[y, x] != DAMAGE}
            _require(len(visible) == 1, "erased or contradictory reflection orbit")
            value = visible.pop()
            for y, x in orbit:
                tile[y, x] = value
    colors = set(map(int, np.unique(tile))) - {bg}
    _require(len(colors) == 1, "a mold must have one material color")
    color = colors.pop()
    mask = tile == color
    _require(len(_components(mask)) == 1, "repaired mold must be connected")
    rr, cc = np.nonzero(mask)
    _require((rr.min(), rr.max(), cc.min(), cc.max()) == (0, h-1, 0, w-1),
             "mold reflection box must be tight")
    return tile, color


def _scene(a):
    candidates = []
    for turn in range(4):
        work = np.rot90(a, turn)
        h, w = work.shape
        for col in range(1, w-1):
            rail = int(work[0, col])
            if rail == DAMAGE or not np.all(work[:, col] == rail):
                continue
            if np.count_nonzero(work == rail) != h:
                continue
            bg = int(work[0, -1])
            if bg in (rail, DAMAGE) or not np.all(work[:, col+1:] == bg):
                continue
            if not np.any(work[:, :col] != bg):
                continue
            candidates.append((work, turn, bg, rail, col))
    _require(len(candidates) == 1, "unique rail and blank casting side required")
    return candidates[0]


def _analyze(a):
    canonical, turn, bg, rail, col = _scene(a)
    objects = []
    for cells in _components(canonical[:, :col] != bg):
        rr, cc = np.array(cells).T
        r0, r1, c0, c1 = int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
        tile = canonical[r0:r1, c0:c1].copy()
        visible_colors = set(map(int, np.unique(tile))) - {bg, DAMAGE}
        _require(len(visible_colors) == 1, "visible material must identify each mold")
        missing = np.argwhere(tile == DAMAGE)
        box = None
        if len(missing):
            low, high = missing.min(axis=0), missing.max(axis=0)+1
            y, x = map(int, low)
            s, t = map(int, high)
            _require((s-y)*(t-x) == len(missing), "one rectangular damage patch per mold")
            box = (y, s, x, t)
        restored, color = _restore(tile, bg)
        mask = restored == color
        holes = _cavities(mask)
        right = mask.shape[1]-1
        active = [hole for hole in holes if any(c == right-1 for r, c in hole)]
        partial = any({r for r, c in hole} != {r for r, c in hole if c == right-1}
                      for hole in active)
        orientation_needed = any(set(hole) != {(r, min(c for _, c in hole)+max(c for _, c in hole)-x)
                                                for r, x in hole} for hole in active)
        stamp = _casting(mask)
        _require(all(1 <= d < canonical.shape[1]-col for r, d in stamp),
                 "whole cast must fit beyond the rail")
        no_repair_bg = _casting(tile == color)
        no_repair_solid = _casting((tile == color) | (tile == DAMAGE))
        objects.append(dict(row=r0, end=r1, start=c0, tile=tile, restored=restored,
                            mask=mask, color=color, damage=box,
                            shift=col-c1, holes=holes, active=active, stamp=stamp,
                            partial=partial, orientation_needed=orientation_needed,
                            a_needed=stamp != no_repair_bg and stamp != no_repair_solid,
                            enclosure_needed=stamp != _casting(mask, "open"),
                            extreme_needed=stamp != _casting(mask, "all")))
    objects.sort(key=lambda o: o['row'])
    _require(bool(objects), "no molds")
    for prev, current in zip(objects, objects[1:]):
        _require(current['row'] > prev['end'], "mold row bands need a blank gap")
    return canonical, turn, bg, rail, col, objects


def _render(a, analysis=None):
    canonical, turn, bg, rail, col, objects = _analyze(a) if analysis is None else analysis
    out = np.full(canonical.shape, bg, dtype=np.int8)
    out[:, col] = rail
    for obj in objects:
        rr, cc = np.nonzero(obj['mask'])
        out[obj['row']+rr, col-obj['mask'].shape[1]+cc] = obj['color']
        for r, depth in obj['stamp']:
            out[obj['row']+r, col+depth] = obj['color']
    return np.rot90(out, -turn).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution; no seed, training examples, files, or hidden palette."""
    return _render(_grid(input_grid))


def _rand(rng, low, high):
    return int(rng.integers(low, high+1))


def _mold(rng, kind, max_h=9):
    """Compose symmetric silhouettes and rooms, then let flood fill classify."""
    minimum = 5 if kind in ('partial', 'tail', 'mixed', 'open') else 4 if kind == 'active' else 3
    h = _rand(rng, minimum, max(minimum, min(9, max_h)))
    w = _rand(rng, 7 if kind in ('partial', 'decoy') else 9 if kind == 'mixed' else 5, 11)
    mask = np.ones((h, w), dtype=bool)
    if kind == 'solid':
        if rng.random() < .65:
            # A cross or stepped outline, still connected and tightly bounded.
            for r in range(h//2):
                cut = _rand(rng, 0, (w-3)//2)
                mask[r, :cut] = mask[h-1-r, :cut] = False
                if cut:
                    mask[r, -cut:] = mask[h-1-r, -cut:] = False
            mask[h//2, :] = mask[h-1-h//2, :] = True
    elif kind == 'open':
        # Two chambers open through top/bottom doors, with a transverse bar
        # joining both extreme walls. Background behind those walls is exterior.
        middle = h//2
        mask[1:middle, 1:-1] = False
        mask[h-middle:h-1, 1:-1] = False
        mask[middle, :] = mask[h-1-middle, :] = True
        left, right = (w-1)//2, w//2+1
        mask[0, left:right] = mask[-1, left:right] = False
    elif kind == 'tail':
        extension = _rand(rng, 1, min(2, (w-3)//2))
        mask[:, :extension] = mask[:, -extension:] = False
        mask[0, :] = mask[-1, :] = True
        for r in range(1, h-1):
            depth = extension+1
            mask[r, depth:w-depth] = False
    elif kind == 'partial':
        # Mirrored but individually asymmetric chambers. Only the right chamber
        # touches the leading extreme; its partner supplies repair witnesses.
        # Both rooms remain connected through a common inner column. Some rows
        # recess from the extreme, resolving selection of the whole chamber.
        inner = w//2+1
        left = _rand(rng, inner, w-3)
        for r in range(1, (h+1)//2):
            end = w-1 if r == 1 else w-2 if r == 2 else _rand(rng, w-2, w-1)
            for y in {r, h-1-r}:
                mask[y, left:end] = False
                mask[y, w-end:w-left] = False
    elif kind == 'mixed':
        # Side rooms reach the extreme wall; central rooms are recessed.
        for r in range(1, (h+1)//2):
            if r == 1 or rng.random() < .7:
                flank = 2 if r == 1 else _rand(rng, 1, 2)
                central = rng.random() < .75
                for y in {r, h-1-r}:
                    mask[y, 1:1+flank] = mask[y, w-1-flank:w-1] = False
                    if central:
                        mask[y, (w-1)//2:w//2+1] = False
    else:
        # Independent separated room bands and variable-width connected rooms.
        room_rows = set()
        for r in range(1, (h+1)//2):
            if r == 1 or rng.random() < .68:
                room_rows.update((r, h-1-r))
        depths = {}
        for r in sorted(room_rows):
            folded = min(r, h-1-r)
            if folded not in depths:
                depths[folded] = _rand(rng, 2 if kind == 'decoy' else 1, (w-1)//2)
            depth = depths[folded]
            mask[r, depth:w-depth] = False
        if kind != 'decoy':
            mask[1, 1:w-1] = mask[h-2, 1:w-1] = False
    _require(np.array_equal(mask, mask[::-1]) and np.array_equal(mask, mask[:, ::-1]),
             "builder reflection failure")
    _require(len(_components(mask)) == 1, "builder disconnected material")
    return mask


def _damage(rng, mask, color, bg, severe, require_a):
    full = np.where(mask, color, bg).astype(np.int8)
    h, w = mask.shape
    stamp = _casting(mask)
    candidates = []
    # Every patch stays in one strict column half, preserving a horizontal mate.
    # Compact patches also preserve a row-reflected mate; severe ones cross it.
    for side in (0, 1):
        for width in range(1, w//2+1):
            for inset in range(w//2-width+1):
                c0 = inset if side == 0 else w-inset-width
                c1 = c0+width
                for r0 in range(h):
                    for r1 in range(r0+1, h+1):
                        crosses = 2*r0 < h-1 < 2*(r1-1)
                        if severe != crosses:
                            continue
                        if not severe and not (r1 <= h//2 or r0 >= (h+1)//2):
                            continue
                        patch = mask[r0:r1, c0:c1]
                        if not patch.any():
                            continue
                        if require_a and patch.all():
                            continue
                        candidates.append((r0, r1, c0, c1))
    # Bounded random order; at most 2*(w/2)^2*(h+1)^2 small candidates.
    for i in rng.permutation(len(candidates)):
        r0, r1, c0, c1 = candidates[int(i)]
        trial = full.copy()
        trial[r0:r1, c0:c1] = DAMAGE
        if len(_components(trial != bg)) != 1:
            continue
        if require_a:
            if _casting(trial == color) == stamp:
                continue
            if _casting((trial == color) | (trial == DAMAGE)) == stamp:
                continue
        repaired, _ = _restore(trial, bg)
        _require(np.array_equal(repaired, full), "damage lost its visible witness")
        return trial
    raise AssertionError("no eligible damage rectangle")


def _pair(rng, kinds, severe, roles, turn, seen):
    bg, rail, palette = roles
    masks = []
    remaining = 28-(len(kinds)-1)
    for i, kind in enumerate(kinds):
        future_min = sum(5 if k in ('partial', 'tail', 'mixed', 'open') else 4 if k == 'active' else 3
                         for k in kinds[i+1:])
        mask = _mold(rng, kind, min(9, remaining-future_min))
        masks.append(mask)
        remaining -= mask.shape[0]
    rng.shuffle(masks)
    widest = max(m.shape[1] for m in masks)
    band = _rand(rng, widest, min(14, 28-widest))
    col = _rand(rng, widest+1, 29-band)
    base_h = sum(m.shape[0] for m in masks)+len(masks)-1
    height = _rand(rng, base_h+1, min(30, base_h+5))
    gaps = rng.multinomial(height-base_h, np.full(len(masks)+1, 1/(len(masks)+1)))
    inp = np.full((height, col+1+band), bg, dtype=np.int8)
    inp[:, col] = rail
    row = int(gaps[0])
    for i, mask in enumerate(masks):
        color = int(rng.choice(palette))
        # Active molds must exhibit a casting difference under both cheap fills.
        tile = _damage(rng, mask, color, bg, severe, bool(_casting(mask)))
        shift = _rand(rng, 0, col-mask.shape[1])
        if i == 0 and shift == 0:
            shift = 1
        start = col-mask.shape[1]-shift
        inp[row:row+mask.shape[0], start:start+mask.shape[1]] = tile
        row += mask.shape[0]+1+int(gaps[i+1])
    inp = np.rot90(inp, turn).copy()
    if _key(inp) in seen:
        return None
    return {'input': inp, 'output': solve(inp)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    available = [c for c in range(10) if c != DAMAGE]
    fixed = tuple(map(int, rng.choice(available, 4, replace=False)))
    vary_roles = rng.random() < .55
    mixed_direction = rng.random() < .45
    base_turn = _rand(rng, 0, 3)
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(nt+nq):
        test = index >= nt
        if index == 0:
            kinds = ['partial', 'open']
        elif index == 1:
            kinds = ['tail', 'mixed'] if rng.random() < .55 else ['tail']
        else:
            count = _rand(rng, 1, 3)
            kinds = [str(rng.choice(['active', 'partial', 'mixed']))]
            kinds += list(map(str, rng.choice(['active', 'decoy', 'open', 'solid', 'mixed', 'tail'], count-1)))
        if vary_roles:
            chosen = tuple(map(int, rng.choice(available, 5, replace=False)))
            roles = (chosen[0], chosen[1], chosen[2:])
        else:
            roles = (fixed[0], fixed[1], fixed[2:])
        turn = _rand(rng, 0, 3) if mixed_direction else base_turn
        for _ in range(12):
            pair = _pair(rng, kinds, test, roles, turn, seen)
            if pair is not None:
                break
        else:
            raise AssertionError("bounded uniqueness attempts exhausted")
        puzzle['test' if test else 'train'].append(pair)
        seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Pixel-derived semantics, repair witnesses, train contrasts and schema."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad example counts")
    seen, train_objects = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair schema")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "casting preserves canvas shape")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            analysis = _analyze(a)
            canonical, turn, bg, rail, col, objects = analysis
            _require(any(o['damage'] is not None for o in objects), "no missing material")
            _require(not np.any(b == DAMAGE), "damage was not repaired")
            _require(np.array_equal(b, _render(a, analysis)), "incorrect repair, docking, or cast")
            oriented_out = np.rot90(b, turn)
            _require(np.all(oriented_out[:, col] == rail), "rail changed")
            for o in objects:
                width = o['mask'].shape[1]
                docked = oriented_out[o['row']:o['end'], col-width:col]
                _require(np.array_equal(docked, o['restored']), "shape/material changed during docking")
                _require(np.count_nonzero(docked == o['color']) == int(o['mask'].sum()), "material area changed")
            if split == 'train':
                train_objects.extend(objects)
    _require(any(o['a_needed'] for o in train_objects), "training fails to require reflection")
    _require(any(o['enclosure_needed'] for o in train_objects), "training lacks open-notch rejection")
    _require(any(o['extreme_needed'] for o in train_objects), "training lacks recessed-hole rejection")
    _require(any(o['partial'] for o in train_objects), "training does not establish whole-cavity selection")
    _require(any(o['orientation_needed'] for o in train_objects), "training does not distinguish cast orientation")
    _require(any(o['shift'] > 0 for o in train_objects), "training fails to show docking")
    return None


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, aggregate diversity, counterfactuals, equivariance, mutations.

    At most three puzzles are live during replacement; no saved grids or traces.
    """
    signatures, shapes, masks, directions, counts, casts = set(), set(), set(), set(), set(), set()
    totals = dict(pairs=0, reflection_essential=0, open_contrasts=0,
                  recessed_contrasts=0, partial_contact=0, asymmetric_casts=0, zero_shifts=0)
    example = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed == 0:
            example = puzzle
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                canonical, turn, bg, rail, col, objects = _analyze(a)
                _require(np.array_equal(solve(a.copy()), pair['output']), "input-only solve mismatch")
                shapes.add(a.shape)
                directions.add(turn)
                counts.add(len(objects))
                totals['pairs'] += 1
                for o in objects:
                    masks.add(_key(o['mask']))
                    casts.add(tuple(sorted(o['stamp'])))
                    totals['reflection_essential'] += o['a_needed']
                    totals['open_contrasts'] += o['enclosure_needed']
                    totals['recessed_contrasts'] += o['extreme_needed']
                    totals['partial_contact'] += o['partial']
                    totals['asymmetric_casts'] += o['orientation_needed']
                    totals['zero_shifts'] += o['shift'] == 0
                    y, s, x, t = o['damage']
                    crosses = 2*y < o['mask'].shape[0]-1 < 2*(s-1)
                    _require(crosses == (split == 'test'), "intended damage shift missing")
    _require(len(signatures) == 42, "cross-seed duplication")
    _require(len(shapes) >= 35 and len(masks) >= 100 and len(casts) >= 60, "structural variation collapsed")
    _require(directions == set(range(4)) and counts == {1, 2, 3}, "geometry variation collapsed")
    _require(min(totals[k] for k in ('reflection_essential', 'open_contrasts', 'recessed_contrasts', 'partial_contact', 'asymmetric_casts')) >= 42,
             "parent/connecting concept coverage collapsed")
    _require(totals['zero_shifts'] > 0, "already-docked molds absent")
    _require(_signature(generate_puzzle(0)) == _signature(example), "nondeterministic seed")
    del puzzle
    # All count combinations, using only one puzzle at a time.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(91+nt*7+nq, nt, nq)
            _require((len(p['train']), len(p['test'])) == (nt, nq), "counts ignored")
    del p
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for turn in range(4):
        for pair in example['train']+example['test']:
            _require(np.array_equal(solve(np.rot90(pair['input'], turn)), np.rot90(pair['output'], turn)),
                     "rotation covariance failed")
    # Visible roles can be permuted; the fixed gray damage convention stays 8.
    lut = np.arange(10, dtype=np.int8)
    other = [c for c in range(10) if c != DAMAGE]
    lut[other] = np.roll(other, 3)
    recolored = {sp: [{k: lut[v] for k, v in pair.items()} for pair in example[sp]] for sp in example}
    validate_generated_puzzle(recolored)
    del recolored
    bad = {sp: [{k: v.copy() for k, v in pair.items()} for pair in example[sp]] for sp in example}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    # Hand-specified rule fixture: the left chamber is an inactive witness for
    # the right one, whose recessed middle row must appear at depth three.
    rows = ('1111111', '1001001', '1101011', '1001001', '1111111')
    mold = np.array([[int(c) for c in row] for row in rows], dtype=bool)
    _require(_casting(mold) == {(1, 2), (1, 3), (2, 3), (3, 2), (3, 3)}, "full-cavity cast fixture")
    inp = np.zeros((7, 17), dtype=np.int8)
    inp[:, 9] = 5
    inp[1:6, :7] = mold.astype(np.int8)*2
    inp[1:3, 5:7] = DAMAGE
    expected = np.zeros_like(inp)
    expected[:, 9] = 5
    expected[1:6, 2:9] = mold.astype(np.int8)*2
    for r, c in ((2, 11), (2, 12), (3, 12), (4, 11), (4, 12)):
        expected[r, c] = 2
    _require(np.array_equal(solve(inp), expected), "repair/docking/casting rule fixture")
    for rows in (('1110111', '1000001', '1111111', '1000001', '1110111'),
                 ('1111111', '0100010', '0100010', '0100010', '1111111')):
        silent = np.array([[int(c) for c in row] for row in rows], dtype=bool)
        _require(not _casting(silent), "open/recessed fixture must remain silent")
    # Destroy a complete orbit and independently corrupt visible agreement.
    for tile in (np.array([[8, 1, 8], [1, 0, 1], [8, 1, 8]], dtype=np.int8),
                 np.array([[1, 1, 1], [1, 0, 1], [1, 2, 1]], dtype=np.int8)):
        try:
            _restore(tile, 0)
        except AssertionError:
            pass
        else:
            raise AssertionError("invalid reflected evidence accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                mold_shapes=len(masks), distinct_casts=len(casts), **totals)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
