"""A marked stripe program spends colored components to weave concentric layers.

Only NumPy and the standard library are used. Generation, solving and validation
do no file I/O. No parent generator is imported or called.
"""
from collections import Counter
import hashlib
import numpy as np


CAP = 9
HOST_SEEDS = (0,760095615,901625792,885785338,386668753,886934224,
              387307595,603911970,727983175,851756128,338510534,466713988,
              980366526,276967088,374958972,466989638,302846137,340621450,
              272161396,134756545,626140557,786373579,616318133,433011364,
              53537670,76621553,893591623,541157198,798233893,677183697,
              877396833,729566605,976875941,651931712,134251615,21110055,
              7687732,917366735,730445469,612865103,374093170,873073102)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= g) & (g <= 9))), "grid colors")
    return g


def _parts(g, bg):
    """Foreground components, ignoring color; objects have a full background moat."""
    seen = np.zeros(g.shape, dtype=bool)
    result = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == bg:
                continue
            seen[r, c] = True
            pending, cells = [(r, c)], []
            while pending:
                rr, cc = pending.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)):
                    if (0 <= nr < h and 0 <= nc < w and not seen[nr,nc]
                            and g[nr,nc] != bg):
                        seen[nr,nc] = True
                        pending.append((nr,nc))
            result.append(cells)
    return result


def _inspect(input_grid):
    """Read the cap, stripes and inventory from input pixels, with no seed/state."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(bg != CAP, "cap cannot be background")
    _require(np.all(g[0] == bg) and np.all(g[-1] == bg)
             and np.all(g[:,0] == bg) and np.all(g[:,-1] == bg), "background border")
    parts = _parts(g, bg)
    panels = [p for p in parts if any(g[r,c] == CAP for r,c in p)]
    _require(len(panels) == 1, "expected one capped program")
    panel = panels[0]
    r0, r1 = min(r for r,c in panel), max(r for r,c in panel)
    c0, c1 = min(c for r,c in panel), max(c for r,c in panel)
    block = g[r0:r1+1, c0:c1+1]
    _require(len(panel) == block.size, "program must be a filled rectangle")
    candidates = []
    for turn in range(4):
        b = np.rot90(block, turn)
        if (3 <= b.shape[0] <= 8 and 3 <= b.shape[1] <= 6
                and np.all(b[0] == CAP) and np.all(b[1:] != CAP)
                and np.all(b[1:] == b[1:, :1])):
            candidates.append((tuple(int(v) for v in b[1:,0]), turn, b.shape[1]))
    _require(len(candidates) == 1, "program cap/direction ambiguous")
    code, turn, stripe_width = candidates[0]
    _require(2 <= len(set(code)) <= 5, "program category count")
    _require(bg not in code, "background stripe")
    labels = np.full(g.shape, -1, dtype=int)
    counts, areas = Counter(), Counter()
    objects = []
    for i, cells in enumerate(parts):
        for r,c in cells:
            labels[r,c] = i
        if cells is panel:
            continue
        colors = {int(g[r,c]) for r,c in cells}
        _require(len(colors) == 1, "objects must be monochromatic")
        color = colors.pop()
        _require(1 <= len(cells) <= 9, "object area")
        rr = [r for r,c in cells]
        cc = [c for r,c in cells]
        _require(max(rr)-min(rr) < 4 and max(cc)-min(cc) < 4, "object extent")
        counts[color] += 1
        areas[color] += len(cells)
        mask = tuple(sorted((r-min(rr),c-min(cc)) for r,c in cells))
        objects.append((color, mask, (min(rr),min(cc))))
    for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
        a = labels[max(0,dr):g.shape[0]+min(0,dr), max(0,dc):g.shape[1]+min(0,dc)]
        b = labels[max(0,-dr):g.shape[0]+min(0,-dr), max(0,-dc):g.shape[1]+min(0,-dc)]
        _require(not np.any((a >= 0) & (b >= 0) & (a != b)), "components touch")
    selected = {c: counts[c] for c in set(code)}
    total = sum(selected.values())
    _require(1 <= total <= 15 and max(selected.values()) <= 5, "inventory bounds")
    _require(len(objects) <= 18, "object budget")
    _require(sum(c not in selected for c,mask,pos in objects) <= 3, "distractor budget")
    return dict(background=bg, code=code, counts=selected, areas=areas,
                objects=objects, total=total, turn=turn, stripe_width=stripe_width,
                panel_box=(r0,c0,r1,c1))


def _spend(code, counts):
    """One stripe visit consumes one component; repeat the entire strip as needed."""
    remaining = dict(counts)
    layers = []
    while any(remaining.values()):
        for color in code:
            if remaining[color] > 0:
                layers.append(color)
                remaining[color] -= 1
    return layers


def _edge_render(layers):
    side = 2 * len(layers)
    d = np.minimum(np.arange(side), np.arange(side)[::-1])
    return np.asarray(layers, dtype=np.int8)[np.minimum(d[:,None], d[None,:])]


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _inspect(input_grid)
    return _edge_render(_spend(d['code'], d['counts']))


def _shape(rng, area=None):
    """Independently sampled connected masks, with bounded 4-by-4 extent."""
    area = int(rng.integers(1, 10)) if area is None else int(area)
    kind = int(rng.integers(4))
    if kind == 0 and area <= 4:
        cells = {(0,c) for c in range(area)}
    elif kind == 1:
        width = int(rng.integers(max(1,(area+3)//4), min(4,area)+1))
        cells = {(i//width,i%width) for i in range(area)}
    else:
        cells = {(1,1)}
        if kind == 3 and area >= 5:
            cells |= {(0,1),(1,0),(1,2),(2,1)}
        while len(cells) < area:
            frontier = sorted({(r+dr,c+dc) for r,c in cells
                               for dr,dc in ((0,1),(0,-1),(1,0),(-1,0))
                               if 0 <= r+dr < 4 and 0 <= c+dc < 4} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c,-r) for r,c in cells}
    if rng.random() < .5:
        cells = {(r,-c) for r,c in cells}
    r0,c0 = min(r for r,c in cells), min(c for r,c in cells)
    return tuple(sorted((r-r0,c-c0) for r,c in cells))


def _place(rng, panel, bg, specs):
    """Try free placement, then a guaranteed bounded shelf fallback; retain masks."""
    ph,pw = panel.shape
    for attempt in range(9):
        if attempt == 8:
            h,w = 30,30
            r0,c0 = 1,1
        else:
            h = min(30, int(rng.integers(14,25)) + 2*(attempt//2))
            w = min(30, int(rng.integers(14,25)) + 2*(attempt//2))
            r0 = int(rng.integers(1,h-ph))
            c0 = int(rng.integers(1,w-pw))
        g = np.full((h,w),bg,dtype=np.int8)
        g[r0:r0+ph,c0:c0+pw] = panel
        occupied = np.zeros((h,w),dtype=bool)
        occupied[0] = occupied[-1] = True
        occupied[:,0] = occupied[:,-1] = True
        occupied[r0-1:r0+ph+1,c0-1:c0+pw+1] = True
        ordered = sorted(range(len(specs)), key=lambda j: (-len(specs[j][1]),j))
        failed = False
        for j in ordered:
            color,mask = specs[j]
            sh,sw = 1+max(r for r,c in mask),1+max(c for r,c in mask)
            candidates = np.ones((h-sh+1,w-sw+1),dtype=bool)
            for dr,dc in mask:
                candidates &= ~occupied[dr:dr+h-sh+1,dc:dc+w-sw+1]
            rr,cc = np.nonzero(candidates)
            if attempt == 8:
                # 5x5 tiles, with 4 tiles reserved for the <=8x6 program.
                slots = [(r,c) for r in (1,6,11,16,21) for c in (1,6,11,16,21)
                         if not (r < 11 and c < 11) and candidates[r,c]]
                if not slots:
                    failed = True
                    break
                r,c = slots[0]
            elif len(rr):
                index = int(rng.integers(len(rr)))
                r,c = int(rr[index]),int(cc[index])
            else:
                failed = True
                break
            for dr,dc in mask:
                g[r+dr,c+dc] = color
                occupied[max(0,r+dr-1):r+dr+2,max(0,c+dc-1):c+dc+2] = True
        if not failed:
            return g
    raise AssertionError("bounded packing fallback exhausted")


def _build(rng, code, counts, bg, rotation, forced=None, distractors=()):
    forced = forced or {}
    stripe_width = int(rng.integers(3,7))
    panel = np.repeat(np.array([CAP]+code,dtype=np.int8)[:,None], stripe_width, axis=1)
    specs = []
    for color,count in counts.items():
        for j in range(count):
            area = forced[color][j] if color in forced else None
            specs.append((color,_shape(rng,area)))
    specs.extend((color,_shape(rng)) for color in distractors)
    rng.shuffle(specs)
    inp = np.rot90(_place(rng,panel,bg,specs),rotation).copy()
    # Generation uses explicit inventories; solving must recover them from pixels.
    return {'input':inp, 'output':_edge_render(_spend(code,counts))}


def _inventory(rng, colors, total, maximum, required=None):
    q = {c:0 for c in colors}
    if required is not None:
        q[required] = 1
    for _ in range(total-sum(q.values())):
        options = [c for c in colors if q[c] < maximum]
        q[int(rng.choice(options))] += 1
    return q


def _program(rng, colors, length):
    code = [int(c) for c in rng.permutation(colors)]
    while len(code) < length:
        code.insert(int(rng.integers(len(code)+1)),int(rng.choice(colors)))
    return code


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,nq = _int(num_train,'num_train'),_int(num_test,'num_test')
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    # Only presentation/evidence is scheduled. All semantic conventions are fixed.
    a,b,c,d,novel,zero_color = [int(v) for v in rng.choice(9,6,replace=False)]
    collision = int(rng.integers(4,9))
    other = int(rng.choice([v for v in range(3,10) if v != collision]))
    split = int(rng.integers(1,collision))
    other_split = int(rng.integers(1,other))
    base_rotation = int(rng.integers(4))
    puzzle,seen = {'train':[], 'test':[]},set()
    previous_bg = None
    for index in range(nt+nq):
        training = index < nt
        first_test = index == nt
        forced = {}
        if index == 0:
            colors = [a,b,c,zero_color]
            code = [int(v) for v in rng.permutation(colors)]
            counts = {a:1,b:int(rng.integers(2,5)),c:2,zero_color:0}
            forced = {a:[collision],c:[other_split,other-other_split]}
        elif index == 1:
            colors = [a,d,c]
            code = ([a,d,a,c],[a,c,a,d],[d,a,c,a])[int(rng.integers(3))]
            first_total = _inspect(puzzle['train'][0]['input'])['total']
            dcount = int(rng.choice([v for v in (2,3,4) if v+3 != first_total]))
            counts = {a:2,d:dcount,c:1}
            forced = {a:[split,collision-split]}
        else:
            n = int(rng.integers(2 if training else 3,6))
            pool = [v for v in range(9) if not training or v != novel]
            colors = [int(v) for v in rng.choice(pool,n,replace=False)]
            if first_test and novel not in colors:
                colors[0] = novel
            length = int(rng.integers(n,8))
            code = _program(rng,colors,length)
            total = int(rng.integers(3, min(10,4*n)+1)) if training else int(rng.integers(11,16))
            counts = _inventory(rng,colors,total,4 if training else 5,
                                novel if first_test else None)
        bg_pool = [v for v in range(9) if v not in colors and v != previous_bg
                   and not (index == 1 and v == b)]
        bg = int(rng.choice(bg_pool))
        dpool = [v for v in range(9) if v != bg and v not in colors]
        distractors = [int(rng.choice(dpool)) for _ in range(int(rng.integers(4)))]
        if index == 1:
            distractors = [b] + distractors[:2]
        rotation = (base_rotation + index) % 4 if index < 2 else int(rng.integers(4))
        if first_test:
            rotation = (base_rotation+2) % 4
        for attempt in range(8):
            pair = _build(rng,code,counts,bg,rotation,forced,distractors)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            raise AssertionError("distinct-input retry bound exceeded")
        seen.add(key)
        puzzle['train' if training else 'test'].append(pair)
        previous_bg = bg
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(g):
    a = np.asarray(g,dtype=np.int8)
    return a.shape,a.tobytes()


def _layer_check(out, code, inventory):
    """Validate shells and resource use directly, without solve or _spend."""
    remaining = dict(inventory)
    total = sum(remaining.values())
    _require(out.shape == (2*total,2*total), "output size is not twice inventory")
    _require(np.array_equal(out,out[::-1]) and np.array_equal(out,out[:,::-1])
             and np.array_equal(out,out.T), "output square symmetries")
    cursor = 0
    for depth in range(total):
        # Cursor moves continuously through the cyclic code, skipping exhausted colors.
        while remaining[code[cursor % len(code)]] == 0:
            cursor += 1
        color = code[cursor % len(code)]
        remaining[color] -= 1
        cursor += 1
        shell = out[depth:2*total-depth,depth:2*total-depth]
        _require(np.all(shell[0] == color) and np.all(shell[-1] == color)
                 and np.all(shell[:,0] == color) and np.all(shell[:,-1] == color),
                 "wrong layer order/color")
    _require(not any(remaining.values()), "unspent inventory")


def _evidence(details):
    items = [(c,d['counts'][c],d['areas'][c]) for d in details for c in set(d['code'])]
    collision = any(c == cc and n != nn and area == aa and area > 0
                    for c,n,area in items for cc,nn,aa in items)
    independence = any(n == nn and n > 0 and area != aa
                       for c,n,area in items for cc,nn,aa in items)
    candidates = {'one_per_stripe':False,'group_by_color':False,'reverse':False,
                  'deduplicate_program':False,'single_pass':False}
    for d in details:
        code,q = d['code'],d['counts']
        actual = _spend(code,q)
        firsts = list(dict.fromkeys(code))
        alternatives = {
            'one_per_stripe':list(code),
            'group_by_color':[c for c in firsts for _ in range(q[c])],
            'reverse':_spend(code[::-1],q),
            'deduplicate_program':_spend(firsts,q),
        }
        remaining = dict(q)
        one_pass = []
        for c in code:
            if remaining[c]:
                one_pass.append(c)
                remaining[c] -= 1
        alternatives['single_pass'] = one_pass
        for name,alternative in alternatives.items():
            candidates[name] |= actual != alternative
    categories = {c for d in details for c in d['code']}
    discards = {c for d in details for c,mask,pos in d['objects'] if c not in d['counts']}
    return collision,independence,candidates,bool(categories & discards)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "split schema")
    nt,nq = len(puzzle['train']),len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen,details = set(),{'train':[],'test':[]}
    for section in ('train','test'):
        for p in puzzle[section]:
            _require(isinstance(p,dict) and set(p) == {'input','output'}, "pair schema")
            inp,out = _grid(p['input']),_grid(p['output'])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _inspect(inp)
            _layer_check(out,d['code'],d['counts'])
            _require(np.array_equal(solve(inp),out), "input-only solve mismatch")
            details[section].append(d)
    tr,te = details['train'],details['test']
    collision,independence,alternatives,role_switch = _evidence(tr)
    _require(collision and independence, "training permits an area lookup")
    _require(all(alternatives.values()), "training leaves a named shortcut unresolved")
    _require(role_switch, "local category/distractor role witness missing")
    _require(len({d['total'] for d in tr}) >= 2, "inventory-size evidence missing")
    _require(len({d['turn'] % 2 for d in tr}) == 2, "both strip axes must be demonstrated")
    _require(len({d['background'] for d in tr}) >= 2, "background role evidence missing")
    _require(any(0 in d['counts'].values() for d in tr), "initial exhaustion evidence missing")
    _require(min(d['total'] for d in te) > max(d['total'] for d in tr), "inventory test shift")
    train_colors = {c for d in tr for c in d['code']}
    _require(any(any(c not in train_colors and n > 0 for c,n in d['counts'].items())
                 for d in te), "new test category missing")


def self_test():
    """42 seeds, all legal counts, shortcut witnesses and semantic mutation tests.

    Retains only small summary sets; emits no grids, traces, logs or files.
    """
    summaries = {name:set() for name in ('backgrounds','orientations','shapes',
                 'program_lengths','categories','counts','output_sides','object_masks',
                 'structures','puzzles','stripe_widths','positions')}
    zero,repeated,adjacent,distractor,passes,area_independent = (0,)*6
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train','test'):
            for p in puzzle[split]:
                g = p['input']
                detail = _inspect(g)
                code,q = detail['code'],detail['counts']
                for name,value in (('backgrounds',detail['background']),('orientations',detail['turn']),
                                   ('shapes',g.shape),('program_lengths',len(code)),
                                   ('categories',len(q)),('output_sides',2*detail['total']),
                                   ('stripe_widths',detail['stripe_width'])):
                    summaries[name].add(value)
                summaries['counts'].update(q.values())
                summaries['object_masks'].update(mask for c,mask,pos in detail['objects'])
                summaries['positions'].update((pos[0]%3,pos[1]%3) for c,mask,pos in detail['objects'])
                normalized = {}
                signature = np.array([normalized.setdefault(int(c),len(normalized)) for c in g.flat],dtype=np.int8)
                summaries['structures'].add((g.shape,hashlib.sha256(signature.tobytes()).hexdigest()))
                digest.update(g.tobytes())
                zero += 0 in q.values()
                repeated += len(set(code)) < len(code)
                adjacent += any(x == y for x,y in zip(code,code[1:]))
                distractor += any(c not in q for c,mask,pos in detail['objects'])
                passes += len(_spend(code,q)) > len(code)
                area_independent += any(q[c] != detail['areas'][c] and q[c] > 0 for c in q)
        summaries['puzzles'].add(digest.hexdigest())
    _require(len(summaries['puzzles']) == 42, "42 distinct puzzles")
    _require(summaries['backgrounds'] == set(range(9)), "background coverage")
    _require(summaries['orientations'] == set(range(4)), "all cap directions")
    _require(summaries['program_lengths'] == set(range(2,8)), "program lengths")
    _require(summaries['categories'] == set(range(2,6)), "category counts")
    _require(summaries['counts'] == set(range(6)), "inventory counts 0..5")
    _require(summaries['stripe_widths'] == set(range(3,7)), "stripe widths")
    _require(len(summaries['shapes']) >= 45 and len(summaries['structures']) >= 180, "structural variety")
    _require(len(summaries['object_masks']) >= 100, "object mask variety")
    _require(summaries['positions'] == {(r,c) for r in range(3) for c in range(3)}, "free placement")
    _require(min(zero,repeated,adjacent,distractor,passes,area_independent) >= 10, "mechanism variation")
    first,again = generate_puzzle(17,7,3),generate_puzzle(17,7,3)
    _require(all(_key(a[k]) == _key(b[k]) for s in first for a,b in zip(first[s],again[s])
                 for k in ('input','output')), "reproducibility")
    for nt in range(2,11):
        for nq in range(1,min(nt-1,4)+1):
            p = generate_puzzle(1000+nt*10+nq,nt,nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "explicit counts")
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),
                  (4.0,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Fixture: duplicate station, zero category, a depleted color and a full restart.
    _require(_spend((1,2,1,3),{1:3,2:0,3:2}) == [1,1,3,1,3], "depletion fixture")
    # All supported boundaries and rotations, independent of training/test schedules.
    fixture_rng = np.random.default_rng(117)
    for code,q in (([1,2],{1:1,2:0}),([1,2,3],{1:5,2:5,3:5}),
                   ([1,2,1,3],{1:3,2:0,3:2})):
        for rotation in range(4):
            fixture = _build(fixture_rng,code,q,0,rotation,distractors=(4,4,5))
            _require(np.array_equal(solve(fixture['input']),fixture['output']), "boundary fixture")
    q = {1:1,2:2,3:1}
    small = _build(fixture_rng,[1,2,3],q,0,0,forced={1:[1],2:[1,1],3:[1]})
    large = _build(fixture_rng,[1,2,3],q,0,3,forced={1:[9],2:[8,9],3:[7]})
    _require(np.array_equal(solve(small['input']),solve(large['input'])), "area/position invariance")
    reordered = _build(fixture_rng,[3,2,1],q,0,2)
    _require(not np.array_equal(solve(small['input']),solve(reordered['input'])), "program order must matter")
    more = _build(fixture_rng,[1,2,3],{1:2,2:2,3:1},0,1)
    _require(not np.array_equal(solve(small['input']),solve(more['input'])), "component count must matter")
    permutation = np.array([8,7,6,5,4,3,2,1,0,9],dtype=np.int8)
    _require(np.array_equal(solve(permutation[small['input']]),permutation[small['output']]), "local palette roles")
    # A corrupted shell and an incorrect dimensional interpretation must fail.
    p = generate_puzzle(29,2,1)
    p['test'][0]['output'][0,0] = (int(p['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {**{name:len(values) for name,values in summaries.items()},
            'zero_inventory_pairs':zero,'repeated_program_pairs':repeated,
            'adjacent_repeat_pairs':adjacent,'distractor_pairs':distractor,
            'longer_than_one_pass_pairs':passes,'pairs_with_nonsingleton_counted_objects':area_independent,
            'seeds_checked':42,'independent_solver':False}
