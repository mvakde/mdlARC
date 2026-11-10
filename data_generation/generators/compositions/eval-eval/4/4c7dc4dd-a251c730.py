"""Counted motif relays: a local analogy chooses the growing port color.

Only standard-library modules and NumPy are used. No parent code is imported.
"""
from __future__ import annotations

from itertools import combinations
import json

import numpy as np


HOST_SEEDS = (0,791418080,17502472,805627091,607574221,122100632,876453137,
516348194,11135326,803681297,636350356,184853090,779892392,682264263,
905515725,133972538,935986512,259873874,513119810,404431198,337121195,
970107062,85524961,743357823,221856894,97089629,46935011,590006790,
402404196,47712714,744593931,475519885,930403475,487620146,665394494,
956318950,438077031,838356112,27997894,664216200,47714123,518975897)
_D4 = ((1,0),(-1,0),(0,1),(0,-1))


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in _D4:
                q = p[0]+dr, p[1]+dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(sorted(cells))
    return result


def _panel_candidates(a):
    """Recognize complete borders in context, even if colors continue outside."""
    h, w = a.shape
    down = np.ones(a.shape, dtype=int)
    right = np.ones(a.shape, dtype=int)
    for r in range(h-2, -1, -1):
        down[r] += (a[r] == a[r+1])*down[r+1]
    for c in range(w-2, -1, -1):
        right[:,c] += (a[:,c] == a[:,c+1])*right[:,c+1]
    panels = []
    for top in range(h-4):
        for bottom in range(top+4, h):
            cols = np.flatnonzero(down[top] >= bottom-top+1)
            for left, end in combinations(map(int, cols), 2):
                width = end-left+1
                if width < 5 or right[top,left] < width or right[bottom,left] < width:
                    continue
                if a[top,left] != a[bottom,left]:
                    continue
                inner = a[top+1:bottom, left+1:end]
                fill = int(inner[0,0])
                if fill == a[top,left]:
                    continue
                if not (np.all(inner[0] == fill) and np.all(inner[-1] == fill)
                        and np.all(inner[:,0] == fill) and np.all(inner[:,-1] == fill)):
                    continue
                if np.count_nonzero(inner == fill) <= inner.size/2:
                    continue
                cells = _components(inner != fill)
                if cells:
                    panels.append(dict(box=(top,left,bottom-top+1,width),
                                       fill=fill, frame=int(a[top,left]),
                                       inner=inner, components=cells))
    return panels


def _disjoint(x, y):
    r,c,h,w = x
    s,t,j,k = y
    return r+h <= s or s+j <= r or c+w <= t or t+k <= c


def _direction(template, key):
    """The root is the sole cell on the back edge of its oriented fan."""
    offsets = [p for p, color in template.items() if color != key]
    candidates = [v for v in _D4 if all(r*v[0]+c*v[1] > 0 for r,c in offsets)]
    assert len(candidates) == 1, 'motif needs one visible growth direction'
    return candidates[0]


def _check_template(template, key):
    assert template.get((0,0)) == key
    assert sum(c == key for c in template.values()) == 1
    direction = _direction(template, key)
    side = (-direction[1], direction[0])
    coords = {(r*direction[0]+c*direction[1], r*side[0]+c*side[1]): color
              for (r,c),color in template.items()}
    depth = max(r for r,c in coords)
    terminal = {c: color for (r,c),color in coords.items() if r == depth}
    assert 2 <= depth <= 4 and len(terminal) == 3
    s = max(terminal)
    assert s in (1,2) and set(terminal) == {-s,0,s}
    assert terminal[-s] == terminal[s] != terminal[0]
    body = {color for (r,c),color in coords.items() if 0 < r < depth}
    assert len(body) == 1 and body.isdisjoint(set(terminal.values()) | {key})
    assert sum(0 < r < depth for r,c in coords) >= 3
    assert set(template.values()) == body | set(terminal.values()) | {key}
    assert all(0 < r <= depth and abs(c) <= s for r,c in coords if (r,c) != (0,0))
    # The final body row supports every port; the rest is a connected shape.
    assert all((depth-1,c) in coords for c in range(-s,s+1))
    remaining = set(template)
    stack = [remaining.pop()]
    while stack:
        r,c = stack.pop()
        for dr,dc in _D4:
            p = r+dr,c+dc
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    assert not remaining, 'template must be connected'
    return direction, set(terminal.values()), depth, s


def _assembly(template, key, active, generations):
    """Union of a finite relay tree, merging coincident buds.

    A successor key replaces its incoming port. Every other overlap must agree.
    This also specifies the simultaneous, order-independent painting rule.
    """
    assert 1 <= generations <= 4
    ports = [p for p,color in template.items() if color == active]
    assert ports and (0,0) not in ports
    frontier, anchors = {(0,0)}, {(0,0)}
    for _ in range(generations-1):
        frontier = {(r+dr,c+dc) for r,c in frontier for dr,dc in ports}
        anchors |= frontier
    painted = {p:key for p in anchors}
    for ar,ac in sorted(anchors):
        for (dr,dc),color in sorted(template.items()):
            p = ar+dr,ac+dc
            if p in anchors:
                continue
            assert p not in painted or painted[p] == color, 'conflicting relay colors'
            painted[p] = color
    return painted, anchors


def _read_reference(panel, keys):
    inner = panel['inner']
    templates, compounds = {}, []
    for comp in panel['components']:
        key_cells = [(p,int(inner[p])) for p in comp if int(inner[p]) in keys]
        assert key_cells and len({k for p,k in key_cells}) == 1
        key = key_cells[0][1]
        if len(key_cells) == 1:
            (ar,ac),_ = key_cells[0]
            template = {(int(r-ar),int(c-ac)):int(inner[r,c]) for r,c in comp}
            _check_template(template,key)
            assert key not in templates or templates[key] == template
            templates[key] = template
        else:
            compounds.append((comp,key,key_cells))
    assert set(templates) == keys, 'every query key needs a complete singleton motif'
    assert len(compounds) == 1, 'one two-generation calibration assembly'
    all_ports = [_check_template(t,k)[1] for k,t in templates.items()]
    assert all(p == all_ports[0] for p in all_ports), 'shared visible port roles'
    assert all(not (set(t.values())-{k}) & keys for k,t in templates.items())
    comp,key,key_cells = compounds[0]
    possibilities = []
    for (ar,ac),_ in key_cells:
        observed = {(int(r-ar),int(c-ac)):int(inner[r,c]) for r,c in comp}
        for active in sorted(all_ports[0]):
            expected,_ = _assembly(templates[key],key,active,2)
            if expected == observed:
                possibilities.append((active,(int(ar),int(ac)),key))
    assert len(possibilities) == 1, 'calibration must uniquely select a port color'
    active,root,calibration_key = possibilities[0]
    return templates, active, calibration_key, root


def _interpret(a):
    panels = _panel_candidates(a)
    interpretations = []
    for target in panels:
        ti = target['inner']
        if not all(len({int(ti[p]) for p in comp}) == 1 for comp in target['components']):
            continue
        keys = {int(ti[comp[0]]) for comp in target['components']}
        for ref in panels:
            if not _disjoint(target['box'],ref['box']):
                continue
            try:
                templates,active,calkey,calroot = _read_reference(ref,keys)
                r,c,h,w = target['box']
                out = a[r:r+h,c:c+w].copy()
                out[1:-1,1:-1] = target['fill']
                used, queries = set(), []
                for comp in target['components']:
                    key = int(ti[comp[0]])
                    direction = _direction(templates[key],key)
                    side = (-direction[1],direction[0])
                    root = min(comp,key=lambda p:p[0]*side[0]+p[1]*side[1])
                    n = len(comp)
                    assert 1 <= n <= 4
                    assert set(comp) == {(root[0]+i*side[0],root[1]+i*side[1]) for i in range(n)}, 'straight count stroke'
                    painted,anchors = _assembly(templates[key],key,active,n)
                    shifted = {(root[0]+dr,root[1]+dc):color for (dr,dc),color in painted.items()}
                    assert not used & set(shifted), 'different query assemblies must be separated'
                    assert all(1 <= y < ti.shape[0]-1 and 1 <= x < ti.shape[1]-1 for y,x in shifted), 'relay must fit inside the fill rim'
                    # Different relays may not touch orthogonally either.
                    assert not any((y+dr,x+dc) in used for y,x in shifted for dr,dc in _D4)
                    used |= set(shifted)
                    for (y,x),color in shifted.items():
                        assert color != target['fill']
                        out[y+1,x+1] = color
                    queries.append(dict(key=key,root=tuple(map(int,root)),n=n,
                                        direction=direction,anchors=len(anchors),
                                        branch=len([p for p,v in templates[key].items() if v == active])))
                interpretations.append((out,dict(reference=ref,target=target,templates=templates,
                    active=active,calibration_key=calkey,calibration_root=calroot,queries=queries)))
            except AssertionError:
                continue
    assert len(interpretations) == 1, f'expected one input interpretation, found {len(interpretations)}'
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read templates, calibration and count strokes entirely from the input."""
    return _interpret(_grid(input_grid))[0]


def _rotate(p, turns):
    r,c = p
    for _ in range(turns):
        r,c = c,-r
    return int(r),int(c)


def _make_templates(rng, keys, body_colors, ports, small=False):
    templates = {}
    for i,key in enumerate(keys):
        depth = int(rng.integers(2,4 if small else 5))
        spread = 1 if small else int(rng.integers(1,3))
        # Spine and a terminal crossbar, with asymmetric connected side growth.
        coords = {(r,0) for r in range(1,depth)} | {(depth-1,c) for c in range(-spread,spread+1)}
        for r in range(1,depth-1):
            low,high = int(rng.integers(-spread,1)),int(rng.integers(0,spread+1))
            coords |= {(r,c) for c in range(low,high+1)}
        body = body_colors[i % len(body_colors)]
        fork,stem = ports if i % 2 == 0 else ports[::-1]
        t = {(0,0):key, **{p:body for p in coords}}
        t.update({(depth,-spread):fork,(depth,0):stem,(depth,spread):fork})
        turns = int(rng.integers(4))
        t = {_rotate(p,turns):color for p,color in t.items()}
        _check_template(t,key)
        templates[key] = t
    return templates


def _bounds(cells):
    r0 = min(p[0] for p in cells)
    c0 = min(p[1] for p in cells)
    return r0,c0,max(p[0] for p in cells)-r0+1,max(p[1] for p in cells)-c0+1


def _pack_options(objects, area_needed=0):
    """Shelf pack isolated objects; rim, gaps and frame have separate roles."""
    bounds = [_bounds(obj) for obj in objects]
    options = []
    for width in range(max(b[3] for b in bounds)+2,29):
        y,x,line_h = 1,1,0
        places = []
        for r0,c0,h,w in bounds:
            if x+w > width-1:
                y += line_h+1
                x,line_h = 1,0
            places.append((y-r0,x-c0))
            x += w+1
            line_h = max(line_h,h)
        height = y+line_h+1
        height = max(height,area_needed//width+1)
        if height <= 28:
            options.append((height+2,width+2,places))
    return options


def _geometry(rng, reference_objects, query_envelopes):
    refopts = _pack_options(reference_objects,2*sum(len(o) for o in reference_objects))
    taropts = _pack_options(query_envelopes)
    choices = [(r,t,axis) for r in refopts for t in taropts for axis in (0,1)
               if r[axis]+t[axis]+1 <= 30]
    if not choices:
        return None
    ref,tar,axis = choices[int(rng.integers(len(choices)))]
    sizes = [list(ref[:2]),list(tar[:2])]
    gap = int(rng.integers(1,min(3,30-sizes[0][axis]-sizes[1][axis])+1))
    required = [max(s[0] for s in sizes),max(s[1] for s in sizes)]
    required[axis] = sizes[0][axis]+sizes[1][axis]+gap
    canvas = [int(rng.integers(n,min(30,n+4)+1)) for n in required]
    if rng.random() < .12:
        canvas = [30,30]
    order = [0,1] if rng.integers(2) else [1,0]
    positions = [[0,0],[0,0]]
    start = int(rng.integers(canvas[axis]-required[axis]+1))
    positions[order[0]][axis] = start
    positions[order[1]][axis] = start+sizes[order[0]][axis]+gap
    for i in range(2):
        cross = 1-axis
        positions[i][cross] = int(rng.integers(canvas[cross]-sizes[i][cross]+1))
    boxes = [tuple(positions[i]+sizes[i]) for i in range(2)]
    return tuple(canvas),boxes,ref[2],tar[2],axis


def _draw_panel(a, box, frame, fill):
    r,c,h,w = box
    a[r:r+h,c:c+w] = frame
    a[r+1:r+h-1,c+1:c+w-1] = fill


def _paint(a, cells, root):
    for (r,c),color in cells.items():
        a[root[0]+r,root[1]+c] = color


def _make_case(rng, palette, types, counts, twin=False):
    """Construct labels from logical motifs; input parsing is a separate path."""
    for attempt in range(80):
        colors = list(map(int,palette))
        bg,rf,tf = colors[:3]
        if rng.random() < .35:
            rf = bg
        if rng.random() < .35 and not twin:
            tf = rf
        keys = colors[6:6+types]
        ports = colors[4:6]
        bodies = [colors[3]] if rng.random() < .5 else [colors[3],colors[9]]
        templates = _make_templates(rng,keys,bodies,ports,attempt >= 20)
        singletons = [templates[k] for k in keys]
        calkey = keys[0] if twin else keys[int(rng.integers(types))]
        calibration = [_assembly(templates[calkey],calkey,p,2)[0] for p in ports]
        cal_envelope = set(calibration[0]) | set(calibration[1])
        queries = [(keys[i % types],n) for i,n in enumerate(counts)]
        envelopes = []
        for key,n in queries:
            direction = _direction(templates[key],key)
            side = (-direction[1],direction[0])
            envelope = {(i*side[0],i*side[1]) for i in range(n)}
            for active in ports:
                envelope |= set(_assembly(templates[key],key,active,n)[0])
            envelopes.append(envelope)
        # The last reference object's envelope has the same placement in twins.
        reference_envelopes = [set(t) for t in singletons]+[cal_envelope]
        # Area check uses the larger actual calibration, not its bounding box.
        area = sum(len(t) for t in singletons)+max(map(len,calibration))
        geom = _geometry(rng,reference_envelopes,envelopes)
        if geom is None:
            continue
        shape,boxes,rp,tp,axis = geom
        # The envelope area bound above is conservative; this separate check
        # also makes the frame parser's majority-fill evidence explicit.
        if (boxes[0][2]-2)*(boxes[0][3]-2) <= 2*area:
            continue
        rr,cc = np.indices(shape)
        mode = int(rng.integers(4))
        word = [bg] if mode == 0 else [bg,colors[9]] if mode < 3 else [bg,colors[9],colors[3]]
        coordinate = rr if mode == 1 else cc if mode == 2 else rr+cc
        canvas = np.asarray(word,dtype=np.int8)[coordinate % len(word)]
        frames = [colors[9],colors[3] if rng.random() < .5 else colors[9]]
        if frames[0] == rf or frames[1] == tf:
            continue
        _draw_panel(canvas,boxes[0],frames[0],rf)
        _draw_panel(canvas,boxes[1],frames[1],tf)
        r,c,_,_ = boxes[0]
        for t,(ar,ac) in zip(singletons,rp):
            _paint(canvas,t,(r+1+ar,c+1+ac))
        tr,tc,th,tw = boxes[1]
        for (key,n),(ar,ac) in zip(queries,tp):
            d = _direction(templates[key],key)
            side = (-d[1],d[0])
            for i in range(n):
                canvas[tr+1+ar+i*side[0],tc+1+ac+i*side[1]] = key
        variants = []
        selected = ports if twin else [ports[int(rng.integers(2))]]
        for active in selected:
            inp = canvas.copy()
            _paint(inp,calibration[ports.index(active)],(r+1+rp[-1][0],c+1+rp[-1][1]))
            out = canvas[tr:tr+th,tc:tc+tw].copy()
            out[1:-1,1:-1] = tf
            for (key,n),(ar,ac) in zip(queries,tp):
                _paint(out,_assembly(templates[key],key,active,n)[0],(1+ar,1+ac))
            try:
                predicted,info = _interpret(inp)
                assert np.array_equal(predicted,out)
                assert info['active'] == active
            except AssertionError:
                break
            variants.append(dict(input=inp,output=out))
        if len(variants) == len(selected):
            return variants
    raise RuntimeError('bounded construction exhausted')


def _counts(a,b):
    for value,name,low,high in ((a,'num_train',2,10),(b,'num_test',1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    return int(a),int(b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a,b = _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    stable = bool(rng.integers(2))
    # Same visible dictionary and target, opposite calibrations. Neither the
    # target nor a memorized key-to-output table can account for both answers.
    twins = _make_case(rng,palette,2,[1,2,2],twin=True)
    puzzle = {'train':twins,'test':[]}
    seen = {_key(p['input']) for p in twins}
    for split,count in (('train',a-2),('test',b)):
        for _ in range(count):
            for retry in range(12):
                local_palette = palette if stable else rng.permutation(10)
                types = int(rng.integers(1,4)) if split == 'train' else int(rng.integers(2,4))
                ns = [int(rng.integers(1,3)) for _ in range(types)]
                if split == 'test':
                    ns[int(rng.integers(types))] = int(rng.integers(3,5))
                elif rng.random() < .3:
                    ns.append(int(rng.integers(1,3)))
                pair = _make_case(rng,local_palette,types,ns)[0]
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    """Schema, local semantics, analogy dependence and demonstrated evidence."""
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,records = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted,info = _interpret(inp)
            assert np.array_equal(predicted,out), 'incorrect relay assembly or crop'
            r,c,h,w = info['target']['box']
            target = inp[r:r+h,c:c+w]
            assert out.shape == target.shape
            assert all(np.array_equal(x,y) for x,y in ((out[0],target[0]),(out[-1],target[-1]),
                                                       (out[:,0],target[:,0]),(out[:,-1],target[:,-1])))
            assert all(info['target']['inner'][q['root']] == q['key'] for q in info['queries'])
            for q in info['queries']:
                root = q['root']
                assert out[root[0]+1,root[1]+1] == q['key']
                assert q['anchors'] == (q['n'] if q['branch'] == 1 else q['n']*(q['n']+1)//2)
            records[split].append(info)
    # The two mandatory contrastive demonstrations keep the target/dictionary
    # fixed. Changing ONLY the worked assembly changes the answer.
    first,second = records['train'][:2]
    assert first['templates'] == second['templates']
    assert first['active'] != second['active']
    assert first['target']['box'] == second['target']['box']
    assert np.array_equal(first['target']['inner'],second['target']['inner'])
    assert first['queries'] != second['queries']  # branching/anchor counts differ
    assert not np.array_equal(puzzle['train'][0]['output'],puzzle['train'][1]['output'])
    assert {q['n'] for f in records['train'] for q in f['queries']} == {1,2}
    assert {q['branch'] for f in records['train'][:2] for q in f['queries']} == {1,2}
    assert first['reference']['fill'] != first['target']['fill']
    assert all(max(q['n'] for q in f['queries']) >= 3 for f in records['test'])
    return None


def _signature(puzzle):
    return tuple((_key(p['input']),_key(p['output'])) for s in ('train','test') for p in puzzle[s])


def _dependency_regressions():
    """Concrete non-generation diagnostics for the two inherited mechanisms."""
    t = {(0,0):1,(1,-1):2,(1,0):2,(1,1):2,(2,-1):3,(2,0):4,(2,1):3}
    fork,fa = _assembly(t,1,3,3)
    stem,sa = _assembly(t,1,4,3)
    assert len(fa) == 6 and len(sa) == 3 and fork != stem
    # Same root/count/port choice, different dictionary body -> different answer.
    other = {p:(5 if color == 2 else color) for p,color in t.items()}
    _check_template(other,1)
    assert _assembly(other,1,3,3)[0] != fork
    # A fork's two routes meet at generation three. Merge one shared bud.
    assert (4,0) in fa and sum(p == (4,0) for p in fa) == 1
    assert fork[(2,-1)] == 1 and fork[(4,0)] == 1  # keys replace active ports
    assert fork[(2,0)] == 4  # inactive port survives
    for turns in range(4):
        turned = {_rotate(p,turns):c for p,c in t.items()}
        actual,anchors = _assembly(turned,1,3,3)
        assert actual == {_rotate(p,turns):c for p,c in fork.items()}
        assert anchors == {_rotate(p,turns) for p in fa}


def self_test():
    """Bounded 42-seed survey; returns compact, pixel-derived coverage metrics."""
    signatures,shapes,outputs,templates,palettes,directions,counts,sizes = (set() for _ in range(8))
    combinations_seen = set()
    branch_merges = changing_fills = reused_frames = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp = pair['input']
                _,info = _interpret(inp)
                shapes.add(inp.shape)
                outputs.add(pair['output'].shape)
                palettes.add(tuple(map(int,np.unique(inp))))
                changing_fills += info['reference']['fill'] != info['target']['fill']
                reused_frames += any(p['frame'] in set().union(*(set(t.values()) for t in info['templates'].values()))
                                     for p in (info['reference'],info['target']))
                for k,t in info['templates'].items():
                    normalized = {k:0,info['active']:1}
                    for p in sorted(t):
                        normalized.setdefault(t[p],len(normalized))
                    templates.add(tuple((r,c,normalized[color]) for (r,c),color in sorted(t.items())))
                    sizes.add(len(t))
                for q in info['queries']:
                    directions.add(q['direction'])
                    counts.add(q['n'])
                    branch_merges += q['branch'] == 2 and q['n'] >= 3
                    combinations_seen.add((len(info['templates']),q['n'],q['branch'],q['direction'],len(info['templates'][q['key']])))
    assert len(signatures) == 42 and len(shapes) >= 20 and len(outputs) >= 15
    assert len(templates) >= 40 and len(sizes) >= 5 and len(combinations_seen) >= 50
    assert directions == set(_D4) and counts == {1,2,3,4}
    assert branch_merges > 0 and changing_fills > 0 and reused_frames > 0
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    # All count combinations, including the minimum evidence budget.
    legal = 0
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(1000+a*10+b,a,b)
            validate_generated_puzzle(p)
            assert len(p['train']) == a and len(p['test']) == b
            legal += 1
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(4,1.0),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    damaged = generate_puzzle(0,2,1)
    damaged['test'][0]['output'][0,0] = (int(damaged['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    _dependency_regressions()
    return dict(seeds=42,pairs=210,distinct_puzzles=len(signatures),input_shapes=len(shapes),
                output_shapes=len(outputs),normalized_templates=len(templates),motif_sizes=sorted(sizes),
                count_lengths=sorted(counts),directions=sorted(directions),joint_profiles=len(combinations_seen),
                branching_merge_queries=branch_merges,different_fill_pairs=changing_fills,
                motif_frame_reuse_pairs=reused_frames,legal_count_combinations=legal,
                verification='generator-owned input reconstruction and invariants')


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
