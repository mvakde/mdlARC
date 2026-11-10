"""Phase-gated beams: incomplete frames launch arithmetic pulse progressions.

Only coincident *painted* cells cause a spatial encounter. A turn transports
each strand's cadence/phase, and bundles add headings without color mixing.
No parent module, external data, or generation state is used by solve().
"""
from __future__ import annotations

from math import gcd
import numpy as np


PORTS = {'N0': (-1, 0), 'N1': (-1, 1), 'S0': (2, 0), 'S1': (2, 1),
         'W0': (0, -1), 'W1': (1, -1), 'E0': (0, 2), 'E1': (1, 2)}
RING = [(r, c) for r in range(-1, 3) for c in range(-1, 3)
        if r in (-1, 2) or c in (-1, 2)]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(point, shape):
    return all(0 <= x < n for x, n in zip(point, shape))


def _add(p, d, k=1):
    return p[0] + k*d[0], p[1] + k*d[1]


def _direction(offset):
    r, c = offset
    return (-1 if r == -1 else 1 if r == 2 else 0,
            -1 if c == -1 else 1 if c == 2 else 0)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), 'invalid grid dimensions')
    _require(np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9)), 'invalid cells')
    return g.astype(np.int8, copy=True)


def _parse(grid):
    colors, counts = np.unique(grid, return_counts=True)
    _require(counts.max() > grid.size/2, 'background must be a strict majority')
    bg = int(colors[counts.argmax()])
    launchers = []
    for r in range(grid.shape[0]-1):
        for c in range(grid.shape[1]-1):
            payload = int(grid[r, c])
            if payload == bg or not np.all(grid[r:r+2, c:c+2] == payload):
                continue
            ring = [(_add((r, c), o), o) for o in RING if _inside(_add((r, c), o), grid.shape)]
            hues = {int(grid[p]) for p, _ in ring if grid[p] != bg}
            gaps = [(p, o) for p, o in ring if grid[p] == bg]
            if len(hues) != 1 or payload in hues or len(ring)-len(gaps) < 4 or not gaps:
                continue
            if any(o not in PORTS.values() for _, o in gaps):
                continue
            launchers.append({'core': (r, c), 'color': payload, 'ring': ring,
                              'structure': next(iter(hues)), 'ports': gaps})
    _require(1 <= len(launchers) <= 6, 'need one to six identifiable launchers')
    _require(len({l['structure'] for l in launchers}) == 1, 'one common ring color required')
    _require(len({l['color'] for l in launchers}) == len(launchers), 'payloads must be distinct')
    footprints, fixed, rays = set(), set(), []
    for li, launcher in enumerate(launchers):
        r, c = launcher['core']
        fp = { (rr, cc) for rr in range(r-1, r+3) for cc in range(c-1, c+3)
               if _inside((rr, cc), grid.shape)}
        _require(not (fp & footprints), 'overlapping launcher footprints')
        footprints |= fp
        fixed |= {p for p in fp if grid[p] != bg}
        for p, offset in launcher['ports']:
            rays.append({'launcher': li, 'origin': p, 'v': _direction(offset), 'color': launcher['color']})
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != bg)} - fixed
    assigned = set()
    for ray in rays:
        beads = []
        for t in range(1, 31):
            p = _add(ray['origin'], ray['v'], t)
            if not _inside(p, grid.shape):
                break
            if p in remaining and int(grid[p]) == ray['color']:
                beads.append((t, p))
        _require(len(beads) == 2, 'each port needs exactly two outward payload beads')
        _require(not (set(p for _, p in beads) & assigned), 'ambiguous bead ownership')
        assigned |= {p for _, p in beads}
        ray.update(a=beads[0][0], p=beads[1][0]-beads[0][0],
                   beads={p for _, p in beads}, second=beads[1][0])
    _require(assigned == remaining, 'unaccounted foreground')
    return {'bg': bg, 'launchers': launchers, 'rays': rays, 'fixed': fixed,
            'footprints': footprints}


def _walk(point, v, shape, raster='floor'):
    """Unit raster steps, major-axis first for a 2:1 heading."""
    major = max(abs(v[0]), abs(v[1]))
    if raster == 'sign':
        v = tuple(int(np.sign(x)) for x in v)
        major = 1
    points = []
    for t in range(3*max(shape)+1):
        offset = major-1 if raster == 'ceil' else 0
        p = (point[0] + int(np.sign(v[0]))*((t*abs(v[0])+offset)//major),
             point[1] + int(np.sign(v[1]))*((t*abs(v[1])+offset)//major))
        if not _inside(p, shape):
            break
        points.append(p)
    return points


def _render(grid, info, *, gate=True, turns=True, reset=False, solid=False,
            raster='floor', weighted=False):
    """Spatial routing; cumulative step counts belong to strands, not a clock.

    Alternative switches are used only as counterfactual validation witnesses.
    """
    rays, bg = info['rays'], info['bg']
    groups = [{'v': ray['v'], 'members': [(i, ray['origin'], 0)], 'depth': 0}
              for i, ray in enumerate(rays)]
    out, occupied, visited = grid.copy(), {}, set()
    events, misses = [], set()

    def active(s, travelled, t):
        ray = rays[s]
        return solid or (travelled+t-ray['a']) % ray['p'] == 0

    def paint(s, travelled, path):
        ray = rays[s]
        for k, point in enumerate(path):
            _require(point not in info['fixed'], 'path goes through a core or ring')
            if grid[point] != bg:
                _require(point in ray['beads'], 'path goes through another beam seed')
            if point in ray['beads']:
                visited.add(point)
            if not active(s, travelled, k):
                continue
            _require(point not in occupied or (k == 0 and occupied[point] == s),
                     'painted strand overlap or a crossing with a past trail')
            occupied[point] = s
            out[point] = ray['color']

    for _ in range(len(rays)+1):
        forecasts = [[_walk(p, g['v'], grid.shape, raster) for _, p, _ in g['members']] for g in groups]
        candidates = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                hits = []
                for ai, ((sa, _, ta), pa) in enumerate(zip(groups[i]['members'], forecasts[i])):
                    lookup = {p:k for k,p in enumerate(pa)}
                    for bi, ((sb, _, tb), pb) in enumerate(zip(groups[j]['members'], forecasts[j])):
                        for kb, p in enumerate(pb):
                            if p not in lookup:
                                continue
                            ka = lookup[p]
                            aa, ab = active(sa,ta,ka), active(sb,tb,kb)
                            if not (aa and ab):
                                misses.add((sa,sb,p))
                            if gate and not (aa and ab):
                                continue
                            hits.append((ka-1,kb-1,ai,bi,p))
                if hits and turns:
                    hit = min(hits, key=lambda h:(h[0]+h[1],max(h[:2]),h[2:4]))
                    _require(hit[0] == min(h[0] for h in hits) and hit[1] == min(h[1] for h in hits),
                             'ambiguous encounter order within a pair of bundles')
                    candidates.append((i,j,hit))
        if not candidates:
            for g, paths in zip(groups, forecasts):
                for (s,_,t), path in zip(g['members'], paths):
                    paint(s,t,path)
            break
        nearest = {}
        for i,j,h in candidates:
            nearest[i] = min(nearest.get(i,1000),h[0])
            nearest[j] = min(nearest.get(j,1000),h[1])
        ready = [(i,j,h) for i,j,h in candidates if h[0] == nearest[i] and h[1] == nearest[j]]
        _require(bool(ready), 'cyclic encounter order')
        i,j,hit = min(ready, key=lambda x:(max(x[2][:2]),sum(x[2][:2]),x[:2]))
        for k,l,h in candidates:
            if (k,l) != (i,j):
                _require(not ((k in (i,j) and h[0] == nearest[k]) or
                              (l in (i,j) and h[1] == nearest[l])), 'unresolved multiway encounter')
        u,v = groups[i]['v'],groups[j]['v']
        _require(max(map(abs,u)) == max(map(abs,v)) == 1, 'staircases do not encounter further beams')
        _require(u != v and u[0]*v[0]+u[1]*v[1] >= 0, 'parallel or cancelling encounter')
        wa,wb = (len(groups[i]['members']),len(groups[j]['members'])) if weighted else (1,1)
        summed = (wa*u[0]+wb*v[0],wa*u[1]+wb*v[1])
        divisor = gcd(abs(summed[0]),abs(summed[1]))
        _require(divisor > 0, 'head-on cancellation')
        heading = (summed[0]//divisor,summed[1]//divisor)
        members = []
        incoming_distances = []
        for gi,cut in ((i,hit[0]),(j,hit[1])):
            _require(cut >= 1, 'encounter at a strand origin')
            g = groups[gi]
            for (s,p,t),path in zip(g['members'],forecasts[gi]):
                _require(cut < len(path), 'bundle partially leaves before encounter')
                _require(g['depth'] or cut >= rays[s]['second'], 'encounter precedes second bead')
                paint(s,t,path[:cut+1])
                new_t = rays[s]['a'] if reset else t+cut
                members.append((s,path[cut],new_t))
            member = hit[2] if gi == i else hit[3]
            incoming_distances.append(g['members'][member][2]+cut+1)
        depth = 1+max(groups[i]['depth'],groups[j]['depth'])
        events.append({'position':hit[4], 'incoming':(u,v), 'outgoing':heading,
                       'depth':depth, 'distances':tuple(incoming_distances),
                       'strands':tuple(s for s,_,_ in members)})
        groups = [g for k,g in enumerate(groups) if k not in (i,j)]
        groups.append({'v':heading,'members':members,'depth':depth})
    else:
        raise ValueError('routing failed to terminate')
    _require(visited == set().union(*(r['beads'] for r in rays)), 'a bend bypasses an input bead')
    return out, {'events':events,'misses':len(misses),'occupied':occupied}


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _render(grid,_parse(grid))[0]


def _at_port(point, name, color, a, period):
    offset = PORTS[name]
    return {'core':_add(point,offset,-1),'color':color,'ports':{name:(a,period)}}


def _build(shape, specs):
    _require(all(1 <= n <= 30 for n in shape), 'canvas outside ARC size limit')
    g = np.zeros(shape,dtype=np.int8)
    used = set()
    for spec in specs:
        r,c = spec['core']
        _require(0 <= r < shape[0]-1 and 0 <= c < shape[1]-1,'core outside canvas')
        fp = {(rr,cc) for rr in range(r-1,r+3) for cc in range(c-1,c+3) if _inside((rr,cc),shape)}
        _require(not fp & used,'overlapping frames')
        used |= fp
        g[r:r+2,c:c+2] = spec['color']
        for o in RING:
            p = _add((r,c),o)
            if _inside(p,shape):
                g[p] = 8
        for name in spec['ports']:
            p = _add((r,c),PORTS[name])
            _require(_inside(p,shape),'port clipped away')
            g[p] = 0
    for spec in specs:
        for name,(a,period) in spec['ports'].items():
            origin = _add(spec['core'],PORTS[name])
            v = _direction(PORTS[name])
            for t in (a,a+period):
                p = _add(origin,v,t)
                _require(_inside(p,shape) and p not in used and g[p] == 0,'seed bead obstructed')
                g[p] = spec['color']
    return g


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _trial(shape,specs):
    g = _build(shape,specs)
    info = _parse(g)
    out,metrics = _render(g,info)
    return g,out,info,metrics


def _base(rng, mode, test=False):
    """Sample approach lengths and rhythms before locating the frames."""
    for _ in range(300):
        if mode == 'isolated':
            h,w = (int(x) for x in rng.integers(10,25,2))
            core = (int(rng.integers(h-2)),int(rng.integers(w-2)))
            names = [str(_pick(rng,list(PORTS)))]
            spec = {'core':core,'color':1,'ports':{}}
            for name in names:
                p = int(rng.integers(1,5))
                spec['ports'][name] = (int(rng.integers(1,p+1)),p)
            specs = [spec]
        else:
            chain = mode == 'chain'
            p1,p2 = (int(x) for x in rng.integers(2,5,2))
            a1,a2 = int(rng.integers(1,p1+1)),int(rng.integers(1,p2+1))
            ds1 = [d for d in range(a1+p1+2,16) if (d-a1)%p1 == 0]
            ds2 = [d for d in range(a2+p2+2,16) if ((d-a2)%p2 == 0) == (mode != 'miss')]
            if not ds1 or not ds2:
                continue
            # Tests often move the crossing farther from its encoded beads.
            d1 = int(_pick(rng,ds1[len(ds1)//2:] if test else ds1))
            d2 = int(_pick(rng,ds2[len(ds2)//2:] if test else ds2))
            if mode == 'live' and d1 == d2:
                continue  # distinguish spatial phase gates from equal-clock motion
            row = int(rng.integers(9 if chain else 4,15))
            col = d1 + int(rng.integers(2,6))
            h = row+d2+int(rng.integers(3,6))
            w = col+int(rng.integers(5,11))
            if h>30 or w>30:
                continue
            specs = [_at_port((row,col-d1),str(_pick(rng,('E0','E1'))),1,a1,p1),
                     _at_port((row+d2,col),str(_pick(rng,('N0','N1'))),2,a2,p2)]
            if chain:
                # The new north beam skips a lower off-phase strand, then meets
                # the upper strand. The bundle as a whole turns into a staircase.
                ks = [k for k in range(3,row-2) if (k-1)%p1 == 0 and
                      (d2-1+(k-1)-a2)%p2 != 0]
                if not ks:
                    continue
                k = int(_pick(rng,ks))
                target = (row-k,col-1+k)
                p3 = int(rng.integers(2,6))
                a3 = int(rng.integers(1,p3+1))
                lengths = [d for d in range(a3+p3+4,16) if (d-a3)%p3 == 0 and target[0]+d+3<=30]
                if not lengths:
                    continue
                d3 = int(_pick(rng,lengths))
                origin = _add(target,(1,0),d3)
                h = max(h,origin[0]+int(rng.integers(3,6)))
                w = max(w,target[1]+4)
                specs.append(_at_port(origin,str(_pick(rng,('N0','N1'))),3,a3,p3))
        try:
            g,out,info,m = _trial((h,w),specs)
            if np.array_equal(g,out):
                continue
            if mode == 'live' and len(m['events']) != 1:
                continue
            if mode == 'miss' and (m['events'] or not m['misses']):
                continue
            if mode == 'chain' and (len(m['events']) != 2 or max(e['depth'] for e in m['events']) != 2):
                continue
            if mode == 'chain' and not all(_different_counterfactual(g,out,**alternative)
                    for alternative in ({'raster':'sign'},{'raster':'ceil'},{'weighted':True})):
                continue
            return (h,w),specs
        except ValueError:
            continue
    raise RuntimeError('bounded scene construction exhausted for '+mode)


def _augment(rng,shape,specs,chain_allowed):
    """Extra ports belong to the same physical network, never separate panels."""
    result = [{'core':s['core'],'color':s['color'],'ports':dict(s['ports'])} for s in specs]
    for _ in range(10):
        idx = int(rng.integers(len(result)))
        s = result[idx]
        # A second side lane shares its core but has an independently encoded rhythm.
        first = next(iter(s['ports']))
        name = first[0]+str(1-int(first[1]))
        if name in s['ports']:
            continue
        period = int(rng.integers(1,6))
        a = int(rng.integers(1,period+1))
        candidate = [{'core':x['core'],'color':x['color'],'ports':dict(x['ports'])} for x in result]
        candidate[idx]['ports'][name] = (a,period)
        try:
            _,_,_,m = _trial(shape,candidate)
            if not chain_allowed and any(e['depth']>1 for e in m['events']):
                continue
            result = candidate
            if rng.random()<.5:
                break
        except ValueError:
            continue
    if chain_allowed and len(result)>=2 and rng.random()<.65:
        # A new frame targets an actual pulse in the existing network. Its
        # added color must participate, and the resulting graph stays connected.
        for _ in range(int(rng.integers(1,3))):
            base,out,info,before = _trial(shape,result)
            targets = [p for p in before['occupied'] if base[p]==0]
            added = None
            for attempt in range(70):
                if not targets or len(result)>=6:
                    break
                target = _pick(rng,targets)
                name = str(_pick(rng,list(PORTS)))
                v = _direction(PORTS[name])
                period = int(rng.integers(2,6))
                a = int(rng.integers(1,period+1))
                distances = [d for d in range(a+period+2,16) if (d-a)%period==0]
                if not distances:
                    continue
                origin = _add(target,v,-int(_pick(rng,distances)))
                color = next(c for c in range(1,7) if c not in {x['color'] for x in result})
                candidate = result+[_at_port(origin,name,color,a,period)]
                try:
                    _,_,new_info,m = _trial(shape,candidate)
                    if len(m['events'])<=len(before['events']):
                        continue
                    connected = {0}
                    edges = [{new_info['rays'][s]['launcher'] for s in e['strands']} for e in m['events']]
                    for _pass in range(len(candidate)):
                        for edge in edges:
                            if edge & connected:
                                connected |= edge
                    if connected != set(range(len(candidate))):
                        continue
                    added = candidate
                    break
                except ValueError:
                    continue
            if added is None:
                break
            result = added
    return result


def _key(g):
    return g.shape,g.tobytes()


def _counts(num_train,num_test):
    for name,n,lo,hi in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or not lo<=n<=hi:
            raise ValueError(name+' must be an integer in '+str((lo,hi)))
    if num_train<=num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train),int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    use_chain = nt>=3 and bool(rng.random()<.65)
    base_palette = rng.permutation(10)
    modes = ['live','miss']+(['chain'] if use_chain else [])
    while len(modes)<nt:
        modes.append(str(_pick(rng,['live','miss','isolated']+(['chain'] if use_chain else []))))
    rng.shuffle(modes)
    protected = {modes.index('live'),modes.index('miss')}
    if use_chain:
        protected.add(modes.index('chain'))
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',nt),('test',ne)):
        for index in range(count):
            mode = modes[index] if split=='train' else str(_pick(rng,['live','live','miss']+(['chain','chain'] if use_chain else [])))
            for attempt in range(30):
                shape,specs = _base(rng,mode,split=='test')
                # Leave guaranteed live/miss witnesses readable. Other examples
                # can combine multiple independently paced ports on each frame.
                if mode != 'miss' and (split=='test' or index not in protected) and rng.random()<.7:
                    specs = _augment(rng,shape,specs,use_chain)
                g,out,_,_ = _trial(shape,specs)
                lut = base_palette if rng.random()<.4 else rng.permutation(10)
                g,out = lut[g].astype(np.int8),lut[out].astype(np.int8)
                rotation = int(rng.integers(4))
                g,out = np.rot90(g,rotation),np.rot90(out,rotation)
                if rng.random()<.5:
                    g,out = np.fliplr(g),np.fliplr(out)
                g,out = np.ascontiguousarray(g),np.ascontiguousarray(out)
                if _key(g) in seen:
                    continue
                # Verify the returned orientation, not pre-transform metadata.
                try:
                    expected,m = _render(g,_parse(g))
                    _require(np.array_equal(expected,out),'transform altered raster phase')
                except ValueError:
                    continue
                puzzle[split].append({'input':g,'output':out})
                seen.add(_key(g))
                break
            else:
                raise RuntimeError('unique pair construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _different_counterfactual(g,out,**kwargs):
    try:
        altered,_ = _render(g,_parse(g),**kwargs)
        return not np.array_equal(altered,out)
    except ValueError:
        # A counterfactual that creates an overlap contradicts the output too.
        return True


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen = set()
    evidence = {'hit':False,'miss':False,'cadence':False,'phase_gate':False,
                'routing':False,'phase_carry':False,'unequal_distance':False}
    staircase_evidence = {'sum':False,'raster':False,'bundle_weight':False}
    train_headings,needed = set(),set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            g,out = _grid(pair['input']),_grid(pair['output'])
            assert g.shape==out.shape and _key(g) not in seen
            seen.add(_key(g))
            info = _parse(g)
            expected,m = _render(g,info)
            assert np.array_equal(expected,out) and np.any(g!=out)
            assert np.array_equal(out[g!=info['bg']],g[g!=info['bg']]), 'frames/beads changed'
            # Every new cell is a pulse of an identified strand, with its own hue.
            changes = {tuple(map(int,p)) for p in np.argwhere(g!=out)}
            assert changes <= set(m['occupied'])
            for e in m['events']:
                assert len({info['rays'][s]['color'] for s in e['strands']})>=2
            tags = {'staircase' if max(map(abs,e['outgoing']))==2 else 'diagonal' for e in m['events']}
            if split=='train':
                train_headings |= tags
                evidence['hit'] |= bool(m['events'])
                evidence['unequal_distance'] |= any(e['distances'][0]!=e['distances'][1] for e in m['events'])
                evidence['miss'] |= bool(m['misses']) and not m['events']
                evidence['cadence'] |= _different_counterfactual(g,out,solid=True)
                evidence['phase_gate'] |= not m['events'] and _different_counterfactual(g,out,gate=False)
                evidence['routing'] |= _different_counterfactual(g,out,turns=False)
                evidence['phase_carry'] |= bool(m['events']) and _different_counterfactual(g,out,reset=True)
                if 'staircase' in tags:
                    staircase_evidence['sum'] |= _different_counterfactual(g,out,raster='sign')
                    staircase_evidence['raster'] |= _different_counterfactual(g,out,raster='ceil')
                    staircase_evidence['bundle_weight'] |= _different_counterfactual(g,out,weighted=True)
            else:
                needed |= tags
    assert all(evidence.values()), 'demonstrations lack a required discriminating witness: '+str(evidence)
    assert needed<=train_headings, 'undemonstrated outgoing raster type'
    if 'staircase' in needed:
        assert all(staircase_evidence.values()), 'ambiguous staircase/bundle convention: '+str(staircase_evidence)


def _semantic_fixtures():
    """Small arithmetic fixtures: expected pixels use explicit coordinate laws.

    These do not call the router, encounter detector, or its raster walker to
    construct expected outputs. They are author-written checks, not an external
    semantic reference or a blind prediction audit.
    """
    fixtures = []
    for live in (True,False):
        specs = [_at_port((10,4),'E0',1,2,2),
                 _at_port((18,14),'N0',2,2 if live else 1,2)]
        g = _build((24,24),specs)
        out = g.copy()
        if live:
            for t in range(0,10,2):
                out[10,4+t] = 1
            for t in range(0,8,2):
                out[18-t,14] = 2
            for k in range(1,12,2):
                if 10-k>=0 and 13+k<24:
                    out[10-k,13+k] = 1
                if 11-k>=0 and 14+k<24:
                    out[11-k,14+k] = 2
            assert out[10,14]==0 and out[9,14]==1 and out[10,15]==2
        else:
            for c in range(4,24,2):
                out[10,c] = 1
            for r in range(17,-1,-2):
                out[r,14] = 2
            assert out[10,14]==1 and out[9,14]==2
        fixtures.append((g,out))
        if live:
            # Moving both reference samples one cadence farther out must not
            # shift their progression or remove completion toward the ports.
            shifted = _build((24,24),[_at_port((10,4),'E0',1,4,2),
                                      _at_port((18,14),'N0',2,4,2)])
            fixtures.append((shifted,out.copy()))
    # A legal spacing beyond the sampler's preferred range, plus a second
    # period-one port on the same core: the input-only rule remains general.
    wide = _build((10,30),[{'core':(2,2),'color':1,'ports':{'E0':(3,7),'E1':(1,1)}}])
    wide_out = wide.copy()
    wide_out[2,[7,14,21,28]] = 1
    wide_out[3,4:] = 1
    fixtures.append((wide,wide_out))
    specs = [_at_port((12,2),'E0',1,1,4),
             _at_port((21,15),'N0',2,1,4),
             _at_port((14,19),'N1',3,1,2)]
    g = _build((24,23),specs)
    out = g.copy()
    for t in (1,5,9):
        out[12,2+t] = 1
    for t in (1,5):
        out[21-t,15] = 2
    out[11,15],out[12,16] = 1,2
    for t in (1,3,5):
        out[14-t,19] = 3
    for color,row,col,period in ((1,8,18,4),(2,9,19,4),(3,8,19,2)):
        for k in range(1,30,period):
            if row-k>=0 and col+k//2<23:
                out[row-k,col+k//2] = color
    assert out[7,18]==1 and out[8,19]==2 and out[7,19]==3
    fixtures.append((g,out))
    for g,out in fixtures:
        for rotation in range(4):
            for reflection in (False,True):
                gi,go = np.rot90(g,rotation),np.rot90(out,rotation)
                if reflection:
                    gi,go = np.fliplr(gi),np.fliplr(go)
                assert np.array_equal(solve(gi),go), 'arithmetic routing fixture failed'
                assert np.array_equal(solve(9-gi),9-go), 'palette fixture failed'


def self_test():
    _semantic_fixtures()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures,shapes,layouts,backgrounds,periods,phases,portcounts,counts,headings = (set() for _ in range(9))
    unequal = chains = clipped = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train','test') for p in puzzle[s]))
        for split in ('train','test'):
            for pair in puzzle[split]:
                g = pair['input']
                info = _parse(g)
                out,m = _render(g,info)
                assert np.array_equal(out,pair['output'])
                shapes.add(g.shape)
                backgrounds.add(info['bg'])
                counts.add(len(info['launchers']))
                portcounts.add(len(info['rays']))
                layouts.add((g.shape,tuple((l['core'],tuple(o for _,o in l['ports'])) for l in info['launchers']),
                             tuple((r['a'],r['p']) for r in info['rays'])))
                for r in info['rays']:
                    periods.add(r['p']); phases.add(r['a']%r['p'])
                for e in m['events']:
                    headings.add(e['outgoing'])
                    unequal += e['distances'][0]!=e['distances'][1]
                    chains += e['depth']>1
                clipped += sum(len(l['ring'])<12 for l in info['launchers'])
    assert len(signatures)==42 and len(shapes)>=50 and len(layouts)>=150
    assert len(backgrounds)==10 and {1,2,3,4,5}<=periods and len(phases)>=4
    assert {1,2,3}<=counts and max(portcounts)>=4
    assert unequal and chains and clipped and len(headings)>=8
    assert tuple(_key(p['input']) for p in generate_puzzle(0)['train']) == tuple(_key(p['input']) for p in puzzles[0]['train'])
    # Every legal explicit count, bounded to one seed each.
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+nt*10+ne,nt,ne))
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupt = {s:[{k:v.copy() for k,v in p.items()} for p in puzzles[0][s]] for s in ('train','test')}
    corrupt['test'][0]['output'][0,0] = (int(corrupt['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(corrupt)
    except (AssertionError,ValueError):
        pass
    else:
        raise AssertionError('corrupted label accepted')
    return {'seeds':42,'pairs':210,'shapes':len(shapes),'layouts':len(layouts),
            'periods':sorted(periods),'launcher_counts':sorted(counts),'ray_counts':sorted(portcounts),
            'unequal_encounters':unequal,'later_encounters':chains,'clipped_launchers':clipped}
