"""Periodic pulse interference on two scaffold-registered motif states.

No parent generator is imported. Generation paints sampled templates directly;
solve reconstructs templates, scale, pulse keys and their intersections from pixels.
"""
import math
import numpy as np

HOST_SEEDS = (0,891180106,977261244,152403777,84617243,17214900,706117634,300441006,142242956,850446810,746717599,501044857,900187682,392714184,429909423,743965696,908623909,200286415,838928183,358956953,934693703,278337240,822709963,52031242,757594747,847006316,49916005,887215521,354493106,993472094,461746269,22668830,677855399,469709702,194474371,127212013,235061092,511982260,987512126,711897542,338823562,990127287)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), 'invalid dimensions')
    _require(np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9)), 'invalid colors')
    return g.astype(int, copy=True)


def _components(mask):
    remaining = {(int(y), int(x)) for y, x in zip(*np.nonzero(mask))}
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    p = y + dy, x + dx
                    if p in remaining:
                        remaining.remove(p)
                        group.add(p)
                        stack.append(p)
        groups.append(group)
    return groups


def _bounds(points):
    return min(y for y,x in points), max(y for y,x in points), min(x for y,x in points), max(x for y,x in points)


def _sig(mapping):
    return tuple(sorted((y,x,int(c)) for (y,x),c in mapping.items()))


def _key(g):
    return g.shape, g.tobytes()


def _primitive(crop):
    lengths = []
    for a in (crop, crop.T):
        for row in a:
            starts = [0] + (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist() + [len(row)]
            lengths.extend(b-a for a,b in zip(starts, starts[1:]))
    s = math.gcd(*lengths)
    small = crop[::s, ::s]
    _require(np.array_equal(crop, np.repeat(np.repeat(small,s,0),s,1)), 'nonuniform pixel enlargement')
    return s, small


def _hit(pulse, y, x, scale, bounded=True):
    # A beam is counted once even if several of its sites land in this hub.
    axis, line, a, b, terminal = pulse
    cross, along = (y,x) if axis == 'H' else (x,y)
    if not cross <= line < cross + scale:
        return False
    lo, hi = ((min(a, terminal), max(b, terminal)) if terminal is not None and bounded else (-100,100))
    return any(lo <= p <= hi and (p-a) % (b-a) == 0 for p in range(along, along+scale))


def _read(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(counts.max()*2 > g.size, 'background must be a strict majority')
    objects, markers = [], {}
    for group in _components(g != bg):
        if len(group) <= 4:
            markers.update({p:int(g[p]) for p in group})
            continue
        _require(len(group) >= 6, 'ambiguous five-cell group')
        y0,y1,x0,x1 = _bounds(group)
        crop = g[y0:y1+1,x0:x1+1]
        _require(np.count_nonzero(crop != bg) == len(group), 'interleaved groups')
        s, small = _primitive(crop)
        raw = {(int(y),int(x)):int(small[y,x]) for y,x in zip(*np.nonzero(small != bg))}
        _require(len(raw) >= 6, 'motif needs six logical cells')
        objects.append({'box':(y0,x0), 'scale':s, 'raw':_sig(raw)})
    signatures = sorted({o['raw'] for o in objects})
    _require(len(signatures) == 2, 'exactly two motif states must be visible')
    maps = [{(y,x):c for y,x,c in sig} for sig in signatures]
    shared_colors = set(maps[0].values()) & set(maps[1].values())
    hubs = [c for c in shared_colors if all(list(m.values()).count(c)==1 for m in maps)]
    _require(len(hubs) == 1, 'unique shared one-cell hub required')
    hub_color = hubs[0]
    origins = [next(p for p,c in m.items() if c==hub_color) for m in maps]
    templates = [{(y-oy,x-ox):c for (y,x),c in m.items()} for m,(oy,ox) in zip(maps,origins)]
    common = {p:c for p,c in templates[0].items() if templates[1].get(p)==c}
    _require(len(common) >= 4, 'insufficient invariant scaffold')
    for c in shared_colors:
        _require({p for p,v in templates[0].items() if v==c} == {p for p,v in templates[1].items() if v==c}, 'moving shared scaffold')
    _require(set(markers.values()).isdisjoint(set(maps[0].values()) | set(maps[1].values())), 'control and motif palettes overlap')
    minimum = min(o['scale'] for o in objects)
    _require({o['raw'] for o in objects if o['scale']==minimum}==set(signatures), 'both states need common smallest scale')
    by_color = {}
    for p,c in markers.items():
        by_color.setdefault(c,[]).append(p)
    pulses, references = [], set()
    for c, points in sorted(by_color.items()):
        if len(points)!=2:
            continue
        p,q = sorted(points)
        if p[0]==q[0]:
            pulse = ('H',p[0],p[1],q[1],None)
        elif p[1]==q[1]:
            pulse = ('V',p[1],p[0],q[0],None)
        else:
            continue
        pulses.append(pulse)
        references.update(points)
    _require(1 <= len(pulses) <= 3, 'one to three uniquely paired pulse keys required')
    used = set(references)
    completed = []
    for pulse in pulses:
        axis,line,a,b,_ = pulse
        extras = []
        for (y,x),c in markers.items():
            if (y,x) in references:
                continue
            cross,along = (y,x) if axis=='H' else (x,y)
            if cross==line and (along-a)%(b-a)==0:
                _require(along<a or along>b, 'terminal between keys')
                extras.append(((y,x),along))
        _require(len(extras)<=1, 'multiple terminals on one beam')
        t = extras[0][1] if extras else None
        if extras:
            _require(extras[0][0] not in used, 'terminal shared by beams')
            used.add(extras[0][0])
        completed.append((axis,line,a,b,t))
    _require(used==set(markers), 'unassociated control marker')
    pulses = completed
    for pulse in pulses:
        axis,line,a,b,t = pulse
        own = {(line,a),(line,b)} if axis=='H' else {(a,line),(b,line)}
        if t is not None:
            own.add((line,t) if axis=='H' else (t,line))
        _require(all(p in own or (p[0] if axis=='H' else p[1])!=line for p in markers), 'foreign key on beam line')
    union = set(templates[0]) | set(templates[1])
    ymin,ymax,xmin,xmax = _bounds(union)
    rects, occurrences = [], []
    for o in objects:
        state = signatures.index(o['raw'])
        s = o['scale']
        oy,ox = origins[state]
        y,x = o['box'][0]+oy*s, o['box'][1]+ox*s
        rect = (y+ymin*s, x+xmin*s, (ymax-ymin+1)*s, (xmax-xmin+1)*s)
        ry,rx,rh,rw = rect
        _require(0<=ry and ry+rh<=g.shape[0] and 0<=rx and rx+rw<=g.shape[1], 'replacement clipped')
        _require(all(_separate(rect,r) for r in rects), 'replacement boxes collide')
        _require(all(not (ry-1<=py<=ry+rh and rx-1<=px<=rx+rw) for py,px in markers), 'marker touches replacement region')
        rects.append(rect)
        mask = sum((1<<i) for i,p in enumerate(pulses) if _hit(p,y,x,s))
        occurrences.append({'hub':(y,x), 'scale':s, 'state':state, 'mask':mask, 'rect':rect})
    return {'grid':g,'background':bg,'templates':templates,'common':common,'hub_color':hub_color,
            'markers':markers,'pulses':pulses,'objects':occurrences}


def _paint(g, template, y, x, scale):
    for (dy,dx),c in template.items():
        yy,xx = y+dy*scale, x+dx*scale
        g[yy:yy+scale,xx:xx+scale] = c


def _render(info, mode='parity'):
    out = info['grid'].copy()
    union = set(info['templates'][0]) | set(info['templates'][1])
    for o in info['objects']:
        count = o['mask'].bit_count()
        if mode=='all':
            toggle = True
        elif mode=='union':
            toggle = count>0
        elif mode=='unbounded':
            toggle = sum(_hit(p,*o['hub'],o['scale'],False) for p in info['pulses'])%2==1
        elif mode=='recolor':
            toggle = count%2==1
        else:
            toggle = count%2==1
        if not toggle:
            continue
        y,x = o['hub']
        s = o['scale']
        if mode=='recolor':
            old,new = info['templates'][o['state']],info['templates'][1-o['state']]
            for p in set(old)&set(new):
                _paint(out,{p:new[p]},y,x,s)
        else:
            _paint(out,{p:info['background'] for p in union},y,x,s)
            _paint(out,info['templates'][1-o['state']],y,x,s)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_read(input_grid))


def _separate(a,b):
    y,x,h,w = a
    yy,xx,hh,ww = b
    return y+h<yy or yy+hh<y or x+w<xx or xx+ww<x


def _design(rng, change, color_pool):
    # Procedural small connected scaffold; one distinctive colored hub.
    core = {(0,0),(-1,0),(1,0),(0,-1),(0,1)}
    if rng.random()<.65:
        side = int(rng.integers(4))
        p = [(-1,-1),(-1,1),(1,-1),(1,1)][side]
        core.add(p)
    if rng.random()<.4:
        core.add([(-2,0),(2,0),(0,-2),(0,2)][int(rng.integers(4))])
    carrier,hub,c0,c1 = color_pool[:4]
    common = {p:carrier for p in core}
    common[(0,0)] = hub
    interior = [(-1,0),(1,0),(0,-1),(0,1)][int(rng.integers(4))]
    sides = [(-1,0),(1,0),(0,-1),(0,1)]
    side0 = int(rng.integers(4))
    side1 = side0 if change=='recolor' else (side0+int(rng.integers(1,4)))%4
    def patch(side):
        dy,dx = sides[side]
        root = max(core, key=lambda p:p[0]*dy+p[1]*dx)
        p = root[0]+dy,root[1]+dx
        cells = {p}
        if rng.random()<.5:
            cells.add((p[0]+dx,p[1]-dy))
        return cells
    a = patch(side0)
    b = a if change=='recolor' else patch(side1)
    maps = []
    for state,patch_points in enumerate((a,b)):
        m = dict(common)
        accent = c0 if state==0 else c1
        for p in patch_points:
            m[p]=accent
        if change in ('combined','recolor'):
            m[interior]=accent
        if len(color_pool)>4+state and len(patch_points)>1 and rng.random()<.6:
            m[max(patch_points)]=color_pool[4+state]
        maps.append(m)
    return maps


def _pulses(rng, profile, h, w):
    if profile=='stop':
        d=int(rng.integers(3,6))
        a=int(rng.integers(0,3))
        b=a+d
        feasible=[a+k*d for k in (3,4) if a+(k+2)*d<w]
        if not feasible:
            return None
        return [('H',int(rng.integers(7,h-7)),a,b,int(rng.choice(feasible)))]
    triple = profile=='triple'
    d=int(rng.integers(2,5))
    if triple or profile=='large':
        a=w-1-d
        vline=int(rng.integers(13,min(19,w-7)))
    else:
        a=int(rng.integers(10,15))
        choices=list(range(a+2*d+1,w-4))
        if not choices:
            return None
        vline=int(rng.choice(choices))
    b=a+d
    av=int(rng.integers(1,5)); dv=int(rng.integers(2,5)); bv=av+dv
    hline=int(rng.integers(max(10,bv+4),min(17,h-8)))
    pulses=[('H',hline,a,b,None),('V',vline,av,bv,None)]
    if profile=='mixed':
        terminals=[t for t in range(bv+dv,h-6,dv) if t>hline+3]
        if not terminals:
            return None
        pulses[1]=('V',vline,av,bv,terminals[-1])
    if triple:
        aa=w-5
        bb=w-1
        pulses.append(('H',hline+1,aa,bb,None))
    return pulses


def _build(rng, profile, change):
    h,w = int(rng.integers(27,31)),int(rng.integers(27,31))
    pulses=_pulses(rng,profile,h,w)
    if pulses is None:
        return None
    palette=[int(c) for c in rng.permutation(10)]
    bg=palette[0]
    refs=palette[1:1+len(pulses)]
    terminal_color=palette[1+len(pulses)]
    pool=palette[2+len(pulses):]
    templates=_design(rng,change,pool)
    union=set(templates[0])|set(templates[1])
    ymin,ymax,xmin,xmax=_bounds(union)
    markers={}
    for (axis,line,a,b,t),c in zip(pulses,refs):
        for pos in (a,b):
            p=(line,pos) if axis=='H' else (pos,line)
            if p in markers:
                return None
            markers[p]=c
        if t is not None:
            markers[(line,t) if axis=='H' else (t,line)]=terminal_color
    if any((p not in ({(line,a),(line,b)}|({(line,t)} if t is not None else set()))) and p[0]==line
           for axis,line,a,b,t in pulses if axis=='H' for p in markers):
        return None
    # One enlarged odd-hit motif plus both swap directions, an even-hit motif,
    # and an untouched motif. Stop witnesses instead separate extent and phase.
    if profile=='stop':
        schedule=[(1,1,'inside'),(1,0,'beyond'),(1,0,'offphase')]
    elif profile=='triple':
        schedule=[(3,7,None),(1,1,None),(1,2,None),(1,0,None)]
    else:
        schedule=[(3 if profile=='large' else 2,1,'left'),(1,2,'right'),(1,3,None),(1,0,None)]
    if profile not in ('stop','triple') and rng.random()<.35:
        schedule.append((1,int(rng.choice([0,1,2])),None))
    if profile=='mixed':
        schedule=schedule[:4]+[(1,0,'beyond_v')]
    lists=[]
    for s,target,kind in schedule:
        candidates=[]
        for y in range(-ymin*s,h-(ymax+1)*s+1):
            for x in range(-xmin*s,w-(xmax+1)*s+1):
                rect=(y+ymin*s,x+xmin*s,(ymax-ymin+1)*s,(xmax-xmin+1)*s)
                ry,rx,rh,rw=rect
                if any(ry-1<=py<=ry+rh and rx-1<=px<=rx+rw for py,px in markers):
                    continue
                mask=sum(1<<i for i,p in enumerate(pulses) if _hit(p,y,x,s))
                if mask!=target:
                    continue
                if kind=='left' and x+s-1>=pulses[0][2]:
                    continue
                if kind=='right' and y<=pulses[1][3]:
                    continue
                if kind in ('beyond','offphase'):
                    axis,line,a,b,t=pulses[0]
                    if not y<=line<y+s:
                        continue
                    full=_hit((axis,line,a,b,None),y,x,s)
                    if kind=='beyond' and (not full or x<=t):
                        continue
                    if kind=='offphase' and full:
                        continue
                if kind=='beyond_v':
                    p=pulses[1]
                    if not _hit(p,y,x,s,False) or y<=p[4]:
                        continue
                candidates.append((y,x,s,rect))
        if not candidates:
            return None
        rng.shuffle(candidates)
        lists.append(candidates)
    order=sorted(range(len(lists)),key=lambda i:len(lists[i]))
    chosen=[None]*len(lists)
    budget=[4000]
    def pack(depth):
        if depth==len(order):
            return True
        i=order[depth]
        for item in lists[i]:
            budget[0]-=1
            if budget[0]<0:
                return False
            if all(other is None or _separate(item[3],other[3]) for other in chosen):
                chosen[i]=item
                if pack(depth+1):
                    return True
                chosen[i]=None
        return False
    if not pack(0):
        return None
    inp=np.full((h,w),bg,dtype=int)
    for p,c in markers.items():
        inp[p]=c
    out=inp.copy()
    states=[0,1]+[int(rng.integers(2)) for _ in chosen[2:]]
    for i,((y,x,s,rect),state) in enumerate(zip(chosen,states)):
        _paint(inp,templates[state],y,x,s)
        flip=schedule[i][1].bit_count()%2
        _paint(out,templates[state^flip],y,x,s)
    # Crop only unused margins; retain the full replacement union of every motif.
    occupied=list(markers)
    for y,x,s,(ry,rx,rh,rw) in chosen:
        occupied.extend([(ry,rx),(ry+rh-1,rx+rw-1)])
    y0,y1,x0,x1=_bounds(occupied)
    y0=max(0,y0-int(rng.integers(3))); x0=max(0,x0-int(rng.integers(3)))
    y1=min(h-1,y1+int(rng.integers(3))); x1=min(w-1,x1+int(rng.integers(3)))
    inp,out=inp[y0:y1+1,x0:x1+1],out[y0:y1+1,x0:x1+1]
    k=int(rng.integers(4))
    inp,out=np.rot90(inp,k),np.rot90(out,k)
    if rng.random()<.5:
        inp,out=np.fliplr(inp),np.fliplr(out)
    pair={'input':inp.copy(),'output':out.copy()}
    try:
        info=_read(inp)
        if not np.array_equal(_render(info),out):
            return None
    except ValueError:
        return None
    return pair


def _count(name, value, lo, hi):
    _require(not isinstance(value,(bool,np.bool_)) and isinstance(value,(int,np.integer)) and lo<=value<=hi, name+' is illegal')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt=_count('num_train',num_train,2,10)
    ns=_count('num_test',num_test,1,4)
    _require(nt>ns,'num_train must exceed num_test')
    rng=np.random.default_rng(seed)
    puzzle={'train':[],'test':[]}; seen=set()
    for split,n in (('train',nt),('test',ns)):
        for i in range(n):
            if split=='train' and i==0:
                profile,change='parity','recolor'
            elif split=='train' and i==1:
                profile,change='stop','relocate'
            elif split=='test':
                profile=str(rng.choice(['large','triple','mixed','stop'],p=[.30,.25,.30,.15]))
                change='combined'
            else:
                profile=str(rng.choice(['parity','stop','triple','mixed']))
                change=str(rng.choice(['recolor','relocate','combined']))
            for attempt in range(300):
                pair=_build(rng,profile,change)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input'])); puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded packing exhausted for '+profile)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair,dict) and set(pair)=={'input','output'}, 'invalid pair schema')
    info=_read(pair['input']); out=_grid(pair['output'])
    _require(out.shape==info['grid'].shape and np.array_equal(out,_render(info)), 'incorrect motif exchange')
    _require(not np.array_equal(out,info['grid']), 'identity example')
    # Exact equality above checks all background and erasure; these checks explain
    # the family-specific invariants explicitly, rather than only solver agreement.
    _require(all(out[p]==c for p,c in info['markers'].items()), 'keys changed')
    for o in info['objects']:
        y,x=o['hub']; s=o['scale']; state=o['state']^(o['mask'].bit_count()%2)
        for (dy,dx),c in info['templates'][state].items():
            _require(np.all(out[y+dy*s:y+(dy+1)*s,x+dx*s:x+(dx+1)*s]==c), 'scale or registration broken')
        for (dy,dx),c in info['common'].items():
            _require(np.all(out[y+dy*s:y+(dy+1)*s,x+dx*s:x+(dx+1)*s]==c), 'scaffold changed')
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle)=={'train','test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s],list) for s in puzzle), 'splits must be lists')
    nt=_count('training count',len(puzzle['train']),2,10)
    ns=_count('test count',len(puzzle['test']),1,4)
    _require(nt>ns,'invalid count relation')
    seen=set(); infos={'train':[],'test':[]}
    for split in infos:
        for pair in puzzle[split]:
            info=_inspect_pair(pair)
            k=_key(info['grid']); _require(k not in seen,'duplicate input'); seen.add(k)
            infos[split].append(info)
    train=infos['train']
    _require(any(any(o['mask'].bit_count()==2 for o in f['objects']) and not np.array_equal(_render(f,'union'),_render(f)) for f in train), 'missing interference witness')
    _require(any(not np.array_equal(_render(f,'all'),_render(f)) for f in train), 'missing selection witness')
    _require(any(not np.array_equal(_render(f,'unbounded'),_render(f)) for f in train), 'missing terminal stopping witness')
    _require(any(not np.array_equal(_render(f,'recolor'),_render(f)) for f in train), 'missing attachment relocation witness')
    flipped_states={o['state'] for f in train for o in f['objects'] if o['mask'].bit_count()%2}
    _require(flipped_states=={0,1},'both exchange directions must be demonstrated')
    _require(any(o['scale']>1 and o['mask'].bit_count()%2 for f in train for o in f['objects']), 'enlarged exchange not demonstrated')
    _require(any(any(o['mask']==0 for o in f['objects']) for f in train),'untouched motif not demonstrated')
    _require(any(len(f['pulses'])==1 and any(p[4] is not None for p in f['pulses']) and
                 any(not _hit(p,*o['hub'],o['scale'],False) and
                     ((o['hub'][0]<=p[1]<o['hub'][0]+o['scale']) if p[0]=='H' else (o['hub'][1]<=p[1]<o['hub'][1]+o['scale']))
                     for p in f['pulses'] for o in f['objects']) for f in train),'off-phase preservation not demonstrated')
    return None


def self_test():
    # Hand-constructed reception regressions, with labels painted without _read
    # or the hit function: one beam toggles a large hub once, not once per pixel.
    for period in (1,2):
        a={(0,0):7,(-1,0):4,(1,0):3,(0,-1):3,(0,1):3,(0,2):4}
        b={(0,0):7,(-1,0):8,(1,0):3,(0,-1):3,(0,1):3,(-2,0):8}
        inp=np.zeros((20,30),dtype=int)
        inp[8,1]=inp[8,1+period]=1
        inp[1,19]=inp[3,19]=2
        out=inp.copy()
        for y,x,s,state,flip in ((8,12,2,0,1),(8,25,1,0,1),(16,19,1,1,0)):
            _paint(inp,(a,b)[state],y,x,s)
            _paint(out,(a,b)[state^flip],y,x,s)
        _require(np.array_equal(solve(inp),out),'area reception witness failed')
        f=_read(inp)
        large=next(o for o in f['objects'] if o['scale']==2)
        if period==1:
            _require(large['mask'].bit_count()==1,'one beam was counted twice')
        else:
            _require(not any(_hit(p,*large['hub'],1) for p in f['pulses']), 'top-left shortcut witness failed')
    shapes=set(); designs=set(); scales=set(); periods=set(); counts=set(); palettes=set(); masks=set(); fingerprints=set(); axes=set(); terminals=set()
    for seed in HOST_SEEDS:
        p=generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprint=tuple(_key(_grid(pair[r])) for split in p for pair in p[split] for r in ('input','output'))
        fingerprints.add(fingerprint)
        if seed in (HOST_SEEDS[0],HOST_SEEDS[-1]):
            again=generate_puzzle(seed)
            _require(fingerprint==tuple(_key(_grid(pair[r])) for split in again for pair in again[split] for r in ('input','output')),'nondeterminism')
        for split in p:
            for pair in p[split]:
                f=_read(pair['input'])
                _require(np.array_equal(solve(pair['input']),pair['output']),'input-only solve failure')
                shapes.add(pair['input'].shape)
                # Remove literal hues, but keep registered geometry and state differences.
                role={}; normalized=[]
                for t in f['templates']:
                    normalized.append(tuple((y,x,role.setdefault(c,len(role))) for y,x,c in _sig(t)))
                designs.add(tuple(normalized)); palettes.add(tuple(sorted(np.unique(pair['input']))))
                counts.add(len(f['objects']))
                for o in f['objects']:
                    scales.add(o['scale']); masks.add(o['mask'].bit_count())
                for axis,line,a,b,t in f['pulses']:
                    axes.add(axis); periods.add(b-a)
                    terminals.add(0 if t is None else (-1 if t<a else 1))
    _require(len(fingerprints)==42 and len(shapes)>=25 and len(designs)>=40, 'insufficient structural variation')
    _require(scales>={1,2,3} and masks>={0,1,2,3} and axes=={'H','V'}, 'missing scale/interference/orientation variation')
    _require(len(periods)>=4 and len(counts)>=3 and len(palettes)>=20 and terminals=={-1,0,1}, 'missing cadence/count/palette/bounds variation')
    # The external verifier enumerates every legal combination. Bound the local
    # self-test with the minimum and maximum count regimes instead.
    for a,b in ((2,1),(5,4),(10,1),(10,4)):
        q=generate_puzzle(73,a,b)
        _require(len(q['train'])==a and len(q['test'])==b,'explicit count failure')
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p=generate_puzzle(0,2,1)
    bad={s:[{r:_grid(v) for r,v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0,0]=(bad['test'][0]['output'][0,0]+1)%10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    # Geometry and color symmetries must commute with the transformation.
    for pair in p['train']+p['test']:
        for k in range(4):
            inp,out=np.rot90(pair['input'],k),np.rot90(pair['output'],k)
            _require(np.array_equal(solve(9-inp),9-out),'color/rotation equivariance failed')
    return {'seeds':42,'shapes':len(shapes),'registered_designs':len(designs),'scales':sorted(scales),
            'periods':sorted(periods),'motif_counts':sorted(counts),'hit_counts':sorted(masks),'palettes':len(palettes)}
