"""Ranked socket splicing: ranks identify mates; rims fix translations.

All conventions are visible or fixed. The solver uses only its input pixels.
"""
import hashlib
import json
import numpy as np
from collections import deque


def groups(cells):
    todo = set(cells)
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, group = [p], {p}
        while stack:
            r, c = stack.pop()
            for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def normalize(cells):
    r0 = min(r for r,c in cells)
    c0 = min(c for r,c in cells)
    return {(r-r0,c-c0):v for (r,c),v in cells.items()}


def shape(cells):
    return max(r for r,c in cells)+1,max(c for r,c in cells)+1


def component(rng, kinds, s, ink, paths):
    cells = {}
    terminals = []
    directions = rng.choice(4,len(kinds),replace=False)
    for i,(kind,d) in enumerate(zip(kinds,directions)):
        dr,dc = ((-1,0),(0,1),(1,0),(0,-1))[d]
        length = int(rng.integers(s+1,s+4))
        jitter = int(rng.integers(-2,3))
        cr,cc = dr*length+dc*jitter,dc*length+dr*jitter
        top,left = cr-s//2,cc-s//2
        terminals.append((top,left,kind))
        # Offset the outer arm using a bend, never rotate an existing piece.
        bend = int(rng.integers(1,length))
        p = (0,0)
        waypoints = [(dr*bend,dc*bend),
                     (dr*bend+dc*jitter,dc*bend+dr*jitter),(cr,cc)]
        color = int(paths[i % len(paths)])
        for q in waypoints:
            while p != q:
                cells[p] = color
                p = (p[0]+int(np.sign(q[0]-p[0])),
                     p[1]+int(np.sign(q[1]-p[1])))
            cells[p] = color
    for top,left,kind in terminals:
        for r in range(s):
            for c in range(s):
                p = top+r,left+c
                cells.pop(p,None)
                if kind == 'S' or r in (0,s-1) or c in (0,s-1):
                    cells[p] = ink
    return normalize(cells)


def pack(rng,parts,bg):
    shapes = [shape(p) for p in parts]
    mh,mw = max(h for h,w in shapes),max(w for h,w in shapes)
    area = sum(map(len,parts))
    for attempt in range(16):
        if attempt >= 10:
            h=w=30
        else:
            h = int(rng.integers(max(mh,14),31))
            w = int(rng.integers(max(mw,14),31))
        if area*2 >= h*w:
            continue
        blocked = np.zeros((h,w),dtype=bool)
        grid = np.full((h,w),bg,dtype=np.int8)
        order = sorted(range(len(parts)),key=lambda i:-(shapes[i][0]*shapes[i][1]+rng.random()*20))
        for i in order:
            ph,pw=shapes[i]
            hh,ww=h-ph+1,w-pw+1
            valid = np.ones((hh,ww),dtype=bool)
            for r,c in parts[i]:
                valid &= ~blocked[r:r+hh,c:c+ww]
            coords = np.argwhere(valid)
            if not len(coords):
                break
            r0,c0 = map(int,coords[int(rng.integers(len(coords)))])
            for (r,c),v in parts[i].items():
                r,c=r+r0,c+c0
                grid[r,c]=v
                blocked[max(0,r-1):r+2,max(0,c-1):c+2]=True
        else:
            return grid
    raise ValueError('pack')


def checked_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError('expected a 1..30 by 1..30 integer ARC grid')
    return grid.astype(np.int8, copy=False)


def parse(grid):
    """Find the background, complete fragments, and square registration ports."""
    grid = checked_grid(grid)
    vals,counts=np.unique(grid,return_counts=True)
    bg=int(vals[counts.argmax()])
    if counts.max()*2<=grid.size:
        raise ValueError('background')
    parts=groups(zip(*np.nonzero(grid!=bg)))
    which={p:i for i,part in enumerate(parts) for p in part}
    for (r,c),owner in which.items():
        if any(which.get((r+dr,c+dc),owner)!=owner
               for dr in (-1,0,1) for dc in (-1,0,1)):
            raise ValueError('fragments touch diagonally')
    candidates=[]
    for color in vals:
        if color==bg:
            continue
        tiles=[]
        for region in groups(zip(*np.nonzero(grid==color))):
            t,l=min(r for r,c in region),min(c for r,c in region)
            b,r=max(r for r,c in region)+1,max(c for r,c in region)+1
            s=int(b-t)
            if s!=r-l or not 3<=s<=5:
                break
            full={(t+i,l+j) for i in range(s) for j in range(s)}
            rim={p for p in full if p[0] in (t,b-1) or p[1] in (l,r-1)}
            if region==full:
                kind='S'
            elif region==rim and np.all(grid[t+1:b-1,l+1:r-1]==bg):
                kind='H'
            else:
                break
            tiles.append(dict(top=t,left=l,kind=kind,part=which[t,l],size=s))
        else:
            if len(tiles)>=4 and len({o['size'] for o in tiles})==1 and {o['kind'] for o in tiles}=={'H','S'}:
                candidates.append((int(color),tiles))
    if len(candidates)!=1:
        raise ValueError('tiles')
    ink,tiles=candidates[0]
    s=tiles[0]['size']
    queues=[[o for o in sorted(tiles,key=lambda o:o['top']) if o['kind']==k] for k in 'HS']
    if (not 2<=len(queues[0])<=4 or len(queues[0])!=len(queues[1])
            or len(parts)!=len(queues[0])+1):
        raise ValueError('counts')
    if any(len({o['top'] for o in q})!=len(q) for q in queues):
        raise ValueError('ties')
    if any(not 1<=sum(o['part']==i for o in tiles)<=3 for i in range(len(parts))):
        raise ValueError('degree')
    if any(not any(grid[p]!=ink for p in part) for part in parts):
        raise ValueError('port-only fragment')
    return dict(bg=bg,ink=ink,tiles=tiles,parts=parts,queues=queues,size=s,grid=grid)


def assemble(info,mode='rank'):
    """Propagate rim-coincidence translations through the matching tree.

    Non-default modes are negative controls. Socket interiors are transparent;
    positive material is preserved. Only paired square rims may coincide.
    """
    grid,s=info['grid'],info['size']
    a,b=[list(q) for q in info['queues']]
    if mode=='reverse':
        b.reverse()
    elif mode=='horizontal':
        a.sort(key=lambda o:(o['left'],o['top']))
        b.sort(key=lambda o:(o['left'],o['top']))
    elif mode=='horizontal_reverse':
        a.sort(key=lambda o:(o['left'],o['top']),reverse=True)
        b.sort(key=lambda o:(o['left'],o['top']),reverse=True)
    elif mode!='rank':
        raise ValueError('unknown matching control')
    adjacency=[[] for p in info['parts']]
    for h,t in zip(a,b):
        i,j=h['part'],t['part']
        if i==j:
            raise ValueError('self')
        delta=(h['top']-t['top'],h['left']-t['left'])
        adjacency[i].append((j,delta))
        adjacency[j].append((i,(-delta[0],-delta[1])))
    shifts={0:(0,0)}
    todo=deque([0])
    while todo:
        i=todo.popleft()
        for j,d in adjacency[i]:
            proposed=(shifts[i][0]+d[0],shifts[i][1]+d[1])
            if j in shifts:
                if shifts[j]!=proposed:
                    raise ValueError('cycle')
            else:
                shifts[j]=proposed
                todo.append(j)
    if len(shifts)!=len(info['parts']):
        raise ValueError('disconnected')
    squares=[]
    allowed={}
    for rank,(h,t) in enumerate(zip(a,b)):
        dr,dc=shifts[h['part']]
        top,left=h['top']+dr,h['left']+dc
        box={(top+r,left+c) for r in range(s) for c in range(s)}
        if any(box & old for old in squares):
            raise ValueError('squares overlap')
        squares.append(box)
        for p in box:
            allowed[p]={h['part'],t['part']}
    union={}
    owners={}
    overlap=0
    for i,part in enumerate(info['parts']):
        dr,dc=shifts[i]
        for r,c in part:
            p=(r+dr,c+dc)
            v=int(grid[r,c])
            if p in union:
                if union[p]!=v or allowed.get(p)!=owners[p]|{i}:
                    raise ValueError('collision')
                overlap+=1
            union[p]=v
            owners.setdefault(p,set()).add(i)
    if overlap!=len(a)*4*(s-1):
        raise ValueError('rim count')
    if any(any(union.get(p)!=info['ink'] for p in box) for box in squares):
        raise ValueError('unfilled socket')
    union=normalize(union)
    h,w=shape(union)
    if max(h,w)>30:
        raise ValueError('output bounds')
    out=np.full((h,w),info['bg'],dtype=np.int8)
    for p,v in union.items():
        out[p]=v
    return out


def pairing_ids(info, mode='rank'):
    """Visible port identities test simpler matching conventions."""
    h,s=[list(q) for q in info['queues']]
    if mode=='greedy':
        rows=[[None,None]]
        for tile in sorted(info['tiles'],key=lambda o:(o['top'],o['left'])):
            col=int(tile['kind']=='S')
            if rows[-1][col] is not None:
                rows.append([None,None])
            rows[-1][col]=tile
        pairs=[(a,b) for a,b in rows if a is not None and b is not None]
    else:
        if mode=='reverse':
            s.reverse()
        elif mode in ('horizontal','horizontal_reverse'):
            reverse=mode=='horizontal_reverse'
            for queue in (h,s):
                queue.sort(key=lambda o:(o['left'],o['top']),reverse=reverse)
        elif mode=='nearest':
            s=[min(s,key=lambda b:(abs(a['top']-b['top'])+abs(a['left']-b['left']),
                                   b['top'])) for a in h]
        elif mode!='rank':
            raise ValueError('unknown matching control')
        pairs=list(zip(h,s))
    return frozenset(((a['top'],a['left']),(b['top'],b['left'])) for a,b in pairs)


def ordering_witness(info):
    correct=pairing_ids(info)
    return all(pairing_ids(info,mode)!=correct for mode in
               ('greedy','horizontal','horizontal_reverse','nearest'))


def make(rng,n=3,s=3,star=False,witness=False,background=None):
    """Bounded local stroke sampling and input-only registration.

    Packing sets ordinal identities, but never the final displacement.
    No parent puzzle generator is invoked or imported.
    """
    for attempt in range(800):
        bg=int(rng.integers(10)) if background is None else background
        ink=int(rng.choice([c for c in range(10) if c!=bg]))
        paths=list(map(int,rng.choice([c for c in range(10) if c not in (bg,ink)],int(rng.integers(1,4)),replace=False)))
        if star:
            degrees=[n]+[1]*n
        elif n==2:
            degrees=[2,1,1]
        elif n==3:
            degrees=[2,2,1,1]
        else:
            degrees=[3,2,1,1,1] if rng.integers(2) else [2,2,2,1,1]
        if star:
            k='H' if rng.integers(2) else 'S'
            kinds=[k]*n+(['S'] if k=='H' else ['H'])*n
        else:
            kinds=list(rng.permutation(['H']*n+['S']*n))
        parts=[]
        j=0
        for d in degrees:
            parts.append(component(rng,kinds[j:j+d],s,ink,paths))
            j+=d
        try:
            grid=pack(rng,parts,bg)
            info=parse(grid)
            out=assemble(info)
            if witness:
                if not ordering_witness(info):
                    raise ValueError('missing order witness')
                alt=assemble(info,'reverse')
                if np.array_equal(alt,out):
                    raise ValueError('same alternative')
            return dict(input=grid,output=out)
        except ValueError:
            continue
    raise RuntimeError('bounded construction exhausted')


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full answer using only the input pixels."""
    return assemble(parse(input_grid))


def integer_argument(name, value, low, high=None):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
            or int(value)<low or (high is not None and int(value)>high)):
        raise ValueError(f'illegal {name}')
    return int(value)


def grid_key(grid):
    grid=np.asarray(grid,dtype=np.int8)
    return tuple(grid.shape),grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed=integer_argument('seed',seed,0)
    nt=integer_argument('num_train',num_train,2,10)
    nq=integer_argument('num_test',num_test,1,4)
    if nt<=nq:
        raise ValueError('num_train must exceed num_test')
    rng=np.random.default_rng(seed)
    # These are distribution controls, never hidden solving conventions.
    background=int(rng.integers(10)) if rng.random()<.25 else None
    first_size=int(rng.integers(3,5))
    puzzle={'train':[],'test':[]}
    seen=set()
    for split,count in (('train',nt),('test',nq)):
        for index in range(count):
            if split=='train' and index<2:
                n=3
                s=first_size if index==0 else 7-first_size
                star=index==0
            elif split=='train':
                n=int(rng.integers(2,4))
                s=int(rng.integers(3,6 if n==2 else 5))
                star=n==3 and bool(rng.integers(2))
            else:
                # A longer tree of registrations, using familiar sizes/roles.
                n=4
                s=int(rng.integers(3,5))
                star=False
            for _ in range(8):
                pair=make(rng,n,s,star,split=='train' and index==0,background)
                key=grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    return puzzle


def pair_information(pair):
    """Family checks and descriptive metrics from the returned pixels."""
    inp,out=checked_grid(pair['input']),checked_grid(pair['output'])
    info=parse(inp)
    expected=assemble(info)
    assert np.array_equal(expected,out),'incorrect registration, union, or crop'
    n=len(info['queues'][0])
    s=info['size']
    assert np.count_nonzero(out!=info['bg'])==np.count_nonzero(inp!=info['bg'])-n*4*(s-1),'material lost or duplicated'
    assert len(groups(zip(*np.nonzero(out!=info['bg']))))==1,'disconnected result'
    assert all(np.any(edge!=info['bg']) for edge in (out[0],out[-1],out[:,0],out[:,-1])),'loose crop'
    assert set(np.unique(inp))-{info['bg']}==set(np.unique(out))-{info['bg']},'palette changed'
    assert not np.array_equal(inp,out)
    degrees=tuple(sorted(sum(o['part']==i for o in info['tiles']) for i in range(len(info['parts']))))
    return info,dict(junctions=n,size=s,background=info['bg'],ink=info['ink'],
                     fragments=len(info['parts']),degrees=degrees,
                     input_shape=tuple(inp.shape),output_shape=tuple(out.shape),
                     palette=len(np.unique(inp)),
                     input_foreground=int(np.count_nonzero(inp!=info['bg'])),
                     output_foreground=int(np.count_nonzero(out!=info['bg'])),
                     sequence=''.join(o['kind'] for o in sorted(info['tiles'],key=lambda o:(o['top'],o['left']))))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    nt,nq=len(puzzle['train']),len(puzzle['test'])
    assert 2<=nt<=10 and 1<=nq<=4 and nt>nq
    seen=set()
    training=[]
    training_metrics=[]
    tests=[]
    try:
        for split in ('train','test'):
            for pair in puzzle[split]:
                assert isinstance(pair,dict) and set(pair)=={'input','output'}
                info,metrics=pair_information(pair)
                key=grid_key(pair['input'])
                assert key not in seen,'duplicate input'
                seen.add(key)
                if split=='train':
                    training.append(info)
                    training_metrics.append(metrics)
                else:
                    tests.append(metrics)
    except (ValueError,TypeError,IndexError) as exc:
        raise AssertionError('malformed family member') from exc
    assert any(ordering_witness(i) for i in training),'missing independent-order evidence'
    # The wrong rank pairing must sometimes give a valid different picture.
    # Thus color, rim shape, and geometric feasibility alone cannot pick mates.
    has_ambiguous_rims=False
    for info in training:
        try:
            has_ambiguous_rims |= not np.array_equal(assemble(info,'reverse'),assemble(info))
        except ValueError:
            pass
    assert has_ambiguous_rims,'missing rank-disambiguation witness'
    assert any(max(m['degrees'])==3 for m in training_metrics),'branching not demonstrated'
    assert any(m['degrees']==(1,1,2,2) for m in training_metrics),'multistep path not demonstrated'
    assert {3,4}<={m['size'] for m in training_metrics},'both test scales must be demonstrated'
    assert max(m['junctions'] for m in training_metrics)<min(m['junctions'] for m in tests),'missing tree-length shift'


HOST_SEEDS=(0,854405993,282743624,934636613,820851060,843662089,315985882,
            82383526,26465061,755936868,588998433,448280214,12026637,470884228,
            283331265,467165568,92324644,292528936,355079729,617418909,761762387,
            52119945,246244910,429486788,567645591,587023426,661073600,899878628,
            261265654,432870510,774400956,735254425,449172962,159645637,683548531,
            349879926,53507971,150389953,924232365,747255682,23503176,327311954)


def self_test() -> dict:
    """Stream 42 puzzles; retain hashes, metric sets, and one sample only.

    No sample files, traces, retry history, or unbounded logs are saved.
    These are generator-owned checks, not independent semantic evaluation.
    """
    signatures=set()
    shapes=set()
    output_shapes=set()
    structures=set()
    sizes=set()
    backgrounds=set()
    topologies=set()
    sequences=set()
    palettes=set()
    first=None
    total=0
    for seed in HOST_SEEDS:
        puzzle=generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest=hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp,out=pair['input'],pair['output']
                assert np.array_equal(solve(inp.copy()),out)
                _,m=pair_information(pair)
                shapes.add(m['input_shape'])
                output_shapes.add(m['output_shape'])
                sizes.add(m['size'])
                backgrounds.add(m['background'])
                topologies.add(m['degrees'])
                sequences.add(m['sequence'])
                palettes.add(m['palette'])
                mapping={}
                norm=bytes(mapping.setdefault(int(c),len(mapping)) for c in inp.flat)
                structures.add((tuple(inp.shape),hashlib.sha256(norm).hexdigest()))
                for grid in (inp,out):
                    digest.update(bytes(grid.shape))
                    digest.update(grid.tobytes())
                total+=1
        signatures.add(digest.hexdigest())
        if first is None:
            first=puzzle
    assert len(signatures)==42
    assert sizes=={3,4,5} and backgrounds==set(range(10))
    assert len(shapes)>=35 and len(output_shapes)>=35
    assert len(structures)>=180 and len(sequences)>=30 and len(topologies)>=5
    assert {3,4,5}<=palettes
    again=generate_puzzle(0)
    for split in ('train','test'):
        for a,b in zip(first[split],again[split]):
            assert all(np.array_equal(a[k],b[k]) for k in ('input','output'))
    p=first['test'][0]
    assert np.array_equal(solve(p['input'].tolist()),p['output'])
    validate_generated_puzzle(generate_puzzle(19,10,4))
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),
                  (4,False),(None,1),(4,None),(4,'1')):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for seed in (-1,True,1.5,None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal seed accepted')
    damaged={s:[{k:v.copy() for k,v in pair.items()} for pair in first[s]] for s in ('train','test')}
    damaged['test'][0]['output'][0,0]=(int(damaged['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    mapping=np.array([9,3,6,0,8,2,1,5,7,4],dtype=np.int8)
    for pair in first['train'][:2]:
        assert np.array_equal(solve(mapping[pair['input']]),mapping[pair['output']])
    return dict(seeds=42,pairs=total,distinct_puzzles=len(signatures),
                input_shapes=len(shapes),output_shapes=len(output_shapes),
                normalized_inputs=len(structures),terminal_sizes=sorted(sizes),
                backgrounds=sorted(backgrounds),topologies=len(topologies),
                global_sequences=len(sequences),palette_sizes=sorted(palettes),
                independent_solver=False)


if __name__=='__main__':
    print(json.dumps(self_test(),separators=(',',':')))
