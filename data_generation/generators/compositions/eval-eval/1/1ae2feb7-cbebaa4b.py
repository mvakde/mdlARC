"""Docked timing trees: geometry transports several competing periodic clocks.

The solver reads only the input. Construction stores no metadata in returned
pairs and renders from its assembled scene, before scattering the pieces.
Colors 0/2/5/9 have fixed background/key/pipe/root roles. All other colors are
copied from straight injector bars. See task_spec.md for the complete rule.
"""
from collections import Counter
import numpy as np

BG, PIN, TRACK, ROOT = 0,2,5,9
SOURCE_COLORS=(1,3,4,6,7,8)
DIRS=((-1,0),(1,0),(0,-1),(0,1))

def neighbors(q):
    r,c=q
    return [(r+dr,c+dc) for dr,dc in DIRS]

def box(cells):
    return min(r for r,c in cells),min(c for r,c in cells),max(r for r,c in cells),max(c for r,c in cells)

def norm(cells):
    t,l,_,_=box(cells)
    return tuple(sorted((r-t,c-l) for r,c in cells))

def shift(cells,d):
    return {(r+d[0],c+d[1]) for r,c in cells}

def components(cells):
    left=set(cells); result=[]
    while left:
        q=min(left); left.remove(q); found={q}; todo=[q]
        while todo:
            for p in neighbors(todo.pop()):
                if p in left:
                    left.remove(p); found.add(p); todo.append(p)
        result.append(found)
    return result

def shapes(size):
    states={((0,0),)}
    for _ in range(size-1):
        nxt=set()
        for s in states:
            ss=set(s)
            for p in set().union(*(set(neighbors(q)) for q in ss))-ss:
                if sum(n in ss for n in neighbors(p))==1:
                    nxt.add(norm(ss|{p}))
        states=nxt
    return sorted(states)
KEYS={n:shapes(n) for n in (3,4)}

def tree_info(route,root):
    parent={root:None}; depth={root:0}; order=[root]
    for q in order:
        for p in neighbors(q):
            if p in route and p!=parent[q]:
                assert p not in parent,'route cycle'
                parent[p]=q; depth[p]=depth[q]+1; order.append(p)
    assert len(parent)==len(route),'disconnected route'
    return parent,depth,order

def grow_path(rng, old, start, length, preferred=None):
    path=[start]; used=old|{start}
    for _ in range(length-1):
        options=[p for p in neighbors(path[-1]) if p not in used and sum(n in used for n in neighbors(p))==1]
        if not options: return None
        # Prefer persistence while retaining genuine turns.
        if len(path)>1:
            prev=(path[-1][0]-path[-2][0],path[-1][1]-path[-2][1])
        else: prev=preferred
        straight=None if prev is None else (path[-1][0]+prev[0],path[-1][1]+prev[1])
        p=straight if straight in options and rng.random()<.58 else options[int(rng.integers(len(options)))]
        path.append(p);used.add(p)
    return set(path)

def sample_route(rng,count):
    first=grow_path(rng,set(),(0,0),int(rng.integers(7,12)),(0,1))
    if first is None:return None
    nodes=[{'track':first,'pins':set()}]; edges=[]
    root=(0,0)
    size=int(rng.choice([3,4])); bank=KEYS[size]; used_keys=set()
    for i in range(1,count):
        made=False
        for attempt in range(130):
            parent=i-1 if i<3 or rng.random()<.5 else int(rng.integers(i))
            used=set().union(*(n['track']|n['pins'] for n in nodes))
            q=sorted(nodes[parent]['track'])[int(rng.integers(len(nodes[parent]['track'])))];dr,dc=DIRS[int(rng.integers(4))]
            stemlen=int(rng.integers(0,3)); stem={(q[0]+j*dr,q[1]+j*dc) for j in range(1,stemlen+1)}
            key=set(bank[int(rng.integers(len(bank)))]); lo=[p for p in key if p[1]==0]; entry=lo[int(rng.integers(len(lo)))]
            # map local (tangent,normal) onto chosen outward direction
            origin=(q[0]+(stemlen+1)*dr,q[1]+(stemlen+1)*dc)
            key={(origin[0]+(r-entry[0])*dc+(c-entry[1])*dr, origin[1]-(r-entry[0])*dr+(c-entry[1])*dc) for r,c in key}
            if norm(key) in used_keys:continue
            far=max(r*dr+c*dc for r,c in key); exits=[p for p in key if p[0]*dr+p[1]*dc==far]
            tip=exits[int(rng.integers(len(exits)))];start=(tip[0]+dr,tip[1]+dc)
            new=stem|key|{start}
            if new & used:continue
            if any(sum(p in used for p in neighbors(x))!=(1 if x==(q[0]+dr,q[1]+dc) else 0) for x in new):continue
            try:tree_info(used|new,root)
            except AssertionError:continue
            body=grow_path(rng,used|stem|key,start,int(rng.integers(7,12)),(dr,dc))
            if body is None:continue
            total=used|stem|key|body;t,l,b,r=box(total)
            if b-t>24 or r-l>24:continue
            nodes[parent]['track']|=stem;nodes[parent]['pins']|=key
            nodes.append({'track':body,'pins':set(key)});edges.append((parent,i,key));used_keys.add(norm(key));made=True;break
        if not made:return None
    return nodes,edges,root

def source_options(track,route,occupied,length):
    result=[]
    for q in sorted(track):
        for dr,dc in DIRS:
            cells={(q[0]+j*dr,q[1]+j*dc) for j in range(1,length+1)}
            if cells&occupied:continue
            if any((set(neighbors(p)) & route) - {q} for p in cells):continue
            if sum(q in neighbors(p) for p in cells)!=1:continue
            if any(set(neighbors(p)) & (occupied-route) for p in cells):continue
            total=occupied|cells;t,l,b,r=box(total)
            if b-t>=29 or r-l>=29:continue
            result.append((q,cells))
    return result

def sample_sources(rng,nodes,root,periods,colors,chain=False):
    route=set().union(*(n['track']|n['pins'] for n in nodes));parent,depth,order=tree_info(route,root)
    occupied=set(route); sources=[]
    for i,(length,color) in enumerate(zip(periods,colors)):
        # first two emitters in different successive modules, later arbitrary
        candidates=source_options(nodes[min(i, len(nodes)-1)]['track'],route,occupied,length) if i<2 else source_options(set().union(*(n['track'] for n in nodes)),route,occupied,length)
        rng.shuffle(candidates)
        for q,cells in candidates:
            if q==root:continue
            if any(q==s['q'] for s in sources):continue
            if i==1 or (chain and i>1):
                a=q
                while a is not None and a!=sources[i-1]['q']:a=parent[a]
                if a is None:continue
            sources.append({'q':q,'cells':cells,'period':length,'color':color,'node':next(j for j,n in enumerate(nodes) if q in n['track'])});occupied|=cells;break
        else:return None
    return sources

def pulse_data(nodes,root,sources):
    route=set().union(*(n['track']|n['pins'] for n in nodes));parent,depth,order=tree_info(route,root)
    at={s['q']:i for i,s in enumerate(sources)};up={};events={};winners={};recurrence=Counter();collision=[]
    tracks=set().union(*(n['track'] for n in nodes))
    for q in order:
        up[q]=([] if parent[q] is None else up[parent[q]].copy())
        if q in at:up[q].append(at[q])
        hits=[i for i in reversed(up[q]) if (depth[q]-depth[sources[i]['q']])%sources[i]['period']==0]
        events[q]=hits
        if q in tracks and q!=root and hits:
            winners[q]=hits[0];recurrence[hits[0]]+=1
            if len(hits)>1:collision.append(q)
    return parent,depth,events,winners,recurrence,collision

def _grid(value):
    a=np.asarray(value)
    assert a.ndim==2 and all(1<=n<=30 for n in a.shape), 'invalid grid dimensions'
    assert np.issubdtype(a.dtype,np.integer) and np.all((0<=a)&(a<=9)), 'invalid grid colors'
    return a.astype(np.int8,copy=False)


def _assemble(input_grid):
    """Match complete connected key masks, then propagate rigid translations."""
    inp=_grid(input_grid)
    roots=list(map(tuple,np.argwhere(inp==ROOT).tolist()))
    assert len(roots)==1, 'one root flag required'
    pieces=components(map(tuple,np.argwhere(inp!=BG).tolist()))
    assert len(pieces)>=2, 'need scattered pieces'
    occurrences={}; source_records=[]
    anchor=next(i for i,cells in enumerate(pieces) if roots[0] in cells)
    for i,cells in enumerate(pieces):
        pins={q for q in cells if inp[q]==PIN}
        assert pins and any(inp[q]==TRACK for q in cells), 'each piece needs pipe and keys'
        for key in components(pins):
            assert len(key)>=2, 'keys need a geometric signature'
            occurrences.setdefault(norm(key),[]).append((i,key))
        for color in SOURCE_COLORS:
            for bar in components(q for q in cells if inp[q]==color):
                assert len({q[0] for q in bar})==1 or len({q[1] for q in bar})==1, 'injectors are straight'
                touching={p for q in bar for p in neighbors(q) if p in cells and inp[p] in (TRACK,ROOT)}
                assert len(touching)==1, 'injector must touch one pipe junction'
                assert not any(inp[p]==PIN for q in bar for p in neighbors(q) if p in cells), 'bar touches a key'
                source_records.append(dict(node=i,q=touching.pop(),cells=bar,period=len(bar),color=color))
    assert source_records, 'need an injector'
    assert len(occurrences)==len(pieces)-1, 'piece graph must be a tree'
    links=[[] for _ in pieces]
    for endpoints in occurrences.values():
        assert len(endpoints)==2 and endpoints[0][0]!=endpoints[1][0], 'every key mask occurs on two pieces'
        (i,a),(j,b)=endpoints
        ta,la,_,_=box(a);tb,lb,_,_=box(b)
        d=(ta-tb,la-lb)
        links[i].append((j,d));links[j].append((i,(-d[0],-d[1])))
    translations={anchor:(0,0)}; todo=[anchor]
    for i in todo:
        for j,d in links[i]:
            proposed=(translations[i][0]+d[0],translations[i][1]+d[1])
            if j in translations:assert translations[j]==proposed, 'inconsistent docking'
            else:translations[j]=proposed;todo.append(j)
    assert len(translations)==len(pieces), 'all pieces must connect to the anchor'
    base=np.zeros_like(inp); multiplicity=Counter(); owners={}
    for i,cells in enumerate(pieces):
        d=translations[i]
        for q in sorted(cells):
            p=(q[0]+d[0],q[1]+d[1]);color=int(inp[q])
            assert 0<=p[0]<inp.shape[0] and 0<=p[1]<inp.shape[1], 'assembly exceeds canvas'
            if color==PIN:
                assert base[p] in (BG,PIN), 'key overlaps payload'
                multiplicity[p]+=1
            else:
                assert base[p]==BG, 'payload overlap'
                owners[p]=i
            base[p]=color
    assert set(multiplicity.values())=={2}, 'every connector cell has two owners'
    sources=[]
    for s in source_records:
        d=translations[s['node']]
        sources.append(dict(s,q=(s['q'][0]+d[0],s['q'][1]+d[1]),cells=shift(s['cells'],d)))
    route=set(map(tuple,np.argwhere(np.isin(base,(TRACK,PIN,ROOT))).tolist()))
    parent,depth,order=tree_info(route,roots[0])
    assert len({s['q'] for s in sources})==len(sources), 'injector junctions must differ'
    for s in sources:
        touching={p for q in s['cells'] for p in neighbors(q) if p in route}
        assert touching=={s['q']}, 'assembly changes injector junction'
    return dict(input=inp,base=base,pieces=pieces,keys=occurrences,translations=translations,
                sources=sources,root=roots[0],route=route,parent=parent,depth=depth,order=order,
                owners=owners,anchor=anchor,links=links)


def _predict(info,priority='near',phase='continuous'):
    """Cell-wise input-derived solver, with explicit rival hypotheses for audits."""
    out=info['base'].copy(); by_q={s['q']:s for s in info['sources']}
    for q in info['route']:
        if info['base'][q]!=TRACK:continue
        choices=[];p=q;last_key=None
        while p is not None:
            if info['base'][p]==PIN and last_key is None:last_key=p
            if p in by_q:
                s=by_q[p]
                elapsed=info['depth'][q]-info['depth'][p]
                if phase=='reset' and last_key is not None:
                    elapsed=info['depth'][q]-info['depth'][last_key]-1
                elif phase=='ignore_keys':
                    walk=q;elapsed=0
                    while walk!=p:
                        elapsed+=int(info['base'][walk]!=PIN);walk=info['parent'][walk]
                elif phase=='manhattan':
                    elapsed=abs(q[0]-p[0])+abs(q[1]-p[1])
                if elapsed%s['period']==0:choices.append(s)
            p=info['parent'][p]
        if choices:
            if priority=='near':winner=choices[0]
            elif priority=='far':winner=choices[-1]
            elif priority=='short':winner=min(choices,key=lambda s:s['period'])
            elif priority=='long':winner=max(choices,key=lambda s:s['period'])
            elif priority=='color':winner=max(choices,key=lambda s:s['color'])
            elif priority=='reverse_color':winner=min(choices,key=lambda s:s['color'])
            else:raise ValueError('unknown priority')
            out[q]=winner['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Assemble the input's rigid pipes and evaluate competing downstream clocks."""
    return _predict(_assemble(input_grid))


def _paint_scene(nodes,root,sources,shape,delta):
    """Construction uses far-to-near emission, separate from the cell solver."""
    base=np.zeros(shape,dtype=np.int8)
    for node in nodes:
        for q in shift(node['track'],delta):base[q]=TRACK
        for q in shift(node['pins'],delta):base[q]=PIN
    flagged=(root[0]+delta[0],root[1]+delta[1]);base[flagged]=ROOT
    for s in sources:
        for q in shift(s['cells'],delta):base[q]=s['color']
    route=set().union(*(n['track']|n['pins'] for n in nodes))
    parent,depth,order=tree_info(route,root)
    out=base.copy()
    for s in sorted(sources,key=lambda s:depth[s['q']]):
        descendants={s['q']}
        for q in order:
            if parent[q] in descendants:descendants.add(q)
        for q in descendants:
            p=(q[0]+delta[0],q[1]+delta[1])
            if base[p]==TRACK and (depth[q]-depth[s['q']])%s['period']==0:
                out[p]=s['color']
    return base,out


def _scatter(rng,nodes,root,sources,shape,delta):
    mappings=[]
    for i,node in enumerate(nodes):
        m={q:TRACK for q in node['track']}
        m.update({q:PIN for q in node['pins']})
        if root in m:m[root]=ROOT
        for s in sources:
            if s['node']==i:m.update({q:s['color'] for q in s['cells']})
        mappings.append({(q[0]+delta[0],q[1]+delta[1]):c for q,c in m.items()})
    for restart in range(12):
        chosen={0:mappings[0]};occupied=set(mappings[0])
        forbidden=set().union(*({(r+dr,c+dc) for dr in (-1,0,1) for dc in (-1,0,1)} for r,c in occupied))
        order=list(map(int,rng.permutation(np.arange(1,len(nodes)))))
        for i in order:
            m=mappings[i];t,l,b,r=box(m);h,w=b-t+1,r-l+1
            if h>shape[0] or w>shape[1]:break
            for attempt in range(350):
                d=(int(rng.integers(shape[0]-h+1))-t,int(rng.integers(shape[1]-w+1))-l)
                if d==(0,0):continue
                moved=shift(m,d)
                if moved&forbidden:continue
                chosen[i]={(q[0]+d[0],q[1]+d[1]):color for q,color in m.items()}
                forbidden.update((a+dr,c+dc) for a,c in moved for dr in (-1,0,1) for dc in (-1,0,1))
                break
            else:break
        if len(chosen)==len(nodes):
            inp=np.zeros(shape,dtype=np.int8)
            for m in chosen.values():
                for q,color in m.items():inp[q]=color
            return inp
    return None


def _ancestor(a,b,parent):
    """Whether a is an ancestor of b, including itself."""
    while b is not None:
        if a==b:return True
        b=parent[b]
    return False


def _witness_data(info,out):
    """Measure visible evidence from grids, with no construction metadata."""
    sources=info['sources'];parent=info['parent'];depth=info['depth'];base=info['base']
    visible={i:[] for i in range(len(sources))};collisions=[];far_survival=False
    for q in info['route']:
        if base[q]!=TRACK:continue
        ancestors=[i for i,s in enumerate(sources) if _ancestor(s['q'],q,parent)]
        firing=[i for i in ancestors if (depth[q]-depth[sources[i]['q']])%sources[i]['period']==0]
        if firing:
            winner=max(firing,key=lambda i:depth[sources[i]['q']])
            if int(out[q])==sources[winner]['color']:visible[winner].append(q)
            for j in firing:
                if j!=winner and sources[j]['color']!=sources[winner]['color'] and depth[q]>depth[sources[winner]['q']]:
                    collisions.append((winner,j,q))
            for j in ancestors:
                if depth[sources[j]['q']]>depth[sources[winner]['q']] and sources[j]['color']!=sources[winner]['color']:
                    far_survival=True
    recurrence={sources[i]['period'] for i,qs in visible.items() if any(
        depth[q]==depth[sources[i]['q']]+sources[i]['period'] for q in qs)}
    transported=any(info['owners'].get(q)!=s['node'] for i,s in enumerate(sources) for q in visible[i])
    return dict(visible=visible,collisions=collisions,far_survival=far_survival,
                recurrence=recurrence,transported=transported)


def _make_pair(rng,count,periods,colors,lesson=False,chain=False):
    for attempt in range(360):
        got=sample_route(rng,count)
        if got is None:continue
        nodes,edges,root=got
        sources=sample_sources(rng,nodes,root,periods,colors,chain)
        if sources is None:continue
        _,_,events,winners,recurrence,collisions=pulse_data(nodes,root,sources)
        if not collisions or any(recurrence.get(i,0)<2 for i in range(len(sources))):continue
        allcells=set().union(*(n['track']|n['pins'] for n in nodes),*(s['cells'] for s in sources))
        t,l,b,r=box(allcells);bh,bw=b-t+1,r-l+1
        shape=(min(30,max(bh,15+count)+int(rng.integers(0,6))),min(30,max(bw,15+count)+int(rng.integers(0,6))))
        for enlarge in range(2):
            if enlarge:shape=(30,30)
            delta=(int(rng.integers(shape[0]-bh+1))-t,int(rng.integers(shape[1]-bw+1))-l)
            base,out=_paint_scene(nodes,root,sources,shape,delta)
            inp=_scatter(rng,nodes,root,sources,shape,delta)
            if inp is not None:break
        if inp is None:continue
        try:
            info=_assemble(inp)
            assert np.array_equal(base,info['base'])
            witnesses=_witness_data(info,out)
            assert all(len(qs)>=2 for qs in witnesses['visible'].values())
            if lesson:
                assert witnesses['transported'] and witnesses['collisions']
                assert not np.array_equal(_predict(info,phase='reset'),out)
                assert not np.array_equal(_predict(info,phase='ignore_keys'),out)
                assert witnesses['far_survival']
                assert set(periods)<=witnesses['recurrence']
                assert any(info['sources'][a]['period']!=info['sources'][b]['period']
                           for a,b,q in witnesses['collisions'])
                # Both sources must participate after the later injector.
                assert not np.array_equal(_predict(info,priority='far'),out)
                assert not np.array_equal(_predict(info,phase='manhattan'),out)
        except AssertionError:continue
        k=int(rng.integers(4))
        inp,out=np.rot90(inp,k).copy(),np.rot90(out,k).copy()
        if rng.random()<.5:inp,out=inp[:,::-1].copy(),out[:,::-1].copy()
        return dict(input=inp,output=out)
    raise RuntimeError('bounded timing-tree construction exhausted')


def _check_counts(num_train,num_test):
    for name,value,low,high in [('num_train',num_train,2,10),('num_test',num_test,1,4)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low<=int(value)<=high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train<=num_test:raise ValueError('num_train must exceed num_test')
    return int(num_train),int(num_test)


def _key(grid):
    a=np.asarray(grid,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test=_check_counts(num_train,num_test)
    rng=np.random.default_rng(seed)
    regime=str(rng.choice(['network','injectors']))
    # Reversing these two clocks teaches proximity independently of period/hue.
    periods=[(2,3),(2,5),(3,4)][int(rng.integers(3))]
    colors=list(map(int,rng.choice(SOURCE_COLORS,2,replace=False)))
    puzzle={'train':[],'test':[]};seen=set();taught=set(periods)
    for split,n in [('train',num_train),('test',num_test)]:
        for i in range(n):
            count=int(rng.integers(3,5)) if split=='train' else int(rng.integers(5,7))
            if split=='train' and i>=2:count=int(rng.integers(2,5))
            if split=='train' and i<2:
                ls=list(periods if i==0 else periods[::-1]);cs=list(colors if i==0 else colors[::-1])
            else:
                amount=2 if split=='train' and regime=='injectors' else int(rng.integers(2,4))
                if split=='test' and regime=='injectors':amount=int(rng.integers(3,5))
                pool=sorted(taught) if split=='test' else list(range(1,6))
                ls=list(map(int,rng.choice(pool,amount,replace=True)))
                # A nonconstant upstream clock makes phase transport observable,
                # including in the optional two-piece training examples.
                if ls[0]==1:ls[0]=int(rng.choice([p for p in pool if p>1]))
                # Two contrasting hues always expose a real precedence choice.
                cs=list(map(int,rng.choice(SOURCE_COLORS,2,replace=False)))
                cs+=list(map(int,rng.choice(SOURCE_COLORS,max(0,amount-2),replace=True)))
                if split=='train':taught.update(ls)
            for retry in range(12):
                pair=_make_pair(rng,count,ls,cs,lesson=split=='train' and i<2,
                                chain=split=='test' and regime=='injectors')
                key=_key(pair['input'])
                if key not in seen:break
            else:raise RuntimeError('distinct-input retry exhausted')
            seen.add(key);puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    _check_counts(len(puzzle['train']),len(puzzle['test']))
    seen=set();lessons=[];counts={'train':[],'test':[]};taught=set()
    edges=set();short_near=long_near=False
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp,out=_grid(pair['input']),_grid(pair['output'])
            assert inp.shape==out.shape and not np.array_equal(inp,out)
            ikey=_key(inp);assert ikey not in seen, 'duplicate input';seen.add(ikey)
            info=_assemble(inp);expected=_predict(info)
            assert np.array_equal(out,expected), 'incorrect timing-tree output'
            base=info['base'];fixed=base!=TRACK
            assert np.array_equal(base[fixed],out[fixed]), 'keys, bars, root, and empty canvas must be preserved after assembly'
            assert np.count_nonzero(inp==PIN)==2*np.count_nonzero(out==PIN)
            assert info['translations'][info['anchor']]==(0,0)
            assert all(d!=(0,0) for i,d in info['translations'].items() if i!=info['anchor'])
            assert 2<=len(info['pieces'])<=6 and 2<=len(info['sources'])<=4
            assert len({len(key) for key in info['keys']})==1, 'equal-area signatures defeat area matching'
            assert next(iter(info['keys'])) and all(len(k) in (3,4) for k in info['keys'])
            assert all(1<=s['period']<=5 for s in info['sources'])
            witnesses=_witness_data(info,out)
            assert all(len(qs)>=2 for qs in witnesses['visible'].values()), 'hidden injector'
            counts[split].append(len(info['pieces']))
            if split=='train':
                taught|={s['period'] for s in info['sources']}
                lessons.append((info,out))
                for a,b,q in witnesses['collisions']:
                    near,far=info['sources'][a],info['sources'][b]
                    short_near|=near['period']<far['period'];long_near|=near['period']>far['period']
                    edges.add((near['color'],far['color']))
            else:assert {s['period'] for s in info['sources']}<=taught
    assert short_near and long_near, 'both proximity/length orders must be taught'
    assert any((b,a) in edges for a,b in edges), 'fixed color priority remains plausible'
    for priority in ('far','short','long','color','reverse_color'):
        assert any(not np.array_equal(_predict(info,priority=priority),out) for info,out in lessons), priority
    for phase in ('reset','ignore_keys'):
        assert any(not np.array_equal(_predict(info,phase=phase),out) for info,out in lessons), 'transport timing is not observable'
    assert any(_witness_data(info,out)['far_survival'] for info,out in lessons), 'older signals must survive newer injectors'
    assert any(not np.array_equal(_predict(info,phase='manhattan'),out) for info,out in lessons)
    assert max(counts['test'])>max(counts['train']), 'network-size shift missing'
    assert len(set.union(*(_witness_data(info,out)['recurrence'] for info,out in lessons)))>=2


def _features(pair):
    info=_assemble(pair['input']);depth=info['depth'];route=info['route']
    return dict(shape=tuple(info['input'].shape),pieces=len(info['pieces']),sources=len(info['sources']),
                periods=tuple(sorted(s['period'] for s in info['sources'])),
                key_masks=tuple(sorted(info['keys'])),max_depth=max(depth.values()),
                branches=sum(sum(p in route for p in neighbors(q))>=3 for q in route),
                turns=sum(len([p for p in neighbors(q) if p in route])==2 and
                          len({p[0] for p in neighbors(q) if p in route})>1 and
                          len({p[1] for p in neighbors(q) if p in route})>1 for q in route),
                root=info['root'],palette=tuple(map(int,np.unique(info['input']))),
                assembly_box=(box(route)[2]-box(route)[0]+1,box(route)[3]-box(route)[1]+1))


def _hand_fixture():
    """A small fixed arithmetic regression, not labels computed by solve()."""
    key_a={(7,5),(8,5),(9,5)}
    key_b={(8,11),(9,11),(9,12)}
    maps=[{(8,c):TRACK for c in range(1,5)},
          {(8,c):TRACK for c in range(6,11)},
          {(9,c):TRACK for c in range(13,23)}]
    maps[0].update({q:PIN for q in key_a});maps[0][8,1]=ROOT
    maps[0].update({(6,3):1,(7,3):1})
    maps[1].update({q:PIN for q in key_a|key_b})
    maps[1].update({(r,7):TRACK for r in range(5,8)})
    maps[1].update({(r,9):3 for r in range(5,8)})
    maps[2].update({q:PIN for q in key_b})
    inp=np.zeros((20,26),dtype=np.int8);expected=inp.copy()
    for m,d in zip(maps,[(0,0),(-4,9),(7,-10)]):
        for (r,c),color in m.items():
            inp[r+d[0],c+d[1]]=color;expected[r,c]=color
    # Injector junction depths are 2 and 8; their periods are 2 and 3.
    # The L-shaped second key takes THREE clock ticks to traverse. Its two
    # corners count, even though those cells remain connector-colored.
    for q in [(8,3),(8,7),(6,7),(9,16),(9,18),(9,22)]:expected[q]=1
    for q in [(8,9),(9,14),(9,17),(9,20)]:expected[q]=3
    return dict(input=inp,output=expected)


def self_test():
    fixture=_hand_fixture()
    for k in range(4):
        assert np.array_equal(solve(np.rot90(fixture['input'],k)),np.rot90(fixture['output'],k)), 'hand arithmetic regression'
    puzzles=[generate_puzzle(seed) for seed in range(42)]
    features=[_features(pair) for puzzle in puzzles for split in ('train','test') for pair in puzzle[split]]
    assert len({_key(pair['input']) for puzzle in puzzles for split in ('train','test') for pair in puzzle[split]})==210
    assert len({f['shape'] for f in features})>=35
    assert {f['pieces'] for f in features}=={2,3,4,5,6}
    assert {f['sources'] for f in features}=={2,3,4}
    assert {p for f in features for p in f['periods']}=={1,2,3,4,5}
    assert len({f['assembly_box'] for f in features})>=50
    assert len({f['max_depth'] for f in features})>=25
    assert len({f['key_masks'] for f in features})>=35
    assert len({f['root'] for f in features})>=60
    assert max(f['branches'] for f in features)>=6 and max(f['turns'] for f in features)>=12
    assert len({f['palette'] for f in features})>=25
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p=generate_puzzle(4201,a,b)
            assert len(p['train'])==a and len(p['test'])==b
    for counts in [(1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(3.,1),(None,1),(4,None)]:
        try:generate_puzzle(0,*counts)
        except ValueError:pass
        else:raise AssertionError('illegal count accepted')
    a,b=generate_puzzle(0),puzzles[0]
    assert all(_key(x[k])==_key(y[k]) for split in a for x,y in zip(a[split],b[split]) for k in ('input','output'))
    # Metamorphic checks exercise an input-only solver under every symmetry and
    # source-hue relabeling. Roles retain their fixed family meanings.
    for pair in puzzles[0]['train']:
        for k in range(4):
            inp,out=np.rot90(pair['input'],k),np.rot90(pair['output'],k)
            assert np.array_equal(solve(inp),out)
            assert np.array_equal(solve(inp[:,::-1]),out[:,::-1])
        mapping=np.arange(10,dtype=np.int8);mapping[list(SOURCE_COLORS)]=SOURCE_COLORS[::-1]
        assert np.array_equal(solve(mapping[pair['input']]),mapping[pair['output']])
    bad={split:[{k:np.asarray(v).copy() for k,v in pair.items()} for pair in puzzles[0][split]] for split in ('train','test')}
    info=_assemble(bad['test'][0]['input']);q=next(q for q in info['route'] if info['base'][q]==TRACK and bad['test'][0]['output'][q]!=TRACK)
    bad['test'][0]['output'][q]=TRACK
    try:validate_generated_puzzle(bad)
    except AssertionError:pass
    else:raise AssertionError('missing pulse accepted')


if __name__=='__main__':
    self_test()
    print('self_test passed (42 seeds, all legal counts, semantic and symmetry checks)')
