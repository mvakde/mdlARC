"""ARC lattice motif replication with independent row and column parity."""
import numpy as np

_SOURCE = {
    (23,23,0,2,4,2,1,'010101010'),
    (27,27,0,1,3,1,1,'110111010'),
    (25,25,0,8,2,0,5,'110010011'),
    (29,29,0,3,8,1,4,'010110001'),
}


def _seed(seed):
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a non-negative integer')
    return int(seed)


def _count(name,value,lo,hi):
    if value is None: return None
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo<=value<=hi:
        raise ValueError(f'{name} must be an integer in [{lo},{hi}]')
    return int(value)


def _counts(seed,rng,num_train,num_test):
    nt=_count('num_train',num_train,2,10)
    ns=_count('num_test',num_test,1,4)
    if nt is None and ns is None:
        ns=1+seed%4
        nt=ns+1+(seed//4)%(10-ns)
    elif nt is None:
        nt=int(rng.integers(ns+1,11))
    elif ns is None:
        ns=int(rng.integers(1,min(4,nt-1)+1))
    if nt<=ns: raise ValueError('num_train must exceed num_test')
    return nt,ns


def _symmetric(mask):
    if any(np.array_equal(mask,np.rot90(mask,k)) for k in (1,2,3)): return True
    reflected=np.fliplr(mask)
    return any(np.array_equal(mask,np.rot90(reflected,k)) for k in range(4))


def _mask_ok(mask,rc,cc):
    if rc is not None and not (mask[:rc].any() and mask[rc:].any()): return False
    if cc is not None and not (mask[:,:cc].any() and mask[:,cc:].any()): return False
    if rc is not None and cc is not None and not mask[:rc,:cc].any(): return False
    return True


def _motif(rng,size,row_cut,col_cut):
    maximum=min(size*size-1,max(3,2*size*size//3))
    wanted=int(rng.integers(2,maximum+1))
    sym=bool(rng.integers(2))
    if size<=3:
        candidates=[]
        for bits in range(1,1<<(size*size)):
            mask=np.array([(bits>>i)&1 for i in range(size*size)],dtype=bool).reshape(size,size)
            if 2<=mask.sum()<=maximum and _mask_ok(mask,row_cut,col_cut): candidates.append(mask)
        for filtered in (
            [m for m in candidates if m.sum()==wanted and _symmetric(m)==sym],
            [m for m in candidates if _symmetric(m)==sym],candidates):
            if filtered: return filtered[int(rng.integers(len(filtered)))].copy()
    else:
        for _ in range(240):
            mask=np.zeros((size,size),dtype=bool)
            mask.flat[rng.choice(size*size,size=wanted,replace=False)]=True
            if sym:
                mask=mask|(np.fliplr(mask) if rng.integers(2) else np.rot90(mask,2))
            if 2<=mask.sum()<=maximum and _mask_ok(mask,row_cut,col_cut): return mask
    mask=np.zeros((size,size),dtype=bool)
    mask[0,0]=mask[-1,-1]=True
    if size>=3: mask[0,-1]=True
    assert _mask_ok(mask,row_cut,col_cut)
    return mask


def _axis(rng,size,phase,ending,regime,size_mode):
    pitch=size+1
    options=[]
    for complete in range(4,11):
        if ending=='full':
            n=complete*pitch-1
            if n<=30: options.append((n,None,complete))
        elif (complete%2==phase)==(ending=='selected'):
            for depth in range(1,size):
                n=complete*pitch+depth
                if n<=30: options.append((n,depth,complete))
    assert options
    if size_mode=='up':
        ceiling=7 if size==3 else 10
        if regime=='source_train':
            options=[x for x in options if x[2]<=ceiling-2]
        else:
            options=[x for x in options if x[2]>=ceiling-1]
        assert options
    elif regime=='target_test' and rng.random()<0.38:
        top=max(x[2] for x in options)
        options=[x for x in options if x[2]>=top-1]
    return options[int(rng.integers(len(options)))]


def _lattice(h,w,bg,sep,size):
    grid=np.full((h,w),bg,dtype=np.int8)
    grid[size::size+1,:]=sep
    grid[:,size::size+1]=sep
    return grid


def _stamp(inp,mask,source,accent):
    out=inp.copy()
    size=len(mask); pitch=size+1
    h,w=out.shape
    for cr,top in enumerate(range(0,h,pitch)):
        if cr%2!=source[0]%2: continue
        for cc,left in enumerate(range(0,w,pitch)):
            if cc%2!=source[1]%2: continue
            for r,c in np.argwhere(mask):
                y,x=top+int(r),left+int(c)
                if y<h and x<w: out[y,x]=accent
    return out


def _pair(rng,size,phase,endings,regime,bg,size_mode):
    h,rc,nr=_axis(rng,size,phase[0],endings[0],regime,size_mode)
    w,cc,nc=_axis(rng,size,phase[1],endings[1],regime,size_mode)
    # Squares are native to the source family. Match axis lengths when the
    # independently chosen ending and parity also permit that length.
    if rng.random()<0.32:
        pitch=size+1
        depth=h%pitch
        compatible=(depth==size if endings[1]=='full' else
                    1<=depth<size and ((h//pitch)%2==phase[1])==(endings[1]=='selected'))
        if compatible:
            w=h; cc=None if endings[1]=='full' else depth
            nc=(h+1)//pitch if endings[1]=='full' else h//pitch
    rows=[i for i in range(nr) if i%2==phase[0]]
    cols=[i for i in range(nc) if i%2==phase[1]]
    source=(int(rng.choice(rows)),int(rng.choice(cols)))
    mask=_motif(rng,size,rc if endings[0]=='selected' else None,cc if endings[1]=='selected' else None)
    colors=[v for v in range(10) if v!=bg]
    sep,accent=(int(v) for v in rng.choice(colors,size=2,replace=False))
    inp=_lattice(h,w,bg,sep,size)
    pitch=size+1
    for r,c in np.argwhere(mask): inp[source[0]*pitch+int(r),source[1]*pitch+int(c)]=accent
    return {'input':inp,'output':_stamp(inp,mask,source,accent)}


def _key(grid): return grid.shape,grid.tobytes()


def _analyze(inp,out):
    """Infer all roles and independently check exact unchanged-mask translation."""
    assert inp.shape==out.shape
    h,w=inp.shape
    possibilities=[]
    for size in (2,3,4):
        pitch=size+1
        if min(h,w)<=size: continue
        sep=int(inp[size,size])
        rr=np.arange(h)[:,None]; cc=np.arange(w)[None,:]
        lines=(rr%pitch==size)|(cc%pitch==size)
        if np.all(inp[lines]==sep) and np.all(inp[~lines]!=sep): possibilities.append((size,sep,lines))
    assert len(possibilities)==1
    size,sep,lines=possibilities[0]
    vals,counts=np.unique(inp[~lines],return_counts=True)
    assert len(vals)==2
    bg=int(vals[np.argmax(counts)]); accent=int(vals[np.argmin(counts)])
    assert len({bg,sep,accent})==3
    pos=np.argwhere(inp==accent)
    pitch=size+1
    cells={(int(r)//pitch,int(c)//pitch) for r,c in pos}
    assert len(cells)==1
    source=next(iter(cells))
    assert source[0]*pitch+size<=h and source[1]*pitch+size<=w
    mask=np.zeros((size,size),dtype=bool)
    for r,c in pos: mask[int(r)%pitch,int(c)%pitch]=True
    assert 2<=mask.sum()<=min(size*size-1,max(3,2*size*size//3))
    expected=_stamp(inp,mask,source,accent)
    assert np.array_equal(out,expected) and not np.array_equal(inp,out)
    assert np.array_equal(inp[lines],out[lines])
    assert np.all(out[~lines]!=sep)
    phase=(source[0]%2,source[1]%2)
    added=np.argwhere(out==accent)
    assert len({int(r)//pitch for r,_ in added})>=2
    assert len({int(c)//pitch for _,c in added})>=2
    def edge(length,axis):
        depth=length%pitch
        if not 1<=depth<size: return 'full'
        if (length//pitch)%2!=phase[axis]: return 'unselected'
        before=mask[:depth,:] if axis==0 else mask[:,:depth]
        after=mask[depth:,:] if axis==0 else mask[:,depth:]
        return 'clipped' if before.any() and after.any() else 'partial_other'
    bits=''.join('1' if v else '0' for v in mask.flat)
    signature=(h,w,bg,sep,accent,source[0],source[1],bits)
    assert signature not in _SOURCE
    return {'size':size,'shape':(h,w),'background':bg,'separator':sep,'accent':accent,
            'source':source,'phase':phase,'mask':mask,'count':int(mask.sum()),
            'symmetric':_symmetric(mask),'row_edge':edge(h,0),'col_edge':edge(w,1),
            'signature':signature}


def generate_puzzle(seed: int,num_train: int | None = None,num_test: int | None = None) -> dict:
    seed=_seed(seed)
    rng=np.random.default_rng(seed)
    nt,ns=_counts(seed,rng,num_train,num_test)
    # Interior size is shared and shown directly in every input. Colors and
    # motif complexity are local roles, so they vary freely between pairs.
    size=int(rng.choice((2,3,4),p=(.1,.8,.1)))
    size_mode='up' if size==3 and rng.random()<0.25 else 'free'
    phases=((0,0),(0,1),(1,0),(1,1))
    first=[(0,0),(1,1)] if rng.integers(2) else [(0,1),(1,0)]
    rng.shuffle(first)
    train_phases=first+[phases[int(rng.integers(4))] for _ in range(nt-2)]
    test_phases=[phases[int(rng.integers(4))] for _ in range(ns)]
    endings=('full','selected','unselected')
    train_endings=[('selected','full') if rng.integers(2) else ('full','selected')]
    train_endings += [(str(rng.choice(endings)),str(rng.choice(endings))) for _ in range(nt-1)]
    test_endings=[(str(rng.choice(endings)),str(rng.choice(endings))) for _ in range(ns)]
    backgrounds=[int(v) for v in rng.choice(10,size=2,replace=False)]
    backgrounds += [int(rng.integers(10)) for _ in range(nt+ns-2)]
    result={'train':[],'test':[]}; seen=set(); index=0
    train_configs=set(); motif_counts=set()
    for split,regime,count,pair_phases,pair_endings in (
        ('train','source_train',nt,train_phases,train_endings),
        ('test','target_test',ns,test_phases,test_endings)):
        for j in range(count):
            for _ in range(80):
                pair=_pair(rng,size,pair_phases[j],pair_endings[j],regime,backgrounds[index],size_mode)
                k=_key(pair['input'])
                if k in seen: continue
                try: analysis=_analyze(pair['input'],pair['output'])
                except AssertionError: continue
                config=(analysis['shape'],analysis['source'],analysis['phase'])
                if split=='test' and config in train_configs: continue
                if index==nt+ns-1 and len(motif_counts)==1 and analysis['count'] in motif_counts:
                    continue
                seen.add(k); result[split].append(pair); break
            else: raise RuntimeError('unable to construct a distinct pair')
            motif_counts.add(analysis['count'])
            if split=='train': train_configs.add(config)
            index+=1
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert isinstance(puzzle['train'],list) and isinstance(puzzle['test'],list)
    nt,ns=len(puzzle['train']),len(puzzle['test'])
    assert 2<=nt<=10 and 1<=ns<=4 and nt>ns
    seen_inputs=set(); seen_pairs=set(); analyses={'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            for name in ('input','output'):
                grid=pair[name]
                assert isinstance(grid,np.ndarray) and grid.ndim==2
                assert all(1<=n<=30 for n in grid.shape)
                assert np.issubdtype(grid.dtype,np.integer)
                assert 0<=int(grid.min()) and int(grid.max())<=9
            ik=_key(pair['input']); pk=(ik,_key(pair['output']))
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik); seen_pairs.add(pk)
            analyses[split].append(_analyze(pair['input'],pair['output']))
    items=analyses['train']+analyses['test']
    assert len({a['size'] for a in items})==1
    assert len({a['background'] for a in items})>=2
    assert len({a['count'] for a in items})>=2
    train_phases={a['phase'] for a in analyses['train']}
    assert {p[0] for p in train_phases}=={0,1}
    assert {p[1] for p in train_phases}=={0,1}
    assert any(a['row_edge']=='clipped' or a['col_edge']=='clipped' for a in analyses['train'])
    train_configs={(a['shape'],a['source'],a['phase']) for a in analyses['train']}
    assert all((a['shape'],a['source'],a['phase']) not in train_configs for a in analyses['test'])


def _predict_from_training(train,test_input):
    # Validate the demonstrated rule, then use only the visible test input.
    for pair in train: _analyze(pair['input'],pair['output'])
    grid=test_input
    for size in (2,3,4):
        pitch=size+1
        if min(grid.shape)<=size: continue
        sep=int(grid[size,size])
        rr=np.arange(grid.shape[0])[:,None]; cc=np.arange(grid.shape[1])[None,:]
        lines=(rr%pitch==size)|(cc%pitch==size)
        if not (np.all(grid[lines]==sep) and np.all(grid[~lines]!=sep)): continue
        vals,counts=np.unique(grid[~lines],return_counts=True)
        if len(vals)!=2: continue
        accent=int(vals[np.argmin(counts)])
        pos=np.argwhere(grid==accent)
        source=(int(pos[0,0])//pitch,int(pos[0,1])//pitch)
        mask=np.zeros((size,size),dtype=bool)
        for r,c in pos: mask[int(r)%pitch,int(c)%pitch]=True
        return _stamp(grid,mask,source,accent)
    raise AssertionError('test input does not expose a lattice motif')


def self_test() -> None:
    def equal(a,b):
        return all(len(a[s])==len(b[s]) and all(
            np.array_equal(x[k],y[k]) for x,y in zip(a[s],b[s]) for k in ('input','output'))
            for s in ('train','test'))
    assert equal(generate_puzzle(314159),generate_puzzle(314159))
    assert equal(generate_puzzle(271828,6,2),generate_puzzle(271828,6,2))
    for i,(nt,ns) in enumerate(((2,1),(10,1),(10,4),(None,1),
                                (None,4),(2,None),(10,None))):
        puzzle=generate_puzzle(6000+i,nt,ns)
        if nt is not None: assert len(puzzle['train'])==nt
        if ns is not None: assert len(puzzle['test'])==ns
    for args in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(True,None)):
        try: generate_puzzle(8000,*args)
        except ValueError: pass
        else: raise AssertionError('accepted invalid counts')
    train_counts=set(); test_counts=set(); shapes=set(); sources=set(); masks=set()
    sizes=set(); phases=set(); edges=set(); symmetric=asym=0
    square=reverse=larger=var_bg=var_count=var_sym=0; puzzles=set()
    repeated_phase=missing_phase=varied_test_phase=combined=0
    split_edges={'train':set(),'test':set()}
    source_zones={'train':set(),'test':set()}
    connected=scattered=0
    for seed in range(42):
        puzzle=generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train'])); test_counts.add(len(puzzle['test']))
        items=[_analyze(p['input'],p['output']) for split in ('train','test') for p in puzzle[split]]
        training=items[:len(puzzle['train'])]; tests=items[len(puzzle['train']):]
        puzzles.add(tuple(a['signature'] for a in items))
        sizes.add(items[0]['size'])
        var_bg+=len({a['background'] for a in items})>1
        var_count+=len({a['count'] for a in items})>1
        var_sym+=len({a['symmetric'] for a in items})>1
        square+=any(a['shape'][0]==a['shape'][1] for a in items)
        reverse+=any(t['shape'][0]<max(x['shape'][0] for x in training) for t in tests)
        larger+=any(t['shape'][0]>max(x['shape'][0] for x in training) and
                    t['shape'][1]>max(x['shape'][1] for x in training) for t in tests)
        train_phase_set={a['phase'] for a in training}
        repeated_phase+=any(a['phase'] in train_phase_set for a in tests)
        missing_phase+=any(a['phase'] not in train_phase_set for a in tests)
        varied_test_phase+=len({a['phase'] for a in tests})>1
        combined+=any(a['symmetric'] and a['shape'][0]==a['shape'][1] and
                      'clipped' in (a['row_edge'],a['col_edge']) for a in items)
        for split in ('train','test'):
          for a in (training if split=='train' else tests):
            split_edges[split].add((a['row_edge'],a['col_edge']))
            pitch=a['size']+1
            for axis,position in enumerate(a['source']):
                full=(a['shape'][axis]+1)//pitch
                zone='top' if position<=1 else 'late' if position>=full-2 else 'middle'
                source_zones[split].add((axis,zone))
            occupied={tuple(x) for x in np.argwhere(a['mask'])}
            reached={next(iter(occupied))}; pending=list(reached)
            while pending:
                r,c=pending.pop()
                for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if neighbor in occupied and neighbor not in reached:
                        reached.add(neighbor); pending.append(neighbor)
            connected+=len(reached)==len(occupied)
            scattered+=len(reached)<len(occupied)
            shapes.add(a['shape']); sources.add(a['source'])
            masks.add((a['size'],a['mask'].tobytes()))
            phases.add(a['phase']); edges.add((a['row_edge'],a['col_edge']))
            symmetric+=a['symmetric']; asym+=not a['symmetric']
        for pair in puzzle['test']:
            assert np.array_equal(_predict_from_training(puzzle['train'],pair['input']),pair['output'])
    assert len(train_counts)>=4 and len(test_counts)>=3
    assert len(puzzles)==42 and sizes=={2,3,4}
    assert len(shapes)>=30 and len(sources)>=15 and len(masks)>=60
    assert len(phases)==4 and len(edges)>=6
    assert symmetric>=30 and asym>=30
    assert square>=5 and reverse>=5
    assert var_bg==42 and var_count>=30
    assert larger>=3 and var_sym>=20 and combined>=3
    assert repeated_phase>=10 and missing_phase>=5 and varied_test_phase>=5
    assert connected>=30 and scattered>=30
    for split in ('train','test'):
        assert ('clipped','clipped') in split_edges[split]
        assert ('full','full') in split_edges[split]
        assert ('unselected','unselected') in split_edges[split]
        assert source_zones[split]=={(axis,zone) for axis in (0,1)
                                     for zone in ('top','middle','late')}
