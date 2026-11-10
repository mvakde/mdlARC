import numpy as np

_SOURCE_INPUTS = (
 ("0002000000200000","0002000000200004","0402000000200000","0002000400200000","4002000000204000","0002000000200000"),
 ("004000","000040","000000","022220","000000","004000","022220","000000","000000","040440","000000","000000"),
 ("222222","004040","400000","222222","000000","000400","000000","040000","000400","000000","000000","044000","004440","000000"),
 ("20000200000000000","02004020000000000","00200002000040000","00020040200000000","00002000020000004","04000000400404000"),
 ("0000040000","0020000200","0200000020","0020400200","0200000020","0024000200","0200040020","0020000200","4000400000","0000000400"),
)

def _counts(seed, rng, train, test):
    for v, lo, hi in ((train,2,10),(test,1,4)):
        if v is not None and (isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or not lo<=v<=hi):
            raise ValueError("illegal count")
    if train is not None and test is not None and train<=test:
        raise ValueError("training must exceed test")
    if train is None and test is None:
        train=2+int(seed)%9
        test=1+(int(seed)//9)%min(4,train-1)
    elif train is None:
        train=int(rng.integers(max(2,test+1),11))
    elif test is None:
        test=int(rng.integers(1,min(4,train-1)+1))
    return int(train),int(test)

def _size(rng):
    band=int(rng.integers(5))
    if band==0:
        return int(rng.integers(6,12)),int(rng.integers(6,12))
    if band==1:
        return int(rng.integers(22,31)),int(rng.integers(8,31))
    if band==2:
        return int(rng.integers(8,31)),int(rng.integers(22,31))
    return int(rng.integers(7,25)),int(rng.integers(7,25))

def _steps(rng,n,kind,direction):
    if kind=="straight": return np.zeros(n-1,dtype=int)
    if kind=="diagonal": return np.full(n-1,direction,dtype=int)
    if kind=="stepped":
        return np.array([direction if rng.random()<0.6 else 0 for _ in range(n-1)])
    if kind=="zigzag":
        phase=int(rng.integers(2)); hold=int(rng.integers(2,5))
        return np.array([0 if (i+phase)%hold==hold-1 else (1 if (i+phase)%2==0 else -1) for i in range(n-1)])
    return rng.choice([-1,0,1],n-1,p=[0.32,0.36,0.32])

def _geometry(rng,kind,orient):
    for attempt in range(100):
        h,w=_size(rng)
        axis,cross=(h,w) if orient=="vertical" else (w,h)
        tail_len=int(rng.integers(1,4)) if rng.random()<0.16 else 0
        if axis-tail_len<4: continue
        n=axis-tail_len if rng.random()<0.24 else int(rng.integers(4,axis-tail_len+1))
        if kind=="diagonal": n=min(n,cross-1)
        if n<4: continue
        gap=int(rng.integers(2,min(11,cross)))
        direction=int(rng.choice([-1,1]))
        corner=kind=="diagonal" and orient=="vertical" and rng.random()<0.55
        if corner: direction=1
        if kind=="one_sided":
            a=_steps(rng,n,"wander",direction); b=np.zeros(n-1,dtype=int)
        elif kind=="variable":
            a=_steps(rng,n,"wander",direction); b=_steps(rng,n,"wander",direction)
        elif kind=="zigzag":
            a=_steps(rng,n,kind,direction); b=_steps(rng,n,kind,direction)
        else:
            a=_steps(rng,n,kind,direction); b=a.copy()
        low=np.r_[0,np.cumsum(a)]
        high=gap+np.r_[0,np.cumsum(b)]
        if np.any(high-low<2): continue
        minimum=min(int(low.min()),int(high.min()))
        maximum=max(int(low.max()),int(high.max()))
        room=cross-1-(maximum-minimum)
        if room<0: continue
        position=(0 if corner else int(rng.choice([0,room])) if rng.random()<0.31
                  else int(rng.integers(room+1)))
        low+=position-minimum; high+=position-minimum
        axis_room=axis-n-tail_len
        start=(0 if corner else int(rng.choice([0,axis_room])) if rng.random()<0.4
               else int(rng.integers(axis_room+1)))
        tail=None
        if tail_len and not corner:
            end=int(rng.integers(2)); side=int(rng.integers(2))
            if end==0: start+=tail_len
            tail=(end,side,tail_len)
        elif corner and tail_len:
            tail=(1,int(rng.integers(2)),tail_len)
        trial_sep,trial_inside,*_= _masks(h,w,orient,start,low,high,tail)
        if not np.any(~(trial_sep|trial_inside)):
            continue
        return h,w,orient,start,low,high,tail
    h,w=_size(rng); axis,cross=(h,w) if orient=="vertical" else (w,h)
    n=int(rng.integers(4,axis+1)); start=int(rng.integers(axis-n+1))
    first=int(rng.integers(cross-2)); second=int(rng.integers(first+2,cross))
    if first==0 and second==cross-1 and start==0 and n==axis:
        first=1
    return h,w,orient,start,np.full(n,first),np.full(n,second),None

def _masks(h,w,orient,start,low,high,tail=None):
    sep=np.zeros((h,w),bool); inside=sep.copy(); left=sep.copy(); right=sep.copy(); beyond=np.ones((h,w),bool)
    for i,(a,b) in enumerate(zip(low,high)):
        x=start+i
        if orient=="vertical":
            sep[x,[a,b]]=True; inside[x,a+1:b]=True
            left[x,:a]=True; right[x,b+1:]=True; beyond[x,:]=False
        else:
            sep[[a,b],x]=True; inside[a+1:b,x]=True
            left[:a,x]=True; right[b+1:,x]=True; beyond[:,x]=False
    if tail:
        end,side,length=tail
        v=int((low if side==0 else high)[0 if end==0 else -1])
        axes=range(start-length,start) if end==0 else range(start+len(low),start+len(low)+length)
        for x in axes:
            if orient=="vertical": sep[x,v]=True
            else: sep[v,x]=True
    return sep,inside,left,right,beyond

def _add(rng,marked,candidates,n):
    cells=np.argwhere(candidates & ~marked)
    n=min(n,len(cells))
    if n:
        chosen=cells[rng.choice(len(cells),n,replace=False)]
        marked[chosen[:,0],chosen[:,1]]=True

def _pair(rng,kind,orient,roles,style):
    h,w,orient,start,low,high,tail=_geometry(rng,kind,orient)
    sep,inside,left,right,beyond=_masks(h,w,orient,start,low,high,tail)
    marked=np.zeros((h,w),bool)
    n=1 if rng.random()<0.3 else int(rng.integers(2,min(13,int(inside.sum()))+1))
    _add(rng,marked,inside,n)
    exterior=~(sep|inside)
    target={"low":left,"high":right,"beyond":beyond & ~sep,"mixed":exterior}[style]
    if not target.any(): target=exterior
    _add(rng,marked,target,int(rng.integers(1,min(17,int(target.sum()))+1)))
    if rng.random()<0.3:
        cells=np.argwhere(target & ~marked)
        if len(cells):
            y,x=map(int,cells[int(rng.integers(len(cells)))])
            dy,dx=(1,0) if rng.random()<0.5 else (0,1)
            for i in range(int(rng.integers(2,6))):
                yy,xx=y+i*dy,x+i*dx
                if 0<=yy<h and 0<=xx<w and target[yy,xx]: marked[yy,xx]=True
    if rng.random()<0.16:
        end=int(rng.integers(2)); a=start if end==0 else start+len(low)-1
        outside=a-1 if end==0 else a+1
        if 0<=outside<(h if orient=="vertical" else w):
            i=0 if end==0 else -1
            b=int(rng.integers(low[i]+1,high[i]))
            if orient=="vertical" and not sep[outside,b]:
                marked[a,b]=True; marked[outside,b]=True
            if orient=="horizontal" and not sep[b,outside]:
                marked[b,a]=True; marked[b,outside]=True
    marked &= ~sep
    if not np.any(marked & ~inside): _add(rng,marked,exterior,1)
    # Keep scattered markers visually sparse even on the smallest canvases.
    cap=max(2,int(0.28*np.count_nonzero(~sep)))
    if np.count_nonzero(marked)>cap:
        retained=np.zeros_like(marked)
        for region in (marked&inside, marked&~inside):
            cells=np.argwhere(region)
            chosen=cells[int(rng.integers(len(cells)))]
            retained[chosen[0],chosen[1]]=True
        _add(rng,retained,marked,cap-2)
        marked=retained
    background,structural,marker=roles
    inp=np.full((h,w),background,np.int8); inp[sep]=structural; inp[marked]=marker
    out=inp.copy(); out[marked & ~inside]=background
    return {"input":inp,"output":out}

def _parse(mask,orient):
    counts=mask.sum(axis=1 if orient=="vertical" else 0)
    occupied=np.flatnonzero(counts)
    if len(occupied)<4 or occupied[-1]-occupied[0]+1!=len(occupied): return None
    if np.any((counts[occupied]<1)|(counts[occupied]>2)): return None
    common=np.flatnonzero(counts==2)
    if len(common)<4 or common[-1]-common[0]+1!=len(common): return None
    if len(occupied)-len(common)>3: return None
    bounds=np.array([np.flatnonzero(mask[a,:] if orient=="vertical" else mask[:,a]) for a in common])
    low,high=bounds[:,0],bounds[:,1]
    if np.any(high-low<2) or np.any(np.abs(np.diff(bounds,axis=0))>1): return None
    tail=None
    if len(occupied)!=len(common):
        before=common[0]-occupied[0]; after=occupied[-1]-common[-1]
        if before and after: return None
        end=0 if before else 1; length=int(before or after)
        segment=occupied[:before] if before else occupied[-after:]
        values=[int(np.flatnonzero(mask[a,:] if orient=="vertical" else mask[:,a])[0]) for a in segment]
        anchor=bounds[0] if before else bounds[-1]
        if len(set(values))!=1 or values[0] not in anchor: return None
        tail=(end,int(values[0]==anchor[1]),length)
    return int(common[0]),low,high,tail

def _inspect(pair):
    assert isinstance(pair,dict) and set(pair)=={"input","output"}
    inp,out=pair["input"],pair["output"]
    for grid in (inp,out):
        assert isinstance(grid,np.ndarray) and grid.ndim==2
        assert 1<=grid.shape[0]<=30 and 1<=grid.shape[1]<=30
        assert np.issubdtype(grid.dtype,np.integer) and np.all((grid>=0)&(grid<=9))
    assert inp.shape==out.shape
    assert tuple("".join(map(str,row)) for row in inp) not in _SOURCE_INPUTS
    changed=inp!=out
    assert np.any(changed)
    old,new=np.unique(inp[changed]),np.unique(out[changed])
    assert len(old)==len(new)==1 and old[0]!=new[0]
    marker,background=int(old[0]),int(new[0])
    other=set(np.unique(inp).tolist())-{marker,background}
    assert len(other)==1
    structural=other.pop()
    assert set(np.unique(out).tolist())=={background,structural,marker}
    assert np.count_nonzero(inp==background)>max(np.count_nonzero(inp==structural),
                                                np.count_nonzero(inp==marker))
    sep=inp==structural
    assert _parse(inp==marker,"vertical") is None
    assert _parse(inp==marker,"horizontal") is None
    candidates=[]
    for orient in ("vertical","horizontal"):
        parsed=_parse(sep,orient)
        if parsed is not None: candidates.append((orient,*parsed))
    assert len(candidates)==1
    orient,start,low,high,tail=candidates[0]
    masks=_masks(*inp.shape,orient,start,low,high,tail)
    assert np.array_equal(sep,masks[0])
    assert np.any((inp==marker)&masks[1])
    expected=inp.copy(); expected[(inp==marker)&~masks[1]]=background
    assert np.array_equal(out,expected)
    return (background,structural,marker),orient,start,low,high,tail

def generate_puzzle(seed:int,num_train:int|None=None,num_test:int|None=None)->dict:
    rng=np.random.default_rng(seed)
    num_train,num_test=_counts(seed,rng,num_train,num_test)
    fixed=bool(rng.random()<0.47)
    shared=tuple(map(int,rng.choice(10,3,replace=False)))
    regime=int(rng.integers(4))
    familiar_orientation="vertical" if rng.random()<0.5 else "horizontal"
    families=("straight","diagonal","zigzag","stepped","one_sided","variable")
    puzzle={"train":[],"test":[]}; seen=set()
    for section,count in (("train",num_train),("test",num_test)):
        for index in range(count):
            if section=="train" and regime==0: kind="straight"
            elif index==0 and section=="train" and regime in (1,3): kind="straight"
            elif index==0 and section=="train" and regime==2: kind=families[int(rng.integers(1,6))]
            elif index==0 and section=="test" and regime==0: kind=families[int(rng.integers(1,6))]
            elif index==0 and section=="test" and regime==2: kind="straight"
            else: kind=families[int(rng.choice(6,p=[0.32,0.13,0.14,0.13,0.14,0.14]))]
            if regime==2 and num_train<=5:
                orient=(familiar_orientation if section=="train" else
                        "horizontal" if familiar_orientation=="vertical" else "vertical")
            else:
                orient="vertical" if rng.random()<0.5 else "horizontal"
            roles=shared if fixed or (section=="train" and index==0) else tuple(map(int,rng.choice(10,3,replace=False)))
            style=("mixed","low","high","beyond")[int(rng.integers(4))]
            for attempt in range(120):
                pair=_pair(np.random.default_rng(int(rng.integers(0,2**63-1))),kind,orient,roles,style)
                key=(pair["input"].shape,pair["input"].tobytes())
                if key in seen: continue
                try: _inspect(pair)
                except AssertionError: continue
                seen.add(key); puzzle[section].append(pair); break
            else: raise RuntimeError("pair construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle

def validate_generated_puzzle(puzzle:dict)->None:
    assert isinstance(puzzle,dict) and set(puzzle)=={"train","test"}
    assert isinstance(puzzle["train"],list) and isinstance(puzzle["test"],list)
    assert 2<=len(puzzle["train"])<=10 and 1<=len(puzzle["test"])<=4
    assert len(puzzle["train"])>len(puzzle["test"])
    seen_inputs=set(); seen_pairs=set()
    for section in ("train","test"):
        for pair in puzzle[section]:
            _inspect(pair)
            inp,out=pair["input"],pair["output"]
            ik=(inp.shape,inp.tobytes()); pk=(ik,out.tobytes())
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik); seen_pairs.add(pk)
    for pair in puzzle["train"]:
        inp,out=pair["input"],pair["output"]
        assert np.any(inp!=out) and np.any(out==_inspect(pair)[0][2])

def _fingerprint(p):
    return tuple(tuple((x["input"].shape,x["input"].tobytes(),x["output"].tobytes()) for x in p[s]) for s in ("train","test"))

def self_test()->None:
    puzzles=[generate_puzzle(i) for i in range(42)]
    assert len({_fingerprint(p) for p in puzzles})==42
    assert len({len(p["train"]) for p in puzzles})>=4
    assert len({len(p["test"]) for p in puzzles})>=3
    shapes=set(); layouts=set(); palettes=set()
    categories=set(); one=side=edge=narrow=large=varied=0
    review={name:0 for name in ('tail','variable_gap','one_sided_bend','zigzag',
                                  'partial_span','full_span','small','only_beyond',
                                  'horizontal_bend','one_survivor_train','same_family',
                                  'bent_train_straight_test','straight_only_train_bent_test',
                                  'opposed_orientation','diagonal_corner','crosses_end')}
    for p in puzzles:
        validate_generated_puzzle(p); roles=[]; family_by_section={'train':set(),'test':set()}
        orientations_by_section={'train':set(),'test':set()}
        for sec in ("train","test"):
            for pair in p[sec]:
                role,orient,start,low,high,tail=_inspect(pair)
                roles.append(role); palettes.add(role)
                inp,out=pair["input"],pair["output"]
                shapes.add(inp.shape); layouts.add((orient,start,tuple(low),tuple(high),tail,inp.shape))
                straight=bool(np.all(low==low[0]) and np.all(high==high[0]))
                categories.add((sec,orient,straight))
                family_by_section[sec].add('straight' if straight else 'bent')
                orientations_by_section[sec].add(orient)
                one+=np.count_nonzero(out==role[2])==1
                narrow+=np.min(high-low)==2
                large+=max(inp.shape)>=25
                edge+=np.any(low==0) or np.any(high==(inp.shape[1] if orient=="vertical" else inp.shape[0])-1)
                masks=_masks(*inp.shape,orient,start,low,high,tail)
                side+=np.any((inp==role[2])&masks[2])!=np.any((inp==role[2])&masks[3])
                axis=inp.shape[0] if orient=='vertical' else inp.shape[1]
                facts={
                    'tail':tail is not None,
                    'variable_gap':np.ptp(high-low)>0,
                    'one_sided_bend':np.any(np.diff(low))!=np.any(np.diff(high)),
                    'zigzag':(np.any(np.diff(low)>0) and np.any(np.diff(low)<0)) or (np.any(np.diff(high)>0) and np.any(np.diff(high)<0)),
                    'partial_span':len(low)<axis,
                    'full_span':len(low)==axis,
                    'small':min(inp.shape)<=7,
                    'only_beyond':np.any((inp==role[2])&masks[4]) and not np.any((inp==role[2])&(masks[2]|masks[3])),
                    'horizontal_bend':orient=='horizontal' and not straight,
                    'one_survivor_train':sec=='train' and np.count_nonzero(out==role[2])==1,
                    'diagonal_corner':orient=='vertical' and start==0 and low[0]==0 and np.all(np.diff(low)==1) and np.all(np.diff(high)==1),
                }
                for name,yes in facts.items(): review[name]+=bool(yes)
                for endpoint in (0,-1):
                    a=start if endpoint==0 else start+len(low)-1
                    outside=a-1 if endpoint==0 else a+1
                    if 0<=outside<axis:
                        for b in range(int(low[endpoint])+1,int(high[endpoint])):
                            if orient=='vertical':
                                crossing=(inp[a,b]==role[2] and inp[outside,b]==role[2] and masks[4][outside,b])
                            else:
                                crossing=(inp[b,a]==role[2] and inp[b,outside]==role[2] and masks[4][b,outside])
                            if crossing:
                                review['crosses_end']+=1
                                break
        varied+=len(set(roles))>1
        review['same_family']+=bool(family_by_section['train']&family_by_section['test'])
        review['bent_train_straight_test']+=('bent' in family_by_section['train'] and 'straight' in family_by_section['test'])
        review['straight_only_train_bent_test']+=(family_by_section['train']=={'straight'} and 'bent' in family_by_section['test'])
        review['opposed_orientation']+=not bool(orientations_by_section['train']&orientations_by_section['test'])
    assert len(shapes)>=35 and len(layouts)>=100 and len(palettes)>=20
    assert len(categories)==8
    assert min(one,side,edge,narrow,large,varied)>=3
    assert min(review.values())>=3
    assert _fingerprint(generate_puzzle(7))==_fingerprint(generate_puzzle(7))
    for tr,te in ((2,1),(10,1),(10,4)):
        p=generate_puzzle(300+tr*10+te,tr,te)
        assert (len(p["train"]),len(p["test"]))==(tr,te)
    for tr,te in ((None,1),(None,4),(2,None),(10,None)):
        validate_generated_puzzle(generate_puzzle(401,tr,te))
    for tr,te in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4)):
        try: generate_puzzle(502,tr,te)
        except ValueError: pass
        else: raise AssertionError("illegal counts accepted")
