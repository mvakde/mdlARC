"""Generate straight-bar recoloring puzzles."""
import numpy as np

_SOURCE = {
 ('021111','020000','020003','020003','000003','888803'),
 ('000001','030001','030001','030001','036666','009999'),
 ('608000','608000','608000','608000','009999','003333'),
 ('777700','300008','300008','300008','300008','007777'),
 ('605555','650000','650000','650000','066666','000000'),
}

def _rows(a):
    return tuple(''.join(str(int(x)) for x in row) for row in a)

def _counts(rng, a, b):
    for x, lo, hi in ((a,2,10),(b,1,4)):
        if x is not None and (isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not lo <= x <= hi):
            raise ValueError('illegal pair count')
    if a is None and b is None:
        a = int(rng.integers(2,11))
        b = int(rng.integers(1,min(4,a-1)+1))
    elif a is None:
        a = int(rng.integers(max(2,b+1),11))
    elif b is None:
        b = int(rng.integers(1,min(4,a-1)+1))
    if a <= b:
        raise ValueError('train count must exceed test count')
    return int(a),int(b)

def _segments(h,w):
    out=[]
    for d in (0,1):
        for axis in range(h if d==0 else w):
            span=w if d==0 else h
            for n in range(2,span+1):
                for start in range(span-n+1):
                    cells=tuple((axis,start+k) if d==0 else (start+k,axis) for k in range(n))
                    mask=sum(1 << (r*w+c) for r,c in cells)
                    out.append((d,cells,mask))
    return out

def _orientations(inp,bg):
    """Unique cell orientations among monochrome straight-bar covers."""
    h,w=inp.shape
    occupied=sum(1 << (r*w+c) for r in range(h) for c in range(w) if inp[r,c]!=bg)
    by_cell=[[] for _ in range(h*w)]
    for d,cells,mask in _segments(h,w):
        color=inp[cells[0]]
        if color==bg or any(inp[cell]!=color for cell in cells):
            continue
        for r,c in cells:
            by_cell[r*w+c].append((d,mask))
    memo={}
    def search(left):
        if not left:
            return {0}
        if left in memo:
            return memo[left]
        options=None
        bits=left
        while bits:
            bit=bits & -bits
            candidates=[(d,m) for d,m in by_cell[bit.bit_length()-1] if m & left==m]
            if options is None or len(candidates)<len(options):
                options=candidates
                if len(options)<=1:
                    break
            bits^=bit
        answers=set()
        for d,m in options:
            for horizontal in search(left^m):
                answers.add(horizontal|m if d==0 else horizontal)
                if len(answers)>1:
                    memo[left]=answers
                    return answers
        memo[left]=answers
        return answers
    results=search(occupied)
    return (occupied,next(iter(results))) if len(results)==1 else None

def _runs(orientation):
    h,w=orientation.shape
    bars=[]
    for d in (0,1):
        for axis in range(h if d==0 else w):
            span=w if d==0 else h
            start=None
            for k in range(span+1):
                active=k<span and orientation[axis,k]==d if d==0 else k<span and orientation[k,axis]==d
                if active and start is None:
                    start=k
                if not active and start is not None:
                    bars.append((d,tuple((axis,j) if d==0 else (j,axis) for j in range(start,k))))
                    start=None
    return bars

def _contacts(bars):
    found=set()
    for i,(d1,a) in enumerate(bars):
        for d2,b in bars[i+1:]:
            if any(abs(r-s)+abs(c-t)==1 for r,c in a for s,t in b):
                found.add('cross' if d1!=d2 else 'parallel')
    return found

def _geometry(rng,h,w,n,preference,familiar=False):
    segments=_segments(h,w)
    directions=[0,1]+[int(rng.integers(2)) for _ in range(n-2)]
    rng.shuffle(directions)
    bars=[]
    mask=0
    for d in directions:
        choices=[]
        weights=[]
        old=_contacts(bars)
        for direction,cells,m in segments:
            if direction!=d or m&mask or len(cells)>9 or (familiar and len(cells)!=4) or (mask|m).bit_count()>h*w*.54:
                continue
            contact=_contacts(bars+[(d,cells)])
            new=bool(contact-old)
            if preference=='isolated' and new:
                continue
            weight=5 if (preference=='touch' and new or preference in contact) else 1
            choices.append((direction,cells,m))
            weights.append(weight)
        if not choices:
            return None
        p=np.array(weights,dtype=float)
        p/=p.sum()
        _,cells,m=choices[int(rng.choice(len(choices),p=p))]
        bars.append((d,cells))
        mask|=m
    contacts=_contacts(bars)
    if preference in ('cross','parallel') and preference not in contacts:
        return None
    if preference=='touch' and not contacts:
        return None
    return bars

def _colors(rng,bars,bg,style,v,h):
    choices=[x for x in range(10) if x!=bg]
    n=len(bars)
    if style=='random':
        pool=rng.choice(choices,size=int(rng.integers(2,min(n,6)+1)),replace=False)
        colors=[int(x) for x in rng.choice(pool,size=n)]
    else:
        colors=[int(x) for x in rng.choice(choices,size=n,replace=False)]
        if style=='cross':
            i=next(i for i,(d,_) in enumerate(bars) if d==0)
            j=next(i for i,(d,_) in enumerate(bars) if d==1)
            colors[j]=colors[i]
        if style=='parallel':
            indices=[i for i,(d,_) in enumerate(bars) if d==0]
            if len(indices)<2:
                indices=[i for i,(d,_) in enumerate(bars) if d==1]
            if len(indices)<2:
                return None
            colors[indices[1]]=colors[indices[0]]
    for d,target in ((0,h),(1,v)):
        if all(colors[i]==target for i,(direction,_) in enumerate(bars) if direction==d):
            return None
    return colors

def _pair(rng,v,h,used,style,preference,familiar=False):
    for _ in range(400):
        if familiar or rng.integers(5)==0:
            height=width=6
        else:
            height,width=int(rng.integers(5,13)),int(rng.integers(5,13))
        n=4 if familiar else int(rng.integers(2,min(7,height*width//8+1)))
        if preference=='isolated':
            n=min(n,4)
        bars=_geometry(rng,height,width,n,preference,familiar)
        if bars is None:
            continue
        bg_choices=[x for x in range(10) if x not in (v,h)]
        bg=0 if 0 in bg_choices and rng.integers(3)==0 else int(rng.choice(bg_choices))
        colors=_colors(rng,bars,bg,style,v,h)
        if colors is None:
            continue
        inp=np.full((height,width),bg,dtype=np.int8)
        out=inp.copy()
        for (d,cells),color in zip(bars,colors):
            for r,c in cells:
                inp[r,c]=color
                out[r,c]=h if d==0 else v
        if np.count_nonzero(inp==bg)<=max(np.count_nonzero(inp==color) for color in set(colors)):
            continue
        key=(inp.shape,inp.tobytes())
        if key in used or _rows(inp) in _SOURCE:
            continue
        found=_orientations(inp,bg)
        if found is None:
            continue
        occupied,horizontal=found
        orient=np.full(inp.shape,-1,dtype=np.int8)
        valid=True
        for r in range(height):
            for c in range(width):
                bit=1 << (r*width+c)
                if occupied&bit:
                    orient[r,c]=0 if horizontal&bit else 1
                    if out[r,c]!=(h if horizontal&bit else v):
                        valid=False
        if not valid:
            continue
        inferred=_runs(orient)
        if len(inferred)!=n or any(len(cells)<2 for _,cells in inferred):
            continue
        used.add(key)
        return {'input':inp,'output':out}
    raise RuntimeError('could not sample an unambiguous pair')

def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng=np.random.default_rng(seed)
    nt,ns=_counts(rng,num_train,num_test)
    v,h=(int(x) for x in rng.choice(10,size=2,replace=False))
    puzzle={'train':[],'test':[]}
    used=set()
    cross_index=int(rng.integers(nt))
    familiar_section=('train','test',None)[int(rng.integers(3))]
    familiar_index=int(rng.integers(nt if familiar_section=='train' else ns))
    preferences=('any','isolated','cross','parallel','touch')
    for section,count in (('train',nt),('test',ns)):
        for i in range(count):
            styles=('distinct','cross','parallel','random')
            style='cross' if section=='train' and i==cross_index else styles[int(rng.choice(4,p=(.35,.25,.15,.25) if section=='train' else (.2,.4,.1,.3)))]
            preference=preferences[int(rng.integers(5))] if section=='train' else preferences[int(rng.choice(5,p=(.15,.1,.3,.2,.25)))]
            familiar=section==familiar_section and i==familiar_index
            pair_rng=np.random.default_rng(int(rng.integers(0,2**63-1)))
            try:
                pair=_pair(pair_rng,v,h,used,style,preference,familiar)
            except RuntimeError:
                pair=_pair(pair_rng,v,h,used,style,'any',familiar)
            puzzle[section].append(pair)
    return puzzle

def _inspect(pair):
    assert isinstance(pair,dict) and set(pair)=={'input','output'}
    inp,out=pair['input'],pair['output']
    for grid in (inp,out):
        assert isinstance(grid,np.ndarray) and grid.ndim==2
        assert all(1<=x<=30 for x in grid.shape)
        assert np.issubdtype(grid.dtype,np.integer)
        assert np.all((0<=grid)&(grid<=9))
    assert inp.shape==out.shape and not np.array_equal(inp,out)
    assert _rows(inp) not in _SOURCE
    values,counts=np.unique(inp,return_counts=True)
    bg=int(values[np.argmax(counts)])
    assert list(counts).count(max(counts))==1
    assert np.array_equal(inp==bg,out==bg)
    result=_orientations(inp,bg)
    assert result is not None
    occupied,horizontal=result
    orient=np.full(inp.shape,-1,dtype=np.int8)
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            bit=1 << (r*inp.shape[1]+c)
            if occupied&bit:
                orient[r,c]=0 if horizontal&bit else 1
    bars=_runs(orient)
    assert 2<=len(bars)<=6 and {d for d,_ in bars}=={0,1}
    assert all(len(cells)>=2 for _,cells in bars)
    mapping={}
    for d,cells in bars:
        assert len({int(inp[cell]) for cell in cells})==1
        out_colors={int(out[cell]) for cell in cells}
        assert len(out_colors)==1
        color=next(iter(out_colors))
        assert d not in mapping or mapping[d]==color
        mapping[d]=color
    assert len(set(mapping.values()))==2 and bg not in mapping.values()
    assert any(int(inp[cell])!=mapping[d] for d,cells in bars for cell in cells)
    return bg,mapping,bars

def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert isinstance(puzzle['train'],list) and isinstance(puzzle['test'],list)
    nt,ns=len(puzzle['train']),len(puzzle['test'])
    assert 2<=nt<=10 and 1<=ns<=4 and nt>ns
    keys=set()
    mapping=None
    cross=False
    for section in ('train','test'):
        for pair in puzzle[section]:
            key=(pair['input'].shape,pair['input'].tobytes())
            assert key not in keys
            keys.add(key)
            _,colors,bars=_inspect(pair)
            if mapping is None:
                mapping=colors
            assert colors==mapping
            if section=='train':
                a={int(pair['input'][cells[0]]) for d,cells in bars if d==0}
                b={int(pair['input'][cells[0]]) for d,cells in bars if d==1}
                cross|=bool(a&b)
    assert cross
    for pair in puzzle['test']:
        inp=pair['input']
        values,counts=np.unique(inp,return_counts=True)
        bg=int(values[np.argmax(counts)])
        occupied,horizontal=_orientations(inp,bg)
        predicted=inp.copy()
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                bit=1 << (r*inp.shape[1]+c)
                if occupied&bit:
                    predicted[r,c]=mapping[0 if horizontal&bit else 1]
        assert np.array_equal(predicted,pair['output'])

def _signature(puzzle):
    return tuple(tuple((p['input'].shape,p['input'].tobytes(),p['output'].tobytes()) for p in puzzle[s]) for s in ('train','test'))

def self_test() -> None:
    signatures=set(); train_counts=set(); test_counts=set(); shapes=set(); layouts=set(); bar_counts=set(); lengths=set(); maps=set()
    cross=distinct=unchanged=isolated=touching=0
    familiar_train=familiar_test=binary_ambiguous=same_hue_contact=combined=0
    for seed in range(42):
        puzzle=generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature=_signature(puzzle)
        assert signature==_signature(generate_puzzle(seed))
        signatures.add(signature)
        train_counts.add(len(puzzle['train'])); test_counts.add(len(puzzle['test']))
        pair_shapes=set(); pair_counts=set()
        for section in ('train','test'):
            for pair in puzzle[section]:
                bg,mapping,bars=_inspect(pair)
                inp=pair['input']
                shapes.add(inp.shape); pair_shapes.add(inp.shape)
                bar_counts.add(len(bars)); pair_counts.add(len(bars))
                lengths.update(len(cells) for _,cells in bars)
                maps.add((mapping[0],mapping[1]))
                layouts.add((inp.shape,tuple(bars)))
                hues=[(d,int(inp[cells[0]])) for d,cells in bars]
                cross+=bool({h for d,h in hues if d==0}&{h for d,h in hues if d==1})
                distinct+=len({h for _,h in hues})==len(hues)
                unchanged+=any(h==mapping[d] for d,h in hues) and any(h!=mapping[d] for d,h in hues)
                isolated+=not _contacts(bars)
                touching+=bool(_contacts(bars))
                familiar=inp.shape==(6,6) and len(bars)==4 and all(len(cells)==4 for _,cells in bars)
                if section=='train': familiar_train+=familiar
                else: familiar_test+=familiar
                binary=np.where(inp==bg,0,1).astype(np.int8)
                binary_ambiguous+=_orientations(binary,0) is None
                same_hue_contact+=any(int(inp[a[0]])==int(inp[b[0]]) and
                                      any(abs(r-s)+abs(c-t)==1 for r,c in a for s,t in b)
                                      for j,(_,a) in enumerate(bars) for _,b in bars[j+1:])
                combined+=bool({h for d,h in hues if d==0}&{h for d,h in hues if d==1}) and bool(_contacts(bars)) and any(h==mapping[d] for d,h in hues)
        assert len(pair_shapes)>=2 or len(pair_counts)>=2
    assert len(signatures)==42
    assert len(train_counts)>=4 and len(test_counts)>=3
    assert len(shapes)>=25 and len(layouts)>=100
    assert len(bar_counts)>=4 and len(lengths)>=6 and len(maps)>=12
    assert min(cross,distinct,unchanged,isolated,touching)>0
    assert min(familiar_train,familiar_test,binary_ambiguous,same_hue_contact,combined)>0
    for a,b in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        puzzle=generate_puzzle(271,a,b)
        validate_generated_puzzle(puzzle)
        assert a is None or len(puzzle['train'])==a
        assert b is None or len(puzzle['test'])==b
        assert _signature(puzzle)==_signature(generate_puzzle(271,a,b))
    for a,b in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4)):
        try:
            generate_puzzle(271,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
