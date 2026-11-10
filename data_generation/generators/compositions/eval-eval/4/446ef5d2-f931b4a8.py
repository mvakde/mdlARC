"""Registered textile jigsaws: translation constrained by a two-scale weave.

Only the input is consulted by solve. No parent code or data is imported.
"""
import numpy as np

class Limit(Exception):
    pass

def search(pieces, anchor, point, corner, canvas, stencil, bg, mode='full', limit=60000):
    """Return up to two distinct (shape, origin, picture, positions) answers."""
    p,q=stencil.shape
    area=sum(int((m>=0).sum()) for m,pos in pieces)
    answers={}; nodes=0
    configs=[]
    for m,pos in pieces:
        ys,xs=np.nonzero(m>=0)
        configs.append((m.shape,ys,xs,m[ys,xs],int(xs[0])))
    for h in range(p,canvas[0]+1,p):
        if area%h: continue
        w=area//h
        if w%q or w>canvas[1]: continue
        oy=point[0]-(h-1 if corner[0]=='B' else 0)
        ox=point[1]-(w-1 if corner[1]=='R' else 0)
        if oy<0 or ox<0 or oy+h>canvas[0] or ox+w>canvas[1]: continue
        if any(ph>h or pw>w for (ph,pw),*rest in configs): continue
        grid=np.full((h,w),-1,int)
        blocks=np.full((h//p,w//q),-1,int)
        positions=[None]*len(pieces)
        bits=[]
        for (ph,pw),ys,xs,vals,first in configs:
            bits.append(sum(1<<int(y*w+x) for y,x in zip(ys,xs)))
        full=(1<<area)-1

        def put(i,r,c):
            (ph,pw),ys,xs,vals,first=configs[i]
            if r<0 or c<0 or r+ph>h or c+pw>w: return None
            yy,xx=ys+r,xs+c
            stamp=stencil[yy%p,xx%q]
            opaque=stamp!=bg
            if mode not in ('no_stencil','geometry'):
                if np.any(vals[opaque]!=stamp[opaque]): return None
                if np.any(np.isin(vals[~opaque], stencil[stencil!=bg])): return None
            changes=[]
            if mode not in ('no_blocks','geometry'):
                for y,x,v in zip(yy[~opaque]//p,xx[~opaque]//q,vals[~opaque]):
                    old=blocks[y,x]
                    if old>=0 and old!=v:
                        for by,bx in changes: blocks[by,bx]=-1
                        return None
                    if old<0:
                        blocks[y,x]=v;changes.append((y,x))
            grid[yy,xx]=vals
            positions[i]=(r+oy,c+ox)
            return yy,xx,changes

        def undo(data):
            yy,xx,changes=data
            grid[yy,xx]=-1
            for y,x in changes: blocks[y,x]=-1

        ar,ac=int(pieces[anchor][1][0]-oy),int(pieces[anchor][1][1]-ox)
        initial=put(anchor,ar,ac)
        if initial is None: continue
        occ=bits[anchor]<<(ar*w+ac)

        def visit(occupied,remaining):
            nonlocal nodes
            nodes+=1
            if nodes>limit: raise Limit()
            if not remaining:
                key=(h,w,oy,ox,grid.tobytes())
                answers.setdefault(key,((h,w),(oy,ox),grid.copy(),tuple(positions)))
                return len(answers)>1
            free=full^occupied
            loc=(free&-free).bit_length()-1
            r,c=divmod(loc,w)
            for i in remaining:
                (ph,pw),ys,xs,vals,first=configs[i]
                x=c-first
                if x<0 or r+ph>h or x+pw>w: continue
                b=bits[i]<<(r*w+x)
                if b&occupied: continue
                data=put(i,r,x)
                if data is None: continue
                stop=visit(occupied|b,tuple(j for j in remaining if i!=j))
                undo(data)
                if stop: return True
            return False
        if visit(occ,tuple(i for i in range(len(pieces)) if i!=anchor)): break
    return list(answers.values()),nodes

def comp(mask):
    unseen=set(map(tuple,np.argwhere(mask)));out=[]
    while unseen:
        v=min(unseen);unseen.remove(v);todo=[v]
        for y,x in todo:
            for t in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if t in unseen: unseen.remove(t);todo.append(t)
        out.append(todo)
    return out

def partition(rng,h,w,n,steps=False):
    rects=[(0,0,h,w)]
    for _ in range(n-1):
        choices=[(i,d) for i,(r,c,hh,ww) in enumerate(rects) for d,z in enumerate((hh,ww)) if z>=4]
        if not choices: return None
        i,d=choices[int(rng.integers(len(choices)))];r,c,hh,ww=rects.pop(i)
        cut=int(rng.integers(2,(hh if d==0 else ww)-1))
        rects.extend([(r,c,cut,ww),(r+cut,c,hh-cut,ww)] if d==0 else [(r,c,hh,cut),(r,c+cut,hh,ww-cut)])
    lab=np.full((h,w),-1,int)
    for i,(r,c,hh,ww) in enumerate(rects): lab[r:r+hh,c:c+ww]=i
    if steps:
        opts=[]
        for y in range(1,h-1):
            for x in range(1,w-1):
                for yy,xx in ((y+1,x),(y,x+1)):
                    if lab[y,x]!=lab[yy,xx]: opts.append((y,x,int(lab[yy,xx])))
        rng.shuffle(opts)
        for y,x,dest in opts:
            old=int(lab[y,x]);lab[y,x]=dest
            if len(comp(lab==old))==len(comp(lab==dest))==1: break
            lab[y,x]=old
    return lab


_CORNERS=('TL','TR','BL','BR')


def _grid(value):
    g=np.asarray(value)
    assert g.ndim==2 and all(1<=d<=30 for d in g.shape), 'invalid shape'
    assert np.issubdtype(g.dtype,np.integer), 'integer grids required'
    assert np.all((g>=0)&(g<=9)), 'invalid color'
    return g.astype(int,copy=False)


def _key(g):
    return (g.shape,g.astype(np.uint8).tobytes())


def _marker_cells(point,corner):
    r,c=point
    dy=-1 if corner[0]=='T' else 1
    dx=-1 if corner[1]=='L' else 1
    return ((r+dy,c),(r,c+dx),(r+dy,c+dx))


def _extract(value):
    g=_grid(value)
    bg=int(g[0,0])
    assert np.all(g[0]==bg) and np.all(g[-1]==bg)
    assert np.all(g[:,0]==bg) and np.all(g[:,-1]==bg), 'blank outer border required'
    frames=[]
    for color in np.unique(g):
        if color==bg: continue
        cells=np.argwhere(g==color)
        r,c=map(int,cells.min(0));y,x=map(int,cells.max(0)+1)
        h,w=y-r,x-c
        if not (4<=h<=6 and 4<=w<=6): continue
        ring=np.ones((h,w),bool);ring[1:-1,1:-1]=False
        if np.array_equal(g[r:y,c:x]==color,ring):
            frames.append((int(color),(r,c,y,x),g[r+1:y-1,c+1:x-1].copy()))
    assert len(frames)==1, 'one isolated framed swatch required'
    frame,box,stencil=frames[0]
    assert np.any(stencil==bg) and np.any(stencil!=bg)
    assert frame not in stencil
    r,c,y,x=box
    payload=(g!=bg).copy();payload[r:y,c:x]=False
    markers=[]
    for color in np.unique(g):
        if color in (bg,frame): continue
        cells=np.argwhere(g==color)
        if len(cells)!=3: continue
        a,b=map(int,cells.min(0));z,w=map(int,cells.max(0))
        if z-a!=1 or w-b!=1: continue
        missing={(a,b),(a,w),(z,b),(z,w)}-set(map(tuple,cells.tolist()))
        point=missing.pop()
        if not payload[point] or g[point]==color: continue
        corner={(1,1):'TL',(1,0):'TR',(0,1):'BL',(0,0):'BR'}[(point[0]-a,point[1]-b)]
        markers.append((int(color),point,corner))
    assert len(markers)==1, 'one unambiguous external L required'
    marker,point,corner=markers[0]
    assert marker not in stencil
    payload[g==marker]=False
    pieces=[];anchor=None
    for cells in comp(payload):
        coords=np.array(cells);r,c=map(int,coords.min(0));y,x=map(int,coords.max(0)+1)
        m=np.full((y-r,x-c),-1,int)
        for yy,xx in cells: m[yy-r,xx-c]=g[yy,xx]
        if point in cells: anchor=len(pieces)
        pieces.append((m,(r,c)))
    assert 3<=len(pieces)<=6 and anchor is not None
    m,(r,c)=pieces[anchor]
    expected=(r+(m.shape[0]-1 if corner[0]=='B' else 0),
              c+(m.shape[1]-1 if corner[1]=='R' else 0))
    assert point==expected, 'marker must point to anchor bounding corner'
    # A clear gap separates each component's bounding box from every other one.
    rects=[(*pos,*m.shape) for m,pos in pieces]+[(box[0],box[1],box[2]-box[0],box[3]-box[1])]
    for i,a in enumerate(rects):
        for b in rects[i+1:]: assert _gap(a,b)
    return dict(grid=g,bg=bg,marker=marker,frame=frame,box=box,stencil=stencil,
                point=point,corner=corner,pieces=pieces,anchor=anchor)


def _infer(info,mode='full'):
    answers,nodes=search(info['pieces'],info['anchor'],info['point'],info['corner'],
                         info['grid'].shape,info['stencil'],info['bg'],mode)
    return answers,nodes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Find the unique unrotated rectangular assembly satisfying both lattices."""
    info=_extract(input_grid)
    answers,_=_infer(info)
    assert len(answers)==1, 'no unique registered textile assembly'
    (h,w),(r,c),picture,_=answers[0]
    output=np.full(info['grid'].shape,info['bg'],int)
    output[r:r+h,c:c+w]=picture
    return output


def _gap(a,b):
    r,c,h,w=a;y,x,hh,ww=b
    return r+h<y or y+hh<r or c+w<x or x+ww<c


def _stamp(rng,p,q):
    rr,cc=np.indices((p,q))
    style=int(rng.integers(6))
    if style==0: mask=rng.random((p,q))<rng.uniform(.20,.70)
    elif style==1: mask=(rr+cc+int(rng.integers(2)))%2==0
    elif style==2: mask=rr==int(rng.integers(p))
    elif style==3: mask=cc==int(rng.integers(q))
    elif style==4:
        mask=np.zeros((p,q),bool)
        mask.ravel()[rng.choice(p*q,int(rng.integers(1,p*q)),replace=False)]=True
    else:
        mask=(rr==0)|(rr==p-1)|(cc==0)|(cc==q-1)
        mask &= rng.random((p,q))>.30
    if not mask.any(): mask.ravel()[int(rng.integers(p*q))]=True
    if mask.all(): mask.ravel()[int(rng.integers(p*q))]=False
    st=np.zeros((p,q),int)
    inks=int(rng.integers(1,min(2,int(mask.sum()))+1))
    vals=rng.integers(1,inks+1,int(mask.sum()))
    vals[:inks]=np.arange(1,inks+1);rng.shuffle(vals);st[mask]=vals
    return st


def _carrier(rng,h,w):
    k=int(rng.integers(2,5))
    rr,cc=np.indices((h,w));style=int(rng.integers(5))
    if style==0: a=rng.integers(k,size=(h,w))
    elif style==1: a=(rr+cc)%k
    elif style==2: a=np.tile(rng.integers(k,size=w),(h,1))
    elif style==3: a=np.tile(rng.integers(k,size=(h,1)),(1,w))
    else:
        a=np.zeros((h,w),int)
        for i in range(1,k):
            r,c=int(rng.integers(h)),int(rng.integers(w))
            a[r:,c:]=i
    if len(np.unique(a))<2:
        a[int(rng.integers(h)),int(rng.integers(w))]=(int(a[0,0])+1)%k
    return a+3


def _make_pieces(labels,picture):
    pieces=[]
    for i in range(int(labels.max())+1):
        coords=np.argwhere(labels==i);r,c=map(int,coords.min(0));y,x=map(int,coords.max(0)+1)
        m=np.where(labels[r:y,c:x]==i,picture[r:y,c:x],-1)
        pieces.append((m,(r,c)))
    return pieces


def _scatter(rng,pieces,anchor,origin,point,corner,stencil,frame,marker,bg,canvas):
    H,W=canvas;ph,pw=stencil.shape
    am,apos=pieces[anchor];fixed=(origin[0]+apos[0],origin[1]+apos[1])
    markers=_marker_cells(point,corner)
    for _ in range(10):
        rects=[(*fixed,*am.shape)]
        # Marker cells are protected with an extra blank-cell halo.
        protected=[(r,c,1,1) for r,c in markers]
        positions=[None]*len(pieces);positions[anchor]=fixed
        objects=[('key',ph+2,pw+2)]
        others=[i for i in range(len(pieces)) if i!=anchor]
        rng.shuffle(others);others.sort(key=lambda i:-pieces[i][0].size)
        objects += [(i,*pieces[i][0].shape) for i in others]
        good=True;keypos=None
        for obj,h,w in objects:
            options=[(r,c) for r in range(1,H-h) for c in range(1,W-w)]
            found=False
            for idx in rng.permutation(len(options)):
                r,c=options[int(idx)];rect=(r,c,h,w)
                if all(_gap(rect,b) for b in rects+protected):
                    rects.append(rect)
                    if obj=='key': keypos=(r,c)
                    else: positions[obj]=(r,c)
                    found=True;break
            if not found: good=False;break
        if not good: continue
        g=np.full(canvas,bg,int)
        for (m,_),(r,c) in zip(pieces,positions):
            sub=g[r:r+m.shape[0],c:c+m.shape[1]];sub[m>=0]=m[m>=0]
        r,c=keypos
        g[r:r+ph+2,c:c+pw+2]=frame
        g[r+1:r+ph+1,c+1:c+pw+1]=stencil
        for loc in markers: g[loc]=marker
        return g
    return None


def _render_candidate(rng,labels,carrier,stencil,corner):
    p,q=stencil.shape;mh,mw=carrier.shape;h,w=labels.shape
    base=np.repeat(np.repeat(carrier,p,0),q,1)
    native=np.tile(stencil,(mh,mw))
    picture=np.where(native!=0,native,base)
    # Reassign all semantic roles per example, including literal black as ink.
    palette=rng.permutation(10)
    bg,marker,frame=map(int,palette[[0,7,8]])
    picture=palette[picture];stencil=palette[stencil]
    pieces=_make_pieces(labels,picture)
    loc=(h-1 if corner[0]=='B' else 0,w-1 if corner[1]=='R' else 0)
    anchor=int(labels[loc])
    # Choose the absolute placement independently of scattering and swatch location.
    H=min(30,max(h+5,int(np.ceil(np.sqrt(h*w*3)))+int(rng.integers(4,8))))
    W=min(30,max(w+5,int(np.ceil(h*w*3/max(H,1)))+int(rng.integers(5,9))))
    H=max(H,p+7);W=max(W,q+7)
    for _ in range(8):
        origin=(int(rng.integers(2,H-h-1)),int(rng.integers(2,W-w-1)))
        point=(origin[0]+loc[0],origin[1]+loc[1])
        g=_scatter(rng,pieces,anchor,origin,point,corner,stencil,frame,marker,bg,(H,W))
        if g is None: continue
        try:
            info=_extract(g);answers,_=_infer(info)
        except (AssertionError,Limit): continue
        if len(answers)!=1: return None
        expected=np.full((H,W),bg,int);expected[origin[0]:origin[0]+h,origin[1]:origin[1]+w]=picture
        shape,pos,recovered,_=answers[0]
        if shape!=(h,w) or pos!=origin or not np.array_equal(recovered,picture): return None
        return {'input':g,'output':expected}
    return None


def _candidate(rng,periods,coarse_shapes,counts=(3,4,5,6)):
    p,q=periods[int(rng.integers(len(periods)))]
    mh,mw=coarse_shapes[int(rng.integers(len(coarse_shapes)))]
    h,w=p*mh,q*mw
    labels=partition(rng,h,w,int(rng.choice(counts)),bool(rng.integers(2)))
    if labels is None: return None
    return _render_candidate(rng,labels,_carrier(rng,mh,mw),_stamp(rng,p,q),str(rng.choice(_CORNERS)))


def _evidence(info):
    cover=set()
    for mode in ('geometry','no_blocks','no_stencil'):
        answers,_=_infer(info,mode)
        if len(answers)>1: cover.add(mode)
    return cover


def _fallback(rng):
    """Two proved geometry/texture witnesses, recolored and freely scattered.

    Used only after the bounded teaching pool fails; never hidden answer lookup.
    """
    rows1=('003334444444','003234444444','112222222222','112222222222')
    rows2=('112233','112233','112233','112244','112244','112244','000000','000000','000000')
    fixtures=((rows1,((3,3,3),(5,4,4)),((2,1,0,1),(0,0,0,0))),
              (rows2,((4,4),(4,5),(4,5)),((1,0,2),(1,1,2),(1,0,0))))
    pairs=[]
    for index,(rows,car,st) in enumerate(fixtures):
        labels=np.array([[int(c) for c in r] for r in rows])
        car=np.array(car);st=np.array(st)
        # Reflect the whole first construction before cutting, not its pieces.
        # Thus the witnesses also demonstrate two different marker corners.
        corner='TR'
        if index==0:
            labels=labels[:,::-1].copy();car=car[:,::-1].copy();st=st[:,::-1].copy()
            corner='TL'
        pair=None
        for _ in range(30):
            pair=_render_candidate(rng,labels,car,st,corner)
            if pair is not None: break
        if pair is None: raise RuntimeError('failed to scatter teaching witness')
        pairs.append(pair)
    return pairs


def _count(value,low,high,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(name+' must be an integer')
    value=int(value)
    if not low<=value<=high: raise ValueError(name+' out of range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt=_count(num_train,2,10,'num_train');ne=_count(num_test,1,4,'num_test')
    if nt<=ne: raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    rng=np.random.default_rng(seed)
    periods=[(p,q) for p in range(2,5) for q in range(2,5)]
    small=[(h,w) for h in (2,3) for w in (2,3)]
    pool=[];train=None;seen=set()
    required={'geometry','no_blocks','no_stencil'}
    # Choose demonstrations by their input-visible evidence, never pair index.
    for _ in range(180):
        pair=_candidate(rng,periods,small)
        if pair is None or _key(pair['input']) in seen: continue
        info=_extract(pair['input']);cover=_evidence(info)
        for old,oldinfo,oldcover in pool:
            if (required<=cover|oldcover and info['stencil'].shape!=oldinfo['stencil'].shape
                    and info['corner']!=oldinfo['corner']
                    and any(np.any(m<0) for details in (info,oldinfo) for m,pos in details['pieces'])):
                train=[old,pair];break
        pool.append((pair,info,cover));seen.add(_key(pair['input']))
        if train is not None: break
    if train is None: train=_fallback(rng)
    puzzle={'train':train,'test':[]};seen={_key(p['input']) for p in train}
    # Additional demonstrations draw from the same structural vocabulary.
    for _extra in range(nt-2):
        pair=None
        for _ in range(200):
            trial=_candidate(rng,periods,small)
            if trial is not None and _key(trial['input']) not in seen:
                pair=trial;break
        if pair is None:
            for _ in range(20):
                trial=_fallback(rng)[int(rng.integers(2))]
                if _key(trial['input']) not in seen: pair=trial;break
        if pair is None: raise RuntimeError('bounded training pool exhausted')
        puzzle['train'].append(pair);seen.add(_key(pair['input']))
    used={_extract(p['input'])['stencil'].shape for p in puzzle['train']}
    regime=int(rng.integers(3))
    if regime==0:
        test_shapes=[(4,2),(4,3),(4,4),(2,4),(3,4)]
        test_periods=periods
    elif regime==1:
        test_shapes=[(2,4),(2,5),(4,2),(5,2)]
        test_periods=periods
    else:
        test_periods=[s for s in periods if s not in used]
        test_shapes=small
        if not test_periods:
            test_periods=periods;test_shapes=[(2,5),(5,2)]
    for _extra in range(ne):
        pair=None
        for _ in range(400):
            trial=_candidate(rng,test_periods,test_shapes)
            if trial is not None and _key(trial['input']) not in seen:
                pair=trial;break
        if pair is None: raise RuntimeError('bounded test candidate pool exhausted')
        puzzle['test'].append(pair);seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train']);rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    g=_grid(pair['input']);out=_grid(pair['output']);info=_extract(g)
    assert out.shape==g.shape
    answers,nodes=_infer(info)
    assert len(answers)==1, 'ambiguous or impossible assembly'
    (h,w),(r,c),pic,positions=answers[0]
    expected=np.full(g.shape,info['bg'],int);expected[r:r+h,c:c+w]=pic
    assert np.array_equal(expected,out), 'incorrect assembled output'
    assert not np.any(out==info['marker']) and not np.any(out==info['frame'])
    assert all(out[p]==info['bg'] for p in _marker_cells(info['point'],info['corner']))
    assert positions[info['anchor']]==info['pieces'][info['anchor']][1]
    occupied=np.zeros((h,w),int);labels=np.full((h,w),-1,int)
    payload_values=[]
    for i,((m,old),(yy,xx)) in enumerate(zip(info['pieces'],positions)):
        yy-=r;xx-=c;mask=m>=0
        assert len(comp(mask))==1
        assert np.array_equal(pic[yy:yy+m.shape[0],xx:xx+m.shape[1]][mask],m[mask])
        occupied[yy:yy+m.shape[0],xx:xx+m.shape[1]][mask]+=1
        labels[yy:yy+m.shape[0],xx:xx+m.shape[1]][mask]=i
        payload_values.extend(m[mask].tolist())
    assert np.all(occupied==1), 'must use every cell once without gaps'
    assert np.array_equal(np.bincount(payload_values,minlength=10),np.bincount(pic.ravel(),minlength=10))
    p,q=info['stencil'].shape
    tile=info['stencil'][np.arange(h)[:,None]%p,np.arange(w)[None,:]%q]
    assert np.array_equal(pic[tile!=info['bg']],tile[tile!=info['bg']])
    carrier=np.empty((h//p,w//q),int)
    for by in range(h//p):
        for bx in range(w//q):
            values=pic[by*p:(by+1)*p,bx*q:(bx+1)*q][info['stencil']==info['bg']]
            assert len(np.unique(values))==1
            carrier[by,bx]=values[0]
    assert len(np.unique(carrier))>=2, 'nontrivial coarse image required'
    assert not set(np.unique(carrier))&set(info['stencil'][info['stencil']!=info['bg']])
    unequal=False
    for a,b,la,lb in ((pic[1:],pic[:-1],labels[1:],labels[:-1]),
                      (pic[:,1:],pic[:,:-1],labels[:,1:],labels[:,:-1])):
        unequal |= bool(np.any((la!=lb)&(a!=b)))
    info.update(shape=(h,w),coarse_shape=carrier.shape,carrier=carrier,labels=labels,
                positions=positions,nodes=nodes,unequal_seam=unequal)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2<=len(puzzle['train'])<=10 and 1<=len(puzzle['test'])<=4
    assert len(puzzle['train'])>len(puzzle['test'])
    seen=set();infos={'train':[],'test':[]};evidence=set()
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            key=_key(_grid(pair['input']));assert key not in seen;seen.add(key)
            info=_analyze(pair);infos[split].append(info)
            if split=='train': evidence|=_evidence(info)
    assert {'geometry','no_blocks','no_stencil'}<=evidence, 'missing teaching evidence'
    assert len({i['stencil'].shape for i in infos['train']})>=2
    assert len({i['corner'] for i in infos['train']})>=2
    assert any(np.any(m<0) for i in infos['train'] for m,pos in i['pieces']), 'demonstrate notched occupancy'
    assert any(i['unequal_seam'] for i in infos['train']), 'show that equal edges are unnecessary'
    # Test members must actually exercise the promised structural extrapolation.
    train_periods={i['stencil'].shape for i in infos['train']}
    train_shapes={i['coarse_shape'] for i in infos['train']}
    for i in infos['test']:
        assert i['stencil'].shape not in train_periods or i['coarse_shape'] not in train_shapes


def _normalized(g):
    roles={};return (g.shape,bytes(roles.setdefault(int(c),len(roles)) for c in g.flat))


def self_test():
    """Bounded tests: 42 seeds, meaningful ablations, counts and corruption."""
    puzzles=[generate_puzzle(seed) for seed in range(42)]
    records=[];fingerprints=set()
    for puzzle in puzzles:
        fingerprints.add(tuple(_key(p['input']) for s in ('train','test') for p in puzzle[s]))
        for split in ('train','test'):
            for pair in puzzle[split]:
                info=_analyze(pair);records.append((split,info))
                assert np.array_equal(solve(pair['input']),pair['output'])
    assert len(fingerprints)==42
    assert {i['corner'] for s,i in records}==set(_CORNERS)
    assert len({i['bg'] for s,i in records})==10
    assert len({i['grid'].shape for s,i in records})>=25
    assert len({i['shape'] for s,i in records})>=25
    assert {len(i['pieces']) for s,i in records}=={3,4,5,6}
    assert len({_normalized(i['stencil']) for s,i in records})>=35
    assert len({_normalized(i['carrier']) for s,i in records})>=30
    assert len({_normalized(i['labels']) for s,i in records})>=80
    assert any(np.any(m<0) for s,i in records for m,pos in i['pieces'])
    assert any(i['stencil'].shape[0]!=i['stencil'].shape[1] for s,i in records)
    assert any(i['bg']!=0 and np.any(i['carrier']==0) for s,i in records)
    assert any(i['stencil'][0,0]==i['bg'] for s,i in records)
    assert any(i['stencil'][0,0]!=i['bg'] for s,i in records)
    # Reversed segmentation order must not choose a different answer.
    info=_extract(puzzles[0]['test'][0]['input'])
    before,_=_infer(info)
    info['pieces']=list(reversed(info['pieces']))
    info['anchor']=len(info['pieces'])-1-info['anchor']
    after,_=_infer(info)
    assert before[0][:2]==after[0][:2] and np.array_equal(before[0][2],after[0][2])
    again=generate_puzzle(0)
    assert all(_key(a[k])==_key(b[k]) for s in ('train','test') for a,b in zip(puzzles[0][s],again[s]) for k in ('input','output'))
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p=generate_puzzle(9000+10*a+b,a,b)
            assert len(p['train'])==a and len(p['test'])==b
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3,1.5),(None,1),(4,None)):
        try: generate_puzzle(0,a,b)
        except ValueError: pass
        else: raise AssertionError('illegal counts accepted')
    for j,pair in enumerate(_fallback(np.random.default_rng(173))):
        assert np.array_equal(solve(pair['input']),pair['output'])
        assert ('no_blocks' if j==0 else 'no_stencil') in _evidence(_extract(pair['input']))
    # The key cannot determine the carrier. Swap two carrier inks only in the
    # fragments: the same swatch, geometry and anchor now give a different answer.
    pair=puzzles[0]['train'][0];info=_analyze(pair)
    a,b=map(int,np.unique(info['carrier'])[:2]);changed=pair['input'].copy()
    changed[pair['input']==a]=b;changed[pair['input']==b]=a
    predicted=solve(changed)
    expected=pair['output'].copy()
    expected[pair['output']==a]=b;expected[pair['output']==b]=a
    assert np.array_equal(predicted,expected) and not np.array_equal(predicted,pair['output'])
    r,c,y,x=info['box'];assert np.array_equal(changed[r:y,c:x],pair['input'][r:y,c:x])
    # Four whole macro cells can be permuted freely. A tie is rejected, not
    # resolved by piece extraction order or the first successful packing.
    ambiguous=np.zeros((18,24),int);st=np.array([[1,0,0],[0,0,0]])
    ambiguous[2:6,18:23]=8;ambiguous[3:5,19:22]=st
    for color,(r,c) in zip((3,4,5,6),((5,3),(2,10),(10,3),(10,11))):
        ambiguous[r:r+2,c:c+3]=np.where(st==0,color,st)
    for loc in _marker_cells((5,3),'TL'): ambiguous[loc]=7
    try: solve(ambiguous)
    except AssertionError: pass
    else: raise AssertionError('ambiguous macro-cell permutation accepted')
    # Incorrect labels and duplicates must fail, without generation metadata.
    for mode in ('color','marker','shift','duplicate'):
        damaged={s:[{k:v.copy() for k,v in pair.items()} for pair in puzzles[0][s]] for s in ('train','test')}
        pair=damaged['test'][0];info=_extract(pair['input'])
        if mode=='color':
            y,x=np.argwhere(pair['output']!=info['bg'])[0]
            pair['output'][y,x]=(int(pair['output'][y,x])+1)%10
        elif mode=='marker': pair['output'][_marker_cells(info['point'],info['corner'])[0]]=info['marker']
        elif mode=='shift': pair['output']=np.roll(pair['output'],1,0)
        else: damaged['train'][1]={k:v.copy() for k,v in damaged['train'][0].items()}
        try: validate_generated_puzzle(damaged)
        except AssertionError: pass
        else: raise AssertionError('corruption accepted: '+mode)


if __name__=='__main__':
    self_test()
    print('self_test passed (42 seeds and all legal explicit counts)')
