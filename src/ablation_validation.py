"""Bounded native-CUDA validation before executing a changed science recipe.

Runs inside the ordinary launch process, in its own disposable CUDA context.
No checkpoint, optimizer state, data, or RNG is shared with the experiment.
"""
import json
import sys
from pathlib import Path

import torch

from common import encode_context, compute_positions_3d
from evaluate import batched_greedy_generate
from tinytransformer import TinyTransformer, TinyTransformerConfig
from train import _enable_compile_scalar_capture


def validate(output, version=2):
    assert torch.cuda.is_available()
    _enable_compile_scalar_capture()
    tokens, sep = encode_context([
        {'input': [[1,2]], 'output': [[2,1]]},
        {'input': [[3]], 'output': [[4]]}],
        {'input': [[5,6]], 'output': [[6,5]]})
    pos = compute_positions_3d(tokens[None]).cuda()
    padded = dict(input_ids=tokens[None].cuda(), example_ids=torch.tensor([0],device='cuda'),
        sep_indices=torch.tensor([sep],device='cuda',dtype=torch.int32),positions_3d=pos)
    packed = dict(padded,input_ids=tokens.cuda(),positions_3d=pos[0],
        cu_seqlens=torch.tensor([0,len(tokens)],device='cuda',dtype=torch.int32),max_seqlen=len(tokens))
    report=[]
    # FA4 is validated on the split two-lookup recipe it trains (head dim 64).
    variants = [(True, False)] if version == 4 else [(False,False),(True,False),(True,True)]
    for split, simplex in variants:
        torch.manual_seed(42)
        m=TinyTransformer(TinyTransformerConfig(d_model=128 if version == 4 else 96,
            n_heads=2 if version == 4 else 4,d_ff=256 if version == 4 else 192,n_layers=2,
            num_examples=2,num_task_embedding_latents=3,task_embeddings=False,FA=version,
            dropout=0,embedding_dropout=0,attention_dropout=0,n_predict=4,
            attention_pattern='pair_group',split_target_demo_attention=split,
            pair_identity_vertices=11 if simplex else 0)).cuda().bfloat16().eval()
        assert m._has_varlen_flash_attention(), 'Native FlashAttention is required'
        a,b=dict(padded),dict(packed)
        if simplex:
            z=pos[...,2]
            ids=torch.where((z>0)&(z<=4),(z+1)//2,0).long()
            a['pair_identity_ids']=ids;b['pair_identity_ids']=ids[0]
        with torch.no_grad(): expected=m(**a)['logits'][0]
        compiled=torch.compile(m)
        actual=compiled(**b)
        torch.testing.assert_close(actual['logits'],expected,rtol=.03,atol=.03)
        actual['loss'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
        prompt=tokens[:sep+1].tolist()
        generated=batched_greedy_generate(m,[prompt],[0],torch.device('cuda'),max_new_tokens=4,
            pair_identity_ids=[ids[0,:sep+1].tolist()] if simplex else None)
        assert len(generated)==1 and len(generated[0])>len(prompt)
        report.append(dict(FA=version,split_target_demo_attention=split,simplex=simplex,
                           finite_loss=float(actual['loss'].detach()),native_packed_backward=True,
                           compiled_generation=True))
        Path(output).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),checks=report),indent=2)+'\n')
        del compiled,m,actual
        torch._dynamo.reset();torch.cuda.empty_cache()
    Path(output).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),checks=report),indent=2)+'\n')
    print('Native CUDA ablation validation passed.',flush=True)


if __name__=='__main__': validate(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 2)
