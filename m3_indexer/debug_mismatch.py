import sys, random, torch
sys.path.insert(0, "/flydsl")
from m3_indexer.vllm_ops import import_ops
import_ops("index_bf16")
from index_bf16.host import index_score_prefill
from vllm.models.minimax_m3.amd.ops.index_topk import minimax_m3_index_score, minimax_m3_index_topk
BLK=128; D=128; TOPK=16
rng = random.Random(0)
q_lens=[rng.randint(6554, 8192) for _ in range(4)]; ctxs=[rng.randint(500000, 800000) for _ in range(4)]
g = torch.Generator(device="cuda"); g.manual_seed(4)
seq_lens=[c+q for c,q in zip(ctxs,q_lens)]; nblocks=[-(-s//BLK) for s in seq_lens]; pool=sum(nblocks)+8
cache=(torch.randn(pool,BLK,D,device="cuda",generator=g)*0.1).to(torch.bfloat16)
bt=torch.zeros(len(q_lens),max(nblocks),dtype=torch.int32,device="cuda")
perm=torch.randperm(pool,device="cuda",generator=g).to(torch.int32); o=0
for i,n in enumerate(nblocks): bt[i,:n]=perm[o:o+n]; o+=n
total_q=sum(q_lens); q=(torch.randn(total_q,1,D,device="cuda",generator=g)*0.1).to(torch.bfloat16)
cu=torch.tensor([0]+list(torch.cumsum(torch.tensor(q_lens),0)),dtype=torch.int32,device="cuda")
t=lambda x: torch.tensor(x,dtype=torch.int32,device="cuda")
args=(q,cache,bt,cu,t(seq_lens),t(ctxs),max(q_lens),max(seq_lens),1)
ref=minimax_m3_index_score(*args); out=index_score_prefill(*args); torch.cuda.synchronize()
tk_ref=minimax_m3_index_topk(ref,cu,t(ctxs),max(q_lens),TOPK,0,1)
tk_ref2=minimax_m3_index_topk(ref,cu,t(ctxs),max(q_lens),TOPK,0,1)
tk_out=minimax_m3_index_topk(out,cu,t(ctxs),max(q_lens),TOPK,0,1)
tk_out2=minimax_m3_index_topk(out,cu,t(ctxs),max(q_lens),TOPK,0,1)
print("ref topk deterministic:", torch.equal(tk_ref,tk_ref2), " out topk deterministic:", torch.equal(tk_out,tk_out2))
mism=(tk_ref!=tk_out).any(dim=-1)[0].nonzero().flatten().tolist(); print("mismatch rows:", mism)
for r in mism[:4]:
    b=int((cu[1:]<=r).sum()); rl=r-int(cu[b]); vb=(ctxs[b]+rl+BLK)//BLK
    a=set(tk_out[0,r].tolist()); e=set(tk_ref[0,r].tolist()); diff=sorted(x for x in (a^e) if x>=0)
    print(f"row {r} req {b} local {rl} prefix {ctxs[b]} valid_blocks {vb} S={ref.shape[2]}")
    print("  ref topk:", sorted(tk_ref[0,r].tolist())); print("  out topk:", sorted(tk_out[0,r].tolist()))
    for x in diff: print(f"   blk {x}: ref {ref[0,r,x].item():.8f} out {out[0,r,x].item():.8f}  in_ref={x in e} in_out={x in a}")
    dall=(ref[0,r,:vb]-out[0,r,:vb]).abs(); print("  max|d| valid cols:", dall.max().item(), " argmax:", int(dall.argmax()))
    # are there other columns (>= vb) that differ?
    d2=(ref[0,r,vb:]-out[0,r,vb:]); fin=torch.isfinite(ref[0,r,vb:])&torch.isfinite(out[0,r,vb:]); print("  cols>=vb: both finite:", int(fin.sum()), " ref finite:", int(torch.isfinite(ref[0,r,vb:]).sum()), " out finite:", int(torch.isfinite(out[0,r,vb:]).sum()))
    # 16th vs 17th score gap in ref
    v=ref[0,r,:vb].clone(); v[vb-1]=float('inf'); srt=torch.sort(v,descending=True).values; print("  ref sorted 14..18:", [f"{x:.7f}" for x in srt[13:19].tolist()])
print("=== positional order of the mismatched rows ===")
for r in mism[:4]:
    print("row", r)
    print("  ref:", [(int(i), f"{ref[0,r,i].item():.7f}") for i in tk_ref[0,r].tolist()])
    print("  out:", [(int(i), f"{out[0,r,i].item():.7f}") for i in tk_out[0,r].tolist()])
