# 38% on ARC-AGI-1: trained from scratch for just ~$0.60 
- Takes <2hrs on a 5090
- Uses a standard tranformer
- 75M parameters

Deploy:
- rent a 5090, upload the interactive run notebook or runscript, run it

---
## Self supervised compression on ARC

Every DL approach on ARC today trains a supervised algorithm (other than compressARC)

I think this is suboptimal.  
A self-supervised compression step will obviously perform better:
- There is new information in the input grids and private puzzles that is currently uncompressed
- Test grids have distribution shifts. Compression will push these grids into distribution

Implementation details: [New pareto frontier on ARC-AGI](https://mvakde.github.io/blog/new-pareto-frontier-arc-agi/)
For why I chose these specific implementations, read my blog on [Why all ARC solvers fail today](https://mvakde.github.io/blog/why-all-ARC-solvers-fail-today/)

## Details
**UPDATED: (details to follow)**
Performance: **38%** on ARC-1 public eval
Total compute cost: **~$0.60**  (<2hrs on a 5090 rented on vast.ai)

**Old:**
Performance: 27.5% on ARC-1 public eval
Total Compute cost: $1.8 (<3hrs on an A100 rented on Google Colab)

50% should be possible with the next research ideas