#!/usr/bin/env python3
"""Analyse run files: J per generated token by condition; paired differences with bootstrap CIs;
regime sanity (clock, temperature); trace statistics (spectral entropy via FFT, LZ76, slope)
with paired CIs; optional plot. Warm-up handling: --drop-first N (per file) and/or --drop-rep R."""
import argparse, json, glob, collections, random, os, math
p=argparse.ArgumentParser(); p.add_argument("--runs",default="runs/")
p.add_argument("--drop-first",type=int,default=0,help="discard the first N requests of each run file (warm-up until plateau)")
p.add_argument("--drop-rep",type=int,default=0,help="discard repetitions below this index")
p.add_argument("--no-trace",action="store_true",help="skip trace statistics (fast)"); p.add_argument("--boot",type=int,default=2000); a=p.parse_args()
R=[]
for f in sorted(glob.glob(os.path.join(a.runs,"**","run_*.jsonl"),recursive=True)):
    rows=[json.loads(l) for l in open(f)]; R+=rows[a.drop_first:]
R=[r for r in R if r["rep"]>=a.drop_rep]
if not R: raise SystemExit("no measurements")
mean=lambda xs: sum(xs)/len(xs)
print(f"{len(R)} measurements after dropping the first {a.drop_first} requests per file and reps < {a.drop_rep}")
subs=sorted({r.get("substrate","?") for r in R}); tags=sorted({r.get("tag","") for r in R}); print("substrate:",subs,"| tags:",tags)
g=collections.Counter(); cnt=collections.Counter(r["cond"] for r in R)
for r in R: g[r["cond"]]+=r["gen_tok"]
print("mean gen_tok by cond:",{c:round(g[c]/cnt[c],1) for c in cnt},"(should equal max_tokens with ignore_eos)")
by=collections.defaultdict(lambda: collections.defaultdict(list))
for r in R: by[r["cond"]][r["id"]].append(r.get("j_per_gen_tok",r["j_per_tok"]))
conds=sorted(by); ids=sorted(set.intersection(*[set(by[c]) for c in conds])) if conds else []
print(f"\n== J per GENERATED token by condition (mean over {len(ids)} complete prompt ids) ==")
for c in conds: print(f"  {c:20s} {mean([mean(by[c][i]) for i in ids]):.4f}")
def paired(get,c1,c2):
    d=[get(c1,i)-get(c2,i) for i in ids if get(c1,i) is not None and get(c2,i) is not None]
    if not d: return None
    random.seed(0); bs=sorted(mean(random.choices(d,k=len(d))) for _ in range(a.boot)); return mean(d),bs[int(0.025*a.boot)],bs[int(0.975*a.boot)],len(d)
PAIRS=[("emotional","neutral"),("emotional","shuffled_emotional"),("stance_feel","stance_count")]
gj=lambda c,i: mean(by[c][i]) if i in by[c] else None
print("\n== paired differences (c1 − c2), 95% bootstrap CI ==")
for c1,c2 in PAIRS:
    r=paired(gj,c1,c2)
    if r:
        base=mean([gj(c2,i) for i in ids]); m,lo,hi,n=r
        print(f"  {c1} − {c2}: {m:+.4f} J/tok  [{lo:+.4f}, {hi:+.4f}]  = {100*m/base:+.3f}% [{100*lo/base:+.3f}, {100*hi/base:+.3f}]  n={n} {'*' if lo>0 or hi<0 else ''}")
cl=[r["mean_clock"] for r in R]; tp=[r["mean_temp"] for r in R]; W=[r["mean_w"] for r in R]
print(f"\nregime: clock {min(cl)}–{max(cl)} (locked? spread should be small) | temp {min(tp)}–{max(tp)} | W {min(W):.0f}–{max(W):.0f}")
if not a.no_trace and "trace_w" in R[0]:
    import numpy as np
    def se_(x):
        x=np.asarray(x,float)
        if len(x)<8: return float("nan")
        P=np.abs(np.fft.rfft(x-x.mean()))[1:]**2; s=P.sum() or 1e-12; q=P/s; q=q[q>0]; return float(-(q*np.log(q)).sum()/np.log(len(P)))
    def lz76(x):
        if len(x)<8: return float("nan")
        med=sorted(x)[len(x)//2]; s="".join("1" if v>med else "0" for v in x); i,c,n=0,0,len(s)
        while i<n:
            l=1
            while i+l<=n and s[i:i+l] in s[:i+l-1]: l+=1
            c+=1; i+=l
        return c/(n/math.log2(n))
    def slope(x):
        n=len(x); xm=(n-1)/2; ym=sum(x)/n; return sum((t-xm)*(x[t]-ym) for t in range(n))/max(sum((t-xm)**2 for t in range(n)),1e-9)
    ts=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R:
        tr=r["trace_w"]; k=(r["id"],r["cond"]); ts[k]["se"].append(se_(tr)); ts[k]["lz"].append(lz76(tr)); ts[k]["sl"].append(slope(tr))
    print("\n== trace statistics by condition (spectral entropy / LZ76 / slope) ==")
    for c in conds:
        f=lambda k: mean([v for i in ids for v in ts[(i,c)][k] if v==v]); print(f"  {c:20s} SE {f('se'):.3f}  LZ {f('lz'):.3f}  slope {f('sl'):+.4f}")
    print("  paired differences, 95% bootstrap CI (the MCC paper reported emotional HIGHER on SE and LZ):")
    for k,name in (("se","spectral entropy"),("lz","LZ76 complexity")):
        gk=lambda c,i,k=k: (mean([v for v in ts[(i,c)][k] if v==v]) if ts[(i,c)][k] else None)
        for c1,c2 in PAIRS:
            r=paired(gk,c1,c2)
            if r: m,lo,hi,n=r; print(f"    {name:17s} {c1} − {c2}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] n={n} {'*' if lo>0 or hi<0 else ''}")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    data=[[mean(by[c][i]) for i in ids] for c in conds]
    plt.boxplot(data,labels=conds); plt.ylabel("J / generated token"); plt.xticks(rotation=20); plt.tight_layout()
    out=os.path.join(a.runs,"j_per_token.png"); plt.savefig(out,dpi=150); print("plot ->",out)
except ImportError: pass
