"""Second readout: with clocks UNLOCKED, tighten the power cap and measure time/token per class.
Run 20_run.py once per cap (set the cap with nvidia-smi -pl between runs); this script just
summarises time/token by cap from the run files' names. Rename each run file to run_<cap>W.jsonl."""
import argparse, json, glob, collections, os, re
p=argparse.ArgumentParser(); p.add_argument("--runs",default="runs/"); a=p.parse_args()
rows=collections.defaultdict(list)
for f in glob.glob(os.path.join(a.runs,"run_*W.jsonl")):
    cap=int(re.search(r"run_(\d+)W",f).group(1))
    for l in open(f):
        r=json.loads(l); rows[(cap,r["cond"])].append(r["seconds"]/max(r["gen_tok"],1))
caps=sorted({k[0] for k in rows}); conds=sorted({k[1] for k in rows})
print("time per generated token (s) by cap and condition")
print("cap  "+" ".join(f"{c:>18s}" for c in conds))
for cap in caps: print(f"{cap:4d} "+" ".join(f"{sum(rows[(cap,c)])/len(rows[(cap,c)]):18.4f}" for c in conds))
print("Prediction if the effect is real: the higher-draw class slows MORE as the cap tightens.")
