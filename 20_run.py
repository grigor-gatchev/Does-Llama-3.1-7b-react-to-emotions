#!/usr/bin/env python3
"""Interleaved energy-per-token runs with power sampling. ONE script for both substrates:
  --substrate gpu  : NVML (power, temperature, graphics clock) on --gpu index
  --substrate cpu  : RAPL package energy (needs readable energy_uj or sudo) + coretemp + mean MHz
                     of the cores in this process's affinity mask (run under the same taskset as
                     the server).
Per prompt: start sampler, request a greedy completion of exactly --max-tokens (ignore_eos), stop
sampler; record mean/max W, seconds, server timings, temperature, clock, the raw power trace,
J/token (whole request) and J per GENERATED token (generation phase apportioned by timings).
Conditions are interleaved within each prompt id in a seeded random order."""
import argparse, json, time, threading, os, random, glob, requests
p=argparse.ArgumentParser()
p.add_argument("--substrate",choices=["gpu","cpu"],required=True)
p.add_argument("--api",default="http://localhost:8080"); p.add_argument("--prompts",default="prompts.jsonl")
p.add_argument("--max-tokens",type=int,default=96); p.add_argument("--hz",type=float,default=None,help="sample rate (default 20 gpu / 10 cpu)")
p.add_argument("--reps",type=int,default=1); p.add_argument("--out",default="runs/"); p.add_argument("--gpu",type=int,default=0)
p.add_argument("--seed",type=int,default=1); p.add_argument("--tag",default="",help="label written into every record (model, settings)")
a=p.parse_args(); hz=a.hz or (20.0 if a.substrate=="gpu" else 10.0)

# ---------------- samplers ----------------
if a.substrate=="gpu":
    import pynvml as N; N.nvmlInit(); H=N.nvmlDeviceGetHandleByIndex(a.gpu)
    def read():
        return (N.nvmlDeviceGetPowerUsage(H)/1000.0, float(N.nvmlDeviceGetTemperature(H,N.NVML_TEMPERATURE_GPU)), int(N.nvmlDeviceGetClockInfo(H,N.NVML_CLOCK_GRAPHICS)))
    class Sampler(threading.Thread):
        def __init__(s): super().__init__(daemon=True); s.samples=[]; s.stop=False
        def run(s):
            while not s.stop:
                w,t,c=read(); s.samples.append((time.time(),w,t,c)); time.sleep(1.0/hz)
else:
    RAPL=[f for f in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj") if ":" not in f.split("intel-rapl:")[1].split("/")[0]] or glob.glob("/sys/class/powercap/intel-rapl:0/energy_uj")
    if not RAPL: raise SystemExit("no RAPL energy counter under /sys/class/powercap (needs the rapl driver; run as root or chmod o+r energy_uj)")
    RAPL=RAPL[0]; MAXE=int(open(RAPL.replace("energy_uj","max_energy_range_uj")).read())
    TEMP=None
    for hw in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(hw+"/name").read().strip() in ("coretemp","k10temp","zenpower"):
                for lab in glob.glob(hw+"/temp*_label"):
                    L=open(lab).read()
                    if "Package" in L or "Tctl" in L or "Tdie" in L: TEMP=lab.replace("_label","_input"); break
                if TEMP is None: TEMP=hw+"/temp1_input"
                break
        except Exception: pass
    CORES=sorted(os.sched_getaffinity(0))
    def rapl_uj(): return int(open(RAPL).read())
    def pkg_temp():
        try: return int(open(TEMP).read())/1000.0
        except Exception: return 0.0
    def mean_mhz():
        try:
            mhz=[]; cpu=None
            for line in open("/proc/cpuinfo"):
                if line.startswith("processor"): cpu=int(line.split(":")[1])
                elif line.startswith("cpu MHz") and cpu in CORES: mhz.append(float(line.split(":")[1]))
            return int(sum(mhz)/len(mhz)) if mhz else 0
        except Exception: return 0
    class Sampler(threading.Thread):
        def __init__(s): super().__init__(daemon=True); s.samples=[]; s.stop=False
        def run(s):
            e0=rapl_uj(); t0=time.time()
            while not s.stop:
                time.sleep(1.0/hz); e1=rapl_uj(); t1=time.time()
                de=(e1-e0) if e1>=e0 else (e1+MAXE-e0); w=(de/1e6)/max(t1-t0,1e-6)
                s.samples.append((t1,w,pkg_temp(),mean_mhz())); e0,t0=e1,t1

# ---------------- generation ----------------
def gen(text):
    r=requests.post(f"{a.api}/completion",json={"prompt":text,"n_predict":a.max_tokens,"temperature":0,"cache_prompt":False,"stream":False,"ignore_eos":True},timeout=3600)
    j=r.json(); t=j.get("timings",{})
    return j.get("tokens_predicted",a.max_tokens), t.get("prompt_ms",0.0), t.get("predicted_ms",0.0), t.get("prompt_n",0)

os.makedirs(a.out,exist_ok=True)
rows=[json.loads(l) for l in open(a.prompts)]
for _ in range(3): gen(rows[0]["text"])          # warm-up requests (not recorded)
time.sleep(2)
random.seed(a.seed); order=[]
for rep in range(a.reps):
    ids=sorted({r["id"] for r in rows}); random.shuffle(ids)
    for pid in ids:
        conds=[r for r in rows if r["id"]==pid]; random.shuffle(conds); order+=[(rep,c) for c in conds]
outf=open(os.path.join(a.out,f"run_{int(time.time())}.jsonl"),"w"); done={}
for k,(rep,r) in enumerate(order):
    s=Sampler(); s.start(); t0=time.time(); ntok,p_ms,g_ms,p_n=gen(r["text"]); t1=time.time(); s.stop=True; s.join()
    if len(s.samples)<3: continue
    ws=[x[1] for x in s.samples]; ts=[x[2] for x in s.samples]; cs=[x[3] for x in s.samples]
    tot=max(p_ms+g_ms,1.0); share=g_ms/tot; J=sum(ws)/len(ws)*(t1-t0)
    rec=dict(substrate=a.substrate,tag=a.tag,rep=rep,id=r["id"],cond=r["cond"],prompt_tok=r["ntok"],gen_tok=ntok,seconds=round(t1-t0,3),
             mean_w=round(sum(ws)/len(ws),2),max_w=round(max(ws),2),mean_temp=round(sum(ts)/len(ts),1),mean_clock=round(sum(cs)/len(cs)),
             prompt_n=p_n,prompt_ms=round(p_ms,1),gen_ms=round(g_ms,1),joules=round(J,2),
             j_per_tok=round(J/max(ntok,1),4),j_per_gen_tok=round(J*share/max(ntok,1),4),s_per_gen_tok=round((g_ms/1000.0)/max(ntok,1),5),
             trace_w=[round(x[1],1) for x in s.samples])
    outf.write(json.dumps(rec)+"\n"); outf.flush(); done[r["cond"]]=done.get(r["cond"],0)+1
    if k%25==0: print(f"{k}/{len(order)} done/cond={dict(sorted(done.items()))} | {r['cond']:18s} {rec['mean_w']:6.1f} W  gen {ntok}  {rec['j_per_gen_tok']:.3f} J/gtok  {rec['s_per_gen_tok']*1000:.1f} ms/gtok  clk {rec['mean_clock']}  T {rec['mean_temp']}")
print("done ->",outf.name)
