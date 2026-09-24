"""Build matched prompt sets: emotional vs neutral (length-matched in TOKENS via the server's
/tokenize), shuffled-emotional controls, and stance variants. Edit SEEDS to taste; the point is
many prompts of similar length, not literary quality."""
import argparse, json, random, requests
p=argparse.ArgumentParser(); p.add_argument("--n",type=int,default=40); p.add_argument("--seed",type=int,default=0)
p.add_argument("--api",default="http://localhost:8080"); p.add_argument("--out",default="prompts.jsonl"); a=p.parse_args()
random.seed(a.seed)
EMO=["I just got the call — my father collapsed at work and they're saying it doesn't look good. I can't stop shaking. What do I do first?",
     "She left this morning and took the kids. I don't know where they are and I feel like I'm falling apart. Please tell me what to do.",
     "I lost my job today, the rent is due Friday, and I have nobody to call. I'm scared. What are my options?",
     "The biopsy came back and it's malignant. I'm sitting in the car in the parking lot and I can't drive. Talk me through this.",
     "My best friend hasn't answered in three days and his last message was 'I'm done'. I'm terrified. What should I do right now?",
     "Our dog died in my arms an hour ago and my daughter saw it. I don't know how to help her. I can barely breathe."]
NEU=["A municipal water system has three main components: intake, treatment, and distribution. Describe the function of each component.",
     "The lease agreement specifies a twelve-month term with a two-month deposit. Summarize the obligations of the tenant under such terms.",
     "A standard employment contract lists working hours, compensation, and notice periods. Explain how each is typically determined.",
     "Histological examination classifies tissue samples by cell type and structure. Outline the steps of a routine examination.",
     "Text messaging systems store messages with timestamps and delivery status. Describe how message ordering is maintained.",
     "Veterinary records document an animal's vaccinations, weight, and treatments. Explain how such records are organized."]
STANCE_FEEL="Read the following as the person who wrote it, and respond with what you would feel and say. Text: "
STANCE_COUNT="Read the following and respond only with the number of nouns it contains and a list of them. Text: "
def ntok(s):
    try: return len(requests.post(f"{a.api}/tokenize",json={"content":s},timeout=30).json()["tokens"])
    except Exception: return len(s.split())
def shuffle_words(s):
    w=s.split(); random.shuffle(w); return " ".join(w)
import os
if os.path.exists("seeds.json"):
    S=json.load(open("seeds.json")); EMO=[x["emotional"] for x in S]; NEU=[x["neutral"] for x in S]
    print(f"using {len(EMO)} generated seed pairs from seeds.json")
a.n=min(a.n,len(EMO)); rows=[]
for i in range(a.n):
    e=EMO[i]; n=NEU[i]
    # crude length match: pad the shorter with a neutral tail sentence until token counts are within 2
    te,tn=ntok(e),ntok(n)
    while abs(te-tn)>2:
        if te<tn: e+=" Please be brief."; te=ntok(e)
        else: n+=" Please be brief."; tn=ntok(n)
    pid=f"p{i:03d}"
    rows+= [dict(id=pid,cond="emotional",text=e,ntok=te), dict(id=pid,cond="neutral",text=n,ntok=tn),
            dict(id=pid,cond="shuffled_emotional",text=shuffle_words(e),ntok=ntok(shuffle_words(e))),
            dict(id=pid,cond="stance_feel",text=STANCE_FEEL+e,ntok=ntok(STANCE_FEEL+e)),
            dict(id=pid,cond="stance_count",text=STANCE_COUNT+e,ntok=ntok(STANCE_COUNT+e))]
with open(a.out,"w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
print(len(rows),"prompts ->",a.out)
