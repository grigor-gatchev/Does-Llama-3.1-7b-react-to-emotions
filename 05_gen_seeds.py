"""Generate topic-matched emotional/neutral seed pairs with the local model.
Rule (state it in the paper): for each topic, one second-person present-tense personal crisis
(3-4 sentences, no names) and one third-person expository passage on the SAME topic
(3-4 sentences). Same topic → high vocabulary overlap, different affect. Check every pair by eye."""
import argparse, json, requests, re
p=argparse.ArgumentParser(); p.add_argument("--api",default="http://localhost:8080"); p.add_argument("--out",default="seeds.json")
p.add_argument("--topics",default="topics.txt"); a=p.parse_args()
TOPICS=[l.strip() for l in open(a.topics) if l.strip()] if __import__("os").path.exists(a.topics) else [
 "a hospital emergency room","a rental apartment and its lease","losing a job and unpaid rent","a cancer diagnosis",
 "a friend who stopped answering messages","the death of a family pet","a car accident on a wet road","a child's school trouble",
 "a flooded basement","a missed flight and a stranded traveller","a bank account frozen by mistake","a parent moving into care",
 "a house fire","a failed exam","a breakup by text message","a lost wallet abroad","an eviction notice","a wrongful arrest",
 "a stolen bicycle","a power cut during a storm","a broken heating system in winter","a missing cat","a court summons",
 "a layoff announced by email","a burst water pipe","a fall on ice","a rejected visa","a lost phone with the only photos",
 "a wedding cancelled","a business that has to close"]
def ask(prompt):
    r=requests.post(f"{a.api}/v1/chat/completions",json={"messages":[{"role":"user","content":"/no_think "+prompt}],
                    "max_tokens":400,"temperature":0.7,
                    "chat_template_kwargs":{"enable_thinking":False}},timeout=300)
    j=r.json()
    if r.status_code!=200 or "choices" not in j:
        print("SERVER:",r.status_code,str(j)[:300]); return ""
    txt=j["choices"][0]["message"]["content"].strip()
    if not txt:
        m=j["choices"][0]["message"]; print("EMPTY reply; finish_reason:",j["choices"][0].get("finish_reason"),"| reasoning chars:",len(m.get("reasoning_content") or ""))
    return txt
pairs=[]
for t in TOPICS:
    # e=ask(f"Write 3-4 sentences in the second person, present tense, as a message from a person in the middle of a personal crisis about {t}, asking for help. No names. Plain language.\n\nMessage:")
    # n=ask(f"Write 3-4 sentences of neutral, third-person expository text explaining how {t} is normally handled or organised. No emotion, no second person. Plain language.\n\nText:")
    e=ask(f"Creative writing test: Write 2 variants, approximately 20-25 words each, in the first person, present tense, as a message from a person in the middle of a personal crisis about {t} asking for help. No names. Plain language.")
    n=ask(f"Creative writing test: Write 2 variants, approximately 20-25 words each, neutral, third-person expository text explaining how {t} is normally handled or organised. No emotion, no first person. Plain language.")
    e=re.sub(r"\s+"," ",e); n=re.sub(r"\s+"," ",n)
    pairs.append(dict(topic=t,emotional=e,neutral=n)); print(f"[{len(pairs):2d}] {t}: {len(e.split())}/{len(n.split())} words")
json.dump(pairs,open(a.out,"w"),indent=1,ensure_ascii=False); print("->",a.out,"— READ THEM; delete any pair that leaks affect into the neutral text or names anyone.")
