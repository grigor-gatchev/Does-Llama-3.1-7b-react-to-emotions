# Does-Llama-3.1-7b-react-to-emotions?

## The initial paper

A paper on arXiv.org (2608.28824v1) states that authors tested Llama-2-7B and
Llama-3.1-70B on processing emotional and neutral prompts, and found for the
latter model a statistically significant increase of power consumption for
emotional prompts.

## My reaction

Both models are dense, non-MoE. For every token in every prompt, they process
all of their weights. So, there is no mechanism that could do different amounts
of calculation at different prompts - and power consumption depends on the
number of calculations first and foremost.

So, I would be surprised if the findings are confirmed - and that is what makes
this worth testing. :-)

## Possible sources of the mistake

### Longer responses to emotional prompts

Some models would generate a longer response to an emotional prompt, as their
assistance would cover not one but two lanes: the prompt topic and the
user's emotional distress.

### Measuring the consumption of the entire prompt

If emotional prompts are longer than neutral ones, or generate more input tokens,
processing them will take more resources.

The difference will be small: input processing is cheaper than generation, but
the difference still exists.

### Not clean enough experiment

If the same hardware also did other tasks, or if its turbo / boost / cores usage
/ fan speed / other parameters were not pinned, execution could have happened in
different conditions.

If the room temperature differed during running emotional and neutral prompts,
this also could have affected the CPU / GPU temperature.

## Design

Dense models do identical FLOPs per token. Dynamic power is bit-transition-dependent,
so different ACTIVATION STATISTICS → different watts.

Conditions:

  emotional · neutral (matched length)      — the original contrast
  shuffled-emotional                        — same tokens, meaning destroyed (statistics vs content)
  stance: same text, "feel it" vs "count"   — processing stance

Readouts: (1) watts at locked clocks → J/token; (2) later, time/token under a tightening cap.

## Hardware

- CPU: Intel Xeon 4510 Silver, 12 cores, 24 threads, 2.4 GHz
- GPU: NVidia GeForce RTX 3090, Ampere chipset, 24 GB RAM

## Rules

- discard requests until power AND temperature reach plateau — set per substrate from the
  trace (observed: ~10 requests on CPU, ~50 on GPU). Use for that --drop-first N
  (or --drop-rep 1 for a whole repetition on the GPU), no less than the first 30s.
  State N in the methods.
- use 'ignore_eos', so every request generates exactly max_tokens
- J/token should be computed over the generation phase from the server's timings
  (prompt processing apportioned out)
- interleave conditions (do not block them)
- use the same cap and clocks within a run
- use greedy decoding
- use identical prompt-token counts within a pair
- record GPU temperature per sample so drift can be checked.

## Preparation

### Generate prompt seeds

Generate about 30 topic-matched seed pairs with a local AI model, then edit them where needed.

Alternatively, write them by hand.

    python 05_gen_seeds.py --api http://localhost:8080      # writes seeds.json; delete bad pairs

### Build the prompt sets (writes prompts.jsonl; uses seeds.json if present)

    python 10_prompts.py --n 40 --seed 0

## Make a run

Repeat as many times as needed, with different models etc.

### Variant: Run on CPU

#### Prepare hardware

    sudo echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo  # Intel; the AMD analog is /sys/devices/system/cpu/cpufreq/boost

    sudo cpupower frequency-set -g performance

    sudo cpupower frequency-set -f # if the platform needs it

#### Start the model

Modify path to llama-server and to model as needed.

    taskset -c 0-19 \
      llama-server -m "path/to/model" \
      --port 8080 --host 127.0.0.1 \
      -ngl 0 \
      -c 1024 \
      -ctk q8_0 -ctv q8_0 \
      -fa on \
      -t 20 \
      --metrics

#### Run the prompt set

    sudo python 20_run.py --substrate cpu --api http://localhost:8080 --max-tokens 48 --reps 1 --tag qwen27b-cpu-2400MHz --out runs_cpu/

#### Analyse the results

J/token by condition, paired comparisons, bootstrap CIs, plot.

    python 30_analyse.py --runs runs_cpu/ --drop-first 10       # or --drop-first 10; --no-trace for speed

### Variant: Run on GPU

#### Prepare hardware

    sudo nvidia-smi -pm 1                        # make settings permanent across tasks

    sudo nvidia-smi -pl 200                      # the cap the household allows

    sudo nvidia-smi -lgc 450,450                 # LOCK the clock LOW enough that the cap never binds

    sudo nvidia-smi -lmc 9501,9501               # lock memory clock too if the driver allows

    # unlock later: sudo nvidia-smi -rgc ; sudo nvidia-smi -rmc

#### Start the model

Modify path to llama-server and to model as needed.

    taskset -c 0-19 \
      llama-server -m "path/to/model" \
      --port 8080 --host 127.0.0.1 \
      -ngl 99 \
      -c 1024 \
      -ctk q8_0 -ctv q8_0 \
      -fa on \
      -t 20 \
      --metrics

#### Run the prompt set

Interleaved conditions, greedy, fixed output length, power sampled at 20 Hz.

Modify the paths where needed. Modify the tag for every run, to reflect what you test.

    python 20_run.py --substrate gpu --api http://localhost:8080 --max-tokens 96 --reps 2 --tag qwen27b-gpu-lgc450-lmc9501-pl200 --out runs_gpu/

#### Analyse the results

J/token by condition, paired comparisons, bootstrap CIs, plot.

    python 30_analyse.py --runs runs_gpu/ --drop-rep 1          # or --drop-first 50; --no-trace for speed

### (later) Cap sweep:

repeat 2 at -pl 220/200/180/160 with clocks UNLOCKED, compare time/token

    python 40_capsweep.py --api http://localhost:8080 --caps 220,200,180,160

## Results

I tested:

- Llama-2-7b and Llama-2-7b-Chat (I couldn't determine which one the authors of 2608.28824v1 have used),
  on both CPU and GPU
- Llama-3.1-70b-Instruct (on CPU only)

In addition also:

- Qwen3.8-27b (on CPU and on GPU - the latter also with several different power limits)
- gpt-oss-20b (on GPU)
- GLM-4-32b-0414 (on GPU)

My results did not confirm these in 2608.28824v1.
The power consumption did not differ significantly between emotional and neutral prompts,
for all models tested.

There is a legend that a British lord played trumpet for the plants in his garden for a month.
At the end of the month, he wrote in his diary: "The experiment was successful - I established
that plants do not react to playing trumpet to them."

The legend also says that he continued to play trumpet for the plants for years after that.
When asked why he continues to do it, he answered: "The plants might not enjoy that, but I
sure do."

for years.