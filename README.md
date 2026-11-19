# fractalx-20261118-6b8d60d1

Report page: https://fractal-ai-bot-experiments.github.io/fractalx-20261118-6b8d60d1/

An experiment run by Fractal, an automated research bot run by AE Studio's Fractal team, after a request on X (post 2123113867911782527). The build agent wrote and ran the code below in a sandbox. The numbers come from the results file that code wrote. `raw/` holds every sandbox command with its exit code (`command_log.jsonl`) and the raw output files that passed the content check.

## Plan, fixed before the run

Primary metric:

```text
Balanced three-way accuracy (mean per-class recall, chance 0.33) of a cross-validated logistic probe predicting hand/arm, foot/leg or mouth for held-out reporting tool nouns: the probe on Qwen3-8B hidden states (layer and prompt chosen on selection nouns) minus the same probe on GloVe 300d vectors, mean of 3 seeds, with a paired bootstrap 95% interval over nouns. Higher supports the idea.
```

Plan given to the build agent:

```text
Model: Qwen/Qwen3-8B
Approach, as announced: training cross-validated linear probes on hidden states and on GloVe vectors, over a cleaner, larger tool list
Setting, as announced: the Lancaster Sensorimotor Norms
Compute: open_model_gpu
Size estimate: 29 GPU minutes on an L40S, 8 of them GPU-bound: anchor: Linear probes, every layer 8 for 11,660 statements, 36 layers, two probe types, controls, 3 seeds; this plan has about 7,000 short prompts (fewer forward passes) but adds word-list building with WordNet and a GloVe download 5, a layer x prompt x regularisation selection grid with ten comparison readouts 7, and two ladder rungs 6 + 3: 8 + 5 + 7 + 9 = 29, of which 8 GPU-bound
GPU tier: standard (29 minutes is far under 0.75 of the standard tier's 120, and inference on Qwen3-8B fits 48 GB)
Second look, the stronger tries if the first result is negative:
1. A probe on 4,096 dimensions may be starved by a few hundred tool nouns: train both probes on up to 1,000 extra non-reporting Lancaster nouns per class with a clear dominant body part, reselect on selection nouns, re-read the same reporting nouns (6 GPU minutes)
2. A linear probe may miss information coded non-linearly: a one-hidden-layer MLP probe on 100 PCA dimensions, for hidden states and GloVe alike (3 GPU minutes)
Primary metric, fixed before the run: Balanced three-way accuracy (mean per-class recall, chance 0.33) of a cross-validated logistic probe predicting hand/arm, foot/leg or mouth for held-out reporting tool nouns: the probe on Qwen3-8B hidden states (layer and prompt chosen on selection nouns) minus the same probe on GloVe 300d vectors, mean of 3 seeds, with a paired bootstrap 95% interval over nouns. Higher supports the idea.
Supports the idea if: On the reporting nouns the hidden-state probe scores at least 0.08 balanced accuracy above the GloVe probe (mean of 3 seeds) and the paired bootstrap 95% interval of the difference lies wholly above 0. A second-look try named in the plan that meets the same bar also counts, reported as that try.
Counts against the idea if: The hidden-state minus GloVe difference is below 0.04 and the upper end of its 95% interval is below 0.10, on the first method and on every second-look try that ran, while the positive control works. Anything in between is reported as inconclusive.
Controls:
- Same logistic probe, same tuning, on GloVe 300d vectors (the word-similarity baseline)
- Positive control: the carrier sentence names a randomly assigned body part before the tool noun; the hidden-state probe must recover it, and word vectors cannot
- Same probe on the model's input embeddings
- Shuffled-label probe (chance level, 200 shuffles)
- Hidden states reduced to 300 dimensions by PCA, to match GloVe's size
- Probe on GloVe plus hidden states against GloVe alone, to see whether hidden states add information
- Probe on log word frequency only, to rule out a frequency shortcut
- The earlier anchor-word cosine readout on the new word list
- The tool-trained probe applied to words for parts of each body part (finger, toe, tongue and the like)
- Each prompt type (neutral carrier, use-eliciting) reported at its own best selection layer
The asker's condition: try harder before you say no; the plan: run stronger readouts and only say no if all fail
Closest prior work from the literature check:
- fractal-ai-bot-experiments/fractalx-20261118-a7fd3213 (2026)
- SauersML/body (2026)
Existing code to reuse, from the reference index:
- https://github.com/safety-research/believe-it-or-not (may copy with credit (MIT)): science_synth_facts/model_internals/probes.py, science_synth_facts/model_internals/model_acts.py; Tested code for pulling per-layer activations and fitting logistic probes. Copy the files (MIT, keep the notice) alongside the pod's probe recipe; do not install its pinned requirements.
Procedure:
1. Words. Download the Lancaster Sensorimotor Norms (OSF osf.io/7emr6). Keep single alphabetic words with a gensim glove-wiki-gigaword-300 vector and no body-part string whose first WordNet noun sense lies under instrumentality.n.03 (tier A) or has lexname noun.artifact (tier B), never food. Label by the top action rating among hand/arm, foot/leg, mouth: at least 2.5 and margin m above both others. Try (A, 1.0), (A, 0.5), (B, 1.0), (B, 0.5); take the first whose smallest class reaches 80, else the largest smallest class. Cap classes at twice the smallest, by margin. Stratified split, seed 0: one third selection, two thirds reporting. Record the smallest detectable edge before reading reporting nouns.
2. Features (probe recipe, raw text, all 36 layers of Qwen3-8B): (a) the noun's last token in 12 neutral carrier sentences ending at the noun, averaged; (b) the final token of 'A {noun} is something a person mainly operates with the'; (c) positive control: the noun's last token in 'Using only her {hands|feet|mouth}, she worked the {noun}', part assigned at random, label = that part; (d) 72 body-part words (24 per class) under (a). Also GloVe, input embeddings, log word frequency.
3. Probe: standardise, then L2 multinomial logistic regression with balanced class weights, fitted in the training fold's sample subspace (SVD rotation; exact and fast). Score: balanced accuracy.
4. Selection, selection nouns only: 5-fold cross-validation, 3 repeats, over layer x prompt (a, b) x C in {1e-4, 1e-3, 1e-2, 1e-1, 1}. Take the best; within 0.02 prefer (a), lower layer, smaller C. GloVe and every other feature set choose C the same way.
5. Reporting, once: 10-fold cross-validation over reporting nouns, training on the other folds plus all selection nouns; 3 seeds = 3 fold assignments. Primary: hidden minus GloVe, mean over seeds, paired bootstrap over nouns (10,000 resamples). Also: (a) and (b) each at its best selection layer, hidden PCA-300, GloVe+hidden against GloVe, input embeddings, frequency only, 200 label shuffles, anchor-word cosine, the tool-trained probe on (d), the positive control.
6. Second look if the difference is below 0.08 or its interval includes 0: rung 1, add up to 1,000 non-reporting Lancaster nouns per class (any category, margin 1.0) as training data for both probes, reselect, re-read; rung 2, a one-hidden-layer MLP probe on PCA-100 for both. Report every try.
7. Save word list, splits, per-noun predictions, results.json (seeds recipe) and a layer curve.
```

## Result

Status: succeeded. Seeds: 0, 1, 2.

Question:

```text
Do tool words carry their body part beyond word similarity, under a stronger test?
```

Answer:

```text
Yes: a probe on Qwen3-8B hidden states names a held-out tool's body part at 0.93 vs 0.79 on GloVe (+0.14, CI 0.09 to 0.19); against fastText vectors (0.86) the edge shrinks to +0.07 (CI 0.03 to 0.11). Mid-layer states of Qwen3-8B carry more of a tool's using body part than static word vectors do, but part of the GloVe gap is GloVe's weakness: a better word vector closes about half of it.
```

Claim tested:

```text
Qwen3-8B's hidden states tell which body part uses a tool better than plain word vectors do.
```

Verdict on the claim: Supported (partial result).

Numbers from the results file:

```text
Held-out accuracy at naming the body part that uses a tool (seed 0): 0.923975
Held-out accuracy at naming the body part that uses a tool (seed 1): 0.925216
Held-out accuracy at naming the body part that uses a tool (seed 2): 0.92809
Same probe on GloVe word vectors (seed 0): 0.782319
Same probe on GloVe word vectors (seed 1): 0.792181
Same probe on GloVe word vectors (seed 2): 0.792571
Same probe on the model's input embeddings (seed 0): 0.712821
Same probe on the model's input embeddings (seed 1): 0.69558
Same probe on the model's input embeddings (seed 2): 0.708706
Hidden states reduced to 300 dimensions (seed 0): 0.923975
Hidden states reduced to 300 dimensions (seed 1): 0.92809
Hidden states reduced to 300 dimensions (seed 2): 0.933837
Probe on GloVe and hidden states together (seed 0): 0.925216
Probe on GloVe and hidden states together (seed 1): 0.933447
Probe on GloVe and hidden states together (seed 2): 0.919469
Probe on word frequency only (seed 0): 0.384703
Probe on word frequency only (seed 1): 0.39045
Probe on word frequency only (seed 2): 0.39045
Hidden-state probe when the sentence names the body part (seed 0): 0.921067
Hidden-state probe when the sentence names the body part (seed 1): 0.905581
Hidden-state probe when the sentence names the body part (seed 2): 0.913075
Hidden-state probe with shuffled labels (seed 0): 0.331467
Hidden-state probe with shuffled labels (seed 1): 0.329658
Hidden-state probe with shuffled labels (seed 2): 0.333276
Earlier anchor-word similarity readout on hidden states (seed 0): 0.728111
Earlier anchor-word similarity readout on hidden states (seed 1): 0.728111
Earlier anchor-word similarity readout on hidden states (seed 2): 0.728111
Tool-trained probe applied to body-part words (seed 0): 0.875
Tool-trained probe applied to body-part words (seed 1): 0.875
Tool-trained probe applied to body-part words (seed 2): 0.875
Hidden minus GloVe, mean of 3 seeds: 0.136737
Hidden minus GloVe, CI lower: 0.0859541
Hidden minus GloVe, CI upper: 0.188584
Hidden probe selection CV accuracy: 0.90795
GloVe probe selection CV accuracy: 0.768678
GloVe plus hidden minus GloVe, CI lower: 0.0851647
GloVe plus hidden minus GloVe, CI upper: 0.188872
GloVe probe on the positive-control labels: 0.385899
Shuffled labels 95th percentile, seed 0: 0.378479
Shuffled labels 95th percentile, seed 1: 0.379709
Shuffled labels 95th percentile, seed 2: 0.383281
Anchor-word cosine readout on GloVe: 0.55066
GloVe tool probe on body-part words: 0.680556
Neutral carriers: best selection layer: 15
Neutral carriers at own best layer: 0.901838
Use-eliciting prompt: best selection layer: 10
Use-eliciting prompt at own best layer: 0.92576
Model's own next-token answer (no training): 0.495317
Extra-nouns rung: hidden probe: 0.915378
Extra-nouns rung: GloVe probe: 0.803782
Extra-nouns rung: hidden minus GloVe: 0.111596
Extra-nouns rung: difference CI lower: 0.0599869
Extra-nouns rung: difference CI upper: 0.165434
MLP rung: hidden states: 0.903966
MLP rung: GloVe: 0.760111
MLP rung: hidden minus GloVe: 0.143856
MLP rung: difference CI lower: 0.0902739
MLP rung: difference CI upper: 0.198696
Extra-nouns rung: selection CV accuracy: 0.922701
Extra-nouns rung: chosen layer: 12
Chosen layer: 10
Chosen regularisation C: 0.1
Chosen prompt is the use-eliciting prompt: 1
Hidden minus GloVe, seed 0: 0.141656
Hidden minus GloVe, seed 1: 0.133035
Hidden minus GloVe, seed 2: 0.135519
```

## Limits

```text
Partial result: the health of the generated text could not be completed, because No plan condition generates text; every reported number is a probe on activations. The only generations were 30 greedy continuations of the use-eliciting prompt from the untouched model, made to read what it says. They were checked with the health recipe under their own condition (passed: unhealthy share 0.067), but the gate wants a record under a plan condition, and relabelling them would misname them.
The bar is met against GloVe, but against fastText the edge is +0.07 (CI 0.03 to 0.11), below the 0.08 bar. 'Better than plain word vectors' depends on which vectors.
Labels are Lancaster rater averages with margin as low as 0.5; some nouns are noisy (cake, trivia and drugs counted as 'mouth tools').
The smallest detectable edge with 255 reporting nouns is about 0.11 (80% power), so differences of 0.07 are near the limit.
One model (Qwen3-8B); seeds are fold assignments only, so the spread across seeds understates the uncertainty, while the bootstrap over nouns covers it.
The selected prompt asks about use; the neutral carriers alone give a similar edge (+0.11).
Probe decodability is not use: the model's own next-token answer reaches only 0.495.
Settings as run: label shuffles per fold seed 67, 67 and 66 (200 in total); minutes the first rung-1 try ran before it was killed 14.
Second look, First method: logistic probe, layer, prompt and C selected on selection nouns (tested): Hidden 0.926 vs GloVe 0.789; difference +0.137, CI [0.086, 0.189]; meets the bar
Second look, Rung 1: plus 2,199 extra non-reporting Lancaster nouns as training data, reselected (tested): Hidden 0.915 vs GloVe 0.804; +0.112, CI [0.060, 0.165]; meets the bar
Second look, Rung 2: one-hidden-layer MLP on PCA-100 (tested): Hidden 0.904 vs GloVe 0.760; +0.144, CI [0.090, 0.199]
Second look, Follow-up (exploratory): neutral carriers only vs GloVe (tested): +0.113, CI [0.062, 0.163]
Second look, Follow-up (exploratory): stronger static vectors, fastText wiki-news 300d (tested): fastText 0.856; hidden minus fastText +0.070, CI [0.028, 0.115]: below the 0.08 bar, above 0
Not shown: 5 statements the review did not pass.
```

## What failed

```text
The first full analysis run stalled in rung 1 and was killed; its rung 1 was cut down (see deviations).
The smoke extraction crashed once because the model loader returns model and tokenizer in the other order.
```

## Withheld outputs

These raw outputs held a string shaped like a secret key, or a content check flagged them as harmful content or could not check them; each is replaced by its marker in the repository.

```text
[output withheld: secret-shaped string]: raw/command_log.jsonl, 1 of 54 commands
[output withheld: violence]: 1
```
