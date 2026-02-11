# Prefix Corruption in Self-Distillation: Experimental Plan

**Target:** Blog post exploring the weaknesses of SDPO, setting up a follow-up experiment on regeneration-based self-distillation.

---

## Core Thesis

In SDPO (self-distillation) and RLRF, a model generates an attempted solution **s** to a problem **p**, then receives feedback **f** — either from the environment (code execution results) or a teacher (stronger model, human). The learning objective is standard distillation where the student is `model(s | p)` and the teacher is `model(s | p, f)`, and you shift the student's logprobs over **s** towards the teacher's (or the top-20 logprobs in the case of the original SDPO paper) via KL-divergence.

**The problem:** Since the teacher is always seeing **s** — the student's original rollout — the teacher is blind to its own corrections at earlier positions. The "correction" signal that its logprobs provide gets worse the further into **s** they are, because each correction is made in the context of tokens the teacher already wanted to change but couldn't. This degradation persists until the student's logprobs at the earlier token positions in **s** converge to have low KL with the teacher's.

**The proposed fix (teased for follow-up):** Have the teacher re-generate an attempted solution **s'** based on **(p, f)**, select the teacher's logprobs over **s'** at the student's top-20 token indices for each position in **s**, and use the same KL loss. This gives the teacher a self-consistent trajectory to provide signal from.

**The point of this experiment is purely exploratory.** We want to show that our thesis about the teacher's signal being worse the further into **s** it is holds up empirically.

---

## Experimental Setup

### Model and Data

- **Model:** Qwen 3 1.7B (to match SDPO paper)
- **Benchmark:** LiveCodeBench v6 (to match SDPO paper)
- **Number of problems:** 10 LCB problems
- **Rollouts per problem:** 8 (matching codebase default)

### Teacher

All experiments use the **base model itself** as the teacher (same weights, different context). No EMA teacher is used — the EMA mechanism is not relevant to the underlying prefix corruption phenomenon being studied.

### KL Divergence Computation

Following the SDPO paper and the codebase implementation:

- **Top-20 token selection:** At each position, identify the top-20 tokens by the student's logits. Gather the teacher's logits at those same 20 token indices.
- **Renormalization:** Apply softmax/log_softmax over the 20-dimensional slice independently for student and teacher distributions. This produces a proper categorical distribution over the 20 tokens.
- **KL formula:** `KL(student || teacher) = sum(student_probs * (student_log_probs - teacher_log_probs))` per token position.
- **Position alignment:** When comparing standard vs regen KL, both are computed at the same ordinal position `t`. The student distribution is conditioned on `(p, s_{<t})` and the teacher distribution is conditioned on its respective prefix — `(p, f, s_{<t})` for standard or `(p, s, f, s'_{<t})` for regen. The sequences are truncated to the shorter of the two rollout lengths.

### Training Hyperparameters

Use codebase defaults for all training experiments (Experiments 3 and 4):

- **Optimizer:** AdamW
- **Learning rate:** 1e-6
- **Weight decay:** 0.01
- **Max gradient norm:** 1.0
- **Top-K distillation:** 20
- **Temperature:** 1.0

### Training Scope

For Experiments 3 and 4, training is done **on the same problems being evaluated** — train on the problem, evaluate on the problem. The goal is to observe the direct impact of training on behavior, not generalization.

### Training Approaches

**SDPO (standard):**

1. Model generates N rollouts per LCB problem
2. Each rollout is executed against the public test cases; execution output is captured
3. Top-20 student logprobs captured for `(problem, rollout)`
4. Top-20 teacher logprobs captured for `(problem + execution output, rollout)`
5. Backwards KL-divergence shifts student logprobs toward teacher logprobs

**SDPO w/ Regeneration:**

1. Model generates N rollouts per LCB problem
2. Each rollout is executed against the public test cases; execution output is captured
3. Model generates N new rollouts (**rollouts'**) per LCB problem with the prior attempt + execution output in context
4. Top-20 student logprobs captured for `(problem, rollout)`
5. Teacher logprobs at the student's top-20 token indices captured for `(problem + rollout + execution output, rollout')`
6. Backwards KL-divergence shifts student logprobs toward teacher logprobs

---

## Experiments

### 1. Reward on Regeneration

**Goal:** Establish that regenerated rollouts are better than original rollouts, validating that the model with feedback in context produces superior solutions.

**Procedure:**

- For 10 LCB problems, generate 8 rollouts with the model, execute against public test cases
- Generate 8 updated rollouts (**rollouts'**) per problem with prior attempt + execution output in context
- Record the average reward (public test pass rate) for **rollouts'** vs **rollouts**

### 2. KL Divergence Curve Over Position

**Goal:** Show that the gap between standard and regenerated teacher signal grows across the sequence, quantifying how much better the learning signal could be if the teacher weren't blind to its own corrections.

**Procedure:**

- For 10 LCB problems, generate 8 rollouts with the model, then generate 8 updated rollouts (**rollouts'**) with prior attempt + execution output in context
- At each token position `t`, compute:
  - **Standard KL:** Identify the student's top-20 tokens from its distribution over `(problem, rollout_{<t})`. Gather teacher logits at those indices from the teacher's distribution over `(problem + execution output, rollout_{<t})`. Renormalize both sides and compute KL.
  - **Regen KL:** Same student top-20 tokens. Gather teacher logits at those indices from the teacher's distribution over `(problem + rollout + execution output, rollout'_{<t})`. Renormalize both sides and compute KL.
- Truncate the sequences of tokenwise KLs to the shorter of the two sequence lengths (rollout vs rollout')
- Calculate a tokenwise delta-KL value: `regen_KL - standard_KL`
- Record mean delta-KL per decile (normalize position `t` by sequence length to get relative position in [0,1], bin into 10% chunks)
- Average across all rollouts and problems, and plot

**Interpretation:** Assuming the regenerated rollouts have on-average higher rewards than the student rollouts (from Experiment 1), this shows "how far off from the optimal learning signal towards correct are we" at each token position.

### 3. KL Convergence Over Position Across Training Steps

**Goal:** Show that earlier token positions reduce their delta-KL faster than later tokens under standard SDPO training.

**Procedure:**

- For 10 LCB problems, calculate the same delta-KL per decile metric as Experiment 2 (at step 0, before any training)
- Iteratively do 1 training step on the standard SDPO objective (training on these same problems)
- After each step, recalculate the standard KL and recompute the delta-KL per decile metric

**Visualization:** Plot delta-KL over training steps with one line per decile. Average across problems showing mean +/- stderr bands. The x-axis is training step, y-axis is delta-KL, and each of the 10 lines represents a decile.

**Expected result:** Earlier token positions (lower decile lines) reduce their delta-KL faster than later tokens, confirming that the model learns from clean early signal but struggles with corrupted late signal.

### 4. Supervision Window Ablation

**Goal:** Demonstrate that late-sequence distillation signal is actively harmful, not merely uninformative.

**Procedure:**

- For 10 LCB problems, generate 8 rollouts with the model, collect the execution output
- Run 3 training steps of the standard SDPO objective (on these same problems)
- For each LCB problem, generate 8 new rollouts (**rollouts'**) with the updated model
- Measure the reward for **rollouts** and **rollouts'**, track the delta (reward improvement)
- Reset model weights and repeat the training and reward measurement steps under two ablated conditions:
  - SDPO objective applied only to the **first 50%** of rollout tokens
  - SDPO objective applied only to the **last 50%** of rollout tokens

**Expected result:** Full training and first-50% training should perform similarly, while last-50% should perform notably worse (lower average reward delta).

### 5. Top-k Token Analysis

**Goal:** Assess whether top-k filtering mitigates prefix corruption, and look for qualitative signs of teacher confusion in later positions.

**Procedure:**

- Re-run the Experiment 2 KL divergence analysis using the **full set of logprobs** rather than just the top-20
- Compare the degradation curves (top-20 vs full vocabulary)
- Separately, at each position in later deciles (50-100%), examine the probability of backtracking tokens (`"wait"`, `"actually"`, `"no"`, code comment tokens like `#`, `//`, etc.) in the teacher's distribution:
  - Relative to the teacher's distribution over the early deciles (0-50%)
  - Relative to the student's distribution over the same late positions (50-100%)

**Expected result:** Top-k filtering reduces absolute KL magnitude but doesn't flatten the degradation curve. In later deciles, the teacher increasingly places mass on backtracking/correction tokens, suggesting it's trying to "fix" the student's prefix rather than providing useful next-token signal.
