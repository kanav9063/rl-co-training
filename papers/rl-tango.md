# RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning

**Paper:** [arXiv:2505.15034](https://arxiv.org/abs/2505.15034) (NeurIPS 2025)
**Authors:** Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi (MIT)
**Code:** [github.com/kaiwenzha/rl-tango](https://github.com/kaiwenzha/rl-tango)

## Problem

Current RL post-training for LLMs uses **fixed verifiers** (rule-based or frozen pretrained reward models). These are:
- Susceptible to reward hacking
- Poor generalization beyond training distribution
- Even PRIME (the only prior co-training approach) uses discriminative, SFT-trained verifiers

## Core Innovation

**Tango** = interleaved RL training of both generator and verifier. The key breakthrough is the **generative, process-level LLM verifier trained via RL** (not SFT).

### Architecture
- **Generator (πg):** Standard LLM producing step-by-step solutions
- **Verifier (πv):** Generative LLM that outputs natural language feedback with step-level assessments + overall judgment
- Both initialized from base models (e.g., Qwen2.5-7B)

### Training Loop
1. Train generator for Ng steps using:
   - Rule-based outcome rewards (correct/incorrect final answer)
   - Step-level rewards from verifier (process-level feedback)
2. Train verifier for Nv steps using:
   - Outcome-level verification correctness rewards only
   - NO process-level annotations needed
3. Repeat interleaved

### Why This Works
1. **RL > SFT for verifiers:** Just as RL produces better generators than SFT, RL-trained verifiers develop stronger reasoning and generalization
2. **Stochastic rewards:** Generative sampling-based verification introduces randomness, making reward hacking harder (vs. deterministic PRM scores)
3. **Mutual reinforcement:** Generator gets better → harder problems for verifier → verifier improves → better feedback for generator

## Results

- **Generator:** SOTA among 7B/8B models on 5 competition-level math benchmarks + 4 OOD reasoning tasks
- **Verifier:** SOTA on ProcessBench (step-level verification)
- **AIME 2025:** Doubled accuracy vs. vanilla GRPO
- 25.5% average relative improvement on competition math vs. vanilla RL
- Outperforms PRIME, ORM-based, and PRM-based baselines

## Key Insights

1. **Co-evolution is key:** Neither generator-only nor verifier-only RL achieves the same results — the mutual reinforcement loop is essential
2. **Process supervision emerges from outcome supervision:** The verifier learns step-level assessment purely from outcome rewards
3. **Generative > Discriminative for verification:** Text-based reasoning about correctness beats scalar score prediction
4. **Compatible with multiple RL algorithms:** Works with GRPO, RLOO, and REINFORCE++

## Relevance to Our Research

- Direct template for implementing co-training on nanochat
- Their codebase is open source — can study implementation details
- Key question: Can we reproduce on smaller scale with nanochat's training setup?
