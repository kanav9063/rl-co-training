# GitHub Analysis: RL Co-Training & Generator-Verifier Implementations
*Analysis Date: 2026-02-12*

---

## Executive Summary

The open-source landscape for RL co-training and generator-verifier systems in LLMs is dominated by **three large frameworks** (veRL, TRL, OpenRLHF) that provide the infrastructure, with **specialized implementations** (SPIN, PRIME, SSP) demonstrating specific co-training paradigms. The field has matured rapidly since DeepSeek-R1's release in early 2025.

---

## Top Repos Ranked

| Rank | Repo | Stars | Co-Training Pattern | Maturity |
|------|------|-------|---------------------|----------|
| 1 | [verl-project/verl](https://github.com/verl-project/verl) | 19.2k | Framework (GRPO/PPO/DAPO) | Production |
| 2 | [huggingface/trl](https://github.com/huggingface/trl) | 17.3k | Framework (GRPO/DPO/Reward) | Production |
| 3 | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 9.0k | Framework (PPO/REINFORCE++/GRPO) | Production |
| 4 | [PRIME-RL/PRIME](https://github.com/PRIME-RL/PRIME) | 1.8k | Implicit PRM + RL co-training | Research |
| 5 | [uclaml/SPIN](https://github.com/uclaml/SPIN) | 1.2k | Self-play iterative refinement | Research |
| 6 | [Alibaba-Quark/SSP](https://github.com/Alibaba-Quark/SSP) | 90 | Adversarial proposer-solver self-play | Research |

---

## How Generator-Verifier Co-Training Is Implemented

### Pattern 1: Policy + Reward Model Co-Training (veRL, OpenRLHF, TRL)

The dominant pattern. A **generator** (policy LLM) produces outputs; a **verifier** (reward model or rule-based function) scores them; RL optimizes the generator.

**veRL implementation:**
- Generator and verifier run on separate GPU sets with flexible device mapping
- 3D-HybridEngine switches model between training and generation modes without memory duplication
- Hybrid-controller programming model: define RL dataflows (GRPO, PPO) in few lines
- Key file: dataflow definitions in recipe configs
- Co-evolution: PRIME recipe shows implicit PRM updating online alongside policy

**OpenRLHF implementation:**
- Ray orchestrates Actor (generator), Reward (verifier), Reference, Critic across GPUs
- Unified agent-based execution: `SingleTurnExecutor` and `MultiTurnExecutor`
- Custom reward functions enable any verifier pattern ([`--agent_func_path`](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_ray_agent_async.sh))
- Async training: generator and verifier can run asynchronously (`--async_train`)

**TRL implementation:**
- `GRPOTrainer` accepts `reward_funcs` parameter — plug in any verifier
- `RewardTrainer` trains a discriminative reward model
- Simplest entry point: `GRPOTrainer(model=..., reward_funcs=accuracy_reward, ...)`
- No explicit co-evolution of verifier; reward model is typically fixed

### Pattern 2: Implicit PRM Co-Evolution (PRIME)

**The most explicit generator-verifier co-training:**
- Generator (policy) produces reasoning chains
- Verifier (implicit PRM) provides **per-token dense rewards** via learned Q-function
- Key insight: PRM is trained as ORM with outcome labels only, then provides step-level rewards
- **Co-evolution loop:** Policy rollouts → outcome verification → PRM online update → denser rewards → better policy
- No separate PRM pre-training needed; SFT model is the starting PRM
- Implemented on veRL: [`recipe/prime`](https://github.com/volcengine/verl/tree/main/recipe/prime)
- Paper shows REINFORCE++ more stable than GRPO for this co-training

### Pattern 3: Self-Play Iterative Refinement (SPIN)

**Self-as-verifier pattern:**
- Current model generates responses; previous iteration's ground truth is the "real" signal
- Training objective: distinguish real (SFT data) from self-generated responses
- No external verifier — the training loss itself acts as implicit verification
- Iterative: model_t generates data → train model_{t+1} to tell apart real vs generated → repeat
- Converges when model can no longer distinguish its outputs from target distribution
- Simple but effective: outperforms DPO without labeled preference data

### Pattern 4: Adversarial Dual-Agent Self-Play (SSP)

**Most sophisticated co-evolution:**
- **Proposer** (generator of hard problems) and **Solver** (generator of solutions) co-evolve
- Both have search engine access for multi-turn reasoning
- Different RL algorithms per role (e.g., GRPO for solver, PPO for proposer)
- Rule-based outcome rewards: solver gets reward for correct answers; proposer gets reward when solver fails
- Built on veRL + Search-R1; requires LLM-as-Judge + retrieval infrastructure
- Zero human annotation needed

---

## Cross-Repo Comparison

### Infrastructure Stack

| Feature | veRL | OpenRLHF | TRL |
|---------|------|----------|-----|
| Distribution | FSDP/Megatron + vLLM/SGLang | Ray + vLLM + DeepSpeed | Accelerate + DeepSpeed |
| Scale | 671B+ MoE demonstrated | 70B+ | Single-node typical |
| Generation Engine | vLLM, SGLang | vLLM | HF generate / vLLM |
| Async RL | Experimental modules | Yes (`--async_train`) | No |
| Multi-turn Agent | Via recipes | Yes (MultiTurnExecutor) | OpenEnv integration |
| Co-training recipes | PRIME, DAPO, SSP | MARTI, custom rewards | Reward + GRPO trainers |

### Suitability for Co-Training Research

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| Quick prototyping | **TRL** | Simplest API, `GRPOTrainer` + custom reward in 10 lines |
| Large-scale production | **veRL** | Best throughput, Megatron support, 671B demonstrated |
| Async co-evolution | **OpenRLHF** | Native async training, Ray-based separation of concerns |
| Process reward co-training | **PRIME** (on veRL) | Only implementation of online implicit PRM updating |
| Self-play without external rewards | **SPIN** | Standalone, no reward model needed |
| Adversarial dual-agent | **SSP** (on veRL) | Only open implementation of proposer-solver co-evolution |

### Key Architectural Insight

All three frameworks separate **generation** from **training** as a core design principle:
- veRL: 3D-HybridEngine resharding
- OpenRLHF: Ray actors separate generation/training GPUs
- TRL: vLLM integration for generation, HF Trainer for updates

This separation is what enables generator-verifier co-training: the verifier (reward model) occupies its own compute while the generator alternates between producing samples and updating weights.

---

## Dependencies & Setup Comparison

| Repo | Python | Key Deps | GPU Requirement |
|------|--------|----------|-----------------|
| veRL | 3.10+ | torch, vLLM/SGLang, FSDP/Megatron | Multi-GPU (8+ for large models) |
| OpenRLHF | 3.10+ | torch, Ray, vLLM, DeepSpeed | Multi-GPU via Ray |
| TRL | 3.9+ | transformers, accelerate, PEFT | Single GPU possible |
| SPIN | 3.10 | transformers, flash-attn | 4-8 GPUs (7B model) |
| PRIME | 3.10+ | veRL, transformers | Multi-GPU |
| SSP | 3.10 | veRL, SGLang, FAISS, torch 2.6 | Multi-GPU + retrieval server |

---

## Key Findings

1. **veRL is the gravitational center**: PRIME, SSP, DAPO all build on it. Its recipe system makes it the go-to for co-training research at scale.

2. **PRIME is the purest generator-verifier co-training**: Online PRM updating alongside policy is the closest implementation to the theoretical "generator-verifier co-evolution" paradigm.

3. **SSP is the most novel**: Dual-agent adversarial self-play with tool use — extends co-training beyond reward models into adversarial curriculum generation.

4. **SPIN proves self-play works without external rewards**: Important baseline showing iterative self-improvement converges without any reward model.

5. **TRL democratizes access**: While less powerful for co-training research, it's where most practitioners will first experiment with GRPO + custom rewards.

6. **GenRM (Generative Verifiers)** is a key concept but only has data releases, not full training code. The idea of verifier-as-generator (next-token prediction for reward modeling) bridges generator and verifier architectures.

---

## Recommended Reading Order for Implementation

1. Start with [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/grpo_trainer) — understand the basic loop
2. Read [PRIME paper](https://arxiv.org/abs/2502.01456) — understand implicit PRM co-training
3. Study [veRL recipe/prime](https://github.com/volcengine/verl/tree/main/recipe/prime) — see co-training at scale
4. Read [SPIN paper](https://arxiv.org/abs/2401.01335) — understand self-play without rewards
5. Read [SSP paper](https://arxiv.org/abs/2510.18821) — understand adversarial dual-agent co-evolution
