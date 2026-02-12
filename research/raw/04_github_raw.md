# GitHub Repos: RL Co-Training, Generator-Verifier Co-Evolution & Self-Play LLM Training
## Raw Data Collection — 2026-02-12

---

## 1. verl (Volcano Engine Reinforcement Learning for LLMs)
- **Repo:** https://github.com/verl-project/verl
- **Stars:** 19,170 | **Updated:** 2026-02-12
- **Description:** Flexible, efficient, production-ready RL training library for LLMs. Open-source version of HybridFlow (EuroSys paper).
- **Paper:** https://arxiv.org/abs/2409.19256v2

### Key Features
- Hybrid-controller programming model for flexible RL dataflow representation
- Supports GRPO, PPO, REINFORCE++, DAPO, VAPO, PF-PPO algorithms
- Integrates FSDP, Megatron-LM, vLLM, SGLang backends
- 3D-HybridEngine for efficient actor model resharding (eliminates memory redundancy between training/generation)
- Flexible device mapping across GPU sets
- Native HuggingFace model integration
- Experimental modules: `transfer_queue`, `fully_async_policy`, `one_step_off_policy`, `vla`

### Co-Training Relevance
- PRIME recipe integrated into veRL main branch: https://github.com/volcengine/verl/tree/main/recipe/prime
- DAPO recipe: achieves 50 pts on AIME 2024 with Qwen2.5-32B
- SSP (Search Self-Play) is built on top of veRL
- Seed-Thinking-v1.5 trained with veRL (86.7 AIME 2024)
- ReTool recipe for multi-round tool-augmented RL
- Trillion-parameter GRPO LoRA training demonstrated on 64 H800s

### Architecture
- ByteDance Seed team initiated, community maintained
- Decouples computation and data dependencies
- Modular APIs for seamless LLM framework integration

---

## 2. huggingface/trl (Transformer Reinforcement Learning)
- **Repo:** https://github.com/huggingface/trl
- **Stars:** 17,348 | **Updated:** 2026-02-12
- **Description:** Comprehensive library to post-train foundation models

### Key Trainers
- `SFTTrainer` — Supervised Fine-Tuning
- `GRPOTrainer` — Group Relative Policy Optimization (DeepSeek R1 algorithm)
- `DPOTrainer` — Direct Preference Optimization
- `RewardTrainer` — Reward model training
- OpenEnv integration (Meta's RL environment framework)

### Co-Training Relevance
- `RewardTrainer` + `GRPOTrainer` enable generator-verifier pipelines
- GRPO implementation directly usable with custom reward functions (accuracy_reward, reasoning_accuracy_reward)
- Built on HuggingFace ecosystem (Accelerate, PEFT, Unsloth)
- CLI interface for quick experimentation
- Most accessible entry point for co-training experiments

### Example (GRPO with custom reward):
```python
from trl import GRPOTrainer
from trl.rewards import accuracy_reward
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

---

## 3. OpenRLHF/OpenRLHF
- **Repo:** https://github.com/OpenRLHF/OpenRLHF
- **Stars:** 8,988 | **Updated:** 2026-02-12
- **Description:** First high-performance RLHF framework with Ray + vLLM distributed architecture and unified agent-based design paradigm

### Architecture
- **Ray** — distributed scheduler/controller, separates Actor/Reward/Reference/Critic across GPUs
- **vLLM** — high-performance inference (AutoTP, PP); 80% of RLHF time is generation
- **DeepSpeed** — ZeRO-3, deepcompile, AutoTP, RingAttention
- **Hybrid Engine Scheduling** — all models + vLLM share GPU resources
- **Agent-Based Execution** — unified token-in-token-out paradigm

### Algorithms
- PPO, REINFORCE++, REINFORCE++-baseline, GRPO, RLOO
- Async RLHF training (`--async_train`)
- Async agent RLHF (`--agent_func_path`)

### Agent Architecture
```
AgentExecutorBase (Token-in-Token-out Core)
  ├── SingleTurnExecutor
  │     ├── Standard RLHF (One-shot gen)
  │     └── Custom Reward Function
  └── MultiTurnExecutor
        ├── Multi-Step Reasoning
        └── External Env (NeMo Gym)
```

### Co-Training Relevance
- ProRL V2 (NVIDIA): REINFORCE++-baseline for 1.5B reasoning model
- PRIME integration demonstrated
- Logic-RL: validates REINFORCE++ > GRPO stability
- MARTI fork: multi-agent RL training (centralized multi-agent interactions + distributed policy training)
- LMM-R1 fork: multimodal DeepSeek-R1 reproduction
- simpleRL-reason: DeepSeek-R1-Zero reproduction on small models
- Custom reward functions enable verifier-as-reward patterns

---

## 4. uclaml/SPIN (Self-Play Fine-Tuning)
- **Repo:** https://github.com/uclaml/SPIN
- **Stars:** 1,233 | **Updated:** 2026-02-11
- **Paper:** https://arxiv.org/abs/2401.01335 (ICML 2024)
- **Authors:** UCLA — Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu

### Core Mechanism
- LLM improves by playing against its **previous iterations**
- No additional human-annotated preference data needed (only SFT dataset)
- Model generates training data from previous iteration; refines policy by distinguishing self-generated responses from original SFT data

### Training Loop (2-step iterative):
1. **Step 1: Generation** — Current model generates responses to SFT prompts (supports vLLM for speed)
2. **Step 1.5: Gather** — Collect generations, convert to training format
3. **Step 2: Fine-tuning** — Train to distinguish real (SFT) from generated (self-play) responses

### Data Format:
```json
{
    "real": [{"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<ground truth>"}],
    "generated": [{"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<generation>"}]
}
```

### Results
- Outperforms DPO on multiple benchmarks after iteration 1
- Converges as model approaches target data distribution
- Released models: zephyr-7b-sft-full-SPIN-iter{0,1,2,3}
- Released datasets: SPIN_iter{0,1,2,3}

### Dependencies
- Python 3.10, flash-attn, HuggingFace alignment-handbook base

---

## 5. PRIME-RL/PRIME (Process Reinforcement through IMplicit rEwards)
- **Repo:** https://github.com/PRIME-RL/PRIME
- **Stars:** 1,805 | **Updated:** 2026-02-11
- **Paper:** https://arxiv.org/abs/2502.01456

### Core Innovation: Implicit Process Reward Model
- Trained as ORM (outcome reward model), used as PRM (process reward model)
- No process labels needed — learns Q-function providing per-token rewards
- SFT model itself serves as strong PRM starting point

### Three Benefits for RL:
1. **Dense Reward:** Per-token rewards via Q-function, no extra value model needed
2. **Scalability:** Online-updated with only outcome labels; on-policy rollouts mitigate distribution shift
3. **Simplicity:** PRM is a language model; no separate PRM pre-training needed

### Generator-Verifier Co-Training Pattern:
- **Generator (Policy):** Generates reasoning chains
- **Verifier (Implicit PRM):** Scores each token/step using implicit rewards
- **Co-evolution:** PRM updates online with policy rollouts → both improve together
- Integrated into veRL: https://github.com/volcengine/verl/tree/main/recipe/prime

### Code Structure:
- `training/` — Training scripts
- `data_preprocessing/` — Math data preparation
- `eval/` — Evaluation scripts
- Separate repo for Implicit PRM: https://github.com/PRIME-RL/ImplicitPRM

---

## 6. Alibaba-Quark/SSP (Search Self-Play)
- **Repo:** https://github.com/Alibaba-Quark/SSP
- **Stars:** 90 | **Updated:** 2026-02-05
- **Paper:** https://arxiv.org/abs/2510.18821

### Architecture: Adversarial Self-Play with Search
- **Proposer agent:** Generates increasingly challenging problems with search engine access
- **Solver agent:** Develops stronger search + reasoning to solve problems
- Both agents do multi-turn search engine calling and reasoning
- Different RL algorithms per role (PPO, GRPO, REINFORCE)
- Rule-based outcome rewards for RL training

### Co-Evolution Mechanism:
- Proposer learns to generate harder problems requiring search/reasoning
- Solver develops stronger capabilities to tackle them
- Pure self-play: no human-annotated QA pairs needed

### Built On:
- veRL (specific commit: a970718ea525b161e1c5c4285e5f8e7ea7663813)
- Search-R1 (https://github.com/PeterGriffinJin/Search-R1)

### Infrastructure:
- LLM-as-a-Judge service for problem evaluation
- Local dense retriever (e5-base-v2 + FAISS)
- SGLang for multi-turn inference

### Dependencies:
- torch 2.6.0, flash-attn 2.5.8, flashinfer 0.2.0, sglang 0.4.6.post5
- transformers 4.55.0, math_verify 0.8.0

---

## 7. GenRM (Generative Reward Models / Generative Verifiers)
- **Data Repo:** https://github.com/genrm-star/genrm-critiques
- **Paper:** https://arxiv.org/abs/2408.15240 ("Generative Verifiers: Reward Modeling as Next-Token Prediction")
- **Website:** https://sites.google.com/corp/view/generative-reward-models
- **Code Repo:** https://github.com/nishadsinghi/sc-genrm-scaling (COLM 2025 — "When To Solve, When To Verify")

### Concept:
- Reward modeling reframed as next-token prediction
- Verifier generates chain-of-thought critiques before scoring
- GenRM-CoT: verification rationales dataset (GSM8K based)

---

## 8. sanowl/OmegaPRM
- **Repo:** https://github.com/sanowl/OmegaPRM
- **Stars:** 44
- **Description:** Implementation of "Improve Mathematical Reasoning in Language Models by Automated Process Supervision" (Google DeepMind)

---

## 9. Other Notable Repos

### UCSC-VLAA/MedVLSynther
- **Repo:** https://github.com/UCSC-VLAA/MedVLSynther
- **Paper:** ICLR 2026 — Generator-Verifier LMMs for medical VQA
- Generator creates synthetic QA; Verifier validates quality

### hellonish/SmolSolver
- **Repo:** https://github.com/hellonish/SmolSolver
- **Stars:** 1
- **Description:** SLM-based math reasoning with Generator + Verifier architecture

### TsinghuaC3I/MARTI
- **Repo:** https://github.com/TsinghuaC3I/MARTI
- **Description:** LLM-based multi-agent reinforced training (OpenRLHF fork)

### thomasgauthier/LLM-self-play
- **Repo:** https://github.com/thomasgauthier/LLM-self-play
- **Description:** Minimal SPIN implementation with GPT-Neo

### martyn/spin-toy
- **Repo:** https://github.com/martyn/spin-toy
- **Description:** Toy SPIN implementation
