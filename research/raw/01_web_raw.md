# RL Co-Training: Generator-Verifier Paradigms — Raw Web Research

*Collected: 2026-02-12*

---

## Search Results & Scraped Content

### 1. RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning
- **URL:** https://arxiv.org/abs/2505.15034
- **Authors:** Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi (MIT)
- **Venue:** NeurIPS 2025
- **Code:** https://github.com/kaiwenzha/rl-tango

**Abstract:** RL to concurrently train both an LLM generator and a verifier in an interleaved manner. Central innovation: generative, process-level LLM verifier trained via RL that co-evolves with the generator. Verifier trained solely on outcome-level verification correctness rewards without process-level annotations. SOTA results among 7B/8B-scale models on 5 competition-level math benchmarks and 4 OOD reasoning tasks. Verifier leads on ProcessBench.

**Key Technical Details (from HTML):**
- Generator produces multi-step reasoning trajectories; verifier offers natural language feedback with step-level assessments + overall correctness judgment
- Generator uses gold outcome-level correctness + step-level rewards from verifier
- Verifier trained exclusively using outcome-level verification correctness rewards (no process-level annotations)
- Through RL, verifier progressively refines chain-of-thought verification reasoning
- Training the verifier via RL (not SFT) → stronger reasoning skills, better generalization
- Generative sampling-based verifier introduces stochasticity → robustness against reward hacking
- Interleaved training: generator and verifier mutually reinforce each other
- Compared to PRIME: PRIME uses discriminative logit-based PRM (deterministic rewards → reward hacking), trained with SFT despite online collection
- 25.5% average relative improvement on math benchmarks

---

### 2. MARS: Co-evolving Dual-System Deep Research via Multi-Agent Reinforcement Learning
- **URL:** https://arxiv.org/abs/2510.04935
- **Authors:** Guoxin Chen, Zile Qiao, Wenqing Wang, et al.
- **Year:** 2025 (ongoing work, v2 Feb 2026)

**Abstract:** MARS (Multi-Agent System for Deep ReSearch) jointly optimizes dual cognitive systems through multi-agent RL. System 1 (fast, intuitive processing) and System 2 (deliberate reasoning) co-adapt through shared trajectory rewards. System 1 learns to distill information useful for System 2's reasoning.

**Key innovations:**
1. Decoupled gradient computation for proper credit assignment despite shared rewards
2. Bin-packing optimization for efficient parallel information processing
3. Advantage-weighted balanced sampling preventing training imbalance
- Extends GRPO for multi-agent settings
- MARS (8B) under Zero RL setting (no SFT): 8.17% on HLE, outperforming WebThinker (32B with SFT, 6.87%)
- Average gain of 8.9% across 7 knowledge-intensive tasks

---

### 3. PRIME: Process Reinforcement through Implicit Rewards
- **URL:** https://arxiv.org/abs/2502.01456
- **Authors:** Ganqu Cui, Lifan Yuan, Zefan Wang, et al.
- **Code:** https://github.com/PRIME-RL/PRIME

**Abstract:** Online PRM updates using only policy rollouts and outcome labels through implicit process rewards. Combines with various advantage functions. Forgoes dedicated reward model training phase. Starting from Qwen2.5-Math-7B-Base, achieves 15.1% average improvement across reasoning benchmarks. Eurus-2-7B-PRIME surpasses Qwen2.5-Math-7B-Instruct with 10% of training data.

**Key details:**
- Policy model and PRM both initialized from SFT model
- Each RL iteration: policy generates rollouts → implicit PRM + outcome verifier score → PRM updated on rollouts with outcome reward → dense rewards estimate advantages
- Online updating prevents reward hacking (vs frozen reward models)
- 2.5× sample efficiency vs RLOO with outcome verifier

---

### 4. Multi-Agent Evolve (MAE): LLM Self-Improve through Co-evolution
- **URL:** https://arxiv.org/abs/2510.23595
- **Authors:** Yixing Chen, Yiding Wang, Siqi Zhu, et al. (UIUC, Peking U, NVIDIA)
- **Code:** https://github.com/ulab-uiuc/Multi-agent-Evolve

**Abstract:** Three interacting agents (Proposer, Solver, Judge) from a single LLM forming a closed self-improving loop. Proposer generates questions, Solver answers, Judge evaluates. Adversarial interaction: Solver rewarded for accurate answers, Proposer gets quality reward + difficulty reward when Solver fails.

**Results:** 4.54% average improvement on Qwen2.5-3B-Instruct across multiple benchmarks. Extends self-play to general domains (not just code/math).

---

### 5. Agentic Self-Learning (ASL) / Towards Agentic Self-Learning LLMs in Search Environment
- **URL:** https://arxiv.org/abs/2510.14253
- **Authors:** Wangtao Sun, Xiang Cheng, et al.
- **Code:** https://github.com/forangel2014/Towards-Agentic-Self-Learning

**Abstract:** Multi-role RL framework: Prompt Generator, Policy Model, Generative Reward Model (GRM) form virtuous cycle. Co-evolving GRM with policy boosts performance. Key finding: **GRM verification capacity is the main bottleneck** — if frozen, induces reward hacking and stalls progress. Continual GRM training on evolving data mitigates this. Small late-stage injection of real verification data raises performance ceiling.

**Critical insight:** Rewards from GRM outperform rigid rule-based signals for open-domain learning. When GRM co-evolved with policy → stronger generative discrimination capabilities.

---

### 6. Self-Rewarding Language Models
- **URL:** https://arxiv.org/abs/2401.10020
- **Authors:** Weizhe Yuan, Richard Yuanzhe Pang, et al. (Meta, NYU)
- **Venue:** 2024

**Abstract:** LLM itself provides its own rewards via LLM-as-a-Judge prompting during training. Iterative DPO: not only instruction following improves, but also ability to provide high-quality rewards. Llama 2 70B outperforms Claude 2, Gemini Pro, GPT-4 0613 on AlpacaEval 2.0.

**Key mechanism:** Self-instruction creation → candidate responses scored by same model → preference dataset → DPO training → next iteration. Both generation and reward modeling improve across iterations.

---

### 7. Absolute Zero Reasoner (AZR)
- **URL:** https://arxiv.org/abs/2505.03335
- **Authors:** Andrew Zhao, Yiran Wu, et al. (Tsinghua)
- **Venue:** NeurIPS 2025
- **Code:** https://github.com/LeapLabTHU/Absolute-Zero-Reasoner

**Abstract:** Single model learns to propose tasks maximizing its own learning progress + solves them, without external data. Uses code executor as unified source of verifiable reward. SOTA on coding and math reasoning, outperforming zero-setting models using tens of thousands of human-curated examples.

**Dual roles:** Proposer generates challenging coding tasks, Solver solves them. Co-evolution through shared model.

---

### 8. R-Zero: Self-Evolving Reasoning LLM from Zero Data
- **URL:** https://arxiv.org/abs/2508.05004
- **Authors:** Chengsong Huang et al.

**Abstract:** Two independent models: Challenger + Solver, co-evolve through interaction. Challenger rewarded for proposing tasks near edge of Solver capability. Solver rewarded for solving increasingly challenging tasks. Targeted self-improving curriculum without pre-existing tasks/labels. Qwen3-4B-Base: +6.49 on math, +7.54 on general reasoning.

---

### 9. SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong
- **URL:** https://arxiv.org/abs/2401.01335
- **Authors:** Zixiang Chen, Yihe Deng, et al. (UCLA)
- **Venue:** ICML 2024
- **Code:** https://github.com/uclaml/SPIN

**Abstract:** Self-play mechanism where LLM refines by playing against previous iterations. Generates own training data from previous iterations, distinguishes self-generated from human-annotated. Global optimum achieved when LLM aligns with target distribution. Outperforms DPO with extra GPT-4 preference data.

---

### 10. SPPO: Self-Play Preference Optimization
- **URL:** https://uclaml.github.io/SPPO/
- **Authors:** UCLA ML
- **Code:** https://github.com/uclaml/SPPO

Two-player constant-sum game with squared-loss objective. Using PairRM (0.4B), achieves 28.53% win-rate against GPT-4-Turbo on AlpacaEval 2.0.

---

### 11. APRM: Adversarial Training for Process Reward Models
- **URL:** https://arxiv.org/abs/2511.22888
- **Authors:** Gurusha Juneja, Deepak Nathani, William Yang Wang (UCSB)
- **Code:** https://gurusha01.github.io/PRM_NIPS/

**Abstract:** Generator learns to produce reasoning errors to deceive PRM; PRM concurrently learns to detect them. Two-player general-sum non-cooperative game (not zero-sum). Game-aware optimizers + symmetric policy regularization → linear-rate convergence to Nash Equilibrium. +3.4pp average, +5.3pp on OOD tasks.

**Key difference from GANs:** General-sum game yields stationary-point NE rather than minimax saddle point.

---

### 12. MAPoRL: Multi-Agent Post-Co-Training for Collaborative LLMs with RL
- **URL:** https://arxiv.org/abs/2502.18439
- **Authors:** Chanwoo Park et al.
- **Venue:** ACL

**Abstract:** Multiple LLMs generate responses independently → multi-turn discussion → MAPoRL verifier evaluates answer + discussion (correctness + incentives for corrective/persuasive discussions). Score maximized through multi-agent RL. Key finding: training individual LLMs alone insufficient for collaboration; multi-agent co-training boosts across benchmarks with generalization to unseen domains.

---

### 13. GenRM: Generative Verifiers - Reward Modeling as Next-Token Prediction
- **URL:** https://openreview.net/forum?id=Ccwp4tFEtE
- **Venue:** ICLR 2025

**Abstract:** Train verifiers using next-token prediction objective on verification + solution generation jointly. GenRM enables CoT reasoning for verification, majority voting for better verification. Outperforms discriminative/DPO verifiers: 5%→45.3% on algorithmic tasks, 73%→93.4% on GSM8K, 28%→44.6% on MATH easy-to-hard.

**Foundation for Tango:** Tango builds on GenRM concept by co-training the generative verifier with the generator via RL.

---

### 14. Spark: Stepwise Process-Aware Rewards for Reference-Free RL
- **URL:** https://arxiv.org/abs/2512.03244
- **Authors:** Salman Rahman et al. (Amazon AGI, UCLA)

**Abstract:** Three-stage framework: (1) generator produces solutions, verifier evaluates via self-consistency + meta-critique; (2) synthetic training data fine-tunes generative PRMs; (3) PRM-CoT as reward in RL. Achieves 47.4% across 6 math benchmarks vs 43.9% for ground-truth RLVR. Reference-free RL exceeding ground-truth methods.

**Notes:** Mentions TANGO and PRIME as co-evolving approaches. Points out both still depend on ground truth references.

---

### 15. ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf
- **Venue:** NeurIPS 2024

MuZero-style learning for LLMs. Iteratively employs policy + process reward model, using search trees for self-improvement.

---

### 16. EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
- **URL:** https://arxiv.org/abs/2511.16672

Extends self-improvement ideas to multimodal setting. Uses agreement among multiple sampled answers as self-reward signal.

---

### 17. RAGEN: Understanding Self-Evolution in LLM Agents via RL
- **URL:** https://ragen-ai.github.io/pdf/RAGEN.pdf

Frames agent problem as MDP, studies self-evolution through RL training.

---

## Patterns Identified

1. **Core paradigm convergence:** Multiple independent groups converging on generator-verifier co-training (Tango, PRIME, ASL, APRM, MAE)
2. **Verifier type matters:** Generative verifiers (Tango, GenRM) > discriminative verifiers. RL-trained > SFT-trained.
3. **Frozen verifiers fail:** ASL explicitly shows frozen GRM → reward hacking + stalled progress. Tango argues same.
4. **Self-play as dual roles:** AZR, R-Zero, MAE all use proposer/challenger + solver from single model
5. **GAN analogy but different:** APRM explicitly uses general-sum (not zero-sum) game theory. Adversarial but cooperative.
6. **Zero-data frontier:** AZR, R-Zero, MAE pushing toward self-improving without human data
7. **Process vs outcome rewards:** Dense process rewards (PRIME, Tango) >> sparse outcome rewards for training efficiency
8. **Credit assignment challenge:** MARS uses decoupled gradients for multi-agent settings with shared rewards

## Gaps & Questions

1. **Scaling laws for co-training:** No paper explicitly studies how co-training scales with model size
2. **Training instability:** Only APRM provides convergence guarantees (Nash equilibrium). Others rely on empirical stability.
3. **Beyond math/code:** Most work is on math. ASL extends to search/agentic, MAE to general QA, but open-ended reasoning largely unexplored.
4. **Mode collapse:** Risk of generator-verifier collusion not well studied
5. **Compute overhead:** Co-training is ~2x compute. No work on whether it's worth it per-FLOP vs just training generator longer.
