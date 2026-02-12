# RL Co-Training: Generator-Verifier Paradigms — Distilled Sources (Top 20%)

*Collected: 2026-02-12*

---

## Taxonomy of Approaches

The field splits into three related paradigms:

### A. Generator-Verifier Co-Training (direct mutual reinforcement)
### B. Multi-Role Self-Play (proposer/solver/judge from same model)
### C. Online Reward Model Training (verifier updates alongside policy)

---

## Tier 1 Papers (Must-Read)

### 1. RL Tango (Zha et al., NeurIPS 2025)
- **Paper:** https://arxiv.org/abs/2505.15034
- **Code:** https://github.com/kaiwenzha/rl-tango
- **Category:** A — Generator-Verifier Co-Training

**Core idea:** Interleaved RL training of LLM generator + generative process-level LLM verifier. Both trained via RL (not SFT). Verifier provides natural-language step-level feedback + overall judgment. Generator uses outcome correctness + verifier's step rewards.

**Why it matters:**
- First to train BOTH generator and verifier via RL (not just online SFT for verifier)
- Generative verifier introduces stochasticity → resists reward hacking (vs discriminative PRMs)
- Verifier needs only outcome-level correctness rewards, yet learns process-level verification
- 25.5% relative improvement on math; SOTA on 7B/8B scale across 9 benchmarks
- Directly addresses limitations of PRIME's discriminative approach

**Training loop:** Generator rollouts → verifier evaluates steps → generator updated with combined outcome+step rewards → verifier updated with outcome-level verification correctness → repeat interleaved

---

### 2. PRIME (Cui et al., 2025)
- **Paper:** https://arxiv.org/abs/2502.01456
- **Code:** https://github.com/PRIME-RL/PRIME
- **Category:** C — Online Reward Model Training

**Core idea:** Online PRM updated using only policy rollouts + outcome labels through implicit process rewards. No dedicated reward model training phase.

**Why it matters:**
- Precursor to Tango's co-training; demonstrated online PRM updating is critical
- 2.5× sample efficiency over outcome-only rewards (RLOO)
- Shows PRM initialized from SFT model + online updates outperforms all alternatives
- Limitation (identified by Tango): still discriminative/logit-based → deterministic rewards → reward hacking vulnerable

**Training loop:** Policy generates rollouts → implicit PRM scores + outcome verifier scores → PRM updated on rollouts with outcome reward → advantages computed from dense PRM rewards → policy updated

---

### 3. Absolute Zero Reasoner (Zhao et al., NeurIPS 2025)
- **Paper:** https://arxiv.org/abs/2505.03335
- **Code:** https://github.com/LeapLabTHU/Absolute-Zero-Reasoner
- **Category:** B — Multi-Role Self-Play

**Core idea:** Single model plays dual roles (Proposer + Solver). Proposer generates code reasoning tasks; Solver solves them. Code executor provides unified verifiable reward. Zero external data.

**Why it matters:**
- Demonstrates self-improving curriculum generation without any human data
- SOTA on coding + math reasoning among zero-setting models
- Proposer reward based on task validity + difficulty calibration (learnability)
- Shows co-evolution works even within a single model playing multiple roles

---

### 4. ASL: Agentic Self-Learning (Sun et al., 2025)
- **Paper:** https://arxiv.org/abs/2510.14253
- **Code:** https://github.com/forangel2014/Towards-Agentic-Self-Learning
- **Category:** A+B — Full Multi-Role Co-Evolution

**Core idea:** Prompt Generator + Policy Model + Generative Reward Model (GRM) form virtuous cycle in search-agent setting. All co-evolve.

**Critical finding: GRM is the bottleneck.** If frozen, it induces reward hacking and stalls progress. Continual GRM training on evolving distribution is essential. Small late-stage injection of real data raises ceiling.

**Why it matters:**
- Empirically proves frozen verifiers are the failure mode
- Shows GRM > rule-based rewards for open-domain learning
- Extends co-training beyond math/code to agentic search tasks
- Demonstrates round-over-round gains while baselines plateau

---

### 5. APRM: Adversarial Training for PRMs (Juneja et al., 2025)
- **Paper:** https://arxiv.org/abs/2511.22888
- **Code:** https://gurusha01.github.io/PRM_NIPS/
- **Category:** A — Generator-Verifier Co-Training (adversarial)

**Core idea:** Generator learns to produce plausible reasoning errors; PRM learns to detect them. Two-player general-sum game with convergence guarantees to Nash Equilibrium.

**Why it matters:**
- Only paper with theoretical convergence guarantees for generator-verifier co-training
- General-sum (not zero-sum) framing avoids GAN instabilities
- +5.3pp on OOD tasks — strongest generalization gains
- Adaptive curriculum: negative sample hardness dynamically increases with PRM capability

---

## Tier 2 Papers (Important Context)

### 6. MARS (Chen et al., 2025)
- **Paper:** https://arxiv.org/abs/2510.04935
- **Category:** A — Multi-Agent Co-Training for Deep Research

System 1 (summarizer) + System 2 (reasoner) co-evolve via shared trajectory rewards in multi-agent GRPO. Key technical contribution: decoupled gradient computation for proper credit assignment. 8B model matches Claude 3.7 Sonnet on HLE benchmark.
- https://arxiv.org/abs/2510.04935

### 7. Multi-Agent Evolve / MAE (Chen et al., 2025)
- **Paper:** https://arxiv.org/abs/2510.23595
- **Code:** https://github.com/ulab-uiuc/Multi-agent-Evolve
- **Category:** B — Proposer/Solver/Judge Self-Play

Three roles from single LLM. Extends self-play to general domains. Adversarial: Proposer rewarded when Solver fails. 4.54% improvement on Qwen2.5-3B.
- https://arxiv.org/abs/2510.23595

### 8. Self-Rewarding Language Models (Yuan et al., Meta 2024)
- **Paper:** https://arxiv.org/abs/2401.10020
- **Category:** C — Self-Judging

Foundational work showing same model can be both generator and judge. Iterative DPO improves both capabilities. Precursor to all co-training work.
- https://arxiv.org/abs/2401.10020

### 9. R-Zero (Huang et al., 2025)
- **Paper:** https://arxiv.org/abs/2508.05004
- **Category:** B — Challenger-Solver Co-Evolution

Two independent models: Challenger proposes tasks at edge of Solver's capability. Clean separation of roles. +6.49 math, +7.54 general reasoning on Qwen3-4B-Base.
- https://arxiv.org/abs/2508.05004

### 10. GenRM: Generative Verifiers (ICLR 2025)
- **Paper:** https://openreview.net/forum?id=Ccwp4tFEtE
- **Category:** Foundation — Generative Verification

Trains verifiers via next-token prediction (not classification). Enables CoT reasoning for verification. Foundation that Tango builds upon for co-training.
- https://openreview.net/forum?id=Ccwp4tFEtE

---

## Key Findings & Synthesis

### The Co-Training Loop (Consensus Design)
All successful approaches share this core loop:
1. **Generator produces rollouts** (solutions, reasoning traces)
2. **Verifier evaluates** (step-level or outcome-level)
3. **Generator updated** using verifier signals as rewards
4. **Verifier updated** using outcome correctness on generator's outputs
5. **Repeat** — both improve each iteration

### Critical Design Choices

| Choice | Better | Worse | Evidence |
|--------|--------|-------|----------|
| Verifier type | Generative (CoT) | Discriminative (logit) | Tango vs PRIME |
| Verifier training | RL | SFT | Tango ablations |
| Verifier state | Online/co-evolving | Frozen | ASL (frozen → reward hacking) |
| Reward granularity | Process-level (dense) | Outcome-level (sparse) | PRIME (2.5× efficiency) |
| Game formulation | General-sum | Zero-sum | APRM (stability) |

### Failure Modes
1. **Frozen verifier → reward hacking** (ASL, Tango both demonstrate) — Source: https://arxiv.org/abs/2510.14253
2. **Discriminative PRM → deterministic rewards → exploitable** — Source: https://arxiv.org/abs/2505.15034
3. **SFT-trained verifier → limited reasoning, poor generalization** — Source: https://arxiv.org/abs/2505.15034
4. **Training imbalance** between components (addressed by MARS with advantage-weighted balanced sampling) — Source: https://arxiv.org/abs/2510.04935

### Open Questions
1. **Scaling laws:** No systematic study of how co-training benefits scale with model size
2. **Compute efficiency:** Is 2× compute for co-training better than 2× compute for generator-only RL?
3. **Beyond verifiable domains:** Math/code have ground truth. How to co-train for open-ended reasoning?
4. **Collusion/mode collapse:** Could generator and verifier converge to mutually reinforcing but wrong patterns?
5. **Optimal interleaving schedule:** How often to update each component? Tango does interleaved; PRIME does each iteration.

### Implementation Resources
- **RL Tango:** https://github.com/kaiwenzha/rl-tango (full training code)
- **PRIME:** https://github.com/PRIME-RL/PRIME (scalable RL with process rewards)
- **AZR:** https://github.com/LeapLabTHU/Absolute-Zero-Reasoner (zero-data self-play)
- **MAE:** https://github.com/ulab-uiuc/Multi-agent-Evolve (multi-agent self-evolution)
- **ASL:** https://github.com/forangel2014/Towards-Agentic-Self-Learning (agentic co-learning)
- **SPIN:** https://github.com/uclaml/SPIN (self-play fine-tuning)
- **SPPO:** https://github.com/uclaml/SPPO (self-play preference optimization)
