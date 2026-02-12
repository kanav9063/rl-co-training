# RL Co-Training: Generator-Verifier Co-Evolution for LLM Reasoning
## Distilled Paper Analysis & Synthesis
Generated: 2026-02-11

---

## 1. Core Co-Training Papers (Generator + Verifier Jointly Trained)

### 1.1 RL Tango (arXiv: 2505.15034) ⭐ PRIMARY
**"RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning"**
Zha, Gao, Shen, Hong, Boning, Katabi (MIT) | 2025 | 24 citations

**Training Loop:** Interleaved RL training of both generator and generative process-level verifier. The verifier is an LLM trained via RL (not SFT) to produce chain-of-thought verification judgments. Both are updated in alternating phases within the same training run.

**Generator-Verifier Interaction:** The generator produces reasoning chains; the verifier evaluates each step generatively (producing verification rationales). The verifier's training signal comes solely from outcome-level correctness — no process-level annotations needed. The verifier's improving judgments create better reward signals for the generator, while the generator's improving outputs create harder verification challenges.

**Key Innovation:** Generative RL-trained verifier (vs. discriminative/SFT verifiers). This makes the verifier more robust to reward hacking and better at generalizing beyond training distributions.

**Results:** SOTA among 7B/8B models on 5 competition-level math benchmarks + 4 OOD reasoning tasks. Verifier leads on ProcessBench. Both components show especially large gains on hardest problems.

**Difference from standard RLHF:** Standard RLHF freezes the reward model. Tango co-evolves both, with the verifier itself trained via RL (not just SFT on preference data).

---

### 1.2 Generative Adversarial Reasoner / GAR (arXiv: 2512.16917) ⭐ PRIMARY
**"Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning"**
Liu et al. | 2025

**Training Loop:** On-policy joint training framework. GAN-inspired adversarial RL where reasoner and discriminator co-evolve. Reasoning chains are partitioned into "logically complete slices" for efficient evaluation.

**Generator-Verifier Interaction:** The discriminator evaluates each slice's soundness with structured justifications. Two reward signals for discriminator: (1) alignment reward — correctly detecting errors, (2) discriminative reward — distinguishing reasoner traces from reference rationales. The reasoner is rewarded for logically consistent steps yielding correct answers.

**Key Innovation:** Dense, well-calibrated, on-policy step-level rewards without explicit process annotations. The adversarial setup prevents reward hacking — the discriminator must stay ahead of the improving reasoner.

**Results:** AIME24: DS-R1-Distill-Qwen-7B 54.0→61.3 (+7.3), DS-R1-Distill-Llama-8B 43.7→53.7 (+10.0). +22.9% AIME24, +19.5% AIME25 (Llama), +35.3% LiveMathBench-Hard (Qwen).

**Difference from RL Tango:** GAR is adversarial (GAN-style min-max), while Tango is cooperative (both optimized toward shared reasoning goal). GAR uses slice-level evaluation; Tango uses process-level generative verification.

---

### 1.3 CURE (arXiv: 2506.03136)
**"Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning"**
Wang et al. | 2025 | NeurIPS 2025 Spotlight

**Training Loop:** RL framework where coder and unit tester co-evolve based on interaction outcomes. No ground-truth code needed as supervision.

**Generator-Verifier Interaction:** The coder generates code; the unit tester generates tests. Rewards are based on mutual interaction — does the code pass the tests? Do the tests catch bugs? The tester learns directly from the coder's mistakes.

**Results:** +5.3% code generation accuracy, +9.0% Best-of-N, +8.1% downstream agentic coding. Outperforms Qwen-Coder, DeepSeek-Coder, Seed-Coder.

**Key Insight:** Extends generator-verifier co-training to the code domain, where verification is naturally executable (unit tests).

---

### 1.4 V-STaR (arXiv: 2402.06457)
**"V-STaR: Training Verifiers for Self-Taught Reasoners"**
Hosseini et al. | 2024

**Training Loop:** Iterative self-improvement where both reasoner and verifier improve across rounds. Each iteration: (1) generate solutions, (2) train verifier on correct+incorrect solutions via DPO, (3) use verifier to filter/select for next round of reasoner training.

**Generator-Verifier Interaction:** Unlike prior work that only uses correct solutions for self-improvement, V-STaR exploits incorrect solutions to train the verifier. The verifier then guides the reasoner through best-of-N selection.

**Results:** 4-17% test accuracy improvement over existing self-improvement and verification approaches on code generation and math reasoning (LLaMA2).

**Difference from Tango/GAR:** V-STaR uses iterative SFT/DPO rounds (not joint RL). The co-evolution happens across discrete iterations rather than within a single training run.

---

## 2. Self-Play & Adversarial Approaches

### 2.1 SPIN (arXiv: 2401.01335)
**"Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"**
Chen, Deng, Yuan, Ji, Gu (UCLA) | 2024 | ICML 2024

**Mechanism:** The LLM plays against previous versions of itself. Current model acts as discriminator distinguishing self-generated responses from human-annotated data. Previous iteration acts as generator.

**Theoretical Guarantee:** Global optimum achieved only when LLM policy matches target data distribution — a fixed-point argument analogous to Nash equilibrium in two-player games.

**Relevance to co-training:** SPIN is a single-model self-play (same model plays both roles), unlike Tango/GAR which train separate generator and verifier models. It's a precursor to the co-training paradigm.

---

### 2.2 APO (arXiv: 2311.08045)
**"Adversarial Preference Optimization: Enhancing Your Alignment via RM-LLM Game"**
Cheng et al. | 2023 | ACL 2024 Findings

**Mechanism:** Min-max game between LLM and reward model. RM distinguishes LLM responses from golden annotations; LLM maximizes RM score. Alternating updates.

**Key Contribution:** Shows the RM can continuously gain accuracy improvement through adversarial co-training without additional preference annotation data. Addresses the distribution shift problem in standard RLHF.

**Relevance:** Direct precursor to Tango and GAR. Establishes the adversarial co-training principle for LLM alignment, which later papers extend to reasoning.

---

## 3. Verifier/Reward Model Foundations

### 3.1 GenRM (arXiv: 2408.15240)
**"Generative Verifiers: Reward Modeling as Next-Token Prediction"**
ICLR 2025

**Innovation:** Trains verifiers using next-token prediction (not discriminative classification). Enables chain-of-thought verification, majority voting for verification, and unification of generation + verification.

**Results:** Massive gains: 5%→45.3% (algorithmic), 73%→93.4% (GSM8K), 28%→44.6% (MATH hard-to-easy).

**Relevance to co-training:** GenRM provides the architectural foundation for Tango's generative verifier. Shows that generative verification is strictly superior to discriminative, motivating the use of generative verifiers in co-training loops.

---

### 3.2 Math-Shepherd (arXiv: 2312.08935)
**"Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"**
Wang et al. | 2023 | ACL 2024

**Innovation:** Automatically constructs process-level supervision data (no human annotation). Uses Monte Carlo rollouts to estimate step-level correctness.

**Results:** Mistral-7B: GSM8K 77.9%→84.1%, MATH 28.6%→33.0% (PPO). With verification: 89.1%, 43.5%.

**Relevance:** Establishes that PRM training data can be automatically generated, removing the bottleneck that makes co-training expensive.

---

### 3.3 Let's Verify Step by Step (arXiv: 2305.20050)
**Lightman et al. (OpenAI) | 2023 | ICLR 2024**

**Foundational Result:** Process supervision significantly outperforms outcome supervision. Released PRM800K (800K step-level labels).

**Relevance:** The paper that established process reward models as the dominant verification paradigm. All subsequent co-training work builds on the insight that step-level feedback > outcome-level feedback.

---

## 4. Self-Evolution with Policy-Verifier Iteration

### 4.1 rStar-Math (arXiv: 2501.04519)
**"rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking"**
Microsoft | 2025 | 266 citations

**Training Loop:** Self-evolution recipe across 4 rounds. Each round: (1) policy SLM generates solutions via MCTS, (2) process preference model (PPM) is re-trained on new data, (3) both models are updated and used for next round.

**Results:** Qwen2.5-Math-7B: 58.8%→90.0% (MATH), Phi3-mini-3.8B: 41.4%→86.4%. Solves 53.3% of AIME problems. Rivals o1-preview.

**Key Insight:** Demonstrates that iterative co-evolution of policy + verifier can achieve frontier-level reasoning from small models without distillation from larger models.

---

### 4.2 SCoRe (arXiv: 2409.12917)
**"Training Language Models to Self-Correct via Reinforcement Learning"**
Kumar et al. (Google DeepMind) | 2024

**Mechanism:** Multi-turn online RL where the model learns to self-correct using entirely self-generated data. The model acts as both generator and verifier across turns.

**Relevance:** Single-model approach where the generation and verification functions are internalized within the same model across turns, rather than being separate models.

---

## 5. Synthesis: Taxonomy of Co-Training Paradigms

### Dimension 1: Architecture
| Paradigm | Generator | Verifier | Coupling |
|----------|-----------|----------|----------|
| **RL Tango** | Separate LLM | Separate generative LLM | Interleaved RL |
| **GAR** | Separate LLM | Separate LLM discriminator | Adversarial RL |
| **CURE** | Coder LLM | Tester LLM | Co-evolutionary RL |
| **V-STaR** | Single LLM (iterated) | DPO-trained verifier | Iterative rounds |
| **SPIN** | Current LLM | Previous LLM iteration | Self-play |
| **APO** | LLM | Reward model | Min-max game |
| **rStar-Math** | Policy SLM | Process preference model | Iterative self-evolution |
| **SCoRe** | Single LLM (turn 1) | Same LLM (turn 2+) | Multi-turn RL |

### Dimension 2: Training Signal
- **Outcome-only:** RL Tango (verifier), SPIN, APO
- **Process-level (automatic):** GAR (slice-level), Math-Shepherd (MC rollouts), rStar-Math (PPM)
- **Process-level (human):** Let's Verify Step by Step (PRM800K)
- **Execution-based:** CURE (unit test pass/fail)

### Dimension 3: Co-Evolution Dynamics
- **Cooperative:** RL Tango (both optimize for reasoning quality), V-STaR, rStar-Math
- **Adversarial:** GAR (GAN-style), APO (min-max), SPIN (self-play discrimination)
- **Hybrid:** CURE (cooperative goal, but tester must catch coder's mistakes)

### Dimension 4: Reward Hacking Mitigation
| Paper | Strategy |
|-------|----------|
| RL Tango | Generative RL-trained verifier (harder to hack than discriminative) |
| GAR | Adversarial training keeps discriminator calibrated |
| APO | Alternating updates prevent RM exploitation |
| V-STaR | Discrete iteration prevents runaway co-adaptation |
| rStar-Math | MCTS exploration + iterative data refresh |

---

## 6. Key Findings & Open Questions

### Converging Insights
1. **Co-training > frozen verifiers:** RL Tango, GAR, and V-STaR all demonstrate that co-evolving verifiers outperform static/frozen reward models, primarily through better calibration and reduced reward hacking.

2. **Generative verification > discriminative:** GenRM and RL Tango show that having the verifier generate rationales (not just scores) significantly improves verification quality and enables process-level reasoning without explicit annotations.

3. **Process-level feedback > outcome-level:** The lineage from "Let's Verify Step by Step" → Math-Shepherd → GAR → RL Tango shows progressive automation of process supervision, from human annotation to Monte Carlo estimation to learned co-evolutionary signals.

4. **Self-play/adversarial dynamics prevent reward hacking:** The fundamental challenge in RLHF — reward model exploitation — is addressed by keeping the verifier "in the loop" and continuously updating it.

5. **Small models can achieve frontier performance:** rStar-Math (7B→90% MATH) and RL Tango (7B/8B SOTA) demonstrate that co-training unlocks capabilities previously requiring much larger models.

### Open Questions
1. **Scaling co-training:** Most papers demonstrate at 7B-8B scale. How does co-training scale to 70B+ models?
2. **Beyond math:** Most results are on math/code. Can co-training work for open-ended reasoning (e.g., legal, scientific)?
3. **Computational cost:** Co-training requires 2x the models. Is the cost justified vs. simply scaling the generator?
4. **Adversarial vs. cooperative:** GAR (adversarial) vs. Tango (cooperative) — which dynamics are more stable at scale?
5. **Convergence guarantees:** Only SPIN provides theoretical convergence guarantees. Can this be extended to multi-model co-training?

### Evolution Timeline
```
2023: Let's Verify Step by Step (process supervision concept)
      APO (adversarial RM-LLM game concept)
      Math-Shepherd (automatic process supervision)
2024: SPIN (self-play for LLM improvement)
      V-STaR (iterative reasoner+verifier improvement)
      GenRM (generative verification)
      SCoRe (self-correction via RL)
2025: DeepSeek-R1 (pure RL for reasoning, 5463 cites)
      rStar-Math (self-evolved policy+PRM, 266 cites)
      RL Tango (true co-training via RL)
      GAR (adversarial co-training)
      CURE (co-evolving coder+tester, NeurIPS Spotlight)
```

The field is rapidly converging on the insight that **generators and verifiers should be jointly optimized**, with 2025 representing a phase transition from "train verifier, then use it" to "train them together."

---

## Note on "ARS"
The task mentioned an "ARS" paper. No paper with the acronym "ARS" matching the description of "co-evolving dual system" was found in Semantic Scholar, arXiv, or web search. The closest candidates are:
- **AReaL** (2505.24298): Large-scale asynchronous RL system for language reasoning (infrastructure paper, not co-training)
- **MARS** (2510.04935): Multi-Agent RL for dual-system deep research
- The concept may refer to a paper not yet indexed or may use a different title than expected.
