# Deep Research: RL Co-Training — Generator-Verifier Co-Evolution
Generated: 2026-02-11
Sources: 17 web | 18 tweet clusters | 11 papers | 6 repos

## TL;DR
The field is converging on a powerful paradigm: **jointly training generators and verifiers via RL produces dramatically better results than training either in isolation.** The key papers are RL Tango (cooperative interleaved RL, NeurIPS 2025), GAR (adversarial RL), CURE (code+tester co-evolution, NeurIPS Spotlight), and MARS (multi-agent deep research). Three critical insights emerged: (1) **generative RL-trained verifiers >> discriminative SFT-trained verifiers**, (2) **frozen verifiers always lead to reward hacking** (ASL proves this), and (3) **process-level rewards are 2.5× more sample-efficient than outcome-only** (PRIME). DeepSeek-Math-V2 is the first production deployment (685B, 5/6 IMO 2025). Implementation stack: veRL (19K★) is the gravitational center, with PRIME being the purest open-source co-training implementation.

---

## Complete Paper Landscape (15+ papers, ranked by relevance)

### 🔴 Tier 1: Core Co-Training (Must-Read)

| Paper | Year | Venue | Citations | Paradigm | Key Result |
|-------|------|-------|-----------|----------|------------|
| **RL Tango** | 2025 | NeurIPS | 24 | Cooperative interleaved RL | SOTA 7B/8B, 25.5% relative improvement on math |
| **GAR** | 2025 | arXiv | — | Adversarial GAN-style RL | +10pts AIME24 on 8B, +35% LiveMathBench-Hard |
| **CURE** | 2025 | NeurIPS Spotlight | — | Coder-Tester co-evolution | +5.3% code gen, +9% Best-of-N |
| **PRIME** | 2025 | arXiv | — | Implicit PRM online | 2.5× sample efficiency, +16.7% avg |
| **MARS** | 2025 | arXiv | — | System1+System2 multi-agent GRPO | 8B matches Claude 3.7 Sonnet on HLE |

### 🟡 Tier 2: Self-Play & Foundations

| Paper | Year | Venue | Citations | Paradigm | Key Result |
|-------|------|-------|-----------|----------|------------|
| **V-STaR** | 2024 | arXiv | — | Iterative DPO reasoner+verifier | 4-17% accuracy gains |
| **SPIN** | 2024 | ICML | — | Self-play fine-tuning | Converges without external rewards |
| **APO** | 2023 | ACL | — | Adversarial RM-LLM game | RM improves without new data |
| **Absolute Zero** | 2025 | NeurIPS | — | Zero-data proposer-solver | SOTA zero-setting reasoning |
| **rStar-Math** | 2025 | arXiv | 266 | Policy+PPM iterative evolution | 7B→90% MATH, rivals o1-preview |

### 🟢 Tier 3: Verifier Foundations

| Paper | Year | Venue | Citations | Key Contribution |
|-------|------|-------|-----------|------------------|
| **GenRM** | 2024 | ICLR 2025 | — | Generative verification >> discriminative |
| **Math-Shepherd** | 2023 | ACL 2024 | — | Automatic process-level supervision |
| **Let's Verify Step by Step** | 2023 | ICLR 2024 | — | Process supervision >> outcome (foundational) |
| **Self-Rewarding LMs** | 2024 | Meta | — | Same model as generator + judge |
| **SCoRe** | 2024 | DeepMind | — | Self-correction via multi-turn RL |

### ⭐ Production Deployment
| System | Scale | Result |
|--------|-------|--------|
| **DeepSeek-Math-V2** | 685B | 5/6 IMO 2025, 118/120 Putnam 2024 |

---

## Taxonomy of Co-Training Paradigms

### By Architecture
| Paradigm | Generator | Verifier | Coupling | Example |
|----------|-----------|----------|----------|---------|
| Explicit dual-model | Separate LLM | Separate generative LLM | Interleaved RL | **RL Tango** |
| Adversarial dual-model | Separate LLM | Separate discriminator | GAN-style RL | **GAR** |
| Domain-specific co-evolution | Coder LLM | Tester LLM | Mutual interaction RL | **CURE** |
| Multi-agent co-evolution | System 1 LLM | System 2 LLM | Multi-agent GRPO | **MARS** |
| Implicit co-training | Policy LLM | Implicit PRM (Q-function) | Online interleaved | **PRIME** |
| Single-model multi-role | Same LLM (Solver) | Same LLM (Judge) | Joint RL | **Multi-Agent Evolve** |
| Self-play | Current LLM | Previous iteration | Iterative DPO/SFT | **SPIN, V-STaR** |
| Adversarial self-play | Bug-injector | Bug-solver | Binary RL | **Self-Play SWE-RL** |
| Task self-play | Proposer | Solver + RAG | RL | **SSP** |

### By Training Dynamics
| Dynamics | Papers | Stability | Reward Hacking Risk |
|----------|--------|-----------|---------------------|
| **Cooperative** | RL Tango, V-STaR, rStar-Math | High | Medium (mitigated by generative verifier) |
| **Adversarial** | GAR, APO, SSP | Medium | Low (adversarial keeps verifier calibrated) |
| **Self-play** | SPIN, AZR, MAE | High | Low (no external verifier to hack) |

### By Reward Granularity
| Level | Papers | Efficiency |
|-------|--------|------------|
| **Process-level (dense)** | RL Tango, GAR, PRIME, Math-Shepherd | 2.5× better (PRIME) |
| **Outcome-level (sparse)** | SPIN, V-STaR, basic GRPO | Baseline |
| **Execution-based** | CURE (unit tests), AZR (code exec) | Domain-specific, very reliable |

---

## The Critical Design Choices

### 1. Generative Verifier > Discriminative Verifier
| | Generative (RL Tango) | Discriminative (PRIME) |
|---|---|---|
| Output | Natural language rationale + judgment | Scalar logit score |
| Training | RL | Online SFT/implicit |
| Reward hacking | Resistant (stochastic, harder to exploit) | Vulnerable (deterministic, exploitable) |
| Generalization | Strong OOD | Weaker OOD |
| Cost | Higher (full generation) | Lower (single forward pass) |

**Verdict:** Generative wins on quality; discriminative wins on cost. RL Tango explicitly addresses PRIME's limitation.

### 2. Online Co-Evolution > Frozen Verifier
**ASL (Sun et al., 2025) proves empirically:** If you freeze the verifier/reward model while training the generator, the generator learns to exploit the verifier's blind spots → reward hacking → performance plateaus or degrades.

**Every successful co-training system keeps the verifier updating:**
- RL Tango: interleaved RL updates
- GAR: adversarial alternating updates  
- PRIME: online PRM updates each iteration
- rStar-Math: re-train PPM each round
- ASL: continual GRM training on evolving distribution

### 3. Process Rewards > Outcome Rewards
PRIME demonstrates 2.5× sample efficiency with process-level (step-by-step) rewards vs outcome-only. The lineage:
```
Let's Verify (human annotations) → Math-Shepherd (automatic MC) → PRIME (implicit learned) → RL Tango (generative RL)
```
Each step removes a bottleneck while maintaining or improving signal quality.

---

## The Co-Training Training Loop (Consensus Design)

```
┌─────────────────────────────────────────────┐
│                TRAINING LOOP                 │
│                                              │
│  1. Generator produces rollouts              │
│     (reasoning chains / code / solutions)    │
│                    ↓                         │
│  2. Verifier evaluates                       │
│     (step-level or outcome-level)            │
│                    ↓                         │
│  3. Generator updated via RL                 │
│     (using verifier signals as rewards)      │
│                    ↓                         │
│  4. Verifier updated                         │
│     (on generator's new outputs)             │
│                    ↓                         │
│  5. Repeat (interleaved or alternating)      │
│     Both improve each iteration              │
└─────────────────────────────────────────────┘
```

**Variants:**
- **Cooperative (Tango):** Both optimize toward shared reasoning goal
- **Adversarial (GAR):** Verifier tries to catch errors; generator tries to fool verifier while being correct
- **Self-play (SPIN/AZR):** Model plays against previous version of itself

---

## Implementation Stack

### Frameworks (ranked by suitability for co-training)

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| Quick prototyping | **TRL** (17K★) | `GRPOTrainer` + custom reward in 10 lines |
| Large-scale production | **veRL** (19K★) | Best throughput, Megatron, 671B demonstrated |
| Async co-evolution | **OpenRLHF** (9K★) | Native async training, Ray separation |
| Process reward co-training | **PRIME** (on veRL) | Only open impl of online implicit PRM |
| Self-play without rewards | **SPIN** (1.2K★) | Standalone, no reward model needed |
| Adversarial dual-agent | **SSP** (on veRL) | Only open proposer-solver co-evolution |

### Paper Implementations
| Paper | Code | Framework |
|-------|------|-----------|
| RL Tango | github.com/kaiwenzha/rl-tango | Custom |
| PRIME | github.com/PRIME-RL/PRIME | veRL |
| SPIN | github.com/uclaml/SPIN | Custom |
| CURE | — (not found) | — |
| AZR | github.com/LeapLabTHU/Absolute-Zero-Reasoner | Custom |
| MAE | github.com/ulab-uiuc/Multi-agent-Evolve | Custom |
| ASL | github.com/forangel2014/Towards-Agentic-Self-Learning | Custom |
| SSP | github.com/Alibaba-Quark/SSP | veRL |

---

## Failure Modes & Mitigations

| Failure Mode | Cause | Mitigation | Evidence |
|---|---|---|---|
| **Reward hacking** | Frozen verifier | Online co-evolution | ASL paper |
| **Deterministic exploitation** | Discriminative verifier (scalar scores) | Generative verifier (CoT reasoning) | RL Tango |
| **Training imbalance** | One component improves faster | Advantage-weighted balanced sampling | MARS |
| **Mode collapse** | Generator+verifier converge to narrow patterns | Adversarial dynamics, diverse rollouts | GAR, APO |
| **Collusion** | Co-trained models agree on wrong answers | External ground-truth signal (math correctness, code execution) | All papers use verifiable domains |

---

## Open Questions

1. **Scaling laws for co-training:** Most papers demonstrate at 7B-8B. How does benefit scale to 70B+? DeepSeek-Math-V2 (685B) suggests it works but no ablations published.
2. **Beyond verifiable domains:** Math/code have ground truth. Can co-training work for law, science, open-ended reasoning where verification is subjective?
3. **Compute efficiency:** Is 2× compute for co-training better than 2× compute for generator-only RL? No systematic comparison exists.
4. **Adversarial vs cooperative:** GAR (adversarial) vs Tango (cooperative) — which dynamics win at scale?
5. **Convergence guarantees:** Only APRM provides theoretical guarantees (Nash equilibrium via general-sum game). Can this extend to multi-model systems?
6. **Collusion risk:** Could generator and verifier converge to mutually reinforcing but wrong patterns in non-verifiable domains?

---

## Evolution Timeline

```
2023 ─── Let's Verify Step by Step (OpenAI) ── process supervision concept
   │     APO ── adversarial RM-LLM game concept
   │     Math-Shepherd ── automatic process supervision
   │
2024 ─── SPIN (ICML) ── self-play without external rewards
   │     V-STaR ── iterative reasoner+verifier
   │     GenRM (ICLR 2025) ── generative verification
   │     Self-Rewarding LMs (Meta) ── same model as judge
   │     SCoRe (DeepMind) ── self-correction via RL
   │
2025 ─── DeepSeek-R1 ── pure RL for reasoning (5463 cites)
   │     PRIME ── implicit PRM co-training (online)
   │     rStar-Math ── policy+PPM self-evolution (7B→90% MATH)
   │     RL Tango (NeurIPS) ── true cooperative co-training via RL ★
   │     GAR ── adversarial co-training
   │     CURE (NeurIPS Spotlight) ── coder+tester co-evolution
   │     MARS ── multi-agent System1+System2 for research
   │     ASL ── proves frozen verifiers fail
   │     APRM ── convergence guarantees for co-training
   │     AZR (NeurIPS) ── zero-data self-play
   │     MAE ── 3-role single-model co-evolution
   │     DeepSeek-Math-V2 ── first production deployment (685B, IMO)
   │
2026 ─── ??? (field is moving fast)
```

---

## Recommended Reading Order

1. **Start:** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — understand why process supervision matters
2. **Foundation:** [GenRM](https://arxiv.org/abs/2408.15240) — understand generative vs discriminative verification
3. **Self-play:** [SPIN](https://arxiv.org/abs/2401.01335) — simplest co-training paradigm
4. **Implicit PRM:** [PRIME](https://arxiv.org/abs/2502.01456) — online reward model co-training
5. **Core:** [RL Tango](https://arxiv.org/abs/2505.15034) — the canonical generator-verifier co-training paper ★
6. **Adversarial:** [GAR](https://arxiv.org/abs/2512.16917) — adversarial alternative to Tango
7. **Code domain:** [CURE](https://arxiv.org/abs/2506.03136) — extends to coder+tester
8. **Failure modes:** [ASL](https://arxiv.org/abs/2510.14253) — why frozen verifiers fail
9. **Theory:** [APRM](https://arxiv.org/abs/2511.22888) — convergence guarantees
10. **Production:** DeepSeek-Math-V2 — scaling to 685B

---

## Note on "ARS"
The paper you mentioned as "ARS" is likely **MARS** (arXiv:2510.04935) — "Multi-Agent RL for co-evolving dual-system deep research." It trains System 1 (summarizer) + System 2 (reasoner) via multi-agent GRPO with decoupled gradient computation. 8B model matches Claude 3.7 Sonnet on HLE.
