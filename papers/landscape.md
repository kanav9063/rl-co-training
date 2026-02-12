# The RL Co-Training Landscape (2024-2026)

## The Big Picture

The LLM reasoning field has evolved through several phases:

1. **2023-2024: Single-model RL** — DeepSeek-R1, o1, QwQ showed RL post-training dramatically improves reasoning
2. **2024-2025: Better rewards** — Generative reward models, process reward models, self-rewarding
3. **2025-2026: Co-training** — Training multiple models together with RL (where we are now)

## Why Co-Training Matters Now

The single-model RL paradigm has a fundamental ceiling: **the verifier/reward model is fixed**. This creates:
- **Reward hacking:** The generator learns to exploit the reward model's blind spots
- **Distribution shift:** As the generator improves, the verifier becomes stale
- **Capability bottleneck:** The generator can only be as good as the reward signal

Co-training solves all three by making the verifier a co-evolving partner.

## Taxonomy of Co-Training Approaches

### 1. Generator-Verifier Co-Training
**Exemplar:** RL Tango, PRIME, Cooper

The most direct approach: train a generator and a verifier/reward model together.

**Spectrum of verifier training:**
- Fixed/frozen (baseline) → Online SFT (PRIME) → Online RL (Tango)
- Discriminative scores → Generative reasoning about correctness

### 2. Dual-System Co-Evolution
**Exemplar:** MARS

Inspired by Kahneman's System 1/System 2. Train two complementary cognitive systems:
- Fast processor (System 1) + Deep reasoner (System 2)
- Shared rewards but decoupled gradients

### 3. Multi-Agent Self-Play
**Exemplar:** MAE, SPIN, Debate

Multiple agents (potentially from the same model) that train against each other:
- Proposer/Solver/Judge triplets
- Adversarial debate
- Playing against previous iterations

### 4. Self-Rewarding / Self-Critique
**Exemplar:** Constitutional AI, Self-Rewarding LMs, Critique-GRPO

Single model learns to both generate AND evaluate:
- Implicit co-training within one model
- Constitutional principles as meta-verifier

## Key Technical Challenges

### Credit Assignment
When multiple agents share rewards, how do you assign credit properly?
- MARS: Decoupled gradient computation
- Tango: Separate reward signals (outcome for verifier, outcome + process for generator)

### Training Stability
Co-training is notoriously unstable (cf. GAN training):
- Interleaved training (Tango) vs. simultaneous (MARS)
- KL regularization to prevent drift
- Balanced sampling to prevent one agent dominating

### Scalability
- Does co-training benefit scale with model size?
- Can you co-train heterogeneous model sizes (small verifier + large generator)?
- Compute overhead of maintaining two models

### Reward Hacking in Co-Training
Does co-training solve reward hacking or just push it to a higher level?
- The verifier could learn to give high rewards to patterns rather than correctness
- Need to ground at least one signal in objective reality (e.g., answer correctness)

## Open Questions (Research Opportunities)

1. **Optimal interleaving schedule:** How many generator steps per verifier step? Is there a theory?
2. **Initialization:** Should both models start from the same base? Different bases?
3. **Asymmetric architectures:** Small verifier + large generator? Or equal?
4. **Beyond pairs:** Can we co-train 3+ models (generator + verifier + critic + ...)?
5. **Transfer:** Does a co-trained verifier transfer to new generators?
6. **Curriculum:** Should the training data difficulty increase as both models improve?
7. **Domain generalization:** Co-training shown mainly for math — does it work for code, general reasoning, agents?

## The Nanochat Opportunity

Nanochat is uniquely positioned for this research because:
- Full stack from pretraining to RL in one codebase
- Single-node design makes iteration fast
- Already has GRPO implementation
- Educational/hackable design philosophy
- Can test co-training ideas at small scale before investing in large compute
