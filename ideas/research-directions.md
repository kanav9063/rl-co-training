# Research Directions

**Priority: HIGH**


- Add a second model (verifier) — same architecture, separate weights
- Implement interleaved training: N_g generator steps, N_v verifier steps
- Verifier generates text-based correctness judgments
- Use verifier output as additional reward signal for generator


## Direction 2: Self-Play Without Ground Truth
**Priority: HIGH**

Combine MAE's self-supervised approach with Tango's generative verifier:
- Train on problems without known answers
- Verifier learns from consistency across multiple generator samples
- Generator learns from verifier's evolving judgment

This is the holy grail: co-training that works without labeled data.

## Direction 3: Asymmetric Co-Training
**Priority: MEDIUM**

Explore heterogeneous architectures:
- Small verifier (1B) + larger generator (7B)
- Or small generator + large verifier
- Question: Which asymmetry is more beneficial?
- Cheaper to train — potentially more practical

## Direction 4: Multi-Agent GRPO
**Priority: MEDIUM**

Implement MARS-style multi-agent GRPO:
- Shared trajectory rewards between agents
- Advantage-weighted balanced sampling

## Direction 5: Co-Training for Code
**Priority: MEDIUM**

Apply co-training to code generation:
- Generator writes code, verifier reasons about correctness
- Code has natural outcome verification (unit tests)
- But process-level verification (is this approach sound?) benefits from co-trained verifier

## Direction 6: Curriculum Co-Evolution
**Priority: LOW**

Study how training data difficulty should change as both models improve:
- Start with easy problems, increase difficulty
- Let the proposer agent (à la MAE) control curriculum
- Three-way co-training: proposer + solver + judge

## Direction 7: Co-Training Scaling Laws
**Priority: LOW (needs more compute)**

Systematic study of:
- Co-training benefit vs. model size
- Optimal generator:verifier size ratio
- Compute-optimal interleaving schedule
- When does co-training become worth the 2x compute?
