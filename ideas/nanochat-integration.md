# Nanochat Integration Plan

## Overview

[nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy is a minimal, hackable LLM training pipeline covering tokenization → pretraining → SFT → RL → evaluation → inference. It runs on a single 8×H100 node (~$100).

We want to extend nanochat's RL stage to support **co-training** of two models.

## What Nanochat Already Has

- Full pretraining pipeline (GPT-2 scale to ~1B)
- SFT stage
- RL with GRPO
- Evaluation harness
- Single-node, single-codebase design
- wandb integration for training visualization

## What We Need to Add

### 1. Second Model Management
```python
# Two separate models, potentially different sizes
generator = NanochatModel(config_g)
verifier = NanochatModel(config_v)

# Separate optimizers
opt_g = torch.optim.AdamW(generator.parameters(), ...)
opt_v = torch.optim.AdamW(verifier.parameters(), ...)
```

### 2. Interleaved Training Loop
```python
for epoch in range(num_epochs):
    # Generator phase: Ng steps
    for step in range(Ng):
        # Generate solutions
        solutions = generator.generate(questions)
        # Get verifier rewards (frozen during generator training)
        with torch.no_grad():
            process_rewards = verifier.verify(questions, solutions)
        # Compute outcome rewards
        outcome_rewards = check_answers(solutions, gold_answers)
        # Combined reward
        rewards = outcome_rewards + alpha * process_rewards
        # GRPO update for generator
        grpo_step(generator, opt_g, solutions, rewards)
    
    # Verifier phase: Nv steps
    for step in range(Nv):
        # Generate solutions with current generator (frozen)
        with torch.no_grad():
            solutions = generator.generate(questions)
        # Verifier generates verification reasoning
        verifications = verifier.verify(questions, solutions)
        # Reward: did the verifier correctly identify correct/incorrect solutions?
        verification_rewards = check_verification_correctness(verifications, gold_answers)
        # GRPO update for verifier
        grpo_step(verifier, opt_v, verifications, verification_rewards)
```

### 3. Generative Verifier Format
The verifier should output structured reasoning:
```
Step 1: [analysis of first step] → Correct/Incorrect
Step 2: [analysis of second step] → Correct/Incorrect
...
Overall: Correct/Incorrect
```

### 4. Reward Computation
- **Generator reward:** `r = r_outcome + α * r_verifier_process`
  - r_outcome: 1 if final answer correct, 0 otherwise
  - r_verifier_process: Step-level rewards from verifier's judgments
- **Verifier reward:** `r = r_verification_correctness`
  - 1 if verifier's overall judgment matches ground truth
  - Could add format reward for proper step-level output

### 5. Memory Management
Running two models on one node is tight. Options:
- Use smaller models (both ~350M or generator 1B + verifier 350M)
- Gradient checkpointing
- Freeze one model while training the other (already done in interleaved setup)
- Mixed precision throughout

## Implementation Phases

### Phase 1: Proof of Concept (Week 1-2)
- Fork nanochat
- Add second model loading
- Implement basic interleaved loop
- Test on GSM8K with small models

### Phase 2: Generative Verifier (Week 2-3)
- Implement verification prompt format
- Train verifier to generate step-level judgments
- Parse verifier output into reward signals

### Phase 3: Full Co-Training (Week 3-4)
- Full Tango-style training loop
- Proper advantage estimation for both models
- Evaluation on math benchmarks
- Compare vs. single-model GRPO baseline

### Phase 4: Extensions (Week 4+)
- Multi-agent GRPO (MARS-style)
- Self-play without ground truth
- Different generator/verifier sizes
