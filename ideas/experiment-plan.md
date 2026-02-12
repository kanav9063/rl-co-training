# Experiment Plan

## Phase 1: Reproduce Baselines (1-2 weeks)

### Experiment 1.1: Nanochat GRPO Baseline
- Train standard nanochat model with GRPO on math data
- Dataset: GSM8K or MATH subset
- Model: nanochat default (~350M-1B)
- Metric: pass@1 accuracy
- Purpose: Establish single-model RL baseline

### Experiment 1.2: Fixed Verifier Baseline  
- Train generator with GRPO
- Add a frozen pretrained model as verifier (inference only)
- Compare vs. 1.1 to see if even a fixed verifier helps

## Phase 2: Basic Co-Training (2-3 weeks)

### Experiment 2.1: Interleaved RL (Tango-lite)
- Two models, same size (~350M each)
- Interleaved training: Ng=5, Nv=2
- Generator gets outcome + verifier rewards
- Verifier gets verification correctness rewards
- Compare vs. Phase 1 baselines

### Experiment 2.2: Interleaving Schedule Ablation
- Vary Ng/Nv ratios: 1:1, 3:1, 5:1, 10:1
- Question: How sensitive is co-training to the schedule?

### Experiment 2.3: Reward Mixing Ablation
- Vary α (weight of verifier reward for generator): 0.1, 0.3, 0.5, 1.0
- Question: How much should the generator trust the verifier?

## Phase 3: Advanced Co-Training (3-4 weeks)

### Experiment 3.1: Asymmetric Sizes
- Generator 1B + Verifier 350M
- Generator 350M + Verifier 1B
- Question: Which asymmetry works better?

### Experiment 3.2: Zero RL (MARS-style)
- Skip SFT entirely, train both models from base with RL
- Compare vs. SFT-then-RL pipeline
- Question: Is SFT warmup necessary for co-training?

### Experiment 3.3: Multi-Agent Evolve
- Implement MAE triplet: Proposer + Solver + Judge
- All from same base model
- No ground truth labels
- Question: Can we match supervised co-training without labels?

## Phase 4: Analysis (ongoing)

### Analysis 4.1: Training Dynamics
- Plot generator accuracy & verifier accuracy over time
- Look for: mutual improvement, collapse, oscillation
- Visualize reward hacking indicators

### Analysis 4.2: Generalization
- Train on GSM8K, test on MATH (harder, OOD)
- Does co-training improve OOD generalization vs. single-model RL?

### Analysis 4.3: Verifier Quality
- Evaluate verifier independently on ProcessBench-style tasks
- Does co-training produce a better verifier than fixed training?

## Compute Budget

Assuming single 8×H100 node (nanochat default):
- Phase 1: ~$50 (2 training runs)
- Phase 2: ~$200 (multiple ablation runs)
- Phase 3: ~$300 (larger models, more experiments)
- Phase 4: Included in above (evaluation is cheap)
- **Total: ~$550**

## Success Criteria

1. **Minimum:** Co-training matches single-model GRPO (doesn't hurt)
2. **Good:** Co-training improves over single-model GRPO by 5%+
3. **Great:** Co-training shows improved OOD generalization
4. **Excellent:** Self-play co-training works without ground truth labels
