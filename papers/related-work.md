# Related Work: RL Co-Training for LLMs

## Direct Co-Training Papers

### PRIME — Process Reinforcement through Implicit Rewards (Jan 2025)
- [GitHub](https://github.com/PRIME-RL/PRIME)
- First approach to jointly train generator + PRM online
- PRM trained via SFT on outcome labels collected online
- Limitation: Discriminative, deterministic rewards → still vulnerable to reward hacking
- RL Tango directly improves upon this

### Cooper — Co-Optimizing Policy and Reward Models (Aug 2025)
- [arXiv:2508.05613](https://arxiv.org/abs/2508.05613) | [GitHub](https://github.com/zju-real/cooper)
- Joint optimization of policy model and reward model
- From Zhejiang University
- Directly relevant — different approach to the same co-training problem

### Multi-Agent Evolve (MAE) (Oct 2025)
- [arXiv:2510.23595](https://arxiv.org/abs/2510.23595)
- **Proposer → Solver → Judge** triplet, all from single LLM, co-evolved via RL
- No human-curated data needed
- Works on math, reasoning, and general Q&A
- 4.54% average improvement on Qwen2.5-3B-Instruct

### MAGRPO — Multi-Agent Group Relative Policy Optimization (Aug 2025)
- [arXiv:2508.04652](https://arxiv.org/abs/2508.04652)
- Multi-agent, multi-turn RL algorithm for LLM collaboration
- Builds on GRPO + MARL techniques
- General framework for multi-agent LLM training

### MARTI — Multi-Agent Reinforced Training and Inference (May 2025)
- [GitHub](https://github.com/TsinghuaC3I/MARTI)
- Framework from Tsinghua for multi-agent RL training
- Used in ReviewRL for scientific review

## Reward Model Co-Evolution

### RM-R1 — Reward Modeling as Reasoning (May 2025)
- [arXiv:2505.02387](https://arxiv.org/abs/2505.02387) | [GitHub](https://github.com/RM-R1-UIUC/RM-R1)
- Train reward models to reason about quality, not just score

### Think-RM — Long-Horizon Reasoning in Generative Reward Models (May 2025)
- [arXiv:2505.16265](https://arxiv.org/abs/2505.16265) | [GitHub](https://github.com/IlgeeHong/Think-RM)
- Enabling deep reasoning in reward models themselves

### GenPRM — Generative Process Reward Models (Apr 2025)
- [arXiv:2504.00891](https://arxiv.org/abs/2504.00891) | [GitHub](https://github.com/RyanLiu112/GenPRM)
- Scaling test-time compute with generative PRMs

### DeepSeek-GRM — Inference-Time Scaling for Reward Modeling (Apr 2025)
- [arXiv:2504.02495](https://arxiv.org/abs/2504.02495)
- Generalist reward modeling with inference-time scaling

### GRAM — Generative Foundation Reward Model (Jun 2025)
- [arXiv:2506.14175](https://arxiv.org/abs/2506.14175)
- Foundational approach to generative reward modeling

## Self-Play Approaches

### SPIN — Self-Play Fine-Tuning (Jan 2024)
- [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) | [GitHub](https://github.com/uclaml/SPIN)
- LLM plays against previous iterations (GAN-like)
- Converges when model matches target distribution
- Foundational self-play paper for LLMs

### Self-Rewarding Language Models (Meta, Jan 2024)
- LLM generates its own reward signal for iterative DPO
- Early form of model-as-judge co-training

### Debate for Scalable Oversight (Anthropic/OpenAI, 2024)
- [OpenReview](https://openreview.net/forum?id=gAEEjGv5Oa)
- Train models to debate via self-play
- Judge accuracy improves when evaluating optimized debaters
- A form of adversarial co-training

### AlphaProof (DeepMind, 2024)
- Self-play RL for formal mathematical reasoning
- Proved IMO problems at silver-medal level
- Used formal language (Lean) as verifier
- Key insight: formal verification provides perfect reward signal for self-play

### "Towards Understanding Self-Play for LLM Reasoning" (Oct 2025)
- [arXiv:2510.27072](https://arxiv.org/abs/2510.27072)
- Theoretical analysis of when/why self-play works for LLM reasoning

## Constitutional / Self-Critique

### Constitutional AI (Anthropic, 2022)
- AI self-critique guided by constitutional principles
- Implicit co-training: the model learns to both generate and evaluate
- Foundation for RLAIF (RL from AI Feedback)

### Critique-GRPO (Jun 2025)
- [arXiv:2506.03106](https://arxiv.org/abs/2506.03106)
- Natural language critique as reward signal in GRPO
- Bridge between generative verification and RL training

## Multi-Agent RL for LLMs

### TTRL — Test-Time Reinforcement Learning (Apr 2025)
- [arXiv:2504.16084](https://arxiv.org/abs/2504.16084) | [GitHub](https://github.com/PRIME-RL/TTRL)
- Online RL without ground-truth labels
- Related: how do you train when you don't have a fixed reward?

### SSRL — Self-Supervised RL for Agentic Search (Aug 2025)
- [arXiv:2508.10874](https://arxiv.org/abs/2508.10874)
- Investigation of agentic search RL without external search engine

## Surveys

### "A Survey of Reinforcement Learning for Large Reasoning Models" (Sep 2025)
- [arXiv:2509.08827](https://arxiv.org/abs/2509.08827) | [GitHub](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs)
- Comprehensive survey covering reward design, policy optimization, sampling, applications
- Essential reference for the field
