# RL Co-Training: Generator-Verifier Co-Evolution for LLM Reasoning

Research project exploring reinforcement learning co-training paradigms where multiple models (generator + verifier, or dual cognitive systems) are trained together via RL to achieve mutual improvement.

## Motivation

Current RL post-training for LLMs relies on **fixed verifiers** (rule-based or frozen reward models), which are susceptible to reward hacking and generalize poorly. The key insight driving this research: **if we train the generator with RL, why not train the verifier with RL too?**

Co-training creates a virtuous cycle where both models push each other to improve — similar to GANs, self-play in games, or biological co-evolution. This is potentially the next frontier in LLM reasoning after single-model RL (DeepSeek-R1, QwQ, etc.).

## Seed Papers

### 1. RL Tango (NeurIPS 2025)
**"RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning"**
- Authors: Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi (MIT)
- Paper: [arXiv:2505.15034](https://arxiv.org/abs/2505.15034)
- Code: [github.com/kaiwenzha/rl-tango](https://github.com/kaiwenzha/rl-tango)
- Key idea: Interleaved RL training of generator + generative process-level verifier. Both trained with RL (not SFT). SOTA on 7B/8B math reasoning.

### 2. MARS (Oct 2025)
**"MARS: Co-evolving Dual-System Deep Research via Multi-Agent Reinforcement Learning"**
- Authors: Guoxin Chen et al.
- Paper: [arXiv:2510.04935](https://arxiv.org/abs/2510.04935)
- Key idea: Co-evolve System 1 (fast summarizer) and System 2 (deliberate reasoner) via MARL with shared trajectory rewards. Extended GRPO for multi-agent settings.

## Related Work

| Paper | Year | Key Idea |
|-------|------|----------|
| **PRIME** | 2025 | Process reward via implicit rewards, online PRM co-training with generator |
| **Cooper** | 2025 | Co-optimizing policy and reward models jointly ([arXiv:2508.05613](https://arxiv.org/abs/2508.05613)) |
| **Multi-Agent Evolve (MAE)** | 2025 | Proposer-Solver-Judge triplet from single LLM, co-evolved via RL ([arXiv:2510.23595](https://arxiv.org/abs/2510.23595)) |
| **MAGRPO** | 2025 | Multi-agent group relative policy optimization for LLM collaboration ([arXiv:2508.04652](https://arxiv.org/abs/2508.04652)) |
| **SPIN** | 2024 | Self-play fine-tuning — LLM plays against previous iterations ([arXiv:2401.01335](https://arxiv.org/abs/2401.01335)) |
| **GenPRM** | 2025 | Generative process reward models with chain-of-thought verification |
| **Think-RM** | 2025 | Long-horizon reasoning in generative reward models |
| **RM-R1** | 2025 | Reward modeling as reasoning |
| **Self-Rewarding LMs** | 2024 | LLM generates its own reward signal for iterative DPO (Meta) |
| **Debate (Anthropic)** | 2024 | Self-play debate improves judge accuracy for scalable oversight |
| **AlphaProof** | 2024 | Self-play RL for formal mathematical reasoning (DeepMind) |
| **MARTI** | 2025 | Multi-agent reinforced training and inference framework (Tsinghua) |
| **TTRL** | 2025 | Test-time RL without ground-truth labels |
| **Constitutional AI** | 2022 | AI self-critique as implicit co-training (Anthropic) |

## Research Directions

2. **Self-play without ground truth**: Combine MAE's judge-free approach with Tango's generative verifier
3. **Scaling laws for co-training**: How does co-training benefit scale with model size?
4. **Beyond math**: Apply co-training to code, general reasoning, agentic tasks


- A second model (verifier) trained in parallel
- Interleaved training schedules (Ng generator steps, Nv verifier steps)
- Generative verification with process-level rewards
- Shared replay buffers between generator and verifier

## Project Structure

```
├── README.md
├── papers/              # Paper summaries and landscape analysis
│   ├── rl-tango.md
│   ├── mars.md
│   ├── related-work.md
│   └── landscape.md
├── ideas/               # Research directions and experiment plans
│   ├── research-directions.md
│   └── experiment-plan.md
└── src/                 # Implementation (placeholder)
    └── __init__.py
```

## Setup

```bash
git clone https://github.com/kanav9063/rl-co-training.git
cd rl-co-training
# Implementation coming soon — see ideas/ for planned experiments
```

## License

MIT
