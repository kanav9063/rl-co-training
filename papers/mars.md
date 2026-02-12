# MARS: Co-evolving Dual-System Deep Research via Multi-Agent Reinforcement Learning

**Paper:** [arXiv:2510.04935](https://arxiv.org/abs/2510.04935) (Oct 2025, ongoing work)
**Authors:** Guoxin Chen, Zile Qiao, Wenqing Wang, et al.

## Problem

Large Reasoning Models (LRMs) have two fundamental limitations:
1. **Excessive token consumption** when overanalyzing simple information processing tasks
2. **Inability to access up-to-date knowledge** beyond training data

## Core Innovation

**MARS** (Multi-Agent System for Deep ReSearch) — a co-evolution framework jointly optimizing **dual cognitive systems** through multi-agent RL:
- **System 1:** Fast, intuitive processing (summarizer)
- **System 2:** Deliberate reasoning

Unlike prior work with fixed or independently-trained summarizers, MARS enables System 1 and System 2 to co-adapt through **shared trajectory rewards**.

### Key Technical Contributions

1. **Decoupled gradient computation:** Proper credit assignment despite shared rewards between agents
2. **Bin-packing optimization:** Efficient parallel information processing
3. **Advantage-weighted balanced sampling:** Prevents training imbalance between the two systems

### Training Setup
- Extended GRPO for multi-agent settings
- **Zero RL setting:** No supervised fine-tuning at all — pure RL from base model
- System 1 learns to distill information specifically useful for System 2's reasoning

## Results

- **MARS (8B)** achieves 8.17% on HLE (Humanity's Last Exam)
- Outperforms WebThinker (32B with SFT, 6.87%)
- Narrows gap with Claude 3.7 Sonnet (7.89%)
- Average gain of 8.9% across 7 knowledge-intensive tasks

## Key Insights

1. **Dual-system co-evolution > independent training:** The systems develop complementary strategies when trained together
2. **Zero RL is viable:** You don't need SFT warmup — pure RL co-training works
3. **Credit assignment matters:** Naive shared rewards fail; need decoupled gradients
4. **Multi-agent GRPO extension is general:** Could apply to other multi-agent LLM settings

## Differences from RL Tango

| Aspect | RL Tango | MARS |
|--------|----------|------|
| Agents | Generator + Verifier | System 1 + System 2 |
| Task | Math reasoning | Deep research (knowledge-intensive) |
| Reward sharing | Separate rewards | Shared trajectory rewards |
| SFT warmup | Yes | No (Zero RL) |
| Focus | Verification quality | Information distillation |

## Relevance to Our Research

- Shows co-training works beyond verification → dual cognitive architectures
- The multi-agent GRPO extension is directly applicable
- Zero RL setting is interesting — could we skip SFT in nanochat and go straight to co-training?
- Credit assignment techniques are essential for any multi-agent RL setup
