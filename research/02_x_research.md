# X/Twitter Research: RL Co-Training Paradigms — Generator-Verifier Co-Evolution
## Distilled Findings | 2026-02-11

---

## Executive Summary

The X/ML research community shows strong convergence on a key idea: **jointly training generators and verifiers via RL produces better results than training either in isolation**. This spans from explicit dual-model co-training (RL Tango) to single-model multi-role self-play (Multi-Agent Evolve) to implicit reward model co-evolution (PRIME). The trend accelerated through 2025 into early 2026, with DeepSeek-Math-V2 being the first major production deployment of a generator-verifier training loop.

---

## Key Systems & Papers

### 1. RL Tango (NeurIPS 2025) — Canonical Generator-Verifier Co-Training
- **Authors:** Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi (MIT)
- **Core idea:** Interleaved RL training of generator and generative process-level verifier. Both improve together.
- **Key insight:** RL-trained generative verifiers are more robust and generalize better than SFT-trained or deterministic verifiers.
- **RL algorithm:** GRPO-based, outperforms vanilla GRPO.
- **Source:** [@iScienceLuvr](https://x.com/iScienceLuvr/status/1925515344309039586) | [arxiv:2505.15034](https://arxiv.org/abs/2505.15034)

### 2. DeepSeek-Math-V2 — First Production Generator-Verifier Loop
- **Organization:** DeepSeek
- **Core idea:** 685B-parameter model uses generator-verifier loop in training: writes proofs → verifier scores → RL closes the loop. First model to deploy this at scale.
- **Results:** IMO 2025 (5/6 problems), Putnam 2024 (118/120)
- **Significance:** Validates generator-verifier co-training as production-ready, not just research.
- **Source:** [@zhs05232838](https://x.com/zhs05232838/status/1994063966566519237) (DeepSeek author), [@jenzhuscott](https://x.com/jenzhuscott/status/1994035158203396404)

### 3. Multi-Agent Evolve / MAE (arxiv:2510.23595) — Self-Play Co-Evolution
- **Authors:** Jiaxuan You et al. (NVIDIA-affiliated)
- **Core idea:** Single LLM instantiated as 3 roles (Proposer, Solver, Judge) that co-evolve via RL. No external supervision needed.
- **Results:** +4.54% avg accuracy on Qwen2.5-3B; beats SFT, rivals strong self-play baselines.
- **Paradigm:** Beyond RLHF/RLVR — purely self-generated curriculum + self-generated rewards.
- **Source:** [@youjiaxuan](https://x.com/youjiaxuan/status/1983293231879393695) | Open-sourced with checkpoints

### 4. PRIME — Implicit PRM Co-Training
- **Core idea:** Interleaved training of policy LLM and implicit process reward model. The PRM (Q-function) provides per-token rewards and is updated online alongside the policy — "an expectation-minimization dance."
- **Results:** 2.5x faster training, 6.9% higher final rewards, 16.7% avg improvement on Eurus-2-7B-PRIME, >20% on AMC/AIME.
- **Key insight (Petr Baudis):** "The reward model LLM and the main LLM epochs are interleaved — the estimates are learned in parallel with finetuning the main model."
- **Source:** [@_akhaliq](https://x.com/_akhaliq/status/1875039314771660989), [@xpasky](https://x.com/xpasky/status/1875362293539570146), [@_philschmid](https://x.com/_philschmid/status/1876551158749135259)

### 5. MAPoRL — Multi-Agent Post-Co-Training
- **Author:** Chanwoo Park
- **Core idea:** Instead of prompting pre-trained models to collaborate, explicitly train LLMs to discover collaboration strategies via RL.
- **Source:** [@chanwoopark20](https://x.com/chanwoopark20/status/1947834449346887955), endorsed by [@rm_rafailov](https://x.com/rm_rafailov/status/1897750148010455366) (Rafael Rafailov, DPO co-author)

### 6. Self-Rewarding LLMs (Meta FAIR)
- **Author:** Jason Weston, Aaron Defazio et al.
- **Core idea:** LLM provides its own rewards via LLM-as-a-Judge during Iterative DPO. Reward modeling ability improves during training (not fixed).
- **Significance:** Early (Jan 2024) demonstration that verifier/judge capability can co-improve with generation.
- **Source:** [@aaron_defazio](https://x.com/aaron_defazio/status/1748525189447524827)

### 7. Self-Play SWE-RL / SSR (Meta FAIR, Dec 2025)
- **Authors:** Yuxiang Wei, Zhiqing Sun, Emily McMilin, Jonas Gehring et al.
- **Core idea:** Single LLM self-plays between bug-injection and bug-repair on real repos. No human-labeled issues or tests needed.
- **Paradigm:** Adversarial self-play (generator creates bugs, solver fixes them; reward is binary win/loss).
- **Source:** [@EdwardSun0909](https://x.com/EdwardSun0909/status/2004434784307859577) | [arxiv:2512.18552](https://arxiv.org/abs/2512.18552)

### 8. Search Self-Play / SSP (Alibaba)
- **Core idea:** Agent is both task proposer and problem solver. Proposer generates deep search queries with verifiable answers; solver attempts to answer; RAG check validates.
- **Significance:** Extends self-play co-training to agentic web search tasks.
- **Source:** [@jiqizhixin](https://x.com/jiqizhixin/status/1992439966258151743)

---

## Taxonomy of Co-Training Paradigms

| Paradigm | Generator | Verifier | Training | Example |
|----------|-----------|----------|----------|---------|
| **Explicit dual-model** | Separate LLM | Separate generative verifier | Interleaved RL | RL Tango |
| **Single-model multi-role** | Same LLM (Solver) | Same LLM (Judge) | Joint RL on all roles | Multi-Agent Evolve |
| **Implicit co-training** | Policy LLM | Implicit PRM (Q-function) | Online interleaved | PRIME |
| **Self-rewarding** | LLM | Same LLM (as Judge) | Iterative DPO | Self-Rewarding LLMs |
| **Adversarial self-play** | Bug-injector | Bug-solver | RL with binary rewards | Self-Play SWE-RL |
| **Task self-play** | Proposer | Solver + RAG verifier | RL | Search Self-Play (SSP) |
| **Production loop** | Proof generator | Proof verifier | RL closed loop | DeepSeek-Math-V2 |

---

## Key Themes & Insights

### 1. Interleaving > Sequential Training
Multiple systems (RL Tango, PRIME, DeepSeek-Math-V2) show that interleaved training of generator and verifier outperforms sequential pipelines (train verifier → freeze → train generator). The co-evolution creates a positive feedback loop.

### 2. The Generator-Discriminator Gap Is the Secret Sauce
Karpathy's insight ([tweet](https://x.com/karpathy/status/1821277264996352246)) explains why: verification is fundamentally easier than generation. Co-training exploits this asymmetry — the verifier provides a richer signal than outcome-only rewards.

### 3. Self-Play Removes the Human Bottleneck
MAE, SSP, Self-Play SWE-RL all demonstrate that LLMs can generate their own training curriculum AND their own reward signal. This removes dependence on human-labeled data for both tasks and evaluations.

### 4. Process Rewards > Outcome Rewards
PRIME and RL Tango both use process-level (step-by-step) reward signals rather than just outcome rewards. This provides denser gradients and faster learning. The Qwen PRM team confirms PRMs are "the unhidden secret" of reasoning models. ([@_philschmid](https://x.com/_philschmid/status/1879101437663154194))

### 5. Scaling to Production
DeepSeek-Math-V2 (685B params, IMO-level performance) proves generator-verifier co-training works at scale. Infrastructure like LlamaRL (Meta) and DAPO enables this at 8B–405B+ scale.

---

## Notable Researchers & Labs

| Researcher | Affiliation | Contribution |
|-----------|-------------|-------------|
| Kaiwen Zha, Dina Katabi | MIT | RL Tango |
| Zhihong Shao | DeepSeek | DeepSeek-Math-V2 |
| Jiaxuan You | NVIDIA/Stanford | Multi-Agent Evolve |
| Chanwoo Park | (with Rafailov endorsement) | MAPoRL |
| Jason Weston, Aaron Defazio | Meta FAIR | Self-Rewarding LLMs, Self-Taught Evaluators |
| Zhiqing Sun, Yuxiang Wei | Meta FAIR | Self-Play SWE-RL |
| Petr Baudis | Independent | Key PRIME explainer thread |
| Qwen Team | Alibaba | PRM development, SSP |

---

## Open Questions Discussed on X

1. **Can co-training scale indefinitely?** MAE claims "more compute → closer to AGI" but evidence is limited to small models so far.
2. **Reward hacking in self-play?** Multiple researchers note the risk when the verifier is co-trained with the generator.
3. **Does RL in pretraining (RLP, ICLR 2026) subsume post-training co-evolution?** Ali Hatamizadeh's RLP suggests RL can be pushed earlier.
4. **Is the verifier even necessary?** Harvard paper (discussed by [@rohanpaul_ai](https://x.com/rohanpaul_ai/status/1979436283060723797)) shows training-free sampling can rival RL on reasoning — but without generalization.

---

*All claims traced to specific X posts with URLs. Paper references included where available.*
