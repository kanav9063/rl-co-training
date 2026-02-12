# X/Twitter Raw Data: RL Co-Training Paradigms
## Collected: 2026-02-11

---

## 1. RL Tango: Generator-Verifier Co-Training

### Tweet: Tanishq Mathew Abraham (@iScienceLuvr)
- **URL:** https://x.com/iScienceLuvr/status/1925515344309039586
- **Content:** "RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning — we propose Tango, a novel framework that uses RL to concurrently train both an LLM generator and a verifier in an interleaved manner. A central innovation of Tango is its generative, process-level [verifier]"
- **Paper:** arxiv.org/abs/2505.15034 (Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi — MIT)
- **Venue:** NeurIPS 2025
- **GitHub:** github.com/kaiwenzha/RL-Tango
- **Key details (from HuggingFace paper page):** "This generative RL-trained verifier exhibits improved robustness and superior generalization compared to deterministic or SFT-trained verifiers, fostering effective mutual reinforcement with the generator."
- **From AlphaXiv:** Tango uses GRPO as its primary RL algorithm and contrasts performance against vanilla GRPO.

---

## 2. DeepSeek-Math-V2: Generator-Verifier Loop at Scale

### Tweet: Zhihong Shao (@zhs05232838) — DeepSeek author
- **URL:** https://x.com/zhs05232838/status/1994063966566519237
- **Content:** "We just shared some thoughts and results on self-verifiable mathematical reasoning. The released model, DeepSeekMath-V2, is strong on IMO-ProofBench and competitions like IMO 2025 (5/6 problems) and Putnam 2024 (a near-perfect score of 118/120)."
- **GitHub:** linked in tweet

### Tweet: Jen Zhu (@jenzhuscott)
- **URL:** https://x.com/jenzhuscott/status/1994035158203396404
- **Content:** "DeepSeek-Math-V2: 685B-parameter math monster built on V3.2-Exp-Base, fully open under Apache 2.0 - 1st model to use a generator-verifier loop in training: writes proofs → verifier scores them → RL closes the loop for self-verifiable reasoning. - Focuses on verifiable full proofs, not just final answers"

### Tweet: Adina Yakup (@AdinaYakup)
- **URL:** https://x.com/AdinaYakup/status/1994032043848610188
- **Content:** "Self-verifiable reasoning, Step-by-step proof checking, More faithful + reliable math logic, Scales verification for harder problems"

---

## 3. Multi-Agent Evolve (MAE): Proposer-Solver-Judge Co-Evolution

### Tweet: Jiaxuan You (@youjiaxuan) — Paper author
- **URL:** https://x.com/youjiaxuan/status/1983293231879393695
- **Content:** "Introducing Multi-Agent Evolve 🧠 A new paradigm beyond RLHF and RLVR: More compute → closer to AGI. No need for expensive data or handcrafted rewards. We show that an LLM can self-evolve — improving itself through co-evolution among roles (Proposer, Solver, Judge) via RL — all without external supervision"
- **Paper:** arxiv.org/abs/2510.23595

### Tweet: Jiaxuan You — Open-source release
- **URL:** https://x.com/youjiaxuan/status/1988399857459994642
- **Content:** "Multi-Agent Evolve is now fully open-source 🚀 With our codebase, you can pick your favorite LLM checkpoint and let it self-evolve, WITHOUT external supervision"

### Tweet: Rohan Paul (@rohanpaul_ai)
- **URL:** https://x.com/rohanpaul_ai/status/1983804313387594144
- **Content:** "New Nvidia paper shows how a single LLM can teach itself to reason better. It creates 3 roles from the same model, a Proposer, a Solver, and a Judge. The Proposer writes hard but solvable questions that stretch the model. The Solver answers those questions with clear steps... On Qwen2.5-3B, the approach lifts average accuracy by 4.54%, and it beats simple supervised fine tuning while rivaling a strong self play baseline without outside tools."

---

## 4. MAPoRL: Multi-Agent Post-Co-Training

### Tweet: Chanwoo Park (@chanwoopark20)
- **URL:** https://x.com/chanwoopark20/status/1947834449346887955
- **Content:** "(1/x) Excited to share our new work on MAPoRL🍁: Multi-Agent Post-Co-Training for Collaborative LLMs with RL. Most current approaches just prompt pre-trained models and hope they'll work together. But can we train LLM to discover the collaboration strategy?"

### Tweet: Rafael Rafailov (@rm_rafailov)
- **URL:** https://x.com/rm_rafailov/status/1897750148010455366
- **Content:** "This is a really cool project where we trained a multi-agent system of 3 LLMs to do cooperative problem-solving end-to-end with reinforcement learning! MARL holds a lot of promise to teach models to be more cooperative with real collaborators!"

---

## 5. MARTI: Multi-Agent Reinforced Training and Inference

### Tweet: Kaiyan Zhang (@OkhayIea)
- **URL:** https://x.com/OkhayIea/status/1927739357383086438
- **Content:** "Introducing: MARTI — Multi-Agent Reinforced Training and Inference. A unified framework for LLM-based Multi-Agent Systems with centralized interaction & distributed policy training. Supports structured workflows (debate, MoA, chain), custom rewards, and 3rd-party MAS"

---

## 6. PRIME: Implicit Process Reward Model Co-Training

### Tweet: AK (@_akhaliq)
- **URL:** https://x.com/_akhaliq/status/1875039314771660989
- **Content:** "Process Reinforcement through Implicit Rewards — We present PRIME, an open-source solution for online RL with process rewards, to advance reasoning abilities of language models beyond imitation or distillation."

### Tweet: Philipp Schmid (@_philschmid)
- **URL:** https://x.com/_philschmid/status/1876551158749135259
- **Content:** "PRIME directly learns a Q-function (scoring) that provides rewards for each token; it can be updated online with only the outcome improving math reasoning up to 27%"
- **Details:** "Filter prompts based on accuracy (keep only those with 20-80% success rate). Calculate outcome (binary reward) and process rewards (likelihood for between tokens) to update the policy model. Update the implicit PRM on the rollouts with the outcome reward. Perform advantage estimation with RLOO, separately for outcome and process rewards."

### Tweet: Petr Baudis (@xpasky) — Key explanatory thread
- **URL:** https://x.com/xpasky/status/1875362293539570146
- **Content:** "Quick primer for non-wizards about the post-MCTS LLM reasoning future (I'm kinda PRIME-pilled rn)... This LLM is then used as a Process RM assigning a reward to each token based on its continuously learned estimate of how much that token is helpful. 'Continuously' learned? The reward model LLM and the main LLM epochs are interleaved - the estimates are learned in parallel with finetuning the main model, a sort of expectation-minimization dance."

---

## 7. Self-Rewarding LLMs (Meta)

### Tweet: Aaron Defazio (@aaron_defazio) — Meta FAIR
- **URL:** https://x.com/aaron_defazio/status/1748525189447524827
- **Content:** "Self-Rewarding LMs - LM itself provides its own rewards on own generations via LLM-as-a-Judge during Iterative DPO - Reward modeling ability improves during training rather than staying fixed ...opens the door to superhuman feedback?"

---

## 8. Self-Play Fine-Tuning (SPIN)

### Tweet: John Nay (@johnjnay)
- **URL:** https://x.com/johnjnay/status/1742391053552930993
- **Content:** "LLM Self-Play Fine-Tuning - LLM generates its own training data from its previous iterations, refining its policy by discerning its self-generated responses vs human ones - Unlocks full potential of human data - Significantly improves perf across benchmarks"

---

## 9. Self-Taught Evaluators (Meta)

### Tweet: Jason Weston (@jaseweston) — Meta FAIR
- **URL:** https://x.com/jaseweston/status/1839669299859886175
- **Content:** "Today we are releasing code, models & data from the Self-Taught Evaluator paper, a method to train LLM judges with synthetic preference data... Our DPO model is a strong LLM judge on RewardBench, despite not using any human annotation in training data creation."

### Tweet: elvis/omarsar (@omarsar0)
- **URL:** https://x.com/omarsar0/status/1820849115607044401
- **Content:** "Meta presents Self-Taught Evaluators — It first generates contrasting outputs (good and bad model responses) and trains an LLM-as-a-Judge to produce reasoning traces and final judgments. The self-improvement scheme repeats the training process in an iterative way using its improved [judge]."

---

## 10. Search Self-Play (SSP) — Alibaba

### Tweet: 机器之心 JIQIZHIXIN (@jiqizhixin)
- **URL:** https://x.com/jiqizhixin/status/1992439966258151743
- **Content:** "What if LLM agents could scale RL training without human-crafted tasks or ground-truth answers? Alibaba's 'search self-play' (SSP) framework turns the agent into both task proposer and problem solver. The proposer generates deep search queries with verifiable answers; the solver attempts to answer them; and a RAG check validates each task using all retrieved evidence."

---

## 11. Self-Play SWE-RL (Meta FAIR)

### Tweet: Zhiqing Sun (@EdwardSun0909) — Paper co-author
- **URL:** https://x.com/EdwardSun0909/status/2004434784307859577
- **Content:** "Software agents can self-improve via self-play RL. Introducing Self-play SWE-RL (SSR): training a single LLM agent to self-play between bug-injection and bug-repair, grounded in real-world repositories, no human-labeled issues or tests."
- **Paper:** arxiv.org/abs/2512.18552

### Tweet: Axel Darmouni (@ADarmouni)
- **URL:** https://x.com/ADarmouni/status/2003810285866409995
- **Content:** "Self-Play on Coding tasks... Solver reward is 1 if wins, -1 if fails. This adversarial game was tested on the CodeGen team's Code World Model (CWM) 32B, and showed pretty good results!"

### Tweet: Dr. Alex Wissner-Gross (@alexwg)
- **URL:** https://x.com/alexwg/status/2004198585798103231
- **Content:** "Meta has trained an agent via self-play to autonomously inject and repair software bugs, outperforming humans on SWE-Bench"

---

## 12. RLP: RL in Pretraining (ICLR 2026)

### Tweet: Ali Hatamizadeh (@ahatamiz1)
- **URL:** https://x.com/ahatamiz1/status/2015867794626380146
- **Content:** "RLP re-imagines the foundations of LLM training by bringing reinforcement learning directly into the pretraining stage."
- **Venue:** Accepted at ICLR 2026

---

## 13. Process Reward Models — Qwen Team

### Tweet: Qwen (@Alibaba_Qwen)
- **URL:** https://x.com/Alibaba_Qwen/status/1879966399499759661
- **Content:** "Our latest research tackles the challenges of data annotation and evaluation in PRMs for better mathematical reasoning in LLMs... Our consensus filtering mechanism integrates MC with LLM-as-a-judge, improving both performance and data efficiency."
- **Blog:** qwenlm.github.io/blog/qwen2.5-math-prm/
- **Model:** Qwen2.5-Math-PRM-72B

### Tweet: Philipp Schmid (@_philschmid)
- **URL:** https://x.com/_philschmid/status/1879101437663154194
- **Content:** "Process Reward Models (PRM) and online RLHF are the unhidden secret to creating reasoning models like o1."

---

## 14. SWEET-RL: Multi-Turn Agent RL (Meta)

### Tweet: Yifei Zhou (@YifeiZhou02)
- **URL:** https://x.com/YifeiZhou02/status/1903166337691816316
- **Content:** "SWEET-RL: New paper from Meta introduces a new multi-turn LLM agent benchmark and a novel RL algorithm for training multi-turn LLM agents with effective credit assignment over the multiple turns."

---

## 15. Karpathy on Generator-Discriminator Gap

### Tweet: Andrej Karpathy (@karpathy)
- **URL:** https://x.com/karpathy/status/1821277264996352246
- **Content:** "RLHF is just barely RL... the LLM Assistant benefits from the generator-discriminator gap. That is, for many problem types, it is a significantly easier task for a human labeler to select the best of few candidate answers, instead of writing the ideal answer from scratch."

---

## 16. DAPO: Open-Source LLM RL at Scale

### Tweet: Qiying Yu (@qiying_yu)
- **URL:** https://x.com/qiying_yu/status/1902405115082104875
- **Content:** "Based on the Qwen-2.5-32B pretrained model, DAPO gets 50 points on AIME. This achieves the new SOTA performance using 50% training steps (previous SOTA achieved by DeepSeek R1's GRPO, 47 points on AIME). Fully open-sourced!"

---

## 17. LlamaRL: Distributed RL Framework (Meta)

### Tweet: DAIR.AI (@dair_ai)
- **URL:** https://x.com/dair_ai/status/1934311358184509683
- **Content:** "LlamaRL is a fully-distributed, asynchronous reinforcement learning framework designed for efficient large-scale LLM training (8B to 405B+ models)."

---

## 18. Alex Dimakis on RL with One Training Example

### Tweet: Alex Dimakis (@AlexGDimakis)
- **URL:** https://x.com/AlexGDimakis/status/1921348214525219206
- **Content:** "'RL with only one training example' and 'Test-Time RL' are two recent papers that I found fascinating. In the 'One Training example' paper the authors find one question and ask the model to solve it again and again. Every time, the model tries 8 times (the Group in GRPO), and a gradient step is performed, to increase the reward which is a very simple verification of the correct [answer]."
