# Raw Paper Data: RL Co-Training Generator-Verifier Paradigms
Generated: 2026-02-11

## Semantic Scholar API Results

### Query: "RL Tango reinforcing generator verifier"
Found: RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning
- paperId: 5becc8d0a5066c58d2785798519f2cddad7f4482
- year: 2025, citations: 24
- Authors: Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi
- venue: arXiv.org
- Abstract: Reinforcement learning (RL) has recently emerged as a compelling approach for enhancing the reasoning capabilities of large language models (LLMs), where an LLM generator serves as a policy guided by a verifier (reward model). However, current RL post-training methods for LLMs typically use verifiers that are fixed (rule-based or frozen pretrained) or trained discriminatively via supervised fine-tuning (SFT). Such designs are susceptible to reward hacking and generalize poorly beyond their training distributions. To overcome these limitations, we propose Tango, a novel framework that uses RL to concurrently train both an LLM generator and a verifier in an interleaved manner. A central innovation of Tango is its generative, process-level LLM verifier, which is trained via RL and co-evolves with the generator. Importantly, the verifier is trained solely based on outcome-level verification correctness rewards without requiring explicit process-level annotations. This generative RL-trained verifier exhibits improved robustness and superior generalization compared to deterministic or SFT-trained verifiers, fostering effective mutual reinforcement with the generator. Extensive experiments demonstrate that both components of Tango achieve state-of-the-art results among 7B/8B-scale models: the generator attains best-in-class performance across five competition-level math benchmarks and four challenging out-of-domain reasoning tasks, while the verifier leads on the ProcessBench dataset. Remarkably, both components exhibit particularly substantial improvements on the most difficult mathematical reasoning problems. Code: https://github.com/kaiwenzha/rl-tango

### Query: "multi-agent reinforcement learning language model co-training"
Notable results:
- [43 cites] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning (2025)
- [52 cites] Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning (2025)

### Other queries returned rate-limited (429) responses from Semantic Scholar API.

---

## RL Tango Citation Graph (from Semantic Scholar)

### Papers CITING RL Tango (15 results):
- [0 cites] rePIRL: Learn PRM with Inverse RL for LLM Reasoning (2026)
- [0 cites] CPMobius: Iterative Coach-Player Reasoning for Data-Free Reinforcement Learning (2026)
- [0 cites] Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration (2026)
- [1 cites] RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System (2026)
- [0 cites] TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment (2026)
- [0 cites] Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning (2026)
- [2 cites] Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning (2025)
- [0 cites] Stepwise Think-Critique: A Unified Framework for Robust and Interpretable LLM Reasoning (2025)
- [10 cites] PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory (2025)
- [0 cites] SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning (2025)
- [2 cites] The Era of Agentic Organization: Learning to Organize with Language Models (2025)
- [1 cites] Foundational Automatic Evaluators: Scaling Multi-Task Generative Evaluator Training for Reasoning-Centric Domains (2025)
- [3 cites] LaSeR: Reinforcement Learning with Last-Token Self-Rewarding (2025)
- [4 cites] Hard2Verify: A Step-Level Verification Benchmark for Open-Ended Frontier Math (2025)
- [2 cites] Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning (2025)

### Papers REFERENCED by RL Tango (15 results):
- [52 cites] Process Reward Models That Think (2025)
- [59 cites] Efficient Reinforcement Finetuning via Adaptive Curriculum Learning (2025)
- [170 cites] Inference-Time Scaling for Generalist Reward Modeling (2025)
- [53 cites] GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning (2025)
- [10 cites] Dancing with Critiques: Enhancing LLM Reasoning with Stepwise Natural Language Self-Critique (2025)
- [1190 cites] DAPO: An Open-Source LLM Reinforcement Learning System at Scale (2025)
- [44 cites] Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning (2025)
- [39 cites] Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search (2025)
- [252 cites] Process Reinforcement through Implicit Rewards (2025)
- [434 cites] SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training (2025)
- [5463 cites] DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning (2025)
- [266 cites] rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking (2025)
- [3127 cites] Qwen2.5 Technical Report (2024)
- [169 cites] ProcessBench: Identifying Process Errors in Mathematical Reasoning (2024)
- [168 cites] Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning (2024)

---

## Full Paper Abstracts & Details

### Paper 1: RL Tango (arXiv: 2505.15034)
**RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning**
Authors: Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi (MIT)
Year: 2025 | Citations: 24 | Venue: arXiv
[Full abstract above in SS section]

### Paper 2: Generative Adversarial Reasoner / GAR (arXiv: 2512.16917)
**Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning**
Year: 2025 | Venue: arXiv
Abstract: LLMs with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors. We introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality. On AIME24: DeepSeek-R1-Distill-Qwen-7B 54.0→61.3 (+7.3), DeepSeek-R1-Distill-Llama-8B 43.7→53.7 (+10.0). +22.9% on AIME24 and +19.5% on AIME25 for Llama backbone, +35.3% on LiveMathBench-Hard for Qwen.

Key technical details from HTML:
- Partitions reasoning chains into "logically complete slices" for efficient review
- Discriminator generates structured rationales, not just scores
- Two complementary discriminator rewards: alignment reward (error detection) + discriminative reward (distinguishing from reference)
- Inspired by GANs (Goodfellow et al., 2014)
- Addresses reward hacking through adversarial training
- 400 training steps with AdamW, 8 H100 GPUs

### Paper 3: CURE (arXiv: 2506.03136)
**Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning**
Year: 2025 | Venue: NeurIPS 2025 Spotlight
Abstract: We propose CURE, a novel reinforcement learning framework with a dedicated reward design that co-evolves coding and unit test generation capabilities based on their interaction outcomes, without any ground-truth code as supervision. This approach enables flexible and scalable training and allows the unit tester to learn directly from the coder's mistakes. ReasonFlux-Coder-7B and 14B improve code generation accuracy by 5.3% and Best-of-N accuracy by 9.0% after optimization on Qwen2.5-Instruct models, outperforming Qwen-Coder, DeepSeek-Coder, and Seed-Coder. 8.1% improvement for downstream agentic coding.

### Paper 4: SPIN (arXiv: 2401.01335)
**Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models**
Authors: Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu (UCLA)
Year: 2024 | Venue: ICML 2024
Abstract: We propose Self-Play fIne-tuNing (SPIN), which starts from a supervised fine-tuned model. At the heart of SPIN lies a self-play mechanism, where the LLM refines its capability by playing against instances of itself. The LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from those obtained from human-annotated data. Theoretically, the global optimum is achieved only when the LLM policy aligns with the target data distribution. SPIN can significantly improve LLM performance and even outperform models trained through DPO supplemented with extra GPT-4 preference data.

### Paper 5: V-STaR (arXiv: 2402.06457)
**V-STaR: Training Verifiers for Self-Taught Reasoners**
Year: 2024 | Venue: arXiv
Abstract: V-STaR utilizes both the correct and incorrect solutions generated during the self-improvement process to train a verifier using DPO that judges correctness of model-generated solutions. This verifier is used at inference time to select one solution among many candidates. Running V-STaR for multiple iterations results in progressively better reasoners and verifiers, delivering a 4% to 17% test accuracy improvement over existing self-improvement and verification approaches on code generation and math reasoning benchmarks with LLaMA2 models.

### Paper 6: APO (arXiv: 2311.08045)
**Adversarial Preference Optimization: Enhancing Your Alignment via RM-LLM Game**
Authors: Pengyu Cheng et al.
Year: 2023 | Venue: ACL 2024 Findings
Abstract: We propose Adversarial Preference Optimization (APO), where the LLM and the reward model update alternatively via a min-max game. Through adversarial training, the reward model can adapt to the shifted generation distribution of the LLM without any additional annotation. Enhances existing alignment baselines in helpfulness and harmlessness.

### Paper 7: GenRM (arXiv: 2408.15240)
**Generative Verifiers: Reward Modeling as Next-Token Prediction**
Year: 2024 | Venue: ICLR 2025
Abstract: We propose training verifiers using the next-token prediction objective, jointly on verification and solution generation. GenRM outperforms discriminative, DPO verifiers, and LLM-as-a-Judge: 5%→45.3% on algorithmic tasks, 73%→93.4% on GSM8K. Easy-to-hard: 28%→44.6% on MATH, 37.9%→53.5% on MMLU abstract algebra. Training GenRM with synthetic verification rationales is sufficient to pick out subtle errors on math problems.

### Paper 8: Math-Shepherd (arXiv: 2312.08935)
**Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations**
Authors: Peiyi Wang, Lei Li, Zhihong Shao et al.
Year: 2023 | Venue: ACL 2024
Abstract: An innovative process-oriented math process reward model that assigns a reward score to each step of math problem solutions. Training achieved using automatically constructed process-wise supervision data. Results: Step-by-step PPO with Math-Shepherd improves Mistral-7B (77.9%→84.1% on GSM8K, 28.6%→33.0% on MATH). With verification: 89.1% and 43.5% on GSM8K and MATH respectively.

### Paper 9: SCoRe (arXiv: 2409.12917)
**Training Language Models to Self-Correct via Reinforcement Learning**
Authors: Aviral Kumar, Vincent Zhuang, Rishabh Agarwal et al. (Google DeepMind)
Year: 2024 | Venue: arXiv
Abstract: A multi-turn online RL approach, SCoRe, that significantly improves an LLM's self-correction ability using entirely self-generated data. Addresses shortcomings of prior self-correction methods that depend on multiple models or more advanced models.

### Paper 10: rStar-Math (arXiv: 2501.04519)
**rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking**
Year: 2025 | Citations: 266 | Venue: arXiv
Abstract: Small LMs can rival or surpass OpenAI o1 math reasoning without distillation. Uses MCTS where a math policy SLM performs test-time search guided by an SLM-based process reward model. Three innovations: (1) code-augmented CoT data synthesis via MCTS rollouts; (2) novel PRM training avoiding naive step-level annotation, yielding a process preference model (PPM); (3) self-evolution recipe where policy SLM and PPM are iteratively evolved. Through 4 rounds of self-evolution with millions of solutions for 747k problems: Qwen2.5-Math-7B 58.8%→90.0% on MATH, Phi3-mini-3.8B 41.4%→86.4%. Solves 53.3% of AIME problems.

### Paper 11: Let's Verify Step by Step (arXiv: 2305.20050)
**Let's Verify Step by Step**
Authors: Lightman et al. (OpenAI)
Year: 2023 | Venue: ICLR 2024
Abstract: Process supervision significantly outperforms outcome supervision for training models to solve problems from the MATH dataset. Process-supervised model solves 78% of problems. Active learning significantly improves efficacy of process supervision. Released PRM800K dataset of 800,000 step-level human feedback labels.

---

## Web Search Results

### "generator verifier co-training arxiv 2024 2025 2026"
- Generative Verifiers: Reward Modeling as Next-Token Prediction (2408.15240) - ICLR 2025
- An Efficient Rubric-based Generative Verifier (2510.14660)
- Benchmarking and Improving Generator-Validator Consistency (2310.01846)

### "self-improving LLM generator discriminator co-training"
- Generative Adversarial Reasoner (2512.16917) - key find
- RLSR: Reinforcement Learning from Self Reward (2505.08827)
- SCoRe (2409.12917)
- Generative Adversarial Distillation / GAD (2511.10643) - frames student as generator, trains discriminator

### "co-evolving verifier generator RL LLM 2025"
- CURE: Co-Evolving LLM Coder and Unit Tester (2506.03136) - NeurIPS 2025 Spotlight
- Generative Adversarial Reasoner (2512.16917)

### "ARS co-evolving dual system"
- No specific "ARS" paper found matching this description. Closest: MARS (2510.04935) - dual-system deep research via multi-agent RL. The "ARS" reference in the task prompt may not correspond to a specific published paper.
