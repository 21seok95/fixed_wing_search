# RL 구조·학습 시나리오 문헌 조사 (2026-11-17)

R13 사용자 보고(비행 중·끝 무렵 원 선회, 입자 없는 곳으로 이동)와 사용자 제안 두 가지(학습 시나리오 구성으로 일반화, 2배씩 커지는 입력 지도와
상황 인지를 위한 모델 구조)를 위해 맥에서 조사했다. 아래 본문은 조사 보고서 그대로다(영어). "Reported"는 출처에 있는 내용, "Judgement"는 조사자의 판단이다.
일부 항목은 조사 중 다시 확인하지 못했다고 끝에 적혀 있다(별표 * 학회명 등). 우리 문제에 쓰기 전에 원문을 확인한다.

요약 (우리 문제와의 관계):
- 원 선회: 결정적 정책(평균 행동)은 관측이 반복되면 고리에 빠진다는 보고가 있다([8], PPO, 지도 기반 UAV). 행동 샘플링이나 지나온 경로를 흐리게 그린 채널이
  고리를 대부분 없앴고, 경로 채널이 가장 좋았다. 우리 평가는 평균 행동으로 날리므로 같은 기제인지 먼저 확인할 수 있다.
- 시나리오 구성: 중간 상태에서 다시 시작(PPO 근거 [44], [45]), 성적이 나쁜 시나리오를 더 자주(PLR [38]), 범위를 자동으로 넓히기(ADR [37]).
  숨은 변수에 대한 커리큘럼은 분포를 틀어지게 할 수 있다는 주의([42]).
- 지도 피라미드: 지도마다 따로 된 CNN이 하나로 합친 CNN보다 낫다는 보고([1], 4배 간격 4장). 2배 간격 자체를 시험한 논문은 없어 실험으로 정할 일.
- 상황 인지: 벡터(남은 확률, 에너지, 시각)로 지도 특징을 조절(FiLM, 게이트)하는 방법, 여러 시간 뒤의 ΔPOS를 예측하는 보조 과제.
- 고정익 + 60분 + 움직이는 표적 + 입자 필터 믿음을 함께 다룬 논문은 찾지 못했다.

---

# Survey: multi-scale map encoders and training options for the fixed-wing search agent

**Conventions.** "Reported" means a source says it, checked in this session unless marked; "Judgement" is my inference for your setup. Effort
assumes SB3 2.9.2 (project venv; sb3-contrib is not installed on this Mac). Problem tags: (a) circling, (b) flying to empty areas, (c) weak late
signal, (d) scenario generalisation. Numbers in brackets point to the reference list. No project files were changed.

## 1. Multi-scale egocentric maps

- **Multi-scale input** (several agent-centred maps of growing extent, same pixel count).
  - Reported: [1] (robot coverage, SAC) used 4 heading-up scales, factor 4 apart, 32×32 each. One or two scales were "not sufficient"; exploration
    needed ≥3. The multi-scale input took 40 ms per training step against 2,500 ms for one full-resolution map.
  - Reported: ANS [2] gives its global policy two scales stacked as 8×240×240: a full-resolution crop around the agent and the whole map
    max-pooled (2× smaller in training, 4× at test).
  - Reported: in [3] (UAV, DDQN) a pooled global map plus a 17×17 local crop trained 1.3–4.7× faster; without it, agents in one mission type
    never learned to land reliably. Relevance: (b)(d).
- **Factor 2.** Reported: the glimpse in RAM [4] stacks patches "each successive patch having twice the width of the previous", resized to one
  size; ANS uses 2× and 4×, [1] uses 4×. No paper found that tests the factor itself. Judgement: the proposal covers the same 64× range with 7
  encoders instead of 5, adding about 0.6M parameters and about 40% encoder compute (each map's flatten-to-linear layer is 2048→128). Plausible
  but without direct evidence: test it as a single change. Effort: low.
- **Separate vs shared encoders vs stacking scales as channels.**
  - Reported: in [1] a CNN with separate filters per scale beat one CNN convolving all scales together (coverage 97.8/93.0% against 93.2/88.4%,
    mowing/exploration; an MLP reached 81.4/27.3%), because the same pixel at different scales is a different place.
  - Reported: the StarCraft II FullyConv agent [5] stacks screen and minimap features as channels after separate convolutions and calls this
    "likely non-optimal" for the same reason.
  - Judgement: an independent CNN per map is the better-supported design; `Conv2d(groups=n_scales)` computes the same in one call. No RL evidence
    found for sharing weights across scales; crossing a map edge to edge takes 48 s on the 1.6 km map and 52 min on the 102.4 km map (64×), so keep
    separate weights.
- **What a coarse cell shows.** Reported: [1] marks a coarse cell as frontier if it contains any frontier point; ANS uses max pooling.
  Judgement: for (b)(c), check that sparse particles late in the flight still show up after per-map normalisation.
- **Log-polar maps.** Reported: only [6] (log-polar-like sensor, A3C+LSTM, 3 Atari games): 5× fewer pixels at the same score; no use found in RL
  search or navigation. Reported: heading-up centring is supported: [7] aligned the belief image with the aircraft "to help the algorithm learn
  better policies", and [8] calls centring "essential for generalization".

## 2. Encoders and architecture

- **IMPALA ResNet** (15-layer residual CNN [9]). Reported: beat the 3-layer Nature-CNN at test time for every training-set size on CoinRun [10];
  larger models improved sample efficiency and generalisation on Procgen with PPO [11]. Relevance (d). Effort low (custom extractor), several
  times the convolution compute.
- **CoordConv** [12]. Reported: with A2C on 9 Atari games, better on 6, similar on 2, slightly worse on 1; no change in Ape-X. Judgement: the
  flatten-to-linear layer already gives each cell its own weights, so it matters only with pooling or attention. Effort low.
- **Spatial softmax** [13]. Reported: robot visuomotor control, not map-based RL. Judgement: an explicit bearing and range to the mass at each
  scale might help (b). Effort low.
- **Self-attention.** Reported: a relational module over CNN feature cells improved Box-World and set the state of the art on 6 StarCraft II
  minigames [14]; vision-transformer methods stayed below CNN with augmentation on pixel-based control [15]. Judgement: joint attention over
  tokens from all scales is the natural way to compare remaining mass across scales. Effort medium–high, more compute.
- **FiLM or gating** (the vector scales and shifts the CNN channels) [16]. Reported: gated attention beat concatenation for multitask and
  zero-shot generalisation [17]; ANS reports that feeding time, distance and angle as binned embeddings improved sample efficiency over raw
  values [2]. Judgement: conditioning the map features on remaining probability and energy is the most direct "awareness of the current
  situation" change for (a)(c). Effort low–medium.
- **Separate policy and value networks.**
  - Reported: Phasic Policy Gradient (PPG) [18] trains separate networks and periodically distils value features into the policy network;
    "significantly better sample efficiency than PPO in nearly every environment" on Procgen. Its Appendix B shows plain PPO with separate
    networks worse than shared on Procgen (the authors suspect high-dimensional inputs).
  - Reported: separate networks were better on 4 of 5 MuJoCo tasks [19]; shared representations can overfit to value-specific features, and
    decoupling improved Procgen generalisation [20].
  - Effort: `share_features_extractor=False` low but doubles the extractor; PPG medium–high (not in SB3/sb3-contrib; code [R1], [R2]; its
    auxiliary phase replays the last 32 rollouts, tens of GB at about 0.4 MB per float32 observation unless cut down).

## 3. Memory and recurrence

- **RecurrentPPO** [R3] (PPO with an LSTM; dict observations; separate actor/critic LSTMs by default; evaluation must pass `lstm_states` and
  `episode_start`). Reported: DRQN matched frame stacking on Atari with "no systematic advantage" but degraded less when observation quality
  changed [21]; well-tuned recurrent model-free RL beat specialised POMDP methods on 18 of 21 tasks [22]. Effort low–medium, slower training.
- **Evidence on circling and commitment to far goals.**
  - Reported: in ANS [2] the PPO+GRU baselines "often got stuck in local areas"; ANS keeps an explicit map and picks a new long-term goal every
    25 steps (relative coverage 0.948 against 0.789).
  - Reported: in [8] (PPO, map-based UAV) a deterministic policy loops when an observation repeats (4.5% of in-distribution tasks failed).
    Sampled actions, a random channel, or a decaying position-history channel (α 0.99) removed most loops; the history channel gave the best
    paths, and with sampled actions it also solved 27.5% of scenarios on an unseen map against at most 13.4% for the others.
  - Reported: [3] lists "small-scale decision alternation" as open and proposes macro-actions; [7] penalised bank angle "to prevent the aircraft
    from banking in tight circles". TempoRL [23] learns an action and how long to keep it (up to 10× faster than plain Q-learning).
- **Judgement.** No source shows an LSTM curing circling, and the maps already carry the belief and coverage. A decaying own-track channel is the
  cheapest memory and targets repeated observations (a); the zero-initialised warm start for a new channel (`load_expanded`) applies. Frame
  stacking would multiply the 25–35 map channels; previous actions in the vector are cheaper. A hierarchy has the best evidence for committing
  to far goals but changes the action interface (the 5 s heading control would stay as the low level).

## 4. Auxiliary tasks, augmentation and value scale

- **Auxiliary prediction.** Reported: UNREAL [24] (pixel control, reward prediction, value replay): 880% of human on Atari, 10× faster and 87% of
  human on Labyrinth. Direct Future Prediction [25] predicts changes of measurements at several future horizons ("a rich supervisory signal";
  won the Full Deathmatch track of the VizDoom competition). Auxiliary tasks reached DD-PPO's 40M-frame PointNav performance 5.5× faster [26].
  Judgement: predict ΔPOS over the next 1, 4, 16, 64 decisions normalised by the remaining mass, a target that does not shrink late (c).
  Effort medium (auxiliary head and loss in a PPO subclass).
- **Data augmentation.** Reported: RAD improved Procgen generalisation [27]; DrAC adds the policy and value regularisers actor-critic needs for
  augmentation (about 40% better Procgen test) [28]; CURL and DrQ were made mainly for off-policy methods [29], [30]. Judgement: random shifts
  and crops break the egocentric geometry; the valid symmetry is a left–right mirror with the turn sign flipped (networks built to respect such
  reflections converged faster in [31]). Effort medium.
- **Value normalisation.** Reported: PopArt [32], [33]; the MAPPO authors say value normalisation "often helps and never hurts" [34]; in [19] it
  was crucial on 2 MuJoCo tasks and harmful on 1. Effort low (`VecNormalize`), medium (PopArt).
- **Judgement on (c).** Global reward scaling cannot undo the decay within an episode, and SB3's per-minibatch advantage normalisation keeps late
  advantages small next to early ones. An untested option: r_t = log(1−POS_t) − log(1−POS_{t+1}) ≈ ΔPOS/(1−POS_t); its undiscounted sum is
  −log(1−POS_T), which rises with final POS, so it ranks outcomes the same while amplifying late gains (not exactly equivalent with γ < 1). With
  γ 0.995 and 5 s steps a gain 20 min ahead is weighted 0.30; [8] found that raising γ from 0.99 to 0.999 during training improved final
  performance. Running out of energy already ends the episode as terminated (`rl_real.py`), as [35] recommends.

## 5. Generalisation through scenario design

- **Domain randomisation** [36], with ranges widened automatically as the agent improves (ADR) [37]. Reported: PPO agents overfit "surprisingly
  large" sets of levels and generalise better with more diverse levels [10], [11].
- **Automatic curricula.** Reported: Prioritized Level Replay (PLR) [38] replays levels with high value error (average |GAE|); PPO on Procgen,
  combined with UCB-DrAC "over 76% improvement in test return"; code [R4]. PAIRED [39]; Robust PLR [40] updates only on curated levels and
  transferred better zero-shot; ACCEL [41] edits levels; code [R5] (PPO, archived 2025). SAMPLR [42]: a curriculum over hidden parameters shifts
  training away from the true distribution and can leave the policy suboptimal.
- **Starting from intermediate states.** Reported: reverse curriculum generation [43]; [44] used PPO and moved the start state backwards along a
  demonstration (74,500 on Montezuma's Revenge); [45] used PPO restarted from stored visited states and improved all 4 tasks, most in hard
  exploration (evaluation used only the true start states; the authors warn that the optimum can shift within a restricted policy class);
  Go-Explore [46] returns to saved simulator states before exploring.
- **Judgement.** The scenario generator is continuous, so the issue is where samples go, not overfitting to a fixed level set: that points to PLR
  over (distribution shape, launch distance, launch delay) seeds or ADR-style widening. These parameters are largely visible to the agent (type
  probe 94%), so the SAMPLR concern applies mainly to hidden factors such as true target motion differing from the particle model. Restarting
  from saved mid- and late-flight states would multiply late-flight samples (a)(c). Effort medium for both (snapshot and restore of the particle
  filter and UAV state; PLR needs a seed buffer, scores from the rollout buffer and `env_method` to set the next seed).

## 6. RL for UAV probabilistic search (2018–2026)

| Source | Algorithm, platform | Input and network | Loop / circling handling reported |
|---|---|---|---|
| [7] | DQN; fixed-wing, ±5° bank at 10 Hz | 100×100 belief and time-since-visit image, centred and heading-aligned; 3 × (conv + max-pool) | Bank-angle penalty |
| [3] | DDQN; grid UAV | Pooled global map + local crop, separate conv stacks | Macro-actions proposed |
| [8] | PPO; separate 5M-parameter actor and critic | Global and local maps + position history | History channel, sampled actions, γ raised during training |
| [1] | SAC; ground robot | 4-scale egocentric maps, separate filters per scale | Total-variation reward (mowing coverage 85.1→97.8%); episode cut after 1,000 steps without progress |
| [47] | SAC (SB3); drone with heading action | Gaussian-mixture parameters + 64-waypoint path (CNN), frame stacking | Penalty when little new probability is seen; found 19.0% of missing persons vs 11.9% for the best benchmark |
| [48] | SAC beat PPO | Recurrent auto-encoder | Not described (abstract only) |
| [49] | Recurrent DDQN + A2C | Not checked | 46% faster than zig-zag search (abstract) |
| [50] | PPO, multi-UAV, moving targets | Four 21×21 egocentric crops with decaying history maps | Map of when each cell was last observed |

No paper found combines a fixed-wing aircraft, 60-minute missions, moving targets and a particle-filter belief.

## Ranked for this project (judgement)

1. Decaying own-track channel, plus a check whether circling persists with sampled rather than deterministic actions (direct PPO/UAV evidence
   [8]; low effort; (a)).
2. Restarts from saved mid- and late-flight states (PPO evidence [44], [45]; more late-flight samples; (a)(c); medium).
3. Late-signal fixes: reward relative to remaining mass or value normalisation, plus a multi-horizon ΔPOS auxiliary head [25], [26], [34] ((c);
   low–medium).
4. Conditioning on mission phase (FiLM, gating or binned embeddings) [17], [2] (the situation-awareness change with the most evidence; low–medium).
5. Scale pyramid: keep separate encoders per scale [1]; test adding the 3.2 and 12.8 km maps as a single change; make sure sparse mass stays
   visible on coarse maps ((b)(d); low).
6. PLR or ADR over scenario seeds [38], [37] ((d); medium).
7. PPG [18] (good evidence for image-like inputs; medium–high effort, large memory).
8. Waypoint or commitment hierarchy [2], [23] (best evidence for committing to far goals; high effort, changes the action interface).
9. RecurrentPPO, IMPALA-CNN, attention, CoordConv (lower expected payoff here).

## Open questions

- Does circling persist with sampled actions, and do the observations repeat over one loop (the mechanism of [8])?
- Are the remaining particles still visible in the 51.2 and 102.4 km maps late in the flight after per-map normalisation?
- How large are late-flight advantages compared with early ones inside PPO minibatches?
- Factor-2 spacing against the current pyramid is unanswered by the literature: only an experiment can settle it.
- Not re-checked in the session (from memory): UNREAL task names, the IMPALA layer count, the PLR + UCB-DrAC pairing and the meaning of its
  `value_l1` score, RecurrentPPO dict support, venues marked *, and the author list of [49] (inferred from how [47] cites it).

## References

1. Jonnarth, Zhao, Felsberg, "Learning Coverage Paths in Unknown Environments with Deep Reinforcement Learning", ICML 2024.
2. Chaplot et al., "Learning to Explore using Active Neural SLAM", ICLR 2020.
3. Theile, Bayerlein, Nai, Gesbert, Caccamo, "UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning", ICAR 2021 (code: https://github.com/theilem/uavSim).
4. Mnih et al., "Recurrent Models of Visual Attention", NIPS 2014*.
5. Vinyals et al., "StarCraft II: A New Challenge for Reinforcement Learning", arXiv 2017.
6. Göransson, Krueger, "Learning Actions and Control of Focus of Attention with a Log-Polar-like Sensor", arXiv 2023.
7. Julian, Kochenderfer, "Distributed Wildfire Surveillance with Autonomous Aircraft using Deep Reinforcement Learning", Journal of Guidance, Control, and Dynamics, 2019.
8. Theile, Bayerlein, Caccamo, Sangiovanni-Vincentelli, "Learning to Recharge: UAV Coverage Path Planning through Deep Reinforcement Learning", arXiv 2309.03157, 2023.
9. Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures", ICML 2018*.
10. Cobbe et al., "Quantifying Generalization in Reinforcement Learning", ICML 2019*.
11. Cobbe et al., "Leveraging Procedural Generation to Benchmark Reinforcement Learning", ICML 2020*.
12. Liu et al., "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution", NeurIPS 2018.
13. Levine et al., "End-to-End Training of Deep Visuomotor Policies", JMLR 2016*.
14. Zambaldi et al., "Deep Reinforcement Learning with Relational Inductive Biases", ICLR 2019* (arXiv title "Relational Deep Reinforcement Learning").
15. Tao, Reda, van de Panne, "Evaluating Vision Transformer Methods for Deep Reinforcement Learning from Pixels", arXiv 2022.
16. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018*.
17. Chaplot et al., "Gated-Attention Architectures for Task-Oriented Language Grounding", AAAI 2018.
18. Cobbe, Hilton, Klimov, Schulman, "Phasic Policy Gradient", ICML 2021*.
19. Andrychowicz et al., "What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study", ICLR 2021*.
20. Raileanu, Fergus, "Decoupling Value and Policy for Generalization in Reinforcement Learning", ICML 2021*.
21. Hausknecht, Stone, "Deep Recurrent Q-Learning for Partially Observable MDPs", AAAI Fall Symposium 2015*.
22. Ni, Eysenbach, Salakhutdinov, "Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs", ICML 2022.
23. Biedenkapp et al., "TempoRL: Learning When to Act", ICML 2021.
24. Jaderberg et al., "Reinforcement Learning with Unsupervised Auxiliary Tasks", ICLR 2017*.
25. Dosovitskiy, Koltun, "Learning to Act by Predicting the Future", ICLR 2017.
26. Ye, Batra, Wijmans, Das, "Auxiliary Tasks Speed Up Learning PointGoal Navigation", CoRL 2020.
27. Laskin et al., "Reinforcement Learning with Augmented Data", NeurIPS 2020.
28. Raileanu et al., "Automatic Data Augmentation for Generalization in Reinforcement Learning", NeurIPS 2021*.
29. Laskin, Srinivas, Abbeel, "CURL: Contrastive Unsupervised Representations for Reinforcement Learning", ICML 2020*.
30. Kostrikov, Yarats, Fergus, "Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels", ICLR 2021*.
31. van der Pol et al., "MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning", NeurIPS 2020*.
32. van Hasselt et al., "Learning Values Across Many Orders of Magnitude", NeurIPS 2016*.
33. Hessel et al., "Multi-task Deep Reinforcement Learning with PopArt", AAAI 2019*.
34. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022 Datasets and Benchmarks (quote from the authors' BAIR blog post).
35. Pardo, Tavakoli, Levdik, Kormushev, "Time Limits in Reinforcement Learning", ICML 2018.
36. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017*.
37. OpenAI et al., "Solving Rubik's Cube with a Robot Hand", arXiv 2019.
38. Jiang, Grefenstette, Rocktäschel, "Prioritized Level Replay", ICML 2021*.
39. Dennis et al., "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design", NeurIPS 2020*.
40. Jiang et al., "Replay-Guided Adversarial Environment Design", NeurIPS 2021.
41. Parker-Holder et al., "Evolving Curricula with Regret-Based Environment Design", ICML 2022.
42. Jiang et al., "Grounding Aleatoric Uncertainty for Unsupervised Environment Design", NeurIPS 2022.
43. Florensa et al., "Reverse Curriculum Generation for Reinforcement Learning", CoRL 2017.
44. Salimans, Chen, "Learning Montezuma's Revenge from a Single Demonstration", NeurIPS 2018 Deep RL Workshop.
45. Tavakoli et al., "Exploring Restart Distributions", RLDM 2019.
46. Ecoffet et al., "First return, then explore", Nature 2021.
47. Ewers, Anderson, Thomson, "Deep Reinforcement Learning for Time-Critical Wilderness Search and Rescue Using Drones", Frontiers in Robotics and AI, 2025.
48. Ewers, Anderson, Thomson, "Recurrent Auto-Encoders for Enhanced Deep Reinforcement Learning in Wilderness Search and Rescue Planning", arXiv 2025.
49. Peake et al., "Wilderness Search and Rescue Missions using Deep Reinforcement Learning", IEEE SSRR 2020.
50. Yan, Jia, Bai, "Searching and Tracking an Unknown Number of Targets: A Learning-Based Method Enhanced with Maps Merging", Sensors 2021.

Repositories: [R1] https://github.com/openai/phasic-policy-gradient (archived), [R2] https://github.com/vwxyzjn/cleanrl (`ppg_procgen.py`),
[R3] https://github.com/Stable-Baselines-Team/stable-baselines3-contrib (RecurrentPPO), [R4] https://github.com/facebookresearch/level-replay,
[R5] https://github.com/facebookresearch/dcd (archived).
