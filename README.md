# fixed_wing_search

Fixed-wing UAV search (lawnmower vs RL). Simulation and scenario code live under
`rl_search/`; root scripts are the unchanged reference.

See **[ENV_SETUP.md](ENV_SETUP.md)** for the two environments:

- **`fw-train`** — learning / scenario generation
- **`fw-gz`** — Gazebo Harmonic + PX4 SITL (`gz_rc_cessna`)

```bash
source envs/activate_train.sh   # or activate_gz.sh
```

단일 무인기 강화학습 설계·구현 진행 상황은
**[RL TODO 체크리스트](docs/single_uav_rl_todo.md)**에서 관리합니다.
