#!/bin/bash
# Interim push of R16 without the symmetric policy at 06:40 whatever the step (user 2026-11-24: "6시 40분에는 학습 완료 상관없이 가장 최신
# 모델과 텐서보드 기록 푸쉬"); training goes on, the PC stays on (the final push and power-off come from r16u_nosym_push.sh).
cd /home/lee/fixed_wing_search
T=r16u_chungju_nosym; D=rl_search/data/rl_real; L=$D/${T}_train.log; P=$D/r16u_nosym_push.log
last=$(ls -t $D/$T/ckpt/model_*.zip 2>/dev/null | head -1)
step=$(basename "$last" .zip | sed 's/model_//')
echo "$(date '+%m-%d %H:%M') interim push, latest checkpoint $step" >> $P
source envs/activate_train.sh > /dev/null
summary=$(python - <<'PY'
import json
from pathlib import Path
h = json.loads(Path("rl_search/data/rl_real/r16u_chungju_nosym/state.json").read_text())["history"].get("3", [])
print(" · ".join(f"{e['step']/1e4:.0f}만 {e['ratio']:.2f}" for e in h) or "none yet")
PY
)
git add -f $D/$T/config.json $D/$T/state.json $D/$T/tb $D/$T/tb_eval $last $L $D/r16u_nosym_push.sh $D/r16u_nosym_interim_push.sh \
  $D/r16_start.sh $D/r17_start.sh rl_search/data/real/amboy/rl_eval/report/r16_sym_removed.png docs/rl_experiments.md 2>> $P
git commit -q -m "R16 without the symmetric policy (Ubuntu, r16u_chungju_nosym): interim at $step steps (training goes on)

Jirisan validation (ratio to SAROPS): $summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >> $P 2>&1
if git pull -q --rebase --autostash origin feat/rl-env >> $P 2>&1 && git push -q origin HEAD:feat/rl-env >> $P 2>&1; then
  echo "$(date '+%m-%d %H:%M') interim pushed to feat/rl-env: $(git log --oneline -1)" >> $P
else
  git rebase --abort 2>/dev/null
  git push -q origin HEAD:refs/heads/r16u-nosym-results >> $P 2>&1 && echo "$(date '+%m-%d %H:%M') interim: feat/rl-env push failed; pushed to branch r16u-nosym-results" >> $P \
    || echo "$(date '+%m-%d %H:%M') INTERIM PUSH FAILED (see above)" >> $P
fi
