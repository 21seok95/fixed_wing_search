#!/usr/bin/env bash
# Learning / scenario env (rl_search). Usage: source envs/activate_train.sh
# shellcheck disable=SC1091
source "$HOME/.local/miniforge3/etc/profile.d/conda.sh"
conda activate fw-train
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
echo "fw-train active | python=$(command -v python) | cwd=$(pwd)"
