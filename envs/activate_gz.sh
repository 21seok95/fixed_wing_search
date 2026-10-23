#!/usr/bin/env bash
# Gazebo Harmonic (gz-sim 8) + PX4 SITL rc_cessna / advanced_plane. Usage: source envs/activate_gz.sh
# Paths exported by activate_sim.sh (Gazebo Classic) pull in a different protobuf and break the build.
unset CMAKE_PREFIX_PATH PKG_CONFIG_PATH LD_LIBRARY_PATH CPATH CXXFLAGS CFLAGS \
      GAZEBO_PLUGIN_PATH GAZEBO_MODEL_PATH GAZEBO_RESOURCE_PATH GAZEBO_IP GAZEBO_MASTER_URI
# shellcheck disable=SC1091
source "$HOME/.local/miniforge3/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
conda activate fw-gz

export PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
# PX4 builds with -Werror; conda gcc 15 and the newer conda protobuf trip these
export CXXFLAGS="${CXXFLAGS:-} -Wno-error=dangling-pointer -Wno-dangling-pointer -Wno-error=deprecated-declarations"
export CFLAGS="${CFLAGS:-} -Wno-error=dangling-pointer -Wno-dangling-pointer"
export GZ_IP=127.0.0.1

cd "$PX4_DIR"
echo "fw-gz active | gz sim $(gz sim --versions 2>/dev/null | head -1) | PX4_DIR=$PX4_DIR"
echo "Cessna:          make px4_sitl gz_rc_cessna"
echo "Advanced plane:  make px4_sitl gz_advanced_plane"
echo "Headless:        HEADLESS=1 make px4_sitl gz_rc_cessna"
