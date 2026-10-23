#!/usr/bin/env bash
# Superseded by activate_gz.sh: PLAN validates on gz_rc_cessna (Gazebo Harmonic), and building PX4
# from the old Gazebo Classic env (fw-sim) would reconfigure ~/PX4-Autopilot/build with other compilers.
echo "activate_sim.sh is replaced by activate_gz.sh; switching." >&2
# shellcheck disable=SC1091
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/activate_gz.sh"
