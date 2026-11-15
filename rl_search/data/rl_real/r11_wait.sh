#!/bin/bash
# R11 start (user 2026-11-14 23:25): after the R10 continuation ends, the web stays up for the user to look; R11 starts when the user says so
# (touch rl_search/data/rl_real/r11_go) or at 00:30 at the latest; the web server is stopped first (memory: web + page ~4 GB).
cd /home/lee/fixed_wing_search
while systemctl --user is-active --quiet rl-r10; do sleep 20; done
deadline=$(date -d "tomorrow 00:30" +%s); [ $(date +%s) -lt $(date -d "today 00:30" +%s) ] && [ $(date +%H) -lt 1 ] && deadline=$(date -d "today 00:30" +%s)
while [ ! -e rl_search/data/rl_real/r11_go ] && [ $(date +%s) -lt $deadline ]; do sleep 15; done
systemctl --user stop rl-web 2>/dev/null
exec rl_search/data/rl_real/r11_start.sh
