#!/bin/bash
gcloud compute ssh slurm0-login-5vui2c4n-001 --command '/storage3/GENIAC_haijima/tools/monitor/monitor.sh' >| result.txt
python3 monitor.py result.txt
