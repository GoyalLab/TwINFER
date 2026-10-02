cd /projects/b1255/hzhang/TwINFER_KA/code/TwINFER/drift_multiple_state
# Shared output dir with the b1042 job (5374312) and the login-node run;
# --skip-existing (default) means the three cooperate -- each grabs reps not
# yet written. Interleaving is fine.
~/.conda/envs/twinfer-code/bin/python regenerate_drift_A_B.py \
    --out-dir /projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_simulation \
    --reps 20 --jobs 1 --cores-per 5