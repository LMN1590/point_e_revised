import os
import json
from tqdm import tqdm
import subprocess

env_config_dir = 'script/benchmark/benchmark_config'

success, fail = [], []

try:
    for env_config in tqdm(os.listdir(env_config_dir), desc="Processing"):
        result = subprocess.run(
            [
                "python", "-m", "script.benchmark.finger_rep_benchmark.fingerrep_benchmark_topdown",
                "--env_config_path", env_config,
                '--gripper_dir', 'data/grippers',
            ],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            success.append((env_config))
        else:
            fail.append({
                "env_config_path": env_config,
                "reasons": result.stderr.strip() or "Unknown error"
            })

except KeyboardInterrupt:
    print("\n\n⏹️ Interrupted by user! Stopping early...\n")

# Always write out fail cases, even if interrupted
if fail:
    with open("script/benchmark/failures.json", "w") as f:
        json.dump(fail, f, indent=2)

print("\n✅ Successful runs:", success)
print("\n❌ Failed runs written to failures.json")
