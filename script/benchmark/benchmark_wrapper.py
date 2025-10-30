import os
import json
from tqdm import tqdm
import subprocess

env_config_dir = 'benchmark/benchmark_config'

pairs = [(env_config, num_fingers)
         for env_config in os.listdir(env_config_dir)
         for num_fingers in range(2, 5)]

success, fail = [], []

try:
    for env_config, num_fingers in tqdm(pairs, desc="Processing"):
        result = subprocess.run(
            [
                "python", "-m", "benchmark.benchmark",
                "--num_finger", str(num_fingers),
                "--env_config_path", env_config
            ],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            success.append((env_config, num_fingers))
        else:
            fail.append({
                "num_fingers": num_fingers,
                "env_config_path": env_config,
                "reasons": result.stderr.strip() or "Unknown error"
            })

except KeyboardInterrupt:
    print("\n\n⏹️ Interrupted by user! Stopping early...\n")

# Always write out fail cases, even if interrupted
if fail:
    with open("benchmark/failures.json", "w") as f:
        json.dump(fail, f, indent=2)

print("\n✅ Successful runs:", success)
print("\n❌ Failed runs written to failures.json")
