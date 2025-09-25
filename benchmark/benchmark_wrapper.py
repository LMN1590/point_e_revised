
import os

env_config_dir = 'benchmark/benchmark_config'

for env_config in os.listdir(env_config_dir):
    for num_fingers in range(2,4):
        os.system(f'python -m benchmark.benchmark --num_finger {num_fingers} --env_config_path {env_config}')