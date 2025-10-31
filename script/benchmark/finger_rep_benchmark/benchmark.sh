# Generate Grippers
python -m script.benchmark.finger_rep_benchmark.generate_gripper --num_finger 4 --variance_scale 5.0 --max_segment_count 10 --hidden_dim 10 --num_grippers 100 --gripper_dir data/grippers

# Config Gen
python -m script.benchmark.config_gen

# Run Benchmark Wrapper
python -m script.benchmark.benchmark_wrapper

# Run Benchmark
python -m script.benchmark.finger_rep_benchmark.fingerrep_benchmark_topdown --env_config_path gripping_a_5_HTP_a_0.5000000000000001_0.5_-0.4999999999999999_0.5_a_0.3.yaml --gripper_dir data/grippers