import os
import json
import pandas as pd

# Directory containing the JSON files
directory = '/media/aioz-nghiale/data1/Proj/point_e_revised/archive/2025_10_31_Training_Data/gripper_logs/debug_benchmark_topdown/benchmark/gripping_a_5_HTP_a_0.4999999999999999_-0.5_-0.5_0.5000000000000001_a_0.4'

records = []

for filename in os.listdir(directory):
    if filename.startswith("gripping_result_id") and filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Extract index from filename (e.g. gripping_result_id12.json → 12)
            index = ''.join([c for c in filename if c.isdigit()])

            reward = data.get("reward", None)
            design_loss = data.get("design_loss", None)
            all_loss = data.get("all_loss", None)

            records.append({
                "index": int(index),
                "reward": reward,
                "design_loss": design_loss,
                "all_loss": all_loss
            })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Convert to DataFrame and save
df = pd.DataFrame(records).sort_values(by="index")
output_csv = os.path.join(directory, "gripping_results_summary.csv")
df.to_csv(output_csv, index=False)

print(f"Saved summary CSV to: {output_csv}")
