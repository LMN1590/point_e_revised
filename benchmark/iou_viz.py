import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
csv_path = "logs/experiment_results_4_with_iou_final - experiment_results_4_with_iou.csv"  # Change to your CSV file path
pdf_output = "graspability_plot.pdf"
user_slope = -0.07892270641
user_intercept = 0.06845819726

# === GLOBAL FONT SETTINGS ===
plt.rcParams.update({
    "font.size": 18,          # Base font size (affects ticks, legend, etc.)
    "axes.titlesize": 18,     # Title size
    "axes.labelsize": 17,     # Axis label size
    "legend.fontsize": 15,    # Legend text
    "xtick.labelsize": 14,    # X tick labels
    "ytick.labelsize": 14     # Y tick labels
})

# === LOAD CSV ===
df = pd.read_csv(csv_path)

# === SCATTER PLOT ===
plt.figure(figsize=(8, 6))
plt.scatter(df["aspect_val"], df["reward"], alpha=0.6, label="Data points", color="steelblue")

# === CUSTOM LINE ===
x_line = np.linspace(df["aspect_val"].min(), df["aspect_val"].max(), 100)
y_line = user_slope * x_line + user_intercept
plt.plot(x_line, y_line, color="red", linewidth=2.0, label=f"y = {user_slope:.3f}x + {user_intercept:.3f}")

# === LABELS & STYLE ===
# plt.title("Reward vs Aspect_val with Custom Linear Approximation")
plt.xlabel("Geometric Graspability Score")
plt.ylabel("Final CoM Height")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
# plt.show()

# === SAVE TO PDF ===
plt.tight_layout()
plt.savefig(pdf_output, format="pdf", bbox_inches="tight")
plt.close()

print(f"✅ Plot with larger text saved to {pdf_output}")
