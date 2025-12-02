#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = "/root/taste_assignment/output_step2/step_log.csv"
OUT_PATH = "/root/taste_assignment/output_step2/batch_loss_curve_clean.png"

df = pd.read_csv(LOG_PATH)

# 只保留 step >= 5
df = df[df["step"] >= 5]

# ---- 🔥 关键：对相同步长 step 的 loss 求平均，只保留一个 ----
df_grouped = df.groupby("step", as_index=False)["loss"].mean()

# 重新按 step 排序
df_grouped = df_grouped.sort_values("step").reset_index(drop=True)

plt.figure(figsize=(11,5))

# 蓝线：单一 batch loss 曲线
plt.plot(df_grouped["step"], df_grouped["loss"], 
         label="batch loss (avg per step)", 
         linewidth=1.2, alpha=0.8)

# 橙线：平滑后的曲线
window = 20
df_grouped["loss_smooth"] = df_grouped["loss"].rolling(window, min_periods=1).mean()

plt.plot(df_grouped["step"], df_grouped["loss_smooth"], 
         label=f"moving avg (window={window})", 
         linewidth=2.2, color="orange")

plt.xlabel("Batch step")
plt.ylabel("Loss")
plt.title("Batch-level Training Loss Curve")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
print(f"Saved → {OUT_PATH}")
