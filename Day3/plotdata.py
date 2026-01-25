import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- Data (unit: 100 million kWh = 亿千瓦时) ----
df = pd.DataFrame({
    "year": [2019, 2020, 2021, 2022, 2023, 2024],
    "electricity": [824, 939, 1116, 1300, 1500, 1660],
})
df["yoy_pct"] = (df["electricity"].pct_change() * 100).round(2)

# ---- Plot style (try common Chinese fonts; fallback safely) ----
plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax1 = plt.subplots(figsize=(8.2, 4.6), dpi=180)

# Bars: electricity consumption
bars = ax1.bar(df["year"].astype(str), df["electricity"], width=0.4, zorder=1)
ax1.set_ylabel("用电量（亿千瓦时）")
ax1.yaxis.set_label_coords(-0.06, 0.8)
ax1.set_ylim(0, max(df["electricity"]) * 1.18)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
ax1.set_axisbelow(True)

# Value labels on bars
for b, v in zip(bars, df["electricity"]):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + max(df["electricity"])*0.015,
             f"{v}", ha="center", va="bottom", fontsize=14)

# Line: YoY growth
ax2 = ax1.twinx()
line_offset = max(df["yoy_pct"].dropna()) * 0.08
line_y = df["yoy_pct"] + line_offset
line_y_max = line_y.dropna().max()
ax2.plot(df["year"].astype(str), line_y, color="red", marker="o", markersize=6,
         markerfacecolor="red", markeredgecolor="red",
         linewidth=2.2, zorder=3)
ax2.set_ylabel("增长率（%）")
ax2.yaxis.set_label_coords(1.04, 0.9)
ax2.set_ylim(0, line_y_max * 1.2)
ax2.set_zorder(ax1.get_zorder() + 1)
ax2.patch.set_visible(False)

# Annotate YoY points (skip first NaN)
for x, y in zip(df["year"].astype(str), df["yoy_pct"]):
    if pd.notna(y):
        adjusted_y = y + line_offset
        ax2.text(x, adjusted_y + line_y_max*0.02, f"{y:.2f}%",
                 ha="center", va="bottom", fontsize=14, color="black")

# Title + source note
# ax1.set_title("中国数据中心（算力中心）用电量及增长率（2019–2024）", pad=10)
# fig.text(0.01, 0.01,
#          "来源：国家能源局、中国信息通信研究院（2019–2023见《算力电力协同发展研究报告（2025年）》图1；"
#          "2024为《算力经济发展研究报告（2025年）》信通院测算）。",
#         #  fontsize=8)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
