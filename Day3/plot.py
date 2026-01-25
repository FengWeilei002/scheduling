# -*- coding: utf-8 -*-
"""
Figure: 2024 global data center impactful outage causes (long-tail breakdown)
Source: Uptime Institute, Global Data Center Survey 2024, Figure 12 (n=97, rounded %)
"""

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_outage_causes_longtail(
    out_dir: str = "figs",
    filename_stem: str = "fig2_outage_causes_2024_longtail",
) -> None:
    # -----------------------------
    # Data (percent share)
    # -----------------------------
    df = pd.DataFrame(
        {
            "cause": [
                "电力（站内供配电）",
                "制冷",
                "网络",
                "IT系统（软/硬件）",
                "信息安全相关",
                "火灾",
                "原因未知",
                "火灾抑制系统",
                "托管服务商（Colocation）",
            ],
            "share_pct": [54, 13, 12, 11, 3, 2, 2, 1, 1],
        }
    )

    # Sort ascending for horizontal bar chart (bottom small → top large after invert)
    df = df.sort_values("share_pct", ascending=True).reset_index(drop=True)

    # -----------------------------
    # Plot setup
    # -----------------------------
    # Chinese font fallback list (Windows/macOS/Linux common options)
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9.2, 5.3), dpi=220)

    y = range(len(df))
    bars = ax.barh(y, df["share_pct"], height=0.62)

    # y labels
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["cause"], fontsize=15)

    # x axis
    ax.set_xlim(0, 60)
    ax.set_xlabel("占比（%）", fontsize=15)
    ax.xaxis.set_label_coords(0.95, -0.05)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    # cleaner frame
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # value labels
    for b, v in zip(bars, df["share_pct"]):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{int(v)}%", va="center", fontsize=15)


    # Footnote (source + definition)
    # fig.text(
    #     0.01,
    #     0.01,
    #     "数据来源：Uptime Institute《Global Data Center Survey 2024》Figure 12；"
    #     "口径：受访者“最近一次影响较大停机/中断事件”的主因；n=97；百分比四舍五入。\n"
    #     "提示：IT 系统 + 网络合计约 23%。",
    #     fontsize=9,
    # )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{filename_stem}.png")
    svg_path = os.path.join(out_dir, f"{filename_stem}.svg")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved:\n- {png_path}\n- {svg_path}")


if __name__ == "__main__":
    plot_outage_causes_longtail(out_dir="figs")
