#!/usr/bin/env python3
"""
Problem 2: Cluster Usage Analysis
Outputs:
  data/output/problem2_timeline.csv
  data/output/problem2_cluster_summary.csv
  data/output/problem2_stats.txt
  data/output/problem2_bar_chart.png
  data/output/problem2_density_plot.png
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession, functions as F

# -------------------- Spark setup --------------------
def create_spark(master_url: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName("Problem2_ClusterUsage")
        .master(master_url)
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.6,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.678")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider,"
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .getOrCreate()
    )

# -------------------- Spark pass: build CSVs --------------------
def run_spark_and_write_csvs(master_url: str, net_id: str, outdir: Path):
    spark = create_spark(master_url)
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    s3_bucket = f"s3a://{net_id}-assignment-spark-cluster-logs"
    input_root = f"{s3_bucket}/data/"

    logs = (
        spark.read.format("text")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.log")
        .load(input_root)
    )

    TS_RE  = r"^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"
    APP_RE = r"(application_\d+_\d+)"

    parsed = (
        logs.select(
            F.regexp_extract("value", TS_RE, 1).alias("ts_str"),
            F.regexp_extract("value", APP_RE, 1).alias("application_id"),
        )
        .filter((F.col("ts_str") != "") & (F.col("application_id") != ""))
        .withColumn("ts", F.to_timestamp("ts_str", "yy/MM/dd HH:mm:ss"))
        .withColumn("cluster_id", F.regexp_extract("application_id", r"application_(\d+)_\d+", 1))
        .withColumn("app_number", F.regexp_extract("application_id", r"application_\d+_(\d+)", 1))
    )

    # Per-application start/end
    apps = (
        parsed.groupBy("application_id")
        .agg(
            F.first("cluster_id", ignorenulls=True).alias("cluster_id"),
            F.first("app_number", ignorenulls=True).alias("app_number"),
            F.min("ts").alias("start_time"),
            F.max("ts").alias("end_time"),
        )
        .select(
            "cluster_id",
            "application_id",
            "app_number",
            F.date_format("start_time", "yyyy-MM-dd HH:mm:ss").alias("start_time"),
            F.date_format("end_time",   "yyyy-MM-dd HH:mm:ss").alias("end_time"),
        )
        .orderBy("cluster_id", "app_number")
    )

    outdir.mkdir(parents=True, exist_ok=True)
    timeline_pdf = apps.toPandas()
    timeline_csv = outdir / "problem2_timeline.csv"
    timeline_pdf.to_csv(timeline_csv, index=False)

    # Cluster summary
    summary = (
        timeline_pdf
        .groupby("cluster_id", as_index=False)
        .agg(
            num_applications=("application_id", "count"),
            cluster_first_app=("start_time", "min"),
            cluster_last_app=("end_time", "max"),
        )
        .sort_values("cluster_id")
    )
    summary_csv = outdir / "problem2_cluster_summary.csv"
    summary.to_csv(summary_csv, index=False)

    spark.stop()
    return timeline_csv, summary_csv

# -------------------- Visualization & stats --------------------
def make_visuals_and_stats(timeline_csv: Path, summary_csv: Path, outdir: Path):
    timeline = pd.read_csv(timeline_csv, dtype={"cluster_id": str, "app_number": str})
    summary = pd.read_csv(summary_csv, dtype={"cluster_id": str})

    # Stats file
    total_clusters = summary["cluster_id"].nunique()
    total_apps = len(timeline)
    avg_apps = (summary["num_applications"].mean() if len(summary) else 0.0)

    lines = [
        f"Total unique clusters: {total_clusters}",
        f"Total applications: {total_apps}",
        f"Average applications per cluster: {avg_apps:.2f}",
        "",
        "Most heavily used clusters:",
    ]
    heavy = summary.sort_values("num_applications", ascending=False)
    for _, r in heavy.iterrows():
        lines.append(f"  Cluster {r['cluster_id']}: {int(r['num_applications'])} applications")
    (outdir / "problem2_stats.txt").write_text("\n".join(lines), encoding="utf-8")

    # Bar chart: applications per cluster
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=summary.sort_values("num_applications", ascending=False),
        x="cluster_id", y="num_applications"
    )
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("# Applications")
    ax.set_title("Applications per Cluster")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{int(h)}", (p.get_x() + p.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(outdir / "problem2_bar_chart.png", dpi=150)
    plt.close()

    # Density plot (largest cluster): duration distribution
    # Compute per-app duration (seconds) from start/end
    tline = timeline.copy()
    tline["start_time"] = pd.to_datetime(tline["start_time"])
    tline["end_time"] = pd.to_datetime(tline["end_time"])
    tline["duration_sec"] = (tline["end_time"] - tline["start_time"]).dt.total_seconds().clip(lower=1)

    top_cluster = (
        summary.sort_values("num_applications", ascending=False)
        .iloc[0]["cluster_id"]
        if len(summary) else None
    )
    if top_cluster is not None:
        durs = tline.loc[tline["cluster_id"] == str(top_cluster), "duration_sec"]
        plt.figure(figsize=(9, 5))
        ax = sns.histplot(durs, bins=30, kde=True)
        ax.set_xscale("log")
        ax.set_xlabel("App Duration (seconds, log scale)")
        ax.set_ylabel("Count")
        ax.set_title(f"Duration Distribution — Cluster {top_cluster} (n={len(durs)})")
        plt.tight_layout()
        plt.savefig(outdir / "problem2_density_plot.png", dpi=150)
        plt.close()

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Problem 2: Cluster Usage Analysis")
    parser.add_argument("master_url", nargs="?", help="Spark master URL (e.g., spark://host:7077)")
    parser.add_argument("--net-id", help="Your net id, e.g., yw1150")
    parser.add_argument("--skip-spark", action="store_true", help="Use existing CSVs to regenerate stats/plots")
    args = parser.parse_args()

    outdir = Path("data/output")

    if args.skip_spark:
        timeline_csv = outdir / "problem2_timeline.csv"
        summary_csv = outdir / "problem2_cluster_summary.csv"
        make_visuals_and_stats(timeline_csv, summary_csv, outdir)
        return

    timeline_csv, summary_csv = run_spark_and_write_csvs(args.master_url, args.net_id, outdir)
    make_visuals_and_stats(timeline_csv, summary_csv, outdir)

if __name__ == "__main__":
    main()