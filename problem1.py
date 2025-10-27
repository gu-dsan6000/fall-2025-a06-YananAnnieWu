#!/usr/bin/env python3
"""
Problem 1: Log Level Distribution
"""

import argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

def create_spark(master_url: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName("Problem1_toPandas")
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

def main():
    parser = argparse.ArgumentParser(description="Problem 1: Log Level Distribution")
    parser.add_argument("master_url", help="Spark master URL")
    parser.add_argument("--net-id", required=True, help="Your net id, e.g., yw1150")
    args = parser.parse_args()

    spark = create_spark(args.master_url)
    spark.sparkContext.setLogLevel("WARN")

    s3_bucket = f"s3a://{args.net_id}-assignment-spark-cluster-logs"
    input_root = f"{s3_bucket}/data/"

    logs = (
        spark.read.format("text")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.log")
        .load(input_root)
    )

    outdir = Path("data/output")
    outdir.mkdir(parents=True, exist_ok=True)

    TS_RE = r"^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"
    LEVEL_RE = r"(INFO|WARN|ERROR|DEBUG)"
    COMP_RE = r"(?:INFO|WARN|ERROR|DEBUG)\s+([^:]+):"

    parsed = logs.select(
        F.regexp_extract("value", TS_RE, 1).alias("timestamp"),
        F.regexp_extract("value", LEVEL_RE, 1).alias("log_level"),
        F.regexp_extract("value", COMP_RE, 1).alias("component"),
        F.col("value").alias("message"),
    )

    parsed_nonempty = parsed.filter(F.col("log_level") != "")

    counts = (
        parsed_nonempty.groupBy("log_level")
        .count()
        .orderBy(F.desc("count"))
    )

    sample10 = (
        parsed_nonempty
        .select(F.col("message").alias("log_entry"), "log_level")
        .orderBy(F.rand())
        .limit(10)
    )

    counts.toPandas().to_csv(outdir / "problem1_counts.csv", index=False)
    sample10.toPandas().to_csv(outdir / "problem1_sample.csv", index=False)

    total_lines = logs.count()
    with_level = parsed_nonempty.count()
    unique_levels = counts.select(F.countDistinct("log_level")).first()[0]

    lines = [
        f"Total log lines processed: {total_lines}",
        f"Total lines with log levels: {with_level}",
        f"Unique log levels found: {unique_levels}",
        "",
        "Log level distribution:",
    ]
    for row in counts.collect():
        lvl, cnt = row["log_level"], row["count"]
        pct = (cnt / with_level * 100) if with_level else 0
        lines.append(f"  {lvl:<5}: {cnt:>10,} ({pct:5.2f}%)")

    (outdir / "problem1_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    spark.stop()

if __name__ == "__main__":
    main()