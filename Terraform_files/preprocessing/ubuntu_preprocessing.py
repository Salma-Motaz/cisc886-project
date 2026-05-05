# -*- coding: utf-8 -*-
"""
Ubuntu Dialogue Corpus — Chatbot Fine-Tuning Preprocessing
EMR-compatible version — figures saved to S3
NetID: 25nplx
"""

import sys
import os
import subprocess

def _ensure():
    for pkg in ["matplotlib", "boto3"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet", "--user"])

_ensure()

import site
sys.path.insert(0, site.getusersitepackages())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io
import boto3
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import LongType

# ── Paths (S3) ────────────────────────────────────────────────────────────
INPUT_PATH = "s3://25nplx-cisc886-bucket/ubuntu_raw.parquet"
OUTPUT_DIR = "s3://25nplx-cisc886-bucket/output/processed"
FIGURES_DIR = "s3://25nplx-cisc886-bucket/output/figures"
S3_BUCKET  = "25nplx-cisc886-bucket"
AWS_REGION = "us-east-1"

# ── SparkSession — clean EMRFS-only config ────────────────────────────────
# s3:// is handled natively by EMRFS on EMR — no s3a configs needed.
# Removed: s3a.impl, s3a.credentials.provider, commitProtocolClass,
#          S3ACommitterFactory, EmrOptimizedSparkSqlParquetOutputCommitter
# Those configs are for s3a:// on non-EMR clusters and cause
# NoSuchMethodException / class-not-found errors on EMR 6.x with Spark 3.4.
spark = (
    SparkSession.builder
    .appName("UbuntuChatbotPreprocessing")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.hadoop.fs.s3.impl",
            "com.amazon.ws.emr.hadoop.fs.EmrFileSystem")   # EMRFS for s3://
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print(f"Spark version : {spark.version}")
print(f"Input         : {INPUT_PATH}")
print(f"Output dir    : {OUTPUT_DIR}")
print(f"Figures       : s3://{S3_BUCKET}/figures/")


def save_figure_to_s3(fig, filename):
    """Save a matplotlib figure directly to S3 without touching local disk."""
    s3  = boto3.client("s3", region_name=AWS_REGION)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    key = f"output/figures/{filename}"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf, ContentType="image/png")
    print(f"Saved figure -> s3://{S3_BUCKET}/{key}")
    plt.close(fig)


STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn",
    "ok", "okay", "yeah", "yes", "hi", "hello", "hey", "thanks",
    "thankyou", "thx", "bye", "lol"
}

try:

    # ── Step 1 — Load ─────────────────────────────────────────────────────
    df = spark.read.parquet(INPUT_PATH)
    df = df.withColumn(
        "timestamp_epoch",
        F.unix_timestamp(F.col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").cast(LongType())
    )
    df.cache()
    total_rows = df.count()
    print(f"Loaded rows: {total_rows:,}")

    # ── Step 2 — Inspect ──────────────────────────────────────────────────
    print("=== Schema ===")
    df.printSchema()
    print(f"\nTotal rows: {total_rows:,}")

    print("\n=== Null / Empty Counts ===")
    for col in ["dialogue_id", "timestamp", "speaker", "addressee", "message"]:
        n_null = df.filter(F.col(col).isNull() | (F.trim(F.col(col)) == "")).count()
        print(f"  {col:<15}: {n_null:>10,}  ({100 * n_null / total_rows:.2f}%)")

    n_bad_ts   = df.filter(F.col("timestamp_epoch").isNull()).count()
    n_dialogues = df.select("dialogue_id").distinct().count()
    print(f"\n  Unparseable timestamps: {n_bad_ts:,}  ({100 * n_bad_ts / total_rows:.2f}%)")
    print(f"  Unique dialogue_ids   : {n_dialogues:,}")
    df.show(10, truncate=80)

    # ── Step 3 — Clean ────────────────────────────────────────────────────
    df_clean = df.filter(F.col("message").isNotNull() & (F.trim(F.col("message")) != ""))
    after_msg = df_clean.count()
    print(f"After empty-message filter : {after_msg:,}  (removed {total_rows - after_msg:,})")

    df_clean = df_clean.filter(~F.col("message").startswith("-!-"))
    after_sys = df_clean.count()
    print(f"After system-message filter: {after_sys:,}  (removed {after_msg - after_sys:,})")

    stop_words_list = list(STOP_WORDS)
    df_clean = df_clean.withColumn("word_count", F.size(F.split(F.trim(F.col("message")), r"\s+")))
    df_clean = df_clean.filter(
        ~((F.col("word_count") == 1) & F.lower(F.trim(F.col("message"))).isin(stop_words_list))
    ).drop("word_count")

    after_words = df_clean.count()
    print(f"After low-value 1-word filter : {after_words:,}  (removed {after_sys - after_words:,})")
    df_clean.cache()
    print(f"\nRetained: {after_words:,} / {total_rows:,} rows ({100 * after_words / total_rows:.1f}%)")

    # ── Step 3b — Infer Missing Speakers ──────────────────────────────────
    window_spec = Window.partitionBy("dialogue_id").orderBy("timestamp_epoch")

    df_clean = (
        df_clean
        .withColumn("prev_speaker", F.lag("speaker",  1).over(window_spec))
        .withColumn("next_speaker", F.lead("speaker", 1).over(window_spec))
    )
    df_clean = df_clean.withColumn(
        "speaker",
        F.when(
            F.col("speaker").isNull() | (F.trim(F.col("speaker")) == ""),
            F.coalesce(
                F.when(F.col("next_speaker").isNotNull() & (F.col("next_speaker") != ""), F.col("next_speaker")),
                F.when(F.col("prev_speaker").isNotNull() & (F.col("prev_speaker") != ""), F.col("prev_speaker")),
                F.lit("unknown")
            )
        ).otherwise(F.col("speaker"))
    ).drop("prev_speaker", "next_speaker")

    df_clean = (
        df_clean
        .withColumn("prev_speaker", F.lag("speaker", 1).over(window_spec))
        .withColumn("next_speaker", F.lead("speaker", 1).over(window_spec))
    )
    df_clean = df_clean.withColumn(
        "addressee",
        F.when(
            F.col("addressee").isNull() | (F.trim(F.col("addressee")) == ""),
            F.coalesce(F.col("prev_speaker"), F.lit("unknown"))
        ).otherwise(F.col("addressee"))
    ).drop("prev_speaker", "next_speaker")

    empty_speakers   = df_clean.filter(F.col("speaker").isNull() | (F.trim(F.col("speaker")) == "")).count()
    unknown_speakers = df_clean.filter(F.col("speaker") == "unknown").count()
    print(f"Remaining empty speakers  : {empty_speakers:,}")
    print(f"Rows labeled 'unknown'    : {unknown_speakers:,}")
    df_clean.cache()

    # ── Step 4 — Build Conversation Pairs ─────────────────────────────────
    window_spec = Window.partitionBy("dialogue_id").orderBy("timestamp_epoch")

    df_pairs = (
        df_clean
        .withColumn("context",          F.lag("message",         1).over(window_spec))
        .withColumn("context_speaker",  F.lag("speaker",         1).over(window_spec))
        .withColumn("context_ts_epoch", F.lag("timestamp_epoch", 1).over(window_spec))
        .withColumnRenamed("message",         "response")
        .withColumnRenamed("speaker",         "response_speaker")
        .withColumnRenamed("timestamp_epoch", "response_ts_epoch")
        .filter(F.col("context").isNotNull())
    )

    pairs_before_gap = df_pairs.count()
    print(f"Pairs before gap filter: {pairs_before_gap:,}")
    df_pairs.cache()

    # ── Step 5 — Gap Filter ───────────────────────────────────────────────
    MAX_GAP_SECONDS = 10 * 60
    df_pairs = df_pairs.withColumn(
        "time_gap_seconds",
        F.col("response_ts_epoch") - F.col("context_ts_epoch")
    )
    df_pairs = df_pairs.filter(
        F.col("time_gap_seconds").isNotNull() &
        (F.col("time_gap_seconds") >= 0) &
        (F.col("time_gap_seconds") <= MAX_GAP_SECONDS)
    )
    after_gap = df_pairs.count()
    print(f"Pairs after gap filter: {after_gap:,}  (removed {pairs_before_gap - after_gap:,})")
    df_pairs.cache()

    # ── Step 6 — Length Filter ────────────────────────────────────────────
    MAX_CHARS = 7500
    df_pairs = df_pairs.withColumn(
        "combined_length",
        F.length(F.col("context")) + F.length(F.col("response"))
    )
    df_pairs = df_pairs.filter(F.col("combined_length") <= MAX_CHARS)
    after_len = df_pairs.count()
    print(f"Pairs after length filter: {after_len:,}  (removed {after_gap - after_len:,})")
    df_pairs.cache()

    # ── Step 7 — Format Prompt ────────────────────────────────────────────
    df_formatted = df_pairs.withColumn(
        "text",
        F.concat(
            F.lit("### Human:\n"), F.col("context"),
            F.lit("\n### Assistant:\n"), F.col("response")
        )
    )
    df_formatted = df_formatted.select(
        "dialogue_id", "context", "response", "text",
        "combined_length", "time_gap_seconds"
    )
    print(f"Total formatted pairs: {df_formatted.count():,}")

    # ── Step 8 — Deduplicate ──────────────────────────────────────────────
    before_dedup = df_formatted.count()
    df_deduped   = df_formatted.dropDuplicates(["context", "response"])
    after_dedup  = df_deduped.count()
    print(f"Before dedup : {before_dedup:,}")
    print(f"After dedup  : {after_dedup:,}")
    print(f"Removed dupes: {before_dedup - after_dedup:,}")
    df_deduped.cache()

    # ── Step 9 — Train / Val / Test Split ─────────────────────────────────
    SEED = 42
    train_df, val_df, test_df = df_deduped.randomSplit([0.8, 0.1, 0.1], seed=SEED)
    train_df.cache()
    val_df.cache()
    test_df.cache()

    n_train = train_df.count()
    n_val   = val_df.count()
    n_test  = test_df.count()
    n_total = n_train + n_val + n_test

    print(f"Train : {n_train:>10,}  ({100 * n_train / n_total:.1f}%)")
    print(f"Val   : {n_val:>10,}  ({100 * n_val   / n_total:.1f}%)")
    print(f"Test  : {n_test:>10,}  ({100 * n_test  / n_total:.1f}%)")
    print(f"Total : {n_total:>10,}")

    # ── Step 10 — EDA Figures ─────────────────────────────────────────────
    SAMPLE_SIZE = 50_000
    train_pdf = train_df.limit(SAMPLE_SIZE).toPandas()
    print(f"Sampled {len(train_pdf):,} rows for EDA")

    train_pdf["approx_tokens"] = train_pdf["combined_length"] / 4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train_pdf["approx_tokens"], bins=80, color="#2E86AB", edgecolor="white", linewidth=0.3)
    ax.axvline(train_pdf["approx_tokens"].median(), color="#E84855", linestyle="--",
               linewidth=1.5, label=f'Median: {train_pdf["approx_tokens"].median():.0f} tokens')
    ax.axvline(1875, color="#F4A261", linestyle=":", linewidth=1.5, label="Max allowed (1,875 tokens)")
    ax.set_xlabel("Approximate Token Count (context + response)", fontsize=12)
    ax.set_ylabel("Number of Pairs", fontsize=12)
    ax.set_title("Figure 1: Token Length Distribution of Training Pairs", fontsize=14, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    save_figure_to_s3(fig, "fig1_token_distribution.png")
    print(f"Mean tokens  : {train_pdf['approx_tokens'].mean():.1f}")
    print(f"Median tokens: {train_pdf['approx_tokens'].median():.1f}")
    print(f"95th pct     : {train_pdf['approx_tokens'].quantile(0.95):.1f}")

    turns_per_dialogue = train_df.groupBy("dialogue_id").count().limit(SAMPLE_SIZE).toPandas()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(turns_per_dialogue["count"], bins=60, color="#52796F", edgecolor="white", linewidth=0.3)
    ax.axvline(turns_per_dialogue["count"].median(), color="#E84855", linestyle="--",
               linewidth=1.5, label=f'Median: {turns_per_dialogue["count"].median():.0f} turns')
    ax.set_xlabel("Number of Pairs per Dialogue", fontsize=12)
    ax.set_ylabel("Number of Dialogues", fontsize=12)
    ax.set_title("Figure 2: Turns per Dialogue Distribution (Training Set)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    save_figure_to_s3(fig, "fig2_turns_per_dialogue.png")
    print(f"Mean turns/dialogue  : {turns_per_dialogue['count'].mean():.1f}")
    print(f"Median turns/dialogue: {turns_per_dialogue['count'].median():.1f}")

    split_counts = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Count": [n_train, n_val, n_test],
        "Color": ["#2E86AB", "#F4A261", "#E84855"]
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(split_counts["Split"], split_counts["Count"],
                  color=split_counts["Color"], edgecolor="white", linewidth=0.5, width=0.5)
    for bar, count in zip(bars, split_counts["Count"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(split_counts["Count"]) * 0.01,
                f"{count:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Number of Pairs", fontsize=12)
    ax.set_title("Figure 3: Sample Count per Split", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, max(split_counts["Count"]) * 1.12)
    plt.tight_layout()
    save_figure_to_s3(fig, "fig3_split_counts.png")

    gap_pdf = train_df.select("time_gap_seconds").limit(SAMPLE_SIZE).toPandas()
    gap_pdf["gap_minutes"] = gap_pdf["time_gap_seconds"] / 60
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(gap_pdf["gap_minutes"], bins=80, color="#8338EC", edgecolor="white", linewidth=0.3)
    ax.axvline(10, color="#E84855", linestyle="--", linewidth=1.5, label="10-min cutoff")
    ax.set_xlabel("Time Gap between Context and Response (minutes)", fontsize=12)
    ax.set_ylabel("Number of Pairs", fontsize=12)
    ax.set_title("Figure 4: Time Gap Distribution Between Context-Response Pairs", fontsize=14, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    save_figure_to_s3(fig, "fig4_time_gap.png")

    # ── Step 11 — Save Parquet splits to S3 ──────────────────────────────
    # Free intermediate frames before writing to avoid OOM on executors
    df.unpersist()
    df_clean.unpersist()
    df_pairs.unpersist()
    df_formatted.unpersist()
    df_deduped.unpersist()

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = f"{OUTPUT_DIR}/{split_name}"
        (
            split_df
            .repartition(8)        
            .write
            .mode("overwrite")
	        .option("compression", "none")
            .parquet(path)
        )
        print(f"Saved {split_name} -> {path}")

    # ── Step 12 — Verify ──────────────────────────────────────────────────
    print("=== Verification ===")
    expected = {"train": n_train, "val": n_val, "test": n_test}
    for split_name in ["train", "val", "test"]:
        path    = f"{OUTPUT_DIR}/{split_name}"
        df_read = spark.read.parquet(path)
        count   = df_read.count()
        match   = "OK" if count == expected[split_name] else "MISMATCH"
        print(f"{match} {split_name:<6}: {count:,} rows  (expected {expected[split_name]:,})")

    # ── Step 13 — Summary ─────────────────────────────────────────────────
    print("=" * 55)
    print("  PIPELINE SUMMARY")
    print("=" * 55)
    print(f"  Raw rows loaded          : {total_rows:>12,}")
    print(f"  After cleaning           : {after_words:>12,}")
    print(f"  Pairs built (before gap) : {pairs_before_gap:>12,}")
    print(f"  After gap filter (10min) : {after_gap:>12,}")
    print(f"  After length filter      : {after_len:>12,}")
    print(f"  After deduplication      : {after_dedup:>12,}")
    print(f"  Train                    : {n_train:>12,}")
    print(f"  Validation               : {n_val:>12,}")
    print(f"  Test                     : {n_test:>12,}")
    print("=" * 55)
    print(f"  Figures saved to         : s3://{S3_BUCKET}/figures/")
    print("=" * 55)

except Exception as e:
    import traceback
    print("\n" + "=" * 60)
    print("  PIPELINE FAILED — full traceback below")
    print("=" * 60)
    traceback.print_exc()
    print("=" * 60 + "\n")
    raise

finally:
    spark.stop()
    print("Spark session closed.")