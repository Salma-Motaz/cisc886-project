import os
import time
import pandas as pd

root        = r"C:\Users\DPQUAI250118\Downloads\version 2\dialogs"
output_file = r"C:\Users\DPQUAI250118\Downloads\ubuntu_raw.parquet"

print("Starting conversion...")
start      = time.time()
all_records = []
files_count = 0
total_lines = 0

for folder, _, files in os.walk(root):
    for file in files:
        files_count += 1
        path        = os.path.join(folder, file)
        dialogue_id = os.path.relpath(path, root).replace("\\", "/")

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    record = {
                        "dialogue_id": dialogue_id,
                        "timestamp"  : parts[0] if len(parts) > 0 else "",
                        "speaker"    : parts[1] if len(parts) > 1 else "",
                        "addressee"  : parts[2] if len(parts) > 2 else "",
                        "message"    : "\t".join(parts[3:]) if len(parts) > 3 else ""
                    }
                    all_records.append(record)
                    total_lines += 1

        except Exception as e:
            print(f"Error: {path}: {e}")

        # Progress every 10,000 files
        if files_count % 10000 == 0:
            elapsed = round((time.time() - start) / 60, 2)
            print(f"Files: {files_count:,} | Lines: {total_lines:,} | Minutes: {elapsed}")

# ── Convert to DataFrame and save as Parquet ──────────────────────────────
print("\nConverting to DataFrame...")
df = pd.DataFrame(all_records)

print(f"Total rows: {len(df):,}")
print(f"Columns   : {df.columns.tolist()}")
print(f"Sample:\n{df.head(3)}")

print("\nSaving to Parquet...")
df.to_parquet(output_file, index=False, engine="pyarrow")

elapsed = round((time.time() - start) / 60, 2)
print(f"\n✓ Done in {elapsed} minutes")
print(f"✓ Saved to: {output_file}")
print(f"✓ File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")