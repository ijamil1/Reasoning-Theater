# Cloud Archive Workflow

Upload the `data/` directory to cloud storage for long-term archival after all experiments are complete. This covers Stage 1 rollout JSONs and Stage 2 activation `.pt` tensors only — the analysis outputs in `results/` are not included.

---

## What gets uploaded

```
data/
  deepseek_r1_qwen_32b/
    mmlu/
      stage1_responses/       # ~5330 JSON files (~690 MB)
      stage2_activations/
        layer_63/             # one subdir per question hash, one .pt per file (~83 GB)
    arc_challenge/
      stage1_responses/       # ~1165 JSON files (~117 MB)
      stage2_activations/
        layer_63/             # (~18 GB)
    medqa/
      stage1_responses/       # 2000 JSON files (~320 MB)
      stage2_activations/
        layer_63/             # (~31 GB)
    gpqa/
      stage1_responses/       # 198 JSON files (~63 MB)
      stage2_activations/
        layer_63/             # (~3 GB)

  gpt_oss_120b/
    mmlu/
      stage1_responses/       # ~5330 JSON files (~690 MB)
      stage2_activations/
        layer_35/             # (~67 GB)
    arc_challenge/
      stage1_responses/       # ~1165 JSON files (~117 MB)
      stage2_activations/
        layer_35/             # (~15 GB)
    medqa/
      stage1_responses/       # 2000 JSON files (~320 MB)
      stage2_activations/
        layer_35/             # (~25 GB)
    gpqa/
      stage1_responses/       # 198 JSON files (~63 MB)
      stage2_activations/
        layer_35/             # (~2.5 GB)
```

**Estimated total: ~248 GB** (~2.4 GB Stage 1 + ~245 GB Stage 2)

---

## Storage provider

Use **AWS S3** with the **S3 Glacier Instant Retrieval** storage class.

- Cost: ~$0.004/GB/month — for 250 GB that is ~$1/month
- Retrieval: available within milliseconds if ever needed
- Alternative: **Backblaze B2** at $0.006/GB/month with `rclone` if you want to avoid AWS

> Standard S3 (`s3:// ` without a storage class flag) costs ~$0.023/GB/month (~$5.75/month for 250 GB). Glacier Instant Retrieval is 6× cheaper for data you do not plan to read.

---

## Setup

### 1. Install the AWS CLI on the pod

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

### 2. Configure credentials

```bash
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region:        us-east-1   (or whichever region your bucket is in)
# Default output format: json
```

### 3. Create a bucket (one-time, from your local machine or the AWS console)

```bash
aws s3 mb s3://reasoning-theater-data --region us-east-1
```

---

## Upload

Run this from the repo root on the RunPod network volume pod after all data generation is complete for both models:

```bash
aws s3 sync /runpod-volume/Reasoning-Theater/data/ \
    s3://reasoning-theater-data/data/ \
    --storage-class GLACIER_IR \
    --no-progress \
    --exclude "*.tmp"
```

`s3 sync` is resumable — if the pod goes down mid-upload, re-run the same command and it will skip already-uploaded files.

Expected upload time at ~300 MB/s (RunPod datacenter egress): roughly 15–30 minutes for 250 GB.

---

## Verify

After the sync completes, confirm the object count and approximate size match what is on disk:

```bash
# Count objects in S3
aws s3 ls s3://reasoning-theater-data/data/ --recursive --summarize | tail -2

# Count files on disk
find /runpod-volume/Reasoning-Theater/data/ -type f | wc -l

# Spot check: confirm one activation file is present per model
aws s3 ls s3://reasoning-theater-data/data/deepseek_r1_qwen_32b/mmlu/stage2_activations/layer_63/ | head -5
aws s3 ls s3://reasoning-theater-data/data/gpt_oss_120b/mmlu/stage2_activations/layer_35/ | head -5
```

The object counts should match. If they diverge, re-run `s3 sync` — it is idempotent.

---

## Delete the network volume

Only do this after the verification step above passes.

1. In the RunPod console, terminate the pod.
2. Go to **Storage → Network Volumes**, select `reasoning-theater-data`, and delete it.

The volume is billed from the moment it is created, so delete it promptly once the upload is confirmed.

---

## Retrieval (if ever needed)

```bash
aws s3 sync s3://reasoning-theater-data/data/ ./data/ --no-progress
```

Objects in GLACIER_IR are immediately accessible — no restore step required.

To delete the S3 bucket entirely when the project is over:

```bash
aws s3 rm s3://reasoning-theater-data --recursive
aws s3 rb s3://reasoning-theater-data
```