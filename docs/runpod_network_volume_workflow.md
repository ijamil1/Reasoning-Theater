# RunPod Network Volume Workflow

This document describes how to use RunPod Network Volumes for persistent storage of Stage 2 `.pt` activation tensors across pod launches.

## Why Network Volumes

Container disk and pod volumes are wiped on pod termination. Network Volumes are independent infrastructure that persist until explicitly deleted — survives pod stop, restart, and termination. This makes them the right target for large activation files that take hours to generate.

## Constraints

- A Network Volume can only be attached to **one running pod at a time**.
- A Network Volume must be in the **same region** as the pod it is attached to.
- Because of the one-pod-at-a-time constraint, the five model runs must either:
  - Run **sequentially** on a single machine sharing one volume, or
  - Run **in parallel** on five separate machines, each with its own volume.

See [Parallel vs. sequential](#parallel-vs-sequential) below.

---

## Step-by-step setup

### 1. Create a Network Volume

In the RunPod console:

1. Go to **Storage → Network Volumes → New Network Volume**.
2. Choose a **region** — must match the region of the pods you will launch.
3. Set **size** — use the table in the README as a guide. 300 GB covers one model; 1.5 TB covers all five models sequentially on one volume.
4. Give it a name (e.g. `reasoning-theater-data`).

The volume begins billing at ~$0.07/GB/month from creation. Delete it when you no longer need the data.

### 2. Launch a pod with the volume attached

When creating a pod:

1. Select your GPU template (A100 80GB, B200, etc.).
2. Under **Volumes**, select your Network Volume. It will be mounted at `/runpod-volume/`.
3. Set **container disk** to the minimum (16–20 GB) — model weights and data will live on the Network Volume, not container disk.

### 3. Set up the environment on the Network Volume

```bash
# Verify the volume is mounted
df -h /runpod-volume

# Point all HuggingFace / temp caches to the volume
export HF_HOME=/runpod-volume/hf_cache
export HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache/hub
export TRANSFORMERS_CACHE=/runpod-volume/hf_cache/transformers
export TMPDIR=/runpod-volume/tmp

mkdir -p $HF_HOME $HUGGINGFACE_HUB_CACHE $TRANSFORMERS_CACHE $TMPDIR
```

Pointing HF caches to the volume means model weights are also persistent — subsequent pod launches for the same model skip re-downloading weights.

### 4. Clone the repo onto the Network Volume

```bash
cd /runpod-volume
git clone <repo-url>
cd Reasoning-Theater
uv venv && uv sync
```

Because the repo lives on the volume, all output paths (e.g. `data/deepseek_r1_llama_8b/arc-easy/stage2_activations/`) resolve to `/runpod-volume/Reasoning-Theater/data/...` — writes go to the Network Volume automatically. No config changes needed.

### 5. Login to HuggingFace

```bash
hf auth login
# Paste your token when prompted
```

### 6. Run data generation

```bash
# Example: DeepSeek-R1-Distill-Llama-8B
bash scripts/run_datagen_deepseek_r1_llama_8b.sh
```

Both stages run sequentially per dataset. Stage 2 `.pt` files land on the Network Volume.

### 7. Terminate the pod when done

Once the runner script completes, **terminate the pod** (stops GPU billing). The Network Volume and all data on it persist.

```bash
# Optional: verify data before terminating
du -sh /runpod-volume/Reasoning-Theater/data/
```

### 8. Re-attach for the next run or analysis

Launch a new pod in the same region, attach the same Network Volume. The repo, weights, and all previously generated data are immediately available at `/runpod-volume/Reasoning-Theater/`.

---

## Parallel vs. sequential

### Sequential (one volume, one machine at a time)

Simplest setup. Run all five models one after another on the same volume.

- 1 Network Volume (~1.5 TB to be safe)
- 1 pod at a time (different GPU config per model)
- Total wall-clock time: sum of all five runs

### Parallel (five volumes, five machines simultaneously)

Run all five models at the same time, one pod per model.

- 5 Network Volumes (~300 GB each, same region)
- 5 pods running simultaneously
- After all runs complete, consolidate data:

```bash
# On a CPU-only pod or local machine with all 5 volumes accessible:
# (Volumes cannot be attached to two pods simultaneously, so consolidate sequentially)
rsync -av /runpod-volume-8b/Reasoning-Theater/data/   /runpod-volume-main/Reasoning-Theater/data/
rsync -av /runpod-volume-14b/Reasoning-Theater/data/  /runpod-volume-main/Reasoning-Theater/data/
rsync -av /runpod-volume-32b/Reasoning-Theater/data/  /runpod-volume-main/Reasoning-Theater/data/
rsync -av /runpod-volume-70b/Reasoning-Theater/data/  /runpod-volume-main/Reasoning-Theater/data/
```

Or transfer each volume's `data/` to S3 and pull everything down to one place for analysis.

---

## Cost reference

| Resource | Rate | 1.5 TB volume for 30 days |
|---|---|---|
| Network Volume storage | ~$0.07/GB/month | ~$105/month |
| A100 80GB pod | ~$1.64/hr | pay only while running |

Delete the volume once data has been transferred to your local machine or S3.
