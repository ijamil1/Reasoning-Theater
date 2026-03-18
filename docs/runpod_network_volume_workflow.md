# RunPod Network Volume Workflow

This document describes how to use RunPod Network Volumes for persistent storage of Stage 2 `.pt` activation tensors across pod launches.

## Why Network Volumes

Container disk and pod volumes are wiped on pod termination. Network Volumes are independent infrastructure that persist until explicitly deleted — survives pod stop, restart, and termination. This makes them the right target for large activation files that take hours to generate.

## Constraints

- A Network Volume can only be attached to **one running pod at a time**.
- A Network Volume must be in the **same region** as the pod it is attached to.
- Because of the one-pod-at-a-time constraint, the 2 model runs must:
  - Run **sequentially** on a single machine sharing one volume, or

---

## Step-by-step setup

### 1. Create a Network Volume

In the RunPod console:

1. Go to **Storage → Network Volumes → New Network Volume**.
2. Choose a **region** — must match the region of the pods you will launch.
3. Set **size** — use the README as a guide.
4. Give it a name (e.g. `reasoning-theater-data`).

The volume begins billing at ~$0.07/GB/month from creation. Delete it when you no longer need the data.

### 2. Launch a pod with the volume attached

When creating a pod:

1. Select your GPU template (A100 80GB, B200, etc.).
2. Under **Volumes**, select your Network Volume. It will be mounted at `/workspace`.
 Model data will live on the Network Volume as will model weights.

### 3. Set up the environment on the Network Volume

```bash
# Verify the volume is mounted
df -h /workspace
```

### 4. Clone the repo onto the Network Volume

```bash
cd /workspace
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
du -sh /workspace/Reasoning-Theater/data/
```

### 8. Re-attach for the next run or analysis

Launch a new pod in the same region, attach the same Network Volume. The repo, weights, and all previously generated data are immediately available at `/workspace/Reasoning-Theater/`.

---

## Sequential (one volume, one machine at a time)

Simplest setup. Run all five models one after another on the same volume.

- 1 Network Volume (~1.5 TB to be safe)
- 1 pod at a time (different GPU config per model)
- Total wall-clock time: sum of all five runs

---

## Cost reference

| Resource | Rate | 1.5 TB volume for 30 days |
|---|---|---|
| Network Volume storage | ~$0.07/GB/month | ~$105/month |
| A100 80GB pod | ~$1.64/hr | pay only while running |

Delete the volume once data has been transferred to your local machine or S3.
