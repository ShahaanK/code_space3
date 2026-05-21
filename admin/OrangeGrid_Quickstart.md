# OrangeGrid Quickstart Guide

Reference for connecting to, navigating, and submitting jobs on Syracuse University's OrangeGrid HTCondor cluster for the CAMEL project. Captures cluster-specific quirks and gotchas discovered during the 5/7–5/19 setup sequence.

---

## Cluster Identity

- **Login node:** `its-og-login5.syr.edu`
- **Username:** `szkhan`
- **Access:** SU RDS jumphost required (cluster not directly reachable from external networks)
- **Home directory:** `/home/szkhan` on NFS share `10.5.0.207`
- **Quota:** 3.6 TB soft / 4 TB hard

## Connection

```bash
ssh szkhan@its-og-login5.syr.edu
```

Goes through SU RDS. Long-running operations on the login node must be wrapped in `tmux` or `nohup` because RDS sessions die on laptop disconnect.

---

## Environment

### Conda

Miniforge 24.7.1 installed at `/home/szkhan/miniconda3`. Auto-activates `base` on login via the `.bash_profile` shim that sources `.bashrc`.

```bash
conda activate camel    # CAMEL project env
conda activate base     # default
conda deactivate
```

### camel env contents

Python 3.11 with: pandas, pyarrow, pyyaml, huggingface_hub, openai, requests. No vLLM or torch in conda; those live inside the Singularity container.

### Adding packages

```bash
conda activate camel
pip install <package>
```

Default to pip over `conda install` for non-binary packages.

---

## vLLM via Singularity

### SIF location

`/home/szkhan/vllm-openai.sif` (11GB). Built from upstream `vllm/vllm-openai:v0.10.0` Docker image, Singularity 3.7.1, December 9 2025. Provided by Prof. Introne; original on Apophis at `/DATA/vllm-openai.sif` as backup.

### Version inside SIF

```
vLLM 0.10.1.dev1+gbcc0a3cbe (commit bcc0a3c)
```

Not pristine 0.10.0. Cite as `vLLM 0.10.1.dev (commit bcc0a3c)` in any methods writing. Apophis runs `0.16.0`; the version delta across clusters is intentional and forced by OrangeGrid's CUDA 12.8 driver (vLLM 0.10.x is the highest version that supports pre-12.9 CUDA).

### Quick version check

```bash
singularity exec /home/szkhan/vllm-openai.sif python3 -c "import vllm; print(vllm.__version__)"
```

### CRITICAL: use `python3`: not `python`

The vLLM Docker base image ships `python3` but no `python` symlink. Any container invocation referencing `python` will fail with `FATAL: "python": executable file not found in $PATH`. Always use `python3` inside `singularity exec`.

### `--nv` flag behavior

The `--nv` flag mounts host NVIDIA driver libraries into the container. On the login node this prints `INFO: Could not find any nv files on this host!` which is harmless because there's no GPU there. On compute nodes the flag is required for GPU access. Rule of thumb: include `--nv` for any actual inference; drop it for sanity checks on the login node.

### HuggingFace cache bind

For inference workloads, bind the host HF cache into the container:

```bash
HF_CACHE=$HOME/.cache/huggingface
singularity exec --nv --bind $HF_CACHE:$HF_CACHE /home/szkhan/vllm-openai.sif \
    python3 -m vllm.entrypoints.openai.api_server --model <model_id> ...
```

This pattern was shared by Prof. Introne in the 5/14 meeting from his fraction code.

---

## Model Weights

Pre-cached on the login node via `huggingface_hub.snapshot_download`. Default cache at `~/.cache/huggingface`. Download from the login node (it has reliable outbound) rather than compute nodes for speed and to avoid re-downloads.

Run long downloads inside tmux:

```bash
tmux new -s hf-download
conda activate camel
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='casperhansen/llama-3.3-70b-instruct-awq', cache_dir='$HOME/.cache/huggingface')
"
# Ctrl-b d to detach; tmux a -t hf-download to reattach
```

CAMEL models to cache:
- `casperhansen/llama-3.3-70b-instruct-awq` (~37 GB)
- `Qwen/Qwen2.5-72B-Instruct-AWQ` (~38 GB)
- `casperhansen/deepseek-r1-distill-qwen-32b-awq` (~17 GB)

Total ~92 GB, comfortable within the 4 TB quota.

---

## HTCondor Job Submission

### Submit a job

```bash
condor_submit my_job.sub
```

### Required ClassAds for CAMEL annotation jobs

```
universe          = vanilla
executable        = wrapper.sh
output            = logs/job_$(Cluster).$(Process).out
error             = logs/job_$(Cluster).$(Process).err
log               = logs/job_$(Cluster).$(Process).log
request_cpus      = 8
request_memory    = 64 GB
request_gpus      = 1
requirements      = (CUDADeviceName == "NVIDIA A100 80GB PCIe")
queue
```

Notes:
- **Do NOT set `request_disk`.** OrangeGrid has `JOB_DEFAULT_REQUESTDISK = DiskUsage` so explicit disk requests are not needed.
- **Advertise GPU memory accurately.** Under-advertising and over-consuming will clobber concurrent jobs on the same GPU. The 70B AWQ uses ~37 GB weights plus KV cache, fitting comfortably under 80 GB.
- **No PREEMPT.** `PREEMPT = False` on OrangeGrid means jobs are not evicted by competing higher-priority work. Long-running jobs run to completion.
- **No wall-time limit.** Confirmed in 5/14 meeting: jobs run as needed; the queue absorbs the rest.

### Wrapper script pattern

HTCondor jobs do not inherit your shell environment. Wrappers must re-init conda before running code. Template:

```bash
#!/bin/bash
eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash hook)"
conda activate camel

HF_CACHE=$HOME/.cache/huggingface
singularity exec --nv --bind $HF_CACHE:$HF_CACHE /home/szkhan/vllm-openai.sif \
    python3 my_annotation_script.py
```

The CAMEL batch scripts will follow this pattern, layered with vLLM server invocation.

---

## Monitoring

### Live queue status

```bash
condor_q                          # your jobs only
condor_q -nobatch                 # per-job detail
condor_q -better-analyze <job_id> # explain why job is idle
watch -n 5 condor_q               # auto-refresh
```

### Completed jobs

```bash
condor_history                    # recent completions
condor_history <cluster_id>       # specific job
```

### Hold and release

If a job goes into hold state (`H`), inspect the reason:

```bash
condor_q -hold
condor_q -analyze <job_id>
```

After fixing whatever is wrong:

```bash
condor_release <job_id>
```

---

## GPU Pool Inventory

Pool as of 5/11/2026 (after ITS expanded A100 visibility):

- **9× A100 80GB PCIe nodes**: 2 GPUs each (18 GPUs total). Target for 70B/72B models (TP=1).
- **7× A40 nodes**: 4 GPUs each (28 GPUs total). 45 GB usable VRAM. Usable for 32B/24B at TP=1.
- **11× L40S nodes**: 2 GPUs each (22 GPUs total). 45 GB usable VRAM. Usable for 32B/24B at TP=1.

Total effective: 27 nodes, 68 GPUs, all CUDA 12+ vLLM-compatible.

**Do not target Quadro RTX 6000 nodes.** They run CUDA driver 11.6, incompatible with vLLM 0.10+ which requires CUDA 12.

### Targeting specific GPUs

For 70B/72B models (require A100 80GB):

```
requirements = (CUDADeviceName == "NVIDIA A100 80GB PCIe")
```

For 32B/24B models (any modern GPU):

```
requirements = (CUDADeviceName == "NVIDIA A100 80GB PCIe" || \
                CUDADeviceName == "NVIDIA L40S" || \
                CUDADeviceName == "NVIDIA A40")
```

### Important ClassAd naming quirk

OrangeGrid runs HTCondor 9.8.0, which uses `CUDA*` prefix attributes (`CUDADeviceName`, `CUDAGlobalMemoryMb`, `CUDAClockMhz`, `CUDADriverVersion`). Newer Condor 10.x uses `GPUs_*` prefix. Queries written against the newer schema will return empty.

---

## Software Inventory

- **OS:** Ubuntu 20.04.4 LTS
- **HTCondor:** 9.8.0 (April 2022 build)
- **Singularity:** 3.7.1
- **Apptainer:** 1.1.3
- **CUDA driver:** 12.8 (caps vLLM at 0.10.x)
- **No Lmod or Environment Modules.** Batch scripts must be self-contained; nothing to `module load`.

---

## Filesystem

| Path                           | Purpose                                     |
|--------------------------------|---------------------------------------------|
| `/home/szkhan/`                | User home; 4TB NFS                          |
| `/home/szkhan/vllm-openai.sif` | vLLM Singularity container                  |
| `/home/szkhan/miniconda3/`     | Miniforge install                           |
| `~/.cache/huggingface/`        | HF model weight cache                       |
| `/var/lib/condor/execute/`     | Compute-node scratch (ephemeral per job)    |

Compute-node scratch is wiped between jobs. Anything that needs to persist must be written back to `/home/szkhan`. Compute nodes can pull from external resources (HuggingFace, GitHub) but cannot push outbound (per Prof. Introne, 5/14).

---

## Common Pitfalls

1. **`python` not in container PATH**: use `python3` for all SIF invocations.
2. **`--nv` warning on login node**: expected, harmless. No GPUs on login.
3. **Login shells read `.bash_profile`, not `.bashrc`**: conda activation requires the shim in `.bash_profile` that sources `.bashrc`.
4. **GPU memory must be advertised accurately**: over-consuming clobbers concurrent jobs.
5. **RDS sessions die on laptop disconnect**: wrap any long-running login-node work in `tmux` or `nohup`.
6. **HTCondor 9.8.0 uses `CUDA*` ClassAds, not `GPUs_*`**, queries written for Condor 10.x will return empty.
7. **Compute nodes can pull but not push**: design output flow to write back to `/home/szkhan` rather than push elsewhere from compute.
8. **No `request_disk` needed**: `JOB_DEFAULT_REQUESTDISK = DiskUsage` handles it.

---

## Cluster Coordination Protocol (per 5/14 meeting with Prof. Introne)

Prof. Introne runs his own work on OrangeGrid. To avoid stepping on his jobs:

- Submit CAMEL production runs in **subsets** (not all 36 jobs at once).
- After the first chunk completes, send Prof. Introne a timing estimate so he can sequence his CPU clustering and GPU jobs around the CAMEL load.
- If he needs the GPUs and CAMEL jobs are queued, hold the queued jobs (`condor_hold`) so his work gets prioritized.

---

## Reference URLs

- OrangeGrid examples repo: https://github.com/SyracuseUniversity/OrangeGridExamples
- ITS HTCondor docs: https://answers.syr.edu (search "HTCondor")
- Research Computing contact: researchcomputing@syr.edu

---

## Version History

- **2026-05-19**: Initial draft. Day 0 setup complete on OrangeGrid (SIF, Miniforge, camel env, vLLM verification).
