# AIRS MoE Remote Execution Scaffold

This workspace is a control scaffold for MoE inference experiments where code is
managed through GitHub and GPU execution happens on a remote Linux server.

The intended loop is:

1. Edit and commit code locally.
2. Push the selected branch to GitHub.
3. Ask the remote server to fetch, checkout, and pull that branch.
4. Run the experiment on the remote GPU server.
5. Download the run directory back to the local results folder.
6. Analyze locally and iterate.

## Layout

```text
configs/
  remote.example.json          Example local/remote experiment config.
scripts/
  local_push_run_fetch.ps1     Windows-side orchestrator.
  remote_pull_run.sh           Linux-side runner uploaded by the orchestrator.
analysis/
  summarize_runs.py            Local summary helper for downloaded runs.
results/
  .gitkeep                     Local output placeholder; real results are ignored.
reports/
  .gitkeep                     Keep lightweight Markdown reports here.
```

## First-Time Setup

If this scaffold was generated under a Codex scratch directory, migrate it to
the durable project path first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/migrate_to_d_moe_research.ps1 `
  -InitGit `
  -Push
```

Generate the private config from the SSH command and GitHub repo URL:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_from_info.ps1 `
  -SshCommand 'ssh -p 50473 root@121.48.170.6' `
  -RepoUrl 'https://github.com/Yuelinfeng/moe_research.git' `
  -InitGit `
  -Push
```

The password is intentionally not stored in any config file. Without an SSH key,
Windows OpenSSH will prompt for the remote password when it runs `ssh` or `scp`.

Recommended one-time key setup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_ssh_key.ps1 `
  -SshCommand 'ssh -p 50473 root@121.48.170.6'
```

Expected SSH access shape:

```powershell
ssh -p 50473 root@121.48.170.6
```

The remote repository is cloned automatically if `remote.repoPath` does not
exist yet.

To edit the generated config later:

```powershell
notepad configs/remote.local.json
```

## Run One Experiment

From this scaffold directory:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/local_push_run_fetch.ps1 `
  -Config configs/remote.local.json
```

The default experiment is a synthetic Tiny-MoE GPU baseline with real router,
top-k dispatch, and expert MLP compute. It records environment checks,
`nvidia-smi`, latency, token throughput, peak CUDA memory, and route counts.
This keeps the first remote experiment independent from Hugging Face network
availability. Set `BASELINE_MODE=hf` when a real pretrained MoE checkpoint is
available locally or through a working mirror.

Useful overrides:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/local_push_run_fetch.ps1 `
  -Config configs/remote.local.json `
  -RunLabel qwen_moe_baseline `
  -RunCommand 'BASELINE_MODE=hf MODEL_ID=Qwen/Qwen1.5-MoE-A2.7B-Chat MAX_NEW_TOKENS=32 bash experiments/run_baseline.sh --output-dir "$RUN_DIR"'
```

If the target local repo has uncommitted changes and you want the script to
commit them before pushing:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/local_push_run_fetch.ps1 `
  -Config configs/remote.local.json `
  -CommitMessage "run utility-aware prefetch experiment"
```

Override the run command without editing JSON:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/local_push_run_fetch.ps1 `
  -Config configs/remote.local.json `
  -RunLabel utility_gate `
  -RunCommand 'bash experiments/run_utility_gate.sh --output-dir "$RUN_DIR"'
```

## Analyze Downloaded Runs

```powershell
python analysis/summarize_runs.py --results-root D:\MoE-results --out reports\runs.csv
```

## Contract

The remote command can rely on these environment variables:

```bash
RUN_ID
RUN_DIR
COMMIT
SHORT_COMMIT
REPO_PATH
```

Every run directory contains:

```text
commit.txt
exit_code.txt
git_status_before.txt
manifest.env
manifest.json
run_command.sh
stdout.log
stderr.log
```

Keep large model files, datasets, traces, and profiles out of Git. Commit code,
configs, small summaries, and Markdown reports.
