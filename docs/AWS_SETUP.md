# AWS Setup Guide — Hyperparams & Geometry Cloud Runs

Instances are launched manually via the AWS Console (no CLI required).
Each instance clones the repo, installs dependencies, runs one task, uploads
`bo_state.json` to S3 after every iteration, and self-terminates.

S3 bucket: `tom-hyperparams-representations`
Region: `eu-west-3` (Paris)

---

## One-time setup (already done)

### S3 bucket
`tom-hyperparams-representations` — stores `bo_state.json` per task.

### IAM role: `ec2-s3-access`
Attached policies:
- `AmazonS3FullAccess`
- Inline policy `ec2-self-terminate`: allows `ec2:TerminateInstances`

### Security group
Allows SSH (port 22) from anywhere. Used for debugging.

### Key pair: `hyperparams-key`
Private key at `secrets/hyperparams-key.pem` (gitignored).

### Launch template: `gp-test-spirals`
Created from a past instance. Use as a starting point, updating User Data per task.

---

## Running the smoke test

Use `scripts/aws_smoke_test_userdata.sh` as User Data verbatim. It clones `main`,
installs deps, runs all 9 tasks with small budgets, saves activations to S3, then
self-terminates.

**Before launching:** ensure S3 is clean (no existing task directories).
**After the smoke test:** delete all S3 content again before the real run.

---

## Launching a run

### 1. Upload existing state to S3 (if resuming)

Go to **S3 → tom-hyperparams-representations**, create a folder named after the
task (e.g. `mnist_dual/`), and upload the local `bo_state.json` into it.

### 2. Launch instance from EC2 console

**EC2 → Launch Instance** (or use the launch template as a starting point):

| Setting | Value |
|---|---|
| AMI | Ubuntu 24.04 LTS (x86_64) |
| Instance type | `m7i-flex.large` or similar CPU instance |
| Key pair | `hyperparams-key` |
| Security group | existing SSH group |
| IAM instance profile | `ec2-s3-access` |
| Storage | **20 GiB** (default 8 GiB is too small for PyTorch) |
| User Data | contents of `scripts/aws_startup_gp_test.sh` with `TASK_NAME` set |

### 3. Edit User Data per task

Open `scripts/aws_startup_gp_test.sh` and change the `TASK_NAME` line at the top:
```bash
TASK_NAME="mnist_dual"   # ← one of: spirals, mnist_dual, mnist_10way, etc.
```
Paste the full script into the **Advanced → User Data** field.

---

## Monitoring

**SSH in for live logs:**
```bash
ssh -i secrets/hyperparams-key.pem ubuntu@<public-ip>
tail -f /var/log/cloud-init-output.log
```

**Check S3 for progress:**
Go to **S3 → tom-hyperparams-representations → <task> → bo_state.json → Download**,
then count observations:
```bash
python3 -c "import json; d=json.load(open('bo_state.json')); print(len(d), 'obs')"
```

---

## Downloading results

Download `bo_state.json` per task from the S3 console into
`output/experiments/<task>/`. Network weights are not uploaded to S3.

---

## After the run

Instances self-terminate on completion. Verify in **EC2 → Instances** that all
show as **terminated**. If any show `running` or `stopped`, terminate manually.

**Billing alert** is set at $50 — check **Billing → Budgets** if uncertain.

---

## Cost reference

| Instance type | vCPU | RAM | Approx spot price |
|---|---|---|---|
| `m7i-flex.large` | 2 | 8 GB | ~$0.03/hr |
| `c5.xlarge` | 4 | 8 GB | ~$0.04/hr |

S3 storage and transfer: negligible.
