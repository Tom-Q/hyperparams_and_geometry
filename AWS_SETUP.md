# AWS Setup Guide — Hyperparams & Geometry Cloud Runs

This document covers setting up AWS to run the 9-task Bayesian optimisation pipeline
in parallel on cloud machines. Expected cost: ~$20 for a full 9-task run using spot
instances. Expected wall-clock time: ~13 hours (parallel).

---

## Step 0 — Sign up

1. Go to [aws.amazon.com](https://aws.amazon.com) and create an account.
2. Give a credit card. You will be billed monthly for what you use.
3. During sign-up, choose the **free tier** option. It covers small usage for 12 months
   but will not cover a 9-machine parallel run — expect real charges.
4. Once logged in, you land in the **AWS Console** (the web interface). Most of the
   one-time setup below can be done here.

**Region**: in the top-right corner of the console, select a region close to you
(e.g. `us-east-1` or `eu-west-1`). Stick to the same region for everything —
S3 bucket, EC2 instances, everything. Cross-region transfers add cost and complexity.

---

## Step 1 — Set a billing alert (do this first)

This is the closest AWS gets to a hard spending limit. It won't stop your instances,
but it will email you if spending looks wrong.

1. In the console, search for **Billing and Cost Management**.
2. Go to **Budgets → Create budget**.
3. Choose "Cost budget", set the amount to **$50** (well above expected ~$20 cost).
4. Set alert at 80% of budget ($40) and enter your email.
5. Create the budget.

Also enable **Cost Anomaly Detection** (same section) — it alerts you when spending
looks unusual relative to your history. Free to enable.

After every run: check the EC2 console and confirm all instances show as **terminated**.
That 30-second check is your main protection against runaway spending.

---

## Step 2 — Install and configure the AWS CLI locally

The CLI (Command Line Interface) is a program that lets you control AWS from your
terminal instead of clicking through the website. You need it to launch instances
with a single command rather than clicking through forms 9 times.

**Install:**
```bash
pip install awscli
```

Or download the installer from the AWS website if you prefer.

**Verify:**
```bash
aws --version
```

**Authenticate:**

First, create access keys so the CLI knows who you are:
1. In the console, click your name (top-right) → **Security credentials**.
2. Under "Access keys", click **Create access key**.
3. Download or copy the **Access Key ID** and **Secret Access Key**.
   Store these somewhere safe — the secret is only shown once.

Then configure the CLI:
```bash
aws configure
```
It will ask for:
- Access Key ID: paste it
- Secret Access Key: paste it
- Default region: enter the same region you chose in Step 0 (e.g. `us-east-1`)
- Default output format: `json`

Test that it works:
```bash
aws s3 ls
```
Should return nothing (empty list) without an error.

---

## Step 3 — One-time AWS resource setup

These are created once and reused for every run.

### 3a. S3 bucket

S3 is AWS's file storage service. We use it to store `bo_state.json` continuously
during training — so if a spot instance is terminated mid-run, no progress is lost.

```bash
# Replace "thomas-hyperparams" with any globally unique name (lowercase, no spaces)
aws s3 mb s3://thomas-hyperparams --region us-east-1
```

Verify:
```bash
aws s3 ls
# Should show your bucket name
```

### 3b. IAM role (permission for EC2 to write to S3)

EC2 instances don't automatically have permission to touch S3. We create a role
that grants this permission and attach it to instances at launch.

This is easiest to do in the console:
1. Search for **IAM** → **Roles** → **Create role**.
2. Trusted entity type: **AWS service** → use case: **EC2**.
3. Attach permission policy: search for and select **AmazonS3FullAccess**.
   (In production you'd scope this down; for a personal research project this is fine.)
4. Name the role `ec2-s3-access` and create it.

### 3c. Security group

A security group is a firewall. We need to allow SSH connections (port 22) so we
can log into instances for debugging if needed.

```bash
aws ec2 create-security-group \
    --group-name hyperparams-sg \
    --description "SSH access for hyperparams BO runs"

# Note the GroupId that's returned (e.g. sg-0abc123...), you'll need it at launch.

# Allow SSH from anywhere (0.0.0.0/0)
# Replace sg-XXXXXXXX with your GroupId
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXXXXX \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0
```

### 3d. Key pair (for SSH access)

A key pair lets you SSH into instances if you need to debug.

```bash
aws ec2 create-key-pair \
    --key-name hyperparams-key \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/hyperparams-key.pem

chmod 400 ~/.ssh/hyperparams-key.pem
```

---

## Step 4 — Find the right AMI (machine image)

An AMI is the operating system snapshot that new instances boot from. We want a
standard Ubuntu 22.04 image. AMI IDs are region-specific.

```bash
aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text
```

Note the returned ID (e.g. `ami-0abcdef1234567890`). Use it in the launch step.

---

## Step 5 — Write the startup script

This script runs automatically when an instance boots. It installs Python dependencies,
clones the repo, and starts the training run. The instance shuts itself down when done.

Create a file `aws_startup.sh` in the repo root:

```bash
#!/bin/bash
set -e

# --- Configuration (set at launch time via environment or sed substitution) ---
TASK_NAME="__TASK__"          # replaced at launch time
S3_BUCKET="thomas-hyperparams"
REPO_URL="https://github.com/Tom-Q/hyperparams_and_geometry.git"
N_ITER=400
BETA=8.0

# --- System setup ---
apt-get update -q
apt-get install -y -q python3-pip python3-venv git

# --- Clone repo ---
cd /home/ubuntu
git clone "$REPO_URL" project
cd project

# --- Python environment ---
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet -r requirements.txt

# --- Sync any existing state from S3 (allows resuming interrupted runs) ---
aws s3 sync "s3://$S3_BUCKET/$TASK_NAME/" "experiments/$TASK_NAME/" || true

# --- Run ---
python run_bo.py \
    --task "$TASK_NAME" \
    --n-iter "$N_ITER" \
    --output-dir experiments \
    --beta "$BETA"

# --- Upload final results ---
aws s3 sync "experiments/$TASK_NAME/" "s3://$S3_BUCKET/$TASK_NAME/"

# --- Terminate this instance ---
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region us-east-1

shutdown -h now
```

Note: `run_bo.py` already calls `save_state` after every iteration. We need to add
an S3 upload there so state is continuously synced (see Step 6 below).

---

## Step 6 — Add continuous S3 sync to save_state

The startup script syncs at start and end, but we want continuous sync in case
of spot termination mid-run. Add this to `src/bo.py`'s `save_state` function:

```python
def save_state(path, observations, s3_bucket=None, task_name=None):
    def _default(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        raise TypeError(type(obj))
    with open(path, "w") as f:
        json.dump(observations, f, indent=2, default=_default)
    if s3_bucket and task_name:
        import boto3
        boto3.client("s3").upload_file(str(path), s3_bucket, f"{task_name}/bo_state.json")
```

And pass `s3_bucket` and `task_name` through from `run_bo.py`
(read from environment variables so the script stays general):

```python
# In run_bo.py, near the top of main():
import os
s3_bucket = os.environ.get("S3_BUCKET")   # None if not set → local run unchanged
task_name  = args.task
```

This means local runs are completely unaffected (no boto3 needed locally).
Cloud runs set the `S3_BUCKET` environment variable and get automatic sync.

---

## Step 7 — Launch the runs

Replace the placeholder values (AMI ID, security group ID, IAM role name) with
your actual values from the setup steps above.

```bash
#!/bin/bash
# launch_all_tasks.sh

AMI_ID="ami-XXXXXXXXXXXXXXX"      # from Step 4
SECURITY_GROUP="sg-XXXXXXXX"      # from Step 3c
IAM_ROLE="ec2-s3-access"          # from Step 3b
KEY_NAME="hyperparams-key"        # from Step 3d
INSTANCE_TYPE="c5.xlarge"         # 4 vCPU, 8GB RAM, ~$0.04/hr spot
S3_BUCKET="thomas-hyperparams"

TASKS="spirals parity mnist_10way mnist_dual fashion_10way mnist_rnn adding cartpole fourrooms"

for TASK in $TASKS; do
    # Inject the task name into the startup script
    USERDATA=$(sed "s/__TASK__/$TASK/" aws_startup.sh | base64 -w 0)

    aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --iam-instance-profile "Name=$IAM_ROLE" \
        --instance-market-options '{"MarketType":"spot"}' \
        --user-data "$USERDATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=task,Value=$TASK}]" \
        --count 1 \
        --query 'Instances[0].InstanceId' \
        --output text

    echo "Launched instance for task: $TASK"
done
```

Run it:
```bash
bash launch_all_tasks.sh
```

---

## Step 8 — Monitor and retrieve results

**Check instance status:**
```bash
aws ec2 describe-instances \
    --filters "Name=tag-key,Values=task" "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[].Instances[].[Tags[?Key==`task`].Value|[0],State.Name,LaunchTime]' \
    --output table
```

**Check S3 for progress** (how many iterations done per task):
```bash
for TASK in spirals parity mnist_10way mnist_dual fashion_10way mnist_rnn adding cartpole fourrooms; do
    COUNT=$(aws s3 cp "s3://thomas-hyperparams/$TASK/bo_state.json" - 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d))" 2>/dev/null || echo "no data yet")
    echo "$TASK: $COUNT iterations"
done
```

**Download all results when done:**
```bash
aws s3 sync s3://thomas-hyperparams/ experiments/
```

**SSH into an instance for debugging** (get the IP from the console or CLI):
```bash
ssh -i ~/.ssh/hyperparams-key.pem ubuntu@<instance-public-ip>
```

---

## Step 9 — After the run: verify termination

Instances should self-terminate. Verify:
```bash
aws ec2 describe-instances \
    --filters "Name=tag-key,Values=task" \
    --query 'Reservations[].Instances[].[Tags[?Key==`task`].Value|[0],State.Name]' \
    --output table
```

All should show `terminated`. If any show `running` or `stopped`, terminate manually:
```bash
aws ec2 terminate-instances --instance-ids i-XXXXXXXXXXXXXXXXX
```

---

## Cost reference

| Instance type | vCPU | RAM   | Spot price (approx) |
|---------------|------|-------|----------------------|
| c5.large      | 2    | 4 GB  | ~$0.02/hr            |
| c5.xlarge     | 4    | 8 GB  | ~$0.04/hr            |
| c5.2xlarge    | 8    | 16 GB | ~$0.08/hr            |

Spot prices vary by region and time. Check current prices:
```bash
aws ec2 describe-spot-price-history \
    --instance-types c5.xlarge \
    --product-descriptions "Linux/UNIX" \
    --max-items 5
```

**Expected total for a full 9-task run on c5.xlarge spot:**
~13 hours × 9 instances × $0.04/hr ≈ **$5–8**

S3 storage and transfer: negligible (a few cents).

---

## What needs to be done before the first cloud run

- [ ] Step 0: Create AWS account
- [ ] Step 1: Set billing alert at $50
- [ ] Step 2: Install and configure AWS CLI
- [ ] Step 3: Create S3 bucket, IAM role, security group, key pair
- [ ] Step 4: Find AMI ID for your region
- [ ] Step 5–6: Finalise `aws_startup.sh` and add S3 sync to `save_state`
- [ ] Step 7: Test with a single task and 10 iterations before launching all 9
- [ ] Step 8: Full launch
