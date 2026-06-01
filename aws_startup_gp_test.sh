#!/bin/bash
set -e

# Single-task GP test run — spirals, starting from scratch (Sobol then GP).
# Paste this as EC2 User Data verbatim.
#
# NOTE: spirals generates all data procedurally — no downloads needed.
# For MNIST/Fashion tasks, torchvision downloads ~200MB on first run.
# For RNN/RL tasks, check tasks/<name>.py for data requirements before running.

TASK_NAME="spirals"          # ← change this per instance
S3_BUCKET="tom-hyperparams-representations"
REPO_URL="https://github.com/Tom-Q/hyperparams_and_geometry.git"
BRANCH="saturating-bo"
N_ITER=400   # ~300 primaries accounting for ~25% repeats
BETA=4.0
H=0.15

# --- System setup ---
apt-get update -q
apt-get install -y -q python3-pip python3-venv python3-full git

# --- Clone repo (specific branch) ---
cd /home/ubuntu
git clone --branch "$BRANCH" "$REPO_URL" project
cd project

# --- Python environment ---
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# --- Download any existing state from S3 (resume if interrupted) ---
mkdir -p "experiments/$TASK_NAME"
python - <<PYEOF
import boto3
try:
    boto3.client("s3").download_file(
        "$S3_BUCKET", "$TASK_NAME/bo_state.json",
        "experiments/$TASK_NAME/bo_state.json"
    )
    print("Resumed from existing S3 state.")
except Exception as e:
    print(f"No existing state, starting fresh. ({e})")
PYEOF

# --- Run (boto3 handles per-iteration S3 uploads; crashes on S3 failure by design) ---
export S3_BUCKET
python run_bo.py \
    --task "$TASK_NAME" \
    --n-iter "$N_ITER" \
    --output-dir experiments \
    --beta "$BETA" \
    --h "$H"

# --- Self-terminate via boto3 (requires ec2:TerminateInstances on IAM role) ---
python - <<'EOF'
import urllib.request, boto3

# IMDSv2: get a session token first
token_req = urllib.request.Request(
    "http://169.254.169.254/latest/api/token",
    method="PUT",
    headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
)
token = urllib.request.urlopen(token_req).read().decode()

def imds(path):
    req = urllib.request.Request(
        f"http://169.254.169.254/latest/meta-data/{path}",
        headers={"X-aws-ec2-metadata-token": token},
    )
    return urllib.request.urlopen(req).read().decode()

instance_id = imds("instance-id")
region      = imds("placement/region")
boto3.client("ec2", region_name=region).terminate_instances(InstanceIds=[instance_id])
EOF
