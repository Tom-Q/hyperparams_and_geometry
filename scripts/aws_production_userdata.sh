#!/bin/bash
set -e

# ============================================================
# PER-TASK SETTINGS — change these 3 lines before pasting
# ============================================================
TASK_NAME="spirals"       # mnist_dual | mnist_10way | fashion_10way | spirals | parity
                          # cartpole | fourrooms | adding | mnist_rnn
N_ITER=1000               # 1000 for all tasks except mnist_rnn (200)
H=0.162                   # supervised: 0.162 | RL: 0.116 | RNN: 0.147 | mnist_rnn: 0.218
# ============================================================

S3_BUCKET="tom-hyperparams-representations"
REPO_URL="https://github.com/Tom-Q/hyperparams_and_geometry.git"
BRANCH="main"
BETA=4.0

# --- System setup ---
apt-get update -q
apt-get install -y -q python3-pip python3-venv python3-full git

# --- Clone repo ---
cd /home/ubuntu
git clone --branch "$BRANCH" "$REPO_URL" project
cd project

# --- Python environment ---
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet -r requirements.txt

# --- Resume from S3 if interrupted ---
mkdir -p "output/experiments/$TASK_NAME"
python3 - <<PYEOF
import boto3
try:
    boto3.client("s3").download_file(
        "$S3_BUCKET", "$TASK_NAME/bo_state.json",
        "output/experiments/$TASK_NAME/bo_state.json"
    )
    print("Resumed from existing S3 state.")
except Exception as e:
    print(f"No existing state, starting fresh. ({e})")
PYEOF

# --- Run ---
export S3_BUCKET
python run_bo.py \
    --task       "$TASK_NAME" \
    --n-iter     "$N_ITER" \
    --output-dir output/experiments \
    --beta       "$BETA" \
    --h          "$H"

# --- Self-terminate ---
python3 - <<'EOF'
import urllib.request, boto3
token = urllib.request.urlopen(urllib.request.Request(
    "http://169.254.169.254/latest/api/token", method="PUT",
    headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
)).read().decode()
def imds(p):
    return urllib.request.urlopen(urllib.request.Request(
        f"http://169.254.169.254/latest/meta-data/{p}",
        headers={"X-aws-ec2-metadata-token": token}
    )).read().decode()
boto3.client("ec2", region_name=imds("placement/region")).terminate_instances(
    InstanceIds=[imds("instance-id")])
EOF
