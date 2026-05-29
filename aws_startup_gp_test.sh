#!/bin/bash
set -e

# Single-task GP test run — spirals, starting from scratch (Sobol then GP).
# Paste this as EC2 User Data verbatim.

TASK_NAME="spirals"
S3_BUCKET="thomas-hyperparams-bo"
REPO_URL="https://github.com/Tom-Q/hyperparams_and_geometry.git"
BRANCH="saturating-bo"
N_ITER=300   # ~100 Sobol + 200 GP iterations
BETA=4.0
H=0.1

# --- System setup ---
apt-get update -q
apt-get install -y -q python3-pip python3-venv python3-full git awscli

# --- Clone repo (specific branch) ---
cd /home/ubuntu
git clone --branch "$BRANCH" "$REPO_URL" project
cd project

# --- Python environment ---
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet -r requirements.txt

# --- Download any existing state from S3 (allows resuming) ---
mkdir -p "experiments/$TASK_NAME"
aws s3 cp "s3://$S3_BUCKET/$TASK_NAME/bo_state.json" \
    "experiments/$TASK_NAME/bo_state.json" 2>/dev/null || true

# --- Run ---
export S3_BUCKET
python run_bo.py \
    --task "$TASK_NAME" \
    --n-iter "$N_ITER" \
    --output-dir experiments \
    --beta "$BETA" \
    --h "$H"

# --- Final upload ---
aws s3 cp "experiments/$TASK_NAME/bo_state.json" \
    "s3://$S3_BUCKET/$TASK_NAME/bo_state.json"

# --- Self-terminate ---
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
shutdown -h now
