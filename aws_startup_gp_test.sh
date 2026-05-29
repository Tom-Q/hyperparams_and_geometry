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
apt-get install -y -q python3-pip python3-venv python3-full git

# --- Clone repo (specific branch) ---
cd /home/ubuntu
git clone --branch "$BRANCH" "$REPO_URL" project
cd project

# --- Python environment ---
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet -r requirements.txt

# --- Run (boto3 handles per-iteration S3 uploads) ---
export S3_BUCKET
python run_bo.py \
    --task "$TASK_NAME" \
    --n-iter "$N_ITER" \
    --output-dir experiments \
    --beta "$BETA" \
    --h "$H"

# --- Self-terminate via boto3 ---
python - <<'EOF'
import urllib.request, boto3
meta = "http://169.254.169.254/latest/meta-data/"
instance_id = urllib.request.urlopen(meta + "instance-id").read().decode()
region      = urllib.request.urlopen(meta + "placement/region").read().decode()
boto3.client("ec2", region_name=region).terminate_instances(InstanceIds=[instance_id])
EOF
