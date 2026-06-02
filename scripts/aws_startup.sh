#!/bin/bash
set -e

# --- Configuration ---
TASK_NAME="__TASK__"          # replaced at launch time by launch_all_tasks.sh
S3_BUCKET="tom-hyperparams-representations"
REPO_URL="https://github.com/Tom-Q/hyperparams_and_geometry.git"
N_ITER=400
BETA=4.0

# --- System setup ---
apt-get update -q
apt-get install -y -q python3-pip python3-venv python3-full git awscli

# --- Clone repo ---
cd /home/ubuntu
git clone "$REPO_URL" project
cd project

# --- Python environment ---
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet -r requirements.txt

# --- Sync any existing BO state from S3 (allows resuming interrupted runs) ---
aws s3 sync "s3://$S3_BUCKET/$TASK_NAME/" "experiments/$TASK_NAME/" || true

# --- Run ---
export S3_BUCKET
python run_bo.py \
    --task "$TASK_NAME" \
    --n-iter "$N_ITER" \
    --output-dir experiments \
    --beta "$BETA"

# --- Terminate this instance ---
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"

shutdown -h now
