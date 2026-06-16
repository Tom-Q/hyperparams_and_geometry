#!/bin/bash
set -e

S3_BUCKET="tom-hyperparams-representations"
REPO_URL="https://github.com/Tom-Q/hyperparams_and_geometry.git"
BRANCH="main"

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

# --- Run smoke test (all 9 tasks, activations saved to S3) ---
export S3_BUCKET
python scripts/aws_smoke_test.py \
    --output-dir output/smoke \
    --data-dir data \
    2>&1 | tee /home/ubuntu/smoke_test.log

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
