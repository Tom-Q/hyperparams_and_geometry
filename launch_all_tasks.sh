#!/bin/bash
# Launch one spot instance per task.
# Fill in the four placeholder values below before running.

AMI_ID="ami-XXXXXXXXXXXXXXX"      # Ubuntu 24.04 LTS in eu-west-3 (from Step 4)
SECURITY_GROUP="sg-XXXXXXXX"      # from Step 3c
IAM_ROLE="ec2-s3-access"          # from Step 3b
KEY_NAME="hyperparams-key"        # from Step 3d
INSTANCE_TYPE="c5.xlarge"         # 4 vCPU, 8 GB RAM
S3_BUCKET="thomas-hyperparams-bo" # from Step 3a
REGION="eu-west-3"

TASKS="spirals parity mnist_10way mnist_dual fashion_10way mnist_rnn adding cartpole fourrooms"

for TASK in $TASKS; do
    USERDATA=$(sed "s/__TASK__/$TASK/" aws_startup.sh | base64 -w 0)

    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
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
        --output text)

    echo "Launched $TASK → $INSTANCE_ID"
done
