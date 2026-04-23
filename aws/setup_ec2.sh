#!/bin/bash
# setup_ec2.sh — AWS EC2 c5.large provisioning guide for MPI experiment
# See README.md for full instructions.

echo "AWS EC2 Setup for MPI Scaling Experiment"
echo "Instance: c5.large (2 vCPU, 4GB, \$0.085/hr)"
echo ""
echo "1) Launch: aws ec2 run-instances --instance-type c5.large --count 4 --image-id ami-0c7217cdde317cfec --key-name matmul-keypair"
echo "2) Install on each: sudo apt install -y build-essential openmpi-bin libopenmpi-dev"
echo "3) Deploy: scp build/matmul_mpi ubuntu@<IP>:~/"
echo "4) Run: ./scripts/cloud_scaling.sh"
echo "5) Cleanup: aws ec2 terminate-instances --instance-ids <IDs>"
