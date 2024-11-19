#!/bin/bash

# Define the remote server details
SERVER_IP="agate.msi.umn.edu"      # Replace with your remote server IP
SERVER_PORT=6006                          # Default TensorBoard port (or change if needed)
LOCAL_PORT=6006                           # Local port to view TensorBoard

# Optionally, set the directory for TensorBoard logs
LOG_DIR="./runs"                           # Log directory on the remote machine

# Step 1: Start TensorBoard on the remote machine
echo "Starting TensorBoard on remote server $SERVER_IP..."

ssh sylve057@$SERVER_IP "tensorboard --logdir=$LOG_DIR --host=$SERVER_IP --port=$SERVER_PORT &"

# Step 2: Port forwarding from your local machine to the remote server
echo "Setting up port forwarding from localhost:$LOCAL_PORT to $SERVER_IP:$SERVER_PORT..."
ssh -N -L $LOCAL_PORT:localhost:$SERVER_PORT username@$SERVER_IP &

# Step 3: Provide a URL for the local machine to access TensorBoard
echo "TensorBoard is now accessible at http://localhost:$LOCAL_PORT"
