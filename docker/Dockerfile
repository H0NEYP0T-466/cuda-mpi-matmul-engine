# ==============================================================================
# Dockerfile — MPI + OpenMP CPU Build Environment
#
# This container is for LOCAL Docker-based MPI execution.
# CUDA kernels run on Google Colab, NOT in these containers.
# ==============================================================================

FROM ubuntu:22.04

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools + OpenMPI + OpenMP
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    openmpi-bin \
    libopenmpi-dev \
    openssh-server \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for MPI communication between containers
RUN mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#StrictHostKeyChecking ask/StrictHostKeyChecking no/' /etc/ssh/ssh_config && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
    echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config

# Generate SSH key pair for passwordless MPI
RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -N "" && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/
COPY Makefile /app/Makefile
COPY docker/hostfile /app/hostfile

# Build CPU + MPI targets
RUN make cpu && make mpi

# Expose SSH port for MPI
EXPOSE 22

# Start SSH daemon in background
CMD ["/usr/sbin/sshd", "-D"]
