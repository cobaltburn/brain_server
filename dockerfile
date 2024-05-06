FROM tensorflow/tensorflow:2.15.0-gpu

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RUST_VERSION=1.77.1
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION # Install Rust
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory inside the container
WORKDIR /app

# Copy the Rust project files
COPY . .

# Find and copy the TensorFlow libraries to the current directory
# Find and copy the TensorFlow libraries to the current directory
RUN find . -type f -name libtensorflow.so.2 -exec cp {} . \;
RUN find . -type f -name libtensorflow_framework.so.2 -exec cp {} . \;

RUN mv /app/*.so.2 /usr/lib

# Set the environment variable for the library path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib

# Build the Rust server
RUN cargo build

# Specify the command to run your Rust server
CMD ["./target/debug/brain_server"]

EXPOSE 5000
