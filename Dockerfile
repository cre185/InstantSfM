FROM nvcr.io/nvidia/pytorch:25.11-py3

ARG TARGETARCH
ARG DEBIAN_FRONTEND=noninteractive

# Core build tooling and runtime libs needed by scikit-sparse / OpenCV / fused-ssim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git wget ca-certificates xz-utils \
    libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev \
    libsuitesparse-dev libmetis-dev liblapack-dev libblas-dev \
    libgl1 libglib2.0-0 colmap \
    && rm -rf /var/lib/apt/lists/*

# Install cuDSS for CUDA 13 (arch-aware)
ARG CUDSS_VERSION=0.7.1.4
ENV CUDSS_ROOT=/opt/cudss-${CUDSS_VERSION}
RUN set -euo pipefail; \
    arch="${TARGETARCH:-$(uname -m)}"; \
    case "${arch}" in \
        amd64|x86_64) cudss_arch="x86_64"; deb_arch_dir="x86_64-linux-gnu";; \
        arm64|aarch64) cudss_arch="aarch64"; deb_arch_dir="aarch64-linux-gnu";; \
        *) echo "Unsupported architecture: ${arch}"; exit 1;; \
    esac; \
    CUDSS_TARBALL="libcudss-linux-${cudss_arch}-${CUDSS_VERSION}_cuda13-archive.tar.xz"; \
    CUDSS_URL_BASE="https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-${cudss_arch}"; \
    wget -q "${CUDSS_URL_BASE}/${CUDSS_TARBALL}" -O /tmp/cudss.tar.xz; \
    tar -xJf /tmp/cudss.tar.xz -C /opt; \
    ln -s "/opt/libcudss-linux-${cudss_arch}-${CUDSS_VERSION}_cuda13-archive" "${CUDSS_ROOT}"; \
    mkdir -p /usr/include/libcudss/12 "/usr/lib/${deb_arch_dir}/libcudss/12"; \
    cp -r "${CUDSS_ROOT}/include/"* /usr/include/libcudss/12/; \
    cp -r "${CUDSS_ROOT}/lib/"* "/usr/lib/${deb_arch_dir}/libcudss/12/"; \
    ldconfig; \
    rm /tmp/cudss.tar.xz

# cuDSS toolchain paths
ENV CUDSS_LIBCUDSS_PATHS=/usr/lib/aarch64-linux-gnu/libcudss/12:/usr/lib/x86_64-linux-gnu/libcudss/12
ENV LD_LIBRARY_PATH=${CUDSS_ROOT}/lib:${CUDSS_LIBCUDSS_PATHS}:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDSS_ROOT}/lib:${CUDSS_LIBCUDSS_PATHS}:${LIBRARY_PATH}
ENV CPATH=${CUDSS_ROOT}/include
ENV CUDA_HOME=/usr/local/cuda
ENV FUSED_SSIM_FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0;8.6;8.9;11.0"

# Build and install Ceres Solver (needed by pyceres)
ARG CERES_VERSION=2.1.0
RUN wget -q http://ceres-solver.org/ceres-solver-${CERES_VERSION}.tar.gz \
    && tar zxf ceres-solver-${CERES_VERSION}.tar.gz \
    && mkdir ceres-build \
    && cd ceres-build \
    && cmake ../ceres-solver-${CERES_VERSION} -GNinja \
         -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON \
         -DMINIGLOG=OFF -DSUITESPARSE=OFF -DCXSPARSE=OFF \
    && ninja install \
    && cd / \
    && rm -rf ceres-build ceres-solver-${CERES_VERSION} ceres-solver-${CERES_VERSION}.tar.gz

WORKDIR /workspace/InstantSfM

# Install Python deps separately so code edits don't bust the cache.
COPY pyproject.toml README.md ./
RUN set -euo pipefail \
    && python -c "import tomllib, pathlib; py=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); reqs=py['project']['dependencies']; pathlib.Path('/tmp/requirements.txt').write_text('\\n'.join(reqs) + '\\n')" \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && git clone --depth=1 https://github.com/rahul-goel/fused-ssim /tmp/fused-ssim \
    && python -c "from pathlib import Path; p=Path('/tmp/fused-ssim/setup.py'); t=p.read_text(); t=t.replace('elif torch.mps.is_available():','elif False and torch.mps.is_available():'); t=t.replace('elif hasattr(torch, \\'xpu\\'):', 'elif False and hasattr(torch, \\'xpu\\'):'); p.write_text(t)" \
    && pip install --no-cache-dir --no-build-isolation /tmp/fused-ssim \
    && MAX_JOBS=$(nproc) pip install --no-cache-dir --no-build-isolation git+https://github.com/zitongzhan/bae.git

# Install the project itself (editable, no deps) after source copy.
COPY instantsfm instantsfm
COPY demo.py demo.py
RUN pip install --no-cache-dir -e . --no-deps

# Bring in the rest of the project (assets, configs, etc.)
COPY . .

CMD ["/bin/bash"]
