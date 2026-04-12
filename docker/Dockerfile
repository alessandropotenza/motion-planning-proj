# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile for yimingli1998/cdf + yimingli1998/RDF
# Target: NVIDIA DGX Spark (CUDA 12.x, Blackwell/Hopper GPU)
#
# Python 3.8 · PyTorch 2.0.1 · CUDA 11.7 base image
#
# Supports running:
#   - frankaemika/data_generator.py  (offline data generation)
#   - frankaemika/nn_cdf.py          (neural CDF training + evaluation)
#
# NOTE: PyTorch 1.13 (listed in the READMEs) only supports CUDA ≤11.7 and will
# NOT run on DGX Spark GPUs. PyTorch 2.0.1 is fully compatible with all code
# in these repos and runs on CUDA 12.x via NVIDIA driver forward-compatibility
# (requires host driver ≥ 520, standard on any DGX Spark).
#
# Directory layout inside container:
#   /workspace/
#     cdf/          ← main project, default WORKDIR
#     RDF/          ← placed alongside cdf (required by cdf's sys.path logic)
#     data/         ← mount your host directory here to retrieve all outputs
#
# ── Build ─────────────────────────────────────────────────────────────────────
#   docker build -t cdf-rdf .
#
# ── Step 1: Generate offline data ─────────────────────────────────────────────
#   docker run --gpus all --rm \
#     -v $(pwd)/output_data:/workspace/data \
#     cdf-rdf \
#     python frankaemika/data_generator.py
#
# ── Step 2: Train the neural CDF ──────────────────────────────────────────────
#   docker run --gpus all --rm \
#     -v $(pwd)/output_data:/workspace/data \
#     cdf-rdf \
#     python frankaemika/nn_cdf.py
#
# Both data.npy → data.pt → model_dict.pt will be written to /workspace/data
# and will appear in ./output_data/ on the host after each step.
#
# ── Interactive shell ─────────────────────────────────────────────────────────
#   docker run --gpus all -it --rm \
#     -v $(pwd)/output_data:/workspace/data \
#     cdf-rdf
# ──────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel


# ---------- system packages ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ninja-build \
        # libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
        libglew-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYOPENGL_PLATFORM=egl

# ---------- working directory -------------------------------------------------
WORKDIR /workspace

# ---------- clone both repositories ------------------------------------------
RUN git clone https://github.com/yimingli1998/cdf.git cdf && \
    git clone https://github.com/yimingli1998/RDF.git RDF


# ---------- upgrade pip -------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip --break-system-packages

# ---------- CDF + RDF shared dependencies ------------------------------------
RUN pip install -r /workspace/cdf/requirements.txt --break-system-packages
RUN pip install -r /workspace/RDF/requirements.txt --break-system-packages

RUN pip install --no-cache-dir --break-system-packages \
        numpy \
        matplotlib \
        pybullet \
        pytorch-minimize \
        pyyaml \
        "trimesh[easy]" \
        "pyglet==1.5.0" \
        git+https://github.com/marian42/mesh_to_sdf.git

# ---------- pytorch3d (required by nn_cdf.py → process_data) -----------------
# We build from source to guarantee compatibility with the installed torch/CUDA.
# This step takes ~10 minutes; it is cached after the first build.
RUN pip install --no-cache-dir --break-system-packages \
        fvcore \
        iopath \
        "numpy<2" \
# RUN pip install --no-cache-dir \
        # "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable"

# ---------- PYTHONPATH -------------------------------------------------------
# cdf/frankaemika scripts do: sys.path.append('../../RDF')
# Setting PYTHONPATH ensures imports work regardless of CWD.
# ENV PYTHONPATH="/workspace/cdf:/workspace/cdf/frankaemika:/workspace/RDF:${PYTHONPATH}"

# ---------- output / data directory ------------------------------------------
# All generated files (data.npy, data.pt, model_dict.pt) should be written
# to /workspace/data so they persist on the host via the volume mount.
RUN mkdir /workspace/data

# ──────────────────────────────────────────────────────────────────────────────
# ⚠️  REQUIRED SOURCE EDITS before running — two commented-out save lines must
#     be uncommented and paths redirected to /workspace/data:
#
# 1. data_generator.py  →  generate_offline_data():
#      CHANGE:  # np.save(os.path.join(save_path,'data.npy'),data)
#      TO:        np.save(os.path.join(save_path,'data.npy'),data)
#      AND in __main__:
#      CHANGE:  gen.generate_offline_data()
#      TO:      gen.generate_offline_data(save_path='/workspace/data')
#
# 2. nn_cdf.py  →  __init__:
#      CHANGE:  self.data_path = os.path.join(CUR_PATH,'data.pt')
#      TO:      self.data_path = '/workspace/data/data.pt'
#      (so it reads the data generated in Step 1)
#
# 3. nn_cdf.py  →  process_data():
#      CHANGE:  torch.save(final_data,os.path.join(CUR_PATH,'data.pt'))
#      TO:      torch.save(final_data,'/workspace/data/data.pt')
#
# 4. nn_cdf.py  →  train_nn():
#      CHANGE:  # torch.save(model_dict, os.path.join(CUR_PATH,'model_dict.pt'))
#      TO:        torch.save(model_dict, '/workspace/data/model_dict.pt')
#
# 5. nn_cdf.py  →  eval_nn() and eval_nn_noise():
#      Both load:  torch.load(os.path.join(CUR_PATH,'model_dict.pt'))
#      Change to:  torch.load('/workspace/data/model_dict.pt')
#
# 6. nn_cdf.py  →  __main__ (to run the full pipeline in one go):
#      Uncomment process_data and update epoch + model load key as needed.
# ──────────────────────────────────────────────────────────────────────────────

COPY ../data_generator.py /workspace/data_generator.py
COPY ../nn_cdf.py /workspace/nn_cdf.py

WORKDIR /workspace/cdf

CMD ["/bin/bash"]
