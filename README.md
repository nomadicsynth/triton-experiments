# Triton Experiments

This repository serves as a playground for experimenting with [Triton](https://github.com/openai/triton) kernels. Triton is an open-source programming language and compiler for writing highly efficient GPU kernels, particularly optimized for NVIDIA GPUs.

While the current focus is on raytracing implementations, this repo is intended for general experimentation with Triton kernels across various computational tasks.

## Features

- **Raytracing Examples**: Multiple raytracing implementations using Triton kernels
  - `ray_tracer_gl.py`: OpenGL-based raytracer
  - `ray-tracer-1.py`: Basic raytracing implementation
  - `record_ray_tracer.py`: Recording functionality for raytracing

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nomadicsynth/triton-experiments.git
   cd triton-experiments
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   **Note**: This project requires CUDA-compatible hardware and drivers for Triton to work properly.

## Usage

Run the raytracing examples:

```bash
python ray_tracer_gl.py
```

Or other scripts as needed for your experiments.

## Requirements

- Python 3.8+
- CUDA 11.0+ compatible GPU
- NVIDIA drivers

## Contributing

This is an experimental repository. Feel free to add your own Triton kernel experiments, optimizations, or new computational examples.

## License

triton-experiments by nomadicsynth is marked CC0 1.0. To view a copy of this mark, visit <https://creativecommons.org/publicdomain/zero/1.0/>
