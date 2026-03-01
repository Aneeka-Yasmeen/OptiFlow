# OptiFlow
An adaptive AI Orchestrator for heterogeneous computing.

##Overview
OptiFlow is a middleman that sits between an AI model and the hardware. It uses a Bayesian-Reinforcement Learning approach to decide which parts of a model should run on the CPU, GPU, or FPGA based on real-time performance and battery state.

Current AI software usually dumps everything on the GPU. OptiFlow helps by keeping small tasks on the CPU and heavy math on the GPU and redundant tasks on FPGA which saves battery and reduces heat.

##Key Functionalities
Scans ONNX models to extract complexity (FLOPs) and memory data.
Uses a Bayesian brain to predict the fastest hardware path.
Monitors system health (temp, battery, load) to adjust scheduling in real-time.
Self-learning: it uses a reward system to improve its decisions after every run.

##Tech Stack
Language: Python
AI/Inference: ONNX Runtime, Scikit-optimize
Hardware Backends: DirectML (for AMD Windows), ROCm (for AMD Linux), OpenCL (to identify hardware as it's not vendor specific)
Telemetry: psutil, ROCm-SMI
UI: Streamlit

##AMD Implementation
This project is built to take advantage of AMD's unified ecosystem. It uses DirectML to talk to Radeon GPUs and Ryzen AI (NPU), and leverages Zen CPU cores for logic-heavy tasks where GPUs usually struggle with latency.
