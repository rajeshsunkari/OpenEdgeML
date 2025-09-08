# 📘 OpenEdgeML Architecture

This document outlines the high-level architecture of the OpenEdgeML pipeline, which automates model training, compression, and deployment for edge devices.

## 🧬 AutoML Pipeline for Edge Devices

```mermaid
graph TD
    A[Input Dataset] --> B[Preprocessing]
    B --> C[Feature Selection]
    C --> D[Model Selection (AutoML Search)]
    D --> E[Hyperparameter Optimization]
    E --> F[Model Compression (Quantization, Pruning)]
    F --> G[Export Format (ONNX, TFLite)]
    G --> H[Edge Deployment (Raspberry Pi, Jetson Nano)]
    H --> I[Monitoring & Dashboard]
