# AI-Based Predictive Thermal & Load Management for Edge Computing 

## Overview

This project presents an AI-driven system that predicts CPU temperature **10 seconds in advance** using a lightweight stacked LSTM model.
Unlike traditional reactive systems, it enables **proactive workload throttling before thermal thresholds are breached**, preventing performance drops and hardware stress.

---

## Why This Matters

Edge devices (IoT, Raspberry Pi, embedded AI systems) lack active cooling.
Reactive thermal management acts **too late**, after overheating occurs.

This project solves that by **predicting and preventing overheating**, improving system reliability and lifespan.

---

## Key Results

| Metric              | Proactive (Ours) | Reactive Baseline |
| ------------------- | ---------------- | ----------------- |
| Time above 80°C     | 212 s            | 584 s             |
| Time above 75°C     | 513 s            | 1365 s            |
| Peak Temperature    | 82.8°C           | 84.6°C            |
| Temp Prediction MAE | 2.57°C           | —                 |
| Model Size          | ~120 KB          | —                 |

**✔ Achieved 63.7% reduction in time above 80°C**

---

## How It Works

* **Phase 1:** Simulate edge device sensor data using thermal RC model + Markov workload states
* **Phase 2:** Train stacked LSTM using 30-second history to predict temperature & load 10 seconds ahead
* **Phase 3:** Proactive controller uses predictions to prevent overheating and compares against reactive baseline

---

## Controller Logic

| Predicted Temperature | Action                    |
| --------------------- | ------------------------- |
| < 70°C                | No action                 |
| 70–75°C               | Soft throttle (−25% load) |
| > 75°C                | Hard throttle (−50% load) |

Reactive baseline triggers only when **actual temperature reaches 80°C**.

---

## Model Architecture

* Input: 30 timesteps × 5 features
* LSTM (64 units)
* Dropout (0.2)
* LSTM (32 units)
* Dropout (0.2)
* Dense → Output (CPU temp, CPU load)

**Total parameters:** 30,898 (~120 KB)
Optimized for deployment on edge devices.

---

## Tech Stack

* Python 3.11
* TensorFlow 2.21 / Keras 3.13
* scikit-learn
* pandas, NumPy
* matplotlib

---

## How to Run

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib

python simulation.py   # Generate dataset
python lstm.py         # Train model
python controller.py   # Run comparison
```

---

## Output Files

| File                      | Description                             |
| ------------------------- | --------------------------------------- |
| edge_simulation_data.csv  | Simulated sensor dataset (3600 samples) |
| lstm_thermal.keras        | Trained LSTM model                      |
| lstm_evaluation.png       | Prediction vs actual (temp & load)      |
| training_loss.png         | Training loss curve                     |
| controller_comparison.png | Proactive vs reactive comparison        |

---

## Key Contribution

* Predicts **temperature directly** (not just load or energy)
* Combines **physics-based thermal modeling + AI**
* Enables **proactive control instead of reactive response**
* Lightweight model suitable for **real edge deployment**

---

## Summary

This project demonstrates how combining **thermal physics and deep learning** enables early prediction of overheating and significantly reduces thermal risk.
It provides a **practical, deployable solution** for next-generation edge computing systems.
