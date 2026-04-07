AI-Based Predictive Thermal & Load Management for Edge Computing

Predicts CPU temperature 10 seconds ahead using a lightweight stacked LSTM, enabling proactive workload throttling before thermal thresholds are breached. Achieves 63.7% reduction in time above 80°C vs reactive baseline.


Results
MetricProactive (Ours)Reactive BaselineTime above 80°C212s584sTime above 75°C513s1365sPeak Temperature82.8°C84.6°CTemp Prediction MAE2.57°C—Model Size120 KB—

How It Works
Phase 1 → Simulate edge device sensor data (thermal RC circuit model, Markov workload states)
Phase 2 → Train stacked LSTM on 30s sliding windows to predict temp + load 10s ahead
Phase 3 → Proactive controller acts on predictions; compared against reactive baseline
Controller logic:

Predicted temp < 70°C → No action
Predicted temp 70–75°C → Soft throttle (−25% load)
Predicted temp > 75°C → Hard throttle (−50% load)
Reactive baseline only fires when actual temp hits 80°C


Model Architecture
Input: (30 timesteps × 5 features)
  → LSTM 64 units (long-range patterns)
  → Dropout 0.2
  → LSTM 32 units (short-term spikes)
  → Dropout 0.2
  → Dense 16 → Dense 2 (cpu_temp_c, cpu_load_pct)

Total params: 30,898 (~120 KB)

Stack
Python 3.11 · TensorFlow 2.21 · Keras 3.13 · scikit-learn · pandas · NumPy · matplotlib

Run
bashpip install tensorflow scikit-learn pandas numpy matplotlib

python phase1_simulation.py   # generates edge_simulation_data.csv
python phase2_lstm.py         # trains lstm_thermal.keras
python phase3_controller.py   # runs proactive vs reactive comparison

Output Files
FileDescriptionedge_simulation_data.csv3,600-sample simulated sensor datasetlstm_thermal.kerasTrained LSTM modellstm_evaluation.pngPrediction vs actual (temp + load)training_loss.pngLoss curve over epochscontroller_comparison.pngProactive vs reactive comparison
