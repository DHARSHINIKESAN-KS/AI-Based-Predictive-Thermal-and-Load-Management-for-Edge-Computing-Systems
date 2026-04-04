"""
Phase 3: Proactive Thermal Controller
AI-Based Predictive Thermal and Load Management for Edge Computing Systems
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = r"C:\Users\chan3\Documents\CLG\sensors\sp\data\edge_simulation_data.csv"
MODEL_PATH  = r"C:\Users\chan3\Documents\CLG\sensors\sp\data\lstm_thermal.keras"
OUTPUT_DIR  = r"C:\Users\chan3\Documents\CLG\sensors\sp\data"

LOOK_BACK       = 30
LOOK_AHEAD      = 10
FEATURE_COLS    = ['cpu_load_pct', 'cpu_temp_c', 'gpu_temp_c', 'memory_pct', 'power_watts']
TARGET_COLS     = ['cpu_temp_c', 'cpu_load_pct']

# Controller thresholds
SOFT_THROTTLE_TEMP  = 70.0   # start reducing workload here (predicted)
HARD_THROTTLE_TEMP  = 75.0   # aggressive reduction here (predicted)
REACTIVE_THRESHOLD  = 80.0   # reactive system only kicks in here (actual)

SOFT_REDUCTION      = 0.25   # reduce load by 25%
HARD_REDUCTION      = 0.50   # reduce load by 50%

# ─────────────────────────────────────────────
# LOAD DATA AND MODEL
# ─────────────────────────────────────────────
print("=" * 55)
print("  Phase 3: Proactive Thermal Controller")
print("=" * 55)

df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
model = load_model(MODEL_PATH)
print(f"\n[1/3] Loaded {len(df)} samples and trained LSTM model")

# Fit scalers (same as Phase 2 — must match training)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[FEATURE_COLS].values)

target_scaler = MinMaxScaler()
target_scaler.fit(df[TARGET_COLS].values)

# ─────────────────────────────────────────────
# PROACTIVE CONTROLLER SIMULATION
# ─────────────────────────────────────────────
def run_proactive_controller(df, data_scaled, model, scaler, target_scaler):
    """
    Proactive controller with feedback loop.
    When load is reduced, temperature actually drops in subsequent steps.
    """
    n = len(df)
    results = []

    # Work with mutable copies
    current_temp = df['cpu_temp_c'].values.copy().astype(float)
    current_load = df['cpu_load_pct'].values.copy().astype(float)
    data_live    = data_scaled.copy()

    for t in range(LOOK_BACK, n - LOOK_AHEAD):
        actual_temp = current_temp[t]
        actual_load = current_load[t]

        # Build window from live (potentially modified) data
        window = data_live[t - LOOK_BACK : t].reshape(1, LOOK_BACK, len(FEATURE_COLS))
        pred_scaled = model.predict(window, verbose=0)
        pred = target_scaler.inverse_transform(pred_scaled)[0]
        predicted_temp = pred[0]

        # Controller decision based on predicted temp
        if predicted_temp >= HARD_THROTTLE_TEMP:
            action = 'hard_throttle'
            load_factor = 1 - HARD_REDUCTION
        elif predicted_temp >= SOFT_THROTTLE_TEMP:
            action = 'soft_throttle'
            load_factor = 1 - SOFT_REDUCTION
        else:
            action = 'none'
            load_factor = 1.0

        effective_load = actual_load * load_factor

        # ── Feedback: if we throttled, cool down future temps ──
        if load_factor < 1.0:
            cooling = (1 - load_factor) * 0.4  # thermal relief factor
            for future in range(t + 1, min(t + LOOK_AHEAD + 5, n)):
                decay = cooling * (1 - (future - t) / (LOOK_AHEAD + 5))
                current_temp[future] = max(
                    current_temp[future] - decay * 10,
                    df['cpu_temp_c'].values[future] * 0.85
                )
                # Update live data array so LSTM sees the cooled state
                col_idx = FEATURE_COLS.index('cpu_temp_c')
                temp_min = data_scaled[:, col_idx].min()
                temp_max_val = data_scaled[:, col_idx].max()
                data_live[future, col_idx] = (
                    (current_temp[future] - df['cpu_temp_c'].min()) /
                    (df['cpu_temp_c'].max() - df['cpu_temp_c'].min())
                )

        results.append({
            'timestep':       t,
            'actual_temp':    actual_temp,
            'controlled_temp': current_temp[t],
            'actual_load':    actual_load,
            'predicted_temp': predicted_temp,
            'action':         action,
            'effective_load': effective_load,
        })

    return pd.DataFrame(results)


def run_reactive_controller(df):
    """
    Reactive controller with feedback loop.
    Only acts AFTER temperature actually hits 80°C.
    Gets cooler too, but only after the threshold is already breached.
    """
    n = len(df)
    results = []
    current_temp = df['cpu_temp_c'].values.copy().astype(float)
    current_load = df['cpu_load_pct'].values.copy().astype(float)

    for t in range(LOOK_BACK, n - LOOK_AHEAD):
        actual_temp  = current_temp[t]
        actual_load  = current_load[t]

        if actual_temp >= REACTIVE_THRESHOLD:
            action      = 'throttle'
            load_factor = 1 - HARD_REDUCTION
            # Feedback cooling — but only triggers AFTER 80°C breach
            cooling = (1 - load_factor) * 0.4
            for future in range(t + 1, min(t + LOOK_AHEAD + 5, n)):
                decay = cooling * (1 - (future - t) / (LOOK_AHEAD + 5))
                current_temp[future] = max(
                    current_temp[future] - decay * 10,
                    df['cpu_temp_c'].values[future] * 0.85
                )
        else:
            action      = 'none'
            load_factor = 1.0

        effective_load = actual_load * load_factor

        results.append({
            'timestep':        t,
            'actual_temp':     df['cpu_temp_c'].values[t],
            'controlled_temp': actual_temp,
            'actual_load':     actual_load,
            'predicted_temp':  actual_temp,
            'action':          action,
            'effective_load':  effective_load,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# PART 2 — COMPARE AND VISUALIZE
# ─────────────────────────────────────────────

# ── Compute comparison metrics ──

def compute_metrics(df_ctrl, label):
    throttle_events = (df_ctrl['action'] != 'none').sum()
    throttle_pct    = throttle_events / len(df_ctrl) * 100
    avg_temp        = df_ctrl['controlled_temp'].mean()
    max_temp        = df_ctrl['controlled_temp'].max()
    time_above_75   = (df_ctrl['controlled_temp'] >= 75).sum()
    time_above_80   = (df_ctrl['controlled_temp'] >= 80).sum()
    avg_load        = df_ctrl['effective_load'].mean()

    print(f"\n── {label} ──────────────────────────")
    print(f"  Throttle events      : {throttle_events} ({throttle_pct:.1f}% of time)")
    print(f"  Avg effective load   : {avg_load:.1f}%")
    print(f"  Avg temperature      : {avg_temp:.1f}°C")
    print(f"  Max temperature      : {max_temp:.1f}°C")
    print(f"  Time above 75°C      : {time_above_75}s ({time_above_75/len(df_ctrl)*100:.1f}%)")
    print(f"  Time above 80°C      : {time_above_80}s ({time_above_80/len(df_ctrl)*100:.1f}%)")

    return {
        'label':           label,
        'throttle_events': throttle_events,
        'throttle_pct':    throttle_pct,
        'avg_temp':        avg_temp,
        'max_temp':        max_temp,
        'time_above_75':   time_above_75,
        'time_above_80':   time_above_80,
        'avg_load':        avg_load,
    }

print("[2/3] Running controller simulations...")
print("      (Proactive controller predicts at each timestep — takes ~1 min)\n")

proactive_df = run_proactive_controller(df, data_scaled, model, scaler, target_scaler)
reactive_df  = run_reactive_controller(df)

print("\n✓ Both controllers simulated successfully")
print(f"  Proactive results : {len(proactive_df)} timesteps")
print(f"  Reactive results  : {len(reactive_df)} timesteps")

print("\n[3/3] Computing comparison metrics...")
m_proactive = compute_metrics(proactive_df, "Proactive LSTM Controller")
m_reactive  = compute_metrics(reactive_df,  "Reactive Baseline Controller")

# ── Summary comparison table ──
print("\n── Head-to-Head Comparison ──────────────────────────")
print(f"  {'Metric':<25} {'Proactive':>12} {'Reactive':>12} {'Winner':>10}")
print(f"  {'-'*62}")
metrics_compare = [
    ("Throttle events",    m_proactive['throttle_events'], m_reactive['throttle_events'],  'lower'),
    ("Time above 75°C (s)",m_proactive['time_above_75'],  m_reactive['time_above_75'],    'lower'),
    ("Time above 80°C (s)",m_proactive['time_above_80'],  m_reactive['time_above_80'],    'lower'),
    ("Max temperature",    m_proactive['max_temp'],        m_reactive['max_temp'],         'lower'),
    ("Avg load (%)",       m_proactive['avg_load'],        m_reactive['avg_load'],         'higher'),
]
for name, p_val, r_val, better in metrics_compare:
    if better == 'lower':
        winner = 'Proactive' if p_val <= r_val else 'Reactive'
    else:
        winner = 'Proactive' if p_val >= r_val else 'Reactive'
    print(f"  {name:<25} {p_val:>12.1f} {r_val:>12.1f} {winner:>10}")

# ── Plot 1: Controller comparison over time ──
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Proactive vs Reactive Controller — Full Simulation Comparison",
             fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(3, 1, hspace=0.45)

t_axis = proactive_df['timestep'].values / 60  # convert to minutes

# Temperature comparison
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_axis, proactive_df['controlled_temp'],
         color='#3498DB', linewidth=0.9, label='Proactive controlled temp')
ax1.plot(t_axis, reactive_df['controlled_temp'],
         color='#E74C3C', linewidth=0.9, linestyle='--', label='Reactive controlled temp')
ax1.plot(t_axis, proactive_df['actual_temp'],
         color='#95A5A6', linewidth=0.6, alpha=0.4, label='Uncontrolled temp')
ax1.axhline(75, color='#F39C12', linestyle=':', linewidth=1.2,
            label='Proactive trigger (75°C)')
ax1.axhline(80, color='#C0392B', linestyle='--', linewidth=1.2,
            alpha=0.7, label='Reactive threshold (80°C)')
# Shade proactive interventions
soft = proactive_df['action'] == 'soft_throttle'
hard = proactive_df['action'] == 'hard_throttle'
ax1.fill_between(t_axis, 0, 100,
                 where=soft.values, alpha=0.12,
                 color='#F39C12', label='Soft throttle active')
ax1.fill_between(t_axis, 0, 100,
                 where=hard.values, alpha=0.18,
                 color='#E74C3C', label='Hard throttle active')
ax1.set_ylabel('Temperature (°C)', fontsize=9)
ax1.set_title('CPU Temperature with Controller Interventions', fontsize=10, loc='left')
ax1.legend(fontsize=7.5, loc='upper right', ncol=2)
ax1.set_ylim(20, 90)
ax1.grid(True, alpha=0.2)

# Effective load comparison
ax2 = fig.add_subplot(gs[1])
ax2.plot(t_axis, proactive_df['effective_load'],
         color='#3498DB', linewidth=0.8, label='Proactive effective load')
ax2.plot(t_axis, reactive_df['effective_load'],
         color='#E74C3C', linewidth=0.8, linestyle='--',
         alpha=0.8, label='Reactive effective load')
ax2.plot(t_axis, proactive_df['actual_load'],
         color='#95A5A6', linewidth=0.6, alpha=0.5, label='Actual load (no control)')
ax2.set_ylabel('CPU Load (%)', fontsize=9)
ax2.set_title('Effective Load After Controller Action', fontsize=10, loc='left')
ax2.legend(fontsize=7.5, loc='upper right')
ax2.grid(True, alpha=0.2)

# Bar chart: key metrics side by side
ax3 = fig.add_subplot(gs[2])
bar_metrics  = ['Throttle\nevents', 'Time above\n75°C (s)', 'Time above\n80°C (s)']
pro_vals     = [m_proactive['throttle_events'],
                m_proactive['time_above_75'],
                m_proactive['time_above_80']]
react_vals   = [m_reactive['throttle_events'],
                m_reactive['time_above_75'],
                m_reactive['time_above_80']]

x      = np.arange(len(bar_metrics))
width  = 0.35
bars1  = ax3.bar(x - width/2, pro_vals,   width, label='Proactive LSTM', color='#3498DB', alpha=0.85)
bars2  = ax3.bar(x + width/2, react_vals, width, label='Reactive baseline', color='#E74C3C', alpha=0.85)

# Add value labels on bars
for bar in bars1:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)

ax3.set_xticks(x)
ax3.set_xticklabels(bar_metrics, fontsize=9)
ax3.set_ylabel('Count (seconds)', fontsize=9)
ax3.set_title('Key Metrics: Proactive vs Reactive', fontsize=10, loc='left')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.2, axis='y')

plt.savefig(os.path.join(OUTPUT_DIR, 'controller_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Comparison plot → {OUTPUT_DIR}\\controller_comparison.png")

# ── Save results CSV ──
proactive_df.to_csv(os.path.join(OUTPUT_DIR, 'proactive_results.csv'), index=False)
reactive_df.to_csv(os.path.join(OUTPUT_DIR, 'reactive_results.csv'),  index=False)

print("\n✓ Phase 3 complete!")
print("  Files saved to your data folder:")
print("  - controller_comparison.png  ← main paper figure")
print("  - proactive_results.csv      ← proactive controller log")
print("  - reactive_results.csv       ← reactive controller log")
print("\n  All 3 phases done. Ready to write the paper!")