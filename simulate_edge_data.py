"""
Phase 1: Edge Computing Thermal & Load Data Simulator
AI-Based Predictive Thermal and Load Management for Edge Computing Systems
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DURATION_SECONDS = 3600        # 1 hour of simulated data
SAMPLE_INTERVAL  = 1           # 1 reading per second
THERMAL_LAG      = 8           # seconds for temp to respond to load changes
AMBIENT_TEMP     = 32.0        # baseline ambient temperature (°C)
NOISE_TEMP       = 0.4         # sensor noise on temperature
NOISE_LOAD       = 1.2         # sensor noise on CPU load (%)

# Workload thresholds
THROTTLE_TEMP    = 80.0        # reactive system throttles here
SAFE_TEMP        = 75.0        # predictive system targets staying below this
MAX_TEMP         = 95.0        # critical shutdown threshold

# ─────────────────────────────────────────────
# WORKLOAD PROFILE GENERATOR
# ─────────────────────────────────────────────
def generate_workload_profile(n_samples):
    """
    Simulates realistic edge device workload patterns.
    Models 4 states: idle, normal, burst, spike.
    State transitions follow a Markov-like process.
    """
    # State definitions: (mean_load, std, min_duration_s, max_duration_s)
    states = {
        'idle':   (15,  3,  30,  120),
        'normal': (55,  8,  20,   90),
        'burst':  (80,  5,  10,   45),
        'spike':  (95,  2,  10,   30),
    }

    # Transition probabilities from each state
    transitions = {
        'idle':   {'idle': 0.05, 'normal': 0.85, 'burst': 0.08, 'spike': 0.02},
        'normal': {'idle': 0.10, 'normal': 0.45, 'burst': 0.35, 'spike': 0.10},
        'burst':  {'idle': 0.05, 'normal': 0.35, 'burst': 0.40, 'spike': 0.20},
        'spike':  {'idle': 0.10, 'normal': 0.60, 'burst': 0.20, 'spike': 0.10},
    }

    load_profile = np.zeros(n_samples)
    state_labels = [''] * n_samples
    current_state = 'idle'
    t = 0

    while t < n_samples:
        mean, std, min_dur, max_dur = states[current_state]
        duration = np.random.randint(min_dur, max_dur)
        end = min(t + duration, n_samples)

        # Smooth ramp between states (5-second transition)
        ramp_len = min(5, end - t)
        base_load = np.clip(np.random.normal(mean, std), 5, 100)

        for i in range(t, end):
            ramp_factor = min((i - t) / ramp_len, 1.0)
            load_profile[i] = base_load * ramp_factor + (load_profile[t-1] if t > 0 else 0) * (1 - ramp_factor)
            state_labels[i] = current_state

        t = end
        # Pick next state
        trans = transitions[current_state]
        current_state = np.random.choice(list(trans.keys()), p=list(trans.values()))

    return load_profile, state_labels


# ─────────────────────────────────────────────
# THERMAL MODEL
# ─────────────────────────────────────────────
def simulate_temperature(load_profile, ambient=AMBIENT_TEMP, lag=THERMAL_LAG):
    """
    Models CPU temperature as a function of load with thermal lag.
    Uses a first-order thermal RC circuit approximation:
      T(t) = T_ambient + k * load(t) + thermal_inertia
    The lag simulates heat capacitance — temp rises/falls slower than load.
    """
    n = len(load_profile)
    temp = np.zeros(n)
    temp[0] = ambient + 0.35 * load_profile[0]

    # Thermal time constant (higher = slower thermal response)
    tau = lag

    for t in range(1, n):
        # Steady-state temperature for current load
        target_temp = ambient + 0.65 * load_profile[t] + 0.002 * (load_profile[t] ** 1.8)
        # Exponential approach toward target (thermal RC model)
        temp[t] = temp[t-1] + (target_temp - temp[t-1]) / tau

    return temp


# ─────────────────────────────────────────────
# ADD SENSOR NOISE
# ─────────────────────────────────────────────
def add_noise(signal, noise_std):
    return signal + np.random.normal(0, noise_std, len(signal))


# ─────────────────────────────────────────────
# SECONDARY METRICS
# ─────────────────────────────────────────────
def compute_secondary_metrics(load_profile, temp_profile, n):
    """
    Adds memory usage, power draw, and GPU temp as additional features
    to make the dataset richer for multi-feature LSTM input.
    """
    # Memory usage: loosely correlated with load, but slower moving
    memory = 30 + 0.5 * load_profile + np.random.normal(0, 2, n)
    memory = np.clip(memory, 10, 95)

    # Power draw (Watts): non-linear function of load
    power = 2.5 + 0.08 * load_profile + 0.0005 * load_profile**2
    power = add_noise(power, 0.3)

    # GPU temperature: lower than CPU, similar lag
    gpu_load = 0.6 * load_profile + np.random.normal(0, 5, n)
    gpu_load = np.clip(gpu_load, 0, 100)
    gpu_temp = simulate_temperature(gpu_load, ambient=AMBIENT_TEMP - 2, lag=THERMAL_LAG + 3)
    gpu_temp = add_noise(gpu_temp, NOISE_TEMP * 0.8)

    return memory, power, gpu_temp


# ─────────────────────────────────────────────
# MAIN SIMULATION
# ─────────────────────────────────────────────
def run_simulation():
    print("=" * 55)
    print("  Edge Thermal & Load Data Simulator")
    print("  AI-Based Predictive Thermal Management")
    print("=" * 55)

    n_samples = DURATION_SECONDS // SAMPLE_INTERVAL
    timestamps = pd.date_range(start="2025-01-01 00:00:00", periods=n_samples, freq=f"{SAMPLE_INTERVAL}s")

    print(f"\n[1/4] Generating workload profile ({n_samples} samples)...")
    raw_load, state_labels = generate_workload_profile(n_samples)
    cpu_load = np.clip(add_noise(raw_load, NOISE_LOAD), 0, 100)

    print("[2/4] Simulating thermal response...")
    cpu_temp = simulate_temperature(raw_load)
    cpu_temp_noisy = add_noise(cpu_temp, NOISE_TEMP)
    cpu_temp_noisy = np.clip(cpu_temp_noisy, AMBIENT_TEMP - 2, MAX_TEMP)

    print("[3/4] Computing secondary metrics...")
    memory, power, gpu_temp = compute_secondary_metrics(raw_load, cpu_temp, n_samples)

    # Throttle flags (for labeling — useful for supervised learning later)
    throttle_reactive   = (cpu_temp_noisy >= THROTTLE_TEMP).astype(int)
    overheat_risk       = (cpu_temp_noisy >= SAFE_TEMP).astype(int)

    print("[4/4] Assembling dataset...")
    df = pd.DataFrame({
        'timestamp':        timestamps,
        'cpu_load_pct':     np.round(cpu_load, 2),
        'cpu_temp_c':       np.round(cpu_temp_noisy, 2),
        'gpu_temp_c':       np.round(gpu_temp, 2),
        'memory_pct':       np.round(memory, 2),
        'power_watts':      np.round(power, 2),
        'workload_state':   state_labels,
        'throttle_flag':    throttle_reactive,
        'overheat_risk':    overheat_risk,
    })

    # ── Summary stats ──
    print("\n── Dataset Summary ─────────────────────────")
    print(f"  Total samples      : {len(df):,}")
    print(f"  Duration           : {DURATION_SECONDS // 60} minutes")
    print(f"  CPU temp range     : {df.cpu_temp_c.min():.1f}°C – {df.cpu_temp_c.max():.1f}°C")
    print(f"  CPU load range     : {df.cpu_load_pct.min():.1f}% – {df.cpu_load_pct.max():.1f}%")
    print(f"  Throttle events    : {df.throttle_flag.sum()} samples above {THROTTLE_TEMP}°C")
    print(f"  Overheat-risk pct  : {df.overheat_risk.mean()*100:.1f}% of time above {SAFE_TEMP}°C")
    print(f"\n── Workload State Distribution ─────────────")
    for state in ['idle', 'normal', 'burst', 'spike']:
        count = (df.workload_state == state).sum()
        print(f"  {state:<8}: {count:>5} samples ({count/len(df)*100:.1f}%)")

    return df


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def plot_simulation(df, save_path):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Edge Device Simulation — 1 Hour of Sensor Data", fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    colors = {'idle': '#4CAF50', 'normal': '#2196F3', 'burst': '#FF9800', 'spike': '#F44336'}
    time_min = np.arange(len(df)) / 60

    # ── Plot 1: CPU Temperature ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_min, df['cpu_temp_c'], color='#E74C3C', linewidth=0.8, label='CPU temp')
    ax1.axhline(80, color='#C0392B', linestyle='--', linewidth=1, alpha=0.7, label=f'Throttle threshold (80°C)')
    ax1.axhline(75, color='#F39C12', linestyle=':', linewidth=1, alpha=0.8, label=f'Predictive target (75°C)')
    ax1.fill_between(time_min, 80, df['cpu_temp_c'], where=(df['cpu_temp_c'] >= 80), color='#E74C3C', alpha=0.2)
    ax1.set_ylabel('Temp (°C)', fontsize=9)
    ax1.set_title('CPU Temperature', fontsize=10, loc='left')
    ax1.legend(fontsize=7.5, loc='upper right')
    ax1.set_xlim(0, len(df) / 60)
    ax1.grid(True, alpha=0.2)

    # ── Plot 2: CPU Load with state coloring ──
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time_min, df['cpu_load_pct'], color='#3498DB', linewidth=0.7, alpha=0.8)
    state_prev = df['workload_state'].iloc[0]
    start_idx = 0
    for i in range(1, len(df)):
        if df['workload_state'].iloc[i] != state_prev or i == len(df) - 1:
            ax2.axvspan(start_idx / 60, i / 60, alpha=0.08, color=colors.get(state_prev, '#999'))
            state_prev = df['workload_state'].iloc[i]
            start_idx = i
    ax2.set_ylabel('Load (%)', fontsize=9)
    ax2.set_title('CPU Load (shaded by workload state)', fontsize=10, loc='left')
    ax2.set_xlim(0, len(df) / 60)
    ax2.grid(True, alpha=0.2)
    # Legend for states
    for state, color in colors.items():
        ax2.bar(0, 0, color=color, alpha=0.4, label=state)
    ax2.legend(fontsize=7.5, loc='upper right', ncol=4)

    # ── Plot 3: Memory + GPU temp ──
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time_min, df['memory_pct'], color='#9B59B6', linewidth=0.7, alpha=0.9, label='Memory (%)')
    ax3_r = ax3.twinx()
    ax3_r.plot(time_min, df['gpu_temp_c'], color='#1ABC9C', linewidth=0.7, alpha=0.8, label='GPU temp (°C)')
    ax3.set_ylabel('Memory (%)', fontsize=9, color='#9B59B6')
    ax3_r.set_ylabel('GPU temp (°C)', fontsize=9, color='#1ABC9C')
    ax3.set_title('Memory Usage & GPU Temperature', fontsize=10, loc='left')
    ax3.set_xlim(0, len(df) / 60)
    ax3.grid(True, alpha=0.2)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc='upper right')

    # ── Plot 4: Power draw ──
    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(time_min, df['power_watts'], color='#E67E22', alpha=0.5)
    ax4.plot(time_min, df['power_watts'], color='#E67E22', linewidth=0.7)
    ax4.set_ylabel('Power (W)', fontsize=9)
    ax4.set_xlabel('Time (minutes)', fontsize=9)
    ax4.set_title('Power Draw', fontsize=10, loc='left')
    ax4.set_xlim(0, len(df) / 60)
    ax4.grid(True, alpha=0.2)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    output_dir = r"C:\Users\chan3\Documents\CLG\sensors\sp\data"
    os.makedirs(output_dir, exist_ok=True)

    df = run_simulation()

    csv_path  = r"C:\Users\chan3\Documents\CLG\sensors\sp\data\edge_simulation_data.csv"
    plot_path = r"C:\Users\chan3\Documents\CLG\sensors\sp\data\edge_simulation_plot.png"

    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved  → {csv_path}")

    plot_simulation(df, plot_path)

    print("\n✓ Phase 1 complete. Ready for Phase 2 (LSTM training).")
    print("  Next: load edge_simulation_data.csv → preprocess → train LSTM")