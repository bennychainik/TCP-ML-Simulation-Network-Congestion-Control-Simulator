# src/run_ml_simulation.py
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# --- Imports ---
try: 
    from src.networkml import NetworkML 
    from src.hostml import HostML 
except ImportError: 
     current_dir=os.path.dirname(os.path.abspath(__file__));parent_dir=os.path.join(current_dir, os.pardir)
     if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
     from src.networkml import NetworkML; from src.hostml import HostML

# --- Metric Calculation and Plotting Functions ---
def calculate_metrics(host, total_duration):
    """ Calculates performance metrics from host logs. """
    results = {}
    if host.rtt_log: results['avg_rtt'] = np.mean([rtt for ts, rtt in host.rtt_log])
    else: results['avg_rtt'] = 0
    results['total_timeouts'] = len(host.timeout_log)
    results['total_acks_received'] = len(host.ack_received_log)
    if total_duration > 0: results['avg_throughput_acks_per_tick'] = results['total_acks_received'] / total_duration
    else: results['avg_throughput_acks_per_tick'] = 0
    if (results['total_timeouts'] + results['total_acks_received']) > 0:
        results['loss_rate_approx'] = results['total_timeouts'] / (results['total_timeouts'] + results['total_acks_received'])
    else: results['loss_rate_approx'] = 0
    return results

def plot_metrics(results_dict, duration, title_suffix=""):
    """ Creates a plot for CWND over time (no RTT). """
    if not results_dict:
        return

    fig, ax = plt.subplots(figsize=(12, 6))  # One plot only

    for host_id, data in results_dict.items():
        if 'cwnd_ts' in data and data['cwnd_ts']:
            time_ticks = np.arange(len(data['cwnd_ts'])) * 10  # Adjust x-axis scale if needed
            ax.plot(time_ticks, data['cwnd_ts'], label=f'{host_id} CWND')

    ax.set_ylabel("Congestion Window Size")
    ax.set_xlabel("Time (ticks)")
    ax.set_title(f"CWND Over Time {title_suffix}")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# --- Simulation Parameters ---
SIMULATION_DURATION = 1000  
NUM_PACKETS_PER_FLOW = 800
HOST_BUFFER_CAP = 64
ROUTER_BUFFER_CAP = 32     

def run_ml_network(duration=SIMULATION_DURATION, num_packets=NUM_PACKETS_PER_FLOW):
    """Sets up, runs simulation using HostML with retrained model, calculates metrics."""
    print("Setting up NetworkML simulation (using RETRAINED model)...")
    net = NetworkML() # Uses HostML which loads ml_host_pipeline.pkl

    # --- Network Topology ---
    net.add_host("h1_ml", buffer_cap=HOST_BUFFER_CAP) 
    net.add_host("h2_ml", buffer_cap=HOST_BUFFER_CAP) 
    net.add_host("h3_recv", buffer_cap=HOST_BUFFER_CAP) 
    net.add_host("h4_recv", buffer_cap=HOST_BUFFER_CAP) 
    net.add_router("r1_ml", buffer_cap=ROUTER_BUFFER_CAP)
    net.add_router("r2_ml", buffer_cap=ROUTER_BUFFER_CAP)
    net.link("h1_ml", "r1_ml"); net.link("h2_ml", "r1_ml") 
    net.link("h3_recv", "r2_ml"); net.link("h4_recv", "r2_ml")
    net.link("r1_ml", "r2_ml") # Bottleneck
    net.generate_forwarding_table_entries() 

    # --- Traffic Generation ---
    print(f"Queueing {num_packets} packets for Flow 1 (h1_ml -> h3_recv)...")
    sender1 = net.hosts["h1_ml"]; receiver1 = net.hosts["h3_recv"]
    for _ in range(num_packets): sender1.send_random_packet(receiver1)
    print(f"Queueing {num_packets} packets for Flow 2 (h2_ml -> h4_recv)...")
    sender2 = net.hosts["h2_ml"]; receiver2 = net.hosts["h4_recv"]
    for _ in range(num_packets): sender2.send_random_packet(receiver2)
        
    # --- Run Simulation & Record Data ---
    print(f"Running simulation for {duration} ticks...")
    simulation_results = {
        "h1_ml": {"cwnd_ts": [], "rtt_log": [], "timeout_log": [], "ack_received_log": []},
        "h2_ml": {"cwnd_ts": [], "rtt_log": [], "timeout_log": [], "ack_received_log": []},
    }
    for i in range(duration): # Main simulation loop
        net.step() 
        if i % 10 == 0: # Record CWND periodically
             try: 
                 simulation_results["h1_ml"]["cwnd_ts"].append(net.hosts["h1_ml"].tcp.window_size)
                 simulation_results["h2_ml"]["cwnd_ts"].append(net.hosts["h2_ml"].tcp.window_size)
             except KeyError: print(f"Error accessing host tick {i}"); break
        if i > 0 and i % 100 == 0: print(f"  Tick {i}/{duration}") # Progress
        elif i == 0: print(f"  Tick 0/{duration}")
    print("Simulation finished.")

    # --- Collect Logs & Calculate Metrics ---
    print("\nCalculating performance metrics for ML Hosts...")
    h1 = net.hosts["h1_ml"]; h2 = net.hosts["h2_ml"]
    # Store raw logs from hosts into results dictionary
    simulation_results["h1_ml"]["rtt_log"] = h1.rtt_log
    simulation_results["h1_ml"]["timeout_log"] = h1.timeout_log
    simulation_results["h1_ml"]["ack_received_log"] = h1.ack_received_log
    simulation_results["h2_ml"]["rtt_log"] = h2.rtt_log
    simulation_results["h2_ml"]["timeout_log"] = h2.timeout_log
    simulation_results["h2_ml"]["ack_received_log"] = h2.ack_received_log
    # Calculate summary metrics
    metrics_h1 = calculate_metrics(h1, duration)
    metrics_h2 = calculate_metrics(h2, duration)

    print("\n--- Performance Summary (ML Hosts - Retrained Model) ---")
    print(f"Host h1_ml:")
    print(f"  Avg Throughput: {metrics_h1['avg_throughput_acks_per_tick']:.4f} ACKs/tick")
    print(f"  Avg RTT:        {metrics_h1['avg_rtt']:.2f} ticks")
    print(f"  Total Timeouts: {metrics_h1['total_timeouts']}")
    print(f"  Approx Loss Rate: {metrics_h1['loss_rate_approx']:.4f}")
    print(f"\nHost h2_ml:")
    print(f"  Avg Throughput: {metrics_h2['avg_throughput_acks_per_tick']:.4f} ACKs/tick")
    print(f"  Avg RTT:        {metrics_h2['avg_rtt']:.2f} ticks")
    print(f"  Total Timeouts: {metrics_h2['total_timeouts']}")
    print(f"  Approx Loss Rate: {metrics_h2['loss_rate_approx']:.4f}")
    print("-------------------------------------------------------")
    
    # --- Plot Results ---
    plot_metrics(simulation_results, duration, title_suffix="(ML Host - Retrained Model)")

if __name__ == "__main__":
    print("--- Running ML Host Simulation with RETRAINED Model & Performance Analysis ---")
    run_ml_network()
    print("\nSimulation plot and metrics displayed.")