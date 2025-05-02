# src/run_baseline_comparison.py
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# --- Imports ---
try: 
    # Assumes network.py uses the host.py WITH ADDED LOGGING
    from src.network import Network 
    from src.host import Host # Import host to access logs
except ImportError: 
     current_dir=os.path.dirname(os.path.abspath(__file__));parent_dir=os.path.join(current_dir, os.pardir)
     if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
     from src.network import Network; from src.host import Host

# --- Metric Calculation and Plotting Functions ---
# (Copied from run_ml_simulation.py for standalone execution)
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

def plot_comparison_metrics(results_dict, duration, title_suffix=""):
    """ Creates a single plot for CWND over time for multiple hosts (no RTT). """
    if not results_dict:
        return

    fig, ax = plt.subplots(figsize=(12, 6))  # Single axis for CWND

    for host_id, data in results_dict.items():
        if 'cwnd_ts' in data and data['cwnd_ts']:
            time_ticks = np.arange(len(data['cwnd_ts'])) * 10
            ax.plot(time_ticks, data['cwnd_ts'], label=f'{host_id} CWND')

    ax.set_ylabel("Congestion Window Size")
    ax.set_xlabel("Time (ticks)")
    ax.set_title(f"CWND Comparison Over Time {title_suffix}")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# --- Simulation Parameters ---
SIMULATION_DURATION = 1000  
NUM_PACKETS_PER_FLOW = 800
HOST_BUFFER_CAP = 64
ROUTER_BUFFER_CAP = 32     

def run_baseline_network(duration=SIMULATION_DURATION, num_packets=NUM_PACKETS_PER_FLOW):
    """Runs simulation with standard Reno and Vegas hosts."""
    print("Setting up Baseline (Reno/Vegas) network simulation...")
    # Use the original Network class with the host.py that has logging
    net = Network() 

    # --- Network Topology (Identical setup to ML sim for fair comparison) ---
    net.add_host("h1_reno", buffer_cap=HOST_BUFFER_CAP, tcp_variant="reno") 
    net.add_host("h2_vegas", buffer_cap=HOST_BUFFER_CAP, tcp_variant="vegas")
    net.add_host("h3_recv", buffer_cap=HOST_BUFFER_CAP, tcp_variant="reno") # Receiver
    net.add_host("h4_recv", buffer_cap=HOST_BUFFER_CAP, tcp_variant="vegas") # Receiver
    net.add_router("r1_base", buffer_cap=ROUTER_BUFFER_CAP)
    net.add_router("r2_base", buffer_cap=ROUTER_BUFFER_CAP)
    net.link("h1_reno", "r1_base"); net.link("h2_vegas", "r1_base") 
    net.link("h3_recv", "r2_base"); net.link("h4_recv", "r2_base")
    net.link("r1_base", "r2_base") # Bottleneck
    net.generate_forwarding_table_entries() 

    # --- Traffic Generation ---
    print(f"Queueing {num_packets} packets for Reno flow (h1 -> h3)...")
    sender1 = net.hosts["h1_reno"]; receiver1 = net.hosts["h3_recv"]
    for _ in range(num_packets): sender1.send_random_packet(receiver1)
    print(f"Queueing {num_packets} packets for Vegas flow (h2 -> h4)...")
    sender2 = net.hosts["h2_vegas"]; receiver2 = net.hosts["h4_recv"]
    for _ in range(num_packets): sender2.send_random_packet(receiver2)
        
    # --- Run Simulation & Record ---
    print(f"Running simulation for {duration} ticks...")
    simulation_results = {
        "h1_reno": {"cwnd_ts": [], "rtt_log": [], "timeout_log": [], "ack_received_log": []},
        "h2_vegas": {"cwnd_ts": [], "rtt_log": [], "timeout_log": [], "ack_received_log": []},
    }
    for i in range(duration): # Main simulation loop
        net.step() 
        if i % 10 == 0: # Record CWND periodically
             try: 
                 simulation_results["h1_reno"]["cwnd_ts"].append(net.hosts["h1_reno"].tcp.window_size)
                 simulation_results["h2_vegas"]["cwnd_ts"].append(net.hosts["h2_vegas"].tcp.window_size)
             except KeyError: print(f"Error accessing host tick {i}"); break
        if i > 0 and i % 100 == 0: print(f"  Tick {i}/{duration}") # Progress
        elif i == 0: print(f"  Tick 0/{duration}")
    print("Simulation finished.")

    # --- Collect Logs & Calculate Metrics ---
    print("\nCalculating performance metrics for Baseline Hosts...")
    h_reno = net.hosts["h1_reno"]
    h_vegas = net.hosts["h2_vegas"]
    
    # Store raw logs from hosts
    simulation_results["h1_reno"]["rtt_log"] = h_reno.rtt_log
    simulation_results["h1_reno"]["timeout_log"] = h_reno.timeout_log
    simulation_results["h1_reno"]["ack_received_log"] = h_reno.ack_received_log
    simulation_results["h2_vegas"]["rtt_log"] = h_vegas.rtt_log
    simulation_results["h2_vegas"]["timeout_log"] = h_vegas.timeout_log
    simulation_results["h2_vegas"]["ack_received_log"] = h_vegas.ack_received_log

    # Calculate summary metrics
    metrics_h1 = calculate_metrics(h_reno, duration)
    metrics_h2 = calculate_metrics(h_vegas, duration)

    print("\n--- Performance Summary (Baseline Hosts) ---")
    print(f"Host h1_reno:")
    print(f"  Avg Throughput: {metrics_h1['avg_throughput_acks_per_tick']:.4f} ACKs/tick")
    print(f"  Avg RTT:        {metrics_h1['avg_rtt']:.2f} ticks")
    print(f"  Total Timeouts: {metrics_h1['total_timeouts']}")
    print(f"  Approx Loss Rate: {metrics_h1['loss_rate_approx']:.4f}")
    print(f"\nHost h2_vegas:")
    print(f"  Avg Throughput: {metrics_h2['avg_throughput_acks_per_tick']:.4f} ACKs/tick")
    print(f"  Avg RTT:        {metrics_h2['avg_rtt']:.2f} ticks")
    print(f"  Total Timeouts: {metrics_h2['total_timeouts']}")
    print(f"  Approx Loss Rate: {metrics_h2['loss_rate_approx']:.4f}")
    print("-------------------------------------------")
    
    # --- Plot Results ---
    plot_comparison_metrics(simulation_results, duration, title_suffix="(Baseline Reno/Vegas)")


if __name__ == "__main__":
    print("--- Running Baseline (Reno/Vegas) Simulation & Performance Analysis ---")
    run_baseline_network()
    print("\nSimulation plot and metrics displayed.")