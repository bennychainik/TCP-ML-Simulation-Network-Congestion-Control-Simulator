# src/generate_dataset.py
import os
import csv
import random
# Ensure this imports the Network class which uses the modified host.py above
from src.network import Network 
from src.host import Host 

# --- Simulation Parameters ---
SIMULATION_DURATION = 1500 # Ticks (Adjust as needed)
NUM_PACKETS_TO_SEND = 1000 # Total packets each flow attempts to send
HOST_BUFFER_CAP = 64       
ROUTER_BUFFER_CAP = 32     # Make router buffer smaller to cause congestion

# --- Define NEW output filename ---
OUTPUT_DIR = "data"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "ml_host_training_data.csv") # New name!

def generate_data(duration=SIMULATION_DURATION, num_packets=NUM_PACKETS_TO_SEND):
    """Runs simulation with Reno/Vegas hosts and collects SIMPLIFIED data for HostML training."""

    print(f"Setting up network for HostML dataset generation ({duration} ticks)...")
    net = Network() # Uses the modified Host class with logging hooks

    # --- Network Topology (Dumbbell - Same setup) ---
    net.add_host("h1_reno", buffer_cap=HOST_BUFFER_CAP, tcp_variant="reno")
    net.add_host("h2_recv", buffer_cap=HOST_BUFFER_CAP, tcp_variant="reno") 
    net.add_host("h3_vegas", buffer_cap=HOST_BUFFER_CAP, tcp_variant="vegas")
    net.add_host("h4_recv", buffer_cap=HOST_BUFFER_CAP, tcp_variant="vegas")
    net.add_router("r1", buffer_cap=ROUTER_BUFFER_CAP)
    net.add_router("r2", buffer_cap=ROUTER_BUFFER_CAP)
    net.link("h1_reno", "r1"); net.link("h3_vegas", "r1")
    net.link("h2_recv", "r2"); net.link("h4_recv", "r2")
    net.link("r1", "r2") # Bottleneck
    
    print("Generating forwarding tables...")
    net.generate_forwarding_table_entries() # Assumes this method exists in Network class

    # --- Traffic Generation ---
    print(f"Queueing {num_packets} packets for Reno flow (h1 -> h2)...")
    h1 = net.hosts["h1_reno"]; h2 = net.hosts["h2_recv"]
    for _ in range(num_packets): h1.send_random_packet(h2)
    print(f"Queueing {num_packets} packets for Vegas flow (h3 -> h4)...")
    h3 = net.hosts["h3_vegas"]; h4 = net.hosts["h4_recv"]
    for _ in range(num_packets): h3.send_random_packet(h4)

    # --- Run Simulation ---
    print(f"Running simulation for {duration} ticks...")
    for i in range(duration):
        if i > 0 and i % 100 == 0: # Print progress update every 100 ticks
             reno_cwnd = net.hosts["h1_reno"].tcp.window_size
             vegas_cwnd = net.hosts["h3_vegas"].tcp.window_size
             print(f"  Tick {i}/{duration} : Reno CWND={reno_cwnd:.1f}, Vegas CWND={vegas_cwnd:.1f}")
        elif i == 0: print(f"  Tick 0/{duration}")
        net.step() # Run one tick of the simulation for all devices
    print("Simulation finished.")

    # --- Collect Data ---
    print("Collecting data logs from hosts...")
    dataset = []
    total_logs = 0
    # Collect the data_log which now contains simplified features
    for host_ip, host_obj in net.hosts.items():
        if host_ip in ["h1_reno", "h3_vegas"]: # Only collect from active senders
             print(f"  - Found {len(host_obj.data_log)} log entries from {host_ip} ({host_obj.tcp_variant})")
             dataset.extend(host_obj.data_log)
             total_logs += len(host_obj.data_log)
             
    print(f"Total log entries collected: {total_logs}")
    if total_logs == 0:
        print("\n--- WARNING: No data logged. Check simulation/logging logic. ---")
        return # Exit if no data

    # --- Save Data to CSV ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Define fieldnames - MUST match SIMPLIFIED keys logged in host.py ---
    fieldnames = [
        "timestamp", 
        "cwnd_before_adj", 
        "ack_received", 
        "timeout_occurred", 
        "loss_indicator",
        "target_cwnd_after_adj" # Target column
    ]

    print(f"\nSaving dataset for HostML training to '{OUTPUT_FILENAME}'...")
    try:
        with open(OUTPUT_FILENAME, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Check if dataset is empty before writing rows
            if dataset:
                # Verify keys in first row match fieldnames (optional but good practice)
                if not all(key in dataset[0] for key in fieldnames):
                     print("--- ERROR: Keys in dataset do not match fieldnames! ---")
                     print(f"Fieldnames: {fieldnames}")
                     print(f"Dataset[0] Keys: {list(dataset[0].keys())}")
                else:
                     writer.writerows(dataset)
            else:
                print("Warning: Dataset is empty, CSV file will only contain header.")
        print("Dataset saved successfully.")
    except IOError as e: 
        print(f"Error saving dataset: {e}")
    except KeyError as e: 
         print(f"\n--- ERROR: Log Key Mismatch during CSV Write ---")
         print(f"Data log entry missing expected key: '{e}'.")
         print("Check keys logged in host.py vs 'fieldnames' here.")
         print("First few data entries:")
         for i in range(min(5, len(dataset))): print(dataset[i])
         print("--------------------------------")

if __name__ == "__main__":
    generate_data()