# src/hostml.py
import pickle
import os
import math
import random
import numpy as np
import pandas as pd 

# Assuming these imports work
from . import context
from src.device import Device, Device_Type
from src.packet import Packet, Packet_Type

# --- Define the SIMPLIFIED feature columns the RETRAINED model expects ---
EXPECTED_FEATURE_COLUMNS = [
    'cwnd_before_adj', 
    'ack_received', 
    'timeout_occurred', 
    'loss_indicator',
]

class TCPML():
    """ Simplified TCP state for the ML-controlled host """
    def __init__(self):
        self.packets_to_send = list()
        self.packets_in_flight = list() # List of tuples: (packet, sent_time)
        self.window_size = 1.0 # Start with a window of 1
        self.timeout = 15      # Initial/Fallback timeout value   

class HostML(Device):
    """ Represents a Host controlled by a pre-trained ML model """
    def __init__(self, ip:str, buffer_cap=5):
        super().__init__(ip)
        self.connected_router = None
        self.outgoing_buffer = list()
        self.incoming_buffer = list()
        self.buffer_cap = buffer_cap
        self.tcp = TCPML()
        self.def_seg_no = 1
        self.clock = 0 

        # --- Performance Logging Lists ---
        self.rtt_log = [] # Stores (timestamp, rtt_value)
        self.timeout_log = [] # Stores timestamp of each timeout event
        self.ack_received_log = [] # Stores timestamp of each ACK received

        # --- Load the RETRAINED Model Pipeline ---
        model_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'model')
        pipeline_filename = 'ml_host_pipeline.pkl' # Model trained for HostML
        pipeline_path = os.path.join(model_dir, pipeline_filename)
        
        print(f"HostML {self.ip}: Loading model pipeline from: {pipeline_path}")
        try:
            with open(pipeline_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            print(f"HostML {self.ip}: Pipeline loaded successfully.")
        except FileNotFoundError:
             print(f"--- ERROR: HostML {self.ip}: Model file not found: '{pipeline_path}' ---")
             print("Please ensure the retraining train.py has run successfully.")
             raise FileNotFoundError(f"Could not load ML Host pipeline: {pipeline_path}")
        except Exception as e:
            print(f"--- ERROR: HostML {self.ip}: Error loading pipeline: {e} ---")
            raise e
            
    def link(self, other:Device):
        """ Connects this host to another device """
        self.connected_router = other
    
    def get_connected_router(self):
        """ Returns the connected router """
        return self.connected_router
    
    def device_type(self):
        """ Returns the device type """
        return Device_Type.HOST

    def send_pckt(self, pckt:Packet):
        """ Adds a packet to the list waiting to be sent """
        self.tcp.packets_to_send.append(pckt)
    
    def send_random_packet(self, to_device:Device):
        """ Creates and queues a data packet """
        pckt = Packet(self.def_seg_no, self, to_device, Packet_Type.DATA)
        self.send_pckt(pckt)
        self.def_seg_no += 1

    def receive_pckt(self, pckt:Packet):
        """ Adds a received packet to the incoming buffer """
        if len(self.incoming_buffer) < self.buffer_cap:
            self.incoming_buffer.append(pckt)
    
    def __str__(self):
        """ String representation of the HostML """
        msg = f"HostML IP: {self.ip}\r\n"
        if self.connected_router:
             msg += f"Connected to {self.connected_router.get_ip()}\r\n"
        else:
             msg += "Not connected to router.\r\n"
        return msg

    def step(self):
        """ Main logic executed at each simulation tick for the HostML """
        super().step() # Increments self.clock

        # --- State before processing ---
        cwnd_before_adj = self.tcp.window_size
        ack_received_this_step = False
        timeout_occurred_this_step = False
        
        # --- Handle incoming packets ---
        latest_ack_seg_no = -1
        indices_to_remove_inflight = []
        for pckt in self.incoming_buffer:
            if pckt.get_pckt_type() == Packet_Type.DATA:
                # Send ACK for received data
                ack_pack = Packet(pckt.get_seg_no(), pckt.get_to(), pckt.get_from(), Packet_Type.ACK)
                self.outgoing_buffer.append(ack_pack)
            
            elif pckt.get_pckt_type() == Packet_Type.ACK:
                ack_received_this_step = True 
                self.ack_received_log.append(self.clock) # <<< Log ACK reception time
                seg_no = pckt.get_seg_no()
                latest_ack_seg_no = max(latest_ack_seg_no, seg_no)
                
                # Remove acknowledged packet from packets_in_flight and estimate RTT
                found_ack_index = -1
                for i, (pkt_in_flight, sent_time) in enumerate(self.tcp.packets_in_flight):
                    if pkt_in_flight.get_seg_no() == seg_no:
                        rtt = self.clock - sent_time
                        if rtt > 0: # Only log valid RTTs
                            self.rtt_log.append((self.clock, rtt)) # <<< Log RTT
                        self.tcp.timeout = max(rtt * 2, 5) # Example adaptive timeout
                        found_ack_index = i
                        break 
                
                if found_ack_index != -1:
                    indices_to_remove_inflight.append(found_ack_index) 

        # Remove acknowledged packets
        for i in sorted(indices_to_remove_inflight, reverse=True):
            if i < len(self.tcp.packets_in_flight):
                 del self.tcp.packets_in_flight[i]
        self.incoming_buffer.clear()

        # --- Handle Timeouts ---
        timed_out_segments = []
        indices_to_remove_timeout = []
        for i in range(len(self.tcp.packets_in_flight)):
            pckt, sent_time = self.tcp.packets_in_flight[i]
            if self.clock - sent_time > self.tcp.timeout: 
                timed_out_segments.append(pckt)
                indices_to_remove_timeout.append(i)

        if len(indices_to_remove_timeout) > 0:
            timeout_occurred_this_step = True
            self.timeout_log.append(self.clock) # <<< Log Timeout Event Time
            # Add timed-out packets back to send queue
            for pckt in reversed(timed_out_segments):
                 self.tcp.packets_to_send.insert(0, pckt)
            # Remove from packets_in_flight
            for i in sorted(indices_to_remove_timeout, reverse=True):
                 if i < len(self.tcp.packets_in_flight):
                     del self.tcp.packets_in_flight[i]

        # --- Prepare Features for Prediction (Using SIMPLIFIED Features) ---
        loss_indicator_ml = 1 if timeout_occurred_this_step else 0
        current_features_dict = {
            'cwnd_before_adj': cwnd_before_adj,
            'ack_received': int(ack_received_this_step),
            'timeout_occurred': int(timeout_occurred_this_step),
            'loss_indicator': loss_indicator_ml,
        }
        
        # Create DataFrame using the RETRAINED model's expected columns list
        features_df = None
        try:
            features_df = pd.DataFrame([current_features_dict], columns=EXPECTED_FEATURE_COLUMNS)
        except ValueError as e:
             print(f"--- ERROR: HostML {self.ip} Feature Formatting ---")
             print(f"Mismatch features vs expected: {e}")
             features_df = None

        # --- Predict New Window Size using NEWLY Trained Pipeline ---
        predicted_cwnd = cwnd_before_adj # Default to previous value
        if features_df is not None: 
            try:
                # *** Potential Overflow Point ***
                predicted_cwnd_array = self.pipeline.predict(features_df)
                predicted_cwnd = predicted_cwnd_array[0] 
            except Exception as e:
                print(f"--- ERROR: HostML {self.ip} Prediction Failed: {e} ---")
                # Check if it's an overflow error specifically
                if "infinity or a value too large" in str(e):
                    print(f"    Overflow likely caused by large feature value. Input features:")
                    print(features_df) 
                predicted_cwnd = cwnd_before_adj # Fallback on error

        # --- Apply Predicted Window Size ---
        
        # Define a reasonable maximum CWND value for this simulation
        # Adjust this based on expected link capacity / buffer sizes if known, 
        # otherwise pick a large but finite number.
        MAX_CWND_CAP = 1000.0 # Example Cap - Adjust as needed! 
                              # Needs to be large enough to allow growth,
                              # but small enough to prevent float overflow.

        # Ensure window size is at least 1
        new_cwnd = max(float(predicted_cwnd), 1.0) 
        
        # *** ADD CWND CAPPING LOGIC HERE ***
        if new_cwnd > MAX_CWND_CAP:
            # Optional: Print a warning when capping occurs
            # print(f"Clock {self.clock}: HostML {self.ip} CWND prediction ({new_cwnd:.2f}) capped at {MAX_CWND_CAP}")
            self.tcp.window_size = MAX_CWND_CAP
        elif not np.isfinite(new_cwnd): # Check for NaN or infinity explicitly
             print(f"Clock {self.clock}: HostML {self.ip} Predicted CWND is not finite ({new_cwnd}). Resetting to 1.")
             self.tcp.window_size = 1.0
        else:
             self.tcp.window_size = new_cwnd
        
        # --- Send packets based on new window size ---
        allowed_to_send_count = math.floor(self.tcp.window_size) - len(self.tcp.packets_in_flight)
        packets_sent_this_step = 0
        while packets_sent_this_step < allowed_to_send_count and len(self.tcp.packets_to_send) > 0:
            pckt_to_send = self.tcp.packets_to_send.pop(0)
            self.outgoing_buffer.append(pckt_to_send)
            self.tcp.packets_in_flight.append((pckt_to_send, self.clock))
            packets_sent_this_step += 1

        if self.connected_router:
            for pckt in self.outgoing_buffer:
                self.connected_router.receive_pckt(pckt)
        self.outgoing_buffer.clear()