# src/host.py
import math 
import random 

# Assuming these imports work based on your project structure
from . import context 
from src.device import Device, Device_Type
from src.packet import Packet, Packet_Type

class TCP():
    """ Holds TCP state variables for a Host """
    def __init__(self):
        self.packets_to_send = list()
        self.packets_in_flight = list() # List of tuples: (packet, sent_time)
        
        # Core TCP State
        self.window_size = 1.0 # Use float for fractional increments
        self.ssthresh = 18     # A high initial value
        self.timeout = 10      # Initial/Fallback timeout value (adaptive logic added below)

        # Reno Specific State
        self.fast_recovery = False
        self.dup_ack_count = 0 
        self.last_ack = None   # Stores the highest ACK number received

        # Vegas Specific State
        self.base_rtt = float('inf') # Minimum RTT seen
        self.current_rtt = None      # Last measured RTT
        self.alpha = 1   # Vegas threshold for increasing CWND
        self.beta = 3    # Vegas threshold for decreasing CWND

        # State captured *before* decision logic (used for ML logging)
        self.cwnd_before_decision = 1.0 
        self.current_rtt_for_decision = None
        self.base_rtt_for_decision = float('inf')
        self.loss_flag_for_decision = False
        self.dup_ack_count_for_decision = 0

class Host(Device):
    """ Represents a network host with TCP Reno/Vegas/Tahoe logic """
    def __init__(self, ip:str, buffer_cap=5, tcp_variant="tahoe"):
        super().__init__(ip)
        self.connected_router = None
        self.outgoing_buffer = list()
        self.incoming_buffer = list()
        self.buffer_cap = buffer_cap
        self.tcp = TCP()
        self.def_seg_no = 1 # Default segment number counter
        self.tcp_variant = tcp_variant  # "tahoe", "reno", or "vegas"
        self.clock = 0 # Simulation clock passed from Network/Device step
        
        # Data log used ONLY when generating data for HostML training
        self.data_log = [] 

        # --- Performance Logging Lists ---
        self.rtt_log = [] # Stores (timestamp, rtt_value)
        self.timeout_log = [] # Stores timestamp of each timeout event
        self.ack_received_log = [] # Stores timestamp of each ACK received by sender
        
    def link(self,other:Device):
        """ Connects this host to another device (typically a router) """
        self.connected_router = other
    
    def get_connected_router(self):
        """ Returns the connected router """
        return self.connected_router
    
    def device_type(self):
        """ Returns the device type """
        return Device_Type.HOST

    def send_pckt(self,pckt:Packet):
        """ Adds a packet to the list waiting to be sent. """
        self.tcp.packets_to_send.append(pckt)
    
    def send_random_packet(self,to_device:Device):
        """ Convenience method to create and queue a data packet. """
        pckt = Packet(self.def_seg_no,self,to_device,Packet_Type.DATA)
        self.send_pckt(pckt)
        self.def_seg_no = self.def_seg_no + 1

    def receive_pckt(self,pckt:Packet):
        """ Adds a received packet to the incoming buffer if space allows. """
        if len(self.incoming_buffer) < self.buffer_cap:
            self.incoming_buffer.append(pckt)
    
    def __str__(self):
        """ String representation of the Host """
        msg = f"Host IP: {self.ip}\r\n"
        if self.connected_router:
             msg += f"Connected to {self.connected_router.get_ip()}\r\n"
        else:
             msg += "Not connected to router.\r\n"
        msg += f"TCP Variant: {self.tcp_variant}\r\n"
        return msg
    
    def step(self):
        """ Main logic executed at each simulation tick for the Host. """
        # --- Start of step ---
        super().step() # Increments self.clock
        self.tcp.cwnd_before_decision = self.tcp.window_size # Capture state before changes
        
        # Flags set by events within this step
        ack_received_this_step = False
        timeout_occurred_this_step = False
        loss_detected_by_dupacks_this_step = False # Reno specific fast retransmit trigger
        
        acks_processed = 0 # Count ACKs received this step
        latest_ack_seg_no = self.tcp.last_ack if self.tcp.last_ack is not None else -1

        # --- Incoming Packet Handling ---
        indices_to_remove_inflight = []
        for pckt in self.incoming_buffer:
            if pckt.get_pckt_type() == Packet_Type.DATA:
                # Send ACK for received data packet
                ack_pack = Packet(pckt.get_seg_no(), pckt.get_to(), pckt.get_from(), Packet_Type.ACK)
                self.outgoing_buffer.append(ack_pack)
            
            elif pckt.get_pckt_type() == Packet_Type.ACK:
                acks_processed += 1
                ack_received_this_step = True
                self.ack_received_log.append(self.clock) # <<< Log ACK reception time

                seg_no = pckt.get_seg_no()
                latest_ack_seg_no = max(latest_ack_seg_no, seg_no)

                # RTT estimation and Packet Removal
                rtt_calculated = None
                acked_packet_index = -1
                for i, (pkt_in_flight, sent_time) in enumerate(self.tcp.packets_in_flight):
                    if pkt_in_flight.get_seg_no() == seg_no:
                        # Calculate RTT
                        rtt_calculated = self.clock - sent_time
                        self.tcp.current_rtt = rtt_calculated 
                        if rtt_calculated > 0: 
                            self.tcp.base_rtt = min(self.tcp.base_rtt, rtt_calculated) 
                            self.rtt_log.append((self.clock, rtt_calculated)) # <<< Log RTT
                        
                        # Adaptive Timeout Example
                        self.tcp.timeout = max(rtt_calculated * 2, 5) 
                        
                        acked_packet_index = i
                        break 
                
                # Mark acknowledged packet for removal
                if acked_packet_index != -1:
                     indices_to_remove_inflight.append(acked_packet_index)

                # Reno: Duplicate ACK Handling
                if self.tcp_variant == "reno":
                    if self.tcp.last_ack is not None and seg_no <= self.tcp.last_ack: 
                        if seg_no == self.tcp.last_ack: self.tcp.dup_ack_count += 1
                    else: # New ACK
                        self.tcp.dup_ack_count = 0
                        self.tcp.last_ack = seg_no 
                        if self.tcp.fast_recovery: # Exit Fast Recovery on new ACK
                            self.tcp.window_size = self.tcp.ssthresh 
                            self.tcp.fast_recovery = False
                    # Check for Fast Retransmit threshold
                    if self.tcp.dup_ack_count == 3:
                        loss_detected_by_dupacks_this_step = True 
                        if not self.tcp.fast_recovery: # Enter FR only once
                            self.tcp.ssthresh = max(self.tcp.window_size // 2, 2)
                            self.tcp.window_size = self.tcp.ssthresh + 3 
                            self.tcp.fast_recovery = True
                    elif self.tcp.fast_recovery and self.tcp.dup_ack_count > 3: 
                         self.tcp.window_size += 1 # Inflate window in FR
                else: # Not Reno, reset state
                    self.tcp.dup_ack_count = 0
                    self.tcp.last_ack = seg_no

        # Remove acknowledged packets from flight list (after iterating)
        for i in sorted(indices_to_remove_inflight, reverse=True):
            if i < len(self.tcp.packets_in_flight):
                del self.tcp.packets_in_flight[i]
        self.incoming_buffer.clear() # Processed all incoming packets

        # --- Timeout Handling ---
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
            # Reset Reno fast recovery state on timeout
            if self.tcp_variant == "reno":
                self.tcp.fast_recovery = False 
                self.tcp.dup_ack_count = 0

        # --- Store final state needed for decision making before applying rules ---
        self.tcp.current_rtt_for_decision = self.tcp.current_rtt 
        self.tcp.base_rtt_for_decision = self.tcp.base_rtt       
        self.tcp.loss_flag_for_decision = timeout_occurred_this_step or loss_detected_by_dupacks_this_step
        self.tcp.dup_ack_count_for_decision = self.tcp.dup_ack_count 

        # --- Congestion Control Logic ---
        cwnd_after_decision = self.tcp.cwnd_before_decision 

        if self.tcp_variant == "tahoe":
            if timeout_occurred_this_step: 
                self.tcp.ssthresh = max(self.tcp.window_size // 2, 2) 
                cwnd_after_decision = 1.0 
            elif ack_received_this_step: 
                if self.tcp.window_size < self.tcp.ssthresh: 
                    cwnd_after_decision = self.tcp.window_size + acks_processed 
                else: 
                    if self.tcp.window_size > 0: 
                         cwnd_after_decision = self.tcp.window_size + (acks_processed / self.tcp.window_size) 
        
        elif self.tcp_variant == "reno":
            if timeout_occurred_this_step: 
                self.tcp.ssthresh = max(self.tcp.window_size // 2, 2)
                cwnd_after_decision = 1.0 
                self.tcp.fast_recovery = False 
            elif loss_detected_by_dupacks_this_step: 
                 # Window adjustments handled when FR was entered/incremented
                 cwnd_after_decision = self.tcp.window_size 
            elif ack_received_this_step and not self.tcp.fast_recovery: # New ACK, not in FR
                 if self.tcp.window_size < self.tcp.ssthresh: # Slow Start
                     cwnd_after_decision = self.tcp.window_size + acks_processed
                 else: # Congestion Avoidance
                     if self.tcp.window_size > 0:
                          cwnd_after_decision = self.tcp.window_size + (acks_processed / self.tcp.window_size)

        elif self.tcp_variant == "vegas":
            if timeout_occurred_this_step: 
                self.tcp.ssthresh = max(self.tcp.window_size // 2, 2)
                cwnd_after_decision = 1.0 # Reset like others on timeout
            elif ack_received_this_step: 
                # Check RTTs only if they are valid
                if self.tcp.current_rtt is not None and self.tcp.current_rtt > 0 and \
                   self.tcp.base_rtt != float('inf') and self.tcp.base_rtt > 0:
                    expected_rate = self.tcp.window_size / self.tcp.base_rtt
                    actual_rate = self.tcp.window_size / self.tcp.current_rtt
                    queue_diff = (expected_rate - actual_rate) * self.tcp.base_rtt 
                    if queue_diff < self.tcp.alpha: 
                        cwnd_after_decision = self.tcp.window_size + 1.0 
                    elif queue_diff > self.tcp.beta: 
                        cwnd_after_decision = self.tcp.window_size - 1.0 
                else: # Fallback to Slow Start if RTTs not ready
                     cwnd_after_decision = self.tcp.window_size + acks_processed 

        # --- Apply the calculated window size ---
        self.tcp.window_size = max(cwnd_after_decision, 1.0) 
        
        # --- Log data for ML (ONLY WHEN GENERATING DATA for HostML training) ---
        # This logs the SIMPLIFIED features that HostML can know, along with the
        # target CWND decided by the actual Reno/Vegas logic above.
        if self.tcp_variant in ["reno", "vegas"]: # Check added for safety, though usually called from generate_dataset
            loss_indicator_ml = 1 if timeout_occurred_this_step else 0
            features_for_ml_host = {
                "timestamp": self.clock,
                "cwnd_before_adj": self.tcp.cwnd_before_decision,
                "ack_received": int(ack_received_this_step),
                "timeout_occurred": int(timeout_occurred_this_step), 
                "loss_indicator": loss_indicator_ml, 
            }
            target_cwnd = self.tcp.window_size 
            self.data_log.append({**features_for_ml_host, "target_cwnd_after_adj": target_cwnd})
        
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