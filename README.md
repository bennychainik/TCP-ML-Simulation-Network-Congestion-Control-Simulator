# TCP Congestion Control - A Machine Learning Approach

# TCP Congestion Control Simulation (Reno, Vegas, ML)

## Overview

This project simulates Congestion Control Simulation (Reno, Vegas, ML)

## Overview

This project simulates and compares classic TCP congestion control algorithms (Reno, Vegas) alongside a custom Machine Learning-based approach within a discrete-time Python network simulator. The simulation focuses on congestion window (`cwnd`) dynamics and performance metrics under network congestion induced by limited router buffer capacity.

*Note: A custom simulator was developed due to platform compatibility issues encountered with Mininet (missing kernel modules in WSL2, instability in dual-boot Linux).*

## Features Implemented

*   **Custom Network Simulator:** Python classes for Host, Router, Packet, Network. Congestion via buffer drops.
*   **TCP Reno:** AIMD, Slow Start, Fast Ret and compares standard TCP congestion control algorithms (Reno, Vegas) with a basic Machine Learning (Polynomial Regression) based controller. It uses a custom, discrete-time Python simulator, developed due to platform challenges encountered with Mininet (WSL2/macOS). Congestion is simulated via router buffer drops.

## Technologies Used

*   Python 3
*   NumPy
*   Pandas
*   Scikit-learn
*   Matplotlib

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bennychainik/TCP-ML-Simulation-Network-Congestion-Control-Simulator.git 
    cd TCP-ML-Simulation-Network-Congestion-Control-Simulator
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Using a virtual environment is recommended)*

## How to Run

1.  **Generate Training Data (for ML Model - optional, pre-generated included):**
    ```bash
    python src/generate_dataset.py 
    ```
    *(Creates `data/ml_host_training_data.csv`)*

2.  **Train ML Model (optional, pre-trained included):**
    ```bash
    python src/train.py
    ```
    *(Creates `model/ml_host_pipeline.pkl`)*

3.  **Run Baseline Comparison (Reno vs Vegas):**
    ```bash
    python src/comparision.py
    ```
    *(Outputs performance metrics table and shows CWND/RTT plots)*

4.  **Run ML Host Simulation (uses retrained model):**
    ```bash
    python src/run_ml_simulation.py
    ```
    *(Outputs performance metrics table and shows CWND/RTT plots)*

## Key Findings

*   The custom simulator provided a platform to implement and observe TCP variant behaviors.
*ransmit/Recovery, Timeout.
*   **TCP Vegas:** RTT-based adjustments (`alpha`/`beta`), Slow Start, Timeout.
*   **ML Controller:** Uses a trained Polynomial Regression model (`model/ml_host_pipeline.pkl`) to predict `cwnd` based on simple features (ACKs, Timeouts). Trained on data generated from Reno/Vegas runs (`data/ml_host_training_data.csv`).
*   **Performance Logging:** Tracks CWND, RTT, ACKs received, and Timeouts for analysis.

## Technologies Used

*   Python 3
*   NumPy
*   Pandas
*   Scikit-learn
*   Matplotlib

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bennychainik/TCP-ML-Simulation-Network-Congestion-Control-Simulator.git
    ```
2.  **Navigate to the directory:**
    ```bash
    cd TCP-ML-Simulation-Network-Congestion-Control-Simulator
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure you have `pip` for Python 3)*

## How to Run

The trained model (`model/ml_host_pipeline.pkl`) and the dataset it was trained on (`data/ml_host_training_data.csv`) are included.

1.  **Run Baseline Comparison (Reno vs Vegas):**
    ```bash
    python src/comparision.py
    ```
    *(This script runs the simulation and shows performance metrics + plots for Reno and Vegas.)*

2.  **Run ML Host Simulation:**
    ```bash
    python src/run_ml_simulation.py
    ```
    *(This script runs the simulation using the pre-trained ML model for hosts, shows metrics + plots.)*

3.  **(Optional) Regenerate Data & Retrain Model:**
    ```bash
    # 1. Generate fresh training data (over   The **retrained** ML model (using simple features: ACKs/Timeouts) successfully learned a stable, AIMD-like control pattern, avoiding the instability seen in initial tests.
*   In this simulation, the retrained ML hosts achievedwrites existing CSV)
    python src/generate_dataset.py 
    # 2. Train the model (overwrites existing PKL)
    python src/train.py 
    ```

## Key Results Summary

*   The retrained ML model learned a stable AIMD-like pattern using only simple ACK/timeout signals.
*    **low loss rates comparable to TCP Vegas** and significantly better than TCP Reno, although they converged to a lower final throughput.
*   Simulator limitations (constant measured RTT) likely impacted the proactive behaviour of TCP Vegas.
