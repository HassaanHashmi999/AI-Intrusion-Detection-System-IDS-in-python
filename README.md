# AI Intrusion Detection System (IDS)

This project comprises four components aimed at creating an AI-based Intrusion Detection System (IDS) for network security. Each component serves a specific function in the system. Below is an overview of each component along with its functionalities.

## Components

### 1. AI Model (AI_Model.py)

This Python script contains code for training and evaluating machine learning models on network traffic data. It includes the following functionalities:
- Data preprocessing: Loading, analyzing, and preprocessing the training and testing datasets.
- Feature selection: Using Recursive Feature Elimination (RFE) to select the best features for model training.
- Model training: Training various classifiers such as Decision Tree, Random Forest, XGBoost, etc., using the selected features.
- Hyperparameter optimization: Utilizing Optuna for optimizing hyperparameters of the Decision Tree classifier.
- Model evaluation: Evaluating the trained model's accuracy on the test dataset and saving the best-performing model.

### 2. Attacker (Attacker.py)

This Python script simulates a network attacker by crafting and sending malicious TCP packets. It includes the following functionalities:
- Creating malicious TCP packets with specified characteristics such as source/destination IP, port numbers, flags, etc.
- Sending crafted packets to the target destination IP address using the Scapy library.

### 3. IDS Server (IDS_Server.py)

This Python script acts as the server-side component of the IDS. It continuously monitors network traffic, analyzes incoming packets, and detects potential intrusions. Key functionalities include:
- Sniffing packets on the network interface using Scapy.
- Processing captured packets and extracting relevant features.
- Making predictions on incoming packets using the trained machine learning model.
- Logging packet details and intrusion predictions to a CSV file for further analysis.

### 4. Traffic Control (Traffic_Control.py)

This Python script serves as a control mechanism for the IDS system. It captures network traffic in real-time and selectively forwards packets to the IDS server for analysis. Key functionalities include:
- Using PyShark to capture live network traffic on a specified interface.
- Filtering captured packets based on predefined criteria (e.g., destination port).
- Forwarding filtered packets to the IDS server for intrusion detection.

## Usage

1. Ensure all required Python libraries are installed (`Scapy`, `PyShark`, `Optuna`, `Pandas`, `Numpy`, `Joblib`, `Matplotlib`, `Seaborn`, `LightGBM`, `XGBoost`).
2. Run each component in the following sequence:
   - Start the AI Model script to train the machine learning model and save it.
   - Run the Traffic Control script to capture live network traffic.
   - Execute the Attacker script to simulate malicious network activity.
   - Run the IDS Server script to analyze captured packets and detect intrusions.

## Notes

- This IDS system serves as a demonstration and may require further refinement for production use.
- Ensure proper permissions and network configurations before running the scripts, especially for capturing live network traffic.

## Disclaimer

- Usage of this IDS system should comply with applicable laws and ethical guidelines.
- The developers are not liable for any misuse or unauthorized access resulting from the use of this system.


Feel free to enhance, modify, or extend the functionality of this IDS system according to your requirements and feedback.
