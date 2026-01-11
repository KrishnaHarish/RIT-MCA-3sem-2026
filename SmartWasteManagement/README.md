# Smart Waste Management System 🗑️🤖

A complete AIoT solution for monitoring waste levels and predicting when bins need emptying.
**Tech Stack:** NodeMCU (ESP8266), Flask (Python), SQLite, Linear Regression (AI), HTML/CSS/JS.

## 📁 Project Structure
- **backend/**: Contains the Python Flask server, Database, and AI Model.
- **firmware/**: Contains the C++ code for your NodeMCU.
- **simulate_device.py**: A script to test the system without hardware.

## 🚀 How to Run

### 1. Setup Environment
Open a terminal in the `SmartWasteManagement` folder:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r backend\requirements.txt
```

### 2. Start the Backend Server
```bash
cd backend
python app.py
```
*The server will start at `http://localhost:5000`*

### 3. Open the Dashboard
Open your browser and go to: [http://localhost:5000](http://localhost:5000)

### 4. Feed Data
**Option A: Using Hardware (NodeMCU)**
1. Open `firmware/smart_bin/smart_bin.ino` in Arduino IDE.
2. Edit `ssid`, `password`, and `serverUrl` (use your PC's IP).
3. Flash to NodeMCU.

**Option B: Using Simulator (No Hardware)**
Open a new terminal:
```bash
# In SmartWasteManagement folder
.\venv\Scripts\activate
python simulate_device.py
```
*You will see the dashboard update in real-time!*

## 🤖 AI Prediction features
The system uses **Linear Regression** to analyze the rate of filling.
- As the bin fills up, the "Time to Full" prediction will become more accurate.
- If the bin is being emptied, it detects the slope and resets status.
