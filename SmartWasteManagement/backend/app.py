from flask import Flask, request, jsonify, render_template
from database import init_db, log_data, get_recent_data, get_all_data
from model import predict_time_to_full
import os

app = Flask(__name__)

# BIN CONFIGURATION
BIN_HEIGHT_CM = 30.0  # Adjust based on your actual bin

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['POST'])
def receive_data():
    try:
        data = request.json
        distance = float(data.get('distance'))
        
        # Calculate fill percentage
        # Distance is from top. so 0 distance = 100% full. 
        # BIN_HEIGHT distance = 0% full.
        
        fill_val = BIN_HEIGHT_CM - distance
        if fill_val < 0: fill_val = 0
        
        fill_percentage = (fill_val / BIN_HEIGHT_CM) * 100
        if fill_percentage > 100: fill_percentage = 100
        
        log_data(distance, round(fill_percentage, 2))
        
        return jsonify({"status": "success", "fill": fill_percentage}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/status', methods=['GET'])
def get_status():
    recent = get_recent_data(1)
    if not recent:
        return jsonify({
            "current_fill": 0,
            "prediction": "No Data",
            "last_updated": "Never"
        })
        
    current = recent[0]
    
    # Get all data for AI prediction
    all_data = get_all_data()
    prediction = predict_time_to_full(all_data[-50:]) # Use last 50 points
    
    return jsonify({
        "current_fill": current['fill_percentage'],
        "prediction": prediction,
        "last_updated": current['timestamp']
    })

if __name__ == '__main__':
    if not os.path.exists('waste_management.db'):
        init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
