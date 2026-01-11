import requests
import time
import random
import sys

URL = "http://127.0.0.1:5000/api/data"
BIN_HEIGHT = 30.0

def simulate():
    print(f"Simulating fill-up... Sending data to {URL}")
    current_distance = BIN_HEIGHT # Empty
    
    while True:
        try:
            # Simulate bin filling up (distance decreases)
            current_distance -= random.uniform(0.5, 2.0) 
            if current_distance < 0:
                current_distance = BIN_HEIGHT # Reset if full
                print("\n--- Bin Emptied ---\n")
                
            payload = {"distance": round(current_distance, 2)}
            response = requests.post(URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Sent: {payload['distance']}cm | Server: Fill {data['fill']}%")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Connection Error: {e}")
            print("Make sure Flask app is running! (cd backend && python app.py)")
            
        time.sleep(2)

if __name__ == "__main__":
    simulate()
