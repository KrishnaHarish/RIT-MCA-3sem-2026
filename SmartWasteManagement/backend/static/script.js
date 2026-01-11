function updateDashboard() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            const fill = data.current_fill;
            const pred = data.prediction;
            const lastUpdated = data.last_updated;

            // Update Gauge Text
            document.getElementById('gauge-text').innerText = Math.round(fill) + '%';
            
            // Update Gauge Rotation
            const rotation = (fill / 100) * 180;
            const gaugeFill = document.getElementById('gauge-fill');
            gaugeFill.style.transform = `rotate(${rotation}deg)`;
            gaugeFill.style.top = '0'; // Snap to top once rotated

            // Update Color based on fill
            if (fill < 50) {
                gaugeFill.style.backgroundColor = 'var(--success)';
                document.getElementById('status-text').innerText = "Normal";
            } else if (fill < 80) {
                gaugeFill.style.backgroundColor = 'var(--warning)';
                document.getElementById('status-text').innerText = "Filling Up";
            } else {
                gaugeFill.style.backgroundColor = 'var(--danger)';
                document.getElementById('status-text').innerText = "Critical / Full";
            }

            // Update Prediction
            document.getElementById('prediction-text').innerText = pred;

            // Update Time
            if (lastUpdated !== 'Never') {
              const date = new Date(lastUpdated);
              document.getElementById('update-time').innerText = date.toLocaleTimeString();
            }
        })
        .catch(error => console.error('Error fetching data:', error));
}

// Poll every 2 seconds
setInterval(updateDashboard, 2000);
updateDashboard(); // Initial call
