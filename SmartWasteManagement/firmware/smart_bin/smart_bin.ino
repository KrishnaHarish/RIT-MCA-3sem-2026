#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>

// ---------------- USER CONFIG ----------------
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
// REPLACE WITH YOUR PC'S IP ADDRESS (Run 'ipconfig' to find it)
const char* serverUrl = "http://192.168.1.100:5000/api/data"; 

// ---------------- PINS ----------------
#define TRIG_PIN D1
#define ECHO_PIN D2

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected, IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  long duration;
  float distance;

  // Clear trig
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  
  // Trigger
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // Read Echo
  duration = pulseIn(ECHO_PIN, HIGH);
  
  // Calculate distance (Sound speed = 0.034 cm/us)
  distance = duration * 0.034 / 2;
  
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // Send to Server
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;
    
    http.begin(client, serverUrl);
    http.addHeader("Content-Type", "application/json");
    
    String jsonPayload = "{\"distance\": " + String(distance) + "}";
    
    int httpResponseCode = http.POST(jsonPayload);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println(httpResponseCode);
      Serial.println(response);
    } else {
      Serial.print("Error on sending POST: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  } else {
    Serial.println("WiFi Disconnected");
  }

  delay(2000); // Send data every 2 seconds
}
