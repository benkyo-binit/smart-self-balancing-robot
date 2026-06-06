import cv2
import numpy as np
import requests
import time
import threading
import serial
from RPLCD.i2c import CharLCD
import speech_recognition as sr
import pyttsx3
import random

# === Initialize LCD ===
lcd = CharLCD('PCF8574', 0x27)  # Adjust I2C address if needed
lcd.clear()

# === Initialize Serial Communication with Arduino ===
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Default baud rate

# === Initialize Speech Recognition and TTS ===
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# List of emotions
emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# List of objects to detect and track (5 objects)
object_labels = ["book", "whiteboard", "bottle", "phone", "face"]

# === Generate Random Emotion ===
def generate_random_emotion():
    return random.choice(emotion_labels)

# === Send Animation Command to Arduino ===
def send_animation_command(emotion):
    try:
        if emotion == "Happy":
            arduino.write("HAPPY_ANIMATION\n".encode())
        elif emotion == "Sad":
            arduino.write("SAD_ANIMATION\n".encode())
        elif emotion == "Angry":
            arduino.write("ANGRY_ANIMATION\n".encode())
        elif emotion == "Neutral":
            arduino.write("NEUTRAL_ANIMATION\n".encode())
        elif emotion == "Surprise":
            arduino.write("SURPRISE_ANIMATION\n".encode())
    except Exception as e:
        print(f"Error sending animation command to Arduino: {e}")

# === Send Balancing Command to Arduino ===
def send_balancing_command(object_type):
    try:
        if object_type == "book":
            arduino.write("BALANCE_BOOK\n".encode())
        elif object_type == "whiteboard":
            arduino.write("BALANCE_WHITEBOARD\n".encode())
        elif object_type == "bottle":
            arduino.write("BALANCE_BOTTLE\n".encode())
        elif object_type == "phone":
            arduino.write("BALANCE_PHONE\n".encode())
        elif object_type == "face":
            arduino.write("BALANCE_FACE\n".encode())
    except Exception as e:
        print(f"Error sending balancing command to Arduino: {e}")

# === Q&A Bot with Voice Input/Output ===
def qna_bot():
    try:
        # Capture voice input
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        # Convert speech to text
        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")

        # Get response from Gemini API
        response = requests.post("https://api.gemini.com/qna", json={"query": query}, timeout=5)
        if response.ok:
            answer = response.json().get("answer", "No answer available")
            print(f"Bot: {answer}")

            # Convert answer to speech
            engine.say(answer)
            engine.runAndWait()
            return answer
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"Error in Q&A bot: {e}")
    return "Error fetching response"

# === Object Detection ===
def object_detection(frame, net, output_layer_names):
    try:
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (160, 160), (0, 0, 0), True, crop=False)  # Smaller input size
        net.setInput(blob)
        layer_outputs = net.forward(output_layer_names)

        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and object_labels[class_id] in object_labels:  # Detect only specific objects
                    center_x, center_y, w, h = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return [(boxes[i], class_ids[i], confidences[i]) for i in indices.flatten()]
    except Exception as e:
        print(f"Error in object detection: {e}")
        return []

# === Main Code ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load YOLOv3-Tiny model for object detection
try:
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    if net.empty():
        print("Error: Could not load YOLOv3-Tiny model.")
        exit()
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except Exception as e:
    print(f"Error loading YOLOv3-Tiny model: {e}")
    exit()

# Initialize tracker
tracker = None
tracking = False
tracked_object = None

# Variables for emotion display
current_emotion = None
emotion_start_time = None

# FPS calculation variables
fps_start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.resize(frame, (320, 240))  # Lower resolution

    # Object Detection (every 5 frames)
    frame_count += 1
    if frame_count % 5 == 0 or not tracking:
        detections = object_detection(frame, net, output_layers)
        if len(detections) > 0:
            # Select the first detected object to track
            box, class_id, confidence = detections[0]
            x, y, w, h = box
            bbox = (x, y, w, h)
            tracker = cv2.TrackerKCF_create()  # Use KCF tracker
            tracker.init(frame, bbox)
            tracked_object = object_labels[class_id]
            tracking = True

            # Send balancing command to Arduino
            send_balancing_command(tracked_object)

    # Object Tracking
    if tracking:
        success, bbox = tracker.update(frame)
        if success:
            # Draw bounding box and label
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, tracked_object, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # If the tracked object is a face, display random emotion
            if tracked_object == "face":
                if current_emotion is None or (time.time() - emotion_start_time) >= 5:
                    current_emotion = generate_random_emotion()
                    emotion_start_time = time.time()

                    # Display emotion on LCD
                    lcd.clear()
                    lcd.write_string(f"Emotion: {current_emotion}")

                    # Send animation command to Arduino
                    send_animation_command(current_emotion)

                # Draw emotion on the frame
                cv2.putText(frame, current_emotion, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Tracking failed, reset
            tracking = False
            tracked_object = None

    # FPS Calculation
    fps_end_time = time.time()
    fps = 1 / (fps_end_time - fps_start_time)
    fps_start_time = fps_end_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Raspberry Pi System", frame)

    # Add delay to reduce frame rate
    time.sleep(0.2)  # 200ms delay (~5 FPS)

    # Break on Esc key
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
lcd.clear()
arduino.close()