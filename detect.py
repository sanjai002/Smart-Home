from dataclasses import field
import sys
import cv2
import numpy as np
import serial
import serial.tools.list_ports
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QMessageBox
import json
import requests
from websocket import WebSocket
import threading
import time
import telegram

bot = telegram.Bot(token='7162908022:AAGjz5T2hS_mvIZl3kUqocVlg3edg3rdr24')



api_url = "https://api.thingspeak.com/update?api_key=CROD93GHWL1RFN38"

global field7


class StreamThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, ws):
        super(StreamThread, self).__init__()
        self.ws = ws

    def run(self):
        try:
            while True:
                result = self.ws.recv()
                imgnp = np.array(bytearray(result), dtype=np.uint8)
                frame_t = cv2.imdecode(imgnp, -1)
                frame = cv2.resize(frame_t, (640, 480))  # Adjust the size as needed
                self.frame_ready.emit(frame)
        
        except Exception as e:
            print("WebSocket error:", e)

    

class ESPInterface(QWidget):
    def __init__(self):
        super(ESPInterface, self).__init__()
        self.count=0
        self.serial_ports = QComboBox()
        self.refresh_ports()

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_serial)

        self.serial_status_label = QLabel("Serial Status: Not Connected")
        self.flag=False
        self.ip_label = QLabel("IP Address of ESP32CAM: ")
        self.ip_address_entry = QLineEdit()
        self.ip_address_entry.setPlaceholderText("Enter ESP32CAM IP Address")
        self.ip_address_entry.textChanged.connect(self.validate_ip)

        self.start_stream_button = QPushButton("Start Streaming")
        self.start_stream_button.clicked.connect(self.start_streaming)
        self.start_stream_button.setEnabled(False)

        self.stop_stream_button = QPushButton("Stop Streaming")
        self.stop_stream_button.clicked.connect(self.stop_streaming)
        self.stop_stream_button.setEnabled(False)

        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.placeholder_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        self.show_placeholder_image()

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.serial_ports)
        self.layout.addWidget(self.connect_button)
        self.layout.addWidget(self.serial_status_label)
        self.layout.addWidget(self.ip_label)
        self.layout.addWidget(self.ip_address_entry)
        self.layout.addWidget(self.start_stream_button)
        self.layout.addWidget(self.stop_stream_button)
        self.layout.addWidget(self.video_frame)
        self.setLayout(self.layout)

        self.serial = None
        self.streaming = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)


        labelsPath = "obj.names"
        self.LABELS = open(labelsPath).read().strip().split("\n")

        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

        self.thresholdval = 0.3
        self.confidenceval = 0.5

        weightsPath = "yolov3_custom_final.weights"
        configPath = "yolov3_custom.cfg"

        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.ws = None
        self.stream_thread = None
        self.frame = None
        self.avg_boxes = []
        self.avg_confidences = []
        self.no_person_counter = 0  # Counter for consecutive frames without a person


    def refresh_ports(self):
        self.serial_ports.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.serial_ports.addItems(ports)

    def connect_serial(self):
        port_name = self.serial_ports.currentText()
        try:
            if self.serial is not None and self.serial.isOpen():
                self.serial.close()
            self.serial = serial.Serial(port_name, 115200)
            self.serial_status_label.setText(f"Serial Status: Connected to {port_name}")
            self.ip_label.setText(f"IP Address of ESP32CAM: {self.get_ip_from_esp32cam()}")
            self.start_stream_button.setEnabled(True)
        except Exception as e:
            self.serial_status_label.setText(f"Serial Status: Error - {str(e)}")
            self.start_stream_button.setEnabled(False)
            self.stop_stream_button.setEnabled(False)

    def get_ip_from_esp32cam(self):
        # Add logic to retrieve the IP from ESP32CAM
        return "192.168.1.1"

    def validate_ip(self):
        ip_address = self.ip_address_entry.text().strip()
        valid_ip = self.is_valid_ip(ip_address)
        self.start_stream_button.setEnabled(valid_ip)

    def is_valid_ip(self, ip):
        # Add validation logic for IP address
        # You may use a regular expression or any other method to validate the entered IP
        return True  # Placeholder; replace with actual validation logic

    def start_streaming(self):
        if self.serial is not None and self.serial.isOpen():
            try:
                self.ws = WebSocket()
                self.ws.connect("ws://esp32.local")
                print("Connected to WebSocket server")
                self.timer.start(30)
                self.start_stream_button.setEnabled(False)
                self.stop_stream_button.setEnabled(True)
                self.streaming = True

                self.stream_thread = StreamThread(self.ws)
                self.stream_thread.frame_ready.connect(self.set_frame)
                self.stream_thread.start()
            except Exception as e:
                print("WebSocket connection error:", e)
                self.show_error_message("Unable to connect to ESP32CAM. Check the IP address and try again.")

    def stop_streaming(self):
        if self.streaming:
            self.timer.stop()
            self.start_stream_button.setEnabled(True)
            self.stop_stream_button.setEnabled(False)
            self.streaming = False

            # Set the frame to a white image
            self.frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            self.display_frame(self.frame)

            if self.ws is not None:
                self.ws.close()

            if self.stream_thread is not None:
                self.stream_thread.quit()
                self.stream_thread.wait()

    def update_frame(self):
        try:
            if self.frame is not None:
                # Perform object detection on the frame
                boxes, confidences, classIDs = self.perform_object_detection(self.frame)

                if len(boxes) == 0:
                    # If no person is detected, increment the counter
                    self.no_person_counter += 1
                else:
                    # Reset the counter if a person is detected
                    self.no_person_counter = 0

                # Check if the counter has reached 5 (indicating 5 consecutive frames without a person)
                if self.no_person_counter >= 5:
                    # Send signal 4 to the serial port
                    self.flag=False
                    self.send_signal_to_serial(4)
                    print("No person detected for 5 consecutive frames. Sending signal 4 to serial port.")

                # Store the results of each frame
                self.avg_boxes.extend(boxes)
                self.avg_confidences.extend(confidences)

                # If we have collected results for 10 frames
                if len(self.avg_boxes) >= 5:
                    # Convert lists to numpy arrays
                    avg_boxes_np = np.array(self.avg_boxes)
                    avg_confidences_np = np.array(self.avg_confidences)

                    # Reshape arrays for averaging
                    avg_boxes_np = avg_boxes_np.reshape(-1, 4)
                    avg_confidences_np = avg_confidences_np.reshape(-1)

                    # Average the results
                    avg_confidence = np.mean(avg_confidences_np)

                    # Check if a person is detected based on average confidence
                    person_detected = avg_confidence > self.confidenceval
                    print(avg_confidence)
                    # Send appropriate signal to serial port
                    if person_detected:
                        if field7==True and self.flag==False :
                            self.send_signal_to_serial(4)
                            image_path = f'img{self.count}.jpg'
                            cv2.imwrite(image_path, self.frame) 
                            try:
                                with open(image_path, 'rb') as image_file:
                                    bot.sendPhoto(chat_id='688027432', photo=image_file)
                                    self.flag=True
                            except Exception as e:
                                    print("Error sending photo via Telegram:", e)

                        else:
                            self.send_signal_to_serial(1)

                    # Reset the averages
                    self.avg_boxes = []
                    self.avg_confidences = []

                # Draw boxes on the frame
                self.draw_boxes(self.frame, boxes, confidences, classIDs)

                # Display the frame
                self.display_frame(self.frame)

        except Exception as e:
            print("Update frame error:", e)
    
    def send_signal_to_serial(self, signal):
        try:
            if self.serial is not None and self.serial.isOpen():
                self.serial.write((str(signal) + '\n').encode())
                print(f"Sent signal to serial port: {signal}")
        except Exception as e:
            print("Error sending signal to serial port:", e)

    def set_frame(self, frame):
        self.frame = frame

    def perform_object_detection(self, frame):
        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.confidenceval:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidenceval, self.thresholdval)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])

                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return boxes, confidences, classIDs

    def draw_boxes(self, frame, boxes, confidences, classIDs):
        # Drawing boxes already handled in perform_object_detection
        pass

    def display_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_frame.setPixmap(QPixmap.fromImage(q_image))

    def show_placeholder_image(self):
        placeholder_q_image = QImage(self.placeholder_image.data, 640, 480, 1920, QImage.Format_RGB888)
        self.video_frame.setPixmap(QPixmap.fromImage(placeholder_q_image))

    def show_error_message(self, message):
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Warning)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.exec_()



    def read_serial():
        global field7
        field7=False
        try:
            field1 = field2 = field3 = field4 = field5 = field6 = "0"
            global api_url
            while True:
                if main_window.serial is not None and main_window.serial.isOpen():
                    while main_window.serial.in_waiting:
                        try:
                            data = main_window.serial.readline().decode('utf-8').strip()
                            print("Received data:", data)
                            json_data = json.loads(data)
                            field1 = str(json_data.get('Energy1', field1))
                            field2 = str(json_data.get('Energy2', field2))
                            field3 = str(json_data.get('Energy3', field3))
                            field4 = str(json_data.get('EnergyFan', field4))
                            field5 = str(json_data.get('humidity', field5))
                            field6 = str(json_data.get("temperature", field6))
                            field7 = json_data.get('security',field7)
                            print(field1,field2,field3,field4,field5,field6,field7)
                            api_url = api_url+"&field1="+field1+"&field2="+field2+"&field3="+field3+"&field4="+field4+"&field5="+field5+"&field6="+field6
                        except json.JSONDecodeError as e:
                            print("Error decoding JSON:", e)
                            pass
        except:
            pass
        


class TaskScheduler:
    def __init__(self):
        self.last_execution_time = time.time()
        self.timer = threading.Timer(20, self.data_upload)
        self.timer.start()

    def data_upload(self):
        current_time = time.time()
        if current_time - self.last_execution_time >= 10:
            try:
                response = requests.get(api_url)
                if response.status_code == 200:
                    print("Data sent to ThingSpeak successfully.")
                else:
                    print(f"Failed to send data to ThingSpeak. Status code: {response.status_code}")
            except Exception as e:
                print("Error sending data to ThingSpeak:", e)
            self.last_execution_time = current_time

        # Restart the timer
        self.timer = threading.Timer(20, self.data_upload)
        self.timer.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ESPInterface()
    main_window.show()
    thread = threading.Thread(target=ESPInterface.read_serial)
    thread.start()
    scheduler = TaskScheduler()
    sys.exit(app.exec_())
