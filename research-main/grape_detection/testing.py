import cv2
import numpy as np
import tensorflow as tf
import os

# Define constants
TFLITE_MODEL_PATH = "tomato.tflite"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
CLASS_LABELS_FILE = "class_labels.txt"  # Path to your class labels file

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]["shape"]

# Create a ClassLabelMapper class to map class IDs to labels
class ClassLabelMapper:
    def __init__(self, labels_file):
        self.labels_file = labels_file
        self.class_labels = self.load_labels()

    def load_labels(self):
        try:
            with open(self.labels_file, "r") as file:
                class_labels = [line.strip() for line in file]
            return class_labels
        except FileNotFoundError:
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")

    def get_label(self, class_id):
        if class_id < 0 or class_id >= len(self.class_labels):
            return "Unknown"
        return self.class_labels[class_id]

# Create an instance of the ClassLabelMapper
class_label_mapper = ClassLabelMapper(CLASS_LABELS_FILE)

# Open a video capture from the webcam (change the parameter to specify a different camera)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = input_data / 255.0  # Normalize the input data
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Assuming input_data is your input data of type FLOAT64
    input_data = input_data.astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output
    output = interpreter.get_tensor(output_details[0]["index"])

    # Process the output and display results
    class_id = np.argmax(output)
    confidence = output[0, class_id]

    # Get the class label based on the class_id
    class_label = class_label_mapper.get_label(class_id)

    # Display the class label and confidence score on the frame
    label = f"Class: {class_label}, Confidence: {confidence:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Real-time Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
