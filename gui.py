import sys
import numpy as np
import tensorflow.lite as tflite
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL import Image

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="fashion_mnist_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names for Fashion MNIST dataset
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class ImageClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fashion MNIST Classifier")
        self.setGeometry(100, 100, 400, 500)

        # Layout
        self.layout = QVBoxLayout()

        # Image Label
        self.image_label = QLabel("Drag and Drop an Image Here", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Prediction Label
        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

        # Enable drag-and-drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.load_image(file_path)
            self.classify_image(file_path)

    def load_image(self, file_path):
        """ Loads the dropped image into the GUI """
        pixmap = QPixmap(file_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def classify_image(self, file_path):
        """ Preprocesses and classifies the dropped image """
        image = Image.open(file_path).convert("L").resize((28, 28))
        image = np.array(image, dtype=np.float32) / 255.0
        image = image.reshape(1, 28, 28, 1)

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = np.argmax(interpreter.get_tensor(output_details[0]['index']))

        self.result_label.setText(f"Predicted Class: {class_names[prediction]}")


# Run the GUI
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec())
