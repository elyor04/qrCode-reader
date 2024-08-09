import sys
import cv2 as cv
from numpy import ndarray
from random import randint
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

colors: dict[str, tuple[int, int, int]] = dict()


def visualize_box_and_labels(
    image: ndarray, decoded_info: tuple, points: ndarray
) -> ndarray:
    imHg, imWd = image.shape[:2]

    for name, p in zip(decoded_info, points.astype(int)):
        xmin, ymin = p[0, 0], p[0, 1]
        xmax, ymax = p[2, 0], p[2, 1]
        if not name:
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            continue

        if name in colors:
            color = colors[name]
        else:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            while (True not in [(pxl > 210) for pxl in color]) or (
                color in colors.values()
            ):
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
            colors[name] = color
            print(name)

        gts = cv.getTextSize(name, cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, 1)
        gtx = gts[0][0] + xmin
        gty = gts[0][1] + ymin

        cv.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        image[
            max(ymin - 5, 0) : min(gty + 5, imHg), max(xmin - 3, 0) : min(gtx + 3, imWd)
        ] = color
        cv.putText(
            image, name, (xmin, gty), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0), 1
        )

    return image


class QRCodeReaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.cam = cv.VideoCapture(0)
        self.qcd = cv.QRCodeDetector()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def initUI(self):
        self.setWindowTitle("QR-Code Reader")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)

    def update_frame(self):
        ret, img = self.cam.read()
        if not ret:
            return

        img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        imGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret_qr, decoded_info, points, _ = self.qcd.detectAndDecodeMulti(imGray)
        if ret_qr:
            img = visualize_box_and_labels(img, decoded_info, points)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width, channel = img.shape
        step = channel * width
        qImg = QImage(img.data, width, height, step, QImage.Format.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(qImg))

    def closeEvent(self, event):
        self.cam.release()
        cv.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = QRCodeReaderApp()
    ex.show()
    sys.exit(app.exec())
