import cv2 as cv
from numpy import ndarray
from random import randint

colors: dict[str, tuple[int, int, int]] = dict()


def visualize_box_and_labels(
    image: ndarray, decoded_info: tuple, points: ndarray
) -> ndarray:
    imHg, imWd = image.shape[:2]

    for name, p in zip(decoded_info, points.astype(int)):
        xmin, ymin = p[0, 0], p[0, 1]
        xmax, ymax = p[2, 0], p[2, 1]
        if not name:
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
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

        gts = cv.getTextSize(name, cv.FONT_HERSHEY_COMPLEX, 1.5, 2)
        gtx = gts[0][0] + xmin
        gty = gts[0][1] + ymin

        cv.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
        image[
            max(ymin - 5, 0) : min(gty + 5, imHg), max(xmin - 3, 0) : min(gtx + 3, imWd)
        ] = color
        cv.putText(image, name, (xmin, gty), cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)

    return image


cap = cv.VideoCapture(0)
qcd = cv.QRCodeDetector()

while True:
    img = cap.read()[1]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(gray)
    if ret_qr:
        visualize_box_and_labels(img, decoded_info, points)

    cv.imshow("QR-Code Reader", img)
    if cv.waitKey(2) == 27:  # esc
        break

cap.release()
cv.destroyAllWindows()
