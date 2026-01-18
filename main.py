import os
import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime
from src.ocr import paddle_ocr
from src.storage import save_json
from src.detector import detect_plates

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ["FLAGS_use_mkldnn_bfloat16"] = "0"
os.environ["CPU_NUM"] = "4"
print("✅ MKLDNN FLAGS SET:", os.environ["FLAGS_use_mkldnn"])

className = ["License"]





if __name__ == "__main__":
    cap = cv2.VideoCapture("data/carLicence4.mp4")

    startTime = datetime.now()
    license_plates = set()
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        print(f"Frame Number: {count}")

        results = detect_plates(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = paddle_ocr(frame, x1, y1, x2, y2)
                if label:
                    license_plates.add(label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame, label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        if (datetime.now() - startTime).seconds >= 20:
            save_json(license_plates, startTime, datetime.now())
            license_plates.clear()
            startTime = datetime.now()

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    cap.release()
    cv2.destroyAllWindows()

    
