"""
Wildfire Detection Pipeline

Uses two YOLO models (primary and validation) to detect wildfire smoke/fire from a video stream.
Includes frame throttling, temporal confirmation logic, alert rate limiting, and email notifications with annotated images.
"""
import argparse
from datetime import datetime
import time
from pathlib import Path
import yaml
import cv2
import os
from ultralytics import YOLO  
from utils import logger, can_alert, send_email

def load_cfg(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    # Load config from YAML file
    cfg = load_cfg(Path(args.config))

    # Load and fuse primarly model (Yolo S)
    model_a = YOLO(Path("model") / "yolos.pt")
    model_a.fuse()

    # load and fuse secondary validation model (Yolo M)
    model_b = YOLO(Path("model") / "yolom.pt")
    model_b.fuse()

    # Determine source: use test video if available, else use live stream
    if "test_video" in cfg:
        video_path = Path("testvideo") / cfg["test_video"]
        cap = cv2.VideoCapture(str(video_path))
        print(video_path)
    else:
        cap = cv2.VideoCapture(cfg["stream_url"])
        if not cap.isOpened():
            logger.error("Unable to open stream.")
            return

    # Maintain frame rate based on target fps
    target_fps = 5
    interval = 1 / target_fps
    
    last_time = time.time()

    # Counters for symmetric decay with conditional reinforcement
    smoke_counter = 0
    fire_counter = 0
    # Flag used for keeping printed updates reasonable
    validation_in_progress = False

    # Main loop: get frame, detect, validate, and alert
    try:
        while True:
            ret, frame = cap.read()
            # Skip if video feed fails
            if not ret:
                logger.warning("Stream read failure.")
                time.sleep(1)
                continue
            # Skip frame if not enough time has passed since last frame (FPS)
            now = time.time()
            if now - last_time < interval:
                time.sleep(0.001)
                continue
            last_time = now

            # Run primary infrence using Primary model (YOLO S)
            results_a = model_a.predict(
                source=frame,
                imgsz=cfg["img_size"],
                conf=cfg["confidence_threshold"],
                iou=cfg["iou_threshold"],
                verbose=False
            )[0]
            detections = results_a.boxes
            
            # Decay counters if no detections from primary model then skip validation infrence (Reduce number of computations)
            if detections.shape[0] == 0:
                smoke_counter = max(smoke_counter - 3, 0)
                fire_counter = max(fire_counter - 3, 0) 
                validation_in_progress = False
                continue
                
            # Save detection names to det_cls_a
            det_cls_a = [results_a.names[int(cls)] for cls in detections.cls]
            
            # If smoke or fire detected by primary model, run secondary validation 
            if "Smoke" in det_cls_a or "Fire" in det_cls_a:
                if not validation_in_progress: # We only want to print this once per detection group
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Potential Wildfire Detected. Running validation tests...")
                    validation_in_progress = True
                # Secondary validation
                results_b = model_b.predict(
                    source=frame,
                    conf=cfg["confidence_threshold"],
                    iou=cfg["iou_threshold"],
                    verbose=False
                )[0]
                det_cls_b = [results_b.names[int(cls)] for cls in results_b.boxes.cls]
                
                # Increment or decay smoke counter based on validation model output
                if "Smoke" in det_cls_a:
                    if "Smoke" in det_cls_b:
                        smoke_counter = min(smoke_counter + 1, 20)
                    else:
                        smoke_counter = max(smoke_counter - 2, 0)
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Secondary validation for 'Smoke' returned negative — possible false positive from the primary model.")
                        print(f"Current smoke danger level: {smoke_counter}/20. Requires {smoke_counter // 2 + 1} more failed validations to fully suppress the alert.")
                        
                # Increment or decay fire counter based on validation model output
                if "Fire" in det_cls_a:
                    if "Fire" in det_cls_b:
                        fire_counter = min(fire_counter + 1, 20)
                    else:
                        fire_counter = max(fire_counter - 2, 0) 
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Secondary validation for 'Fire' returned negative — possible false positive from the primary model.")
                        print(f"Current fire danger level: {fire_counter}/20. Requires {fire_counter // 2 + 1} more failed validations to fully suppress the alert.")
            else:
                # Redundant but i felt it didnt hurt to add this block
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Detections found, but no smoke or fire.")
                smoke_counter = 0
                fire_counter = 0
                validation_in_progress = False
            # Upon fire or smoke counters reaching the specified threshold check if alert cooldown has passed
            if (smoke_counter > 5 or fire_counter > 5) and can_alert(cfg["alert_cooldown_seconds"]):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = Path("detections")
                output_path.mkdir(exist_ok=True)
                
                # Draw bounding boxes and confidence scores for confirmed detections
                for box in results_b.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = results_b.names[int(box.cls[0])]
                    conf = box.conf[0].item()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # Save annotated frame to 'detections/' folder
                cv2.imwrite(str(output_path / f"wildfire_{timestamp}.jpg"), frame)
                
                # Trigger email alert with saved image as attachment
                image_path = output_path / f"wildfire_{timestamp}.jpg"
                subject = "Wildfire Smoke Detected"
                msg = f"Smoke / Fire detected at {time.strftime('%Y-%m-%d %H:%M:%S')}."
                if "email" in cfg:
                    send_email(cfg["email"], subject, msg, image_path = str(image_path))
                validation_in_progress = False
                

    finally:
        # Release video capture and close any OpenCV windows
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
