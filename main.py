import cv2
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. TRAFFIC LIGHT STATE FROM DATASET FOLDER
# ============================================================

def infer_light_state(video_path):
    """Infer the fixed traffic-light state from the video's parent folder.

    Supported folder names:
      - dendo1, dendo2       -> RED
      - denxanh1, denxanh 2  -> GREEN

    Spaces, underscores, and hyphens are ignored when matching names.
    """
    red_folders = {"dendo1", "dendo2"}
    green_folders = {"denxanhhannam", "vt1denxanh"}

    for parent in Path(video_path).resolve().parents:
        normalized = "".join(
            ch for ch in parent.name.lower()
            if ch.isalnum()
        )

        if normalized in red_folders:
            return "RED"
        if normalized in green_folders:
            return "GREEN"

    raise ValueError(
        f"Cannot infer traffic-light state from folder: {video_path}. "
        "Expected a parent folder named Dendo1, dendo2, "
        "denxanh_hannam, or vt1_denxanh. You can also set --light-state manually."
    )


# ============================================================
# 2. INTERACTIVE STOP LINE SELECTION
# ============================================================

def select_stop_line(frame, json_path, video_path):
    """
    Let the user select the two endpoints of the stop line.
    Left click: select endpoint.
    R: reset selection.
    Enter: confirm and save top_line.json.
    Q / ESC: cancel.
    """
    points = []
    window_name = "Select Stop Line"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((int(x), int(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    selected_line = None

    while True:
        preview = frame.copy()

        # Draw selected points and line preview
        for index, point in enumerate(points):
            cv2.circle(preview, point, 7, (0, 255, 255), -1)
            cv2.putText(
                preview, f"P{index + 1}",
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2
            )

        if len(points) == 2:
            cv2.line(
                preview, points[0], points[1],
                (0, 255, 255), 3
            )

        instructions = [
            "Click 2 points to define the stop line",
            "R: reset | ENTER: confirm and save | Q: cancel"
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(
                preview, instruction, (20, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('r'), ord('R')):
            points.clear()

        elif key in (10, 13):  # Enter / Return
            if len(points) == 2:
                selected_line = points.copy()
                break
            print("Please click exactly two points before confirming.")

        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyWindow(window_name)

    if selected_line is None:
        return None

    data = {
        "video": str(video_path),
        "line_start": list(selected_line[0]),
        "line_end": list(selected_line[1])
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"Stop line saved to: {json_path}")
    print(f"Start: {selected_line[0]} | End: {selected_line[1]}")

    return tuple(selected_line[0]), tuple(selected_line[1])


# ============================================================
# 3. LINE CROSSING DETECTION
# ============================================================

def signed_distance(point, line_start, line_end):
    """
    Determine which side of the stop line the vehicle is on.
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1

    return dx * (py - y1) - dy * (px - x1)


def check_line_crossing(
    previous_point,
    current_point,
    line_start,
    line_end,
    direction
):
    """
    Detect whether a vehicle crosses the stop line.

    direction:
        positive_to_negative
        negative_to_positive
        both
    """

    d1 = signed_distance(
        previous_point, line_start, line_end
    )

    d2 = signed_distance(
        current_point, line_start, line_end
    )

    if direction == "positive_to_negative":
        crossed = d1 > 0 and d2 <= 0

    elif direction == "negative_to_positive":
        crossed = d1 < 0 and d2 >= 0

    else:
        crossed = (
            (d1 > 0 and d2 <= 0) or
            (d1 < 0 and d2 >= 0)
        )

    if not crossed:
        return False

    # Check whether the crossing occurs on the finite line segment
    denominator = d1 - d2

    if abs(denominator) < 1e-9:
        return False

    t = d1 / denominator

    x = previous_point[0] + t * (
        current_point[0] - previous_point[0]
    )
    y = previous_point[1] + t * (
        current_point[1] - previous_point[1]
    )

    x1, y1 = line_start
    x2, y2 = line_end

    line_dx = x2 - x1
    line_dy = y2 - y1

    length_squared = line_dx**2 + line_dy**2

    if length_squared == 0:
        return False

    projection = (
        (x - x1) * line_dx +
        (y - y1) * line_dy
    ) / length_squared

    return 0 <= projection <= 1


# ============================================================
# 3. MAIN DETECTION PIPELINE
# ============================================================

def main(args):

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # Select stop line interactively on the first frame.
    top_line_path = output_path.parent / "top_line.json"

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        writer.release()
        raise RuntimeError("Cannot read the first frame of the input video.")

    selected_line = select_stop_line(
        first_frame, top_line_path, args.video
    )

    if selected_line is None:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print("Stop line selection cancelled. Detection was not started.")
        return

    line_start, line_end = selected_line

    # Restart video from the first frame after line selection.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read the dataset label from the video folder name.
    # dendo1/dendo2 -> RED; denxanh1/denxanh 2 -> GREEN.
    light_roi = select_light_roi(first_frame)

    print("Traffic light detection initialized.")

    # Tracking history
    track_history = {}

    # Avoid reporting the same track multiple times
    processed_tracks = set()

    frame_id = 0
    events = []

    vehicle_classes = {
        1: "BICYCLE",
        2: "CAR",
        3: "MOTORCYCLE",
        5: "BUS",
        7: "TRUCK"
    }

    print("Starting vehicle tracking and stop-line crossing detection...")

    PROCESS_FPS = 10

    frame_interval = max(1, round(fps / PROCESS_FPS))

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        if frame_id % frame_interval != 0:
            continue

        # ----------------------------------------------------
        # A. TRAFFIC LIGHT STATE (FROM DATASET FOLDER LABEL)
        # ----------------------------------------------------

        light_state = detect_light_state(frame, light_roi)

        # ----------------------------------------------------
        # B. YOLO + BYTETRACK
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[1, 2, 3, 5, 7],
            conf=0.05,
            imgsz=960,
            verbose=False
        )

        annotated = results[0].plot()

        # Draw stop line
        cv2.line(
            annotated,
            line_start,
            line_end,
            (255, 255, 0),
            3
        )

        # ----------------------------------------------------
        # C. PROCESS TRACKED VEHICLES
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            # Kiểm tra YOLO có phát hiện xe máy, xe đạp không
            if frame_id % 30 == 0:
                print(f"\n--- Frame {frame_id} ---")

                for class_id, confidence in zip(classes, confidences):
                    print(
                        f"{model.names[class_id]} "
                        f"(ID={class_id}) - Confidence: {confidence:.3f}"
                    )

            # Chỉ lấy ID sau khi đã kiểm tra detection
            if result.boxes.id is None:
                continue

            ids = result.boxes.id.cpu().numpy().astype(int)

            for box, track_id, class_id in zip(
                boxes, ids, classes
            ):

                if class_id not in vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Bottom-center point of the bounding box
                center_x = (x1 + x2) // 2
                bottom_y = y2

                current_point = (center_x, bottom_y)

                # ------------------------------------------------
                # D. TRACKING HISTORY
                # ------------------------------------------------

                previous_point = track_history.get(track_id)

                if previous_point is not None:

                    if track_id not in processed_tracks:

                        crossed = check_line_crossing(
                            previous_point,
                            current_point,
                            line_start,
                            line_end,
                            args.direction
                        )

                        if crossed:

                            violation = (
                                light_state == "RED"
                            )

                            event = {
                                "frame_id": frame_id,
                                "timestamp": round(
                                    frame_id / fps, 3
                                ),
                                "vehicle_id": track_id,
                                "vehicle_type": vehicle_classes[class_id],
                                "light_state": light_state,
                                "violation": violation,
                                "x": center_x,
                                "y": bottom_y
                            }

                            events.append(event)
                            processed_tracks.add(track_id)

                            print(
                                f"[EVENT] Vehicle {track_id} | "
                                f"Light: {light_state} | "
                                f"Violation: {violation}"
                            )

                track_history[track_id] = current_point

                # ------------------------------------------------
                # E. DRAW TRACKING RESULTS
                # ------------------------------------------------

                color = (0, 255, 0)

                label = (
                    f"{vehicle_classes[class_id]} "
                    f"ID:{track_id}"
                )

                if track_id in processed_tracks:

                    event = next(
                        (
                            e for e in reversed(events)
                            if e["vehicle_id"] == track_id
                        ),
                        None
                    )

                    if event and event["violation"]:
                        color = (0, 0, 255)
                        label += " VIOLATION"

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.circle(
                    annotated,
                    current_point,
                    5,
                    (255, 0, 0),
                    -1
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        writer.write(annotated)

        if args.show:
            cv2.imshow("Red Light Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ========================================================
    # 4. EXPORT EVENTS TO CSV
    # ========================================================

    csv_path = output_path.with_name(
        output_path.stem + "_events.csv"
    )

    fieldnames = [
        "frame_id",
        "timestamp",
        "vehicle_id",
        "vehicle_type",
        "light_state",
        "violation",
        "x",
        "y"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        csv_writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        csv_writer.writeheader()
        csv_writer.writerows(events)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nDetection completed!")
    print(f"Output video: {output_path}")
    print(f"Event CSV: {csv_path}")
    print(f"Total crossing events: {len(events)}")
    print(
        "Red-light violations:",
        sum(e["violation"] for e in events)
    )

# ============================================================
# TRAFFIC LIGHT DETECTION USING HSV
# ============================================================

def select_light_roi(frame):
    """
    Select the traffic light region manually.
    The selected ROI will be used throughout the video.
    """

    roi = cv2.selectROI(
        "Select Traffic Light ROI",
        frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow("Select Traffic Light ROI")

    x, y, w, h = map(int, roi)

    if w == 0 or h == 0:
        raise ValueError("Traffic light ROI selection cancelled.")

    print(f"Traffic light ROI: x={x}, y={y}, w={w}, h={h}")

    return x, y, w, h


def detect_light_state(frame, roi):
    """
    Detect RED or GREEN traffic light using HSV color thresholding.
    """

    x, y, w, h = roi

    light_crop = frame[y:y+h, x:x+w]

    if light_crop.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(
        light_crop,
        cv2.COLOR_BGR2HSV
    )

    # RED color ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # GREEN color range
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([90, 255, 255])

    # Create color masks
    red_mask1 = cv2.inRange(
        hsv, lower_red1, upper_red1
    )

    red_mask2 = cv2.inRange(
        hsv, lower_red2, upper_red2
    )

    red_mask = cv2.bitwise_or(
        red_mask1, red_mask2
    )

    green_mask = cv2.inRange(
        hsv, lower_green, upper_green
    )

    # Count detected pixels
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    print(
        f"Red pixels: {red_pixels} | "
        f"Green pixels: {green_pixels}"
    )

    # Minimum number of colored pixels
    MIN_PIXELS = 2

    if red_pixels < MIN_PIXELS and green_pixels < MIN_PIXELS:
        return "UNKNOWN"

    if red_pixels > green_pixels:
        return "RED"

    if green_pixels > red_pixels:
        return "GREEN"

    return "UNKNOWN"

# ============================================================
# 5. CONFIGURATION
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        # required=True,
        default="real_traffic/1108.mov",
        help="Input video path (must be inside a labeled dataset folder)"
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path"
    )

    parser.add_argument(
        "--output",
        default="output/red_light_result.mp4"
    )

    parser.add_argument(
        "--direction",
        choices=[
            "positive_to_negative",
            "negative_to_positive",
            "both"
        ],
        default="positive_to_negative"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--show",
        action="store_true"
    )

    args = parser.parse_args()

    main(args)

