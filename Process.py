import json
import csv
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
import torch

IMAGE_SIZE = (224, 224)
MIN_SIGN_SIZE = 24


def resize_with_padding(img):
    h, w = img.shape[:2]
    scale = min(IMAGE_SIZE[0] / w, IMAGE_SIZE[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    delta_w = IMAGE_SIZE[0] - new_w
    delta_h = IMAGE_SIZE[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0))


def process_dataset(base_dir, output_dir, dataset_name, split='train'):
    processors = {
        'gtsrb-german-traffic-sign': process_gtsrb,
        'lisa-road-sign': process_lisa,
        'Mapillary': process_mapillary,
        'CURE-TSD': process_cure_tsd,
        'roboflow-traffic-signs-dataset': process_roboflow
    }
    return processors[dataset_name](base_dir, output_dir, split) if dataset_name in processors else []


def process_gtsrb(base_dir, output_dir, split='train'):
    base_dir = Path(base_dir) / 'versions' / '1'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    CLASS_MAPPING = {
        '0': 'speed_limit',       # 20 km/h
        '1': 'speed_limit',       # 30 km/h
        '2': 'speed_limit',       # 50 km/h
        '3': 'speed_limit',       # 60 km/h
        '4': 'speed_limit',       # 70 km/h
        '5': 'speed_limit',       # 80 km/h
        '6': 'other',             # End of 80 km/h limit
        '7': 'speed_limit',       # 100 km/h
        '8': 'speed_limit',       # 120 km/h
        '9': 'no_overtaking',     # No overtaking (general)
        '10': 'no_overtaking',    # No overtaking (trucks)
        '11': 'priority_road',    # Priority road
        '12': 'priority_road',    # Priority at next intersection
        '13': 'yield',            # Yield
        '14': 'stop',             # Stop
        '15': 'no_vehicles',      # No vehicles
        '16': 'goods_vehicles',   # Trucks prohibited
        '17': 'no_entry',         # No entry
        '18': 'other',            # General caution
        '19': 'curve',            # Dangerous left curve
        '20': 'curve',            # Dangerous right curve
        '21': 'curve',            # Double curve
        '22': 'bump',             # Bumpy road
        '23': 'slippery_road',    # Slippery road
        '24': 'warning',          # Road narrows on right
        '25': 'road_work',        # Road work
        '26': 'warning',          # Traffic signals
        '27': 'pedestrian_crossing', # Pedestrian crossing
        '28': 'school_zone',      # Children crossing
        '29': 'bicycle_crossing', # Bicycle crossing
        '30': 'slippery_road',    # Beware of ice/snow
        '31': 'wild_animals',     # Wild animals crossing
        '32': 'other',            # End of all restrictions
        '33': 'turn_right',       # Turn right ahead
        '34': 'turn_left',        # Turn left ahead
        '35': 'ahead_only',       # Ahead only
        '36': 'directional',      # Go straight or right
        '37': 'directional',      # Go straight or left
        '38': 'keep_right',       # Keep right
        '39': 'keep_left',        # Keep left
        '40': 'roundabout',       # Roundabout
        '41': 'no_overtaking',    # End of no overtaking
        '42': 'no_overtaking',    # End of no overtaking (trucks)
        "__default__": "other"    # Default for unmatched classes
    }

    csv_file = 'Train.csv' if split == 'train' else 'Test.csv'
    csv_path = base_dir / csv_file

    if not csv_path.exists():
        csv_path = base_dir / csv_file.lower()
        if not csv_path.exists():
            return records

    with open(csv_path, 'r') as f:
        data = list(csv.DictReader(f))

    for row in tqdm(data, desc=f'Processing GTSRB {split}', unit='img'):
        img_rel_path = row['Path']

        if split == 'train':
            parts = img_rel_path.split('/')
            if len(parts) < 3: continue
            img_path = base_dir / 'Train' / parts[1] / parts[2]
        else:
            parts = img_rel_path.split('/')
            if len(parts) < 2: continue
            img_path = base_dir / 'Test' / parts[1]

        if not img_path.exists(): continue
        img = cv2.imread(str(img_path))
        if img is None: continue

        try:
            x1 = int(row.get('Roi.X1', row.get('roi.x1', 0)))
            y1 = int(row.get('Roi.Y1', row.get('roi.y1', 0)))
            x2 = int(row.get('Roi.X2', row.get('roi.x2', 0)))
            y2 = int(row.get('Roi.Y2', row.get('roi.y2', 0)))
            if x2 <= x1 or y2 <= y1: continue

            sign = img[y1:y2, x1:x2]
            padded = resize_with_padding(sign)

            class_id = row['ClassId']
            unified_class = CLASS_MAPPING.get(class_id, CLASS_MAPPING["__default__"])

            save_path = output_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(save_path), padded)

            records.append({
                "source": "gtsrb",
                "image_path": str(save_path),
                "original_class": f"Class_{class_id}",
                "unified_class": unified_class
            })
        except Exception:
            pass

    return records


def process_lisa(base_dir, output_dir, split='train'):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    CLASS_MAPPING = {
        0: 'directional',           # addedLane
        1: 'curve',                 # curveLeft
        2: 'curve',                 # curveRight
        3: 'bump',                  # dip
        4: 'no_entry',              # doNotEnter
        5: 'no_overtaking',         # doNotPass
        6: 'warning',               # intersection
        7: 'keep_right',            # keepRight
        8: 'warning',               # laneEnds
        9: 'warning',               # merge
        10: 'no_left_turn',         # noLeftTurn
        11: 'no_right_turn',        # noRightTurn
        12: 'pedestrian_crossing',  # pedestrianCrossing
        13: 'speed_limit',          # rampSpeedAdvisory20
        14: 'speed_limit',          # rampSpeedAdvisory35
        15: 'speed_limit',          # rampSpeedAdvisory40
        16: 'speed_limit',          # rampSpeedAdvisory45
        17: 'speed_limit',          # rampSpeedAdvisory50
        18: 'speed_limit',          # rampSpeedAdvisory
        19: 'directional',          # rightLaneMustTurn
        20: 'roundabout',           # roundabout
        21: 'school_zone',          # school
        22: 'speed_limit',          # schoolSpeedLimit25
        23: 'warning',              # signalAhead
        24: 'warning',              # slow
        25: 'speed_limit',          # speedLimit15
        26: 'speed_limit',          # speedLimit25
        27: 'speed_limit',          # speedLimit30
        28: 'speed_limit',          # speedLimit35
        29: 'speed_limit',          # speedLimit40
        30: 'speed_limit',          # speedLimit45
        31: 'speed_limit',          # speedLimit50
        32: 'speed_limit',          # speedLimit55
        33: 'speed_limit',          # speedLimit65
        34: 'speed_limit',          # speedLimit
        35: 'stop',                 # stop
        36: 'warning',              # stopAhead
        37: 'directional',          # thruMergeLeft
        38: 'directional',          # thruMergeRight
        39: 'directional',          # thruTrafficMergeLeft
        40: 'speed_limit',          # truckSpeedLimit55
        41: 'turn_left',            # turnLeft
        42: 'turn_right',           # turnRight
        43: 'yield',                # yield
        44: 'warning',              # yieldAhead
        45: 'warning',              # zoneAhead25
        46: 'warning'               # zoneAhead45
    }

    images_dir = base_dir / split / 'images'
    labels_dir = base_dir / split / 'labels'

    if not images_dir.exists() or not labels_dir.exists():
        return records

    for img_path in tqdm(list(images_dir.glob('*')), desc=f"Processing LISA {split}", unit='img'):
        img = cv2.imread(str(img_path))
        if img is None: continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists(): continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        img_height, img_width = img.shape[:2]

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5: continue

            try:
                class_id = int(parts[0])
                if class_id not in CLASS_MAPPING: continue
                unified_class = CLASS_MAPPING[class_id]

                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = max(0, int(x_center - width / 2))
                y1 = max(0, int(y_center - height / 2))
                x2 = min(img_width, int(x_center + width / 2))
                y2 = min(img_height, int(y_center + height / 2))

                if x2 <= x1 or y2 <= y1 or (x2 - x1) < MIN_SIGN_SIZE or (y2 - y1) < MIN_SIGN_SIZE:
                    continue

                sign = img[y1:y2, x1:x2]
                sign = cv2.resize(sign, IMAGE_SIZE)

                save_path = output_dir / f"{img_path.stem}_{idx}.png"
                cv2.imwrite(str(save_path), sign)

                records.append({
                    "source": "lisa",
                    "image_path": str(save_path),
                    "original_class": f"Class_{class_id}",
                    "unified_class": unified_class
                })
            except Exception:
                pass

    return records


def process_mapillary(base_dir, output_dir, split='train'):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    CLASS_MAPPING = {
        "speed-limit": "speed_limit",
        "speed-limit-zone": "speed_limit",
        "minimum-speed-limit": "speed_limit",
        "stop": "stop",
        "yield": "yield",
        "give-way": "yield",
        "no-entry": "no_entry",
        "no-parking": "no_parking",
        "no-stopping": "no_stopping",
        "no-overtaking": "no_overtaking",
        "no-left-turn": "no_left_turn",
        "no-right-turn": "no_right_turn",
        "no-u-turn": "no_u_turn",
        "priority-road": "priority_road",
        "one-way": "one_way",
        "weight-limit": "goods_vehicles",
        "pedestrian-crossing": "pedestrian_crossing",
        "children-crossing": "school_zone",
        "bicycle-crossing": "bicycle_crossing",
        "animal-crossing": "wild_animals",
        "slippery-road": "slippery_road",
        "curve-left": "curve",
        "curve-right": "curve",
        "double-curve": "curve",
        "bump": "bump",
        "dip": "bump",
        "hump": "bump",
        "road-narrows": "warning",
        "road-work": "road_work",
        "traffic-signals": "warning",
        "railway-crossing": "railway_crossing",
        "roundabout": "roundabout",
        "keep-right": "keep_right",
        "keep-left": "keep_left",
        "turn-left": "turn_left",
        "turn-right": "turn_right",
        "ahead-only": "ahead_only",
        "go-straight": "ahead_only",
        "go-straight-or-right": "directional",
        "go-straight-or-left": "directional",
        "parking": "parking",
        "bus-stop": "bus_stop",
        "tram-stop": "bus_stop",
        "rest-area": "rest_area",
        "__default__": "other"
    }

    fully_ann_dir = base_dir / 'mtsd_fully_annotated_annotation' / 'mtsd_v2_fully_annotated'
    partial_ann_dir = base_dir / 'mtsd_partially_annotated_annotation' / 'mtsd_v2_partially_annotated'

    if split == 'train':
        fully_img_dirs = [base_dir / f'mtsd_fully_annotated_images.train.{i}' / 'images' for i in range(3)]
        partial_img_dirs = [base_dir / f'mtsd_partially_annotated_images.train.{i}' / 'images' for i in range(4)]
    elif split == 'val':
        fully_img_dirs = [base_dir / 'mtsd_fully_annotated_images.val' / 'images']
        partial_img_dirs = [base_dir / 'mtsd_partially_annotated_images.val' / 'images']
    elif split == 'test':
        fully_img_dirs = [base_dir / 'mtsd_fully_annotated_images.test' / 'images']
        partial_img_dirs = [base_dir / 'mtsd_partially_annotated_images.test' / 'images']
    else:
        return records

    for dataset_type, ann_dir, img_dirs in [
        ('fully', fully_ann_dir, fully_img_dirs),
        ('partial', partial_ann_dir, partial_img_dirs)
    ]:
        if not ann_dir.exists():
            continue

        split_file = ann_dir / 'splits' / f'{split}.txt'
        annotations_dir = ann_dir / 'annotations'

        if not split_file.exists() or not annotations_dir.exists():
            print(f"Annotation directory does not exist: {annotations_dir}")
            continue

        valid_img_dirs = [d for d in img_dirs if d.exists()]
        if not valid_img_dirs:
            print(f"Image directory does not exist: {img_dirs}")
            continue

        with open(split_file, 'r') as f:
            image_keys = [line.strip() for line in f]

        image_key_to_path = {}
        for img_dir in valid_img_dirs:
            for img_path in img_dir.glob('*.jpg'):
                image_key_to_path[img_path.stem] = img_path

        for image_key in tqdm(image_keys, desc=f"Processing Mapillary {dataset_type} {split}"):
            img_path = image_key_to_path.get(image_key)
            ann_path = annotations_dir / f"{image_key}.json"

            if (not img_path or not img_path.exists()):
                continue
            elif not ann_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            with open(ann_path, 'r') as f:
                data = json.load(f)

            for obj in data.get('objects', []):
                bbox = obj.get('bbox', {})
                if not bbox or 'cross_boundary' in bbox:
                    continue

                x1 = max(0, int(bbox.get('xmin', 0)))
                y1 = max(0, int(bbox.get('ymin', 0)))
                x2 = min(img.shape[1], int(bbox.get('xmax', 0)))
                y2 = min(img.shape[0], int(bbox.get('ymax', 0)))

                if (x2 <= x1 or y2 <= y1 or
                        (x2 - x1) < MIN_SIGN_SIZE or
                        (y2 - y1) < MIN_SIGN_SIZE):
                    continue

                sign = img[y1:y2, x1:x2]
                sign = cv2.resize(sign, IMAGE_SIZE)

                save_path = output_dir / f"{dataset_type}_{image_key}_{x1}_{y1}.png"
                cv2.imwrite(str(save_path), sign)

                original_label = obj.get('label', 'unknown')
                sign_type = original_label.split('--')[1] if '--' in original_label else original_label

                if any(char.isdigit() for char in sign_type) and "speed" in sign_type:
                    sign_type = "speed-limit"

                unified_class = CLASS_MAPPING.get(sign_type, CLASS_MAPPING["__default__"])

                records.append({
                    "source": f"mapillary_{dataset_type}",
                    "image_path": str(save_path),
                    "original_class": original_label,
                    "unified_class": unified_class
                })

    return records


def process_cure_tsd(base_dir, output_dir, split='train'):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    CLASS_MAPPING = {
        '01': 'speed_limit',
        '02': 'goods_vehicles',
        '03': 'no_overtaking',
        '04': 'no_stopping',
        '05': 'no_parking',
        '06': 'stop',
        '07': 'bicycle_crossing',
        '08': 'bump',
        '09': 'no_left_turn',
        '10': 'no_right_turn',
        '11': 'priority_road',
        '12': 'no_entry',
        '13': 'yield',
        '14': 'parking',
        "__default__": "other"
    }

    test_sequences = {
        '01_04', '01_05', '01_06', '01_07', '01_08', '01_18', '01_19', '01_21',
        '01_24', '01_26', '01_31', '01_38', '01_39', '01_41', '01_47', '02_02',
        '02_04', '02_06', '02_09', '02_12', '02_13', '02_16', '02_17', '02_18',
        '02_20', '02_22', '02_28', '02_31', '02_32', '02_36'
    }

    data_dir = base_dir / 'data'
    labels_dir = base_dir / 'labels'

    if not data_dir.exists() or not labels_dir.exists():
        print(f"CURE-TSD directories not found: {data_dir} or {labels_dir}")
        return records

    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA acceleration available: {cuda_available}")

    gpu_resizer = None
    if cuda_available:
        try:
            gpu_resizer = cv2.cuda_Resize(IMAGE_SIZE[0], IMAGE_SIZE[1], interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Failed to initialize GPU resizer: {e}")
            cuda_available = False

    ann_files = {}
    for ann_file in labels_dir.glob('*.txt'):
        base_id = "_".join(ann_file.stem.split('_')[:2])
        ann_files[base_id] = ann_file

    video_files = list(data_dir.glob('*.mp4'))
    if not video_files:
        print(f"No video files found in {data_dir}")
        return records

    for video_path in tqdm(video_files, desc="Processing CURE-TSD videos"):
        video_parts = video_path.stem.split('_')
        if len(video_parts) < 2:
            continue
        base_seq_id = f"{video_parts[0]}_{video_parts[1]}"

        is_test = base_seq_id in test_sequences
        if (split == 'test' and not is_test) or (split == 'train' and is_test):
            continue

        ann_file = ann_files.get(base_seq_id)
        if not ann_file or not ann_file.exists():
            continue

        frame_to_annots = {}
        try:
            with open(ann_file, 'r') as f:
                next(f)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('_')
                    if len(parts) < 10:
                        continue

                    frame_num = int(parts[0]) - 1
                    if frame_num not in frame_to_annots:
                        frame_to_annots[frame_num] = []
                    frame_to_annots[frame_num].append(line)
        except Exception as e:
            print(f"Error parsing {ann_file}: {str(e)}")
            continue

        if not frame_to_annots:
            continue

        if cuda_available:
            try:
                cuda_reader = cv2.cudacodec.createVideoReader(str(video_path))
                if not cuda_reader.nextFrame():
                    print(f"Failed to open video with CUDA: {video_path}")
                    continue
            except Exception as e:
                print(f"CUDA video reader failed: {e}")
                cuda_available = False
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"Failed to open video: {video_path}")
                    continue
        else:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not cuda_available else -1
        processed_frames = 0

        for frame_idx, annotations in frame_to_annots.items():
            if not cuda_available and (frame_idx < 0 or frame_idx >= total_frames):
                continue

            if cuda_available:
                cuda_reader.set(cv2.cudacodec.VideoReaderProps.PROP_POS_FRAMES, frame_idx)
                ret, gpu_frame = cuda_reader.nextFrame()
                if not ret:
                    continue
                frame = gpu_frame.download()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

            processed_frames += 1

            for ann in annotations:
                parts = ann.split('_')
                if len(parts) < 10:
                    continue

                sign_type = parts[1]
                unified_class = CLASS_MAPPING.get(sign_type, CLASS_MAPPING["__default__"])
                if unified_class == "other":
                    continue

                try:
                    coords = list(map(int, parts[2:10]))
                    xs = [coords[0], coords[2], coords[4], coords[6]]
                    ys = [coords[1], coords[3], coords[5], coords[7]]
                    xmin, ymin = min(xs), min(ys)
                    xmax, ymax = max(xs), max(ys)

                    if (xmax - xmin) < MIN_SIGN_SIZE or (ymax - ymin) < MIN_SIGN_SIZE:
                        continue

                    sign = frame[ymin:ymax, xmin:xmax]

                    if cuda_available:
                        gpu_sign = cv2.cuda_GpuMat()
                        gpu_sign.upload(sign)

                        resized = gpu_resizer.apply(gpu_sign)

                        h, w = resized.size()

                        delta_w = IMAGE_SIZE[0] - w
                        delta_h = IMAGE_SIZE[1] - h
                        top = delta_h // 2
                        bottom = delta_h - top
                        left = delta_w // 2
                        right = delta_w - left

                        padded = cv2.cuda.copyMakeBorder(
                            resized,
                            top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0))

                        result = padded.download()
                    else:
                        result = resize_with_padding(sign)

                        save_path = output_dir / f"{video_path.stem}_f{frame_idx + 1}_{xmin}_{ymin}.png"
                        cv2.imwrite(str(save_path), result)

                        records.append({
                            "source": "cure_tsd",
                            "image_path": str(save_path),
                            "original_class": sign_type,
                            "unified_class": unified_class
                        })
                except Exception as e:
                    print(f"Error processing annotation: {ann} in {video_path.name} frame {frame_idx}: {str(e)}")
                    continue

        if cuda_available:
            pass
        else:
            cap.release()

        if processed_frames == 0:
            print(f"No valid frames processed for {video_path.name}")

    return records


def process_roboflow(base_dir, output_dir, split='train'):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    CLASS_MAPPING = {
        0: 'warning',               # -Road narrows on right
        1: 'speed_limit',           # 50 mph speed limit
        2: 'warning',               # Attention Please-
        3: 'school_zone',           # Beware of children
        4: 'bicycle_crossing',      # CYCLE ROUTE AHEAD WARNING
        5: 'curve',                 # Dangerous Left Curve Ahead
        6: 'curve',                 # Dangerous Rright Curve Ahead
        7: 'warning',               # End of all speed and passing limits
        8: 'yield',                 # Give Way
        9: 'directional',           # Go Straight or Turn Right
        10: 'directional',          # Go straight or turn left
        11: 'keep_left',            # Keep-Left
        12: 'keep_right',           # Keep-Right
        13: 'warning',              # Left Zig Zag Traffic
        14: 'no_entry',             # No Entry
        15: 'no_overtaking',        # No_Over_Taking
        16: 'no_overtaking',        # Overtaking by trucks is prohibited
        17: 'pedestrian_crossing',  # Pedestrian Crossing
        18: 'roundabout',           # Round-About
        19: 'slippery_road',        # Slippery Road Ahead
        20: 'speed_limit',          # Speed Limit 20 KMPh
        21: 'speed_limit',          # Speed Limit 30 KMPh
        22: 'stop',                 # Stop_Sign
        23: 'ahead_only',           # Straight Ahead Only
        24: 'warning',              # Traffic_signal
        25: 'goods_vehicles',       # Truck traffic is prohibited
        26: 'turn_left',            # Turn left ahead
        27: 'turn_right',           # Turn right ahead
        28: 'bump'                  # Uneven Road
    }

    images_dir = base_dir / split / 'images'
    labels_dir = base_dir / split / 'labels'

    if not images_dir.exists() or not labels_dir.exists():
        return records

    for img_path in tqdm(list(images_dir.glob('*.*')), desc=f"Processing Roboflow {split}", unit='img'):
        img = cv2.imread(str(img_path))
        if img is None: continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists(): continue

        img_height, img_width = img.shape[:2]

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5: continue

            try:
                class_id = int(parts[0])
                if class_id not in CLASS_MAPPING: continue
                unified_class = CLASS_MAPPING[class_id]

                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = max(0, int(x_center - width / 2))
                y1 = max(0, int(y_center - height / 2))
                x2 = min(img_width, int(x_center + width / 2))
                y2 = min(img_height, int(y_center + height / 2))

                if x2 <= x1 or y2 <= y1 or (x2 - x1) < MIN_SIGN_SIZE or (y2 - y1) < MIN_SIGN_SIZE:
                    continue

                sign = img[y1:y2, x1:x2]
                padded = resize_with_padding(sign)

                save_path = output_dir / f"{img_path.stem}_{idx}.png"
                cv2.imwrite(str(save_path), padded)

                records.append({
                    "source": "roboflow",
                    "image_path": str(save_path),
                    "original_class": f"Class_{class_id}",
                    "unified_class": unified_class
                })
            except Exception:
                pass

    return records


def save_metadata(records, output_path):
    if not records: return
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'source', 'original_class', 'unified_class'])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved metadata for {len(records)} images to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Traffic Sign Dataset Preprocessor')
    parser.add_argument('--base_dir', default='./Datasets', help='Base directory for datasets')
    parser.add_argument('--output_dir', default='./processed', help='Output directory for processed data')
    parser.add_argument('--datasets', nargs='+', default=['CURE-TSD', 'gtsrb-german-traffic-sign', 'lisa-road-sign',
                                                          'roboflow-traffic-signs-dataset', 'Mapillary'],
                        choices=['gtsrb-german-traffic-sign', 'lisa-road-sign', 'CURE-TSD',
                                 'roboflow-traffic-signs-dataset', 'Mapillary'], help='Datasets to process')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'], help='Splits to process')

    args = parser.parse_args()

    for split in args.splits:
        (Path(args.output_dir) / split / 'images').mkdir(parents=True, exist_ok=True)

    total_images = 0
    for split in args.splits:
        records = []
        output_dir = Path(args.output_dir) / split / 'images'

        for dataset in args.datasets:
            dataset_records = process_dataset(
                base_dir=Path(args.base_dir) / dataset,
                output_dir=output_dir,
                dataset_name=dataset,
                split=split
            )
            records.extend(dataset_records)
            print(f"{dataset} {split}: {len(dataset_records)} images processed")

        save_metadata(records, Path(args.output_dir) / split / 'metadata.csv')
        total_images += len(records)

    print(f"\nTotal images processed: {total_images}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    main()