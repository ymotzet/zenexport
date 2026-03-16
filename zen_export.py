import pickle
import numpy as np
from tqdm import tqdm
import cv2

from zendar.io.zenplay import Zenplay

# zen-cmd runpy zen_export.py


# TODO read extrinsic calibration
# TODO read vehicle name to proberly name file (timestamp is not sufficient!)
# TODO read radar pointcloud from EPC


def read_all_data(zenplay):
    data = {
        "can": {"wheel_speeds": {}, "vehicle_speed": {}},
        "gnss": [],
        "camera": [],
        "lidar": [],
    }

    # CAN
    if zenplay.vehicle_can:
        can = zenplay.vehicle_can[0]
        for i in range(can._nframes):
            if can._frames[i] is not None:
                data["can"]["wheel_speeds"].update({
                    a.host_timestamp_ns: [a.wheel_speed_fl_kmh, a.wheel_speed_fr_kmh, a.wheel_speed_rl_kmh, a.wheel_speed_rr_kmh]
                    for a in can._frames[i].wheel_speeds.measurements
                })
                data["can"]["vehicle_speed"].update({
                    a.host_timestamp_ns: a.cluster_speed_kmh for a in can._frames[i].vehicle_speed.measurements
                })

    # GNSS
    if not zenplay.navigations:
        print("No navigation data found.")
    else:
        nav = zenplay.navigations[0]
        for i in range(nav._nframes):
            if nav._frames[i] is not None:
                data["gnss"].append({
                    "stamp_sensor": nav._tindex.time.sensor_ns[i],
                    "stamp_host": nav._tindex.time.host_ns[i],
                    "latitude": nav._frames[i].position[0],
                    "longitude": nav._frames[i].position[1],
                    "altitude": nav._frames[i].position[2],
                    "status": nav._frames[i].status,
                    "service": 0,  # SERVICE_GPS=1 if known; 0 = unknown
                })

    # Camera
    camera_names = ["front_center", "front_left", "front_right", "rear_center"]

    if not zenplay.cameras:
        print("No camera data found.")
    else:
        for cam_idx, cam in enumerate(zenplay.cameras):
            cam_name = camera_names[cam_idx] if cam_idx < len(camera_names) else f"camera_{cam_idx}"
            for i in tqdm(range(cam._nframes), desc=f"Reading {cam_name} frames"):
                if cam._frames[i] is not None:
                    img = np.asarray(cam._frames[i].data, dtype=np.uint8)
                    _, jpg_data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    data["camera"].append({
                        "camera_name": cam_name,
                        "stamp_sensor": cam._tindex.time.sensor_ns[i],
                        "stamp_host": cam._tindex.time.host_ns[i],
                        "data": jpg_data.tobytes(),
                    })

    # Lidar
    if not zenplay.lidars:
        print("No lidar data found.")
    else:
        for lidar_idx, lidar in enumerate(zenplay.lidars):
            for i in tqdm(range(lidar._nframes), desc=f"Reading lidar {lidar_idx} frames"):
                if lidar._frames[i] is not None:
                    data["lidar"].append({
                        "lidar_index": lidar_idx,
                        "stamp_sensor": lidar._tindex.time.sensor_ns[i],
                        "stamp_host": lidar._tindex.time.host_ns[i],
                        "points": lidar._frames[i],  # numpy array
                    })

    return data


path = "/mnt/zen-lager/CarLo_logs/2026-03-12_12-31-37/005"
pickle_path = "/mnt/zen-lager/CarLo_logs/2026-03-12_12-31-37/005/zenexport/005.pkl"

zenplay = Zenplay(path)

data = read_all_data(zenplay)

with open(pickle_path, "wb") as f:
    pickle.dump(data, f)

print(f"Saved {len(data)} entries to {pickle_path}")
