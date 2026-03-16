import pickle

from mcap_ros2.writer import Writer as McapWriter

from utils.schema_ros import schema_ros

_NUMPY_TO_ROS_TYPE = {
    ('f', 4): 7,  # FLOAT32
    ('f', 8): 8,  # FLOAT64
    ('i', 1): 1,  # INT8
    ('u', 1): 2,  # UINT8
    ('i', 2): 3,  # INT16
    ('u', 2): 4,  # UINT16
    ('i', 4): 5,  # INT32
    ('u', 4): 6,  # UINT32
    # uint64 / int64 have no PointField equivalent — skipped
}


def _pointcloud2_fields(arr):
    """Return (ros_fields, point_step) for a numpy structured array.

    Array sub-fields (e.g. position shape=(3,)) are expanded into individual
    fields. 'position' is expanded to x/y/z for RViz2 compatibility.
    Unsupported dtypes (e.g. uint64) are skipped.
    """
    fields = []
    for name in arr.dtype.names:
        base_dtype, offset = arr.dtype.fields[name]
        sub_shape = base_dtype.shape
        scalar_dtype = base_dtype.base
        ros_type = _NUMPY_TO_ROS_TYPE.get((scalar_dtype.kind, scalar_dtype.itemsize))

        if ros_type is None:
            continue  # unsupported (e.g. uint64)

        if sub_shape:
            count = sub_shape[0]
            axis_names = ["x", "y", "z"] if name == "position" else [f"{name}_{j}" for j in range(count)]
            for j, axis in enumerate(axis_names):
                fields.append({
                    "name": axis,
                    "offset": offset + j * scalar_dtype.itemsize,
                    "datatype": ros_type,
                    "count": 1,
                })
        else:
            fields.append({
                "name": name,
                "offset": offset,
                "datatype": ros_type,
                "count": 1,
            })

    return fields, arr.dtype.itemsize


def write_to_mcap(output_file, data):
    with open(output_file, "wb") as fo:
        writer = McapWriter(fo)

        schema_float = writer.register_msgdef(
            "std_msgs/msg/Float64",
            schema_ros["std_msgs/msg/Float64"],
        )
        schema_bool = writer.register_msgdef(
            "std_msgs/msg/Bool",
            schema_ros["std_msgs/msg/Bool"],
        )
        schema_nav_sat_fix = writer.register_msgdef(
            "sensor_msgs/msg/NavSatFix",
            schema_ros["sensor_msgs/msg/NavSatFix"],
        )
        schema_compressed_image = writer.register_msgdef(
            "sensor_msgs/msg/CompressedImage",
            schema_ros["sensor_msgs/msg/CompressedImage"],
        )
        schema_pointcloud2 = writer.register_msgdef(
            "sensor_msgs/msg/PointCloud2",
            schema_ros["sensor_msgs/msg/PointCloud2"],
        )

        # CAN — wheel_speeds: {host_ns: [fl, fr, rl, rr]}
        wheel_speed_topics = [
            "/can/wheel_speed_front_left",
            "/can/wheel_speed_front_right",
            "/can/wheel_speed_rear_left",
            "/can/wheel_speed_rear_right",
        ]
        for ts, speeds in sorted(data["can"]["wheel_speeds"].items()):
            for topic, ws in zip(wheel_speed_topics, speeds):
                writer.write_message(
                    topic=topic,
                    schema=schema_float,
                    message={"data": float(ws)},
                    log_time=ts,
                    publish_time=ts,
                )

        # CAN — vehicle_speed: {host_ns: speed_ms}
        for ts, speed in sorted(data["can"]["vehicle_speed"].items()):
            writer.write_message(
                topic="/can/vehicle_speed",
                schema=schema_float,
                message={"data": float(speed)},
                log_time=ts,
                publish_time=ts,
            )

        # CAN — steering_angle: {host_ns: angle_rad}
        for ts, angle in sorted(data["can"]["steering_angle"].items()):
            writer.write_message(
                topic="/can/steering_angle",
                schema=schema_float,
                message={"data": float(angle)},
                log_time=ts,
                publish_time=ts,
            )

        # CAN — vehicle_dynamics: {host_ns: {longitudinal_acceleration, lateral_acceleration, yaw_rate}}
        for ts, dyn in sorted(data["can"]["vehicle_dynamics"].items()):
            for field, topic in (
                ("longitudinal_acceleration", "/can/longitudinal_acceleration"),
                ("lateral_acceleration", "/can/lateral_acceleration"),
                ("yaw_rate", "/can/yaw_rate"),
            ):
                writer.write_message(
                    topic=topic,
                    schema=schema_float,
                    message={"data": float(dyn[field])},
                    log_time=ts,
                    publish_time=ts,
                )

        # CAN — turn_indicators: {host_ns: {turn_signal_left, turn_signal_right}}
        for ts, ind in sorted(data["can"]["turn_indicators"].items()):
            for field, topic in (
                ("turn_signal_left", "/can/turn_signal_left"),
                ("turn_signal_right", "/can/turn_signal_right"),
            ):
                writer.write_message(
                    topic=topic,
                    schema=schema_bool,
                    message={"data": bool(ind[field])},
                    log_time=ts,
                    publish_time=ts,
                )

        # CAN — gear_position: {host_ns: gear}
        for ts, gear in sorted(data["can"]["gear_position"].items()):
            writer.write_message(
                topic="/can/gear_position",
                schema=schema_float,
                message={"data": float(gear)},
                log_time=ts,
                publish_time=ts,
            )

        # CAN — accelerator_pedal: {host_ns: percent}
        for ts, pedal in sorted(data["can"]["accelerator_pedal"].items()):
            writer.write_message(
                topic="/can/accelerator_pedal",
                schema=schema_float,
                message={"data": float(pedal)},
                log_time=ts,
                publish_time=ts,
            )

        # GNSS
        for entry in data["gnss"]:
            stamp_sec = entry["stamp_sensor"] // 1_000_000_000
            stamp_nanosec = entry["stamp_sensor"] % 1_000_000_000
            writer.write_message(
                topic="/gnss",
                schema=schema_nav_sat_fix,
                message={
                    "header": {
                        "stamp": {"sec": stamp_sec, "nanosec": stamp_nanosec},
                        "frame_id": "gnss",
                    },
                    "status": {
                        "status": entry["status"],
                        "service": entry["service"],
                    },
                    "latitude": entry["latitude"],
                    "longitude": entry["longitude"],
                    "altitude": entry["altitude"],
                    "position_covariance": [0.0] * 9,
                    "position_covariance_type": 0,  # COVARIANCE_TYPE_UNKNOWN
                },
                log_time=entry["stamp_host"],
                publish_time=entry["stamp_host"],
            )

        # Camera
        for entry in data["camera"]:
            stamp_sec = entry["stamp_sensor"] // 1_000_000_000
            stamp_nanosec = entry["stamp_sensor"] % 1_000_000_000
            writer.write_message(
                topic=f"/camera/{entry['camera_name']}/compressed",
                schema=schema_compressed_image,
                message={
                    "header": {
                        "stamp": {"sec": stamp_sec, "nanosec": stamp_nanosec},
                        "frame_id": entry["camera_name"],
                    },
                    "format": "jpeg",
                    "data": entry["data"],
                },
                log_time=entry["stamp_host"],
                publish_time=entry["stamp_host"],
            )

        # Lidar
        for entry in data["lidar"]:
            stamp_sec = entry["stamp_sensor"] // 1_000_000_000
            stamp_nanosec = entry["stamp_sensor"] % 1_000_000_000
            arr = entry["points"]
            fields, point_step = _pointcloud2_fields(arr)
            flat = arr.tobytes()
            writer.write_message(
                topic=f"/lidar_{entry['lidar_index']}/points",
                schema=schema_pointcloud2,
                message={
                    "header": {
                        "stamp": {"sec": stamp_sec, "nanosec": stamp_nanosec},
                        "frame_id": f"lidar_{entry['lidar_index']}",
                    },
                    "height": 1,
                    "width": len(flat) // point_step,
                    "fields": fields,
                    "is_bigendian": False,
                    "point_step": point_step,
                    "row_step": len(flat),
                    "data": flat,
                    "is_dense": False,
                },
                log_time=entry["stamp_host"],
                publish_time=entry["stamp_host"],
            )

        writer.finish()

    n_can = len(data["can"]["wheel_speeds"]) + len(data["can"]["vehicle_speed"])
    return n_can + len(data["gnss"]) + len(data["camera"])


pickle_path = "/mnt/zen-lager/CarLo_logs/2026-03-12_12-31-37/005/zenexport/ZenCarLo_2026-03-12_12-31-37_005.pkl"
output_path = "/mnt/zen-lager/CarLo_logs/2026-03-12_12-31-37/005/zenexport/ZenCarLo_2026-03-12_12-31-37_005.mcap"

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

n = write_to_mcap(output_path, data)
print(f"Wrote {n} messages to {output_path}")
