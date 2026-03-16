#!/usr/bin/env python3
"""CLI tool to download sensor data from remote storage to a local folder."""

# Download all segments, all sensors
# python download_recording.py 2026-03-09_18-38-49 /local/data

# Download specific segments only
# python download_recording.py 2026-03-09_18-38-49 /local/data --segments 000 001

# Download only camera and lidar
# python download_recording.py 2026-03-09_18-38-49 /local/data --sensors camera lidar

# Combine both
# python download_recording.py 2026-03-09_18-38-49 /local/data --segments 000 --sensors radar lidar

# Preview without copying
# python download_recording.py 2026-03-09_18-38-49 /local/data --dry-run

# List available segments
# python download_recording.py 2026-03-09_18-38-49 /local/data --list-segments 

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

REMOTE_ROOT = Path("/mnt/buckets/raw-acquisitions")

SENSOR_TYPES = ["camera", "lidar", "radar", "radar_epc", "navigation", "vehicle_can"]

SEGMENT_FILES = ["acquisition.pt", "metadata.pt", "vehicle.pt", "recording_description.txt",
                 "data_validation.json", "ingestion.json"]


def list_segments(recording_path: Path) -> list[str]:
    return sorted(p.name for p in recording_path.iterdir() if p.is_dir() and p.name.isdigit())


def collect_files(
    recording_path: Path,
    dest_path: Path,
    segment: str,
    sensors: list[str],
) -> list[tuple[Path, Path]]:
    """Return list of (src, dst) pairs for a segment."""
    src_seg = recording_path / segment
    dst_seg = dest_path / segment
    pairs = []

    for fname in SEGMENT_FILES:
        src = src_seg / fname
        if src.exists():
            pairs.append((src, dst_seg / fname))

    for sensor in sensors:
        src_sensor = src_seg / sensor
        if not src_sensor.exists():
            continue
        for sensor_dir in src_sensor.iterdir():
            if not sensor_dir.is_dir():
                continue
            for f in sensor_dir.iterdir():
                rel = f.relative_to(recording_path)
                pairs.append((f, dest_path / rel))

    return pairs


def download_segment(
    recording_path: Path,
    dest_path: Path,
    segment: str,
    sensors: list[str],
    dry_run: bool,
) -> None:
    pairs = collect_files(recording_path, dest_path, segment, sensors)
    total_bytes = sum(src.stat().st_size for src, _ in pairs)

    with tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Segment {segment}",
        leave=True,
    ) as bar:
        for src, dst in pairs:
            bar.set_postfix_str(src.name, refresh=False)
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            bar.update(src.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sensor data from remote storage to a local folder."
    )
    parser.add_argument(
        "recording",
        help="Recording name (e.g. 2026-03-09_18-38-49) or full path.",
    )
    parser.add_argument(
        "destination",
        help="Local destination folder.",
    )
    parser.add_argument(
        "--segments", "-s",
        nargs="+",
        metavar="SEG",
        help="Segments to download (e.g. 000 001). Downloads all if omitted.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        choices=SENSOR_TYPES,
        default=SENSOR_TYPES,
        metavar="SENSOR",
        help=(
            f"Sensor types to download. Choices: {', '.join(SENSOR_TYPES)}. "
            "Downloads all if omitted."
        ),
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be copied without actually copying.",
    )
    parser.add_argument(
        "--list-segments",
        action="store_true",
        help="List available segments in the recording and exit.",
    )

    args = parser.parse_args()

    # Resolve recording path
    recording_path = Path(args.recording)
    if not recording_path.is_absolute():
        recording_path = REMOTE_ROOT / args.recording
    if not recording_path.exists():
        print(f"Error: recording not found: {recording_path}", file=sys.stderr)
        sys.exit(1)

    # List segments mode
    if args.list_segments:
        segments = list_segments(recording_path)
        print(f"Segments in {recording_path.name}:")
        for seg in segments:
            print(f"  {seg}")
        return

    # Resolve segments
    available = list_segments(recording_path)
    if args.segments:
        segments = args.segments
        unknown = set(segments) - set(available)
        if unknown:
            print(f"Error: unknown segments: {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"Available: {', '.join(available)}", file=sys.stderr)
            sys.exit(1)
    else:
        segments = available

    dest_path = Path(args.destination) / recording_path.name

    print(f"Recording : {recording_path}")
    print(f"Destination: {dest_path}")
    print(f"Segments  : {', '.join(segments)}")
    print(f"Sensors   : {', '.join(args.sensors)}")
    if args.dry_run:
        print("Mode      : dry-run")
    print()

    for segment in segments:
        download_segment(recording_path, dest_path, segment, args.sensors, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
