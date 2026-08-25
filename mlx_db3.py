from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import struct
from typing import Callable, Final

import numpy as np


POINTCLOUD_FLAG: Final = 0x01
AMBIENT_FLAG: Final = 0x02
DEPTH_FLAG: Final = 0x04
INTENSITY_FLAG: Final = 0x08
MAX_ECHO_COUNT: Final = 8
MAX_ITEMS_PER_ECHO: Final = 1_000_000
FRAME_SEQUENCE_RESET_GAP_NS: Final = 1_000_000_000


@dataclass(frozen=True)
class SensorFrame:
    sequence: int
    timestamp_ns: int
    frame_id: int
    rows: int
    cols: int
    echo_count: int
    xyz_m: np.ndarray
    intensity: np.ndarray
    valid_mask: np.ndarray

    @property
    def valid_point_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))


@dataclass(frozen=True)
class ReplaySession:
    path: Path
    device_name: str
    frames: tuple[SensorFrame, ...]
    metadata: dict[str, str]
    raw_message_count: int
    invalid_message_count: int
    nominal_hz: float

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def start_time_ns(self) -> int:
        return self.frames[0].timestamp_ns

    @property
    def end_time_ns(self) -> int:
        return self.frames[-1].timestamp_ns

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.end_time_ns - self.start_time_ns) / 1_000_000_000)


@dataclass(frozen=True)
class _DecodedScene:
    record_timestamp_ns: int
    status: int
    frame_id: int
    rows: int
    cols: int
    flags: int
    timestamps: np.ndarray
    pointcloud: tuple[np.ndarray, ...]
    intensity: tuple[np.ndarray, ...]


class _Cursor:
    def __init__(self, data: bytes | memoryview) -> None:
        self.data = memoryview(data)
        self.position = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.position

    def _take(self, size: int) -> memoryview:
        if size < 0 or self.remaining < size:
            raise ValueError("truncated mlx_binary_v1 payload")
        start = self.position
        self.position += size
        return self.data[start : start + size]

    def read_u8(self) -> int:
        return int(self._take(1)[0])

    def read_u16(self) -> int:
        return struct.unpack_from("<H", self._take(2))[0]

    def read_u32(self) -> int:
        return struct.unpack_from("<I", self._take(4))[0]

    def read_u64(self) -> int:
        return struct.unpack_from("<Q", self._take(8))[0]

    def read_bytes(self, size: int) -> memoryview:
        return self._take(size)

    def skip(self, size: int) -> None:
        self._take(size)


def _skip_repeated_section(
    data: memoryview,
    offset: int,
    item_size: int,
) -> int | None:
    try:
        cursor = _Cursor(data[offset:])
        echo_count = cursor.read_u16()
        if echo_count == 0 or echo_count > MAX_ECHO_COUNT:
            return None
        for _ in range(echo_count):
            count = cursor.read_u32()
            if count == 0 or count > MAX_ITEMS_PER_ECHO:
                return None
            cursor.skip(count * item_size)
        return offset + cursor.position
    except (ValueError, struct.error):
        return None


def _can_parse_tail(
    data: memoryview,
    offset: int,
    has_depth: bool,
    has_intensity: bool,
) -> bool:
    if has_depth:
        next_offset = _skip_repeated_section(data, offset, 4)
        if next_offset is None:
            return False
        offset = next_offset
    if has_intensity:
        next_offset = _skip_repeated_section(data, offset, 2)
        if next_offset is None:
            return False
        offset = next_offset
    return offset == len(data)


def decode_scene(blob: bytes) -> _DecodedScene:
    """Decode one custom ``mlx_binary_v1`` scene payload.

    The payload contains row timestamps, one or more XYZ float32 point arrays,
    and optional ambient/depth/intensity sections. XYZ values are stored in mm.
    """

    cursor = _Cursor(blob)
    if bytes(cursor.read_bytes(4)) != b"MLXS":
        raise ValueError("invalid MLXS magic")
    version = cursor.read_u16()
    if version != 1:
        raise ValueError(f"unsupported MLXS version {version}")

    record_timestamp_ns = cursor.read_u64()
    status = cursor.read_u64()
    frame_id = cursor.read_u8()
    rows = cursor.read_u16()
    cols = cursor.read_u16()
    flags = cursor.read_u8()
    if rows == 0 or cols == 0 or flags & 0xF0:
        raise ValueError("invalid MLXS frame layout")

    timestamp_count = cursor.read_u16()
    timestamps = np.frombuffer(
        cursor.read_bytes(timestamp_count * 8),
        dtype="<u8",
    ).copy()

    pointcloud_echo_count = cursor.read_u16()
    if pointcloud_echo_count > MAX_ECHO_COUNT:
        raise ValueError("invalid pointcloud echo count")
    if bool(flags & POINTCLOUD_FLAG) != (pointcloud_echo_count > 0):
        raise ValueError("pointcloud flag and echo count disagree")

    pointcloud: list[np.ndarray] = []
    for _ in range(pointcloud_echo_count):
        point_count = cursor.read_u32()
        if point_count == 0 or point_count > MAX_ITEMS_PER_ECHO:
            raise ValueError("invalid point count")
        points = np.frombuffer(
            cursor.read_bytes(point_count * 3 * 4),
            dtype="<f4",
        ).reshape(point_count, 3)
        pointcloud.append(points.copy())

    has_ambient = bool(flags & AMBIENT_FLAG)
    has_depth = bool(flags & DEPTH_FLAG)
    has_intensity = bool(flags & INTENSITY_FLAG)
    data = cursor.data

    if has_ambient:
        base_pixels = rows * cols
        ambient_pixels: int | None = None
        for multiplier in range(1, MAX_ECHO_COUNT + 1):
            candidate_pixels = base_pixels * multiplier
            candidate_offset = cursor.position + candidate_pixels * 4
            if candidate_offset <= len(data) and _can_parse_tail(
                data,
                candidate_offset,
                has_depth,
                has_intensity,
            ):
                ambient_pixels = candidate_pixels
                break
        if ambient_pixels is None:
            raise ValueError("could not infer ambient image size")
        cursor.skip(ambient_pixels * 4)

    if has_depth:
        depth_echo_count = cursor.read_u16()
        if depth_echo_count == 0 or depth_echo_count > MAX_ECHO_COUNT:
            raise ValueError("invalid depth echo count")
        for _ in range(depth_echo_count):
            pixel_count = cursor.read_u32()
            if pixel_count == 0 or pixel_count > MAX_ITEMS_PER_ECHO:
                raise ValueError("invalid depth pixel count")
            cursor.skip(pixel_count * 4)

    intensity: list[np.ndarray] = []
    if has_intensity:
        intensity_echo_count = cursor.read_u16()
        if intensity_echo_count == 0 or intensity_echo_count > MAX_ECHO_COUNT:
            raise ValueError("invalid intensity echo count")
        for _ in range(intensity_echo_count):
            pixel_count = cursor.read_u32()
            if pixel_count == 0 or pixel_count > MAX_ITEMS_PER_ECHO:
                raise ValueError("invalid intensity pixel count")
            values = np.frombuffer(
                cursor.read_bytes(pixel_count * 2),
                dtype="<u2",
            )
            intensity.append(values.copy())

    if cursor.remaining != 0:
        raise ValueError(f"unexpected {cursor.remaining} trailing payload bytes")

    return _DecodedScene(
        record_timestamp_ns=record_timestamp_ns,
        status=status,
        frame_id=frame_id,
        rows=rows,
        cols=cols,
        flags=flags,
        timestamps=timestamps,
        pointcloud=tuple(pointcloud),
        intensity=tuple(intensity),
    )


def _read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            )
        }
    except sqlite3.OperationalError:
        return {}


def load_session(
    path: str | Path,
    progress: Callable[[int, str], None] | None = None,
) -> ReplaySession:
    """Load a complete DB3 replay into memory without any ROS dependency."""

    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"DB3 file does not exist: {db_path}")

    uri = f"{db_path.as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        metadata = _read_metadata(connection)
        topics = list(
            connection.execute(
                """
                SELECT id
                FROM topics
                WHERE type='mlx/SceneData'
                  AND serialization_format='mlx_binary_v1'
                ORDER BY id
                """
            )
        )
        if len(topics) != 1:
            raise ValueError(
                "DB3 must contain exactly one mlx/SceneData mlx_binary_v1 topic"
            )
        topic_id = int(topics[0][0])
        raw_message_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM messages WHERE topic_id=?",
                (topic_id,),
            ).fetchone()[0]
        )

        frames: list[SensorFrame] = []
        invalid = 0
        previous_frame_id: int | None = None
        previous_timestamp_ns: int | None = None
        rows_cursor = connection.execute(
            """
            SELECT id, timestamp, data
            FROM messages
            WHERE topic_id=?
            ORDER BY timestamp, id
            """,
            (topic_id,),
        )
        for raw_index, (_, timestamp_ns, blob) in enumerate(rows_cursor):
            try:
                scene = decode_scene(bytes(blob))
                expected = scene.rows * scene.cols
                if not scene.pointcloud or any(
                    points.shape != (expected, 3)
                    for points in scene.pointcloud
                ):
                    raise ValueError("point cloud size does not match frame grid")

                timestamp_value = int(timestamp_ns)
                if previous_frame_id is not None:
                    frame_delta = (scene.frame_id - previous_frame_id) & 0xFF
                    sequence_restarted = (
                        previous_timestamp_ns is not None
                        and timestamp_value - previous_timestamp_ns
                        >= FRAME_SEQUENCE_RESET_GAP_NS
                    )
                    if (frame_delta == 0 or frame_delta >= 128) and not sequence_restarted:
                        raise ValueError("duplicate or out-of-order frame id")

                xyz_m = np.concatenate(scene.pointcloud, axis=0).astype(
                    np.float32,
                    copy=False,
                )
                xyz_m *= np.float32(0.001)
                valid_mask = (
                    np.isfinite(xyz_m).all(axis=1)
                    & (np.einsum("ij,ij->i", xyz_m, xyz_m) > 1.0e-12)
                )

                intensity_parts: list[np.ndarray] = []
                for echo_index, points in enumerate(scene.pointcloud):
                    if (
                        echo_index < len(scene.intensity)
                        and scene.intensity[echo_index].shape[0] == points.shape[0]
                    ):
                        values = scene.intensity[echo_index].astype(np.float32)
                    else:
                        values = np.zeros(points.shape[0], dtype=np.float32)
                    intensity_parts.append(values)

                frames.append(
                    SensorFrame(
                        sequence=len(frames),
                        timestamp_ns=timestamp_value,
                        frame_id=scene.frame_id,
                        rows=scene.rows,
                        cols=scene.cols,
                        echo_count=len(scene.pointcloud),
                        xyz_m=xyz_m,
                        intensity=np.concatenate(intensity_parts),
                        valid_mask=valid_mask,
                    )
                )
                previous_frame_id = scene.frame_id
                previous_timestamp_ns = timestamp_value
            except (ValueError, struct.error):
                invalid += 1

            if progress is not None:
                percent = round((raw_index + 1) * 100 / max(raw_message_count, 1))
                progress(percent, f"Reading frames {raw_index + 1}/{raw_message_count}")

        if not frames:
            raise ValueError("DB3 contains no valid complete MLX point-cloud frames")

        timestamps = np.asarray(
            [frame.timestamp_ns for frame in frames],
            dtype=np.int64,
        )
        intervals = np.diff(timestamps)
        intervals = intervals[intervals > 0]
        nominal_hz = (
            1_000_000_000 / float(np.median(intervals))
            if intervals.size
            else 0.0
        )
        return ReplaySession(
            path=db_path,
            device_name=metadata.get("device_name", db_path.stem),
            frames=tuple(frames),
            metadata=metadata,
            raw_message_count=raw_message_count,
            invalid_message_count=invalid,
            nominal_hz=nominal_hz,
        )
    finally:
        connection.close()
