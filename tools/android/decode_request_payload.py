import argparse
import sqlite3
from pathlib import Path


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def parse_message(buf: bytes) -> list[tuple[int, int, int | bytes]]:
    fields: list[tuple[int, int, int | bytes]] = []
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field_no = key >> 3
        wire_type = key & 0x7
        if wire_type == 0:
            value, pos = read_varint(buf, pos)
            fields.append((field_no, wire_type, value))
        elif wire_type == 2:
            length, pos = read_varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
            fields.append((field_no, wire_type, value))
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type} at field {field_no}")
    return fields


def decode_packed_int32(buf: bytes) -> list[int]:
    values: list[int] = []
    pos = 0
    while pos < len(buf):
        value, pos = read_varint(buf, pos)
        values.append(value)
    return values


def tensor_summary(raw: bytes) -> dict[str, object]:
    data_len: int | None = None
    shape: list[int] = []
    dtype: str | None = None
    for field_no, wire_type, value in parse_message(raw):
        if field_no == 1 and wire_type == 2:
            data_len = len(value)  # type: ignore[arg-type]
        elif field_no == 2:
            if wire_type == 0:
                shape.append(value)  # type: ignore[arg-type]
            elif wire_type == 2:
                shape.extend(decode_packed_int32(value))  # type: ignore[arg-type]
        elif field_no == 3 and wire_type == 2:
            dtype = value.decode("utf-8")  # type: ignore[union-attr]
    return {"dtype": dtype, "shape": shape, "bytes": data_len}


def request_summary(raw: bytes) -> dict[str, object]:
    summary: dict[str, object] = {"tensors": {}}
    tensor_names = {
        3: "hidden_states",
        4: "attention_mask",
        5: "position_ids",
        6: "labels",
        7: "shift_log_p_prev",
    }
    tensors = summary["tensors"]
    assert isinstance(tensors, dict)
    for field_no, wire_type, value in parse_message(raw):
        if field_no == 1:
            summary["batch_id"] = value
        elif field_no == 2:
            summary["chunk_idx"] = value
        elif field_no in tensor_names and wire_type == 2:
            tensors[tensor_names[field_no]] = tensor_summary(value)  # type: ignore[arg-type]
        elif field_no == 8 and wire_type == 2:
            summary["request_id"] = value.decode("utf-8")  # type: ignore[union-attr]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode stored coordinator ForwardChunkRequest payloads.")
    parser.add_argument("request_ids", nargs="+")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("coordinator/coordinator/data/coordinator.db"),
        help="Path to coordinator.db.",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    for request_id in args.request_ids:
        row = conn.execute(
            "SELECT payload_proto FROM request_payloads WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            print(f"{request_id}: no stored payload")
            continue

        summary = request_summary(row[0])
        print(
            f"{request_id}: batch_id={summary.get('batch_id')} "
            f"chunk_idx={summary.get('chunk_idx', 0)} "
            f"request_id={summary.get('request_id')}"
        )
        tensors = summary["tensors"]
        assert isinstance(tensors, dict)
        for name, tensor in tensors.items():
            print(
                f"  {name}: dtype={tensor['dtype']} "
                f"shape={tensor['shape']} bytes={tensor['bytes']}"
            )


if __name__ == "__main__":
    main()
