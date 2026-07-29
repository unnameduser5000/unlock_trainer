import argparse
import struct
from pathlib import Path


MARKERS = (
    b"__et_training",
    b"aten::empty_permuted",
    b"XnnpackBackend",
    b"aten::_softmax",
    b"aten::_log_softmax",
)


def read_program_region(path: Path) -> bytes:
    with path.open("rb") as f:
        header = f.read(64)
        if len(header) >= 40 and header[8:12] == b"eh00":
            program_size = struct.unpack_from("<Q", header, 16)[0]
            f.seek(0)
            return f.read(program_size)
        f.seek(0)
        return f.read(min(path.stat().st_size, 1024 * 1024))


def inspect(path: Path) -> None:
    program = read_program_region(path)
    counts = {marker.decode("ascii"): program.count(marker) for marker in MARKERS}
    summary = " ".join(f"{key}={value}" for key, value in counts.items())
    print(f"{path}: size={path.stat().st_size} program_bytes={len(program)} {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect key markers in ExecuTorch .pte artifacts.")
    parser.add_argument("pte", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.pte:
        inspect(path)


if __name__ == "__main__":
    main()
