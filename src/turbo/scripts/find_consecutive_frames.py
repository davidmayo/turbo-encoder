"""
Find consecutive chunks of valid frames in a file

USAGE:

python find_consecutive_frames.py \
    --input some_file.bin \
    --output some_file.report.json \
    --denominator 6 \
    --word-size 8920

This will classify every bit of the file as either being part of a valid frame (including the leading ASM of a frame),
part of an invalid frame (i.e., located in an area where the ASM indicates that there should be a frame, but the decode fails),
or in an area where no ASM's have been found close enough for this to be a frame.


Will create a file like this:

```
{
    "path": "/absolute/path/to/some_file.bin",
    "denominator": 6,
    "word_size": 8920,
    "segments": [
        {
            "start": 0,
            "end": 9112,
            "status": "INVALID-DECODE-FAILURE"
        },
        {
            "start": 9112,
            "end": 91120,
            "status": "VALID"
        },
        {
            "start": 91120,
            "end": 100000,
            "status": "INVALID-NO-ASM"
        },
        {
            "start": 100000,
            "end": 191120,
            "status": "VALID"
        },
    ]
}

```
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Literal

from turbo import (
    PseudorandomizationMode,
    Turbo2Frame,
    Turbo3Frame,
    Turbo4Frame,
    Turbo6Frame,
    find_frames_from_file,
)

Denominator = Literal[2, 3, 4, 6]
CodewordSizeBits = Literal[1784, 3568, 7136, 8920]

VALID_STATUS = "VALID"
INVALID_DECODE_STATUS = "INVALID-DECODE-FAILURE"
INVALID_NO_ASM_STATUS = "INVALID-NO-ASM"

FRAME_TYPE_BY_DENOMINATOR: dict[Denominator, type[Turbo2Frame | Turbo3Frame | Turbo4Frame | Turbo6Frame]] = {
    2: Turbo2Frame,
    3: Turbo3Frame,
    4: Turbo4Frame,
    6: Turbo6Frame,
}


@dataclasses.dataclass(frozen=True)
class Segment:
    start: int
    end: int
    status: str


def _frame_bit_length(denominator: Denominator, word_size: CodewordSizeBits) -> int:
    frame_type = FRAME_TYPE_BY_DENOMINATOR[denominator]
    asm_bits = len(frame_type.ASM) * 8
    return asm_bits + (int(word_size) + 4) * int(denominator)


def _build_segments(
    *,
    total_bits: int,
    discovered_windows: list[tuple[int, int, bool]],
) -> list[Segment]:
    if total_bits <= 0:
        return []

    # Sweep-line event map: [delta_valid, delta_invalid]
    events: dict[int, list[int]] = {}
    for start, end, decode_success in discovered_windows:
        if end <= start:
            continue
        start = max(0, start)
        end = min(total_bits, end)
        if end <= start:
            continue
        if start not in events:
            events[start] = [0, 0]
        if end not in events:
            events[end] = [0, 0]

        idx = 0 if decode_success else 1
        events[start][idx] += 1
        events[end][idx] -= 1

    boundaries = sorted({0, total_bits, *events.keys()})
    active_valid = 0
    active_invalid = 0
    segments: list[Segment] = []

    for idx, current in enumerate(boundaries[:-1]):
        deltas = events.get(current)
        if deltas is not None:
            active_valid += deltas[0]
            active_invalid += deltas[1]

        nxt = boundaries[idx + 1]
        if nxt <= current:
            continue

        if active_valid > 0:
            status = VALID_STATUS
        elif active_invalid > 0:
            status = INVALID_DECODE_STATUS
        else:
            status = INVALID_NO_ASM_STATUS

        if segments and segments[-1].status == status and segments[-1].end == current:
            last = segments[-1]
            segments[-1] = Segment(start=last.start, end=nxt, status=last.status)
        else:
            segments.append(Segment(start=current, end=nxt, status=status))

    return segments


def _build_report(
    *,
    path: Path,
    denominator: Denominator,
    word_size: CodewordSizeBits,
    pseudorandomization: PseudorandomizationMode,
) -> dict:
    resolved_path = path.resolve()
    total_bits = resolved_path.stat().st_size * 8
    frame_bits = _frame_bit_length(denominator, word_size)

    discovered_windows: list[tuple[int, int, bool]] = []
    decode_successes = 0
    decode_failures = 0

    for discovered in find_frames_from_file(
        path=resolved_path,
        denominator=denominator,
        pseudorandomization=pseudorandomization,
        codeword_size_bits=word_size,
    ):
        start = discovered.start_bit_index
        status = discovered.decode_success
        if start is None or status is None:
            continue
        discovered_windows.append((start, start + frame_bits, status))
        if status:
            decode_successes += 1
        else:
            decode_failures += 1

    segments = _build_segments(
        total_bits=total_bits,
        discovered_windows=discovered_windows,
    )

    return {
        "path": str(resolved_path),
        "denominator": int(denominator),
        "word_size": int(word_size),
        "pseudorandomization": pseudorandomization,
        "segments": [dataclasses.asdict(segment) for segment in segments],
        "summary": {
            "total_bits": total_bits,
            "frame_bits": frame_bits,
            "total_matches": len(discovered_windows),
            "decode_successes": decode_successes,
            "decode_failures": decode_failures,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify bit regions in a file based on Turbo ASM hits and decode validity."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input binary file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSON report file.",
    )
    parser.add_argument(
        "--denominator",
        type=int,
        choices=(2, 3, 4, 6),
        required=True,
        help="Turbo code denominator.",
    )
    parser.add_argument(
        "--word-size",
        dest="word_size",
        type=int,
        choices=(1784, 3568, 7136, 8920),
        default=8920,
        help="Information block size in bits.",
    )
    parser.add_argument(
        "--pseudorandomization",
        choices=("none", "legacy", "modern"),
        default="legacy",
        help="Pseudorandomization mode for decode validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a regular file: {input_path}")

    denominator: Denominator = args.denominator
    word_size: CodewordSizeBits = args.word_size
    pseudorandomization: PseudorandomizationMode = args.pseudorandomization

    report = _build_report(
        path=input_path,
        denominator=denominator,
        word_size=word_size,
        pseudorandomization=pseudorandomization,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "Wrote report to "
        f"{output_path.resolve()} "
        f"({report['summary']['total_matches']} matches, "
        f"{report['summary']['decode_successes']} valid, "
        f"{report['summary']['decode_failures']} decode-fail)"
    )


if __name__ == "__main__":
    main()
