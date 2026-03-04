import math
from pathlib import Path

import bitstring
import pytest

from turbo import (
    DiscoveredTurboFrame,
    Turbo2Frame,
    Turbo3Frame,
    Turbo4Frame,
    Turbo6Frame,
    TurboDecodeError,
    TurboEncodeError,
    find_frames_from_bits,
    find_frames_from_bytes,
    find_frames_from_file,
)


FRAME_CASES = (
    (Turbo2Frame, 2),
    (Turbo3Frame, 3),
    (Turbo4Frame, 4),
    (Turbo6Frame, 6),
)
PAYLOAD_SIZES = (223, 446, 892, 1115)


def _payload(length: int, seed: int) -> bytes:
    return bytes((seed + idx * 17) % 256 for idx in range(length))


@pytest.mark.parametrize(
    ("frame_cls", "denominator", "payload_size"),
    [
        (frame_cls, denominator, payload_size)
        for frame_cls, denominator in FRAME_CASES
        for payload_size in PAYLOAD_SIZES
    ],
)
def test_encode_success_matrix(
    frame_cls: type[Turbo2Frame | Turbo3Frame | Turbo4Frame | Turbo6Frame],
    denominator: int,
    payload_size: int,
) -> None:
    payload = _payload(payload_size, denominator + payload_size)
    frame = frame_cls(decoded=payload)

    frame.encode(codeword_size_bits=payload_size * 8)  # type: ignore[arg-type]

    assert frame.encoded is not None
    assert frame.encoded.startswith(frame.ASM)

    information_bits = payload_size * 8
    expected_codeword_bits = (information_bits + 4) * denominator
    expected_total_bytes = len(frame.ASM) + math.ceil(expected_codeword_bits / 8)
    assert len(frame.encoded) == expected_total_bytes


@pytest.mark.parametrize(
    ("frame_cls", "denominator", "payload_size"),
    [
        (frame_cls, denominator, payload_size)
        for frame_cls, denominator in FRAME_CASES
        for payload_size in PAYLOAD_SIZES
    ],
)
def test_decode_roundtrip_matrix(
    frame_cls: type[Turbo2Frame | Turbo3Frame | Turbo4Frame | Turbo6Frame],
    denominator: int,
    payload_size: int,
) -> None:
    payload = _payload(payload_size, denominator + payload_size + 99)

    encoded_frame = frame_cls(decoded=payload)
    encoded_frame.encode(codeword_size_bits=payload_size * 8)  # type: ignore[arg-type]
    assert encoded_frame.encoded is not None

    decoded_frame = frame_cls(encoded=encoded_frame.encoded)
    decoded_frame.decode(codeword_size_bits=payload_size * 8)  # type: ignore[arg-type]
    assert decoded_frame.decoded == payload


@pytest.mark.parametrize(
    ("frame_cls", "payload_size"),
    [
        (frame_cls, payload_size)
        for frame_cls, _ in FRAME_CASES
        for payload_size in PAYLOAD_SIZES
    ],
)
def test_decode_accepts_raw_codeword_without_asm(
    frame_cls: type[Turbo2Frame | Turbo3Frame | Turbo4Frame | Turbo6Frame],
    payload_size: int,
) -> None:
    payload = _payload(payload_size, payload_size + 7)

    encoded_frame = frame_cls(decoded=payload)
    encoded_frame.encode(codeword_size_bits=payload_size * 8)  # type: ignore[arg-type]
    assert encoded_frame.encoded is not None

    raw_codeword = encoded_frame.encoded[len(encoded_frame.ASM) :]
    decoded_frame = frame_cls(encoded=raw_codeword)
    decoded_frame.decode(codeword_size_bits=payload_size * 8)  # type: ignore[arg-type]
    assert decoded_frame.decoded == payload


def test_decode_rejects_corrupted_codeword() -> None:
    payload = _payload(223, 123)
    frame = Turbo4Frame(decoded=payload)
    frame.encode(codeword_size_bits=1784)
    assert frame.encoded is not None

    tampered = bytearray(frame.encoded)
    tampered[len(frame.ASM)] ^= 0x80

    with pytest.raises(TurboDecodeError):
        Turbo4Frame(encoded=bytes(tampered)).decode(codeword_size_bits=1784)


def test_invalid_payload_size_raises_encode_error() -> None:
    with pytest.raises(TurboEncodeError):
        Turbo2Frame(decoded=b"\x00").encode()


def test_invalid_encoded_size_raises_decode_error() -> None:
    with pytest.raises(TurboDecodeError):
        Turbo2Frame(encoded=b"\x00" * 10).decode()


def test_rate_1_3_padding_bits_zero_and_rejected_if_nonzero() -> None:
    payload = _payload(223, 44)
    frame = Turbo3Frame(decoded=payload)
    frame.encode(codeword_size_bits=1784)
    assert frame.encoded is not None

    raw_codeword = frame.encoded[len(frame.ASM) :]
    assert raw_codeword[-1] & 0x0F == 0

    tampered_raw = bytearray(raw_codeword)
    tampered_raw[-1] |= 0x0F
    tampered_encoded = frame.ASM + bytes(tampered_raw)

    with pytest.raises(TurboDecodeError):
        Turbo3Frame(encoded=tampered_encoded).decode(codeword_size_bits=1784)


def test_missing_inputs_raise_errors() -> None:
    with pytest.raises(TurboEncodeError):
        Turbo2Frame().encode()

    with pytest.raises(TurboDecodeError):
        Turbo2Frame().decode()


def _successes(
    discovered: list[DiscoveredTurboFrame],
) -> list[DiscoveredTurboFrame]:
    return [frame for frame in discovered if frame.decode_success is True]


@pytest.mark.parametrize(
    ("frame_cls", "payload_size", "mode"),
    (
        (Turbo2Frame, 223, "none"),
        (Turbo3Frame, 223, "legacy"),
        (Turbo6Frame, 1115, "modern"),
    ),
)
def test_encode_decode_accepts_explicit_pseudorandomization_modes(
    frame_cls: type[Turbo2Frame | Turbo3Frame | Turbo6Frame],
    payload_size: int,
    mode: str,
) -> None:
    payload = _payload(payload_size, payload_size + 123)
    encoded = frame_cls(decoded=payload)
    encoded.encode(
        pseudorandomization=mode,
        codeword_size_bits=payload_size * 8,  # type: ignore[arg-type]
    )
    assert encoded.encoded is not None

    decoded = frame_cls(encoded=encoded.encoded)
    decoded.decode(
        pseudorandomization=mode,
        codeword_size_bits=payload_size * 8,  # type: ignore[arg-type]
    )
    assert decoded.decoded == payload


def test_decode_fails_when_pseudorandomization_mode_mismatch() -> None:
    payload = _payload(223, 141)
    encoded = Turbo2Frame(decoded=payload)
    encoded.encode(pseudorandomization="legacy", codeword_size_bits=1784)
    assert encoded.encoded is not None

    with pytest.raises(TurboDecodeError):
        Turbo2Frame(encoded=encoded.encoded).decode(
            pseudorandomization="none", codeword_size_bits=1784
        )


def test_find_frames_from_bits_non_byte_aligned_start() -> None:
    payload = _payload(223, 31)
    encoded = Turbo2Frame(decoded=payload)
    encoded.encode(codeword_size_bits=1784)
    assert encoded.encoded is not None

    data = (
        bitstring.Bits("0b101")
        + bitstring.Bits(bytes=encoded.encoded)
        + bitstring.Bits("0b11")
    )
    discovered = list(
        find_frames_from_bits(data=data, denominator=2, codeword_size_bits=1784)
    )

    successful = _successes(discovered)
    assert successful
    assert any(
        frame.start_bit_index == 3
        and frame.parsed is not None
        and frame.parsed.decoded == payload
        for frame in successful
    )


def test_find_frames_from_bits_yields_one_candidate_per_asm() -> None:
    valid_payload = _payload(1115, 53)
    valid = Turbo2Frame(decoded=valid_payload)
    valid.encode()
    assert valid.encoded is not None

    invalid_payload = _payload(1115, 99)
    invalid = Turbo2Frame(decoded=invalid_payload)
    invalid.encode()
    assert invalid.encoded is not None
    tampered_invalid = bytearray(invalid.encoded)
    tampered_invalid[len(invalid.ASM)] ^= 0x80

    data = bitstring.Bits(bytes=valid.encoded) + bitstring.Bits(
        bytes=bytes(tampered_invalid)
    )
    discovered = list(
        find_frames_from_bits(data=data, denominator=2, codeword_size_bits=8920)
    )

    at_start_zero = [frame for frame in discovered if frame.start_bit_index == 0]
    assert len(at_start_zero) == 1
    assert any(frame.decode_success is True for frame in at_start_zero)
    assert any(
        frame.decode_success is False and frame.parsed is None for frame in discovered
    )


def test_find_frames_from_bytes_delegates_and_decodes() -> None:
    payload = _payload(223, 71)
    encoded = Turbo3Frame(decoded=payload)
    encoded.encode(codeword_size_bits=1784)
    assert encoded.encoded is not None

    bits = (
        bitstring.Bits("0b10101")
        + bitstring.Bits(bytes=encoded.encoded)
        + bitstring.Bits("0b111")
    )
    discovered = list(
        find_frames_from_bytes(
            data=bits.tobytes(), denominator=3, codeword_size_bits=1784
        )
    )

    successful = _successes(discovered)
    assert any(
        frame.start_bit_index == 5
        and frame.parsed is not None
        and frame.parsed.decoded == payload
        for frame in successful
    )


def test_find_frames_from_file_sets_path(tmp_path: Path) -> None:
    payload = _payload(223, 81)
    encoded = Turbo2Frame(decoded=payload)
    encoded.encode(codeword_size_bits=1784)
    assert encoded.encoded is not None

    bits = (
        bitstring.Bits("0b111")
        + bitstring.Bits(bytes=encoded.encoded)
        + bitstring.Bits("0b0")
    )
    path = tmp_path / "stream.bin"
    path.write_bytes(bits.tobytes())

    discovered = list(
        find_frames_from_file(path=path, denominator=2, codeword_size_bits=1784)
    )
    successful = _successes(discovered)

    assert any(
        frame.path == path.resolve()
        and frame.start_bit_index == 3
        and frame.parsed is not None
        and frame.parsed.decoded == payload
        for frame in successful
    )


def test_find_frames_empty_when_no_asm() -> None:
    discovered = list(
        find_frames_from_bits(data=bitstring.Bits("0b0" * 500), denominator=2)
    )
    assert discovered == []


def test_find_frames_invalid_denominator_raises() -> None:
    with pytest.raises(ValueError):
        list(find_frames_from_bits(data=bitstring.Bits("0b0" * 300), denominator=5))  # type: ignore[arg-type]


def test_find_frames_respects_pseudorandomization_mode() -> None:
    payload = _payload(223, 155)
    encoded = Turbo2Frame(decoded=payload)
    encoded.encode(pseudorandomization="modern", codeword_size_bits=1784)
    assert encoded.encoded is not None

    data = bitstring.Bits("0b101") + bitstring.Bits(bytes=encoded.encoded)
    wrong_mode = list(
        find_frames_from_bits(
            data=data,
            denominator=2,
            pseudorandomization="legacy",
            codeword_size_bits=1784,
        )
    )
    assert not _successes(wrong_mode)

    correct_mode = list(
        find_frames_from_bits(
            data=data,
            denominator=2,
            pseudorandomization="modern",
            codeword_size_bits=1784,
        )
    )
    assert any(
        frame.start_bit_index == 3
        and frame.parsed is not None
        and frame.parsed.decoded == payload
        for frame in _successes(correct_mode)
    )
