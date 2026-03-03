import math

import pytest

from turbo import (
    Turbo2Frame,
    Turbo3Frame,
    Turbo4Frame,
    Turbo6Frame,
    TurboDecodeError,
    TurboEncodeError,
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

    frame.encode()

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
    encoded_frame.encode()
    assert encoded_frame.encoded is not None

    decoded_frame = frame_cls(encoded=encoded_frame.encoded)
    decoded_frame.decode()
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
    encoded_frame.encode()
    assert encoded_frame.encoded is not None

    raw_codeword = encoded_frame.encoded[len(encoded_frame.ASM) :]
    decoded_frame = frame_cls(encoded=raw_codeword)
    decoded_frame.decode()
    assert decoded_frame.decoded == payload


def test_decode_rejects_corrupted_codeword() -> None:
    payload = _payload(223, 123)
    frame = Turbo4Frame(decoded=payload)
    frame.encode()
    assert frame.encoded is not None

    tampered = bytearray(frame.encoded)
    tampered[len(frame.ASM)] ^= 0x80

    with pytest.raises(TurboDecodeError):
        Turbo4Frame(encoded=bytes(tampered)).decode()


def test_invalid_payload_size_raises_encode_error() -> None:
    with pytest.raises(TurboEncodeError):
        Turbo2Frame(decoded=b"\x00").encode()


def test_invalid_encoded_size_raises_decode_error() -> None:
    with pytest.raises(TurboDecodeError):
        Turbo2Frame(encoded=b"\x00" * 10).decode()


def test_rate_1_3_padding_bits_zero_and_rejected_if_nonzero() -> None:
    payload = _payload(223, 44)
    frame = Turbo3Frame(decoded=payload)
    frame.encode()
    assert frame.encoded is not None

    raw_codeword = frame.encoded[len(frame.ASM) :]
    assert raw_codeword[-1] & 0x0F == 0

    tampered_raw = bytearray(raw_codeword)
    tampered_raw[-1] |= 0x0F
    tampered_encoded = frame.ASM + bytes(tampered_raw)

    with pytest.raises(TurboDecodeError):
        Turbo3Frame(encoded=tampered_encoded).decode()


def test_missing_inputs_raise_errors() -> None:
    with pytest.raises(TurboEncodeError):
        Turbo2Frame().encode()

    with pytest.raises(TurboDecodeError):
        Turbo2Frame().decode()
