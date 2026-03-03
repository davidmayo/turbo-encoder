import dataclasses
from typing import ClassVar, Literal


class TurboError(RuntimeError):
    """An error with a Turbo encoding/decoding process"""


class TurboEncodeError(TurboError):
    """An error with a Turbo encoding process"""


class TurboDecodeError(TurboError):
    """An error with a Turbo decoding process"""


_VALID_INFORMATION_BITS = (1784, 3568, 7136, 8920)
_VALID_INFORMATION_BYTES = tuple(value // 8 for value in _VALID_INFORMATION_BITS)
_K2_BY_INFORMATION_BITS = {
    1784: 223,
    3568: 446,
    7136: 892,
    8920: 1115,
}
_TERMINATION_BITS = 4
_INTERLEAVER_PRIMES = (31, 37, 43, 47, 53, 59, 61, 67)

_G1 = (1, 1, 0, 1, 1)
_G2 = (1, 0, 1, 0, 1)
_G3 = (1, 1, 1, 1, 1)

_INTERLEAVER_CACHE: dict[int, tuple[int, ...]] = {}


def _bytes_to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    if not bits:
        return b""

    out = bytearray((len(bits) + 7) // 8)
    for idx, bit in enumerate(bits):
        out[idx // 8] |= bit << (7 - (idx % 8))
    return bytes(out)


def _truncate_bits_with_padding_check(data: bytes, bit_length: int) -> list[int]:
    bits = _bytes_to_bits(data)
    if len(bits) < bit_length:
        raise TurboDecodeError(
            f"Not enough bits in codeword: expected {bit_length}, got {len(bits)}"
        )
    if any(bits[bit_length:]):
        raise TurboDecodeError("Non-zero padding bits found after codeword")
    return bits[:bit_length]


def _packed_codeword_bytes(information_bits: int, denominator: int) -> int:
    return ((information_bits + _TERMINATION_BITS) * denominator + 7) // 8


def _interleaver_indices(information_bits: int) -> tuple[int, ...]:
    cached = _INTERLEAVER_CACHE.get(information_bits)
    if cached is not None:
        return cached

    if information_bits not in _K2_BY_INFORMATION_BITS:
        raise TurboEncodeError(
            f"Unsupported information block length: {information_bits} bits"
        )

    k1 = 8
    k2 = _K2_BY_INFORMATION_BITS[information_bits]
    half_k1 = k1 // 2

    indices = [0] * information_bits
    for s in range(1, information_bits + 1):
        m = (s - 1) % 2
        i = (s - 1) // (2 * k2)
        j = ((s - 1) // 2) - (i * k2)

        t = (19 * i + 1) % half_k1
        q = (t % 8) + 1
        c = (_INTERLEAVER_PRIMES[q - 1] * j + 21 * m) % k2
        pi = 2 * (t + c * half_k1 + 1) - m

        indices[s - 1] = pi - 1

    if len(set(indices)) != information_bits:
        raise TurboEncodeError(
            "Interleaver permutation generation failed to produce a valid permutation"
        )

    result = tuple(indices)
    _INTERLEAVER_CACHE[information_bits] = result
    return result


def _component_encode(
    information_bits: list[int],
    forward_vectors: dict[str, tuple[int, int, int, int, int]],
) -> tuple[list[int], dict[str, list[int]]]:
    m1 = m2 = m3 = m4 = 0
    systematic: list[int] = []
    parity = {name: [] for name in forward_vectors}

    for idx in range(len(information_bits) + _TERMINATION_BITS):
        feedback = m3 ^ m4  # G0=10011
        switch_value = (
            information_bits[idx] if idx < len(information_bits) else feedback
        )
        recursive_input = switch_value ^ feedback

        systematic.append(switch_value)
        taps = (recursive_input, m1, m2, m3, m4)

        for name, vector in forward_vectors.items():
            value = 0
            for include, tap in zip(vector, taps):
                if include:
                    value ^= tap
            parity[name].append(value)

        m1, m2, m3, m4 = recursive_input, m1, m2, m3

    return systematic, parity


def _build_codeword_bits(information_bits: list[int], denominator: int) -> list[int]:
    information_bit_count = len(information_bits)
    if information_bit_count not in _VALID_INFORMATION_BITS:
        raise TurboEncodeError(
            f"Unsupported information block length: {information_bit_count} bits"
        )

    permutation = _interleaver_indices(information_bit_count)
    interleaved = [information_bits[index] for index in permutation]

    if denominator in (2, 3):
        out0a, parity_a = _component_encode(information_bits, {"1": _G1})
        _, parity_b = _component_encode(interleaved, {"1": _G1})
    elif denominator == 4:
        out0a, parity_a = _component_encode(information_bits, {"2": _G2, "3": _G3})
        _, parity_b = _component_encode(interleaved, {"1": _G1})
    elif denominator == 6:
        out0a, parity_a = _component_encode(
            information_bits, {"1": _G1, "2": _G2, "3": _G3}
        )
        _, parity_b = _component_encode(interleaved, {"1": _G1, "3": _G3})
    else:
        raise TurboEncodeError(
            f"Unsupported Turbo denominator: {denominator}. Expected one of 2, 3, 4, 6."
        )

    symbol_count = information_bit_count + _TERMINATION_BITS
    codeword: list[int] = []

    if denominator == 2:
        for idx in range(symbol_count):
            codeword.append(out0a[idx])
            codeword.append(parity_a["1"][idx] if idx % 2 == 0 else parity_b["1"][idx])
    elif denominator == 3:
        for idx in range(symbol_count):
            codeword.extend((out0a[idx], parity_a["1"][idx], parity_b["1"][idx]))
    elif denominator == 4:
        for idx in range(symbol_count):
            codeword.extend(
                (out0a[idx], parity_a["2"][idx], parity_a["3"][idx], parity_b["1"][idx])
            )
    else:  # denominator == 6
        for idx in range(symbol_count):
            codeword.extend(
                (
                    out0a[idx],
                    parity_a["1"][idx],
                    parity_a["2"][idx],
                    parity_a["3"][idx],
                    parity_b["1"][idx],
                    parity_b["3"][idx],
                )
            )

    expected_length = (information_bit_count + _TERMINATION_BITS) * denominator
    if len(codeword) != expected_length:
        raise TurboEncodeError(
            f"Invalid codeword length generated: expected {expected_length}, got {len(codeword)}"
        )

    return codeword


@dataclasses.dataclass
class TurboFrame:
    DENOMINATOR: ClassVar[Literal[2, 3, 4, 6]]
    ASM: ClassVar[bytes]
    encoded: bytes | None = None
    decoded: bytes | None = None

    def encode(self) -> None:
        """Encode the `decoded` value according to the specific turbo encoding algorithm

        Resulting value will be stored in `encoded`.

        Encoding errors will raise a `TurboEncodeException`"""
        if self.decoded is None:
            raise TurboEncodeError("Cannot encode: 'decoded' is not set")

        payload_length = len(self.decoded)
        if payload_length not in _VALID_INFORMATION_BYTES:
            raise TurboEncodeError(
                "Invalid payload length for Turbo encoding. "
                f"Got {payload_length} bytes; expected one of {_VALID_INFORMATION_BYTES} bytes."
            )

        information_bits = _bytes_to_bits(self.decoded)
        codeword_bits = _build_codeword_bits(information_bits, self.DENOMINATOR)
        self.encoded = self.ASM + _bits_to_bytes(codeword_bits)

    def decode(self) -> None:
        """Decode the `encoded` value according to the specific turbo decoding algorithm.

        Resulting value will be stored in `decoded`.

        Decoding errors will raise a `TurboDecodeException`"""
        if self.encoded is None:
            raise TurboDecodeError("Cannot decode: 'encoded' is not set")

        codeword_bytes_by_information_bits = {
            information_bits: _packed_codeword_bytes(information_bits, self.DENOMINATOR)
            for information_bits in _VALID_INFORMATION_BITS
        }
        information_bits_by_codeword_bytes = {
            codeword_bytes: information_bits
            for information_bits, codeword_bytes in codeword_bytes_by_information_bits.items()
        }

        codeword_bytes: bytes | None = None
        information_bits_count: int | None = None

        if self.encoded.startswith(self.ASM):
            candidate = self.encoded[len(self.ASM) :]
            information_bits_count = information_bits_by_codeword_bytes.get(
                len(candidate)
            )
            if information_bits_count is not None:
                codeword_bytes = candidate

        if codeword_bytes is None:
            information_bits_count = information_bits_by_codeword_bytes.get(
                len(self.encoded)
            )
            if information_bits_count is not None:
                codeword_bytes = self.encoded

        if codeword_bytes is None or information_bits_count is None:
            raw_lengths = tuple(sorted(information_bits_by_codeword_bytes))
            asm_lengths = tuple(length + len(self.ASM) for length in raw_lengths)
            raise TurboDecodeError(
                "Invalid encoded length for Turbo decoding. "
                f"Got {len(self.encoded)} bytes; expected raw codeword lengths {raw_lengths} "
                f"or ASM-prefixed lengths {asm_lengths}."
            )

        total_codeword_bits = (
            information_bits_count + _TERMINATION_BITS
        ) * self.DENOMINATOR
        codeword_bits = _truncate_bits_with_padding_check(
            codeword_bytes, total_codeword_bits
        )

        if self.DENOMINATOR == 2:
            systematic = codeword_bits[::2]
        else:
            systematic = codeword_bits[:: self.DENOMINATOR]

        expected_systematic_length = information_bits_count + _TERMINATION_BITS
        if len(systematic) != expected_systematic_length:
            raise TurboDecodeError(
                "Invalid systematic stream length extracted from codeword. "
                f"Expected {expected_systematic_length}, got {len(systematic)}."
            )

        candidate_information_bits = systematic[:information_bits_count]
        expected_codeword_bits = _build_codeword_bits(
            candidate_information_bits, self.DENOMINATOR
        )
        if expected_codeword_bits != codeword_bits:
            raise TurboDecodeError(
                "Codeword parity/termination validation failed; frame cannot be decoded"
            )

        self.decoded = _bits_to_bytes(candidate_information_bits)


@dataclasses.dataclass
class Turbo2Frame(TurboFrame):
    DENOMINATOR = 2
    ASM = bytes.fromhex("034776C7272895B0")


@dataclasses.dataclass
class Turbo3Frame(TurboFrame):
    DENOMINATOR = 3
    ASM = bytes.fromhex("25D5C0CE8990F6C9461BF79C")


@dataclasses.dataclass
class Turbo4Frame(TurboFrame):
    DENOMINATOR = 4
    ASM = bytes.fromhex("034776C7272895B0 FCB88938D8D76A4F")


@dataclasses.dataclass
class Turbo6Frame(TurboFrame):
    DENOMINATOR = 6
    ASM = bytes.fromhex("25D5C0CE8990F6C9461BF79C DA2A3F31766F0936B9E40863")


if __name__ == "__main__":
    pass
