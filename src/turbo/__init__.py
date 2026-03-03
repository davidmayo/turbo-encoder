import dataclasses
from typing import ClassVar, Literal


class TurboError(RuntimeError):
    """An error with a Turbo encoding/decoding process"""


class TurboEncodeError(TurboError):
    """An error with a Turbo encoding process"""


class TurboDecodeError(TurboError):
    """An error with a Turbo decoding process"""


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
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the 'encode()' method"
        )

    def decode(self) -> None:
        """Decode the `encoded` value according to the specific turbo decoding algorithm.

        Resulting value will be stored in `decoded`.

        Decoding errors will raise a `TurboDecodeException`"""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the 'decode()' method"
        )


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
