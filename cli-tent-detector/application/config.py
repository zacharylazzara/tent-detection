from enum import StrEnum
import enum
from pathlib import Path


class IOFormat(StrEnum):
    pass


class OFormat(IOFormat):
    model = 'pth'
    spreadsheet = 'csv'
    image = 'png'


class PathEnum(type(Path()), enum.Enum):
    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, (Path, enum.auto)):
            raise TypeError(f"Values of PathEnums must be of type {type(Path())}: {value!r} is of type {type(value)}")
        return super().__new__(cls, value, *args, **kwargs)
    
    def __truediv__(self, arg):
        return Path(self) / arg

    def __str__(self):
        return str(Path(self))