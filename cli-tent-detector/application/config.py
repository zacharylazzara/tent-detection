from enum import StrEnum

# class StrEnum(str, Enum):
#     def __str__(self) -> str:
#         return str(self.value)

class IOFormat(StrEnum):
    pass

# TODO: remove IFormat (at least for image, as we should get the file extension from the file itself)
class IFormat(IOFormat):
    model = 'pth'
    spreadsheet = 'csv'
    image = 'jpg'


class OFormat(IOFormat):
    model = 'pth'
    spreadsheet = 'csv'
    image = 'png'
    