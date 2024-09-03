from dataclasses import dataclass, field
import typing

from ezmsg.util.messages.axisarray import AxisArray


@dataclass
class CustomAxis(AxisArray.Axis):
    labels: typing.List[str] = field(default_factory=lambda: [])

    @classmethod
    def SpaceAxis(cls, labels: typing.List[str]):  # , locs: typing.Optional[npt.NDArray] = None):
        return cls(unit="mm", labels=labels)


# Monkey-patch AxisArray with our customized Axis
AxisArray.Axis = CustomAxis
