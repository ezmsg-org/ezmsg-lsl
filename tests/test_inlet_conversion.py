"""Stream-level ``<desc>`` scalars arrive as attrs; ``try_convert`` undoes a digitization.

The outlet writes every ``message.attrs`` entry as a top-level ``<desc>``
element (``populate_desc_from_axisarray``); ezmsg-sigproc's ``Digitize``
records ``conversion`` and ``offset`` there. The inlet reads them back, and
with ``try_convert`` emits ``sample * conversion + offset``.
"""

import time
import uuid

import numpy as np
import pylsl
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.lsl.inlet import LSLInfo, LSLInletProducer, LSLInletSettings, _parse_stream_attrs, digitization
from ezmsg.lsl.outlet import populate_desc_from_axisarray


def _info(name: str = "conv-test", n_ch: int = 2, fmt=pylsl.cf_int16, **desc: str) -> pylsl.StreamInfo:
    info = pylsl.StreamInfo(name, "EEG", n_ch, 100.0, fmt, name)
    for key, value in desc.items():
        info.desc().append_child_value(key, value)
    return info


class TestStreamAttrs:
    def test_top_level_scalars_become_attrs(self):
        info = _info(conversion="0.25", offset="-2", unit="microvolts", note="1.5")
        info.desc().append_child("channels").append_child("channel").append_child_value("label", "a")
        info.desc().append_child("acquisition").append_child_value("model", "Hub")

        attrs = _parse_stream_attrs(info.desc())

        # conversion/offset are numbers; other values stay strings, as written.
        assert attrs == {"conversion": 0.25, "offset": -2.0, "unit": "microvolts", "note": "1.5"}

    def test_unparseable_conversion_stays_a_string_and_is_not_a_digitization(self):
        attrs = _parse_stream_attrs(_info(conversion="n/a").desc())

        assert attrs == {"conversion": "n/a"}
        assert digitization(attrs) is None

    def test_round_trips_what_the_outlet_writes(self):
        conversion, offset = 1000.0 / 65535.0, -0.007629510948348184
        msg = AxisArray(
            data=np.zeros((1, 2), dtype=np.int16),
            dims=["time", "ch"],
            axes={"time": AxisArray.TimeAxis(fs=100.0), "ch": CoordinateAxis(np.array(["a", "b"]), dims=["ch"])},
            attrs={"conversion": conversion, "offset": offset, "unit": "uV"},
        )
        info = _info()
        populate_desc_from_axisarray(info, msg, out_size=2)

        attrs = _parse_stream_attrs(info.desc())

        assert attrs == {"conversion": conversion, "offset": offset, "unit": "uV"}

    def test_digitization_defaults_offset_to_zero(self):
        assert digitization({"conversion": 0.5}) == (0.5, 0.0)
        assert digitization({"unit": "uV"}) is None


def _first_message(producer: LSLInletProducer, outlet: pylsl.StreamOutlet, chunk: np.ndarray) -> AxisArray:
    deadline = time.monotonic() + 10.0
    for msg in producer:
        outlet.push_chunk(chunk)
        if msg is not None and msg.data.size:
            return msg
        assert time.monotonic() < deadline, "no data from the outlet"
    raise AssertionError("producer stopped")


@pytest.mark.parametrize("try_convert", [False, True])
def test_try_convert_undoes_the_declared_digitization(try_convert: bool):
    name = f"conv-{uuid.uuid4().hex[:8]}"
    outlet = pylsl.StreamOutlet(_info(name, conversion="0.5", offset="1", unit="uV"))
    producer = LSLInletProducer(settings=LSLInletSettings(info=LSLInfo(name=name), try_convert=try_convert))
    try:
        msg = _first_message(producer, outlet, np.full((5, 2), 10, dtype=np.int16))
    finally:
        producer.shutdown()
        del outlet

    assert msg.attrs["unit"] == "uV"
    if try_convert:
        assert msg.data.dtype == np.float64
        np.testing.assert_array_equal(msg.data, 6.0)
        assert "conversion" not in msg.attrs and "offset" not in msg.attrs
    else:
        np.testing.assert_array_equal(msg.data, 10)
        assert (msg.attrs["conversion"], msg.attrs["offset"]) == (0.5, 1.0)


def test_try_convert_leaves_an_undeclared_stream_alone():
    name = f"conv-{uuid.uuid4().hex[:8]}"
    outlet = pylsl.StreamOutlet(_info(name, unit="uV"))
    producer = LSLInletProducer(settings=LSLInletSettings(info=LSLInfo(name=name), try_convert=True))
    try:
        msg = _first_message(producer, outlet, np.full((5, 2), 10, dtype=np.int16))
    finally:
        producer.shutdown()
        del outlet

    assert msg.data.dtype == np.int16
    np.testing.assert_array_equal(msg.data, 10)
