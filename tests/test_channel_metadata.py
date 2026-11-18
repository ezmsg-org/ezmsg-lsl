"""Tests for structured channel metadata roundtrip through LSL stream descriptors."""

import numpy as np
import pylsl
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.lsl.inlet import _parse_channel_metadata
from ezmsg.lsl.outlet import populate_desc_from_axisarray

CHANNEL_DTYPE = np.dtype(
    [
        ("x", "f4"),
        ("y", "f4"),
        ("label", "U16"),
        ("bank", "U1"),
        ("elec", "i4"),
    ]
)


def _write_outlet_metadata(info: pylsl.StreamInfo, ch_data: np.ndarray) -> None:
    """Write *ch_data* into *info*'s desc exactly as the outlet does."""
    msg = AxisArray(
        data=np.zeros((1, len(ch_data))),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=100.0), "ch": AxisArray.CoordinateAxis(data=ch_data, dims=["ch"])},
    )
    populate_desc_from_axisarray(info, msg, out_size=len(ch_data))


def test_roundtrip_structured_metadata():
    """Structured channel data survives outlet write → inlet parse."""
    n_ch = 4
    ch_data = np.zeros(n_ch, dtype=CHANNEL_DTYPE)
    ch_data[0] = (1.0, 2.0, "ch1", "A", 1)
    ch_data[1] = (3.0, 4.0, "ch2", "A", 2)
    ch_data[2] = (5.0, 6.0, "ch3", "B", 1)
    ch_data[3] = (7.0, 8.0, "ch4", "B", 2)

    info = pylsl.StreamInfo("test", "EEG", n_ch, 100.0, pylsl.cf_float32, "test_roundtrip")
    _write_outlet_metadata(info, ch_data)

    result = _parse_channel_metadata(info.desc().child("channels"), n_ch)

    assert result is not None
    assert "label" in result.dtype.names
    assert "x" in result.dtype.names
    assert "y" in result.dtype.names
    assert "bank" in result.dtype.names
    assert "elec" in result.dtype.names

    for i in range(n_ch):
        assert result[i]["label"] == ch_data[i]["label"]
        assert np.isclose(result[i]["x"], ch_data[i]["x"])
        assert np.isclose(result[i]["y"], ch_data[i]["y"])
        assert result[i]["bank"] == ch_data[i]["bank"]
        assert result[i]["elec"] == ch_data[i]["elec"]


def test_roundtrip_label_only():
    """Channels with only a label field produce a single-field struct."""
    n_ch = 3
    info = pylsl.StreamInfo("test", "EEG", n_ch, 100.0, pylsl.cf_float32, "test_label_only")
    chans = info.desc().append_child("channels")
    for name in ("Fp1", "Fp2", "Cz"):
        chans.append_child("channel").append_child_value("label", name)

    result = _parse_channel_metadata(info.desc().child("channels"), n_ch)

    assert result is not None
    assert result.dtype.names == ("label",)
    assert list(result["label"]) == ["Fp1", "Fp2", "Cz"]


def test_parse_empty_channels():
    """An empty <channels> element returns None."""
    info = pylsl.StreamInfo("test", "EEG", 2, 100.0, pylsl.cf_float32, "test_empty")
    # desc has no <channels> child
    result = _parse_channel_metadata(info.desc().child("channels"), 2)
    assert result is None


def test_parse_channel_count_mismatch():
    """Returns None when XML channel count doesn't match expected n_ch."""
    info = pylsl.StreamInfo("test", "EEG", 2, 100.0, pylsl.cf_float32, "test_mismatch")
    chans = info.desc().append_child("channels")
    chans.append_child("channel").append_child_value("label", "ch1")
    # Only 1 channel in XML but n_ch=2

    result = _parse_channel_metadata(info.desc().child("channels"), 2)
    assert result is None


def test_dtype_inference():
    """Integer, float, and string fields get the expected dtypes."""
    n_ch = 2
    info = pylsl.StreamInfo("test", "EEG", n_ch, 100.0, pylsl.cf_float32, "test_dtype")
    chans = info.desc().append_child("channels")

    ch1 = chans.append_child("channel")
    ch1.append_child_value("label", "ch1")
    ch1.append_child_value("elec", "42")
    loc1 = ch1.append_child("location")
    loc1.append_child_value("X", "1.5")
    loc1.append_child_value("Y", "2.5")

    ch2 = chans.append_child("channel")
    ch2.append_child_value("label", "ch2")
    ch2.append_child_value("elec", "43")
    loc2 = ch2.append_child("location")
    loc2.append_child_value("X", "3.5")
    loc2.append_child_value("Y", "4.5")

    result = _parse_channel_metadata(info.desc().child("channels"), n_ch)

    assert result is not None
    # label should be a Unicode string type
    assert result.dtype["label"].kind == "U"
    # elec should be inferred as int, at full width
    assert result.dtype["elec"] == np.int64
    # x, y should be inferred as float, at full precision
    assert result.dtype["x"] == np.float64
    assert result.dtype["y"] == np.float64

    assert result[0]["elec"] == 42
    assert np.isclose(result[1]["x"], 3.5)


def _roundtrip(ch_data: np.ndarray) -> np.ndarray:
    info = pylsl.StreamInfo("test", "EEG", len(ch_data), 100.0, pylsl.cf_float32, "test_roundtrip")
    _write_outlet_metadata(info, ch_data)
    result = _parse_channel_metadata(info.desc().child("channels"), len(ch_data))
    assert result is not None
    return result


def _assert_same_values(result: np.ndarray, ch_data: np.ndarray) -> None:
    """Same field names and values; dtypes (and field order) may differ."""
    assert set(result.dtype.names) == set(ch_data.dtype.names)
    for f in ch_data.dtype.names:
        assert result[f].tolist() == ch_data[f].tolist(), f


@pytest.mark.parametrize(
    "ch_data",
    [
        pytest.param(np.array([("1",), ("2",)], dtype=[("label", "U4")]), id="numeric-labels"),
        pytest.param(np.array([("a", "007"), ("b", "010")], dtype=[("label", "U4"), ("id", "U3")]), id="zero-padded"),
        pytest.param(
            np.array(
                [("a", 2**40, 2**64 - 1), ("b", -(2**62), 3)], dtype=[("label", "U1"), ("big", "i8"), ("ubig", "u8")]
            ),
            id="large-ints",
        ),
        pytest.param(np.array([("a", 0.1), ("b", 1 / 3)], dtype=[("label", "U1"), ("v", "f8")]), id="float64"),
        pytest.param(np.array([("a", True), ("b", False)], dtype=[("label", "U1"), ("flag", "?")]), id="bools"),
        pytest.param(
            np.array(
                [("1", "uV", "EEG", 3, 0.25, True, "A"), ("2", "mV", "EMG", 4, -1.5, False, "1")],
                dtype=[
                    ("label", "U2"),
                    ("unit", "U2"),
                    ("type", "U3"),
                    ("elec", "i4"),
                    ("gain", "f8"),
                    ("ok", "?"),
                    ("bank", "U1"),
                ],
            ),
            id="mixed",
        ),
        pytest.param(
            np.array(
                [("a", 1.5, -2.0, 0.125), ("b", 3.25, 4.0, 5.5)],
                dtype=[("label", "U1"), ("x", "f8"), ("y", "f8"), ("z", "f8")],
            ),
            id="xyz",
        ),
    ],
)
def test_outlet_inlet_roundtrip_preserves_values(ch_data):
    _assert_same_values(_roundtrip(ch_data), ch_data)


def test_roundtrip_name_is_published_as_label():
    ch_data = np.array([("1", 7), ("2", 8)], dtype=[("name", "U1"), ("elec", "i8")])
    result = _roundtrip(ch_data)
    assert result["label"].tolist() == ["1", "2"]
    assert result["elec"].tolist() == [7, 8]


def test_roundtrip_bytes_fields_are_decoded():
    ch_data = np.array([(b"a", b"x1"), (b"b", b"x2")], dtype=[("label", "S1"), ("kind", "S2")])
    result = _roundtrip(ch_data)
    assert result["label"].tolist() == ["a", "b"]
    assert result["kind"].tolist() == ["x1", "x2"]


@pytest.mark.parametrize("bad", ["1st", "has space", "xmlish"])
def test_outlet_rejects_invalid_xml_field_names(bad):
    ch_data = np.zeros(2, dtype=[("label", "U1"), (bad, "i4")])
    info = pylsl.StreamInfo("test", "EEG", 2, 100.0, pylsl.cf_float32, "test_bad_name")
    with pytest.raises(ValueError, match="XML element names"):
        _write_outlet_metadata(info, ch_data)


def test_field_missing_on_some_channels():
    """A numeric field absent on some channels is NaN there, not 0; a bool one stays text."""
    info = pylsl.StreamInfo("test", "EEG", 3, 100.0, pylsl.cf_float32, "test_gaps")
    chans = info.desc().append_child("channels")
    for label, extra in (
        ("a", {"elec": "5", "gain": "0.5", "flag": "True"}),
        ("b", {}),
        ("c", {"elec": "7", "flag": "False"}),
    ):
        chan = chans.append_child("channel")
        chan.append_child_value("label", label)
        for k, v in extra.items():
            chan.append_child_value(k, v)

    result = _parse_channel_metadata(info.desc().child("channels"), 3)

    assert result is not None
    assert result["label"].tolist() == ["a", "b", "c"]
    np.testing.assert_array_equal(result["elec"], [5.0, np.nan, 7.0])
    np.testing.assert_array_equal(result["gain"], [0.5, np.nan, np.nan])
    assert result["flag"].tolist() == ["True", "", "False"]


def test_foreign_desc_with_labels_only():
    """A desc written by another LSL application: label text is kept as-is."""
    info = pylsl.StreamInfo("test", "EEG", 3, 100.0, pylsl.cf_float32, "test_foreign")
    chans = info.desc().append_child("channels")
    for name in ("01", "2", "Cz"):
        chans.append_child("channel").append_child_value("label", name)

    result = _parse_channel_metadata(info.desc().child("channels"), 3)

    assert result is not None
    assert result.dtype.names == ("label",)
    assert result["label"].tolist() == ["01", "2", "Cz"]
