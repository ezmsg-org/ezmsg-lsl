"""Tests for structured channel metadata roundtrip through LSL stream descriptors."""

import numpy as np
import pylsl

from ezmsg.lsl.inlet import _parse_channel_metadata

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
    """Replicate the outlet's structured-array → XML logic."""
    label_field = next((f for f in ("label", "name") if f in ch_data.dtype.names), None)
    loc_fields = {f: f.upper() for f in ("x", "y", "z") if f in ch_data.dtype.names}
    other_fields = [f for f in ch_data.dtype.names if f not in (label_field, "x", "y", "z")]

    chans = info.desc().append_child("channels")
    for ch in ch_data:
        chan = chans.append_child("channel")
        if label_field is not None:
            chan.append_child_value("label", str(ch[label_field]))
        if loc_fields:
            loc = chan.append_child("location")
            for struct_f, xml_f in loc_fields.items():
                loc.append_child_value(xml_f, str(ch[struct_f]))
        for f in other_fields:
            chan.append_child_value(f, str(ch[f]))


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
    # elec should be inferred as int
    assert result.dtype["elec"].kind == "i"
    # x, y should be inferred as float
    assert result.dtype["x"].kind == "f"
    assert result.dtype["y"].kind == "f"

    assert result[0]["elec"] == 42
    assert np.isclose(result[1]["x"], 3.5)
