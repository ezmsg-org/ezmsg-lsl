"""Unit tests for the LSL outlet's StreamInfo desc population.

Exercises :func:`ezmsg.lsl.outlet.populate_desc_from_axisarray` in
isolation — without spinning up a real :class:`pylsl.StreamOutlet`, which
would touch the LSL network.  The function takes a fresh ``StreamInfo``
and an :class:`AxisArray`, and writes both:

* well-known stream-level ``attrs`` (``conversion``, ``offset``,
  ``unit``) as top-level desc XML elements; and
* a per-channel ``<channel>`` block when the message's ``"ch"`` axis is
  a :class:`CoordinateAxis`.
"""

import numpy as np
import pylsl
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.lsl.outlet import populate_desc_from_axisarray


def _make_int16_msg(
    channel_count: int = 3,
    n_time: int = 5,
    ch_labels=None,
    attrs=None,
) -> AxisArray:
    if ch_labels is None:
        ch_labels = [f"c{i}" for i in range(channel_count)]
    return AxisArray(
        data=np.zeros((n_time, channel_count), dtype=np.int16),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
            "ch": CoordinateAxis(data=np.array(ch_labels), dims=["ch"]),
        },
        attrs=attrs or {},
        key="test",
    )


def _make_info(
    channel_count: int = 3,
    name: str = "test",
    stream_type: str = "Features",
) -> pylsl.StreamInfo:
    return pylsl.StreamInfo(
        name=name,
        type=stream_type,
        channel_count=channel_count,
        nominal_srate=100.0,
        channel_format=pylsl.cf_int16,
        source_id=f"ezmsg-test-{name}",
    )


class TestStreamLevelAttrs:
    def test_known_attrs_emitted_as_desc_children(self):
        msg = _make_int16_msg(
            attrs={
                "conversion": 0.000244,
                "offset": 0.0,
                "unit": "a.u.",
            },
        )
        info = _make_info()
        populate_desc_from_axisarray(info, msg, out_size=3)
        xml = info.as_xml()
        # Each promoted attr appears as a top-level <key>value</key> in desc.
        assert "<conversion>0.000244</conversion>" in xml
        assert "<offset>0.0</offset>" in xml
        assert "<unit>a.u.</unit>" in xml

    def test_missing_attrs_do_not_appear(self):
        """Only attrs that are *present* on the message land in desc.

        A pipeline without a Digitize upstream — e.g. a raw int16 source
        with no scaling metadata — should produce a desc with no scaling
        elements at all (rather than placeholder zeros).
        """
        msg = _make_int16_msg(attrs={})
        info = _make_info()
        populate_desc_from_axisarray(info, msg, out_size=3)
        xml = info.as_xml()
        assert "<conversion>" not in xml
        assert "<offset>" not in xml
        assert "<unit>" not in xml

    def test_non_known_attrs_are_dropped(self):
        """Only the well-known attr keys ride the LSL XML.

        Other AxisArray.attrs entries stay on the message but don't
        appear in the stream descriptor, so the on-the-wire XML stays
        bounded regardless of how the upstream graph populates attrs.
        """
        msg = _make_int16_msg(
            attrs={"conversion": 0.5, "ignore_me": "private"},
        )
        info = _make_info()
        populate_desc_from_axisarray(info, msg, out_size=3)
        xml = info.as_xml()
        assert "<conversion>0.5</conversion>" in xml
        assert "ignore_me" not in xml


class TestPerChannelLabels:
    def test_simple_string_channel_labels(self):
        msg = _make_int16_msg(ch_labels=["alpha", "beta", "gamma"])
        info = _make_info(channel_count=3)
        populate_desc_from_axisarray(info, msg, out_size=3)
        xml = info.as_xml()
        for label in ("alpha", "beta", "gamma"):
            assert f"<label>{label}</label>" in xml

    def test_attrs_and_channel_labels_coexist(self):
        msg = _make_int16_msg(
            ch_labels=["c1-spk", "c1-sbp", "c2-spk"],
            attrs={"conversion": 0.000244, "unit": "a.u."},
        )
        info = _make_info(channel_count=3)
        populate_desc_from_axisarray(info, msg, out_size=3)
        xml = info.as_xml()
        # Stream-level scaling metadata...
        assert "<conversion>0.000244</conversion>" in xml
        assert "<unit>a.u.</unit>" in xml
        # ...and per-channel labels.
        for label in ("c1-spk", "c1-sbp", "c2-spk"):
            assert f"<label>{label}</label>" in xml
