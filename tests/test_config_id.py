"""Every message declares its stream's ``config_id``, digested once per connection, and
``segment`` numbers start at 0 in every inlet."""

from unittest.mock import patch

import numpy as np
from ezmsg.util.messages.axisarray import chain_config_id, compute_config_id
from test_inlet import _FakeStreamInfo
from test_segments import SRATE, _producer, _pull, _reconnect, _stamps

from ezmsg.lsl import inlet


def expected_config_id(m) -> int:
    """The layout digest with the stream's identity folded in."""
    return chain_config_id(compute_config_id(m), m.attrs["lsl_source_id"], m.attrs["lsl_hostname"])


def _declares_its_config(msgs) -> int:
    ids = {m.config_id for m in msgs}
    assert len(ids) == 1
    for m in msgs:
        assert m.config_id != 0
        assert m.config_id == expected_config_id(m)
    return ids.pop()


def test_regular_stream_declares_one_config_id_computed_once():
    with patch.object(inlet, "compute_config_id", wraps=compute_config_id) as spy:
        producer = _producer()
        msgs = _pull(producer, _stamps(100.0, 50))
        msgs += _pull(producer, np.concatenate([_stamps(100.05, 30), _stamps(103.0, 20)]))  # cut at a break
        msgs += _pull(producer, _stamps(103.02, 50))
    assert len(msgs) == 4
    assert msgs[0].data.dtype == np.float32
    _declares_its_config(msgs)
    assert spy.call_count == 1


def test_irregular_stream_declares_one_config_id():
    producer = _producer(_FakeStreamInfo(srate=0.0))
    msgs = _pull(producer, np.array([100.0, 100.4, 160.0])) + _pull(producer, np.array([161.0, 170.0]))
    _declares_its_config(msgs)


def test_config_id_follows_the_stream_across_reconnects():
    producer = _producer()
    (first,) = _pull(producer, _stamps(100.0, 50))
    _reconnect(producer, _FakeStreamInfo(srate=SRATE, uid="uid-2"))  # same stream, restarted outlet
    (same,) = _pull(producer, _stamps(100.05, 50))
    _reconnect(producer, _FakeStreamInfo(srate=SRATE, n_ch=3))
    (wider,) = _pull(producer, _stamps(100.10, 50))
    _reconnect(producer, _FakeStreamInfo(srate=SRATE / 2, n_ch=3))
    (slower,) = _pull(producer, _stamps(100.15, 25))
    _reconnect(producer, _FakeStreamInfo(srate=SRATE / 2, n_ch=3, source_id="src-2"))
    (other,) = _pull(producer, _stamps(100.20, 25))  # another stream of the same layout
    assert same.config_id == first.config_id
    assert len({first.config_id, wider.config_id, slower.config_id, other.config_id}) == 4
    for m in (wider, slower, other):
        assert m.config_id == expected_config_id(m)
    # A reconnect is no break: none was measured in the timestamps.
    assert {m.segment for m in (first, same, wider, slower, other)} == {first.segment}


def test_two_fresh_inlets_both_start_at_segment_0():
    a = _pull(_producer(), _stamps(100.0, 10))[0]
    b = _pull(_producer(), _stamps(100.0, 10))[0]
    assert a.segment == b.segment == 0
