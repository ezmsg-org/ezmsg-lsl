Using ezmsg-lsl
===============

This guide explains how to use ezmsg-lsl to receive and send data via Lab Streaming Layer (LSL).

.. contents:: On this page
   :local:
   :depth: 2


Overview
--------

ezmsg-lsl provides two main components:

- **LSLInletUnit**: Receives data from an LSL stream and publishes it as ``AxisArray`` messages
- **LSLOutletUnit**: Subscribes to ``AxisArray`` messages and sends them to an LSL outlet

Both components handle clock synchronization between LSL time and system time automatically.


Receiving Data with LSLInletUnit
--------------------------------

Basic Usage
^^^^^^^^^^^

To receive data from an LSL stream:

.. code-block:: python

   import ezmsg.core as ez
   from ezmsg.lsl.inlet import LSLInletUnit, LSLInletSettings, LSLInfo

   # Create the inlet
   inlet = LSLInletUnit(
       LSLInletSettings(
           info=LSLInfo(
               name="MyEEGStream",  # Name of the LSL stream
               type="EEG",          # Type of the LSL stream
           ),
       )
   )

   # Use in a pipeline
   components = {"INLET": inlet, ...}
   connections = ((inlet.OUTPUT_SIGNAL, next_unit.INPUT_SIGNAL), ...)
   ez.run(components=components, connections=connections)


Finding Streams
^^^^^^^^^^^^^^^

You can match streams by various criteria. All fields are optional - leave empty to match any:

.. code-block:: python

   LSLInfo(
       name="MyStream",        # Match by stream name
       type="EEG",             # Match by stream type
       host="localhost",       # Match by hostname
       channel_count=8,        # Match by number of channels
       nominal_srate=500.0,    # Match by sampling rate
       channel_format="float32",  # Match by data format
   )

If multiple streams match, the first one found is used.


Output Format
^^^^^^^^^^^^^

``LSLInletUnit`` produces ``AxisArray`` messages with:

- **dims**: ``["time", "ch"]``
- **data**: numpy array of shape ``(n_samples, n_channels)``
- **axes**:
  - ``time``: ``TimeAxis`` for regular streams, ``CoordinateAxis`` for irregular streams
  - ``ch``: ``CoordinateAxis`` with channel labels from the LSL stream metadata


Regular vs Irregular Streams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LSL streams can have regular sampling (e.g., EEG at 500 Hz) or irregular sampling
(e.g., event markers). ezmsg-lsl handles both:

**Regular streams** (``nominal_srate > 0``):

.. code-block:: python

   # Output axes["time"] is a TimeAxis
   axes["time"] = AxisArray.TimeAxis(fs=500.0, offset=start_time)

**Irregular streams** (``nominal_srate == 0``):

.. code-block:: python

   # Output axes["time"] is a CoordinateAxis with individual timestamps
   axes["time"] = AxisArray.CoordinateAxis(
       data=np.array([t1, t2, t3, ...]),  # Timestamp per sample
       dims=["time"],
       unit="s",
   )


Clock Synchronization
^^^^^^^^^^^^^^^^^^^^^

By default, timestamps are converted from LSL clock to system time (``time.time()``).
You can control this behavior:

.. code-block:: python

   LSLInletSettings(
       info=LSLInfo(name="MyStream"),
       use_arrival_time=False,   # Use LSL send timestamps (default)
       use_lsl_clock=False,      # Convert to system time (default)
   )

   # Alternative: Use arrival time instead of send time
   LSLInletSettings(
       info=LSLInfo(name="MyStream"),
       use_arrival_time=True,    # Use time.time() when data arrives
   )

   # Alternative: Keep LSL clock (useful when both ends use LSL)
   LSLInletSettings(
       info=LSLInfo(name="MyStream"),
       use_lsl_clock=True,       # Don't convert to system time
   )


Buffer Size
^^^^^^^^^^^

Control how much data is pulled at once:

.. code-block:: python

   LSLInletSettings(
       info=LSLInfo(name="MyStream"),
       local_buffer_dur=1.0,  # Buffer up to 1 second of data
   )


Sending Data with LSLOutletUnit
-------------------------------

Basic Usage
^^^^^^^^^^^

To send data to an LSL outlet:

.. code-block:: python

   import ezmsg.core as ez
   from ezmsg.lsl.outlet import LSLOutletUnit, LSLOutletSettings

   outlet = LSLOutletUnit(
       LSLOutletSettings(
           stream_name="MyOutput",
           stream_type="Markers",
       )
   )

   # Use in a pipeline
   components = {..., "OUTLET": outlet}
   connections = (..., (prev_unit.OUTPUT_SIGNAL, outlet.INPUT_SIGNAL))
   ez.run(components=components, connections=connections)


Input Format
^^^^^^^^^^^^

``LSLOutletUnit`` accepts any ``AxisArray``. It automatically:

- Detects sampling rate from ``TimeAxis`` (or uses irregular rate for ``CoordinateAxis``)
- Flattens multi-dimensional data to channels
- Extracts channel labels from the ``ch`` axis if present


Timestamp Handling
^^^^^^^^^^^^^^^^^^

By default, the incoming message timestamps are preserved and converted to LSL clock:

.. code-block:: python

   LSLOutletSettings(
       stream_name="MyOutput",
       stream_type="Data",
       use_message_timestamp=True,   # Use timestamps from AxisArray (default)
       assume_lsl_clock=False,       # Convert from system time to LSL (default)
   )

   # Alternative: Ignore message timestamps, use current time
   LSLOutletSettings(
       stream_name="MyOutput",
       stream_type="Data",
       use_message_timestamp=False,  # Use current pylsl.local_clock()
   )


Complete Example
----------------

Here's a complete pipeline that receives EEG, processes it, and sends results:

.. code-block:: python

   import ezmsg.core as ez
   from ezmsg.lsl.inlet import LSLInletUnit, LSLInletSettings, LSLInfo
   from ezmsg.lsl.outlet import LSLOutletUnit, LSLOutletSettings
   from ezmsg.sigproc.butterworthfilter import ButterworthFilter, ButterworthFilterSettings

   components = {
       "INLET": LSLInletUnit(
           LSLInletSettings(
               info=LSLInfo(name="RawEEG", type="EEG"),
           )
       ),
       "FILTER": ButterworthFilter(
           ButterworthFilterSettings(order=4, cuton=1.0, cutoff=40.0)
       ),
       "OUTLET": LSLOutletUnit(
           LSLOutletSettings(stream_name="FilteredEEG", stream_type="EEG")
       ),
   }

   connections = (
       (components["INLET"].OUTPUT_SIGNAL, components["FILTER"].INPUT_SIGNAL),
       (components["FILTER"].OUTPUT_SIGNAL, components["OUTLET"].INPUT_SIGNAL),
   )

   if __name__ == "__main__":
       ez.run(components=components, connections=connections)


Multiple Streams
----------------

You can receive from multiple LSL streams by creating multiple inlet units:

.. code-block:: python

   components = {
       "EEG": LSLInletUnit(
           LSLInletSettings(info=LSLInfo(name="EEGStream", type="EEG"))
       ),
       "MARKERS": LSLInletUnit(
           LSLInletSettings(info=LSLInfo(name="MarkerStream", type="Markers"))
       ),
       # ... processing units ...
   }

Each inlet runs independently and produces messages as data becomes available.


Using Without ezmsg Pipeline
----------------------------

You can also use the generator directly for scripting or testing:

.. code-block:: python

   from ezmsg.lsl.inlet import LSLInletGenerator, LSLInletSettings, LSLInfo

   # Create generator
   gen = LSLInletGenerator(
       settings=LSLInletSettings(
           info=LSLInfo(name="MyStream"),
       )
   )

   # Pull data
   for _ in range(100):
       msg = next(gen)
       if msg is not None and msg.data.size > 0:
           print(f"Received {msg.data.shape[0]} samples")

   # Clean up
   gen.shutdown()


Troubleshooting
---------------

**Stream not found**:
  - Verify the stream is running with ``pylsl.resolve_streams()``
  - Check that name/type match exactly (case-sensitive)
  - Ensure no firewall is blocking LSL traffic (UDP ports)

**High latency**:
  - Reduce ``local_buffer_dur`` for faster updates
  - Check system load and network conditions

**Clock drift**:
  - Use ``use_arrival_time=True`` if send timestamps are unreliable
  - Ensure both systems have synchronized clocks for best results

**Empty messages**:
  - This is normal when no new data is available
  - Downstream units should handle empty arrays gracefully
