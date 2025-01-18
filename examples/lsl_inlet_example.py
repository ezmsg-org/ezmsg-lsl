import ezmsg.core as ez
from ezmsg.lsl.units import LSLInletUnit, LSLInfo
from ezmsg.util.debuglog import DebugLog
import typer
# from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings


def main(stream_name: str = "", stream_type: str = "EEG"):
    comps = {
        "SRC": LSLInletUnit(info=LSLInfo(name=stream_name, type=stream_type)),
        "LOGGER": DebugLog(name="DEBUG", max_length=400),
        #  MessageLogger(output=file_path),
    }
    conns = ((comps["SRC"].OUTPUT_SIGNAL, comps["LOGGER"].INPUT),)
    ez.run(components=comps, connections=conns)


if __name__ == "__main__":
    typer.run(main)
