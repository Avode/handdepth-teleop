"""Append-only packet log, associated pose JSONL and session metadata."""
import datetime
import json
import os
from pathlib import Path
import struct
import sys
import time
from . import __version__
from .protocol import MAX_MESSAGE, ProtocolError

DEFAULT_ROOT = Path("/mnt/robotics-data/robotics/agibot-g2/datasets/handdepth")

def check_recording_root(root, fixture=False):
    root = Path(root).expanduser().resolve()
    if fixture:
        # Tests use caller-selected temporary directories, never the Linux production root on Mac.
        if str(root).startswith("/mnt/"):
            raise ValueError("fixture root must be local")
    else:
        if sys.platform != "linux":
            raise ValueError("production recording requires Linux; use --fixture-storage for local tests")
        # Require an actual mounted ancestor other than /. Avoid silent writes onto root SSD.
        mount = next((p for p in [root, *root.parents] if p != Path("/") and os.path.ismount(p)), None)
        if mount is None:
            raise ValueError("recording HDD is not mounted at any ancestor of " + str(root))
        stat = os.statvfs(mount)
        if stat.f_flag & os.ST_RDONLY:
            raise ValueError("recording mount is read-only")
    root.mkdir(parents=True, exist_ok=True)
    probe = root / (".write-probe-" + str(os.getpid()))
    try:
        with probe.open("xb") as f:
            f.write(b"handdepth")
        probe.unlink()
    except OSError as e:
        raise ValueError("recording root not writable") from e
    return root

class Recorder:
    def __init__(self, root, fixture=False):
        root = check_recording_root(root, fixture)
        self.path = root / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + os.urandom(4).hex())
        self.path.mkdir()
        (self.path/"metadata.json").write_text(json.dumps({"application_version": __version__, "protocol_version": 1,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "format": "HDREC1",
            "receiver_clock": "python_monotonic_seconds", "fixture_storage": fixture}, indent=2))
        self.packets = (self.path/"packets.hdr").open("wb")
        self.packets.write(b"HDREC1\n")
        self.poses = (self.path/"poses.jsonl").open("w")
        self.calibrations = set()
        self.closed = False
        self.refinements = None

    def refinement(self,pose):
        if self.refinements is None:self.refinements=(self.path/'pc-poses.jsonl').open('w')
        self.refinements.write(json.dumps(pose,allow_nan=False)+'\n');self.refinements.flush()

    def append(self, channel, raw, received_s, pose=None):
        if len(raw) > MAX_MESSAGE or channel not in ("sensor", "preview"):
            raise ProtocolError("recording message")
        # network-endian: channel byte, receive monotonic Float64, packet UInt32 length
        self.packets.write(struct.pack(">BdI", channel == "preview", received_s, len(raw)))
        self.packets.write(raw)
        if pose is not None:
            self.poses.write(json.dumps(dict(pose, received_monotonic_s=received_s), allow_nan=False)+"\n")
            from .protocol import decode
            h = decode(raw).header
            identity = (h["session_id"], h["calibration_id"])
            if identity not in self.calibrations:
                self.calibrations.add(identity)
                folder = self.path/"calibration"
                folder.mkdir(exist_ok=True)
                # Hash file names rather than allowing sender-controlled paths.
                import hashlib
                name = hashlib.sha256(repr(identity).encode()).hexdigest()
                (folder/(name+".json")).write_text(json.dumps({"session_id": h["session_id"], "calibration_id": h["calibration_id"], "calibration": h.get("calibration")}, allow_nan=False, indent=2))

    def event(self, pose):
        self.poses.write(json.dumps(pose, allow_nan=False)+"\n")

    def close(self):
        if not self.closed:
            self.packets.close(); self.poses.close(); self.closed = True
            if self.refinements:self.refinements.close()


def replay_packets(path):
    with (Path(path)/"packets.hdr").open("rb") as f:
        if f.read(7) != b"HDREC1\n":
            raise ProtocolError("recording magic")
        while raw := f.read(13):
            if len(raw) != 13:
                raise ProtocolError("truncated record")
            channel, received, n = struct.unpack(">BdI", raw)
            if channel > 1 or not 4 <= n <= MAX_MESSAGE or not received >= 0:
                raise ProtocolError("record length/channel/time")
            packet = f.read(n)
            if len(packet) != n:
                raise ProtocolError("truncated packet")
            yield ("preview" if channel else "sensor"), received, packet
