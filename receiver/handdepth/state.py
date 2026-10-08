import collections
import math
import time
from .protocol import ProtocolError, finite

class ClockAlignment:
    """NTP-style remote-to-receiver offset with conservative freshness/jitter gates."""
    def __init__(self):
        self.samples = collections.deque(maxlen=8)
        self.last_sample = -math.inf

    def update(self, raw, now):
        if not isinstance(raw, dict):
            return
        offset = finite(raw.get("offset_s"), "clock offset")
        rtt = finite(raw.get("rtt_s"), "clock RTT")
        sample = finite(raw.get("ios_sample_s"), "clock sample")
        if not 0 <= rtt <= 1 or sample > now-offset+0.1:
            raise ProtocolError("clock estimate bounds")
        if sample > self.last_sample and 0 <= now-(sample+offset) <= 10:
            self.samples.append((offset, rtt, sample))
            self.last_sample = sample

    def age(self, capture_s, now):
        recent = [s for s in self.samples if 0 <= now-(s[2]+s[0]) <= 10 and s[1] <= .1]
        if len(recent) < 3 or max(s[0] for s in recent)-min(s[0] for s in recent) > .02:
            return None, None
        best = min(recent, key=lambda s: s[1])
        age = now-(capture_s+best[0])
        # RTT/2 plus observed offset spread bounds our estimate, not hardware sensor latency.
        uncertainty = best[1]/2 + (max(s[0] for s in recent)-min(s[0] for s in recent))
        if age < -uncertainty-.005:
            return None, None
        return max(age, 0), uncertainty

class FrameGate:
    def __init__(self, max_age=.25):
        self.max_age = max_age
        self.highwater = {"sensor": -1, "preview": -1}
        self.last_capture = {"sensor": -math.inf, "preview": -math.inf}
        self.session_id = None
        self.clock = ClockAlignment()
        self.dropped = 0
        self.last_rejection = None

    def accept(self, header, channel, now=None):
        now = time.monotonic() if now is None else now
        self.last_rejection = None
        sid = header["session_id"]
        if self.session_id != sid:
            if channel == "preview" and self.session_id is not None:
                self.dropped += 1
                self.last_rejection='preview_session_mismatch'
                return False, None, None
            self.__init__(self.max_age)
            self.session_id = sid
        self.clock.update(header.get("clock_estimate"), now)
        age, uncertainty = self.clock.age(header["capture_time_s"], now)
        frame = header["frame_id"]
        capture = header["capture_time_s"]
        reason=('frame_order' if frame<=self.highwater[channel] else
                'capture_order' if capture<=self.last_capture[channel] else
                'phone_processing_age' if header['processing_end_s']-capture>self.max_age else
                'capture_age' if age is not None and age>self.max_age+uncertainty else None)
        if reason:
            self.dropped += 1
            self.last_rejection=reason
            return False, age, uncertainty
        self.highwater[channel] = frame
        self.last_capture[channel] = capture
        return True, age, uncertainty
