"""Single-reader MAVLink connection for an ArduRover boat."""
import time
import serial
from pymavlink.dialects.v20 import ardupilotmega as mav

class BoatLink:
    def __init__(self, port, baud=115200):
        self.port = serial.Serial(port, baud, timeout=0.05, write_timeout=1)
        self.mav = mav.MAVLink(self.port, srcSystem=255, srcComponent=190)
        self.mav.robust_parsing = True
        self.target = None
        self.heartbeat = None
        self.last_heartbeat = 0.0
        self.next_heartbeat = 0.0
        self.latest = {}
        self.messages = []

    def poll(self):
        now = time.monotonic()
        if now >= self.next_heartbeat:
            self.mav.heartbeat_send(6, 8, 0, 0, 4, force_mavlink1=True)
            self.next_heartbeat = now + 1
        result = []
        for message in self.mav.parse_buffer(self.port.read(self.port.in_waiting or 1)) or []:
            source = (message.get_srcSystem(), message.get_srcComponent())
            kind = message.get_type()
            if self.target is None and kind == "HEARTBEAT":
                if message.type == 11 and message.autopilot == 3 and source[1] == 1:
                    self.target = source
            if self.target is None or source != self.target:
                continue
            self.latest[kind] = message
            if kind == "HEARTBEAT":
                self.heartbeat = message
                self.last_heartbeat = time.monotonic()
            if kind == "STATUSTEXT":
                self.messages.append(message.text)
                print("Vehicle:", message.text, flush=True)
            result.append(message)
        return result

    def wait(self, predicate, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for message in self.poll():
                if predicate(message):
                    return message
        return None

    def connect(self):
        if self.wait(lambda p: p.get_type() == "HEARTBEAT", 8) is None:
            raise RuntimeError("No ArduRover boat heartbeat. Check port, baud and radio.")
        print("Connected:", self.target, "mode:", self.heartbeat.custom_mode,
              "armed:", bool(self.heartbeat.base_mode & 128))

    def require_fresh(self):
        if time.monotonic() - self.last_heartbeat > 3:
            raise RuntimeError("Boat heartbeat lost for more than 3 seconds.")

    def command(self, command, *parameters):
        values = list(parameters) + [0] * (7 - len(parameters))
        self.mav.command_long_send(*self.target, command, 0, *values, force_mavlink1=True)

    def parameter(self, name):
        for attempt in range(3):
            self.mav.param_request_read_send(*self.target, name.encode(), -1, force_mavlink1=True)
            answer = self.wait(lambda p: p.get_type() == "PARAM_VALUE" and p.param_id == name, 2)
            if answer is not None:
                return answer.param_value
        raise RuntimeError("No parameter response: " + name)

    def set_mode(self, mode):
        self.mav.set_mode_send(self.target[0], 1, mode, force_mavlink1=True)
        answer = self.wait(lambda p: p.get_type() == "HEARTBEAT" and p.custom_mode == mode, 4)
        if answer is None:
            raise RuntimeError("Mode change was not confirmed.")

    def stop(self):
        # A mode command can be lost. The caller must verify the resulting state.
        if self.target:
            self.mav.set_mode_send(self.target[0], 1, 4, force_mavlink1=True)

    def close(self):
        self.port.close()
