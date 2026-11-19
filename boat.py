"""Status, finite motor test and dead-man keyboard control."""
import argparse
import csv
import datetime
import math
import time
from pathlib import Path
from serial.tools import list_ports
from USV_Serial import BoatLink, mav


def status(link, seconds):
    names = ["RC_PROTOCOLS", "FS_THR_ENABLE", "FS_GCS_ENABLE", "FS_GCS_TIMEOUT",
             "FS_ACTION", "ARMING_REQUIRE", "SERVO1_FUNCTION", "SERVO3_FUNCTION"]
    for name in names:
        try:
            print(name, link.parameter(name))
        except RuntimeError as error:
            print(error)
    link.command(401)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for message in link.poll():
            if message.get_type() in ("GPS_RAW_INT", "EKF_STATUS_REPORT", "COMMAND_ACK"):
                print(message)


def motor_test(link, args):
    link.require_fresh()
    if link.heartbeat.custom_mode != 4:
        raise RuntimeError("Select Hold before the motor test.")
    if link.heartbeat.base_mode & 128:
        raise RuntimeError("Disarm before the motor test.")
    if not 0 < args.percent <= 10 or not 0 < args.duration <= 8:
        raise RuntimeError("Bench test limits: 0 < throttle <= 10%, 0 < duration <= 8 seconds.")
    number = {"left": 3, "right": 4}[args.motor]
    link.command(209, number, 0, args.percent, args.duration)
    print("Sent once:", args.motor, args.percent, "%", args.duration, "seconds.")
    ack = None
    changed = False
    returned = False
    start = time.monotonic()
    while time.monotonic() - start < args.duration + 6:
        for message in link.poll():
            kind = message.get_type()
            if kind == "COMMAND_ACK" and message.command == 209:
                ack = message.result
                print("Motor-test ACK:", ack, "(0 = accepted)")
            if kind == "SERVO_OUTPUT_RAW" and message.port == 0:
                print("PWM left/right:", message.servo1_raw, message.servo3_raw)
                if message.servo1_raw != 1500 or message.servo3_raw != 1500:
                    changed = True
                if time.monotonic() - start > args.duration and message.servo1_raw == 1500 and message.servo3_raw == 1500:
                    returned = True
    print("ACK:", ack, "output changed:", changed, "neutral observed after timeout:", returned)
    print("Confirm physical motion and stopping on site. Missing ACK does not prove non-execution.")


def keyboard_control(link, args):
    import msvcrt
    if not 0 < args.percent <= 10:
        raise RuntimeError("Keyboard bench throttle must be between 0 and 10%.")
    # Parameter reads never alter the vehicle configuration.
    if link.parameter("FS_GCS_ENABLE") != 1 or link.parameter("FS_ACTION") != 2:
        raise RuntimeError("Configure GCS failsafe enabled with Hold action before keyboard control.")
    throttle = int(link.parameter("RCMAP_THROTTLE"))
    steering = int(link.parameter("RCMAP_ROLL"))
    if throttle == steering or not 1 <= throttle <= 8 or not 1 <= steering <= 8:
        raise RuntimeError("Unsupported RC channel mapping.")
    channels = {}
    for channel in (throttle, steering):
        channels[channel] = [link.parameter(f"RC{channel}_MIN"),
                             link.parameter(f"RC{channel}_TRIM"),
                             link.parameter(f"RC{channel}_MAX")]
    link.set_mode(4)
    link.command(400, 1)
    armed = link.wait(lambda p: p.get_type() == "HEARTBEAT" and bool(p.base_mode & 128), 5)
    if armed is None:
        raise RuntimeError("Normal arming failed. Read vehicle messages.")
    try:
        link.set_mode(0)
        print("Hold arrow keys or W/S/A/D. Space stops. Esc exits. Release returns neutral within 0.3s.")
        deadline = time.monotonic() + args.duration
        last_key = float("-inf")
        key = ""
        due = 0.0
        while time.monotonic() < deadline:
            link.poll()
            link.require_fresh()
            if link.heartbeat.custom_mode != 0:
                raise RuntimeError("Manual mode lost. Control stopped.")
            while msvcrt.kbhit():
                value = msvcrt.getwch().lower()
                if value in ("\x00", "\xe0"):
                    value = {"H": "w", "P": "s", "K": "a", "M": "d"}.get(msvcrt.getwch(), "")
                if value == "\x1b":
                    return
                key = value
                last_key = time.monotonic()
            if time.monotonic() >= due:
                values = [65535] * 8
                for channel, (minimum, trim, maximum) in channels.items():
                    values[channel - 1] = int(trim)
                if time.monotonic() - last_key < 0.3:
                    channel = throttle if key in ("w", "s") else steering
                    direction = {"w": 1, "s": -1, "a": -1, "d": 1}.get(key, 0)
                    minimum, trim, maximum = channels[channel]
                    span = maximum - trim if direction > 0 else trim - minimum
                    values[channel - 1] = int(trim + direction * span * args.percent / 100)
                link.mav.rc_channels_override_send(*link.target, *values, force_mavlink1=True)
                due = time.monotonic() + 0.1
    finally:
        # Try neutral first, then Hold, then release overrides. All can fail on a broken link.
        for attempt in range(3):
            values = [65535] * 8
            for channel, (_, trim, _) in channels.items():
                values[channel - 1] = int(trim)
            link.mav.rc_channels_override_send(*link.target, *values, force_mavlink1=True)
            link.stop()
            time.sleep(0.1)
        link.mav.rc_channels_override_send(*link.target, *([0] * 8), force_mavlink1=True)
        stopped = link.wait(lambda p: p.get_type() == "HEARTBEAT" and p.custom_mode == 4, 4)
        print("Hold confirmed:", stopped is not None)
        link.command(400, 0)
        disarmed = link.wait(lambda p: p.get_type() == "HEARTBEAT" and not p.base_mode & 128, 4)
        print("Disarm confirmed:", disarmed is not None)


def record(link, args):
    folder = Path(__file__).parent / "data"
    folder.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = folder / f"telemetry_{stamp}.csv"
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["monotonic_s", "message", "fields"])
        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            for message in link.poll():
                if message.get_type() in ("ATTITUDE", "GLOBAL_POSITION_INT", "GPS_RAW_INT", "SERVO_OUTPUT_RAW"):
                    writer.writerow([time.monotonic(), message.get_type(), str(message.to_dict())])
    print("Recorded:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["ports", "status", "motor-test", "keyboard", "record"])
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--percent", type=float, default=5)
    parser.add_argument("--motor", choices=["left", "right"], default="left")
    args = parser.parse_args()
    if args.action == "ports":
        for port in list_ports.comports():
            print(port.device, port.description)
        return
    if not args.port:
        parser.error("--port is required. Use ports to list devices.")
    if args.duration is None:
        args.duration = 4 if args.action == "motor-test" else 30
    if not math.isfinite(args.duration) or not 0 < args.duration <= 600 or not math.isfinite(args.percent):
        parser.error("Use finite values; duration must be 0..600 seconds.")
    link = BoatLink(args.port, args.baud)
    try:
        link.connect()
        if args.action == "status":
            status(link, args.duration)
        elif args.action == "motor-test":
            motor_test(link, args)
        elif args.action == "keyboard":
            keyboard_control(link, args)
        else:
            record(link, args)
    finally:
        link.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error))
