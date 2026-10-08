import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys
import time
from .recording import DEFAULT_ROOT, Recorder, replay_packets
from .server import Receiver

async def replay(args, receiver):
    start = None
    for channel, recorded_s, raw in replay_packets(args.session):
        if start is None:
            start = recorded_s, time.monotonic()
        if args.speed > 0:
            delay = (recorded_s-start[0])/args.speed - (time.monotonic()-start[1])
            while delay > 0:
                await asyncio.sleep(min(delay, .033 if not args.headless else .1))
                receiver.expire()
                if not args.headless:
                    import cv2
                    cv2.imshow("HandDepth replay", receiver.dashboard.render())
                    if cv2.waitKey(1) & 255 in (27,ord('q')):
                        receiver.invalidate("replay_stopped")
                        cv2.destroyAllWindows()
                        return
                delay = (recorded_s-start[0])/args.speed - (time.monotonic()-start[1])
        receiver.process(channel, raw, recorded_s, replay=True)
        if not args.headless:
            import cv2
            cv2.imshow("HandDepth replay", receiver.dashboard.render())
            if cv2.waitKey(1) & 255 in (27,ord('q')): break
    if args.snapshot:
        import cv2
        if not cv2.imwrite(str(args.snapshot), receiver.dashboard.render()):
            raise ValueError("cannot write snapshot")
    receiver.invalidate("replay_complete")
    if not args.headless:
        import cv2
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="HandDepth local-network TrueDepth receiver")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--record", action="store_true")
    serve.add_argument("--recording-root", type=Path, default=DEFAULT_ROOT)
    serve.add_argument("--fixture-storage", action="store_true", help="explicit local test storage override; never use /mnt")
    serve.add_argument("--duration", type=float)
    play = sub.add_parser("replay")
    play.add_argument("session", type=Path)
    play.add_argument("--speed", type=float, default=1, help="0 = as fast as possible")
    play.add_argument("--snapshot", type=Path)
    tele = sub.add_parser("teleop", help="measured upper-body poses to G2 in MuJoCo (simulation only)")
    tele.add_argument("--host", default="0.0.0.0")
    tele.add_argument("--port", type=int, default=8765)
    tele.add_argument("--scene", type=Path, default=Path('/mnt/robotics-data/robotics/agibot-g2/projects/handdepth/scene.xml'))
    tele.add_argument("--recording-root", type=Path, default=DEFAULT_ROOT)
    tele.add_argument("--fixture-storage", action="store_true")
    tele.add_argument("--record", action="store_true")
    tele.add_argument("--duration", type=float)
    tele.add_argument("--auto-calibrate", action="store_true", help="acquire relaxed neutral per source session; preserve it through occlusion; Space pauses")
    tele.add_argument("--arm-mapping", choices=('limb','shoulder','cartesian'), default='limb',
                      help="limb: upper-arm and forearm directions; shoulder: upper arm only; cartesian: legacy wrist IK")
    tele.add_argument('--wrist-mapping',choices=('relative','hold'),default='relative',
                      help='limb mode: measured palm rotation relative to forearm, or hold wrist joints')
    tele.add_argument('--grip-mapping',choices=('fist','pinch'),default='fist',
                      help='fist: curled fingers close, open hand releases; pinch: legacy thumb-index distance')
    tele.add_argument("--replay", type=Path)
    tele.add_argument("--speed", type=float, default=1)
    tele.add_argument("--status-file", type=Path, help="bounded current simulation status JSON; use HDD path")
    tele.add_argument("--headless", action="store_true")
    tele.add_argument("--no-pc-pose", action="store_true",help="disable independent Ubuntu RGB pose inference")
    tele.add_argument("--pc-pose-model",type=Path,default=Path('/mnt/robotics-data/robotics/agibot-g2/models/handdepth/pose_landmarker_full.task'))
    for p in (serve, play):
        p.add_argument("--headless", action="store_true")
        p.add_argument("--pose-jsonl", type=Path, help="machine-readable poses; '-' writes to stdout")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    if args.command == "teleop":
        if not 1 <= args.port <= 65535 or args.speed <= 0 or args.duration is not None and args.duration <= 0:
            parser.error("invalid port/speed/duration")
        try:
            from .teleop import run
            run(args)
        except KeyboardInterrupt:
            pass
        except (ValueError, OSError, RuntimeError, ImportError) as e:
            parser.exit(2, "handdepth teleop: " + str(e) + "\n")
        return
    if args.command == "serve" and (not 1 <= args.port <= 65535 or args.duration is not None and args.duration <= 0):
        parser.error("invalid port/duration")
    if args.command == "replay" and args.speed < 0:
        parser.error("speed must be nonnegative")
    recorder = None
    pose_stream = None
    try:
        if args.pose_jsonl:
            pose_stream = sys.stdout if str(args.pose_jsonl) == "-" else args.pose_jsonl.open("w")
        if args.command == "serve" and args.record:
            recorder = Recorder(args.recording_root, args.fixture_storage)
            logging.info("recording %s", recorder.path)
        receiver = Receiver(recorder, pose_stream)
        if args.command == "serve":
            asyncio.run(receiver.run(args.host, args.port, args.headless, args.duration))
        else:
            asyncio.run(replay(args, receiver))
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError) as e:
        parser.exit(2, "handdepth: " + str(e) + "\n")
    finally:
        if recorder: recorder.close()
        if pose_stream and pose_stream is not sys.stdout: pose_stream.close()

if __name__ == "__main__":
    main()
