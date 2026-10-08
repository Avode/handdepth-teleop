"""Render an existing simulation status snapshot without changing the live process.

Set MUJOCO_GL=egl for offscreen use. No historical pose is presented as fresh.
"""
import argparse
import json
from pathlib import Path
import cv2
import mujoco
from handdepth.recording import check_recording_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = check_recording_root(args.output)
    status = json.loads(args.status.read_text())
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    if len(status['qpos']) != model.nq:
        raise ValueError('status and model coordinate sizes differ')
    data.qpos[:] = status['qpos']
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.05, 0, .95]
    camera.distance = 2.9
    camera.elevation = -12
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        for azimuth, name in [(135, 'runtime-three-quarter.png'), (180, 'runtime-front.png')]:
            camera.azimuth = azimuth
            renderer.update_scene(data, camera)
            cv2.imwrite(str(output / name), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    report = {key: status.get(key) for key in ('state', 'reason', 'source_frame_id',
              'measurement_age_s', 'active_parts', 'torso_mode', 'base_displacement_m',
              'ik_failures', 'physics_warnings')}
    report['capture_kind'] = 'Offscreen render of running simulation state; no human overlay; no live-process changes'
    (output / 'runtime-capture.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
