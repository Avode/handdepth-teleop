# Attribution and release boundaries

Original HandDepth Ubuntu source and documentation are released under [MIT](LICENSE).
This project was developed with AI coding assistance (Codex); framework and model
work below remains credited to its authors.

| Component | Role | Upstream |
| --- | --- | --- |
| MuJoCo | Physics, model import and visualization | https://github.com/google-deepmind/mujoco |
| Mink | Differential inverse kinematics and task framework | https://github.com/kevinzakka/mink |
| AgiBot Genie G2 descriptions | Robot geometry, kinematics and mesh source | https://github.com/AgibotTech/genie_sim_robot_model |
| Genie Sim SRDF | Adjacent-link collision exclusions used by the local converter | https://github.com/AgibotTech/genie_sim |
| MediaPipe Pose Landmarker | Independent Ubuntu RGB pose observations | https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker |
| Apple AVFoundation / Vision | TrueDepth capture and phone body/hand observations | https://developer.apple.com/documentation/avfoundation |
| OpenCV, NumPy, SciPy, websockets, trimesh, xacro, pycollada | Image, geometry, transport and asset tools | See dependency manifests |

Upstream robot assets, framework source, model weights, Apple SDKs and the Swift
app are **not bundled** or relicensed by this repository. Fetch dependencies from
their upstream sources and retain their applicable licenses and notices.
The inspected AgiBot robot-model repository declares Apache-2.0 at its root;
its other robot/vendor assets may have different terms. The converter targets
only `genie/g2/g2_crsB_swiftpicker`.

The local adaptation uses fixed-base MuJoCo simulation, position servos, gravity
compensation and documented inertia repair. It is not the official Genie Sim
engine and its dynamics have not been validated on physical G2 hardware.
Brand names do not imply affiliation or endorsement.

The selected demo pictures are the author's supplied project screenshots.
They are portfolio illustrations, not a training dataset or motion-capture ground
truth. Raw RGB-D sessions and personal device/calibration logs are excluded.
