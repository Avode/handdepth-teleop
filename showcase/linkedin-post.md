# LinkedIn post draft

I started with a goal: teach a robot to pick up and place everyday objects. My first milestone became building a way to demonstrate motion to it.

This is HandDepth: my iPhone-to-humanoid teleoperation prototype, connecting front TrueDepth RGB/depth to Ubuntu and an AgiBot G2 model in MuJoCo, using Mink for inverse kinematics.

The most useful lessons came from things that went wrong:

• Matching the hands did not guarantee that the arms matched. I moved to shoulder and forearm directions relative to the human torso, while keeping the robot's torso and base fixed.

• An elbow disappearing behind a hand should not erase the whole skeleton. I added coherent arm reconstruction using retained metric lengths and the previous bend plane. Inferred geometry stays labelled and separate from control measurements.

• A frozen pose was not always a disconnected phone. RGB and depth sometimes arrived with different source frame IDs. Bounded fallback now keeps current phone observations flowing while preserving strict matching for Ubuntu fusion.

The prototype also includes forearm-relative wrist control and a fist-close/open-hand-release gripper gesture. Missing or stale measurements hold the affected commands.

I’m releasing the Ubuntu source under MIT, with tests, setup instructions and a draft engineering journal. The local regression suite has 268 passing tests with the G2 model. The Swift capture app is still a separate component.

This remains a simulation prototype. Tracking quality varies with visibility and latency; autonomous pick-and-place and physical robot validation are next steps, not completed results.

Built with Codex assistance, on the work of the MuJoCo, Mink, AgiBot, MediaPipe and Apple Vision communities.

Code and journey: https://github.com/Avode/handdepth-teleop

I’m looking for robotics engineering or research opportunities focused on perception, manipulation and human–robot interaction. I’d welcome feedback from people building and testing these systems.

#Robotics #MuJoCo #ComputerVision #Teleoperation #OpenSource
