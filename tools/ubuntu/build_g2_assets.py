"""Build a local MuJoCo tabletop adaptation of AgiBot's official G2 assets."""

import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
import xacro




DATA = Path(os.environ["HANDDEPTH_DATA_ROOT"]).expanduser().resolve()
SOURCE = DATA / "assets/genie_sim_robot_model"
OUTPUT = DATA / "assets/g2-mujoco"
PROJECT = DATA / "projects/tabletop"


def write_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")


def build():
    from handdepth.recording import check_recording_root
    check_recording_root(DATA)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PROJECT.mkdir(parents=True, exist_ok=True)
    staged = OUTPUT / "xacro"
    for source in (SOURCE / "robots/genie/g2").rglob("*.xacro"):
        destination = staged / source.relative_to(SOURCE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text().replace("$(find genie_sim_robot_model)", str(staged))
        text = text.replace("package://genie_sim_robot_model", str(SOURCE))
        destination.write_text(text.replace('collision_type="coarse"', 'collision_type="fine"'))
    doc = xacro.process_file(str(staged / "robots/genie/g2/g2_crsB_swiftpicker.urdf.xacro"))
    (OUTPUT / "g2-original.urdf").write_text(doc.toprettyxml())
    urdf = ET.fromstring(doc.toxml())
    joint_sources = {j.attrib["name"]: j for j in urdf.findall("joint")}
    frozen = []
    for joint in joint_sources.values():
        if joint.get("type") != "fixed" and not any(s in joint.get("name") for s in ("_arm_", "_gripper_")):
            frozen.append(joint.get("name"))
            joint.set("type", "fixed")
    mesh_paths = {}
    for link in urdf.findall("link"):
        for kind in ("visual", "collision"):
            for i, element in enumerate(link.findall(kind)):
                element.set("name", f'{link.get("name")}_{kind}_{i}')
                mesh = element.find("geometry/mesh")
                if mesh is None:
                    continue
                source = Path(mesh.get("filename"))
                relative = source.relative_to(SOURCE)
                unique_name = "__".join(relative.with_suffix("").parts) + "_" + relative.suffix[1:].lower() + ".obj"
                destination = OUTPUT / "meshes" / unique_name
                if str(source) not in mesh_paths:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        loaded = trimesh.load(source, force="scene")
                        geometry = loaded.to_geometry()
                        geometry.export(destination, file_type="obj", include_color=False, include_texture=False)
                    mesh_paths[str(source)] = str(destination)
                mesh.set("filename", str(destination))
                if kind == "visual" and element.find("material") is None:
                    name = link.get("name")
                    dark = any(s in name for s in ("gripper", "chassis", "wheel", "head_link3"))
                    material = ET.SubElement(element, "material", name=f"local_{name}")
                    ET.SubElement(material, "color", rgba="0.12 0.15 0.18 1" if dark else "0.78 0.82 0.86 1")
    extension = ET.SubElement(urdf, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="false", fusestatic="false", strippath="false", balanceinertia="true")
    write_xml(urdf, OUTPUT / "g2-mujoco.urdf")
    model = mujoco.MjModel.from_xml_path(str(OUTPUT / "g2-mujoco.urdf"))
    mujoco.mj_saveLastXML(str(OUTPUT / "g2-imported.xml"), model)
    robot = ET.parse(OUTPUT / "g2-imported.xml").getroot()
    robot.set("model", "AgiBot Genie G2 | MuJoCo tabletop adaptation")
    ET.SubElement(robot, "option", timestep="0.002", integrator="implicitfast", iterations="80", cone="elliptic")
    ET.SubElement(robot, "size", memory="32M")
    for body in robot.findall(".//body"):
        body.set("gravcomp", "1")
    for geom in robot.findall(".//geom"):
        if geom.get("contype") != "0":
            geom.set("group", "3")
            geom.set("friction", "0.8 0.005 0.0001")
            geom.set("condim", "4")
    equality = ET.SubElement(robot, "equality")
    actuator = ET.SubElement(robot, "actuator")
    mimic = {}
    for joint in robot.findall(".//body/joint"):
        name = joint.get("name")
        source = joint_sources[name]
        joint.set("damping", "0.15" if "gripper" in name else "1")
        joint.set("armature", "0.002" if "gripper" in name else "0.05")
        coupling = source.find("mimic")
        if coupling is not None:
            master = coupling.get("joint")
            multiplier = float(coupling.get("multiplier", "1"))
            offset = float(coupling.get("offset", "0"))
            mimic[name] = {"master": master, "multiplier": multiplier, "offset": offset}
            ET.SubElement(equality, "joint", name=f"couple_{name}", joint1=name, joint2=master,
                          polycoef=f"{offset} {multiplier} 0 0 0", solref="0.004 1")
        else:
            limit = source.find("limit")
            effort = float(limit.get("effort"))
            ET.SubElement(actuator, "position", name=name, joint=name,
                          kp="120" if "gripper" in name else "250", dampratio="1",
                          ctrllimited="true", ctrlrange=f'{limit.get("lower")} {limit.get("upper")}',
                          forcelimited="true", forcerange=f"{-effort} {effort}")
    # The gripper linkage has adjacent, interleaved support meshes. Preserve
    # collisions with objects while excluding collision within each mechanism.
    contact = ET.SubElement(robot, "contact")
    srdf = Path(os.environ["HANDDEPTH_SRDF"]).expanduser().resolve()
    body_names = {b.get("name") for b in robot.findall(".//body")}
    adjacent_pairs = []
    for pair in ET.parse(srdf).getroot().iter("disable_collisions"):
        first, second = pair.get("link1"), pair.get("link2")
        if pair.get("reason") == "Adjacent" and first in body_names and second in body_names:
            ET.SubElement(contact, "exclude", body1=first, body2=second)
            adjacent_pairs.append([first, second])
    for side in ("l", "r"):
        bodies = [b.get("name") for b in robot.findall(".//body") if b.get("name", "").startswith(f"gripper_{side}_")]
        for i, first in enumerate(bodies):
            for second in bodies[i + 1:]:
                ET.SubElement(contact, "exclude", body1=first, body2=second)
    for side in ("l", "r"):
        body = robot.find(f'.//body[@name="gripper_{side}_base_link"]')
        ET.SubElement(body, "site", name=f"{side}_grasp", pos="0 0 0.13", size="0.004", rgba="0.1 0.8 0.6 0.5")
    write_xml(robot, OUTPUT / "robot.xml")
    scene = ET.Element("mujoco", model="AgiBot Genie G2 - tabletop")
    ET.SubElement(scene, "include", file=str(OUTPUT / "robot.xml"))
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0.1 0.1 0.1")
    asset = ET.SubElement(scene, "asset")
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker", rgb1="0.16 0.19 0.24", rgb2="0.21 0.25 0.3", width="512", height="512")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="4 4", reflectance="0.1")
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", pos="1 -2 3", dir="-0.3 0.4 -1", diffuse="0.8 0.8 0.8", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="4 4 0.1", material="floor_mat")
    table = ET.SubElement(world, "body", name="table", pos="0.8 0 0.70")
    ET.SubElement(table, "geom", name="table_top", type="box", size="0.35 0.6 0.025", rgba="0.48 0.33 0.21 1", friction="0.8 0.005 0.0001")
    for x in (-0.28, 0.28):
        for y in (-0.53, 0.53):
            ET.SubElement(table, "geom", type="box", pos=f"{x} {y} -0.35", size="0.025 0.025 0.325", rgba="0.2 0.24 0.28 1")
    for name, pos, shape, size, mass, color in [
        ("can", "0.65 0.20 0.79", "cylinder", "0.028 0.06", "0.15", "0.12 0.55 0.8 1"),
        ("box", "0.85 0.32 0.77", "box", "0.035 0.035 0.04", "0.12", "0.88 0.4 0.12 1"),
        ("bottle", "0.68 -0.25 0.82", "cylinder", "0.025 0.09", "0.18", "0.22 0.65 0.45 1"),
    ]:
        body = ET.SubElement(world, "body", name=name, pos=pos)
        ET.SubElement(body, "freejoint", name=f"{name}_free")
        ET.SubElement(body, "geom", name=f"{name}_geom", type=shape, size=size, mass=mass, rgba=color, condim="4", friction="0.8 0.005 0.0001")
    tray = ET.SubElement(world, "body", name="tray", pos="0.93 -0.18 0.736")
    ET.SubElement(tray, "geom", type="box", size="0.13 0.13 0.01", rgba="0.5 0.6 0.68 1")
    for x, y, sx, sy in [(0.13, 0, 0.008, 0.13), (-0.13, 0, 0.008, 0.13), (0, 0.13, 0.13, 0.008), (0, -0.13, 0.13, 0.008)]:
        ET.SubElement(tray, "geom", type="box", pos=f"{x} {y} 0.025", size=f"{sx} {sy} 0.025", rgba="0.5 0.6 0.68 1")
    write_xml(scene, OUTPUT / "starter-scene.xml")
    model = mujoco.MjModel.from_xml_path(str(OUTPUT / "starter-scene.xml"))
    data = mujoco.MjData(model)
    for side, prefix in [("l", 20), ("r", 60)]:
        for index, angle in [(2, -45), (4, -75), (6, -50)]:
            name = f"idx{prefix + index}_arm_{side}_joint{index}"
            data.qpos[model.joint(name).qposadr[0]] = np.deg2rad(angle)
        master = f"idx{31 if side == 'l' else 71}_gripper_{side}_inner_joint1"
        data.qpos[model.joint(master).qposadr[0]] = -0.05
    for name, coupling in mimic.items():
        data.qpos[model.joint(name).qposadr[0]] = coupling["offset"] + coupling["multiplier"] * data.qpos[model.joint(coupling["master"]).qposadr[0]]
    for i in range(model.nu):
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
    keyframe = ET.SubElement(scene, "keyframe")
    ET.SubElement(keyframe, "key", name="home", qpos=" ".join(map(str, data.qpos)), ctrl=" ".join(map(str, data.ctrl)))
    write_xml(scene, OUTPUT / "starter-scene.xml")
    if not (PROJECT / "scene.xml").exists():
        write_xml(scene, PROJECT / "scene.xml")
    provenance = {
        "source": "https://github.com/AgibotTech/genie_sim_robot_model",
        "commit": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
        "variant": "g2_crsB_swiftpicker", "simulator": f"MuJoCo {mujoco.__version__}",
        "frozen_joints": frozen, "base": "fixed", "robot_gravity_compensation": True,
        "inertia_repair": "MuJoCo balanceinertia enabled because upstream body_link4 and head_link1 violate the principal-inertia triangle inequality. Both links are fixed in this adaptation.",
        "mimic_constraints": mimic, "mesh_conversion": "DAE/STL to OBJ using trimesh; simplified visual colors",
        "adjacent_collision_exclusions": adjacent_pairs,
        "limitations": "Local tabletop adaptation, not the official Genie Sim engine. No learned policy or sim-to-real validation. Position servo gains are local settings; torso, head, and wheels fixed.",
        "mesh_files": len(mesh_paths), "actuators": model.nu,
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"scene": str(PROJECT / "scene.xml"), "actuators": model.nu, "meshes": model.nmesh, "mimic_constraints": len(mimic)}, indent=2))


if __name__ == "__main__":
    build()
