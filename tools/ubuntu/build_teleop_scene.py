"""Create a separate G2 scene; reuse installed official meshes, never edit them."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(source / 'robot.xml').getroot()
    original = ET.parse(source / 'g2-original.urdf').getroot()
    restored = []
    for joint in original.findall('joint'):
        name = joint.get('name')
        if not ('_body_joint' in name or '_head_joint' in name):
            continue
        body = robot.find(f'.//body[@name="{joint.find("child").get("link")}"]')
        limit = joint.find('limit')
        ET.SubElement(body, 'joint', name=name, type='hinge', axis=joint.find('axis').get('xyz'),
                      limited='true', range=f'{limit.get("lower")} {limit.get("upper")}',
                      damping='3', armature='0.15' if '_body_' in name else '0.02')
        effort = float(limit.get('effort'))
        ET.SubElement(robot.find('actuator'), 'position', name=name, joint=name,
                      kp='400' if '_body_' in name else '100', dampratio='1',
                      ctrllimited='true', ctrlrange=f'{limit.get("lower")} {limit.get("upper")}',
                      forcelimited='true', forcerange=f'{-effort} {effort}')
        restored.append(name)
    # Frame origins at kinematic shoulder/elbow/wrist centers, not fingertips.
    for side in ('l', 'r'):
        for label, link in [('shoulder',2),('elbow',4),('wrist',7)]:
            body=robot.find(f'.//body[@name="arm_{side}_link{link}"]')
            ET.SubElement(body,'site',name=f'{side}_{label}',size='.008',rgba='0 .8 .8 .5')
    ET.SubElement(robot.find('.//body[@name="body_link5"]'),'site',name='torso_frame',pos='0 0 .3085',size='.012')
    ET.SubElement(robot.find('.//body[@name="head_link3"]'),'site',name='head_frame',size='.012')
    for i in range(1,6):
        ET.SubElement(robot.find('equality'), 'joint', name=f'fixed_torso_{i}',
                      joint1=f'idx0{i}_body_joint{i}', polycoef='0 0 0 0 0',
                      solref='0.004 1', solimp='0.9999 0.9999 0.001')
    robot.set('model','AgiBot G2 | HandDepth simulation teleoperation')
    # Existing balanced inertias are retained and explicitly recorded in provenance.
    ET.indent(robot);ET.ElementTree(robot).write(output/'robot.xml',encoding='unicode')
    scene=ET.Element('mujoco',model='HandDepth G2 teleoperation')
    ET.SubElement(scene,'include',file=str((output/'robot.xml').resolve()))
    vis=ET.SubElement(scene,'visual');ET.SubElement(vis,'global',offwidth='1280',offheight='960')
    ET.SubElement(vis,'headlight',diffuse='.7 .7 .7',ambient='.3 .3 .3')
    world=ET.SubElement(scene,'worldbody')
    ET.SubElement(world,'light',pos='0 -2 3',dir='0 .5 -1',directional='true')
    ET.SubElement(world,'geom',name='floor',type='plane',size='4 4 .1',rgba='.17 .21 .26 1')
    ET.indent(scene);ET.ElementTree(scene).write(output/'scene.xml',encoding='unicode')
    m=mujoco.MjModel.from_xml_path(str(output/'scene.xml'));d=mujoco.MjData(m)
    # Reachable relaxed pose: elbows below shoulders, forearms forward.
    relaxed={'l':[1.07712,-1.23077,-1.23290,-1.74608,-.11308,-.57435,.10076],
             'r':[-1.07849,-1.23576,1.23770,-1.74579,.20732,-.57441,-.11105]}
    for side,prefix in [('l',20),('r',60)]:
        for i,v in enumerate(relaxed[side],1):d.qpos[m.joint(f'idx{prefix+i}_arm_{side}_joint{i}').qposadr[0]]=v
    # Pitch zero is almost at one limit; center it so a neutral gaze can nod both ways.
    pitch=m.joint('idx13_head_joint3')
    d.qpos[pitch.qposadr[0]]=np.mean(pitch.range)
    coupling=json.loads((source/'provenance.json').read_text())['mimic_constraints']
    for side,idx in [('l',31),('r',71)]:d.qpos[m.joint(f'idx{idx}_gripper_{side}_inner_joint1').qposadr[0]]=-.05
    for name,c in coupling.items():d.qpos[m.joint(name).qposadr[0]]=c['offset']+c['multiplier']*d.qpos[m.joint(c['master']).qposadr[0]]
    for i in range(m.nu):d.ctrl[i]=d.qpos[m.jnt_qposadr[m.actuator_trnid[i,0]]]
    key=ET.SubElement(scene,'keyframe');ET.SubElement(key,'key',name='home',qpos=' '.join(map(str,d.qpos)),ctrl=' '.join(map(str,d.ctrl)))
    ET.indent(scene);ET.ElementTree(scene).write(output/'scene.xml',encoding='unicode')
    provenance={'source_assets':str(source),'restored_joints':restored,'base':'fixed',
        'simulation_only':True,'actuators':m.nu,'mujoco':mujoco.__version__,
        'inertias':'Preserves existing balanced inertias, including isotropic repairs on body_link4 and head_link1; local simulation adaptation, not validated robot dynamics.',
        'gravity_compensation':True, 'torso':'fixed zero configuration with joint equality locks; frozen in IK',
        'wrist_sites':'link7 origins include all seven arm joints, including final axial wrist rotation',
        'environment':'Empty floor; original tabletop/pick-place scenes untouched.'}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.source,a.output)
