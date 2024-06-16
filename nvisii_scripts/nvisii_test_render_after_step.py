import os
os.environ['MUJOCO_GL'] = 'egl' #'glfw'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# The basic mujoco wrapper.
from dm_control import mujoco as dm_mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.core import MjvOption
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf


import numpy as np
from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet


# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import IPython.display as display
import PIL.Image

## nvisii class
import nvisii_renderer
import nvisii


## Import and append path directories
import sys
import os

# Get the absolute path of the rocobench folder
rocobench_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'rocobench'))

# Add rocobench_path to sys.path
if rocobench_path not in sys.path:
    sys.path.append(rocobench_path)

sys.path.append('/home/adityadutt/Desktop/robot-collab-v2/')
sys.path.append('/home/adityadutt/Desktop/robot-collab-v2/rocobench/envs')

from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import UR5E_ROBOTIQ_CONSTANTS, PANDA_CONSTANTS, KINOVA_CONSTANTS

#Handle Configs to work with Curobo, NVISII
def append_robot_names(name, constants):
    result = dict()
    result["name"] = name
    for key, value in constants.items():
        #print(key)
        if key=='name':
            continue

        if key == "actuator_info":
            result[key] = {name + "/" + x: name + "/" + y for x, y in value.items()}
        elif key == "mesh_to_geoms":
            result[key] = {x: [name + "/" + y for y in z] for x, z in value.items()}

        elif isinstance(constants[key],str):
            result[key]= name+'/'+value
        else:
            result[key] = [name + "/" + x for x in value]

    return result

class BaseTask():
    # model = mjcf.RootElement()
    # physics = None
    def __init__(self, model_name:str = "", project_root_dir = ""):
        self.model = mjcf.RootElement(model=model_name)

        self.project_root_dir = project_root_dir
        ## Set the gloabl parameters
        self.global_params()

    def global_params(self,):
        self.model.compiler.angle = 'radian'
        self.model.compiler.autolimits = 'true'

        #Visual options
        self.model.visual.headlight.diffuse = [0.6, 0.6, 0.6] #Can use a list format or a string format
        self.model.visual.headlight.ambient=[0.1, 0.1, 0.1]
        self.model.visual.headlight.specular=[0, 0, 0]
        self.model.visual.rgba.haze = "0.15 0.25 0.35 1"
        self.model.visual.__getattr__('global').azimuth = 120 #Only way to access global attribute since gloabl is a Python keyword
        self.model.visual.__getattr__('global').elevation = -20
        self.model.visual.__getattr__('global').offwidth = 2000
        self.model.visual.__getattr__('global').offheight = 2000
        self.model.visual.quality.shadowsize = 4096

    def set_dclasses(self,):
        pass

    def render(self,):
        pass

    def set_physics(self,):
        self.physics = mjcf.Physics.from_mjcf_model(self.model)
    
    def build_scene(self, ):
        pass

class DoorCabinet(BaseTask):
    def __init__(self, model_name, project_root_dir, filepath = None, robots = None, reset_to_home_pose = False, create_weld=False):
        
        super().__init__(model_name, project_root_dir) 

        #Set default classes
        self.set_dclasses()

        #Build the scene
        self.build_scene()
        
        #Store Robot Configs
        self.robot_configs = {"panda": PANDA_CONSTANTS.copy(), "ur5e": UR5E_ROBOTIQ_CONSTANTS.copy(), 
                              "kinova": KINOVA_CONSTANTS.copy()}
    
        #Add Robots
        self.add_robots(robot_dict=robots, create_weld=create_weld)

        print("Add Robots Works!")

        #If add cameras:
        self.add_cameras()

        self.physics = mjcf.Physics.from_mjcf_model(self.model)

        self.create_SimRobots(robots= robots)

        print("Sim Robot Works!")

        ## Compute Door Pose for Open-Door
        self.door_name = "cabinet/right_door_handle"
        # doorcabinet.physics.named.data.xpos["cabinet/cabinet"] += [0., 0., 0.51]
        self.cabinet_pos = self.physics.data.body("cabinet/cabinet").xpos.copy()
        self.open_door_pose = self.compute_door_open_pose(self.door_name) #[3, 4]

        ## Reset to home pose for now
        if reset_to_home_pose:
            self.reset_to_home_pose()
        
    def set_dclasses(self):
        #Default class values
        self.obj_visual_class = self.model.default.add('default', dclass="object_visual")
        self.obj_site_class = self.model.default.add('default', dclass='object_sites')
        self.obj_collision_class = self.model.default.add('default', dclass='object_collision')
        # obj_hingecabinet = self.model.default.add('default', dclass="hingecabinet")

        self.obj_visual_class.geom.type = 'mesh'
        self.obj_visual_class.geom.conaffinity = '0'
        self.obj_visual_class.geom.contype = '0'
        self.obj_visual_class.geom.group = '1'
        self.obj_visual_class.geom.mass = '0.00001'

        #add('site', type='cylinder', size='0.003 0.006', group=3)
        self.obj_site_class.site.type = 'cylinder'
        self.obj_site_class.site.size = '0.003 0.006'
        self.obj_site_class.site.group = 3           

        # <geom density="500" rgba="0.8 0.8 0.8 0.9" group="3"/>
        self.obj_collision_class.geom.density = '500'
        self.obj_collision_class.geom.rgba = '0.8 0.8 0.8 0.9'
        self.obj_collision_class.geom.group = 3

    def build_scene(self):
        #Floor and initial light
        self.model.worldbody.add('light', pos="0 0 1.5", dir="0 0 -1", directional="true")

        self.groundplane_texture = self.model.asset.add('texture', type="2d", name="groundplane", builtin="checker", mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
        self.groundplane_mat = self.model.asset.add('material', name="groundplane", texture="groundplane", texuniform="true", texrepeat="2 2", reflectance="0.2")
        self.groundplane = self.model.worldbody.add('geom', name="floor", pos="0 0 -0.5", size="0 0 0.05", type="plane", material="groundplane")

        #Adding the table box
        self.table_box = self.model.worldbody.add('body', name="table", pos="0 0.5 0")
        self.table_box.add('geom', name="table_collision", pos="0 0 0.1", size="1.6 .5 0.05", type="box", group=3) #friction="1 0.005 0.0001")
        # self.table_box.add('geom', material="white-wood", name="table-mat")

        self.table_top = self.table_box.add('body', name="table_top", pos="0 0 0.11")
        self.table_top.add('geom', name="table_top", size="1.6 0.4 0.05", type="box", conaffinity="0", contype="0", group="1")
        # table_top.add('site', name="table_top", size="0.001 0.001 0.001", class_="site_top")

        self.table_box.add('geom', name="table_left", pos="-1.63 0 1", size="0.02 1.6 1.5", rgba="1 1 1 0", type="box")
        self.table_box.add('geom', name="table_right", pos="1.63 0 1", size="0.02 1.6 1.5", rgba="1 1 1 0", type="box")
        self.table_box.add('geom', name="table_front", pos="0 1.63 1", size="1.7 0.02 1.5", rgba="1 1 1 0", type="box")
        self.table_box.add('geom', name="table_back", pos="0 -1.63 1", size="1.7 0.02 1.5", rgba="1 1 1 0", type="box")

        #Adding a key card on the table
        self.key_card = self.model.worldbody.add('body', name="keycard", pos="0.5 0.6 0.1")

        #Can also do add_free_joint()
        self.key_card.add('freejoint', name="keycard_joint") #, type="free", name="keycard_joint", pos="0 0 0", axis="0 0 1")
        self.key_card.add('geom', name='keycard', type="box", rgba="0. 0. 0. 1", size="0.05 0.1 0.01")

        #Add the cabinet
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the texture file
        # texture_file_path = os.path.join(script_dir, 'envs', 'assets', 'objects', 'textures', 'white-wood.png')
            
        self.cabinet_path = self.project_root_dir + "rocobench/envs/cabinet_mjcf.xml"
        self.cabinet =  mjcf.from_file(self.cabinet_path)
        self.cabinet.model = "cabinet"
        self.model.attach(self.cabinet)

        #Add the camera
        self.camera = self.model.worldbody.add('camera', name='my_camera', pos="2.1 -0.15 1.731", xyaxes="0.487 0.874 0.000 -0.500 0.278 0.820")

    def add_cameras(self):
        self.model.worldbody.add('camera', mode="fixed", name='face_panda1', pos="0.062 -2.806 0.768", xyaxes="1.000 0.009 -0.000 0.001 -0.131 0.991")
        self.model.worldbody.add('camera', mode="fixed", name='face_panda2', pos="0.084 3.711 0.772", xyaxes="-1.000 0.016 0.000 0.002 0.111 0.994")
        self.model.worldbody.add('camera', mode="fixed", name='top_cam', pos="-0.001 0.652 2.057", xyaxes="-1.000 -0.000 -0.000 0.000 -1.000 0.019")
        self.model.worldbody.add('camera', mode="fixed", name='right_cam', pos="1.873 0.605 0.926", xyaxes="0.014 1.000 0.000 -0.242 0.003 0.970")
        self.model.worldbody.add('camera', mode="fixed", name='left_cam', pos="-1.873 0.605 0.926", xyaxes="-0.000 -1.000 0.000 0.177 -0.000 0.984")
        self.model.worldbody.add('camera', name='teaser', pos="2.675 -0.747 1.997", xyaxes="0.496 0.869 -0.000 -0.429 0.245 0.869")
        self.model.worldbody.add('camera', name='video', pos="1.957 -0.281 1.731", xyaxes="0.487 0.874 0.000 -0.500 0.278 0.820")

    def render_scene(self, frames, framerate=30):
        # Simulate and display video.
        print("Not yet implemented!")
        # display_video(frames, framerate)

    def render_image(self, cam_id:int, qpos = None, qpos_id = None, width=1080, height=800):
        if qpos is None:
            # self.physics.reset()
            img = PIL.Image.fromarray(self.physics.render(camera_id=cam_id, width = width, height = height))
        else:
            with self.physics.reset_context():
                self.physics.data.qpos[qpos_id] = qpos
                self.physics.step()

                img = PIL.Image.fromarray(self.physics.render(camera_id=cam_id, width = width, height = height))
        return img

    def add_robots(self, robot_dict, create_weld = False):
        #Generalize for multiple robots/arms
        self.robot_models = {}

        for robot_name, robot_data in robot_dict.items():   
            
            #Create robot model to attach to mujoco world model
            robot = mjcf.from_path(robot_data["file"])
            robot.model = robot_name
            robot_site = self.model.worldbody.add('site', name=robot_name, pos=robot_data["worldbody_site_pos"], size=[0.001, 0.001, 0.001])
            robot_site.attach(robot)
            self.robot_models[robot_name] = deepcopy(robot)
            del robot, robot_site
        
        # self.panda_arm1 = Panda("panda1", arm_pos=[0.6, 1.02, 0.1], xml_file='rocobench/envs/assets/panda/panda.xml')
        # self.panda_arm2 = Panda("panda2", arm_pos=[0.6, -0.05, 0.1], xml_file='rocobench/envs/assets/panda/panda.xml')
        
        # self.panda_arm1 = mjcf.from_path('rocobench/envs/assets/panda/panda.xml')
        # self.panda_arm1.model = "panda1"
        # self.ur5e = mjcf.from_path('rocobench/envs/assets/ur5e_robotiq/ur5e_robotiq_full.xml')
        # self.ur5e.model = "ur5e"
        
        # ur5e_site = self.model.worldbody.add('site', name='ur5e', pos=[0.2, 1.2, 0.1], size=[0.001, 0.001, 0.001])
        # ur5e_site.attach(self.ur5e)

        # self.panda_arm2 = mjcf.from_path('rocobench/envs/assets/panda/panda.xml')
        # self.panda_arm2.model = "panda2"    
        
        # # panda_arm1_site = self.model.worldbody.add('site', name='panda1', pos=[0.2, 1.15, 0.1], size=[0.001, 0.001, 0.001])
        # # panda_arm1_site.attach(self.panda_arm1)

        # panda_arm2_site = self.model.worldbody.add('site', name='panda2', pos=[0.4, -.15, 0.1], size=[0.001, 0.001, 0.001])
        # panda_arm2_site.attach(self.panda_arm2)

        ## Add Kinova Arm 
        # self.kinovagen3_arm = mjcf.from_path("rocobench/envs/assets/kinova_with_base/base_with_kinova_gripper.xml")
        # self.kinovagen3_arm.model = "kinovagen3"
        # kinovagen3_site = self.model.worldbody.add('site', name="kinovagen3", pos= [1.3, -0.15, -0.5], size=[0.001, 0.001, 0.001])
        # kinovagen3_site.attach(self.kinovagen3_arm)

        #Add weld connections -> Add Adhesion instead
        if create_weld:
            self.weld_true = True
            self.model.equality.add("weld", name="door_handle_panda", body1="cabinet/right_door_handle", body2="ur5e/robotiq_tip", relpose=[0,0, 0, 1, 0, 0, 0], active=False)
            self.model.equality.add("weld", name="keycard_panda", body1="keycard", body2="panda/panda_palm", relpose=[0,0, 0, 1, 0, 0, 0], active=False)
        else:
            self.weld_true = False
    
    def create_SimRobots(self, robots = None):
        #Create new config based on which robot it is
        #Dictionary storing the SimRobot instances of all robots
        self.sim_robots = {}
        self.joint_ids = {}

        for robot_name, robot_data in robots.items():  
            # new_config = append_robot_names(robot_name, self.robot_configs[robot_data["robot_type"]])

            #Create SimRobot instances
            self.sim_robots[robot_name] = SimRobot(physics=self.physics, **robot_data["robot_config"])

            self.joint_ids[robot_name] = {
                                        "qpos":self.sim_robots[robot_name].joint_idxs_in_qpos, 
                                          "ctrl":self.sim_robots[robot_name].joint_idxs_in_ctrl
                                          }

    def get_robot_pose(self, robot_name):
        return np.concatenate((self.physics.named.data.xpos[robot_name],self.physics.named.data.xquat[robot_name]),axis=0)

    def compute_door_open_pose(self, door_name: str = "cabinet/right_door_handle"):
        physics = self.physics.copy(share_model=True)
        # if door_name == "left_door_handle":p
        #     qpos_slice = self.physics.named.data.qpos._convert_key("leftdoorhinge")
        #     if self.cabinet_pos[0] > 0:
        #         physics.data.qpos[qpos_slice.start] = -2.2
        #     else:
        #         physics.data.qpos[qpos_slice.start] = -2.6
        # elif door_name == "right_door_handle":
        qpos_slice = self.physics.named.data.qpos._convert_key("cabinet/rightdoorhinge")
        if self.cabinet_pos[0] > 0:
            physics.data.qpos[qpos_slice.start] = 1.8    
        else:
            physics.data.qpos[qpos_slice.start] = 1.
        # else:
        #     raise NotImplementedError
        physics.forward()
        desired_handle_pose = np.concatenate(
            [physics.data.body(door_name).xpos, physics.data.body(door_name).xquat]
        ) 
        # img = physics.render(camera_id="teaser")
        # plt.imshow(img)
        # plt.show()
        del physics 
        return desired_handle_pose   

    def reset_to_home_pose(self, joint_ids = None, render=False):

        if joint_ids is None:
            joint_ids = self.joint_ids.copy()

        # self.physics.reset()
        self.physics.named.data.ctrl["ur5e/robotiq_fingers_actuator"] = 255
        self.physics.named.data.ctrl["panda/panda_gripper_actuator"] = 255
        # with doorcabinet.physics.reset_context():
        if self.weld_true:
            self.physics.named.model.eq_active["keycard_panda"] = False
            self.physics.named.model.eq_active["door_handle_panda"] = False
        
        self.physics.named.data.qpos["keycard_joint"] = [0.5, 0.5, 0.17, 1., 0., 0., 0.]
        
            # qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04"
            # physics.data.qpos[joint_ids["panda1"]["qpos"] + joint_ids["panda2"]["qpos"]] = [0., 0., 0., 0., -1.57079, 0, 1.57079, -0.7853]*2 
        if "ur5e" in joint_ids.keys():
            self.physics.data.qpos[joint_ids["ur5e"]["qpos"]] = [0., -1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.] # (8, 9, 10, 11, 12, 13, 14)
        if "panda" in joint_ids.keys():
            self.physics.data.qpos[joint_ids["panda"]["qpos"]] = [0., 0., -1.3, 0., -2.5, 0, 1., 0., ]
        if "kinova" in joint_ids.keys():
            self.physics.data.qpos[joint_ids["kinova"]["qpos"]] = [0.0411018, 1.57, -0.31415, 0.06283, -1.19377, -0.06283, -1.44509, 1.67465]
        ##Kinova home pos
        # physics.data.qpos[np.arange(33, 41)] = [0.0411018, 1.57, -0.31415, 0.06283, -1.19377, -0.06283, -1.44509, 1.67465]

        # self.physics.data.ctrl = 0.
        # self.physics.data.qvel = 0.
        self.physics.step()

        if render:
            img = PIL.Image.fromarray(self.physics.render(camera_id=6, width=1080, height=800))
            return img
        
        return None
    
    def nvisii_render(self, width=800, height=800, output_file = None):
        # self.physics.step()
        r = nvisii_renderer.NVISIIRenderer(env=self, width=width, height=height)
        if output_file is None:
            r.render(render_type="png")
        else:
            r.render_to_file(output_file)

        r.close()


def main():
    ### Load Cabinet Model ###
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # print("Script dir:", script_dir)

    # Get the absolute path of the parent directory (one level up from nvisii_scripts)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(parent_dir)

    # Ensure parent_dir ends with a separator
    if not parent_dir.endswith(os.path.sep):
        parent_dir += os.path.sep
    print(parent_dir)
    # Construct the full path to panda.xml
    panda_xml_path = os.path.join(parent_dir, 'rocobench', 'envs', 'assets', 'panda', 'panda.xml')
    ur5e_xml_path = os.path.join(parent_dir, 'rocobench', 'envs', 'assets', 'ur5e_robotiq', 'ur5e_robotiq_full.xml')

    robots = { 
                "panda": {
                    "robot_config": append_robot_names("panda", PANDA_CONSTANTS.copy()),
                    "robot_type": "panda",
                    "file": panda_xml_path,
                    "worldbody_site_pos": [0.4, -0.2, 0.2]
                
                },
                "ur5e":{
                    "robot_config": append_robot_names("ur5e", UR5E_ROBOTIQ_CONSTANTS.copy()),
                    "robot_type": "ur5e",
                    "file": ur5e_xml_path,
                    "worldbody_site_pos": [0.2, 1.2, 0.1]
                }
            }

    doorcabinet = DoorCabinet(model_name="doorcabinet", project_root_dir = parent_dir + "/", robots=robots, create_weld=False)

    ### Use NVISII Renderer 
    doorcabinet.nvisii_render(width=800, height=800, output_file="images/test_physics_before_step.png")

    ## Render with MuJoCo
    # plt.imshow(doorcabinet.render_image(cam_id=0, width = 1080, height=800))
    # plt.show()

    ## Take some action ##
    ## Move the panda: 8 DOFs, ur5e: 7 DOFs ##
    doorcabinet.physics.data.qpos[doorcabinet.joint_ids["panda"]["qpos"]] = 0.5*np.ones(8)
    doorcabinet.physics.step()

    ## Render with MuJoCo
    # plt.imshow(doorcabinet.render_image(cam_id=0, width = 1080, height=800))
    # plt.show()

    ## Render the scene again: Don't need to specify output file, will default
    doorcabinet.nvisii_render(width=800, height=800, output_file="images/test_physics_after_step.png")
    
    ## Render the scene using Mujoco Physics Renderer for comparison
    # plt.imshow(doorcabinet.render_image(cam_id=0, width = 1080, height=800))
    # plt.show()


if __name__=="__main__":
    main()