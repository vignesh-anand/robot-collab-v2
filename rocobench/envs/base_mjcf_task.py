from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet
import numpy as np

from dm_control import mujoco as dm_mujoco
from dm_control.mujoco.wrapper.core import MjvOption
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control import mjcf
import os


class BaseTask():
    def __init__(
            self, 
            model_name:str = "", 
            project_root_dir:str  = "", 
            filepath:str = None
    ):
        self.model = mjcf.RootElement(model=model_name) if filepath is None else mjcf.from_file(filepath)
        
        if not project_root_dir:
            parent_dir = os.path.abspath(os.path.join(os.path.dirname("__file__")))

            # Ensure parent_dir ends with a separator
            if not parent_dir.endswith(os.path.sep):
                parent_dir += os.path.sep
            
            self.project_root_dir = parent_dir
        else:
            self.project_root_dir = project_root_dir


        # self.project_root_dir = os.getcwd() if not project_root_dir else project_root_dir
        ## Set the gloabl parameters
        self.global_params()

    """ 
    Set the global parameters for the MuJoCo Model
    """
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

    """
    Set the default classes for the MuJoCo Model
    """
    def set_dclasses(self,):
        pass
    
    """
    Renders the scene through MuJoCo Renderer or NVISII
    """
    def render(self, enable_nvisii = False):
        pass

    """
    Sets the Physics for the Model using dm control
    """
    def set_physics(self,):
        self.physics = mjcf.Physics.from_mjcf_model(self.model)
    
    """
    Defines robot models and adds them to the scene 
    """
    def add_robots(
            self,
            robot_dict: Dict[str, Dict[str, Any]] = {}, 
            create_weld=False
    ):
        #Generalize for multiple robots/arms
        self.robot_models = {}

        assert robot_dict != {}, "Robot Dictionary is empty. Please provide a dictionary of robot models to add to the scene."

        for robot_name, robot_data in robot_dict.items():   
            
            #Create robot model to attach to mujoco world model
            robot = mjcf.from_path(robot_data["file"])
            robot.model = robot_name
            robot_site = self.model.worldbody.add('site', name=robot_name, pos=robot_data["worldbody_site_pos"], size=[0.001, 0.001, 0.001])
            robot_site.attach(robot)
            self.robot_models[robot_name] = deepcopy(robot)
            del robot, robot_site
        
        if create_weld:
            pass
    
    """
    Gets the pose of the specified robot
    """
    def get_robot_pose(self, robot_name):
            return np.concatenate((self.physics.named.data.xpos[robot_name],self.physics.named.data.xquat[robot_name]),axis=0)