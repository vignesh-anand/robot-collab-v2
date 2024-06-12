from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet


from dm_control import mujoco as dm_mujoco
from dm_control.mujoco.wrapper.core import MjvOption
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control import mjcf


class BaseTask():
    def __init__(self, model_name:str = "", project_root_dir = ""):
        self.model = mjcf.RootElement(model=model_name)

        self.project_root_dir = project_root_dir
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
    Adds bodies, geoms, sites, joints, actuators using MJCF
    """
    def build_scene(self, ):
        pass
    
    """
    Defines robot models and adds them to the scene 
    """
    def add_robots(self,):
        pass
    
    """
    Gets the pose of the specified robot
    """
    def get_robot_pose(self, robot_name):
        pass