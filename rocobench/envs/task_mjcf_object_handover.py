from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet
import numpy as np
import PIL

from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control import mjcf
from rocobench.envs.robot import SimRobot

from typing import Any, Dict
from rocobench.envs.base_mjcf_task import BaseTask
from rocobench.envs.constants import *
from copy import deepcopy


class TaskObjectHandover(BaseTask):
    def __init__(
            self, 
            model_name : str = "task", 
            project_root_dir : str = "", 
            filepath : str = None, 
            robots = None, 
            reset_to_home_pose = False, 
            create_weld=False
    ):
        
        super().__init__(model_name, project_root_dir, filepath) 
        
        #Set default classes
        self.set_dclasses()

        #Build the scene
        self.build_scene()
        
        #Store Robot Configs
        self.robot_configs = {"panda": PANDA_CONSTANTS.copy(), "ur5e": UR5E_ROBOTIQ_CONSTANTS.copy(), 
                              "kinova": KINOVA_CONSTANTS.copy()}
    
        # #Add Robots
        self.add_robots(robot_dict=robots, create_weld=create_weld)

        self.home_pose()

        # #If add cameras:
        self.add_cameras()

        self.set_physics()

        # self.create_SimRobots(robots= robots)
        # print("Sim Robot Works!")

        # Reset to home pose for now
        if reset_to_home_pose:
            self.reset_to_home_pose()

    def build_scene(self,):
        self.light_wood_texture=self.model.asset.add('texture',file= self.project_root_dir + 'assets/objects/textures/light-wood.png',type="2d",name='tex-light-wood')
        self.light_wood=self.model.asset.add('material',name='light-wood',reflectance=0.2,texrepeat="15 15",texture='tex-light-wood',texuniform='true')
        checker = self.model.asset.add('texture', type='2d', builtin='checker', width=300, height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
        groundplane = self.model.asset.add('material', name='grid', texture=checker, texrepeat=[2, 2], reflectance=.2)

        self.model.worldbody.add('light',pos="0 0 1.5",dir="0 0 -1",directional="true")
        self.model.worldbody.add('geom',name="floor",pos="0 0 -0.5",size="0 0 0.05", type="plane",material=groundplane)

        self.table_body=self.model.worldbody.add('body',name='table',pos='0 0.5 0')

        self.table_body.add('geom',name='table_visual',pos='0 0 0.1',size='1.6 0.4 0.05', type='box', group='1', friction='1 0.5 0.0001',rgba='1 1 1 1')
        self.table_body.add('geom',name='table_collision',pos='0 0 0.1',size='1.6 0.4 0.05', type='box', group='3', friction='1 0.5 0.0001',rgba='1 1 1 1')

        self.table_top_body = self.table_body.add('body',name='table_top',pos='0 0 0.1')
        self.table_top_body.add('geom',name='table_top',size='1.6 0.4 0.05',type='box',conaffinity='0',contype='0', group='1',rgba='1 1 1 1')
        self.table_top_body.add('geom',name='table_top_collision',size='1.6 0.4 0.05',type='box',conaffinity='0',contype='0', group='3',rgba='1 1 1 1')

        self.table_body.add('geom',name='table_left',pos='-1.63 0 1', size='0.02 1.6 1.5', rgba='1 1 1 0', type='box')
        self.table_body.add('geom',name='table_right',pos='1.63 0 1', size='0.02 1.6 1.5', rgba='1 1 1 0', type='box')
        self.table_body.add('geom',name='table_front',pos='0 1.63 1', size='1.7 0.02 1.5', rgba='1 1 1 0', type='box')
        self.table_body.add('geom',name='table_back',pos='0 -1.63 1', size='1.7 0.02 1.5', rgba='1 1 1 0', type='box')

        #Collision Meshes
        self.table_body.add('geom',name='table_left_c',pos='-1.63 0 1', size='0.02 1.6 1.5', rgba='1 1 1 0', type='box', group = 3)
        self.table_body.add('geom',name='table_right_c',pos='1.63 0 1', size='0.02 1.6 1.5', rgba='1 1 1 0', type='box', group = 3)
        self.table_body.add('geom',name='table_front_c',pos='0 1.63 1', size='1.7 0.02 1.5', rgba='1 1 1 0', type='box', group = 3)
        self.table_body.add('geom',name='table_back_c',pos='0 -1.63 1', size='1.7 0.02 1.5', rgba='1 1 1 0', type='box', group = 3)

        self.bin_body1 = self.table_top_body.add('body',name='bin1',pos='-0.5 0 0.05')
        
        self.bin_body1.add('geom',name='bottom1',type='box',size='0.35 0.2 0.02',friction="1 0.005 0.0001",material='light-wood')
        self.bin_body1.add('geom',name='front1',type='box',pos='0 0.2 0.1',size='0.35 0.02 0.06',friction="1 0.005 0.0001",material='light-wood')
        self.bin_body1.add('geom',name='back1',type='box',pos='0 -0.2 0.1',size='0.35 0.02 0.06',friction="1 0.005 0.0001",material='light-wood')
        self.bin_body1.add('geom',name='right1',type='box',pos='0.35 0 0.1',size='0.02 0.2 0.06',friction="1 0.005 0.0001",material='light-wood')
        self.bin_body1.add('geom',name='left1',type='box',pos='-0.35 0 0.1',size='0.02 0.2 0.06',friction="1 0.005 0.0001",material='light-wood')

        #Collision Meshes
        self.bin_body1.add('geom',name='bottom1_c',type='box',size='0.35 0.2 0.02',friction="1 0.005 0.0001",material='light-wood', group = 3)
        self.bin_body1.add('geom',name='front1_c',type='box',pos='0 0.2 0.1',size='0.35 0.02 0.06',friction="1 0.005 0.0001",material='light-wood', group = 3)
        self.bin_body1.add('geom',name='back1_c',type='box',pos='0 -0.2 0.1',size='0.35 0.02 0.06',friction="1 0.005 0.0001",material='light-wood', group = 3)
        self.bin_body1.add('geom',name='right1_c',type='box',pos='0.35 0 0.1',size='0.02 0.2 0.06',friction="1 0.005 0.0001",material='light-wood', group = 3)
        self.bin_body1.add('geom',name='left1_c',type='box',pos='-0.35 0 0.1',size='0.02 0.2 0.06',friction="1 0.005 0.0001",material='light-wood', group = 3)

        self.box = self.model.worldbody.add('body',name='black_box',pos='-0.5 0.5 0.1')
        self.box.add('freejoint', name='box_free')
        self.box.add('body',name='box_top',pos='0 0 0.03')
        self.box.add('body',name='box_bottom',pos='0 0 -0.03')
        self.box.add('site',name='box_top',type='sphere',pos='0 0 0.03')
        self.box.add('site',name='box_bottom',type='sphere',pos='0 0 -0.03')
        self.box.add('geom',name='box',type='box',size='0.03 0.03 0.03',rgba='0 0 0 1')
        self.box.add('geom',name='box_collision',type='box',size='0.03 0.03 0.03',rgba='0 0 0 1', group = 3)

        self.bin_body2 = self.table_top_body.add('body',name='bin2',pos='0.5 0 0.05')
        
        self.bin_body2.add('geom',name='bottom2',type='box',size='0.35 0.2 0.02',friction="1 0.005 0.0001",material='light-wood')
        self.bin_body2.add('geom',name='front2',type='box',pos='0 0.2 0.1',size='0.35 0.02 0.06',friction="1 0.005 0.0001",material='light-wood')
        self.bin_body2.add('geom',name='back2',type='box',pos='0 -0.2 0.1',size='0.35 0.02 0.06',friction="1 0.005 0.0001",material='light-wood')
        self.bin_body2.add('geom',name='right2',type='box',pos='0.35 0 0.1',size='0.02 0.2 0.06',friction="1 0.005 0.0001",material='light-wood')
        self.bin_body2.add('geom',name='left2',type='box',pos='-0.35 0 0.1',size='0.02 0.2 0.06',friction="1 0.005 0.0001",material='light-wood')

        #Collision Meshes
        self.bin_body2.add('geom',name='bottom2_c',type='box',size='0.35 0.2 0.02',friction="1 0.005 0.0001",material='light-wood')
        self.bin_body2.add('geom',name='front2_c',type='box',pos='0 0.2 0.1',size='0.35 0.02 0.06',friction="1 0.005 0.0001",material='light-wood', group = 3)
        self.bin_body2.add('geom',name='back2_c',type='box',pos='0 -0.2 0.1',size='0.35 0.02 0.06',friction="1 0.005 0.0001",material='light-wood', group = 3)
        self.bin_body2.add('geom',name='right2_c',type='box',pos='0.35 0 0.1',size='0.02 0.2 0.06',friction="1 0.005 0.0001",material='light-wood', group = 3)
        self.bin_body2.add('geom',name='left2_c',type='box',pos='-0.35 0 0.1',size='0.02 0.2 0.06',friction="1 0.005 0.0001",material='light-wood', group = 3)

        self.bin_body = self.model.worldbody.add('body', name="bin", pos="0.45 0.5 0.16")
        self.bin_inside_body = self.bin_body.add('body', name="bin_inside", pos="0 0 0")
        
        self.bin_inside_body.add('geom', name="bin_inside_bottom", pos="0 0 0", size="0.35 0.2 0.02", type="box", friction="1 0.005 0.0001", material="light-wood")
        self.bin_inside_body.add('geom', name="bin_inside_front", pos="0 0.2 0.03", size="0.35 0.01 0.06", type="box", friction="1 0.005 0.0001", material="light-wood")
        self.bin_inside_body.add('geom', name="bin_inside_back", pos="0 -0.2 0.03", size="0.35 0.01 0.06", type="box", friction="1 0.005 0.0001", material="light-wood")
        self.bin_inside_body.add('geom', name="bin_inside_right", pos="0.38 0 0.03", size="0.01 0.18 0.06", type="box", friction="1 0.005 0.0001", rgba="1 0 0 0")
        self.bin_inside_body.add('geom', name="bin_inside_left", pos="-0.38 0 0.03", size="0.01 0.18 0.06", type="box", friction="1 0.005 0.0001", rgba="1 0 0 0" )

        #Collision Meshes
        self.bin_inside_body.add('geom', name="bin_inside_bottom_c", pos="0 0 0", size="0.35 0.2 0.02", type="box", group="3", friction="1 0.005 0.0001", material="light-wood")
        self.bin_inside_body.add('geom', name="bin_inside_front_c", pos="0 0.2 0.03", size="0.35 0.01 0.06", type="box", group="3", friction="1 0.005 0.0001", material="light-wood")
        self.bin_inside_body.add('geom', name="bin_inside_back_c", pos="0 -0.2 0.03", size="0.35 0.01 0.06", type="box", group="3", friction="1 0.005 0.0001", material="light-wood")
        self.bin_inside_body.add('geom', name="bin_inside_right_c", pos="0.38 0 0.03", size="0.01 0.18 0.06", type="box", group="3", friction="1 0.005 0.0001", rgba="1 0 0 0")
        self.bin_inside_body.add('geom', name="bin_inside_left_c", pos="-0.38 0 0.03", size="0.01 0.18 0.06", type="box", group="3", friction="1 0.005 0.0001", rgba="1 0 0 0" )

        self.bin_body.add('geom', name="bin_right", pos="0.35 0 0.03", size="0.01 0.2 0.06", type="box", friction="1 0.005 0.0001", material="light-wood", margin="0.01")
        self.bin_body.add('geom', name="bin_left", pos="-0.35 0 0.03", size="0.01 0.2 0.06", type="box", friction="1 0.005 0.0001", material="light-wood", margin="0.01")

        #Collision Meshes
        self.bin_body.add('geom', name="bin_right_c", pos="0.35 0 0.03", size="0.01 0.2 0.06", type="box", group="3", friction="1 0.005 0.0001", material="light-wood", margin="0.01")
        self.bin_body.add('geom', name="bin_left_c", pos="-0.35 0 0.03", size="0.01 0.2 0.06", type="box", group="3", friction="1 0.005 0.0001", material="light-wood", margin="0.01")


    def home_pose(self,):
        panda_qpos0='0 0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04'
        ur5_qpos0='0 -1.5708 -1.5708 1.5708 -1.5708 -1.5708 0 0 0 0 0 0 0 0 0'
        box_qpos0='-0.5 0.5 0.22 0 0 0 0'
        panda_ctrl0='0 0 0 0 -1.57079 0 1.57079 -0.7853 255'
        ur5_ctrl0='0 1.5708 -1.5708 1.5708 -1.5708 -1.5708 0 0'
        self.model.keyframe.add('key',name='home',
                                qpos=box_qpos0 + ' ' + panda_qpos0 + ' ' + ur5_qpos0, 
                                ctrl=panda_ctrl0+' '+ur5_ctrl0)

    def add_cameras(self):
        self.model.worldbody.add('camera', mode="fixed", name='face_panda1', pos="0.062 -2.806 0.768", xyaxes="1.000 0.009 -0.000 0.001 -0.131 0.991")
        self.model.worldbody.add('camera', mode="fixed", name='face_panda2', pos="0.084 3.711 0.772", xyaxes="-1.000 0.016 0.000 0.002 0.111 0.994")
        self.model.worldbody.add('camera', mode="fixed", name='top_cam', pos="-0.001 0.652 2.057", xyaxes="-1.000 -0.000 -0.000 0.000 -1.000 0.019")
        self.model.worldbody.add('camera', mode="fixed", name='right_cam', pos="1.873 0.605 0.926", xyaxes="0.014 1.000 0.000 -0.242 0.003 0.970")
        self.model.worldbody.add('camera', mode="fixed", name='left_cam', pos="-1.873 0.605 0.926", xyaxes="-0.000 -1.000 0.000 0.177 -0.000 0.984")
        self.model.worldbody.add('camera', name='teaser', pos="2.675 -0.747 1.997", xyaxes="0.496 0.869 -0.000 -0.429 0.245 0.869")
        self.model.worldbody.add('camera', name='video', pos="1.957 -0.281 1.731", xyaxes="0.487 0.874 0.000 -0.500 0.278 0.820")

    def render_image(
            self, 
            cam_id : int = 0, 
            qpos = None, 
            qpos_id = None, 
            width=1080, 
            height=800
    ):
        if qpos is None:
            # self.physics.reset()
            img = PIL.Image.fromarray(self.physics.render(camera_id=cam_id, width = width, height = height))
        else:
            with self.physics.reset_context():
                self.physics.data.qpos[qpos_id] = qpos
                self.physics.step()

                img = PIL.Image.fromarray(self.physics.render(camera_id=cam_id, width = width, height = height))
        return img
    
    def reset_to_home_pose(self,):
        self.physics.reset(0)