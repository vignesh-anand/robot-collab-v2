import numpy as np
import torch
from copy import deepcopy
from transforms3d import quaternions
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet

from rocobench.envs.env_utils import Pose
from rocobench.envs.base_env import MujocoSimEnv
from rocobench.envs.world_config import WorldConfig

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from curobo.geom.types import WorldConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    )


class SimRobot:
    """ Stores the info for a single arm, doesn't store or change physics state """
    def __init__(
        self,
        physics: Any, # use only for gathering more arm infos
        name: str,
<<<<<<< HEAD
        robot_constants: dict,
        curobo_path: str,
        urdf_path: str,
        yaml_path: str,
        mjcf_model, 
=======
        all_joint_names: List[str],
        ik_joint_names: List[str],
        arm_joint_names: List[str],
        actuator_info: Dict[str, Any],
        all_link_names: List[str],
        arm_link_names: List[str], # 
        ee_link_names: List[str],
        base_joint: str,
        ee_site_name: str,
        grasp_actuator: str,
        mesh_to_geoms: Dict[str, Any],
        weld_body_name: str = "rhand", # or gripper
        ee_rest_quat: np.ndarray = np.array([0, 1, 0, 0]),
        use_ee_rest_quat: bool = False,
>>>>>>> aditya
    ):
        
        pass
        self.name=name
        self.constants=self.prepend_robot_name(name,robot_constants)
        self.curobo_path=curobo_path
        self.urdf_path=urdf_path
        self.yaml_path=yaml_path
        self.ik_joint_names = self.constants['ik_joint_names']
        self.ee_site_name = self.constants['ee_site_name']
        self.ee_link_names = self.constants['ee_link_names']
        self.ee_rest_quat = self.constants['ee_rest_quat']
        self.arm_link_names = self.constants['arm_link_names']
        self.use_ee_rest_quat = self.constants['use_ee_rest_quat']
        self.grasp_actuator = self.constants['grasp_actuator']
        self.grasp_idx_in_ctrl = physics.named.data.ctrl._convert_key(self.grasp_actuator)

        self.actuator_info = self.constants['actuator_info']
        self.weld_body_name = self.constants['weld_body_name']
        self.curobo_robot_config=RobotConfig.from_dict(
        load_yaml(yaml_path)["robot_cfg"]
        )
        self.collision_world=WorldConfig(mjcf_model,physics,skip_robot_name=name)
        self.joint_ranges = []
        self.joint_idxs_in_qpos = [] 
        self.joint_idxs_in_ctrl = []
        self.mjcf_model=mjcf_model
        for _name in self.ik_joint_names:
            qpos_slice = physics.named.data.qpos._convert_key(_name)
            assert int(qpos_slice.stop - qpos_slice.start) == 1, "Only support single joint for now"
            idx_in_qpos = qpos_slice.start
            self.joint_idxs_in_qpos.append(idx_in_qpos)
            self.joint_ranges.append(physics.model.joint(_name).range)

            assert _name in self.actuator_info, f"Joint {_name} not in actuator_info"
            actuator_name = self.actuator_info[_name]
            idx_in_ctrl = physics.named.data.ctrl._convert_key(actuator_name)
            self.joint_idxs_in_ctrl.append(idx_in_ctrl)
        
        
        self.ee_link_body_ids = []
        for _name in self.ee_link_names:
            try:
                link = physics.model.body(_name)
            except Exception as e:
                print(f'link name: {_name} does NOT have a body in env.physics.model.body')
                raise e
            self.ee_link_body_ids.append(link.id)
        
        self.all_link_body_ids = []
        for _name in self.all_link_names:
            try:
                link = physics.model.body(_name)
            except Exception as e:
                print(f'link name: {_name} does NOT have a body in env.physics.model.body')
                raise e
            self.all_link_body_ids.append(link.id)
        
        self.ee_link_pairs = set()
        for _id1 in self.ee_link_body_ids:
            for _id2 in self.ee_link_body_ids:
                if _id1 != _id2:
                    self.ee_link_pairs.add(
                        frozenset([_id1, _id2])
                    )
        self.collision_link_names = self.arm_link_names + self.ee_link_names
        self.collision_link_ids = [
            physics.model.body(_name).id for _name in self.collision_link_names
        ]
        self.home_qpos = physics.data.qpos[self.joint_idxs_in_qpos].copy()
    def prepend_robot_name(self,name:str,constants: dict):
        result = dict()
        result["name"] = self.name
        for key, value in self.constants.items():
            #print(key)
            if key=='name':
                continue

            if key == "actuator_info":
                result[key] = {self.name + "/" + x: self.name + "/" + y for x, y in value.items()}
            elif key == "mesh_to_geoms":
                result[key] = {x: [self.name + "/" + y for y in z] for x, z in value.items()}

            elif isinstance(self.constants[key],str):
                result[key]= self.name+'/'+value
            else:
                result[key] = [self.name + "/" + x for x in value]

        return result
    def set_home_qpos(self, env: MujocoSimEnv):
        env_cp = deepcopy(env)
        env_cp.reset()
        self.home_qpos = env_cp.physics.data.qpos[self.joint_idxs_in_qpos].copy()        
    
    def get_home_qpos(self) -> np.ndarray:
        return self.home_qpos.copy()

    def get_ee_pose(
        self,
        physics: Any,
    ) -> Pose:
        """ Get the pose of the end effector """
        ee_site = physics.data.site(self.ee_site_name)
        ee_pos = ee_site.xpos.copy()
        ee_quat = quaternions.mat2quat(
                    physics.named.data.site_xmat[self.ee_site_name].copy()
                )
        if self.use_ee_rest_quat:
            ee_quat = quaternions.qmult(ee_quat, self.ee_rest_quat)
        return Pose(position=ee_pos, orientation=ee_quat)
 
    def map_qpos_to_joint_ctrl(self, qpos: np.ndarray) -> Dict[str, np.ndarray]:
        """ Map the full qpos to the joint ctrl """
        assert len(qpos) > len(self.joint_idxs_in_qpos), f"qpos: {qpos} should be full state"
        desired_joint_qpos = qpos[self.joint_idxs_in_qpos]
        return {
            'ctrl_idxs': self.joint_idxs_in_ctrl,
            'ctrl_vals': desired_joint_qpos,
        }

    @property
    def grasp_idx(self) -> int:
        """ Get the grasp idx of the end effector """
        return self.grasp_idx_in_ctrl

    def get_grasp_ctrl_val(self, grasp: bool):
        # Hard code for now
        if self.grasp_actuator == 'adhere_gripper':
            grasp_ctrl_val = 0.0 if grasp else 0.0 # disabled!
        elif self.grasp_actuator == "panda_gripper_actuator":
            grasp_ctrl_val = 0 if grasp else 255 
        elif self.grasp_actuator == "adhere_hand":
            grasp_ctrl_val = 0 if grasp else 0
        elif self.grasp_actuator == "robotiq_fingers_actuator":
            grasp_ctrl_val = 255 if grasp else 0
        else:
            raise NotImplementedError(f"Grasp actuator {self.grasp_actuator} not implemented")
        return grasp_ctrl_val
        
    def solve_ik(
        self, 
        physics, 
        target_pos, 
        target_quat = None,
        number_seeds=20,
        position_threshhold=1e-3,
        roation_threshhold=5e-2,
        check_self_collision=True,
        check_world_collision=False,
        collision_world=None,
        use_primitive_collisions=True
        ):
        """ solves single arm IK, helpful to check if a pose is achievable """
        ## Update world config
        target_quat = np.array([1, 0, 0, 0]) if target_quat is None else target_quat 
        tensor_args = TensorDeviceType()
        if collision_world is None and check_world_collision:
            collision_world.update_curobo_world(physics)
            if use_primitive_collisions:
                collision_world=self.collision_world.get_as_obb()
            else:
                collision_world=self.collision_world.get_as_class()
        elif not check_world_collision:
            collision_world=None
        ik_config = IKSolverConfig.load_from_robot_config(
            self.curobo_robot_config,
            collision_world,
            rotation_threshold=roation_threshhold,
            position_threshold=position_threshhold,
            num_seeds=number_seeds,
            self_collision_check=check_self_collision,
            self_collision_opt=check_self_collision,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        ik_solver = IKSolver(ik_config)
        goal_pose=Pose(position=target_pos,quaternion=target_quat[[3,0,1,2]])
        ik_result=ik_solver.solve_single(goal_pose=goal_pose)
        ik_result.get_unique_solution()
        return ik_result if ik_result.succes else None 
    
