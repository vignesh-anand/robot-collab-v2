import logging
import numpy as np
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt
from transforms3d import euler, quaternions
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet

from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control.utils.transformations import mat_to_quat, quat_to_euler, euler_to_quat 

from rocobench.rrt import direct_path, smooth_path, birrt, NearJointsUniformSampler, CenterWaypointsUniformSampler
from rocobench.envs import SimRobot 
#from rocobench.envs.env_utils import Pose
from curobo.types.math import Pose
from curobo.util_file import (
    get_robot_configs_path,
    get_assets_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    )
from curobo.types.robot import RobotConfig
import urdfpy
import yaml
from urdfpy import URDF,Link,Joint
import os
import copy
from scipy.spatial.transform import Rotation
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig as c_world_config
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from rocobench.envs.world_config import WorldConfig
from curobo.types.base import TensorDeviceType

class MultiArmCurobo:
    """ Stores the info for a group of arms and plan all the combined joints together """
    def __init__(
        self,
        physics,
        mjcf_model=None,
        robots: Dict[str, SimRobot] = {},
        seed: int = 0,
    ):
        self.robots = robots
        self.physics = physics 
        self.np_random = np.random.RandomState(seed)
        
        self.all_joint_names = []
        self.all_joint_ranges = []
        self.all_joint_idxs_in_qpos = []
        self.all_collision_link_names = []
        self.inhand_object_info = dict()
        urdf_list=[]
        config_list=[]
        pose_list=[]
        self.names_list=[]

        for name, robot in self.robots.items():
            self.names_list.append(name)
            with open(robot.yaml_path, 'r') as f:
                config_list.append(yaml.load(f, Loader=yaml.SafeLoader))
            
            urdf_list.append(URDF.load(robot.urdf_path))
            pose_list.append(
                np.concatenate((    physics.named.data.xpos[name+"/"],
                                    physics.named.data.xquat[name+"/"]), 
                                    axis=0).tolist()
            )
            self.all_joint_names.extend(
                robot.ik_joint_names
            ) 
            self.all_joint_idxs_in_qpos.extend(
                robot.joint_idxs_in_qpos
            )
            self.all_joint_ranges.extend(
                robot.joint_ranges
            )
            self.all_collision_link_names.extend(
                robot.collision_link_names
            )

        self.primary_robot_name=self.names_list[0]
        #self.set_inhand_info(physics, inhand_object_info)
        combined_urdf=self.add_robots_urdf(urdf_list=urdf_list,pose_list=pose_list,name='_'.join(list(self.robots.keys())))
        combined_urdf.save(os.path.join(get_assets_path(),'robot/','_'.join(list(self.robots.keys()))))
        combined_yaml_dict=self.combine_yaml_kinematics(config_list=config_list, combined_urdf_path=os.path.join("robot",'_'.join(list(self.robots.keys()))))
        
        self.robot_config=RobotConfig.from_dict(combined_yaml_dict)
        
        if mjcf_model is not None:
            self.collision_world=WorldConfig(mjcf_model,physics,skip_robot_name=list(self.robots.keys()))
        
        self.joint_minmax = np.array([jrange for jrange in self.all_joint_ranges])
        self.joint_ranges = self.joint_minmax[:, 1] - self.joint_minmax[:, 0]

        # assign a list of allowed grasp ids to each robot
        tensor_args = TensorDeviceType()
        
    def pose_list_to_pose_matrix(self,pose):
        pose_matrix=np.eye(4)
        rot=Rotation.from_quat(np.array(pose)[[4,5,6,3]])
        
        pose_matrix[0:3,3]=pose[0:3]
        pose_matrix[0:3,0:3]=rot.as_matrix()
        return pose_matrix
    
    def clean_robot(self,urdf_object,append_string="_1"):
        for link in urdf_object.links:
            link.name=link.name+append_string
        for joint in urdf_object.joints:
            joint.name=joint.name+append_string
            joint.parent=joint.parent+append_string
            joint.child=joint.child+append_string
        return urdf_object
    
    def add_robots_urdf(self,urdf_list,pose_list,name):
        assert len(urdf_list)==len(pose_list),'inputs should be of same length'
        base_link=Link('base_fixture_link',None,visuals=[urdfpy.Visual(urdfpy.Geometry(sphere=urdfpy.Sphere(0.1)))],collisions=None)
        new_links=[base_link]
        new_joints=[]
        for i in range(len(urdf_list)):
            robot_pose=pose_list[i]
            urdf_list[i]=self.clean_robot(urdf_list[i],"_"+str(i+1))
            new_links+=urdf_list[i].links
            new_joints+=[Joint(name=base_link.name+"_j_"+urdf_list[i].base_link.name,joint_type='fixed',parent=base_link.name,child=urdf_list[i].base_link.name,origin=list(self.pose_list_to_pose_matrix(robot_pose)))]
            new_joints+=urdf_list[i].joints
        #for links in new_links:
            #print(link.name)
        #kinematics={}
        return URDF(name,links=new_links,joints=new_joints)
    
    def combine_yaml_kinematics(self,config_list,combined_urdf_path):
        kinematic_list=[]
        for config in config_list:
            kinematic_list.append(config['robot_cfg']['kinematics'])
        new_kinematics={'urdf_path':combined_urdf_path,'asset_root_path':'/robot','base_link':'base_fixture_link'}
        #new_lock_joints={}
        collison_sphere={} #
        ee_link=kinematic_list[0]['ee_link']+"_1"
        links_names=[]#
        lock_joints={}#
        extra_links={}#
        collison_sphere_buffer=0#
        extra_collison_spheres={}#
        self_collison_ignore={}#
        self_collison_buffer={}#
        mesh_link_names=[]#
        collision_link_names=[]#
        use_global_cumul=False#
        cspace={'joint_names':[],'retract_config':[],'null_space_weight':[],'cspace_distance_weight':[],'max_jerk':float('inf'),'max_acceleration':float('inf')}
        for i in range(len(kinematic_list)):
            if isinstance(kinematic_list[i]['collision_spheres'],str):
                with open(os.path.join(get_robot_configs_path(),'spheres'), 'r') as f:
                    spheres_i = yaml.load(f, Loader=yaml.SafeLoader)['collision_spheres']
                    
            else:
                spheres_i=kinematic_list[i]['collision_spheres']
            spheres_i={key+"_"+str(i+1):value for key,value in spheres_i.items()}
            collison_sphere.update(spheres_i)
            links_names.append(kinematic_list[i]['ee_link']+'_'+str(i+1))
            if kinematic_list[i]['lock_joints'] is not None:
                lock_joints.update({key+"_"+str(i+1):value for key,value in kinematic_list[i]['collision_spheres'].items()})
            if kinematic_list[i]['extra_links'] is not None:
                for k,v in kinematic_list[i]['extra_links'].items():
                    v['parent_link_name']+="_"+str(i+1)
                    v['link_name']+="_"+str(i+1)
                    v['joint_name']+="_"+str(i+1)
                    extra_links[k+"_"+str(i+1)]=v
            collison_sphere_buffer=max(collison_sphere_buffer,kinematic_list[i]['collision_sphere_buffer'])
            if kinematic_list[i]['extra_collision_spheres'] is not None:
                extra_collison_spheres.update({key+"_"+str(i+1):value for key,value in kinematic_list[i]['extra_collision_spheres'].items()})
            self_collison_ignore.update({key+"_"+str(i+1):[vi+"_"+str(i+1) for vi in value] for key,value in kinematic_list[i]['self_collision_ignore'].items()})
            self_collison_buffer.update({key+"_"+str(i+1):value for key,value in kinematic_list[i]['self_collision_buffer'].items()})
            mesh_link_names+=[name+"_"+str(i+1) for name in kinematic_list[i]['mesh_link_names']]
            collision_link_names+=[name+"_"+str(i+1) for name in kinematic_list[i]['collision_link_names']]
            use_global_cumul=(use_global_cumul or kinematic_list[i]['use_global_cumul'])
            cspace['joint_names']+=[name+"_"+str(i+1) for name in kinematic_list[i]['cspace']['joint_names']]
            cspace['retract_config']+=kinematic_list[i]['cspace']['retract_config']
            cspace['null_space_weight']+=kinematic_list[i]['cspace']['null_space_weight']
            cspace['cspace_distance_weight']+=  kinematic_list[i]['cspace']['cspace_distance_weight']
            cspace['max_jerk']=min(cspace['max_jerk'],kinematic_list[i]['cspace']['max_jerk'])
            cspace['max_acceleration']=min(cspace['max_acceleration'],kinematic_list[i]['cspace']['max_acceleration'])
        new_kinematics['collision_spheres']=collison_sphere
        new_kinematics['ee_link']=ee_link
        new_kinematics['link_names']=links_names
        new_kinematics['lock_joints']=lock_joints
        new_kinematics['extra_links']=extra_links
        new_kinematics['collision_sphere_buffer']=collison_sphere_buffer
        new_kinematics['extra_collision_spheres']=extra_collison_spheres
        new_kinematics['self_collision_ignore']=self_collison_ignore
        new_kinematics['self_collision_buffer']=self_collison_buffer
        new_kinematics['mesh_link_names']=mesh_link_names
        new_kinematics['collision_link_names']=collision_link_names
        new_kinematics['use_global_cumul']=use_global_cumul
        new_kinematics['cspace']=cspace
        return {'robot_cfg':{'kinematics':new_kinematics}}
    
    def return_world_config_checker(self, physics, collision_world = None):
        
        collision_checker = CollisionCheckerType.PRIMITIVE if self.use_primitive_collision else CollisionCheckerType.MESH

        if collision_world is None and self.check_world_collision:
            self.collision_world.update_curobo_world(physics)
            if self.use_primitive_collision:
                collision_world=self.collision_world.get_as_obb()
            else:
                collision_world=self.collision_world.get_as_class()
        elif not self.check_world_collision:
                
            collision_checker=CollisionCheckerType.MESH
            collision_world = c_world_config()
        
        print(collision_world, collision_checker)
        return collision_world , collision_checker
    
    def forward_kinematics_all(
        self,
        q: np.ndarray,
        physics = None,
        return_ee_pose: bool = False,
    ) -> Optional[Dict[str, Pose]]:
        if physics is None:
            physics = self.physics.copy(share_model=True)
        physics = physics.copy(share_model=True)
        
        # transform inhand objects!
        obj_transforms = dict()
        for robot_name, obj_info in self.inhand_object_info.items():
            gripper_pose = self.robots[robot_name].get_ee_pose(physics)
            if obj_info is not None:
                body_name, site_name, joint_name, (start, end) = obj_info
                obj_quat = mat_to_quat(
                    physics.data.site(site_name).xmat.reshape((3, 3))
                )
                obj_pos = physics.data.site(site_name).xpos
                rel_rot = quaternions.qmult( 
                    quaternions.qinverse(
                        gripper_pose.orientation
                        ),
                    obj_quat,
                    )
                rel_pos = obj_pos - gripper_pose.position 
                obj_transforms[robot_name] = (rel_pos, rel_rot)
            else:
                obj_transforms[robot_name] = None
        
        physics.data.qpos[self.all_joint_idxs_in_qpos] = q
        physics.forward()

        ee_poses = {}
        for robot_name, robot in self.robots.items():
            ee_poses[robot_name] = robot.get_ee_pose(physics)
        
        # also transform inhand objects!
        for robot_name, obj_info in self.inhand_object_info.items():
            if obj_info is not None:
                body_name, site_name, joint_name, (start, end) = obj_info
                rel_pos, rel_rot = obj_transforms[robot_name] 
                new_ee_pos = ee_poses[robot_name].position
                new_ee_quat = ee_poses[robot_name].orientation 
                target_pos = new_ee_pos + rel_pos 
                target_quat = quaternions.qmult(new_ee_quat, rel_rot) 
                result = self.solve_ik(
                    physics,
                    site_name,
                    target_pos,
                    target_quat,
                    joint_names=[joint_name], 
                    max_steps=300,
                    inplace=0,   
                    )
                if result is not None:
                    new_obj_qpos = result.qpos[start:end]
                    physics.data.qpos[start:end] = new_obj_qpos
                    physics.forward()
        if return_ee_pose:
            return ee_poses
        # physics.step(10) # to make sure the physics is stable
        return physics # a copy of the original physics object 
 
    def initalize_ik(self, 
        physics, 
        number_seeds=20,
        position_threshold=1e-3,
        rotation_threshold=5e-2,
        check_self_collision=True,
        check_world_collision=False,
        collision_world=None,
        use_primitive_collisions=True
        ):
        
        #print(new_target_pose)
        self.check_self_collision=check_self_collision
        self.check_world_collision=check_world_collision
        self.use_primitive_collision=use_primitive_collisions
        tensor_args = TensorDeviceType()
        
            
        collision_world,collision_checker=self.return_world_config_checker(physics=physics,collision_world=collision_world)
        

        ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_config,
            collision_world,
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            collision_checker_type=collision_checker,
            num_seeds=number_seeds,
            self_collision_check=check_self_collision,
            self_collision_opt=check_self_collision,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(ik_config)
    
    def solve_ik(
        self,
        physics, 
        target_pos, 
        collision_world = None
    ):
        physics_cp = physics.copy(share_model=True) 
        
        
        collision_world,collision_checker=self.return_world_config_checker(physics, collision_world)
        self.ik_solver.update_world(collision_world)
        primary_goal_pose=Pose.from_list(list(target_pos[self.primary_robot_name]))
        other_poses={}
        for name,pose in target_pos.items():
            if name is not self.primary_robot_name:
                other_poses[self.robots[name].curobo_robot_config.kinematics.kinematics_config.ee_link]=Pose.from_list(list(pose))
        #print(goal_pose.quaternion)
        ik_result=self.ik_solver.solve_single(goal_pose=primary_goal_pose,link_poses=other_poses)
        ik_result.get_unique_solution()
        
        return ik_result.solution.detach().cpu().squeeze().numpy() if ik_result.success else None 

    def initialize_motion_planner(
        self,
        physics,
        position_threshold=1e-3,
        rotation_threshold=5e-2,
        interpolation_dt=0.02,
        trajopt_dt=0.25,
        collision_activation_distance=0.01,
        check_world_collision=False,
        collision_world=None,
        use_primitive_collisions=True
        ):
        #self.check_self_collision=check_self_collision
        self.check_world_collision=check_world_collision
        self.use_primitive_collision=use_primitive_collisions
        tensor_args = TensorDeviceType()
        collision_world,col_checker=self.return_world_config_checker(physics=physics,collision_world=collision_world)
        
        print(col_checker)
        motion_gen_config=MotionGenConfig.load_from_robot_config(
            self.robot_config,
            collision_world,
            interpolation_dt=interpolation_dt,
            collision_checker_type=col_checker,
            collision_activation_distance=collision_activation_distance,
            trajopt_dt=trajopt_dt,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold
        )
        
        self.motion_generator=MotionGen(motion_gen_config)
        self.motion_generator.warmup()
    
    def plan(
        self,
        physics, 
        target_pos, 
        target_quat = None,
        start_state=None,
        max_attempts=5,
        time_dilation=0.5
        ):
        collision_world,collision_checker=self.return_world_config_checker(physics=physics,collision_world=collision_world)
        self.motion_generator.update_world(collision_world)
        
        tensor_args = TensorDeviceType()
        primary_goal_pose=Pose.from_list(list(target_pos[self.primary_robot_name]))
        other_poses={}
        for name,pose in target_pos.items():
            if name is not self.primary_robot_name:
                other_poses[self.robots[name].curobo_robot_config.kinematics.kinematics_config.ee_link]=Pose.from_list(list(pose))
        if start_state is None:
            start_qpos=np.array([])
            for name in self.names_list:
                start_qpos=np.concatenate((start_qpos,physics.data.qpos(self.robots[name].joint_idxs_in_qpos)))  
        else:
            start_qpos=start_state
        start_state=JointState.from_list(position=[start_qpos.tolist()],
                                             velocity=[np.zeros_like(start_qpos).tolist()],
                                             acceleration=[np.zeros_like(start_qpos).tolist()],
                                             tensor_args=tensor_args)
        result = self.motion_generator.plan_single(start_state, primary_goal_pose, MotionGenPlanConfig(max_attempts=max_attempts,time_dilation_factor=time_dilation),link_poses=other_poses)
        
        if result.success:
            traj = result.get_interpolated_plan()
            return traj.position.detach().cpu().numpy()
        else:
            return None
        