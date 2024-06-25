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
from rocobench.envs.env_utils import Pose
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
from rocobench.envs.world_config import WorldConfig

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
        for name, robot in self.robots.items():
            with open('/home/vignesh/curobo/src/curobo/content/configs/robot/ur5e.yml', 'r') as f:
                config_list.append(yaml.load(f, Loader=yaml.SafeLoader))
            urdf_list.append(URDF.load(robot.urdf_path))
            pose_list.append(np.concatenate((physics.named.data.xpos[self.name+"/"],
                                    physics.named.data.xquat[self.name+"/"]), axis=0).tolist())
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
 
    
    def check_joint_range(
        self, 
        physics,
        joint_names,
        qpos_idxs,
        ik_result,
        allow_err=0.03,
    ) -> bool:
        _lower, _upper = physics.named.model.jnt_range[joint_names].T
        qpos = ik_result.qpos[qpos_idxs]
        assert len(qpos) == len(_lower) == len(_upper), f"Shape mismatch: qpos: {qpos}, _lower: {_lower}, _upper: {_upper}"
        for i, name in enumerate(joint_names):
            if qpos[i] < _lower[i] - allow_err or qpos[i] > _upper[i] + allow_err:
                # print(f"Joint {name} out of range: {_lower[i]} < {qpos[i]} < {_upper[i]}")
                return False 
        return True

    def solve_ik(
        self,
        physics,
        site_name,
        target_pos,
        target_quat,
        joint_names, 
        tol=1e-14,
        max_steps=300,
        max_resets=20,
        inplace=True, 
        max_range_steps=0,
        qpos_idxs=None,
        allow_grasp=True,
        check_grasp_ids=None,
        check_relative_pose=False
    ):
        physics_cp = physics.copy(share_model=True)
        
        def reset_fn(physics):
            model = physics.named.model 
            _lower, _upper = model.jnt_range[joint_names].T
            
            curr_qpos = physics.named.data.qpos[joint_names]
            # deltas = (_upper - _lower) / 2
            # new_qpos = self.np_random.uniform(low=_lower, high=_upper)
            new_qpos = self.np_random.uniform(low=curr_qpos-0.5, high=curr_qpos + 0.5)
            new_qpos = np.clip(new_qpos, _lower, _upper)
            physics.named.data.qpos[joint_names] = new_qpos
            physics.forward()

        for i in range(max_resets):
            # print(f"Resetting IK {i}")
            if i > 0:
                reset_fn(physics_cp)
                
            result = qpos_from_site_pose(
                physics=physics_cp,
                site_name=site_name,
                target_pos=target_pos,
                target_quat=target_quat,
                joint_names=joint_names,
                tol=tol,
                max_steps=max_steps,
                inplace=True,
            )
            need_reset = False
            if result.success:
                in_range = True 
                collided = False
                if qpos_idxs is not None:
                    in_range = self.check_joint_range(physics_cp, joint_names, qpos_idxs, result)
                    ik_qpos = result.qpos.copy()
                    _low, _high = physics_cp.named.model.jnt_range[joint_names].T
                    ik_qpos[qpos_idxs] = np.clip(
                        ik_qpos[qpos_idxs], _low, _high
                    )
                    ik_qpos = ik_qpos[self.all_joint_idxs_in_qpos]
                    # print('checking collision on IK result: step {}'.format(i))
                    collided = self.check_collision(
                        physics=physics_cp,
                        robot_qpos=ik_qpos,
                        check_grasp_ids=check_grasp_ids,
                        allow_grasp=allow_grasp,
                        check_relative_pose=check_relative_pose,
                        )

                need_reset = (not in_range) or collided

            else:
                need_reset = True
            if not need_reset:
                break
        # img = physics_cp.render(camera_id='teaser', height=400, width=400)
        # plt.imshow(img)
        # plt.show()

        return result if result.success else None

    def inverse_kinematics_all(
        self,
        physics,
        ee_poses: Dict[str, Pose],
        inplace=False, 
        allow_grasp=True, 
        check_grasp_ids=None,
        check_relative_pose=False,
    ) -> Dict[str, Union[None, np.ndarray]]:

        if physics is None:
            physics = self.physics
        physics = physics.copy(share_model=True)
        results = dict() 
        for robot_name, target_ee in ee_poses.items():
            assert robot_name in self.robots, f"robot_name: {robot_name} not in self.robots"
            robot = self.robots[robot_name]
            pos = target_ee.position 
            quat = target_ee.orientation
            if robot.use_ee_rest_quat:
                quat = quaternions.qmult(
                    quat, robot.ee_rest_quat
                ), # TODO 
            # print(robot.ee_site_name, pos, quat, robot.joint_names)
            qpos_idxs = robot.joint_idxs_in_qpos     
            result = self.solve_ik(
                physics=physics,
                site_name=robot.ee_site_name,
                target_pos=pos,
                target_quat=quat,
                joint_names=robot.ik_joint_names,
                tol=1e-14,
                max_steps=300,
                inplace=inplace,  
                qpos_idxs=qpos_idxs,
                allow_grasp=allow_grasp, 
                check_grasp_ids=check_grasp_ids,
                check_relative_pose=check_relative_pose,
            )
            if result is not None:
                result_qpos = result.qpos[qpos_idxs].copy()
                _lower, _upper = physics.named.model.jnt_range[robot.ik_joint_names].T
                result_qpos = np.clip(result_qpos, _lower, _upper)
                results[robot_name] = (result_qpos, qpos_idxs) 
            else:
                results[robot_name] = None
        return results      


    def plan(
        self, 
        start_qpos: np.ndarray,  # can be either full length or just the desired qpos for the joints 
        goal_qpos: np.ndarray,
        init_samples: Optional[List[np.ndarray]] = None,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, int]] = None,
        skip_endpoint_collision_check: bool = False,
        skip_direct_path: bool = False,
        skip_smooth_path: bool = False,
        timeout: int = 200,
        check_relative_pose: bool = False,
    ) -> Tuple[Optional[List[np.ndarray]], str]:

        if len(start_qpos) != len(goal_qpos):
            return None, "RRT failed: start and goal configs have different lengths."
        if len(start_qpos) != len(self.all_joint_idxs_in_qpos):
            start_qpos = start_qpos[self.all_joint_idxs_in_qpos]
        if len(goal_qpos) != len(self.all_joint_idxs_in_qpos):
            goal_qpos = goal_qpos[self.all_joint_idxs_in_qpos]
  
        def collision_fn(q: np.ndarray, show: bool = False):
            return self.check_collision(
                robot_qpos=q,
                physics=self.physics,
                allow_grasp=allow_grasp,           
                check_grasp_ids=check_grasp_ids,  
                check_relative_pose=check_relative_pose,
                show=show,
                # detect_grasp=False, TODO?
            )
        if not skip_endpoint_collision_check:
            if collision_fn(start_qpos, show=1):
                # print("RRT failed: start qpos in collision.")
                return None, f"ReasonCollisionAtStart_time0_iter0"
            elif collision_fn(goal_qpos, show=1): 
                # print("RRT failed: goal qpos in collision.")
                return None, "ReasonCollisionAtGoal_time0_iter0"
        paths, info = birrt(
                start_conf=start_qpos,
                goal_conf=goal_qpos,
                distance_fn=self.ee_l2_distance,
                sample_fn=CenterWaypointsUniformSampler(
                    bias=0.05,
                    start_conf=start_qpos,
                    goal_conf=goal_qpos,
                    numpy_random=self.np_random,
                    min_values=self.joint_minmax[:, 0],
                    max_values=self.joint_minmax[:, 1],
                    init_samples=init_samples,
                ),
                extend_fn=self.extend_ee_l2,
                collision_fn=collision_fn,
                iterations=800,
                smooth_iterations=200,
                timeout=timeout,
                greedy=True,
                np_random=self.np_random,
                smooth_extend_fn=self.extend_ee_l2,
                skip_direct_path=skip_direct_path,
                skip_smooth_path=skip_smooth_path, # enable to make sure it passes through the valid init_samples 
            )
        if paths is None:
            return None, f"RRT failed: {info}"
        return paths, f"RRT succeeded: {info}"
 