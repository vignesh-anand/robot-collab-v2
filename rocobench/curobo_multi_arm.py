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


class MultiArmCurobo:
    """ Stores the info for a group of arms and plan all the combined joints together """
    def __init__(
        self,
        physics,
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

        for name, robot in self.robots.items():
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


        self.joint_minmax = np.array([jrange for jrange in self.all_joint_ranges])
        self.joint_ranges = self.joint_minmax[:, 1] - self.joint_minmax[:, 0]

        # assign a list of allowed grasp ids to each robot

    
    def set_inhand_info(self, physics, inhand_object_info: Optional[Dict[str, Tuple]] = None):
        """ Set the inhand object info """
        self.inhand_object_info = dict()
        if inhand_object_info is not None:
            for name, robot in self.robots.items():
                self.inhand_object_info[name] = None
                
                obj_info = inhand_object_info.get(name, None)
                
                if obj_info is not None:
                    if 'rope' in obj_info[0] or 'CB' in obj_info[0]:
                        continue
                    assert len(obj_info) == 3, f"inhand obj info: {obj_info} should be a tuple of (obj_body_name, obj_site_name, obj_joint_name)"
                    body_name, site_name, joint_name = obj_info
                    try:
                        mjsite = physics.data.site(site_name)
                        qpos_slice = physics.named.data.qpos._convert_key(joint_name) 
                    except: 
                        print(f"Error: site_name: {site_name} joint_name {joint_name} not found in mujoco model")
                        breakpoint() 
                    self.inhand_object_info[name] = (body_name, site_name, joint_name, (qpos_slice.start, qpos_slice.stop))
        return 
 
    
    def set_ungraspable(
        self, 
        graspable_object_dict: Optional[Dict[str, List[str]]]
    ):
        """ Find all sim objects that are not graspable """
        
        all_bodies = []
        for i in range(self.physics.model.nbody):
            all_bodies.append(self.physics.model.body(i))

        # all robot link bodies are ungraspable:
        ungraspable_ids = [0]  # world
        for name, robot in self.robots.items():
            ungraspable_ids.extend(
                robot.collision_link_ids
            )
        # append all children of ungraspable body
        ungraspable_ids += [
            body.id for body in all_bodies if body.rootid[0] in ungraspable_ids
        ]
 
        if graspable_object_dict is None or len(graspable_object_dict) == 0:
            graspable = set(
                [body.id for body in all_bodies if body.id not in ungraspable_ids]
            )
            ungraspable = set(ungraspable_ids)
            self.graspable_body_ids = {name: graspable for name in self.robots.keys()}
            self.ungraspable_body_ids = {name: ungraspable for name in self.robots.keys()}
        else: 
            # in addition to robots, everything else would be ungraspable if not in this list of graspable objects
            self.graspable_body_ids = {}
            self.ungraspable_body_ids = {}
            for robot_name, graspable_object_names in graspable_object_dict.items():
                graspable_ids = [
                    body.id for body in all_bodies if body.name in graspable_object_names
                ]
                graspable_ids += [
                    body.id for body in all_bodies if body.rootid[0] in graspable_ids
                ]
                self.graspable_body_ids[robot_name] = set(graspable_ids)
                robot_ungraspable = ungraspable_ids.copy()
                robot_ungraspable += [
                    body.id for body in all_bodies if body.rootid[0] not in graspable_ids
                ]
                self.ungraspable_body_ids[robot_name] = set(ungraspable_ids)
            # breakpoint()

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
 