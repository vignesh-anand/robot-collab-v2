import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet
import os
from copy import deepcopy
from curobo.geom.types import WorldConfig as curobo_WorldConfig
from curobo.util_file import get_assets_path

class WorldConfig():

    def __init__(
        self, 
        mjcf_model:Any, 
        physics:Any, 
        mesh_dir:str = "", 
        skip_robot_name: Union[List[str], str] = "", 
        mesh_test=False
    ):
        self.world_config = {"cylinder": {}, "cuboid": {}, "mesh": {}, "capsule": {}}
        
        self.mjcf_model = mjcf_model
        self.physics = physics.copy(share_model=True)
        self.robot_name = skip_robot_name if isinstance(skip_robot_name,str) else "world"

        ### Generate World Config and Update the world###
        self.store_assets_in_dir(out_dir = mesh_dir)

        self.generate_world_config(mesh_test=mesh_test, skip_robot_name=skip_robot_name)
        
        self.update_curobo_world(physics=physics, robot_name=skip_robot_name)
    
    def store_assets_in_dir(self, out_dir = None):
        from dm_control.mujoco.wrapper import util
    

        if out_dir is None or out_dir == "":
            out_dir = self.physics.model.name + '_scene/'
            
        out_dir=os.path.join(get_assets_path(),out_dir)
        
        self.mesh_dir = out_dir

        assets = self.mjcf_model.get_assets() # this model is mjcf-created
        assets[self.physics.model.name + '_task.xml'] = self.mjcf_model.to_xml_string(precision=4)
        os.makedirs(out_dir, exist_ok=True)
        for filename, contents in assets.items():
            with open(os.path.join(out_dir, filename), 'wb') as f:
                f.write(util.to_binary_string(contents))


    def get_as_dict(self):
        return self.world_config
    
    def get_as_obb(self):
        world_model = curobo_WorldConfig.from_dict(self.world_config)
        return curobo_WorldConfig.create_obb_world(world_model)

    def get_as_class(self):
        return curobo_WorldConfig.from_dict(self.world_config)

    def save_obb_mesh(self, filename):
        w = self.get_as_obb()
        w.save_world_as_mesh(filename)
    
    def save_world_mesh(self, filename):
        w = self.get_as_class()
        w.save_world_as_mesh(filename)        

    def get_mesh_filename(self, geom):
        mesh_id = geom.dataid.item()
        mesh_name = self.physics.model.mesh(mesh_id).name
        mesh_file = self.mjcf_model.asset.mesh[mesh_name].file.get_vfs_filename()
        return self.mesh_dir + mesh_file

    """
    Generate Base Curobo World Config Function from MJCF Physics Model
    """
    def generate_world_config(
            self, 
            geom_group: int = 3, 
            mesh_test: bool = False, 
            skip_robot_name: Union[List[str], str] = ""
    ):

        collision_geom_idx = np.where(self.physics.model.geom_group == geom_group)[0]
        
        ### Looping over Robot Meshes for world_config ####
        for i in collision_geom_idx:
            
            # Access the geom model and data
            geom = self.physics.model.geom(i)
            geom_data = self.physics.data.geom(i)

            if(geom.name.split("/")[0] in skip_robot_name):
                continue
            
            ## Check for different geom types ##

            ##  Mujoco Description: Sphere 
            #   Centered at geom's position 
            #   Size Params: radius of sphere

            if geom.type[0] == 2: #Sphere
                self.world_config["sphere"][geom.name] = {  "radius": float(geom.size[0]),
                                                            "pose": np.concatenate([geom_data.xpos, [1, 0, 0, 0]])
                                                        }
            
            ##  Mujoco Description: Capsule is a cylinder capped with two half-spheres. 
            #   Orientation along z-axis of geom's frame. 
            #   Size Params: radius, half-height of cylinder 
            #   Note: However capsules as well as cylinders can also be thought of as connectors, 
            #   allowing an alternative specification with the fromto attribute below. 
            #   In that case only one size parameter is required, namely the radius of the capsule.

            if geom.type[0] ==  3: #Capsule       
                self.world_config["capsule"][geom.name] = { "radius": float(geom.size[0]),
                                                            "base": np.array([0.0, 0.0, 0.0]),
                                                            "tip": np.array([0., 0.0, 2.0*float(geom.size[1])]),
                                                            "pose": np.concatenate([geom_data.xpos, [1, 0, 0, 0]])
                                                        }

            ##  Mujoco Description: Cylinder
            #   Orientation along z-axis of geom's frame
            #   Size Params: radius, half-height 
            #   Note: can be specified with the fromto attribute

            if geom.type[0] == 5: #Cylinder
                self.world_config["cylinder"][geom.name] = {    "radius": float(geom.size[0]),
                                                                "height": 2.0*float(geom.size[1]),
                                                                "pose" : np.concatenate([geom_data.xpos, [1, 0, 0, 0]])
                                                            }
            
            ##  Mujoco Description: Box
            #   Size Params: half-sizes of X, Y, Z along the xyz axes of the geom's frame
            if geom.type[0] ==  6: #Cuboid
                # print(i)
                self.world_config["cuboid"][geom.name] = {  "dims": list(geom.size*2),
                                                            "pose": np.concatenate([geom_data.xpos, [1, 0,0,0]])
                                                        }

            ## Mujoco Description: Mesh
            #   A mesh file has to be provided 
            #   Geom sizes are ignored!!!

            if geom.type[0] ==  7: #Mesh

                mesh_file = self.get_mesh_filename(geom)
                
                self.world_config["mesh"][geom.name] = { "file_path": mesh_file,
                                                    "pose": np.concatenate([geom_data.xpos, [1, 0, 0, 0]])
                                                }

        ### Testing Meshes Work ###
        if mesh_test:
            for key, value in self.world_config["mesh"].items():
                print(key, ":", value["file_path"])
    
    """
    Transform Object Pose to Robot Frame:
    
    Notes:
    - params are using Mujoco Convention for Orientation when passed in: (wxyz)
    - returned in Mujoco Convention
    """
    def transform_pose_robot(self, object_pose: np.ndarray, robot_pose: np.ndarray, verbose: bool = False):
        robot_pose_x = robot_pose[:3]
        robot_pose_q = robot_pose[[4,5,6,3]]
        object_pose_x = object_pose[:3]
        object_pose_q = object_pose[[4,5,6,3]]
        
        robot_r = R.from_quat(robot_pose_q)
        object_r = R.from_quat(object_pose_q)
        
        new_pose_x = robot_r.inv().apply(object_pose_x - robot_pose_x)
        new_quat = robot_r.inv()*object_r
        
        if verbose:
            print(object_pose_q,robot_pose_q,robot_r.inv().as_matrix(),object_r.as_matrix())
        
        return np.concatenate(( new_pose_x, -new_quat.as_quat()[[3,0,1,2]] ))
    
    """
    Update the Curobo World Config after making Changes to MJCF Environment
    """
    def update_curobo_world(self, physics:Any = None, robot_name:str = None):
        
        assert physics is not None, "Physics Model is None. Please provide a valid physics model."

        if robot_name is None:
            robot_name = self.robot_name
        robot_name=robot_name+ "/"

        robot_pos = np.concatenate((physics.named.data.xpos[robot_name],
                                    physics.named.data.xquat[robot_name]), axis=0)
        

        for object_type in self.world_config:
            
            print("Updating Type: ", object_type)

            object_dict = self.world_config[object_type]
            
            if object_type == "mesh":
                for name in object_dict:
                    new_xpos = physics.named.data.xpos[physics.named.model.geom_bodyid[name]].copy()
                    new_quat = physics.named.data.xquat[physics.named.model.geom_bodyid[name]]
                    xpos_robot_frame = self.transform_pose_robot(np.concatenate((new_xpos,new_quat), axis=0).copy(),robot_pos.copy())
                    object_dict[name]['pose'] = xpos_robot_frame.copy()

            else:
                for name in object_dict:
                    new_xpos = physics.named.data.geom_xpos[name].copy()
                    new_quat = R.from_matrix(physics.named.data.geom_xmat[name].reshape(3,3)).as_quat()[[3,0,1,2]]
                    xpos_robot_frame = self.transform_pose_robot(np.concatenate((new_xpos,new_quat), axis=0).copy(),robot_pos.copy())
                    object_dict[name]['pose'] = xpos_robot_frame.copy()
        




