import abc
import xml.etree.ElementTree as ET


class BaseParser(object):
    """
    Base class for Parser objects used by renderers.
    """

    def __init__(self, renderer, env):
        """
        Parse the mujoco xml and initialize iG renderer objects.
        Env is an object describing the task and the environment. 
        Must contain the following attributes:
        1. MJCF model
        2. MJCF physics

        Args:
            renderer: the renderer
            env : Mujoco env
        """
        self.renderer = renderer
        self.env = env
        self.xml_root = ET.fromstring(self.env.model.to_xml_string())
        self.parent_map = {c: p for p in self.xml_root.iter() for c in p}
        self.visual_objects = {}

    @abc.abstractmethod
    def parse_textures(self):
        """
        Parse and load all textures and store them
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parse_materials(self):
        """
        Parse all materials and use texture mapping to initialize materials
        """
        raise NotImplementedError

    def parse_cameras(self):
        """
        Parse cameras and initialize the cameras.
        """
        raise NotImplementedError

    def parse_meshes(self):
        """
        Create mapping of meshes.
        """
        raise NotImplementedError

    def parse_geometries(self):
        """
        Iterate through each geometry and load it in the renderer.
        """
        raise NotImplementedError
