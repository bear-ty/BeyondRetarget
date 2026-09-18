import xml.etree.ElementTree as ET

import numpy as np
import torch

from lib.robot import torch_utils


def _identity_dof_mapping(dof: torch.Tensor) -> torch.Tensor:
    return dof


G1_NMR_TO_XML_JOINT_MAPPING = [
    0, 6, 12,
    1, 7, 13,
    2, 8, 14,
    3, 9, 15, 22,
    4, 10, 16, 23,
    5, 11, 17, 24,
    18, 25,
    19, 26,
    20, 27,
    21, 28,
]


def _rpy_to_quat_xyzw(rpy):
    roll, pitch, yaw = [float(v) for v in rpy]
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return np.array([x, y, z, w], dtype=np.float32)


class Joint:
    def __init__(self, name, dof_dim, axis):
        self._name = name
        self._dof_dim = dof_dim
        self._axis = axis
        self._dof_idx = -1

    def set_dof_idx(self, dof_idx):
        if self._dof_dim == 0:
            raise ValueError(f"Joint {self._name} has no dof")
        self._dof_idx = dof_idx

    def dof_to_rot(self, dof):
        rot_shape = list(dof.shape[:-1]) + [4]
        ret_rot = torch.zeros(rot_shape, dtype=dof.dtype, device=dof.device)
        if self._dof_dim == 0:
            ret_rot[..., -1] = 1.0
        elif self._dof_dim == 1:
            axis = torch.broadcast_to(self._axis, ret_rot[..., 0:3].shape)
            ret_rot[:] = torch_utils.axis_angle_to_quat(axis, dof.squeeze(-1))
        elif self._dof_dim == 3:
            ret_rot[:] = torch_utils.exp_map_to_quat(dof)
        else:
            raise ValueError(f"Unsupported dof_dim={self._dof_dim} for joint {self._name}")
        return ret_rot

    @property
    def dof_dim(self):
        return self._dof_dim

    @property
    def dof_idx(self):
        return self._dof_idx


class RobotKinematicsModel:
    def __init__(self, file_path, device, model_to_xml_dof=None, neutral_dof=None):
        self._device = device
        self._file_path = file_path
        self._model_to_xml_dof = model_to_xml_dof
        self._neutral_dof = None if neutral_dof is None else torch.as_tensor(neutral_dof, dtype=torch.float32, device=device)
        self._build_kinematics_model()
        self._set_dof_indices()
        if self._neutral_dof is not None and int(self._neutral_dof.numel()) != int(self._num_dof):
            raise ValueError(f"Expected neutral_dof with {self._num_dof} values, got {self._neutral_dof.numel()}")

    def _build_kinematics_model(self):
        self._body_names = []
        self._parent_indices = []
        self._local_translation = []
        self._local_rotation = []
        self._joints = []
        self._dof_size = []
        self._dof_upper_limits = []
        self._dof_lower_limits = []

        if not (self._file_path.endswith(".xml") or self._file_path.endswith(".urdf")):
            raise NotImplementedError("Only MuJoCo XML or URDF is supported")

        self._parse_xml()
        self._parent_indices = torch.tensor(self._parent_indices, dtype=torch.long, device=self._device)
        self._local_translation = torch.tensor(np.array(self._local_translation), dtype=torch.float32, device=self._device)
        self._local_rotation = torch.tensor(np.array(self._local_rotation), dtype=torch.float32, device=self._device)
        self._num_dof = sum(self._dof_size)
        self._dof_lower_limits = torch.tensor(self._dof_lower_limits, dtype=torch.float32, device=self._device)
        self._dof_upper_limits = torch.tensor(self._dof_upper_limits, dtype=torch.float32, device=self._device)
        if self._rot_unit == "degree":
            self._dof_lower_limits = torch.deg2rad(self._dof_lower_limits)
            self._dof_upper_limits = torch.deg2rad(self._dof_upper_limits)

    def _parse_xml(self):
        tree = ET.parse(self._file_path)
        xml_doc_root = tree.getroot()
        if xml_doc_root.tag == "mujoco":
            xml_world_body = xml_doc_root.find("worldbody")
            assert xml_world_body is not None, "worldbody not found"

            xml_body_root = xml_world_body.find("body")
            assert xml_body_root is not None, "body not found"

            compiler_data = xml_doc_root.find("compiler")
            self._rot_unit = compiler_data.attrib.get("angle", "degree") if compiler_data is not None else "degree"

            def _add_mujoco_body(xml_node, parent_index, body_index):
                body_name = xml_node.attrib.get("name")
                pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
                rot = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
                rot_w = rot[..., 0].copy()
                rot[..., 0:3] = rot[..., 1:]
                rot[..., 3] = rot_w

                if body_index == 0:
                    curr_joint = Joint(name=body_name, dof_dim=0, axis=None)
                else:
                    curr_joints = xml_node.findall("joint")
                    num_joints = len(curr_joints)
                    if num_joints == 0:
                        curr_joint = Joint(name=body_name, dof_dim=0, axis=None)
                    elif num_joints == 1:
                        axis = torch.from_numpy(np.fromstring(curr_joints[0].attrib.get("axis"), dtype=float, sep=" ")).to(self._device)
                        curr_joint = Joint(name=body_name, dof_dim=1, axis=axis)
                        dof_limits = np.fromstring(curr_joints[0].attrib.get("range"), dtype=float, sep=" ")
                        self._dof_lower_limits.append(dof_limits[0])
                        self._dof_upper_limits.append(dof_limits[1])
                    elif num_joints == 3:
                        curr_joint = Joint(name=body_name, dof_dim=3, axis=None)
                        for joint in curr_joints:
                            dof_limits = np.fromstring(joint.attrib.get("range"), dtype=float, sep=" ")
                            self._dof_lower_limits.append(dof_limits[0])
                            self._dof_upper_limits.append(dof_limits[1])
                    else:
                        raise ValueError(f"Invalid number of joints: {num_joints} of body: {body_name}")

                self._body_names.append(body_name)
                self._parent_indices.append(parent_index)
                self._local_rotation.append(rot)
                self._local_translation.append(pos)
                self._joints.append(curr_joint)
                self._dof_size.append(curr_joint.dof_dim)

                curr_index = body_index
                body_index += 1
                for child in xml_node.findall("body"):
                    body_index = _add_mujoco_body(child, curr_index, body_index)
                return body_index

            _add_mujoco_body(xml_body_root, -1, 0)
            return

        if xml_doc_root.tag == "robot":
            self._rot_unit = "radian"
            link_lookup = {}
            joint_lookup = {}

            def _joint_to_body_name(joint_name: str) -> str:
                if joint_name.endswith("_joint"):
                    return joint_name[:-6]
                return joint_name

            for link_node in xml_doc_root.findall("link"):
                link_name = link_node.attrib["name"]
                link_lookup[link_name] = link_node
            for joint_node in xml_doc_root.findall("joint"):
                joint_lookup[joint_node.attrib["name"]] = joint_node

            child_links = set()
            for joint_node in xml_doc_root.findall("joint"):
                child = joint_node.find("child")
                if child is not None:
                    child_links.add(child.attrib["link"])

            root_links = [link.attrib["name"] for link in xml_doc_root.findall("link") if link.attrib["name"] not in child_links]
            if len(root_links) != 1:
                raise ValueError(f"Expected a single root link in URDF, got {root_links}")
            root_link = root_links[0]

            def _add_urdf_body(link_name, parent_index):
                link_node = link_lookup[link_name]
                joint_node = None
                if parent_index >= 0:
                    for candidate in xml_doc_root.findall("joint"):
                        child = candidate.find("child")
                        if child is not None and child.attrib.get("link") == link_name:
                            joint_node = candidate
                            break

                pos = np.zeros(3, dtype=np.float32)
                rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                if joint_node is not None:
                    origin = joint_node.find("origin")
                    if origin is not None:
                        pos = np.fromstring(origin.attrib.get("xyz", "0 0 0"), dtype=float, sep=" ").astype(np.float32)
                        rot = _rpy_to_quat_xyzw(np.fromstring(origin.attrib.get("rpy", "0 0 0"), dtype=float, sep=" "))

                joint_nodes = []
                if joint_node is not None and joint_node.attrib.get("type", "fixed") != "fixed":
                    joint_nodes = [joint_node]

                if parent_index < 0:
                    curr_joint = Joint(name=link_name, dof_dim=0, axis=None)
                elif len(joint_nodes) == 0:
                    curr_joint = Joint(name=link_name, dof_dim=0, axis=None)
                elif len(joint_nodes) == 1:
                    axis = torch.from_numpy(np.fromstring(joint_nodes[0].find("axis").attrib.get("xyz"), dtype=float, sep=" ")).to(self._device)
                    curr_joint = Joint(name=link_name, dof_dim=1, axis=axis)
                    limit = joint_nodes[0].find("limit")
                    if limit is not None:
                        self._dof_lower_limits.append(float(limit.attrib.get("lower", "0")))
                        self._dof_upper_limits.append(float(limit.attrib.get("upper", "0")))
                    elif joint_nodes[0].attrib.get("type") == "continuous":
                        self._dof_lower_limits.append(float(-np.pi))
                        self._dof_upper_limits.append(float(np.pi))
                    else:
                        self._dof_lower_limits.append(0.0)
                        self._dof_upper_limits.append(0.0)
                else:
                    raise ValueError(f"Unsupported URDF joint count for link {link_name}: {len(joint_nodes)}")

                self._body_names.append(link_name)
                self._parent_indices.append(parent_index)
                self._local_rotation.append(rot)
                self._local_translation.append(pos)
                self._joints.append(curr_joint)
                self._dof_size.append(curr_joint.dof_dim)

                curr_index = len(self._body_names) - 1
                for joint_node in xml_doc_root.findall("joint"):
                    child = joint_node.find("child")
                    parent = joint_node.find("parent")
                    if child is None or parent is None:
                        continue
                    if parent.attrib.get("link") != link_name:
                        continue
                    child_link = child.attrib.get("link")
                    _add_urdf_body(child_link, curr_index)
                return curr_index

            _add_urdf_body(root_link, -1)
            return

        raise NotImplementedError(f"Unsupported robot file format: {xml_doc_root.tag}")

    def _set_dof_indices(self):
        curr_dof_idx = 0
        for joint in self._joints:
            if joint.dof_dim > 0:
                joint.set_dof_idx(curr_dof_idx)
                curr_dof_idx += joint.dof_dim

    def model_dof_to_xml_dof(self, dof):
        if self._model_to_xml_dof is None:
            return dof
        if callable(self._model_to_xml_dof):
            return self._model_to_xml_dof(dof)
        if dof.shape[-1] != len(self._model_to_xml_dof):
            raise ValueError(f"Expected {len(self._model_to_xml_dof)} robot dofs, got {dof.shape[-1]}")
        if self._neutral_dof is None:
            xml_dof = torch.zeros(*dof.shape[:-1], self._num_dof, dtype=dof.dtype, device=dof.device)
        else:
            neutral = self._neutral_dof.to(dtype=dof.dtype, device=dof.device)
            xml_dof = neutral.view(*([1] * (dof.dim() - 1)), self._num_dof).expand(*dof.shape[:-1], self._num_dof).clone()
        xml_dof[..., self._model_to_xml_dof] = dof
        return xml_dof

    def dof_to_rot(self, dof):
        dof = self.model_dof_to_xml_dof(dof)
        rot_shape = list(dof.shape[:-1]) + [self.num_joint - 1, 4]
        joint_rot = torch.zeros(rot_shape, dtype=dof.dtype, device=dof.device)
        for j in range(1, self.num_joint):
            joint = self._joints[j]
            if joint.dof_idx == -1:
                joint_rot[..., j - 1, -1] = 1.0
            else:
                joint_rot[..., j - 1, :] = joint.dof_to_rot(dof[..., joint.dof_idx:joint.dof_idx + joint.dof_dim])
        return joint_rot

    def forward_kinematics(self, root_pos, root_rot, dof_pos):
        joint_rot = self.dof_to_rot(dof_pos)
        body_pos = [None] * self.num_joint
        body_rot = [None] * self.num_joint
        body_pos[0] = root_pos
        body_rot[0] = root_rot

        for j in range(1, self.num_joint):
            j_rot = joint_rot[..., j - 1, :]
            local_trans = self._local_translation[j]
            local_rot = self._local_rotation[j]
            parent_idx = self._parent_indices[j]

            parent_pos = body_pos[parent_idx]
            parent_rot = body_rot[parent_idx]

            local_trans_broadcast = torch.broadcast_to(local_trans, parent_pos.shape)
            local_rot_broadcast = torch.broadcast_to(local_rot, parent_rot.shape)

            world_trans = torch_utils.quat_rotate(parent_rot, local_trans_broadcast)
            curr_pos = parent_pos + world_trans
            curr_rot = torch_utils.quat_mul(local_rot_broadcast, j_rot)
            curr_rot = torch_utils.quat_mul(parent_rot, curr_rot)

            body_pos[j] = curr_pos
            body_rot[j] = curr_rot

        return torch.stack(body_pos, dim=-2), torch.stack(body_rot, dim=-2)

    @property
    def body_names(self):
        return self._body_names

    @property
    def num_joint(self):
        return len(self._body_names)

    @property
    def num_dof(self):
        return self._num_dof

    @property
    def dof_lower_limits(self):
        return self._dof_lower_limits

    @property
    def dof_upper_limits(self):
        return self._dof_upper_limits
