import concurrent.futures
import typing
import sys
from copy import deepcopy

import numpy as np
from solver import Solver

try:
    if sys.platform != "win32":
        import pinocchio as pin
except ImportError:
    raise ImportError(
        "pinocchio not installed. Install with `pip install pin`")

__all__ = ["PinocchioSolver"]


class PinocchioSolver(Solver):
    def __init__(
        self,
        urdf_path: str,
        end_link_name: str,
        **kwargs,
    ):
        r"""Initializes the Pinocchio kinematics and dynamics solver.

            This class leverages Pinocchio to perform kinematic and dynamic
            computations for the specified robot model.

        Args:
            urdf_path (str, optional): Path to the robot's URDF file.
            end_link_name (str): The name of the end-effector link.
            **kwargs: Additional arguments for the base solver.
        """
        super().__init__(
            urdf_path=urdf_path,
            end_link_name=end_link_name,
            **kwargs,
        )
        gravity = kwargs.get("gravity", [0.0, 0.0, -9.81])

        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.model.gravity.vector[:] = [*gravity, 0, 0, 0]

        self.model_data = self.model.createData()

        self.link_id_to_frame_index = []

        self.index_s2p = np.arange(self.model.nv, dtype=np.int64)

        self.index_p2s = np.zeros_like(self.index_s2p)
        for s, p in enumerate(self.index_s2p):
            self.index_p2s[p] = s

        self.ids_qpos = np.zeros(self.model.nq, dtype=np.int64)
        self.num_qpos = np.zeros(self.model.nq, dtype=np.int64)
        self.num_vel = np.zeros(self.model.nq, dtype=np.int64)

        for i in range(self.model.nq):
            self.ids_qpos[i] = self.model.idx_qs[i + 1]
            self.num_qpos[i] = self.model.nqs[i + 1]
            self.num_vel[i] = self.model.nvs[i + 1]

        self.dof = self.model.nq
        self.ik_nearst_weight = np.ones(self.dof)

        self.upper_position_limits = self.model.upperPositionLimit
        self.lower_position_limits = self.model.lowerPositionLimit

        # Extract the names of all frames from the model and store them in the frame_names list
        self.frame_names = [frame.name for frame in self.model.frames.tolist()]

        # Check if the specified end link name exists in the frame names list
        if self.end_link_name not in self.frame_names:
            self.end_link_name = self.frame_names[-1]

    def get_dof(self) -> int:
        r"""Returns the degree of freedom (DOF) of the robot.

        Returns:
            int: The degree of freedom of the robot.
        """
        return self.dof

    def get_urdf_path(self) -> str:
        r"""Returns the file path to the URDF (Unified Robot Description Format) file.

        Returns:
            str: The file path to the URDF file.
        """
        return self.urdf_path

    def set_joint_order(self, names: typing.List[str]):
        r"""Sets the order of joints based on the provided joint names.

        This function updates the internal representation of joint indices
        based on the specified order. It also initializes arrays for
        joint position indices, dimensions, and velocity dimensions.

        Args:
            names (list of str): A list of joint names in the desired order.

        Raises:
            ValueError: If any provided joint name is invalid (i.e., does not exist
                    in the robot model).

        Notes:
            The function constructs a mapping between the internal and external
            representations of joint positions and velocities.
        """
        v = np.zeros(self.model.nv, dtype=np.int64)
        count = 0
        for name in names:
            i = self.model.getJointId(name)
            if i >= self.model.njoints:
                raise ValueError(f"Invalid joint name: {name}.")
            size = self.model.nvs[i]
            qi = self.model.idx_vs[i]
            for s in range(size):
                v[count] = qi + s
                count += 1

        self.index_s2p = v

        self.index_p2s = np.zeros_like(self.index_s2p)
        for s, p in enumerate(self.index_s2p):
            self.index_p2s[p] = s

        self.ids_qpos = np.zeros(len(names),
                                 dtype=np.int64)  # joint position start index
        self.num_qpos = np.zeros(len(names),
                                 dtype=np.int64)  # joint position dim
        self.num_vel = np.zeros(len(names),
                                dtype=np.int64)  # joint velocity dim
        for i in range(len(names)):
            index = self.model.getJointId(names[i])
            if index >= self.model.njoints:
                raise ValueError(f"Invalid joint name: {name}")
            self.num_qpos[i] = self.model.nqs[index]
            self.num_vel[i] = self.model.nvs[index]
            self.ids_qpos[i] = self.model.idx_qs[index]

    def set_link_order(self, names):
        r"""Sets the order of links based on the provided link names.

        This function updates the internal representation of link indices
        based on the specified order. It creates an array that maps link names
        to their corresponding frame indices in the robot model.

        Args:
            names (list of str): A list of link names in the desired order.

        Raises:
            ValueError: If any provided link name is invalid (i.e., does not exist
                    in the robot model).
        """
        v = []
        for name in names:
            i = self.model.getFrameId(name, pin.BODY)
            if i >= self.model.nframes:
                raise ValueError(f"invalid joint name {name}")

            v.append(i)

        self.link_id_to_frame_index = np.array(v, dtype=np.int64)

    def q_p2s(self, qint) -> np.ndarray:
        r"""Converts internal joint position representation to external representation.

        Args:
            qint (numpy.ndarray): An array containing the internal joint positions.
                                The size of this array should match the number of
                                joints in the robot model.

        Returns:
            numpy.ndarray: An array containing the external joint positions (qext).
                        The size of this array is based on the number of velocity
                        variables in the robot model.

        Notes:
            For revolute joints with 2 degrees of freedom, the function calculates
            the angle using `arctan2` for proper angle representation.
        """
        qext = np.zeros(self.model.nv)
        count = 0
        for i in range(len(self.ids_qpos)):
            start_idx = self.ids_qpos[i]
            if self.num_qpos[i] == 1:
                qext[count] = qint[start_idx]
            elif self.num_qpos[i] == 2:
                qext[count] = np.arctan2(qint[start_idx + 1], qint[start_idx])
            elif self.num_qpos[i] > 2:
                raise ValueError(
                    f"Unsupported joint in computation. Currently support: fixed, revolute, prismatic"
                )
            count += self.num_vel[i]

        assert count == len(self.num_vel)
        return qext

    def q_s2p(self, qext) -> np.ndarray:
        r"""Converts external joint position representation back to internal representation.

        Args:
            qext (numpy.ndarray): An array containing the external joint positions.
                                The size of this array should match the number of
                                velocity variables in the robot model.

        Returns:
            numpy.ndarray: An array containing the internal joint positions (qint).
                        The size of this array is based on the number of joints
                        in the robot model.

        Notes:
            For revolute joints with 2 degrees of freedom, the function uses
            cosine and sine to compute the internal position representation from the external angle.
        """
        qint = np.zeros(self.model.nq)
        count = 0
        for i in range(len(self.ids_qpos)):
            start_idx = self.ids_qpos[i]
            if self.num_qpos[i] == 1:
                qint[start_idx] = qext[count]
            elif self.num_qpos[i] == 2:
                qint[start_idx] = np.cos(qext[count])
                qint[start_idx + 1] = np.sin(qext[count])
            elif self.num_qpos[i] > 2:
                raise ValueError(
                    "Unsupported joint in computation. Currently support: fixed, revolute, prismatic"
                )
            count += self.num_vel[i]

        assert count == len(self.num_vel)
        return qint

    def get_random_qpos(self) -> np.ndarray:
        return self.q_p2s(pin.randomConfiguration(self.model))

    def get_all_link_pose(self, qpos=np.ndarray) -> typing.Dict:
        r"""Computes and returns the poses of all links in the articulation base frame.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.

        Returns:
            Dict[int, np.ndarray]: A dictionary where the keys are link indices and the values
                               are 4x4 transformation matrices representing the poses of the links.
        """
        if not isinstance(qpos, np.ndarray) or qpos.ndim != 1:
            raise ValueError("qpos must be a 1D numpy array")
        self.compute_forward_kinematics(qpos)

        link_poses = {}

        # TODO: Link name as key
        # Retrieve the pose of each link
        for index in range(len(self.link_id_to_frame_index)):
            link_poses[index] = self.get_link_pose(index)

        return link_poses

    def get_link_pose(self, qpos: np.ndarray, link_name: str) -> np.ndarray:
        r"""Gets the pose of a specified link in the articulation base frame.

        This function computes the forward kinematics to get the pose of a specified
        link in the robot's base frame. It first converts the input joint positions 
        to PyTorch tensors, computes the forward kinematics for all links, and then 
        extracts the pose for the requested link.

        Args:
            qpos (np.ndarray): Joint positions array with shape (dof,). Must match
                the robot's degrees of freedom.
            link_name (str): Name of the link to get pose for. Must be a valid
                link name in the robot's kinematic chain.

        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix representing the 
                link's pose in the base frame, where:
                - The 3x3 upper-left block represents rotation (R)
                - The 3x1 upper-right block represents translation (t)
                - The bottom row is [0, 0, 0, 1]
                Returns identity matrix if computation fails.
        """
        qpos = np.array(qpos)
        if qpos.size != self.dof:
            rlia.utility.log_warning(
                f"qpos size {qpos.size} does not match robot DOF {self.dof}")
            return np.eye(4)

        self.compute_forward_kinematics(qpos)

        frame_index = self.model.getFrameId(link_name, pin.BODY)

        if 0 >= frame_index >= self.model.nframes:
            rlia.utility.log_warning(
                f"Link '{link_name}' not found in model. Available links: {self.frame_names}"
            )
            return np.eye(4)

        try:
            parent_joint = self.model.frames[frame_index].parent
            link2joint = self.model.frames[frame_index].placement
            joint2world = self.model_data.oMi[parent_joint]
            link2world = joint2world * link2joint
            p = link2world.translation
            q = link2world.rotation
            T = np.eye(4)
            T[:3, :3] = q
            T[:3, 3] = p
            return T

        except Exception as e:
            rlia.utility.log_warning(
                f"Failed to get pose for link '{link_name}': {str(e)}")
            return np.eye(4)

    def compute_full_jacobian(self, qpos: np.ndarray):
        r"""Compute and cache the Jacobian for all links.

        This function computes the Jacobian matrix for all links in
        the robot model based on the provided joint positions.
        The result is stored in the internal data structure for later retrieval.

        Args:
            qpos (np.ndarray): The joint positions as a 1D numpy array, which defines the configuration of the robot.
        """
        pin.computeJointJacobians(self.model, self.model_data,
                                  self.q_s2p(qpos))

    def get_link_jacobian(self, index: int, local: int = False) -> np.ndarray:
        r"""Given a link index, get the Jacobian. Must be called after compute_full_jacobian.

        This function retrieves the Jacobian matrix for a specified link in the robot model.
        The Jacobian can be expressed either in the world (spatial) frame or the link (body) frame,
        based on the `local` parameter.

        Args:
            index (int): The index of the link for which the Jacobian is to be retrieved.
            local (bool, optional): If True, the Jacobian is transformed to the world (spatial) frame;
                                    if False, it is expressed in the link (body) frame. Defaults to False.

        Raises:
            AssertionError: If the provided index is out of bounds (i.e., not valid for the link indices).

        Returns:
            np.ndarray: The Jacobian matrix corresponding to the specified link.
        """
        assert 0 <= index < len(self.link_id_to_frame_index)
        frame = int(self.link_id_to_frame_index[index])
        parent_joint = self.model.frames[frame].parent

        link2joint = self.model.frames[frame].placement
        joint2world = self.model_data.oMi[parent_joint]
        link2world = joint2world * link2joint

        J = pin.getJointJacobian(self.model, self.model_data, parent_joint,
                                 pin.ReferenceFrame.WORLD)
        if local:
            J = link2world.toActionMatrixInverse() @ J

        return J[:, self.index_s2p]

    def limit_robot_config(self, qpos_list: np.ndarray) -> np.ndarray:
        r"""Limit the robot configuration based on the elbow position.

        If the elbow is in the up position, it checks the positions of specific
        links to determine if the configuration is valid.

        Args:
            qpos_list (np.ndarray): The list of joint positions to be limited.

        Returns:
            np.ndarray: The limited list of joint positions if the elbow is up,
                        otherwise returns the original list.
        """
        if not self._is_elbow_up:
            return qpos_list

        limit_qpos_list = []
        for q in qpos_list:
            pin.forwardKinematics(self.model, self.model_data, q)

            link_xpos_list = []
            for i in range(self.model.nv):
                # Get the transformation matrix (SE3) of the i-th joint/link
                T = pin.SE3(self.model_data.oMi[i]
                            )  # oMi gives the pose of the i-th joint/link
                link_xpos_list.append(T)

            if (link_xpos_list[3].translation[2] <=
                    link_xpos_list[2].translation[2]):
                if (link_xpos_list[4].translation[2] <=
                        link_xpos_list[3].translation[2]):
                    limit_qpos_list.append(q)
            else:
                limit_qpos_list.append(q)

        limit_qpos_list = np.array(limit_qpos_list)
        return limit_qpos_list

    def compute_single_link_local_jacobian(self, qpos: np.ndarray,
                                           index: int) -> np.ndarray:
        r"""Compute the link (body) Jacobian for a single link.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.
            index (int): The index of the link for which to compute the Jacobian.

        Returns:
            np.ndarray: The Jacobian matrix for the specified link in its local frame.

        Raises:
            AssertionError: If the provided index is out of bounds (i.e., not valid for the link indices).
        """
        assert 0 <= index < len(self.link_id_to_frame_index)
        frame = int(self.link_id_to_frame_index[index])
        joint = self.model.frames[frame].parent
        link2joint = self.model.frames[frame].placement
        J = pin.computeJointJacobian(self.model, self.model_data,
                                     self.q_s2p(qpos), joint)
        J = link2joint.toActionMatrixInverse() @ J
        return J[:, self.index_s2p]

    def compute_generalized_mass_matrix(self, qpos: np.ndarray) -> np.ndarray:
        r"""Compute the generalized mass matrix.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.

        Returns:
            np.ndarray: The generalized mass matrix, a 2D array of shape [dof, dof].
        """
        return pin.crba(self.model, self.model_data,
                        self.q_s2p(qpos))[self.index_s2p, :][:, self.index_s2p]

    def compute_coriolis_matrix(self, qpos: np.ndarray,
                                qvel: np.ndarray) -> np.ndarray:
        r"""Compute the Coriolis matrix.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.
            qvel (np.ndarray): A 1D array of shape [dof,] representing the joint velocities.

        Returns:
            np.ndarray: The Coriolis matrix, a 2D array of shape [dof, dof].
        """
        return pin.computeCoriolisMatrix(
            self.model, self.model_data, self.q_s2p(qpos),
            qvel[self.index_p2s])[self.index_s2p, :][:, self.index_s2p]

    def compute_inverse_dynamics(self, qpos: np.ndarray, qvel: np.ndarray,
                                 qacc: np.ndarray) -> np.ndarray:
        r"""Compute the inverse dynamics.

        This function calculates the joint torques required to achieve the specified accelerations,
        given the current joint positions and velocities.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.
            qvel (np.ndarray): A 1D array of shape [dof,] representing the joint velocities.
            qacc (np.ndarray): A 1D array of shape [dof,] representing the joint accelerations.

        Returns:
            np.ndarray: The computed joint torques, a 1D array of shape [dof].
        """
        return pin.rnea(
            self.model,
            self.model_data,
            self.q_s2p(qpos),
            qvel[self.index_p2s],
            qacc[self.index_p2s],
        )[self.index_s2p]

    def compute_forward_dynamics(self, qpos: np.ndarray, qvel: np.ndarray,
                                 qf: np.ndarray) -> np.ndarray:
        r"""Compute the forward dynamics.

        This function calculates the joint accelerations resulting from the specified joint
        positions, velocities, and external forces.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.
            qvel (np.ndarray): A 1D array of shape [dof,] representing the joint velocities.
            qf (np.ndarray): A 1D array of shape [dof,] representing the external forces applied to the joints.

        Returns:
            np.ndarray: The computed joint accelerations, a 1D array of shape [dof].
        """
        return pin.aba(
            self.model,
            self.model_data,
            self.q_s2p(qpos),
            qvel[self.index_p2s],
            qf[self.index_p2s],
        )[self.index_s2p]

    def get_fk(self, qpos: np.ndarray, index: int = -1) -> np.ndarray:
        r"""Get the forward kinematics for a given joint state.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.
            index (int, optional): The index of the link for which to retrieve the pose.
                                Defaults to -1, which typically corresponds to the end-effector.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the pose of the specified link.
        """
        # Compute forward kinematics for the provided joint positions
        self.compute_forward_kinematics(qpos)

        if index == -1:
            # If index is -1, use the last link (end-effector)
            new_index = len(self.index_p2s)  # Get the last index
        else:
            frame_index = int(self.link_id_to_frame_index[index])
            joint_index = self.model.frames[frame_index].parent
            new_index = joint_index

        # Retrieve the pose of the specified link
        xpos_se3 = self.model_data.oMi.tolist()[new_index]

        xpos = np.eye(4)
        xpos[:3, :3] = xpos_se3.rotation
        xpos[:3, 3] = xpos_se3.translation.T
        return np.dot(xpos, self.tcp_xpos)

    def compute_forward_kinematics(self, qpos: np.ndarray):
        r"""Compute and cache forward kinematics.

        Args:
            qpos (np.ndarray): A 1D array of shape [dof,] representing the joint state.
        """
        # Update the model's forward kinematics with the joint positions
        pin.forwardKinematics(self.model, self.model_data, self.q_s2p(qpos))

    def compute_inverse_kinematics(
        self,
        target_pose: np.ndarray,
        joint_seed: np.ndarray,
        link_index: int = None,
        active_qmask=None,
    ) -> (bool, np.ndarray):
        r"""Computes the inverse kinematics for a given target pose.

        This function attempts to find a joint configuration that achieves the specified target pose
        using an iterative approach based on the CLIK (Constrained Linear Inverse Kinematics) algorithm.
        It utilizes forward kinematics and the Jacobian to adjust the joint positions iteratively until
        convergence is reached or the maximum number of iterations is exceeded.

        For more details, refer to:
        - CLIK algorithm documentation: https://gepettoweb.laas.fr/doc/stack-of-tasks/pin/master/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html
        - Bug fix information: https://github.com/stack-of-tasks/pin/pull/1963/files

        Args:
            target_pose (np.ndarray): The target pose represented as a 4x4 transformation matrix.
            joint_seed (np.ndarray): The initial joint positions used as a seed for the IK computation.
            link_index (int, optional): The index of the link for which to compute the IK.
                                        If None, defaults to the base link.
            active_qmask (np.ndarray, optional): A mask indicating which joints are active in the IK computation.
                                                 If None, all joints are considered.

        Raises:
            ValueError: If `target_pose` is not a 4x4 numpy array or if `joint_seed` is not a numpy array.

        Returns:
            Tuple[bool, np.ndarray]: A tuple containing:
                - A boolean indicating whether convergence to the desired pose was achieved.
                - The computed joint positions that correspond to the target pose,
                  or an empty array if convergence was not achieved.
        """
        if joint_seed is None:
            joint_seed = pin.neutral(self.model)

        active_qmask = (np.ones(self.model.nv) if active_qmask is None else
                        np.array(active_qmask)[self.index_p2s])
        # Apply the active mask to the joint seed
        mask = np.diag(active_qmask)

        link_index = -1 if link_index is None else link_index

        if 0 == len(self.link_id_to_frame_index):
            joint_index = self.model.nq
            oMdes = pin.SE3(target_pose)
            current_pose_se3 = self.model_data.oMi[joint_index]
        else:
            frame_index = int(self.link_id_to_frame_index[link_index])
            joint_index = self.model.frames[frame_index].parent

            l2w = pin.SE3()
            l2w.translation[:] = target_pose[:3, 3]
            l2w.rotation[:] = target_pose[:3, :3]
            l2j = self.model.frames[frame_index].placement
            oMdes = l2w * l2j.inverse()

        # Deep copy joint seed to avoid modifying the original seed
        q = deepcopy(joint_seed).astype(np.float64)

        for i in range(self._max_iterations):
            # Perform forward kinematics to compute the current pose
            pin.forwardKinematics(self.model, self.model_data, q)
            current_pose_se3 = self.model_data.oMi[joint_index]

            if self._is_only_position_constraint:
                # Fix the rotation part of the pose
                fixed_pose = np.eye(4)
                fixed_pose[:3, :3] = target_pose[:3, :3]  # Use target rotation
                fixed_pose[:3,
                           3] = current_pose_se3.translation.T  # Use current position
                fixed_pose_SE3 = pin.SE3(fixed_pose)
                current_pose_se3 = pin.SE3(fixed_pose_SE3)

            iMd = current_pose_se3.actInv(oMdes)  # Calculate the pose error
            err = pin.log6(iMd).vector  # Get the error vector

            # Check position convergence
            pos_converged = np.linalg.norm(err[:3]) < self._pos_eps

            if self._is_only_position_constraint:
                if pos_converged:
                    # Convergence achieved, apply joint limits
                    q = self.qpos_to_limits(q, joint_seed, active_qmask)
                    if 0 == len(q):
                        return False, []
                    return True, self.q_p2s(q)
            else:
                # Check rotation convergence
                rot_converged = np.linalg.norm(err[3:]) < self._rot_eps

                # Check for overall convergence
                if pos_converged and rot_converged:
                    # Convergence achieved, apply joint limits
                    q = self.qpos_to_limits(q, joint_seed, active_qmask)
                    if 0 == len(q):
                        return False, []
                    return True, self.q_p2s(q)

            # Compute the Jacobian
            J = pin.computeJointJacobian(self.model, self.model_data, q,
                                         joint_index)
            Jlog = pin.Jlog6(iMd.inverse())
            J = -Jlog @ J
            J = J @ mask

            # Damped least squares
            JJt = J @ J.T
            JJt[np.diag_indices_from(JJt)] += self._damp
            # Compute the velocity update
            v = -(J.T @ np.linalg.solve(JJt, err))

            # Update joint positions
            new_q = pin.integrate(self.model, q, v * self._dt)
            q[active_qmask.astype(bool)] = new_q[active_qmask.astype(bool)]

        return False, self.q_p2s(
            q)  # Return failure and the last computed joint positions

    def get_ik(
        self,
        target_pose: np.ndarray,
        joint_seed: np.ndarray,
        link_index: int = None,
        active_qmask: np.ndarray = None,
        num_samples: int = None,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> (bool, np.ndarray):
        r"""Computes the inverse kinematics for a given target pose.

        This function generates random joint configurations within the specified limits,
        including the provided joint_seed, and attempts to find valid inverse kinematics solutions.
        It then identifies the joint position that is closest to the joint_seed.

        Args:
            target_pose (np.ndarray): The target pose represented as a 4x4 transformation matrix.
            joint_seed (np.ndarray): The initial joint positions used as a seed, providing a reference for the solution.
            link_index (int, optional): An index that specifies which link of the robot to consider for the IK computation.
            active_qmask (np.ndarray, optional): A mask indicating which joints are active for the computation;
                                                 if None, all joints are considered.
            num_samples (int, optional): Number of samples, must be positive.
            return_all_solutions (bool, optional): If True, return all IK results. If False, return the first IK result.
                                        Defaults to False.

            **kwargs: Additional arguments for future extensions.

        Returns:
            Tuple[bool, np.ndarray]: A tuple containing:
                - A boolean indicating whether a valid solution was found (True) or not (False).
                - The closest joint position to the joint_seed as a numpy array,
                  or an empty array if no valid solutions were found.

        Notes:
            - The function samples multiple random joint configurations and evaluates them to find a suitable solution.
            - If no valid configurations are found, warnings are logged to provide feedback on the failure.
            - The closest joint configuration to the provided joint_seed is returned if a solution exists.
        """
        target_xpos = np.dot(target_pose, np.linalg.inv(self.tcp_xpos))

        if joint_seed is None:
            joint_seed = pin.neutral(self.model)

        joint_seed = self.q_s2p(joint_seed)

        if num_samples is not None:
            self._num_samples = num_samples

        # Generate random joint configurations
        random_joint_seeds = np.empty((self._num_samples, len(joint_seed)))
        random_joint_seeds[0] = joint_seed

        if active_qmask is None:
            mask = np.ones(self.model.nv)
        else:
            mask = np.array(active_qmask)

        active_qmask = np.array(mask, dtype=bool)

        # Generate random configurations based on active_qmask
        upper_plimits = self.upper_position_limits
        lower_plimits = self.lower_position_limits
        for i in range(1, self._num_samples):
            random_joint_seeds[i] = np.random.uniform(upper_plimits,
                                                      lower_plimits,
                                                      size=len(joint_seed))

            # Apply active_qmask
            if mask is not None:
                for j in range(len(joint_seed)):
                    if mask[j] == 0:
                        random_joint_seeds[i][j] = joint_seed[j]

        valid_qpos_list = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.compute_inverse_kinematics,
                    target_xpos,
                    q,
                    link_index,
                    active_qmask,
                ): q
                for q in random_joint_seeds
            }
            for future in concurrent.futures.as_completed(futures):
                res, qpos = future.result()
                if res:
                    valid_qpos_list.append(qpos)

        # Check for valid solutions
        if not valid_qpos_list:
            rlia.utility.log_warning(
                "Pin: The iterative algorithm has not reached convergence to the desired precision."
            )
            return False, []

        limited_qpos_array = self.limit_robot_config(np.array(valid_qpos_list))

        if 0 == len(limited_qpos_array):
            rlia.utility.log_warning(
                "Pin:It is estimated that none of the axis configurations are met, elbow_up enable: {}"
                .format(self._is_elbow_up))
            return False, []

        # Calculate the distances to joint_seed
        joint_seed = self.q_p2s(joint_seed)

        weighted_distances = np.linalg.norm(
            (limited_qpos_array[:, active_qmask] - joint_seed[active_qmask]) *
            self.ik_nearst_weight[active_qmask],
            axis=1)

        if return_all_solutions:
            # Sort the solutions by weighted distances
            sorted_indices = np.argsort(weighted_distances)
            sorted_qpos_array = limited_qpos_array[sorted_indices]

            return True, sorted_qpos_array
        else:
            # Find the index of the closest solution
            closest_index = np.argmin(weighted_distances)

            # Return the closest joint position
            closest_qpos = limited_qpos_array[closest_index]

            return True, closest_qpos
