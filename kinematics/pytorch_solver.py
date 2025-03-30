import sys
import itertools
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from os import devnull
import logging

import numpy as np
import torch
from solver import Solver

try:
    if sys.platform != "win32":
        import pytorch_kinematics as pk
except ImportError:
    raise ImportError(
        "pytorch_kinematics_ms not installed. Install with `pip install pytorch_kinematics_ms`"
    )

__all__ = ["PytorchSolver"]

logger = logging.getLogger(__name__)


class PytorchSolver(Solver):
    def __init__(
        self,
        urdf_path: str,
        end_link_name: str,
        **kwargs,
    ):
        r"""Initializes the PyTorch kinematics solver.

            This constructor sets up the kinematics solver using PyTorch,
            allowing for efficient computation of robot kinematics based on
            the specified URDF model.

        Args:
            urdf_path (str, optional): Path to the robot's URDF file.
            end_link_name (str): The name of the end-effector link.
            **kwargs: Additional keyword arguments passed to the base solver.

        """

        super().__init__(
            urdf_path=urdf_path,
            end_link_name=end_link_name,
            **kwargs,
        )

        self.device = kwargs.get(
            "device",
            torch.device("cuda:0")
            if torch.cuda.is_available() else torch.device("cpu"),
        )

        self.root_link_name = kwargs.get("root_link_name", None)

        with open(self.urdf_path, "rb") as f:
            urdf_str = f.read()

        # NOTE It seems that the pk library currently always outputs some complaints if there are unknown attributes in a URDF. Hide it with this contextmanager here.
        @contextmanager
        def suppress_stdout_stderr():
            """A context manager that redirects stdout and stderr to devnull"""
            with open(devnull, "w") as fnull:
                with redirect_stderr(fnull) as err, redirect_stdout(
                        fnull) as out:
                    yield (err, out)

        with suppress_stdout_stderr():
            if self.root_link_name is None:
                self.pk_chain = pk.build_serial_chain_from_urdf(
                    urdf_str,
                    end_link_name=self.end_link_name,
                ).to(device=self.device)
            else:
                self.pk_chain = pk.build_serial_chain_from_urdf(
                    urdf_str,
                    end_link_name=self.end_link_name,
                    root_link_name=self.root_link_name,
                ).to(device=self.device)

        # Get agent joint limits.
        self.lim = torch.tensor(self.pk_chain.get_joint_limits(),
                                device=self.device)

        # Inverse kinematics is available via damped least squares (iterative steps with Jacobian pseudo-inverse damped to avoid oscillation near singularlities).
        self.pik = pk.PseudoInverseIK(
            self.pk_chain,
            pos_tolerance=self._pos_eps,
            rot_tolerance=self._rot_eps,
            joint_limits=self.lim.T,
            early_stopping_any_converged=True,
            max_iterations=self._max_iterations,
            lr=self._dt,
            num_retries=1,
        )

        self.dof = self.pk_chain.n_joints
        self.ik_nearst_weight = torch.ones(self.dof)

        self.upper_position_limits = self.pk_chain.high
        self.lower_position_limits = self.pk_chain.low

    def get_link_names(self) -> dict:
        return self.pk_chain.get_link_names()

    def set_iteration_params(
        self,
        pos_eps: float = 5e-4,
        rot_eps: float = 5e-4,
        max_iterations: int = 1000,
        dt: float = 0.1,
        damp: float = 1e-6,
        num_samples: int = 30,
        is_only_position_constraint: bool = False,
    ) -> bool:
        """Sets the iteration parameters for the kinematics solver.

        Args:
            pos_eps (float): Pos convergence threshold, must be positive.
            rot_eps (float): Rot convergence threshold, must be positive.
            max_iterations (int): Maximum number of iterations, must be positive.
            dt (float): Time step size, must be positive.
            damp (float): Damping factor, must be non-negative.
            num_samples (int): Number of samples, must be positive.
            is_only_position_constraint (bool):

        Returns:
            bool: True if all parameters are valid and set, False otherwise.
        """
        res = super().set_iteration_params(
            pos_eps=pos_eps,
            rot_eps=rot_eps,
            max_iterations=max_iterations,
            dt=dt,
            damp=damp,
            num_samples=num_samples,
            is_only_position_constraint=is_only_position_constraint,
        )

        if res:
            self.pik = pk.PseudoInverseIK(
                self.pk_chain,
                pos_tolerance=self._pos_eps,
                rot_tolerance=self._rot_eps,
                joint_limits=self.lim.T,
                early_stopping_any_converged=True,
                max_iterations=self._max_iterations,
                lr=self._dt,
                num_retries=1,
            )

        return res

    def limit_robot_config(self, qpos_list: torch.tensor) -> np.ndarray:
        r"""Limit the robot configuration based on the elbow position.

        If the elbow is in the up position, it checks the positions of specific
        links to determine if the configuration is valid.

        Args:
            qpos_list (torch.tensor): The list of joint positions to be limited.

        Returns:
            np.ndarray: The limited list of joint positions if the elbow is up,
                        otherwise returns the original list.
        """
        if self._is_elbow_up:
            return qpos_list

        def process_qpos(q):
            ret = self.pk_chain.forward_kinematics(q, end_only=False)
            link_xpos_list = list(ret.values())

            # Extract the z positions of specific links
            link_1_z = link_xpos_list[2].get_matrix()[:, 2, 3]
            link_2_z = link_xpos_list[3].get_matrix()[:, 2, 3]
            link_3_z = link_xpos_list[4].get_matrix()[:, 2, 3]

            if link_2_z <= link_1_z:
                if link_3_z <= link_2_z:
                    return q
            else:
                return q

            return None

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_qpos, qpos_list))

        limit_qpos_list = [q for q in results if q is not None]

        return (torch.stack(limit_qpos_list)
                if limit_qpos_list else torch.empty((0, ), device=self.device))

    @staticmethod
    def _qpos_to_limits_single(
        q: torch.Tensor,
        joint_seed: torch.Tensor,
        lower_position_limits: torch.Tensor,
        upper_position_limits: torch.Tensor,
        ik_nearst_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adjusts the given joint positions (q) to fit within the specified limits while minimizing the difference to the seed position.

        Args:
            q (torch.Tensor): The initial joint positions.
            joint_seed (torch.Tensor): The seed joint positions for comparison.
            lower_position_limits (torch.Tensor): The lower bounds for the joint positions.
            upper_position_limits (torch.Tensor): The upper bounds for the joint positions.
            ik_nearst_weight (torch.Tensor): The weights for the inverse kinematics nearest calculation.

        Returns:
            torch.Tensor: The adjusted joint positions that fit within the limits.
        """
        device = q.device
        joint_seed = joint_seed.to(device)
        lower_position_limits = lower_position_limits.to(device)
        upper_position_limits = upper_position_limits.to(device)
        ik_nearst_weight = ik_nearst_weight.to(device)

        best_qpos_limit = q.clone()
        best_total_q_diff = float("inf")

        # Generate possible values for each joint
        possible_arrays = [[
            q[i] + offset * (2 * torch.pi) for offset in range(-5, 6)
            if lower_position_limits[i] <= q[i] + offset *
            (2 * torch.pi) <= upper_position_limits[i]
        ] for i in range(q.size(0))]

        if any(not values for values in possible_arrays):
            return torch.tensor([]).to(device)

        # Create all possible combinations of joint values
        all_possible_combinations = itertools.product(*possible_arrays)
        for combination in all_possible_combinations:
            combination_tensor = torch.tensor(combination).to(device)
            total_q_diff = torch.sum(
                torch.abs(combination_tensor - joint_seed) * ik_nearst_weight)

            # Update the best solution if a smaller total difference is found
            if total_q_diff < best_total_q_diff:
                best_total_q_diff = total_q_diff
                best_qpos_limit = combination_tensor

        return best_qpos_limit

    def qpos_to_limits(self, qpos_list_split: torch.Tensor,
                       joint_seed: torch.Tensor) -> torch.Tensor:
        r"""Adjusts a list of joint positions to fit within the specified limits for each joint.

        Args:
            qpos_list_split (torch.Tensor): The list of joint positions to be adjusted.
            joint_seed (torch.Tensor): The seed joint positions for comparison.

        Returns:
            torch.Tensor: The adjusted list of joint positions.
        """
        if self.ik_nearst_weight is None:
            self.ik_nearst_weight = torch.ones_like(qpos_list_split[-1],
                                                    device=self.device)
        else:
            self.ik_nearst_weight = (self.ik_nearst_weight.clone().detach().to(
                self.device))

        adjusted_qpos_list = []
        # Use a ThreadPoolExecutor to parallelize the processing
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._qpos_to_limits_single,
                    q,
                    joint_seed,
                    self.lower_position_limits,
                    self.upper_position_limits,
                    self.ik_nearst_weight,
                ) for q in qpos_list_split
            ]

            for future in as_completed(futures):
                result = future.result()
                if result.numel() > 0:
                    adjusted_qpos_list.append(result)

        return (torch.stack(adjusted_qpos_list).to(qpos_list_split.device) if
                adjusted_qpos_list else torch.tensor([], device=self.device))

    def get_ik(
        self,
        target_pose: torch.Tensor,
        joint_seed: torch.Tensor,
        num_samples: int = None,
        return_all_solutions: bool = False,
        **kwargs,
    ):
        r"""Computes the inverse kinematics for given target poses.

        This function generates random joint configurations within the specified limits,
        including the provided joint_seed, and attempts to find valid inverse kinematics solutions.
        It then identifies the joint positions that are closest to the joint_seed.

        Args:
            target_pose (torch.Tensor): The target poses represented as a (batch_size, 4, 4) tensor or a single 4x4 transformation matrix.
            joint_seed (torch.Tensor): The initial joint positions used as a seed. It can be either a 1D tensor of shape (dof,) or a 2D tensor of shape (batch_size, dof).
            num_samples (int, optional): Number of samples, must be positive.
            return_all_solutions (bool, optional): If True, return all IK results. If False, return the first IK result.
                                        Defaults to False.
            **kwargs: Additional arguments for future extensions.

        Returns:
            Tuple[bool, np.ndarray]: A tuple containing:
                - A boolean indicating whether a valid solution was found for all target poses.
                - A list of the closest joint positions to the joint_seed for each target pose, or an empty list if no valid solutions were found.
        """
        # Check dimensions of target_pose
        target_pose = torch.as_tensor(target_pose)

        if target_pose.dim() == 2:
            assert target_pose.shape == (
                4,
                4,
            ), "`target_pose` must be of shape (4, 4) or (n, 4, 4)."
            target_pose = target_pose.unsqueeze(0)
        elif target_pose.dim() == 3:
            assert target_pose.shape[1:] == (
                4,
                4,
            ), "`target_pose` must be of shape (4, 4) or (n, 4, 4)."
        else:
            raise ValueError(
                "`target_pose` must be a tensor of shape (4, 4) or (n, 4, 4).")

        if num_samples is not None:
            self._num_samples = num_samples

        if joint_seed is None:
            joint_seed = torch.zeros(self.dof)
        else:
            joint_seed = torch.as_tensor(joint_seed)

        # Check dimensions of joint_seed
        if joint_seed.dim() == 1:
            joint_seed = joint_seed.unsqueeze(0)
            joint_seed_ndim = 1
        elif joint_seed.dim() == 2:
            joint_seed_ndim = 2
            if len(joint_seed) != len(target_pose):
                raise ValueError(
                    "Batch size of joint_seed must match batch size of target_pose when joint_seed is a 2D tensor."
                )
        else:
            raise ValueError(
                "`joint_seed` must be a tensor of shape (n,) or (n, n).")
        tcp_xpos = deepcopy(self.tcp_xpos)
        tcp_xpos = torch.as_tensor(tcp_xpos)
        tcp_xpos = tcp_xpos.to(self.device).float()

        target_pose = target_pose.to(self.device).float()
        target_pose = target_pose @ torch.inverse(tcp_xpos)
        joint_seed = joint_seed.to(self.device).float()

        # Get qpos limits
        upper_limits = self.upper_position_limits.float()
        lower_limits = self.lower_position_limits.float()

        batch_size = target_pose.shape[0]
        dof = joint_seed.shape[1]
        # num_samples = self._num_samples * batch_size

        random_joint_seeds_part = lower_limits + (
            upper_limits - lower_limits) * torch.rand(
                (self._num_samples - 3, dof), device=self.device)

        # Initialize random_joint_seeds as an empty tensor
        random_joint_seeds = torch.empty((0, dof), device=self.device)
        target_pose_repeated = torch.empty((0, 4, 4), device=self.device)

        # Handle different dimensions of joint_seed and iterate over target_pose
        for i in range(batch_size):
            current_joint_seed = (joint_seed[i].unsqueeze(0)
                                  if joint_seed_ndim == 2 else joint_seed)
            joint_seeds = torch.vstack([
                current_joint_seed,
                lower_limits.unsqueeze(0),
                random_joint_seeds_part,
                upper_limits.unsqueeze(0),
            ])
            random_joint_seeds = torch.vstack(
                [random_joint_seeds, joint_seeds])
            current_target_pose = (target_pose[i].unsqueeze(0).repeat(
                self._num_samples, 1, 1))
            target_pose_repeated = torch.vstack(
                [target_pose_repeated, current_target_pose])

        res_list, qpos_list = self.compute_inverse_kinematics(
            target_pose_repeated, random_joint_seeds)

        # Split res_list and qpos_list according to self._num_samples
        res_list_split = torch.split(res_list, self._num_samples)
        qpos_list_split = torch.split(qpos_list, self._num_samples)

        # Initialize the final results and the closest joint positions
        final_results = []
        final_qpos = []

        for i in range(batch_size):
            target_joint_seed = (joint_seed[i]
                                 if joint_seed_ndim == 2 else joint_seed)

            if not res_list_split[i].any():
                final_results.append(False)
                final_qpos.append(torch.empty(0, device=self.device))
                continue

            result_qpos_limit = self.qpos_to_limits(qpos_list_split[i],
                                                    target_joint_seed)

            limited_qpos_t = self.limit_robot_config(result_qpos_limit)

            if len(limited_qpos_t) == 0:
                logger.warning(
                    "Pk: It is estimated that none of the axis configurations are met, elbow_up enable: {}"
                    .format(self._is_elbow_up))
                final_results.append(False)
                final_qpos.append(torch.empty(0, device=self.device))
                continue

            distances = torch.norm(limited_qpos_t - target_joint_seed, dim=1)
            if return_all_solutions:
                # Sort the solutions by distances
                sorted_indices = torch.argsort(distances)
                sorted_qpos_array = limited_qpos_t[sorted_indices]
                final_qpos.append(sorted_qpos_array)
            else:
                # Find the index of the closest solution
                closest_index = torch.argmin(distances)
                # Return the closest joint position
                closest_qpos = limited_qpos_t[closest_index]
                final_qpos.append(closest_qpos)
            final_results.append(True)

        # Check if all elements in final_results are True
        all_true = all(final_results)

        return all_true, final_qpos

    def compute_inverse_kinematics(
            self, target_pose: torch.Tensor,
            joint_seed: torch.Tensor) -> typing.Tuple[bool, np.ndarray]:
        r"""Computes the inverse kinematics solutions for the given target poses and joint seeds.

        Args:
            target_pose (torch.Tensor): The target poses represented as a (batch_size, 4, 4) tensor.
            joint_seed (torch.Tensor): The initial joint positions used as a seed. It can be either a 1D tensor of shape (dof,) or a 2D tensor of shape (batch_size, dof).

        Returns:
            Tuple[bool, torch.Tensor]: A tuple containing:
                - A boolean indicating whether any valid solution was found.
                - The solutions tensor of shape (batch_size, dof) if a solution is found, otherwise an empty tensor.
        """
        target_pose = target_pose.to(self.device).float()
        joint_seed = joint_seed.to(self.device).float()

        # Extract translation and rotation parts
        pos = target_pose[:, :3, 3]
        rot = target_pose[:, :3, :3]

        tf = pk.Transform3d(
            pos=pos,
            rot=rot,
            device=self.device,
        )
        self.pik.initial_config = joint_seed

        result = self.pik.solve(tf)

        if result.converged_any.any().item():
            return result.converged_any, result.solutions[:, 0, :].squeeze(0)

        return False, torch.empty(0)

    def get_link_pose(self, qpos: np.ndarray, link_name: str) -> np.ndarray:
        r"""Gets the pose of a specified link in the articulation base frame.

        This function computes the pose of the specified link using forward kinematics.
        It first updates the robot's kinematic state using the provided joint positions,
        then retrieves the pose of the requested link in the world frame.

        Args:
            qpos (np.ndarray): Joint positions array with shape (dof,). Must match
                            the robot's degrees of freedom.
            link_name (str): Name of the link to get pose for. Must be a valid
                            link name in the robot model.

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
            logger.warning(
                f"qpos size {qpos.size} does not match robot DOF {self.dof}")
            return np.eye(4)

        qpos_t = torch.from_numpy(qpos).float().to(self.device)

        try:
            ret = self.pk_chain.forward_kinematics(qpos_t, end_only=False)
        except Exception as e:
            raise RuntimeError(f"Forward kinematics failed: {str(e)}")

        if link_name not in ret:
            logger.warning(
                f"Link '{link_name}' not found. Available links: {list(ret.keys())}"
            )
            return np.eye(4)

        link_pose = ret[link_name]

        pose_m = link_pose.get_matrix()

        return pose_m.cpu().numpy()

    def get_fk(self, qpos: torch.tensor) -> torch.tensor:
        r"""Get the forward kinematics for the end link.

        Args:
            qpos (torch.Tensor): The joint positions.

        Returns:
            torch.Tensor: A 4x4 homogeneous transformation matrix representing the pose of the end link.
        """
        tcp_xpos = deepcopy(self.tcp_xpos)
        tcp_xpos = torch.as_tensor(tcp_xpos)
        tcp_xpos = tcp_xpos.to(self.device).float()

        if self.end_link_name is None:
            return self.compute_forward_kinematics(qpos) @ tcp_xpos
        else:
            return (self.compute_forward_kinematics(
                qpos, link_name=self.end_link_name) @ tcp_xpos)

    def get_all_fk(self, qpos: torch.tensor) -> torch.tensor:
        r"""Get the forward kinematics for all links from root to end link.

        Args:
            qpos (torch.Tensor): The joint positions.

        Returns:
            list: A list of 4x4 homogeneous transformation matrices representing the poses of all links from root to end link.
        """
        qpos = torch.as_tensor(qpos)
        qpos = qpos.to(self.device)

        ret = self.pk_chain.forward_kinematics(qpos, end_only=False)
        link_names = list(ret.keys())

        if self.root_link_name is not None:
            try:
                start_index = link_names.index(self.root_link_name)
            except ValueError:
                raise KeyError(
                    f"Root link name '{self.root_link_name}' not found in the kinematic chain"
                )
        else:
            start_index = 0

        if self.end_link_name is not None:
            try:
                end_index = link_names.index(self.end_link_name) + 1
            except ValueError:
                raise KeyError(
                    f"End link name '{self.end_link_name}' not found in the kinematic chain"
                )
        else:
            end_index = len(link_names)

        poses = []
        for link_name in link_names[start_index:end_index]:
            xpos = ret[link_name]
            if not hasattr(xpos, "get_matrix"):
                raise AttributeError(
                    f"The result for link '{link_name}' must have 'get_matrix' attributes."
                )
            xpos_t = torch.eye(4, device=xpos.get_matrix().device)
            m = xpos.get_matrix()
            xpos_t[:3, 3] = m[:, :3, 3]
            xpos_t[:3, :3] = m[:, :3, :3]
            poses.append(xpos_t)

        return poses

    def compute_forward_kinematics(self,
                                   qpos: torch.tensor,
                                   link_name=None) -> torch.tensor:
        r"""Computes the forward kinematics for the given joint positions.

        Args:
            qpos (torch.Tensor): The joint positions.
            link_name (str, optional): The name of the link for which to compute the forward kinematics.
                                       If None, computes for the end-effector.

        Returns:
            torch.Tensor: A 4x4 homogeneous transformation matrix representing the pose of the specified link or end-effector.
        """
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.tensor(qpos, dtype=torch.float)
        qpos = qpos.to(self.device)

        if link_name is None:
            xpos = list(self.pk_chain.forward_kinematics(qpos,
                                                         end_only=True))[-1]
        else:
            xpos = self.pk_chain.forward_kinematics(
                qpos, end_only=False)[link_name][-1]

        if not hasattr(xpos, "get_matrix"):
            logger.warning(
                "Get FK failed, the result from forward_kinematics must have 'get_matrix' attributes."
            )
            return torch.eye(4, device=xpos.device)

        xpos_t = torch.eye(4, device=xpos.device)
        m = xpos.get_matrix()
        xpos_t[:3, 3] = m[:, :3, 3]
        xpos_t[:3, :3] = m[:, :3, :3]

        return xpos_t

    def get_jacobian(self, qpos: torch.tensor) -> torch.tensor:
        r"""Compute the Jacobian matrix for the given joint positions.

        Args:
            qpos (torch.Tensor): The joint positions.

        Returns:
            torch.Tensor: The Jacobian matrix.
        """
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.tensor(qpos, dtype=torch.float)
        qpos = qpos.to(self.device)

        J = self.pk_chain.jacobian(qpos)
        return J
