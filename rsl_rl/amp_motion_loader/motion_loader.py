# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import yaml
import os
import torch
from typing import Optional


class AMPTocabiMotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, num_amp_obs_steps: int, batch_dt: float, device: torch.device) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        self.device = device
        self.num_amp_obs_steps = num_amp_obs_steps
        self.num_amp_obs = num_amp_obs_steps * (4 + 12 + 12 + 6)
        
        self._motion_lengths_s = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_dt = []
        self._motions = []
        self._motion_joint_pos = []
        self._motion_joint_vel = []
        self._motion_base_pos = []
        self._motion_base_rot = []
        self._motion_base_lin_vel = []
        self._motion_base_ang_vel = []
        self._motion_key_body_pos = []
        self.batch_dt = batch_dt

        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        motion_files, motion_weight, clip_motion_ranges = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print(f"Loading motion file {f}/{num_motion_files}: {curr_file}")
            curr_data = np.loadtxt(curr_file)
            motion_hz = 1.0 / (curr_data[1,0] - curr_data[0,0])
            motion_dt = curr_data[1,0] - curr_data[0,0]
            curr_data = curr_data[clip_motion_ranges[f][0]:clip_motion_ranges[f][1], :]
            print(f"Motion length: {(curr_data.shape[0]-1)*motion_dt} sec")
            self._motion_lengths_s.append((curr_data.shape[0]-1)*motion_dt)
            self._motion_lengths.append(curr_data.shape[0])
            self._motion_weights.append(motion_weight[f])
            self._motion_dt.append(motion_dt)
            self._motions.append(curr_data)
            self._motion_joint_pos.append(curr_data[:,1:13])
            self._motion_joint_vel.append(curr_data[:,13:25])
            self._motion_base_pos.append(curr_data[:,25:28])
            self._motion_base_rot.append(curr_data[:,28:32])
            self._motion_base_lin_vel.append(curr_data[:,32:35])
            self._motion_base_ang_vel.append(curr_data[:,35:38])
            self._motion_key_body_pos.append(curr_data[:,38:])
        
        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_lengths_s = np.array(self._motion_lengths_s)
        self._motion_weights = np.array(self._motion_weights)
        self._motion_weights /= np.sum(self._motion_weights)
        self._motion_dt = np.array(self._motion_dt)

        print(f"Loaded {len(self._motions)} motions with a total duration of {sum(self._motion_lengths_s)} seconds")


        self._dof_names = ["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                           "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"]
        # self._body_names = data["body_names"].tolist()

        # self.dof_positions = torch.tensor(data[:,1:13], dtype=torch.float32, device=self.device)
        # self.dof_velocities = torch.tensor(data[:,13:25], dtype=torch.float32, device=self.device)
        # self.root_positions = torch.tensor(data[:,25:28], dtype=torch.float32, device=self.device)
        # self.root_rotations = torch.tensor(data[:,28:32], dtype=torch.float32, device=self.device)
        # self.root_linear_velocities = torch.tensor(data[:,32:35], dtype=torch.float32, device=self.device)
        # self.root_angular_velocities = torch.tensor(data[:,35:38], dtype=torch.float32, device=self.device)
        # self.key_body_positions = torch.tensor(data[:,38:], dtype=torch.float32, device=self.device)

        # self.dt = 1.0 / 2000
        # self.num_frames = self.dof_positions.shape[0]
        # self.duration = self.dt * (self.num_frames - 1)
        # print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []
            clip_motion_range = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                curr_clip_motion_range = motion_entry['clip_motion_range']
                assert curr_weight >= 0
                assert len(curr_clip_motion_range) == 2, f"Invalid clip motion range: {curr_clip_motion_range}"
                assert curr_clip_motion_range[0] < curr_clip_motion_range[1], f"Invalid clip motion range: {curr_clip_motion_range}"

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
                clip_motion_range.append(curr_clip_motion_range)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
            clip_motion_range = [[0, -1]]

        return motion_files, motion_weights, clip_motion_range
    
    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)
       
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            motion_ids = self._sample_motion_ids(mini_batch_size)
            motion_times_0 = self._sample_times(motion_ids)

            # repeat motion_ids and make size (mini_batch_size, num_step_amp_obs)
            motion_ids_steps = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self.num_amp_obs_steps])
            motion_times_steps = np.expand_dims(motion_times_0, axis=-1)
            time_steps_sim_dt = -self.batch_dt * np.arange(0, self.num_amp_obs_steps)
            motion_times_steps = motion_times_steps + time_steps_sim_dt

            # flatten motion_ids_steps and motion_times_steps to get motion steps
            motion_ids_steps = motion_ids_steps.flatten()
            motion_times_steps = motion_times_steps.flatten()

            # sample motion data
            joint_pos, joint_vel, base_pos, base_rot, base_lin_vel, base_ang_vel, key_body_pos = self.sample(motion_ids_steps, motion_times_steps)

            # build root states
            root_states = torch.cat((base_pos, base_rot, base_lin_vel, base_ang_vel), dim=-1)

            # build amp obs demo buffer
            amb_obs_demo_buffer_ = self._build_amp_observations(root_states, joint_pos, joint_vel, key_body_pos)
            amb_obs_demo_buffer = amb_obs_demo_buffer_.view(-1, self.num_amp_obs)
            yield amb_obs_demo_buffer

    def _sample_times(self, motion_ids: np.ndarray) -> np.ndarray:
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)
        motions_len = self._motion_lengths[motion_ids]
        return motions_len * phase

    def _sample_motion_ids(self, num_samples: int) -> np.ndarray:
        return np.random.choice(len(self._motions), size=num_samples, replace=True, p=self._motion_weights)
    
    def sample(self, motion_ids: np.ndarray, motion_times: np.ndarray
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.
        """
        n = len(motion_ids)
        joint_pos = torch.zeros((n, 12), dtype=torch.float32, device=self.device)
        joint_vel = torch.zeros((n, 12), dtype=torch.float32, device=self.device)
        base_pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        base_rot = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        base_lin_vel = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        base_ang_vel = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        key_body_pos = torch.zeros((n, 6), dtype=torch.float32, device=self.device)

        motions_len = self._motion_lengths[motion_ids]
        motions_len_s = self._motion_lengths_s[motion_ids]
        motions_dt = self._motion_dt[motion_ids]

        index_0, index_1, blend = self._compute_frame_blend(motion_times, motions_len, motions_len_s, motions_dt)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)
        
        unique_ids = np.unique(motion_ids)
        for uid in unique_ids:
            ids = np.where(motion_ids == uid)
            # curr_motion = self._motions[uid]

            joint_pos_tensor = torch.tensor(self._motion_joint_pos[uid], dtype=torch.float32, device=self.device)
            joint_vel_tensor = torch.tensor(self._motion_joint_vel[uid], dtype=torch.float32, device=self.device)
            base_pos_tensor = torch.tensor(self._motion_base_pos[uid], dtype=torch.float32, device=self.device)
            base_rot_tensor = torch.tensor(self._motion_base_rot[uid], dtype=torch.float32, device=self.device)
            base_lin_vel_tensor = torch.tensor(self._motion_base_lin_vel[uid], dtype=torch.float32, device=self.device)
            base_ang_vel_tensor = torch.tensor(self._motion_base_ang_vel[uid], dtype=torch.float32, device=self.device)
            key_body_pos_tensor = torch.tensor(self._motion_key_body_pos[uid], dtype=torch.float32, device=self.device)

            joint_pos[ids] = self._interpolate(joint_pos_tensor, blend=blend[ids], start=index_0[ids], end=index_1[ids])
            joint_vel[ids] = self._interpolate(joint_vel_tensor, blend=blend[ids], start=index_0[ids], end=index_1[ids])
            base_pos[ids] = self._interpolate(base_pos_tensor, blend=blend[ids], start=index_0[ids], end=index_1[ids])
            base_rot[ids] = self._slerp(base_rot_tensor, blend=blend[ids], start=index_0[ids], end=index_1[ids])
            base_lin_vel[ids] = self._interpolate(base_lin_vel_tensor, blend=blend[ids], start=index_0[ids], end=index_1[ids])
            base_ang_vel[ids] = self._interpolate(base_ang_vel_tensor, blend=blend[ids], start=index_0[ids], end=index_1[ids])
            key_body_pos[ids] = self._interpolate(key_body_pos_tensor, blend=blend[ids], start=index_0[ids], end=index_1[ids])
        
        return joint_pos, joint_vel, base_pos, base_rot, base_lin_vel, base_ang_vel, key_body_pos

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        # qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        qw, qx, qy, qz = 3, 0, 1, 2  # xyzw
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray, duration: np.ndarray, num_frames: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase = np.clip(times / duration, 0.0, 1.0)
        index_0 = (phase * (num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, num_frames - 1)
        blend = ((times - index_0 * dt) / dt).round(decimals=5)
        return index_0, index_1, blend
    
    def _build_amp_observations(self, root_states, dof_pos, dof_vel, key_pos_):
        # print("root_states: ", root_states.shape)
        # print("dof_pos: ", dof_pos.shape)
        # print("dof_vel: ", dof_vel.shape)
        # print("key_pos: ", key_pos.shape)
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]
        root_vel = root_states[:, 7:10]
        root_ang_vel = root_states[:, 10:13]

        heading_rot = quat_conjugate(yaw_quat(root_rot))

        # need to convert to xyzw -> wxyz
        root_rot_obs = root_rot[:, [3, 0, 1, 2]]

        local_root_vel = quat_rotate(heading_rot, root_vel)
        local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

        key_pos = key_pos_.reshape(key_pos_.shape[0], 2, 3)
        root_pos_expand = root_pos.unsqueeze(-2)
        local_key_body_pos = key_pos - root_pos_expand
    
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
        flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
        flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                    heading_rot_expand.shape[2])
        local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
        flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
        
        obs = torch.cat((root_rot_obs, dof_pos, dof_vel, flat_local_key_pos), dim=-1)
        return obs
    
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    # quat order is xyzw
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((-q[..., 0:3], q[..., 3:4]), dim=-1).view(shape)   

def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    # quat order is xyzw
    shape = quat.shape
    quat_yaw = quat.view(-1, 4)
    qw = quat_yaw[:, 3]
    qx = quat_yaw[:, 0]
    qy = quat_yaw[:, 1]
    qz = quat_yaw[:, 2]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = torch.zeros_like(quat_yaw)
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)

def quat_rotate(q, v):
    shape = v.shape
    # reshape to (N, 3) for multiplication
    quat = q.reshape(-1, 4)
    vec = v.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, :3]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 3:4] * t + xyz.cross(t, dim=-1)).view(shape)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    # motion = AMPTocabiMotionLoader(args.file, 10, torch.device("cuda"))

    # print("- number of DOFs:", motion.num_dofs)
    # # print("- duration:", motion.)

    # # print(motion.sample(1, times=torch.tensor([2.8]).cpu().numpy())[-1][0])
    # # print(motion.sample(1, times=torch.tensor([6.4]).cpu().numpy())[-1][0])
    # dof_index = motion.get_dof_index(["L_AnkleRoll_Joint", "L_HipRoll_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint"])
    # print(dof_index)
