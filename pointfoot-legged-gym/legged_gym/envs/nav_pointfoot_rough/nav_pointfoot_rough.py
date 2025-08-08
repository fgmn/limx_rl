import torch
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    torch_rand_sqrt_float,
)
from .nav_pointfoot_rough_config import BipedCfgPF

class BipedPF(BaseTask):
    
    def __init__(
        self, cfg: BipedCfgPF, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            obs (torch.Tensor): Tensor of shape (num_envs, num_observations_per_env)
            rewards (torch.Tensor): Tensor of shape (num_envs)
            dones (torch.Tensor): Tensor of shape (num_envs)
        """
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):    # 物理仿真频率 / 策略控制频率
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )   # 将动作推入 FIFO 缓冲以实现动作延迟
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # 每个step都重新计算相对命令
        self._update_commands()

        # # Debug printout for the first environment after each step
        # if hasattr(self, 'step_count') and self.step_count % 100 == 0:  # Print every 100 steps to avoid spam
        #     env_id = 0  # First environment
        #     robot_pos = self.root_states[env_id, :2]  # x, y position of robot
        #     target_pos = self.global_targets[env_id]  # 全局目标位置
        #     relative_target = self.commands[env_id, :2] * self.commands_scale  # 相对目标位置
        #     distance_to_target = torch.norm(target_pos - robot_pos).item()  # 使用全局位置计算距离，与奖励函数一致
        #     reward = self.rew_buf[env_id].item()
        #     done = self.reset_buf[env_id].item()
        #     print(f"[DEBUG Step {self.step_count}] Env 0: Robot pos=({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
        #           f"Target pos=({target_pos[0]:.2f}, {target_pos[1]:.2f}), "
        #           f"Relative=({relative_target[0]:.2f}, {relative_target[1]:.2f}), "
        #           f"Distance={distance_to_target:.2f}m, Reward={reward:.3f}, Done={done}")
        
        # if not hasattr(self, 'step_count'):
        #     self.step_count = 0
        # self.step_count += 1

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :2] * self.commands_scale,  # 返回 x, y 目标位置
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )

    def _resample_commands(self, env_ids):
        """随机选择目标位置的全局坐标，commands将在每个step中计算相对位置

        Args:
            env_ids (List[int]): 需要重新采样命令的环境ID
        """
        if len(env_ids) == 0:
            return
            
        # 获取当前位置（全局坐标）
        current_pos = self.root_states[env_ids, :2]  # x, y 坐标（全局）
        
        # 随机生成距离和角度
        distances = torch.rand(len(env_ids), device=self.device) * (
            self.cfg.commands.target_distance_max - self.cfg.commands.target_distance_min
        ) + self.cfg.commands.target_distance_min
        
        angles = torch.rand(len(env_ids), device=self.device) * 2 * np.pi
        
        # 计算全局目标位置
        target_x = current_pos[:, 0] + distances * torch.cos(angles)
        target_y = current_pos[:, 1] + distances * torch.sin(angles)
        
        # 存储全局目标位置
        self.global_targets[env_ids, 0] = target_x
        self.global_targets[env_ids, 1] = target_y
        
        # 初始计算相对位置作为命令（将在每个step中更新）
        relative_x = distances * torch.cos(angles)
        relative_y = distances * torch.sin(angles)
        self.commands[env_ids, 0] = relative_x
        self.commands[env_ids, 1] = relative_y  
        
        # 更新目标点可视化（如果启用）
        if self.cfg.commands.visualize_targets and not self.headless:
            self._update_target_visualization(env_ids.cpu().numpy() if hasattr(env_ids, 'cpu') else env_ids)
        
        # 记录任务开始时间，用于计算奖励持续时间
        if not hasattr(self, 'task_start_time'):
            self.task_start_time = torch.zeros(self.num_envs, device=self.device)
        self.task_start_time[env_ids] = self.episode_length_buf[env_ids].float() * self.dt

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12:18] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[18:24] = 0.0  # previous actions
        noise_vec[24:26] = 0.0  # clock signals (sin, cos)
        noise_vec[26:30] = 0.0  # gait parameters
        # 移除了相对目标位置的噪声设置，现在观测是30维
        return noise_vec
    
    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        
        # 重置导航任务相关的缓冲区
        if hasattr(self, 'task_start_time'):
            self.task_start_time[env_ids] = 0.0
        # global_targets 由 _resample_commands 设置，不需要手动重置
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,                    # 3维：角速度
                self.projected_gravity,                                         # 3维：重力投影
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 6维：关节位置
                self.dof_vel * self.obs_scales.dof_vel,                        # 6维：关节速度
                self.actions,                                                   # 6维：上一步动作
                self.clock_inputs_sin.view(self.num_envs, 1),                 # 1维：步态时钟sin
                self.clock_inputs_cos.view(self.num_envs, 1),                 # 1维：步态时钟cos
                self.gaits,                                                     # 4维：步态参数
            ),
            dim=-1,
        )
        # 总计：3+3+6+6+6+1+1+4 = 30维 (移除了相对目标位置的2维)
        
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf), dim=-1)
        return obs_buf, critic_obs_buf
    
    # --------------------------- reward functions---------------------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_navigation_task(self):
        """主任务奖励：到达目标位置
        
        公式：r_task = (1/T_r) * (1 / (1 + ||x_b - x_b*||^2))  当 t > T - T_r 时
               否则为 0
        """
        # 计算当前时间
        current_time = self.episode_length_buf.float() * self.dt
        max_episode_time = self.cfg.env.episode_length_s
        reward_duration = self.cfg.rewards.reward_duration
        
        # 检查是否在奖励时间窗口内
        in_reward_window = current_time > (max_episode_time - reward_duration)
        
        # 直接使用机器人位置和全局目标位置计算距离
        current_pos = self.root_states[:, :2]  # x, y 坐标（全局）
        target_pos = self.global_targets       # 全局目标位置
        distance_to_target = torch.norm(target_pos - current_pos, dim=1)
        
        # 计算奖励
        reward = torch.zeros_like(distance_to_target)
        reward[in_reward_window] = (1.0 / reward_duration) * (1.0 / (1.0 + distance_to_target[in_reward_window] ** 2))
        
        return reward

    def _reward_navigation_bias(self):
        """辅助奖励：朝目标方向移动
        
        公式：r_bias = (x_dot_b · (x_b* - x_b)) / (||x_dot_b|| * ||x_b* - x_b||)
        
        当主任务奖励达到最大值的50%后，此奖励会被移除
        """
        # 检查是否应该移除辅助奖励
        if hasattr(self, 'max_task_reward'):
            current_task_reward = self._reward_navigation_task()
            if torch.mean(current_task_reward) > self.cfg.rewards.bias_removal_threshold * self.max_task_reward:
                return torch.zeros(self.num_envs, device=self.device)
        else:
            # 记录最大任务奖励
            self.max_task_reward = 1.0 / self.cfg.rewards.reward_duration
        
        # 直接使用机器人位置和全局目标位置计算
        current_pos = self.root_states[:, :2]  # x, y 坐标（全局）
        target_pos = self.global_targets       # 全局目标位置
        current_vel = self.base_lin_vel[:, :2] # x, y 速度
        
        # 计算到目标的向量
        to_target = target_pos - current_pos
        to_target_norm = torch.norm(to_target, dim=1)
        vel_norm = torch.norm(current_vel, dim=1)
        
        # 避免除零
        valid_mask = (to_target_norm > 1e-6) & (vel_norm > 1e-6)
        
        reward = torch.zeros(self.num_envs, device=self.device)
        
        if torch.any(valid_mask):
            # 计算方向一致性
            cos_similarity = torch.sum(current_vel[valid_mask] * to_target[valid_mask], dim=1) / (
                vel_norm[valid_mask] * to_target_norm[valid_mask]
            )
            reward[valid_mask] = cos_similarity
        
        return reward

    def _reward_tracking_lin_vel(self):
        # 导航任务中不使用速度跟踪，返回零奖励
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_tracking_ang_vel(self):
        # 导航任务中不使用角速度跟踪，返回零奖励
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
        else:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * (
                    1 - torch.exp(-foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))

        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * (
                    1 - torch.exp(-foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma))
        return reward / len(self.feet_indices)

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.001
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _init_buffers(self):
        """Initialize additional buffers for navigation task"""
        super()._init_buffers()
        
        # 初始化导航任务相关的缓冲区
        self.task_start_time = torch.zeros(self.num_envs, device=self.device)
        self.max_task_reward = 1.0 / self.cfg.rewards.reward_duration
        
        # 为导航任务重新定义命令缩放（只有目标位置）
        self.commands_scale = torch.tensor(
            [1.0, 1.0],  # x, y 位置不缩放
            device=self.device,
            requires_grad=False,
        )
        
        # 确保基础位置缓冲区存在
        if not hasattr(self, 'base_position'):
            self.base_position = self.root_states[:, :3]
        
        # 初始化全局目标位置缓冲区
        self.global_targets = torch.zeros(self.num_envs, 2, device=self.device)
        
        # 初始化目标点可视化
        self._init_target_visualization()
        
    def _init_target_visualization(self):
        """初始化目标点可视化标记"""
        if not self.cfg.commands.visualize_targets or self.headless:
            self.target_spheres = None
            return
            
        # 将颜色列表转换为元组
        color_tuple = tuple(self.cfg.commands.target_marker_color)
        
        # 创建目标点几何体（线框球体）
        self.target_sphere_geom = gymutil.WireframeSphereGeometry(
            self.cfg.commands.target_marker_size, 
            8, 8, None, 
            color=color_tuple
        )
        
        # 标记可视化已初始化
        self.target_visualization_enabled = True
            
    def _update_target_visualization(self, env_ids=None):
        """更新目标点可视化位置
        
        Args:
            env_ids: 需要更新的环境ID列表，如果为None则更新所有环境
        """
        if not self.cfg.commands.visualize_targets or self.headless:
            return
            
        # 调用调试可视化方法绘制目标点
        self._draw_target_debug_vis()

    def _get_terrain_height_at_position(self, x, y):
        """获取指定位置的地形高度
        
        Args:
            x (float): 全局X坐标
            y (float): 全局Y坐标
            
        Returns:
            float: 该位置的地形高度
        """
        if self.cfg.terrain.mesh_type == "plane":
            return 0.0
        elif self.cfg.terrain.mesh_type == "none":
            return 0.0
        elif not hasattr(self, 'height_samples'):
            return 0.0
            
        # 将全局坐标转换为地形坐标系（参考_get_heights函数的实现）
        # 添加边界偏移
        terrain_x = x + self.terrain.cfg.border_size
        terrain_y = y + self.terrain.cfg.border_size
        
        # 转换为网格索引
        px = int(terrain_x / self.terrain.cfg.horizontal_scale)
        py = int(terrain_y / self.terrain.cfg.horizontal_scale)
        
        # 确保索引在有效范围内
        px = max(0, min(px, self.height_samples.shape[0] - 2))
        py = max(0, min(py, self.height_samples.shape[1] - 2))
        
        # 获取周围三个点的高度并取最小值（与_get_heights保持一致）
        height1 = self.height_samples[px, py]
        height2 = self.height_samples[px + 1, py]
        height3 = self.height_samples[px, py + 1]
        
        height = min(height1, height2, height3)
        
        return height.item() * self.terrain.cfg.vertical_scale

    def _draw_target_debug_vis(self):
        """绘制目标点调试可视化
        类似于_draw_debug_vis的实现方式，使用gymutil绘制目标点
        """
        if not self.cfg.commands.visualize_targets or self.headless:
            return
        if not hasattr(self, 'viewer') or self.viewer is None:
            return
            
        # 清除之前的线条
        self.gym.clear_lines(self.viewer)
        
        # 创建目标点几何体（线框球体）
        sphere_geom = gymutil.WireframeSphereGeometry(
            self.cfg.commands.target_marker_size, 
            6, 6, None, 
            color=tuple(self.cfg.commands.target_marker_color)
        )
        
        # 为所有环境绘制目标点，完全模仿原始_draw_debug_vis的实现
        for i in range(self.num_envs):
            if i < len(self.global_targets):
                # 直接使用全局目标位置
                target_x = self.global_targets[i, 0].item()
                target_y = self.global_targets[i, 1].item()
                
                # 使用新的地形高度获取函数
                terrain_height = self._get_terrain_height_at_position(target_x, target_y)
                target_z = terrain_height + self.cfg.commands.target_marker_height
                
                # 创建目标点位姿（使用全局坐标）
                target_pose = gymapi.Transform(
                    gymapi.Vec3(target_x, target_y, target_z), 
                    r=None
                )
                
                # 绘制目标点
                gymutil.draw_lines(
                    sphere_geom, 
                    self.gym, 
                    self.viewer, 
                    self.envs[i], 
                    target_pose
                )
                
    def render(self, sync_frame_time=True):
        """重写render方法以包含目标点可视化"""
        # 调用父类的render方法
        super().render(sync_frame_time)
        
        # 如果有viewer且启用了目标点可视化，则绘制目标点
        if self.viewer and self.cfg.commands.visualize_targets and not self.headless:
            self._draw_target_debug_vis()

    def _update_commands(self):
        """每步重新计算相对位置命令
        
        将全局目标位置转换为相对于当前机器人位置的向量
        """
        # 获取当前机器人位置
        current_pos = self.root_states[:, :2]  # x, y 坐标（全局）
        
        # 计算相对位置命令 = 全局目标位置 - 当前位置
        self.commands[:, :2] = self.global_targets - current_pos