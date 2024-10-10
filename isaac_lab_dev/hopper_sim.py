"""Script to simulate Go2 Robot in simulation"""
from dataclasses import MISSING

"""Launch the Isaac Sim Simulator First"""
import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadruped base environment.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import csv
import os
from pytorch3d.transforms import quaternion_invert, quaternion_multiply, so3_log_map, quaternion_to_matrix, Rotate, \
    euler_angles_to_matrix, matrix_to_quaternion
import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs.mdp.actions.joint_actions import JointAction
from omni.isaac.lab.envs.mdp.actions.actions_cfg import JointActionCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from collections.abc import Sequence

##
# Pre-defined configs
##
from isaac_lab_dev.hopper_config import HOPPER_CFG


def zero_traj_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[0, 0, 0, 0, 0]], device=env.device).repeat(env.num_envs, 1)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""
    # terrain - flat terrain plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    # add robot
    robot: ArticulationCfg = HOPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/foot", update_period=0.0, history_length=1, debug_vis=False
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


class HopperGeometricPD(JointAction):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._raw_actions[:, 0] = 1.
        self._processed_actions = torch.clone(self._raw_actions.detach())

        self.actuator_transform = Rotate(torch.tensor([
            [-0.8165, 0.2511, 0.2511],
            [-0, -0.7643, 0.7643],
            [-0.5773, -0.5939, -0.5939]
        ]), device=torch.device("cuda"))

        # self.actuator_transform = Rotate(torch.tensor([
        #     [-1., 0., 0.],
        #     [0., 1., 0.],
        #     [0., 0., -1.]
        # ]), device=torch.device("cuda"))

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions / torch.linalg.norm(self._raw_actions, axis=-1, keepdims=True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids, 1:] = 0.0
        self._raw_actions[env_ids, 0] = 1.

    def apply_actions(self):
        quat_d = self.processed_actions
        quat = self._asset.data.root_quat_w
        contact = torch.greater(torch.linalg.norm(self._env.scene.sensors['contact_forces'].data.net_forces_w, axis=-1), self.cfg.contact_threshold).squeeze()
        omega = self._asset.data.root_ang_vel_b
        wheel_vel = self._asset.data.joint_vel[:, 1:]

        not_contact = torch.logical_not(contact)
        torques = torch.zeros((quat.shape[0], 3), device=quat.device)
        # Spindown, when in contact
        if torch.any(contact):
            torques[contact, :] = -self.cfg.Kspindown * wheel_vel[contact, :]
        # Orientation Tracking
        if torch.any(not_contact):
            quat_d = quat_d[not_contact, :] / torch.linalg.norm(quat_d[not_contact, :], dim=-1, keepdim=True)
            quat = quat[not_contact, :] / torch.linalg.norm(quat[not_contact, :], dim=-1, keepdim=True)
            omega = omega[not_contact, :]
            err = quaternion_multiply(quaternion_invert(quat_d), quat)
            log_err = so3_log_map(quaternion_to_matrix(err))
            local_tau = -self.cfg.Kp * log_err - self.cfg.Kd * omega
            tau = self.actuator_transform.transform_points(local_tau)
            torques[not_contact, :] = tau

        # with open('data/wheel_torque.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(torques[0, :].tolist())
        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)


@configclass
class HopperGeometricPDCfg(JointActionCfg):

    class_type: type[ActionTerm] = HopperGeometricPD

    Kp: float = MISSING
    Kd: float = MISSING
    Kspindown: float = MISSING
    contact_threshold: float = MISSING


class HopperFoot(JointAction):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 0

    def apply_actions(self):
        foot_pos = self._asset.data.joint_pos[:, self._joint_ids]
        foot_vel = self._asset.data.joint_vel[:, self._joint_ids]
        not_contact = torch.less(torch.linalg.norm(self._env.scene.sensors['contact_forces'].data.net_forces_w, axis=-1), self.cfg.contact_threshold).squeeze()

        torques = torch.zeros_like(foot_pos, device=foot_pos.device)
        if torch.any(not_contact):
            torques[not_contact] = self.cfg.spring_stiffness * self.cfg.foot_pos_des - self.cfg.Kp * (
                        foot_pos[not_contact] - self.cfg.foot_pos_des) - self.cfg.Kd * foot_vel[not_contact]
        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)


@configclass
class HopperFootCfg(JointActionCfg):

    class_type: type[ActionTerm] = HopperFoot
    Kp: float = MISSING
    Kd: float = MISSING
    spring_stiffness: float = MISSING
    foot_pos_des: float = MISSING
    contact_threshold: float = MISSING


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    wheel_eff = HopperGeometricPDCfg(
        asset_name="robot", joint_names=["wheel.*"],
        Kp=60., Kd=8., Kspindown=0.1,
        contact_threshold=0.1
    )
    foot_eff = HopperFootCfg(
        asset_name="robot", joint_names=["foot_slide"],
        Kp=25, Kd=10,
        spring_stiffness=HOPPER_CFG.actuators['foot'].stiffness,
        foot_pos_des=0.02, contact_threshold=0.1
    )


def get_body_acc(env):
    return env.scene['robot'].data.body_ang_acc_w[:, 0, :]


def get_joint_acc(env):
    return env.scene['robot'].data.joint_acc


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        root_pos = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.0, n_max=0.0))    # [:3]
        root_quat = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.0, n_max=0.0))    # [3:7] (w, x, y, z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.0, n_max=0.0))  # [7:10]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0., n_max=0.))  # [10:13]

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0., n_max=0.))  # [13:17]
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0., n_max=0.))    # [17:21]
        traj_commands = ObsTerm(func=zero_traj_commands)                                    # [21:26]
        contact_force = ObsTerm(                                                            # [26]
            func=lambda e: mdp.contact_forces(e, threshold=0., sensor_cfg=SceneEntityCfg("contact_forces"))[:, None],
            # params={"threshold": 0, "sensor_cfg": SceneEntityCfg("contact_forces")},
            noise=Unoise(n_min=-0.0, n_max=0.0)
        )
        actions = ObsTerm(func=mdp.last_action)                                             # [27:]
        body_acc = ObsTerm(func=get_body_acc, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_acc = ObsTerm(func=get_joint_acc, noise=Unoise(n_min=-0.0, n_max=0.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


def quat2yaw(quat):
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return yaw

def rpy2quat(r, p, y):
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)
    cp = torch.cos(p * 0.5)
    sp = torch.sin(p * 0.5)
    cr = torch.cos(r * 0.5)
    sr = torch.sin(r * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack((w, x, y, z), dim=-1)


class RaibertHeuristic:
    CMD_IDX = 21

    def __init__(
            self, Kp, Kd, Kff, clip_pos, clip_vel, clip_ff, clip_ang
    ):
        self.Kp = Kp
        self.Kd = Kd
        self.clip_pos = clip_pos
        self.clip_vel = clip_vel
        self.clip_ff = clip_ff
        self.clip_ang = clip_ang
        self.Kff = Kff

    def __call__(self, obs):
        e_x = -(obs[:, RaibertHeuristic.CMD_IDX] - obs[:, 0])
        e_y = obs[:, RaibertHeuristic.CMD_IDX + 1] - obs[:, 1]
        e_vx = obs[:, 7]
        e_vy = -obs[:, 8]
        vx_cmd = -obs[:, RaibertHeuristic.CMD_IDX + 3]
        vy_cmd = obs[:, RaibertHeuristic.CMD_IDX + 1]
        quat = obs[:, 3:7]
        yaw = quat2yaw(quat)

        pitch_d = torch.clip(
            - self.Kp * torch.clip(e_x, -self.clip_pos, self.clip_pos) \
            - self.Kd * torch.clip(e_vx, -self.clip_vel, self.clip_vel) \
            + self.Kff * torch.clip(vx_cmd, -self.clip_ff, self.clip_ff),
            -self.clip_ang, self.clip_ang)
        roll_d = torch.clip(
            - self.Kp * torch.clip(e_y, -self.clip_pos, self.clip_pos) \
            - self.Kd * torch.clip(e_vy, -self.clip_vel, self.clip_vel) \
            + self.Kff * torch.clip(vy_cmd, -self.clip_ff, self.clip_ff),
            -self.clip_ang, self.clip_ang)
        yaw_d = torch.clip(obs[:, RaibertHeuristic.CMD_IDX + 2], yaw - self.clip_ang, yaw + self.clip_ang)

        quat_d = rpy2quat(roll_d, pitch_d, yaw_d)

        return quat_d


@configclass
class EventCfg:
    """Configuration for events."""
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material


def main():
    """Main function."""
    # setup base environment
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # Raibert Policy
    raibert_policy = RaibertHeuristic(
        0.2, 0.4, 0.1, 0.5, 1, 0.2, 0.4
    )

    # simulate physics
    count = 0
    obs, _ = env.reset()
    if os.path.exists('data/wheel_torque.csv'):
        os.remove('data/wheel_torque.csv')
    with open('data/output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)


        while simulation_app.is_running():
            with torch.inference_mode():
                writer.writerow(obs['policy'][0].tolist())
                print(count)
                # reset
                if count % 250 == 0:
                    obs, _ = env.reset()
                    count = 0
                    print("-" * 80)
                    print("[INFO]: Resetting environment...")
                # step env
                action = raibert_policy(obs["policy"])

                obs, _ = env.step(action)
                # update counter
                count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()