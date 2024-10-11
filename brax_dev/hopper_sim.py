import jax
from jaxlie import SO3
import jax.numpy as jnp

import mujoco as mjc
from mujoco import mjx

import mediapy as media
import time
import matplotlib.pyplot as plt

visualize = False

rh_Kp = 0.2
rh_Kd = 0.4
rh_Kff = 0.15
clip_pos = 0.5
clip_vel = 1.0
clip_ff = 0.2
clip_ang = 0.4
contact_threshold = 0.1
Kspindown = 0.1

Kp = 60
Kd = 8

actuator_transform = jnp.linalg.inv(jnp.array([
    [-0.8165, 0.2511, 0.2511],
    [-0, -0.7643, 0.7643],
    [-0.5773, -0.5939, -0.5939]
]))

num_envs = 1024

def quat2yaw(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.atan2(siny_cosp, cosy_cosp)
    return yaw

def rpy2quat(r, p, y):
    cy = jnp.cos(y * 0.5)
    sy = jnp.sin(y * 0.5)
    cp = jnp.cos(p * 0.5)
    sp = jnp.sin(p * 0.5)
    cr = jnp.cos(r * 0.5)
    sr = jnp.sin(r * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return jax.numpy.array((w, x, y, z))

def raibert_heuristic(qpos, qvel, pos_d, vel_d, yaw_d):
    e_x = -(pos_d[0] - qpos[0])
    e_y = pos_d[1] - qpos[1]
    e_vx = qvel[0]
    e_vy = -qvel[1]
    vx_cmd = -vel_d[0]
    vy_cmd = vel_d[1]
    quat = qpos[3:7]
    yaw = quat2yaw(quat)

    pitch_d = jnp.clip(
        - rh_Kp * jnp.clip(e_x, -clip_pos, clip_pos) \
        - rh_Kd * jnp.clip(e_vx, -clip_vel, clip_vel) \
        + rh_Kff * jnp.clip(vx_cmd, -clip_ff, clip_ff),
        -clip_ang, clip_ang)
    roll_d = jnp.clip(
        - rh_Kp * jnp.clip(e_y, -clip_pos, clip_pos) \
        - rh_Kd * jnp.clip(e_vy, -clip_vel, clip_vel) \
        + rh_Kff * jnp.clip(vy_cmd, -clip_ff, clip_ff),
        -clip_ang, clip_ang)
    yaw_d = jnp.clip(yaw_d, yaw - clip_ang, yaw + clip_ang)

    quat_d = rpy2quat(roll_d, pitch_d, yaw_d)
    return quat_d


def geometric_pd(quat_d, qpos, qvel, contact_dist):
    quat_d = SO3(quat_d / jnp.linalg.norm(quat_d))
    quat = SO3(qpos[3:7])
    contact = contact_dist[0] < contact_threshold
    omega = qvel[3:6]
    wheel_vel = qvel[-3:]

    err = quat_d.inverse().multiply(quat)
    log_err = err.log()
    local_tau = -Kp * log_err - Kd * omega
    tau = actuator_transform @ local_tau

    torques = jnp.where(contact, -Kspindown * wheel_vel, tau)
    return torques


def controller(qpos, qvel, contact_dist, pos_d, vel_d, yaw_d):
    quat_d = raibert_heuristic(qpos, qvel, pos_d, vel_d, yaw_d)
    return geometric_pd(quat_d, qpos, qvel, contact_dist)

def main():
    mj_model = mjc.MjModel.from_xml_path('../rsc/hopper/hopper.xml')
    mj_data = mjc.MjData(mj_model)
    if visualize:
        renderer = mjc.Renderer(mj_model, width=1920, height=1080)
    
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    duration = 4  # (seconds)
    framerate = 60  # (Hz)
    
    def change_loc(mjx_data_, x, y):
        qpos = mjx_data.qpos
        qpos = qpos.at[0].set(x)
        qpos = qpos.at[1].set(y)
        return mjx_data_.replace(qpos=qpos)
    
    batch_data = jax.vmap(lambda ind: change_loc(mjx_data, ind // int(num_envs ** 0.5), ind % int(num_envs ** 0.5)))(jax.numpy.arange(num_envs))
    plt.scatter(batch_data.qpos[:, 0], batch_data.qpos[:, 1])
    plt.show()
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    jit_control = jax.jit(jax.vmap(controller, in_axes=(0, 0, 0, 0, 0, 0)))

    pos_d = jnp.zeros((num_envs, 2))
    vel_d = jnp.zeros((num_envs, 2))
    yaw_d = jnp.zeros((num_envs,))


    controller(batch_data.qpos[1], batch_data.qvel[1], batch_data.contact.dist[1], pos_d[1],  vel_d[1], yaw_d[1])
    frames = []
    t0 = time.perf_counter()
    i = 0

    while jax.numpy.min(batch_data.time) < duration:
        torques = jit_control(batch_data.qpos, batch_data.qvel, batch_data.contact.dist, pos_d,  vel_d, yaw_d)
        batch_data = batch_data.replace(ctrl=batch_data.ctrl.at[:, 1:].set(torques))
        batch_data = jit_step(mjx_model, batch_data)
        print(f"Step {i}: {time.perf_counter() - t0}s")
        t0 = time.perf_counter()
        i += 1
        if visualize and len(frames) < jax.numpy.min(batch_data.time) * framerate:
            mj_data = mjx.get_data(mj_model, jax.tree.map(lambda x: x[1], batch_data))
            renderer.update_scene(mj_data)
            pixels = renderer.render()
            frames.append(pixels)

    if visualize:
        # Simulate and display video.
        media.write_video("data/hopper_sim_video.mp4", frames, fps=framerate)
        renderer.close()

    plt.scatter(batch_data.qpos[:, 0], batch_data.qpos[:, 1])
    plt.show()
    

if __name__ == '__main__':
    main()
