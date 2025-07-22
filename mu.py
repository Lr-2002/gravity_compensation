import mujoco
import numpy as np
import mujoco.viewer
from pathlib import Path


class MujocoController:
    def __init__(self, model_path=None):
        """
        初始化 MuJoCo 控制器
        :param model_path: 模型文件路径
        """
        if model_path is None:
            model_path = "/Users/lr-2002/project/instantcreation/pin/mujoco_menagerie/franka_emika_panda/mjx_panda.xml"

        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # PD 控制参数
        self.kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 4500])
        self.kd = np.array([450, 450, 350, 350, 200, 200, 200, 450])

        # 初始化关节位置
        self.data.qpos[:] = 0.5

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        print(f"Model loaded: {self.model.nq} joints, {self.model.nv} DOF")

    def set_joint_angles(self, target_angles):
        """
        设置关节目标角度并计算控制力矩
        :param target_angles: 目标关节角度数组
        """
        target_angles = np.asarray(target_angles)

        # PD 控制：tau = kp * (q_target - q_current) - kd * qvel
        position_error = (
            target_angles - self.data.qpos[:-1]
        )  # 排除最后一个自由度（夹爪）
        velocity_damping = self.kd * self.data.qvel[:-1]

        # self.data.ctrl[:] = self.kp * position_error - velocity_damping
        self.set_force(self.kp * position_error - velocity_damping)

    def set_force(self, torque):
        self.data.ctrl[:] = torque

    def set_direct_force(self, torque):
        self.data.qfrc_applied[:] = torque

    def running_check(self):
        return self.viewer.is_running()

    def step(self):
        if self.running_check():
            # breakpoint()
            print(self.data.ctrl)
            mujoco.mj_step(self.model, self.data)
            print("---constraint force is", self.data.qfrc_constraint)
            print("---qfrc actuator is ", self.data.qfrc_actuator)
            self.viewer.sync()

    def run_simulation(self):
        """
        运行仿真，使用 mujoco.viewer
        """

        while self.running_check():
            # 设置目标关节角度
            target_angles = [0, 0, 1.2, 0, 0, 1, 0, 0]
            self.set_joint_angles(target_angles)
            self.step()

    @property
    def q(
        self,
    ):
        return self.data.qpos[:-1]

    @property
    def dq(
        self,
    ):
        return self.data.qvel[:-1]


if __name__ == "__main__":
    controller = MujocoController()
    controller.run_simulation()
