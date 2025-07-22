import numpy as np
import pinocchio as pin


class Gravity_Compensation:
    def __init__(
        self,
        urdf="/Users/lr-2002/project/instantcreation/pin/mujoco_menagerie/franka_emika_panda/mjx_panda.xml",
    ) -> None:
        self.urdf = urdf
        models_tuple = pin.buildModelsFromMJCF(self.urdf)
        self.model = models_tuple[0]  # 只取主 model
        self.data = self.model.createData()
        self.gravity = 9.8
        self._q = np.zeros(self.model.nq)
        self._dq = np.zeros(self.model.nv)

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        value = np.array(value)
        assert value.shape == self._q.shape, "you have input wrong shape value "
        self._q = value
        print("set q to", self._q)

    @property
    def dq(self):
        return self._dq

    @dq.setter
    def dq(self, value):
        value = np.array(value)
        assert value.shape == self._dq.shape, "you have input wrong shape value "
        self._dq = value
        print("set dq to", self._dq)

    def compute_gravity_torque(self, q=None):
        """
        计算当前关节位置下的重力补偿力矩
        :param q: 关节位置向量，可选，若不提供则使用当前 self.q
        :return: 重力补偿力矩（numpy 数组）
        """
        g = pin.computeGeneralizedGravity(self.model, self.data, self._q)
        return g

    def compuate_c_g(self):
        return pin.nonLinearEffects(self.model, self.data, self._q, self._dq)

    def eef(self):
        print(" the input q is", self.q)
        pose = pin.forwardKinematics(self.model, self.data, self._q)
        return self.data.oMf[-1]


if __name__ == "__main__":
    gc = Gravity_Compensation(
        urdf="/Users/lr-2002/project/instantcreation/openarm_simulation/assets/mujoco/openarm.mjcf.xml"
    )
    # 示例：设置关节位置为零，计算重力补偿力矩
    q_zero = np.random.uniform(0, 1, size=gc.model.nq)
    print(q_zero)
    gc.q = q_zero
    tau_g = gc.compute_gravity_torque()
    print("Gravity compensation torque:", tau_g)
    print(gc.eef())
