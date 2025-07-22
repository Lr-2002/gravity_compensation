from mu import MujocoController
from gravity import Gravity_Compensation

openarm = "/Users/lr-2002/project/instantcreation/openarm_simulation/assets/mujoco/openarm_force.mjcf.xml"

mc = MujocoController(openarm)
model = Gravity_Compensation(openarm)
while True:
    qq = mc.q
    dqq = mc.dq
    model.q = [*qq, qq[-1]]
    model.dq = [*dqq, dqq[-1]]
    # breakpoint()
    torque = model.compute_gravity_torque()
    # torque = model.compuate_c_g()
    mc.set_force(torque[:-1])
    # mc.set_direct_force(torque)
    mc.step()
