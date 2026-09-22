"""AIM-Robots — Step 1 bonus: a scripted policy that actually picks the cube.

`robot.py`'s docstring ends by claiming "a VLA model and a scripted controller are
interchangeable here". This is the scripted controller.

It exists as a control experiment. The ACT checkpoint run in
`notebooks/01_any_policy.ipynb` is a *learned* policy, and on a hand-built scene it
fails — its weights were fitted to `gym-aloha`'s table, cube spawn region and camera,
not ours. Change exactly one thing (the brain) and keep the robot, the physics and the
`run_policy()` call: the cube comes up, so the scene was never the problem.

It works because it *cheats*: it reads the cube's ground-truth pose straight out of
MuJoCo, which no real camera gives you. That trade is the whole point of the contrast —

    learned policy   generalises from pixels, needs its training distribution
    scripted policy  needs ground truth, works every time inside its assumptions

Three things about ALOHA that the code has to respect:

1. **At the all-zeros default pose the two arms collide** — the right gripper is
   pressed into the left forearm (`aloha/left/lower_forearm_link`). Nothing tracks
   until you move them apart, so we start both arms at ALOHA's home pose, the same
   one every demonstration episode starts from.
2. **The arm actuators are P-only position servos** (`biastype=1`, no damping term)
   with soft gains — `wrist_angle` has kp=37 and droops ~0.3 rad under gravity. So
   commanding an IK solution is not enough; a joint-space **integral** term is
   needed to remove the steady-state error.
3. **The gripper actuator is in METRES**, `ctrlrange=[0.002, 0.037]` — not the 0…1
   convention the LeRobot datasets use. Send 0.5 and it is silently clamped wide open.

Walkthrough with plots and per-position success rates:
`notebooks/01b_scripted_pick.ipynb`.
"""
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco

from strands_robots import Robot
from strands_robots.policies.base import Policy

ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")

# ALOHA's home pose, per arm. Every episode of lerobot/aloha_sim_transfer_cube_human
# starts here; it also happens to be the pose that keeps the two arms out of each
# other's way. Puts each tool centre point at [+/-0.188, -0.019, 0.325].
HOME = (0.0, -0.96, 1.16, 0.0, -0.30, 0.0)

# Gripper actuator limits, in METRES.
GRIP_OPEN, GRIP_SHUT = 0.037, 0.006

# Target tool orientation for the grasp, lifted from a real demonstration
# (episode 0, frame 187 — the frame where the human's gripper closes). Using a
# demonstrated wrist pose instead of a hand-guessed "straight down" keeps the
# solution inside ALOHA's comfortable wrist range.
GRASP_R = np.array([
    [-0.717407, -0.155863, -0.678995],
    [+0.089033, -0.987171, +0.132535],
    [-0.690942, +0.034628, +0.722081],
])


class ArmModel:
    """Kinematics for one arm: name lookups, forward kinematics, and IK.

    Knows nothing about picking cubes — everything task-shaped lives in ScriptedPick.
    Splitting it out is what makes a port to another robot tractable: only this class
    cares how many joints there are or where the tool point lives.
    """

    def __init__(self, sim, prefix, joints=ARM_JOINTS, site=None):
        self.sim, self.m = sim, sim.mj_model
        m = self.m
        site_name = site or f"{prefix}/gripper"

        # mj_name2id returns -1 for anything it cannot find, so a typo (or the wrong
        # robot) silently binds every joint to whatever sits at index -1. Check it.
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/{j}") for j in joints]
        missing = [j for j, i in zip(joints, jids) if i == -1]
        if missing:
            raise ValueError(f"no joint(s) {missing} under prefix '{prefix}' — wrong robot?")
        # the true tool centre point is a SITE, not a body: <prefix>/gripper
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site == -1:
            raise ValueError(f"no site '{site_name}' (this model has {m.nsite} site(s)); "
                             f"a robot without a tool site needs mj_jacBody instead")

        self.n = len(jids)
        self.qadr = [m.jnt_qposadr[i] for i in jids]
        self.dof = [m.jnt_dofadr[i] for i in jids]
        self.lo = np.array([m.jnt_range[i][0] for i in jids])
        self.hi = np.array([m.jnt_range[i][1] for i in jids])
        self.scratch = mujoco.MjData(m)   # IK sandbox: never mutate sim.mj_data

    def measured_q(self):
        """Where the joints actually are right now (not where they were told to go)."""
        d = self.sim.mj_data
        return np.array([d.qpos[a] for a in self.qadr])

    def _pose(self, q):
        """Put q on the scratch state and refresh its kinematics."""
        ds = self.scratch
        ds.qpos[:] = self.sim.mj_data.qpos
        for a, v in zip(self.qadr, q):
            ds.qpos[a] = v
        mujoco.mj_kinematics(self.m, ds)
        mujoco.mj_comPos(self.m, ds)
        return ds

    def fk(self, q):
        """Forward: joint angles -> where the tool point ends up. The easy direction."""
        return self._pose(q).site_xpos[self.site].copy()

    def ik(self, q_seed, target, ref_R, iters=80, damp=1e-2, w_rot=0.35, gain=0.6):
        """Inverse: iterate dq = J.T (J J.T + damp I)^-1 e until the tool point lands.

        Damped least squares. The damping is what stops the solver demanding infinite
        joint velocity near a singularity (an arm stretched straight, say).
        """
        q = np.array(q_seed, float)
        for _ in range(iters):
            ds = self._pose(q)
            e_pos = target - ds.site_xpos[self.site]
            quat = np.zeros(4)          # orientation error, as a rotation vector
            mujoco.mju_mat2Quat(
                quat, (ref_R @ ds.site_xmat[self.site].reshape(3, 3).T).flatten())
            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            if np.linalg.norm(e_pos) < 5e-5 and np.linalg.norm(e_rot) < 5e-3:
                break
            jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, ds, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dof], w_rot * jacr[:, self.dof]])
            err = np.concatenate([e_pos, w_rot * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(J.shape[0]), err)
            q = np.clip(q + gain * dq, self.lo, self.hi)   # respect the joint limits
        return q


class ScriptedPick(Policy):
    """hover → descend → close → lift, re-planned from the live cube pose every chunk.

    The Cartesian setpoint marches on its own clock (a persistent trajectory); joint
    targets come from IK plus an integral term that absorbs servo droop. Phase
    transitions are gated on the *measured* tool pose, which is what makes this
    closed-loop rather than a recorded animation.
    """

    PHASES = ("hover", "descend", "close", "lift", "done")

    def __init__(self, sim, side="right", obj="cube", horizon=20, hold=None,
                 hover_h=0.12, grasp_dz=0.004, lift_h=0.25,
                 step_m=0.0035, tol=0.012, close_steps=60, ki=0.45, bias_max=0.9):
        super().__init__()
        self.sim, self.side, self.horizon = sim, side, horizon
        self.arm = ArmModel(sim, f"aloha/{side}")
        self.hold = dict(hold or {})       # the idle arm, re-commanded every step
        self.hover_h, self.grasp_dz, self.lift_h = hover_h, grasp_dz, lift_h
        self.step_m, self.tol, self.close_steps = step_m, tol, close_steps
        self.ki, self.bias_max = ki, bias_max

        self.obj_bid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, obj)
        if self.obj_bid == -1:
            raise ValueError(f"no body named '{obj}' — did you call sim.add_object()?")

        self.phase, self.sp, self.q_ik = 0, None, None
        self.qbias = np.zeros(self.arm.n)   # the I of a PI controller, in joint space
        self._closing, self._grasp_xy, self.log = 0, None, []

    # -- what the Policy ABC asks for ---------------------------------------
    @property
    def provider_name(self):
        return "scripted-ik"

    @property
    def requires_images(self):
        return False              # this brain reads state, not pixels

    @property
    def execution_horizon(self):
        return self.horizon       # actions per call, same contract as ACT's 100

    def set_robot_state_keys(self, keys):
        pass

    # -- the state machine --------------------------------------------------
    def _goal(self, cube):
        phase = self.PHASES[self.phase]
        if phase == "hover":
            return cube + [0, 0, self.hover_h]
        if phase in ("descend", "close"):
            return cube + [0, 0, self.grasp_dz]
        base = self._grasp_xy if self._grasp_xy is not None else cube
        return np.array([base[0], base[1], self.lift_h])

    async def get_actions(self, observation, instruction, **kwargs):
        cube = self.sim.mj_data.xpos[self.obj_bid].copy()   # <- the cheat: ground truth
        q_live = self.arm.measured_q()
        tcp = self.arm.fk(q_live)

        if self.sp is None:                       # first call: start from where we are
            self.sp, self.q_ik = tcp.copy(), q_live.copy()

        # integral: push harder wherever the servo has drooped behind the command
        self.qbias = np.clip(self.qbias + self.ki * (self.q_ik - q_live),
                             -self.bias_max, self.bias_max)

        actions = []
        for _ in range(self.horizon):
            phase = self.PHASES[self.phase]
            goal = self._goal(cube)

            if phase == "close":
                self._closing += 1
                if self._closing >= self.close_steps:
                    self._grasp_xy = cube.copy()
                    self.log.append(("close", np.round(tcp, 4)))
                    self.phase += 1
            elif phase != "done" and np.linalg.norm(goal - tcp) < self.tol:
                self.log.append((phase, np.round(tcp, 4)))   # gated on the MEASURED pose
                self.phase += 1
                self.sp = tcp.copy()              # re-anchor once, at a phase boundary
                goal = self._goal(cube)

            # the setpoint advances on ITS OWN clock, independent of tracking error --
            # rebase it on the measurement instead and a lagging arm never builds demand
            gap = goal - self.sp
            dist = np.linalg.norm(gap)
            if dist > 1e-9:
                self.sp = self.sp + gap * min(1.0, self.step_m / dist)
            self.q_ik = self.arm.ik(self.q_ik, self.sp, GRASP_R)

            q_cmd = np.clip(self.q_ik + self.qbias, self.arm.lo, self.arm.hi)
            action = {f"{self.side}/{j}": float(v) for j, v in zip(ARM_JOINTS, q_cmd)}
            action[f"{self.side}/gripper"] = (
                GRIP_OPEN if self.PHASES[self.phase] in ("hover", "descend") else GRIP_SHUT)
            action.update(self.hold)
            actions.append(action)
        return actions


def go_home(sim, idle="left"):
    """Put both arms at HOME and return a hold dict that parks the idle arm.

    Without this the arms start at all-zeros, which is a COLLISION state, and the
    working arm cannot move at all.
    """
    m, d = sim.mj_model, sim.mj_data
    qadr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"aloha/{n}")]
    act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"aloha/{n}")
    for sd in ("left", "right"):
        for joint, value in zip(ARM_JOINTS, HOME):
            d.qpos[qadr(f"{sd}/{joint}")] = value   # move it there...
            d.ctrl[act(f"{sd}/{joint}")] = value    # ...AND tell the servo to hold it
        d.ctrl[act(f"{sd}/gripper")] = GRIP_OPEN
    mujoco.mj_forward(m, d)
    hold = {f"{idle}/{j}": v for j, v in zip(ARM_JOINTS, HOME)}
    hold[f"{idle}/gripper"] = GRIP_OPEN
    return hold


def main():
    # Off-centre on purpose: nowhere near where ACT parks, so a pick here cannot be a
    # memorised pose. This position is also repeatable -- three runs all land the cube
    # at z = 0.2465. Marginal positions like (0.08, 0.04) lift ~4 times in 5: the
    # closing fingers sometimes sweep the cube out instead of pinching it.
    cube_xy = (0.15, -0.05)
    sim = Robot("aloha", mesh=False)
    sim.add_object(name="cube", shape="box", position=[*cube_xy, 0.0125],
                   size=[0.025, 0.025, 0.025], color=[1, 0, 0, 1], mass=0.05)
    sim.add_camera(name="cam", position=[0.0, -0.85, 0.52], target=[0.04, 0.0, 0.15], fov=50)

    hold = go_home(sim)
    policy = ScriptedPick(sim, side="right", obj="cube", hold=hold)

    print(f"Policy: {type(policy).__name__} (provider={policy.provider_name})")
    print(f"Requires images: {policy.requires_images}")

    result = sim.run_policy(
        robot_name="aloha",
        policy_object=policy,
        instruction="pick up the red cube",
        n_steps=800,
        video={"path": "scripted_pick.mp4", "camera": "cam", "fps": 50},
    )
    print(f"Status: {result['status']}")

    cube_z = float(sim.mj_data.xpos[policy.obj_bid][2])
    print(f"Phases reached: {[str(e[0]) for e in policy.log]}")
    print(f"Cube z: 0.0125 -> {cube_z:.4f}   PICKED: {cube_z > 0.06}")


if __name__ == "__main__":
    main()
