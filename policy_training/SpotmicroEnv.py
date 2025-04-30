import pybullet
import pybullet_data
import numpy as np
import gymnasium as gym
from collections import deque

class SpotmicroEnv(gym.Env):
    def __init__(self, use_gui=False):
        super().__init__()

        self._OBS_SPACE_SIZE = 94
        self._ACT_SPACE_SIZE = 12
        self._MAX_EPISODE_LEN = 1000
        self._SIM_FREQUENCY = 240
        self._CONTROL_FREQUENCY = 60

        self._step_counter = 0
        self._robot_id = None
        self._plane_id = None
        self._motor_joints_id = []
        self._joint_history = deque(maxlen=5)
        self._previous_action = np.zeros(self._ACT_SPACE_SIZE, dtype=np.float32)
        self.physics_client = None
        self.use_gui = use_gui

        self._tilt_step = None #Keep track of where the plane is at (only needed in the tilting plane simulation)
        self._tilt_phase = None

        #Declaration of observation and action spaces
        self.observation_space = gym.spaces.Box(
            low = -np.inf, 
            high = np.inf, 
            shape = (self._OBS_SPACE_SIZE,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low = -1.0, 
            high = 1.0, 
            shape = (self._ACT_SPACE_SIZE,),
            dtype = np.float32
        )

        self._agent_state = {
            "base_position": None,
            "base_orientation": None,
            "linear_velocity": None,
            "angular_velocity": None
        }

        #If the agents is in this state, we terminate the simulation. Should quantize the fact that it has fallen, maybe a threshold?
        self._target_state = {
            "min_height": 0.10, #meters?
        } 
    
    def close(self):
        """
        Method exposed and used by SB3.
        Clean up the simulation.
        """
        if self.physics_client is not None:
            pybullet.disconnect(self.physics_client)
            self.physics_client = None

    def reset(self, seed: int |None = None, options: dict | None = None) -> tuple[gym.spaces.Box, dict]:
        """
        Method exposed and called by SB3 before starting each episode.
        Sets the all parameters and puts the agent in place
        """
        
        super().reset(seed=seed)
        self._step_counter = 0
        self._agent_state["base_position"] = (0.0 , 0.0, 0.24)
        self._agent_state["base_orientation"] = pybullet.getQuaternionFromEuler([0,0,0])
        self._agent_state["linear_velocity"] = 0.0
        self._agent_state["angular_velocity"] = 0.0

        self._tilt_step = 0
        self._tilt_phase = np.random.uniform(0, 2 * np.pi)

        self._joint_history.clear()
        dummy_joint_state = [0.0] * 24
        for _ in range(5):
            self._joint_history.append(dummy_joint_state)

        #Initialize pybullet
        if self.physics_client is None:
            self.physics_client = pybullet.connect(pybullet.GUI if self.use_gui else pybullet.DIRECT)

        pybullet.resetSimulation(physicsClientId=self.physics_client)
        pybullet.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        pybullet.setTimeStep(1/self._SIM_FREQUENCY, physicsClientId=self.physics_client)

        #load robot URDF here
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = pybullet.loadURDF(
            "plane.urdf", 
            basePosition = [0,0,0],
            baseOrientation = pybullet.getQuaternionFromEuler([0,0,0]),
            physicsClientId=self.physics_client
        )
        
        self._robot_id = pybullet.loadURDF(
            "spotmicroai.urdf",
            basePosition = self._agent_state["base_position"],
            baseOrientation = self._agent_state["base_orientation"],
            physicsClientId = self.physics_client
        )

        # Builld it oonce and make it immutable
        if not isinstance(self._motor_joints_id, tuple):
            for i in range(pybullet.getNumJoints(self._robot_id)):
                joint_info = pybullet.getJointInfo(self._robot_id, i)
                joint_type = joint_info[2]

                if joint_type == pybullet.JOINT_REVOLUTE:
                    self._motor_joints_id.append(i)
        self._motor_joints_id = tuple(self._motor_joints_id)

        #this is just to let the physics stabilize? -> might need to remove this loop
        for _ in range(10):
            pybullet.stepSimulation(physicsClientId=self.physics_client)
        
        self._update_agent_state()
        observation = self._get_observation()

        return (observation, self._get_info())

    def step(self, action: np.ndarray) -> tuple[gym.spaces.Box, float, bool, bool, dict]:
        """
        Method exposed and used bby SB3 to execute one time step within the environment.

        Args:
            action (gym.spaces.Box): The action to take in the environment.

        Returns:
            tuple containing
                - observation (np.ndarray): Agent's observation of the environment.
                - reward (float): Amount of reward returned after previous action.
                - terminated (bool): Whether the episode naturally ended.
                - truncated (bool): Whether the episode was artificially terminated.
                - info (dict): Contains auxiliary diagnostic information.
        """
        counter = 0
        if counter == int(self._SIM_FREQUENCY / self._CONTROL_FREQUENCY) - 1:
            observation = self._step_simulation(action)
            counter = 0
        else:
            counter += 1
            observation = self._step_simulation(self._previous_action)

        reward = self._calculate_reward(action)
        terminated = self._is_target_state(self._agent_state) # checks wether the agent has fallen or not
        truncated = self._is_terminated()
        info = self._get_info() 
        
        self._previous_action = action.copy()
        return (observation, reward, terminated, truncated, info)
    
    #@TODO: should also tilt the plane, look up the code fromm the notebook
    def _step_simulation(self, action: np.ndarray) -> np.ndarray:
        """
        Private method that calls the API to execute the given action in PyBullet.
        It should sinchronize the state of the agent in the simulation with the state recorded here!
        Accepts an action and returns an observation
        """

        self._step_counter += 1 #updates the step counter (used to check against timeouts)

        # Execute the action in pybullet
        for i in range(len(self._motor_joints_id)):
            pybullet.setJointMotorControl2(
                bodyUniqueId = self._robot_id,
                jointIndex = self._motor_joints_id[i],
                controlMode = pybullet.POSITION_CONTROL,
                targetPosition = action[i]
            ) #can also set maxTorque, positionGain, velocityGain (tunable)
        
        self._tilt_plane()
        pybullet.stepSimulation()

        self._update_agent_state()
        self._update_history()

        return self._get_observation()
    
    def _get_joint_states(self) -> tuple[list[float], list[float]]:
        """
        Function that returns a list of current positions and a list of current velocities of all the joints of the robot
        """
        positions = []
        velocities = []
        for joint_id in self._motor_joints_id:
            joint_state = pybullet.getJointState(self._robot_id, joint_id)
            positions.append(joint_state[0])  # position
            velocities.append(joint_state[1])  # velocity
        
        return positions, velocities

    def _get_gravity_vector(self) -> np.ndarray:
        """
        Returns the gravity vector in the robot's base frame.
        
        Returns:
            np.ndarray: 3D vector representing gravity direction in robot base frame
        """
        # World frame gravity vector (pointing down)
        gravity_world = np.array([0, 0, -1])
        
        # Get the rotation matrix from base orientation quaternion
        base_orientation = self._agent_state["base_orientation"]
        rot_matrix = np.array(pybullet.getMatrixFromQuaternion(base_orientation)).reshape(3, 3)
        
        # Transform gravity vector from world frame to base frame
        gravity_base = rot_matrix.T @ gravity_world
        
        return gravity_base   
    
    def _update_history(self):
        hist = []
        pos, vel = self._get_joint_states()
        hist.extend(pos)
        hist.extend(vel)

        self._joint_history.appendleft(hist)


    def _get_observation(self) -> np.ndarray:
        """
        - 0-2: gravity vector
        - 3: height of the robot
        - 4-6: linear velocity of the base
        - 7-9: angular velocity of the base
        - 10-21: positions of the joints
        - 22-33: velocities of the joints
        - 34-81: history
        - 82-93: previous action
        """

        obs = []
        positions, velocities = self._get_joint_states()

        obs.extend(self._get_gravity_vector())
        obs.append((self._agent_state["base_position"])[2])
        obs.extend(self._agent_state["linear_velocity"])
        obs.extend(self._agent_state["angular_velocity"])
        obs.extend(positions)
        obs.extend(velocities)
        obs.extend(self._joint_history[1])
        obs.extend(self._joint_history[4])
        obs.extend(self._previous_action)

        assert len(obs) == self._OBS_SPACE_SIZE, f"Expected 94 elements, got {len(obs)}"

        return np.array(obs, dtype=np.float32)

    def _update_agent_state(self) -> None:
        """
        Update position, orientation and linear and angular velocities
        """

        self._agent_state["base_position"], self._agent_state["base_orientation"] = pybullet.getBasePositionAndOrientation(self._robot_id)
        self._agent_state["linear_velocity"], self._agent_state["angular_velocity"] = pybullet.getBaseVelocity(self._robot_id)

        return  

    #@TODO: discuss the target state and rework this
    def _is_target_state(self, agent_state) -> bool:
        """
        Private method that returns wether the state of the agent is a target state (one in which to end the simulation) or not
        """

        base_pos = agent_state["base_position"]
        base_ori = agent_state["base_orientation"]
        height = base_pos[2]


        return (
            height <= self._target_state["min_height"]
        )
    
    #@TODO: check all edge cases
    def _is_terminated(self) -> bool:
        """
        Function that returns wether an episode was terminated artificially or timed out
        """
        return (self._step_counter >= self._MAX_EPISODE_LEN)

    def _get_info(self) -> dict:
        """
        Function that returns a dict containing the following fields:
            - height (of the body)
            - pitch: (of the base)
            - episode_step
        """
        return {
            "height": self._agent_state["base_position"][2],
            "pitch": pybullet.getEulerFromQuaternion(self._agent_state["base_orientation"])[1],
            "episode_step": self._step_counter
        }

    def _tilt_plane(self):
        """
        Smoothly tilts the plane in one direction
        """

        self._tilt_step += 1

        freq = 0.01
        max_angle = np.radians(6)

        tilt_x = max_angle * np.sin(freq * self._tilt_step + self._tilt_phase)
        tilt_y = max_angle * np.cos(freq * self._tilt_step + self._tilt_phase)

        quat = pybullet.getQuaternionFromEuler([tilt_x, tilt_y, 0])

        pybullet.resetBasePositionAndOrientation(
            self._plane_id,
            posObj=[0, 0, 0],
            ornObj=quat,
            physicsClientId=self.physics_client
        )

    #@TODO: implement a well thought reward function
    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Just a dummy generated with ChatGPT, to test the whole thing
        """
        height = self._agent_state["base_position"][2]
        roll, pitch, _ = pybullet.getEulerFromQuaternion(self._agent_state["base_orientation"])

        # Encourage upright posture
        uprightness = 1.0 - (abs(roll) + abs(pitch))
        height_bonus = max(0.0, height - 0.15)  # Don't reward when falling

        return height_bonus + uprightness