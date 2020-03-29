import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym_tools.map_generator import FieldMap
from gym_tools.plot import plot_observations, plot2dcov, plot_field, plot_robot, get_plots_figure
from gym_tools.tools import get_movie_writer, get_dummy_context_mgr, get_landmark_coords_global
import matplotlib.pyplot as plt
from gym_tools.observation_generator import get_state
from gym_tools.observation_generator import get_observation, get_noisy_observation, get_state
from tools.data import generate_data as generate_input_data
from tools.data import load_data
from gym_tools.tools import Gaussian
from sqrtSAM import SqrtSAM
from gym_tools.tools import wrap_angle

DEFAULT_WIDTH_SIZE = 600
DEFAULT_HEIGHT_SIZE = 400


class DataAssociationEnv(gym.Env):
    def __init__(self, input_data_file, solver, n_possible_observations=10, n_possible_LMs=10, num_landmarks_per_side=5,
                 should_show_plots=True, should_write_movie=False, num_steps=100, alphas=(0.05, 0.001, 0.05, 0.01),
                 beta=(10., 10.), random_state_generator=True, dt=0.1, movie_file="Gym.mp4"):

        # Current observation
        # (before the new observation we have info about LM coordinates and their IDs if they were observed previously)
        self.observations = np.zeros((n_possible_observations, 2))  # format: [range, bearing]
        self.robot_coordinates = np.zeros(3)  # format: [x, y, theta]

        self.noisy_observations = np.zeros((0, 3))  # format: [range, bearing, ID]
        self.noise_free_observations = np.zeros((0, 3))  # format: [range, bearing, ID]
        self.observations_IDs = []

        # Observed landmarks
        self.LM_data = np.zeros((n_possible_LMs, 3))  # format: [x, y, ID]. For unknown ID: -1
        # self.LM_IDs = np.zeros((n_possible_LMs, 1))  # format: [ID]

        self.data_association = np.zeros((n_possible_observations, 1))  # format: [ID]
        self.n_possible_LMs = n_possible_LMs
        self.n_possible_observations = n_possible_observations

        self.viewer = None
        self._viewers = {}

        self.dt = 1.0 / 30.
        self.t = 0

        self.movie_file = movie_file

        self.should_write_movie = should_write_movie
        self.should_show_plots = should_show_plots

        self.num_steps = num_steps
        self.beta = beta

        '''
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        '''

        self.action_space = spaces.Box(0, n_possible_LMs, shape=self.data_association.shape, dtype='float32')

        self.observation_space = spaces.Dict(dict(
            observations=spaces.Box(-np.inf, np.inf, shape=self.observations.shape, dtype='float32'),
            robot_coordinates=spaces.Box(-np.inf, np.inf, shape=self.robot_coordinates.shape, dtype='float32'),
            LM_data=spaces.Box(-np.inf, np.inf, shape=self.LM_data.shape, dtype='float32')))

        self.field_map = FieldMap(num_landmarks_per_side)

        self.fig = get_plots_figure(self.should_show_plots, self.should_write_movie)
        self.movie_writer = get_movie_writer(should_write_movie, 'Simulation SLAM', int(np.round(1.0 / self.dt)), 0.01)

        alphas = np.array(alphas)
        beta = np.array(beta)

        mean_prior = np.array([180., 50., 0.])
        Sigma_prior = 1e-12 * np.eye(3, 3)
        initial_state = Gaussian(mean_prior, Sigma_prior)

        self.random_state_generator = random_state_generator

        if not random_state_generator:
            if input_data_file:
                self.data_sam = load_data(input_data_file)
            elif num_steps:
                # Generate data, assuming `--num-steps` was present in the CL args.
                self.data_sam = generate_input_data(initial_state.mu.T,
                                                    num_steps,
                                                    num_landmarks_per_side,
                                                    n_possible_observations,
                                                    alphas,
                                                    beta,
                                                    dt)
            else:
                raise RuntimeError('')

        self.sam = SqrtSAM(mean_prior,
                           Sigma_prior,
                           alphas ** 2,
                           np.diag([beta[0] ** 2, np.deg2rad(beta[1]) ** 2]),
                           solver=solver)

        self.unique_observations_IDs = []
        self.LM_data_flex = np.zeros((0, 3))

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self._set_action(action)
        reward = self._compute_reward()
        observation = self._get_observation()

        done = False
        self.t += 1

        info = {'reward': reward}

        return observation, reward, done, info

    def reset(self):
        self.observations.fill(0)
        self.robot_coordinates.fill(0)
        self.LM_data.fill(0)
        self.LM_data[:, 2].fill(-1)  # No ID information yet

        self.observations_IDs = []

        self.noisy_observations = np.zeros((0, 3))
        self.noise_free_observations = np.zeros((0, 3))

        return self.observations, self.robot_coordinates, self.LM_data

    def render(self, mode='live', screen_width=DEFAULT_WIDTH_SIZE, screen_height=DEFAULT_HEIGHT_SIZE):
        with self.movie_writer.saving(self.fig, self.movie_file, self.num_steps) if self.should_write_movie else get_dummy_context_mgr():

            plt.cla()
            plot_field(self.field_map, self.noise_free_observations[:, 2])
            # plot_robot(get_state(self.field_map))
            plot_robot(self.robot_coordinates)

            plot_observations(self.robot_coordinates,
                              self.noise_free_observations,
                              self.noisy_observations)

            if not self.random_state_generator:
                tp1 = self.t + 1
                t = self.t
                # plt.plot(self.data_sam.debug.real_robot_path[1:tp1, 0], self.data_sam.debug.real_robot_path[1:tp1, 1], 'm')
                # plt.plot(self.data_sam.debug.noise_free_robot_path[1:tp1, 0], self.data_sam.debug.noise_free_robot_path[1:tp1, 1], 'g')

                # plt.plot([self.data_sam.debug.real_robot_path[t, 0]], [self.data_sam.debug.real_robot_path[t, 1]], '*r')
                # plt.plot([self.data_sam.debug.noise_free_robot_path[t, 0]], [self.data_sam.debug.noise_free_robot_path[t, 1]], '*g')

            if self.should_show_plots:
                # Draw all the plots and pause to create an animation effect.
                plt.draw()
                plt.pause(0.01)

            if self.should_write_movie:
                self.movie_writer.grab_frame()

    def close(self):
        '''
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}
        '''

        pass

    def _set_action(self, action):
        action = action.copy()
        for i in range(action.shape[0]):
            self.data_association[i] = action[i]

    def _compute_reward(self):
        n_correct = 0
        n_incorrect = 0
        counter = 0
        for observation in self.observations_IDs:
            if observation == self.data_association[counter]:
                n_correct += 1
            else:
                n_incorrect += 1
            counter += 1

        if n_incorrect + n_correct != 0:
            return n_correct/(n_incorrect + n_correct)
        else:
            return 10000000

    def _get_observation(self):
        self.observations_IDs = []
        self.LM_data_flex = np.zeros((0, 3))

        self.observations.fill(0)
        self.robot_coordinates.fill(0)
        self.LM_data.fill(0)
        self.LM_data[:, 2].fill(-1)  # No ID information yet

        self.noisy_observations = np.zeros((0, 3))
        self.noise_free_observations = np.zeros((0, 3))

        if self.random_state_generator:
            self.robot_coordinates = get_state(self.field_map)
            n_observations = int(np.round(np.random.random_sample() * self.n_possible_observations))

            flag = True
            for n_obs in range(n_observations):
                id = int(np.round(np.random.random_sample() * (self.n_possible_LMs - 1)))
                while id in self.observations_IDs:
                    id = int(np.round(np.random.random_sample() * (self.n_possible_LMs - 1)))
                noise_free_observation = get_observation(self.robot_coordinates, self.field_map, id)
                noisy_observation = get_noisy_observation(self.robot_coordinates, self.field_map, id, self.beta)

                self.noisy_observations = np.vstack((self.noisy_observations, noisy_observation))
                self.noise_free_observations = np.vstack((self.noise_free_observations, noise_free_observation))

                self.observations_IDs.append(id)

                self.observations[n_obs] = noisy_observation[:2]
                # self.LM_data[id, :2] = get_landmark_coords_global(self.robot_coordinates, self.noisy_observations)
                # We assume correct DA on previous step

                if flag:
                    for old_id in self.unique_observations_IDs:
                        self.LM_data_flex = np.vstack(
                            (self.LM_data_flex, np.array([self.field_map.landmarks_poses_x[old_id],
                                                      self.field_map.landmarks_poses_y[old_id], old_id])))
                    flag = False

                if id not in self.unique_observations_IDs:
                    '''
                    self.LM_data_flex = np.vstack((self.LM_data_flex, np.array(
                        [self.field_map.landmarks_poses_x[id], self.field_map.landmarks_poses_y[id], -1])))
                    '''
                    self.unique_observations_IDs.append(id)

            for i in range(self.LM_data_flex.shape[0]):
                self.LM_data[i] = self.LM_data_flex[i]

            # self.LM_coordinates = ... (for SLAM - taking from theta)

        else:
            map = self.sam.get_current_map()
            counter = 0
            ids = list(self.sam.landmarks_index_map_.keys())
            for landmark, _ in map:
                self.LM_data[counter, :2] = landmark
                self.LM_data[counter, 2] = ids[counter]
                counter += 1

            # Control at the current step.
            u = self.data_sam.filter.motion_commands[self.t]

            # Observation at the current step.
            z = self.data_sam.filter.observations[self.t]

            # SLAM predict(u)
            self.sam.predict(u)

            # SLAM update
            self.sam.update(z)

            self.robot_coordinates = np.array([self.sam.states_[-1, 0], self.sam.states_[-1, 1], wrap_angle(self.sam.states_[-1, 2])])
            # self.robot_coordinates = self.data_sam.debug.real_robot_path[self.t]  # Real states (not available for DA algorithms)

            self.noise_free_observations = self.data_sam.debug.noise_free_observations[self.t]
            self.noisy_observations = self.data_sam.filter.observations[self.t]

            for id in self.noisy_observations[:, 2]:
                self.observations_IDs.append(id)

            self.observations = self.noisy_observations[:, :2]

        return {
            'observations': self.observations.copy(),
            'robot_coordinates': self.robot_coordinates.copy(),
            'LM_data': self.LM_data.copy(),
        }
