import contextlib
import os
from argparse import ArgumentParser
import numpy as np
import gym
from gym_tools.gym import DataAssociationEnv
import matplotlib.pyplot as plt
from progress.bar import FillingCirclesBar
from gym_tools.map_generator import FieldMap
from gym_tools.plot import plot_observations, plot2dcov, plot_field, plot_robot, get_plots_figure
from gym_tools.tools import get_movie_writer, get_dummy_context_mgr
from tools.data import generate_data as generate_input_data
from tools.data import load_data
from gym_tools.tools import Gaussian
from sqrtSAM import SqrtSAM
from gym_tools.observation_generator import get_state
from tools.task import wrap_angle

from rl_da.da_estimators import ReinforceEstimator

def get_cli_args():
    parser = ArgumentParser('Perception in Robotics FP')
    parser.add_argument('-i',
                        '--input-data-file',
                        type=str,
                        action='store',
                        help='File with generated data to simulate the filter '
                             'against. Supported format: "npy", and "mat".')
    parser.add_argument('-n',
                        '--num-steps',
                        type=int,
                        action='store',
                        help='The number of time steps to generate data for the simulation. '
                             'This option overrides the data file argument.',
                        default=100)
    parser.add_argument('-f',
                        '--filter',
                        dest='filter_name',
                        choices=['ekf', 'sam'],
                        action='store',
                        help='The slam filter use for the SLAM problem.',
                        default='sam')
    parser.add_argument('-a',
                        '--alphas',
                        type=float,
                        nargs=4,
                        metavar=('A1', 'A2', 'A3', 'A4'),
                        action='store',
                        help='Diagonal of Standard deviations of the Transition noise in action space (M_t).',
                        default=(0.05, 0.001, 0.05, 0.01))
    parser.add_argument('-b',
                        '--beta',
                        type=float,
                        nargs=2,
                        metavar=('range', 'bearing (deg)'),
                        action='store',
                        help='Diagonal of Standard deviations of the Observation noise (Q).',
                        default=(10., 10.))
    parser.add_argument('--dt', type=float, action='store', help='Time step (in seconds).', default=0.1)
    parser.add_argument('-s', '--animate', action='store_true',
                        help='Show and animation of the simulation, in real-time.')
    parser.add_argument('--plot-pause-len',
                        type=float,
                        action='store',
                        help='Time (in seconds) to pause the plot animation for between frames.',
                        default=0.01)
    parser.add_argument('--num-landmarks-per-side',
                        type=int,
                        help='The number of landmarks to generate on one side of the field.',
                        default=4)
    parser.add_argument('--max-obs-per-time-step',
                        type=int,
                        help='The maximum number of observations to generate per time step.',
                        default=2)
    parser.add_argument('--data-association',
                        type=str,
                        choices=['known', 'ml', 'jcbb'],
                        default='known',
                        help='The type of data association algorithm to use during the update step.')
    parser.add_argument('--update-type',
                        type=str,
                        choices=['batch', 'sequential'],
                        default='batch',
                        help='Determines how to perform update in the SLAM algorithm.')
    parser.add_argument('-m',
                        '--movie-file',
                        type=str,
                        help='The full path to movie file to write the simulation animation to.',
                        default=None)
    parser.add_argument('--movie-fps',
                        type=float,
                        action='store',
                        help='The FPS rate of the movie to write.',
                        default=10.)
    parser.add_argument('--solver',
                        type=str,
                        help='Least squares solving method: build-in numpy or Cholesky factorization with back-substitution.',
                        choices=['numpy', 'cholesky'],
                        default='numpy')
    parser.add_argument('-r', '--random', action='store_true',
                        help='Generate random robots state each step.')
    return parser.parse_args()


def validate_cli_args(args):
    if args.input_data_file and not os.path.exists(args.input_data_file):
        raise OSError('The input data file {} does not exist.'.format(args.input_data_file))

    if not args.input_data_file and not args.num_steps:
        raise RuntimeError('Neither `--input-data-file` nor `--num-steps` were present in the arguments.')


def main():
    args = get_cli_args()
    validate_cli_args(args)

    alphas = np.array(args.alphas)
    beta = np.array(args.beta)
    solver = args.solver

    mean_prior = np.array([180., 50., 0.])
    Sigma_prior = 1e-12 * np.eye(3, 3)
    initial_state = Gaussian(mean_prior, Sigma_prior)

    should_show_plots = True if args.animate else False
    should_generate_random_state = True if args.random else False
    should_write_movie = True if args.movie_file else False
    should_update_plots = True if should_show_plots or should_write_movie else False
    num_steps = args.num_steps

    env = DataAssociationEnv(input_data_file=args.input_data_file, solver=solver, n_possible_observations=args.max_obs_per_time_step,
                             n_possible_LMs=args.num_landmarks_per_side*2, num_landmarks_per_side=args.num_landmarks_per_side, should_show_plots=should_show_plots,
                             should_write_movie=should_write_movie, num_steps=num_steps, alphas=alphas,
                             beta=beta, random_state_generator=should_generate_random_state, dt=args.dt, movie_file=args.movie_file)

    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(num_steps):
    #         env.render()
    #         print(observation)
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             break

    estimator = ReinforceEstimator(observation_dim=1,
                                   n_observations=args.max_obs_per_time_step,
                                   max_landmarks=args.num_landmarks_per_side*2,
                                   hidden_state_size=10)


    env.reset()
    n_skipped = 0
    step_count = 0

    total_rewards = []
    log_probabilities = []
    rewards = []

    for i_episode in range(num_steps):
        env.render()

        if n_skipped <= 2:
            n_skipped += 1
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            continue

        # print(observation)
        action, log_probability = estimator.get_action(create_distance_matrix(observation))
        observation, reward, done, info = env.step(action)

        print("Reward: " + str(reward))

        total_rewards.append(reward)
        log_probabilities.append(log_probability)
        rewards.append(reward)

        step_count += 1
        if step_count % 5 == 0:
            estimator.update_policy(rewards, log_probabilities)

    env.close()

def create_distance_matrix(observation):
    measurements = observation['observations']
    landmarks = observation['LM_data']
    distance_matrix = np.zeros((landmarks.shape[0], measurements.shape[0]))
    for i in range(landmarks.shape[0]):
        for j in range(measurements.shape[0]):
            observed_coords = get_landmark_position(observation['robot_coordinates'],
                                                    measurements[j][0], measurements[j][1])
            predicted_coords = landmarks[i][0:2]
            distance_matrix[i, j] = np.sqrt(
                (predicted_coords[0] - observed_coords[0])**2 + (predicted_coords[1] - observed_coords[1])**2)
    return distance_matrix

def get_landmark_position(state, range, bearing):
    angle = wrap_angle(state[2] + bearing)
    x_rel = range * np.cos(angle)
    y_rel = range * np.sin(angle)
    x = x_rel + state[0]
    y = y_rel + state[1]
    return np.array([x, y])


if __name__ == '__main__':
    main()

