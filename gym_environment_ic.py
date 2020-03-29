import os
from argparse import ArgumentParser
import numpy as np
from gym_tools.gym import DataAssociationEnv
from gym_tools.tools import Gaussian
from ic_da.ic import IC
import matplotlib.pyplot as plt
from gym_tools.tools import get_movie_writer, get_dummy_context_mgr

def isNaN(num):
    return num != num

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

    x_list = []
    x_list.append(mean_prior)
    DA = IC(env.field_map, beta)

    all_rewards = []

    env.reset()
    action = env.action_space.sample()
    with env.movie_writer.saving(env.fig, args.movie_file, env.num_steps) if should_write_movie else get_dummy_context_mgr():
        for t in range(num_steps):
            print("STEP: ", t)
            env.render()
            # print(observation)
            observation, reward, done, info = env.step(action)
            print("REWARD: ", reward)
            all_rewards.append(reward)

            x = observation['robot_coordinates']
            z = observation['observations']
            lm_data = observation['LM_data']
            associated_data, all_data = DA.get_association(z, x, lm_data[:, 2].copy())
            print("All data: ", lm_data[:, 2], all_data)
            print("Associated data: ", associated_data)

            action_data = np.zeros(env.action_space.shape[0])
            action_data.fill(args.max_obs_per_time_step)

            counter = 0
            for obs in associated_data:
                if obs is not None:
                    action_data[int(obs)] = counter
                    counter += 1

            print("calculated observations_IDs: ", action_data)
            action = action_data
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    plt.cla()
    plt.plot(all_rewards)
    plt.show()
    plt.pause(5)


if __name__ == '__main__':
    main()

