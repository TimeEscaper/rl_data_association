import numpy as np
from gym_tools.map_generator import FieldMap
from gym_tools.tools import wrap_angle
from numpy.random import multivariate_normal as sample2d


def sense_landmarks(state, field_map, max_observations):
    """
    Observes num_observations of landmarks for the current time step.
    The observations will be in the front plan of the robot.

    :param state: The current state of the robot (format: np.array([x, y, theta])).
    :param field_map: The FieldMap object. This is necessary to extract the true landmark positions in the field.
    :param max_observations: The maximum number of observations to generate per time step.
    :return: np.ndarray or size num_observations x 3. Each row is np.array([range, bearing, lm_id]).
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(field_map, FieldMap)

    assert state.shape == (3,)

    M = field_map.num_landmarks
    noise_free_observations_list = list()
    for k in range(M):
        noise_free_observations_list.append(get_observation(state, field_map, k))
    noise_free_observation_tuples = [(x[0], np.abs(x[1]), int(x[2])) for x in noise_free_observations_list]

    dtype = [('range', float), ('bearing', float), ('lm_id', int)]
    noise_free_observations = np.array(noise_free_observations_list)
    noise_free_observation_tuples = np.array(noise_free_observation_tuples, dtype=dtype)

    ii = np.argsort(noise_free_observation_tuples, order='bearing')
    noise_free_observations = noise_free_observations[ii]
    noise_free_observations[:, 2] = noise_free_observations[:, 2].astype(int)

    c1 = noise_free_observations[:, 1] > -np.pi / 2.
    c2 = noise_free_observations[:, 1] <  np.pi / 2.
    ii = np.nonzero((c1 & c2))[0]

    if ii.size <= max_observations:
        return noise_free_observations[ii]
    else:
        return noise_free_observations[:max_observations]


def get_noisy_observation(state, field_map, lm_id, beta=(10., 10.)):
    """
    Generates a sample noisy observation given the current state of the robot and the marker id of which to observe.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param field_map: A map of the field.
    :param lm_id: The landmark id indexing into the landmarks list in the field map.
    :param beta: Diagonal of Standard deviations of the Observation noise (Q).
    :return: The observation to the landmark (format: np.array([range, bearing, landmark_id])).
             The bearing (in rad) will be in [-pi, +pi].
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(field_map, FieldMap)

    assert state.shape == (3,)

    beta = np.array(beta)
    beta[1] = np.deg2rad(beta[1])
    Q = np.diag([*(beta ** 2), 0])

    lm_id = int(lm_id)

    observation_dim = 3

    # Generate observation noise.
    observation_noise = sample2d(np.zeros(observation_dim), Q)

    dx = field_map.landmarks_poses_x[lm_id] - state[0]
    dy = field_map.landmarks_poses_y[lm_id] - state[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)
    bearing = np.arctan2(dy, dx) - state[2]

    noise_free_observation = np.array([distance, wrap_angle(bearing), lm_id])

    # Generate noisy observation as observed by the robot.
    noisy_observation = noise_free_observation + observation_noise

    return noisy_observation


def get_observation(state, field_map, lm_id):
    """
    Generates a sample observation given the current state of the robot and the marker id of which to observe.

    :param state: The current state of the robot (format: [x, y, theta]).
    :param field_map: A map of the field.
    :param lm_id: The landmark id indexing into the landmarks list in the field map.
    :return: The observation to the landmark (format: np.array([range, bearing, landmark_id])).
             The bearing (in rad) will be in [-pi, +pi].
    """

    assert isinstance(state, np.ndarray)
    assert isinstance(field_map, FieldMap)

    assert state.shape == (3,)

    lm_id = int(lm_id)

    dx = field_map.landmarks_poses_x[lm_id] - state[0]
    dy = field_map.landmarks_poses_y[lm_id] - state[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)
    bearing = np.arctan2(dy, dx) - state[2]

    return np.array([distance, wrap_angle(bearing), lm_id])


def get_state(field_map):
    """
    Generates robots state (format: [x, y, theta]).

    :param field_map: A map of the field.
    """

    assert isinstance(field_map, FieldMap)

    x = np.random.random_sample() * field_map.complete_size_x
    y = np.random.random_sample() * field_map.complete_size_y
    theta = np.random.random_sample() * 50

    return np.array([x, y, wrap_angle(theta)])
