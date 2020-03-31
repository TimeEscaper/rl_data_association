import numpy as np
import contextlib
from matplotlib import animation as anim


def wrap_angle(angle):
    """
    Wraps the given angle to the range [-pi, +pi].

    :param angle: The angle (in rad) to wrap (can be unbounded).
    :return: The wrapped angle (guaranteed to in [-pi, +pi]).
    """

    pi2 = 2 * np.pi

    while angle < -np.pi:
        angle += pi2

    while angle >= np.pi:
        angle -= pi2

    return angle


def get_movie_writer(should_write_movie, title, movie_fps, plot_pause_len):
    """
    :param should_write_movie: Indicates whether the animation of SLAM should be written to a movie file.
    :param title: The title of the movie with which the movie writer will be initialized.
    :param movie_fps: The frame rate of the movie to write.
    :param plot_pause_len: The pause durations between the frames when showing the plots.
    :return: A movie writer that enables writing MP4 movie with the animation from SLAM.
    """

    get_ff_mpeg_writer = anim.writers['ffmpeg']
    metadata = dict(title=title, artist='matplotlib', comment='PS3: EKF SLAM')
    movie_fps = min(movie_fps, float(1. / plot_pause_len))

    return get_ff_mpeg_writer(fps=movie_fps, metadata=metadata)


@contextlib.contextmanager
def get_dummy_context_mgr():
    """
    :return: A dummy context manager for conditionally writing to a movie file.
    """
    yield None


class Gaussian(object):
    """
    Represents a multi-variate Gaussian distribution representing the state of the robot.
    """

    def __init__(self, mu, Sigma):
        """
        Sets the internal mean and covariance of the Gaussian distribution.

        :param mu: A 1-D numpy array (size 3x1) of the mean (format: [x, y, theta]).
        :param Sigma: A 2-D numpy ndarray (size 3x3) of the covariance matrix.
        """

        assert isinstance(mu, np.ndarray)
        assert isinstance(Sigma, np.ndarray)
        assert Sigma.shape == (3, 3)

        if not isinstance(mu, np.ndarray):
            raise TypeError('mu should be of type np.ndarray.')

        if mu.ndim < 1:
            raise ValueError('The mean must be a 1D numpy ndarray of size 3.')
        elif mu.shape == (3,):
            # This transforms the 1D initial state mean into a 2D vector of size 3x1.
            mu = mu[np.newaxis].T
        elif mu.shape != (3, 1):
            raise ValueError('The mean must be a vector of size 3x1.')
        if not isinstance(Sigma, np.ndarray):
            raise TypeError('Sigma should be of type np.ndarray.')

        self.mu = mu
        self.Sigma = Sigma


def get_gaussian_statistics(samples):
    """
    Computes the parameters of the samples assuming the samples are part of a Gaussian distribution.

    :param samples: The samples of which the Gaussian statistics will be computed (shape: N x 3).
    :return: Gaussian object from utils.objects with the mean and covariance initialized.
    """

    assert isinstance(samples, np.ndarray)
    assert samples.shape[1] == 3

    # Compute the mean along the axis of the samples.
    mu = np.mean(samples, axis=0)

    # Compute mean of angles.
    angles = samples[:, 2]
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    mu[2] = np.arctan2(sin_sum, cos_sum)

    # Compute the samples covariance.
    mu_0 = samples - np.tile(mu, (samples.shape[0], 1))
    mu_0[:, 2] = np.array([wrap_angle(angle) for angle in mu_0[:, 2]])
    Sigma = mu_0.T @ mu_0 / samples.shape[0]

    return Gaussian(mu, Sigma)


def get_landmark_coords_global(x, z):
    """
    Transforms local observation to global coordinates.
    :param x: State.
    :param z: Observation in local frame.
    :return:
    """

    mx = x[0] + z[0]*np.cos(wrap_angle(z[1]) + wrap_angle(x[2]))
    my = x[1] + z[0]*np.sin(wrap_angle(z[1]) + wrap_angle(x[2]))

    return np.array([mx, my])


def get_landmark_coords_local(x, m):
    """
    Transforms global coordinates to local observation.
    :param x: State.
    :param z: Observation in global frame (landmark coordinates).
    :return:
    """

    dx = m[0] - x[0]
    dy = m[1] - x[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)
    bearing = np.arctan2(dy, dx) - wrap_angle(x[2])

    return np.array([distance, wrap_angle(bearing)])
