import numpy as np
import dsi

def init_bgn_hist_cpp(filename_noise_pos, filename_noise_neg):
    """ Load the distribution of the noise for the cpp simulator
        Pick randomly one noise distribution for each pixel and initialize also randomly the phases of the
        background noise
        Args:
            filename_noise_pos: path of the positive noise's filename
            filename_noise_neg: path of the negative noise's filename
        """
    noise_pos = np.load(filename_noise_pos)
    if len(noise_pos) == 0: raise NameError(filename_noise_pos + " is not correct")
    noise_pos = np.reshape(noise_pos, (noise_pos.shape[0] * noise_pos.shape[1], noise_pos.shape[2]))
    ind_n = np.where(noise_pos[:, -2] == 0)
    noise_pos[ind_n, 1] = 1
    noise_pos[ind_n, -2] = 1
    div = np.tile(noise_pos[:, -2], [noise_pos.shape[1], 1])
    noise_pos = noise_pos / div.transpose()

    noise_neg = np.load(filename_noise_neg)
    if len(noise_neg) == 0: raise NameError(filename_noise_neg + " is not correct")
    noise_neg = np.reshape(noise_neg, (noise_neg.shape[0] * noise_neg.shape[1], noise_neg.shape[2]))
    ind_n = np.where(noise_neg[:, -2] == 0)
    noise_neg[ind_n, 1] = 1
    noise_neg[ind_n, -2] = 1
    div = np.tile(noise_neg[:, -2], [noise_neg.shape[1], 1])
    noise_neg = noise_neg / div.transpose()
    dsi.initNoise(noise_pos, noise_neg)