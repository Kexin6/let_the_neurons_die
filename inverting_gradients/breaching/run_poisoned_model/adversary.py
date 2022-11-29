import logging, sys, math
import numpy as np

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logging.disable(logging.DEBUG)

def get_target_weights(initial_weights, r):
    w_t = []
    cross = False
    for w_x in initial_weights:
        logging.debug('w_x.shape: ')
        logging.debug(w_x.shape)
        n_row = w_x.shape[0]
        n_col = w_x.shape[1]
        w_size = n_col * n_row
        logging.debug('w_size: ')
        logging.debug(w_size)
        logging.debug('small num (r * w_size): ')
        logging.debug(r * w_size)
        logging.debug('large num: w_size - int(r * w_size): ')
        logging.debug(w_size - int(r * w_size))

        w_flat = w_x.flatten()
        logging.debug('w_flat: {}'.format(w_flat))
        w_sorted = w_flat[np.argsort(w_flat)]
        logging.debug('w_sorted: {}'.format(w_sorted))

        w_s = w_sorted[:int(r * w_size)]  # smallest N values, where N = r * #_of_rows
        w_l = w_sorted[int(r * w_size):]  # remaining larger ones
        logging.debug('w_s: {}'.format(w_s))
        logging.debug('w_l: {}'.format(w_l))

        if cross:
            logging.debug('entering cross')
            k = math.floor(r * n_col)
            logging.debug('k is {}'.format(k))

            w_s_new = w_s[:k * n_row]
            w_l_new = np.concatenate([w_s[k * n_row:], w_l])
            S = np.reshape(w_s_new, (n_row, k))
            L = np.reshape(w_l_new, (n_row, n_col - k))
            w_t_x = np.concatenate((L, S), axis=1)
            logging.debug('check L|S: {}'.format(w_t_x))
        else:
            logging.debug('entering NOT cross')
            j = math.floor(r * n_row)
            w_s_new = w_s[:j * n_col]
            w_l_new = np.concatenate([w_s[j * n_col:], w_l])
            S = np.reshape(w_s_new, (j, n_col))
            L = np.reshape(w_l_new, (n_row - j, n_col))
            w_t_x = np.concatenate((S, L))
            logging.debug('check S on L: {}'.format(w_t_x))
        w_t.append(w_t_x)
        cross = not cross
    return w_t
