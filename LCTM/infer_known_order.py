import numpy as np
from numba import float64, jit, int16, boolean, int64

@jit("float64[:,:](float64[:,:], int32[:])")
def segmental_forward_known(x, segs):
    # Assumes segment function is additive: f(x)=sum_t'=t^t+d x_t'
    T, n_classes = x.shape
    n_segs = len(segs)
    LARGE_NUMBER = 99999.
    scores = np.zeros([n_segs, T], np.float) - LARGE_NUMBER

    # initialize first segment scores
    scores[0] = np.cumsum(x[:,segs[0]], 0)

    # Compute scores per segment
    for m in range(1, n_segs):
        # Compute scores per timestep
        for t in range(1, T):
            c = segs[m]

            # Score for staying in same segment or coming from previous
            best_same = scores[m,   t-1]
            best_prev = scores[m-1, t-1]

            # Add cost of curent frame to best incoming cost
            if best_same > best_prev:
                scores[m, t] = best_same + x[t, c]
                # print(m, t)
            else:
                scores[m, t] = best_prev + x[t, c]
                # print(m, t)

    # Set nonzero entries to 0 for visualization
    scores[scores<0] = 0

    return scores

# scores = segmental_forward_known(x, segs)


@jit("int16[:,:](float64[:,:], int32[:])")
def segmental_backward_known(scores, segs):
    n_segs, T = scores.shape

    # Start at end
    seq_c = [segs[-1]] # Class
    seq_t = [T] # Time
    m = n_segs-1

    for t in range(T, 0, -1):
        # Scores of previous timestep in current segment
        score_same = scores[m, t-1]
        score_prev = scores[m-1, t-1]

        # Check if it's better to stay or switch segments
        if score_prev >= score_same:
            next_class = segs[m-1]
            seq_c += [next_class]
            seq_t += [t-1]
            m -= 1

            if m == 0:
                break
    seq_t += [0]

    if m != 0:
        print("# segs (m) not zero!", m)

    seq_c = list(reversed(seq_c))
    seq_t = list(reversed(seq_t))

    y_out = np.empty(T, np.int)
    for i in range(len(seq_c)):
        y_out[seq_t[i]:seq_t[i+1]] = seq_c[i]

    return y_out    

def infer_known_ordering(x, segs):

    scores = segmental_forward_known(x, segs)
    y_out = segmental_backward_known(scores, segs)

    return y_out


