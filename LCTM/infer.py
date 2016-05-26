import numpy as np
from numba import float64, jit, int16, boolean, int64, autojit

@jit("float64[:,:](float64[:,:], int16, float64[:,:])")
def segmental_viterbi(x, max_dur, pw=None):
	# From S&C NIPS 2004
	T, n_classes = x.shape
	scores = np.zeros([T, n_classes], np.float) - np.inf
	lengths = np.ones([T, n_classes], np.int)
	classes_prev = np.ones([T, n_classes], np.int)
	if pw is None:
		pw = np.zeros([n_classes, n_classes], np.float)

	# initialize first segment scores
	integral_scores = np.cumsum(x, 0)
	scores[0] = integral_scores[0]

	# -------- Forward -----------
	# Compute scores per timestep
	for t_end in range(1, T):
		# Compute scores per class
		for c in range(n_classes):
			# Compute over all durations
			best_dur = 0
			best_score = -np.inf
			best_class = -1
			for duration in range(1, min(t_end, max_dur)+1):
				t_start = t_end - duration
				current_segment = integral_scores[t_end, c] - integral_scores[t_start, c]
				
				if t_start == 0 and current_segment > best_score:
					best_dur = duration
					best_score = current_segment
					best_class = -1
					continue

				# Check if it is cheaper to create a new segment or stay in same class
				for c_prev in range(n_classes):
					if c_prev == c:
						continue

					# Previous segment, other class
					tmp = scores[t_start, c_prev] + current_segment + pw[c_prev,c]
					if tmp > best_score:
						best_dur = duration
						best_score = tmp
						best_class = c_prev


				# Add cost of curent frame to best previous cost
			scores[t_end, c] = best_score
			lengths[t_end, c] = best_dur
			# classes_prev[t_end, c] = best_class

	# Set nonzero entries to 0 for visualization
	# scores[scores<0] = 0
	scores[np.isinf(scores)] = 0

	# -------- Backward -----------
	classes = [scores[-1].argmax()]
	times = [T]
	t = T - lengths[-1, classes[-1]]
	while t > 0:
		class_prev = scores[t].argmax()
		length = lengths[t, class_prev]
		classes.insert(0, class_prev)
		times.insert(0, t)
		t -= length

	y_out = np.zeros(T, np.int)
	t = 0
	for c,l in zip(classes, times):
		y_out[t:t+l] = c
		t += l


	return scores


""" This version maximizes!!! """
@jit("float64[:,:](float64[:,:], int16, float64[:,:])")
def segmental_forward_normalized(x, max_segs, pw=None):
	# Assumes segment function is normalized by duration: f(x)= 1/d sum_t'=t^t+d x_t'
	T, n_classes = x.shape
	scores = np.zeros([max_segs, T, n_classes], np.float) - np.inf
	if pw is None:
		pw = np.zeros([n_classes, n_classes], np.float)

	integral_scores = np.cumsum(x, 0)

	# Intial scores
	scores[0] = integral_scores.copy()
	starts = np.zeros([max_segs, n_classes], np.int)+1

	# Compute scores for each segment in sequence
	for m in range(1, max_segs):
		# Compute score for each class
		for c in range(n_classes):
			best_score = -np.inf
			for c_prev in range(n_classes):
				if c_prev == c:
					continue

				# Compute scores for each timestep
				for t in range(1, T):
					new_segment = integral_scores[t, c] - integral_scores[starts[m,c], c]

					# Previous segment, other class
					score_change = scores[m-1, t, c_prev] + pw[c_prev,c]
					if score_change > best_score:
						best_score = score_change
						starts[m,c] = t

					# Add cost of curent frame to best previous cost
					scores[m, t, c] = best_score + new_segment

	# Set nonzero entries to 0 for visualization
	scores[np.isinf(scores)] = 0

	return scores	


def sparsify_incoming_pw(pw):
    # Output is INCOMING transitions
    n_classes = pw.shape[0]
    valid = np.nonzero(~np.isinf(pw.T))
    sparse_idx = [[] for i in range(n_classes)]
    for i,j in zip(valid[0], valid[1]):
        sparse_idx[i] += [j]

    return sparse_idx
    
@jit("float64[:,:](float64[:,:], int16, float64[:,:])")
def segmental_forward_eccv(x, max_segs, pw=None):
    # Assumes segment function is additive: f(x)=sum_t'=t^t+d x_t'
    T, n_classes = x.shape
    scores = np.zeros([max_segs, T, n_classes], np.float) - np.inf
    lengths = np.zeros([max_segs, T, n_classes], np.float)
    if pw is None:
        pw = np.log(1 - np.eye(n_classes))

    # initialize first segment scores
    scores[0] = np.cumsum(x, 0)

    # Compute scores per segment
    for m in range(1, max_segs):
        # scores[m, 0, c] = scores[m-1, 0, c]
        # Compute scores per timestep
        for t in range(1, T):
            # Compute scores per class
            for c in range(n_classes):
                # Score for staying in same segment
                best_prev = scores[m, t-1, c]
                length = lengths[m,t-1,c] + 1

                # Check if it is cheaper to create a new segment or stay in same class
                for c_prev in range(n_classes):
                    # Previous segment, other class
                    tmp = scores[m-1, t-1, c_prev] + pw[c_prev,c]
                    if tmp > best_prev:
                        best_prev = tmp
                        length = 1

                # Add cost of curent frame to best previous cost
                scores[m, t, c] = best_prev + x[t, c]
                lengths[m,t,c] = length

    # Set nonzero entries to 0 for visualization
    scores[np.isinf(scores)] = 0

    return scores


@jit("int16[:,:](float64[:,:], float64[:,:])")
def segmental_backward_eccv(scores, pw=None):

    n_segs, T, n_classes = scores.shape

    if pw is None:
        pw = np.log(1 - np.eye(n_classes))

    best_scores = scores[:,-1].max(1)
    n_segs = np.argmax(best_scores)

    # Start at end
    seq_c = [scores[n_segs, -1].argmax()] # Class
    seq_t = [T] # Time
    m = n_segs

    for t in range(T, -1, -1):
        # Scores of previous timestep in current segment
        score_same = scores[m, t-1, seq_c[0]]
        score_diff = scores[m-1, t-1] + pw[:,seq_c[0]]

        # Check if it's better to stay or switch segments
        if any(score_diff > score_same):
            next_class = score_diff.argmax()
            score_diff = score_diff[next_class]
            seq_c.insert(0, next_class)
            seq_t.insert(0, t)
            m -= 1

            if m == 0:
                break
    seq_t.insert(0,0)

    if m != 0:
        print("# segs (m) not zero!", m)

    y_out = np.empty(T, np.int)
    for i in range(len(seq_c)):
        y_out[seq_t[i]:seq_t[i+1]] = seq_c[i]

    return y_out    


def segmental_inference(x, max_segs, pw=None, normalized=False, verbose=False):
    scores = segmental_forward_eccv(x, max_segs, pw)
    return segmental_backward_eccv(scores, pw)

@jit("float64[:,:](float64[:,:], int16, float64[:,:], float64[:], float64[:,:])")
def segmental_forward_oracle(x, max_segs, pw, y_oracle, oracle_valid):
    # Assumes segment function is additive: f(x)=sum_t'=t^t+d x_t'
    T, n_classes = x.shape
    scores = np.zeros([max_segs, T, n_classes], np.float) - np.inf
    lengths = np.zeros([max_segs, T, n_classes], np.float)
    if pw is None:
        pw = np.log(1 - np.eye(n_classes))

    # initialize first segment scores
    scores[0] = np.cumsum(x, 0)

    # Compute scores per segment
    for m in range(1, max_segs):
        # scores[m, 0, c] = scores[m-1, 0, c]
        # Compute scores per timestep
        for t in range(1, T):
            # Compute scores per class
            for c in range(n_classes):
                # Score for staying in same segment
                best_prev = scores[m, t-1, c]
                length = lengths[m,t-1,c] + 1

                # Check if it is cheaper to create a new segment or stay in same class
                for c_prev in range(n_classes):
                    # Previous segment, other class
                    tmp = scores[m-1, t-1, c_prev] + pw[c_prev,c]
                    if tmp > best_prev:
                        best_prev = tmp
                        length = 1

                if oracle_valid[y_oracle[t], c] == 0:
                    best_prev = -np.inf

                # Add cost of curent frame to best previous cost
                scores[m, t, c] = best_prev + x[t, c]
                lengths[m,t,c] = length

    # Set nonzero entries to 0 for visualization
    scores[np.isinf(scores)] = 0

    return scores


def segmental_inference_oracle(x, max_segs, pw, y_oracle, oracle_valid):
    scores = segmental_forward_oracle(x, max_segs, pw, y_oracle, oracle_valid)
    return segmental_backward_eccv(scores, pw)


# Toy examples
if 0:
	from LCTM.energies import pairwise
	x = np.zeros([100, 3])
	y = np.zeros([100], np.int)
	y[:30] = 0
	y[30:60] = 1
	y[60:] = 2
	x[:30,0] = 1
	x[30:60,1] = 1
	x[60:,2] = 1
	pw = np.zeros([3,3])-1000
	pw[0,1] = 1
	pw[1,2] = 1

	# x = -np.log(np.maximum(x, 0.05))
	# pw = -np.log(np.maximum(pw, 0.05))

	scores = segmental_forward_normalized(x, 10)
	y_ = segmental_backward(scores)

	# x = np.random.random([7, 1000]);
	# %timeit tmp = segmental_forward(x.T, 20)
	# %timeit tmp_b = segmental_forward_pw(x.T, pw, 20)
