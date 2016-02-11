import numpy as np
from numba import float64, jit, int16, boolean, int64

@jit("float64[:,:](float64[:,:], int16, float64[:,:], float64[:], float64[:])")
def segmental_forward(x, max_segs, pw=None, start_prior=None, end_prior=None):
	# Assumes segment function is additive: f(x)=sum_t'=t^t+d x_t'
	T, n_classes = x.shape
	LARGE_NUMBER = 99999.
	scores = np.zeros([max_segs, T, n_classes], np.float) - LARGE_NUMBER
	if pw is None:
		pw = np.zeros([n_classes, n_classes], np.float)

	# initialize first segment scores
	scores[0] = np.cumsum(x, 0)

	if start_prior is not None:
		scores[0] += start_prior.T

	# Compute scores per segment
	for m in range(1, max_segs):
		# Compute scores per timestep
		for t in range(1, T):
			# Compute scores per class
			for c in range(n_classes):
				# Score for staying in same segment
				best_prev = scores[m, t-1, c]

				# Check if it is cheaper to create a new segment or stay in same class
				for c_prev in range(n_classes):
					if c_prev == c:
						continue

					# Previous segment, other class
					tmp = scores[m-1, t-1, c_prev] + pw[c_prev,c]
					if tmp > best_prev:
						best_prev = tmp

				# Add cost of curent frame to best previous cost
				scores[m, t, c] = best_prev + x[t, c]

	if end_prior is not None:
		scores[-1] += end_prior.T

	# Set nonzero entries to 0 for visualization
	scores[scores<0] = 0

	return scores

# @jit("float64[:,:](float64[:,:], float64[:,:], int16, float64[:], float64[:])")
# def segmental_forward_normalized(x, max_segs, pw=None, start_prior=None, end_prior=None):
# 	# Assumes segment function is normalized by duration: f(x)= 1/d sum_t'=t^t+d x_t'
# 	T, n_classes = x.shape
# 	LARGE_NUMBER = 99999.
# 	scores = np.zeros([max_segs, T, n_classes], np.float) - LARGE_NUMBER
# 	durations = np.ones([max_segs, T, n_classes], np.int)
# 	if pw is None:
# 		pw = np.zeros([n_classes, n_classes], np.float)

# 	integral_scores = np.cumsum(x, 0)

# 	# initialize first segment scores
# 	scores[0][0,:] = x[0,:]
# 	for t in range(1, T):
# 		scores[0][t] = scores[0][t-1]+x[t]
# 		durations[0][t] = t+1
# 	scores[0] /= durations[0]

# 	if start_prior is not None:
# 		scores[0] += start_prior.T

# 	# Compute scores for each segment in sequence
# 	for m in range(1, max_segs):
# 		# Compute scores for each timestep
# 		for t in range(1, T):
# 			# Compute score for each class
# 			for c in range(n_classes):
# 				# Score for staying in same segment
# 				best_prev = scores[m, t-1, c] / durations[m,t-1,c]
# 				t_prev = t - durations[m,t-1,c]
# 				score_stay = integral_scores[m, t, c] - integral_scores[m, t_prev, c]
# 				# durations[m,t,c] = durations[m,t-1,c] + 1

# 				# Check if it is cheaper to create a new segment or stay in same class
# 				for c_prev in range(n_classes):
# 					if c_prev == c:
# 						continue

# 					# Previous segment, other class
# 					tmp = scores[m-1, t-1, c_prev] + pw[c_prev,c]
# 					if tmp > best_prev:
# 						best_prev = tmp
# 						durations[m,t,c] = 1
# 						switch = True

# 				if switch:
# 					durations[m,t-1,c]	

# 				# Add cost of curent frame to best previous cost
# 				scores[m, t, c] = best_prev + x[t, c]
# 		scores[m] /= durations[m]

# 	if end_prior is not None:
# 		scores[-1] += end_prior.T

# 	# Set nonzero entries to 0 for visualization
# 	scores[scores<0] = 0

# 	return scores	

# x = np.random.random([7, 1000]);
# %timeit tmp = segmental_forward(x.T, 20)
# %timeit tmp_b = segmental_forward_pw(x.T, pw, 20)


# @jit("int16[:,:](float64[:,:])")
# def segmental_backward(scores):
# 	n_segs, T, n_classes = scores.shape

# 	# Start at end
# 	seq_c = [scores[-1, -1].argmax()] # Class
# 	seq_t = [T] # Time
# 	m = n_segs-1

# 	for t in range(T, 0, -1):
# 		# Scores of previous timestep in current segment
# 		score_left = scores[m, t-1, seq_c[-1]]

# 		# Check if it's better to stay or switch segments
# 		if any(scores[m-1, t-1] > score_left):
# 			next_class =    scores[m-1, t-1].argmax()			
# 			score_topleft = scores[m-1, t-1, next_class]
# 			seq_c += [next_class]
# 			seq_t += [t]
# 			m -= 1

# 			if m == 0:
# 				break
# 	seq_t += [0]

# 	if m != 0:
# 		print("# segs (m) not zero!", m)

# 	seq_c = list(reversed(seq_c))
# 	seq_t = list(reversed(seq_t))

# 	y_out = np.empty(T, np.int)
# 	for i in range(len(seq_c)):
# 		y_out[seq_t[i]:seq_t[i+1]] = seq_c[i]

# 	return y_out

@jit("int16[:,:](float64[:,:], float64[:,:])")
def segmental_backward(scores, pw=None):
	n_segs, T, n_classes = scores.shape

	if pw is None:
		pw = np.zeros([n_classes, n_classes], np.float)		

	# Start at end
	seq_c = [scores[-1, -1].argmax()] # Class
	seq_t = [T] # Time
	m = n_segs-1

	for t in range(T, 0, -1):
		# Scores of previous timestep in current segment
		score_left = scores[m, t-1, seq_c[-1]]
		score_topleft = scores[m-1, t-1] + pw[:,seq_c[-1]]

		# Check if it's better to stay or switch segments
		if any(score_topleft > score_left):
			next_class =    score_topleft.argmax()			
			score_topleft = score_topleft[next_class]
			seq_c += [next_class]
			seq_t += [t]
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

def segmental_inference(x, max_segs, pw=None, start_prior=None, end_prior=None, verbose=False):

	scores = segmental_forward(x, max_segs, pw=pw, 
							   start_prior=start_prior, 
							   end_prior=end_prior)

	best_n_segs = scores[:,-1].max(1).argmax()+1
	if verbose:
		print("Best # segs: {}".format(best_n_segs))

	y_out = segmental_backward(scores[:best_n_segs+1, :], pw)

	return y_out


