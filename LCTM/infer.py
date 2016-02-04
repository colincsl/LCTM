import numpy as np
from numba import float64, jit, int16, boolean, int64

@jit("float64[:,:](float64[:,:], float64[:,:], int16, float64[:], float64[:])")
def segmental_forward(x, max_segs, pw=None, start_prior=None, end_prior=None):
	T, n_classes = x.shape
	LARGE_NUMBER = 99999.
	scores = np.zeros([max_segs, T, n_classes], np.float) - LARGE_NUMBER
	if pw is None:
		pw = np.zeros([n_classes, n_classes], np.float)

	# initialize first segment scores
	scores[0] = np.cumsum(x, 0)

	if start_prior is not None:
		scores[0] += start_prior.T

	# Compute scores for each segment in sequence
	for m in range(1, max_segs):
		# Compute scores for each timestep
		for t in range(m+1, T):
			# Compute score for each class
			for c in range(n_classes):
				# Score for staying in same segment
				best_prev = scores[m, t-1, c]

				# Check if it is cheaper to create a new segment or stay in same class
				for c_prev in range(n_classes):
					if c_prev == c:
						continue
					tmp = scores[m-1, t-1, c_prev] + pw[c_prev,c]
					if tmp > best_prev:
						best_prev = tmp
				
				# if c==4 and m==1:
				# 	print(t, best_prev)

				# Add cost of curent frame to best previous cost
				if best_prev > 0:
					scores[m, t, c] = best_prev + x[t, c]

	if end_prior is not None:
		scores[-1] += end_prior.T

	# Set nonzero entries to 0 for visualization
	scores[scores==-LARGE_NUMBER] = 0

	return scores

# x = np.random.random([7, 1000]);
# %timeit tmp = segmental_forward(x.T, 20)
# %timeit tmp_b = segmental_forward_pw(x.T, pw, 20)

@jit("int16[:,:](float64[:,:])")
def segmental_backward(scores):
	n_segs, T, n_classes = scores.shape

	# Start at end
	seq_c = [scores[-1, -1].argmax()] # Class
	seq_t = [T] # Time
	m = n_segs-1
	for t in range(T, 0, -1):
		# Get scores for previous timestep in current segment or previous
		score_left =    scores[m,   t-1, seq_c[-1]]
		score_topleft = scores[m-1, t-1].max()
		next_class =    scores[m-1, t-1].argmax()
		# Choose whether it should stay in same segment or transition to previous
		if (score_topleft > score_left) and seq_c[-1] != next_class:
			seq_c += [next_class]
			seq_t += [t]
			m -= 1

			if m == 0:
				break
	seq_t += [0]

	if m != 0:
		print("# segs (m) not zero!", m)
		# plot(np.diff(seq_t))

	seq_c = list(reversed(seq_c))
	seq_t = list(reversed(seq_t))

	y_out = np.empty(T, np.int)
	for i in range(len(seq_c)):
		y_out[seq_t[i]:seq_t[i+1]] = seq_c[i]

	return y_out

def segmental_inference(x, max_segs, pw=None, start_prior=None, end_prior=None, verbose=False):

	scores = segmental_forward(x, max_segs, pw=pw, 
						start_prior=start_prior, end_prior=end_prior)

	best_n_segs = scores[:,-1].max(1).argmax()+1
	if verbose:
		print("Best # segs: {}".format(best_n_segs+1))

	y_out = segmental_backward(scores[:best_n_segs, :])

	return y_out


# def segment_data(y, x):
# 	segs = np.hstack([0, np.nonzero(np.diff(y)!=0)[0]+1, y.shape[0]])
# 	x_segs = np.array([x[segs[i-1]:segs[i],:] for i in range(1,len(segs))])
# 	y_labels = np.array([y[s] for s in segs[:-1]])

# 	return x_segs, y_labels





# @jit("float64[:,:](float64[:,:], int16, float64[:], float64[:])")
# def segmental_forward(x, max_segs, start_prior=None, end_prior=None):
# 	T, n_classes = x.shape
# 	LARGE_NUMBER = 99999.
# 	scores = np.zeros([max_segs, T, n_classes], np.float) - LARGE_NUMBER

# 	# initialize first segment scores
# 	scores[0,:,:] = np.cumsum(x, 0)

# 	# Forward pass
# 	# Compute scores for each segment in sequence
# 	for m in range(1, max_segs):
# 		# Compute scores for each timestep
# 		for t in range(m+1, T):
# 			# Compute score for each class
# 			for c in range(n_classes):
# 				# Score for staying in same segment
# 				best_prev = scores[m, t-1, c]

# 				# Check if it is cheaper to create a new segment or stay in same class
# 				for c_prev in range(n_classes):
# 					if c_prev == c:
# 						continue
# 					elif scores[m-1, t-1, c_prev] > best_prev:
# 						best_prev = scores[m-1, t-1, c_prev]
				
# 				# Add cost of curent frame to best previous cost
# 				scores[m, t, c] = best_prev + x[t, c]

# 	# Set nonzero entries to 0 for visualization
# 	scores[scores==-LARGE_NUMBER] = 0

# 	return scores

# @jit("float64[:,:](float64[:,:], float64[:,:], int16)")
# def segmental_forward(x, pw, max_segs):
