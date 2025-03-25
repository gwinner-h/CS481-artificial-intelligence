from math import sqrt
from numpy import asarray, arange, meshgrid
from numpy.random import rand, seed
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with adam
def adam(objective, derivative, bounds, n_i, alpha, beta1, beta2, eps=1e-8):
	solutions = list()
	
    # generate an initial point
	x = bounds[:,0] + rand(len(bounds)) * (bounds[:,1] - bounds[:,0])
	score = objective(x[0], x[1])
	
    # initialize first and second moments
	m = [0.0 for _ in range(bounds.shape[0])]
	v = [0.0 for _ in range(bounds.shape[0])]
	
    # run the gradient descent updates
	for t in range(n_i):
		# calculate gradient g(t)
		g = derivative(x[0], x[1])
		
        # build a solution one var at a time for each parameter that is being optimized
		for i in range(bounds.shape[0]):
			m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]  # calculate first moment
			v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2  # calculate the second moment
			mhat = m[i] / (1.0 - beta1**(t+1))  # bias correction for first moment
			vhat = v[i] / (1.0 - beta2**(t+1))  # bias correction for second moment
			x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps) # update variable value
			
        # evaluate candidate point
		score = objective(x[0], x[1])
		
        # keep track of solutions
		solutions.append(x.copy())
		
        # report progress
		print('>%d f(%s) = %.5f' % (t, x, score))
		
	return x, score, solutions

# demo with visualization
def demo(bounds, solutions):
    # sample input range uniformly at 0.1 increments
    xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
    yaxis = arange(bounds[1,0], bounds[1,1], 0.1)

    # create a mesh from the axis
    x, y = meshgrid(xaxis, yaxis)

    # compute targets
    results = objective(x, y)

    # create a filled contour plot with 50 levels and jet color scheme
    plt.contourf(x, y, results, levels=50, cmap='jet')
	
    # plot the sample as black circles
    solutions = asarray(solutions)
    plt.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
    plt.show()

def main():
    seed(1) # seed the pseudo random number generator

    bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])    # define range for input
    n_iter = 60 # the total iterations
    alpha = 0.02 # step size
    beta1 = 0.8 # factor for average gradient
    beta2 = 0.999 # factor for average squared gradient

    # perform the gradient descent search with adam
    best, score, solutions = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)

    print('Optimal Solution Found: f(%s) = %f' % (best, score))

    # demo(bounds, solutions)
	

if __name__ == '__main__':
    main()