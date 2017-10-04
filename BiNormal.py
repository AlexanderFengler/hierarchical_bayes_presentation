import numpy as np
import matplotlib.pyplot as plt

#### Gibbs sampler for bivariate normal
rho = 0.9
y = [0,0]
theta_guess1 = [-2.5,-2.5]
theta_guess2 = [-2.5,2.5]
theta_guess3 = [2.5,-2.5]
theta_guess4 = [2.5,2.5]
theta_guess  = np.array([theta_guess1,theta_guess2,theta_guess3,theta_guess4])

def Gibbs_sampler_BiNorm (theta_guess_, y_, rho_, N_iter = 1000, N_chain = 4, seed = 134):
  np.random.seed(seed)
  theta_chain =  np.empty(shape = (N_iter + 1, 2 * N_chain))
  for chain in range(N_chain):
    theta = theta_guess_[chain,:].copy()
    theta_chain[0, [chain * 2,chain * 2 + 1]] = theta.copy()
    for i in range(N_iter):
      theta[0] = np.random.normal(size = 1,
                                  loc = y_[0] + rho_ * (theta[1] - y_[1]),
                                  scale = np.sqrt(1 - np.power(rho_, 2)))
      theta[1] = np.random.normal(size = 1,
                                  loc = y_[1] + rho_ * (theta[0] - y_[0]),
                                  scale = np.sqrt(1 - np.power(rho_, 2)))
      theta_chain[i + 1, [chain*2,chain*2 + 1]] = theta.copy()
  return theta_chain

par_chain = Gibbs_sampler_BiNorm(theta_guess_ = theta_guess.copy(), y_ = y,rho_ = rho)

colors = ['black', 'green', 'yellow', 'blue']
# First 10 iteration
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.scatter(theta_guess[:, 0], theta_guess[:, 1], marker = "s", color = colors, s = 100)
for chain in range(4):
    plt.step(par_chain[range(10), (chain * 2)],
             par_chain[range(10), (chain * 2) + 1],
             color = colors[chain])
plt.show()

# All 1000 iterations
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.scatter(theta_guess[:, 0], theta_guess[:, 1], marker = "s", color = colors, s = 100)
for chain in range(4):
  plt.step(par_chain[:, chain * 2],
           par_chain[:, chain * 2 + 1],
           color = colors[chain],
           alpha = 0.2)
plt.show()

# After 500 burn-in
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.scatter(theta_guess[:, 0], theta_guess[:, 1], marker = "s", color = colors, s = 100, alpha = 0)
for chain in range(4):
    plt.plot(par_chain[range(500,1001), chain * 2],
             par_chain[range(500,1001), chain * 2 + 1],
             "o",
             color = colors[chain],
             alpha = 0.2)
plt.show()
