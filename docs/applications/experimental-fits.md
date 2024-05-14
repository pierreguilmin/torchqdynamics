# Fitting a cat qubit experiment with real-world data

dynamiqs enables fast simulations of quantum systems with automatic gradient computation, making it highly suitable for data fitting tasks. This tutorial will guide you through the process of simulating and fitting real-world experimental data, step by step.

## Context of the experiment

!!! note "Context"
    This section introduces the kind of data and process that we will fit in the notebook. If you are only interested in the code you can skip the next section.

In the experiment of Réglade, Bocquet _et al._ (arXiv, 2023)[^1], the authors aim to perform quantum control of a cat qubit while preserving its exponential bias in bit-flip errors. One challenging aspect of the experiment is to calibrate the two-photon exchange rate $g_2$ between the cat qubit oscillator (the memory) and the ancillary reservoir mode (the buffer). The object of this tutorial is to perform this calibration with dynamiqs.

The experiment can be modelled by the following master equation:

\begin{aligned}
    \frac{d\hat\rho}{dt} = -i[\hat H, \hat \rho] &+ \kappa_a (1 + n_\mathrm{th, a})\mathcal{D}[\hat a] \rho + \kappa_a n_\mathrm{th, a} \mathcal{D}[\hat a^\dagger] \rho \\
    &+ \kappa_b (1 + n_\mathrm{th, b})\mathcal{D}[\hat b] \rho + \kappa_b n_\mathrm{th, b} \mathcal{D}[\hat b^\dagger] \rho,
\end{aligned}

with
$$
    \hat H = g_2 \hat a^2 \hat b^\dagger + g_2^* \hat a^{\dagger 2} \hat b
$$
where $\hat a$ and $\hat b$ denote the annihilation operators of the memory and the buffer, respectively. Both the memory and buffer dissipation rates and their average thermal populations are assumed to be known from previous experimental calibrations. Here, they are given by $\kappa_a / 2\pi = 9.3\:\mathrm{kHz}$, $\kappa_b / 2\pi = 2.6\:\mathrm{MHz}$, $n_\mathrm{th, a} = 0.1$, and $n_\mathrm{th, b} = 0.011$.

At the beginning of the experiment the memory is populated with a coherent state and the buffer is in thermal vacuum. We then turn on the two-photon exchange mechanism, such that pairs of photons leave the memory and are converted into single buffer photons that are quickly dissipated into the environment. The number of photons in the memory, $\langle a^\dagger a \rangle$ is measured at different times and saved into the `data` variable. We then infer $g_2$ from this data.

![Pulse sequence](/figs-docs/g2-pulse-sequence.png)
_Pulse sequence of the experiment, taken from [^1]_

[^1]:  [Réglade, Bocquet _et al._, "Quantum control of a cat-qubit with bit-flip times exceeding ten seconds" (2023), arxiv:2307.06617](https://arxiv.org/pdf/2307.06617.pdf)

## Fitting real-world data, step by step

In this notebook, we will load the previously described experimental data (or generate synthetic data), encode the two-photon exchange model in dynamiqs, simulate the system evolution and measure the number of photons at different times. We will then use gradient descent to fit the $g_2$ parameter from the data.

We start by importing all necessary libraries, and defining the physical parameters of the system.

```python
import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
dq.mplstyle()

# Declare useful helpers to handle units
MHz = 2 * jnp.pi
kHz = 2 * jnp.pi * 1e-3
us = 1.0
ns = 1.0e-3

# Define physical values that describe our system
nth_a, nth_b = 0.1, 0.011
kappa_a = 9.3 * kHz
kappa_b = 2.6 * MHz

# Instanciate our bosonic annihilation operators
Na, Nb = 15, 7
a, b = dq.destroy(Na, Nb)
```

### Loading the data

We now initialize the experimental data with which we will work. There are two options to do so:

- load the actual experimental data from [^1]_, or
- generate synthetic data from a noisy master equation simulation.

#### Option 1: Real world data

First, download the experimental data from the dynamiqs github repository (TODO: add link). Once this is done, we load the experimental data and plot it.

```python
# Load the data
def load(file):
    x = np.loadtxt(file)
    x = jnp.array(x)
    return x

data = load("short.npy")
tsave = load("time_short.npy")
alphas = jnp.sqrt(data[0])

# Plot experimental data
plt.figure()
plt.plot(tsave / us, data, "+")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
renderfig('g2-real-data')
```

![Loaded data](/figs-docs/g2-real-data.png)

#### Option 2: Synthetic data

Alternatively, we can generate synthetic data from a master equation simulation with dynamiqs, and add artificial noise on top. Below, we define the simulation code and generate this synthetic data.

```python
# True value of g2 which we aim to fit
g2_true = 0.8 * MHz

# Define several initial states, one for each coherent state amplitude
alphas = jnp.sqrt(jnp.linspace(0.4, 2.0, 5))
psi0_b = dq.unit(dq.fock(Nb, 0) + nth_b ** 0.5 * dq.fock(Nb, 1))
psi0 = jnp.stack([dq.tensor(dq.coherent(Na, alpha), psi0_b) for alpha in alphas])

# Define the Hamiltonian
H = g2_true * a @ a @ dq.dag(b)
H += dq.dag(H)

# Define jump operators
jump_ops = [
    jnp.sqrt(kappa_a * (1 + nth_a)) * a,
    jnp.sqrt(kappa_a * nth_a) * dq.dag(a),
    jnp.sqrt(kappa_b * (1 + nth_b)) * b,
    jnp.sqrt(kappa_b * nth_b) * dq.dag(b)
]

# Define expectation operator
exp_ops = [dq.dag(a) @ a]

# Define time variable
tsave = jnp.linspace(0.0 * ns, 515 * ns, 501)

# Perform the master equation simulation
result = dq.mesolve(
    H,
    jump_ops,
    psi0,
    tsave,
    exp_ops=exp_ops
)

# Extract final data and add noise
key = jax.random.key(42)
data = result.expects[:, 0].real.T
data += data * (jax.random.normal(key, data.shape) + 1.0) * 0.01 # multiplicative noise
data += jax.random.normal(key, data.shape) * 0.015 # additive noise

# Plot the synthetic data
plt.figure()
plt.plot(tsave / us, data, "+")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
renderfig('g2-synthetic-data')
```
![Synthetic data](/figs-docs/g2-synthetic-data.png)


### Fitting the data

Now that we have the data, we can perform the numerical fit of $g_2$. We first define the simulation function. Note that we have to redefine the Hamiltonian with the new $g_2$ value at each iteration of the optimization loop.

```python
def simulate(g2):
    # Redefine Hamiltonian with the new g2 value
    H = g2 * a @ a @ dq.dag(b)
    H += dq.dag(H)

    # Perform the master equation simulation
    return dq.mesolve(
        H,
        jump_ops,
        psi0,
        tsave,
        exp_ops=exp_ops,
    )
```

We choose an initial guess that is reasonable with respect to the data we have. We converge to a steady state in roughly $1\:\mu\mathrm{s}$ so $g_2$ should be of the order of $1\:\mathrm{MHz}$. Let us begin from this estimated value.
```python
# Initial guess
g2 = jnp.array(1.0 * MHz)

# Run the simulation with the initial guess
result = simulate(g2)

# Plot the initial guess simulation and the data
plt.figure()
for i in range(len(alphas)):
    plt.plot(tsave / us, data[:, i].real, f"+C{i}")
    plt.plot(tsave / us, result.expects[i, 0].real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
renderfig('g2-initial-guess')
```
![g2 initial guess](/figs-docs/g2-initial-guess.png)

% skip: start

Next, we run the optimization, with a straightforward cost function (distance between the simulated and experimental data), and with a plain gradient descent. There are many smarter ways to perform gradient descent, for instance the ones provided in [Optax](https://github.com/google-deepmind/optax). The optimization loop is defined as follow
```python
def cost_fn(g2, data):
    """Simulate the system with the given g2 and return the cost."""
    res = simulate(g2)
    return jnp.sum((data - res.expects[:, 0].real.T) ** 2)

costs, g2s = [], [g2]
lr = 0.01 # learning rate
for _ in tqdm(range(40)):
    # Compute the cost and its gradient
    cost, cost_grad = jax.value_and_grad(cost_fn)(g2, data)

    # Clip the gradient to avoid numerical instability
    cost_grad = jnp.clip(cost_grad, -100, 100)

    # Perform the gradient descent step
    g2 = g2 - lr * cost_grad

    # Save the cost and the g2 value
    costs.append(cost)
    g2s.append(g2)

# Save the final cost and convert to jnp arrays
costs.append(cost_fn(g2, data))
costs, g2s = jnp.array(costs), jnp.array(g2s)
```

!!! note Optimisation time
    The full optimisation runs in about 20 seconds on a NVIDIA GeForce RTX 4090 and around 15 minutes on a M2 Macbook air.

### Plotting the results

Once the optimization is finished, we can plot the loss to ensure we converged correctly.
```python
fig, (ax0, ax1) = plt.subplots(1, 2, tight_layout=True)

ax0.plot(costs)
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Cost")

ax1.plot(g2s / MHz)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("g2 [MHz]")
```
![Fit result](/figs-docs/g2-fit-value-cost.png)

Finally we plot the fit and compare with the data
```python
# Print the final g2 value
print(f"g2 = {g2 / MHz:.3f} MHz")

# Run a final simulation with the fitted g2 value
result = simulate(g2)

# Plot the fit result
plt.figure()
for i in range(len(alphas)):
    plt.plot(tsave / us, data[:, i].real, f"+C{i}")
    plt.plot(tsave / us, result.expects[i, 0].real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
```
```text
g2 = 0.785 MHz
```
![Fit result](/figs-docs/g2-fit-result.png)

!!! example "Final note"
    $g_2$ value differs from the one reported in
    [arxiv:2307.06617](https://arxiv.org/pdf/2307.06617.pdf)
    by approximately 15%. This discrepancy is attributed to the unavailability
    of dynamiqs for performing fits at the time of writing the article.
    Consequently, the authors were unable to employ gradient descent for their
    fit and had to resort to alternative methods.


## Full code

```python
import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
dq.mplstyle()

# Declare useful helpers to handle units
MHz = 2 * jnp.pi
kHz = 2 * jnp.pi * 1e-3
us = 1.0
ns = 1.0e-3

# Define physical values that describe our system
nth_a, nth_b = 0.1, 0.011
kappa_a = 9.3 * kHz
kappa_b = 2.6 * MHz

# Instanciate our bosonic annihilation operators
Na, Nb = 15, 7
a, b = dq.destroy(Na, Nb)

# =============================================================================
# Option 1: Load experimental data
# =============================================================================

# Load the data
def load(file):
    x = np.loadtxt(file)
    x = jnp.array(x)
    return x

data = load("short.npy")
tsave = load("time_short.npy")
alphas = jnp.sqrt(data[0])

# Plot experimental data
plt.figure()
plt.plot(tsave / us, data, "+")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")

# =============================================================================
# Option 2: Generate synthetic data
# =============================================================================

# True value of g2 which we aim to fit
g2_true = 0.8 * MHz

# Define several initial states, one for each coherent state amplitude
alphas = jnp.sqrt(jnp.linspace(0.4, 2.0, 5))
psi0_b = dq.unit(dq.fock(Nb, 0) + nth_b ** 0.5 * dq.fock(Nb, 1))
psi0 = jnp.stack([dq.tensor(dq.coherent(Na, alpha), psi0_b) for alpha in alphas])

# Define the Hamiltonian
H = g2_true * a @ a @ dq.dag(b)
H += dq.dag(H)

# Define jump operators
jump_ops = [
    jnp.sqrt(kappa_a * (1 + nth_a)) * a,
    jnp.sqrt(kappa_a * nth_a) * dq.dag(a),
    jnp.sqrt(kappa_b * (1 + nth_b)) * b,
    jnp.sqrt(kappa_b * nth_b) * dq.dag(b)
]

# Define expectation operator
exp_ops = [dq.dag(a) @ a]

# Define time variable
tsave = jnp.linspace(0.0 * ns, 515 * ns, 501)

# Perform the master equation simulation
result = dq.mesolve(
    H,
    jump_ops,
    psi0,
    tsave,
    exp_ops=exp_ops
)

# Extract final data and add noise
key = jax.random.key(42)
data = result.expects[:, 0].real.T
data += data * (jax.random.normal(key, data.shape) + 1.0) * 0.01 # multiplicative noise
data += jax.random.normal(key, data.shape) * 0.015 # additive noise

# Plot the synthetic data
plt.figure()
plt.plot(tsave / us, data, "+")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
plt.show()

# =============================================================================
# Fitting the data
# =============================================================================

# Simulation function
def simulate(g2):
    # Redefine Hamiltonian with the new g2 value
    H = g2 * a @ a @ dq.dag(b)
    H += dq.dag(H)

    # Perform the master equation simulation
    return dq.mesolve(
        H,
        jump_ops,
        psi0,
        tsave,
        exp_ops=exp_ops,
    )

# Define the model to fit
g2 = jnp.array(1.0 * MHz)
result = simulate(g2)

plt.figure()

for i in range(len(alphas)):
    plt.plot(tsave, data[:, i].real, f"+C{i}")
    plt.plot(tsave, result.expects[i, 0].real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
plt.show()

# Define the cost function
def cost_fn(g2, data):
    """Simulate the system with the given g2 and return the cost."""
    res = simulate(g2)
    return jnp.sum((data - res.expects[:, 0].real.T) ** 2)

costs, g2s = [], [g2]
lr = 0.01 # learning rate
for _ in tqdm(range(40)):
    # Compute the cost and its gradient
    cost, cost_grad = jax.value_and_grad(cost_fn)(g2, data)

    # Clip the gradient to avoid numerical instability
    cost_grad = jnp.clip(cost_grad, -100, 100)

    # Perform the gradient descent step
    g2 = g2 - lr * cost_grad

    # Save the cost and the g2 value
    costs.append(cost)
    g2s.append(g2)

# Save the final cost and convert to jnp arrays
costs.append(cost_fn(g2, data))
costs, g2s = jnp.array(costs), jnp.array(g2s)

# =============================================================================
# Plotting the results
# =============================================================================

# Plot the cost and the g2 values
fig, (ax0, ax1) = plt.subplots(1, 2, tight_layout=True)

ax0.plot(costs)
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Cost")

ax1.plot(g2s / MHz)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("g2 [MHz]")

# Print the final g2 value
print(f"g2 = {g2 / MHz:.3f} MHz")

# Run a final simulation with the fitted g2 value
result = simulate(g2)

# Plot the fit result
plt.figure()
for i in range(len(alphas)):
    plt.plot(tsave / us, data[:, i].real, f"+C{i}")
    plt.plot(tsave / us, result.expects[i, 0].real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
```

% skip: end
