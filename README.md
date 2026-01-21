# Reinforcement Learning for Adaptive Signal Weighting

A demonstration of Proximal Policy Optimization (PPO) for learning optimal signal combination weights in a sequential decision-making framework.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the RL agent
python main.py train --episodes 1000

# Compare RL agent vs baseline strategies
python main.py compare

# Run full pipeline (train + evaluate + compare)
python main.py all
```

---

## Mathematical Framework

### 1. Problem Formulation as a Markov Decision Process

We formulate the signal weighting problem as a finite-horizon MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, H)$ where:

- **State space** $\mathcal{S} \subseteq \mathbb{R}^{4n}$: For $n$ signals, the state vector at time $t$ is:
$$s_t = \left[ \tilde{\sigma}_t, \bar{\sigma}_t^{(k)}, \hat{\sigma}_t^{(k)}, m_t \right] \in \mathbb{R}^{4n}$$

  where:
  - $\tilde{\sigma}_t^{(i)} = \frac{\sigma_t^{(i)} - \bar{\sigma}_t^{(i)}}{\hat{\sigma}_t^{(i)}}$ is the z-normalized signal
  - $\bar{\sigma}_t^{(k)} = \frac{1}{k}\sum_{j=t-k}^{t} \sigma_j$ is the rolling mean over lookback window $k$
  - $\hat{\sigma}_t^{(k)} = \sqrt{\frac{1}{k}\sum_{j=t-k}^{t}(\sigma_j - \bar{\sigma}_t^{(k)})^2}$ is the rolling standard deviation
  - $m_t = \frac{\sigma_t - \bar{\sigma}_t^{(k)}}{\hat{\sigma}_t^{(k)}}$ is the momentum indicator

- **Action space** $\mathcal{A} = \Delta^{n-1}$: The $(n-1)$-simplex representing portfolio weights:
$$\mathcal{A} = \left\{ w \in \mathbb{R}^n : w_i \geq 0, \sum_{i=1}^{n} w_i = 1 \right\}$$

- **Transition dynamics** $P: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$: Determined by exogenous market dynamics (model-free setting).

- **Reward function** $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$:
$$R(s_t, a_t) = 100 \cdot \left( \phi_t \cdot r_t^{(p)} - c \cdot |\phi_t - \phi_{t-1}| \right)$$

  where:
  - $\phi_t = \tanh\left(3 \cdot \sum_{i=1}^{n} w_t^{(i)} \sigma_t^{(i)}\right) \in [-1, 1]$ is the position
  - $r_t^{(p)} = \frac{p_{t+1} - p_t}{p_t}$ is the price return
  - $c$ is the transaction cost coefficient

### 2. Policy Parameterization

We employ a stochastic policy $\pi_\theta: \mathcal{S} \rightarrow \Delta(\mathcal{A})$ parameterized by a neural network with shared feature extraction:

$$f_\psi(s) = \tanh(W_2 \cdot \tanh(W_1 s + b_1) + b_2) \in \mathbb{R}^d$$

The policy outputs parameters of a factorized Gaussian in the pre-softmax space:

$$\pi_\theta(a|s) = \mathcal{N}\left(\mu_\theta(s), \text{diag}(\sigma_\theta^2)\right)$$

where $\mu_\theta(s) = W_\mu f_\psi(s) + b_\mu \in \mathbb{R}^n$ and $\log \sigma_\theta \in \mathbb{R}^n$ is a learnable parameter vector.

The actual weight vector is obtained via the softmax transformation:
$$w = \text{softmax}(z), \quad z \sim \pi_\theta(\cdot|s)$$

This reparameterization ensures $w \in \Delta^{n-1}$ while allowing unconstrained optimization in $\mathbb{R}^n$.

### 3. Proximal Policy Optimization

#### 3.1 Policy Gradient Foundation

The objective is to maximize the expected discounted return:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{H} \gamma^t R(s_t, a_t)\right]$$

By the policy gradient theorem (Sutton et al., 2000):
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{H} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t)\right]$$

where $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ is the advantage function.

#### 3.2 Generalized Advantage Estimation (GAE)

We estimate advantages using GAE (Schulman et al., 2016) which provides a bias-variance tradeoff controlled by $\lambda \in [0,1]$:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where the TD residual is:
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

In practice, we compute this recursively:
$$\hat{A}_t = \delta_t + \gamma \lambda (1 - d_t) \hat{A}_{t+1}$$

where $d_t \in \{0, 1\}$ is the terminal indicator.

#### 3.3 Clipped Surrogate Objective

PPO (Schulman et al., 2017) optimizes a clipped surrogate objective to ensure stable updates:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where the probability ratio is:
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

The clipping mechanism with $\epsilon = 0.2$ prevents destructively large policy updates by bounding $r_t(\theta) \in [1-\epsilon, 1+\epsilon]$ when the advantage is positive (and inversely when negative).

#### 3.4 Value Function Learning

The critic $V_\phi(s)$ is trained to minimize the mean squared error against empirical returns:

$$L^{V}(\phi) = \mathbb{E}_t\left[\left(V_\phi(s_t) - \hat{R}_t\right)^2\right]$$

where $\hat{R}_t = \hat{A}_t + V_{\phi_{\text{old}}}(s_t)$ are the GAE-based return targets.

#### 3.5 Entropy Regularization

To encourage exploration and prevent premature convergence, we add an entropy bonus:

$$H[\pi_\theta(\cdot|s)] = -\mathbb{E}_{a \sim \pi_\theta}[\log \pi_\theta(a|s)]$$

For our Gaussian policy:
$$H[\mathcal{N}(\mu, \sigma^2)] = \frac{1}{2}\log(2\pi e \sigma^2) = \frac{1}{2}(1 + \log(2\pi) + 2\log\sigma)$$

#### 3.6 Combined Objective

The full PPO objective is:
$$L(\theta, \phi) = \mathbb{E}_t\left[L^{\text{CLIP}}(\theta) - c_1 L^V(\phi) + c_2 H[\pi_\theta(\cdot|s_t)]\right]$$

with coefficients $c_1 = 0.5$ (value loss) and $c_2 = 0.01$ (entropy).

### 4. Theoretical Guarantees

#### 4.1 Monotonic Improvement Bound

PPO provides an approximate trust region guarantee. Under mild assumptions, for any policies $\pi, \pi'$:

$$J(\pi') \geq J(\pi) + \mathbb{E}_{s \sim d^{\pi}}\left[\mathbb{E}_{a \sim \pi'}[A^\pi(s,a)] - C \cdot D_{\text{KL}}^{\max}(\pi || \pi')\right]$$

where $D_{\text{KL}}^{\max} = \max_s D_{\text{KL}}(\pi(\cdot|s) || \pi'(\cdot|s))$ and $C = \frac{4\gamma\epsilon}{(1-\gamma)^2}$ with $\epsilon = \max_{s,a}|A^\pi(s,a)|$.

The clipping mechanism implicitly constrains the KL divergence, providing stability without explicit KL penalty computation.

#### 4.2 Sample Complexity

For $\epsilon$-optimal policy with probability $1-\delta$, PPO achieves sample complexity:
$$\tilde{O}\left(\frac{|\mathcal{S}||\mathcal{A}|}{(1-\gamma)^3\epsilon^2}\log\frac{1}{\delta}\right)$$

in the tabular setting, with function approximation bounds depending on the complexity class of the policy and value networks.

### 5. Baseline Strategies (Non-RL)

For comparison, we implement several heuristic weighting schemes:

| Strategy | Weight Formula |
|----------|----------------|
| Equal Weight | $w_i = \frac{1}{n}$ |
| Signal Strength | $w_i = \frac{\exp(\|\sigma_i\|)}{\sum_j \exp(\|\sigma_j\|)}$ |
| Inverse Volatility | $w_i = \frac{1/\hat{\sigma}_i^{(k)}}{\sum_j 1/\hat{\sigma}_j^{(k)}}$ |
| Momentum | $w_i = \frac{\exp(\rho_i)}{\sum_j \exp(\rho_j)}$ where $\rho_i = \text{corr}(\sigma^{(i)}_{t-k:t}, r_{t-k:t}^{(p)})$ |

### 6. Performance Metrics

We evaluate strategies using standard risk-adjusted metrics:

- **Sharpe Ratio**: $\text{SR} = \frac{\mathbb{E}[r] - r_f}{\sigma_r}\sqrt{252}$ (annualized)
- **Maximum Drawdown**: $\text{MDD} = \max_t \frac{\max_{\tau \leq t} P_\tau - P_t}{\max_{\tau \leq t} P_\tau}$
- **Annualized Volatility**: $\sigma_{\text{ann}} = \sigma_r \sqrt{252}$
- **Win Rate**: $\mathbb{P}(r_t > 0)$

---

## Simple Explanation (ELI5 Version)

### What This Project Does

Imagine you have 5 different weather forecasters, and you need to decide how much to trust each one when planning your day. Some forecasters are better than others, and their accuracy might change over time.

**The RL agent learns to figure out which forecasters to trust and by how much.**

### How It Works

1. **The Problem**: We have 5 trading signals (like the forecasters). Each signal suggests whether the market will go up or down. We need to combine them into one decision.

2. **The Simple Approach (Baselines)**:
   - *Equal Weight*: Trust everyone equally (20% each)
   - *Random*: Randomly pick who to trust
   - *Loudest Voice*: Only listen to whoever is most confident

3. **The Smart Approach (RL Agent)**:
   - The agent observes how well each signal has been doing
   - It learns patterns: "Signal 3 is usually right when Signal 1 is wrong"
   - It adjusts the weights automatically based on what it learned

### Why RL Beats Simple Rules

- **Adapts**: If Signal 2 starts performing poorly, RL reduces its weight
- **Finds Hidden Patterns**: Discovers complex relationships humans might miss
- **Optimizes for the Goal**: Directly learns what makes money, not just what seems logical

### The Training Process

```
Episode 1:    Agent guesses randomly         -> Loses money
Episode 100:  Agent notices Signal 3 is good -> Starts trusting it more
Episode 500:  Agent finds optimal balance    -> Makes consistent profits
Episode 1000: Agent masters the strategy     -> Beats all simple rules
```

### Running the Comparison

```bash
python main.py compare
```

This shows you a table like:

```
Strategy              Return    Sharpe
------------------------------------
RL Agent (PPO)        +15.2%     1.85    <-- Best!
Momentum Weight        +8.1%     1.23
Equal Weight           +5.4%     0.89
Random Weight          +1.2%     0.31
```

The RL agent wins because it learned the optimal strategy through trial and error, while the other methods just follow fixed rules.

### Key Takeaway

**Without RL**: You pick a fixed rule and hope it works.

**With RL**: The computer tries millions of combinations, learns from mistakes, and finds the best strategy automatically.

---

## Project Structure

```
rl-demo/
├── main.py              # CLI entry point
├── src/
│   ├── agent.py         # PPO Actor-Critic implementation
│   ├── environment.py   # Gymnasium environment
│   ├── baselines.py     # Non-RL baseline strategies
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Single strategy evaluation
│   ├── compare.py       # RL vs baselines comparison
│   └── data_generator.py
├── checkpoints/         # Saved models
└── data/                # Synthetic data
```

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)
2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." ICLR (2016)
3. Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. "Policy Gradient Methods for Reinforcement Learning with Function Approximation." NeurIPS (2000)
