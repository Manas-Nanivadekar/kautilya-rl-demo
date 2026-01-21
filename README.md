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
$$s\_t = \left[ \tilde{\sigma}\_t, \bar{\sigma}\_t^{(k)}, \hat{\sigma}\_t^{(k)}, m\_t \right] \in \mathbb{R}^{4n}$$

  where:
  - $\tilde{\sigma}\_t^{(i)} = \frac{\sigma\_t^{(i)} - \bar{\sigma}\_t^{(i)}}{\hat{\sigma}\_t^{(i)}}$ is the z-normalized signal
  - $\bar{\sigma}\_t^{(k)} = \frac{1}{k}\sum\_{j=t-k}^{t} \sigma\_j$ is the rolling mean over lookback window $k$
  - $\hat{\sigma}\_t^{(k)} = \sqrt{\frac{1}{k}\sum\_{j=t-k}^{t}(\sigma\_j - \bar{\sigma}\_t^{(k)})^2}$ is the rolling standard deviation
  - $m\_t = \frac{\sigma\_t - \bar{\sigma}\_t^{(k)}}{\hat{\sigma}\_t^{(k)}}$ is the momentum indicator

- **Action space** $\mathcal{A} = \Delta^{n-1}$: The $(n-1)$-simplex representing portfolio weights:
$$\mathcal{A} = \left\{ w \in \mathbb{R}^n : w\_i \geq 0, \sum\_{i=1}^{n} w\_i = 1 \right\}$$

- **Transition dynamics** $P: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$: Determined by exogenous market dynamics (model-free setting).

- **Reward function** $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$:
$$R(s\_t, a\_t) = 100 \cdot \left( \phi\_t \cdot r\_t^{(p)} - c \cdot |\phi\_t - \phi\_{t-1}| \right)$$

  where:
  - $\phi\_t = \tanh\left(3 \cdot \sum\_{i=1}^{n} w\_t^{(i)} \sigma\_t^{(i)}\right) \in [-1, 1]$ is the position
  - $r\_t^{(p)} = \frac{p\_{t+1} - p\_t}{p\_t}$ is the price return
  - $c$ is the transaction cost coefficient

### 2. Policy Parameterization

We employ a stochastic policy $\pi\_\theta: \mathcal{S} \rightarrow \Delta(\mathcal{A})$ parameterized by a neural network with shared feature extraction:

$$f\_\psi(s) = \tanh(W\_2 \cdot \tanh(W\_1 s + b\_1) + b\_2) \in \mathbb{R}^d$$

The policy outputs parameters of a factorized Gaussian in the pre-softmax space:

$$\pi\_\theta(a|s) = \mathcal{N}\left(\mu\_\theta(s), \text{diag}(\sigma\_\theta^2)\right)$$

where $\mu\_\theta(s) = W\_\mu f\_\psi(s) + b\_\mu \in \mathbb{R}^n$ and $\log \sigma\_\theta \in \mathbb{R}^n$ is a learnable parameter vector.

The actual weight vector is obtained via the softmax transformation:

$$w = \text{softmax}(z), \quad z \sim \pi\_\theta(\cdot|s)$$

This reparameterization ensures $w \in \Delta^{n-1}$ while allowing unconstrained optimization in $\mathbb{R}^n$.

### 3. Proximal Policy Optimization

#### 3.1 Policy Gradient Foundation

The objective is to maximize the expected discounted return:

$$J(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}\left[\sum\_{t=0}^{H} \gamma^t R(s\_t, a\_t)\right]$$

By the policy gradient theorem (Sutton et al., 2000):

$$\nabla\_\theta J(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}\left[\sum\_{t=0}^{H} \nabla\_\theta \log \pi\_\theta(a\_t|s\_t) \cdot A^{\pi\_\theta}(s\_t, a\_t)\right]$$

where $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ is the advantage function.

#### 3.2 Generalized Advantage Estimation (GAE)

We estimate advantages using GAE (Schulman et al., 2016) which provides a bias-variance tradeoff controlled by $\lambda \in [0,1]$:

$$\hat{A}\_t^{\text{GAE}(\gamma, \lambda)} = \sum\_{l=0}^{\infty} (\gamma \lambda)^l \delta\_{t+l}$$

where the TD residual is:

$$\delta\_t = r\_t + \gamma V\_\phi(s\_{t+1}) - V\_\phi(s\_t)$$

In practice, we compute this recursively:

$$\hat{A}\_t = \delta\_t + \gamma \lambda (1 - d\_t) \hat{A}\_{t+1}$$

where $d\_t \in \{0, 1\}$ is the terminal indicator.

#### 3.3 Clipped Surrogate Objective

PPO (Schulman et al., 2017) optimizes a clipped surrogate objective to ensure stable updates:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\_t\left[\min\left(r\_t(\theta)\hat{A}\_t, \text{clip}(r\_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}\_t\right)\right]$$

where the probability ratio is:

$$r\_t(\theta) = \frac{\pi\_\theta(a\_t|s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t|s\_t)}$$

The clipping mechanism with $\epsilon = 0.2$ prevents destructively large policy updates by bounding $r\_t(\theta) \in [1-\epsilon, 1+\epsilon]$ when the advantage is positive (and inversely when negative).

#### 3.4 Value Function Learning

The critic $V\_\phi(s)$ is trained to minimize the mean squared error against empirical returns:

$$L^{V}(\phi) = \mathbb{E}\_t\left[\left(V\_\phi(s\_t) - \hat{R}\_t\right)^2\right]$$

where $\hat{R}\_t = \hat{A}\_t + V\_{\phi\_{\text{old}}}(s\_t)$ are the GAE-based return targets.

#### 3.5 Entropy Regularization

To encourage exploration and prevent premature convergence, we add an entropy bonus:

$$H[\pi\_\theta(\cdot|s)] = -\mathbb{E}\_{a \sim \pi\_\theta}[\log \pi\_\theta(a|s)]$$

For our Gaussian policy:
$$H[\mathcal{N}(\mu, \sigma^2)] = \frac{1}{2}\log(2\pi e \sigma^2) = \frac{1}{2}(1 + \log(2\pi) + 2\log\sigma)$$

#### 3.6 Combined Objective

The full PPO objective is:
$$L(\theta, \phi) = \mathbb{E}\_t\left[L^{\text{CLIP}}(\theta) - c\_1 L^V(\phi) + c\_2 H[\pi\_\theta(\cdot|s\_t)]\right]$$

with coefficients $c\_1 = 0.5$ (value loss) and $c\_2 = 0.01$ (entropy).

### 4. Theoretical Guarantees

#### 4.1 Monotonic Improvement Bound

PPO provides an approximate trust region guarantee. Under mild assumptions, for any policies $\pi, \pi'$:

$$J(\pi') \geq J(\pi) + \mathbb{E}\_{s \sim d^{\pi}}\left[\mathbb{E}\_{a \sim \pi'}[A^\pi(s,a)] - C \cdot D\_{\text{KL}}^{\max}(\pi || \pi')\right]$$

where $D\_{\text{KL}}^{\max} = \max\_s D\_{\text{KL}}(\pi(\cdot|s) || \pi'(\cdot|s))$ and $C = \frac{4\gamma\epsilon}{(1-\gamma)^2}$ with $\epsilon = \max\_{s,a}|A^\pi(s,a)|$.

The clipping mechanism implicitly constrains the KL divergence, providing stability without explicit KL penalty computation.

#### 4.2 Sample Complexity

For $\epsilon$-optimal policy with probability $1-\delta$, PPO achieves sample complexity:
$$\tilde{O}\left(\frac{|\mathcal{S}||\mathcal{A}|}{(1-\gamma)^3\epsilon^2}\log\frac{1}{\delta}\right)$$

in the tabular setting, with function approximation bounds depending on the complexity class of the policy and value networks.

### 5. Baseline Strategies (Non-RL)

For comparison, we implement several heuristic weighting schemes:

| Strategy | Weight Formula |
|----------|----------------|
| Equal Weight | $w\_i = \frac{1}{n}$ |
| Signal Strength | $w\_i = \frac{\exp(\|\sigma\_i\|)}{\sum\_j \exp(\|\sigma\_j\|)}$ |
| Inverse Volatility | $w\_i = \frac{1/\hat{\sigma}\_i^{(k)}}{\sum\_j 1/\hat{\sigma}\_j^{(k)}}$ |
| Momentum | $w\_i = \frac{\exp(\rho\_i)}{\sum\_j \exp(\rho\_j)}$ where $\rho\_i = \text{corr}(\sigma^{(i)}\_{t-k:t}, r\_{t-k:t}^{(p)})$ |

### 6. Performance Metrics

We evaluate strategies using standard risk-adjusted metrics:

- **Sharpe Ratio**: $\text{SR} = \frac{\mathbb{E}[r] - r\_f}{\sigma\_r}\sqrt{252}$ (annualized)
- **Maximum Drawdown**: $\text{MDD} = \max\_t \frac{\max\_{\tau \leq t} P\_\tau - P\_t}{\max\_{\tau \leq t} P\_\tau}$
- **Annualized Volatility**: $\sigma\_{\text{ann}} = \sigma\_r \sqrt{252}$
- **Win Rate**: $\mathbb{P}(r\_t > 0)$

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
