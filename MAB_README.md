# Multi-Armed Bandit Experiment: Profit-Aware Product Recommendation

A Reinforcement Learning project that implements and compares multiple Multi-Armed Bandit (MAB) strategies to maximize net profit in an e-commerce product recommendation setting.

---

## 📌 Problem Statement

A large e-commerce organization wants to maximize long-term net profit from homepage product recommendations. Each of the 6 products has varying revenue and cost per user session. The goal is to find the best MAB strategy that learns which product to recommend while balancing exploration and exploitation.

> **Net Reward = Product Revenue − Cost**

---

## 📂 Project Structure

```
├── Multi_Armed_Bandit_Experiment.ipynb     # Main notebook
├── Dataset_Product_Recommendation.csv      # Input dataset (required)
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 📊 Dataset

- **File:** `Dataset_Product_Recommendation.csv`
- **Rows:** 498 user sessions
- **Features:** Revenue and cost columns for 6 products per user
- **Arms (Products):** 6

| Product | Avg Net Reward | Always Profitable? |
|---|---|---|
| Product 1 | ~7.06 | ✅ Yes |
| Product 2 | ~66.99 | ✅ Yes (Best) |
| Product 3 | ~31.42 | ✅ Yes |
| Product 4 | ~9.60 | ✅ Yes |
| Product 5 | ~-17.06 | ❌ No (Worst) |
| Product 6 | ~36.99 | ✅ Yes |

---

## 🤖 Algorithms Implemented

### 1. Random Policy
- Randomly recommends one of the 6 products each round
- Baseline for comparison — no learning involved

### 2. Try-Then-Exploit (TTE)
- Explores all arms for a fixed number of initial rounds
- Then commits to the best-performing arm
- Sensitive to early sampling luck

### 3. Epsilon-Greedy (ε = 0.02, 0.1, 0.25)
- Exploits the best-known arm most of the time
- Explores a random arm with probability ε
- ε = 0.1 gives the best balance

### 4. UCB1 (Upper Confidence Bound)
- Selects arms based on reward estimate + exploration bonus
- Dynamically balances uncertainty vs. known reward
- Best overall performer

---

## 📈 Results (300 Rounds)

| Strategy | Total Cumulative Reward |
|---|---|
| Random | ~6,141 |
| Try-Then-Exploit | ~17,424 |
| Epsilon-Greedy (ε=0.02) | ~16,396 |
| Epsilon-Greedy (ε=0.1) | ~17,654 |
| Epsilon-Greedy (ε=0.25) | ~13,152 |
| **UCB1** | **~19,827 ✅ Best** |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/multi-armed-bandit-recommendation.git
cd multi-armed-bandit-recommendation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook Multi_Armed_Bandit_Experiment.ipynb
```

> ⚠️ Make sure `Dataset_Product_Recommendation.csv` is in the same directory as the notebook before running.

---

## 🛠️ Tech Stack

- **Python 3.12**
- **NumPy** — simulations and reward computation
- **Pandas** — dataset loading and analysis
- **Matplotlib** — cumulative reward plots

---

## 💡 Key Takeaways

- UCB1 consistently achieves the highest cumulative reward by intelligently handling uncertainty
- Epsilon-Greedy with ε=0.1 is a simpler yet competitive alternative
- Try-Then-Exploit can fail if the exploration phase is unlucky
- Random policy is always the worst — confirms the value of learning-based strategies
- All algorithms assume **stationary reward distributions**; non-stationary settings would require contextual or sliding-window bandits

---

## 🔮 Future Improvements

- Implement Thompson Sampling for Bayesian exploration
- Extend to Contextual Bandits using user features
- Handle non-stationary rewards with sliding-window UCB
- Scale to more products / arms
