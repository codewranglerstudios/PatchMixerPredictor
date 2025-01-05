# **PatchMixerPredictor**  
### *An Experimental Stock Trading Strategy Leveraging the PatchMixer CNN-Based Model*  
### *This repository is under development*  

---

## **Overview**  
**PatchMixerPredictor** is an innovative stock trading strategy that integrates:  
- A **Convolutional Neural Network (CNN)** for price forecasting Using the PatchMixer model.  
- A **Reinforcement Learning (RL)** agent for decision-making and trading execution.  

This experimental approach combines predictive analytics and adaptive learning to identify profitable trading opportunities while managing risks effectively.  
---

## **Key Features**  
- **Advanced CNN Model:** Captures historical trends to predict future price movements.  
- **Dynamic RL Agent:** Adapts to changing market conditions to maximize portfolio returns.  
- **Multi-Step Forecasting:** Enables both short- and medium-term trading strategies.  
- **Risk-Aware Trading:** Considers transaction costs and market volatility in its decisions.  

---

## **How It Works**

### **1. CNN-Based Forecasting Model**
- **Purpose:** Predicts stock Close prices over the next 128 time steps.  
- **How It Works:**  
  - Analyzes 512 time steps of historical data to detect actionable patterns.  
  - Outputs a sequence of future prices to forecast trends (upward, downward, or stable).  
- **Why It Matters:**  
  - Filters noise to focus on meaningful trends.  
  - Provides a solid foundation for informed trading decisions.  

### **2. Reinforcement Learning Agent**
- **Purpose:** Makes trading decisions based on CNN predictions and additional metrics.  
- **How It Works:**  
  - **State Variables:** Predictions, account balance, transaction costs, and risk metrics.  
  - **Actions:**  
    - **Buy:** When significant price increases are forecasted.  
    - **Sell:** To lock in profits or minimize losses during expected declines.  
    - **Hold:** When the outlook is neutral or uncertain.  
  - **Reward:** Optimized based on portfolio performance or risk-adjusted metrics like the Sharpe ratio.  
- **Why It Matters:**  
  - Adapts to new market conditions using real-time feedback.  
  - Balances risk and reward to avoid overtrading or risky decisions.  

### **3. Profit Generation**
The strategy's design maximizes profit potential through:  
- **Predictive Power:** Reliable trends forecasted by the CNN.  
- **Dynamic Adaptation:** RL agent adjusts to evolving market conditions.  
- **Risk Management:** Transaction costs and volatility are considered in every decision.  
- **Multi-Step Predictions:** Captures opportunities across various time horizons.  
- **Data-Driven Actions:** Combines statistical accuracy with policy optimization.  

 ### ** Credits**
 The prediction Model variant used in the first step, PatchMixer, has been chosen because it is
 designed specifically to capture long term trends making it perfect for multi-year data
 driven analyzation.
 The original research paper can be found here: https://github.com/Zeying-Gong/PatchMixer

---

## **Getting Started**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/codewranglerstudios/PatchMixerPredictor.git
cd PatchMixerPredictor
```
### **Step 2: Install Dependencies**
Ensure you have Python installed. Then, run the following command to install the required packages:

```bash
pip install -r requirements.txt
```
### **Step 3: Run the Model**
Start the model training and evaluation process by running:

```bash
python main.py
```
### **Step 4: Visualize Results**
Explore predictions and trading performance using the provided Jupyter Notebook:

```bash
# note: under construction
jupyter notebook notebooks/PatchMixerAnalysis.ipynb
```
