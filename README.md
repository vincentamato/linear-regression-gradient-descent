# Linear Regression with Gradient Descent

This is a simple and straightforward project to predict California housing prices using linear regression with gradient descent.

## Design Choices
**Gradient Descent:** I chose to use gradient descent instead of an analytical solution because gradient descent is scalable to larger datasets and can be easily extended to other machine learning problems. It also provided me with a hands-on way to understand the optimization process. 

**Normalizing the Data:** Normalizing the data was crucial for this dataset because the features have vastly different scales. For example, the median income feature ranges from 0.5 to 15, while the population feature ranges from 3 to 35,682. Without normalization, features with larger scales would dominate the gradient updates, leading to poor convergence and suboptimal performance of the gradient descent algorithm.

## Usage
**1. Clone the repository:**
```
git clone https://github.com/vincentamato/linear-regression-gradient-descent.git
```

**2. Install the required dependencies:**
```
pip install numpy pandas matplotlib scikit-learn
```

**3. Run the Jupyter Notbook:**
```
jupyter notebook linear-regression-gradient-descent.ipynb
```

**4. Train the model:**
The notebook will guide you through loading the data, preprocessing it, and training the linear regression model using gradient descent.

## Equations
**1. Prediction (Forward Pass):**
   $$\hat{y} = X\textbf{w} + b$$
   where:
   - $\hat{y}$ is the predicted value
   - $X$ is the input feature matrix
   - $\textbf{w}$ is the weight vector
   - $b$ is the bias term

**2. Mean Squared Error (Loss Function):**
   $$L = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y_i})^2$$
   where:
   - $L$ is the loss
   - $m$ is the number of samples
   - $y_i$ is the actual value
   - $\hat{y_i}$ is the predicted value

**3. Gradients (Vectorized):**
   $$\frac{\partial L}{\partial W} = -\frac{2}{m}X^T(y-\hat{y})$$
   $$\frac{\partial L}{\partial b} = -\frac{2}{m}(y-\hat{y})$$
   where:
   - $\frac{\partial L}{\partial W}$ is the weight gradient
   - $\frac{\partial L}{\partial b}$ is the bias gradient
  
**4. Optimization (Gradient Descent):**
   $$W \leftarrow W - \alpha\frac{\partial L}{\partial W}$$
   $$b \leftarrow b - \alpha\frac{\partial L}{\partial b}$$
   where:
   - $\alpha$ is the learning rate

Feel free to tweak the learning rate, number of iterations, and other parameters to see how they affect the model's performance.
