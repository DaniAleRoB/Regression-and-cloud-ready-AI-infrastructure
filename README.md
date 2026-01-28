---

# Stellar Luminosity Modeling

### Linear and Polynomial Regression from First Principles

## 1. Project Overview

This repository contains the implementation of **linear regression and polynomial regression from first principles**, applied to a simplified astrophysical problem: **modeling stellar luminosity as a function of stellar mass and temperature**.

The project is part of a **Machine Learning Bootcamp embedded in a Digital Transformation and Enterprise Architecture course**, where machine learning is treated as a **core architectural capability** rather than a black-box tool.

Instead of relying on high-level machine learning libraries, all models are built explicitly by defining:

* Hypothesis functions
* Loss (cost) functions
* Gradient-based optimization
* Training loops
* Convergence analysis

This approach ensures a deep understanding of how learning systems behave, scale, and fail—an essential skill for enterprise architects and engineers.

---

## 2. Problem Description

### Astrophysical Motivation

Astronomy is a data-driven science where physical laws are inferred from observation. One of the most important relationships in stellar astrophysics is the connection between:

* **Stellar mass (M)**
* **Effective temperature (T)**
* **Luminosity (L)**

For main-sequence stars, luminosity grows rapidly with mass and is influenced by temperature and nonlinear interactions between physical quantities.

In this project, we model stellar luminosity using:

* A **linear model** with one feature (mass)
* A **polynomial model** with multiple features and interaction terms

---

## 3. Repository Structure

```
/
├── README.md
├── imges
├── noteBooks
    ├── 01_part1_linreg_1feature.ipynb
    ├── 02_part2_polyreg.ipynb

```

* All datasets are **hard-coded NumPy arrays**
* All code lives **inside the notebooks**
* No external ML libraries are used

---

## 4. Tools and Libraries Used

Allowed libraries only:

* **Python 3**
* **NumPy** – numerical computation and vectorization
* **Matplotlib** – inline data visualization

❌ Not used:

* scikit-learn
* TensorFlow / PyTorch
* statsmodels
* Any automatic fitting or optimization library

---

## 5. Dataset and Notation

### Variables

* **M**: Stellar mass (in units of solar mass, M⊙)
* **T**: Effective stellar temperature (Kelvin)
* **L**: Stellar luminosity (in units of solar luminosity, L⊙)

### Part I Dataset (One Feature)

```python
M = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
L = [0.15, 0.35, 1.00, 2.30, 4.10, 7.00, 11.2, 17.5, 25.0, 35.0]
```

### Part II Dataset (Two Features)

```python
M = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
T = [3800, 4400, 5800, 6400, 6900, 7400, 7900, 8300, 8800, 9200]
L = [0.15, 0.35, 1.00, 2.30, 4.10, 7.00, 11.2, 17.5, 25.0, 35.0]
```

---

## 6. Notebook 1 — Linear Regression with One Feature

### Objective

Model stellar luminosity as a linear function of mass:

$$\hat{L} = wM + b$$

### Key Steps Implemented

1. **Dataset visualization**

   * Scatter plot of M vs L
   * Discussion of approximate linearity and physical plausibility

     ![alt text](<images/M vs L scatter plot.png>)

2. **Model and loss function**

   * Mean Squared Error (MSE):
     $$J(w,b) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{L}_i - L_i)^2$$

3. **Cost surface analysis (mandatory)**

   * Cost evaluated over a grid of (w) and (b)
   * Visualization via contour or surface plot
   * Interpretation of the global minimum

     ![alt text](<images/Cost surface.png>)

4. **Gradient derivation and implementation**

   * Analytical gradients for (w) and (b)
   * Both non-vectorized and vectorized implementations

5. **Gradient descent experiments**

   * Multiple learning rates tested
   * Loss vs iteration plots to analyze convergence behavior

   ![alt text](<images/Loss vs iterations for different learning rates.png>)

6. **Final fit visualization**

   * Regression line plotted over the data
   * Discussion of systematic errors

   ![alt text](<images/Final linear fit.png>)


### Conceptual Discussion

* Physical interpretation of the slope (w)
* Limitations of linear models for stellar physics

---

## 7. Notebook 2 — Polynomial Regression with Interaction Terms

### Objective

Capture nonlinear and interaction effects using polynomial feature engineering:

$$\hat{L} = Xw + b$$

### Feature Map

The design matrix includes:

$$X = [M,\ T,\ M^2,\ M \cdot T]$$

(No constant column; bias is handled explicitly.)

---

### Key Steps Implemented

1. **Dataset visualization**

   * Luminosity vs mass with temperature encoded as color

     ![alt text](<images/L vs M colored by T.png>)

2. **Feature engineering**

   * Vectorized construction of polynomial and interaction terms

3. **Vectorized loss and gradients**

   * Fully vectorized gradient descent implementation

4. **Model comparison (mandatory)**

| Model | Features        | Description            |
| ----- | --------------- | ---------------------- |
| M1    | [M, T]          | Linear baseline        |
| M2    | [M, T, M²]      | Nonlinear mass effect  |
| M3    | [M, T, M², M·T] | Full interaction model |

* Final loss and parameters reported
* Predicted vs actual plots for each model

![alt text](<images/Predicted vs actual for M1, M2, M3.png>)

5. **Cost vs interaction coefficient (mandatory)**

   * Cost evaluated as the interaction weight varies
   * Demonstrates the importance of interaction effects

    ![alt text](<images/Cost vs interaction weight.png>)

6. **Inference demonstration (mandatory)**

   * Prediction for a new star:

     * (M = 1.3), (T = 6600)
     * Visualization showing where the prediction lies relative to the data
     
     ![alt text](<images/Inference visualization.png>)

---

## 8. Results Summary

* Polynomial regression significantly reduces loss compared to linear models
* Interaction terms capture meaningful physical relationships
* Vectorization improves numerical stability and convergence speed
* The inferred luminosity values are physically reasonable and consistent with observed trends

---

## 9. AWS SageMaker Execution Evidence

### Execution Environment

Both notebooks were uploaded and executed in **AWS SageMaker** (Studio / Notebook Instance).

### Evidence Included

* Screenshots showing:

  * Both notebooks visible in SageMaker
  * Successful execution of all cells
  * Rendered plots
* Comparison between local execution and SageMaker:

  * No differences in numerical results
  * SageMaker provides a controlled, scalable execution environment suitable for enterprise workflows

*(Screenshots inserted here)*

---

## 10. Final Remarks

This project demonstrates how machine learning models can be:

* Built from first principles
* Interpreted physically
* Validated numerically
* Executed reliably in cloud environments

Such understanding is essential when integrating intelligent components into **enterprise-scale digital architectures**.

---

