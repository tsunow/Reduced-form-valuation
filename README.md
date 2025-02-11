# Two‐Factor CIR Model Estimation Using a (Quasi) Kalman Filter

This document explains how and **why** we apply a Kalman Filter to estimate a **two‐factor CIR** (Cox–Ingersoll–Ross) term‐structure model from observed yields. We will:

1. **Motivate** the goal of the Kalman Filter in this context.  
2. **Outline** the state‐space structure, main variables, and steps.  
3. **Walk through** each part of the provided code, showing how it maps onto the Kalman Filter procedures.  

---

## 1. Goal of the Kalman Filter in Multi‐Factor CIR Estimation

In a **two‐factor CIR model**, we have two latent state variables,
\[
   x_{1,t} \quad \text{and} \quad x_{2,t}.
\]
These follow continuous‐time stochastic differential equations of the form
\[
  \mathrm{d}x_{j}(t) 
    = \alpha_{j}\bigl[\beta_{j} - x_{j}(t)\bigr]\;\mathrm{d}t 
      \;+\; \sigma_{j}\,\sqrt{x_{j}(t)}\;\mathrm{d}W_{j}(t),
  \quad j = 1,2.
\]
Meanwhile, we observe **bond yields** at discrete times \(t=1,\dots,T\) for various maturities \(\tau\). The yields are linear functions of the states in **risk‐neutral** form, with “speed” parameters \(\alpha_j + \lambda_j\). We typically denote:
\[
  Y_{t}(\tau_i) 
    = -\frac{1}{\tau_i}
      \sum_{j=1}^{2}\ln A_{j}(\tau_i)
      \;+\;\frac{1}{\tau_i}
      \sum_{j=1}^{2}B_{j}(\tau_i)\,x_{j}(t).
\]

A **Kalman Filter** is used here to:

1. **Infer** the latent states \(x_{1,t}, x_{2,t}\) at each date \(t\) from the observed yields (measurement step).  
2. **Accumulate** the probability (likelihood) of these observations under a (quasi)‐Gaussian assumption, so we can perform **Maximum Likelihood Estimation (MLE)** of the model parameters \(\{\alpha_j,\beta_j,\sigma_j,\lambda_j\}\) **and** the measurement noise.  

Because a CIR model is *not perfectly linear* (the variance depends on \(\sqrt{x}\)), the code below uses an **approximate** or **extended** Kalman Filter method. Nonetheless, it follows the standard KF structure of “predict” and “update” steps.

---

## 2. Main Variables and Steps

1. **State Vector**: 
   \[
     \mathbf{x}_{t} 
       = \begin{pmatrix}
           x_{1,t} \\[6pt]
           x_{2,t}
         \end{pmatrix}.
   \]
   We treat it in **discrete time** with step size \(\Delta t\).

2. **State Transition** \(\mathbf{x}_{t+1} = F \mathbf{x}_{t} + \mathbf{c} + \varepsilon_{t}\):
   - \(F\) is roughly \(\mathrm{diag}\bigl[e^{-\alpha_1 \Delta t},\, e^{-\alpha_2 \Delta t}\bigr]\).  
   - \(\mathbf{c}\) is \(\bigl[\beta_1(1-e^{-\alpha_1\Delta t}),\,\beta_2(1-e^{-\alpha_2\Delta t})\bigr]^\top\).  
   - \(\varepsilon_{t}\) is an approximate **Gaussian** noise with covariance \(Q_t\).  

3. **Measurement Equation** \(\mathbf{Y}_{t} = A_{\mathrm{vec}} + B_{\mathrm{mat}}\,\mathbf{x}_{t} + \eta_{t}\):
   - We compute \(\mathbf{A}_{\mathrm{vec}}, \mathbf{B}_{\mathrm{mat}}\) from the CIR **closed‐form** for yields:
     \[
       Y_{t}(\tau)
         = A(\tau) + B_{1}(\tau)\,x_{1,t} + B_{2}(\tau)\,x_{2,t}.
     \]
   - \(\eta_{t}\) is measurement noise, modeled as \(\mathcal{N}(\mathbf{0},\,R)\).

4. **Kalman Filter Steps**  
   - **Predict**:  
     \[
       \hat{\mathbf{x}}_{t|t-1} = F\,\hat{\mathbf{x}}_{t-1|t-1} + \mathbf{c},
       \quad
       P_{t|t-1} = F\,P_{t-1|t-1}\,F^\top + Q_{t-1}.
     \]
   - **Update**:
     - Innovation: \(\nu_{t} = \mathbf{Y}_{t} - \bigl[A_{\mathrm{vec}} + B_{\mathrm{mat}}\,\hat{\mathbf{x}}_{t|t-1}\bigr].\)
     - Covariance: \(S_{t} = B_{\mathrm{mat}}\,P_{t|t-1}\,B_{\mathrm{mat}}^\top + R.\)
     - Kalman Gain: \(K_{t} = P_{t|t-1}\,B_{\mathrm{mat}}^\top\,S_{t}^{-1}.\)
     - State Update: \(\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + K_{t}\,\nu_{t}.\)
     - Cov. Update: \(P_{t|t} = (I - K_{t}\,B_{\mathrm{mat}})\,P_{t|t-1}\,(I - K_{t}\,B_{\mathrm{mat}})^\top + K_{t}\,R\,K_{t}^\top.\)

5. **Log‐Likelihood**  
   - Each time step \(t\), the innovation \(\nu_{t}\) is assumed \(\mathcal{N}(0,S_{t})\).  
   - So we accumulate
     \[
       \ell_t
         = -\frac{1}{2}\Bigl[
           \ln\bigl|S_{t}\bigr|
           + \nu_{t}^\top\,S_{t}^{-1}\,\nu_{t}
           + N \ln(2\pi)
         \Bigr].
     \]
   - The code **sums** \(\ell_t\) over \(t\). Minimizing the negative of that sum yields the MLE estimates of the CIR parameters.

---

## 3. Code Walk‐Through

Below, we highlight each section of the provided code and how it reflects the above steps.

### 3.1 `log_likelihood_2factor(params, Y, tau_vec, dt)`

- **Unpack Parameters**:  
  \(\alpha_1, \beta_1, \sigma_1, \lambda_1,\; \alpha_2, \beta_2, \sigma_2, \lambda_2,\; \{\text{meas\_noise}_i\}\).  
- **Compute Measurement Coefficients** (\(A_{j}, B_{j,1}, B_{j,2}\) for each maturity). Internally, it does:
  \[
    \Gamma_j = \sqrt{(\alpha_j + \lambda_j)^2 + 2\,\sigma_j^2},
    \quad
    B_{j}(\tau),\; A_{j}(\tau) \quad\text{(CIR bond pricing)}.
  \]
  Then uses  
  \[
    \frac{-\bigl(A_{1}(\tau)+A_{2}(\tau)\bigr)}{\tau}
    \quad \text{and} \quad
    \frac{B_{1}(\tau)}{\tau},\;\frac{B_{2}(\tau)}{\tau}.
  \]

- **Initialize the Filter**:  
  - \(\hat{x}_{0|0}\approx (\beta_1,\beta_2)\).  
  - \(P_{0|0}\) from unconditional variance \(\sigma_j^2\beta_j/(2\alpha_j)\).

- **Time Loop**: For \(t=1,\dots,T\) (the discrete observation times):
  1. **Predict** \(\hat{x}_{t|t-1} = F \hat{x}_{t-1|t-1} + c\), \(\;P_{t|t-1}=F\,P_{t-1|t-1}F^\top + Q\).  
     The code approximates \(Q\) by 
     \(\mathrm{diag}\Bigl[\sigma_1^2\,\max(\hat{x}_{1},0)\,\Delta t,\;\sigma_2^2\,\max(\hat{x}_{2},0)\,\Delta t\Bigr].\)
  2. **Measurement**: 
     \[
       y_{\mathrm{pred}} = A_{\mathrm{vec}} + B_{\mathrm{mat}}\,x_{\mathrm{pred}},
       \quad
       \nu = Y_{t} - y_{\mathrm{pred}},
       \quad
       S = B_{\mathrm{mat}}\,P_{\mathrm{pred}}\,B_{\mathrm{mat}}^\top + R.
     \]
  3. **Update**: 
     \[
       K = P_{\mathrm{pred}}\,B_{\mathrm{mat}}^\top\,S^{-1},
       \quad
       \hat{x}_{t|t} = x_{\mathrm{pred}} + K \,\nu,
       \quad
       P_{t|t} = \dots 
     \]
  4. **Log‐likelihood**: 
     \[
       \ell_{t} 
         = -\tfrac{1}{2}\bigl[N\ln(2\pi) + \ln|S| + \nu^\top S^{-1}\nu\bigr].
     \]
     Accumulate in `logL`.

- **Return** the negative log‐likelihood for the optimizer.

### 3.2 `estimate_2factor_CIR(Y, tau_vec, dt)`

- Sets an **initial guess** for all parameters.  
- Defines **bounds** (e.g.\ \(\alpha_j,\beta_j,\sigma_j>0\), measurement noises>0, etc.).  
- Calls:
  \[
    \text{res} = \min_{\text{params}} 
      \Bigl[\text{log\_likelihood\_2factor}(\text{params},Y,\tau\_vec,dt)\Bigr].
  \]
  In other words, it does MLE.

- On **success**, `res.x` has the final parameter estimates, and `res.fun` is the minimized negative log‐likelihood.

---

## 4. Summary

- **We have two latent states** evolving under approximate CIR dynamics in discrete steps (\(F, c, Q\) in the “predict” step).  
- **Bond yields** are measured from the **CIR closed‐form** expression, forming a *linear measurement equation* in the states with noise covariance \(R\).  
- **Kalman Filter** logic then: (1) predict the next‐step mean & covariance of states, (2) update them with new yield data, and (3) accumulate the *Gaussian* log‐likelihood of the yield “innovations.”  
- **Optimizer**: By finding parameter values that maximize this likelihood, we get an approximate MLE of \(\{\alpha_j,\beta_j,\sigma_j,\lambda_j\}\) plus measurement noise.

