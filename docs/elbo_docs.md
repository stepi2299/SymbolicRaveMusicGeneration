# ELBO Loss in Variational Autoencoders (VAEs)

## What is ELBO Loss?

**ELBO** = Evidence Lower Bound.  
It is the training objective of **Variational Autoencoders (VAEs)** and comes from variational inference.  
Since maximizing the exact data likelihood is intractable, VAEs maximize a *lower bound* on it, called ELBO.

The ELBO loss is minimized during training and consists of two terms:

\[
\mathcal{L}_{\text{ELBO}}(x) = 
\underbrace{\mathbb{E}_{q_\phi(z|x)}[-\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} +
\underbrace{D_{KL}(q_\phi(z|x) \,\|\, p(z))}_{\text{KL Divergence}}
\]

---

## Components of the ELBO

### 1. Reconstruction Loss
- Measures how well the decoder can reconstruct the input \(x\) from latent vector \(z\).
- Often implemented as:
  - **MSE** (continuous data, e.g. images).
  - **Cross-Entropy** (discrete/categorical data, e.g. text or music tokens).
- **Good reconstruction → small value.**
- **Bad reconstruction → large value.**

### 2. KL Divergence Term
- Regularizes the encoder distribution \(q_\phi(z|x)\) to be close to the prior \(p(z)\) (usually \(N(0, I)\)).
- **KL = 0** means encoder matches the prior exactly.
- If KL is too small → risk of **posterior collapse** (latent variables unused).
- If KL is too large → encoder deviates too much from the prior, making sampling less meaningful.

### 3. Total ELBO Loss
- Sum of reconstruction loss and KL divergence.
- Represents a **tradeoff**:
  - Minimize reconstruction error.
  - Keep latent space regularized.

---

## Training Behavior

Typical dynamics during training:

- **Reconstruction Loss**  
  - High at the beginning (poor reconstruction).  
  - Decreases as the model learns.  

- **KL Divergence**  
  - Small at the beginning (encoder close to prior, little information in latent).  
  - Increases as the encoder starts using latent variables.  
  - Should stabilize, not collapse or explode.  

- **Total ELBO Loss**  
  - High at the beginning.  
  - Decreases as reconstruction improves.  
  - Stabilizes at a balance point between reconstruction accuracy and latent space regularization.  

---

## Example (Toy Training Log)

| Epoch | Recon Loss | KL Divergence | Total ELBO |
|-------|------------|---------------|------------|
| 1     | 5.2        | 0.01          | 5.21       |
| 10    | 2.5        | 0.3           | 2.8        |
| 50    | 1.2        | 1.1           | 2.3        |
| 100   | 1.0        | 1.5           | 2.5        |

**Interpretation:**
- Early: dominated by reconstruction.  
- Mid: KL starts to rise as latent variables become informative.  
- Later: balance between good reconstruction and regularized latent space.  

---

## Summary

- **Reconstruction Loss**: measures accuracy of reconstructions. Should **decrease** with training.  
- **KL Divergence**: measures how much the encoder’s latent distribution deviates from the prior. Should not vanish (collapse) or blow up.  
- **Total ELBO**: decreases early, then stabilizes at a tradeoff point.  
- This balance ensures VAEs can both reconstruct inputs and generate meaningful samples.
