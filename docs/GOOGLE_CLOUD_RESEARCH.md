# Google AI Pro & Google Cloud Compute Research for Titanium Ataxx

Written 2026-10-27. Analysis of the Google AI Pro / Google One AI Premium subscription and bundled Google Cloud credits for Ataxx NNUE training and engine compute.

---

## 1. Credit Structure & Value Assessment

The subscription benefits include monthly Google Cloud / Vertex AI / AI Studio credits through the **Google Developer Program**:
- **Google AI Pro / One AI Premium**: Provides **$10 to $100** in recurring Google Cloud credits depending on tier and promotions.
- **New Account Bonus**: Stacks with the standard **$300 Google Cloud Free Trial** if linked to a fresh GCP billing profile.

---

## 2. Applicability to Titanium Ataxx

| Workflow | Suitability | Verdict |
| :--- | :--- | :--- |
| **CPU Relabeling (`labelbin` 5k nodes)** | **Poor** | Google Cloud Compute Engine (GCE) e2/c2/c3 VMs cost ~$0.035–$0.05/vCPU-hr. While usable, Docker cloud sandboxes (already 40 free vCPUs) and Modal (faster launch, zero setup) are superior. Not recommended to waste GCP credits here. |
| **NNUE PyTorch Training (Stage 1 / HPO)** | **High** | Vertex AI Custom Jobs or Compute Engine with NVIDIA L4/T4/A100 GPUs can run `train_nnue.py` directly without drawing from the $30 Modal budget. |
| **Gemini LLM APIs (AI Studio)** | **Limited** | LLMs cannot evaluate 100,000 chess/ataxx leaf nodes/sec. However, useful for offline search analysis, rule tuning prompts, and automated commit/bench summaries. |

---

## 3. Recommended Setup (When needed for GPU Training)

If activating GCP credits to train the new NNUE net after the relabel run completes:
1. **Activate Benefits**:
   - Go to [developers.google.com/program/my-benefits](https://developers.google.com/program/my-benefits) with the Google account owning the subscription.
   - Claim the monthly Cloud Credit promo code.
2. **Apply to Billing Profile**:
   - In Google Cloud Console (`console.cloud.google.com/billing`), apply the credit code to the active billing account.
3. **Compute Allocation**:
   - Provision an `n1-standard-8` or `g2-standard-8` instance with an NVIDIA L4 GPU (~$0.70/hr).
   - $100 of credits pays for **~140 hours of continuous GPU training**, saving Modal GPU compute entirely for time-critical gates.
