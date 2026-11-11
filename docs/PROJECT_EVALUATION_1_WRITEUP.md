# BML MUNJAL UNIVERSITY
### School of Engineering & Technology | Department of Computer Science & Engineering
**B.Tech CSE — Project-IV (PRJ4) / Capstone Project**  
**Evaluation 1 Write-up (Weightage: 5 Marks)**

---

* **Project Title**: Multimodal Emotion Recognition & Explainable Assistive Agent: Context-Aware Cross-Modal Modeling with Foundation Speech Representations and Generative Explanation
* **Supervisor**: Dr. Yogesh Gupta, Professor, Dept. of CSE
* **Team Members**:
  1. **Suhani Aggarwal** (Roll No: 230547)
  2. **Aditya Parecha** (Roll No: 230522)
  3. **Harsh Pratap Singh** (Roll No: 230518)
* **Date & Time of Evaluation**: September 11, 2026 | 3:15 PM (Supervisor's Cabin)

---

## 1. Problem Definition

Human emotional expression in multi-party conversations is an inherently multi-layered, multimodal phenomenon that text alone cannot reliably decipher. In natural conversations, the **"textual dominance"** effect often masks or outright contradicts a speaker’s true affective state—e.g., a semantically neutral or polite statement (*"Oh, that's just fantastic"*) delivered with sharp pitch variance, flat intensity, or vocal strain indicative of sarcasm, frustration, or suppressed anger.

While Project-III established an initial multimodal baseline (fine-tuned BERT + Librosa MFCCs with scalar softmax gating), rigorous empirical failure analysis revealed four fundamental bottlenecks that define the problem space for Project-IV:

1. **Acoustic Feature Flattening**: Traditional audio feature engineering (e.g., 13 MFCCs averaged over time into 15 static scalars) discards continuous pitch contours, vocal tremor, pauses, and speech cadence. Consequently, the audio stream loses discriminative power, causing multimodal models to underperform unimodal text baselines.
2. **Severe Class Imbalance & Mode Collapse**: Conversational datasets (such as MELD) exhibit extreme skew (Neutral ~47%, while Fear/Disgust <3%). Standard Focal Loss ($\gamma=2$) failed to penalize majority bias, resulting in catastrophic mode collapse where the model predicted only high-arousal classes (*Joy* and *Anger*), scoring **0.00 F1 on Sadness, Surprise, and Neutral**.
3. **Conversational Amnesia (Turn-Level Isolation)**: Utterances are conventionally evaluated in total isolation. Disregarding the preceding conversational trajectory ($K$ previous turns), speaker identity, and emotional inertia (the psychological resistance to abrupt affective transitions) leads to severe misclassifications during nuanced dialogue.
4. **Black-Box Opacity in Assistive Settings**: Current Emotion Recognition in Conversation (ERC) architectures merely predict categorical labels with confidence scores without explaining *why* an emotion was inferred, rendering them unsuitable for clinical, neurodiverse, or real-time human-agent assistive interaction.

---

## 2. Objectives and Outcomes

### Primary Objectives
1. **High-Definition Speech Feature Representation**: Replace handcrafted static acoustic features with self-supervised speech foundation models (**HuBERT / Wav2Vec 2.0**) to extract frame-level prosodic, pitch, and cadence dynamics.
2. **Fine-Grained Cross-Modal Alignment**: Implement **Bidirectional Cross-Modal Attention (CMA)** to dynamically align textual token embeddings (**DeBERTa-v3**) with acoustic temporal frames, resolving semantic-paralinguistic contradictions (e.g., sarcasm).
3. **Contextual Dialogue Tracking**: Develop a **Contextual Reaffirmation Layer** with a sliding turn-history buffer ($K=3-5$ turns) and speaker tracking to model emotional inertia and conversation trajectory.
4. **Imbalance Elimination**: Restore the complete 7-class MELD taxonomy and eliminate mode collapse using **Class-Balanced Focal Loss (CB-Focal)** coupled with **Post-Hoc Logit Calibration**.
5. **Explainable AI (XAI) & Generative Assistance**: Formulate an attention-rollout evidence extraction engine that extracts salient linguistic and acoustic triggers, feeding them into a quantized local Small Language Model (**Llama 3.2 3B / Phi-3.5 via Ollama**) to generate compassionate, conversational explanations vocalized via real-time Text-to-Speech (**TTS**).

### Quantifiable Outcomes
* **Target Benchmark Performance**: Achieve $\ge 68.0\%$ Weighted F1-Score across all 7 emotion classes on the full MELD benchmark (surpassing our PRJ3 baseline of 63.26% on 5 classes and exceeding recent published literature such as GAT-CRESA at 65.04%).
* **Minority Class Viability**: Achieve $> 0.35$ per-class F1-score on historically collapsed low-arousal classes (*Sadness*, *Surprise*, *Fear*, *Disgust*), verifying zero mode collapse.
* **Assistive XAI Pipeline**: Generate structured, verifiable reasoning JSON packets (acoustic cues + textual triggers + turn shifts) with human-aligned natural explanations.
* **Low-Latency Edge Deployment**: Achieve an end-to-end inference latency under **450 ms** on edge/consumer hardware for real-time interactive turn evaluation.

---

## 3. Methodology

The PRJ4 system is structured into four integrated, modular processing stages:

```
[User Audio & Text / Whisper ASR]
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Deep Feature Representation                             │
│    • Audio Stream : HuBERT / Wav2Vec 2.0 (Frame-level 768d)  │
│    • Text Stream  : DeBERTa-v3-base (Contextual Tokens 768d)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Bidirectional Cross-Modal Attention (CMA) Fusion         │
│    • Text attends to acoustic pitch/energy spikes           │
│    • Audio attends to salient affective keywords            │
│    • Computes Initial Turn Emotion Probability P(E_t)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Contextual Reaffirmation & Class-Balanced Optimization   │
│    • Sliding Turn Memory (Last K turns + Speaker IDs)       │
│    • Bi-GRU / Temporal Smoothing for Emotional Inertia      │
│    • Loss: Class-Balanced Focal Loss + Logit Adjustment     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Explainable AI (XAI) & Generative Assistive Delivery     │
│    • Salience Extraction (Attention Rollout + Pitch Peaks)  │
│    • Structured Evidence Packet -> Local SLM (Llama 3.2 3B) │
│    • Natural Spoken Explanation via Local TTS (Piper / Edge)│
└─────────────────────────────────────────────────────────────┘
```

### Module Breakdown
1. **Deep Feature Representation**:
   * *Acoustic Backbone*: Raw 16 kHz audio is processed through `facebook/hubert-base-ls960` or `wav2vec2-base`, producing temporal sequence embeddings $H_a \in \mathbb{R}^{T_a \times 768}$, preserving intonation, micro-tremors, and pauses.
   * *Textual Backbone*: Utterance text (or Whisper ASR output) is encoded via `microsoft/deberta-v3-base`, producing contextual token vectors $H_t \in \mathbb{R}^{T_t \times 768}$.
2. **Bidirectional Cross-Modal Attention (CMA)**:
   * Instead of naive scalar gating, we formulate scaled multi-head cross-attention where text acts as Query ($Q_t$) over audio Key-Value ($K_a, V_a$), and vice versa:
     $$\text{CMA}(T, A) = \text{Softmax}\left(\frac{Q_t K_a^T}{\sqrt{d_k}}\right)V_a$$
   * This aligns specific words (e.g., *"Sure"*) with temporal pitch dips or flat prosody, isolating paralinguistic sarcasm.
3. **Contextual Reaffirmation & Imbalance Optimization**:
   * Initial turn estimates are passed into a conversational memory module tracking dialogue turns:
     $$h_t = \text{GRU}( [e_t; s_t], h_{t-1} )$$
     where $e_t$ is the cross-modal turn embedding and $s_t$ is the speaker identity encoding.
   * Optimization employs Class-Balanced Focal Loss with effective number of samples $E_n = \frac{1 - \beta^n}{1 - \beta}$:
     $$\mathcal{L}_{\text{CB-Focal}} = - \frac{1 - \beta}{1 - \beta^{n_y}} (1 - p_t)^\gamma \log(p_t)$$
   * Post-hoc logit adjustment enforces mathematically fair decision margins for minority classes: $\hat{z}_c = z_c - \tau \log(\pi_c)$.
4. **Explainable AI (XAI) & Generative Delivery**:
   * Attention maps and acoustic prosody deviations are parsed into a structured evidence packet (JSON) containing: primary emotion, acoustic anomalies (pitch variation, speech rate), and textual triggers.
   * A quantized local SLM (Llama 3.2 3B via Ollama) translates this packet into warm, concise assistive feedback without hallucinating predictions.
   * Piper TTS or Edge-TTS synthesizes the output into natural audio for real-time auditory assistance.

---

## 4. Progress (Work Done)

Since completing PRJ3, the team has completed the foundational diagnostic, data preparation, and architectural prototyping phases for PRJ4:

* **Comprehensive Failure Diagnosis of PRJ3 Model**:
  * Conducted exhaustive post-mortem evaluation of the PRJ3 checkpoint (`best3.pt`). Quantified the exact failure modes: severe mode collapse (`RESULTS1.jpeg`) yielding 61% false Joy and 39% false Anger on neutral samples, alongside flat 0.00 F1 scores for Sadness and Surprise (`results2.jpeg`, `results3.jpeg`).
  * Traced the architectural cause directly to 15-dimensional Librosa MFCC mean-pooling and uncalibrated Focal Loss.
* **Dataset Alignment & Caching Pipeline**:
  * Restructured the complete MELD dataset, reinstating all 7 emotion classes (re-incorporating *Fear* and *Disgust* previously discarded in PRJ3).
  * Built an automated audio-text alignment and feature-caching pipeline using PyTorch and HuggingFace Transformers, caching HuBERT audio embeddings and DeBERTa-v3 token states to accelerate model training iterations.
* **Architectural Blueprint & Prototype Components**:
  * Formulated the complete mathematical framework for Bidirectional Cross-Modal Attention and Class-Balanced loss functions.
  * Implemented preliminary standalone PyTorch modules for the cross-attention matrix computation and turn-memory buffer.
  * Tested local SLM inference (Ollama running `llama3.2:3b`) with prototype structured XAI prompt templates, validating latency under 300 ms on consumer GPUs.

---

## 5. Project Timeline (16-Week Roadmap)

| Phase | Milestone / Key Activities | Target Duration | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1: Diagnosis & Data Pipeline** | Empirical error analysis of PRJ3, 7-class MELD dataset alignment, HuBERT & DeBERTa feature extraction/caching pipeline. | Weeks 1–3 | **Completed / In Progress** |
| **Phase 2: Cross-Modal Model Development** | PyTorch implementation of Bidirectional Cross-Modal Attention (CMA) layer and Dialogue Turn Memory Buffer. | Weeks 4–6 | **Upcoming** |
| **Phase 3: Loss Engineering & Optimization** | Integration of Class-Balanced Focal Loss and logit calibration; training, hyperparameter sweeps, and ablation studies. | Weeks 7–9 | **Upcoming** |
| **Phase 4: Explainability & Edge Generative Pipeline** | Fact-extraction engine (XAI salience), Ollama local SLM integration, and low-latency TTS synthesis (Piper/Edge-TTS). | Weeks 10–12 | **Upcoming** |
| **Phase 5: Full-Stack Interface & Benchmarking** | Development of interactive Streamlit dashboard; comprehensive benchmarking against IEMOCAP/MELD state-of-the-art; final documentation. | Weeks 13–16 | **Upcoming** |

---

## 6. Team Responsibility Matrix

To guarantee individual accountability and seamless parallel execution, tasks are divided across core technical specializations:

| Team Member | Core Area of Responsibility | Specific Deliverables & Ownership |
| :--- | :--- | :--- |
| **Suhani Aggarwal** *(230547)* | **Foundation Feature Extraction & Cross-Modal Fusion** | • Audio feature caching pipeline using pretrained HuBERT / Wav2Vec 2.0.<br>• Textual token processing via DeBERTa-v3-base.<br>• Architectural design, PyTorch coding, and training of the **Bidirectional Cross-Modal Attention (CMA)** network. |
| **Aditya Parecha** *(230522)* | **Contextual Modeling & Loss Optimization** | • Implementation of the **Dialogue Memory Buffer & Contextual Reaffirmation Layer** (temporal GRU/attention over conversation turns).<br>• Formulation and tuning of **Class-Balanced Focal Loss** and post-hoc logit adjustments to eliminate mode collapse.<br>• Benchmark statistical validation and comparative metrics analysis. |
| **Harsh Pratap Singh** *(230518)* | **Explainable AI (XAI) & Assistive Agent Deployment** | • Development of the **XAI Fact Extraction Engine** (attention rollout & prosody anomaly parser).<br>• Local **Ollama SLM (Llama 3.2 3B)** prompt orchestration for natural conversational explanations.<br>• Edge **TTS synthesis** integration and full-stack interactive dashboard development. |

---
