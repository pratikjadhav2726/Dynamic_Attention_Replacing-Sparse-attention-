# Clustered Dynamic Attention for Efficient Multi-Head Attention

## 🚀 Introduction
This repository implements **Clustered Dynamic Attention**, a novel technique that optimizes traditional **multi-head self-attention (MHA)** by **reducing redundant tokens** dynamically while preserving critical information. This approach significantly improves computational efficiency by **merging similar tokens** and broadcasting attention results back to the original token positions.

## 🔥 Intuition

### **Why Reduce Tokens?**
Standard self-attention in transformers processes all tokens equally, even when some tokens contain **redundant or highly similar information**. This leads to:
- **High computational cost** 💸 (quadratic complexity in `O(n²d)`)
- **Unnecessary redundancy** in attention processing

### **Solution**
- **Cluster Similar Tokens**: Identify groups of tokens that are **highly similar** based on cosine similarity.
- **Reduce Token Set**: Keep **one representative token per cluster** while remembering mappings.
- **Compute Attention Only on the Reduced Set**: Apply `nn.MultiheadAttention` only to the reduced token set.
- **Broadcast Attention Back**: Restore the attention results to **all original positions** using the saved mappings.

## 🏗️ Approach

### **1. Token Clustering**
- Compute **cosine similarity** between all tokens.
- Merge **highly similar tokens** into clusters.
- Keep **one representative** token per cluster.
- **Store mapping** of original tokens to reduced tokens.

### **2. Apply Multihead Attention (MHA) on Reduced Tokens**
- Project hidden states into **queries (Q), keys (K), values (V)**.
- Perform **MHA only on the reduced token set**.

### **3. Broadcast Attention to Full Length**
- After attention, use the **stored mapping** to **propagate the results** back to all original tokens.

---

## 📌 Key Features

✅ **Reduces Computation** - Instead of computing attention on `seq_len` tokens, it only computes on `new_seq_len ≤ seq_len`.  
✅ **Preserves Information** - Ensures **merged tokens receive attention** from their cluster representatives.  
✅ **Improves Transformer Efficiency** - Reduces complexity from **O(n²d) → O(m²d)** (where `m ≤ n`).  
✅ **Works with Any Transformer Architecture** - Can be integrated into **LLMs, Vision Transformers, and NLP models**.  

---

## 🔬 Possible Enhancements

🚀 Adaptive Token Clustering: Use learnable thresholds instead of fixed cosine similarity.
🚀 Soft Attention Weight Redistribution: Instead of hard clustering, blend attention weights between similar tokens.
🚀 Efficient Kernel Implementation: Implement clustering in CUDA for even faster processing.
🚀 Pre-Trained Model Fine-Tuning: Apply this technique to existing Transformer models like GPT, BERT, or Llama.

 ## 👀 Visualization (Coming Soon)

## We can visualize:
	•	Token Clustering (Which tokens were merged)
	•	Attention Weight Redistribution (Before & After merging)
	•	Computation Speed Gains (Benchmark comparisons)

## 📢 Citation & Credits

This approach is inspired by recent dynamic token merging techniques like:
	•	Token Merging (ToMe)
	•	Dynamic Token Merging for Byte-Level Models
	•	Spectrum-Preserving Token Merging

## 🤝 Contributing

Feel free to open issues, submit PRs, or discuss improvements & new ideas in this repository.

🌟 If you like this work, please give a ⭐ on GitHub!

---


