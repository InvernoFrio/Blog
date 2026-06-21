---
title: 推荐算法笔记
published: 2026-04-02
description: 协同过滤与矩阵分解的完整推导，包括 SGD、ALS、SVD++ 等方法。
tags: [推荐系统, 协同过滤, 矩阵分解]
category: 矩阵优化与计算
draft: false
image: /images/recommendation-cover.jpg
---

## 1. 协同过滤概述

协同过滤（Collaborative Filtering）是推荐系统中最经典的方法之一，其核心思想是利用用户的历史行为数据来预测用户对未交互物品的偏好。

### 1.1 基于用户的协同过滤

基于用户的协同过滤（User-based CF）通过找到与目标用户兴趣相似的用户群体，然后将这些用户喜欢的物品推荐给目标用户。

相似度计算方法：

- **余弦相似度**：
$$\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui} \cdot r_{vi}}{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2} \cdot \sqrt{\sum_{i \in I_{uv}} r_{vi}^2}}$$

- **皮尔逊相关系数**：
$$\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$

### 1.2 基于物品的协同过滤

基于物品的协同过滤（Item-based CF）通过计算物品之间的相似度，然后根据用户历史喜欢的物品推荐相似物品。

## 2. 矩阵分解

矩阵分解（Matrix Factorization）将用户-物品评分矩阵分解为两个低秩矩阵的乘积，是现代推荐系统的核心技术。

### 2.1 基本模型

设评分矩阵 $R \in \mathbb{R}^{m \times n}$，其中 $m$ 为用户数，$n$ 为物品数。矩阵分解的目标是找到两个低秩矩阵 $P \in \mathbb{R}^{m \times k}$ 和 $Q \in \mathbb{R}^{n \times k}$，使得

$$R \approx P Q^{\mathsf T}$$

其中 $k$ 为隐因子维度，通常 $k \ll m, n$。

预测评分公式：

$$\hat{r}_{ui} = p_u^{\mathsf T} q_i = \sum_{f=1}^k p_{uf} \cdot q_{if}$$

### 2.2 优化方法

#### SGD（随机梯度下降）

损失函数：

$$\min_{P, Q} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - p_u^{\mathsf T} q_i)^2 + \lambda (\|p_u\|^2 + \|q_i\|^2)$$

其中 $\mathcal{K}$ 为已知评分的集合，$\lambda$ 为正则化参数。

SGD更新规则：

$$p_u \leftarrow p_u + \alpha (e_{ui} \cdot q_i - \lambda \cdot p_u)$$
$$q_i \leftarrow q_i + \alpha (e_{ui} \cdot p_u - \lambda \cdot q_i)$$

其中 $e_{ui} = r_{ui} - \hat{r}_{ui}$ 为预测误差，$\alpha$ 为学习率。

#### ALS（交替最小二乘法）

ALS通过交替固定 $P$ 和 $Q$ 来优化损失函数：

1. 固定 $Q$，优化 $P$：
$$p_u = (Q^{\mathsf T} Q + \lambda I)^{-1} Q^{\mathsf T} r_u$$

2. 固定 $P$，优化 $Q$：
$$q_i = (P^{\mathsf T} P + \lambda I)^{-1} P^{\mathsf T} r_i$$

### 2.3 SVD++

SVD++在基本矩阵分解的基础上加入了隐式反馈信息：

$$\hat{r}_{ui} = \mu + b_u + b_i + q_i^{\mathsf T} \left( p_u + |N(u)|^{-1/2} \sum_{j \in N(u)} y_j \right)$$

其中：
- $\mu$：全局平均评分
- $b_u, b_i$：用户和物品的偏置项
- $N(u)$：用户 $u$ 交互过的物品集合
- $y_j$：物品 $j$ 的隐式反馈因子

## 3. 总结

推荐系统的核心技术演进：

1. **传统协同过滤**：基于用户/物品的相似度计算
2. **矩阵分解**：将评分矩阵分解为低秩矩阵
3. **深度学习**：使用神经网络学习用户和物品的表示

矩阵分解方法的优势：
- 可以处理稀疏的评分矩阵
- 可以学习到用户和物品的潜在特征
- 预测精度高，计算效率好
