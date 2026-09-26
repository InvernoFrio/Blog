---
title: 推荐算法笔记
published: 2026-04-02
description: 协同过滤与矩阵分解的完整推导，包括 SGD、ALS、SVD++ 等方法。
tags: [推荐系统, 矩阵方法]
category: 矩阵优化与计算
draft: false
image: /images/recommendation-cover.jpg
---

## 1. 问题定义与低秩假设

给定用户-物品评分矩阵

$$R \in \mathbb{R}^{m \times n},$$

其中 $m$ 为用户数、$n$ 为物品数。实际场景中，$R$ 仅在观测集合

$$\Omega = \{(u, i) \mid R_{ui}\ \text{已观测}\}$$

上有值。协同过滤常用低秩假设：

$$R \approx P Q^{\top},\quad P \in \mathbb{R}^{m \times k},\ Q \in \mathbb{R}^{n \times k},\ k \ll \min(m, n).$$

其中 $p_u$（$P$ 的第 $u$ 行）是用户潜在向量，$q_i$（$Q$ 的第 $i$ 行）是物品潜在向量。

## 2. 矩阵分解目标函数（SVD 思想 + 正则化）

基础预测模型：

$$\hat{r}_{ui} = p_u^{\top} q_i.$$

在观测集合上做带 $\ell_2$ 正则的最小二乘：

$$\min_{P, Q}\ \sum_{(u, i) \in \Omega} \bigl(r_{ui} - p_u^{\top} q_i\bigr)^2 + \lambda \left( \|P\|_F^2 + \|Q\|_F^2 \right).$$

正则化的作用是抑制参数过大、缓解过拟合，提高在噪声数据上的泛化稳定性。

## 3. SGD 更新推导

对单个样本 $(u, i)$，记误差

$$e_{ui} = r_{ui} - p_u^{\top} q_i,$$

单样本目标可写为

$$\ell_{ui} = e_{ui}^2 + \lambda \left( \|p_u\|_2^2 + \|q_i\|_2^2 \right).$$

### 梯度计算

$$\frac{\partial \ell_{ui}}{\partial p_u} = -2 e_{ui} q_i + 2 \lambda p_u,$$

$$\frac{\partial \ell_{ui}}{\partial q_i} = -2 e_{ui} p_u + 2 \lambda q_i.$$

### 梯度下降更新

吸收常数 $2$ 到学习率 $\eta$，得

$$p_u \leftarrow p_u + \eta \bigl(e_{ui} q_i - \lambda p_u\bigr),$$

$$q_i \leftarrow q_i + \eta \bigl(e_{ui} p_u - \lambda q_i\bigr).$$

循环遍历样本，迭代至验证集误差收敛。

## 4. Biased-SVD 与 SVD++

### Biased-SVD

考虑全局均值和用户/物品偏置：

$$\hat{r}_{ui} = \mu + b_u + b_i + p_u^{\top} q_i.$$

其中 $\mu$ 为全局平均分，$b_u, b_i$ 分别表示用户和物品偏置。

### SVD++

进一步利用隐式反馈集合 $N(u)$（如点击/浏览/购买）：

$$\hat{r}_{ui} = \mu + b_u + b_i + q_i^{\top} \left( p_u + \frac{1}{\sqrt{|N(u)|}} \sum_{j \in N(u)} y_j \right).$$

其中 $y_j$ 为物品 $j$ 的隐式反馈向量。

## 5. 交替最小二乘（ALS）

SGD 是联合更新，ALS 则采用"固定一侧、求解另一侧"的交替策略。目标函数仍为

$$\min_{P, Q}\ \sum_{(u, i) \in \Omega} \bigl(r_{ui} - p_u^{\top} q_i\bigr)^2 + \lambda \left( \|P\|_F^2 + \|Q\|_F^2 \right).$$

### 固定 $Q$，求 $P$（法方程推导）

对某个用户 $u$，设其已评分物品集合为 $\Omega_u$，令

$$Q_u \in \mathbb{R}^{|\Omega_u| \times k}$$

为对应物品因子堆叠矩阵，$r_u \in \mathbb{R}^{|\Omega_u|}$ 为该用户的观测评分向量。则子问题为

$$\min_{p_u}\ \|r_u - Q_u p_u\|_2^2 + \lambda \|p_u\|_2^2.$$

其一阶最优条件为

$$\left( Q_u^{\top} Q_u + \lambda I \right) p_u = Q_u^{\top} r_u.$$

### 固定 $P$，求 $Q$

完全对称地，对每个物品 $i$ 有

$$\left( P_i^{\top} P_i + \lambda I \right) q_i = P_i^{\top} r_i.$$

### ALS 流程

1. 初始化 $P, Q$（小随机数或常数）。
2. 固定 $Q$，逐个用户求解 $p_u$。
3. 固定 $P$，逐个物品求解 $q_i$。
4. 重复步骤 2–3，直到目标函数或验证误差收敛。

由于每个子问题都是凸二次问题并被精确求解，ALS 每一步都不增大目标函数，最终收敛到局部最优点。

## 6. 隐式反馈 ALS（Weighted-ALS）

对隐式反馈（如观看时长、点击次数）常用偏好-置信度建模：

$$p_{ui} = \mathbb{I}[r_{ui} > 0], \qquad c_{ui} = 1 + \alpha r_{ui}.$$

优化问题写为

$$\min_{X, Y} \sum_{u, i} c_{ui} \bigl(p_{ui} - x_u^{\top} y_i\bigr)^2 + \lambda \left( \|X\|_F^2 + \|Y\|_F^2 \right).$$

该模型同样可用 ALS 高效求解，是工业推荐系统中的经典方案。
