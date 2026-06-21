---
title: PCA 与核 PCA 分析
published: 2026-03-17
description: 从线性 PCA 到核方法，推导协方差矩阵特征分解与核技巧的完整过程。
tags: [PCA, 核方法, 机器学习]
category: 矩阵优化与计算
draft: false
image: /images/cover.jpg
---

## 1. PCA 回顾

在主成分分析（PCA）中，我们寻找投影后方差最大的方向。该方向恰好对应样本协方差矩阵的最大特征值对应的特征向量。

首先介绍协方差矩阵。对于两个向量 $x, y$，协方差定义为

$$\operatorname{Var}(x, y) = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}),$$

用于衡量两者的相关性。

考虑中心化后的数据矩阵 $X \in \mathbb{R}^{m \times d}$（$m$ 个样本，$d$ 个特征）。则协方差矩阵为

$$C = \frac{1}{n} X^{\mathsf T} X,$$

其中 $C_{ij} = \operatorname{Var}(X_i, X_j)$。它将 $d$ 个特征在 $m$ 个样本上的相关信息浓缩在一起。

设单位向量 $u \in \mathbb{R}^{d}$（$\|u\|=1$），样本在 $u$ 方向上的投影为 $v = X u$。投影后的样本方差为 $\operatorname{Var}(X u) = u^{\mathsf T} C u$。我们要最大化该方差：

$$\max_{\|u\|=1} u^{\mathsf T} C u.$$

构造拉格朗日函数：

$$L(u, \lambda) = u^{\mathsf T} C u - \lambda (u^{\mathsf T} u - 1).$$

对 $u$ 求导并令其为零：

$$\frac{\partial L}{\partial u} = 2 C u - 2 \lambda u = 0 \quad \Longrightarrow \quad C u = \lambda u.$$

代入目标函数得 $u^{\mathsf T} C u = u^{\mathsf T} (\lambda u) = \lambda$。因此，最大方差等于最大特征值，且最优投影方向为对应的特征向量。

因此，只需求出协方差矩阵 $\frac{1}{n} X^{\mathsf T} X$ 的特征值和特征向量，取最大特征值对应的特征向量作为第一主成分方向，次大特征值对应第二主成分方向，以此类推。

## 2. 核 PCA

PCA 是线性方法，在处理非线性问题时表现不佳。因此引入映射

$$\Phi: \mathbb{R}^d \to \mathcal{F},\quad \boldsymbol{x} \mapsto \Phi(\boldsymbol{x}),$$

将原始数据映射到高维（甚至无穷维）特征空间 $\mathcal{F}$。

设原始数据矩阵 $X = (\boldsymbol{x}_1, \boldsymbol{x}_2, \dots, \boldsymbol{x}_n)^{\mathsf T} \in \mathbb{R}^{n \times d}$，映射后的数据矩阵为 $\Phi(X) = (\Phi(\boldsymbol{x}_1), \dots, \Phi(\boldsymbol{x}_n))^{\mathsf T}$（每行是一个样本的像）。在特征空间中，协方差矩阵为

$$C_{\Phi} = \frac{1}{n} \Phi(X)^{\mathsf T} \Phi(X) = \frac{1}{n} \sum_{i=1}^n \Phi(\boldsymbol{x}_i) \Phi(\boldsymbol{x}_i)^{\mathsf T}.$$

我们希望求 $C_{\Phi}$ 的特征向量 $\boldsymbol{u}$，满足

$$C_{\Phi} \boldsymbol{u} = \lambda \boldsymbol{u}.$$

由于 $\Phi$ 可能维数很高甚至无穷，直接计算不可行。由上式可知，所有特征向量 $\boldsymbol{u}$ 均可表示为 $\Phi(\boldsymbol{x}_i)$ 的线性组合（因为 $C_{\Phi}$ 的每一列都是 $\Phi(\boldsymbol{x}_i)$ 的线性组合）。设

$$\boldsymbol{u} = \sum_{i=1}^n \alpha_i \Phi(\boldsymbol{x}_i) = \Phi(X)^{\mathsf T} \boldsymbol{\alpha},$$

其中 $\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_n)^{\mathsf T}$。将其代入特征方程：

$$\frac{1}{n} \sum_{i=1}^n \Phi(\boldsymbol{x}_i) \Phi(\boldsymbol{x}_i)^{\mathsf T} \left( \sum_{j=1}^n \alpha_j \Phi(\boldsymbol{x}_j) \right) = \lambda \sum_{k=1}^n \alpha_k \Phi(\boldsymbol{x}_k).$$

两边左乘 $\Phi(\boldsymbol{x}_l)^{\mathsf T}$，得

$$\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^n \alpha_j \left( \Phi(\boldsymbol{x}_l)^{\mathsf T} \Phi(\boldsymbol{x}_i) \right) \left( \Phi(\boldsymbol{x}_i)^{\mathsf T} \Phi(\boldsymbol{x}_j) \right) = \lambda \sum_{k=1}^n \alpha_k \Phi(\boldsymbol{x}_l)^{\mathsf T} \Phi(\boldsymbol{x}_k).$$

定义核矩阵 $K \in \mathbb{R}^{n \times n}$，其元素 $K_{ij} = \Phi(\boldsymbol{x}_i)^{\mathsf T} \Phi(\boldsymbol{x}_j) = k(\boldsymbol{x}_i, \boldsymbol{x}_j)$。上式可写为

$$\frac{1}{n} K K \boldsymbol{\alpha} = \lambda K \boldsymbol{\alpha}.$$

通常假设 $K$ 可逆（或正定），两边左乘 $K^{-1}$ 得

$$\frac{1}{n} K \boldsymbol{\alpha} = \lambda \boldsymbol{\alpha} \quad \Longrightarrow \quad K \boldsymbol{\alpha} = n \lambda \boldsymbol{\alpha}.$$

因此，$\boldsymbol{\alpha}$ 是核矩阵 $K$ 的特征向量，对应的特征值为 $n\lambda$。求出 $\boldsymbol{\alpha}$ 后，即可得到特征空间中的主方向 $\boldsymbol{u} = \sum_i \alpha_i \Phi(\boldsymbol{x}_i)$，但 $\boldsymbol{u}$ 本身无法显式写出（除非映射已知），我们只需知道样本在 $\boldsymbol{u}$ 上的投影即可。

### 2.1 核函数

核技巧的核心是直接定义核函数 $k(\boldsymbol{x}, \boldsymbol{y}) = \Phi(\boldsymbol{x})^{\mathsf T} \Phi(\boldsymbol{y})$，而不必显式构造映射 $\Phi$。常用核函数有：

- **多项式核：** $k(\boldsymbol{x}, \boldsymbol{y}) = (\boldsymbol{x}^{\mathsf T} \boldsymbol{y} + c)^d$，$d \in \mathbb{N}$，$c \ge 0$。
- **高斯核（RBF）：** $k(\boldsymbol{x}, \boldsymbol{y}) = \exp\left( -\frac{\|\boldsymbol{x} - \boldsymbol{y}\|^2}{2\sigma^2} \right)$，$\sigma > 0$。
- **Sigmoid 核：** $k(\boldsymbol{x}, \boldsymbol{y}) = \tanh(a \boldsymbol{x}^{\mathsf T} \boldsymbol{y} + r)$。

### 2.2 投影的计算

我们希望投影方向 $\boldsymbol{u}$ 满足归一化条件 $\|\boldsymbol{u}\| = 1$。由 $\boldsymbol{u} = \sum_i \alpha_i \Phi(\boldsymbol{x}_i)$ 得

$$\|\boldsymbol{u}\|^2 = \boldsymbol{u}^{\mathsf T} \boldsymbol{u} = \sum_{i,j} \alpha_i \alpha_j \Phi(\boldsymbol{x}_i)^{\mathsf T} \Phi(\boldsymbol{x}_j) = \boldsymbol{\alpha}^{\mathsf T} K \boldsymbol{\alpha}.$$

若 $\boldsymbol{\alpha}$ 是 $K$ 的特征向量且特征值为 $\lambda_K$（即 $K \boldsymbol{\alpha} = \lambda_K \boldsymbol{\alpha}$），则

$$\|\boldsymbol{u}\|^2 = \boldsymbol{\alpha}^{\mathsf T} (\lambda_K \boldsymbol{\alpha}) = \lambda_K \|\boldsymbol{\alpha}\|^2.$$

为使 $\|\boldsymbol{u}\| = 1$，可取 $\|\boldsymbol{\alpha}\|^2 = 1 / \lambda_K$，即令 $\boldsymbol{\alpha} = \frac{1}{\sqrt{\lambda_K}} \boldsymbol{\alpha}_{\text{raw}}$，其中 $\boldsymbol{\alpha}_{\text{raw}}$ 是单位特征向量（$\|\boldsymbol{\alpha}_{\text{raw}}\|=1$）。

样本 $\boldsymbol{x}$ 在 $\boldsymbol{u}$ 上的投影为

$$\text{proj} = \boldsymbol{u}^{\mathsf T} \Phi(\boldsymbol{x}) = \sum_{i=1}^n \alpha_i \Phi(\boldsymbol{x}_i)^{\mathsf T} \Phi(\boldsymbol{x}) = \sum_{i=1}^n \alpha_i k(\boldsymbol{x}_i, \boldsymbol{x}).$$

这就是核 PCA 的投影公式。我们只需要计算新样本与所有训练样本的核函数值，而不需要知道映射 $\Phi$ 的具体形式。

## 3. 总结

核 PCA 通过核技巧将 PCA 扩展到非线性情况，其核心思想是：

1. **隐式映射**：通过核函数 $k(\boldsymbol{x}, \boldsymbol{y})$ 隐式地将数据映射到高维特征空间
2. **核矩阵**：计算核矩阵 $K$ 并求解其特征向量
3. **投影计算**：利用核函数值计算新样本的投影

核 PCA 的优势在于：
- 无需显式构造映射 $\Phi$
- 可以处理非线性可分数据
- 议算复杂度与样本数 $n$ 相关，与特征维度无关

常用核函数的选择：
- **高斯核**：适用于大多数情况，参数 $\sigma$ 控制核的宽度
- **多项式核**：适用于多项式可分的数据
- **Sigmoid核**：类似于神经网络中的激活函数
