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

$$\langle \Phi(\boldsymbol{x}), \boldsymbol{u} \rangle = \sum_{i=1}^n \alpha_i \langle \Phi(\boldsymbol{x}), \Phi(\boldsymbol{x}_i) \rangle = \sum_{i=1}^n \alpha_i k(\boldsymbol{x}, \boldsymbol{x}_i).$$

对于训练集，所有样本在某个主成分上的投影值构成向量

$$\boldsymbol{y} = K \boldsymbol{\alpha}.$$

若选取前 $m$ 个主成分，对应的特征向量矩阵为 $U_m = (\boldsymbol{\alpha}_1, \dots, \boldsymbol{\alpha}_m) \in \mathbb{R}^{n \times m}$，特征值对角矩阵为 $\Lambda_m = \operatorname{diag}(\lambda_1, \dots, \lambda_m)$（其中 $\lambda_j$ 是 $K$ 的特征值），则训练样本在 $m$ 个主成分上的投影矩阵为

$$Y = K U_m \Lambda_m^{-1/2} \in \mathbb{R}^{n \times m}.$$

### 2.3 中心化

在实际应用中，我们需要在特征空间中对数据进行中心化，即用

$$\tilde{\Phi}(\boldsymbol{x}) = \Phi(\boldsymbol{x}) - \frac{1}{n} \sum_{i=1}^n \Phi(\boldsymbol{x}_i)$$

代替 $\Phi(\boldsymbol{x})$。相应地，核矩阵也需要中心化。记 $\boldsymbol{1}_n$ 为全1列向量，$I_n$ 为 $n \times n$ 单位矩阵，定义矩阵

$$H = I_n - \frac{1}{n} \boldsymbol{1}_n \boldsymbol{1}_n^{\mathsf T},$$

则中心化后的核矩阵为

$$\tilde{K} = H K H.$$

具体推导如下：设 $\Phi = (\Phi(\boldsymbol{x}_1), \dots, \Phi(\boldsymbol{x}_n))^{\mathsf T}$（每行是一个样本的像），中心化后的数据矩阵为

$$\tilde{\Phi} = \Phi - \frac{1}{n} \boldsymbol{1}_n \boldsymbol{1}_n^{\mathsf T} \Phi = H \Phi.$$

则中心化核矩阵

$$\tilde{K} = \tilde{\Phi} \tilde{\Phi}^{\mathsf T} = (H \Phi)(H \Phi)^{\mathsf T} = H (\Phi \Phi^{\mathsf T}) H = H K H.$$

因此，实际计算时，我们首先构造原始核矩阵 $K$，然后用 $H K H$ 代替 $K$，再对其进行特征分解，后续步骤不变。中心化后的核矩阵保证了特征空间中的数据均值为零，从而正确进行 PCA。
