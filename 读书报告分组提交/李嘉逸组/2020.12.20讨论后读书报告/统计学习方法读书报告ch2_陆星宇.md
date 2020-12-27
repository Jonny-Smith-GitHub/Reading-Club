# 统计学习方法

## 第二章 感知机

### 一、问题讨论

#### 1. 自己的问题及理解

#### 2. 他人的问题及理解

1. 对偶形式和原始形式比有什么好处？

   对偶形式每次迭代运算量较小，避免重复劳动。

2. 如果无法判断训练集是否线性可分，能否使用感知机算法分类？

   一方面对于不严格线性可分的数据，可以引入松弛变量。另一方面，可以通过核方法，将数据转化为线性可分数据。

3. 感知机和之前我们学的支持向量机（SVM）感觉很像，它们是同一种东西吗？如果不是，那它们的区别和联系是什么？

   感知机往往只以分类正确为目的，不考虑正反例范围之间需要有充分的区别，且每个数据都被平等对待；支持向量机追求边界区域的最大化，仅看重边界数据。

4. 极小化损失函数不是一次使M种所有误分类点的梯度下降，而是一次随机选取一个误分类点使其梯度下降，那么是否会出现梯度下降之后，新的w和b对于其他误分类的损失更大了？

   可以证明感知机算法的收敛性(Novikoff定理)，学习率较大时，会出现“过修正”现象，与学习率过小一样，会减慢收敛速度，但最终都会收敛。因此，选择一个适当的学习率很重要。

### 二、内容摘要

#### 1. 感知机定义：

$$
f(x)=sign(w·x+b)
$$

#### 2. 损失函数:

$$
L(w,b)=-\sum_{x_i\in M}y_i(w·x_i+b)
$$

​	M为误分类点的集合

#### 3. 随机梯度下降法：

$$
w⬅w+ηy_ix_i
$$

$$
b⬅b+ηy_i
$$

#### 4. 感知机学习算法的原始形式

算法2.1

Step1. 选取初始$w_0$,$b_0$;

Step2. 在训练集中选取数据($x_i$,$y_i$);

Step3. 如果$y_i$($w·x_i+b$)$≤$0，则执行梯度下降法*;

Step4. 转至Step2,直到训练集中没有误分类点。

可以证明感知机算法原始形式时收敛的，可以在有限次迭代中寻找到可以将线性可分训练数据完全分开的分离超平面。

#### 5. 感知机学习算法的对偶形式

$$
w=\sum_{i=1}^N\alpha_iy_ix_i
$$

$$
b=\sum_{i=1}^N\alpha_iy_i
$$

算法2.2

Step1. $\alpha⬅0$，$b⬅0$;

Step2. 在训练集中选取数据($x_i$,$y_i$);

Step3. 如果$y_i(\sum_{j=1}^N\alpha_jy_jx_j·x_i+b)≤0$,
$$
\alpha_i⬅\alpha_i+η
$$

$$
b⬅b+ηy_i
$$

Step4. 转到Step2直到没有误分类数据

为了方便可事先将实例间内积计算出来并以矩阵形式存储：
$$
G=[x_i·x_j]_{N\times N}
$$

### 三、程序实现

~~~~python
#Date:2020/12/22	Authour:陆星宇
#先随机生成一些线性可分的二维训练数据
import random as rd
import matplotlib.pyplot as plt
w=[rd.uniform(-1,1),rd.uniform(-1,1)]
b=rd.uniform(-1,1)
fun0=lambda x1,x2:w[0]*x1+w[1]*x2+b
T=[]
for i in range(100):
    t=[rd.uniform(-1,1),rd.uniform(-1,1)]
    if fun0(t[0],t[1])>0:
        t.append(1)
        T.append(t)
    elif fun0(t[0],t[1])<0:
        t.append(-1)
        T.append(t)
T1x=[]
T1y=[]
T2x=[]
T2y=[]
for t in T:
    if t[2]==1:
        T1x.append(t[0])
        T1y.append(t[1])
    else:
        T2x.append(t[0])
        T2y.append(t[1])
plt.plot(T1x,T1y,'r.',T2x,T2y,'b.')

#感知机算法
import numpy as np
n=len(T)
a=[0 for i in range(n)]
b=0
s=0.5#学习率
G=[]
gen=0
maxgen=2000
for i in range(n):
    G.append([])
    for j in range(n):
        G[i].append(T[i][0]*T[j][0]+T[i][1]*T[j][1])
gen=0
while gen<maxgen:
    flag=True
    gen+=1
    for i in range(n):
        temp=0
        for j in range(n):
            temp+=a[j]*T[j][2]*G[j][i]
        temp+=b
        temp*=T[i][2]
        if temp<=0:
            flag=False
            a[i]+=s
            b+=s*T[i][2]
    w=[0,0]
    for k in range(n):
        w[0]+=a[k]*T[k][2]*T[k][0]
        w[1]+=a[k]*T[k][2]*T[k][1]
    print("迭代次数：",gen,'   参数：',w,b)
    if flag:
        break
        
#绘制结果
k=-w[0]/w[1]
h=-b/w[1]
fun1=lambda x:k*x+h
x=[t for t in np.arange(-1.1,1.1,0.1)]
y=[fun1(t) for t in x]
plt.plot(T1x,T1y,'r.',T2x,T2y,'b.',x,y,'g-')
~~~~

### 四、运行结果

随机生成一些线性可分的二维二类样本数据：

![image-20201223011024490](C:\Users\27169\AppData\Roaming\Typora\typora-user-images\image-20201223011024490.png)

经过迭代：

迭代次数： 1    参数： [0.8886611693311368, -1.1831809663920225] 0.5 

迭代次数： 2    参数： [0.9367149519841184, -1.6970308361839428] 0.5 

迭代次数： 3    参数： [1.5274651141056617, -1.3803357080981677] 0.5 

迭代次数： 4    参数： [1.2690877108959877, -1.7018844379593858] 0.5 

迭代次数： 5    参数： [1.2690877108959877, -1.7018844379593858] 0.5

最终结果：

![image-20201223011155712](C:\Users\27169\AppData\Roaming\Typora\typora-user-images\image-20201223011155712.png)

结果良好。

### 五、下周计划

《统计学习方法》第三章