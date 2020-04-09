# 3. Logistic Regression and Other Regression

- **Regression: 그래프 상에 무작위하게 흩어져 있는 데이터의 경향을 선으로 나타내는 것**
  - 선이 직선일 경우 Linear Regression

## Linear Regression

### Linear Model for Regression

- Ex) 사람의 age를 통해 SBP를 predict하는 Regression

  - Age는 independent variable = input feature
  - SBP는 dependent variable = ground truth

  ![image-20200405170025728](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200405170025728.png)

  - Loss $\mathcal{L}=(X\theta-Y)^\top(X\theta-Y)$, where $X\in R^{n*d}, Y\in R^n, \theta\in R^{d+1}$
    - $X$: Feature matrix(n*d)
      - n: sample의 수(Ex. 33명의 사람)
      - d: feature의 수(Ex. 1개(age))     (d+1)..
    - $Y$: Ground truth(Ex. SBP)
      - 유추하고자 하는 값에 대해 표본들이 취하고 있는 값들
    - $\theta$: Parameters to learn
      - Ex) $\theta_1$: 81.54, $\theta_2$: 1.222
    - $X\theta-Y$: Error of prediction
      - $X\theta$: Predicted value
      - $(X\theta-Y)^\top$는 $\begin{pmatrix}e_1&e_2&..\end{pmatrix}$ 형태이고, $(X\theta-Y)$는 $\begin{pmatrix}e_1\\e_2\\..\end{pmatrix}$ 형태이므로 둘을 곱하면 $e_1^2+e_2^2+...$ 형태의 하나의 scalar가 나온다. (Sum Squared Error)
        - $(..)^\top(..)$가 일반적으로 Mean Squared Error의 값을 구한다는 것을 알아둘 것
    - **설명과 다르게 저렇게 계산하면 차원이 안맞는다.**
      1. $X$의 가장 왼쪽 열에 $\begin{pmatrix}1\\1\\..\end{pmatrix}$을 추가하여 $\theta$에서의 bias parameter를 더할 수 있게 한다.
         - 이 경우 $X$의 크기가 (n+1)*d가 됨
      2. $X\theta$ 대신 $X\theta+\beta$ 를 사용한다.
         - 이 경우 $\theta$의 크기는 d\*1, $\beta$의 크기는 n\*1이 됨 
  - 이러한 Loss를 가장 작게 만드는 $\theta(\underset{\theta}{\arg\min} \mathcal{L})$를 찾는 것이 가장 적합한 Linear Regression Model을 찾는 것이다.
    - $\mathcal{L}=(X\theta-Y)^\top(X\theta-Y)$
      $=((X\theta)^\top-Y^\top)(X\theta-Y)=(X\theta)^\top X\theta-(X\theta)^\top Y-Y^\top(X\theta)+Y^\top Y$
      $=(X\theta)^\top X\theta-2Y^\top (X\theta)+Y^\top Y$ ($(X\theta)^\top Y$와 $Y^\top(X\theta)$는 scalar이며, 같은 값)
      $=\theta^\top X^\top X\theta-2Y^\top (X\theta)+Y^\top Y$ (parabola 형태이므로, 미분하여 최소값을 찾는다.)
    - vector에 대한 미분(여기서 X는 vector이고 A는 matrix)
      - ![image-20200405185019699](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200405185019699.png)
    - $\frac{\delta\mathcal{L}}{\delta\theta}=X^\top X\theta+X^\top X\theta-2X^\top Y=2X^\top X\theta-2X^\top Y=0$
    - $\therefore \theta^*=(X^\top X)^{-1}X^\top Y$
      - 그러나, 만약 **n<d**일 경우 $(X^\top X)^{-1}$와 같은 식이 나올 수 없다.
        - 샘플(n)의 수가 아주 적을 경우, 이 적은 수의 샘플에서 큰 양의 parameter(d)를 learn해야 하는데, (미지수가 d개인 연립방정식에 n개의 해만을 대입하므로 결정이 되지 않는 느낌???)<span style="color:red">???</span>
        - 이 경우 inverse calculation이 불가능!<span style="color:red">왜???</span>
        - 즉, 이럴 때는 gradient descent와 같은 다른 종류의 optimization technique를 사용해야 함
          - 그러나, 다른 방법을 사용했을 때는 위와 같은 optimal solution에 수렴하지 않거나 여러 가지의 optimal solution이 나옴
          - Unique optimal solution을 얻기 위해서는 linear regression을 사용해야 한다.
            - 그러므로 샘플의 수가 많아야 함
        - Linear Regression에서는 일반적으로 큰 수의 샘플(n)이 있어야 한다.

### Linear Model for Classification

- Ex) 사람의 Age를 통해 CD라는 값(0 또는 1을 가지는 categorical value)을 predict

  ![image-20200405220326187](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200405220326187.png)

- **Categorical value를 predict할 때는 Linear Regression이 적합하지 않다**

  - $P(Y=1|X=\vec{x})=\beta_0+\beta_1x_1+\beta_2x_2+...++\beta_px_p=\vec{\beta}^\top\vec x$
    - 위 식은 $X\theta$에 대응하고, $\theta=\begin{pmatrix}\beta_1\\ \beta_2\\ ..\end{pmatrix}$
    - 위 확률에 대해 learn하게 된다고 해도 문제는 같다..
    - feature vector x가 있을 때 class가 1이 되는 확률

- 이러한 Classification 문제에는 Logistic Model을 사용한다.

## Logistic Regression

### Odds

- odds = $\frac{P(A)}{P(A^C)}=\frac{P(A)}{1-P(A)}$
  - 모든 경우는 A이거나, A가 아니거나로 나뉜다(binary classification)
  - P(A)=1일 때 odds=$+\infin$
  - P(A)=1/2일 때 odds=1
  - P(A)=0일 때 odds=0
- ![image-20200405222500229](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200405222500229.png)

### Logistic Regression for Classification

- Logistic Regression을 크게 나누면 "Linear Regression → Logit → Classification"의 단계를 거친다.
- ![image-20200406023408609](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406023408609.png)

#### Linear Regression → Logit

- Classification에는 Logistic model이 사용됨
  - <u>odd의 log값에 Linear regression을 적용</u>
- logit(log odds) = $\log\left(\frac{P(Y=1|X=\vec{x})}{1-P(Y=1|X=\vec{x})}\right)=\vec{\beta}^\top\vec x $ 꼴로 linear regression을 적용할 수 있음
  - $\vec{\beta}^\top\vec x $ 가 $X\theta$의 역할을 한다.
- 일반적으로, 이러한 logit을 predict하는 것이 더 적절하다.**(Logistic Regression의 장점)**
  - P(Y=1 or 0|X), 즉 probability는 일반적으로 non-linear하며, S모양 곡선의 값을 가짐
  - Linear regression으로 logit을 predict하면, non-linear probability predictor를 build할 수 있다.<span style="color:red">???</span>
  - Logit값이 P(Y=1 or 0|X), 즉 probability보다 더 linear에 가깝다.
  - **Logit은 범위가 $[-\infin,\infin]$에 들어가므로(Unconstrained Optimization) linear regression이 predict하기에 더 적합함**
    - Probability의 경우 [0,1]의 범위를 가지므로 식에 대해 $\theta$에 대한 argmin을 구할 때 $0\le X\theta\le1$이라는 조건 안에서 구해야 한다. (Constrained Optimization Problem to predict P(Y=0 or 1|X))
- 즉, Linear Regression → Logit은 classical regression problem

#### Logit → Classification

- $\log\left(\frac{p(x)}{1-p(x)}\right)=a(=X\theta)$라고 할 때,

  - $\frac{p(x)}{1-p(x)}=e^a$
    $p(x)=e^a(1-p(x))$
    $p(x)=e^a-e^ap(x)$
    $p(x)+e^ap(x)=e^a$
    $p(x)(1+e^a)=e^a$
    $\therefore p(x)=\frac{e^a}{1+e^a}=\frac{1}{1+e^{-a}}$

  - Sigmoid function

    ![image-20200406022339537](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406022339537.png)

    - x(a, $X\theta$, logit)의 범위가 $[-\infin,\infin]$이고, p(x) (probability)의 범위가 [0, 1]이 된다.
    - $P(Y=1|X=\vec{x})=\frac{1}{1+e^{-\vec{\beta}^\top\vec x}}$
      - $\vec{x}$가 주어졌을 때 Y의 class가 1이 될 확률
    - Sigmoid는 S모양의 함수라는 뜻이고, 위의 함수 외에도 여러 종류의 식이 존재한다.
      - <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406024521773.png" alt="image-20200406024521773" style="zoom: 67%;" />

- 주어진 feature vector를 통해 class를 predict하는 방법

  - $P(Y=1|X=\vec{x})>P(Y=0|X=\vec{x})$라면, $\vec{x}$는 class 1이라고 볼 수 있다.
    - 즉, $p(x)>1-p(x)$
      $\frac{p(x)}{1-p(x)}>1$
      $\log\frac{p(x)}{1-p(x)}>0$
      $\therefore \vec{\beta}^\top\cdot\vec x>0$
      - Logit > 0인 경우, $\vec{x}$ belongs to class 1
      - Logit < 0인 경우, $\vec{x}​ belongs to class 0
      - Logit = 0인 경우, cannot decide

### Visualization of Logistic Regression

- Dot product

  - $\vec{a}\cdot\vec{b}=0 \iff$ 두 vector가 직각(perpendicular)
  - $\vec{a}\cdot\vec{b}>0 \iff$ 두 vector가 예각
  - $\vec{a}\cdot\vec{b}<0 \iff$ 두 vector가 둔각

  ![image-20200406031754595](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406031754595.png)

  - $P(Y=1|X=\vec{x})=\beta_0+\beta_1x_1+\beta_2x_2+...++\beta_px_p=\vec{\beta}^\top\cdot\vec x=\vec{\beta}^{'}\cdot\vec x+\beta_0$
    - <span style="color:red">확률인데 왜 갑자기 logit이 되어 범위가 -무한~무한이 되었는지</span>
    - <span style="color:red">베타프라임과 x의 내적의 부호로 classification을 정하는데 왜 내적 식에 + bias가 들어가는지. 이미 내적에서 방향이 나왔는데 거기에 bias를 더하면 의미가 바뀌는거 아닌지.</span>

  - class X(1)와 class O(0)가 있으며, 이 두 class를 구분하는 hyperplane이 있다.

  - Hyperplane의 절편은 $\beta_0$. 이는 위의 행렬 전개식에서 상수 부분

  - 이 Hyperplane에 수직한 vector $\vec{\beta}^{'}=[\beta_1, \beta_2, ...,\beta_p]$

  - 편의상 $\vec{x_1}\sim \vec{x_9}(\neq x_1\sim x_9)$까지 있다고 가정하며, 1~5가 class X이고 6~9는 class O

  - $\vec{x_1}\sim \vec{x_5}$의 경우  $\vec{\beta}^{'}$와 예각을 이뤄 내적값이 양수가 되고, class X이 되고, 반대도 성립.

  - **$\beta_0$의 중요성(매우 중요!)**

    ![image-20200406040303391](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406040303391.png)

    - 만약 $\beta_0$이 위와 이 설정된 경우 어떤 $\vec{\beta}^{'}$도 predict를 할 수 없게 된다.
      - 모든 $\vec{x}$와 예각을 이뤄 모두 class X로 분류하기 때문
    - $\beta_0$을 "bias"라 부른다.
      - 이 값을 조정하며 hyperplane을 이동할 수 있다.

<img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406040944251.png" alt="image-20200406040944251" style="zoom: 67%;" />

- <span style="color:red">Does the distance from the hyperplane has a meaning?</span>

### Analogy between Logistic Regression and Fully Connected Layer

<img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406043102491.png" alt="image-20200406043102491" style="zoom:80%;" />

<img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200324223959491-tile_2.png" alt="image-20200324223959491-tile_2" style="zoom:33%;" />

- $h_i=\sigma(W\cdot h_{i-1}+b)$
  - $h_i$: hidden vector at layer $i$
  - $W\cdot h_{i-1}+b \iff \vec{\beta}^\top\cdot\vec x$
  - $\sigma()$ activation function: sigmoid
  - 위와 같은 관점으로 봤을 때, 하나의 Fully Connected Layer는 Logistic Regression과 같다.

### Logistic Regression for Multi-Class Classification

- Ex) 3개의 class가 있는 경우
  - 2개의 logistic regression model을 사용하는 경우
    - <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406044921292.png" alt="image-20200406044921292" style="zoom: 67%;" />
    - K개의 class에 대해 확장 가능하며, 이를 일반화한 결과는 다음과 같다.
      - <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406045304416.png" alt="image-20200406045304416" style="zoom:67%;" />
        - <span style="color:red">왜 K번째 클래스의 분모만 1?</span>
  - 3개의 logistic regression model을 모두 사용하는 경우: Softmax!

### Bias and Variance

- 개념에 대한 예시

  - 다양한 전공의 학생에게 ML 수업을 가르치려 할 때
  - 수업의 내용이 쉽다면 컴과 전공생들이 complain을 할 것이고,
  - 수업의 내용이 어렵다면 비전공생들이 complain을 할 것이다.
  - 이 때 수업의 만족도를 최대로 올리려면 수업을 "중간 난이도"로 가르쳐야 한다.

- 개념

  - $y=f(x)+e$라는 True model 존재
    - 그러나 $f$(feature vector)와 $e$(error term with zero mean)에 대한 exact form은 알지 못하고, 관측만 할 수 있음
    - 실제 세계의 현상들도 이렇게 설명할 수 있으며, $e$는 noise를 표현한다.

  ![image-20200409145929058](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200409145929058.png)

  - 나만의 Regression algorithm을 만들어 f를 예측하려 한다.
  - m명의 소비자가 나의 algorithm을 사용하려 함
    - 각자의 관측으로 인한 dataset $D_1\sim D_m$이 $m$개의 training data가 된다.
  - Regression algorithm은 각각의 dataset에 대해 approximated model인 $\hat{f_1}\sim\hat{f_m}$을 생성한다.
    - $\hat{f}(x;D_1)\sim\hat{f}(x;D_m)$
  - 이 때, Error term $E_D\left[(y-\hat{f}(x;D))^2\right]$은 <u>새로운(unseen)</u> test case $x$가 주어졌을 때 algorithm의 expected error이다.

## Other Regressions

### Quadratic Regression

### Polynomial Regression

##### q vs h

### Ridge Regression

### Lasso Regression

