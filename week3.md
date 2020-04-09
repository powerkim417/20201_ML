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
    - $\dfrac{\delta\mathcal{L}}{\delta\theta}=X^\top X\theta+X^\top X\theta-2X^\top Y=2X^\top X\theta-2X^\top Y=0$
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

- odds = $\dfrac{P(A)}{P(A^C)}=\dfrac{P(A)}{1-P(A)}$
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
- logit(log odds) = $\log\left(\dfrac{P(Y=1|X=\vec{x})}{1-P(Y=1|X=\vec{x})}\right)=\vec{\beta}^\top\vec x $ 꼴로 linear regression을 적용할 수 있음
  - $\vec{\beta}^\top\vec x $ 가 $X\theta$의 역할을 한다.
- 일반적으로, 이러한 logit을 predict하는 것이 더 적절하다.**(Logistic Regression의 장점)**
  - P(Y=1 or 0|X), 즉 probability는 일반적으로 non-linear하며, S모양 곡선의 값을 가짐
  - Linear regression으로 logit을 predict하면, non-linear probability predictor를 build할 수 있다.<span style="color:red">???</span>
  - Logit값이 P(Y=1 or 0|X), 즉 probability보다 더 linear에 가깝다.
  - **Logit은 범위가 $[-\infin,\infin]$에 들어가므로(Unconstrained Optimization) linear regression이 predict하기에 더 적합함**
    - Probability의 경우 [0,1]의 범위를 가지므로 식에 대해 $\theta$에 대한 argmin을 구할 때 $0\le X\theta\le1$이라는 조건 안에서 구해야 한다. (Constrained Optimization Problem to predict P(Y=0 or 1|X))
- 즉, Linear Regression → Logit은 classical regression problem

#### Logit → Classification

- $\log\left(\dfrac{p(x)}{1-p(x)}\right)=a(=X\theta)$라고 할 때,

  - $\dfrac{p(x)}{1-p(x)}=e^a$
    $p(x)=e^a(1-p(x))$
    $p(x)=e^a-e^ap(x)$
    $p(x)+e^ap(x)=e^a$
    $p(x)(1+e^a)=e^a$
    $\therefore p(x)=\dfrac{e^a}{1+e^a}=\dfrac{1}{1+e^{-a}}$

  - Sigmoid function

    ![image-20200406022339537](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406022339537.png)

    - x(a, $X\theta$, logit)의 범위가 $[-\infin,\infin]$이고, p(x) (probability)의 범위가 [0, 1]이 된다.
    - $P(Y=1|X=\vec{x})=\dfrac{1}{1+e^{-\vec{\beta}^\top\vec x}}$
      - $\vec{x}$가 주어졌을 때 Y의 class가 1이 될 확률
    - Sigmoid는 S모양의 함수라는 뜻이고, 위의 함수 외에도 여러 종류의 식이 존재한다.
      - <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200406024521773.png" alt="image-20200406024521773" style="zoom: 67%;" />

- 주어진 feature vector를 통해 class를 predict하는 방법

  - $P(Y=1|X=\vec{x})>P(Y=0|X=\vec{x})$라면, $\vec{x}$는 class 1이라고 볼 수 있다.
    - 즉, $p(x)>1-p(x)$
      $\dfrac{p(x)}{1-p(x)}>1$
      $\log\dfrac{p(x)}{1-p(x)}>0$
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
    - $\hat{f_1}(x;D_1)\sim\hat{f_m}(x;D_m)$
  - 이 때, Error term $E_D\left[(y-\hat{f}(x;D))^2\right]$은 <u>새로운(unseen)</u> test case $x$가 주어졌을 때 algorithm의 expected error이다. (True mean value)
    - 한편, Sample mean value는 $\dfrac{(y-\hat{f_1}(x;D_1))^2+(y-\hat{f_2}(x;D_2))^2+...+(y-\hat{f_m}(x;D_m))^2}{m}$ 와 같이 계산할 수 있다.
    - <u>m의 크기가 커질 수록 Sample mean value는 True mean에 수렴하게 된다.</u>
  - Expected error of algorithm $E_D\left[(y-\hat{f}(x;D))^2\right]$은 $\left(\mathrm{Bias}_D[(\hat{f}(x;D))^2]\right)^2+\mathrm{Var}_D[\hat{f}(x;D)]+\sigma^2$로 나타낼 수 있다.
    - $\mathrm{Bias}_D[\hat{f}(x;D)]=E_D[\hat{f}(x;D)]-f(x)$
      - Expected prediction value - ground truth value
    - $\mathrm{Var}_D[\hat{f}(x;D)]=E_D[\{\hat{f}(x;D)\}^2]-E_D[\hat{f}(x;D)]^2$
    - $\sigma^2$는 error $e$의 variance
    - 여기서 $E_D[\hat{f}(x;D)]\approx\dfrac{\underset{i}\sum\hat{f_i}(x;D_i)}{m}$로 계산한다.
      - Expectation of prediction by $m$ customers

## Other Regressions

### Quadratic Regression & Polynomial Regression

- Quadratic Regression

  <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200409160539927.png" alt="image-20200409160539927" style="zoom:50%;" />

  - $x^2$ 항이 존재(square of feature vector)
  - 형성되는 선이 더이상 linear가 아닌 parabola 형태
    - data가 더 잘 모델링될 수 있음

- Polynomial Regression

  <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200409161516762.png" alt="image-20200409161516762" style="zoom:50%;" />

  - Quadratic regression의 일반화
  - 항의 차수가 더 높아질 수록 fluctuation(변동)이 커져 model이 더 정확하게 형성될 수 있음

#### Bias-Variance Tradeoff

Expected error = $\left(\mathrm{Bias}_D[(\hat{f}(x;D))^2]\right)^2+\mathrm{Var}_D[\hat{f}(x;D)]+\sigma^2$

$\mathrm{Bias}_D[\hat{f}(x;D)]=E_D[\hat{f}(x;D)]-f(x)$

$\mathrm{Var}_D[\hat{f}(x;D)]=E_D[\{\hat{f}(x;D)\}^2]-E_D[\hat{f}(x;D)]^2$

<img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200409161617297.png" alt="image-20200409161617297" style="zoom:67%;" />

- 알고리즘의 complexity가 높아질 수록 Bias는 줄어들고 Variance는 늘어난다.(Overfitting situation)
  - Polynomial Regression의 경우 알고리즘의 complexity가 높아 model output의 변동성이 커서 variance가 커진다.
    - variance가 크다는 것은, 주어진 x에 대해 모든 customer에 대한 prediction이 다 다르므로 결과의 폭이 넓게 나옴.(not preferred)
- Variance가 커지면 Bias는 감소한다.<span style="color:red">왜??</span>
- $\sigma^2$는 true model에서 나오는 값이므로 control할 수 없다.
- <u>Bias와 Variance 간의 Tradeoff를 통해 Total error가 가장 작게 나오는 적절한 알고리즘을 찾아야 한다!</u>
  - Tradeoff point의 model은 unknown test case에 대해서도 잘 작동할 것

#### Quadratic(2차) vs. Hexic(6차) Regression

- Which one is better?

  - "Make it simple(=Hexic), but not simpler(Quadratic)" - Albert Einstein

    - 즉, Hexic을 택하되, 최대한 simple하게?

    - How to make hexic regression simple?

      <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200409161516762.png" alt="image-20200409161516762" style="zoom:50%;" />

      - $36834x$와 같은 large parameter 존재
      - "Make it simple" → large parameter는 learn하지 않도록!
        - parameter가 크면 Variance도 커진다.
        - 그림 상의 data에 대해서는 잘 맞지만 다른 data에 대해서도 잘 맞을지는 보장할 수 없음

- 결론적으로, Polynomial regression은 bias-variance tradeoff에 대해 익히기 좋은 예시지만, 실제로는 자주 사용되지 않는다.

  - Why? The curse of dimensionality
  - feature vector의 차원이 $1000^{100}$과 같이 아주 큰 경우 polynomial regression을 적용하면 엄청나게 많은 computation resource가 필요하다.
    - 즉, Computation resource가 충분하지 않은 경우는 large parameter를 배제하는 등의 작업을 거치고 사용해야 될 수도.

### Ridge Regression

- squared error loss + Regularization term 의 형태로, 더 큰 weight(coefficient)가 suppress된다.

- $\underset{\theta}{\arg\min}(X\theta-Y)^\top(X\theta-Y)+\lambda(\theta^{\top}\theta)=\underset{\theta}{\arg\min}||X\theta-Y||_2^2+\lambda||\theta||_2^2$
  - Squared L2 norm(Euclidean distance의 제곱) 활용
  - $(X\theta-Y)^\top(X\theta-Y)$: Sum of squared error
  - $\lambda$: a hyperparameter to emphasize the regularization term
  - $(\theta^{\top}\theta)=||\theta||_2^2$ ('L2 norm of $\theta$'를 제곱한 것)
  - $\Gamma=\lambda I$: Tikhonov matrix
    - $||\Gamma\theta||_2^2=\lambda||\theta||_2^2$
- 즉, Error term 뿐만 아니라 learned parameter까지 minimize하는 것
  - Error term과 regularization term의 tradeoff
- $\lambda$가 커지면,
  - model variance는 작아진다.
    - 값이 최소가 되기 위해 $\theta$의 coefficient들이 작아져야 함
  - bias는 커진다.
  - $\lambda$를 조금씩 조정하면서 balanced model을 찾을 수 있다.(hyperparameter tuning)
    - Ridge regression에서 $\lambda$는 hyperparameter이고, $\theta$는 parameter이다.
    - Hyperparameter(e.g. $\lambda$)는 직접적으로 모델에 속해있지 않다.
    - Parameter(e.g. $\theta$)는 모델의 구성 요소이며, hyperparameter에 의해 영향을 받는다.
    - 즉, $\lambda$를 조정하면 $\theta$는 완전히 다른 파라미터가 된다.

#### 결론

- Ridge regression이 linear regression보다 더 잘 작동한다.
  - Overfitting을 유발하는 <u>large parameter를 suppress</u>하므로

#### 문제점

- Ridge regression은 zero weight(coefficient)를 learn하지 못한다.
  - Ex) 두 weight value $\theta_1, \theta_2$가 있다고 가정
    - $\theta_1$는 100에서 95로 줄어들고, $\theta_2$는 10에서 5로 줄어듦
    - 그러면 $||\theta||_2^2$는 각각 $100^2-95^2=975$, $10^2-5^2=75$가 된다.
    - 이 경우, Ridge regression은 $\theta_1$를 suppress하는 데 더 focus한다.
    - 결론적으로, Ridge regression은 weight가 어느 정도까지 작아지면 이를 더 이상 suppress하지 않게 됨(do not focus on minor problems)

### Lasso Regression

Least Absolute Shrinkage and Selection Operator

- regularization(prediction accuracy) + <u>variable selection(interpretability)</u>

- $\underset{\theta}{\arg\min}||X\theta-Y||_2^2+\lambda||\theta||_1$
  - Ridge regression에서 $\lambda||\theta||_2^2$ 대신 $\lambda||\theta||_1$를 더함
  - $||\theta||_1=\underset{i}\sum|\theta_i|$
- 이 경우 regularization term에 L2 norm에 의한 square가 없다.
  - 위 예시($\theta_1, \theta_2$)를 그대로 사용할 때 $||\theta||_1$이 두 경우 모두 5로 동일
  - 즉, 같은 크기만큼 감소했으면 같은 regularization term을 가진다.(Equally important)
- feature가 useful하지 않다면, 이 feature의 weight는 0에 가까워질 것이다 → Better interpretability
  - Ex) $\theta_1=10$, $\theta_3=0.0001$이면 $x_3$은 별로 중요하지 않다는 것을 알 수 있음
  - <span style="color:red">이건 다른 regression도 그런거 아닌가?</span>

