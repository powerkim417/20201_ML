# 5. Support Vector Machines(SVM)

## (Week 4 - Part 3 영상 내용)

- 개요
  - Multi-logic regression의 경우

    ```mermaid
    graph LR
    A(Input feature vector) --> B[Linear Regression]
    B -->|Logit| C[Sigmoid Function]
    C --> D(Class 0)
    C --> E(Class 1)
    ```

    - SVM도 이러한 구조를 따름

  - Linear classifier에 대한 개념은 이전부터 존재했지만, 충분한 성능의 컴퓨터가 존재하지 않았어서 이러한 알고리즘은 개념으로써만 존재했다.

  - 한편, SVM은 90년대 이후로 popular. 가장 성공적인 classification algorithm model

  - 현재는 multi-layer neural network와 같은 다양한 deep learning model이 있지만, SVM 알고리즘 자체의 중요성을 강조하기 위해(ML의 마일스톤과 같음) 배운다.

- Feature의 중요성(Feature Engineering)

  - input feature가 garbage면 어떤 classification을 쓴다 해도 성공하기 힘들다.

  - Ex 1) ![image-20200417170740105](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200417170740105.png) 와 같은 경우

    - 두 class를 linear classification으로 나누기기 위해서는 hyperplane(직선 및 평면)을 사용해야 하지만, 위와 같이 circle 형태로밖에 분류를 하지 못한다.
      - Linear classification: in
    - <u>해법: 새로운 feature를 추가!</u>(e.g., 중심으로부터의 거리(radius))
    - 그러면 위와 같은 공간으로 표현할 수 있고,![image-20200417170957069](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200417170957069.png) 와 같이 hyperplane으로 나눌 수 있는 형태가 된다.

  - Ex 2) 이메일 스팸 분류

    - 이메일을 represent할 수 있는 feature를 설계하고, 스팸 분류에 활용

    1. <u>특정 이메일</u>을 받았을 때, 이 이메일에 대한 feature들을 계산

       ![image-20200418151844736](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200418151844736.png) 

       - wordcount(word가 등장한 횟수), from(보낸 사람), subject(이메일 제목), date(날짜), has(이메일 포맷), mentions(특정 주제 언급 여부), salutation(인삿말 포함 여부)

    2. 이러한 feature를 이용하여 이메일에 대한 <u>vector representation</u>을 구할 수 있음

    3. 이러한 vector를 <u>classifier</u>에 넣어 스팸인지 아닌지 분류

### Linear Classifier in SVM

<img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200418154202311.png" alt="image-20200418154202311" style="zoom:67%;" /> (+b 또는 -b 둘 다 이론적으론 같은 의미)

- <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200418161138078.png" alt="image-20200418161138078" style="zoom:67%;" /> 
  - 두 Class를 linear regression으로 classify
  - $WX + b$ = logit (output of linear regression)
    - $P(+|X)$와 logit의 관계: ![image-20200418163259315](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200418163259315.png) 
  - 즉, $f(x,w,b)$ = (wx+b 의 부호) 로 정할 수 있다.
  - 그런데, 위 그림의 경우 두 class를 구분하는 hyperplane이 매우 다양하게 존재할 수 있다..!
    - 이 경우 다른 data point를 넣었을 때 이에 대한 classification 결과가 hyperplane에 따라 다르게 나와버림
      - One deterministic hyperplane 필요!(이 중에 하나를 골라야 함)
    - classification accuracy에 대해서만 생각하는 경우에는 어느 거여도 상관 없음

### Classifier Margin

- SVM classification의 main novelty

- ![image-20200418171036361](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200418171036361.png)  

  - Margin: 그래프의 노란 부분
    - margin의 center가 hyperplane
    - margin 내에는 point가 없음
  - Support vector: 그래프에서 margin의 경계와 닿아있는 data point

- SVM의 역할은 이렇게 다양한 hyperplane 중 maximum margin을 가지는 hyperplane을 고르는 것!

  - Maximum margin인 경우가 가장 좋음

- 왜 margin을 최대로 가져야 하는가?

  - 직관적으로도 가장 안정적임

- 한편, 이런 경우도 있다

  ![image-20200418205839871](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200418205839871.png) 

  - 하나의 +1 class 때문에 hyperplane의 margin이 매우 좁게 형성된 상황
  - 이 경우 이렇게 hyperplane을 잡는 것이 가장 적절할까? **NO**
  - Margin area를 위해 training dataset에 대한 classification accuracy를 희생하는 것이 더 적절하다.

### Specifying a line and margin

![image-20200418212045400](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200418212045400.png) 

<span style="color:red">왜 우변이 1, -1로 고정되는가.. support vector를 wx+b에 대입했을 때 1이 아닌 값이 나올 수도 있지 않나? </span>

- $w\cdot x+b>=1$ (class +1로 분류)

- **$w\cdot x+b=1$ (positive plane)**
- $0<w\cdot x+b<1$ (해당 data 없음)
- **$w\cdot x+b=0$ (hyperplane)**
- $-1<w\cdot x+b<0$ (해당 data 없음)
- **$w\cdot x+b=-1$ (negative plane)**
- $w\cdot x+b<=-1$ (class -1로 분류)
- <u>이 때, vector $w$는 위의 plane들과 수직!</u>
  - Proof) positive plane과 수직임을 보인다.
    - positive plane 위에 임의의 두 vector $u, v$가 있다고 할 때,
    - $w\cdot (u-v) = w\cdot u - w\cdot v = (1-b) - (1-b) = 0$
    - 즉, $w$와 $u-v$의 내적 값이 0인데,
    - 내적 값이 0이라는 것은 두 vector가 수직이라는 뜻이고, 
    - vector $u-v$ 역시 positive plane 위에 있는 vector가 되므로
    - $w$는 positive plane 위의 임의의 벡터와 모두 수직이 되어 결국 plane과도 수직이 된다.
  - 위 증명을 hyperplane, negative plane에도 적용 가능

### Margin width

![image-20200418230405637](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200418230405637.png) 

- $x^+$를 positive plane 위의 임의의 point로 잡고, $x^-$는 negative plane 위의 point 중 $x^+$로부터 가장 가까운 point로 잡는다.
  - $w\cdot x^++b=1, w\cdot x^-+b=-1$
- 그러면 $x^+$와 $x^-$를 연결하는 line segment는 hyperplane과 수직이 되고,
- 이렇게 잡았을 때 $x^+$와 $x^-$ 사이의 거리가 최소가 된다.
- 이 경우, $x^+ = x^- + \lambda w$ ($\lambda$: scaling factor) 가 성립한다.
- 한편, 여기서의 margin width를 $M$으로 두면 $M=||x^+-x^-||_2$가 성립하는데,
- 이 $M$을 $w$와 $b$로 나타내보려 한다.
- $w\cdot x^++b=w\cdot(x^- + \lambda w)+b=w\cdot x^-+\lambda w\cdot w+b$
  $=(-1-b)+\lambda w\cdot w+b=\lambda w\cdot w-1=1$
- 즉, $\lambda w\cdot w=2, \lambda=\dfrac{2}{w\cdot w}$
- 한편, $M=||x^+-x^-||_2=||\lambda w||_2=\lambda||w||_2=\dfrac{2}{w\cdot w}\sqrt{w\cdot w}=\dfrac{2}{\sqrt{w\cdot w}}$
- $M=\dfrac{2}{\sqrt{w\cdot w}}$
  - <u>즉, Margin width가 learned parameter vector $w$에 의해 결정된다.</u>

이 이후 다양한 형태로 SVM을 train하는 방법이 제시되었지만, 전부 부족했다.

## (Week 5 - Part 2 영상 내용)

### Learning Maximum Margin with Noise

- Classification이 진행된 후 모습

  ![image-20200426194147942](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200426194147942.png)

  - 여기서 분류 결과와 맞지 않는 data set들에 대해 error를 구할 수 있음

- 이 optimization problem은 결국 $\min\left\{\dfrac{1}{2}w\cdot w +c\cdot\overset{R}{\underset{k=1}{\sum}}\epsilon_k\right\}$과 같다.

  - $R$: training sample의 갯수
    - 실제 Red class($y_k=1$)의 data 중 $x_k+b\ge1-\epsilon_k$(Blue class로 분류됨)인 경우와
    - 실제 Blue class($y_k=-1$)의 data 중 $x_k+b\le-1+\epsilon_k$(Red class로 분류됨)인 경우가 해당된다.
  - $\epsilon_k\ge0$
  - Find the smallest allowable error $\epsilon_k$

- 위의 값을 최소로 하되, margin은 최대한 크게 가져가는 것이 SVM의 철학

### An Equivalent QP

위 문제를 동등한 의미의 식으로 다시 쓸 수 있음

- Maximize:
  - $\overset{R}{\underset{k=1}{\sum}}\alpha_k-\dfrac{1}{2}\overset{R}{\underset{k=1}{\sum}}\overset{R}{\underset{l=1}{\sum}}\alpha_k\alpha_lQ_{kl}$
    - $Q_{kl}=y_ky_l(x_k\cdot x_l)$
- Constraints:
  - $0\le \alpha_k\le C$
  - $\overset{R}{\underset{k=1}{\sum}}\alpha_ky_k=0$
- Then define:
  - $w=\overset{R}{\underset{k=1}{\sum}}\alpha_ky_kx_k$
  - $b=y_K(1-\epsilon_K)-x_K\cdot w_K$ where $K=\underset{k}{\arg\max}\alpha_k$
- Then classify with:
  - $f(x,w,b)=sign(w\cdot x-b)$

### SVM Algorithm 적용

- ![image-20200426210845870](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200426210845870.png) 
  - 이런 경우는 쉽게 나눌 수 있음
- ![image-20200426211010616](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200426211010616.png) 
  - 이런 경우는 쉽게 나누기가 힘듦...
  - 이 경우 input에 새로운 차원을 추가한다. 즉, $x \to (x,x^2)$
    - ![image-20200426211121891](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200426211121891.png) 
    - 이렇게 새로운 차원을 추가하면 더 쉬운 classify가 가능하다.
  - 이렇게 input $x_k$를 $(x_k,x_k^2)$와 같이 바꾸는 함수를 kernel function이라고 한다.
    - $z_k=(x_k,x_k^2)$

#### Kernel Function

- 목적: To distort the original feature vector space

- polynomial, RBF, sigmoid 등 다양한 형태 존재
- Ex) Quadratic Kernel
  - ![image-20200426212201646](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200426212201646.png) 
  - input vector가 $m$-dimension일 때, 이 feature로 만들 수 있는 quadratic term이 모두 존재
  - term의 수는 $_{m+2}\mathrm{C}_{2}=\dfrac{(m+2)(m+1)}{2}\approx \dfrac{m^2}{2}$
    - m개의 feature와 '없음'과 '중복' 이라는 $m+2$개 선택지 중 2개를 뽑는다고 생각하면 편함. $x_1$과 중복을 골랐으면 $x_1^2$가 되는거고, 없음과 중복을 골랐으면 1이 되는 셈..

### QP with Basis Functions

- Maximize:
  - $\overset{R}{\underset{k=1}{\sum}}\alpha_k-\dfrac{1}{2}\overset{R}{\underset{k=1}{\sum}}\overset{R}{\underset{l=1}{\sum}}\alpha_k\alpha_lQ_{kl}$
    - $Q_{kl}=y_ky_l(\Phi(x_k)\cdot \Phi(x_l))$
    - 원래 input vector인 $x$ 대신 수많은 artificial information을 함께 가지고 있는 $\Phi(x)$를 넣는다.
- Constraints:
  - $0\le \alpha_k\le C$
  - $\overset{R}{\underset{k=1}{\sum}}\alpha_ky_k=0$
- Then define:
  - $w=\overset{R}{\underset{k=1}{\sum}}\alpha_ky_k\Phi(x_k)$
  - $b=y_K(1-\epsilon_K)-x_K\cdot w_K$ where $K=\underset{k}{\arg\max}\alpha_k$
- Then classify with:
  - $f(x,w,b)=sign(w\cdot \Phi(x)-b)$

- 문제점: Calculation of the entire matrix $Q$ is prohibitive(expensive)
  - 두 kernel간의 dot product를 구하려면 총 $\dfrac{R^2m^2}{4}$회의 연산이 필요하다..
  - 그러나 두 연산을 자세히 살펴보면
  - ![image-20200427012254387](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200427012254387.png) 
  - 즉, $\Phi(a)\cdot\Phi(b)=1+2\overset{m}{\underset{i=1}{\sum}}a_ib_i+\overset{m}{\underset{i=1}{\sum}}a_i^2b_i^2+2\overset{m}{\underset{i=1}{\sum}}\overset{m}{\underset{j=i+1}{\sum}}a_ib_ja_ib_j$
  - 한편, $(a\cdot b+1)^2$ 역시 전개해보면 위와 같은 형태이다! 즉 $\Phi(a)\cdot\Phi(b)=(a\cdot b+1)^2$ (Kernel Trick)
    - 이러한 형태를 맞추려고 kernel의 일부 항에 $\sqrt{2}$가 들어가는듯.
    - 이 연산의 경우 $\dfrac{R^2}{2}$회의 연산만을 필요로 함

### Higher Order Polynomials

![image-20200427013506793](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200427013506793.png) 

위와 같이 $\Phi(a)\cdot\Phi(b)$를 $(a\cdot b+1)^n$ 꼴로 변형하면 아무리 차수가 높은 case여도 연산의 수가 늘어나지 않는다!

### SVM Kernel Functions

$K(a,b)$

- 위에서 구한 $K(a,b)=(a\cdot b+1)^d$는 SVM Kernel Function의 한 예시!(polynomial)
- Radial-Basis-style Kernel Function: $K(a,b)=-exp\left(-\dfrac{(a-b)^2}{2\sigma^2}\right)$
- Neural-net-style Kernel Function: $K(a,b)=tanh(\kappa a\cdot b-\delta)$

### Other applications of SVM

- Regression, Density estimation 등에도 활용될 수 있음
  - SVM의 Output은 raw value가 될 수 있다.(class prediction 대신에)<span style="color:red">???</span>

### 결론

-  SVM classification에서 가장 중요한 2가지
  - Maximize the margin while minimizing training classification errors
  - Apply kernel function to original input space
    - 연산이 expensive해보이지만, 실제로는 좀 더 간단하게 할 수 있다.

## (Week 6 - Part 1 영상 내용)

