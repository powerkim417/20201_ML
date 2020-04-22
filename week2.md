# 2. Probability & Density Estimation

## Probability

- $X$: random variable(변수)
- $P(x)$: $X=x$(특정 값)일 때의 확률
- Ex) Uniform distribution(특정 범위에서 균등한 \_-\_ 모양의 분포)을 따르는 random variable
  - <u>Discrete</u> Uniform Distribution
    - Given $k$ possible options $\{x_1, x_2, ..., x_k\}$
    - $P(X=x_i) = {1\over k}$ for all $i$
    - $\underset{i}\sum P(X=x_i)=1$
    - Probability Mass Function 사용
  - <u>Continuous</u> Uniform Distribution
    - 이 경우는 확률을 $P(X=x)$ 형태로 정의할 수 없음(~~Pointwise probability~~)
      - $P(X=x)=0$이라는 뜻과는 다름
    - 대신 $P(X\in[a,b])$ 형태로 정의함
    - $P(X\in[a,b]) \equiv \int_a^b p(x)dx = \int_a^b P(X=x)dx$ 
    - Probability Density Function
    - Ex) Uniform dist. between 1 and 3 $(U[1,3]$로 표기)
      - ![image-20200328150448907](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200328150448907.png) 
      - 각 point에서의 probability는 제대로 정의할 수 없음
        - 위의 예시의 경우 P(X=1)=0.5 이지만, 그렇다고 X=1일 때의 확률이 0.5라는 것은 아니다.
          - continuous의 경우 확률을 P(X=x) 형태로 정의하는 것이 아니므로!
        - X=1일 때의 확률을 구해보자면 정의 상a,b가 같으므로 확률은 0이 된다.

### Joint, Marginal, Conditional Probability

#### Joint Probability

- Random variable이 2개 이상 존재할 때의 확률
- Random variable이 X, Y 2개일 경우의 joint probability는 $P(X=x, Y=y)$로 표현
  - Probability에서 ,(Comma)는 교집합을 의미
  - X와 Y가 independent한 경우 $P(X=x, Y=y) = P(X=x)*P(Y=y)$
- Ex) 주사위 2개를 던졌을 때의 결과

#### Marginal Probability

- Random variable이 여러 개일 때 probability를 하나의 random variable로 표현하려 할 때 사용

- Sum Rule
  - Discrete Random Variable: $P(X=x)=\underset{y}\sum P(X=x, Y=y)$
  - Continuous Random Variable: $P(X=x)=\int P(X=x, Y=y) dy$

#### Conditional Probability

- 조건부 확률
- $P(Y=y\vert X=x) = {P(Y=y, X=x)\over P(X=x)}$
- Chain Rule
  - Joint probability distribution이 얼마나 많은 random variable들로 표현되어 있어도, contidional probability로 decompose될 수 있다.
  - $P(a, b, c) = P(a|b,c)*P(b,c) = P(a|b,c)*P(b|c)*P(c)$
    - 일반적인 ML에서 $P(a|b,c), P(b|c)$와 같은 값이 주로 계산되어 있으므로, 이 값들을 통해 $P(a,b,c)$를 구하게 된다.
- Conditional Independence
  - a와 b가 independent할 때,  $P(a,b|c) = P(a|c)*P(b|c)$

### Expectation, Variance, and Covariance

#### Expectation

- $E_{X\sim P}[f(x)] = \begin{cases}\underset{x}\sum P(X=x)f(x)\\\int P(X=x)f(x)dx\end{cases}$
  - X follows a distribution P
- $E[\alpha f(x)+\beta g(x)] = \alpha E[f(x)]+\beta E[g(x)]$

#### Variance

- $Var(f(x)) = E\left[(f(x)-E[f(x)])^2\right]=E[\{f(x)\}^2]-(E[f(x)])^2$ (제곱의 평균 - 평균의 제곱)

#### Covariance

각 random variable이 얼마나 linearly related 되어있는지 알 수 있는 지표

- $Cov(f(x), g(y)) = E[{\color{Blue}(f(x)-E[f(x)])}{\color{Red}(g(y)-E[g(y)])}]$
  - $f(x)=g(y)$, 즉 $X=Y$일 경우 $Cov(f(x), g(y)) = Var(f(x))$
  - Ex) 예시
    - $f(x)$: income of person $x\in$[\$10K, \$100K], $E[f(x)]$=30K 
    - $g(y)$: age of person $y\in$[20, 60], $E[g(y)]$=35
    - ${\color{Blue}(f(x)-E[f(x)])}$가 양수인 경우 x는 평균보다 더 많은 수입을 가지며, 음수일 경우 x는 평균보다 적은 수입을 가짐
    - ${\color{Red}(g(y)-E[g(y)])}$가 음수인 경우 x는 평균보다 더 나이가 많으며, 음수일 경우 x는 평균보다 더 나이가 적음
    - $Cov(f(x), g(y))$가 높은 경우 $\iff$ <span style="color:blue">Blue</span>, <span style="color:Red">Red</span> term이 같은 부호를 가지고, <u>동시에</u> 그들의 절대값 또한 크다. 
      - Cov의 값이 높은 경우(양의 방향으로 클 경우): x와 y가 비슷한 양상으로 움직인다.
      - Cov의 값이 낮은 경우(음의 방향으로 클 경우): x와 y가 반대 양상으로 움직인다.
- Covariance의 문제점
  - 예시와 같은 경우 x의 scale이 y의 scale보다 훨씬 크기 때문에, x의 영향을 더 많이 받게 된다.
  - 즉, covariance는 더 scale이 큰 random variable에 bias된다.
    - Fairness 문제 발생!
  - 이러한 문제점을 해결하기 위해 나온 것이 "Correlation"!

#### Correlation

- $Corr(f(x), g(y)) = \dfrac{Cov(f(x), g(y)))}{\sigma_x*\sigma_y}$
  - $Corr(f(x), (y))\in[-1,1]$
- ![image-20200329153001316](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200329153001316.png) 

### Covariance and Independence

- $Cov(F, G)=0 \nRightarrow$ F and G are "independent"
  - 반례)
    - random variable X는 U[-1, 1]을 따르고, random variable S는 1/2 확률로 1 또는 -1의 값을 가짐
    - 그러면 X와 S는 independent.
    - 여기서 새로운 random variable Y=S*X를 정의하면, Y는 X와 S에 dependent.
    - 한편, E[X]=0, E[S]=0 이고, X와 S는 independent하므로 E[Y]=E[XS]=E[X]*E[S]=0.
    - 여기서 Cov(X, Y)를 구하면 Cov(X, Y)=E[(X-0)\*(Y-0)]=E[XY]=E[X\*XS]=E[S]\*E[X]\*E[X]=0.
    - 즉, X와 Y가 independent하지 않음에도 불구하고 Cov(X, Y)가 0이 됨을 알 수 있다.
- F and G are "independent" $\Rightarrow Cov(F, G)=0$

### Bayes’ Rule

- "$P(\theta)$와 $P(X|\theta)$를 통해 **$P(\theta|X)$를 구하는 것**"

  - $P(\theta)$: Prior probability
    - parameter로 구성된 확률모형
    - 일반적으로 미리 주어짐
  - $P(X|\theta)$: Likelihood
    - parameter $\theta$가 주어졌을 때, 관측치 X가 나타날 확률
  - $P(\theta|X)$: Posterior probability
    - 관측치 X가 주어졌을 때, $\theta$의 parameter를 가지는 확률모형
  - 즉, **"Prior probability와 likelihood를 통해 Posterior probability를 구하는 것"**
    - Bayes' rule은 posterior를 구하기 위한 효율적인 방법

- $P(\theta|X) = \frac{P(\theta)P(X|\theta)}{P(X)}$

  - 즉, Posterior는 prior와 likelihood에 비례
  - 참고로, $P(X)=\underset{\theta}\sum P(X|\theta)*P(\theta)$로 구할 수 있으므로 따로 알 필요 없다.

- Random variable의 parameter를 estimate

  - 주어진 데이터를 설명하기 위해, 머신러닝 모델의 파라미터를 learn해야 함

  - 2가지 방법: **MLE**(Maximum <u>Likelihood</u> Estimation), **MAP**(Maximum A <u>Posteriori</u> estimation, Bayes' Rule과 연관)

  - Ex) 동전 뒤집기

    1. X=1일 때 앞면, X=0일 때 뒷면

       - 이 때 동전 뒤집기가 completely random하다면, P(X=1) = P(X=0) = 50%.

    2. 이 때, 동전 뒤집기로 많은 돈을 잃어서, 이 게임이 조작된 동전을 사용한다고 의심한다.

       - 즉, P(X=1)$\neq$P(X=0)인 동전이 사용되고 있다고 의심

    3. <u>현재까지 $\alpha_H$번의 앞면과 $\alpha_T$번의 뒷면이 나왔을 때**(순서 포함)**, 이 동전에 대한 확률 P(X=1)를 어떻게 구할까?</u> 

       - MLE(Likelihood를 최대로)

         1.  $P(X=1)=\theta$로 두면, 자동적으로 $P(X=0)=1-\theta$가 된다. (Bernoulli trial)

         2. 이 모델이는 1개의 scalar parameter $\theta$가 존재하며, 주어진 data($\alpha_H, \alpha_T$)를 통해 이를 learn해야 한다.

         3. 각 flip은 independent하므로, $P(\alpha_H, \alpha_T|\theta)\equiv\theta^{\alpha_H}(1-\theta)^{\alpha_T}$(Binomial Distribution)

            -  여기서 $\alpha_H, \alpha_T$는 $\alpha_H$번의 앞면과 $\alpha_T$번의 뒷면이 나온 모든 경우를 가리키는게 아니라.. 지금까지의 시행을 모두 기록했을 때  $\alpha_H$번의 앞면과 $\alpha_T$번의 뒷면이 나왔고, 이 관측치 D=HHT...HTHT 하나만을 의미하는듯. 설명이 부실했음
            - 조건부 뒤에 붙는 $\theta$는 "동전의 특이한 모양에 의해, 구하고자 하는 P(X=1)의 값이 $\theta$로 주어졌을 경우(given)" 라는 의미!

            - 이 때, $P(\alpha_H, \alpha_T|\theta)$를 likelihood function이라고 부름
            - Likelihood function: probability of observation given model parameter
            - **이 값을 최대로 하는 $\theta$를 찾는다!**
            - 즉, $\alpha_H$+$\alpha_T$번 던졌을 때 앞면이 $\alpha_H$번 나올 가능성이 가장 높도록 하는 $\theta$를 찾는다.

         4. $\theta_{MLE}^* = \underset{\theta}{\arg\max}\ln{P(\alpha_H, \alpha_T|\theta)} = \underset{\theta}{\arg\max}\ln{\theta^{\alpha_H}(1-\theta)^{\alpha_T}}$를 구한다.

            - argmax 함수는 해당 식을 최대로 만드는 변수를 출력한다.
            - *은 optimal value를 의미
            - log를 씌운 이유는 argmax 값이 바뀌지 않는 선에서 계산을 더 쉽게 할 수 있기 때문?

         5. $\ln{\theta^{\alpha_H}(1-\theta)^{\alpha_T}}$이 concave function이므로, 이에 대한 도함수를 0으로 만드는 $\theta$값이 구하고자 하는 argmax 값이 된다.

            - $\frac{d}{d\theta}\ln{\theta^{\alpha_H}(1-\theta)^{\alpha_T}}=0$
            - $\dfrac{\partial}{\partial\theta}(\alpha_H \ln\theta+\alpha_T \ln(1-\theta)) = \dfrac{\alpha_H}{\theta}-\dfrac{\alpha_T}{1-\theta} = 0$
            - $\theta_{MLE}^*=\frac{\alpha_H}{\alpha_H+\alpha_T}$
            - 즉, 전체 시행 중 앞면이 $\alpha_H$번 나올 가능성이 가장 높은 확률 $\theta$는 위의 값이 된다.

       - MAP(Posterior를 최대로)

         1. let $P(\theta)\sim \mathsf{Beta}(\beta_H, \beta_T)\equiv \dfrac{\theta^{\beta_H-1}(1-\theta)^{\beta_T-1}}{B(\beta_H, \beta_T)}$ (prior distribution)
            - Beta distribution 사용
            - 게임에 참여하기 전에, 확률이 $\theta$라고 미리 가정
            - $\beta$는 prior를 의미
              - 시행 횟수가 늘어날 수록, prior에 대한 중요도 감소
         2. $P(\alpha_H, \alpha_T|\theta)\equiv\theta^{\alpha_H}(1-\theta)^{\alpha_T}$ (likelihood)
         3. posterior $\propto$ prior * likelihood = $\theta^{\alpha_H}(1-\theta)^{\alpha_T}\cdot\dfrac{\theta^{\beta_H-1}(1-\theta)^{\beta_T-1}}{B(\beta_H, \beta_T)} $
            - 즉, posterior $\propto \dfrac{\theta^{\alpha_H+\beta_H-1}(1-\theta)^{\alpha_T+\beta_T-1}}{B(\beta_H, \beta_T)}$
         4. $\theta_{MAP}^* = \underset{\theta}{\arg\max}\dfrac{\theta^{\alpha_H+\beta_H-1}(1-\theta)^{\alpha_T+\beta_T-1}}{B(\beta_H, \beta_T)}$를 구한다.
            - 식의 형태가 beta distribution과 비슷한 형태이므로, 이를 활용
              - $\mathsf{Beta}(\alpha_H+\beta_H, \alpha_T+\beta_T)$
         5. 위의 식이 $\theta^{\alpha_H+\beta_H-1}(1-\theta)^{\alpha_T+\beta_T-1}$에 비례하는데, MLE에서 구한 방식과 유사하게 답을 추론할 수 있음.
            - $\theta_{MAP}^* = \dfrac{\alpha_H+\beta_H-1}{\alpha_H+\beta_H-1+\alpha_T+\beta_T-1}$

  - MLE는 관측 결과에 민감하다는 것이 단점이 될 수 있음

    - 정상적인 동전임에도 던졌을 때 모두 앞면만 나왔다면 MLE는 이 동전을 unbalanced coin이라고 판단할 것임

## Density Estimation

### Gaussian Distribution (a.k.a. Normal Distribution)

- $N(x;\mu, \sigma) = \dfrac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$
  - <u>';' 기호는 parameterized by를 의미</u>
    - 즉, $x$의 probability는 $\mu$와 $\sigma$에 의해 parameterized.
    - $\mu$와 $\sigma$는 이 Normal distribution의 parameter이고, $x$는 하나의 data point가 된다.
- Happens frequently in many applications, natural phenomenon
  - 반대로, 많은 자연 현상들이 Gaussian distribution으로 표현 가능하다.
- Favorable characteristics
  - 서로 independent한 여러 distribution의 합은 gaussian distribution으로 수렴함(the central limit theorem)
  - 최소한의 prior knowledge(i.e., max entropy)를 model에 encode한다. <span style="color:red">???</span>
    - Max entropy를 가지는 실수 x의 probability distribution이다.
      - Entropy: $H[P]\equiv-E\log P(x)=-\int P(x)logP(x)dx$인데, 
      - 이 값은 Probability distribution P가 적절한 $\mu$와 $\sigma$를 가진(전부는 아님) gaussian distribution일 때 최대가 된다.
- d-dimensional vector space에서의 multivariate gaussian distribution으로 확장 가능

### Mixture of Distributions

- Real-world environment에서, 하나의 distribution은 여러 종류의 같거나 다른 distribution들이 혼합된 것이다.

- Gaussian mixture model

  - 가장 강력하고 보편적인 mixture model
  - universal approximator
    - "모든 periodic function은 sin, cos function들의 합으로 근사할 수 있다" 라는 Fourier series의 concept
      - <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200402011552041.png" alt="image-20200402011552041" style="zoom:50%;" /> 
    - 하지만, 때로는 특정 periodic function을 나타내기 위해 무한 개의 sin, cos function들이 필요하듯이 <u>특정 distribution을 나타내는 데도 무한 개의 gaussian distribution이 필요할 수 있음.</u>

- Mixture model

  - $P(x) = \underset{i}\sum P(C=i)P(x|C=i)$, $P(C)$는 categorical distribution over items
  - 만약 $P(x|C=i)$가 모든 $i$에 대해 Gaussian distribution이라면, $P(x)$는 mixture of Gaussian
  - $P(C=i)$: prior, 즉 <u>$x$가 관찰되기 전에</u> categorical variable $C$에 대해 가지는 belief
  - $P(C|x)$: posterior probability
  - 이를 이용해서 MLE 또는 MAP를 하면 됨
  - <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200402012941385.png" alt="image-20200402012941385" style="zoom:80%;" /> 
  - 즉, Gaussian mixture model을 적용하는 데는 expectation, maximization algorithm 등이 필요!

- "Fitting" a gaussian mixture model to the following data

  - "Fitting"은 해당 data를 가장 잘 표현할 수 있는 (hyper)parameter를 찾는 것이다.

    ![image-20200402013235892](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200402013235892.png) 

### Self-Information & Entropy

- Information quantity는 probability에 반비례.. by Claude Shannon
- **event $x$에 대한 Self-information: $I(x) = -\log P(x)$**
  - 로그의 밑은 2, 그리고 $I(x)$가 음수임에 주의할 것
  - 'Summer is hot'과 같은 문장은 모두 아는 사실이기 때문에 유용한 정보를 전달하지 않음
    - 'summer is hot'에 대한 확률이 1이라고 하면 quantity는 0이 된다.
  - $P(x)=1$(무조건 일어남)일 때, $I(X)=-\log1=0$
  - $P(x)=0$(절대 일어나지 않음)일 때, $I(X)=-\log0=-\infin$
- **Distribution P에 대한 Entropy: $H[X]=E_{X\sim P}[I(X=x)]=-E_{X\sim P}[\log P(X)]$**
  - 의미: distribution P에서 도출한 event x에 대한 expected amount of information
    - distribution P에서 도출한 정보를 encode하는데 필요한 최소의 bit 수
      - Ex) X가 각각 50% 확률로 0 또는 1의 2가지 경우가 될 때,
        - $H[X]=-\log (0.5)\cdot0.5 + -\log (0.5)\cdot0.5 = 1$ 
        - 즉, 이 경우 expected information은 1이며,
        - random variable X를 encode하는 데 1bit가 필요하다.
          - sending 0 or 1 over communication channel
      - Ex) {0.25, 0.25, 0.25, 0.25}일 때는 2bit
    - distribution P의 정보를 전달하는 가장 좋은 coding scheme

### Cross Entropy & Kullback-Leibler (KL) Divergence

#### Entropy & Cross Entropy

- $H(P)$: distribution P에 대한 entropy
  - minimum number of bits
  - 정보에 대한 놀람도?
  - $-E_{X\sim P}[\log P(x)]$
- $H(P, Q)$: distribution P와 Q에 대한 cross entropy
  - the # of needed bits to identify an event drawn P where the coding scheme is optimized toward Q (coding scheme이 Q에 optimized 되어있을 때, P의 정보를 전달할 때 필요한 bit 수)
    - Ex) 미국과 영국에서 알파벳(A, B만 존재)을 사용할 때, 모스 코드로 이 알파벳의 sequence를 보내려 한다.
      - 미국에서는 A 70% B 30% 빈도, 영국에서는 A 30% B 70% 빈도로 사용
      - 여기서 Optimal coding scheme이란?
        - 미국에서는 더 자주 사용하는 A에 "."을 대응시키고, B에 "_"를 대응시키는 것
          - 더 짧은 signal(여기서는 ".")이 많을 수록 efficiency(transmission rate)가 증가하므로 위와 같이 대응시킬 때 같은 시간 안에 더 많은 정보를 보낼 수 있음(Maximize the volume of information transfered)
        - 반대로, 영국에서는 A에 "-", B에 "."을 대응시켜야 가장 효율적인데...
        - 만약 영국에서 미국의 coding scheme(A=".", B="-")을 사용하게 된다면?
          - 영국에서 가장 적합한 coding scheme을 사용하는 것에 비해 덜 효율적일 것이다(inefficiency)!
      - Cross Entropy는 이러한 "Inefficiency"를 나타내는 지표
  - $H(P)\le H(P, Q)$가 항상 성립 (Gibbs' inequality)
  - $-E_{X\sim P}[\log Q(x)]$
- Ex) P가 small person이고, Q가 large t-shirt라면
  - small person이 large t-shirt를 입는다면 매우 불편할 것이다.
  - small person은 small t-shirt를 입는 것이 가장 적절하며,
  - H(P)는 small person이 가장 적절한 t-shirt를 입었을 때의 entropy
  - H(P, Q)는 small person이 Q에 해당하는 large t-shirt를 입었을 때의 entropy

#### KL Divergence

- 서로 다른 distribution P와 Q에 대해, 두 distribution의 difference를 측정하는 지표
  - $D_{KL}(P||Q)=E_{X\sim P}\left[\log\frac{P(x)}{Q(x)}\right]=E_{X\sim P}[\log P(x)-\log Q(x)]$
    $=\int P(x)(\log P(x)-\log Q(x))dx = \int P(x)\log P(x)-\int P(x)\log Q(x)dx $
    $=-H(P)+H(P,Q)$
- 즉, $D_{KL}(P||Q)=-H(P)+H(P,Q)$
  - Gibbs' inequality에 의해 항상 $D_{KL}(P||Q)\ge0$ (non-negative)

---

## Q&A

### Beta Distribution

- MAP에서 사용(perfect distribution for this problem)

- Definition

  - $f(x;\alpha, \beta)$: **probability of** the success **probability** is $x$ given $\alpha$ heads, $\beta$ tails

    - 여기서는 동전 던지기에서 $\alpha$개의 head와 $\beta$개의 tail가 관측되었을 때 head가 나올 확률

    <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200420210435518.png" alt="image-20200420210435518" style="zoom: 67%;" /> 

    - $\alpha=\beta$일 때, $x$는 0.5일 때 가장 높은 함수값을 가지며, $x$가 0.5가 아닌 경우도 non-zero value를 가진다.
    - $\alpha=5, \beta=5$일 때 $x=0.5$의 함수값이  $\alpha=2, \beta=2$일 때 $x=0.5$의 함수값보다 크다.
      - 더 많이 관찰을 하고 앞뒷면이 같은 갯수가 나온 경우가  $x=0.5$일 확률이 더 높으므로!

    <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200420210443615.png" alt="image-20200420210443615" style="zoom:67%;" /> 

    - $\alpha=5, \beta=1$일 때가 $\alpha=2, \beta=1$일 때보다 true head probability가 1에 근접하며, $x=1$일 때의 함수값 또한 더 크다.
      - 전자의 경우 head를 더 많이 관찰했으므로

- Formal definition

  - $f(x;\alpha, \beta)=\dfrac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1},$ where $B(\alpha,\beta)=\int_0^1x^{\alpha-1}(1-x)^{\beta-1}dx$
    - $x$가 확률이므로 $x\in[0,1]$
    - $x$가 head probability이므로 $x^{\alpha-1},(1-x)^{\beta-1}$은 각각 $(\alpha-1)$번의 head와 $(\beta-1)$번의 tail이 나올 확률을 의미

### MAP Estimation

Probability Distribution Function이 $f(x;\alpha,\beta)$ 형태일 때,

- Prior: **A belief** you have before doing the coin flipping game
  - Probability Distribution Function이 $f(x;\alpha,\beta)$ 형태일 때, prior는 $f(\theta;\beta_H,\beta_T)$로 나타낼 수 있다.
  - ![image-20200423040313608](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200423040313608.png) 
    - 지금까지 head 100회와 tail 100회가 나왔다면 head probability가 0.5일 것이라는 belief가 가장 높을 것이고, head 2회와 tail 200회가 나왔다면 head probability가 0에 가까울 것이라는 belief가 가장 높을 것이다.
    - 여기서 반영하는 $\beta$값(사전 관측값)은 나의 belief를 형성하기 위한 parameter로, likelihood에서 말하는 observation과는 다르다!
      - Likelihood의 observation은 "your observation"으로, 실제 시행한 후 반영할 관측값
- Likelihood: The most probable head probability given **your observations**
  - 실제 시행했을 때 head 50회와 tail 50회를 관측했다면 MLE는 $\theta_{MLE}=0.5$로 도출할 것이다.
  - MLE의 경우 prior를 반영하지 않음
- Posterior: A **combination** of <u>prior(prior belief)</u> <u>and likelihood(observation)</u>
  - $\theta_{MAP}=f(x;\alpha_H+\beta_H,\alpha_T+\beta_T)$
  - 실제 시행 결과($\alpha$)와 사전 정보($\beta$)를 모두 반영
  - coin flipping 예제의 경우 처음에는 시행 횟수가 부족해 사전 정보($\beta$)에 의지하지만, 시행 횟수가 점점 많아질 경우 실제 시행 결과($\alpha$)의 비중이 높아져 MLE의 도출 결과에 수렴하게 된다.
    - 즉, $\alpha_H,\alpha_T$가 매우 큰 값일 때 $\theta_{MAP}=\theta_{MLE}$