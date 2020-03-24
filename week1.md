# 1. Machine Learning Overview and Recap. of Math

## Theoretical vs. Applied Machine Learning

### Classification

- Ex) 소비자의 수입과 나이로 소비 유형 예측

  <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200321013757615.png" alt="image-20200321013757615" style="zoom:80%;" />

  - 문제 조건
    1. 각 소비자는 '수입'과 '나이'라는 2가지 feature를 가짐
    2. 위의 표본들이 주어졌을 때, classification algorithm을 학습시켜라!
       - **두 class를 split하는 가장 적절한 boundary를 찾아야 함! => "Classification"**
  - Simple rule-based classifer를 설계할 수 있음
    - 초록색 원(70k <= income <= 100k && 30 <= age <= 45)일 때 premium
    - 그러면 빨간 별의 새로운 소비자 데이터가 등장했을 때 이 소비자가 premium purchase를 할 것이라고 predict할 수 있음

### Theoretical vs. Applied Machine Learning

- Theoretical Machine Learning
  - 더 적절한 boundary(hyperplane)를 발견하는 **method를 설계**하는 단계
    - 다양한 feature가 있을 수 있기 때문에(high-dimensional vector space) 유용한 feature를 통해 boundary를 설계하는 것은 매우 어려운 작업!
  - 고도의 수학(Linear algebra, mathematical optimization) 필요
- Applied Machine Learning(= Feature Engineering)
  - Feature engineering(데이터의 변수를 통해 feature(Ex. age, income)를 설정하는 것) 후 최첨단 classifier를 활용하는 단계
  - 고도의 domain knowledge???와 다른 종류의 수학(game theory model) 필요

### Feature Engineering

Ex) Data-driven Airline Profit Maximization (Feature Engineering이 많이 들어간 연구)

- 항공사에서 4개의 노선을 운항할 때, 제한된 예산(운항에 들어가는 비용)을 가장 최적으로 할당하여 수익을 최대화 하는 방법 구하기(flight frequency를 조정하여)
- 2가지 필요한 알고리즘
  - profit-frequency 간의 관계를 정확히 **예측(forecasting)**
    - frequency를 각각 배정했을 때 output으로 profit이 계산되도록
  - 위의 예측을 기반으로 **가장 적합하게 할당(optimal allocation)**

#### Forecasting

- Market share(시장 점유율)에 대한 예측(Regression)

  - Regression: 우리에게 주어진 feature로 market share를 예측하는 것
    - 여기서 feature는 ticket price, frequency, delay time, delay ratio, ...
  - 각 항공사에 대해 이 feature를 정의???

- Feature Engineering

  1. 각 항공사를 포함하는 zero-sum game theory model 설계	

     - LA->NY로 가는 초기 ticket price: Delta(\$190), AA(\$180), United($200)
     - 각 항공사는 경쟁사의 가격을 계속 모니터링하며 가격을 조정하려 할 것이다.

  2. 각 항공사가 자신의 equilibrium ticket price(균형 가격)를 찾는다.

     - equilibrium price: 수요와 공급이 일치할 때 성립되는 가격

  3. **이 "equilibrium ticket price와 suggested ticket price(시장 가격)의 차의 절대값"을 새로운 feature로 추가한다. (Feature Engineering)**

     - 시장가격이 더 높으면 상대적으로 비싸다고 생각되므로 시장 점유율이 내려갈 것이고, 시장가격이 더 낮으면 상대적으로 싸다고 생각되므로 시장 점유율이 올라갈 것을 예측할 수 있음

     - 이렇게 새로 추가한 feature는 데이터베이스에서 따온 다른 feature들보다 market share를 predict 하는 데 있어서 더욱 효과적일 것이다.

       ![image-20200321163845744](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200321163845744.png)

       - 데이터베이스에서 다운로드받은 basic feature만을 사용하는 것은 누구나 할 수 있음, 그러나 equilibrium ticket price를 활용하는 feature engineering은 이와 다름!
         - 그래서 더 어렵기도 함

#### Profit Maximization

이부분 잘 모르겠다...???

![image-20200321172538800](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200321172538800.png)

- $f_i$: route $i$에 대한 flight frequency
- $N_i(f_i)$: predicted market share(=profit)
  - $N_i$: regression model trained for route $i$ 
  - $\underset{R_i \in R}\sum N_i(f_i)$: 모든 route에 대해 예측한 market share를 모두 합한 값
    - 이 값이 최대가 되는 $f_i$를 구해야 한다!
    - 조건: $0\le f_i\le f_i^{max}, f_i\in\N, \forall R_i\in R,\underset{R_i \in R}\sum C_if_i\le b$ (지정된 flight frequency의 쌍을 이용해 발생한 총 비용이 budget을 넘지 않아야 함)
- 이렇게 구하는 방법을 prediction-driven optimization 이라고 한다.
  - prediction 작업($N_i$를 구하는 작업)을 통해 최적의 $f_i$의 쌍을 찾는 것???
  - Prediction: Machine learning techniques
  - Decision making: optimization
  - 이 두가지가 query를 통해 interact한다.
  - Prediction(ML) 자체만으로는 의미가 없을 수 있음
    - Decision making을 하기 위한 starting point가 된다.

Feature engineering은 최악의 경우 노력에 비해 아무 성과도 얻지 못 할 수도 있다..

### Artificial Intelligence

- AI가 포함하는 것
  - Reasoning, problem solving
    - Step-by-step reasoning in logical deductions(추론)
  - Knowledge representation
    - Expert knowledge in a narrow domain
    - Ex) IBM Watson
  - Planning, decision making
    - Game theory, optimization
    - Agent-based model
  - Learning
    - Machine learning (우리가 배우는 부분)
  - General(Multi-modal) intelligence

- AI $\supset$ ML $\supset$ DL
  - AI: Any technique that enables computers to <u>mimic human behavior</u>
  - ML: Ability to <u>learn without explicitly being programmed</u>
  - DL: Extract patterns from data <u>using neural networks</u>

## Recap. of Linear Algebra

### Several Types of Mathematical Objects

- Scalar: 숫자 1개(실수)

- Vector: 배열.  Ex) [1, 1.5, 2.278] $\in R^3$ (해당 vector는 3-dimensional vector)

  - d-dimensional vector: d차원 좌표계에서의 점

  ![image-20200323005031502](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200323005031502.png)

  - Basis vector
    - 3-dimensional space일 때, $b_x = (1, 0, 0), b_y = (0, 1, 0), b_z = (0, 0, 1)$ 3개의 벡터가 basis vector가 된다.
  - Linear combination
    - "모든 벡터는 basis vector들의 linear combination으로 표현할 수 있다"
    - 마찬가지로 3-dimensional space일 때, $v=[a, b, c]$이면 $v=a\cdot b_x + b\cdot b_y + c\cdot b_z$

- Matrix: 2차원의 배열

  - Diagonal matrix: 대각선의 원소만 값을 가지고, 나머지 원소는 전부 0인 경우

- Tensor: 3차원 이상의 배열

### Multiplying Matrices and Vectors

- Matrix Product
  - $A$가 **m**\*n matrix이고, $B$가 n\***p** matrix 일 때, $AB$는 **m**\***p** matrix가 된다.
  - Distributive(분배 법칙을 따름). $A(B+C) = AB + AB$
  - Associative(결합 법칙을 따름). $A(BC) = (AB)C$
  - Not commutative(교환 법칙은 따르지 않음). $AB \neq BA$

- Element-wise Product (= Hadamard Product)

  - $A$가 **m\*n** matrix이고, $B$가 **m\*n** matrix 일 때, $A\odot B$는 **m\*n** matrix가 된다.

    - 곱하는 두 행렬의 크기가 같아야 함

  - 결과의 각 원소는 두 피연산자의 각 위치의 원소를 곱한 값을 가진다.

    ![image-20200323011811122](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200323011811122.png)

- Transpose

  - $A^\top$는 $A$의 row와 column을 바꾼 matrix
  - $(AB)^\top = B^\top A^\top$

### Linear dependence and Span

- Linear Independence
  - vector의 집합 $\{v_1, v_2, ..., v_n\}$이 있을 때, 모든 가능한 $a_j$에 대하여 $v_i \neq \underset{j\neq i}\sum a_jv_j$를 만족한다면 해당 집합은 linearly independent!
    - 하나의 vector가 집합의 나머지 vector들로 표현되지 않아야 하는 것!
  - 가장 trivial한 예시: basis vector
- Square Matrix
  - n*m 크기의 matrix A가 square matrix $\iff$ n = m
  - <u>Non-singular</u> square matrix: 각 row vector가 linearly independent한 square matrix
    - Non-singular: $A^{-1}$ 존재
    - Singular: $A^{-1}$ 이 존재하지 않음

### Identity and Inverse Matrices

- Identity Matrix
  - $I = \begin{pmatrix} 1&0&0\\0&1&0\\0&0&1\end{pmatrix}$
  - $AA^{-1} = I$

### Norms

- Vector norm: vector의 size or length or distance를 측정하기 위한 지표
  - 성질
    - $f(0) = 0$
    - $f(x+y) \le f(x)+f(y)$ (Triangle Inequality)
    - $\forall a\in R, f(ax)=\vert a\vert f(x)$
  - $L_p$ norm: $\Vert x\Vert_p = (\underset{i}\sum\vert x_i\vert^p )^{1\over p} (p = 1, 2, ...)$
    - $L_1$ norm (=Manhattan Distance)
      - 각 원소의 절댓값의 합을 더한 것
    - $L_2$ norm (=Euclidean Distance)
      - 원점에서 해당 점까지의 거리
    - $L_3$ norm, $L_4$ norm, ...
    - $L_{\infin}$ norm = $max\vert x_i\vert$
      - vector의 원소의 절댓값 중 가장 큰 값
- Matrix norm
  - Frobenius matrix norm: $\Vert A\Vert_F = \sqrt{(\underset{i, j}\sum {a_{i,j}}^2 )}$
    - Matrix의 모든 원소에 대한 $L_2$ vector norm과 같은 개념

### Special Kinds of Matrices and Vectors

- Symmetric Matrix
  - A가 symmetric matrix $\iff$ $A = A^\top$
- Unit Vector
  - v가 unit vector $\iff$ $\Vert v\Vert_2 = 1$ ($L_2$ norm)

### Eigendecomposition

#### Eigenvector & Eigenvalue

- Square matrix A가 주어졌을 때, $Av = \lambda v $를 만족하는 scalar $\lambda$와 vector $v$가 있다면 <u>$\lambda$가 eigenvalue, $v$가 eigenvector가 된다.</u>

  - A에 대해 여러 쌍의 eigenvalue-eigenvector가 존재할 수 있다.

    - **A가 n*n 크기일 경우, n개의 eigenvalue-eigenvector 쌍이 존재**

  - $Av$는 linear transformation

    - Linear transformation: mapping from a vector space to other vector space

      ![image-20200324010516633](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200324010516633.png)

      - vector $v$가 존재할 때, A(transform matrix)와 곱하면 A에서 의미하는 linear transformation이 일어난 값이 결과가 된다.

- 예시

  - $A = \begin{pmatrix} 2&1\\1&-1\end{pmatrix}, u=\begin{pmatrix} 1\\1\end{pmatrix}$일 때, $u' = Au = \begin{pmatrix}3\\0 \end{pmatrix}$
  - ![image-20200324031949956](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200324031949956.png)
  - $Ae_1 = \lambda_1e_1, Ae_2 = \lambda_2e_2$를 만족!
  - **$(e, \lambda)$ 쌍의 의미**
    - **특정 vector에 대한 Linear transformation을 표현하는 각각의 절차 중 하나가 됨**
      - 각 쌍 하나씩만 수행하면 안되고, n개를 모두 수행해야 u에서 u'이 된다.
      - 다시 말하면, 각 쌍을 모두 찾아내면 Linear transformation에 대한 정보를 알 수 있다.
    - e는 기준 vector가 되므로 Linear transformation A를 수행한 후의 결과 vector도 e와 같은 방향을 유지
    - 각 쌍에 의한 변환의 경우 eigenvalue($\lambda$)의 절댓값만큼 vector가 stretch되며, 부호가 음수면 vector는 해당 eigenvector($e$)로부터 reflect된다. <span style="color:red">**잘못된 설명**</span>

- eigenvector & eigenvalue의 의미를 통해 이를 계산 없이 구할 수도 있다.

  1.  $A = \begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix}$를 예로 들면 A는 identity matrix 이므로
  2. x축을 따르며, reflection 없고, stretch 없음 $\to \lambda_1 = 1, e_1=(1,0,0)$
     y축을 따르며, reflection 없고, stretch 없음 $\to \lambda_2 = 1, e_2=(0,1,0)$
     z축을 따르며, reflection 없고, stretch 없음 $\to \lambda_3= 1, e_3=(0,0,1)$

### Fully Connected Layer

- $\sigma(Wh_i+b)$: Fully Connected Layer in Deep Learning
  - Logistic Regression에서 특별한 의미를 가지고 있음
  - $W$: linear transformation matrix
  - $h_i$: hidden vector at layer $i$
  - $b$: bias vector
  - $\sigma$: <u>non-linear</u> activation function (Ex. ReLU)
- **즉, 하나의 Fully connected layer는 이전의 hidden vector를 linear transform한 후 bias를 더한 값을 activation function에 통과시킨 결과!**
  - $\Rightarrow$Fully Connected Layer는 "non-linear transformation"

<img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200324223959491.png" alt="image-20200324223959491" style="zoom:50%;" />

![image-20200324225542216](C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200324225542216.png)

- Classification 문제에서 빨간 class와 파란 class를 나눠야 할 경우,

  - 어떤 linear hyperplane도 두 class를 성공적으로 구분할 수 없다.

  - 하지만, 여러 Fully Connected Layer를 활용한다면(a series of FC layers)

    <img src="C:\Users\KJH\AppData\Roaming\Typora\typora-user-images\image-20200324223959491-tile_2.png" alt="image-20200324223959491-tile_2" style="zoom:33%;" />

  - 두 class를 나눌 수 있는 hyperplane이 존재할 수 있는 형태로 transform될 수 있다.