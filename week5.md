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

## (Week 6 - Part 1 영상 내용)