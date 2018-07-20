# Fashion-mninst
Fashion-mninst( CNN + Visualization )

Kaggle에 올라와 있는 간단한 데이터 셋입니다. 

fashion 이미지 데이터를  mnist 로 변환한 데이터 셋

https://github.com/zalandoresearch/fashion-mnist

에서 확인 할 수 있습니다. 

기존의 숫자 mnist 처럼 tensorflow에서도 제공하고 있는데, 단순 숫자 mnist 만으로 연습하는게 지겨우셨던 분들은 한번 재미삼아 해보시면 좋을 것 같습니다.


<p align="center">
<img height="600" src="https://github.com/MAKU315/Fashion-mninst/blob/master/img/fashoin-mnist.PNG" />
</p>


해당 그림은 fashion-mnist 데이터를 불러온 결과 입니다.


#
### CNN model 및 Confusion matrix
model 폴더의 코드를 실행하면 epoch마다의 accuracy와 loss, confusion matrix를 저장 및 확인 할수 있습니다. 

다음은 예시 입니다.

epoch 10 까지 만 학습을 하였고, 

각 step 마다의 train, validation accuracy와 loss를 확인 해 보았습니다. 

<p align="center">
<img height="300" src="https://github.com/MAKU315/Fashion-mninst/blob/master/img/Model%2004.png" />
</p>


Confusion matrix는 sklearn 라이브러리를 활용하였습니다. 

우리는 학습된 모덱에서 Shirt와 t-shirt, coat를 잘 맞추지 못하는 것을 알 수 있습니다. 

<p align="center">
<img height="600" src="https://github.com/MAKU315/Fashion-mninst/blob/master/img/Confusion%20matrix4.png" />
</p>


# Visualization

다음은 Network model을 적용해 본 결과 이다.

크게 세 가지 방법을 고려했다.

1. Correlation 2. score function( Lasso ) 3. Euclidean distance(KNN) 

Network model 활용하여, 세 가지 척도 별 시각화를 진행 했다. 

Input image, Scaling image, Last feature map과 FC-layer 를 시각화에 이용해 보았다.

확실히 FC-layer의 정보를 활용했을 때, Network 구조로 표현했을 때 가장 뛰어난 시각화를 보여주었다.

Correlaiton기반 척도를 사용했을때 의류종류 별 로 뭉쳤다. 

이때 betweeness와 degree가 높은 Node(이미지)경우 실제로 CNN 모델이 분류하지 못하는 이미지와 일치 한다는 것을 알 수 있었다.

CNN의 Classification 모델을 좀 더 활용할 수 있는 방안을 고안 할 수 있었다.


## Correlation : Last feature map vs FC layer
<div align='center'>
<img height="300" src="https://github.com/MAKU315/Fashion-mninst/blob/master/img/cor_cnn_1152.png" />
<img height="300" src="https://github.com/MAKU315/Fashion-mninst/blob/master/img/cor_cnn_128.png" />
</div>

## Score function : Last feature map vs FC layer
<div align='center'>
<img height="300" src="https://github.com/MAKU315/Fashion-mninst/blob/master/img/lasso_cnn2_1152.png" />
<img height="300" src="https://github.com/MAKU315/Fashion-mninst/blob/master/img/lasso_cnn2_128.png" />
</div>

## Euclidean Distance : Last feature map vs FC layer
<div align='center'>
<img height="300" src="https://github.com/MAKU315/Fashion-mninst/blob/master/img/knn_cnn2_1152.png" />
<img height="300" src="https://github.com/MAKU315/Fashion-mninst/blob/master/img/knn_cnn2_128.png" />
</div>


















