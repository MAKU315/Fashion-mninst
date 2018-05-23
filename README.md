# Fashion-mninst
Fashion-mninst( CNN + Graphical model )

Kaggle에 올라와 있는 간단한 데이터 셋입니다. 

fashion 이미지 데이터를  mnist 로 변환한 데이터 셋

https://github.com/zalandoresearch/fashion-mnist

에서 확인 할 수 있습니다. 

기존의 숫자 mnist 처럼 tensorflow에서도 제공하고 있는데, 단순 숫자 mnist 만으로 연습하는게 지겨우셨던 분들은 한번 재미삼아 해보시면 좋을 것 같습니다.


<p align="center">
<img height="600" src="https://github.com/MAKU315/Fashion-mninst/blob/master/img/fashoin-mnist.PNG" />
</p>


해당 그림은 fashion-mnist 데이터를 불러온 결과 입니다.



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
