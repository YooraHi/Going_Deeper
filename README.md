# Going_Deeper_Project 18_Reivew

아이펠캠퍼스 온라인4기 피어코드리뷰 [2023-07-20]

- 코더 : 최우정
- 리뷰어 : 김창완

----------------------------------------------

**PRT(PeerReviewTemplate)**

** [ ] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   
|평가문항|상세기준|완료여뷰|
|-------|--------|-------|
|1.Text recognition을 위해 특화된 데이터셋 구성이 체계적으로 진행되었다.| 텍스트 이미지 리사이징, ctc loss 측정을 위한 라벨 인코딩, 배치처리 등이 적절히 수행되었다. |리사이징과 라벨 인코딩, 배치처리가 전부 잘 되었습니다|
|2. CRNN 기반의 recognition 모델의 학습이 정상적으로 진행되었다. |학습결과 loss가 안정적으로 감소하고 대부분의 문자인식 추론 결과가 정확하다. |epoch이 4임에도 불구하고 로스가 줄어드는게 잘 보입니다|
|3. keras-ocr detector와 CRNN recognizer를 엮어 원본 이미지 입력으로부터 text가 출력되는 OCR이 End-to-End로 구성되었다. |샘플 이미지를 원본으로 받아 OCR 수행 결과를 리턴하는 1개의 함수가 만들어졌다.|원본 이미지를 받고 crop까지 해서 제대로 보이는 함수가 다 작성 되었습니다|

** [ ] 주석을 보고 작성자의 코드가 이해되었나요?

 - 네 특히 데이터 불러오기와 라벨 출력부분이 잘 되어 있었습니다.

   ```python
   # env에 데이터 불러오기
   # lmdb에서 데이터를 불러올 때 env라는 변수명 사용
   env = lmdb.open(TRAIN_DATA_PATH, 
                   max_readers=32, # 
                   readonly=True, 
                   lock=False, 
                   readahead=False, 
                   meminit=False)
   
   # 불러온 데이터를 txn(transaction) 변수를 통해 열기
   # txn변수를 통해 직접 데이터에 접근
   with env.begin(write=False) as txn:
       for index in range(1, 5):
           # index를 이용해서 라벨 키와 이미지 키를 만들면
           # txn에서 라벨과 이미지를 읽어올 수 있음
           label_key = 'label-%09d'.encode() % index
           label = txn.get(label_key).decode('utf-8')
           img_key = 'image-%09d'.encode() % index
           imgbuf = txn.get(img_key)
           buf = six.BytesIO()
           buf.write(imgbuf)
           buf.seek(0)
   
           # 이미지는 버퍼를 통해 읽어오기 때문에 
           # 버퍼에서 이미지로 변환하는 과정이 다시 필요
           try:
               img = Image.open(buf).convert('RGB')
   
           except IOError:
               img = Image.new('RGB', (100, 32))
               label = '-'
   
           # 원본 이미지 크기 출력
           width, height = img.size
           print('original image width:{}, height:{}'.format(width, height))
           
           # 이미지 비율을 유지하면서 높이를 32로 바꾸기
           # 하지만 너비를 100보다는 작게하기
           target_width = min(int(width*32/height), 100)
           target_img_size = (target_width,32)        
           print('target_img_size:{}'.format(target_img_size))        
           img = np.array(img.resize(target_img_size)).transpose(1,0,2)
   
           # 높이가 32로 일정한 이미지와 라벨을 함께 출력      
           print('display img shape:{}'.format(img.shape))
           print('label:{}'.format(label))
           display(Image.fromarray(img.transpose(1,0,2).astype(np.uint8)))
           print('--------')
   ```

   

** [ ] 코드가 에러를 유발할 가능성이 있나요?

- 에러가 보이지 않고 안정적으로 전부 동작함을 확인 했습니다.

** [ ] 코드가 간결한가요?

- 네 코드가 딱히 더 줄일점은 보이지 않고 깔끔하게 작성 되었습니다.
- 전체적으로 흠 잡을데 없었습니다

------------------------------------------------
------------------------------------------------

아이펠캠퍼스 온라인4기 피어코드리뷰 [2023-06-30]

- 코더 : 최우정
- 리뷰어 : 최지호

----------------------------------------------

**PRT(PeerReviewTemplate)**

** [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   
|평가문항|상세기준|완료여뷰|
|-------|--------|-------|
|1. CAM을 얻기 위한 기본모델의 구성과 학습이 정상 진행되었는가?|ResNet50 + GAP + DenseLayer 결합된 CAM 모델의 학습과정이 안정적으로 수렴하였다.|로컬 환경에서 지속적으로 학습이 끊겨서 저장, 불러오기를 하면서 학습하셨다고 합니다. 결과는 수렴했다고 합니다.|
|2. 분류근거를 설명 가능한 Class activation map을 얻을 수 있는가?|CAM 방식과 Grad-CAM 방식의 class activation map이 정상적으로 얻어지며, 시각화하였을 때 해당 object의 주요 특징 위치를 잘 반영한다.|이 부분도 다른 환경에서 성공 하셨다는데 아쉽게도 로컬에서 불러오시지 않았네요..|
|CAM 방식과 Grad-CAM 방식의 class activation map이 정상적으로 얻어지며, 시각화하였을 때 해당 object의 주요 특징 위치를 잘 반영한다.|CAM과 Grad-CAM 각각에 대해 원본이미지합성, 바운딩박스, IoU 계산 과정을 통해 CAM과 Grad-CAM의 object localization 성능이 비교분석되었다.|네. IOU 결과 수치를 비교하여 결론으로 작성 해주셨습니다.|


** [o] 주석을 보고 작성자의 코드가 이해되었나요?
  - ![image](https://github.com/YooraHi/Going_Deeper/assets/79844211/6a250d71-3f1c-4151-a353-cd158a144031)
  - md형식과 주석으로 설명이 자세히 되어 있어서 이해하기 편했습니다.

** [x] 코드가 에러를 유발할 가능성이 있나요?
  - 없습니다.

** [o] 코드가 간결한가요?
  - ![image](https://github.com/YooraHi/Going_Deeper/assets/79844211/affaf879-c25f-4a8e-87e4-b09457c9ff5a)

  - 재사용 가능한 부분은 전부 함수로 만들어져 있습니다.

----------------------------------------------

로컬과 다른 환경에서 동시에 프로젝트를 진행하셔서 로컬 기준으로 올라온 노트북 파일에 프로젝트 진척을 시각적으로 확인할 수 없었던 점이 아쉽습니다...
