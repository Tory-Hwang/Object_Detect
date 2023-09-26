import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import cv2

# 카메라 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

# 모델 불러오기
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# 이미지 전처리를 위한 변환 선언
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

while True:
    # 비디오 프레임 읽기
    ret, frame = cap.read()

    # 이미지 전처리
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)

    # 모델 입력을 위해 GPU 사용 여부 확인
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # 추론을 위해 모델에 입력 전달
    with torch.no_grad():
        output = model(input_batch)

    # 클래스 예측 결과 얻기
    _, predicted_idx = torch.max(output, 1)
    predicted_class = predicted_idx.item()

    # 예측된 클래스를 화면에 표시
    cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 비디오 화면 표시
    cv2.imshow('Camera', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체와 창 해제
cap.release()
cv2.destroyAllWindows()
