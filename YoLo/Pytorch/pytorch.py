import torch
import torchvision.transforms as transforms
import cv2
import dlib

# 얼굴 검출기 초기화
detector = dlib.get_frontal_face_detector()

# 눈동자 인식을 위한 모델 경로
pupil_model_path = 'pupil_model.pth'

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 불러오기
pupil_model = torch.load(pupil_model_path)

# 이미지 전처리를 위한 변환기 초기화
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 웹캠 비디오 스트리밍을 위한 초기화
cap = cv2.VideoCapture(0)

# 비디오 스트리밍 시작
while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 얼굴 검출
    faces = detector(frame, 0)
    
    # 얼굴이 검출되지 않았을 경우
    if len(faces) == 0:
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # 얼굴 좌표 추출
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    
    # 얼굴 이미지 추출
    face_image = frame[y:y+h, x:x+w]
    face_image = cv2.cvtColor(cv2.resize(face_image, (32, 32)), cv2.COLOR_BGR2RGB)
    face_tensor = transform(face_image).unsqueeze(0).to(device)
    
    # 눈동자 인식
    pupil_output = pupil_model(face_tensor)
    pupil_output = torch.sigmoid(pupil_output).squeeze().cpu().detach().numpy()
    
    # 눈동자 좌표 계산
    pupil_x, pupil_y = x + int(w * pupil_output[0]), y + int(h * pupil_output[1])
    
    # 결과 시각화
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 얼굴 영역
    cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 0, 255), -1)  # 눈동자
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 스트리밍 종료
cap.release()
cv2.destroyAllWindows()
