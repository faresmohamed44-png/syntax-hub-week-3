import cv2

# تحميل نموذج كشف الوجوه المدمج في OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

print('[INFO] Press "q" to exit...')

while True:
  ret, frame = cap.read()
  if not ret:
    break

  # تحويل الصورة لأبيض وأسود للسرعة
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # كشف الوجوه
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
  )

  # رسم مربع حول كل وجه وكتابة الاسم
  for x, y, w, h in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        'Detected Face',
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

  # عرض الشاشة
  cv2.imshow('Face Detection & Recognition Project', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()