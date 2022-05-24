#gerekli kütüphaneleri import ediyoruz
import cv2

#haarcascade xml dosya yollarını belirtiyoruz
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#kamerayı başlatıyoruz
videoCapture = cv2.VideoCapture(0)

#kameradan sürekli görüntüler geleceği için sonsuz döngü açıyoruz
while True:
    #kameradan gelen görüntü karelerini frame de tutuypruz
    ret, frame = videoCapture.read()

    #frame bilgisini işleyebilmek için gri hale getiriyoruz
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Daha önce dahil ettiğimiz face xml dosyasını detectMultiScale yardımıyla kullnarak karelerdeki yüzleri buluyoruz
    faces = faceCascade.detectMultiScale(gray,1.3,3)

    for (x, y, w, h) in faces:
        #Bulduğumuz yüzleri kare içerisine alıyoruz
        cv2.rectangle(frame, (x, y), (x+w, y+h+60), (0, 255, 0), 1)
        
        roi = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        #yüz bulma işleminin aynısını yüz sınırlarını içerisinde gözler için uyguluyoruz
        eyes = eyeCascade.detectMultiScale(roi, 1.3, 5,maxSize=(30,30))

        for (x1, y1, w1, h1) in eyes:
            #gözlerin açık-kapalı durumunu inceleyerek ekrana yazdırıyoruz
            if(len(eyes)==2):
                cv2.putText(frame, "Goz Acik", (70, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Goz Kapali", (70, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    #işlenmiş görüntüyü gösteriyoruz
    cv2.imshow('Video', frame)
    #q ya basılınca çıkması için
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Donanım kaynağını serbest bırakıp pencereyi kapatıyoruz
videoCapture.release()
cv2.destroyAllWindows()