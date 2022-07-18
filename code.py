import dlib, cv2,os
import matplotlib.pyplot as plt
import numpy as np
from imutils.face_utils import FaceAligner

pose_predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(pose_predictor)
face_encoder=dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()
modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

faces=[]
name=[]
trainpath = "lfw_selected/face"

for im in os.listdir(trainpath):
    print(im)
    img = cv2.imread(os.path.join(trainpath,im))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117,   123], False, False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceAligned =fa.align(img, gray,dlib.rectangle(x1,y1,x2,y2))
            landmark = pose_predictor(faceAligned,dlib.rectangle(0,0,faceAligned.shape[0],faceAligned.shape[1]))
            face_descriptor = face_encoder.compute_face_descriptor(faceAligned, landmark, num_jitters=2)
            faces.append(face_descriptor)
            name.append(im)

        faces = np.array(faces)
name = np.array(name)
np.save('face_repr.npy', faces)
np.save('labels.npy', name)

faces = np.load("face_repr.npy")
name = np.load("labels.npy")
image = cv2.imread("lfw_selected/face2/Johnny_Depp_0002.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

frameHeight = image.shape[0]
frameWidth = image.shape[1]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
net.setInput(blob)
detections = net.forward()
scores=[]
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.7:
        x1 = int(detections[0, 0, i, 3] * frameWidth)
        y1 = int(detections[0, 0, i, 4] * frameHeight)
        x2 = int(detections[0, 0, i, 5] * frameWidth)
        y2 = int(detections[0, 0, i, 6] * frameHeight)
        faceAligned = fa.align(image, gray,dlib.rectangle(x1,y1,x2,y2))
        landmark = pose_predictor(faceAligned,dlib.rectangle(0,0,faceAligned.shape[0],faceAligned.shape[1]))
        face_descriptor = face_encoder.compute_face_descriptor(faceAligned, landmark, num_jitters=2)
        score = np.linalg.norm(faces - np.array(face_descriptor), axis=1)
        scores.append(score)
        imatches = np.argsort(score)
        score = score[imatches]
        print(name[imatches][:10].tolist(), score[:10].tolist())