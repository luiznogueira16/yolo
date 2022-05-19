import cv2
import time

from matplotlib.colors import cnames

# Cores das classes
colors = [(0,255,255), (255,255,0), (0,255,0), (255,0,0)]

# Carrega as classes
class_names = []
with open("./data/coco.names", "r") as f:
    class_names = [cnames.strip() for cnames in f.readlines()]

# Captura do video
cap = cv2.VideoCapture("./data/pexels.mp4")

# Carregando os pesos da rede neural
net = cv2.dnn.readNet("./cfg/yolov4-tiny.weights", "./cfg/yolov4-tiny.cfg")

# Setando os parametros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)

# Lendo os frames do video
while True:
    
    # Captura do Frame
    _, frame = cap.read()
    
    # Começo da contagem dos ms
    start = time.time()
    
    # Deteccao
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    
    # Fim da contação dos ms
    end = time.time()
    
    # Pecorrer cada geração
    for (classid, score, box) in zip(classes, scores, boxes):
        
        # Gerando uma cor para cada classe
        color = colors[int(classid) % len(colors)]
        
        # Pegando o nome da classe pelo id e o seu score de acuracia
        label = f"{class_names[classid[0]]} : {score}"
        
        # Desenhando a box da deteccao
        cv2.rectangle(frame, box, color, 2)
        
        # Escrevendo o nome da classe em cima da box do objeto
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    # Calculando o tempo para fazer a deteccao
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"
    
    # Escrevendo o fps da imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    
    # Mostrando a imagem
    cv2.imshow("detections", frame)
    
    # Espera da resposta
    if cv2.waitKey(1) == 27:
        break

# Libertação da camera e destroi todas as janelas
cap.release()
cv2.destroyAllWindows()