from ultralytics import YOLO

# carrega modelo de pose (pré-treinado no COCO)
model = YOLO("yolov8n-pose.pt")

# roda em vídeo (gera preview e salva resultado)
# results = model.predict(
#     source="luta.mp4",  # caminho do vídeo
#     conf=0.5,           # confiança mínima
#     save=True           # salva o vídeo com esqueletos desenhados
# )

# for r in results:
#     for kp in r.keypoints.xy:
#         print(kp.shape)  # (17, 2) → 17 pontos (x,y)

model.track(
    source="luta.mp4",       # seu vídeo
    tracker="bytetrack.yaml",
    conf=0.5,
    show=True,               # mostra a janela com IDs e esqueletos
    save=True,               # salva o vídeo anotado em runs/pose/track/...
    stream=False
)
