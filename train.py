from ultralytics import YOLO

#
if __name__ == '__main__':
    model = YOLO("./runs/segment/train/weights/last.pt")
    model.train(
                batch=8,
                device=0,
                data="./datasets/data.yaml",
                optimizer = 'Adam',
                epochs=100,
                imgsz=640,
                workers = 1
            )