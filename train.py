from ultralytics import YOLO

if __name__ == '__main__':
    # Load the pre-trained YOLOv8 model 
    model = YOLO('yolov8n.pt')
    
    print("Starting training")
    
    # Training
    results = model.train(
        data='data.yaml',   
        epochs=20,          
        imgsz=640,          
        device='cpu',       
        batch=4,            
        workers=2           
    )
    
    print("Training complete!")