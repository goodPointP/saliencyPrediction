"""
To run:
    create instance of class based on a specific model:
    
        model = heatmap_inference('path_to_model')
        
            e.g: model = heatmap_inference("models/newnet_model")
    run inference on a single instance:
        
        heatmap = model.inline_inference('path_to_image.jpg', sigma)
        (sigma should be between 0 and 1)
        
            e.g: heatmap = model.inline_inference('testPictures/car.jpg', 0.5)
        
"""
import torch
from PIL import Image
from torchvision import transforms

class heatmap_inference():
    def __init__(self, model):
        
        transformer = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.model = torch.load(model, map_location=torch.device('cpu'))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transformer
    
    def inline_inference(self, imgpath, sigma):
        image = Image.open(imgpath)
        image_transformed = self.transform(image).to(self.device)
        image_unsqueezed = image_transformed.unsqueeze(0)
        heatmap = self.model(image_unsqueezed)
        heatmap_sigmoid = torch.sigmoid(heatmap)
        heatmap_sigma = (heatmap_sigmoid > sigma)
        return heatmap_sigma[0][0].cpu().numpy()
    