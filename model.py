import timm
import torch
import torch.nn as nn

class VegetableModel(nn.Module):
    def __init__(self, num_classes):
        super(VegetableModel, self).__init__()

        # EfficientNet pretrained
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        network_out_size = 1280  # EfficientNet B0 last layer dimension

       
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(network_out_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

    
 
    
    

