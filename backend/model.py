# # backend/model.py
# import torch
# from torchvision import transforms, models

# def load_model(model_path: str, device: torch.device):
#     # Create the exact architecture you trained
#     # You stored class_names and model_state_dict in the checkpoint
#     checkpoint = torch.load(model_path, map_location=device)
#     class_names = checkpoint["class_names"]

#     model = models.efficientnet_b0(weights=None)
#     model.classifier[1] = torch.nn.Linear(
#         model.classifier[1].in_features, len(class_names)
#     )
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device)
#     model.eval()

#     return model, class_names

# def get_transform():
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])



# backend/model.py
import torch
from torchvision import transforms, models

def load_model(model_path: str, device: torch.device):
    """
    Loads the trained EfficientNet-B0 model and its class names
    from a checkpoint that contains:
      - checkpoint["class_names"] (list[str])
      - checkpoint["model_state_dict"] (state_dict)
    """
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names

def get_transform():
    """
    Image preprocessing matching training:
      - Resize to 224x224
      - ToTensor()
      - Normalize with ImageNet stats
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])