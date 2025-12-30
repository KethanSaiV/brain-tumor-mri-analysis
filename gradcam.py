import torch
import cv2
import numpy as np

def gradcam(model, img):
    model.eval()
    img.requires_grad = True
    out = model(img)
    cls = out.argmax()
    out[0, cls].backward()

    gradients = img.grad[0].numpy()
    heatmap = np.mean(np.abs(gradients), axis=0)
    heatmap = heatmap / heatmap.max()
    return heatmap
