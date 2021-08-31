import torch

from lucent.optvis import render
from lucent.modelzoo import inceptionv1
import lucent
import torchvision.models as models
resnet18 = models.resnet18()
resnet50 = models.resnet50()

class RedirectedReLU(torch.autograd.Function):
    """
    A workaround when there is no gradient flow from an initial random input
    See https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py
    Note: this means that the gradient is technically "wrong"
    TODO: the original Lucid library has a more sophisticated way of doing this
    """
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_tensor < 0] = grad_input[input_tensor < 0] * 1e-1
        return grad_input


class RedirectedReluLayer(nn.Module):
    def forward(self, tensor):
        return RedirectedReLU.apply(tensor)

def redirect_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, RedirectedReluLayer())
        else:
            redirect_relu(child)
  

resnet50.to(device).eval()  
resnet50 = redirect_relu(resnet50)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print([n for n, x in resnet50.named_modules()])
# print(lucent.modelzoo.util.get_model_layers(resnet50))
render.render_vis(resnet50, "layer4.2.conv2:476")