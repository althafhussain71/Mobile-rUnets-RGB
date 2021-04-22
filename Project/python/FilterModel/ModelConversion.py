import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

from torchvision import transforms
import coremltools as ct

# image must be converted to a tensor, hence using pytorch's transforms.Compose function
# Compose will combine several transforms into one based on the order
compose = transforms.Compose([
   transforms.ToTensor(),
])


# Extending Pytorch's nn module to implement neural network models (using predefined methods functions etc..)
class FILTER_MODEL(nn.Module):
    # Referred from https://debuggercafe.com/image-deblurring-using-convolutional-neural-networks-and-deep-learning/
    def __init__(self):
        nn.Module.__init__(self)
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=9, padding=2)
        self.conv_2 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
        self.conv_3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)


    # Referred from https://debuggercafe.com/image-deblurring-using-convolutional-neural-networks-and-deep-learning/
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = self.conv_3(x)
        return x


model = FILTER_MODEL()
# Referred from: https://pytorch.org/tutorials/beginner/saving_loading_models.html
model.load_state_dict(torch.load('model.pth'))
model.eval()

# for conversion of Pytorh model to core ml, Referred below tutorials
# Referred from: https://developer.apple.com/videos/play/tech-talks/10154/
# Referred from: https://coremltools.readme.io/docs/pytorch-conversion
ip_image = Image.open('image_114.jpg')
ip_tensor = compose(ip_image)
# Adding batch dimension to the tensor to set the shape as the model expects 4D tensor but the image is a 3D tensor
ip_batch = ip_tensor.unsqueeze(0)
# Tracing will capture all the operations involved during the forward pass and will return a torchscript representation of the model
# Torchscript representation then can be used to convert to a coreml model. For more detailed info follow above mentioned tutorials
trace = torch.jit.trace(model, ip_batch)
# Defining input for Torchscript model as we can specify input to be an image or a multiarray. For more detailed info follow above mentioned tutorial
_input = ct.ImageType(
    name='input_1',
    shape=ip_batch.shape,
    scale = 1./(255*0.226)
)
# Converting to a CoreML model to integrate in iOS app
mlmodel = ct.convert(
    trace,
    inputs=[_input],
)

# Saving the CoreML model
mlmodel.save("FilterModel")



