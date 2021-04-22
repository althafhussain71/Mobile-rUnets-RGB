import coremltools

# referred from: https://cutecoder.org/featured/core-ml-image-output/ from line# 4 to 78
coreml_model = coremltools.models.MLModel('FilterModel.mlmodel')
spec = coreml_model.get_spec()
spec_layers = getattr(spec,spec.WhichOneof("Type")).layers

# find the current output layer and save it for later reference
last_layer = spec_layers[-1]

# add the post-processing layer
new_layer = spec_layers.add()
new_layer.name = 'convert_to_image'

# Configure it as an activation layer
new_layer.activation.linear.alpha = 255
new_layer.activation.linear.beta = 0

# Use the original model's output as input to this layer
new_layer.input.append(last_layer.output[0])

# Name the output for later reference when saving the model
new_layer.output.append('image_output')

# Find the original model's output description
output_description = next(x for x in spec.description.output if x.name == last_layer.output[0])

# Update it to use the new layer as output
output_description.name = new_layer.name


# Function to mark the layer as output
# https://forums.developer.apple.com/thread/81571#241998
def convert_multiarray_output_to_image(spec, feature_name, is_bgr=False):
    """
    Convert an output multiarray to be represented as an image
    This will modify the Model_pb spec passed in.
    Example:
        model = coremltools.models.MLModel('MyNeuralNetwork.mlmodel')
        spec = model.get_spec()
        convert_multiarray_output_to_image(spec,'imageOutput',is_bgr=False)
        newModel = coremltools.models.MLModel(spec)
        newModel.save('MyNeuralNetworkWithImageOutput.mlmodel')
    Parameters
    ----------
    spec: Model_pb
        The specification containing the output feature to convert
    feature_name: str
        The name of the multiarray output feature you want to convert
    is_bgr: boolean
        If multiarray has 3 channels, set to True for RGB pixel order or false for BGR
    """
    for output in spec.description.output:
        if output.name != feature_name:
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType':
            raise ValueError("%s is not a multiarray type" % output.name)
        array_shape = tuple(output.type.multiArrayType.shape)
        channels, height, width = array_shape
        from coremltools.proto import FeatureTypes_pb2 as ft
        if channels == 1:
            output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
        elif channels == 3:
            if is_bgr:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('BGR')
            else:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB')
        else:
            raise ValueError("Channel Value %d not supported for image inputs" % channels)
        output.type.imageType.width = width
        output.type.imageType.height = height

    # Mark the new layer as image

convert_multiarray_output_to_image(spec, output_description.name, is_bgr=False)
model_name = 'BlurFilterModel.mlmodel'
model_name.save(model_name)