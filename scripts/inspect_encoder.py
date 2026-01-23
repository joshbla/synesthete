import torchcodec
from torchcodec.encoders import VideoEncoder
import inspect

print(inspect.signature(VideoEncoder.__init__))
print(inspect.getdoc(VideoEncoder))
