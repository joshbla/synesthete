import torchcodec
import inspect

print("torchcodec version:", torchcodec.__version__)
print("\nDir(torchcodec):")
print(dir(torchcodec))

try:
    from torchcodec import decoders
    print("\nDir(torchcodec.decoders):")
    print(dir(decoders))
    
    if hasattr(decoders, 'VideoDecoder'):
        print("\nVideoDecoder methods:")
        print(dir(decoders.VideoDecoder))
except ImportError:
    print("Could not import decoders")

try:
    from torchcodec import encoders
    print("\nDir(torchcodec.encoders):")
    print(dir(encoders))
except ImportError:
    print("Could not import encoders")
