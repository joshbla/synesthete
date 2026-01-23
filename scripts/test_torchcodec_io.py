import torch
import torchcodec
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder
import os

def test_torchcodec():
    print("Testing torchcodec...")
    
    # Create a dummy video tensor (N, C, H, W)
    # torchcodec expects (N, C, H, W)
    T, C, H, W = 30, 3, 64, 64
    video_data = torch.randint(0, 255, (T, C, H, W), dtype=torch.uint8)
    
    output_path = "test_torchcodec.mp4"
    
    # Test Writing
    print("Testing VideoEncoder...")
    try:
        encoder = VideoEncoder(video_data, frame_rate=30)
        encoder.to_file(output_path)
        print(f"Successfully wrote {output_path}")
    except Exception as e:
        print(f"Failed to write video: {e}")
        return

    # Test Reading
    print("Testing VideoDecoder...")
    try:
        decoder = VideoDecoder(output_path)
        print(f"Metadata: {decoder.metadata}")
        
        # Read all frames
        frames = decoder[:]
        print(f"Read frames shape: {frames.shape}")
        
        # torchcodec likely returns (N, C, H, W)
        if frames.shape == (T, C, H, W):
            print("Shape matches (N, C, H, W)!")
        elif frames.shape == (T, H, W, C):
            print("Shape matches (N, H, W, C)!")
        else:
            print(f"Shape mismatch! Expected {(T, C, H, W)} or {(T, H, W, C)}")
            
    except Exception as e:
        print(f"Failed to read video: {e}")
        
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)

if __name__ == "__main__":
    test_torchcodec()
