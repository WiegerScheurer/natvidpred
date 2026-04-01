import cv2
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoVideoProcessor

# Configuration
HF_MODEL_NAME = "facebook/vjepa2-vitg-fpc64-384"
VIDEO_PATH = Path("/project/3018078.02/MEG_ingmar/shorts/iglo.mp4")
TUBELET_SIZE = 2

def debug_shapes():
    """Debug tensor shapes at each step"""
    
    print("=" * 60)
    print("V-JEPA2 Pipeline Debug")
    print("=" * 60)
    
    # Load model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model... (device: {device})")
    proc = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    mdl = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()
    print("✓ Model loaded")
    
    # Load a few frames
    print(f"\nLoading video: {VIDEO_PATH.name}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    raw_frames = []
    
    frame_count = 0
    while frame_count < 10:  # Just load 10 frames for debugging
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB)
        raw_frames.append(frame_rgb)
        frame_count += 1
    cap.release()
    
    print(f"✓ Loaded {len(raw_frames)} frames")
    print(f"  Frame shape: {raw_frames[0].shape}")
    
    # Test with a single tubelet (2 frames)
    print("\n" + "=" * 60)
    print("Testing Single Tubelet Processing")
    print("=" * 60)
    
    tubelet_frames = raw_frames[:TUBELET_SIZE]
    
    # Step 1: Convert to torch tensors and permute
    print("\nStep 1: Convert frames to torch and permute")
    frame_tensors = [torch.from_numpy(f).permute(2, 0, 1) for f in tubelet_frames]
    print(f"  After permute (per frame): {frame_tensors[0].shape}")
    print(f"  dtype: {frame_tensors[0].dtype}")
    
    # Step 2: Stack to create video tensor
    print("\nStep 2: Stack frames")
    video = torch.stack(frame_tensors)
    print(f"  Video shape: {video.shape}")
    print(f"  dtype: {video.dtype}")
    
    # Step 3: Add batch dimension
    print("\nStep 3: Add batch dimension (unsqueeze)")
    video_batched = video.unsqueeze(0)
    print(f"  Video batched shape: {video_batched.shape}")
    
    # Step 4: Pass through processor
    print("\nStep 4: Process through VideoProcessor")
    print(f"  Processor config: {proc.__class__.__name__}")
    try:
        inputs = proc(video_batched, return_tensors="pt")
        print(f"  ✓ Processor successful")
        print(f"  Output keys: {inputs.keys()}")
        for key in inputs.keys():
            print(f"    {key}: {inputs[key].shape}")
            print(f"      dtype: {inputs[key].dtype}")
    except Exception as e:
        print(f"  ✗ Processor failed: {e}")
        return
    
    # Step 5: Move to device and get features
    print("\nStep 5: Get vision features from model")
    try:
        pixel_values = inputs["pixel_values_videos"].to(device)
        print(f"  pixel_values_videos shape: {pixel_values.shape}")
        print(f"  Device: {pixel_values.device}")
        
        # Try getting features
        with torch.inference_mode():
            feats = mdl.get_vision_features(pixel_values)
        
        print(f"  ✓ Feature extraction successful")
        print(f"  Raw features shape: {feats.shape}")
        print(f"  dtype: {feats.dtype}")
        
        # Step 6: Remove batch dimension
        print("\nStep 6: Squeeze batch dimension")
        feats_squeezed = feats.squeeze(0)
        print(f"  Features squeezed shape: {feats_squeezed.shape}")
        
    except Exception as e:
        print(f"  ✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Testing Multiple Tubelets (as in actual processing)")
    print("=" * 60)
    
    try:
        all_feats = []
        num_tubelets = len(raw_frames) // TUBELET_SIZE
        
        for i in range(num_tubelets):
            tubelet = raw_frames[i * TUBELET_SIZE : (i + 1) * TUBELET_SIZE]
            
            # Convert and process
            video = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in tubelet])
            inputs = proc(video.unsqueeze(0), return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].to(device)
            
            with torch.inference_mode():
                feats = mdl.get_vision_features(pixel_values).cpu().squeeze(0)
            
            all_feats.append(feats)
            print(f"  Tubelet {i}: {feats.shape}")
        
        # Find min patches
        min_patches = min([f.shape[0] for f in all_feats])
        print(f"\n  Min patches across all tubelets: {min_patches}")
        
        # Truncate and stack
        all_feats = [f[:min_patches, :] for f in all_feats]
        stacked = torch.stack(all_feats)
        
        print(f"  Final stacked shape: {stacked.shape}")
        print(f"  Interpretation: [num_tubelets={stacked.shape[0]}, num_patches={stacked.shape[1]}, embedding_dim={stacked.shape[2]}]")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✓ All checks passed!")
    print("=" * 60)

if __name__ == "__main__":
    debug_shapes()
