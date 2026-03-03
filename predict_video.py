#!/usr/bin/env python
import sys
import torch
import numpy as np
import os
import glob
import argparse
import json
import cv2
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple

sys.path.append('../physical_envs/OpenSTL')

from openstl import models


class VideoPreprocessor:
    """Handle video loading, preprocessing, and caching."""
    
    def __init__(self, target_size=(64, 64), cache_dir='video_cache'):
        self.target_size = target_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, video_path: str) -> Path:
        """Get cache file path for a video."""
        video_name = Path(video_path).stem
        return self.cache_dir / f"{video_name}_frames.npy"
    
    def get_metadata_path(self, video_path: str) -> Path:
        """Get metadata file path for a video."""
        video_name = Path(video_path).stem
        return self.cache_dir / f"{video_name}_metadata.json"
    
    def preprocess_video(self, video_path: str, fps: Optional[int] = None, 
                        grayscale: bool = True, force_reprocess: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Load and preprocess video into frames.
        
        Args:
            video_path: Path to .mp4 file
            fps: Target FPS (None = keep original)
            grayscale: Convert to grayscale
            force_reprocess: Force reprocessing even if cached
            
        Returns:
            frames (T, C, H, W): Preprocessed frames
            metadata: Dictionary with processing info
        """
        cache_path = self.get_cache_path(video_path)
        metadata_path = self.get_metadata_path(video_path)
        
        # Check cache
        if cache_path.exists() and metadata_path.exists() and not force_reprocess:
            print(f"📦 Loading cached frames from: {cache_path}")
            frames = np.load(cache_path)
            with open(metadata_path) as f:
                metadata = json.load(f)
            return frames, metadata
        
        print(f"📹 Loading video: {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        print(f"   Resolution: {original_w}x{original_h}, FPS: {original_fps:.1f}, Frames: {total_frames}")
        
        # Determine frame sampling
        frame_step = 1
        if fps is not None and fps < original_fps:
            frame_step = int(original_fps / fps)
        
        frames_list = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_step == 0:
                # Resize with aspect ratio preservation
                frame = self._resize_with_aspect_ratio(frame, self.target_size)
                
                # Convert to grayscale if needed
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame, axis=0)  # (1, H, W)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = np.transpose(frame, (2, 0, 1))  # (3, H, W)
                
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames_list.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        frames = np.array(frames_list)  # (T, C, H, W)
        print(f"   Extracted: {len(frames_list)} frames after resampling")
        
        # Save cache
        np.save(cache_path, frames)
        
        metadata = {
            'video_path': video_path,
            'original_resolution': (original_w, original_h),
            'target_resolution': self.target_size,
            'original_fps': original_fps,
            'target_fps': fps or original_fps,
            'total_frames_original': total_frames,
            'total_frames_extracted': len(frames_list),
            'frame_step': frame_step,
            'grayscale': grayscale,
            'cached_at': datetime.now().isoformat(),
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Cached to: {cache_path}")
        return frames, metadata
    
    @staticmethod
    def _resize_with_aspect_ratio(frame: np.ndarray, target_size: Tuple[int, int], 
                                  pad_value: int = 0) -> np.ndarray:
        """
        Resize frame preserving aspect ratio with padding.
        
        Args:
            frame: Input frame (H, W, C)
            target_size: Target (H, W)
            pad_value: Padding color value
            
        Returns:
            Resized frame (target_H, target_W, C)
        """
        h, w = frame.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scale to fit within target
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h_top = (target_h - new_h) // 2
        pad_h_bot = target_h - new_h - pad_h_top
        pad_w_left = (target_w - new_w) // 2
        pad_w_right = target_w - new_w - pad_w_left
        
        padded = cv2.copyMakeBorder(resized, pad_h_top, pad_h_bot, pad_w_left, pad_w_right,
                                    cv2.BORDER_CONSTANT, value=pad_value)
        return padded


class SlidingWindowPredictor:
    """Handle sliding window predictions and frame-level feature extraction."""
    
    def __init__(self, model, window_size: int = 10, stride: int = 1):
        """
        Args:
            model: Trained SimVP model
            window_size: Number of input frames (usually 10)
            stride: Stride between windows (1 = all frames get predictions)
        """
        self.model = model
        self.window_size = window_size
        self.stride = stride
    
    def predict_sliding_window(self, frames: np.ndarray) -> dict:
        """
        Apply sliding window predictions.
        
        Args:
            frames: (T, C, H, W) preprocessed frames
            
        Returns:
            dict with predictions and frame-level features
        """
        T = frames.shape[0]
        if T < self.window_size:
            raise ValueError(f"Video has {T} frames, need at least {self.window_size}")
        
        predictions = []
        frame_features = {}  # Frame index -> prediction feature
        
        num_windows = (T - self.window_size) // self.stride + 1
        print(f"🪟 Processing {num_windows} sliding windows (size={self.window_size}, stride={self.stride})")
        
        for i, start_idx in enumerate(range(0, T - self.window_size + 1, self.stride)):
            window_frames = frames[start_idx : start_idx + self.window_size]  # (10, C, H, W)
            
            # Add batch dimension
            window_batch = np.expand_dims(window_frames, axis=0)  # (1, 10, C, H, W)
            input_tensor = torch.from_numpy(window_batch).cuda().float()
            
            with torch.no_grad():
                pred_frames = self.model(input_tensor)  # (1, T_pred, C, H, W)
            
            pred_np = pred_frames[0].cpu().numpy()  # (T_pred, C, H, W)
            predictions.append(pred_np)
            
            # Register predictions for frames covered by this window
            for j in range(start_idx, start_idx + self.window_size):
                if j not in frame_features:
                    frame_features[j] = []
                frame_features[j].append({
                    'window_idx': i,
                    'position_in_window': j - start_idx
                })
            
            if (i + 1) % max(1, num_windows // 10) == 0:
                print(f"   {i+1}/{num_windows} windows processed")
        
        return {
            'predictions': predictions,
            'frame_features': frame_features,
            'num_windows': num_windows,
            'total_frames': T,
        }


def predict_video(input_path: str, output_dir: str = 'results', run_name: Optional[str] = None,
                 window_size: int = 10, stride: int = 1, fps: Optional[int] = None,
                 force_preprocess: bool = False) -> dict:
    """
    Predict video frames using SimVP model with sliding window.
    
    Args:
        input_path: Path to video (.mp4) or frames (.npy)
        output_dir: Base output directory
        run_name: Custom run name
        window_size: Input frames for model
        stride: Sliding window stride
        fps: Target FPS for video (None = keep original)
        force_preprocess: Force reprocessing video
    """
    from openstl.models import SimVP_Model
    import importlib.util
    
    # Create output directory
    if run_name is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"prediction_{timestamp}"
    
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_path}\n")
    
    # Load or preprocess frames
    if input_path.endswith('.mp4'):
        preprocessor = VideoPreprocessor(target_size=(64, 64))
        frames, video_metadata = preprocessor.preprocess_video(
            input_path, fps=fps, grayscale=True, force_reprocess=force_preprocess
        )
        input_shape = [window_size] + list(frames.shape[1:])  # [10, 1, 64, 64]
    else:  # .npy file
        print(f"📂 Loading frames from: {input_path}")
        frames = np.load(input_path)
        input_shape = list(frames.shape[1:])
        video_metadata = None
    
    print(f"   Loaded shape: {frames.shape}\n")
    
    # Load model config
    config_path = '../physical_envs/OpenSTL/configs/mmnist/simvp/SimVP_gSTA.py'
    spec = importlib.util.spec_from_file_location("simvp_config", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    model_cfg = {k: v for k, v in cfg_module.__dict__.items() if not k.startswith('_')}
    
    # Create model
    model = SimVP_Model(input_shape, **model_cfg).cuda().eval()
    print("✅ SimVP_gSTA model loaded")
    
    # Load checkpoint
    ckpt_dir = '../physical_envs/OpenSTL/work_dirs/mmnist_simvp_gsta_pretrained/checkpoints'
    ckpts = glob.glob(f'{ckpt_dir}/best*.ckpt') or glob.glob(f'{ckpt_dir}/last*.ckpt')
    
    if ckpts:
        latest = max(ckpts, key=os.path.getctime)
        checkpoint = torch.load(latest, map_location='cuda')
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Loaded checkpoint: {os.path.basename(latest)}\n")
    else:
        raise FileNotFoundError("No checkpoints found")
    
    # Run sliding window predictions
    print("🔄 Running sliding window predictions...")
    predictor = SlidingWindowPredictor(model, window_size=window_size, stride=stride)
    
    results = predictor.predict_sliding_window(frames)
    
    print(f"✅ Predictions complete!\n")
    
    # Save outputs
    filename_base = Path(input_path).stem
    
    # Save all predictions as NPY
    predictions_npy = output_path / f"{filename_base}_predictions.npy"
    all_predictions = np.concatenate(results['predictions'], axis=0)
    np.save(predictions_npy, all_predictions)
    print(f"💾 All predictions NPY: {predictions_npy}")
    
    # Save first window as GIF for preview
    gif_path = output_path / f"{filename_base}_preview.gif"
    first_pred = results['predictions'][0][0]
    imgs = [Image.fromarray((f*255).clip(0,255).astype(np.uint8)) for f in first_pred]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=150, loop=0)
    print(f"🎬 Preview GIF (first window): {gif_path}")
    
    # Save frame-level metadata
    frame_metadata = {
        'input_file': input_path,
        'input_shape': list(frames.shape),
        'window_size': window_size,
        'stride': stride,
        'total_frames': results['total_frames'],
        'num_windows': results['num_windows'],
        'frame_coverage': dict(results['frame_features']),
        'video_preprocessing': video_metadata,
    }
    
    metadata_path = output_path / 'results_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(frame_metadata, f, indent=2)
    print(f"📋 Metadata: {metadata_path}")
    
    print(f"\n✨ All outputs saved to: {output_path}\n")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video frame prediction with sliding window')
    parser.add_argument('input', type=str, help='Path to video (.mp4) or frames (.npy)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom run name')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Input frame window size (default: 10)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Sliding window stride (default: 1)')
    parser.add_argument('--fps', type=int, default=None,
                        help='Target FPS for video (default: original)')
    parser.add_argument('--force-preprocess', action='store_true',
                        help='Force reprocessing even if cached')
    
    args = parser.parse_args()
    
    predict_video(
        input_path=args.input,
        output_dir=args.output_dir,
        run_name=args.name,
        window_size=args.window_size,
        stride=args.stride,
        fps=args.fps,
        force_preprocess=args.force_preprocess,
    )
