#!/usr/bin/env python3
"""
Test script to demonstrate dense keypoint visualization.
"""

import numpy as np
import cv2
from tools.vis import vis_dense_kp, vis_img_with_dense_kp, start_gradio
import os

def create_test_image():
    """Create a simple test image."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:, :] = (100, 150, 200)  # Light blue background
    # Draw a simple human silhouette
    cv2.ellipse(img, (200, 120), (40, 60), 0, 0, 360, (50, 50, 50), -1)  # head
    cv2.rectangle(img, (180, 180), (220, 320), (50, 50, 50), -1)  # torso
    cv2.rectangle(img, (190, 320), (210, 380), (50, 50, 50), -1)  # legs
    return img

def create_test_dense_kp():
    """Create test dense keypoints that mimic SMPL's 138 keypoints."""
    # Create 138 keypoints with realistic positions and confidences
    dense_kp = np.zeros((138, 3), dtype=np.float32)
    
    # SMPL body joints (first 24)
    body_joints = [
        [200, 120], [200, 140], [180, 160], [220, 160],  # head, neck, shoulders
        [160, 180], [240, 180], [140, 220], [260, 220],  # arms
        [180, 200], [220, 200], [200, 240],              # hips, spine
        [190, 280], [210, 280], [190, 320], [210, 320],  # legs
        [190, 360], [210, 360], [190, 380], [210, 380],  # feet
        [195, 110], [205, 110], [185, 115], [215, 115], [200, 100] # face points
    ]
    
    # Fill in body joints
    for i, (x, y) in enumerate(body_joints):
        if i < len(dense_kp):
            dense_kp[i] = [x, y, 0.8 + 0.2 * np.random.random()]  # high confidence
    
    # Fill remaining keypoints with lower confidence points around the body
    for i in range(len(body_joints), 138):
        # Random points around the torso and limbs with lower confidence
        x = 150 + 100 * np.random.random()
        y = 100 + 250 * np.random.random() 
        conf = 0.3 + 0.4 * np.random.random()
        dense_kp[i] = [x, y, conf]
    
    return dense_kp

def test_visualization():
    """Test the dense keypoint visualization functions."""
    print("Creating test image and keypoints...")
    img = create_test_image()
    dense_kp = create_test_dense_kp()
    
    print(f"Dense keypoints shape: {dense_kp.shape}")
    print(f"Keypoint confidence range: {dense_kp[:, 2].min():.3f} - {dense_kp[:, 2].max():.3f}")
    
    # Test the visualization function
    img_with_kp = vis_dense_kp(img, dense_kp, conf_threshold=0.3, point_size=3)
    
    # Save test images
    cv2.imwrite('/tmp/test_original.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/tmp/test_with_dense_kp.png', cv2.cvtColor(img_with_kp, cv2.COLOR_RGB2BGR))
    print("Saved test images to /tmp/test_original.png and /tmp/test_with_dense_kp.png")
    
    # Test with Gradio if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--gradio':
        print("Starting Gradio interface...")
        start_gradio(host='0.0.0.0', port=7860)
        
        # Send images to Gradio
        print("Sending original image to Gradio...")
        from tools.vis import vis_img
        vis_img(img)
        
        import time
        time.sleep(2)
        
        print("Sending image with dense keypoints to Gradio...")
        vis_img_with_dense_kp(img, dense_kp, conf_threshold=0.3, point_size=3)
        
        print("Check http://192.168.0.241:7860 to see the visualization")
        print("Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")

if __name__ == "__main__":
    test_visualization()