"""Constants shared between pipeline scripts and their Handler DATs.

Values used on both sides of the device boundary live here so the two can never
drift apart — e.g. the hand pipeline bakes HAND_CROP_PADDING into its on-device
crop Script while hand_handler.py uses the same value to back-map landmarks.
"""

# Normalized padding added around each detected palm rect before cropping.
# hand_pipeline.py grows the crop by it; hand_handler.py undoes it in the back-map.
HAND_CROP_PADDING: float = 0.1
