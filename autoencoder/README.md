# Data Processing Pipeline

## 1. Convert Videos/ImageFolders to Events
```cd ~/projects/IEBCS/examples/00_video_2_events```
```python3 images_to_events.py```

## 2. Convert Events to Event Frames
```cd ~/projects/npc-av-learning2025/utils```
```python3 event2frame.py --input_dir /home/matt/projects/IEBCS/examples/00_video_2_events/outputs --output_dir /media/matt/bigdata/DATA/CRIB/train_event_frames```

## 3. Extract Patches for the Autoencoder
```python extract_patches.py --M 10 --N 100 --test_split 0.2 --crop 80```

## 4. Train the Autoencoder
