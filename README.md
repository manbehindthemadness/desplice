# Desplice

Desplice is a Python library designed to process video files by removing duplicate frames, still images, and certain transitions. This helps in deduplicating video content, making it more efficient for storage or further processing.

## Features

- **Heal**: Drops duplicate frames from the output.
- **Keep**: Retains one instance of each duplicate frame in the output.
- **Explode**: Removes duplicate frames from the video output and returns separate clips for each splice.

## Installation

To install Desplice, you can use pip:

```bash
pip install desplice
```

## Usage

Below are some examples demonstrating how to use Desplice.

### Importing the Library

```python
import cv2
import numpy as np
from antidupe import Antidupe
from desplice import Desplice
from PIL import Image
```

### Loading a Video and Processing it

```python
# Initialize the Desplice class
desplice = Desplice(debug=True)

# Load a video file into memory
video_frames = desplice.load_video_to_memory('path_to_your_video.mp4')

# Process the video to deduplicate frames
result, is_slideshow = desplice.process(video_frames, mode='heal', show=True, show_breaks=True)

# Show the processed video
desplice.show_video_from_frames(result[0][0])
```

### Example of Different Modes:

#### Heal Mode

Removes the detected content from output entirely.

```python
result, is_slideshow = desplice.process('path_to_your_video.mp4', mode='heal')
```

#### Keep Mode

Keeps 1 frame of the duplicated content in the output.

```python
result, is_slideshow = desplice.process('path_to_your_video.mp4', mode='keep')
```

#### Explode Mode

Returns each segment inbetween each duplicated sequence as it's own sequence.

```python
result, is_slideshow = desplice.process('path_to_your_video.mp4', mode='explode')
```

Results are returned in a nested tuple object:

```
(
    (
        [video-1, video-2, video-3...],  # List of videos: list[list[np.ndarray]]
        [still-1, still-2, still-3]  # List of still images: list[np.ndarray]
    ), 
    slideshow  # Is the video a slideshow: Bool
)
```

Deduplication weights can be passed to augment the behavior of the ```antidupe``` module:

```python
custom_thresholds = {
        'ih': 0.0,  # Image Hash
        'ssim': 0.1,  # SSIM
        'cs': 0.02,  # Cosine Similarity
        'cnn': 0.03,  # CNN
        'dedup': 0.005  # Mobilenet
    }

desplice = Desplice(thresholds=custom_thresholds)
```

Further information on using ```antidupe``` can be found [here](https://github.com/manbehindthemadness/antidupe)

## Class and Method Descriptions

### `Desplice`

#### Methods

- **`__init__(self, thresholds: dict = None, device: str = 'cpu', debug: bool = False)`**:
  Initializes the Desplice class with optional threshold values, device, and debug mode.

- **`load_video_to_memory(file_path: str) -> list`**:
  Loads a video from the given file path into memory.

- **`show_video_from_frames(frames: list)`**:
  Displays a video from a list of frames.

- **`deduplicate_frames(self, frames: list) -> tuple`**:
  Deduplicates frames from the input video and returns a tuple containing processed video frames and images.

- **`process(self, file_path_or_array: [str, list[np.ndarray]], mode: modes = 'heal', show: bool = False, show_breaks: bool = False) -> tuple`**:
  Processes the video based on the specified mode and returns the processed result and a boolean indicating if the output is a slideshow.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.