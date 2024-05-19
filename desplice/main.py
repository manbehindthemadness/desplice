import cv2
import numpy as np
from antidupe import Antidupe
from PIL import Image


modes = [
    'heal',         # Drops duplicate frames from the output.
    'keep',         # Keeps a total of 1 instance of a duplicate frames in the output.
    'explode',      # Drops the duplicate frames from the video output and returns a separate clip for each splice.
]


class Desplice:
    """
    Explode video into usable data components.
    """
    show = False
    show_breaks = False
    mode = 'heal'

    images = list()

    default_thresholds = {
        'ih': 0.0,  # Image Hash
        'ssim': 0.1,  # SSIM
        'cs': 0.02,  # Cosine Similarity
        'cnn': 0.03,  # CNN
        'dedup': 0.005  # Mobilenet
    }

    def __init__(self, thresholds: dict = None, device: str = 'cpu', debug: bool = False):
        self.debug = debug
        self.device = device
        if thresholds is None:
            thresholds = self.default_thresholds
        self.antidupe = Antidupe(limits=thresholds, debug=self.debug, device=self.device)

    @staticmethod
    def load_video_to_memory(file_path: str) -> list:
        """
        Aptly named.
        """
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            raise IOError(f"Error opening video file: {file_path}")
        frames = list()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def show_video_from_frames(frames: list):
        while True:
            for frame in frames:
                cv2.imshow('Video', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def deduplicate_frames(self, frames: list) -> tuple:
        """
        Here we will take sections of video that are duplicate frames and reduce them into a single image.
        If there are mixed segments we will return the video data as clips in the output list.
        """
        video = list()
        images = list()
        first_frame = frames[0]
        filled_frame = np.full(first_frame.shape, 0, dtype=first_frame.dtype)
        single = False
        chunk = list()
        for idx, frame in enumerate(frames):
            if idx + 1 == len(frames):
                break
            next_frame = frames[idx + 1]
            is_duplicate = self.antidupe.predict([Image.fromarray(frame), Image.fromarray(next_frame)])
            if is_duplicate and not single:
                single = True
                if chunk:
                    video.extend(chunk)
                chunk = list()
                images.append(frame)
                if self.mode == 'keep':
                    video.append(frame)
                if self.show_breaks:
                    video.append(filled_frame)
                if self.debug:
                    print('----- Duplicates Start')
            elif not is_duplicate:
                if single:
                    if self.debug:
                        print('----- Duplicates End')
                    single = False
                else:
                    chunk.append(frame)
            if self.show:
                cv2.imshow('pw', frame)
                cv2.waitKey(1)
            if self.debug:
                print(is_duplicate)
            frames[idx] = None  # Conserve memory.
        if chunk:
            video.extend(chunk)
        if self.show:
            cv2.destroyAllWindows()
        result = list(), images
        if self.mode == 'explode':
            chunk = list()
            for frame in video:
                if not np.array_equal(frame, filled_frame):
                    chunk.append(frame)
                else:
                    if chunk:
                        result[0].append(chunk)
                        chunk = list()
        else:
            result[0].append(video)
        if self.show:
            self.show_video_from_frames(video)
        return result

    def process(
            self,
            file_path_or_array: [str, list[np.ndarray]],
            mode: modes = 'heal',
            show: bool = False,
            show_breaks: bool = False
    ) -> tuple:
        """
        Process and output video
        """
        self.show = show
        self.mode = mode
        self.show_breaks = show_breaks
        if self.mode == 'explode':
            self.show_breaks = True  # We use the breaks to detect the in and out of the clips.
        if isinstance(file_path_or_array, str):
            frames = self.load_video_to_memory(file_path_or_array)
        else:
            frames = file_path_or_array
        result = self.deduplicate_frames(frames)
        slideshow = False
        total = 0
        for element in result[0]:
            total += len(element)
        if total <= len(result[1]):
            slideshow = True
        if self.debug:
            clips = result[0]
            clip_details = f""
            for idx, clip in enumerate(clips):
                details = f"\tclip: {idx}, with {len(clip)} frames\n"
                clip_details += details
            reel_details = f""
            if not slideshow:
                reel_details = f"clips exported: {len(result[0])}\n{clip_details}"
                if not len(result[1]):
                    reel_type = "uninterrupted video"
                else:
                    reel_type = "interrupted video"
            else:
                reel_type = "slideshow"
            print(
                f"\nREPORT:\n"
                f"input status: {reel_type}\n{reel_details}"
                f"images exported: {len(result[1])}\n"
            )
        return result, slideshow
