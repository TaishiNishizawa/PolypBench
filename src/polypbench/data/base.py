import numpy as np
import cv2


class Sequence:
    """One video: frames in order plus sparse or dense ground-truth masks."""

    def __init__(self, name, n_frames, fps, annotations):
        self.name = name
        self.n_frames = n_frames
        self.fps = fps
        self.annotations = annotations      # {frame_index (0-based): mask Path}

    def frames(self):
        """Yield (index, rgb_frame) for every frame, in order."""
        raise NotImplementedError # each subclass reads frames its own way

    def load_mask(self, index):
        """Return the ground-truth mask at `index` as uint8 {0, 1}."""
        if index not in self.annotations:
            raise KeyError(f"No annotation for frame {index} in sequence {self.name}.")

        path = self.annotations[index]
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask {path}")
        
        return (mask > 127).astype(np.uint8)

class VideoFileSequence(Sequence):
    def __init__(self, name, video_path, n_frames, fps, annotations):
        super().__init__(name, n_frames, fps, annotations)
        self.video_path = video_path

    def frames(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video {self.video_path}")
        try:
            count_read = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield count_read, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                count_read += 1
            if count_read != self.n_frames:
                raise RuntimeError(f"Expected {self.n_frames} frames, but read {count_read} from {self.video_path}")
        finally:
            cap.release()

class Dataset:
    def __init__(self, name, sequences):
        self.name = name
        self.sequences = sequences      

    def __iter__(self):
        return iter(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, name):
        for seq in self.sequences:
            if seq.name == name:
                return seq
        raise KeyError(f"Sequence {name} not found. Valid names: {[seq.name for seq in self.sequences]}")
