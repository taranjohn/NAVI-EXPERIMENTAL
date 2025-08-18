import os
import pandas as pd
from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    """Abstract base class for io loaders."""

    @abstractmethod
    def load_data(self):
        """
        Loads io from the source and returns a DataFrame and class names.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: DataFrame with 'image_path' and 'true_label_name'.
                - list: A list of class names.
        """
        pass


class YoloV8DataLoader(BaseDataLoader):
    """Loads io from a YOLOv8 format dataset."""
    CLASS_NAMES = [
        'Agriculture', 'Airport', 'Beach', 'City', 'Desert', 'Forest',
        'Grassland', 'Highway', 'Lake', 'Mountain', 'Parking', 'Port',
        'Railway', 'River'
    ]

    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir

    def load_data(self):
        """
        Parses YOLOv8 io, ensuring one unique classification label per image.
        """
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])
        image_to_class_map = {}
        for img_file in image_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_file)
            img_path = os.path.join(self.image_dir, img_file)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    line = f.readline()
                    if line:
                        class_label = int(line.strip().split()[0])
                        if 0 <= class_label < len(self.CLASS_NAMES):
                            image_to_class_map[img_path] = class_label
                        else:
                            print(f"Warning: Skipping label '{class_label}' in {label_file} as it's out of range.")

        if not image_to_class_map:
            raise ValueError("No valid io loaded. Check paths and label files.")

        df = pd.DataFrame(list(image_to_class_map.items()), columns=['image_path', 'class_label'])
        df['true_label_name'] = df['class_label'].apply(lambda x: self.CLASS_NAMES[x])
        #drop class_label column
        df = df.drop('class_label', axis=1)
        return df, self.CLASS_NAMES


class AIDDataset(BaseDataLoader):
    def __init__(self, image_dir):
        self.image_dir = image_dir

    def load_data(self):
        # Get the names of the subdirectories, which are the class names
        CLASS_NAMES = [d for d in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, d))]
        CLASS_NAMES.sort()  # Sort for consistency

        all_image_paths = []
        all_true_labels = []

        # Iterate over the class names (which are the directory names)
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.image_dir, class_name)

            # Iterate over all files in the class directory
            for filename in os.listdir(class_dir):
                if filename.lower().endswith('.jpg'):  # Use .lower() for robustness
                    image_path = os.path.join(class_dir, filename)

                    all_image_paths.append(image_path)
                    all_true_labels.append(class_name)

        # Create the DataFrame *after* all loops are finished
        df = pd.DataFrame({
            'image_path': all_image_paths,
            'true_label_name': all_true_labels
        })

        return df, CLASS_NAMES

def get_data_loader(dataset_type, **kwargs):
    """
    Factory function to get a io loader instance.

    Args:
        dataset_type (str): The type of the dataset (e.g., 'yolov8').
        **kwargs: Arguments to pass to the io loader's constructor.

    Returns:
        BaseDataLoader: An instance of a io loader.
    """
    if dataset_type == 'yolov8':
        return YoloV8DataLoader(**kwargs)
    elif dataset_type == 'aid':
        return AIDDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
