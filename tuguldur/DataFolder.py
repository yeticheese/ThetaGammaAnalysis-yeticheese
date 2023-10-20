from pathlib import Path
import numpy as np

class DataFolder:
    def __init__(self, input_dir: str):
        """
        Initialize the Dataset object.

        Args:
            input_dir (str): The directory containing REM files in npz format.
        """
        self.input_dir = input_dir
        self.paths = list(Path(input_dir).rglob('*.npz'))

    def load(self, index: int):
        """
        Load data from the specified file.

        Returns:
            list: List of Numpy arrays.
        """
        file = self.paths[index]
        container = np.load(file)
        data = [container[key] for key in container]

        return data

    def get_name(self, index: int):
        """
        Get metadata from the file name.
        """

        filename = self.paths[index].stem
        name_parts = str(filename).split('_')
        field_names = ['RatID', 'StudyDay', 'condition', 'condition_full', 'treatment', 'posstrial_number']
        return {field: value for field, value in zip(field_names, name_parts)}

    def __getitem__(self, index: int):
        """
        Get both metadata and loaded data for a specific index.

        Args:
            index (int): Index of the file to retrieve data from.

        Returns:
            dict: Dictionary containing metadata (dict) and loaded data (np.ndarray).
        """
        data_dict = self.get_name(index)
        data = {"data" : self.load(index)}
        data["metadata"] = data_dict
        return data
    
    def __len__(self):
        return len(self.paths)
    
    def __str__(self):
        num_files = len(self.paths)
        return f"DataFolder in '{self.input_dir}', contains {num_files} files."