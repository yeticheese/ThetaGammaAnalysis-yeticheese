from pathlib import Path
import numpy as np

import logging
logger = logging.getLogger('runtime')

class DataFolder:
    def __init__(self, input_dir: str):
        """
        Initialize a DataFolder instance.

        Parameters:
            input_dir (str): The directory containing REM recordings in .npz format.
        """
        logger.info("Initializing a DataFolder.")

        self.input_dir = input_dir
        self.paths = list(Path(input_dir).rglob('*.npz'))

        logger.info("DataFolder in {0} contains {1} files.".format(self.input_dir, len(self.paths)))

    def load(self, index: int):
        """
        Load data in .npz format from the specified file.

        Returns:
            list: List of Numpy arrays.
        """
        file = self.paths[index]
        container = np.load(file)
        data = [container[key] for key in container]

        return data

    def get_metadata(self, index: int):
        """
        Extract metadata from the file name.
        """

        filename = self.paths[index].stem
        name_parts = str(filename).split('_')
        field_names = ['RatID', 'StudyDay', 'condition', 'condition_full', 'treatment', 'posstrial_number']
        return {field: value for field, value in zip(field_names, name_parts)}

    def __getitem__(self, index: int):
        """
        Get both metadata and loaded data for a specific index.

        Parameters:
            index (int): Index of the file to retrieve data from.

        Returns:
            dict: Dictionary containing metadata (dict) and loaded data (np.ndarray).
        """
        logger.debug("Retrieving index number: {0}".format(index))

        data_dict = self.get_metadata(index)
        data = {"data" : self.load(index)}
        data["metadata"] = data_dict
        return data
    
    def __len__(self):
        return len(self.paths)
    
    def __str__(self):
        return f"DataFolder in '{self.input_dir}', contains {len(self.paths)} files."