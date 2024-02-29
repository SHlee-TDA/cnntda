"""
Module for topological data analysis vectorization.

This module provides the Vectorization class which is responsible for
transforming dataset samples using topological data analysis techniques,
and subsequently saving the transformed data.

Author: Seong-Heon Lee (Postech MINDS)
Date: 19. 10. 23
"""

import os
import time
import logging

import numpy as np
from tqdm import tqdm

from filtration import BaseConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Vectorization:
    """
        A class to perform topological vectorization on datasets.

        This class provides functionalities to perform filtration, compute
        persistence diagrams, and vectorize the diagrams, and finally save
        the vectorized data.

        Attributes:
            dataset: A PyTorch Dataset.
            filtration: A filtration converter based on BaseConverter.
            persistence: Persistence diagram computation module.
            vectorization: Diagram vectorization module.
            vectors (np.ndarray, optional): Stores the vectorized data after transformation.
    """
    def __init__(self,
                dataset,
                filtration: BaseConverter,
                persistence,
                vectorization
                ):

        """
        Initialize the Vectorization instance.

        Args:
            dataset: A PyTorch Dataset.
            filtration (BaseConverter): Filtration method to convert samples.
            persistence: Persistence diagram computation module.
            vectorization: Diagram vectorization module.
        """
        self.dataset = dataset
        self.filtration = filtration
        if filtration.dataset is not dataset:
            self.filtration.dataset = self.dataset
        self.persistence = persistence
        self.vectorization = vectorization
        self.vectors = None

    def transform(self) -> np.ndarray:
        """
        Transform dataset samples using TDA techniques.

        This method performs the filtration, computes persistence diagrams,
        and vectorizes the diagrams.

        Returns:
            np.ndarray: The vectorized data.
        """
        start_time = time.time()

        # Step 1 : Convert sample to filtered image
        logger.info("Starting filtration...")
        filtered_dataset = self.filtration.convert_dataset()
        filtered_images = np.array([image[0].numpy() for image, target in filtered_dataset])
        
        # Step 2 : Compute Persistence Diagrams
        logger.info("Computing persistence diagrams...")
        persistence_diagrams = self.persistence.fit_transform(filtered_images)
        normalized_diagrams = [np.where(arr == np.inf, 1.0, arr) for arr in persistence_diagrams]
        
        # Step 3 : Vectorize Persistence Diagrams.
        logger.info("Vectorizing persistence diagrams")
        vectors = self.vectorization.fit_transform(normalized_diagrams)
        normalized_vectors = np.zeros_like(vectors, dtype=np.float32)
        for idx, vector in enumerate(vectors):
            min_val = vector.min()
            max_val = vector.max()

            if max_val == min_val:
                normalized_vector = np.zeros_like(vector, dtype=np.float32)
            else:
                normalized_vector = (vector - min_val) / (max_val - min_val)
            normalized_vectors[idx] = normalized_vector
        
        self.vectors = normalized_vectors
        end_time = time.time()
        logger.info(f"Total transformation time: {end_time - start_time} seconds")
        return normalized_vectors
    
    def save_transformed(self, save_dir: str, interval: int = 100):
        """
        Save the vectorized data to a specified directory.

        Args:
            save_dir (str): Directory path where the vectorized data should be saved.
            interval (int, optional): The interval for intermediate saves. Default is 100.
        """
        root_dir = self.dataset.root
        meta = self.dataset.meta
        image_ids = meta['image_id']
        save_dir = os.path.join(root_dir, 'processed', save_dir)
        os.makedirs(save_dir, exist_ok=True)

        if self.vectors is None:
            vectors = self.transform()
        else:
            vectors = self.vectors

        for idx, (vector, image_id) in tqdm(enumerate(zip(vectors, image_ids))):
            file_name = image_id + '.npy'
            path = os.path.join(save_dir, file_name)
            np.save(path, vector)

            # Save intermediate results
            if (idx + 1) % interval == 0:
                logger.info(f"Saved {idx + 1} vectors. Continuing...")
            
        