from torch.utils.data import Dataset

class DatasetWrapper(Dataset):
    """
    A wrapper for PyTorch datasets that allows redefining their behavior
    without modifying the original dataset.
    """
    def __init__(self, original_dataset, transform=None):
        """
        Args:
            original_dataset: The PyTorch dataset to wrap
            transform: Optional transforms to apply to images
            additional_processing: Optional function to apply additional processing to items
        """
        self.dataset = original_dataset
        self.transform = transform

    def __len__(self):
        """Return the number of items in the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an item from the dataset and apply transformations
        
        Args:
            idx: Index of the item to get
            
        Returns:
            The processed item
        """
        # Get item from original dataset
        item = self.dataset[idx]
        
        # Apply custom transform to the image if provided
        if self.transform and isinstance(item, dict) and 'image' in item:
            item['image'] = self.transform(item['image'])

        
        # Remove 'bounding_boxes' from the item if it exists, creates problems with batching
        item.pop('bounding_boxes', None)  # More efficient than if-check + del
        
        return item