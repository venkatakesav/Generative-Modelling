# transforms.py

class FloatLabelsTransform:
    def __call__(self, data):
        # If the data is a dictionary with 'image' and 'label' keys
        if isinstance(data, dict):
            image = data['image']
            label = float(data['label'])  # Convert label to float
            return {'image': image, 'label': label}

        # If the data is a tuple with image and label
        elif isinstance(data, tuple) and len(data) == 2:
            image, label = data
            label = float(label)  # Convert label to float
            return image, label

        # Default: return the data without transformation
        return data
