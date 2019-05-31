# stanford-cars-model (under development)

### Dependency
- PyTorch
- scikit-image
- opencv-python

### Extract training & testing data using annotation bounding box

#### Folder structure
  ```
  data_processing/
  │
  ├── datasets/ - folder contain training & testing data
      ├── training/
          ├── original/ - original cars from training data
          ├── extracted/ - cars after extracted using bounding box label
      ├── testing/
          ├── original/ - original cars from testing data
          ├── extracted/ - cars after extracted using bounding box label
          
  ```
  Download training data from ..., copy every images in training and testing to
#### Running Script
python extract_cars.py



##3. Training

##4. Testing
