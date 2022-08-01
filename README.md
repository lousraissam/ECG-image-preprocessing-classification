# ECG-image-preprocessing-classification
We used the ECG Images dataset of Cardiac Patients created under the auspices
of Ch. Pervaiz Elahi Institute of Cardiology Multan,that aims to help the scientific
community for conducting research for Cardiovascular diseases, and downloaded
from Mendeley. https://data.mendeley.com/datasets/gwbz3fsgp8/2

### Example of ECG image offred by Dataset
![MI(1)](https://user-images.githubusercontent.com/64719616/182239937-6193c96d-1ead-4d61-8771-de4e50131675.jpg)

# Data preprocessing
* Read the RGB color (3D) using openCV library and transfer it to a gray image.
![Gray ECG](https://user-images.githubusercontent.com/64719616/182240771-cce1efdb-92a4-432e-93a5-21b61efa2eda.png)

* Segmentation of ECG line from the background

![ecg final](https://user-images.githubusercontent.com/64719616/182240933-3e6c4563-9637-4ab6-8b5e-952fa629b939.png)

* Resize image
* Normalization

### Model architecture
![our cnn architecture](https://user-images.githubusercontent.com/64719616/182241167-38370cd5-5787-49ed-a463-7371b67b88fb.png)

### Model evaluation
![Acc](https://user-images.githubusercontent.com/64719616/182241281-c22866ee-9417-4cc9-b868-22189e4ac542.PNG)
###
###
![loss](https://user-images.githubusercontent.com/64719616/182241278-da4b77f5-6cf7-4d24-8c04-2cc24a98019b.PNG)


