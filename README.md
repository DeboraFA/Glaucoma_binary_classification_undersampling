# Glaucoma_binary_classification_undersampling

The analysis were performed in Python version 3.8.13

The features and labels to each extraction model are available at https://drive.google.com/file/d/13lrwFIE5s1TdEK1hb0ETaszFMrNgkVfD/view?usp=sharing

In the Classificacao_ML_tradicional folder are: feature extraction model by traditional extraction methods such as LBP, HOG, Zernike, and Gabor filters;
Furthermore, transfer learning models for feature extractions. The classification is done using SVM, MLP, XGB, and voting classifiers.

In the Classificacao_TransferLearning_DL folder are the transfer learning models for extracting and classifying the images.

In the CrossDataset folder are models with the cross dataset, in which two datasets are trained and tested on a different dataset.

In the load_dataset file, the input dataset formats are:

---Folder 

--Class 1 

-Image 1 

-Image 2

... 

--Class 2 

-Image 1 

-Image 2



The classes must have the same name in different directories, the classes have been renamed to 0_Normal and 1_Glaucoma in all datasets.

The analyzes were performed by merging the datasets and balancing the data classes.
The methods for selecting the normal/non-glaucoma class are used:
- Random 
- Cluster centróide 
- Near Miss 

