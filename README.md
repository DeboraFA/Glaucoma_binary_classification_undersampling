# Glaucoma_binary_classification_undersampling

Todas as análises foram realizadas na versão Python 3.8.13

Na pasta Classificacao_ML_tradicional estão: modelo de extração de características por métodos de extração tradicional como LBP, HOG, Zernike e filtros de Gabor; 
e os modelos de transfer learning para extração de características. A classificação é realizada com SVM, MLP, XGB e voting.

Na pasta Classificacao_TransferLearning_DL estão os modelos de transfer learning para extração e classificação das imagens.

Na pasta CrossDataset estão os modelos com cross dataset, no qual são treinados dois dataset e testado em um dataset diferente.

No arquivo load_dataset o formato de entrada dos dataset são:

---Diretório 

--Classe 1 

-Imagem 1 

-Imagem 2

... 

--Classe 2 

-Imagem 1 

-Imagem 2



As classes devem possuir o mesmo nome nos diferentes diretórios, as classes foram renomeadas para 0_Normal e 1_Glaucoma em todos os dataset.

As análises foram realizadas com o merge dos datasets e com o balanceamento entre as classes de dados. 
São utilizados os métodos para selecionar a classe normal/não-glaucoma: 
- Random (separação aleatória) 
- Cluster centróide 
- Near Miss 

