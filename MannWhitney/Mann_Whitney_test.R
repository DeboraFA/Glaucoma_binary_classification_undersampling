
library(tidyr)
library(readxl)
library(readODS)


dados = read_excel("C:/Users/Debora/Documents/jcis_subdataset/MannWhitney/data.xlsx")
head(dados)

metrica = 'F1'

Tradicional1 = dados[dados$Modelo=='Tradicional' & dados$Metricas==metrica & dados$Tipo=='Unbalanced',]
MobileNet1 = dados[dados$Modelo=='MobileNet' & dados$Metricas==metrica & dados$Tipo=='Unbalanced',]
ResNet1 = dados[dados$Modelo=='ResNet50' & dados$Metricas==metrica & dados$Tipo=='Unbalanced',]
VGGA1 = dados[dados$Modelo=='VGGA' & dados$Metricas==metrica & dados$Tipo=='Unbalanced',]
VGGB1 = dados[dados$Modelo=='VGGB' & dados$Metricas==metrica & dados$Tipo=='Unbalanced',]
Xception1 = dados[dados$Modelo=='Xception' & dados$Metricas==metrica & dados$Tipo=='Unbalanced',]


Tradicional2 = dados[dados$Modelo=='Tradicional' & dados$Metricas==metrica & dados$Tipo=='Balanced',]
MobileNet2 = dados[dados$Modelo=='MobileNet' & dados$Metricas==metrica & dados$Tipo=='Balanced',]
ResNet2 = dados[dados$Modelo=='ResNet50' & dados$Metricas==metrica & dados$Tipo=='Balanced',]
VGGA2 = dados[dados$Modelo=='VGGA' & dados$Metricas==metrica & dados$Tipo=='Balanced',]
VGGB2 = dados[dados$Modelo=='VGGB' & dados$Metricas==metrica & dados$Tipo=='Balanced',]
Xception2 = dados[dados$Modelo=='Xception' & dados$Metricas==metrica & dados$Tipo=='Balanced',]


# Teste de Mann-Whitney para comparacao entre duas amostras
# H0: Os dois grupos possuem a mesma medida de tendencia central
# H1: Os grupos estao centrais em pontos diferentes


wilcox.test(Tradicional1$Resultados,Tradicional2$Resultados,correct=FALSE,alternative = "two.sided") 
wilcox.test(VGGA1$Resultados,VGGA2$Resultados,correct=FALSE, alternative = "two.sided") 
wilcox.test(VGGB2$Resultados,VGGB1$Resultados,correct=FALSE, alternative = "two.sided") 
wilcox.test(MobileNet1$Resultados,MobileNet2$Resultados,correct=FALSE, alternative = "two.sided") 
wilcox.test(ResNet1$Resultados,ResNet2$Resultados,correct=FALSE, alternative = "two.sided") 
wilcox.test(Xception1$Resultados, Xception2$Resultados, alternative = "two.sided") 



