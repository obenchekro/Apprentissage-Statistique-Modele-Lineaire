install.packages('glmnet')
library(glmnet)

######################################################
#################  Pokemon  ##########################
######################################################

# Le jeu de données pokemon
pokemon <- read.csv("L3_MIASHS_IDS/Modele Lineaire/pokemon.csv")
pokemon = na.omit(pokemon)
View(pokemon)

# noms des colonnes
colnames(pokemon)

# variable a definir
y=pokemon$speed 

# matrice des variables predictives
X = pokemon[,c("attack","defense","hp","sp_attack","sp_defense")]

x = data.matrix(pokemon[,c("attack","defense","hp","sp_attack","sp_defense")])
View(x)

# estimation des coefficients du modele de regression 
model = glmnet(x, y, alpha=0, standardize = F)
# il montre ; 
# - le nombre de coefficients non nuls (Df)
# - le pourcentage de la deviance expliquÃ©e (%Dev)
# - la valeur de lambda (Lambda)

# desription des diverses sorties disponibles a partir de 'model'
summary(model)

# pour visualiser les coefficients
plot(model, label = T)
# l'axe correspond aux degres de liberte effectifs du lasso

# un resume du chemin de glmnet a chaque etape
print(model)
# il montre ; 
# - le nombre de coefficients non nuls (Df)
# - le pourcentage de la deviance expliquÃ©e (%Dev)
# - la valeur de lambda (Lambda)

# Test des coefficients avec un lambda a 0.1
coef(model, s = 0.1)

# objectif : trouver le lambda optimal tel qu'il
# minimise l'Erreur Quadratique Moyenne

# grace a la procedure dite "validation croisee"
# on arrive a faire la "Moyenne des Erreurs de prediction
# sur une nouvelle donneee" :
cv_model = cv.glmnet(x, y, alpha=0, standardize = F)

# on sort ensuite le lambda optimal :
best_lambda = cv_model$lambda.min ; best_lambda # 505.5

# EQM en fonction de lambda
plot(cv_model, main = "EQM en fonction de lambda")

# calcul des coefficients du modele par regression ridge
best_model = glmnet(x, y, alpha = 0, lambda = best_lambda)

# Coefficients des variables de x correspondant au lambda optimal
coef(best_model, s = "best_lambda")

# Nouvelles donnees surlesquelles on va predire y :
z=c(50,50,85,65,65)

# prediction des valeurs de y disponible avec predict (intervalle de confiance) :
p_conf = predict(best_model, newx = z, s = "lambda.min", interval = "confidence")

# prediction des valeurs de y disponible avec predict (intervalle de prediction) :
p_pred = predict(best_model, newx = z, s = "best_lambda", interval = "prediction") ; p_pred


######################################################
#################  Fifa 19  ##########################
######################################################

# Le jeu de données fifa
fifa <- read.csv("~/L3_MIASHS_IDS/Modele Lineaire/fifa_data.csv")
fifa = na.omit(fifa)
View(fifa)

# variable a definir
y = fifa$Age

# matrice des variables predictives
x = data.matrix(fifa[,c("Overall","Potential","Special","International.Reputation")])
View(x)

# Model d'estimation
model = glmnet(x, y, alpha=0, standardize = F)
print(model)

# Model Cross Validation
cv_model = cv.glmnet(x, y, alpha=0, standardize = F)
print(cv_model)

# On sort ensuite le lambda optimal :
best_lambda = cv_model$lambda.min ; best_lambda

# EQM en fonction de lambda
plot(cv_model, main = "EQM en fonction de lambda")

# calcul des coefficients du modele par regression ridge
best_model = glmnet(x, y, alpha = 0, lambda = best_lambda) ; best_model

# Coefficients des variables de x correspondant au lambda optimal
coef(best_model, s = "best_lambda")

# Nouvelles donnees sur lesquelles on va predire y :
z=c(84,85,1850,4)

# prediction des valeurs de y disponible avec predict (intervalle de confiance) :
p_conf = predict(best_model, newx = z, s = "best_lambda", interval = "confidence") ; p_conf

# prediction des valeurs de y disponible avec predict (intervalle : prediction) :
p_pred = predict(best_model, newx = z, s = "best_lambda", interval = "prediction") ; p_pred


######################################################
#################  Student  ##########################
######################################################

# Le jeu de données student
student_mat <- read.csv("~/L3_MIASHS_IDS/Modele Lineaire/student-mat.csv")
student_por <- read.csv("~/L3_MIASHS_IDS/Modele Lineaire/student-por.csv")
student <- merge(student_mat,student_por,by=c("school","sex","age","address"
      ,"famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
student = na.omit(student)
View(student)

colnames(student)

# variable a definir
y = student$age

# matrice des variables predictives
x = data.matrix(student[,c("Dalc.x","Walc.x","absences.x","health.x")])
View(x)

# Model d'estimation
model = glmnet(x, y, alpha=0, standardize = F)
print(model)

# Model Cross Validation
cv_model = cv.glmnet(x, y, alpha=0, standardize = F)
print(cv_model)

# On sort ensuite le lambda optimal :
best_lambda = cv_model$lambda.min ; best_lambda

# EQM en fonction de lambda
plot(cv_model, main = "EQM en fonction de lambda")

# calcul des coefficients du modele par regression ridge
best_model = glmnet(x, y, alpha = 0, lambda = best_lambda) ; best_model

# Coefficients des variables de x correspondant au lambda optimal
coef(best_model, s = "best_lambda")

# Nouvelles donnees sur lesquelles on va predire y :
z=c(3,5,10,2)

# prediction des valeurs de y disponible avec predict (intervalle de confiance) :
p_conf = predict(best_model, newx = z, s = "best_lambda", interval = "confidence") ; p_conf

# prediction des valeurs de y disponible avec predict (intervalle de prediction) :
p_pred = predict(best_model, newx = z, s = "best_lambda", interval = "prediction") ; p_pred


######################################################
############  Régression Kernel Ridge  ###############
######################################################

# Simulation des donnees du modele
fstar <- function(x){
  return(x-1.2*x^2-0.8*x^3+0.6*cos(2*pi*x))
}

sgma <- 0.1
n <- 80
x <- runif(n)
y <- fstar(x)+sgma*rnorm(n)
plot(x,y)

gridx=0:200/200
fgridx <- fstar(gridx)
lines(gridx,fgridx,col='red') # fonction y

# Ajustement de la regression de la crete du noyau avec un noyau gaussien
install.packages("CVST")
library(CVST)
krr <- constructKRRLearner()

dat <- constructData(x,y)
dat_tst <- constructData(gridx,0)

# Simulation de la regression avec sigma fixe et lambda qui varie
par(mfrow=c(3,3),oma = c(5,4,0,0) + 0.1,mar = c(0,0,1,1) + 0.1)
lambdas= 10^(-8:0)
for(lambda in lambdas){
  param <- list(kernel="rbfdot", sigma=50, lambda=lambda)
  krr.model <- krr$learn(dat,param)
  pred <- krr$predict(krr.model,dat_tst)
  plot(x, y, xaxt='n', yaxt='n', main=paste('lambda =', signif(lambda,digits=3)))
  lines(gridx,fgridx,col='red')
  lines(gridx,pred,col='blue') # fonction de regression ajustee
}
# On remarque que :
#    - lambda petit = sur-ajustement et pas assez de regularisation
#    - lambda grand = sous-ajustement et trop de regularisation


# Simulation de la regression avec sigma qui varie et lambda fixe
par(mfrow=c(3,3),oma = c(5,4,0,0) + 0.1,mar = c(0,0,1,1) + 0.1) 
sigmas=10^((1:9)/3)
for(sigma in sigmas){
  param <- list(kernel="rbfdot", sigma=sigma, lambda=0.01)
  krr.model <- krr$learn(dat,param)
  pred <- krr$predict(krr.model,dat_tst)
  plot(x,y,xaxt='n',yaxt='n',main=paste('sigma =',signif(sigma,digits=3)))
  lines(gridx,fgridx,col='red')
  lines(gridx,pred,col='blue')
}
# On remarque que :
#    - sigma petit = sous-ajustement
#    - sigma grand = sur-ajustement

# Validation croisee
params <- constructParams(kernel="rbfdot", sigma=sigmas, lambda=lambdas)
opt <- CV(dat, krr, params, fold=10, verbose=FALSE)
par(mfrow=c(1,1), mar = c(0,0,0,0))
param <- list(kernel="rbfdot", sigma=opt[[1]]$sigma, lambda=opt[[1]]$lambda)
krr.model <- krr$learn(dat,param)
pred <- krr$predict(krr.model,dat_tst)
plot(x,y,xaxt='n',yaxt='n',main=paste("selected values: sigma=",signif(param$sigma,digits=3),", lambda=",signif(param$lambda,digits=3)))
lines(gridx,fgridx,col='red')
lines(gridx,pred,col='blue')




