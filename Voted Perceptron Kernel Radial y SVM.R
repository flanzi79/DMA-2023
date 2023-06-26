library(data.table)
library(caret)
#Voted Perceptron

# Función de kernel radial (RBF)
rbf_kernel <- function(x1, x2, gamma) {
  exp(-gamma * sum((x1 - x2)^2))
}

# Función de entrenamiento del Voted Perceptron kernelizado con kernel radial
voted_perceptron_radial_train <- function(X, y, gamma, max_iter) {
  n <- nrow(X)
  alpha <- numeric(n)
  biases <- numeric(max_iter)
  counts <- rep(1, max_iter)
  errors <- rep(0, n)
  c <- 0  # Contador de iteraciones
  
  for (iter in 1:max_iter) {
    for (i in 1:n) {
      prediction <- sign(sum(alpha * y * rbf_kernel(X[i, ],X, gamma)) + biases[iter])
      if (prediction != y[i]) {
        alpha[i] <- alpha[i] + 1
        biases <- c(biases, biases[iter] + y[i])
        counts <- c(counts, 1)
        errors[i] <- errors[i] + 1
      } else {
        counts[iter] <- counts[iter] + 1
      }
    }
    c <- c + 1
    if (all(errors == 0)) {
      break
    }
  }
  
  return(list(alpha = alpha, biases = biases[1:c], counts = counts[1:c]))
}

# Función de prediccion del Voted Perceptron kernelizado con kernel radial
voted_perceptron_radial_predict <- function(model, X_test, X_train, gamma) {
  alpha <- model$alpha
  biases <- model$biases
  
  n_test <- nrow(X_test)
  predictions <- numeric(n_test)
  
  for (i in 1:n_test) {
    prediction <- sign(sum(alpha * rbf_kernel(X_test[i, ], X_train, gamma)) + biases)
    predictions[i] <- prediction
  }
  
  return(predictions)
}



# Cargando el data set Titanic
set.seed(798956)
setwd("D:/Maestria Ciencia de Datos/9_Data Mining Avanzado/Ejercicio 10")  #Cambiar el directorio por el direcpotrio donde se encuentre el archivo titanic.csv
dt <- fread("./titanic.csv")
dt <- dt[complete.cases(dt)]

#remover variables no utilizadas
dt<-dt[, -c("Name", "Ticket", "Cabin")]
#cnvertir a numericas las variables tipo caracter
dt[, Sex := ifelse(Sex == "male", 1, 0)]
dt[, Embarked := ifelse(Embarked == "S",1 ,ifelse(Embarked == "C", 2,3)) ]



# Datos de entrenamiento (X_train) y etiquetas (y_train)
inTrain <- dt[,sample(.N, floor(.N*.7))]
Train <- dt[inTrain]
X_train <- scale(as.matrix(Train[,-c("PassengerId", "Survived")]))
y_train <- as.matrix(Train[,"Survived"])


# Datos de prueba (X_test)
Test <- dt[-inTrain]
X_test <- scale(as.matrix(Test[,-c("PassengerId", "Survived")]))


# Entrenamiento del Voted Perceptron kernelizado con kernel radial
gamma <- 0.16
max_iter <- 100
model <- voted_perceptron_radial_train(X_train, y_train, gamma, max_iter)



# Testing con Voted Perceptron kernelizado entrenado
predictions <- voted_perceptron_radial_predict(model, X_test, X_train, gamma)
print(predictions)


#Evaluación de resultados
y_test<-as.factor(Test$Survived)
y_pred<-as.factor(predictions)
confusion_votedper <- confusionMatrix(y_pred, y_test)
precision_votedper <- confusion$overall["Accuracy"]


#Usando SVM

library(e1071)

svm_train <- Train[,-"PassengerId"]
model_svm <- svm(Survived ~ ., data = svm_train,  type= "C-classification", kernel ="radial", gamma=0.16)

svm_test <- Test[,-c("PassengerId", "Survived")]
predictions_svm<-predict(model_svm, Test)

#Evaluación de resultados
confusion_SVM <- confusionMatrix(predictions_svm, y_test)
precision_SVM <- confusion$overall["Accuracy"]



