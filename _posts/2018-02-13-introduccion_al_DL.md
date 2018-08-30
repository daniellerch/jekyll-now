---
layout: post
author: Daniel Lerch
title: Introducción al Deep Learning con Keras
published: true
---

En esta primera receta vamos a ver cómo usar los elementos básicos para construir una red neuronal con [Keras](https://keras.io/). Para ponerlos en práctica usaremos el conocido conjunto de datos [MNIST](http://yann.lecun.com/exdb/mnist/), creando una red de reconocimiento de caracteres.

## ¿Qué es Keras?
Keras es una API de alto nivel escrita en Python que corre sobre diferentes motores de *deep learning* como [Tensorflow](https://www.tensorflow.org/), [CNTK](https://github.com/Microsoft/CNTK) o [Theano](https://github.com/Theano/Theano). Es la herramienta perfecta para el desarrollo rápido en *deep learning* y corre tanto en CPUs como en GPUs.

No sin motivo es una de las herramientas preferidas en las competiciones de [Kaggle](https://www.kaggle.com/).

## ¿Qué es MNIST?
La base de datos MINST es un conjunto de dígitos escritos a mano que ha sido usado durante años para la investigación en machine learning. Contiene un conjunto de entrenamiento de 60.000 muestras y un conjunto de test de 10.000.


<img class="image" src="{{ site.baseurl }}/images/mnist.png"/>
<p class="image-cite">Wikipedia contributors. “MNIST database.” Wikipedia, The Free Encyclopedia.</p>


## Acceso al conjunto de datos MNIST
Lo bueno de usar un conjunto de datos como MNIST para hacer experimentos es que viene de serie con la mayoría de herramientas de *machine learning*. En el caso de Keras, basta con importarlo del módulo “datasets” y cargar los datos.

```python
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

Disponemos de 60.000 muestras en el conjunto de entrenamiento. Cada una de ellas es una pequeña imagen de 28x28.

```python
>>> print x_train.shape
(60000, 28, 28)
```

Como etiquetas tenemos números del 0 al 9, indicando el dígito dibujado en la imagen.

```python
>>> print y_train.shape
(60000,)
>>> print y_train[:10]
[5 0 4 1 9 2 1 3 1 4]
```

## Preparando los datos para Keras
Cada imagen del conjunto MNIST está formada por un array de 28x28 formando una imagen en escala de grises. Es decir, cada píxel está representado por un valor numérico de un byte (un valor de 0 a 255).

En general, en visión artificial se suele trabajar con arrays de 3 dimensiones: alto, ancho y canal. El alto y el ancho de una imagen no hace falta explicarlos pues son evidentes. El canal en imágenes en color corresponde a los componentes rojo, verde y azul. Es decir, que para una imagen en color tendríamos tres bloques de 28x28. Sin embargo, en este caso, como se trata de imágenes en escala de grises solo hay un canal.

Esto es importante tenerlo en cuenta, pues tenemos que preparar los datos incluyendo dicho canal. Como actualmente cada imagen es de 28x28, necesitamos cambiar su forma a 28x28x1. Es decir, necesitamos añadir una dimensión.

```python
x_train = x_train.reshape(x_train.shape+(1,))
x_test = x_test.reshape(x_test.shape+(1,))
```

Por otra parte, tenemos que representar cada imagen mediante un valor normalizado de 0 a 1. Para ello, basta con usar un array de valores reales y dividir entre 255.

```python
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
```

Dado que hay diez dígitos diferentes, la salida de la red neuronal va a estar formada por 10 neuronas. De esta manera, al introducir un carácter a la red (cuando ya esté entrenada) cada neurona nos va a dar la probabilidad de que se trate de un dígito concreto.

Sin embargo, como hemos visto anteriormente, tal y como tenemos los datos de etiquetado actualmente, disponemos de un número del 0 al 9 como etiqueta. Para que la red pueda usar los datos necesitamos tener 10 valores, uno para cada neurona. La conversión la podemos hacer fácilmente con Keras.

```python
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
```

De esta manera cada etiqueta queda ahora representada por un array de 10 valores que estará a 1 para la etiqueta correcta y a 0 para las incorrectas. Si anteriormente las etiquetas de los diez primeras imágenes eran:

```python
[5 0 4 1 9 2 1 3 1 4]
```

Actualmente son:

```python
>>> print y_train.shape[:10]
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
```

## La red
En visión artificial se suelen usar las conocidas como Convolutional Neural Networks o CNN. En este caso usaremos una red sencilla de la que no vamos a hablar en detalle (lo dejo para otra receta). De momento, la idea importante con la que tenemos que quedarnos es que necesitamos definir el tamaño y forma de la entrada. Es decir, necesitamos decirle a Keras que van a entrar muestras de 28x28x1.

Después de la entrada vienen las diferentes capas de la red: capas de convolución, de *pooling*, de *dropout*, etc. Como decía, no entraremos en detalle en esta receta.

El punto aquí en el que conviene fijarse es que ala hora de crear el modelo, Keras necesita que le indiquemos cuales son las entradas y las salidas. Presta atención a la llamada a Model().

```python
from keras.models import Model        
from keras.layers import Input, Dense, Dropout 
from keras.layers import Flatten, Conv2D, MaxPooling2D
inputs = Input(x_train.shape[1:])
x = inputs
x = Conv2D(32, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax')(x)
outputs = x
model = Model(inputs=inputs, outputs=outputs)
```

Una vez creado el modelo lo tenemos que compilar, momento en el cual le indicaremos algunos parámetros importantes:

- La función de pérdida que usará la red. En este caso usamos “categorical_crossentropy” puesto que el problema tiene tres clases y esta es la función a usar en problemas de clasificación con más de dos clases.
- El algoritmo de optimización que se usará para entrenar la red. Usaremos “Adam”, mi favorito. Hay muchos, cada uno con sus pros y sus contras. Keras soporta los más conocidos.
- La métrica a usar. Usaremos el *accuracy*, que representa el porcentaje de aciertos.

```python
model.compile(loss="categorical_crossentropy", 
              optimizer="adam", metrics=['accuracy'])
```


## Entrenamiento de la red
Ha llegado el momento de entrenar la red. Para ello le daremos los datos de entrenamiento como entrada. Le daremos también los datos de test para que nos vaya informando de como evoluciona el entrenamiento, pero estos datos no los va a usar para entrenar.

Le informaremos también del *batch size*, es decir, de la cantidad de imágenes que debe usar en cada iteración. Así como del numero de *epochs*, es decir, de cuantas veces va a recorrer el conjunto entero de datos para entrenar.

```python
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1,
          validation_data=(x_test, y_test))
```

## Resultados
Una vez tenemos la red entrenada podemos evaluar el conjunto de test con la siguiente instrucción, que nos dará la pérdida y el *accuracy*.

```python
>>> score = model.evaluate(x_test, y_test, verbose=0)
>>> print 'loss:', score[0], 'accuracy:', score[1]
loss: 0.0257 accuracy: 0.9933
```

Los resultados mostrados corresponden al entrenamiento durante 10 *epochs*.

## La receta
Finalmente, os dejo la receta completa:

```python
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape+(1,))
x_test = x_test.reshape(x_test.shape+(1,))

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
                                                                                 
inputs = Input(x_train.shape[1:])
x = inputs
x = Conv2D(32, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax')(x)
outputs = x

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print 'loss:', score[0], 'accuracy:', score[1]
```
