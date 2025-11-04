# 🤖 neuroLESSA
### Inteligencia Artificial para la Interpretación en Tiempo Real de palabras en  Lengua de Señas Salvadoreña (LESSA) a Texto y Voz

---

## 📖 Acerca del Proyecto
En **El Salvador**, existe una persistente barrera comunicativa entre personas con discapacidad auditiva y la comunidad oyente.  
A pesar de la existencia de soluciones de IA, muchas **no están adaptadas** a las diferencias lingüísticas, gestuales y culturales de la **Lengua de Señas Salvadoreña (LESSA)**.

**neuroLESSA** surge como una solución desarrollada específicamente para **reconocer y procesar señas de LESSA**, utilizando visión por computadora y aprendizaje profundo entrenado con **datos locales**.  
Este proyecto busca promover la **inclusión tecnológica** y el **respeto por la identidad lingüística** de El Salvador.

---

## 🎯 Objetivo Principal
Desarrollar un **prototipo de Inteligencia Artificial** capaz de interpretar señas de LESSA en tiempo real.  
El sistema utiliza una **cámara estándar** para capturar los gestos y los convierte a **texto y voz**, facilitando la comunicación inclusiva.

---

## ✨ Características Clave
- ⚡ **Interpretación en Tiempo Real:** Reconocimiento instantáneo de señas.  
- 💬 **Traducción a Texto:** Muestra en pantalla la palabra o frase reconocida.  
- 🔊 **Traducción a Voz:** Utiliza *Google Text-to-Speech (gTTS)* para verbalizar la seña.  
- 🌎 **Modelo Local:** Entrenado con datos de LESSA, asegurando relevancia cultural.  
- 🖥️ **Interfaz Simple:** GUI limpia y fácil de usar construida con *Tkinter*.  

---

## 🛠️ Tecnologías Utilizadas
| Tecnología | Descripción |
|-------------|-------------|
| 🐍 **Python** | Lenguaje base del proyecto. |
| 🤖 **TensorFlow / Keras** | Diseño y ejecución del modelo de aprendizaje profundo. |
| 🎥 **OpenCV** | Captura y procesamiento de video en tiempo real. |
| ✋ **MediaPipe** | Extracción de puntos de referencia (*landmarks*) de las manos. |
| 🧩 **Tkinter** | Construcción de la interfaz gráfica de usuario (GUI). |
| 🔉 **gTTS** | Conversión de texto a voz. |
| 📊 **scikit-learn** | Utilidades de *machine learning* complementarias. |

---

## 🚀 Instalación y Puesta en Marcha


### 1️⃣ Clonar el repositorio y crear el entorno virtual
> Se recomienda usar **Python 3.10**

```bash
git clone https://github.com/JorgeMajano/neuroLESSA.git
cd neuroLESSA

# Crear el entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows:
.\venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```
### 2️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```
### 3️⃣ Ejecutar la aplicación
``` bash
python camaraVECTORES.py
```
La aplicación activará tu cámara principal y abrirá la interfaz gráfica.
¡Ya puedes comenzar a realizar señas y ver cómo neuroLESSA las interpreta en texto y voz!

# Requerimientos 
- Procesador Intel® Core™ i5-8250U
- 8 gb de ram
- Espacio en disco 15 gb

# 👥 Autores
- Jorge Ernesto Majano Santos
- Richard Jonathan Quinteros Mendoza





