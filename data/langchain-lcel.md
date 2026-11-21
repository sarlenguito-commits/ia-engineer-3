# Apuntes: LangChain y LCEL

## Qué es un Runnable
En LangChain, casi cada pieza es un Runnable: el prompt que arma el mensaje, el modelo que genera la respuesta, el parser que ordena la salida e incluso una función común envuelta en RunnableLambda. Todos comparten la misma interfaz, con los mismos métodos, y por eso se pueden combinar como bloques.

Los métodos principales son:
- invoke ejecuta una vez y devuelve el resultado.
- batch ejecuta con muchos datos a la vez.
- stream devuelve la respuesta de a fragmentos mientras se genera.
Cada uno tiene su versión asíncrona: ainvoke, abatch y astream. En una aplicación asíncrona se usan las versiones con a, para no bloquear el event loop.

## Componer con el pipe
LCEL (LangChain Expression Language) conecta Runnables con el símbolo |. La expresión prompt | modelo | parser crea una cadena que recibe un diccionario de variables, arma el mensaje, llama al modelo y convierte la salida. La salida de cada paso es la entrada del siguiente. La cadena resultante también es un Runnable, así que tiene invoke, batch y stream, y se puede volver a componer o envolver.

## Prompts
ChatPromptTemplate arma los mensajes de sistema y de usuario a partir de plantillas con variables entre llaves, por ejemplo {texto} o {contexto}. Es preferible a armar strings con f-strings dentro de la cadena, porque LangChain gestiona las variables de entrada y el prompt queda modular y reutilizable. El método partial deja fijas algunas variables, como las instrucciones de formato, y deja abiertas las demás.

## Datos estructurados
Hay dos formas de obtener un objeto Pydantic. with_structured_output(Esquema) usa las herramientas del modelo; con include_raw=True devuelve un diccionario con el mensaje crudo (raw), el objeto parseado (parsed) y el error de parseo (parsing_error), lo que permite leer por qué terminó la respuesta. El argumento method="function_calling" fuerza el mismo mecanismo en distintos proveedores. PydanticOutputParser, en cambio, trabaja sobre el texto: agrega instrucciones de formato al prompt y parsea lo que devuelve el modelo.

## Funciones y ramas dentro de una cadena
RunnableLambda convierte una función en un Runnable, útil para registrar logs, medir tiempos o validar una respuesta dentro de la cadena. RunnableParallel ejecuta varias ramas con la misma entrada y devuelve un diccionario con los resultados de cada una; con operator.itemgetter se puede pasar una clave de la entrada tal cual. Con RunnablePassthrough, la entrada original sigue disponible más adelante.

## Reintentos con with_retry
with_retry envuelve un Runnable y lo vuelve a ejecutar si lanza ciertas excepciones. Sus opciones principales son retry_if_exception_type, que dice qué errores se reintentan; stop_after_attempt, que fija el máximo de intentos; y wait_exponential_jitter, que agrega espera creciente con un poco de azar entre intentos. Si todos los intentos fallan, se relanza el último error.

El reintento debe envolver la cadena entera y no solo el modelo. La validación de la respuesta ocurre después de que el modelo contesta: si el JSON es inválido o está incompleto, hay que volver a llamar al modelo. Si solo se reintentara el modelo, un error de validación posterior nunca provocaría otra llamada. Conviene reintentar únicamente los errores que una nueva llamada puede arreglar, como una respuesta mal formada o cortada, y dejar que los errores de red y los límites de tasa los resuelva el reintento del cliente del proveedor. Un detalle práctico: cada reintento repite el mismo pedido, con el mismo prompt y el mismo límite de tokens.

## Elegir entre invoke, batch y stream
Para una consulta aislada alcanza con invoke o ainvoke. Para procesar muchos textos independientes conviene batch o abatch, con max_concurrency en la configuración para no superar los límites del proveedor. Para mostrar texto en pantalla a medida que se genera se usa stream o astream.

## Probar sin claves
Una cadena se prueba sin llamar a un proveedor real reemplazando el modelo por uno falso que devuelve respuestas preparadas. LangChain incluye modelos falsos para pruebas, como FakeMessagesListChatModel, y se pueden extender con una clase propia. Así se verifican los reintentos, los errores controlados y el orden de los resultados en cada ejecución, sin costo y sin red.
