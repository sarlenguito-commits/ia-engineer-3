# Apuntes: resiliencia, errores y logs

## Qué puede fallar
Una llamada a un modelo puede fallar por la red (conexión caída, timeout), por límites del proveedor (error 429 de límite de tasa), por problemas del servidor (errores 5xx), por un pedido inválido (error 400) o por credenciales incorrectas (401). También puede devolver una respuesta que no sirve: texto en lugar de JSON, JSON incompleto, campos que faltan o una respuesta cortada a la mitad.

## Qué reintentar y qué no
Se reintentan los errores que pueden desaparecer solos: límites de tasa, caídas de red, timeouts y errores 5xx. No se reintentan los que no cambian al repetir: una clave inválida (401) o un pedido mal armado (400) fallarían siempre, y reintentarlos solo gasta tiempo. Los SDK oficiales ya reintentan por su cuenta los errores de conexión, el 429 y los 5xx, con un número de intentos configurable.

## Espera exponencial
Reintentar de inmediato puede empeorar un servicio que ya está saturado. La espera exponencial aumenta el tiempo entre intentos: 1, 2, 4, 8 segundos. El jitter agrega un poco de azar para que muchos clientes no reintenten exactamente al mismo tiempo. Siempre hay un máximo de intentos, y si se agotan, el error se informa de forma clara.

## Errores controlados
Un error controlado es uno que el programa anticipa y convierte en un resultado manejable en lugar de dejar que rompa todo. Por ejemplo, devolver un objeto de respuesta con un campo error y un método para saber si salió bien, o mostrar un mensaje claro al usuario. En un script con varios textos, un fallo en uno no debería impedir procesar los siguientes. Los mensajes no deben revelar claves: si un texto de error trae algo con forma de clave, se reemplaza antes de mostrarlo.

## Respuestas cortadas
Cada respuesta indica por qué terminó. En Anthropic el campo se llama stop_reason y vale max_tokens cuando el modelo se quedó sin tokens; en OpenAI se llama finish_reason y vale length. Una respuesta cortada puede dejar un JSON incompleto, así que hay que detectarlo antes de intentar usar el objeto. Reintentar con el mismo límite de tokens puede volver a fallar; si pasa de forma repetida, la solución es subir el máximo de tokens. El valor de max_tokens es un tope por respuesta, no mide la velocidad del modelo.

## Validar y reintentar la salida
Un paso de control después del modelo revisa que la respuesta esté completa y que cumpla el esquema. Si falla, lanza una excepción propia con un mensaje en español, y el reintento de la cadena vuelve a pedir la respuesta. Si se agotan los intentos, el error llega a quien llamó, que decide qué mostrar.

## Logs útiles
Un logger por componente (por ejemplo "pipeline" o "rag") permite filtrar y cambiar el nivel sin tocar el código. Niveles habituales: INFO para el flujo normal (llamando al modelo, respuesta validada), WARNING para algo que se recuperó (una respuesta que no validó y se reintentó) y ERROR para fallos que no se pudieron resolver. Si "Llamando al modelo" aparece dos veces, hubo un reintento.

Medir tiempos es parte de un buen log: la duración de cada llamada y la del flujo completo, con time.perf_counter. Permite ver dónde está el cuello de botella, por ejemplo cuánto tarda la recuperación de documentos y cuánto el modelo. No se registran claves ni textos sensibles completos: alcanza con el largo del texto y el resultado.

## Tests de resiliencia
Los comportamientos de resiliencia se prueban con modelos falsos que fallan de manera controlada. Casos útiles: la respuesta válida a la primera; la respuesta inválida seguida de una válida (se recupera); siempre inválida (falla con el mensaje correcto después del máximo de intentos); respuesta cortada por tokens; un error de red que no debe reintentarse en la cadena; cero reintentos configurados, que implica un solo intento; y los valores justo en el límite y uno por encima para cada tope. Un test que nunca falla aunque el código esté roto no sirve: cada test debería verificar un número de llamadas o un mensaje concreto.

## Orden del repositorio
Lo que se prueba tiene que estar en el repositorio: las pruebas que quedan en una carpeta temporal no las ve nadie. El README explica cómo instalar, cómo ejecutar, qué resultado se espera y qué se probó con claves reales y qué no.
