# Apuntes: asyncio y clientes de LLM

## Por qué asincronía
Una llamada a un modelo de lenguaje tarda de uno a varios segundos, y casi todo ese tiempo el programa solo espera la respuesta de la red. Con código síncrono, cada llamada bloquea el programa hasta que termina. Con asyncio, mientras una llamada espera, el programa puede avanzar con otras. La asincronía no hace más rápida una llamada individual: permite superponer las esperas.

## Corrutinas y event loop
Una función definida con async def devuelve una corrutina. No se ejecuta hasta que alguien la espera con await o la programa en el event loop. El event loop es un único hilo que alterna entre tareas: cuando una tarea llega a un await y tiene que esperar, cede el control y el loop ejecuta otra. Por eso el programa se arranca con asyncio.run(main()).

El error más común es bloquear el event loop. Un time.sleep(5) dentro de una corrutina detiene todas las tareas durante cinco segundos; la versión correcta es await asyncio.sleep(5). Lo mismo pasa con las llamadas síncronas de un SDK, con cálculo pesado de CPU o con leer archivos grandes: mientras ese código corre, nadie más avanza. Acceder a una variable en memoria no bloquea el loop, porque es instantáneo. Lo que bloquea es el trabajo lento que no usa await.

## Concurrencia: gather, Semaphore y timeout
asyncio.gather(*corrutinas) lanza varias corrutinas a la vez y devuelve una lista con los resultados en el mismo orden en que se pasaron. Si una falla, por defecto la excepción se propaga; con return_exceptions=True, el error vuelve como un resultado más y las otras no se pierden.

Lanzar cien llamadas simultáneas contra un proveedor suele terminar en errores de límite de tasa. Un asyncio.Semaphore(n) limita a n las corrutinas que pueden estar dentro de la sección protegida al mismo tiempo: cada una hace async with semaphore: y, si ya hay n adentro, espera su turno. Es el mecanismo para controlar cuántas llamadas simultáneas se permiten.

asyncio.timeout(segundos) corta una operación que tarda demasiado: cuando se cumple el plazo, la tarea se cancela y se lanza TimeoutError. Conviene calcular el timeout global teniendo en cuenta los reintentos, para no cortar una llamada que todavía tenía intentos por delante.

## Streaming con generadores asíncronos
Con streaming, el modelo manda la respuesta en fragmentos a medida que los genera. En Python se modela con un generador asíncrono: una función async def que usa yield y se consume con async for. El consumidor empieza a mostrar texto de inmediato en lugar de esperar la respuesta completa.

Hay que cerrar el generador explícitamente si el consumidor se detiene antes de tiempo, por ejemplo con contextlib.aclosing, que garantiza que se ejecute el cierre de la conexión aunque el bucle termine con una excepción o un break.

## Métricas de velocidad
- Latencia total: tiempo desde el pedido hasta la respuesta completa.
- TTFT (time to first token): tiempo hasta el primer fragmento. Es lo que más influye en la sensación de velocidad, y el streaming lo reduce sin cambiar el modelo.
- Throughput: tokens por segundo que genera el modelo. Depende del modelo y del hardware del proveedor, no del código del cliente.
- RTT: tiempo de ida y vuelta de la red; suele ser de milisegundos y casi nunca explica una demora de segundos.

Si el TTFT crece de forma lineal a medida que avanza la conversación, la causa más probable es que se reenvía todo el historial en cada turno y el prompt se hace cada vez más largo. La solución es acortarlo: un system prompt breve y una ventana deslizante que conserva solo los últimos mensajes. Los datos importantes, como el destino y la fecha de un viaje, se guardan aparte en variables de contexto (slots) para que no se pierdan cuando el historial se recorta.

## Dónde guardar el historial
Guardar el historial de conversaciones en una variable global de Python funciona con un solo proceso, pero impide el escalado horizontal: cada instancia del servidor tendría su propio estado y el mismo usuario podría caer en otra instancia y encontrarse con una conversación distinta. El estado conversacional debe vivir en un almacén externo compartido, como Redis o una base de datos.

## Clientes de proveedores
Los SDK oficiales ofrecen clientes asíncronos, como AsyncAnthropic y AsyncOpenAI. Una buena práctica es esconderlos detrás de una clase base abstracta con los métodos generate y stream, y elegir el proveedor con una variable de entorno. Así, cambiar de proveedor no obliga a modificar el resto del programa.
