# Sistema de recuperación semántica local (RAG)

Recibe una pregunta, busca los fragmentos más parecidos en una base vectorial local (ChromaDB) y pide al modelo
responder solo con esa información. Verifica que las referencias nombren archivos recuperados y que las frases citadas
existan en lo recuperado. Si la respuesta no está en los documentos, pide responder "No lo sé". Hecho con LangChain
(LCEL), ChromaDB, embeddings locales con `fastembed`, Pydantic y Python 3.12. No hace falta ninguna clave para indexar:
la única clave es la del modelo de lenguaje, para generar la respuesta.

## Correspondencia con la consigna

| Entregable pedido | Dónde está |
|---|---|
| Script de ingesta: lee `/data`, fragmenta (mínimo 500 tokens con 50 de solapamiento) y guarda en ChromaDB | `ingest.py` (`load_documents`, `build_splitter`, `ingest`) |
| Capa de recuperación: embedding de la pregunta y fragmentos más relevantes (`top_k` de 3 a 5) | `retriever.py` (`retrieve`) y `embeddings.py` |
| `get_rag_response(query)` asíncrona: búsqueda, prompt con los fragmentos, llamada asíncrona al LLM y parseo a Pydantic con texto y referencias | `rag.py` (`get_rag_response`) y `schemas.py` (`RespuestaRAG`, `Referencia`) |
| Cadena LCEL con prompt de "filtro de veracidad" ("No lo sé") y `PydanticOutputParser` | `rag.py` (`build_prompt`, `build_chain`) |
| Dos pruebas: pregunta con respuesta y "pregunta trampa" | `ejemplo.py` (2 con respuesta y 2 trampa) y `tests\test_rag.py` |
| Dataset de ejemplo (txt/md) | `data\` (4 apuntes) |
| Entorno virtual, dependencias y claves fuera del repo | `requirements.txt`, `.env.example` y `.gitignore` |

## Qué incluye

- `data\`: 4 apuntes de los módulos 1 y 2 del curso (asyncio, Pydantic, LangChain/LCEL y resiliencia), de unas 700
  palabras cada uno. Con `CHUNK_SIZE=500`, solapamiento 50 y `count_tokens` del modelo real dan 10 fragmentos:
  `asyncio-y-clientes-llm.md`: 3; `langchain-lcel.md`: 2; `pydantic-y-validacion.md`: 2; `resiliencia-y-logs.md`: 3.
  El mayor tiene 500 tokens de contenido, más 2 especiales.
- `ingest.py`: carga los `.txt` y `.md`, los fragmenta con `RecursiveCharacterTextSplitter` contando tokens con
  el tokenizador del propio modelo (`count_tokens`) y los guarda en una colección persistente de ChromaDB con distancia coseno. Si la base ya
  tiene fragmentos no vuelve a indexar (`python ingest.py --force` fuerza la reindexación). Guarda en la colección qué
  modelo de embeddings se usó: si alguien consulta con otro, falla con un mensaje claro en vez de devolver resultados
  sin sentido. Con `--force` primero carga los documentos, fragmenta y calcula los vectores; recién después reemplaza
  la colección. Si falla la lectura o el cálculo, conserva la base anterior. Permite reemplazar el modelo al reindexar.
- `embeddings.py`: embeddings locales con `fastembed` y el modelo `jinaai/jina-embeddings-v2-base-es` (768
  dimensiones, español e inglés). El tokenizador trunca a 512 tokens: su capacidad teórica de 8.192 no se usa.
  `count_tokens` cuenta sin truncar con una copia del tokenizador. La ingesta valida el límite efectivo:
  `CHUNK_SIZE` entre 500 y 510, dejando 2 tokens para los especiales. El mismo modelo indexa y consulta.
- `retriever.py`: busca por similitud y devuelve los fragmentos con su similitud (1 menos la distancia coseno).
- `rag.py`: el prompt, la cadena, el control de la respuesta y `get_rag_response()`.
- `config.py`: validación de la configuración al arrancar, con mensajes en español.
- `models.py`: registro de proveedores del modelo de lenguaje (`anthropic` y `openai`).
- `ejemplo.py`: script que indexa si hace falta y hace las preguntas de prueba.

## Cómo funciona

1. La pregunta se convierte en un embedding con el mismo modelo local con el que se indexó.
2. ChromaDB devuelve los `TOP_K` fragmentos más parecidos (por defecto 4).
3. Los fragmentos se numeran y se pasan como CONTEXTO, junto con la pregunta, a una cadena LCEL:
   `prompt | modelo | control de la respuesta`, envuelta en `.with_retry()`.
4. El prompt de sistema pide responder solo con el contexto, responder exactamente "No lo sé" si no está, y citar en
   `referencias` el archivo y una frase copiada del contexto.
5. `PydanticOutputParser` convierte la respuesta en `RespuestaRAG` (texto y referencias).
6. El control de la respuesta rechaza, y por lo tanto reintenta, estos casos:
   - respuesta cortada por falta de tokens (`stop_reason` igual a `max_tokens` o `finish_reason` igual a `length`);
   - JSON inválido o campos que faltan;
   - una respuesta con contenido que no cita ninguna fuente;
   - una referencia a un archivo que no está entre los fragmentos recuperados (una fuente inventada);
   - una frase citada que no aparece en los fragmentos recuperados de su fuente, normalizando mayúsculas, tildes,
     puntuación y espacios;
   - un "No lo sé" que trae referencias.
   La abstención es estricta: solo "No lo sé", admitiendo mayúsculas, ausencia de tildes y signos en los bordes;
   cualquier texto extra se valida como una respuesta con contenido. Los bloques markdown con JSON válido se aceptan.
   También se acepta un JSON casi completo con todos los campos y citas reales si el modelo no indicó corte por tokens:
   el parser puede completar los cierres que falten.
7. `.with_retry()` envuelve toda la cadena: cada reintento vuelve a llamar al modelo. Hace `LLM_MAX_RETRIES + 1`
   intentos en total y repite el mismo pedido (mismo prompt y mismo límite de tokens). Los errores de red y los límites
   de tasa los reintenta el cliente del proveedor, no la cadena.
8. El arranque de la cadena, los embeddings y la ingesta se ejecuta en hilos para no bloquear el event loop.
   Los constructores por defecto comparten un lock y una caché para inicializar una sola vez entre hilos.

Los logs del logger `rag` incluyen `Llamada al modelo: N ms` en cada intento, también cuando falla. El log final
`Tiempos: recuperación X ms, modelo Y ms (con reintentos), total Z ms` incluye el arranque diferido en el total.

## Cómo correrlo

Instalá `uv` siguiendo las instrucciones para tu sistema en https://docs.astral.sh/uv/ y abrí una terminal nueva.

Desde PowerShell, en la carpeta del proyecto:

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
copy .env.example .env
```

Completar en `.env` la clave del proveedor (`ANTHROPIC_API_KEY` por defecto).

```powershell
.venv\Scripts\python.exe ingest.py
.venv\Scripts\python.exe ejemplo.py
.venv\Scripts\python.exe ejemplo.py trampa-precio
```

En Linux/macOS, desde la carpeta del proyecto:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
cp .env.example .env
```

Completá la clave del proveedor en `.env` y ejecutá:

```bash
.venv/bin/python ejemplo.py
.venv/bin/python -m pytest
```

- La primera vez se descarga el modelo de embeddings (unos 600 MB, una sola vez, a `.cache\modelos`, que no va al repo).
  Después funciona sin internet para indexar y consultar; solo la llamada
  al modelo de lenguaje la necesita.
- `ingest.py` indexa los archivos de `data\` en `vectorstore\` (que tampoco va al repo). `ejemplo.py` también indexa si la
  base no existe, así que correr solo `ejemplo.py` alcanza.
- `ejemplo.py` sin argumentos hace las 4 preguntas; con un nombre hace solo esa (`con-respuesta`, `reintentos`,
  `trampa-precio` o `trampa-despliegue`).

## Salida esperada de ejemplo.py

Con `claude-haiku-5-5`, en la primera corrida (indexa y después consulta). Se muestra la primera pregunta, con sus logs,
y una pregunta trampa:

```
INFO Indexados 10 fragmentos de 4 archivos en 3.6 s
INFO Fragmentos recuperados: asyncio-y-clientes-llm.md#1 (0.42), langchain-lcel.md#0 (0.41), langchain-lcel.md#1 (0.41), asyncio-y-clientes-llm.md#0 (0.36)
INFO Llamando al modelo
INFO Llamada al modelo: 3930 ms
INFO Respuesta validada
INFO Tiempos: recuperación 32 ms, modelo 3972 ms (con reintentos), total 4003 ms

=== con-respuesta ===
Pregunta: ¿Qué diferencia hay entre invoke, batch y stream?
{
  "respuesta": "invoke ejecuta el Runnable una sola vez y devuelve el resultado completo. batch ejecuta el Runnable con muchos datos a la vez, pensado para procesar muchos textos independientes, y puede limitar la concurrencia con max_concurrency. stream devuelve la respuesta en fragmentos mientras se genera, para mostrar el texto en pantalla a medida que llega.",
  "referencias": [
    {
      "fuente": "langchain-lcel.md",
      "fragmento": "invoke ejecuta una vez y devuelve el resultado."
    },
    {
      "fuente": "langchain-lcel.md",
      "fragmento": "batch ejecuta con muchos datos a la vez."
    },
    {
      "fuente": "langchain-lcel.md",
      "fragmento": "stream devuelve la respuesta de a fragmentos mientras se genera."
    },
    {
      "fuente": "langchain-lcel.md",
      "fragmento": "Para procesar muchos textos independientes conviene batch o abatch, con max_concurrency en la configuración"
    },
    {
      "fuente": "langchain-lcel.md",
      "fragmento": "Para mostrar texto en pantalla a medida que se genera se usa stream o astream."
    }
  ]
}

INFO Fragmentos recuperados: resiliencia-y-logs.md#1 (0.15), pydantic-y-validacion.md#1 (0.12), asyncio-y-clientes-llm.md#1 (0.11), langchain-lcel.md#0 (0.10)
INFO Llamando al modelo
INFO Llamada al modelo: 722 ms
INFO Respuesta validada
INFO Tiempos: recuperación 28 ms, modelo 730 ms (con reintentos), total 758 ms

=== trampa-precio ===
Pregunta: ¿Cuánto cuesta el millón de tokens de Claude Haiku?
{
  "respuesta": "No lo sé",
  "referencias": []
}
```

Los logs (logger `rag`) muestran los fragmentos recuperados con su similitud, `Llamando al modelo` y `Llamada al modelo: N ms`
en cada intento (si aparecen dos veces, hubo un reintento), el motivo de cada reintento y los tiempos de recuperación, del
modelo y total.

Una pregunta del mismo tema cuya respuesta no está en los apuntes (el precio de un modelo) y una de otro tema
(Kubernetes) tienen similitud baja con los fragmentos (0,15 y 0,28 como mucho, contra 0,42 a 0,59 en las que sí tienen
respuesta), y el modelo responde "No lo sé" aunque conozca la respuesta por su cuenta.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

Resultado con el modelo local disponible: `91 passed`, sin claves y sin red. Usan embeddings falsos (deterministas),
un modelo de lenguaje falso y tres pruebas con los embeddings reales, que se saltean si no existe `.cache/modelos`.
Sin esa carpeta: `88 passed, 3 skipped`. Los tests bloquean conexiones externas y la lectura del archivo de claves;
el modelo real se carga solo desde la caché, sin descargas. Cubren:

- Ingesta: solo `.txt` y `.md`, carpeta inexistente o vacía, tamaño y solapamiento de los fragmentos, que cada fragmento
  conserve su fuente y posición, que la segunda ingesta no vuelva a indexar, `--force`, persistencia entre sesiones y
  modelo de embeddings distinto, reemplazo con otro modelo, conservación de la base ante fallas de lectura o embeddings,
  conteo por palabras inyectado, falta de contador y límite efectivo de 510 tokens frente a 511.
- Recuperación: orden por similitud, `top_k` y base vacía.
- Cadena RAG: respuesta con referencias, pregunta trampa con "No lo sé" y sin referencias, el contenido del prompt,
  JSON incompleto sin referencias, campo faltante, archivo y cita inventados, normalización de citas, abstención
  estricta, "No lo sé" con referencias, respuesta sin fuentes, respuesta cortada por tokens (Anthropic y OpenAI),
  bloques markdown con JSON válido, JSON casi completo aceptado sin señal de corte, texto irreparable, un error de red
  que no se reintenta, cero reintentos, logs por intento y varias preguntas en paralelo sobre la misma cadena,
  con una barrera que exige cuatro llamadas simultáneas. También verifican inicialización única entre hilos.
- Límites, con el valor justo en el límite y uno por encima: `top_k` (3 y 5, frente a 2 y 6), `chunk_size` (500 y 2.000,
  frente a 499 y 2.001), `chunk_overlap` (499, frente a 500), largo de la pregunta (1.000 y 1.001), cantidad de
  referencias (5 y 6) y largo de la respuesta (2.000 y 2.001).
- Configuración y `ejemplo.py`: valores inválidos, proveedor desconocido, clave faltante, que una consulta fallida no
  corte las siguientes y arranque con ingesta lenta sin bloquear los ticks de asyncio.
- Embeddings reales: fragmentos dentro del límite, conteo sin truncar e igualdad de vectores cuando los textos solo
  cambian después del token 512.

Lo que no cubren: la respuesta real del modelo de lenguaje ni su conexión con el proveedor.

## Variables de entorno

- `LLM_PROVIDER` (`anthropic` por defecto u `openai`), `ANTHROPIC_API_KEY` y `ANTHROPIC_MODEL`, `OPENAI_API_KEY` y
  `OPENAI_MODEL`.
- `LLM_MAX_TOKENS` (64 a 8192, por defecto 1024), `LLM_TIMEOUT` (hasta 300, por defecto 30) y `LLM_MAX_RETRIES` (0 a 5,
  por defecto 2).
- `RAG_DATA_DIR` (`data`), `RAG_VECTORSTORE_DIR` (`vectorstore`), `RAG_COLLECTION` (`documentos`), `EMBEDDING_MODEL` y
  `EMBEDDING_CACHE_DIR` (`.cache/modelos`).
- `CHUNK_SIZE` (rango general de configuración: 500 a 2000, por defecto 500; con el modelo actual el rango efectivo
  es 500 a 510 y la ingesta lo valida), `CHUNK_OVERLAP` (por defecto 50, menor que `CHUNK_SIZE`) y `TOP_K` (3 a 5,
  por defecto 4).

Las rutas relativas salen de la carpeta del proyecto. La clave se guarda como `SecretStr` y el `.env` no se sube al repo.
No se manda `temperature`: la API de Anthropic ya no la acepta (error 400 desde el 21/10/2026).

## Estado de las pruebas

- Anthropic (`claude-haiku-5-5`), con clave real (21/11/2026), con las citas ya verificadas: en tres corridas
  consecutivas de `ejemplo.py` las 4 preguntas devolvieron respuestas válidas. En la última, con los logs completos, las 4
  validaron al primer intento, sin reintentos. Las dos con respuesta citan frases que existen en `langchain-lcel.md` y las
  dos trampa responden "No lo sé".
- Embeddings: el modelo local real se probó con ChromaDB sobre los 4 apuntes, con 7 preguntas con respuesta. En 6 el
  archivo correcto quedó primero. En la restante ("¿Qué diferencia hay entre invoke, batch y stream?") quedó segundo y
  tercero, con una similitud casi igual a la del primero (0,41 contra 0,42), y el modelo respondió bien porque `TOP_K=4`
  incluye ambos.
- OpenAI como modelo de lenguaje: está implementado con la misma interfaz y se verifica que se construye, pero no se
  probó con clave real.

## Límites conocidos

- Si cambian los archivos de `data\` hay que correr `ingest.py --force`: no se detectan cambios solos.
- El tokenizador del modelo de embeddings trunca a 512 tokens, incluidos los especiales; la capacidad teórica de
  8.192 no se usa. Las preguntas largas también pueden truncarse al generar su embedding.
- Verificar archivos y frases citadas aporta trazabilidad; no prueba que cada afirmación de la respuesta esté
  respaldada por esas citas ni que los documentos sean verdaderos.
- El modelo de embeddings está pensado para español e inglés; con otros idiomas habría que elegir otro.
