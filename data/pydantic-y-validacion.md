# Apuntes: Pydantic y validación de datos

## Qué resuelve
Un programa que habla con un modelo de lenguaje recibe y envía datos con forma incierta: configuración que viene del entorno, mensajes de usuarios y respuestas del modelo. Pydantic permite declarar la forma esperada de esos datos con clases y validarlos al crearlos. Si algo no cumple, falla de inmediato con un error que dice qué campo está mal y por qué, en vez de propagar datos inválidos hasta el fondo del programa.

## Modelos y Field
Un modelo hereda de BaseModel y declara sus campos con anotaciones de tipo. Field agrega restricciones: ge y le para rangos numéricos (mayor o igual, menor o igual), gt y lt para rangos estrictos, min_length y max_length para textos y listas, y default para valores por defecto. Por ejemplo, una temperatura entre 0 y 2 o un máximo de tokens entre 64 y 8192 se declaran en una línea cada uno. Pydantic convierte tipos compatibles (el texto "30" se vuelve el número 30) y rechaza lo que no puede convertir.

## Validadores propios
field_validator ejecuta código para un campo. Con mode="before" recibe el valor crudo, antes de la conversión de tipos: sirve para limpiar y para dar mensajes propios. Con mode="after" recibe el valor ya convertido. model_validator con mode="after" revisa relaciones entre campos, como que el solapamiento de un fragmento sea menor que su tamaño o que un modelo de lenguaje sea coherente con el proveedor elegido.

Dentro de un validador se lanza ValueError con un mensaje claro. Los mensajes por defecto de Pydantic están en inglés y son genéricos. Un buen mensaje dice el rango permitido y qué valor se recibió: "tecnologias puede tener hasta 20 elementos; se recibieron 21". Para armar un resumen legible a partir de un ValidationError se recorre error.errors() y se traducen los tipos más comunes, como campo faltante, número inválido o fuera de rango.

## Enums y tipos estrictos
Un Enum (por ejemplo StrEnum) limita un campo a un conjunto cerrado de valores, como los niveles baja, media y alta. Normalizar el valor antes de validarlo, pasando a minúsculas y sacando espacios, evita rechazar respuestas correctas por detalles de forma.

## Claves y datos sensibles
SecretStr guarda un texto sensible. Al imprimir el objeto o su repr se ve ********** en lugar del valor, así que una clave no aparece por accidente en un log o en un traceback. Para usar el valor real hay que pedirlo explícitamente con get_secret_value(). Las claves de API van siempre en variables de entorno o en un archivo .env que no se sube al repositorio, y en el repositorio se deja un .env.example con los nombres vacíos.

## Salidas estructuradas del modelo
Para que un modelo devuelva datos que un programa pueda usar, se le da un esquema Pydantic. Hay dos caminos. Con with_structured_output el esquema se convierte en una herramienta (tool calling) y LangChain devuelve el objeto ya validado. Con PydanticOutputParser, el parser agrega al prompt instrucciones de formato con el esquema JSON y después convierte el texto que devuelve el modelo en un objeto de la clase.

En los dos casos hay que prever el fallo: el modelo puede devolver JSON incompleto, campos faltantes o valores fuera de rango. Esas respuestas no se deben aceptar en silencio; se detectan con la validación y se reintentan. Los límites defensivos también importan: acotar el tamaño del texto de entrada, la cantidad de elementos de una lista y el largo de cada elemento evita respuestas absurdas y gastos de tokens innecesarios.

## Validar la configuración al arrancar
La configuración inválida conviene descubrirla al inicio, no a mitad de una ejecución. Un modelo de configuración que lee las variables de entorno y valida todo junto (proveedor conocido, clave y modelo presentes, números dentro de rango) permite fallar temprano con un mensaje que dice exactamente qué línea del .env corregir. Separar la configuración de la respuesta del modelo en clases distintas mantiene cada una simple y fácil de probar.

## Serialización
model_dump() devuelve un diccionario y model_dump_json() un texto JSON. Los modelos Pydantic son la forma de entregar un resultado validado a otras funciones o de imprimirlo en el formato esperado.
