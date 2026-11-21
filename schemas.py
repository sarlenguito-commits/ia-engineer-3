import unicodedata

from pydantic import BaseModel, Field, ValidationError, field_validator

NO_LO_SE = "No lo sé"
MAX_RESPUESTA = 2000
MAX_REFERENCIAS = 5


def normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFD", texto.strip().lower())
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")


def es_no_lo_se(texto: str) -> bool:
    return normalizar(texto).strip(".,;:!¡?¿\"' \t\n\r") == normalizar(NO_LO_SE)


def normalizar_cita(texto: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in normalizar(texto)).split())


class Referencia(BaseModel):
    fuente: str = Field(description="Nombre del archivo del que sale la información, tal como figura en el contexto.")
    fragmento: str = Field(description="Frase corta copiada del contexto que respalda la respuesta.")

    @field_validator("fuente", "fragmento", mode="before")
    @classmethod
    def validar_texto(cls, valor, info):
        if not isinstance(valor, str) or not valor.strip():
            raise ValueError(f"{info.field_name} tiene que ser un texto no vacío; se recibió {valor!r}")
        return valor.strip()


class RespuestaRAG(BaseModel):
    respuesta: str = Field(
        description="Respuesta basada solo en el contexto, o exactamente 'No lo sé' si el contexto no la contiene."
    )
    referencias: list[Referencia] = Field(
        default_factory=list,
        description="Fuentes que respaldan la respuesta; vacía si la respuesta es 'No lo sé'.",
    )

    @field_validator("respuesta", mode="before")
    @classmethod
    def validar_respuesta(cls, valor):
        if not isinstance(valor, str):
            raise ValueError(f"respuesta tiene que ser un texto; se recibió {valor!r}")
        valor = valor.strip()
        if not 1 <= len(valor) <= MAX_RESPUESTA:
            raise ValueError(
                f"respuesta tiene que tener entre 1 y {MAX_RESPUESTA} caracteres; se recibió una de {len(valor)}"
            )
        return valor

    @field_validator("referencias", mode="before")
    @classmethod
    def validar_referencias(cls, valor):
        if not isinstance(valor, list):
            raise ValueError(f"referencias tiene que ser una lista; se recibió {valor!r}")
        if len(valor) > MAX_REFERENCIAS:
            raise ValueError(f"referencias puede tener hasta {MAX_REFERENCIAS} elementos; se recibieron {len(valor)}")
        return valor

    @property
    def sin_respuesta(self) -> bool:
        return es_no_lo_se(self.respuesta)


LIMITES = {"ge": "mayor o igual a", "le": "menor o igual a", "gt": "mayor que", "lt": "menor que"}


def explicar_errores(error: ValidationError) -> str:
    mensajes = []
    for detalle in error.errors():
        campo = ".".join(str(parte) for parte in detalle["loc"])
        if detalle["type"] == "missing":
            mensajes.append(f"falta el campo {campo}")
        elif detalle["type"] in ("int_parsing", "float_parsing"):
            mensajes.append(f"{campo} tiene que ser un número; se recibió {detalle['input']!r}")
        elif set(detalle.get("ctx", {})) & set(LIMITES):
            clave = next(k for k in detalle["ctx"] if k in LIMITES)
            mensajes.append(
                f"{campo} tiene que ser {LIMITES[clave]} {detalle['ctx'][clave]}; se recibió {detalle['input']!r}"
            )
        else:
            mensajes.append(detalle["msg"].removeprefix("Value error, "))
    return "; ".join(mensajes)
