import pytest
from pydantic import ValidationError

from schemas import MAX_REFERENCIAS, MAX_RESPUESTA, RespuestaRAG, es_no_lo_se, explicar_errores

REFERENCIA = {"fuente": "chroma.md", "fragmento": "una carpeta local"}


def mensaje(**datos):
    with pytest.raises(ValidationError) as error:
        RespuestaRAG(**datos)
    return explicar_errores(error.value)


def test_respuesta_valida():
    respuesta = RespuestaRAG(respuesta=" Se guarda en disco. ", referencias=[REFERENCIA])
    assert respuesta.respuesta == "Se guarda en disco."
    assert respuesta.referencias[0].fuente == "chroma.md"
    assert not respuesta.sin_respuesta


def test_no_lo_se_con_y_sin_tilde_y_mayusculas():
    assert es_no_lo_se("No lo sé")
    assert es_no_lo_se("NO LO SE")
    assert es_no_lo_se("No lo sé.")
    assert es_no_lo_se("¡No lo sé!")
    assert not es_no_lo_se("  NO LO SE, no está en el contexto")
    assert not es_no_lo_se("No lo sé, pero la capital de Francia es París")
    assert not es_no_lo_se("No lo sé. La capital es París")
    assert not es_no_lo_se("Lo sé")
    assert not es_no_lo_se("Lo sé: la respuesta es 4")
    assert RespuestaRAG(respuesta="No lo sé").sin_respuesta


def test_referencias_por_defecto_vacias():
    assert RespuestaRAG(respuesta="No lo sé").referencias == []


def test_respuesta_vacia_o_demasiado_larga():
    assert "respuesta tiene que tener entre 1 y 2000 caracteres; se recibió una de 0" in mensaje(respuesta="  ")
    assert f"se recibió una de {MAX_RESPUESTA + 1}" in mensaje(respuesta="x" * (MAX_RESPUESTA + 1))
    assert len(RespuestaRAG(respuesta="x" * MAX_RESPUESTA).respuesta) == MAX_RESPUESTA


def test_respuesta_que_no_es_texto():
    assert "respuesta tiene que ser un texto; se recibió 5" in mensaje(respuesta=5)


def test_referencia_sin_fuente_o_fragmento():
    assert "fuente tiene que ser un texto no vacío" in mensaje(
        respuesta="ok", referencias=[{"fuente": " ", "fragmento": "x"}]
    )
    assert "fragmento tiene que ser un texto no vacío" in mensaje(
        respuesta="ok", referencias=[{"fuente": "a.md", "fragmento": ""}]
    )


def test_demasiadas_referencias():
    texto = mensaje(respuesta="ok", referencias=[REFERENCIA] * (MAX_REFERENCIAS + 1))
    assert f"referencias puede tener hasta {MAX_REFERENCIAS} elementos; se recibieron {MAX_REFERENCIAS + 1}" in texto
    assert len(RespuestaRAG(respuesta="ok", referencias=[REFERENCIA] * MAX_REFERENCIAS).referencias) == MAX_REFERENCIAS


def test_referencias_que_no_es_lista():
    assert "referencias tiene que ser una lista" in mensaje(respuesta="ok", referencias="chroma.md")


def test_campo_faltante_en_castellano():
    assert mensaje() == "falta el campo respuesta"
