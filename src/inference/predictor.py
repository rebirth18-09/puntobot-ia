"""Cumple el contenido del contrato /clasificar; no consulta inventario."""
from difflib import SequenceMatcher
from src.data.preprocesamiento import normalizar_texto

def candidatos_producto(texto, catalogo):
    tokens = normalizar_texto(texto).split()
    candidatos = []
    for producto in sorted(set(catalogo)):
        nombre = normalizar_texto(producto)
        if not nombre:
            continue
        n = len(nombre.split())
        similitud = max((SequenceMatcher(None, " ".join(tokens[i:i+n]), nombre).ratio()
            for i in range(max(0, len(tokens)-n+1))), default=0.0)
        if similitud >= 0.5:
            candidatos.append({"nombre": producto, "similitud": float(similitud)})
    return sorted(candidatos, key=lambda c: (-c["similitud"], c["nombre"]))[:3]

def predecir(texto, modelo, catalogo, version):
    if not isinstance(texto, str):
        raise TypeError("El texto debe ser una cadena")
    if not texto.strip() or len(texto) > 200:
        raise ValueError("Escribe entre 1 y 200 caracteres")
    valores = modelo.predict_proba([texto])[0]
    candidatos = candidatos_producto(texto, catalogo)
    return {
        "modelo_version": version,
        "probabilidades": {str(c): float(p) for c,p in zip(modelo.classes_, valores)},
        "producto_texto": candidatos[0]["nombre"] if candidatos else None,
        "candidatos": candidatos,
    }
