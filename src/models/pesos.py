from collections import Counter

def calcular_pesos(etiquetas, clases):
    conteos = Counter(etiquetas)
    if set(conteos) != set(clases):
        raise ValueError("Cada clase debe estar presente en entrenamiento")
    n = len(etiquetas)
    return {clase: n / (len(clases) * conteos[clase]) for clase in clases}
