#!/usr/bin/env python3
"""
Analizador de Imágenes con IA - Powered by Claude
Herramienta de línea de comandos para analizar imágenes.
"""

import anthropic
import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Optional

client = anthropic.Anthropic()

MODOS = {
    "describir": {
        "nombre": "Descripción General",
        "prompt": "Describe esta imagen con detalle. Incluye los elementos principales, colores, composición y cualquier detalle relevante. Sé minucioso y descriptivo.",
    },
    "objetos": {
        "nombre": "Detección de Objetos",
        "prompt": "Lista todos los objetos que puedes identificar en esta imagen. Para cada objeto menciona: nombre, posición en la imagen, cantidad si aplica, y características visuales relevantes. Organiza la lista por categorías si hay muchos objetos.",
    },
    "texto": {
        "nombre": "Extracción de Texto (OCR)",
        "prompt": "Extrae y transcribe TODO el texto visible en esta imagen. Mantén el formato original lo mejor posible (columnas, listas, títulos, etc.). Si hay texto en varios idiomas, indícalo. No omitas nada.",
    },
    "colores": {
        "nombre": "Análisis de Colores y Paleta",
        "prompt": "Analiza en detalle la paleta de colores de esta imagen: 1) Colores dominantes con sus tonos aproximados (incluye valores hex si puedes estimarlos), 2) Esquema de color (complementario, análogo, triádico, etc.), 3) Temperatura de color general, 4) Cómo los colores contribuyen al mensaje o atmósfera de la imagen.",
    },
    "codigo": {
        "nombre": "Análisis de Código",
        "prompt": "Analiza el código que aparece en esta imagen: 1) Identifica el lenguaje de programación, 2) Explica qué hace el código paso a paso, 3) Identifica cualquier error, bug o problema, 4) Sugiere mejoras o buenas prácticas. Si es una captura de pantalla de terminal, analiza los comandos y su output.",
    },
    "documento": {
        "nombre": "Análisis de Documento",
        "prompt": "Analiza este documento de forma estructurada: 1) Tipo de documento, 2) Resumen ejecutivo, 3) Puntos clave o secciones principales, 4) Datos o cifras importantes, 5) Conclusiones o acciones requeridas. Extrae toda la información relevante.",
    },
    "sentimientos": {
        "nombre": "Análisis de Sentimientos y Emociones",
        "prompt": "Analiza el tono emocional y los sentimientos transmitidos por esta imagen: 1) Emoción o sentimiento principal, 2) Elementos visuales que generan esa emoción (composición, colores, expresiones, etc.), 3) Mensaje implícito, 4) Audiencia objetivo si aplica, 5) Efectividad comunicativa.",
    },
    "accesibilidad": {
        "nombre": "Texto Alternativo para Accesibilidad",
        "prompt": "Genera un texto alternativo (alt text) completo y accesible para esta imagen siguiendo las pautas WCAG 2.1. Incluye: descripción concisa para atributo alt (máx 125 caracteres), y descripción larga detallada para usuarios con discapacidad visual. Prioriza la información más importante.",
    },
    "comparar": {
        "nombre": "Comparar Imágenes",
        "prompt": "Compara estas imágenes en detalle: 1) Similitudes principales, 2) Diferencias clave, 3) Calidad visual comparativa, 4) Contexto o propósito de cada una, 5) Conclusión sobre cuál logra mejor su objetivo (si aplica). Sé específico y objetivo.",
    },
}


def cargar_imagen(ruta_o_url: str) -> tuple[str, str | tuple[str, str]]:
    """Carga una imagen desde archivo local o URL."""
    if ruta_o_url.startswith(("http://", "https://")):
        return "url", ruta_o_url

    ruta = Path(ruta_o_url)
    if not ruta.exists():
        print(f"\n❌ Error: No se encontró el archivo '{ruta_o_url}'")
        sys.exit(1)

    mime_type, _ = mimetypes.guess_type(str(ruta))
    ext = ruta.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, mime_type or "image/jpeg")

    if mime_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
        print(f"\n❌ Error: Formato no soportado. Use JPG, PNG, GIF o WebP.")
        sys.exit(1)

    with open(ruta, "rb") as f:
        datos = base64.standard_b64encode(f.read()).decode("utf-8")

    return "base64", (mime_type, datos)


def construir_contenido(imagenes: list[str], prompt: str) -> list[dict]:
    """Construye el array de contenido para la API."""
    contenido = []

    for img in imagenes:
        tipo, datos = cargar_imagen(img)
        if tipo == "url":
            contenido.append({
                "type": "image",
                "source": {"type": "url", "url": datos},
            })
        else:
            mime_type, b64_datos = datos
            contenido.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": b64_datos,
                },
            })

    contenido.append({"type": "text", "text": prompt + "\n\nResponde en español."})
    return contenido


def analizar(
    imagenes: list[str],
    modo: str = "describir",
    pregunta: Optional[str] = None,
    modelo: str = "claude-opus-4-7",
    guardar: Optional[str] = None,
) -> str:
    """Analiza imágenes usando Claude con streaming."""

    if modo == "preguntar":
        if not pregunta:
            print("\n❌ Error: El modo 'preguntar' requiere --pregunta")
            sys.exit(1)
        prompt = pregunta
        modo_nombre = "Pregunta Personalizada"
    elif modo in MODOS:
        prompt = MODOS[modo]["prompt"]
        modo_nombre = MODOS[modo]["nombre"]
    else:
        modos_disponibles = ", ".join(list(MODOS.keys()) + ["preguntar"])
        print(f"\n❌ Error: Modo '{modo}' no reconocido.")
        print(f"Modos disponibles: {modos_disponibles}")
        sys.exit(1)

    if modo == "comparar" and len(imagenes) < 2:
        print("\n❌ Error: El modo 'comparar' requiere al menos 2 imágenes.")
        sys.exit(1)

    contenido = construir_contenido(imagenes, prompt)

    num = len(imagenes)
    print(f"\n{'═'*60}")
    print(f"  🔍 Analizando {num} imagen{'es' if num > 1 else ''}")
    print(f"  📋 Modo: {modo_nombre}")
    print(f"  🤖 Modelo: {modelo}")
    print(f"{'═'*60}\n")

    resultado = ""

    try:
        with client.messages.stream(
            model=modelo,
            max_tokens=4096,
            messages=[{"role": "user", "content": contenido}],
        ) as stream:
            for texto in stream.text_stream:
                print(texto, end="", flush=True)
                resultado += texto

        print("\n")

        if guardar:
            salida = {
                "modo": modo,
                "modo_nombre": modo_nombre,
                "modelo": modelo,
                "imagenes": imagenes,
                "pregunta": pregunta,
                "resultado": resultado,
            }
            with open(guardar, "w", encoding="utf-8") as f:
                json.dump(salida, f, ensure_ascii=False, indent=2)
            print(f"✅ Resultado guardado en: {guardar}\n")

        return resultado

    except anthropic.AuthenticationError:
        print("\n❌ Error: API key inválida. Verifica ANTHROPIC_API_KEY.")
        sys.exit(1)
    except anthropic.APIError as e:
        print(f"\n❌ Error de API: {e}")
        sys.exit(1)


def main():
    modos_ayuda = "\n".join(
        f"  {k:<15} {v['nombre']}" for k, v in MODOS.items()
    )

    parser = argparse.ArgumentParser(
        description="🖼️  Analizador de Imágenes con IA - Powered by Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Modos disponibles:
{modos_ayuda}
  preguntar       Hacer una pregunta personalizada sobre la imagen

Ejemplos:
  python analizador.py foto.jpg
  python analizador.py foto.jpg --modo objetos
  python analizador.py screenshot.png --modo codigo
  python analizador.py img1.jpg img2.jpg --modo comparar
  python analizador.py foto.jpg --modo preguntar --pregunta "¿Dónde fue tomada esta foto?"
  python analizador.py https://example.com/imagen.jpg --modo colores
  python analizador.py documento.png --modo texto --guardar resultado.json
  python analizador.py foto.jpg --modelo claude-sonnet-4-6
        """,
    )

    parser.add_argument(
        "imagenes",
        nargs="+",
        help="Rutas locales o URLs de imágenes a analizar",
    )
    parser.add_argument(
        "--modo", "-m",
        default="describir",
        help="Modo de análisis (default: describir)",
    )
    parser.add_argument(
        "--pregunta", "-p",
        help="Pregunta personalizada (para modo 'preguntar')",
    )
    parser.add_argument(
        "--modelo",
        default="claude-opus-4-7",
        choices=[
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ],
        help="Modelo Claude a usar (default: claude-opus-4-7)",
    )
    parser.add_argument(
        "--guardar", "-g",
        metavar="ARCHIVO.json",
        help="Guardar resultado en archivo JSON",
    )

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n❌ Error: Variable de entorno ANTHROPIC_API_KEY no configurada.")
        print("  Windows (CMD):        set ANTHROPIC_API_KEY=tu_clave_aqui")
        print("  Windows (PowerShell): $env:ANTHROPIC_API_KEY='tu_clave_aqui'")
        print("  Linux/Mac:            export ANTHROPIC_API_KEY=tu_clave_aqui")
        sys.exit(1)

    analizar(
        imagenes=args.imagenes,
        modo=args.modo,
        pregunta=args.pregunta,
        modelo=args.modelo,
        guardar=args.guardar,
    )


if __name__ == "__main__":
    main()
