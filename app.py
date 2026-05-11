"""
Analizador de Imágenes con IA - Interfaz Web
Powered by Claude (Anthropic)
"""

import anthropic
import base64
import json
import os
from flask import Flask, Response, render_template, request, stream_with_context

app = Flask(__name__)
client = anthropic.Anthropic()

MODOS = {
    "describir": {
        "nombre": "Descripción General",
        "icono": "🔍",
        "prompt": "Describe esta imagen con detalle. Incluye los elementos principales, colores, composición y cualquier detalle relevante.",
    },
    "objetos": {
        "nombre": "Detección de Objetos",
        "icono": "📦",
        "prompt": "Lista todos los objetos identificables en esta imagen. Para cada objeto menciona su nombre, posición y características relevantes. Organiza por categorías.",
    },
    "texto": {
        "nombre": "Extraer Texto (OCR)",
        "icono": "📝",
        "prompt": "Extrae y transcribe TODO el texto visible en esta imagen, manteniendo el formato original (columnas, listas, títulos). No omitas nada.",
    },
    "colores": {
        "nombre": "Análisis de Colores",
        "icono": "🎨",
        "prompt": "Analiza la paleta de colores: colores dominantes con valores aproximados, esquema cromático, temperatura de color, y cómo los colores afectan el mensaje de la imagen.",
    },
    "codigo": {
        "nombre": "Analizar Código",
        "icono": "💻",
        "prompt": "Analiza el código de la imagen: identifica el lenguaje, explica qué hace paso a paso, detecta errores o bugs, y sugiere mejoras siguiendo buenas prácticas.",
    },
    "documento": {
        "nombre": "Analizar Documento",
        "icono": "📄",
        "prompt": "Analiza este documento: tipo de documento, resumen ejecutivo, puntos clave, datos importantes, y conclusiones o acciones requeridas.",
    },
    "sentimientos": {
        "nombre": "Análisis Emocional",
        "icono": "❤️",
        "prompt": "Analiza las emociones y sentimientos de la imagen: emoción principal, elementos visuales que la generan, mensaje implícito y efectividad comunicativa.",
    },
    "accesibilidad": {
        "nombre": "Alt Text (Accesibilidad)",
        "icono": "♿",
        "prompt": "Genera texto alternativo completo para accesibilidad web (WCAG 2.1): descripción corta para atributo alt (máx 125 caracteres) y descripción larga detallada para lectores de pantalla.",
    },
}

MODELOS = [
    {"id": "claude-opus-4-7", "nombre": "Claude Opus 4.7", "desc": "Más potente"},
    {"id": "claude-opus-4-6", "nombre": "Claude Opus 4.6", "desc": "Muy capaz"},
    {"id": "claude-sonnet-4-6", "nombre": "Claude Sonnet 4.6", "desc": "Rápido y preciso"},
    {"id": "claude-haiku-4-5", "nombre": "Claude Haiku 4.5", "desc": "Más económico"},
]


@app.route("/")
def index():
    return render_template("index.html", modos=MODOS, modelos=MODELOS)


@app.route("/analizar", methods=["POST"])
def analizar():
    modo = request.form.get("modo", "describir")
    pregunta = request.form.get("pregunta", "").strip()
    url_imagen = request.form.get("url_imagen", "").strip()
    modelo = request.form.get("modelo", "claude-opus-4-7")
    system_prompt = request.form.get("system_prompt", "").strip()

    contenido = []

    # Procesar imagen
    archivo = request.files.get("imagen")
    if archivo and archivo.filename:
        datos = archivo.read()
        b64 = base64.standard_b64encode(datos).decode("utf-8")
        mime = archivo.content_type or "image/jpeg"
        if mime not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
            mime = "image/jpeg"
        contenido.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64},
        })
    elif url_imagen:
        contenido.append({
            "type": "image",
            "source": {"type": "url", "url": url_imagen},
        })
    else:
        return Response(
            'data: {"error": "No se proporcionó ninguna imagen"}\n\n',
            mimetype="text/event-stream",
        )

    # Construir prompt
    if modo == "preguntar" and pregunta:
        prompt = pregunta
    elif modo in MODOS:
        prompt = MODOS[modo]["prompt"]
    else:
        prompt = MODOS["describir"]["prompt"]

    contenido.append({
        "type": "text",
        "text": prompt + "\n\nResponde en español con formato claro y estructurado.",
    })

    def generar():
        try:
            kwargs = {
                "model": modelo,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": contenido}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            with client.messages.stream(**kwargs) as stream:
                for texto in stream.text_stream:
                    payload = json.dumps({"texto": texto}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except anthropic.AuthenticationError:
            error = json.dumps({"error": "API key inválida. Verifica ANTHROPIC_API_KEY."})
            yield f"data: {error}\n\n"
        except anthropic.APIError as e:
            error = json.dumps({"error": f"Error de API: {str(e)}"})
            yield f"data: {error}\n\n"

    return Response(
        stream_with_context(generar()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ADVERTENCIA: ANTHROPIC_API_KEY no está configurada.")
        print("  PowerShell: $env:ANTHROPIC_API_KEY='tu_clave_aqui'")
        print("  CMD:        set ANTHROPIC_API_KEY=tu_clave_aqui\n")

    print("🚀 Iniciando Analizador de Imágenes...")
    print("📌 Abre tu navegador en: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
