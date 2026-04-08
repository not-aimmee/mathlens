"""
✏️ Hand-written Math Recognizer
=================================

Recognizes mathematical expressions from images using the Gemini Vision API
and returns both the detected expression and its evaluated result.

DEPENDENCIES:
pip install gradio google-generativeai numpy opencv-python-headless pillow

SETUP:
Set your Gemini API key as an environment variable before running:
  export GEMINI_API_KEY="AIza..."

USAGE:
  python math_recognizer_final.py
  Open http://127.0.0.1:7860 in your browser

"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
import base64
import io
import os
import json

from solver import GeminiVisionClient, MathEngine

# ============================================
# 🔑 GEMINI API SETUP
# ============================================
# Set your API key as an environment variable:
#   export GEMINI_API_KEY="AIza..."
# Or replace the os.environ.get(...) below with your key directly.

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"

# ============================================
# 🎯 STEP 1: ADD YOUR MODEL IMPORTS HERE
# ============================================
# Example imports (uncomment and modify as needed):
# import tensorflow as tf
# from tensorflow import keras
# import torch
# from your_model import MathRecognizerModel

# ============================================
# 📦 STEP 2: LOAD YOUR MODEL
# ============================================

MODEL = "gemini-api"  # Using Gemini API — no local model needed

def load_model():
    """
    Verifies the Gemini API key is set and the client is ready.
    """
    global MODEL
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set! Add it as an environment variable.")
        MODEL = None
    else:
        print(f"✅ Gemini API ready — using {MODEL}")
        MODEL = "gemini-api"
    return MODEL

# ============================================
# 🔧 STEP 3: PREPROCESS YOUR IMAGES
# ============================================

def preprocess_image(image):
    """
    Converts PIL Image to base64 string for the Gemini Vision API.

    Args:
        image: PIL Image object or numpy array

    Returns:
        base64-encoded PNG string
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    buffer = io.BytesIO()
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# ============================================
# 🎯 STEP 4: MAKE PREDICTIONS
# ============================================

def predict_math_expression(image_b64):
    """
    Sends the image to Gemini Vision API and returns the
    recognized expression and the solver metadata.

    Args:
        image_b64: base64-encoded PNG string from preprocess_image()

    Returns:
        tuple: (expression, result_dict)
    """
    if MODEL is None:
        return "⚠️ API key not set", {
            "type": "error",
            "result": "Set GEMINI_API_KEY environment variable",
            "expanded": "N/A",
            "factored": "N/A",
            "latex": "N/A"
        }

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ API key not set", {
            "type": "error",
            "result": "Set GEMINI_API_KEY environment variable",
            "expanded": "N/A",
            "factored": "N/A",
            "latex": "N/A"
        }

    try:
        vision = GeminiVisionClient(api_key=api_key)
        raw_expression = vision.recognise(image_b64)
        engine = MathEngine()
        result = engine.solve(raw_expression)

        expression = result.get("recognised", "Could not read expression")
        return expression, result
    except Exception as e:
        err_text = str(e)
        if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text or "quota" in err_text.lower():
            err_text = "Your Gemini quota is exhausted. Wait a few minutes or check your Google Cloud quota/billing."
        return "❌ API Error", {
            "type": "error",
            "result": err_text,
            "expanded": "N/A",
            "factored": "N/A",
            "latex": "N/A"
        }

# ============================================
# 🎨 MAIN PREDICTION FUNCTION FOR UI
# ============================================

def recognize_math(image):
    """
    Main function called by the Gradio interface.

    Args:
        image: PIL Image uploaded by user

    Returns:
        tuple: (expression_html, answer_html)
    """
    if image is None:
        return (
           "<div style='text-align:center;padding:40px;color:#aab8cc;font-family:Outfit,sans-serif;font-size:15px;font-weight:500;letter-spacing:2px;text-transform:uppercase;'>— Please upload an image —</div>",
            "<div style='text-align:center;padding:40px;color:#aab8cc;font-family:Outfit,sans-serif;font-size:15px;font-weight:500;letter-spacing:2px;text-transform:uppercase;'>— No image provided —</div>"
        )

    try:
        processed_image = preprocess_image(image)
        expression, result = predict_math_expression(processed_image)

        result_value = result.get("result", "Could not evaluate")
        if isinstance(result_value, (dict, list)):
            result_value = json.dumps(result_value, indent=2)

        result_type = result.get("type", "Unknown")
        expanded = result.get("expanded", "N/A")
        factored = result.get("factored", "N/A")
        latex = result.get("latex", "N/A")

        answer_html = f"""
        <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=Outfit:wght@600;700;900&display=swap" rel="stylesheet">
        <div style='
            background: linear-gradient(135deg, #112250 0%, #3C507D 100%);
            padding: 40px;
            border-radius: 24px;
            border: 2px solid #E0C58F;
            text-align: left;
            box-shadow: 0 8px 32px rgba(17, 34, 80, 0.15);
        '>
            <p style='
                font-size: 13px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 3px;
                margin-bottom: 20px;
                font-family: "Outfit", system-ui, sans-serif;
                color: #E0C58F;
            '>✦ Result</p>
            <div style='
                font-size: 28px;
                color: #F5F0E9;
                font-family: "Cormorant Garamondorm", Georgia, serif;
                font-weight: 700;
                line-height: 1.5;
                letter-spacing: 1px;
            '>
                <strong>Recognised</strong>: {expression}<br>
                <strong>Type</strong>: {result_type}<br>
                <strong>Result</strong>: {result_value}<br>
                <strong>Expanded</strong>: {expanded}<br>
                <strong>Factored</strong>: {factored}<br>
                <strong>LaTeX</strong>: {latex}
            </div>
        </div>
        """

        expression_html = f"""
        <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=Outfit:wght@600;700;900&display=swap" rel="stylesheet">
        <div style='
            background: #ffffff;
            padding: 40px;
            border-radius: 24px;
            border: 2px solid #E0C58F;
            text-align: center;
            box-shadow: 0 8px 32px rgba(17, 34, 80, 0.1);
        '>
            <p style='
                color: #3C507D;
                font-size: 13px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 3px;
                margin-bottom: 20px;
                font-family: "Outfit", system-ui, sans-serif;
            '>✦ Predicted Expression</p>
            <p style='
                font-size: 44px;
                font-weight: 700;
                color: #112250;
                font-family: "Cormorant Garamond", Georgia, serif;
                margin: 0;
                word-break: break-word;
                line-height: 1.2;
                letter-spacing: 1px;
            '>{expression}</p>
        </div>
        """
        return expression_html, answer_html

    except Exception as e:
        error_html = f"""
        <div style='
            background: linear-gradient(135deg, #112250 0%, #3C507D 100%);
            padding: 28px 32px;
            border-radius: 20px;
            border: 2px solid #D9CBC2;
            text-align: center;
            color: #F5F0E9;
            font-family: "Outfit", system-ui, sans-serif;
            box-shadow: 0 16px 40px rgba(17,34,80,0.4);
        '>
            <p style='font-size: 18px; font-weight: 700; margin-bottom: 10px; letter-spacing: 2px; text-transform: uppercase; color: #F5F0E9;'>— Error Processing Image —</p>
            <p style='font-size: 14px; margin: 0; font-weight: 500; opacity: 0.75; letter-spacing: 1px;'>{str(e)}</p>
        </div>
        """
        return error_html, "<div style='text-align: center; color: #999;'>Please try again</div>"

# ============================================
# 🎨 CREATE THE BEAUTIFUL UI
# ============================================

def create_ui():
    """Create the Gradio interface with custom styling."""

    # Custom CSS - Luxury Navy & Gold palette (Sapphire, Royal Blue, Quicksand, Swan Wing, Shellstone)
    custom_css ="""
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Outfit:wght@400;500;600;700;900&display=swap');

    /* ── Palette ──────────────────────────────────────
       Sapphire   #3C507D   Royal Blue  #112250
       Quicksand  #E0C58F   Swan Wing   #F5F0E9
       Shellstone #D9CBC2
    ─────────────────────────────────────────────────── */

    /* Global — light, airy base with soft warm whites */
    .gradio-container {
        font-family: "Outfit", system-ui, sans-serif !important;
        background: linear-gradient(160deg, #f8f6f2 0%, #eef2f8 40%, #dce6f5 100%) !important;
        min-height: 100vh !important;
    }

    /* Header — white card, navy text, gold accent */
    #header-box {
        background: #ffffff !important;
        border-radius: 28px !important;
        padding: 52px 36px !important;
        margin-bottom: 28px !important;
        border: 2px solid #E0C58F !important;
        box-shadow: 0 8px 40px rgba(17, 34, 80, 0.1), 0 2px 8px rgba(17,34,80,0.06) !important;
    }

    #header-box h1 {
        color: #112250 !important;
        font-size: 50px !important;
        font-weight: 700 !important;
        margin: 0 0 14px 0 !important;
        text-align: center !important;
        font-family: "Cormorant Garamond", Georgia, serif !important;
        letter-spacing: 1px !important;
    }

    #header-box p {
        color: #3C507D !important;
        font-size: 15px !important;
        text-align: center !important;
        margin: 0 !important;
        font-family: "Outfit", system-ui, sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
    }

    /* Instructions Markdown */
    .gradio-container .prose {
        color: #1a2f68 !important;
        font-family: "Outfit", system-ui, sans-serif !important;
        font-weight: 500 !important;
    }
    .gradio-container .prose h3 {
        color: #050f1e !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        font-family: "Cormorant Garamond", Georgia, serif !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    .gradio-container .prose strong {
        color: #c9a96e !important;
    }
    .gradio-container .prose p, .gradio-container .prose li {
        color: #3C507D !important;
    }
    .gradio-container .prose em {
        color: #7a8fae !important;
    }

    /* Upload Box */
    .image-container {
        border: 2px solid #E0C58F !important;
        border-radius: 20px !important;
        background: #ffffff !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(17,34,80,0.08) !important;
    }

    .image-container:hover {
        border-color: #3C507D !important;
        box-shadow: 0 8px 30px rgba(60, 80, 125, 0.15) !important;
    }

    /* Button */
    .primary-button, button.primary {
        background: linear-gradient(135deg, #112250 0%, #3C507D 100%) !important;
        border: none !important;
        padding: 20px 40px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        color: #E0C58F !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 24px rgba(17, 34, 80, 0.25) !important;
        font-family: "Outfit", system-ui, sans-serif !important;
        letter-spacing: 2.5px !important;
        text-transform: uppercase !important;
    }

    .primary-button:hover, button.primary:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 14px 36px rgba(17, 34, 80, 0.35) !important;
        background: linear-gradient(135deg, #1a2f68 0%, #4a6090 100%) !important;
    }

    /* Results */
    .output-html {
        min-height: 140px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Tips */
    #tips-box {
        background: #ffffff !important;
        border-radius: 24px !important;
        padding: 32px !important;
        margin-top: 28px !important;
        border: 2px solid #dce6f5 !important;
        box-shadow: 0 4px 20px rgba(17, 34, 80, 0.07) !important;
    }

    /* Footer */
    footer {
        text-align: center !important;
        margin-top: 40px !important;
        padding: 20px !important;
    }

    /* Labels */
    label span, .label-wrap span {
        color: #112250 !important;
        font-family: "Outfit", system-ui, sans-serif !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    """

    with gr.Blocks(css=custom_css, title="Math Expression Recognizer", theme=gr.themes.Soft()) as demo:

        with gr.Box(elem_id="header-box"):
            gr.HTML("""
                <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Outfit:wght@400;500;600;700;900&display=swap" rel="stylesheet">
                <h1>✏️ Hand-written Math Recognizer</h1>
                <p>✦ Drop a math image — watch the magic unfold ✦</p>
            """)

        gr.Markdown("""
        ### ✦ How It Works
        **1.** Upload an image of a handwritten or printed math expression

        **2.** Hit the **"Solve It"** button

        **3.** See the predicted expression and its calculated result!

        *Supported formats: PNG, JPG, JPEG, BMP, GIF, and more!*
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Your Math Expression")

                image_input = gr.Image(
                    label="Drop image here or click to upload",
                    type="pil",
                    elem_classes="image-container"
                )

                recognize_btn = gr.Button(
                    "✦ Solve It",
                    variant="primary",
                    size="lg",
                    elem_classes="primary-button"
                )

            with gr.Column(scale=1):
                gr.Markdown("###  Results")

                expression_output = gr.HTML(
                    value="<div style='text-align:center;padding:40px;color:#aab8cc;font-family:Outfit,sans-serif;font-size:16px;font-weight:500;letter-spacing:2px;text-transform:uppercase;'>— Awaiting image —</div>",
                    elem_classes="output-html"
                )

                answer_output = gr.HTML(
                    value="<div style='text-align:center;padding:40px;color:#aab8cc;font-family:Outfit,sans-serif;font-size:16px;font-weight:500;letter-spacing:2px;text-transform:uppercase;'>— Ready to solve —</div>",
                    elem_classes="output-html"
                )

        with gr.Box(elem_id="tips-box"):
            gr.Markdown("""
            ### ✦ Tips for Best Results

            **✓ Clear Images** — Use well-lit, high-contrast photos
            **✓ Center Expression** — Keep the math expression centered in the frame
            **✓ Any Format** — Both handwritten and printed expressions work great!
            **✓ Good Quality** — Higher resolution images give better results
            """)

        gr.Markdown("### 🎯 Try These Examples")
        gr.Examples(
            examples=[
                # Add paths to your example images here:
                # ["examples/example1.png"],
                # ["examples/example2.jpg"],
            ],
            inputs=image_input,
            label="Click an example to try it out!"
        )

        gr.HTML("""
           <footer>
                <p style='font-size: 16px; font-weight: 600; color: #112250; font-family: "Cormorant Garamond", Georgia, serif; letter-spacing: 3px; text-transform: uppercase;'>
                    Crafted with precision · Powered by Machine Learning
                </p>
                <p style='font-size: 13px; color: #112250; margin-top: 8px; font-family: "Outfit", sans-serif; letter-spacing: 1.5px;'>
                    Add your model to make predictions · See instructions in the code
                </p>
            </footer>
        """)

        recognize_btn.click(
            fn=recognize_math,
            inputs=image_input,
            outputs=[expression_output, answer_output]
        )

    return demo

# ============================================
# 🚀 LAUNCH THE APPLICATION
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧮 Math Expression Recognizer")
    print("=" * 60)
    print()
    print("📦 Loading model...")

    # Load the model
    load_model()

    print()
    print("🎨 Creating user interface...")

    # Create the UI
    demo = create_ui()

    print()
    print("🚀 Launching application...")
    print()
    print("=" * 60)
    print("✅ Server is running!")
    print("=" * 60)
    print()
    print("📍 Open this URL in your browser:")
    print("   http://127.0.0.1:7860")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print()
    print("⚙️  Configuration:")
    print("   • Share publicly: Set share=True below")
    print("   • Change port: Set server_port=YOUR_PORT below")
    print()
    print("=" * 60)

    # Launch the app
    demo.launch(
        share=False,           # Set to True to create a public link
        server_name="0.0.0.0",  # Makes it accessible on your network
        server_port=7860,      # Default Gradio port (change if needed)
        show_error=True,       # Show detailed errors for debugging
        favicon_path=None      # Add your favicon path if you have one
    )
