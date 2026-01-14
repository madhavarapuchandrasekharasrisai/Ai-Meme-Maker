import os
import io
import textwrap
import random
from typing import List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont



def create_solid_canvas(width: int, height: int, color: str = "#222222") -> Image.Image:
    return Image.new("RGB", (width, height), color)


def create_linear_gradient(width: int, height: int, start_color: str, end_color: str, direction: str = "vertical") -> Image.Image:
    # Simple gradient without numpy to keep deps minimal
    img = Image.new("RGB", (width, height), start_color)
    start = Image.new("RGB", (1, 1), start_color).getpixel((0, 0))
    end = Image.new("RGB", (1, 1), end_color).getpixel((0, 0))
    dr = (end[0] - start[0])
    dg = (end[1] - start[1])
    db = (end[2] - start[2])

    draw = ImageDraw.Draw(img)
    if direction == "vertical":
        for y in range(height):
            ratio = y / (height - 1) if height > 1 else 1
            color = (int(start[0] + dr * ratio), int(start[1] + dg * ratio), int(start[2] + db * ratio))
            draw.line([(0, y), (width, y)], fill=color)
    else:  # horizontal
        for x in range(width):
            ratio = x / (width - 1) if width > 1 else 1
            color = (int(start[0] + dr * ratio), int(start[1] + dg * ratio), int(start[2] + db * ratio))
            draw.line([(x, 0), (x, height)], fill=color)
    return img


def load_font(preferred: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """
    Try to load a TTF font; fallback to default bitmap font if not found.
    Good cross-platform picks: DejaVuSans, Arial.
    """
    candidates = [preferred] if preferred else []
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            if path:
                return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    # Fallback
    return ImageFont.load_default()


def wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width_px: int) -> str:
    """
    Wrap text so that each line fits within max_width_px.
    """
    if not text:
        return ""
    words = text.split()
    if not words:
        return ""

    lines = []
    current = []
    for w in words:
        trial = " ".join(current + [w])
        bbox = draw.textbbox((0, 0), trial, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= max_width_px:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def fit_text_within_box(draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int, font_path: Optional[str], max_size: int = 64, min_size: int = 16) -> Tuple[ImageFont.ImageFont, str]:
    """
    Reduce font size until text fits into box_w x box_h when wrapped.
    """
    for size in range(max_size, min_size - 1, -2):
        font = load_font(font_path, size=size)
        wrapped = wrap_text_to_width(draw, text, font, box_w)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=6, stroke_width=2)
        w, h = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        if w <= box_w and h <= box_h:
            return font, wrapped
    # If nothing fits, return smallest
    font = load_font(font_path, size=min_size)
    wrapped = wrap_text_to_width(draw, text, font, box_w)
    return font, wrapped


def render_text_on_image(
    image: Image.Image,
    text_top: str,
    text_bottom: str,
    font_path: Optional[str],
    text_color: str,
    stroke_color: str,
    margin_ratio: float = 0.05,
) -> Image.Image:
    """
    Classic meme layout: top and/or bottom text within bounded boxes.
    """
    img = image.copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    margin = int(W * margin_ratio)
    box_width = W - 2 * margin
    box_height = int(H * 0.22)  # allocate ~22% height for top and bottom each

    if text_top.strip():
        font, wrapped = fit_text_within_box(draw, text_top, box_width, box_height, font_path)
        top_bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=6, stroke_width=2)
        tw, th = (top_bbox[2] - top_bbox[0], top_bbox[3] - top_bbox[1])
        x = (W - tw) // 2
        y = margin
        draw.multiline_text(
            (x, y),
            wrapped,
            font=font,
            fill=text_color,
            align="center",
            spacing=6,
            stroke_width=2,
            stroke_fill=stroke_color,
        )

    if text_bottom.strip():
        font, wrapped = fit_text_within_box(draw, text_bottom, box_width, box_height, font_path)
        bot_bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=6, stroke_width=2)
        bw, bh = (bot_bbox[2] - bot_bbox[0], bot_bbox[3] - bot_bbox[1])
        x = (W - bw) // 2
        y = H - bh - margin
        draw.multiline_text(
            (x, y),
            wrapped,
            font=font,
            fill=text_color,
            align="center",
            spacing=6,
            stroke_width=2,
            stroke_fill=stroke_color,
        )

    return img


def add_watermark(img: Image.Image, text: str = "made with streamlit", opacity: int = 160) -> Image.Image:
    if not text.strip():
        return img
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(None, size=max(14, base.size[0] // 40))
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    pad = 12
    x = base.size[0] - w - pad
    y = base.size[1] - h - pad
    draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity), stroke_width=1, stroke_fill=(0, 0, 0, opacity))
    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")


# -----------------------------
# Helpers: captions (local + optional AI)
# -----------------------------

SAFETY_BLOCKLIST = {"kill", "hate", "terrorist", "bomb", "nazi", "racist", "suicide", "self-harm", "nsfw"}

def passes_safety(text: str) -> bool:
    lower = text.lower()
    return not any(bad in lower for bad in SAFETY_BLOCKLIST)


LOCAL_TEMPLATES = [
    "When you finally fix the bug and nothing else breaks",
    "POV: You ran it once and now it works forever (copium)",
    "That moment when your code works on the first try",
    "Deploying on Friday like nothing can go wrong",
    "Me: quick 5-min fix • Also me 4 hours later:",
]

def local_caption_ideas(prompt: str, tone: str, k: int = 6) -> List[str]:
    seed = f"{prompt}-{tone}"
    random.seed(seed)
    pool = LOCAL_TEMPLATES + [
        f"{prompt} but make it {tone}",
        f"{tone.capitalize()} take: {prompt}",
        f"When {prompt.lower()}, and you’re still {tone.lower()}",
        f"{prompt} — expectations vs reality",
        f"Plot twist: {prompt.lower()}",
    ]
    ideas = random.sample(pool, min(k, len(pool)))
    return [i for i in ideas if passes_safety(i)]


def ai_caption_ideas(
    prompt: str,
    tone: str,
    provider: str,
    api_key: str,
    k: int = 6,
    timeout: int = 20
) -> List[str]:
    """
    Optional: call a chat-completions compatible endpoint to get caption ideas.
    This uses simple HTTP calls to avoid extra SDK deps.
    Providers:
      - "openai"  -> https://api.openai.com/v1/chat/completions
      - "openrouter" -> https://openrouter.ai/api/v1/chat/completions
    """
    import requests

    system = "You are a witty meme writer. Return only a JSON array of short caption suggestions (5-10 words each), no prose."
    user = f"Topic: {prompt}\nTone: {tone}\nReturn 6 concise, safe, family-friendly meme captions."

    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.9,
        }
    elif provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.9,
        }
    else:
        return local_caption_ideas(prompt, tone, k=k)

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # best-effort parse: expect a JSON array
        import json
        ideas = json.loads(content)
        if not isinstance(ideas, list):
            return local_caption_ideas(prompt, tone, k=k)
        ideas = [str(x) for x in ideas][:k]
        ideas = [i for i in ideas if passes_safety(i)]
        return ideas if ideas else local_caption_ideas(prompt, tone, k=k)
    except Exception:
        # Fallback gracefully
        return local_caption_ideas(prompt, tone, k=k)


def get_env_secret(name: str) -> Optional[str]:
    # Try Streamlit Secrets, then env
    try:
        return st.secrets.get(name)  # type: ignore
    except Exception:
        return os.getenv(name)


# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(page_title="AI Meme & Poster Creator", page_icon="🖼️", layout="wide")
st.title("🖼️ AI Meme & Poster Creator")

with st.sidebar:
    st.header("Image source")
    src = st.radio("Choose", ["Upload image", "Generate background"], index=0)

    image = None
    if src == "Upload image":
        uploaded = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        st.subheader("Background options")
        width = st.number_input("Width (px)", 480, 2000, 1080, 20)
        height = st.number_input("Height (px)", 480, 2000, 1080, 20)
        bg_type = st.selectbox("Type", ["Solid", "Gradient"], index=1)
        if bg_type == "Solid":
            color = st.color_picker("Color", value="#222222")
            if st.button("Create"):
                image = create_solid_canvas(width, height, color)
        else:
            start_color = st.color_picker("Start color", "#222222")
            end_color = st.color_picker("End color", "#6666aa")
            direction = st.selectbox("Direction", ["vertical", "horizontal"])
            if st.button("Create"):
                image = create_linear_gradient(width, height, start_color, end_color, direction)

    st.markdown("---")
    st.header("Caption generation")
    tone = st.selectbox("Tone", ["Funny", "Sarcastic", "Wholesome", "Bold", "Minimal"], index=0)
    use_ai = st.checkbox("Use AI API for ideas (optional)", value=False)
    provider = st.selectbox("Provider", ["local", "openai", "openrouter"], index=0, disabled=not use_ai)

    # API key handling
    api_key = ""
    if use_ai and provider != "local":
        suggested_env = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
        existing = get_env_secret(suggested_env)
        if existing:
            st.info(f"Found {suggested_env} in secrets/env.")
            api_key = existing
        else:
            api_key = st.text_input(f"Enter {suggested_env}", type="password")

    st.markdown("---")
    st.header("Text style")
    text_color = st.color_picker("Text color", "#ffffff")
    stroke_color = st.color_picker("Outline color", "#000000")
    font_path = st.text_input("Font path (optional, TTF)", value="")
    add_mark = st.checkbox("Add watermark", value=False)


# Main workspace
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Canvas")
    if image is not None:
        st.image(image, use_column_width=True)
    else:
        st.info("Upload or generate a background to begin.")

with col2:
    st.subheader("Captions")
    prompt = st.text_input("Caption prompt/topic", "When you finally fix the bug")
    mode = st.radio("Layout", ["Single caption (bottom)", "Top & Bottom"], index=1)

    if st.button("Suggest captions"):
        if use_ai and provider != "local" and api_key:
            st.session_state["ideas"] = ai_caption_ideas(prompt, tone, provider, api_key, k=6)
        else:
            st.session_state["ideas"] = local_caption_ideas(prompt, tone, k=6)

    ideas = st.session_state.get("ideas", [])
    if ideas:
        selection = st.radio("Pick an idea", list(range(len(ideas))), format_func=lambda i: ideas[i])

    top_text, bottom_text = "", ""
    if mode == "Top & Bottom":
        top_text = st.text_area("Top text", ideas[selection] if ideas else "", height=100)
        bottom_default = "Deploy on Friday? What could go wrong."
        bottom_text = st.text_area("Bottom text", bottom_default, height=100)
    else:
        top_text = ""
        bottom_text = st.text_area("Caption (bottom)", ideas[selection] if ideas else prompt, height=120)

    if st.button("Render"):
        if image is None:
            st.error("No image/canvas. Please upload or generate a background.")
        else:
            rendered = render_text_on_image(
                image=image,
                text_top=top_text,
                text_bottom=bottom_text,
                font_path=font_path if font_path.strip() else None,
                text_color=text_color,
                stroke_color=stroke_color,
            )
            if add_mark:
                rendered = add_watermark(rendered)

            st.image(rendered, caption="Preview", use_column_width=True)

            buf = io.BytesIO()
            rendered.save(buf, format="PNG")
            st.download_button(
                "Download PNG",
                data=buf.getvalue(),
                file_name="meme.png",
                mime="image/png",
            )

st.caption("Note: Keep captions safe and respectful. This tool is for learning and student projects.")
