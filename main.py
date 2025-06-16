import os
import json
import logging
import re
import requests
import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
from openai import OpenAI
from gtts import gTTS

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_tale")

# Load .env file and get API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client with API key
client = OpenAI(api_key=openai_api_key)

# FastAPI App
app = FastAPI()

# Add static files mounting for uploads directory
try:
    # Create uploads directory if it doesn't exist
    uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    # Mount the directory to make it accessible via HTTP
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
    logger.info("Static files directory mounted successfully")
except Exception as e:
    logger.error(f"Error mounting static files directory: {e}")

# Firebase Connection
try:
    cred = credentials.Certificate("firebase_service_account.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase connection established")
except Exception as e:
    logger.error(f"Firebase connection error: {e}")
    # Will continue without Firebase
    db = None


# Health check endpoint
@app.get("/api/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "message": "API is running"}
    )


# Slugify
def slugify(text):
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_]+", "-", text).strip("-")


# Image generation
def generate_image(prompt, path):
    try:
        logger.info(f"🖼️ Image prompt sent:\n{prompt}")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        image_url = response.data[0].url
        image_bytes = requests.get(image_url).content
        with open(path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"✅ Image saved: {path}")
    except Exception as e:
        logger.error(f"❌ Image error: {e}")


# Audio generation
def generate_audio(text, path):
    try:
        logger.info(f"🔊 Generating audio...")
        tts = gTTS(text=text, lang='en')
        tts.save(path)
        logger.info(f"✅ Audio saved: {path}")
    except Exception as e:
        logger.error(f"❌ Audio error: {e}")


# API model
class StoryRequest(BaseModel):
    characters: list
    theme: str
    setting: str
    tone: str
    input_text: str = ""


# API endpoint
@app.post("/generate_story")
async def generate_story(story_input: StoryRequest):
    try:
        model_name = "gpt-3.5-turbo"

        prompt = f"""
        You are a creative AI that writes immersive children's fairy tales in JSON format.
        Characters: {', '.join(story_input.characters)}
        Theme: {story_input.theme}
        Setting: {story_input.setting}
        Tone: {story_input.tone}
        Additional user input: {story_input.input_text}

        Instructions:
        - Write a 3-minute story.
        - Split it into exactly 3 scenes.
        - Each scene must have 5 sentences.
        - Return ONLY valid JSON: {{"story_title": str, "scenes": [{{"scene_id": int, "text": str}}]}}
        """

        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = completion.choices[0].message.content

        if "```" in content:
            extracted = re.search(r"\{.*\}", content, re.DOTALL)
            story_json = json.loads(extracted.group(0)) if extracted else {}
        else:
            story_json = json.loads(content)

        # Check for exactly 3 scenes
        if len(story_json.get("scenes", [])) != 3:
            logger.error(f"❌ Incorrect number of scenes: {len(story_json.get('scenes', []))}. Operation aborted.")
            return JSONResponse(
                status_code=400,
                content={"error": "Story must have exactly 3 scenes. Operation aborted to save tokens."}
            )

        story_title = story_json.get("story_title", "story")
        slug = slugify(story_title)
        img_dir = f"uploads/images/{slug}"
        aud_dir = f"uploads/audio/{slug}"
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(aud_dir, exist_ok=True)

        # Create main story record in Firestore (if Firebase connection exists)
        if db:
            story_ref = db.collection("ai_stories").document(slug)
            story_ref.set({
                "story_title": story_title,
                "slug": slug
            })

        for scene in story_json.get("scenes", []):
            scene_id = scene["scene_id"]
            scene_text = scene["text"]

            image_prompt = (
                f"Flat storybook-style illustration for a children's fairy tale. "
                f"Scene {scene_id}: {scene_text}. "
                f"Colorful, imaginative, whimsical, child-friendly. "
                f"No text, no captions, no letters, no writing, no words. "
                f"Suitable for children aged 5–10. No text in the image."
            )

            image_path = f"{img_dir}/scene_{scene_id}.png"
            audio_path = f"{aud_dir}/scene_{scene_id}.mp3"

            # Generation
            generate_image(image_prompt, image_path)
            generate_audio(scene_text, audio_path)

            scene["image_path"] = image_path
            scene["audio_path"] = audio_path

            # Add scene to Firestore (if Firebase connection exists)
            if db:
                scene_ref = story_ref.collection("scenes").document(f"scene_{scene_id}")
                scene_ref.set({
                    "scene_id": scene_id,
                    "text": scene_text,
                    "image_path": image_path,
                    "audio_path": audio_path
                })

        return story_json

    except Exception as e:
        logger.error(f"❌ General error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Story generation failed: {str(e)}"}
        )


# API endpoint - streaming version
@app.post("/generate_story_stream")
async def generate_story_stream(request: Request, story_input: StoryRequest):
    async def event_generator():
        model_name = "gpt-3.5-turbo"

        prompt = f"""
        You are a creative AI that writes immersive children's fairy tales in JSON format.
        Characters: {', '.join(story_input.characters)}
        Theme: {story_input.theme}
        Setting: {story_input.setting}
        Tone: {story_input.tone}
        Additional user input: {story_input.input_text}

        Instructions:
        - Write a 3-minute story.
        - Split it into exactly 3 scenes.
        - Each scene must have 5 sentences.
        - Return ONLY valid JSON: {{"story_title": str, "scenes": [{{"scene_id": int, "text": str}}]}}
        """

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            content = completion.choices[0].message.content

            if "```" in content:
                extracted = re.search(r"\{.*\}", content, re.DOTALL)
                story_json = json.loads(extracted.group(0)) if extracted else {}
            else:
                story_json = json.loads(content)

            # Check for exactly 3 scenes
            if len(story_json.get("scenes", [])) != 3:
                logger.error(f"❌ Incorrect number of scenes: {len(story_json.get('scenes', []))}. Operation aborted.")
                yield f"data: {json.dumps({'error': 'Story must have exactly 3 scenes. Operation aborted to save tokens.'})}\n\n"
                return

            story_title = story_json.get("story_title", "story")
            slug = slugify(story_title)
            img_dir = f"uploads/images/{slug}"
            aud_dir = f"uploads/audio/{slug}"
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(aud_dir, exist_ok=True)

            # Create main story record in Firestore (if Firebase connection exists)
            if db:
                story_ref = db.collection("ai_stories").document(slug)
                story_ref.set({
                    "story_title": story_title,
                    "slug": slug
                })

            for scene in story_json.get("scenes", []):
                scene_id = scene["scene_id"]
                scene_text = scene["text"]

                image_prompt = (
                    f"Flat storybook-style illustration for a children's fairy tale. "
                    f"Scene {scene_id}: {scene_text}. "
                    f"Colorful, imaginative, whimsical, child-friendly. "
                    f"No text, no captions, no letters, no writing, no words. "
                    f"Suitable for children aged 5–10. No text in the image."
                )

                image_path = f"{img_dir}/scene_{scene_id}.png"
                audio_path = f"{aud_dir}/scene_{scene_id}.mp3"

                # Generation
                generate_image(image_prompt, image_path)
                generate_audio(scene_text, audio_path)

                scene["image_path"] = image_path
                scene["audio_path"] = audio_path

                # Add scene to Firestore (if Firebase connection exists)
                if db:
                    scene_ref = story_ref.collection("scenes").document(f"scene_{scene_id}")
                    scene_ref.set({
                        "scene_id": scene_id,
                        "text": scene_text,
                        "image_path": image_path,
                        "audio_path": audio_path
                    })

                yield f"data: {json.dumps(scene)}\n\n"
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"❌ General error: {e}")
            yield f"data: {json.dumps({'error': f'Story generation failed: {str(e)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
