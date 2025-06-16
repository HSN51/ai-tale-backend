# ai-tale-backend

**AI-powered Backend for Personalized Storytelling**

This repository contains the backend service of the AI-Tale project, responsible for generating interactive stories with accompanying images and audio using advanced AI technologies.

## Features

- **Story Generation**: Uses OpenAI GPT models to create personalized stories based on user input.
- **Image Generation**: Integrates DALL-E 3 to generate visuals for each story scene.
- **Audio Narration**: Converts generated stories to audio using Google Text-to-Speech (gTTS).
- **RESTful API**: Exposes endpoints for story generation, media retrieval, and health checks using FastAPI.
- **Media Management**: Handles static file serving for generated images and audio.
- **Firebase Integration**: Optionally stores story metadata and user interactions.
- **Robust Error Handling**: Ensures reliability with advanced error management and logging.

## System Architecture

- **Backend**: Python, FastAPI, OpenAI, DALL-E, gTTS, Firebase (optional)
- **Endpoints**:
  - `/generate_story` – Generates a story with images and audio
  - `/api/health` – Health check endpoint
  - Serves static media from `/uploads`

## Getting Started

### Prerequisites

- Python 3.8+
- [OpenAI API key](https://platform.openai.com/account/api-keys)
- (Optional) Firebase service account for Firestore integration

### Installation

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file in the backend directory:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Create upload directories:

```bash
mkdir -p uploads/images uploads/audio
```

### Running the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Usage

Send a POST request to `/generate_story` with your desired story parameters.

### Health Check

Access `/api/health` to confirm the backend is running.

## Roadmap

- Cloud deployment for global access
- Authentication and user management
- Analytics and usage monitoring
- Advanced search and filtering
- Offline media support

## License

This project is for educational and non-commercial use. For commercial licensing, please contact the author.

---

**Contributors**: [Your Name]

---
