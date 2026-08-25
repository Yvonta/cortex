# Cortex

A unified REST API service bundling **Chatterbox TTS**, **OpenAI Whisper**, and **Ollama** to power real-time AI voice cloning and avatar intelligence.



## Features

- **Text-to-Speech (TTS):** Zero-shot voice cloning and multilingual speech synthesis using Chatterbox.
- **Speech-to-Text (STT):** Fast audio transcription powered by OpenAI Whisper.
- **LLM Engine:** Local text generation and proxying via Ollama.
- **Deployment:** Fully containerized with Docker and NVIDIA GPU support.

## Prerequisites

- **Git:** Installed on your system to clone the repository.
- **Docker & Docker Compose:** Installed with NVIDIA GPU support enabled (NVIDIA Container Toolkit).

## How to Clone and Run

1. **Clone the Repository**
    

  Fetch the project files from GitHub and navigate into the folder:
    

  Bash
  ```
  git clone https://github.com/Yvonta/cortex.git
  cd cortex

  ```
2. **Set Script Permissions**
    

  Grant execution permissions to the setup entrypoint script:
    

  Bash
  ```
  chmod +x entrypoint.sh

  ```
3. **Build the Docker Container**
    

  Assemble the container image with all GPU drivers, PyTorch, Whisper, Chatterbox, and Ollama dependencies:
    

  Bash
  ```
  docker compose build

  ```
4. **Start the Application**
    

  Launch the API service in the background:
    

  Bash
  ```
  docker compose up -d

  ```
5. **Monitor Initial Startup & Logs**
    

  Track the container initialization (Ollama model downloading, TTS loading, and server startup):
    

  Bash
  ```
  docker compose logs -f

  ```
6. **Stop the Application**
    

  To shut down the container service when finished:
    

  Bash
  ```
  docker compose down

  ```

## License

This project is dedicated to the public domain under the [Creative Commons Zero v1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (CC0 1.0) license. You can copy, modify, distribute, and perform the work, even for commercial purposes, without asking permission.

  


## About Yvonta

**Yvonta** is a technology platform and publication brand focused on personal development, digital identity, biohacking, and artificial intelligence.

  


- **Focus Areas:** The platform explores topics including digital identity, AI voice/video cloning, quantified self, cryonics, virtual reality, and personal tech innovation.
- **Origin & Meaning:** Etymologically derived from Germanic roots (*Yvonna* / *Ivo*), the name relates to the yew tree—a traditional symbol of endurance, vitality, and longevity.

