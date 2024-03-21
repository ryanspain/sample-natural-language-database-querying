# Querying a database using natural language

This sample demonstrates querying a MySQL database using natural language thanks to LangChain, Chainlit, and Ollama.

![Demo](app/demo.gif)

## Requirements

- Docker Desktop
- Visual Studio Code + Dev Containers extension
- A dedicated Nvidia GPU 

## Running locally

1. Open the repository in Visual Studio Code
2. Use the Dev Containers command `Reopen in a Container`.
    - Tip: Use `Ctrl` + `Shift` + `P` to open the command pallete.
3. Install a LLM in the `llm` container using Ollama.
    - Command: `ollama pull mistral:7b-instruct-v0.2-q5_K_M`
4. Set the `LLM_MODEL` in the `.env` file (If you choose a different model).
5. When the container has loaded, launch the app using `F5`.
