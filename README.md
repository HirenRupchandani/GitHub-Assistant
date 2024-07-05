# Git Assistant - Generative AI GitHub Repository Assistant

A system that allows you to chat with your GitHub repository. It is powered by GPT-3.5 Turbo and allows you to ask any questions about the content of a cloned GitHub repository.

## Chat with your GitHub Repository GenAI project

![Alt Text](insert-your-image-link-here)

## Tech Stack

- **Client:** Streamlit
- **Server Side:** LangChain 🦜🔗
- **Vectorstore:** Local (Using Facebook AI Similarity Search (FAISS))
- **Embeddings:** OpenAI
- **Large Language Model:** GPT-3.5 Turbo
- **Runtime:** Local/Cloud Run

## Environment Variables

To run this project, you will need to add the following environment variables to your `.env` file:

- `OPENAI_API_KEY`

## Run Locally

### Clone the project

```bash
git clone https://github.com/your-username/git-assistant.git
```

Go to the project directory

```bash
  cd git-assistant
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the Streamlit server

```bash
  streamlit run github_chat.py
```

NOTE: Make sure `OPENAI_API_KEy` is active when you run the chat.py file. As of May 2024, you need to pay atleast 5 USD to use that API but I guess it is worth it because the API calls are very very cheap.



## 🚀 About Me
Hiren Rupchandani, Aspiring Data Analyst and Machine Learning Engineer

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hiren-rupchandani/) 
