import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_chat import message
from datetime import datetime, timedelta
from authenticate import get_user_session, update_session_query_count, reset_query_count

load_dotenv()
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import re
from git import Repo
import os


class GitAssistant:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        if 'to_path' not in st.session_state:
            st.session_state.to_path = None
        if 'chat_answers_history' not in st.session_state:
            st.session_state["chat_answers_history"] = []
        if 'user_prompt_history' not in st.session_state:
            st.session_state["user_prompt_history"] = []
        if 'chat_history' not in st.session_state:
            st.session_state["chat_history"] = []

    def ingest_repo(self, repo_url: str):
        pattern = r"(?<=github\.com\/)[^\/]+\/([^\.]+)"
        to_path = re.search(pattern, repo_url).group(0)
        print('This is path:', to_path)

        loader = GitLoader(
            clone_url=repo_url,
            repo_path=to_path,
            branch="main",
        )
        if not os.path.exists(to_path):
            repo = Repo.clone_from(repo_url, to_path=to_path, branch="main")
        else:
            repo = Repo(to_path)
            repo.remote().pull('main')
        print('Cloned Successfully')

        raw_documents = loader.load()
        print(f"loaded {len(raw_documents)} documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
        )

        for doc in raw_documents:
            source = doc.metadata["source"]
            cleaned_source = "/".join(source.split("/")[1:])
            doc.page_content = (
                    "FILE NAME: "
                    + cleaned_source
                    + "\n###\n"
                    + doc.page_content.replace("\u0000", "")
            )

        documents = text_splitter.split_documents(raw_documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separators=["\n\n", "\n"])
        docs = text_splitter.split_documents(documents=documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(to_path)
        print("****Loading to vectorstore done ***")
        return to_path

    def run_llm(self, query: str, to_path: str, chat_history: List[Dict[str, Any]] = []):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        new_vectorstore = FAISS.load_local(to_path, embeddings, allow_dangerous_deserialization=True)
        chat = ChatOpenAI(
            verbose=True,
            temperature=0,
        )
        qa = ConversationalRetrievalChain.from_llm(
            llm=chat, retriever=new_vectorstore.as_retriever(), return_source_documents=True
        )
        return qa.invoke({"question": query, "chat_history": chat_history})

    def ingest_repo_url(self, url):
        to_path = self.ingest_repo(url)
        st.success('Repo inserted')
        st.session_state.to_path = to_path
        return to_path

    def chat_with_git(self):
        st.title("Git Assistant")
        input_placeholder = st.empty()
        chat_placeholder = st.empty()
        form_container = st.container()

        self.initialize_session_state()

        user = st.session_state.get('user')
        if not user:
            st.warning("Please login to access this feature.")
            return

        user_id = user[0]
        session_data = get_user_session(user_id)
        if session_data:
            queries_used, last_query_date = session_data
            last_query_datetime = datetime.strptime(last_query_date, '%Y-%m-%d %H:%M:%S')
            if last_query_datetime < datetime.now() - timedelta(days=1):
                reset_query_count(user_id)
                queries_used = 0
        else:
            queries_used = 0

        st.session_state.query_count = queries_used

        with input_placeholder.form(key='repo_input'):
            url = st.text_input("GitHub Repo URL", placeholder="e.g., Enter a GitHub URL")
            submit_url = st.form_submit_button(label='Submit Repo URL')
            if url and submit_url:
                self.ingest_repo_url(url)

        to_path = st.session_state.to_path

        if to_path:
            st.write(f"Current Query Count: {st.session_state.query_count}/10")

            if st.session_state.query_count < 10:
                with form_container.form(key='input_form', clear_on_submit=True):
                    prompt = st.text_input("Prompt", placeholder="e.g., Tell me something about this repo")
                    submit_button = st.form_submit_button(label='Submit')

                if submit_button and prompt:
                    st.session_state.query_count += 1
                    update_session_query_count(user_id, st.session_state.query_count)
                    # print("After Query:", st.session_state.query_count)  # Debug print
                    with st.spinner("Generating response..."):
                        generated_response = self.run_llm(
                            query=prompt, chat_history=st.session_state["chat_history"], to_path=to_path
                        )

                        response_text = generated_response["answer"]
                        st.session_state.chat_history.append((prompt, response_text))
                        st.session_state.user_prompt_history.append(prompt)
                        st.session_state.chat_answers_history.append(response_text)
            else:
                form_container.write("Input is gone. You are out of allowed query limits for the session")

        with chat_placeholder.container():
            if st.session_state["chat_answers_history"]:
                for generated_response, user_query in zip(
                        st.session_state["chat_answers_history"],
                        st.session_state["user_prompt_history"],
                ):
                    message(user_query, is_user=True)
                    message(generated_response)
        st.markdown(
            """
            <script>
            var element = document.getElementById("end-of-chat");
            element.scrollIntoView({behavior: "smooth"});
            </script>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    assistant = GitAssistant()
    assistant.chat_with_git()
