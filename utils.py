from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
from sentence_transformers import SentenceTransformer
from dotenv import main
from pinecone import Pinecone
from openai import OpenAI
import os
main.load_dotenv()
PINECONE_API = os.getenv("pineconeapi")


# code to convert pdf to txt file for ease of use
def convert_pdf2txt(filepath: str = "sample_data.pdf"):
    content = PyPDF2.PdfReader(filepath)
    text = ""
    for i in range(0, len(content.pages)):
        text = text + content.pages[i].extract_text()
    with open("sample_data.txt", "w") as f:
        f.write(text)


# pinecone, hf and openai init
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API)
index_hf = pc.Index("rag-llm")
index_openai = pc.Index("rag-llm-openai")


# code for embeddings using hf
def create_database_hf(file_path="sample_data.txt"):
    docs = TextLoader(file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    content = [splits[i].page_content for i in range(0, len(splits))]

    vectors = []
    for i in range(0, len(splits)):
        vector = {
            "id": str(i),
            "values": model.encode(splits[i].page_content),
            "metadata": {"content": content[i]},
        }
        vectors.append(vector)
    index_hf.upsert(vectors=vectors)
    return index_hf, content


# for embedddings using openai
def create_database_openai(file_path="sample_data.txt", index="rag-llm"):
    docs = TextLoader("sample_data.txt").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    content = [splits[i].page_content for i in range(0, len(splits))]
    embeddings = client.embeddings.create(model="text-embedding-ada-002", input=content)
    vectors = []
    for i in range(0, len(splits)):
        vector = {
            "id": str(i),
            "values": embeddings.data[i].embedding,
            "metadata": {"content": content[i]},
        }
        vectors.append(vector)
    index_openai.upsert(vectors=vectors)
    return index, content


def find_context(prompt: str, method="openai", top_k=2):
    if method == "openai":
        result = client.embeddings.create(
            model="text-embedding-ada-002",
            input=prompt,
        )
        embed = result.data[0].embedding
        content = index_openai.query(
            vector=embed, top_k=top_k, include_values=False, include_metadata=True
        )
        if top_k == 1:
            return (
                content["matches"][0]["id"],
                content["matches"][0]["metadata"]["content"],
            )
        elif top_k > 0:
            content_list = []
            index_list = []
            for i in range(0, top_k):
                index_list.append(content["matches"][i]["id"])
                content_list.append(content["matches"][i]["metadata"]["content"])
            return index_list, content_list
    elif method == "hf":
        embed = model.encode(prompt).tolist()
        content = index_hf.query(
            vector=embed, top_k=top_k, include_values=False, include_metadata=True
        )
        if top_k == 1:
            return (
                content["matches"][0]["id"],
                content["matches"][0]["metadata"]["content"],
            )
        elif top_k > 0:
            content_list = []
            index_list = []
            for i in range(0, top_k):
                index_list.append(content["matches"][i]["id"])
                content_list.append(content["matches"][i]["metadata"]["content"])
            return index_list, content_list
    else:
        raise ValueError("Method can only be hf or openai")
    
def QNA(prompt:str,method = "openai"):
    if(method == "openai"):
        id, content = find_context(prompt)
    elif(method == "hf"):
        id, content = find_context(prompt,method = "hf")
    else:
        raise ValueError("Method can only be hf or openai")
    system_prompt = f"You are Given two pages from a book called 48 laws of power. Answer the users questions based on the context from these two pages: \n\n page 1 : {content[0]} \n\n page 2 : {content[1]}"
    user_prompt = prompt
    msg = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}],
    )
    return msg.choices[0].message.content
