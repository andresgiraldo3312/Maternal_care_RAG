from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import pandas as pd
import fire
import yaml
import os

model_object = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0" : "Bedrock",
    "cohere.command-r-plus-v1:0" : "Bedrock",
    "ai21.jamba-instruct-v1:0" : "Bedrock",
    "meta.llama3-70b-instruct-v1:0" : "Bedrock",
    "meta.llama3-8b-instruct-v1:0" : "Bedrock",
    "mistral.mistral-large-2402-v1:0" : "Bedrock",
    "gemma2:2b"  : "Ollama",
    "llama3.1:8b" : "Ollama",
    "mistral:7b" : "Ollama",
    "qwen:7b" : "Ollama",
    "gpt-4o" : "OpenAI"
}

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main(
    knowledged_base_path: str,
    question_path: str,
    answer_path: str,
    ground_truth_path: str,
    config_path: str
):
    print("Starting RAG application")
    
    # Load the YAML configuration file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    print("--- Configuration:", config)
    print("--- Loading base knowledge from: ", knowledged_base_path)
    
    # List all PDF files in the specified folder
    pdf_files = [f for f in os.listdir(knowledged_base_path) if f.endswith('.pdf')]
    
    pages = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(knowledged_base_path, pdf_file)
        print("--- Processing file: ", pdf_path)
        loader = PyPDFLoader(pdf_path)
        pages.extend(loader.load_and_split())
    
    print("--- Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])
    splits = text_splitter.split_documents(pages)
    
    print("--- Creating database")
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={'k': config["top_k_retriever"]}
    )
    
    print("--- Setting up model and chain")
    prompt = ChatPromptTemplate([
        ("system", config["system_prompt"]),
        ("human", config["rag_prompt"] + """
    Pregunta: {question} 
    Documentos: {context} 
    Respuesta:"""),
    ])
    
    if model_object[config["llm_model"]] == "Bedrock":
        print("--- Using Bedrock model")
        llm = ChatBedrock(
            model_id=config["llm_model"],
            model_kwargs=dict(temperature=config["temperature"]),
        )
    elif model_object[config["llm_model"]] == "Ollama":
        print("--- Using Ollama model")
        llm = OllamaLLM(
            model=config["llm_model"],
            model_kwargs=dict(temperature=config["temperature"]),
        )
    elif model_object[config["llm_model"]] == "OpenAI":
        print("--- Using OpenAI model")
        llm = ChatOpenAI(
            model=config["llm_model"],
            temperature=config["temperature"],
        )
    
    print("--- Running the chain")
    rag_chain = (
        {"context": retriever | format_docs, 
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("--- Ready to answer questions")
    
    # Load the questions from the specified file
    df_question = pd.read_excel(
        question_path,
        header = None,
        names = ['question'],
        )
    
    df_ground_truth = pd.read_excel(
        ground_truth_path,
        header = None,
        names = ['ground_truth'],
        )
    
    # Iterate over the questions and get the answers
    answer_list = []
    for index, row in tqdm(df_question.iterrows(), total=df_question.shape[0], desc="Procesando preguntas"):
        ans = rag_chain.invoke(row['question'])
        answer_list.append(ans)
    
    answer_file = answer_path + config["llm_model"] + ".xlsx"

    # Save top-k
    topks = []

    for query in df_question['question']:
        topk = []
        for i in range(5):
            topk.append(retriever.vectorstore.similarity_search(query, k=5)[i].page_content)
        topks.append(topk)
    

    print("--- Saving answers to file: ", answer_file)
    df_question['answer'] = answer_list
    df_question['contexts'] = topks
    df_question['ground_truth'] = df_ground_truth
    df_question.to_excel(answer_file, index=False)
    
if __name__ == "__main__":
    fire.Fire(main)

