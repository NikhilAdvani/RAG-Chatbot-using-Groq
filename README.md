# RAG-Chatbot-using-Groq

## Overview

Welcome to the RAG Chatbot project! This chatbot leverages the LangChain framework and integrates multiple tools to provide accurate and detailed responses to user queries. By combining the power of the Groq inference engine, the open-source Llama-3 model, and ChromaDB, this chatbot ensures high performance and versatility in information retrieval.

## Features

- **Groq Inference Engine**: Ensures rapid response times for inference.
- **Llama-3 Model**: Utilizes an open-source large language model for generating responses.
- **ChromaDB**: Serves as the vector database for storing and retrieving embeddings.
- **Wikipedia Tool**: Searches Wikipedia for relevant information.
- **PDF Search Tool**: Retrieves information from PDF documents.
- **Arxiv Tool**: Provides information about research papers from Arxiv.

## Technologies Used

- **LangChain**: Framework for building language model applications.
- **Streamlit**: For building an interactive web interface.
- **PyPDFDirectoryLoader**: To load and process PDF documents.
- **OpenAIEmbeddings**: For creating and managing embeddings.
- **RecursiveCharacterTextSplitter**: For splitting documents into manageable chunks.
- **Chroma**: Vector database for efficient document retrieval.
- **WikipediaAPIWrapper**: Utility for interacting with the Wikipedia API.
- **ArxivAPIWrapper**: Utility for interacting with the Arxiv API.


### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/NikhilAdvani/RAG-Chatbot-using-Groq.git
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   ```

3. **Install the required packages**

4. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your Groq API key:

   ```env
   GROQ_API_KEY = your_groq_api_key
   ```

5. **Run the Application**

   ```bash
   streamlit run main.py
   ```

## Usage

1. Open your browser and navigate to `http://localhost:8501`.
2. Enter your query in the input box and click "Get Answer".
3. The chatbot will process your query using the appropriate tool and display the response along with the response time.

## Project Structure

- `main.py`: Main application file containing the Streamlit setup and chatbot logic.
- `us_census_data/`: Directory containing PDF documents for the PDF search tool.
- `.env`: File to store environment variables.
  
```

## Contributing

Contributions/feedbacks are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.

## Acknowledgments

- A big thank you to Krish Naik's youtube tutorials (https://www.youtube.com/@krishnaik06) for guiding me through this project.
- Thanks to the LangChain community for providing the tools and support for building this project.
- Special thanks to Groq for their high-performance inference engine.

---
