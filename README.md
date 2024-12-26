# Simple Retrieval Augmented Generation [RAG] using groq API

## Overview
This repository contains a Gradio-based interface that enables users to ask questions about documents loaded into a vector database. The framework uses the LangChain library for document loading, text splitting, and retrieval-based question-answering (QA). It is tailored for answering queries on documents, such as research papers, with the ability to incorporate additional context for enhanced retrieval.

The primary example in this project demonstrates the creation of a chatbot for answering questions about AI agents' research papers.

## Key Features
- **Document Loader**: Handles PDF documents using LangChain's `UnstructuredPDFLoader`.
- **Text Splitting**: Utilizes `CharacterTextSplitter` for splitting documents into manageable chunks for vectorization.
- **Vector Store**: Builds a persistent Chroma-based vector database using `HuggingFaceEmbeddings`.
- **RetrievalQA**: Enables retrieval-augmented generation (RAG) for accurate and contextually relevant responses.
- **Gradio Interface**: User-friendly interface to ask questions based on the document content.

## How to Run

### Prerequisites
Ensure you have Python installed along with the required dependencies.

1. Clone this repository:
   ```bash
   https://github.com/mayur-ml/Simple-Retrieval-Augmented-Generation-RAG-.git
   cd Simple-Retrieval-Augmented-Generation-RAG-
   ```

2. Install the dependencies:
   ```bash
   pip install -r environment.yaml
   ```

3. Run the application:
   ```bash
   python main.py
   ```

4. Open your browser and navigate to the local Gradio instance (typically `http://127.0.0.1:7860`).

   
 ![Demo Interface](https://github.com/mayur-ml/Simple-Retrieval-Augmented-Generation-RAG-/blob/main/assets/Demo_interface.png)


## Usage

- Place your PDF documents in the `data/` directory.
- Launch the Gradio application and enter your queries in the text input box.
- The chatbot will retrieve and respond based on the loaded documents.

## Future Updates
### Planned Enhancements
1. **Support for Multiple Document Types**:
   Extend the loader to handle a variety of document formats, such as `.txt`, `.docx`, and `.csv`.

2. **Contextual Retrieval**:
   Add contextual tags during text splitting to improve retrieval accuracy for multi-document use cases. Inspired by Anthropic's [blog post on contextual retrieval](https://www.anthropic.com/news/contextual-retrieval).

3. **Improved Models**:
   Support additional large language models and fine-tuned embeddings for domain-specific use cases.

4. **Interactive Context Management**:
   Allow users to dynamically add or modify context during QA sessions for better personalization.


## Contributing
Feel free to open issues or submit pull requests if you have suggestions or improvements.

## Contact
For any questions, reach out to `myringole@gmail.com`.

## License
This project is licensed under the [MIT License](./LICENSE).

