# ATB Chatbot

ATB Chatbot is a command-line and server-based chatbot for interacting with the **AllTheBacteria (ATB)** knowledge base. It supports crawling source materials, ingesting them into a database, and launching a local chat server backed by an LLM for question answering.

## Overview

This tool provides a simple workflow to:

1. Install dependencies
2. Crawl the ATB knowledge base into a local directory
3. Ingest the crawled content into a vector database
4. Launch a chatbot server to query the knowledge base

## Installation

Run the installation script:

```bash
bash install.sh
```

## Workflow

### 1. Crawl the ATB knowledge base

This step collects and stores the ATB materials locally.

```bash
atb-chat crawl --crawl-dir ATB-material
```

### 2. Ingest the crawled content into the vector database

This step processes the crawled materials and builds the local vector database used for retrieval.

```bash
atb-chat ingest --crawl-dir ATB-material --db-dir atb_chat_db
```

### 3. Start the chatbot server

This launches the chatbot using the ingested database and the specified model.

```bash
atb-chat server --db-dir atb_chat_db --model gemma3:12b
```

## Example end-to-end usage

```bash
bash install.sh
atb-chat crawl --crawl-dir ATB-material
atb-chat ingest --crawl-dir ATB-material --db-dir atb_chat_db
atb-chat server --db-dir atb_chat_db --model gemma3:12b
```

## Directory structure

After running the workflow, you should expect a structure similar to:

```text
.
├── install.sh
├── ATB-material/
└── atb_chat_db/
```

- `ATB-material/` contains the crawled source content
- `atb_chat_db/` contains the processed vector database

## Notes

- Run the crawl step again if the ATB knowledge base has been updated and you want to refresh local content.
- Re-run the ingest step after crawling new or updated materials.
- You can swap the model in the server command if other supported local models are available in your environment.

## License

MIT

## Contacts
- Qianxuan(Sean) She, email = sheq@chop.edu
- Ahmed M Moustafa, email = moustafaam@chop.edu
