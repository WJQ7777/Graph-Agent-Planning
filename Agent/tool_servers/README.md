## Wiki Server
### Retriever environment
If you would like to call a local retriever as the search engine, you can install the environment as follows. (We recommend using a seperate environment.)
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```

### download the index file
```bash
cd AFM/tool_servers/wiki_server
# save_path=/the/path/to/save
python ./download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

### start the wiki search tool
set the path of the index file in `AFM/tool_servers/wiki_server/launch_rag_server.sh`, then start wiki_server with
```bash
bash AFM/tool_servers/wiki_server/launch_rag_server.sh
```


## Web Server
We have two servers for web agent:
- **CrawlPageServer**: A web crawling service that takes a URL and returns the page content.
- **SerperCacheServer**: Caches results from the Serper search engine to reduce latency and avoid duplicate searches.

You should first `source environment.sh` to set env variables. Then, use `start_servers_v2.sh` script to manage the servers. After start the server, you should execute test to ensure that the servers start correctly. You can check logs in the `tool_servers/web_server/logs` dir.

- **Start Servers**:
  ```bash
  bash AFM/tool_servers/web_server/start_servers.sh start
  ```
  This command starts the `CrawlPageV2` and `SerperCacheV2` services.

- **Stop Servers**:
  ```bash
  bash AFM/tool_servers/web_server/start_servers.sh stop
  ```

- **Check Server Status**:
  ```bash
  bash AFM/tool_servers/web_server/start_servers.sh status
  ```

- **Run Tests**:
  ```bash
  bash AFM/tool_servers/web_server/start_servers.sh test
  ```
  This command runs the test scripts in the `server_tests` directory to verify that the services are working correctly.
