# nohup bash launch_rag_server.sh > launch_rag_server_8000.log 2>&1 &

file_path=/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/wiki_rag
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/home/jiaqi/Agent_Foundation_Models/AFM/models/e5-base-v2
port=8008

python /home/jiaqi/Agent_Foundation_Models/AFM/tool_servers/wiki_server/wiki_rag_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --host $SERVER_HOST \
                                            --port $port
