export SERVER_HOST=$(hostname -I | awk '{print $1}')
export CRAWL_PAGE_PORT=9000
export WEBSEARCH_PORT=9001

# =====================================================================================================================
#                                      wiki_rag_server
# =====================================================================================================================
export WIKI_RAG_SERVER_URL="http://$SERVER_HOST:8008/retrieve"
# =====================================================================================================================
#                                      GRM
# =====================================================================================================================
# for llm as judge
export GRM_BASE_URL="https://api.tensoropera.ai/v1"
export GRM_API_KEY=""
export GRM_MODEL_NAME="gpt-4o-mini"
# =====================================================================================================================
#                                      Jina
# =====================================================================================================================
# for crawl page
export JINA_BASE_URL="https://s.jina.ai/"
export JINA_API_KEY=""
# =====================================================================================================================
#                                      Serper
# =====================================================================================================================
# for web search
export WEB_SEARCH_METHOD_TYPE="serapi"
export WEB_SEARCH_SERP_NUM="10"
export WEB_SEARCH_SERPER_API_KEY=""
# =====================================================================================================================
#                                      Summary Model
# =====================================================================================================================
# for summary of crawl page content
export SUMMARY_OPENAI_API_BASE_URL="https://api.tensoropera.ai/v1"
export SUMMARY_OPENAI_API_KEY=""
export SUMMARY_MODEL="gpt-4o-mini"
# =====================================================================================================================
#                                      Code testcases
# =====================================================================================================================
# for evaluate code val datasets
export LIVECODEBENCH_DATA_PATH="$(pwd)/data/code_agent/livecodebench_testcases"

export UNI_API_URLS="https://api.tensoropera.ai/v1" # or other qwen3 api provider
export UNI_API_KEY=""