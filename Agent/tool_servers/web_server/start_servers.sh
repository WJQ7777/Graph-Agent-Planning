#!/bin/bash

: "${SERVER_HOST:?Please set SERVER_HOST in environment.sh}"
: "${CRAWL_PAGE_PORT:?Please set CRAWL_PAGE_PORT in environment.sh}"
: "${WEBSEARCH_PORT:?Please set WEBSEARCH_PORT in environment.sh}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$DIR/logs/$SERVER_HOST";   mkdir -p "$LOG_DIR"

cmd=$1
if [[ ! "$cmd" =~ ^(start|stop|status|test)$ ]]; then
  echo "Usage: $0 [start|stop|status|test]"
  echo "  start  : Start training mode servers (SerperV2, CrawlPageV2)"
  echo "  stop   : Stop all servers"
  echo "  status : Check server status"
  echo "  test   : Test server functionality"
  exit 1
fi

# =====================================================================================================================
#                                      start
# =====================================================================================================================

if [[ "$cmd" == "start" ]]; then
  echo "Starting training mode servers..."
  
  # CrawlPageV2
  logf="$LOG_DIR/CrawlPageV2_$CRAWL_PAGE_PORT.log"
  if netstat -tulnp | grep -q ":$CRAWL_PAGE_PORT "; then
    echo "CrawlPageV2 is already running on port $CRAWL_PAGE_PORT"
  else
    echo "Starting CrawlPageV2 on port $CRAWL_PAGE_PORT..."
    nohup python -u "$DIR/v2/crawl_page_server_v2.py" > "$logf" 2>&1 &
  fi

  # SerperCacheV2
  logf="$LOG_DIR/SerperCacheV2_$WEBSEARCH_PORT.log"
  if netstat -tulnp | grep -q ":$WEBSEARCH_PORT "; then
    echo "SerperCacheV2 is already running on port $WEBSEARCH_PORT"
  else
    echo "Starting SerperCacheV2 on port $WEBSEARCH_PORT..."
    nohup python -u "$DIR/v2/cache_serper_server_v2.py" > "$logf" 2>&1 &
  fi

# =====================================================================================================================
#                                      test
# =====================================================================================================================

elif [[ "$cmd" == "test" ]]; then
  echo "--------------------Starting test for serper cache v2 ------------------"
  python -u "$DIR/server_tests/test_cache_serper_server_v2.py"           "http://$SERVER_HOST:$WEBSEARCH_PORT/search"
  echo "-------------------------Test completed--------------------------"
  echo "--------------------Starting test for crawl page v2 -------------------"
  python -u "$DIR/server_tests/test_crawl_page_simple_v2.py"           "http://$SERVER_HOST:$CRAWL_PAGE_PORT/crawl_page"
  echo "-------------------------Test completed--------------------------"

# =====================================================================================================================
#                                      stop
# =====================================================================================================================

elif [[ "$cmd" == "stop" ]]; then
  echo "Stopping all servers..."
  
  # CrawlPageV2 - stop by port
  port_info=$(netstat -tulnp | grep ":$CRAWL_PAGE_PORT ")
  if [[ -n "$port_info" ]]; then
    # Extract PID from netstat output (format: protocol recv-q send-q local-address foreign-address state pid/program)
    pids=($(echo "$port_info" | awk '{print $7}' | grep -o '^[0-9]*' | sort -u))
    echo "Found ${#pids[@]} process(es) on port $CRAWL_PAGE_PORT"
    for pid in "${pids[@]}"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping CrawlPageV2 process (PID $pid)"
        if kill "$pid" 2>/dev/null; then
          # Wait up to 5 seconds for graceful shutdown
          for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
              echo "Stopped CrawlPageV2 (PID $pid)"
              break
            fi
            sleep 0.5
          done
          # Force kill if still running
          if kill -0 "$pid" 2>/dev/null; then
            if kill -9 "$pid" 2>/dev/null; then
              echo "Force stopped CrawlPageV2 (PID $pid)"
            else
              echo "Warning: Unable to force stop CrawlPageV2 (PID $pid)"
            fi
          fi
        else
          echo "Warning: Unable to stop CrawlPageV2 (PID $pid)"
        fi
      fi
    done
  else
    echo "No processes found on port $CRAWL_PAGE_PORT"
  fi

  # SerperCacheV2 - stop by port  
  port_info=$(netstat -tulnp | grep ":$WEBSEARCH_PORT ")
  if [[ -n "$port_info" ]]; then
    # Extract PID from netstat output (format: protocol recv-q send-q local-address foreign-address state pid/program)
    pids=($(echo "$port_info" | awk '{print $7}' | grep -o '^[0-9]*' | sort -u))
    echo "Found ${#pids[@]} process(es) on port $WEBSEARCH_PORT"
    for pid in "${pids[@]}"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping SerperCacheV2 process (PID $pid)"
        if kill "$pid" 2>/dev/null; then
          # Wait up to 5 seconds for graceful shutdown
          for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
              echo "Stopped SerperCacheV2 (PID $pid)"
              break
            fi
            sleep 0.5
          done
          # Force kill if still running
          if kill -0 "$pid" 2>/dev/null; then
            if kill -9 "$pid" 2>/dev/null; then
              echo "Force stopped SerperCacheV2 (PID $pid)"
            else
              echo "Warning: Unable to force stop SerperCacheV2 (PID $pid)"
            fi
          fi
        else
          echo "Warning: Unable to stop SerperCacheV2 (PID $pid)"
        fi
      fi
    done
  else
    echo "No processes found on port $WEBSEARCH_PORT"
  fi

# =====================================================================================================================
#                                      status
# =====================================================================================================================

else
  # CrawlPageV2
  if netstat -tulnp | grep -q ":$CRAWL_PAGE_PORT "; then
    echo "CrawlPageV2 is running on port $CRAWL_PAGE_PORT"
  else
    echo "CrawlPageV2 is not running"
  fi

  # SerperCacheV2
  if netstat -tulnp | grep -q ":$WEBSEARCH_PORT "; then
    echo "SerperCacheV2 is running on port $WEBSEARCH_PORT"
  else
    echo "SerperCacheV2 is not running"
  fi
fi