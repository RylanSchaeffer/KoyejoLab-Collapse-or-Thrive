python3 api_call.py \
  --requests_filepath eval_active_2000_baseline.jsonl \
  --save_filepath results_active_2000_baseline.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 4000 \
  --max_tokens_per_minute 30000 \
  --api_key YOUR_OPEN_AI_API_KEY