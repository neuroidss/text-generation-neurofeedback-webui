source env/bin/activate

#python server.py --chat --model TheBloke_llama2_7b_chat_uncensored-GGML --extensions silero_tts whisper_stt --sdp-attention

#python server.py --chat --model TheBloke_vicuna-AlekseyKorshuk-7B-GPTQ-4bit-128g --sdp-attention
python server.py --chat --model TheBloke_vicuna-AlekseyKorshuk-7B-GPTQ-4bit-128g --extensions silero_tts whisper_stt --sdp-attention

#python server.py --chat --model TehVenom_Pygmalion-7b-4bit-GPTQ-Safetensors --extensions silero_tts whisper_stt --sdp-attention
#python server.py --chat --model anon8231489123_vicuna-13b-GPTQ-4bit-128g --extensions silero_tts whisper_stt --sdp-attention
#python server.py --chat --extensions silero_tts whisper_stt --sdp-attention
