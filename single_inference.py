import argparse
import os
import sys
import re
from functools import partial
import time

import torch
torch.set_num_threads(1)
import torchaudio
import torchaudio.functional as F
import whisper
import opencc
from hyperpyyaml import load_hyperpyyaml
from huggingface_hub import snapshot_download
from g2pw import G2PWConverter

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.frontend_utils import (contains_chinese, replace_blank, replace_corner_mark,remove_bracket, spell_out_number, split_paragraph)
from utils.word_utils import word_to_dataset_frequency, char2phn, always_augment_chars


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

####new normalize
class CustomCosyVoiceFrontEnd(CosyVoiceFrontEnd):
    def text_normalize_new(self,text, split=False):
        text = text.strip()
        def split_by_brackets(input_string):
            # Use regex to find text inside and outside the brackets
            inside_brackets = re.findall(r'\[(.*?)\]', input_string)
            outside_brackets = re.split(r'\[.*?\]', input_string)
            
            # Filter out empty strings from the outside list (result of consecutive brackets)
            outside_brackets = [part for part in outside_brackets if part]
            
            return inside_brackets, outside_brackets
        
        def text_normalize_no_split(text, is_last=False):
            text = text.strip()
            text_is_terminated = text[-1] == "。"
            if contains_chinese(text):
                if self.use_ttsfrd:
                    text = self.frd.get_frd_extra_info(text, 'input')
                else:
                    text = self.zh_tn_model.normalize(text)
                if not text_is_terminated and not is_last:
                    text = text[:-1]
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "、")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,]+$', '。', text)
            else:
                if self.use_ttsfrd:
                    text = self.frd.get_frd_extra_info(text, 'input')
                else:
                    text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
            return text
        
        def join_interleaved(outside, inside):
            # Ensure the number of parts match between outside and inside
            result = []
            
            # Iterate and combine alternating parts
            for o, i in zip(outside, inside):
                result.append(o + '[' + i + ']')
            
            # Append any remaining part (if outside is longer than inside)
            if len(outside) > len(inside):
                result.append(outside[-1])
            
            return ''.join(result)
        inside_brackets, outside_brackets = split_by_brackets(text)
        for n in range(len(outside_brackets)):
            e_out = text_normalize_no_split(outside_brackets[n],is_last = n == len(outside_brackets) - 1)
            outside_brackets[n] = e_out
            
        text = join_interleaved(outside_brackets, inside_brackets)
        if split is False:
            return text
        return texts
    
    
    def frontend_zero_shot(self, prompt_text, prompt_speech_16k , prompt_feature_path: str):
        print("[BreezyVoiceX] >>> frontend_zero_shot 開始")
        t0 = time.time()
        # tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        t1 = time.time()
        print(f"[BreezyVoiceX] text token done in {t1 - t0:.2f}s")

        t2 = time.time()
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        t3 = time.time()
        print(f"[BreezyVoiceX] resample done in {t3 - t2:.2f}s")

        t4 = time.time()
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        t5 = time.time()
        print(f"[BreezyVoiceX] speech_feat done in {t5 - t4:.2f}s")

        t6 = time.time()
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        t7 = time.time()
        print(f"[BreezyVoiceX] speech_token done in {t7 - t6:.2f}s")

        t8 = time.time()
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        t9 = time.time()
        print(f"[BreezyVoiceX] speaker embedding done in {t9 - t8:.2f}s")

        model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
        
        torch.save(model_input, prompt_feature_path)

####model
class CustomCosyVoiceModel(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def inference(self, text, text_len, flow_embedding, llm_embedding=torch.zeros(0, 192),
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32), prompt_text_len=torch.zeros(1, dtype=torch.int32),
                  llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), llm_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), flow_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32)):

        print("[BreezyVoiceX] ========== Start Inference ==========")
        total_start = time.time()

        # [1] LLM acoustic token generation
        print("[BreezyVoiceX] LLM start")
        t0 = time.time()

        tts_speech_token = self.llm.inference(text=text.to(self.device),
                                              text_len=text_len.to(self.device),
                                              prompt_text=prompt_text.to(self.device),
                                              prompt_text_len=prompt_text_len.to(self.device),
                                              prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                              prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
                                              embedding=llm_embedding.to(self.device),
                                              beam_size=1,
                                              sampling=25,
                                              max_token_text_ratio=30,
                                              min_token_text_ratio=3)
        
        t1 = time.time()
        print(f"[BreezyVoiceX] LLM done in {t1 - t0:.2f}s")

        # [2] Flow model to mel
        print("[BreezyVoiceX] FLOW start")
        t2 = time.time()

        tts_mel = self.flow.inference(token=tts_speech_token,
                                      token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
                                      prompt_token=flow_prompt_speech_token.to(self.device),
                                      prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                      prompt_feat=prompt_speech_feat.to(self.device),
                                      prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                      embedding=flow_embedding.to(self.device))
        t3 = time.time()
        print(f"[BreezyVoiceX] FLOW done in {t3 - t2:.2f}s")

        # [3] Vocoder to waveform
        print("[BreezyVoiceX] HIFT (Vocoder) start")
        t4 = time.time()

        tts_speech = self.hift.inference(mel=tts_mel).cpu()

        t5 = time.time()
        print(f"[BreezyVoiceX] HIFT done in {t5 - t4:.2f}s")

        # [4] Total time
        total_end = time.time()
        print(f"[BreezyVoiceX] TOTAL inference time: {total_end - total_start:.2f}s")

        torch.cuda.empty_cache()
        return {'tts_speech': tts_speech}
###CosyVoice
class CustomCosyVoice:

    def __init__(self, model_dir):
        #assert os.path.exists(model_dir), f"model path '{model_dir}' not exist, please check the path: pretrained_models/CosyVoice-300M-zhtw"
        instruct = False
        
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        print("model", model_dir)
        self.model_dir = model_dir
        
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CustomCosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          model_dir,
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          instruct,
                                          configs['allowed_special'])
        self.model = CustomCosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id):
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_sft(i, spk_id)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}
    
    def inference_zero_shot_no_normalize(self, tts_text, prompt_feature_path: str):
        prompt_features = torch.load(prompt_feature_path, map_location='cpu')

        tts_speeches = []
        for i in re.split(r'(?<=[？！。.?!])\s*', tts_text):
            if not len(i):
                continue
            print("Synthesizing:", i)

            tts_text_token, tts_text_token_len = self.frontend._extract_text_token(i)

            model_input = {
                'text': tts_text_token,
                'text_len': tts_text_token_len,
                'prompt_text': prompt_features['prompt_text'],
                'prompt_text_len': prompt_features['prompt_text_len'],
                'llm_prompt_speech_token': prompt_features['llm_prompt_speech_token'],
                'llm_prompt_speech_token_len': prompt_features['llm_prompt_speech_token_len'],
                'flow_prompt_speech_token': prompt_features['flow_prompt_speech_token'],
                'flow_prompt_speech_token_len': prompt_features['flow_prompt_speech_token_len'],
                'prompt_speech_feat': prompt_features['prompt_speech_feat'],
                'prompt_speech_feat_len': prompt_features['prompt_speech_feat_len'],
                'llm_embedding': prompt_features['llm_embedding'],
                'flow_embedding': prompt_features['flow_embedding'],
            }

            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])

        return {'tts_speech': torch.concat(tts_speeches, dim=1)}
        
####wav2text
def transcribe_audio(audio_file):
    #model = whisper.load_model("base")
    #result = model.transcribe(audio_file)
    from transformers import pipeline

    # Load Whisper model
    whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")

    # Perform ASR on an audio file
    result = whisper_asr(audio_file)

    converter = opencc.OpenCC('s2t')
    traditional_text = converter.convert(result["text"])
    return traditional_text

def get_bopomofo_rare(text, converter):
    res = converter(text)
    text_w_bopomofo = [x for x in zip(list(text), res[0])]
    reconstructed_text = ""
    
    for i in range(len(text_w_bopomofo)):
        t = text_w_bopomofo[i]
        try:
            next_t_char = text_w_bopomofo[i+1][0]
        except:
            next_t_char = None
        #print(t[0], word_to_dataset_frequency[t[0]], t[1])
        
        if word_to_dataset_frequency[t[0]] < 500 and t[1] != None and next_t_char != '[':
            # Add the char and the pronunciation
            reconstructed_text += t[0] + f"[:{t[1]}]"
        
        elif len(char2phn[t[0]]) >= 2:
            if t[1] != char2phn[t[0]][0] and (word_to_dataset_frequency[t[0]] < 10000 or t[0] in always_augment_chars) and next_t_char != '[':  # Not most common pronunciation
                # Add the char and the pronunciation
                reconstructed_text += t[0] + f"[:{t[1]}]"
            else:
                reconstructed_text += t[0]
            #print("DEBUG, multiphone char", t[0], char2phn[t[0]])
        else:
            # Add only the char
            reconstructed_text += t[0]
    
    #print("Reconstructed:", reconstructed_text)
    return reconstructed_text

import re

def parse_transcript(text, end):
    pattern = r"<\|(\d+\.\d+)\|>([^<]+)<\|(\d+\.\d+)\|>"
    matches = re.findall(pattern, text)
    
    parsed_output = [(float(start), float(end), content.strip()) for start, content,end in matches]
    count0 = 0
    for i in range(len(parsed_output)):
        if parsed_output[i][0] == 0:
            count0 += 1
        if count0 >= 2:
            parsed_output = parsed_output[:i]
            break
    #print("a", parsed_output)
    for i in range(len(parsed_output)):
        if parsed_output[i][0] >= end:
            parsed_output = parsed_output[:i]
            break
    #print("b", parsed_output)
    for i in range(len(parsed_output)):
        if parsed_output[i][0] < end - 15:
            continue
        else:
            parsed_output = parsed_output[i:]
            break
    #print("c", parsed_output)
    start = parsed_output[0][0]
    parsed_output = "".join([p[2] for p in parsed_output])
    return parsed_output, start

def generate_prompt_features(speaker_prompt_audio_path, prompt_text, prompt_feature_path, cosyvoice, bopomofo_converter):
    prompt_speech_16k = load_wav(speaker_prompt_audio_path, 16000)

    # Normalize
    prompt_text = cosyvoice.frontend.text_normalize_new(prompt_text, split=False)
    prompt_text_bopomo = get_bopomofo_rare(prompt_text, bopomofo_converter)
    print("[BreezyVoiceX] Normalized speaker prompt text:", prompt_text_bopomo)

    cosyvoice.frontend.frontend_zero_shot(prompt_text_bopomo, prompt_speech_16k, prompt_feature_path)
    print(f"[BreezyVoiceX] Prompt features saved to {prompt_feature_path}")


def synthesize_from_features(content_to_synthesize, prompt_feature_path, output_path, cosyvoice, bopomofo_converter):
    content_to_synthesize = cosyvoice.frontend.text_normalize_new(content_to_synthesize, split=False)
    content_to_synthesize_bopomo = get_bopomofo_rare(content_to_synthesize, bopomofo_converter)
    print("[BreezyVoiceX] Content to be synthesized:", content_to_synthesize_bopomo)

    start = time.time()
    output = cosyvoice.inference_zero_shot_no_normalize(content_to_synthesize_bopomo, prompt_feature_path)
    end = time.time()

    print("[BreezyVoiceX] Elapsed time:", end - start)
    print("[BreezyVoiceX] Generated audio length:", output['tts_speech'].shape[1] / 22050, "seconds")
    torchaudio.save(output_path, output['tts_speech'], 22050)
    print(f"[BreezyVoiceX] Generated voice saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run BreezyVoice text-to-speech with speaker feature caching support")

    parser.add_argument("--mode", choices=["cache", "synthesize"], required=True,
                        help="Choose mode: 'cache' to prepare speaker feature, 'synthesize' to generate speech using cached features.")
    
    parser.add_argument("--content_to_synthesize", type=str, help="Text to synthesize (required in synthesize mode)")
    parser.add_argument("--speaker_prompt_audio_path", type=str, help="Path to speaker prompt audio file (required in cache mode)")
    parser.add_argument("--speaker_prompt_text_transcription", type=str, required=False, help="Transcribed prompt text (optional if using Whisper)")
    
    parser.add_argument("--prompt_feature_path", type=str, default="cache/prompt.pt", help="Path to save/load speaker prompt feature (.pt)")
    parser.add_argument("--output_path", type=str, default="results/output.wav", help="Output audio path")
    parser.add_argument("--model_path", type=str, default="MediaTek-Research/BreezyVoice-300M", help="Model directory or Hugging Face model id")

    args = parser.parse_args()
    
    cosyvoice = CustomCosyVoice(args.model_path)
    bopomofo_converter = G2PWConverter()

    if args.mode == "cache":
        assert args.speaker_prompt_audio_path, "You must provide --speaker_prompt_audio_path in cache mode"
        if args.speaker_prompt_text_transcription:
            prompt_text = args.speaker_prompt_text_transcription
        else:
            prompt_text = transcribe_audio(args.speaker_prompt_audio_path)
        generate_prompt_features(args.speaker_prompt_audio_path, prompt_text, args.prompt_feature_path, cosyvoice, bopomofo_converter)

    elif args.mode == "synthesize":
        assert args.content_to_synthesize, "You must provide --content_to_synthesize in synthesize mode"
        synthesize_from_features(args.content_to_synthesize, args.prompt_feature_path, args.output_path, cosyvoice, bopomofo_converter)

if __name__ == "__main__":
    main()




