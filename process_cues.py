#!/usr/bin/env python3
"""
process_cues.py

This script recursively scans a base cues directory for projects (each cue folder must contain a "MIDI"
subfolder with .mid file(s) and a "PT/Audio Files" subfolder). It processes the MIDI and audio files,
tokenizes the audio using EnCodec for MusicGen (the 48 kHz AudioCraft model, from transformers),
and uses the OpenAI Batch API to do the classification (track→group + common category) asynchronously.

Finally, once a MIDI track is matched to an audio group and we have a final "common instrument category,"
we update both:
  - the MIDI track (in `midi_tracks.instrument_category`)
  - the associated `audio_files.instrument_category`
to ensure consistency.

Audio features are serialized as binary (numpy .npy format) for efficiency.
"""

import os
import sys
import glob
import json
import io
import re
import time
import logging  # For optional debug logging
import requests  # For file download
from concurrent.futures import ProcessPoolExecutor

# Third-Party Packages
import mido
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from thefuzz import process as fuzz_process
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# ---------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# OpenAI Batch API Setup
# ---------------------------------------------------------------------
from openai import OpenAI
import openai  # needed for the new interface

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if client.api_key is None:
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

# ---------------------------------------------------------------------
# Helper to download file content using requests
# ---------------------------------------------------------------------
def download_output_file(file_id):
    headers = {"Authorization": f"Bearer {client.api_key}"}
    url = f"https://api.openai.com/v1/files/{file_id}/content"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.content

# ---------------------------------------------------------------------
# AudioCraft EnCodec 48 kHz from transformers
# ---------------------------------------------------------------------
from audiocraft.models.encodec import CompressionModel
 
def convert_audio(audio_tensor: torch.Tensor, src_sr: int, dst_sr: int, dst_channels: int):
    """
    Resample from src_sr to dst_sr using torchaudio and adjust channel count if needed.
    """
    if src_sr != dst_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, src_sr, dst_sr)
    cur_channels = audio_tensor.shape[1]
    if cur_channels != dst_channels:
        if dst_channels == 1:
            audio_tensor = audio_tensor.mean(dim=1, keepdim=True)
        elif dst_channels == 2 and cur_channels == 1:
            audio_tensor = audio_tensor.repeat(1, 2, 1)
        else:
            logger.warning(f"Cannot convert from {cur_channels} to {dst_channels} channels.")
    return audio_tensor

# Load the 48 kHz model
global encodec_model
encodec_model = CompressionModel.get_pretrained("facebook/encodec_48khz")
encodec_model.eval()
logger.info("Loaded AudioCraft EnCodec 48k model.")

# ---------------------------------------------------------------------
# Global Configuration & Constants
# ---------------------------------------------------------------------
INSTRUMENT_CATEGORIES = [
    "Strings", "Woodwinds", "Brass", "Acoustic Guitar", "Electric Guitar",
    "Piano", "Keys", "Bells", "Harp", "Synth Pulse", "Synth Pad", "Synth Bass",
    "Low Percussion", "Mid Percussion", "High Percussion", "Drums", "Orch Percussion",
    "Bass", "Double Bass", "FX", "Choir", "Solo Vocals", "Mallets", "Plucked",
    "Sub Hits", "Guitar FX", "Orch FX", "Ticker", "Extra"
]

MARKER_KEYWORDS = ["NOTES", "CONDUCTOR", "ORCHESTRATOR", "MARKER", "MONITOR", "MIDI"]

CHANNEL_ORDER = {
    ".L": 0, ".C": 1, ".R": 2, ".Ls": 3, ".Rs": 4, ".lfe": 5, ".Lf": 5
}

INSTRUMENT_ABBREVIATIONS = {
    "STG": "Strings", "VCS": "Strings", "VLN": "Strings", "VLA": "Strings", "CBS": "Strings",
    "HARP": "Harp", "HPS": "Harp", "BASS TRB": "Brass", "TRB": "Brass", "TBNS": "Brass",
    "BASS": "Bass", "PERC": "Percussion", "TPT": "Brass", "FHN": "Brass",
    "GONG": "High Percussion", "METALS": "High Percussion", "BASS DRUM": "Low Percussion",
    "TBN": "Brass", "FHNS": "Brass", "FL": "Woodwinds", "OB": "Woodwinds", "CL": "Woodwinds",
    "BSN": "Woodwinds", "SAX": "Woodwinds", "GTR": "Guitar", "BASSOON": "Woodwinds",
    "SYN": "Synth", "FX": "Effects", "PAD": "Synth Pad", "DRONE": "Synth Pad",
    "ORG": "Keys", "UKE": "Plucked", "MARIMBA": "Mallets", "HORN": "Brass",
    "PNO": "Piano", "HITS": "Effects", "PEDRO": "Woodwinds", "MARTIN": "Electric Guitar",
    "BELL": "Bells", "CHOIR": "Choir", "PLUCK": "Plucked", "BASS CLARINET": "Woodwinds",
    "VOX": "Solo Vocals", "MALLET": "Mallets", "BOOM": "Effects",
    "TAIKO": "Low Percussion", "TKO": "Low Percussion",
    "TIMP": "Orch Percussion", "TIMPANI": "Orch Percussion",
    "SNR": "Mid Percussion", "Duduk": "Woodwinds", "SHAKU": "Woodwinds",
    "CYM": "High Percussion", "SHAKER": "High Percussion", "AMB": "Synth Pad", "AMBIENT": "Synth_Pad"
}

CATEGORY_OVERRIDES = {
    "Synth": "Synth Pad",
    "Synth Lead": "Synth Pulse",
    "Piano Pedal": "Piano",
    "Percussion": "Drums",
    "Taiko": "Low Percussion",
    "Timpani": "Orch Percussion",
    "Harp": "Harp"
}

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def clean_name(name):
    cleaned = re.sub(r'[^\w\s]', '', name)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def find_cue_directories(base_dir):
    cue_dirs = []
    for root, dirs, files in os.walk(base_dir):
        midi_dir = os.path.join(root, "MIDI")
        pt_audio_dir = os.path.join(root, "PT", "Audio Files")
        if os.path.isdir(midi_dir) and os.path.isdir(pt_audio_dir):
            mid_files = glob.glob(os.path.join(midi_dir, "*.mid"))
            if mid_files:
                cue_dirs.append(root)
    return cue_dirs

def longest_common_prefix(tokens_list):
    if not tokens_list:
        return []
    min_len = min(len(tokens) for tokens in tokens_list)
    common = []
    for i in range(min_len):
        token = tokens_list[0][i]
        if all(tokens[i] == token for tokens in tokens_list):
            common.append(token)
        else:
            break
    return common

def trim_common_prefix(names):
    if not names:
        return names
    token_lists = [name.split() for name in names]
    common_tokens = longest_common_prefix(token_lists)
    common_prefix = " ".join(common_tokens)
    trimmed = []
    if common_prefix:
        for name in names:
            if name.startswith(common_prefix):
                trimmed.append(name[len(common_prefix):].strip())
            else:
                trimmed.append(name)
    else:
        trimmed = names
    return trimmed

def trim_audio_group_names(names):
    trimmed_list = trim_common_prefix(names)
    mapping = {}
    for original, t in zip(names, trimmed_list):
        mapping[original] = t if t != "" else original
    logger.debug(f"Trimmed audio group names mapping: {mapping}")
    return mapping

def get_instrument_category_with_confidence(expanded_name):
    best_match, score = fuzz_process.extractOne(expanded_name, INSTRUMENT_CATEGORIES)
    return best_match, score

def expand_instrument_abbrev(name):
    cleaned = clean_name(name)
    normalized = cleaned.upper()
    for abbr, full in sorted(INSTRUMENT_ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        pattern = r'\b' + re.escape(abbr) + r'\b'
        normalized = re.sub(pattern, full.upper(), normalized)
    for k, v in CATEGORY_OVERRIDES.items():
        if normalized == k.upper():
            return v
    return normalized

# ---------------------------------------------------------------------
# Database imports
# ---------------------------------------------------------------------
from server import (
    init_db,
    insert_midi_file,
    insert_audio_file,
    insert_audio_feature,
    insert_final_mix,
    insert_midi_track,
    insert_midi_note,
    insert_midi_cc,
    insert_midi_program_change,
    get_cue_group_by_path,
    insert_cue_group,
    session,
    MidiFile,
    AudioFile,
    insert_project
)
init_db()

def get_or_create_cue_group(cue_path):
    from server import CueGroup
    cue_group = get_cue_group_by_path(cue_path)
    if cue_group is None:
        cue_group = insert_cue_group(cue_path)
    return cue_group.id

def get_audio_file_groups(audio_dir, keyword_filter=None):
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]
    found_files = []
    for ext in audio_extensions:
        found_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    groups = {}
    for filepath in found_files:
        base = os.path.basename(filepath)
        name, _ = os.path.splitext(base)
        channel_label = None
        file_order = None
        for suffix, order in CHANNEL_ORDER.items():
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                file_order = order
                channel_label = suffix.strip(".").upper()
                break
        canonical = name
        if "CLK" in canonical.upper():
            continue
        if keyword_filter and not keyword_filter(canonical.lower()):
            continue
        groups.setdefault(canonical, []).append((filepath, file_order, channel_label))
    sorted_groups = {}
    for canonical, file_list in groups.items():
        sorted_files = sorted(file_list, key=lambda x: x[1] if x[1] is not None else 999)
        sorted_groups[canonical] = sorted_files
    return sorted_groups

def normalize_audio(stereo_audio):
    max_val = np.max(np.abs(stereo_audio))
    if max_val > 1.0:
        return stereo_audio / max_val
    return stereo_audio

def downmix_interleaved(y):
    if y.shape[0] < 2:
        return y
    if y.shape[0] == 6:
        left = y[0] + 0.707 * y[2] + 0.5 * y[4] + 0.3548 * y[3]
        right = y[1] + 0.707 * y[2] + 0.5 * y[5] + 0.3548 * y[3]
        stereo = np.stack([left, right], axis=0)
    else:
        left = np.mean(y[: y.shape[0] // 2], axis=0)
        right = np.mean(y[y.shape[0] // 2:], axis=0)
        stereo = np.stack([left, right], axis=0)
    return normalize_audio(stereo)

def downmix_from_separate(file_tuples, sample_rate):
    if len(file_tuples) == 2:
        labels = {file_tuples[0][2], file_tuples[1][2]}
        if (None in labels) or (labels == {"L", "R"}):
            left, _ = librosa.load(file_tuples[0][0], sr=sample_rate, mono=True)
            right, _ = librosa.load(file_tuples[1][0], sr=sample_rate, mono=True)
            min_len = min(len(left), len(right))
            stereo = np.stack([left[:min_len], right[:min_len]], axis=0)
            return normalize_audio(stereo)
    gains = {
        "L": (1.0, 0.0),
        "R": (0.0, 1.0),
        "C": (0.707, 0.707),
        "LS": (0.5, 0.0),
        "RS": (0.0, 0.5),
        "LFE": (0.3548, 0.3548)
    }
    signals = {}
    lengths = []
    for filepath, order, label in file_tuples:
        if label is None:
            label = "UNKNOWN"
        y, sr = librosa.load(filepath, sr=sample_rate, mono=True)
        signals[label.upper()] = y
        lengths.append(len(y))
    if not lengths:
        return None
    min_len = min(lengths)
    left = np.zeros(min_len)
    right = np.zeros(min_len)
    for label, signal in signals.items():
        sig = signal[:min_len]
        if label in gains:
            gl, gr = gains[label]
            left += gl * sig
            right += gr * sig
        else:
            left += 0.5 * sig
            right += 0.5 * sig
    stereo = np.stack([left, right], axis=0)
    return normalize_audio(stereo)

def combine_audio_group(file_list, sample_rate):
    if not file_list:
        return None
    if isinstance(file_list[0], tuple):
        if len(file_list) == 1:
            mono_path = file_list[0][0]
            y, _ = librosa.load(mono_path, sr=sample_rate, mono=True)
            stereo = np.stack([y, y], axis=0)
            return normalize_audio(stereo)
        elif len(file_list) == 2:
            label0 = file_list[0][2]
            label1 = file_list[1][2]
            if (label0 is None and label1 is None) or ({label0, label1} == {"L", "R"}):
                left, _ = librosa.load(file_list[0][0], sr=sample_rate, mono=True)
                right, _ = librosa.load(file_list[1][0], sr=sample_rate, mono=True)
                min_len = min(len(left), len(right))
                stereo = np.stack([left[:min_len], right[:min_len]], axis=0)
                return normalize_audio(stereo)
            else:
                return downmix_from_separate(file_list, sample_rate)
        else:
            return downmix_from_separate(file_list, sample_rate)
    else:
        path = file_list[0]
        y, sr = librosa.load(path, sr=sample_rate, mono=False)
        if y.ndim == 1:
            y = np.expand_dims(y, 0)
        if y.shape[0] > 2:
            if y.shape[0] == 6:
                return downmix_interleaved(y)
            else:
                logger.warning(f"Unexpected # of channels ({y.shape[0]}) for {path}, downmixing by averaging.")
                left = np.mean(y[: y.shape[0] // 2], axis=0)
                right = np.mean(y[y.shape[0] // 2:], axis=0)
                stereo = np.stack([left, right], axis=0)
                return normalize_audio(stereo)
        return y

def process_audio_group_fn(args):
    # args: (canonical, file_tuples, sample_rate)
    canonical, file_tuples, sample_rate = args
    wave_stereo = combine_audio_group(file_tuples, sample_rate)
    if wave_stereo is None:
        return (canonical, None, None)
    tokens = encode_audio_features(wave_stereo, int(sample_rate))
    serialized = serialize_feature_array(tokens)
    return (canonical, wave_stereo, serialized)

def encode_audio_features(y, sr):
    if y is None:
        return None
    audio_tensor = torch.tensor(y, dtype=torch.float32)
    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = convert_audio(audio_tensor,
                                 src_sr=sr,
                                 dst_sr=encodec_model.sample_rate,
                                 dst_channels=encodec_model.channels)
    if hasattr(encodec_model, "model") and hasattr(encodec_model.model, "config"):
        config = encodec_model.model.config
        chunk_length = config.chunk_length
    else:
        chunk_length = 768
    total_length = audio_tensor.shape[-1]
    num_chunks = (total_length + chunk_length - 1) // chunk_length
    padded_length = num_chunks * chunk_length
    if total_length < padded_length:
        pad_needed = padded_length - total_length
        logger.debug(f"Padding audio from {total_length} to {padded_length} samples for full encoding.")
        audio_tensor = F.pad(audio_tensor, (0, pad_needed))
    channels = audio_tensor.shape[1]
    audio_tensor = audio_tensor.reshape(num_chunks, channels, chunk_length)
    with torch.no_grad():
        codes, scale = encodec_model.encode(audio_tensor)
    num_quantizers = codes.shape[1]
    n_frames_per_chunk = codes.shape[2]
    codes_cat = codes.transpose(0,1).reshape(num_quantizers, num_chunks * n_frames_per_chunk)
    codes_flat = codes_cat.flatten()
    return codes_flat.cpu().numpy()

def serialize_feature_array(feature_array):
    if feature_array is None:
        return None
    buf = io.BytesIO()
    np.save(buf, feature_array)
    buf.seek(0)
    return buf.read()

def thin_midi_cc_events(events, tolerance=1, max_interval=0.05):
    if not events:
        return events
    filtered = [events[0]]
    last = events[0]
    for event in events[1:]:
        if (abs(event['cc_value'] - last['cc_value']) <= tolerance and
            (event['time'] - last['time']) < max_interval):
            continue
        filtered.append(event)
        last = event
    return filtered

def tick_to_time(abs_tick, tempo_map, ticks_per_beat):
    if not tempo_map:
        return 0.0
    time_val = 0.0
    prev_tick = 0
    current_tempo = 500000
    for seg_time, seg_tick, seg_tempo in tempo_map:
        if abs_tick > seg_tick:
            delta_ticks = seg_tick - prev_tick
            time_val += (delta_ticks / ticks_per_beat) * (current_tempo / 1e6)
            prev_tick = seg_tick
            current_tempo = seg_tempo
        else:
            delta_ticks = abs_tick - prev_tick
            time_val += (delta_ticks / ticks_per_beat) * (current_tempo / 1e6)
            return time_val
    delta_ticks = abs_tick - prev_tick
    time_val += (delta_ticks / ticks_per_beat) * (current_tempo / 1e6)
    return time_val

def extract_tempo_and_time_signature(midi_path):
    try:
        mid = mido.MidiFile(midi_path)
    except IOError as e:
        print(f"Error opening MIDI file: {e}")
        sys.exit(1)
    ticks_per_beat = mid.ticks_per_beat
    merged = mido.merge_tracks(mid.tracks)
    current_tempo = 500000
    abs_time = 0.0
    abs_ticks = 0
    tempo_map = []
    time_sig_map = []
    for msg in merged:
        abs_ticks += msg.time
        delta_sec = mido.tick2second(msg.time, ticks_per_beat, current_tempo)
        abs_time += delta_sec
        if msg.type == "set_tempo":
            if msg.tempo != 0:
                current_tempo = msg.tempo
                tempo_map.append((abs_time, abs_ticks, msg.tempo))
        elif msg.type == "time_signature":
            time_sig_map.append((abs_time, abs_ticks, msg.numerator, msg.denominator))
    if not tempo_map or tempo_map[0][1] > 0:
        tempo_map.insert(0, (0.0, 0, 500000))
    if not time_sig_map or time_sig_map[0][1] > 0:
        time_sig_map.insert(0, (0.0, 0, 4, 4))
    return tempo_map, time_sig_map, ticks_per_beat

def extract_midi_track_names(midi_path):
    mid = mido.MidiFile(midi_path)
    result = []
    for i, track in enumerate(mid.tracks):
        track_name = None
        for msg in track:
            if msg.type == "track_name":
                track_name = msg.name.strip()
                break
        if not track_name:
            track_name = f"Track {i}"
        track_name = clean_name(track_name)
        if any(k in track_name.upper() for k in MARKER_KEYWORDS):
            continue
        result.append((i, track_name))
    return result

import tiktoken

class BatchManager:
    MAX_PENDING_TOKENS = 250000
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.requests = []
        self.encoding = tiktoken.encoding_for_model(model)
    def _count_tokens(self, text):
        return len(self.encoding.encode(text))
    def get_total_estimated_tokens(self):
        total = 0
        for req in self.requests:
            for msg in req["body"]["messages"]:
                total += self._count_tokens(msg["content"])
        return total
    def queue_chat_request(self, user_prompt, system_prompt, custom_id):
        MAX_REQUEST_TOKENS = 10000
        estimated_user_tokens = self._count_tokens(user_prompt)
        estimated_system_tokens = self._count_tokens(system_prompt)
        total_estimated = estimated_user_tokens + estimated_system_tokens
        if total_estimated > MAX_REQUEST_TOKENS:
            logger.warning(f"Request {custom_id} is too large ({total_estimated} tokens). Truncating user prompt.")
            user_prompt = user_prompt[: MAX_REQUEST_TOKENS * 4]
            total_estimated = self._count_tokens(user_prompt) + estimated_system_tokens
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 20
        }
        request_line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        current_total = self.get_total_estimated_tokens()
        logger.info(f"Current pending tokens before adding '{custom_id}': {current_total}")
        self.requests.append(request_line)
        current_total = self.get_total_estimated_tokens()
        logger.info(f"Pending tokens after adding '{custom_id}': {current_total}")
        logger.info(f"Queued request '{custom_id}' with estimated tokens={total_estimated}.")
        if current_total > self.MAX_PENDING_TOKENS:
            logger.info(f"Pending tokens ({current_total}) exceed threshold, executing batches...")
            self.execute_batches(max_requests_per_batch=12, max_tokens_per_batch=4000)
    def _process_batch(self, batch):
        results = {}
        BATCH_INPUT_DIR = "batch-input"
        if not os.path.exists(BATCH_INPUT_DIR):
            os.makedirs(BATCH_INPUT_DIR)
        temp_filename = os.path.join(BATCH_INPUT_DIR, f"batch_input_{int(time.time()*1000)}.jsonl")
        with open(temp_filename, "w", encoding="utf-8") as f:
            for req in batch:
                f.write(json.dumps(req) + "\n")
        try:
            upload_resp = client.files.create(
                file=open(temp_filename, "rb"),
                purpose="batch"
            )
        except Exception as e:
            logger.error(f"Error uploading batch file {temp_filename}: {e}")
            sys.exit(1)
        input_file_id = upload_resp.id
        try:
            batch_obj = client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
        except Exception as e:
            logger.error(f"Error creating batch for {temp_filename}: {e}")
            sys.exit(1)
        batch_id = batch_obj.id
        logger.info(f"[BatchManager] Created batch {batch_id} for file {temp_filename}. Waiting for completion...")
        while True:
            status_obj = client.batches.retrieve(batch_id)
            status = status_obj.status
            if status == "completed":
                logger.info(f"Batch {batch_id} complete. Waiting an extra 10 seconds before proceeding for safety...")
                time.sleep(10)
                break
            elif status in ["failed", "expired"]:
                logger.error(f"Batch {batch_id} ended with status={status}")
                logger.error(f"Error details: {status_obj}")
                sys.exit(1)
            elif status in ["cancelling", "cancelled"]:
                logger.error(f"Batch {batch_id} was cancelled. Exiting.")
                sys.exit(1)
            else:
                logger.info(f"  Batch {batch_id} status = {status}, waiting 10s...")
                time.sleep(10)
        output_file_id = status_obj.output_file_id
        if not output_file_id:
            logger.error("No output_file_id found. Exiting.")
            sys.exit(1)
        try:
            output_resp = download_output_file(output_file_id)
        except Exception as e:
            logger.error(f"Error downloading output for batch {batch_id}: {e}")
            sys.exit(1)
        for line in output_resp.decode("utf-8").splitlines():
            rec = json.loads(line)
            cid = rec["custom_id"]
            if rec["error"] is not None:
                results[cid] = f"Error: {rec['error']}"
                logger.error(f"Custom_id '{cid}' error: {rec['error']}")
                continue
            choices = rec["response"]["body"].get("choices", [])
            if not choices:
                results[cid] = ""
                continue
            content = choices[0]["message"]["content"].strip()
            results[cid] = content
            usage = rec["response"]["body"].get("usage")
            if usage:
                logger.info(f"Token usage for '{cid}': prompt={usage.get('prompt_tokens',0)}, "
                            f"completion={usage.get('completion_tokens',0)}, total={usage.get('total_tokens',0)}")
        return results
    def execute_batches(self, max_requests_per_batch=12, max_tokens_per_batch=4000):
        results = {}
        if not self.requests:
            return results
        current_batch = []
        current_token_count = 0
        for req in self.requests:
            body = req["body"]
            estimated_tokens = sum(self._count_tokens(msg["content"]) for msg in body["messages"])
            if len(current_batch) >= max_requests_per_batch or (current_token_count + estimated_tokens) > max_tokens_per_batch:
                try:
                    results.update(self._process_batch(current_batch))
                except openai.NotFoundError as e:
                    logger.warning(f"Batch not found, skipping this batch: {e}")
                current_batch = []
                current_token_count = 0
            current_batch.append(req)
            current_token_count += estimated_tokens
        if current_batch:
            try:
                results.update(self._process_batch(current_batch))
            except openai.NotFoundError as e:
                logger.warning(f"Batch not found, skipping this batch: {e}")
        self.requests = []
        return results
    def execute_batches_parallel(self, max_workers=4):
        results = {}
        if not self.requests:
            return results
        total_requests = len(self.requests)
        batch_size = max(1, total_requests // max_workers)
        batches = [self.requests[i:i+batch_size] for i in range(0, total_requests, batch_size)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_batch, batch): batch for batch in batches}
            for future in futures:
                try:
                    results.update(future.result())
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
        self.requests = []
        return results

# Global batch managers:
global_batch_manager_match = BatchManager(model="gpt-4o")
global_batch_manager_common = BatchManager(model="gpt-4o")
global_batch_manager_audio = BatchManager(model="gpt-4o")

# Global dictionary to hold pending cue data:
pending_cue_data = {}

# Global dictionary to hold refined audio group categories.
audio_group_category = {}

def refine_audio_groups(available_groups_original):
    threshold = 80  # Fuzzy match score threshold
    for canonical in available_groups_original:
        name_expanded = expand_instrument_abbrev(canonical)
        best_match, score = get_instrument_category_with_confidence(name_expanded)
        if score < threshold:
            custom_id = "audio_refine_" + canonical
            system_prompt = "You are an expert music-production AI. Given an audio group name and a list of possible instrument categories, choose the best category."
            user_prompt = f"Audio group name: {name_expanded}\nPossible categories: {', '.join(INSTRUMENT_CATEGORIES)}\nSelect the best instrument category."
            global_batch_manager_audio.queue_chat_request(user_prompt, system_prompt, custom_id)
            audio_group_category[canonical] = "Pending"
        else:
            audio_group_category[canonical] = best_match

def queue_match_prompt(batch_manager, track_name, available_groups_trimmed, cue_id, track_idx):
    custom_id = f"{track_idx}_{track_name.replace(' ', '_')}"
    system_prompt = (
        "You are an expert music-production AI. Given a MIDI track name and a list of audio groups, "
        "select the best matching group considering abstract meanings and abbreviations. Output only the group name."
    )
    abbreviations_str = ", ".join([f"{k}:{v}" for k, v in INSTRUMENT_ABBREVIATIONS.items()])
    user_prompt = (
        f"MIDI Track: {track_name}\n"
        f"Audio Groups: {', '.join(available_groups_trimmed)}\n"
        f"List of possible abbreviations: {abbreviations_str}\n"
        "Select the best matching group."
    )
    batch_manager.queue_chat_request(user_prompt, system_prompt, custom_id)
    return custom_id

def process_cue(cue_dir, sample_rate, project_id):
    print(f"\n=== Processing Cue: {cue_dir} ===")
    midi_dir = os.path.join(cue_dir, "MIDI")
    mid_files = glob.glob(os.path.join(midi_dir, "*.mid"))
    if not mid_files:
        print(f"No MIDI file found in {midi_dir}, skipping.")
        return
    midi_path = mid_files[0]
    print("Processing MIDI file:", midi_path)
    from server import MidiFile
    existing = session.query(MidiFile).filter_by(file_path=midi_path).first()
    if existing:
        print(f"MIDI file {midi_path} already processed. Skipping.")
        return
    cue_group_id = get_or_create_cue_group(cue_dir)
    # Compute the chunk length from the model configuration.
    if hasattr(encodec_model, "model") and hasattr(encodec_model.model, "config"):
        config = encodec_model.model.config
        chunk_length = config.chunk_length
    else:
        chunk_length = 768

    tempo_map, time_sig_map, tpb = extract_tempo_and_time_signature(midi_path)
    midi_tracks = extract_midi_track_names(midi_path)
    pt_dir = os.path.join(cue_dir, "PT", "Audio Files")
    if not os.path.exists(pt_dir):
        print("Audio Files folder not found; skipping.")
        return
    audio_groups = get_audio_file_groups(pt_dir)
    available_groups_original = list(audio_groups.keys())
    trim_mapping = trim_audio_group_names(available_groups_original)
    available_groups_trimmed = list(trim_mapping.values())
    logger.info(f"Original audio groups: {available_groups_original}")
    logger.info(f"Trimmed audio groups: {available_groups_trimmed}")
    # --- Refine audio group categories using GPT-4 if needed ---
    refine_audio_groups(available_groups_original)
    # Process audio groups in parallel per audio group.
    composite_audio = {}
    audio_features = {}
    with ProcessPoolExecutor() as executor:
        args = [(canonical, audio_groups[canonical], sample_rate) for canonical in available_groups_original]
        for canonical, wave_stereo, serialized in tqdm(executor.map(process_audio_group_fn, args), total=len(args), desc="Processing Audio Groups"):
            if wave_stereo is not None and serialized is not None:
                composite_audio[canonical] = wave_stereo
                audio_features[canonical] = serialized
    print("Encoding audio with EnCodec complete.")
    midi_file_record = insert_midi_file(
        file_path=midi_path,
        tempo_map=json.dumps(tempo_map),
        time_signature_map=json.dumps(time_sig_map),
        ticks_per_beat=tpb,
        cue_group_id=cue_group_id,
        project_id=project_id
    )
    # Process final mix if applicable.
    final_mix_groups = get_audio_file_groups(
        pt_dir,
        keyword_filter=lambda c: "6mx" in c.lower() or "ref" in c.lower()
    )
    if final_mix_groups:
        fm_canonical = list(final_mix_groups.keys())[0]
        fm_filetuples = final_mix_groups[fm_canonical]
        fm_wave = combine_audio_group(fm_filetuples, sample_rate)
        if fm_wave is not None:
            fm_tokens = encode_audio_features(fm_wave, int(sample_rate))
            fm_data = serialize_feature_array(fm_tokens)
            if fm_data is not None:
                insert_final_mix(
                    midi_file_id=midi_file_record.id,
                    file_path=(fm_filetuples[0][0] if isinstance(fm_filetuples[0], tuple) else fm_filetuples[0]),
                    feature_type="encoded_audio_tokens",
                    feature_data=fm_data,
                    cue_group_id=cue_group_id,
                    project_id=project_id,
                    chunk_size=chunk_length  # store chunk size in final mix
                )
    else:
        print("No final mix found in", pt_dir)
    # --- Bulk insert audio file records ---
    from server import AudioFile, AudioFeature
    audio_files_to_insert = []
    audio_features_to_insert = []
    audio_file_records = {}
    for canonical, file_tuples in audio_groups.items():
        if "ref" in canonical.lower() or "6mx" in canonical.lower():
            continue
        rep = file_tuples[0] if isinstance(file_tuples[0], tuple) else (file_tuples[0], None, None)
        full_path = os.path.join(pt_dir, os.path.basename(rep[0]))
        name_expanded = expand_instrument_abbrev(canonical)
        # Use refined category if available; otherwise, fallback
        audio_cat = audio_group_category.get(canonical, get_instrument_category_with_confidence(name_expanded)[0])
        af = insert_audio_file(
            file_path=full_path,
            canonical_name=canonical,
            instrument_category=audio_cat,
            cue_group_id=cue_group_id,
            project_id=project_id,
            chunk_size=chunk_length  # store chunk size in audio file record
        )
        audio_file_records[canonical] = full_path
        if canonical in audio_features and audio_features[canonical] is not None:
            af_feat = AudioFeature(
                feature_type="encoded_audio_tokens",
                feature_data=audio_features[canonical]
            )
            audio_features_to_insert.append((full_path, af_feat))
    if audio_features_to_insert:
        for file_path, af_feat in audio_features_to_insert:
            audio_obj = session.query(AudioFile).filter_by(file_path=file_path).first()
            if audio_obj:
                af_feat.audio_file_id = audio_obj.id
                session.add(af_feat)
        session.commit()
    # Queue MIDI-to-audio-group matching prompts.
    cue_id = os.path.basename(cue_dir)
    match_requests = {}
    for idx, track_name in midi_tracks:
        cid = queue_match_prompt(global_batch_manager_match, track_name, available_groups_trimmed, cue_id, idx)
        match_requests[cid] = (idx, track_name)
        print(f"Queued matching prompt for track '{track_name}' -> {cid}")
    pending_cue_data[cue_dir] = {
        "cue_id": cue_id,
        "midi_file_record": midi_file_record,
        "midi_tracks": midi_tracks,
        "midi_match_requests": match_requests,
        "audio_file_records": audio_file_records,
        "available_groups_original": available_groups_original,
        "available_groups_trimmed": available_groups_trimmed,
        "trim_mapping": trim_mapping
    }
    # Process MIDI notes & CC events.
    mid = mido.MidiFile(midi_path)
    valid_indices = {i for i, _ in midi_tracks}
    for idx, track in enumerate(mid.tracks):
        if idx not in valid_indices:
            continue
        ctick = 0
        pending_notes = {}
        cc_events = []
        for msg in tqdm(track, desc=f"MIDI events track {idx}", leave=False):
            ctick += msg.time
            evt_time = tick_to_time(ctick, tempo_map, tpb)
            if msg.type == "note_on":
                if msg.velocity > 0:
                    pending_notes[(msg.channel, msg.note)] = (ctick, evt_time, msg.velocity)
                else:
                    if (msg.channel, msg.note) in pending_notes:
                        st_tick, st_time, vel = pending_notes.pop((msg.channel, msg.note))
                        dur = evt_time - st_time
                        insert_midi_note(
                            midi_track_id=None,
                            channel=msg.channel,
                            note=msg.note,
                            velocity=vel,
                            start_tick=st_tick,
                            end_tick=ctick,
                            start_time=st_time,
                            duration=dur
                        )
            elif msg.type == "note_off":
                if (msg.channel, msg.note) in pending_notes:
                    st_tick, st_time, vel = pending_notes.pop((msg.channel, msg.note))
                    dur = evt_time - st_time
                    insert_midi_note(
                        midi_track_id=None,
                        channel=msg.channel,
                        note=msg.note,
                        velocity=vel,
                        start_tick=st_tick,
                        end_tick=ctick,
                        start_time=st_time,
                        duration=dur
                    )
            elif msg.type == "control_change":
                cc_events.append({
                    'channel': msg.channel,
                    'cc_number': msg.control,
                    'cc_value': msg.value,
                    'tick': ctick,
                    'time': evt_time
                })
            elif msg.type == "program_change":
                insert_midi_program_change(
                    midi_track_id=None,
                    channel=msg.channel,
                    program=msg.program,
                    tick=ctick,
                    time=evt_time
                )
        grouped_cc = {}
        for e in cc_events:
            key = (e['channel'], e['cc_number'])
            grouped_cc.setdefault(key, []).append(e)
        final_cc = []
        for key, group_evs in grouped_cc.items():
            if key[1] in [1, 11]:
                sorted_evs = sorted(group_evs, key=lambda x: x['tick'])
                thinned = thin_midi_cc_events(sorted_evs, tolerance=1, max_interval=0.05)
            else:
                thinned = group_evs
            final_cc.extend(thinned)
        final_cc.sort(key=lambda x: x['tick'])
        for e in final_cc:
            insert_midi_cc(
                midi_track_id=None,
                channel=e['channel'],
                cc_number=e['cc_number'],
                cc_value=e['cc_value'],
                tick=e['tick'],
                time=e['time']
            )
    print(f"Finished processing cue: {cue_dir} (Phase 1)")

def finalize_phase_2_3():
    if global_batch_manager_match.requests:
        print("Executing global batch for MIDI matching...")
        midi_match_results = global_batch_manager_match.execute_batches(max_requests_per_batch=12, max_tokens_per_batch=4000)
    else:
        midi_match_results = {}
    print("MIDI matching results:")
    for k, v in midi_match_results.items():
        print(f"  {k}: {v}")
    for cue_dir, data in pending_cue_data.items():
        cue_id = data["cue_id"]
        match_requests = data["midi_match_requests"]
        mapping = {}
        for custom_id, (idx, track_name) in match_requests.items():
            pred = midi_match_results.get(custom_id, "").strip()
            if not pred:
                if data["available_groups_trimmed"]:
                    pred = data["available_groups_trimmed"][0]
                else:
                    pred = "UnknownGroup"
            if pred.lower() not in [grp.lower() for grp in data["available_groups_trimmed"]]:
                pred_g, _ = fuzz_process.extractOne(pred, data["available_groups_trimmed"])
                pred = pred_g
            mapping[track_name] = pred
        data["final_mapping"] = mapping
        group_to_tracks = {}
        for idx, track_name in data["midi_tracks"]:
            grp = mapping.get(track_name)
            if grp:
                group_to_tracks.setdefault(grp, []).append(track_name)
        data["group_to_tracks"] = group_to_tracks
        for group_name, track_names in group_to_tracks.items():
            custom_id = f"assign_common_{cue_id}_{group_name}"
            system_prompt = (
                "You are an expert music-production AI. Given MIDI track names and an audio group, "
                "select the best category from the provided list. MIDI and audio names might be abstract or abbreviated, "
                "Output only the category name."
            )
            user_prompt = (
                f"MIDI Tracks: {', '.join(track_names)}\n"
                f"Audio Group: {group_name}\n"
                f"Categories: {', '.join(INSTRUMENT_CATEGORIES)}\n"
                "Select the best category."
            )
            global_batch_manager_common.queue_chat_request(user_prompt, system_prompt, custom_id)
            print(f"Queued group->commonCategory prompt for {group_name} => {custom_id}")
            
def finalize_phase_4():
    if global_batch_manager_common.requests:
        print("Executing global batch for common category assignment...")
        common_results = global_batch_manager_common.execute_batches(max_requests_per_batch=12, max_tokens_per_batch=4000)
    else:
        common_results = {}
    print("Common category results:")
    for k, v in common_results.items():
        print(f"  {k}: {v}")
    for cue_dir, data in pending_cue_data.items():
        cue_id = data["cue_id"]
        track_mapping = data.get("final_mapping", {})
        group_to_tracks = data.get("group_to_tracks", {})
        audio_file_records = data["audio_file_records"]
        midi_file_record = data["midi_file_record"]
        group_common_map = {}
        for group_name, track_names in group_to_tracks.items():
            custom_id = f"assign_common_{cue_id}_{group_name}"
            cat = common_results.get(custom_id, "").strip()
            if not cat:
                cat = "Strings"
            group_common_map[group_name] = cat
        inv_trim = {v.lower(): k for k, v in data["trim_mapping"].items()}
        for idx, track_name in data["midi_tracks"]:
            chosen_trimmed = track_mapping.get(track_name)
            if chosen_trimmed and chosen_trimmed.lower() in inv_trim:
                chosen_group = inv_trim[chosen_trimmed.lower()]
            else:
                chosen_group = chosen_trimmed
            assigned_audio_id = None
            if chosen_group in audio_file_records:
                assigned_audio_id = audio_file_records[chosen_group]
            if assigned_audio_id is not None:
                audio_file_obj = session.query(AudioFile).filter_by(id=assigned_audio_id).first()
                if audio_file_obj:
                    final_cat = audio_file_obj.instrument_category
                else:
                    final_cat = group_common_map.get(chosen_trimmed, "Strings")
            else:
                final_cat = group_common_map.get(chosen_trimmed, "Strings")
            if assigned_audio_id is not None:
                audio_file_obj = session.query(AudioFile).filter_by(id=assigned_audio_id).first()
                if audio_file_obj:
                    audio_file_obj.instrument_category = final_cat
                    session.commit()
            insert_midi_track(
                midi_file_id=midi_file_record.id,
                track_index=idx,
                track_name=track_name,
                instrument_category=final_cat,
                assigned_audio_file_id=assigned_audio_id
            )
            print(f"Inserted MIDI track '{track_name}' => group='{chosen_group}', category='{final_cat}'.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_cues.py <cue_base_directory> [sample_rate] [project_id]")
        sys.exit(1)
    base_dir = sys.argv[1]
    try:
        sample_rate = float(sys.argv[2]) if len(sys.argv) >= 3 else 48000
    except ValueError:
        print("Invalid sample rate provided. Using default 48000.")
        sample_rate = 48000
    from server import Project
    project = session.query(Project).first()
    if not project:
        project = insert_project("Default Project", sample_rate=sample_rate)
        print(f"Created default project with id={project.id} @ {sample_rate} Hz")
    project_id = project.id
    cue_dirs = find_cue_directories(base_dir)
    if not cue_dirs:
        print("No cues found under", base_dir)
        sys.exit(0)
    print(f"Found {len(cue_dirs)} cue directories. Beginning Phase 1 processing...")
    for cue in tqdm(cue_dirs, desc="Processing Cues"):
        try:
            process_cue(cue, sample_rate, project_id)
        except Exception as e:
            logger.error(f"Error processing cue {cue}: {e}")
            continue
    print("Phase 1 complete. All cues processed.")
    if global_batch_manager_audio.requests:
        print("Refining audio group categories via GPT-4...")
        audio_refine_results = global_batch_manager_audio.execute_batches(max_requests_per_batch=12, max_tokens_per_batch=4000)
        for key, refined in audio_refine_results.items():
            canonical = key.replace("audio_refine_", "")
            audio_group_category[canonical] = refined.strip()
        print("Refined audio group categories:", audio_group_category)
    finalize_phase_2_3()
    finalize_phase_4()
    print("All done. DB insertion complete!")

if __name__ == "__main__":
    main()
