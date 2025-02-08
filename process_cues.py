#!/usr/bin/env python3
"""
process_cues.py

This script recursively scans a base cues directory for projects (each cue folder must contain a "MIDI"
subfolder with a .mid file and a "PT/Audio Files" subfolder). It processes the MIDI and audio files,
tokenizes the audio using EnCodec for MusicGen (using the 48 kHz AudioCraft model rather than mel-spectrograms),
and uses the OpenAI Batch API to do the classification (track->group + common category) asynchronously.

Finally, once a MIDI track is matched to an audio group and we have a final "common instrument category,"
we **update both**:
  - the MIDI track (in `midi_tracks.instrument_category`)
  - the associated `audio_files.instrument_category`
to ensure consistency.

Audio features are serialized as binary (numpy .npy format) for efficiency.

For surround files (whether provided as interleaved multi-channel files or as separate mono files):
  - If the audio group contains exactly two files (with channel labels “L” and “R” or with no labels),
    they are assumed already stereo and no additional downmix is applied.
  - Otherwise (e.g. LCR, Quad, 5.0), a custom downmix is performed using these per-channel gains:
      • L: 0 dB (×1.0)
      • C: –3 dB (×0.707)
      • R: 0 dB (×1.0)
      • Ls: –6 dB (×0.5)
      • Rs: –6 dB (×0.5)
      • LFE: –9 dB (×0.3548)
Since EnCodec (48 kHz) supports only stereo input, any surround file is downmixed to stereo.
"""

import os
import sys
import glob
import json
import io
import re
import time

import mido
import numpy as np
import librosa
import torch
import torchaudio
from tqdm import tqdm
from thefuzz import process as fuzz_process
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# --- OpenAI Batch API Setup ---
# Using 'OpenAI' for the Batch API (from the official docs).
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if client.api_key is None:
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

# --- AudioCraft EnCodec 48 kHz ---
# We install AudioCraft from GitHub and import its EnCodec model:
from audiocraft.models.encodec import CompressionModel

def convert_audio(audio_tensor: torch.Tensor, src_sr: int, dst_sr: int, dst_channels: int):
    """
    Similar to old `encodec.utils.convert_audio`, but here we do:
      1) Resample from src_sr to dst_sr using torchaudio
      2) Convert channel count if needed
    audio_tensor shape = (batch_size, channels, num_samples)
    """
    # Resample if needed:
    if src_sr != dst_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, src_sr, dst_sr)

    # Adjust channels
    cur_channels = audio_tensor.shape[1]
    if cur_channels != dst_channels:
        if dst_channels == 1:
            # average to mono
            audio_tensor = audio_tensor.mean(dim=1, keepdim=True)
        elif dst_channels == 2 and cur_channels == 1:
            # replicate mono -> stereo
            audio_tensor = audio_tensor.repeat(1, 2, 1)
        else:
            print(f"Warning: cannot automatically convert from {cur_channels} ch to {dst_channels} ch.")
    return audio_tensor

# Load the 48 kHz EnCodec model globally
global encodec_model
encodec_model = CompressionModel.get_pretrained("facebook/encodec_48khz")
encodec_model.eval()

# --- Configuration and Global Constants ---
INSTRUMENT_CATEGORIES = [
    "Strings", "Woodwinds", "Brass", "Acoustic Guitar", "Electric Guitar",
    "Piano", "Keys", "Bells", "Harp", "Synth Pulse", "Synth Pad", "Synth Bass",
    "Low Percussion", "Mid Percussion", "High Percussion", "Drums", "Orch Percussion",
    "Bass", "Double Bass", "FX", "Choir", "Solo Vocals", "Mallets", "Plucked",
    "Sub Hits", "Guitar FX", "Orch FX", "Ticker"
]

MARKER_KEYWORDS = ["NOTES", "CONDUCTOR", "ORCHESTRATOR", "MARKER", "MONITOR", "MIDI"]

CHANNEL_ORDER = {
    ".L": 0, ".C": 1, ".R": 2, ".Ls": 3, ".Rs": 4, ".lfe": 5, ".Lf": 5
}

INSTRUMENT_ABBREVIATIONS = {
    "VCS": "Violoncello", "VLN": "Violin", "VLA": "Viola",
    "SHT": "Short", "LG": "Long", "HARP": "Harp", "HPS": "Harps",
    "BASS": "Bass", "PERC": "Percussion", "TPT": "Trumpet",
    "TBN": "Trombone", "FL": "Flute", "OB": "Oboe", "CL": "Clarinet",
    "BSN": "Bassoon", "SAX": "Saxophone", "GTR": "Guitar",
    "SYN": "Synth", "FX": "Effects", "PAD": "Synth Pad",
    "ORG": "Keys",  # Organ, Rhodes, celeste, etc.
    "BELL": "Bells", "CHOIR": "Choir",
    "VOX": "Solo Vocals", "MALLET": "Mallet Percussion",
    "TAKO": "Low Percussion", "TKO": "Low Percussion",
    "TIMP": "Orch Percussion", "TIMPANI": "Orch Percussion",
    "SNR": "Mid Percussion",
    "CYM": "High Percussion"
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

def clean_name(name):
    """Remove non-alphanumeric (except whitespace) and collapse whitespace."""
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

# --- Import database functions and models from server ---
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

def downmix_interleaved(y):
    """Downmix 6-channel 5.1 to stereo. Otherwise fallback to naive averaging."""
    if y.shape[0] < 2:
        return y
    if y.shape[0] == 6:
        left = y[0] + 0.707 * y[2] + 0.5 * y[4] + 0.3548 * y[3]
        right = y[1] + 0.707 * y[2] + 0.5 * y[5] + 0.3548 * y[3]
        return np.stack([left, right], axis=0)
    else:
        left = np.mean(y[: y.shape[0] // 2], axis=0)
        right = np.mean(y[y.shape[0] // 2 :], axis=0)
        return np.stack([left, right], axis=0)

def downmix_from_separate(file_tuples, sample_rate):
    if len(file_tuples) == 2:
        labels = {file_tuples[0][2], file_tuples[1][2]}
        if (None in labels) or (labels == {"L", "R"}):
            left, _ = librosa.load(file_tuples[0][0], sr=sample_rate, mono=True)
            right, _ = librosa.load(file_tuples[1][0], sr=sample_rate, mono=True)
            min_len = min(len(left), len(right))
            return np.stack([left[:min_len], right[:min_len]], axis=0)

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
    return np.stack([left, right], axis=0)

def combine_audio_group(file_list, sample_rate):
    if not file_list:
        return None
    if isinstance(file_list[0], tuple):
        if len(file_list) == 1:
            mono_path = file_list[0][0]
            y, _ = librosa.load(mono_path, sr=sample_rate, mono=True)
            return np.stack([y, y], axis=0)
        elif len(file_list) == 2:
            label0 = file_list[0][2]
            label1 = file_list[1][2]
            if (label0 is None and label1 is None) or ({label0, label1} == {"L","R"}):
                left, _ = librosa.load(file_list[0][0], sr=sample_rate, mono=True)
                right, _ = librosa.load(file_list[1][0], sr=sample_rate, mono=True)
                min_len = min(len(left), len(right))
                return np.stack([left[:min_len], right[:min_len]], axis=0)
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
                print(f"Warning: Unexpected # of channels ({y.shape[0]}) for {path}, downmixing by averaging.")
                left = np.mean(y[: y.shape[0] // 2], axis=0)
                right = np.mean(y[y.shape[0] // 2 :], axis=0)
                return np.stack([left, right], axis=0)
        return y

def encode_audio_features(y, sr):
    """
    Use AudioCraft's EnCodec to encode an audio numpy array.
    1) Convert to torch tensor
    2) Resample to 48 kHz and fix channels to match encodec_model
    3) encodec_model.encode(...) => (codes, scale)
    4) Flatten codes => np array
    """
    if y is None:
        return None
    audio_tensor = torch.tensor(y, dtype=torch.float32)
    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.unsqueeze(0)  # (1, channels, samples)

    # Convert audio to the model's sample_rate/channels
    audio_tensor = convert_audio(audio_tensor, src_sr=sr,
                                 dst_sr=encodec_model.sample_rate,
                                 dst_channels=encodec_model.channels)

    with torch.no_grad():
        codes, scale = encodec_model.encode(audio_tensor)  # shape [B, K, T]
    # Usually B=1 => shape (K, T)
    codes_2d = codes[0]
    codes_flat = codes_2d.flatten()
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
        if any(k in track_name.upper() for k in MARKER_KEYWORDS):
            continue
        result.append((i, track_name))
    return result

def expand_instrument_abbrev(name):
    cleaned = clean_name(name)
    normalized = cleaned.upper()
    for abbr, full in INSTRUMENT_ABBREVIATIONS.items():
        normalized = normalized.replace(abbr, full.upper())
    for k, v in CATEGORY_OVERRIDES.items():
        if normalized == k.upper():
            return v
    return normalized

# --- Our custom Batch Manager (for Chat) ---
class BatchManager:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.requests = []

    def queue_chat_request(self, user_prompt, system_prompt, custom_id):
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 40
        }
        request_line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        self.requests.append(request_line)

    def execute_batch(self):
        if not self.requests:
            return {}

        input_filename = "batch_input.jsonl"
        with open(input_filename, "w", encoding="utf-8") as f:
            for req in self.requests:
                f.write(json.dumps(req) + "\n")

        try:
            upload_resp = client.files.create(
                file=open(input_filename, "rb"),
                purpose="batch"
            )
        except Exception as e:
            print("Error uploading batch file:", e)
            sys.exit(1)
        input_file_id = upload_resp.id

        try:
            batch = client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
        except Exception as e:
            print("Error creating batch:", e)
            sys.exit(1)

        batch_id = batch.id
        print(f"[BatchManager] Created batch {batch_id}. Waiting for completion...")

        while True:
            status_obj = client.batches.retrieve(batch_id)
            status = status_obj.status
            if status == "completed":
                break
            elif status in ["failed", "expired"]:
                print(f"Batch {batch_id} ended with status={status}")
                sys.exit(1)
            elif status in ["cancelling", "cancelled"]:
                print(f"Batch {batch_id} was cancelled. Exiting.")
                sys.exit(1)
            else:
                print(f"  batch status = {status}, waiting 10s...")
                time.sleep(10)

        output_file_id = status_obj.output_file_id
        if not output_file_id:
            print("No output_file_id found. Possibly no lines succeeded. Exiting.")
            sys.exit(1)

        try:
            output_resp = client.files.download(output_file_id)
        except Exception as e:
            print("Error downloading output file:", e)
            sys.exit(1)

        results = {}
        for line in output_resp.decode("utf-8").splitlines():
            rec = json.loads(line)
            cid = rec["custom_id"]
            if rec["error"] is not None:
                results[cid] = ""
                continue
            choices = rec["response"]["body"].get("choices", [])
            if not choices:
                results[cid] = ""
                continue
            content = choices[0]["message"]["content"].strip()
            results[cid] = content

        self.requests = []
        return results

global_batch_manager_match = BatchManager(model="gpt-4")
global_batch_manager_common = BatchManager(model="gpt-4")

pending_cue_data = {}

def queue_match_prompt(batch_manager, track_name, available_groups, cue_id, track_idx):
    custom_id = f"match_{cue_id}_{track_idx}_{track_name.replace(' ', '_')}"
    user_prompt = (
        "You must match this MIDI track to one of these audio group names.\n"
        f"MIDI Track: {track_name}\nAvailable Audio Groups: {', '.join(available_groups)}\n\n"
        "Output only the exact group name from the above list."
    )
    system_prompt = "You match MIDI tracks to audio groups from the provided list."
    batch_manager.queue_chat_request(user_prompt, system_prompt, custom_id)
    return custom_id

def process_cue(cue_dir, sample_rate, project_id):
    print(f"\n=== Processing Cue: {cue_dir} ===")
    midi_dir = os.path.join(cue_dir, "MIDI")
    mid_files = glob.glob(os.path.join(midi_dir, "*.mid"))
    if not mid_files:
        print(f"No MIDI file in {midi_dir}, skipping.")
        return
    midi_path = mid_files[0]
    print("Processing MIDI file:", midi_path)

    from server import MidiFile
    existing = session.query(MidiFile).filter_by(file_path=midi_path).first()
    if existing:
        print(f"MIDI file {midi_path} is already processed. Skipping.")
        return

    cue_group_id = get_or_create_cue_group(cue_dir)
    tempo_map, time_sig_map, tpb = extract_tempo_and_time_signature(midi_path)
    midi_tracks = extract_midi_track_names(midi_path)

    pt_dir = os.path.join(cue_dir, "PT", "Audio Files")
    if not os.path.exists(pt_dir):
        print(f"No PT/Audio Files dir in {cue_dir}, skipping.")
        return

    audio_groups = get_audio_file_groups(pt_dir)
    composite_audio = {}
    print("Combining/downmix audio in", pt_dir)
    for canonical, file_tuples in tqdm(audio_groups.items(), desc="Audio Groups", leave=False):
        wave_stereo = combine_audio_group(file_tuples, sample_rate)
        if wave_stereo is not None:
            composite_audio[canonical] = wave_stereo

    print("Encoding audio with EnCodec...")
    audio_features = {}
    for canonical, wave_stereo in tqdm(composite_audio.items(), desc="Tokenizing", leave=False):
        tokens = encode_audio_features(wave_stereo, int(sample_rate))
        audio_features[canonical] = serialize_feature_array(tokens)

    midi_file_record = insert_midi_file(
        file_path=midi_path,
        tempo_map=json.dumps(tempo_map),
        time_signature_map=json.dumps(time_sig_map),
        ticks_per_beat=tpb,
        cue_group_id=cue_group_id,
        project_id=project_id
    )

    # Final mix
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
                    project_id=project_id
                )
    else:
        print("No final mix (6mx/ref) found in", pt_dir)

    audio_file_records = {}
    for canonical, file_tuples in audio_groups.items():
        rep = file_tuples[0] if isinstance(file_tuples[0], tuple) else (file_tuples[0], None, None)
        full_path = os.path.join(pt_dir, os.path.basename(rep[0]))
        name_expanded = expand_instrument_abbrev(canonical)
        best_match, score = fuzz_process.extractOne(name_expanded, INSTRUMENT_CATEGORIES)
        if not best_match:
            best_match = "Strings"
        audio_cat = best_match

        exist_aud = session.query(AudioFile).filter_by(file_path=full_path).first()
        if exist_aud:
            print(f"Audio file {full_path} in DB, skipping.")
            rec_id = exist_aud.id
        else:
            rec = insert_audio_file(
                file_path=full_path,
                canonical_name=canonical,
                instrument_category=audio_cat,
                cue_group_id=cue_group_id,
                project_id=project_id
            )
            rec_id = rec.id

        audio_file_records[canonical] = rec_id
        if canonical in audio_features and audio_features[canonical] is not None:
            insert_audio_feature(
                audio_file_id=rec_id,
                feature_type="encoded_audio_tokens",
                feature_data=audio_features[canonical]
            )

    available_groups = list(audio_groups.keys())
    cue_id = os.path.basename(cue_dir)
    match_requests = {}
    for idx, track_name in midi_tracks:
        cid = queue_match_prompt(global_batch_manager_match, track_name, available_groups, cue_id, idx)
        match_requests[cid] = (idx, track_name)
        print(f"Queued track→group match: {track_name} => custom_id={cid}")

    pending_cue_data[cue_dir] = {
        "cue_id": cue_id,
        "midi_file_record": midi_file_record,
        "midi_tracks": midi_tracks,
        "midi_match_requests": match_requests,
        "audio_file_records": audio_file_records,
        "available_groups": available_groups
    }

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
            if key[1] in [1, 11]:  # mod or expression
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

    print(f"Finished processing {cue_dir} for Phase 1.")

def finalize_phase_2_3():
    if global_batch_manager_match.requests:
        print("\nExecuting track->group batch requests...")
        track_match_results = global_batch_manager_match.execute_batch()
    else:
        track_match_results = {}

    print("\nTrack->Group match results:")
    for k, v in track_match_results.items():
        print(f"  {k} => {v}")

    for cue_dir, data in pending_cue_data.items():
        cue_id = data["cue_id"]
        match_map = {}
        for custom_id, (idx, track_name) in data["midi_match_requests"].items():
            pred = track_match_results.get(custom_id, "").strip()
            if not pred:
                if data["available_groups"]:
                    pred = data["available_groups"][0]
                else:
                    pred = "UnknownGroup"
            if pred.lower() not in [g.lower() for g in data["available_groups"]]:
                best_g, _ = fuzz_process.extractOne(pred, data["available_groups"])
                pred = best_g
            match_map[track_name] = pred
        data["final_mapping"] = match_map

        group_to_tracks = {}
        for idx, track_name in data["midi_tracks"]:
            assigned_group = match_map.get(track_name)
            if assigned_group:
                group_to_tracks.setdefault(assigned_group, []).append(track_name)
        data["group_to_tracks"] = group_to_tracks

        for group_name, track_names in group_to_tracks.items():
            custom_id = f"assign_common_{cue_id}_{group_name}"
            user_prompt = f"""
You are given a set of MIDI track names and an audio group identifier.
MIDI Tracks: **{', '.join(track_names)}**
Audio Group: **{group_name}**
Available Categories: **{', '.join(INSTRUMENT_CATEGORIES)}**
Select the single best category. Output only the exact category name.
"""
            system_prompt = "You classify musical instruments into a common category."
            global_batch_manager_common.queue_chat_request(user_prompt, system_prompt, custom_id)
            print(f"Queued group->commonCategory prompt for {group_name} => {custom_id}")

def finalize_phase_4():
    if global_batch_manager_common.requests:
        print("\nExecuting group->commonCategory batch requests...")
        common_results = global_batch_manager_common.execute_batch()
    else:
        common_results = {}

    print("\nGroup->Category classification results:")
    for k, v in common_results.items():
        print(f"  {k} => {v}")

    for cue_dir, data in pending_cue_data.items():
        midi_file_rec = data["midi_file_record"]
        group_to_tracks = data.get("group_to_tracks", {})
        track_mapping = data.get("final_mapping", {})
        audio_file_records = data["audio_file_records"]

        group_common_map = {}
        for group_name, track_names in group_to_tracks.items():
            custom_id = f"assign_common_{data['cue_id']}_{group_name}"
            cat = common_results.get(custom_id, "").strip()
            if not cat:
                cat = "Strings"
            group_common_map[group_name] = cat

        for idx, track_name in data["midi_tracks"]:
            chosen_group = track_mapping.get(track_name)
            assigned_audio_id = None
            if chosen_group in audio_file_records:
                assigned_audio_id = audio_file_records[chosen_group]

            final_cat = group_common_map.get(chosen_group, "Strings")

            if assigned_audio_id is not None:
                audio_file_obj = session.query(AudioFile).filter_by(id=assigned_audio_id).first()
                if audio_file_obj:
                    audio_file_obj.instrument_category = final_cat
                    session.commit()

            insert_midi_track(
                midi_file_id=midi_file_rec.id,
                track_index=idx,
                track_name=track_name,
                instrument_category=final_cat,
                assigned_audio_file_id=assigned_audio_id
            )
            print(f"MIDI track '{track_name}': group='{chosen_group}', cat='{final_cat}' => inserted.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_cues.py <cue_base_directory> [sample_rate] [project_id]")
        sys.exit(1)

    base_dir = sys.argv[1]
    try:
        sample_rate = float(sys.argv[2]) if len(sys.argv) >= 3 else 48000
    except ValueError:
        print("Invalid sample_rate, defaulting to 48000.")
        sample_rate = 48000

    if len(sys.argv) >= 4:
        project_id = int(sys.argv[3])
    else:
        proj = insert_project("Current Project", sample_rate=sample_rate)
        project_id = proj.id
        print(f"Created a new Project (id={project_id}) at {sample_rate} Hz.")

    cue_dirs = find_cue_directories(base_dir)
    if not cue_dirs:
        print("No cues found under", base_dir)
        sys.exit(0)
    print(f"Found {len(cue_dirs)} cue dirs. Starting Phase 1...")

    for cue in tqdm(cue_dirs, desc="Processing Cues"):
        process_cue(cue, sample_rate, project_id)

    print("\n--- Phase 1 done. Now creating & executing Batch #1 for track->group matches ---")
    finalize_phase_2_3()

    print("\n--- Phase 2 & 3 done. Now creating & executing Batch #2 for group->category classification ---")
    finalize_phase_4()

    print("\nAll done. DB insertion complete!")


if __name__ == "__main__":
    main()
