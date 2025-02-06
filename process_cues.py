#!/usr/bin/env python3
"""
process_cues.py

This script recursively scans a base cues directory for projects (each cue folder
must contain a "MIDI" subfolder with a .mid file and a "PT/Audio Files" subfolder).
It then processes the MIDI and audio files, extracts features, maps MIDI tracks to audio
groups using OpenAI, and inserts the data into a database.

Duplicate-checking is performed based on file_path so that cues already catalogued are skipped.
"""

import os
import sys
import glob
import json
import mido
import numpy as np
import librosa
from tqdm import tqdm
from openai import OpenAI  # using the migrated interface
from thefuzz import process
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# --- Configuration and Global Constants ---

# Instantiate the OpenAI client.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if client.api_key is None:
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

# Updated list of instrument categories.
INSTRUMENT_CATEGORIES = [
    "Strings", "Woodwinds", "Brass", "Electric Guitar", "Acoustic Guitar",
    "Piano", "Organ", "Bells", "Harp", "Synth Pulse", "Synth Pad", "Synth Bass",
    "Low Percussion", "Mid Percussion", "High Percussion", "Drums", "Orch Percussion",
    "Bass", "Double Bass", "FX", "Choir", "Solo Vocals", "Mallets", "Plucked",
    "Sub Hits", "Guitar FX", "Orch FX", "Ticker"
]

# Marker keywords to filter out non-musical or marker tracks.
MARKER_KEYWORDS = ["notes", "conductor", "orchestrator"]

# --- Function to Find Cue Directories ---

def find_cue_directories(base_dir):
    """
    Recursively search the base directory for cue directories.
    A valid cue directory is one that:
      - Contains a "MIDI" subfolder with at least one .mid file.
      - Contains a "PT/Audio Files" subfolder.
    Returns a list of full paths to valid cue directories.
    """
    cue_dirs = []
    for root, dirs, files in os.walk(base_dir):
        midi_dir = os.path.join(root, "MIDI")
        pt_audio_dir = os.path.join(root, "PT", "Audio Files")
        if os.path.isdir(midi_dir) and os.path.isdir(pt_audio_dir):
            mid_files = glob.glob(os.path.join(midi_dir, "*.mid"))
            if mid_files:
                cue_dirs.append(root)
    return cue_dirs

# --- Database Functions ---
# These functions (and the session & models) are assumed to be defined in server.py.
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
    session,  # SQLAlchemy session object
    MidiFile,   # Database model for MIDI files
    AudioFile   # Database model for Audio files
)

# Initialize the database.
init_db()

def get_or_create_cue_group(cue_path):
    """Retrieve an existing CueGroup by cue_path or create a new one."""
    from server import CueGroup  # Import here to avoid circular import issues.
    cue_group = get_cue_group_by_path(cue_path)
    if cue_group is None:
        cue_group = insert_cue_group(cue_path)
    return cue_group.id

# --- Audio File Grouping and Combination Functions ---

def get_audio_file_groups(audio_dir):
    """
    Scan the given directory for common audio file types and group them by canonical name.
    Returns a dictionary mapping canonical name -> list of full file paths.
    """
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]
    found_files = []
    for ext in audio_extensions:
        found_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    groups = {}
    # Updated list of channel suffixes to include additional variants.
    channel_suffixes = [".L", ".C", ".R", ".Ls", ".Rs", ".lfe", ".Lf", ".Rf"]
    for filepath in found_files:
        base = os.path.basename(filepath)
        name, _ = os.path.splitext(base)
        canonical = name
        for suffix in channel_suffixes:
            if canonical.endswith(suffix):
                canonical = canonical[:-len(suffix)]
                break
        groups.setdefault(canonical, []).append(filepath)
    return groups

def combine_audio_group(file_list, sample_rate):
    """
    Load each file in file_list. If only one file is present, load it with all its channels
    (interleaved mode). Otherwise, assume each file is a mono recording representing one channel,
    trim to the shortest length, and stack them to create a multi-channel composite.
    Returns an array with shape (channels, samples).
    """
    if len(file_list) == 1:
        y, sr = librosa.load(file_list[0], sr=sample_rate, mono=False)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=0)
        return y
    else:
        signals = []
        for f in file_list:
            y, sr = librosa.load(f, sr=sample_rate, mono=True)
            signals.append(y)
        if not signals:
            return None
        min_len = min(len(y) for y in signals)
        signals = [y[:min_len] for y in signals]
        composite = np.stack(signals, axis=0)
        return composite

def extract_audio_features_from_composite(y, sr=48000, n_mels=64, hop_length=512):
    """
    Compute a mel-spectrogram (in dB) from a composite waveform y.
    If y is mono (1D), compute a single spectrogram.
    If y is multi-channel (2D: channels x samples), compute a spectrogram for each channel
    and return a nested list with shape (n_mels, time_steps, channels).
    """
    if y.ndim == 1:
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.tolist()
    else:
        mel_specs = []
        for ch in range(y.shape[0]):
            mel_spec = librosa.feature.melspectrogram(y=y[ch], sr=sr, n_mels=n_mels, hop_length=hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_specs.append(mel_spec_db)
        mel_specs = np.stack(mel_specs, axis=-1)
        return mel_specs.tolist()

# --- MIDI Processing Functions ---

def tick_to_bar_beat(abs_tick, ts_events, ticks_per_beat):
    total_measures = 0
    for i, (start_tick, num, den) in enumerate(ts_events):
        next_tick = ts_events[i+1][0] if i+1 < len(ts_events) else None
        beat_ticks = ticks_per_beat * (4.0 / den)
        measure_ticks = beat_ticks * num
        if next_tick is not None and abs_tick >= next_tick:
            segment_ticks = next_tick - start_tick
            full_measures = int(segment_ticks // measure_ticks)
            total_measures += full_measures
        else:
            segment_ticks = abs_tick - start_tick
            full_measures = int(segment_ticks // measure_ticks)
            total_measures += full_measures
            remainder_ticks = segment_ticks - full_measures * measure_ticks
            beat_in_measure = remainder_ticks / beat_ticks
            return total_measures + 1, beat_in_measure + 1
    return 1, (abs_tick / ticks_per_beat) + 1

def tick_to_time(abs_tick, tempo_map, ticks_per_beat):
    if not tempo_map:
        return 0.0
    time_val = 0.0
    prev_tick = 0
    for seg in tempo_map:
        seg_time, seg_tick, tempo = seg
        if abs_tick > seg_tick:
            delta_ticks = seg_tick - prev_tick
            delta_time = (delta_ticks / ticks_per_beat) * (tempo / 1e6)
            time_val += delta_time
            prev_tick = seg_tick
        else:
            delta_ticks = abs_tick - prev_tick
            delta_time = (delta_ticks / ticks_per_beat) * (tempo / 1e6)
            time_val += delta_time
            return time_val
    last_tempo = tempo_map[-1][2]
    delta_ticks = abs_tick - prev_tick
    time_val += (delta_ticks / ticks_per_beat) * (last_tempo / 1e6)
    return time_val

def extract_tempo_and_time_signature(midi_path):
    try:
        mid = mido.MidiFile(midi_path)
    except IOError as e:
        print(f"Error opening MIDI file: {e}")
        sys.exit(1)
    ticks_per_beat = mid.ticks_per_beat
    merged_track = mido.merge_tracks(mid.tracks)
    current_tempo = 500000  # default tempo in µs per beat
    abs_time = 0.0
    abs_ticks = 0
    tempo_map = []
    time_signature_map = []
    for msg in merged_track:
        abs_ticks += msg.time
        delta_sec = mido.tick2second(msg.time, ticks_per_beat, current_tempo)
        abs_time += delta_sec
        if msg.type == "set_tempo":
            if msg.tempo == 0:
                print(f"Warning: Skipping zero tempo event at {abs_time:.3f} sec.")
            else:
                current_tempo = msg.tempo
                tempo_map.append((abs_time, abs_ticks, msg.tempo))
        elif msg.type == "time_signature":
            time_signature_map.append((abs_time, abs_ticks, msg.numerator, msg.denominator))
    if not tempo_map or tempo_map[0][1] > 0:
        tempo_map.insert(0, (0.0, 0, 500000))
    if not time_signature_map or time_signature_map[0][1] > 0:
        time_signature_map.insert(0, (0.0, 0, 4, 4))
    return tempo_map, time_signature_map, ticks_per_beat

def extract_midi_track_names(midi_path):
    mid = mido.MidiFile(midi_path)
    track_info = []
    for i, track in enumerate(mid.tracks):
        track_name = None
        for msg in track:
            if msg.type == "track_name":
                track_name = msg.name.strip()
                break
        if track_name is None:
            track_name = f"Track {i}"
        if any(keyword in track_name.lower() for keyword in MARKER_KEYWORDS):
            continue
        track_info.append((i, track_name))
    return track_info

def match_track_to_audio(track_name, canonical_names):
    prompt = (
        f"Given a MIDI track named '{track_name}', and a list of audio file canonical names: {', '.join(canonical_names)}, "
        "determine which audio group best corresponds to the MIDI track. The relationship may be abstract. "
        "Output only the exact canonical name from the list that best matches, or 'None' if no match."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that matches MIDI track names to audio file canonical names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        if answer.lower() not in [name.lower() for name in canonical_names]:
            best_match, score = process.extractOne(answer, canonical_names)
            if score >= 80:
                return best_match
            return None
        return answer
    except Exception as e:
        print("Error during OpenAI API call (match_track_to_audio):")
        print(e)
        return None

def assign_instrument_category(item_name, categories):
    allowed = ", ".join(categories)
    prompt = (
        f"Given the following list of instrument categories: {allowed}. "
        f"Assign the item '{item_name}' to exactly one category from the list. "
        "Output only the category name exactly as it appears in the list."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that assigns instrument categories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        category = response.choices[0].message.content.strip()
        if category in categories:
            return category
        else:
            best_match, score = process.extractOne(category, categories)
            if score >= 80:
                return best_match
            return None
    except Exception as e:
        print("Error during OpenAI API call (assign_instrument_category):")
        print(e)
        return None

# --- Main Processing Loop for Each Cue Directory ---

def process_cue(cue_dir, sample_rate):
    print(f"\n=== Processing Cue: {cue_dir} ===")
    # Locate the MIDI file.
    midi_dir = os.path.join(cue_dir, "MIDI")
    mid_files = glob.glob(os.path.join(midi_dir, "*.mid"))
    if not mid_files:
        print("No MIDI file found in", midi_dir)
        return
    midi_file = mid_files[0]
    print("Processing MIDI file:", midi_file)
    
    # Duplicate-check: Skip cue if MIDI file already exists in the database.
    from server import MidiFile  # Ensure MidiFile model is imported
    existing_midi = session.query(MidiFile).filter_by(file_path=midi_file).first()
    if existing_midi:
        print(f"MIDI file {midi_file} is already processed. Skipping cue.")
        return

    # Get or create the cue group record for this cue.
    cue_group_id = get_or_create_cue_group(cue_dir)

    # Process MIDI data.
    tempo_map, time_signature_map, ticks_per_beat = extract_tempo_and_time_signature(midi_file)
    ts_events = sorted([(ts_tick, num, den) for (_, ts_tick, num, den) in time_signature_map], key=lambda x: x[0])
    midi_tracks = extract_midi_track_names(midi_file)

    # Process audio groups from PT/Audio Files.
    pt_dir = os.path.join(cue_dir, "PT", "Audio Files")
    if not os.path.exists(pt_dir):
        print("Audio Files folder not found in", os.path.join(cue_dir, "PT"))
        return
    audio_groups = get_audio_file_groups(pt_dir)
    composite_audio = {}
    print("Processing Audio Groups in", pt_dir)
    for canonical, files in tqdm(audio_groups.items(), desc="Audio Groups", leave=False):
        composite = combine_audio_group(files, sample_rate)
        if composite is not None:
            composite_audio[canonical] = composite

    # Extract audio features.
    audio_features = {}
    for canonical, waveform in tqdm(composite_audio.items(), desc="Extracting Features", leave=False):
        features = extract_audio_features_from_composite(waveform, sr=int(sample_rate))
        audio_features[canonical] = features

    canonical_names = list(audio_groups.keys())
    print("Canonical Audio Groups:")
    for name in canonical_names:
        print("  ", name)

    mapping = {}
    for idx, track_name in midi_tracks:
        best_match = match_track_to_audio(track_name, canonical_names)
        mapping[track_name] = best_match
        if best_match:
            print(f"MIDI Track '{track_name}'  -->  Audio Group '{best_match}'")
        else:
            print(f"MIDI Track '{track_name}'  -->  No matching audio group found.")

    # --- Insert MIDI file record into DB ---
    midi_file_record = insert_midi_file(
        file_path=midi_file,
        tempo_map=json.dumps(tempo_map),
        time_signature_map=json.dumps(time_signature_map),
        ticks_per_beat=ticks_per_beat
    )

    # --- Process and insert final mix file ---
    final_mix_files = []
    for f in glob.glob(os.path.join(pt_dir, "*")):
        if os.path.isfile(f):
            base = os.path.basename(f)
            name, _ = os.path.splitext(base)
            if "6mx" in name.lower():
                final_mix_files.append(f)
    if final_mix_files:
        final_mix_file = final_mix_files[0]
        print("Processing Final Mix file:", final_mix_file)
        # For the final mix, we load with mono=True (adjust if you need multi-channel)
        y_final, sr_final = librosa.load(final_mix_file, sr=sample_rate, mono=True)
        final_mix_features = extract_audio_features_from_composite(y_final, sr=int(sample_rate))
        insert_final_mix(
            midi_file_id=midi_file_record.id,
            file_path=final_mix_file,
            feature_type="mel_spectrogram",
            feature_data=json.dumps(final_mix_features),
            cue_group_id=cue_group_id
        )
    else:
        print("No final mix file (containing '6MX') found in", pt_dir)

    # --- Process individual instrument audio groups ---
    audio_file_records = {}
    for canonical in tqdm(canonical_names, desc="Processing Instrument Audio Groups", leave=False):
        rep_file = audio_groups[canonical][0]
        full_path = os.path.join(pt_dir, os.path.basename(rep_file))
        audio_cat = assign_instrument_category(canonical, INSTRUMENT_CATEGORIES)
        # Duplicate-check for audio files:
        existing_audio = session.query(AudioFile).filter_by(file_path=full_path).first()
        if existing_audio:
            print(f"Audio file {full_path} already exists in the database. Skipping insertion.")
            rec = existing_audio
        else:
            rec = insert_audio_file(
                file_path=full_path,
                canonical_name=canonical,
                instrument_category=audio_cat,
                cue_group_id=cue_group_id
            )
        audio_file_records[canonical] = rec.id
        if canonical in audio_features:
            insert_audio_feature(
                audio_file_id=rec.id,
                feature_type="mel_spectrogram",
                feature_data=json.dumps(audio_features[canonical])
            )
    midi_track_ids = {}
    for idx, track_name in tqdm(midi_tracks, desc="Inserting MIDI Tracks", leave=False):
        midi_cat = assign_instrument_category(track_name, INSTRUMENT_CATEGORIES)
        assigned_audio_id = None
        if mapping.get(track_name):
            canonical = mapping[track_name]
            assigned_audio_id = audio_file_records.get(canonical)
        rec = insert_midi_track(
            midi_file_id=midi_file_record.id,
            track_index=idx,
            track_name=track_name,
            instrument_category=midi_cat,
            assigned_audio_file_id=assigned_audio_id
        )
        midi_track_ids[idx] = rec.id

    # Detailed MIDI event extraction.
    mid = mido.MidiFile(midi_file)
    allowed_track_indices = {idx for idx, _ in midi_tracks}
    for idx, track in enumerate(mid.tracks):
        if idx not in allowed_track_indices:
            continue
        cumulative_tick = 0
        pending_notes = {}
        for msg in track:
            cumulative_tick += msg.time
            event_time = tick_to_time(cumulative_tick, tempo_map, ticks_per_beat)
            if msg.type == "note_on":
                if msg.velocity > 0:
                    pending_notes[(msg.channel, msg.note)] = (cumulative_tick, event_time, msg.velocity)
                else:
                    if (msg.channel, msg.note) in pending_notes:
                        start_tick, start_time, velocity = pending_notes.pop((msg.channel, msg.note))
                        duration = event_time - start_time
                        insert_midi_note(
                            midi_track_id=midi_track_ids.get(idx),
                            channel=msg.channel,
                            note=msg.note,
                            velocity=velocity,
                            start_tick=start_tick,
                            end_tick=cumulative_tick,
                            start_time=start_time,
                            duration=duration
                        )
            elif msg.type == "note_off":
                if (msg.channel, msg.note) in pending_notes:
                    start_tick, start_time, velocity = pending_notes.pop((msg.channel, msg.note))
                    duration = event_time - start_time
                    insert_midi_note(
                        midi_track_id=midi_track_ids.get(idx),
                        channel=msg.channel,
                        note=msg.note,
                        velocity=velocity,
                        start_tick=start_tick,
                        end_tick=cumulative_tick,
                        start_time=start_time,
                        duration=duration
                    )
            elif msg.type == "control_change":
                insert_midi_cc(
                    midi_track_id=midi_track_ids.get(idx),
                    channel=msg.channel,
                    cc_number=msg.control,
                    cc_value=msg.value,
                    tick=cumulative_tick,
                    time=event_time
                )
            elif msg.type == "program_change":
                insert_midi_program_change(
                    midi_track_id=midi_track_ids.get(idx),
                    channel=msg.channel,
                    program=msg.program,
                    tick=cumulative_tick,
                    time=event_time
                )
    print(f"Finished processing cue: {cue_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_cues.py <cue_base_directory> [sample_rate]")
        sys.exit(1)
    
    cue_base_dir = sys.argv[1]
    try:
        sample_rate = float(sys.argv[2]) if len(sys.argv) >= 3 else 48000
    except ValueError:
        print("Invalid sample rate provided. Using default 48000 Hz.")
        sample_rate = 48000

    cue_dirs = find_cue_directories(cue_base_dir)
    if not cue_dirs:
        print("No cue directories found under", cue_base_dir)
        sys.exit(0)

    print(f"Found {len(cue_dirs)} cue directories. Beginning processing...")
    for cue in tqdm(cue_dirs, desc="Processing Cues"):
        process_cue(cue, sample_rate)
    print("All cues have been processed.")

if __name__ == '__main__':
    main()
