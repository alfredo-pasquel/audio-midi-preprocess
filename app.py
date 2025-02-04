#!/usr/bin/env python3
import os
import sys
import glob
import json
import mido
from openai import OpenAI  # Using the migrated interface
from thefuzz import process  # For fuzzy matching

# --- Configuration and Constants ---

# Updated list of instrument categories (you can adjust this list as needed)
INSTRUMENT_CATEGORIES = [
    "Strings", "Woodwinds", "Brass", "Electric Guitar", "Acoustic Guitar",
    "Piano", "Organ", "Bells", "Harp", "Synth Pulse", "Synth Pad", "Synth Bass",
    "Low Percussion", "Mid Percussion", "High Percussion", "Drums", "Orch Percussion", "Bass", "Double Bass",
    "FX", "Choir", "Solo Vocals", "Mallets", "Plucked", "Sub Hits",
    "Guitar FX", "Orch FX", "Ticker"
]

# Marker keywords to filter out non-musical or marker tracks.
MARKER_KEYWORDS = ["notes", "conductor", "orchestrator"]

# Instantiate the OpenAI client.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if client.api_key is None:
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

# --- Import Database Functions from server.py ---
from server import init_db, insert_midi_file, insert_audio_file, insert_midi_track

# Initialize the database (creates tables if they don't exist)
init_db()

# --- Helper Functions for Data Processing ---

def tick_to_bar_beat(abs_tick, ts_events, ticks_per_beat):
    """Convert an absolute tick value into a (bar, beat) position."""
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

def extract_tempo_and_time_signature(midi_path):
    """Extract tempo and time signature events from a MIDI file."""
    try:
        mid = mido.MidiFile(midi_path)
    except IOError as e:
        print(f"Error opening MIDI file: {e}")
        sys.exit(1)
    ticks_per_beat = mid.ticks_per_beat
    merged_track = mido.merge_tracks(mid.tracks)
    current_tempo = 500000  # default tempo (µs per beat) = 120 BPM
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
    """Extract track names from a MIDI file, skipping tracks with marker keywords."""
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

def get_audio_files(audio_dir):
    """
    Scan the given directory for common audio file types.
    For multi-channel files (with suffixes like .L, .C, .R, etc.), return one representative file per canonical name.
    """
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]
    found_files = []
    for ext in audio_extensions:
        found_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    canonical_map = {}
    channel_suffixes = [".L", ".C", ".R", ".Ls", ".Rs", ".lfe"]
    for filepath in found_files:
        base = os.path.basename(filepath)
        name, ext = os.path.splitext(base)
        canonical = name
        for suffix in channel_suffixes:
            if canonical.endswith(suffix):
                canonical = canonical[:-len(suffix)]
                break
        if canonical not in canonical_map:
            canonical_map[canonical] = base
    return list(canonical_map.values())

def match_track_to_audio(track_name, audio_files):
    """
    Use the OpenAI API (with fuzzy matching as needed) to choose which audio file best matches the MIDI track.
    """
    prompt = (
        f"Given a MIDI track named '{track_name}', and a list of audio file names: {', '.join(audio_files)}, "
        "determine which audio file best corresponds to the MIDI track. The names may not match exactly and "
        "the relationship can be abstract (for example, a 'timpani' track might correspond to an audio file "
        "named 'orchestral low perc'). Output only the exact file name from the list that best matches, or 'None' if no match."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that matches MIDI track names to audio file names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        if answer not in audio_files:
            best_match, score = process.extractOne(answer, audio_files)
            if score >= 80:
                return best_match
            return None
        return answer
    except Exception as e:
        print("Error during OpenAI API call (match_track_to_audio):")
        print(e)
        return None

def assign_instrument_category(item_name, categories):
    """
    Use the OpenAI API to assign an instrument category from a fixed list to a given item
    (MIDI track name or audio file name). Output only one category exactly as it appears in the list.
    """
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

# --- Main Application Logic ---

def main():
    if len(sys.argv) < 3:
        print("Usage: python app.py <midi_file> <audio_files_directory> [sample_rate]")
        sys.exit(1)
    
    midi_file = sys.argv[1]
    audio_dir = sys.argv[2]
    sample_rate = 44100  # default sample rate in Hz
    if len(sys.argv) >= 4:
        try:
            sample_rate = float(sys.argv[3])
        except ValueError:
            print("Invalid sample rate provided. Using default 44100 Hz.")
            sample_rate = 44100

    # Process the MIDI file.
    tempo_map, time_signature_map, ticks_per_beat = extract_tempo_and_time_signature(midi_file)
    ts_events = sorted([(ts_tick, num, den) for (_, ts_tick, num, den) in time_signature_map], key=lambda x: x[0])
    
    print("=== Tempo Map ===")
    for abs_time, abs_tick, tempo in tempo_map:
        bpm = mido.tempo2bpm(tempo)
        ms = abs_time * 1000
        sample_pos = abs_time * sample_rate
        bar, beat = tick_to_bar_beat(abs_tick, ts_events, ticks_per_beat)
        print(f"Bar {bar:3d}, Beat {beat:5.2f} | Time: {abs_time:7.3f} sec, {ms:8.1f} ms, Sample: {sample_pos:10.0f} | Tempo: {bpm:6.2f} BPM")
    
    print("\n=== Time Signature Map ===")
    for abs_time, ts_tick, num, den in time_signature_map:
        bar, beat = tick_to_bar_beat(ts_tick, ts_events, ticks_per_beat)
        print(f"Bar {bar:3d}, Beat {beat:5.2f} | Time: {abs_time:7.3f} sec | Time Signature: {num}/{den}")
    
    midi_tracks = extract_midi_track_names(midi_file)
    print("\n=== MIDI Track Names (Filtered) ===")
    for idx, name in midi_tracks:
        print(f"Track {idx}: {name}")
    
    audio_files = get_audio_files(audio_dir)
    print("\n=== Audio Files Found (Merged Multi-Channel Files) ===")
    for f in audio_files:
        print(f"  {f}")
    
    # Determine the mapping between MIDI tracks and audio files.
    mapping = {}
    print("\n=== Mapping MIDI Tracks to Audio Files ===")
    for idx, track_name in midi_tracks:
        best_match = match_track_to_audio(track_name, audio_files)
        mapping[track_name] = best_match
        if best_match:
            print(f"MIDI Track '{track_name}'  -->  Audio File '{best_match}'")
        else:
            print(f"MIDI Track '{track_name}'  -->  No matching audio file found.")
    
    # --- Insert Data into the Database ---
    # Insert the MIDI file record.
    midi_file_record = insert_midi_file(
        file_path=midi_file,
        tempo_map=json.dumps(tempo_map),
        time_signature_map=json.dumps(time_signature_map),
        ticks_per_beat=ticks_per_beat
    )
    
    # Insert AudioFile records (with instrument category assignment).
    audio_file_records = {}
    for af in audio_files:
        audio_cat = assign_instrument_category(af, INSTRUMENT_CATEGORIES)
        rec = insert_audio_file(
            file_path=os.path.join(audio_dir, af),
            canonical_name=os.path.splitext(af)[0],
            instrument_category=audio_cat
        )
        audio_file_records[af] = rec.id
    
    # Insert MidiTrack records (with instrument category assignment).
    for idx, track_name in midi_tracks:
        midi_cat = assign_instrument_category(track_name, INSTRUMENT_CATEGORIES)
        assigned_audio_id = None
        if mapping.get(track_name):
            af_name = mapping[track_name]
            assigned_audio_id = audio_file_records.get(af_name)
        insert_midi_track(
            midi_file_id=midi_file_record.id,
            track_index=idx,
            track_name=track_name,
            instrument_category=midi_cat,
            assigned_audio_file_id=assigned_audio_id
        )
    
    print("\nDatabase records have been created in 'preprocessed_data.db'.")

if __name__ == '__main__':
    main()
