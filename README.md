# Audio-MIDI Preprocessing Tool

This project is a comprehensive Python-based tool for pre-processing and extracting features from MIDI and audio files. It processes individual instrument tracks as well as the final mixed audio cue, extracts detailed timing, expression, and spectral data, and stores all the information in a relational database. The processed data can later be used to train AI/ML models for music analysis, synthesis, or performance.

> **Note:** The tool now features detailed progress bars for each step (using `tqdm`) so you can monitor the progress globally (per cue directory) and for each major processing step (audio loading, feature extraction, MIDI event processing, etc.). Additionally, if no project ID is provided at runtime, a default project is automatically created in the database.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Database Schema](#database-schema)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Deployment & Scaling](#deployment--scaling)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

---

## Overview

The Audio-MIDI Preprocessing Tool processes cue directories (each containing a `MIDI` folder and a `PT/Audio Files` folder) by:

### MIDI Processing:
- Extracting tempo and time signature events.
- Converting MIDI ticks to seconds as well as to musical positions (bars and beats).
- Extracting track names while filtering out non-musical marker tracks.
- Collecting detailed MIDI events (note on/off events, control changes, and program changes) with timing information.

### Audio Processing:
- Recursively scanning cue directories for audio files.
- Grouping multi-channel audio files by their canonical names (removing recognized channel suffixes such as `.L`, `.R`, etc.).
- Combining individual channel files into a composite multi-channel waveform.
- Extracting mel-spectrogram features from both instrument groups and the final mix (e.g. files containing `6MX` or `REF` in their name).

### Mapping & Feature Extraction:
- Using OpenAI’s gpt-4o API to match MIDI tracks to audio groups based on abstract naming relationships.
- Assigning instrument categories (from a fixed list) to both MIDI tracks and audio groups.
- Thinning out redundant MIDI control change events (especially for expression (CC 11) and modulation (CC 1)).

### Database Storage:
- Inserting all extracted information into a relational database using SQLAlchemy.
- Creating tables for MIDI files, MIDI tracks, MIDI notes, MIDI control changes, MIDI program changes, audio files, audio features, and final mixes.
- Automatically creating a default project if one is not provided.

---

## Features

- **Recursive Processing:** Automatically finds and processes all cue directories.
- **Detailed Timing Extraction:** Converts MIDI ticks to seconds and calculates bar/beat positions.
- **Audio Feature Extraction:** Generates mel-spectrograms for both individual instrument groups and final mix cues.
- **MIDI Event Parsing:** Captures note events, control changes (with thinning of redundant events), and program changes.
- **OpenAI Integration:** Uses gpt-4o for matching MIDI track names to audio file groups and for instrument categorization.
- **Progress Visualization:** Displays detailed progress bars for the overall process and for individual processing steps.
- **Database Integration:** Uses SQLAlchemy to store all processed data into a relational database (supports both local SQLite and external PostgreSQL).
- **Default Project Creation:** If no project ID is provided, the tool creates a default project.

---

## Technologies Used

- **Python 3.8+**
- **Mido:** For MIDI file parsing.
- **Librosa & NumPy:** For audio processing and mel-spectrogram extraction.
- **OpenAI Python API:** For gpt-4o-powered matching and categorization.
- **TheFuzz:** For fuzzy string matching.
- **SQLAlchemy & Psycopg2:** For ORM-based database operations.
- **tqdm:** For progress bar visualization.
- **python-dotenv:** For loading configuration from a `.env` file.
- **AWS CLI (optional):** For generating IAM tokens when using Amazon Aurora DSQL.

---

## Database Schema

The database includes the following tables:

- `cue_groups`: Stores cue directory paths and links to related MIDI, audio, and final mix entries.
- `projects`: Stores project information (A default project is created if none is specified).
- `midi_files`: Stores MIDI file metadata (file path, tempo map, time signature map, ticks-per-beat, project and cue group associations).
- `audio_files`: Stores audio group information (file path, canonical name, instrument category).
- `audio_features`: Stores extracted audio features (e.g., mel-spectrograms in binary `.npy` format).
- `final_mixes`: Stores information about the final mixed audio cue (file path and extracted features).
- `midi_tracks`: Stores per-track MIDI information (track name, index, instrument category, and matched audio file).
- `midi_notes`: Stores individual MIDI note events with timing and velocity information.
- `midi_cc`: Stores MIDI control change events (channel, CC number, value, tick, and time).
- `midi_program_changes`: Stores MIDI program change events.

---

## Installation

1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/audio-midi-preprocess.git
cd audio-midi-preprocess
```

2. **Create a Virtual Environment and Install Dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the root directory with at least the following variables:
```ini
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://your_username:your_password@your_db_endpoint:5432/your_database?sslmode=require
```
For local development, you may use SQLite by setting:
```ini
DATABASE_URL=sqlite:///your_database.db
```

---

## Usage

The main script is `process_cues.py`. It will:
- Recursively find cue directories.
- Process MIDI and audio files.
- Extract features and timing information.
- Insert all processed data into the database.
- Display progress bars for each processing stage.

### Command-line Usage:
```bash
python process_cues.py "/path/to/cues/base/directory" [sample_rate] [project_id]
```
- `/path/to/cues/base/directory`: Base directory where cue folders are located.
- `sample_rate`: (Optional, default: 48000) The sample rate to use when loading audio files.
- `project_id`: (Optional) If provided, the processed data will be linked to the given project. If not provided, a default project named "Current Project" will be created.

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainer at info@alfredopasquel.com.

Happy processing!
