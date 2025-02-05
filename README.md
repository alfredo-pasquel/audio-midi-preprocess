# Audio-MIDI Preprocessing Tool

This project is a comprehensive Python-based tool for pre‑processing and extracting features from MIDI and audio files. It processes individual instrument tracks as well as the final mixed audio cue, extracts detailed timing, expression, and spectral data, and stores all the information in a relational database. This processed data can later be used to train AI/ML models for music analysis, synthesis, or performance.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Database Schema](#database-schema)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Deployment & Scaling](#deployment--scaling)
- [Notes & Troubleshooting](#notes--troubleshooting)
- [License](#license)

## Overview

The tool performs the following tasks:

1. **MIDI Processing:**  
   - Extracts tempo and time signature events from MIDI files.
   - Converts MIDI ticks to time (seconds) and musical positions (bar and beat).
   - Extracts track names and filters out non-musical marker tracks.
   - Extracts detailed MIDI events (note on/off events, control changes, program changes) with timing information.

2. **Audio Processing:**  
   - Recursively scans cue directories for audio files.
   - Groups multi-channel audio files by canonical names (e.g., merging files with suffixes such as `.L`, `.R`, `.Ls`, etc.).
   - Combines individual channel files into a composite (mono) waveform.
   - Extracts mel‑spectrogram features from both instrument groups and the final mix (e.g., a 5.1 mix file containing "6mx" in its name).

3. **Mapping & Feature Extraction:**  
   - Uses OpenAI’s GPT‑3.5‑turbo model to match MIDI tracks to audio groups based on abstract naming relationships.
   - Assigns an instrument category (from a fixed list) to both MIDI tracks and audio groups using the OpenAI API.
   - Stores all extracted information into a relational database.

4. **Database Storage:**  
   - Uses SQLAlchemy to create and manage database tables.
   - Supports both a local SQLite database (for development/testing) and external PostgreSQL databases (e.g., Amazon Aurora DSQL) for production.
   - Includes tables for MIDI files, individual MIDI tracks, MIDI notes, MIDI control changes, MIDI program changes, audio files, audio features (mel‑spectrograms), and final mixes.

## Technologies Used

- **Python 3.8+**
- **Mido:** For reading and processing MIDI files.
- **Librosa & NumPy:** For audio loading, waveform processing, and mel‑spectrogram extraction.
- **OpenAI Python API:** To query GPT‑3.5‑turbo for abstract matching and instrument categorization.
- **TheFuzz:** For fuzzy matching between candidate names.
- **SQLAlchemy & Psycopg2:** For ORM database interactions (SQLite for local, PostgreSQL for production).
- **tqdm:** For displaying progress bars.
- **python-dotenv:** For loading environment variables from a `.env` file.
- **AWS CLI:** To generate authentication tokens for Amazon Aurora DSQL (if using AWS for your database).

## Database Schema

The following tables are defined:

- **midi_files:**  
  Stores MIDI file-level information (file path, tempo map, time signature map, ticks-per-beat).

- **audio_files:**  
  Stores individual audio group information (file path, canonical name, assigned instrument category).

- **audio_features:**  
  Stores features (e.g., mel‑spectrograms) extracted from audio files.

- **final_mixes:**  
  Stores the final mixed audio file for each cue (typically a 5.1 mix), with its extracted features.

- **midi_tracks:**  
  Stores information for each MIDI track (track name, index, assigned instrument category, reference to an audio file if matched).

- **midi_notes:**  
  Stores individual MIDI note events (channel, note number, velocity, start/end ticks and times, duration).

- **midi_cc:**  
  Stores MIDI control change events (channel, control number, value, tick, time).

- **midi_program_changes:**  
  Stores MIDI program change events (channel, program, tick, time).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a Virtual Environment and Install Dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root of the project with at least the following variables:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://your_username:your_token@your_db_endpoint:5432/your_database?sslmode=require
```

## Usage

The project includes a main processing script:

- **process_cues.py:**  
  This script recursively finds cue directories (each containing a MIDI folder and a PT/Audio Files folder), processes the MIDI and audio files, extracts features, maps MIDI tracks to audio groups, and stores everything into the database.

   **Example Command:**
   ```bash
   python process_cues.py "/path/to/cues/base/directory" 48000
   ```

## Deployment & Scaling

- **Local vs. Production Database:**  
  It is recommended to use an external PostgreSQL instance (e.g., Amazon Aurora DSQL) for production.

- **Handling Long Processing Times:**  
  Processing a single song can generate a very large database, so consider scaling down feature resolution or processing in batches.

- **Automating Batch Processing:**  
  Organize your cues in a standardized file tree and use `find_cue_directories()` to detect cue folders automatically.

## Notes & Troubleshooting

- **SSL and IAM Tokens:**  
  AWS Aurora DSQL connections require SSL (`sslmode=require`) and valid IAM-generated tokens as passwords.

- **Database Connection Errors:**  
  Errors like "FATAL: unable to accept connection, access denied" indicate authentication issues (e.g., expired tokens).

- **Environment Variables:**  
  Ensure `python-dotenv` is properly loading environment variables with `load_dotenv()`.

## License

This project is released under the [MIT License](LICENSE).

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainer at [your.email@example.com](mailto:your.email@example.com).
