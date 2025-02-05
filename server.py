from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Float
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MidiFile(Base):
    __tablename__ = "midi_files"
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True)
    tempo_map = Column(Text)           # stored as JSON
    time_signature_map = Column(Text)    # stored as JSON
    ticks_per_beat = Column(Integer)
    tracks = relationship("MidiTrack", back_populates="midi_file")
    # Add a one-to-one relationship with the final mix.
    final_mix = relationship("FinalMix", back_populates="midi_file", uselist=False)

class AudioFile(Base):
    __tablename__ = "audio_files"
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True)
    canonical_name = Column(String)
    instrument_category = Column(String, nullable=True)
    features = relationship("AudioFeature", back_populates="audio_file")
    tracks = relationship("MidiTrack", back_populates="audio_file")

class AudioFeature(Base):
    __tablename__ = "audio_features"
    id = Column(Integer, primary_key=True)
    audio_file_id = Column(Integer, ForeignKey("audio_files.id"))
    feature_type = Column(String)  # e.g., "mel_spectrogram"
    feature_data = Column(Text)    # JSON string representing the feature array
    audio_file = relationship("AudioFile", back_populates="features")

class FinalMix(Base):
    __tablename__ = "final_mixes"
    id = Column(Integer, primary_key=True)
    midi_file_id = Column(Integer, ForeignKey("midi_files.id"), unique=True)
    file_path = Column(String, unique=True)
    feature_type = Column(String)  # e.g., "mel_spectrogram"
    feature_data = Column(Text)    # JSON string representing the feature array
    midi_file = relationship("MidiFile", back_populates="final_mix")

class MidiTrack(Base):
    __tablename__ = "midi_tracks"
    id = Column(Integer, primary_key=True)
    midi_file_id = Column(Integer, ForeignKey("midi_files.id"))
    track_index = Column(Integer)
    track_name = Column(String)
    instrument_category = Column(String, nullable=True)
    assigned_audio_file_id = Column(Integer, ForeignKey("audio_files.id"), nullable=True)
    midi_file = relationship("MidiFile", back_populates="tracks")
    audio_file = relationship("AudioFile", back_populates="tracks")
    notes = relationship("MidiNote", back_populates="midi_track")
    cc_events = relationship("MidiCC", back_populates="midi_track")
    program_changes = relationship("MidiProgramChange", back_populates="midi_track")

class MidiNote(Base):
    __tablename__ = "midi_notes"
    id = Column(Integer, primary_key=True)
    midi_track_id = Column(Integer, ForeignKey("midi_tracks.id"))
    channel = Column(Integer)
    note = Column(Integer)  # MIDI note number
    velocity = Column(Integer)
    start_tick = Column(Integer)
    end_tick = Column(Integer)
    start_time = Column(Float)  # seconds
    duration = Column(Float)    # seconds
    midi_track = relationship("MidiTrack", back_populates="notes")

class MidiCC(Base):
    __tablename__ = "midi_cc"
    id = Column(Integer, primary_key=True)
    midi_track_id = Column(Integer, ForeignKey("midi_tracks.id"))
    channel = Column(Integer)
    cc_number = Column(Integer)
    cc_value = Column(Integer)
    tick = Column(Integer)
    time = Column(Float)  # seconds
    midi_track = relationship("MidiTrack", back_populates="cc_events")

class MidiProgramChange(Base):
    __tablename__ = "midi_program_changes"
    id = Column(Integer, primary_key=True)
    midi_track_id = Column(Integer, ForeignKey("midi_tracks.id"))
    channel = Column(Integer)
    program = Column(Integer)
    tick = Column(Integer)
    time = Column(Float)  # seconds
    midi_track = relationship("MidiTrack", back_populates="program_changes")

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")  # Now read from the .env file.
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def init_db():
    Base.metadata.create_all(engine)

def insert_midi_file(file_path, tempo_map, time_signature_map, ticks_per_beat):
    record = MidiFile(
        file_path=file_path,
        tempo_map=tempo_map,
        time_signature_map=time_signature_map,
        ticks_per_beat=ticks_per_beat
    )
    session.add(record)
    session.commit()
    return record

def insert_audio_file(file_path, canonical_name, instrument_category):
    record = AudioFile(
        file_path=file_path,
        canonical_name=canonical_name,
        instrument_category=instrument_category
    )
    session.add(record)
    session.commit()
    return record

def insert_audio_feature(audio_file_id, feature_type, feature_data):
    record = AudioFeature(
        audio_file_id=audio_file_id,
        feature_type=feature_type,
        feature_data=feature_data
    )
    session.add(record)
    session.commit()
    return record

def insert_final_mix(midi_file_id, file_path, feature_type, feature_data):
    record = FinalMix(
        midi_file_id=midi_file_id,
        file_path=file_path,
        feature_type=feature_type,
        feature_data=feature_data
    )
    session.add(record)
    session.commit()
    return record

def insert_midi_track(midi_file_id, track_index, track_name, instrument_category, assigned_audio_file_id):
    record = MidiTrack(
        midi_file_id=midi_file_id,
        track_index=track_index,
        track_name=track_name,
        instrument_category=instrument_category,
        assigned_audio_file_id=assigned_audio_file_id
    )
    session.add(record)
    session.commit()
    return record

def insert_midi_note(midi_track_id, channel, note, velocity, start_tick, end_tick, start_time, duration):
    record = MidiNote(
        midi_track_id=midi_track_id,
        channel=channel,
        note=note,
        velocity=velocity,
        start_tick=start_tick,
        end_tick=end_tick,
        start_time=start_time,
        duration=duration
    )
    session.add(record)
    session.commit()
    return record

def insert_midi_cc(midi_track_id, channel, cc_number, cc_value, tick, time):
    record = MidiCC(
        midi_track_id=midi_track_id,
        channel=channel,
        cc_number=cc_number,
        cc_value=cc_value,
        tick=tick,
        time=time
    )
    session.add(record)
    session.commit()
    return record

def insert_midi_program_change(midi_track_id, channel, program, tick, time):
    record = MidiProgramChange(
        midi_track_id=midi_track_id,
        channel=channel,
        program=program,
        tick=tick,
        time=time
    )
    session.add(record)
    session.commit()
    return record
