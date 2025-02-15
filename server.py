#!/usr/bin/env python3
"""
server.py

This module defines the database schema and helper functions for inserting and retrieving
data from the database. SQLAlchemy is used as the ORM. Environment variables (via python‑dotenv)
are used to supply the DATABASE_URL.
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Float, LargeBinary
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

# --- CueGroup Model ---
class CueGroup(Base):
    __tablename__ = "cue_groups"
    id = Column(Integer, primary_key=True)
    cue_path = Column(String, unique=True)
    # Relationships:
    audio_files = relationship("AudioFile", back_populates="cue_group")
    final_mixes = relationship("FinalMix", back_populates="cue_group")
    midi_files = relationship("MidiFile", back_populates="cue_group")

# --- Project Model ---
class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    sample_rate = Column(Integer, nullable=True)  # New column for sample rate
    midi_files = relationship("MidiFile", back_populates="project")
    audio_files = relationship("AudioFile", back_populates="project")
    final_mixes = relationship("FinalMix", back_populates="project")

# --- MidiFile Model ---
class MidiFile(Base):
    __tablename__ = "midi_files"
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True)
    tempo_map = Column(Text)           # stored as JSON
    time_signature_map = Column(Text)    # stored as JSON
    ticks_per_beat = Column(Integer)
    cue_group_id = Column(Integer, ForeignKey("cue_groups.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    project = relationship("Project", back_populates="midi_files")
    cue_group = relationship("CueGroup", back_populates="midi_files")
    tracks = relationship("MidiTrack", back_populates="midi_file")
    final_mix = relationship("FinalMix", back_populates="midi_file", uselist=False)

# --- AudioFile Model ---
class AudioFile(Base):
    __tablename__ = "audio_files"
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True)
    canonical_name = Column(String)
    instrument_category = Column(String, nullable=True)
    cue_group_id = Column(Integer, ForeignKey("cue_groups.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    chunk_size = Column(Integer, nullable=True)  # NEW: store the chunk size used

    project = relationship("Project", back_populates="audio_files")
    cue_group = relationship("CueGroup", back_populates="audio_files")
    features = relationship("AudioFeature", back_populates="audio_file")
    tracks = relationship("MidiTrack", back_populates="audio_file")

# --- AudioFeature Model ---
class AudioFeature(Base):
    __tablename__ = "audio_features"
    id = Column(Integer, primary_key=True)
    audio_file_id = Column(Integer, ForeignKey("audio_files.id"))
    feature_type = Column(String)  # e.g., "encoded_audio_tokens"
    feature_data = Column(LargeBinary)    # Stored as binary (npy format)
    audio_file = relationship("AudioFile", back_populates="features")

# --- FinalMix Model ---
class FinalMix(Base):
    __tablename__ = "final_mixes"
    id = Column(Integer, primary_key=True)
    midi_file_id = Column(Integer, ForeignKey("midi_files.id"), unique=True)
    file_path = Column(String, unique=True)
    feature_type = Column(String)  # e.g., "encoded_audio_tokens"
    feature_data = Column(LargeBinary)    # Stored as binary (npy format)
    cue_group_id = Column(Integer, ForeignKey("cue_groups.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    chunk_size = Column(Integer, nullable=True)  # NEW: store the chunk size used

    project = relationship("Project", back_populates="final_mixes")
    cue_group = relationship("CueGroup", back_populates="final_mixes")
    midi_file = relationship("MidiFile", back_populates="final_mix")

# --- MidiTrack Model ---
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

# --- MidiNote Model ---
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

# --- MidiCC Model ---
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

# --- MidiProgramChange Model ---
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
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def init_db():
    Base.metadata.create_all(engine)

def insert_cue_group(cue_path):
    cue_group = CueGroup(cue_path=cue_path)
    session.add(cue_group)
    session.commit()
    return cue_group

def get_cue_group_by_path(cue_path):
    return session.query(CueGroup).filter_by(cue_path=cue_path).first()

def insert_project(name, sample_rate=None):
    project = Project(name=name, sample_rate=sample_rate)
    session.add(project)
    session.commit()
    return project

def insert_midi_file(file_path, tempo_map, time_signature_map, ticks_per_beat, cue_group_id=None, project_id=None):
    record = MidiFile(
        file_path=file_path,
        tempo_map=tempo_map,
        time_signature_map=time_signature_map,
        ticks_per_beat=ticks_per_beat,
        cue_group_id=cue_group_id,
        project_id=project_id
    )
    session.add(record)
    session.commit()
    return record

def insert_audio_file(file_path, canonical_name, instrument_category, cue_group_id=None, project_id=None, chunk_size=None):
    record = AudioFile(
        file_path=file_path,
        canonical_name=canonical_name,
        instrument_category=instrument_category,
        cue_group_id=cue_group_id,
        project_id=project_id,
        chunk_size=chunk_size
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

def insert_final_mix(midi_file_id, file_path, feature_type, feature_data, cue_group_id=None, project_id=None, chunk_size=None):
    record = FinalMix(
        midi_file_id=midi_file_id,
        file_path=file_path,
        feature_type=feature_type,
        feature_data=feature_data,
        cue_group_id=cue_group_id,
        project_id=project_id,
        chunk_size=chunk_size
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
