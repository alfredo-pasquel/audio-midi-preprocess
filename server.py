#!/usr/bin/env python3
import os
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text
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

class AudioFile(Base):
    __tablename__ = "audio_files"
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True)
    canonical_name = Column(String)
    instrument_category = Column(String, nullable=True)
    tracks = relationship("MidiTrack", back_populates="audio_file")

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

# Define your database URL (here we use SQLite).
DATABASE_URL = "sqlite:///preprocessed_data.db"

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
