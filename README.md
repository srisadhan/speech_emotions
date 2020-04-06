speech_emotion_learning
==============================

learning representations for emotions in speech

# RAVDESS file information.
The speech files have the following information encoded in the filename. The numbers denote the placeholders in the filename separated by '-'.
- 1) Modality:             (01 = full-AV, 02 = video-only, 03 = audio-only).
- 2) Vocal channel:        (01 = speech, 02 = song).
- 3) Emotion:              (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 =      disgust, 08 = surprised).
- 4) Emotional intensity:  (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
- 5) Statement:            (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
- 6) Repetition:           (01 = 1st repetition, 02 = 2nd repetition).
- 7) Actor:                (01 to 24. Odd numbered actors are male, even numbered actors are female).


# Instructions to run the file:
Replace the 'skip' term with 'run' in the functions that you want to run as shown below
- To **skip** : with skip_run('**skip**', 'download_RAVDESS_data') as check, check():
- To **run**  : with skip_run('**run**', 'download_RAVDESS_data') as check, check():

All the files are stored as .h5 files in data folder for example "data/interim/filename.h5"

the structure of the dictionaries:
 - speech1 ("Kids are talking by the door") and speech2 ("Dogs are sitting by the door") are separatedly stored as dictionaries.
 - the structure of dictionaries are data['Actor_#']['emotion_#']['intensity_#']['repete_#'], here # are the numbers mentioned in the file information above