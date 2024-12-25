from music21 import converter, note, chord

# Load the MusicXML file
score = converter.parse('Original.xml')  # Replace with the path to your XML file

# Iterate through all parts and measures to extract notes
for part in score.parts:
    print(f"Part: {part.id}")
    for measure in part.getElementsByClass('Measure'):
        print(f"  Measure number: {measure.number}")
        for element in measure.notes:
            if isinstance(element, note.Note):
                pitch = element.pitch
                duration = element.duration.quarterLength
                print(f"    Note: {pitch}, Duration: {duration}")
            elif isinstance(element, chord.Chord):
                pitches = [p.nameWithOctave for p in element.pitches]
                duration = element.duration.quarterLength
                print(f"    Chord: {pitches}, Duration: {duration}")
