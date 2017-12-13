import csv
import sys

data_set_path = "./input/songdata.csv"

def process_args(args):
  if args[1]=="-all":
    return True, "**all**", "all.out"
  elif len(args)<5:
    print("Usage:")
    print("  get_lyrics.py -a [artist] -output [output_file]")
    return False, 0, 0
  elif args[1]=="-a" and args[3]=="-output":
    return True, args[2], args[4]
  else:
    print("Usage:")
    print("  get_lyrics.py -a [artist] -output [output_file]")
    return False, 0, 0

def line_is_good(line):
  line = str.replace(line, " ", "")
  if line=="":
    return False
  elif "verse" in str.lower(line):
    return False
  elif "chorus" in str.lower(line):
    return False
  elif "(" in str.lower(line):
    return False
  elif "[" in str.lower(line):
    return False
  else:
    return True

good, in_artist, output_file = process_args(sys.argv)
if good:
  with open("./input/songdata.csv", "r") as csvfile:
    song_reader = csv.reader(csvfile, delimiter=",")
    i=0
    artists = {}
    songs = {}
    for row in song_reader:
      if not i==0:
        artist = str.lower(row[0])
        song = row[3]
        if artist not in artists:
          artists[artist] = 1
          songs[artist] = [song]
        else:
          artists[artist]+=1
          songs[artist].append(song)
      i+=1

with open(output_file, "w") as out:
  for artist, count in artists.items():
    if artist == str.lower(in_artist) or in_artist == "**all**":
      for song in songs[artist]:
        for line in song.split("\n"):
          if line_is_good(line):
            out.write("{0}\n".format(line))

