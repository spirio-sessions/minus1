import h5py
import pandas as pd
import requests
import os

# Open the metadata file
filename = "path_to_your_metadata_file.h5"
metadata_file = h5py.File(filename, 'r')


# Assume that the genre information is stored in a dataset called 'genre'
# and song details are in a dataset called 'songs'
# (You might need to explore the HDF5 structure to find the correct paths)

# Explore the structure to find the relevant datasets
def explore_h5_group(group, indent=0):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print('  ' * indent + f"Group: {key}")
            explore_h5_group(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print('  ' * indent + f"Dataset: {key} - Shape: {item.shape} - Datatype: {item.dtype}")


print("Exploring the HDF5 file structure:")
explore_h5_group(metadata_file)

# Assuming the genre dataset exists and is accessible
# Here, 'genre' is just a placeholder; replace it with the actual path to the genre dataset
genres = metadata_file['genre'][:]
songs = metadata_file['songs'][:]

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(songs)
df['genre'] = genres

# Filter for jazz songs
jazz_songs = df[df['genre'] == 'Jazz']


# Define a function to download songs given their URL or identifier
def download_song(song_id, download_path):
    # This is a placeholder URL format; adjust based on the actual source
    url = f"http://example.com/song/{song_id}.mp3"
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(download_path, f"{song_id}.mp3"), 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download song {song_id}")


# Create a directory to save the downloaded songs
download_path = "path_to_save_jazz_songs"
os.makedirs(download_path, exist_ok=True)

# Download the filtered jazz songs
for song_id in jazz_songs['song_id']:  # Adjust based on the actual column name
    download_song(song_id, download_path)

# Close the metadata file
metadata_file.close()
