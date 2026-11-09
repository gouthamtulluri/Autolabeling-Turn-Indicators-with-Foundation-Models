#-----------------------------------------------------------------------------------
# MIT License

# Copyright (c) 2025 Computer Vision for Intelligent Mobility Systems

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#-----------------------------------------------------------------------------------


import os
import json


def load_annotation_data(labels_path):
    """
    Loads JSON annotation data from a specified file path.
    Args:
        labels_path (str): The absolute or relative path to the JSON labels file.
    Returns:
        dict or None: The parsed JSON data (typically in track-wise format), 
                      or None if the provided path is None.
    """
    if labels_path is None:
        return None
    # Load the JSON file into a Python dictionary
    with open(os.path.join(labels_path), 'r') as f:
        labels_trackwise = json.load(f)
    
    return labels_trackwise


def convert_labels_from_tracks_wise_to_timestamp_wise(labels, skip_tracks=[]):
    """
    Converts track-wise labels into a frame-by-frame (timestamp-wise) format.
    Args:
        labels (dict): The original track-wise dictionary loaded from JSON, 
                       containing a "tracks" list.
    Returns:
        dict: {
            timestamp: [
                {
                    "track_id": ...,
                    "object_type": ...,
                    "position": ...,
                    "orientation": ...,
                    "dimension": ...,
                    "attributes": {...}
                },
                ...
            ],
            ...
        }
    """
    if labels is None:
        return None
    labels_new = {}
    # Iterate through every unique vehicle/object track
    for track in labels['tracks']:
        
        # Ignore explicitly skipped tracks
        if track['track_id'] in skip_tracks:
            continue
            
        # Unpack the track data frame-by-frame
        for timestamp_index, timestamp in enumerate(track['timestamps']):
            
            # Initialize the timestamp array if it doesn't exist yet
            if timestamp not in labels_new.keys():
                labels_new[timestamp] = []
            # Extract geometric and semantic data for this specific frame
            this_dict = {
                'track_id': track['track_id'],
                'object_type': track['object_type'],
                'position': track['positions'][timestamp_index],
                'orientation': track['orientations'][timestamp_index],
                # Dimensions might be static (len == 1) or dynamic across frames
                'dimension': track['dimensions'][0] if len(track['dimensions']) == 1 else track['dimensions'][timestamp_index],
                'attributes': dict(),
            }
            # Handle custom attributes (checking if they are static or dynamic per-frame)
            for k, v in track['attributes'].items():
                if isinstance(v, list):
                    this_dict['attributes'][k] = v[timestamp_index]
                else:
                    this_dict['attributes'][k] = v
            # Append the extracted object to the specific timestamp container
            labels_new[timestamp].append(this_dict)
    return labels_new


def convert_from_timestamp_wise_to_tracks_wise(ts_wise_labels):
    """
    Converts timestamp-wise labels back into track-wise format.

    Args:
        ts_wise_labels (dict): {timestamp: [object_dicts]} — as produced by
                               convert_labels_from_tracks_wise_to_timestamp_wise().

    Returns:
        dict: {
            "tracks": [
                {
                    "track_id": ...,
                    "object_type": ...,
                    "timestamps": [...],
                    "positions": [...],
                    "orientations": [...],
                    "dimensions": [...],
                    "attributes": {...}
                },
                ...
            ]
        }
    """
    if ts_wise_labels is None:
        return None

    tracks_dict = {}

    # Iterate through each timestamp and its annotations
    for timestamp, objects in ts_wise_labels.items():
        for obj in objects:
            tid = obj["track_id"]

            # Initialize a new track if not seen before
            if tid not in tracks_dict:
                tracks_dict[tid] = {
                    "track_id": tid,
                    "object_type": obj["object_type"],
                    "timestamps": [],
                    "positions": [],
                    "orientations": [],
                    "dimensions": [],
                    "attributes": {},
                }

            # Append timestamp data
            tracks_dict[tid]["timestamps"].append(timestamp)
            tracks_dict[tid]["positions"].append(obj["position"])
            tracks_dict[tid]["orientations"].append(obj["orientation"])
            tracks_dict[tid]["dimensions"].append(obj["dimension"])

            # Handle attributes (per-frame or static)
            for k, v in obj["attributes"].items():
                # static
                if k not in ['4', '5']:
                    tracks_dict[tid]["attributes"][k] = v
                    continue

                if k not in tracks_dict[tid]["attributes"]:
                    # Initialize as list if attribute varies per timestamp
                    tracks_dict[tid]["attributes"][k] = [v]
                else:
                    tracks_dict[tid]["attributes"][k].append(v)

    # Convert dictionary to track list format
    tracks_list = {"tracks": list(tracks_dict.values())}

    return tracks_list
