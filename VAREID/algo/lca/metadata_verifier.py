import sqlite3
import math
import pandas as pd
import os

from pathlib import Path


def parse_datetime(value):
    """Parse a datetime string, treating a bare number as Unix epoch seconds.

    pd.to_datetime reads a bare number as nanoseconds, which would put every
    image within seconds of 1970-01-01.
    """
    try:
        return pd.to_datetime(float(value), unit='s')
    except (ValueError, TypeError):
        return pd.to_datetime(value)


class metadata_verifier(object):
    def __init__(self, data_df, human_verifier, node2uuid):
        self.data_df = data_df
        self.node2uuid = node2uuid
        self.uuid2node = {val:key for (key, val) in node2uuid.items()}
        self.human_verifier = human_verifier

    def get_image_metadata(self, uuid):
        # Get the file_name from the dataframe
        longitude = self.data_df.loc[self.data_df['uuid_x'] == uuid, "longitude"].squeeze()
        latitude = self.data_df.loc[self.data_df['uuid_x'] == uuid, "latitude"].squeeze()
        datetime = self.data_df.loc[self.data_df['uuid_x'] == uuid, "datetime"].squeeze()
        file_name = self.data_df.loc[self.data_df['uuid_x'] == uuid, "file_name"].squeeze()
        return [str(longitude), str(latitude), str(datetime), str(file_name)]

    def convert_query(self, n0, n1):
        uuid1 = self.node2uuid[n0]
        uuid2 = self.node2uuid[n1]
        return (uuid1, self.get_image_metadata(uuid1), uuid2, self.get_image_metadata(uuid2))

    def __call__(self, query):
        nodes_to_review = []
        nodes_query = [self.convert_query(n0, n1) for (n0, n1) in query]

        # Thresholds for plausibility (customize as needed)
        MAX_ZEBRA_SPEED_KMH = 65  # Maximum plausible speed in km/h

        for (uuid1, meta1, uuid2, meta2) in nodes_query:
            file1, file2 = meta1[-1], meta2[-1]
            if file1 == file2:
                continue

            # Extract datetimes and locations
            try:
                dt1 = parse_datetime(meta1[2])
                dt2 = parse_datetime(meta2[2])
                time_diff_hours = abs((dt1 - dt2).total_seconds() / 3600)
            except Exception:
                continue

            try:
                lon1, lat1 = float(meta1[0]), float(meta1[1])
                lon2, lat2 = float(meta2[0]), float(meta2[1])
            except Exception:
                continue

            if lon1 == -1 or lat1 == -1 or lon2 == -1 or lat2 == -1:
                nodes_to_review.append((uuid1, uuid2))
                continue

            # Haversine distance calculation
            R = 6371
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            delta_phi = math.radians(lat2 - lat1)
            delta_lambda = math.radians(lon2 - lon1)
            a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance_km = R * c

            # Avoid division by zero for same timestamp
            if time_diff_hours <= 0.1:
                plausible = distance_km < 0.1  # 100 meters, basically the same spot
            else:
                required_speed = distance_km / time_diff_hours
                plausible = required_speed <= MAX_ZEBRA_SPEED_KMH

            if plausible:
                nodes_to_review.append((uuid1, uuid2))

        return nodes_to_review