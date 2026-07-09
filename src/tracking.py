from src.models import Track


class TrackState:
    def __init__(self, n_confirm=4, m_persist=20):
        self.n_confirm = n_confirm
        self.m_persist = m_persist
        self.tracks: dict[int, Track] = {}

    def update(self, seen_ids) -> list[int]:
        removed_ids = []
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            if track_id in seen_ids:
                track.seen_count += 1
                track.missed_count = 0
                if track.seen_count >= self.n_confirm:
                    track.confirmed = True
            else:
                track.missed_count += 1

            if track.missed_count > self.m_persist:
                del self.tracks[track_id]
                removed_ids.append(track_id)
        for track_id in seen_ids:
            if track_id not in self.tracks:
                self.tracks[track_id] = Track(seen_count=1)
        return removed_ids

    def is_visible(self, track_id) -> bool:
        track = self.tracks.get(track_id)
        if track is None:
            return False
        if track.confirmed and track.missed_count <= self.m_persist:
            return True
        return False

    def remove_track(self, track_id) -> None:
        if track_id in self.tracks:
            del self.tracks[track_id]
