import copy
import logging as lg
import random

from mido import MidiFile, MidiTrack

from src.dataset.music_augmenter.base_music_augmenter import BaseMusicAugmenter

logger = lg.getLogger(__name__)


class MidiAugmentor(BaseMusicAugmenter):
    def __init__(
        self,
        allow_track_drop=True,
        allow_humanize=True,
        allow_voicing_change=True,
        humanize_sigma=10,  # Sigma w tickach (dla 480 tpb, 10 to mało, 30 to słyszalne)
        track_drop_probability=0.4,
        octave_shift_prob=0.3,
    ):
        self._allow_track_drop = allow_track_drop
        self._allow_humanize = allow_humanize
        self._allow_voicing_change = allow_voicing_change
        self._humanize_sigma = humanize_sigma
        self._track_drop_probability = track_drop_probability
        self._octave_shift_prob = octave_shift_prob
        logger.info(
            f"MIDI Augmentor initialized with params: "
            f"allow_track_drop={allow_track_drop}, allow_humanize={allow_humanize}, "
            f"allow_voicing_change={allow_voicing_change}, humanize_sigma={humanize_sigma}, "
            f"track_drop_probability={track_drop_probability}, octave_shift_prob={octave_shift_prob}"
        )

    def __call__(self, music_data: MidiFile) -> MidiFile:
        aug_mid = copy.deepcopy(music_data)

        if self._allow_track_drop and len(aug_mid.tracks) > 1:
            aug_mid = self._drop_tracks(aug_mid)

        if self._allow_voicing_change:
            aug_mid = self._change_voicing(aug_mid)

        if self._allow_humanize:
            aug_mid = self._humanize(aug_mid)

        return aug_mid

    def _drop_tracks(self, music_data: MidiFile) -> MidiFile:
        """Usuwa losowe ścieżki, ale zachowuje pierwszą (często tempo map)"""
        if len(music_data.tracks) <= 2:
            return music_data

        new_tracks = [music_data.tracks[0]]
        remaining_tracks = music_data.tracks[1:]
        if not remaining_tracks:
            return music_data

        keep_idx = random.randint(0, len(remaining_tracks) - 1)

        for i, track in enumerate(remaining_tracks):
            if i == keep_idx or random.random() > self._track_drop_probability:
                new_tracks.append(track)

        music_data.tracks = new_tracks
        return music_data

    def _change_voicing(self, music_data):
        """Przesuwa nuty o oktawę w górę/dół"""
        for track in music_data.tracks:
            for msg in track:
                if msg.type in ["note_on", "note_off"]:
                    if random.random() < self._track_drop_probability:
                        shift = random.choice([-12, 12])
                        new_pitch = msg.note + shift
                        if 0 <= new_pitch <= 127:
                            msg.note = new_pitch
        return music_data

    def _humanize(self, music_data):
        """
        Zmienia velocity i timing.
        Dla timingu: Konwertuje na czas absolutny -> dodaje szum -> sortuje -> konwertuje na delta.
        """
        for track in music_data.tracks:
            abs_events = []
            current_time = 0

            for msg in track:
                current_time += msg.time

                if msg.type == "note_on" and msg.velocity > 0:
                    vel_noise = int(random.gauss(0, 5))  # +/- 5 velocity
                    msg.velocity = max(1, min(127, msg.velocity + vel_noise))

                # Humanize Time (dodajemy szum do abs_time)
                # Pomijamy MetaMessages (np. tempo, time signature), żeby nie psuć struktury pliku
                if not msg.is_meta:
                    time_noise = int(random.gauss(0, self._humanize_sigma))
                    # Nie cofamy się w czasie poniżej 0
                    event_time = max(0, current_time + time_noise)
                else:
                    event_time = current_time

                abs_events.append((event_time, msg))

            # --- 2. Sortowanie zdarzeń ---
            # Po dodaniu szumu kolejność nut mogła się zmienić!
            # Sortujemy stabilnie po czasie
            abs_events.sort(key=lambda x: x[0])

            # --- 3. Rekonstrukcja Delta Time ---
            last_time = 0
            new_track = MidiTrack()

            for abs_time, msg in abs_events:
                delta = abs_time - last_time
                # Zabezpieczenie (teoretycznie sortowanie to załatwia, ale dla pewności)
                delta = max(delta, 0)

                msg.time = delta
                new_track.append(msg)
                last_time = abs_time

            # Podmieniamy track w obiekcie (modyfikacja w miejscu dla obiektu mid,
            # ale my pracujemy na kopii w __call__)
            track[:] = new_track

        return music_data
