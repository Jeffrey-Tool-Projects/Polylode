"""Play complete glTF weight tracks when Qt's importer omits later morph channels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gltf_runtime


def scalar(document: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    if accessor["componentType"] != 5126 or accessor["type"] != "SCALAR" or "sparse" in accessor:
        raise ValueError("unsupported morph animation accessor")
    view = document["bufferViews"][accessor["bufferView"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return np.ndarray((accessor["count"],), dtype="<f4", buffer=binary, offset=offset,
                      strides=(view.get("byteStride", 4),)).copy()


@dataclass
class Track:
    targets: list
    times: np.ndarray
    values: np.ndarray
    interpolation: str

    def sample(self, seconds: float) -> np.ndarray:
        cubic = self.interpolation == "CUBICSPLINE"
        values = self.values[:, 1] if cubic else self.values
        if seconds <= self.times[0]:
            return values[0]
        if seconds >= self.times[-1]:
            return values[-1]
        right = int(np.searchsorted(self.times, seconds, side="right"))
        left = right - 1
        if self.interpolation == "STEP":
            return values[left]
        duration = self.times[right] - self.times[left]
        t = (seconds - self.times[left]) / duration
        if cubic:
            return ((2 * t**3 - 3 * t**2 + 1) * values[left]
                    + (t**3 - 2 * t**2 + t) * duration * self.values[left, 2]
                    + (-2 * t**3 + 3 * t**2) * values[right]
                    + (t**3 - t**2) * duration * self.values[right, 0])
        return values[left] * (1 - t) + values[right] * t


class Playback:
    def __init__(self, root, timelines):
        from PySide6.QtQml import QQmlListReference
        path = root.property("source").toLocalFile()
        document, binary = gltf_runtime.read(path)
        models = gltf_runtime.models(root, document, binary)
        targets = {index: [ref.at(i) for i in range(ref.count())]
                   for index, model in models.items()
                   for ref in [QQmlListReference(model, "morphTargets")]}
        widths = {index: len(document["meshes"][document["nodes"][index]["mesh"]]
                             ["primitives"][0].get("targets", [])) for index in targets}
        self.defaults = []
        for index, objects in targets.items():
            node = document["nodes"][index]
            mesh = document["meshes"][node["mesh"]]
            width = widths[index]
            weights = node.get("weights", mesh.get("weights", [0.0] * width))
            if len(weights) != width:
                raise ValueError("invalid default morph weights")
            if width:
                self.defaults.extend((target, weights[i % width]) for i, target in enumerate(objects))
        self.tracks = []
        animations = document.get("animations", [])
        if len(animations) != len(timelines):
            raise ValueError("the imported animation count differs from glTF")
        for animation, (name, _timeline, _player) in zip(animations, timelines):
            if animation.get("name") != name:
                raise ValueError("the imported animation order differs from glTF")
            tracks = []
            for channel in animation["channels"]:
                if channel["target"]["path"] != "weights":
                    continue
                node_index = channel["target"]["node"]
                objects = targets[node_index]
                width = widths[node_index]
                if not width:
                    raise ValueError("weight animation targets a mesh without morphs")
                sampler = animation["samplers"][channel["sampler"]]
                times = scalar(document, binary, sampler["input"])
                mode = sampler.get("interpolation", "LINEAR")
                if mode not in ("LINEAR", "STEP", "CUBICSPLINE") or not len(times) \
                        or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                    raise ValueError("invalid morph animation timing")
                values = scalar(document, binary, sampler["output"])
                shape = (len(times), 3, width) if mode == "CUBICSPLINE" else (len(times), width)
                values = values.reshape(shape)
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite morph weights")
                for start in range(0, len(objects), width):
                    tracks.append(Track(objects[start:start + width], times, values, mode))
            self.tracks.append(tracks)
        self.timelines = timelines
        self.callbacks = []
        for index, (_name, timeline, _player) in enumerate(timelines):
            callback = lambda i=index: self.apply(i)
            timeline.currentFrameChanged.connect(callback)
            self.callbacks.append((timeline, callback))

    def close(self):
        from shiboken6 import isValid
        for timeline, callback in self.callbacks:
            if isValid(timeline):
                timeline.currentFrameChanged.disconnect(callback)
        self.callbacks.clear()

    def reset(self):
        for target, weight in self.defaults:
            target.setProperty("weight", float(weight))

    def apply(self, index):
        timeline = self.timelines[index][1]
        if timeline.property("enabled"):
            seconds = float(timeline.property("currentFrame")) / 1000.0
            for track in self.tracks[index]:
                for target, value in zip(track.targets, track.sample(seconds)):
                    if target.property("weight") != float(value):
                        target.setProperty("weight", float(value))
