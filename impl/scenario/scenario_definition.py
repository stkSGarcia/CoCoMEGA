import random
import uuid
from abc import ABC
from typing import List, Dict

from impl.config import CONFIG


class Boundary:
    def __init__(self, boundary: Dict[str, List]):
        self.boundary = boundary
        # Check type consistency.
        for lower, upper in self.boundary.values():
            if type(lower) is not type(upper):
                raise ValueError(f"Unmatched boundary types: [{type(lower)}, {type(upper)}].")
            if not isinstance(lower, float) and not isinstance(lower, int):
                raise ValueError(f"Unsupported boundary type: {type(lower)}.")

    def get(self, field: str):
        return self.boundary[field][0], self.boundary[field][1]

    def random(self, field: str, none_pb=None):
        lower, upper = self.boundary[field]
        if isinstance(lower, float):
            return None if none_pb and random.random() < none_pb else random.uniform(lower, upper)
        elif isinstance(lower, int):
            return None if none_pb and random.random() < none_pb else random.randint(lower, upper)


def _dist_attrs(this, that, attrs, boundary: Boundary):
    dist = 0
    for attr in attrs:
        lower, upper = boundary.get(attr)
        if isinstance(lower, float):
            dist += abs(getattr(this, attr) - getattr(that, attr)) / (upper - lower)
        elif isinstance(lower, int):
            # TODO: within the same category
            dist += CONFIG["dist_scaling"] * (0.0 if getattr(this, attr) == getattr(that, attr) else 1.0)
    return dist


def _mate_attrs(this, that, attrs):
    for attr in attrs:
        if random.random() < CONFIG["scenario"]["cxpb"]:
            value = getattr(this, attr)
            setattr(this, attr, getattr(that, attr))
            setattr(that, attr, value)


def _mate_actors(this, that):
    size = min(len(this), len(that))
    for i in range(size):
        if random.random() < CONFIG["scenario"]["cxpb"]:
            this[i].id_, that[i].id_ = that[i].id_, this[i].id_
            this[i], that[i] = that[i], this[i]
    return this, that


def _mutate_attrs(this, attrs, boundary: Boundary):
    for attr in attrs:
        if random.random() >= CONFIG["scenario"]["mutpb"]: continue
        lower, upper = boundary.get(attr)
        if isinstance(lower, float):  # Polynomial mutation
            x = getattr(this, attr)
            delta_1 = (x - lower) / (upper - lower)
            delta_2 = (upper - x) / (upper - lower)
            rand = random.random()
            mut_pow = 1.0 / (CONFIG["scenario"]["eta"] + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (CONFIG["scenario"]["eta"] + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (CONFIG["scenario"]["eta"] + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (upper - lower)
            x = min(max(x, lower), upper)
            setattr(this, attr, x)
        elif isinstance(lower, int):
            setattr(this, attr, random.randint(lower, upper))


class ScenarioDefinition:
    _ATTRIBUTES = ["weather", "darkness"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["env"])

    def __init__(self):
        self.id_ = uuid.uuid4().hex
        self.ego_vehicle: Vehicle | None = None
        self.trajectory: List = []
        self.vehicles: List[Vehicle] = []
        self.walkers: List[Walker] = []
        self.statics: List[Static] = []
        self.weather = None
        self.darkness = None
        self._other_actors: List | None = None

    @staticmethod
    def generate_random():
        scenario = ScenarioDefinition()
        scenario.ego_vehicle = Vehicle.generate_random()
        scenario.vehicles = [Vehicle.generate_random()]
        scenario.walkers = [Walker.generate_random()]
        scenario.statics = [Static.generate_random()]
        for attr in ScenarioDefinition._ATTRIBUTES:
            setattr(scenario, attr, ScenarioDefinition._BOUNDARY.random(attr))
        return scenario

    @staticmethod
    def mock():
        scenario = ScenarioDefinition()
        scenario.ego_vehicle = Vehicle(
            id_="ego_vehicle",
            transform=Transform(x=200.0, y=-2.0, z=0.5, pitch=0.0, yaw=0.0, roll=0.0),
            speed=0.0,
            model=0,
            color=0,
            autopilot=0,
        )
        scenario.trajectory = [{
            "x": -188.04,
            "y": 111.89,
            "z": 0.0,
            "yaw": -90.0,
        }, {
            "x": -208.309,
            "y": 87.82,
            "z": 0.0,
            "yaw": 180.0,
        }, {
            "x": -241.01,
            "y": 87.77,
            "z": 0.0,
            "yaw": 180.0,
        }, ]
        scenario.vehicles = [
            Vehicle(
                id_="vehicle_nearby1",
                transform=Transform(x=-195.0, y=120.0, z=0.0, pitch=0.0, yaw=-90.0, roll=0.0),
                speed=2.0,
                model=23,
                color=1,
                autopilot=0,
            ),
            Vehicle(
                id_="vehicle_nearby2",
                transform=Transform(x=-188.0, y=100.0, z=0.0, pitch=0.0, yaw=-90.0, roll=0.0),
                speed=20.0,
                model=23,
                color=2,
                autopilot=1,
            ),
        ]
        scenario.walkers = [
            Walker(
                id_="pedestrian1",
                transform=Transform(x=-182.0, y=100.0, z=0.0, pitch=0.0, yaw=180.0, roll=0.0),
                speed=1,
                model=0,
            ),
            Walker(
                id_="pedestrian2",
                transform=Transform(x=-182.0, y=102.0, z=0.0, pitch=0.0, yaw=180.0, roll=0.0),
                speed=1.5,
                model=10,
            ),
        ]
        return scenario

    def get_trigger_position(self):
        return self.trajectory[0]

    def get_other_actors(self):  # TODO
        if self._other_actors is None:
            self._other_actors = [actor.get_config() for actor in self.vehicles + self.walkers + self.statics]
        return self._other_actors

    def update(self, category: str, value, replace_pb=None):  # TODO: optimize
        if category in ["vehicle", "walker", "static"]:
            actors = getattr(self, f"{category}s")
            exist = False
            for actor in actors:
                if actor.id_ == value.id_:
                    exist = True
                    actor.update(value)
                    break
            if not exist:
                if replace_pb and len(actors) > 0 and random.random() < replace_pb:
                    actor = random.choice(actors)
                    actor.update(value)
                else:
                    actors.append(value)
        elif category in ScenarioDefinition._ATTRIBUTES:
            setattr(self, category, value)
        else:
            raise ValueError(f"Unsupported category: {category}.")

    def dist(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        dist = _dist_attrs(self, other, ScenarioDefinition._ATTRIBUTES, ScenarioDefinition._BOUNDARY)
        length = len(ScenarioDefinition._ATTRIBUTES)
        for actors, other_actors in zip([self.vehicles, self.walkers, self.statics],
                                        [other.vehicles, other.walkers, other.statics]):
            actor_dict = {actor.id_: actor for actor in actors}
            other_actor_dict = {actor.id_: actor for actor in other_actors}
            common_ids = set(actor_dict.keys()).intersection(set(other_actor_dict.keys()))
            size = max(len(actors), len(other_actors))
            length += size
            dist += (size - len(common_ids)) * CONFIG["dist_scaling"]
            for actor_id in common_ids:
                dist += actor_dict[actor_id].dist(other_actor_dict[actor_id])
        return dist / length

    def mate(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        _mate_attrs(self, other, ScenarioDefinition._ATTRIBUTES)
        _mate_actors(self.vehicles, other.vehicles)
        _mate_actors(self.walkers, other.walkers)
        _mate_actors(self.statics, other.statics)

    def mutate(self):
        _mutate_attrs(self, ScenarioDefinition._ATTRIBUTES, ScenarioDefinition._BOUNDARY)
        for actor in self.vehicles + self.walkers + self.statics:
            actor.mutate()

    def __repr__(self):
        return (f"Scenario(id={self.id_}, "
                f"ego_vehicle={self.ego_vehicle}, "
                f"trajectory={self.trajectory}, "
                f"vehicles={self.vehicles}, "
                f"walkers={self.walkers}, "
                f"statics={self.statics}, "
                f"weather={self.weather}, "
                f"darkness={self.darkness})")


class Transform:
    def __init__(self, x, y, z, pitch, yaw, roll):
        self.x = x
        self.y = y
        self.z = z
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    def get_config(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
        }

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.x == other.x and
                self.y == other.y and
                self.z == other.z and
                self.pitch == other.pitch and
                self.yaw == other.yaw and
                self.roll == other.roll)

    def __repr__(self):
        return f"Transform(x={self.x}, y={self.y}, z={self.z}, pitch={self.pitch}, yaw={self.yaw}, roll={self.roll})"


class Actor(ABC):
    _TRANSFORM = ["x", "y", "z", "pitch", "yaw", "roll"]
    _ATTRIBUTES = []
    _BOUNDARY = None
    _BASE_ID = "actor"
    _last_id = 0

    def __init__(self, id_, transform: Transform, *args, **kwargs):
        self.id_ = id_
        self.transform = transform

    @classmethod
    def generate_random(cls, boundary: Boundary = None, none_pb=None, id_=None):
        if id_ is None:
            cls._last_id += 1
            id_ = f"{cls._BASE_ID}{cls._last_id}"
        if boundary is None:
            boundary = cls._BOUNDARY
        return cls(id_=id_, transform=Transform(**{attr: boundary.random(attr, none_pb) for attr in Actor._TRANSFORM}),
                   **{attr: boundary.random(attr, none_pb) for attr in cls._ATTRIBUTES})

    def update(self, other):
        for attr in Actor._TRANSFORM:
            setattr(self.transform, attr, getattr(other.transform, attr))
        for attr in self._ATTRIBUTES:
            setattr(self, attr, getattr(other, attr))

    def dist(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        return ((_dist_attrs(self.transform, other.transform, Actor._TRANSFORM, self._BOUNDARY) +
                 _dist_attrs(self, other, self._ATTRIBUTES, self._BOUNDARY)) /
                (len(Actor._TRANSFORM) + len(self._ATTRIBUTES)))

    def mate(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        _mate_attrs(self.transform, other.transform, Actor._TRANSFORM)
        _mate_attrs(self, other, self._ATTRIBUTES)

    def mutate(self):
        _mutate_attrs(self.transform, Actor._TRANSFORM, self._BOUNDARY)
        _mutate_attrs(self, self._ATTRIBUTES, self._BOUNDARY)

    def get_config(self):
        return {
            "id": self.id_,
            "spawn_point": self.transform.get_config()
        }

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.id_ == other.id_ and
                self.transform == other.transform)

    def __repr__(self):
        return (f"{self.__class__.__name__}(id={self.id_}, transform={self.transform}, " +
                ", ".join(f"{attr}={str(getattr(self, attr))}" for attr in self._ATTRIBUTES) + ")")


class Vehicle(Actor):
    _ATTRIBUTES = ["speed", "model", "color", "autopilot"]
    _BLUEPRINTS = CONFIG["blueprint"]["vehicle"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["vehicle"])
    _BASE_ID = "vehicle"

    def __init__(self, id_, transform, speed, model, color, autopilot):
        super().__init__(id_, transform)
        self.speed = speed
        self.model = model
        self.color = color
        self.autopilot = autopilot

    def get_config(self):
        return {
            **super().get_config(),
            "speed": self.speed,
            "model": Vehicle._BLUEPRINTS["model"][self.model],
            "color": Vehicle._BLUEPRINTS["color"][self.color],
            "autopilot": bool(self.autopilot),
        }

    def __eq__(self, other):
        return (super().__eq__(other) and
                self.speed == other.speed and
                self.model == other.model and
                self.color == other.color and
                self.autopilot == other.autopilot)


class Walker(Actor):
    _ATTRIBUTES = ["speed", "model"]
    _BLUEPRINTS = CONFIG["blueprint"]["walker"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["walker"])
    _BASE_ID = "walker"

    def __init__(self, id_, transform, speed, model):
        super().__init__(id_, transform)
        self.speed = speed
        self.model = model

    def get_config(self):
        return {
            **super().get_config(),
            "speed": self.speed,
            "model": Walker._BLUEPRINTS["model"][self.model],
        }

    def __eq__(self, other):
        return (super().__eq__(other) and
                self.speed == other.speed and
                self.model == other.model)


class Static(Actor):
    _ATTRIBUTES = ["model", "size"]
    _BLUEPRINTS = CONFIG["blueprint"]["static"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["static"])
    _BASE_ID = "static"

    def __init__(self, id_, transform, model, size):
        super().__init__(id_, transform)
        self.model = model
        self.size = size

    def get_config(self):
        return {
            **super().get_config(),
            "model": Static._BLUEPRINTS["model"][self.model],
            "size": self.size,
        }

    def __eq__(self, other):
        return (super().__eq__(other) and
                self.model == other.model and
                self.size == other.size)
