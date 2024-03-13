import random
import uuid
from abc import ABC, abstractmethod
from typing import List

from impl.config import CONFIG


class Boundary:
    def __init__(self, boundary):
        self.boundary = boundary
        # Check type consistency.
        for lower, upper in self.boundary.values():
            if type(lower) is not type(upper):
                raise ValueError(f"Unmatched boundary types: [{type(lower)}, {type(upper)}].")
            if not isinstance(lower, float) and not isinstance(lower, int):
                raise ValueError(f"Unsupported boundary type: {type(lower)}.")

    def get(self, field: str):
        return self.boundary[field][0], self.boundary[field][1]

    def random(self, field: str):
        lower, upper = self.boundary[field]
        if isinstance(lower, float):
            return random.uniform(lower, upper)
        elif isinstance(lower, int):
            return random.randint(lower, upper)


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
    ATTRIBUTES = ["weather", "darkness"]
    BOUNDARY = Boundary(CONFIG["boundary"]["env"])

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
        for attr in ScenarioDefinition.ATTRIBUTES:
            setattr(scenario, attr, ScenarioDefinition.BOUNDARY.random(attr))
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

    def dist(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        dist = _dist_attrs(self, other, ScenarioDefinition.ATTRIBUTES, ScenarioDefinition.BOUNDARY)
        length = len(ScenarioDefinition.ATTRIBUTES)
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
        _mate_attrs(self, other, ScenarioDefinition.ATTRIBUTES)
        _mate_actors(self.vehicles, other.vehicles)
        _mate_actors(self.walkers, other.walkers)
        _mate_actors(self.statics, other.statics)

    def mutate(self):
        _mutate_attrs(self, ScenarioDefinition.ATTRIBUTES, ScenarioDefinition.BOUNDARY)
        for actor in self.vehicles + self.walkers + self.statics:
            actor.mutate()

    def __str__(self):
        return (f"Scenario(id={self.id_}, "
                f"ego_vehicle={self.ego_vehicle}, "
                f"trajectory={self.trajectory}, "
                f"vehicles={self.vehicles}, "
                f"walkers={self.walkers}, "
                f"statics={self.statics}, "
                f"weather={self.weather}, "
                f"darkness={self.darkness})")

    def __repr__(self):
        return self.__str__()


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

    def __str__(self):
        return f"Transform(x={self.x}, y={self.y}, z={self.z}, pitch={self.pitch}, yaw={self.yaw}, roll={self.roll})"

    def __repr__(self):
        return self.__str__()


class Actor(ABC):
    ATTRIBUTES = ["x", "y", "z", "pitch", "yaw", "roll"]
    last_id = 0

    def __init__(self, id_, transform: Transform):
        self.id_ = id_
        self.transform = transform

    @staticmethod
    @abstractmethod
    def generate_random():
        raise NotImplementedError

    @abstractmethod
    def dist(self, other):
        raise NotImplementedError

    def mate(self, other):
        _mate_attrs(self.transform, other.transform, Actor.ATTRIBUTES)

    @abstractmethod
    def mutate(self):
        raise NotImplementedError

    def get_config(self):
        return {
            "id": self.id_,
            "spawn_point": self.transform.get_config()
        }

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.id_ == other.id_ and
                self.transform == other.transform)

    def __str__(self):
        return f"Actor(id={self.id_}, transform={self.transform})"

    def __repr__(self):
        return self.__str__()


class Vehicle(Actor):
    ATTRIBUTES = ["speed", "model", "color", "autopilot"]
    BLUEPRINTS = CONFIG["blueprint"]["vehicle"]
    BOUNDARY = Boundary(CONFIG["boundary"]["vehicle"])

    def __init__(self, id_, transform, speed, model, color, autopilot):
        super().__init__(id_, transform)
        self.speed = speed
        self.model = model
        self.color = color
        self.autopilot = autopilot

    @staticmethod
    def generate_random():
        Vehicle.last_id += 1
        return Vehicle(
            id_=f"vehicle{Vehicle.last_id}",
            transform=Transform(**{attr: Vehicle.BOUNDARY.random(attr) for attr in Actor.ATTRIBUTES}),
            **{attr: Vehicle.BOUNDARY.random(attr) for attr in Vehicle.ATTRIBUTES},
        )

    def dist(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        return ((_dist_attrs(self.transform, other.transform, Actor.ATTRIBUTES, Vehicle.BOUNDARY) +
                 _dist_attrs(self, other, Vehicle.ATTRIBUTES, Vehicle.BOUNDARY)) /
                (len(Actor.ATTRIBUTES) + len(Vehicle.ATTRIBUTES)))

    def mate(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        super().mate(other)
        _mate_attrs(self, other, Vehicle.ATTRIBUTES)

    def mutate(self):
        _mutate_attrs(self.transform, Actor.ATTRIBUTES, Vehicle.BOUNDARY)
        _mutate_attrs(self, Vehicle.ATTRIBUTES, Vehicle.BOUNDARY)

    def get_config(self):
        return {
            **super().get_config(),
            "speed": self.speed,
            "model": Vehicle.BLUEPRINTS["model"][self.model],
            "color": Vehicle.BLUEPRINTS["color"][self.color],
            "autopilot": bool(self.autopilot),
        }

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                super().__eq__(other) and
                self.speed == other.speed and
                self.model == other.model and
                self.color == other.color and
                self.autopilot == other.autopilot)

    def __str__(self):
        return (f"Vehicle(id={self.id_}, "
                f"transform={self.transform}, "
                f"speed={self.speed}, "
                f"model={self.model}, "
                f"color={self.color}, "
                f"autopilot={self.autopilot})")

    def __repr__(self):
        return self.__str__()


class Walker(Actor):
    ATTRIBUTES = ["speed", "model"]
    BLUEPRINTS = CONFIG["blueprint"]["walker"]
    BOUNDARY = Boundary(CONFIG["boundary"]["walker"])

    def __init__(self, id_, transform, speed, model):
        super().__init__(id_, transform)
        self.speed = speed
        self.model = model

    @staticmethod
    def generate_random():
        Walker.last_id += 1
        return Walker(
            id_=f"walker{Walker.last_id}",
            transform=Transform(**{attr: Walker.BOUNDARY.random(attr) for attr in Actor.ATTRIBUTES}),
            **{attr: Walker.BOUNDARY.random(attr) for attr in Walker.ATTRIBUTES},
        )

    def dist(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        return ((_dist_attrs(self.transform, other.transform, Actor.ATTRIBUTES, Walker.BOUNDARY) +
                 _dist_attrs(self, other, Walker.ATTRIBUTES, Walker.BOUNDARY)) /
                (len(Actor.ATTRIBUTES) + len(Walker.ATTRIBUTES)))

    def mate(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        super().mate(other)
        _mate_attrs(self, other, Walker.ATTRIBUTES)

    def mutate(self):
        _mutate_attrs(self.transform, Actor.ATTRIBUTES, Walker.BOUNDARY)
        _mutate_attrs(self, Walker.ATTRIBUTES, Walker.BOUNDARY)

    def get_config(self):
        return {
            **super().get_config(),
            "speed": self.speed,
            "model": Walker.BLUEPRINTS["model"][self.model],
        }

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                super().__eq__(other) and
                self.speed == other.speed and
                self.model == other.model)

    def __str__(self):
        return (f"Walker(id={self.id_}, "
                f"transform={self.transform}, "
                f"speed={self.speed}, "
                f"model={self.model})")

    def __repr__(self):
        return self.__str__()


class Static(Actor):
    ATTRIBUTES = ["model", "size"]
    BLUEPRINTS = CONFIG["blueprint"]["static"]
    BOUNDARY = Boundary(CONFIG["boundary"]["static"])

    def __init__(self, id_, transform, model, size):
        super().__init__(id_, transform)
        self.model = model
        self.size = size

    @staticmethod
    def generate_random():
        Static.last_id += 1
        return Static(
            id_=f"static{Static.last_id}",
            transform=Transform(**{attr: Static.BOUNDARY.random(attr) for attr in Actor.ATTRIBUTES}),
            **{attr: Static.BOUNDARY.random(attr) for attr in Static.ATTRIBUTES},
        )

    def dist(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        return ((_dist_attrs(self.transform, other.transform, Actor.ATTRIBUTES, Static.BOUNDARY) +
                 _dist_attrs(self, other, Static.ATTRIBUTES, Static.BOUNDARY)) /
                (len(Actor.ATTRIBUTES) + len(Static.ATTRIBUTES)))

    def mate(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        super().mate(other)
        _mate_attrs(self, other, Static.ATTRIBUTES)

    def mutate(self):
        _mutate_attrs(self.transform, Actor.ATTRIBUTES, Static.BOUNDARY)
        _mutate_attrs(self, Static.ATTRIBUTES, Static.BOUNDARY)

    def get_config(self):
        return {
            **super().get_config(),
            "model": Static.BLUEPRINTS["model"][self.model],
            "size": self.size,
        }

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                super().__eq__(other) and
                self.model == other.model and
                self.size == other.size)

    def __str__(self):
        return (f"Static("
                f"id={self.id_}, "
                f"transform={self.transform}, "
                f"size={self.size})")

    def __repr__(self):
        return self.__str__()
