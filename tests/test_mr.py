from impl.mr import MR, Perturbation, Relation, RelationType

mr1 = MR(
    [
        Perturbation(2, 10.0, 20.0),  # location x
        Perturbation(3, 10.0, 20.0),  # location y
        Perturbation(4, 0.0, 360.0),  # rotation yaw
    ],
    [
        Relation(0, RelationType.Decreasing, 0.3)  # speed
    ]
)

mr2 = MR(
    [
        Perturbation(0, 1, 6),  # weather
    ],
    [
        Relation(1, RelationType.Invariance, 4)  # steering angle
    ]
)

mr3 = MR(
    [
        Perturbation(2, 10.0, 20.0),  # location x
        Perturbation(3, 10.0, 20.0),  # location y
        Perturbation(4, 170.0, 180.0),  # rotation yaw
    ],
    [
        Relation(0, RelationType.Decreasing, 0.5)  # speed
    ]
)

mr = mr1.merge(mr3)
