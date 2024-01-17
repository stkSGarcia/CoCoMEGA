from impl.mr import MR, Perturbation, Relation, RelationType

mr1 = MR(
    [
        Perturbation(0, 0, 0),  # location x
        Perturbation(0, 0, 0),  # location y
        Perturbation(0, 0, 0),  # location z
        Perturbation(0, 0, 0),  # rotation pitch
        Perturbation(0, 0, 0),  # rotation yaw
        Perturbation(0, 0, 0),  # rotation roll
        Perturbation(0, 0, 0),  # velocity
    ],
    [
        Relation(0, RelationType.Decreasing, 0)  # speed down
    ]
)
