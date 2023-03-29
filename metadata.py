# Subjects and clip numbers. Generally, we split everything into
# the two datasets via a top-level dict selection.
meta_in_crib = { # Clinical in-crib dataset
        'R1': [10, 15, 17, 61, 144],
        'R3': [9, 15, 60, 82, 158],
        'R7': [1, 23, 49, 105, 152],
        'R10': [2, 7, 8, 11, 12],
        'R18': [2, 11, 19, 23, 29],
        'R24': [4, 6, 42, 56, 127]
    }

meta_in_wild = { # YouTube in-the-wild dataset
    '01': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    '03': [0, 6, 9],
    '10-2': [0, 1],
    '10-4': [0, 1, 2, 3, 4, 5],
    '10-5': [0, 1]
}