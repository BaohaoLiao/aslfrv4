import numpy as np

PAD=-100.
LIP = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

NOSE = [1,2,98,327]
LNOSE = [98]
RNOSE = [327]

REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,]
EYE = LEYE + REYE

HAND = np.arange(21)

def get_names(idxs, keyword):
    names = []
    for idx in idxs:
        names.append(keyword + "_" + str(idx))
    return names

def get_xy_names(names, axis):
    output = []
    for n in names:
        output.append(axis + "_" + n)
    return output

LIP_NAMES = get_names(LIP, "face")
LLIP_NAMES = get_names(LLIP, "face")
RLIP_NAMES = get_names(RLIP, "face")

NOSE_NAMES = get_names(NOSE, "face")
LNOSE_NAMES = get_names(LNOSE, "face")
RNOSE_NAMES = get_names(RNOSE, "face")

EYE_NAMES = get_names(EYE, "face")
LEYE_NAMES = get_names(LEYE, "face")
REYE_NAMES = get_names(REYE, "face")

LHAND_NAMES = get_names(HAND, "left_hand")
RHAND_NAMES = get_names(HAND, "right_hand")
HAND_NAMES = LHAND_NAMES + RHAND_NAMES

POINT_LANDMARKS = NOSE_NAMES + LIP_NAMES + EYE_NAMES + LHAND_NAMES + RHAND_NAMES
X_POINT_LANDMARKS = get_xy_names(POINT_LANDMARKS, axis="x")
Y_POINT_LANDMARKS = get_xy_names(POINT_LANDMARKS, axis="y")
XY_POINT_LANDMARKS = X_POINT_LANDMARKS + Y_POINT_LANDMARKS

LLIP_IDXS = np.argwhere(np.isin(POINT_LANDMARKS, LLIP_NAMES)).squeeze()
LNOSE_IDXS = np.argwhere(np.isin(POINT_LANDMARKS, LNOSE_NAMES)).squeeze(axis=0)
LEYE_IDXS = np.argwhere(np.isin(POINT_LANDMARKS, LEYE_NAMES)).squeeze()
LHAND_IDXS = np.argwhere(np.isin(POINT_LANDMARKS, LHAND_NAMES)).squeeze()

RLIP_IDXS = np.argwhere(np.isin(POINT_LANDMARKS, RLIP_NAMES)).squeeze()
RNOSE_IDXS = np.argwhere(np.isin(POINT_LANDMARKS, RNOSE_NAMES)).squeeze(axis=0)
REYE_IDXS = np.argwhere(np.isin(POINT_LANDMARKS, REYE_NAMES)).squeeze()
RHAND_IDXS = np.argwhere(np.isin(POINT_LANDMARKS, RHAND_NAMES)).squeeze()