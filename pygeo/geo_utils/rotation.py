# External modules
import numpy as np

# --------------------------------------------------------------
#                Rotation Functions
# --------------------------------------------------------------


def rotxM(theta):
    """Return x rotation matrix"""
    theta = theta * np.pi / 180
    M = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    return M


def rotyM(theta):
    """Return y rotation matrix"""
    theta = theta * np.pi / 180
    M = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    return M


def rotzM(theta):
    """Return z rotation matrix"""
    theta = theta * np.pi / 180
    M = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    return M


def rotxV(x, theta):
    """Rotate a coordinate in the local x frame"""
    M = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    return np.dot(M, x)


def rotyV(x, theta):
    """Rotate a coordinate in the local y frame"""
    M = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    return np.dot(M, x)


def rotzV(x, theta):
    """Rotate a coordinate in the local z frame"""
    M = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    return np.dot(M, x)


def rotVbyW(V, W, theta):
    """Rotate a vector V, about an axis W by angle theta"""
    ux = W[0]
    uy = W[1]
    uz = W[2]

    c = np.cos(theta)
    s = np.sin(theta)
    if (
        np.array(theta).dtype == np.dtype("D")
        or np.array(W).dtype == np.dtype("D")
        or np.array(V).dtype == np.dtype("D")
    ):
        dtype = "D"
    else:
        dtype = "d"

    R = np.zeros((3, 3), dtype)

    R[0, 0] = ux**2 + (1 - ux**2) * c
    R[0, 1] = ux * uy * (1 - c) - uz * s
    R[0, 2] = ux * uz * (1 - c) + uy * s

    R[1, 0] = ux * uy * (1 - c) + uz * s
    R[1, 1] = uy**2 + (1 - uy**2) * c
    R[1, 2] = uy * uz * (1 - c) - ux * s

    R[2, 0] = ux * uz * (1 - c) - uy * s
    R[2, 1] = uy * uz * (1 - c) + ux * s
    R[2, 2] = uz**2 + (1 - uz**2) * c

    return np.dot(R, V)


# --------------------------------------------------------------
#                Array Rotation and Flipping Functions
# --------------------------------------------------------------


def rotateCCW(inArray):
    """Rotate the inArray array 90 degrees CCW"""
    rows = inArray.shape[0]
    cols = inArray.shape[1]
    output = np.empty([cols, rows], inArray.dtype)

    for row in range(rows):
        for col in range(cols):
            output[cols - col - 1][row] = inArray[row][col]

    return output


def rotateCW(inArray):
    """Rotate the inArray array 90 degrees CW"""
    rows = inArray.shape[0]
    cols = inArray.shape[1]
    output = np.empty([cols, rows], inArray.dtype)

    for row in range(rows):
        for col in range(cols):
            output[col][rows - row - 1] = inArray[row][col]

    return output


def reverseRows(inArray):
    """Flip Rows (horizontally)"""
    rows = inArray.shape[0]
    cols = inArray.shape[1]
    output = np.empty([rows, cols], inArray.dtype)
    for row in range(rows):
        output[row] = inArray[row][::-1].copy()

    return output


def reverseCols(inArray):
    """Flip Cols (vertically)"""
    rows = inArray.shape[0]
    cols = inArray.shape[1]
    output = np.empty([rows, cols], inArray.dtype)
    for col in range(cols):
        output[:, col] = inArray[:, col][::-1].copy()

    return output


def orientArray(index, inArray):
    """Take an input array inArray, and rotate/flip according to the index
    output from quadOrientation (in orientation.py)
    """

    if index == 0:
        outArray = inArray.copy()
    elif index == 1:
        outArray = rotateCCW(inArray)
        outArray = rotateCCW(outArray)
        outArray = reverseRows(outArray)
    elif index == 2:
        outArray = reverseRows(inArray)
    elif index == 3:
        outArray = rotateCCW(inArray)  # Verified working
        outArray = rotateCCW(outArray)
    elif index == 4:
        outArray = rotateCW(inArray)
        outArray = reverseRows(outArray)
    elif index == 5:
        outArray = rotateCCW(inArray)
    elif index == 6:
        outArray = rotateCW(inArray)
    elif index == 7:
        outArray = rotateCCW(inArray)
        outArray = reverseRows(outArray)

    return outArray
