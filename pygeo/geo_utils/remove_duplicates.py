import numpy as np
from .norm import eDist

# --------------------------------------------------------------
#  Functions that removes duplicate entries from a list
# --------------------------------------------------------------


def unique(s):
    r"""Return a list of the elements in s, but without duplicates.

    For example, ``unique([1,2,3,1,2,3])`` is some permutation of ``[1,2,3]``,
    ``unique("abcabc")`` some permutation of ``["a", "b", "c"]``, and
    ``unique(([1, 2], [2, 3], [1, 2]))`` some permutation of
    ``[[2, 3], [1, 2]]``.

    For best speed, all sequence elements should be hashable.  Then
    ``unique()`` will usually work in linear time.

    If not possible, the sequence elements should enjoy a total
    ordering, and if ``list(s).sort()`` doesn't raise ``TypeError`` it's
    assumed that they do enjoy a total ordering.  Then ``unique()`` will
    usually work in :math:`\mathcal{O}(N\log_2(N))` time.

    If that's not possible either, the sequence elements must support
    equality-testing.  Then ``unique()`` will usually work in quadratic
    time.
    """

    n = len(s)
    if n == 0:
        return []

    # Try using a dict first, as that's the fastest and will usually
    # work.  If it doesn't work, it will usually fail quickly, so it
    # usually doesn't np.cost much to *try* it.  It requires that all the
    # sequence elements be hashable, and support equality comparison.
    u = {}
    try:
        for x in s:
            u[x] = 1
    except TypeError:
        pass
    else:
        return sorted(u.keys())

    # We can't hash all the elements.  Second fastest is to sort,
    # which brings the equal elements together; then duplicates are
    # easy to weed out in a single pass.
    # NOTE:  Python's list.sort() was designed to be efficient in the
    # presence of many duplicate elements.  This isn't true of all
    # sort functions in all languages or libraries, so this approach
    # is more effective in Python than it may be elsewhere.

    try:
        t = list(s)
        t.sort()
    except TypeError:
        pass
    else:
        assert n > 0
        last = t[0]
        lasti = i = 1
        while i < n:
            if t[i] != last:
                t[lasti] = last = t[i]
                lasti += 1
            i += 1
        return t[:lasti]

    # Brute force is all that's left.

    u = []
    for x in s:
        if x not in u:
            u.append(x)
    return u


def uniqueIndex(s, sHash=None):
    """
    This function is based on :meth:`unique`.
    The idea is to take a list s, and reduce it as per unique.

    However, it additionally calculates a linking index array that is
    the same size as the original s, and points to where it ends up in
    the the reduced list

    if sHash is not specified for sorting, s is used

    """
    if sHash is not None:
        ind = np.argsort(np.argsort(sHash))
    else:
        ind = np.argsort(np.argsort(s))

    n = len(s)
    t = list(s)
    t.sort()

    diff = np.zeros(n, "bool")

    last = t[0]
    lasti = i = 1
    while i < n:
        if t[i] != last:
            t[lasti] = last = t[i]
            lasti += 1
        else:
            diff[i] = True
        i += 1

    b = np.where(diff)[0]
    for i in range(n):
        ind[i] -= b.searchsorted(ind[i], side="right")

    return t[:lasti], ind


def pointReduce(points, nodeTol=1e-4):
    """Given a list of N points in ndim space, with possible
    duplicates, return a list of the unique points AND a pointer list
    for the original points to the reduced set"""

    # First
    points = np.array(points)
    N = len(points)
    if N == 0:
        return points, None
    dists = []
    for ipt in range(N):
        dists.append(np.sqrt(np.dot(points[ipt], points[ipt])))

    # we need to round the distances to 8 decimals before sorting
    # because 2 points might have "identical" distances to the origin,
    # but they might differ on the 16 significant figure. As a result
    # the argsort might flip their order even though the elements
    # should not take over each other. By rounding them to 8
    # significant figures, we somewhat guarantee that nodes that
    # have similar distances to the origin dont get shuffled
    # because of floating point errors
    dists_rounded = np.around(dists, decimals=8)

    # the "stable" sorting algorithm guarantees that entries
    # with the same values dont overtake each other.
    # The entries with identical distances are fully checked
    # in the brute force check below.
    ind = np.argsort(dists_rounded, kind="stable")

    i = 0
    cont = True
    newPoints = []
    link = np.zeros(N, "intc")
    linkCounter = 0

    while cont:
        cont2 = True
        tempInd = []
        j = i
        while cont2:
            if abs(dists[ind[i]] - dists[ind[j]]) < nodeTol:
                tempInd.append(ind[j])
                j = j + 1
                if j == N:  # Overrun check
                    cont2 = False
            else:
                cont2 = False

        subPoints = []  # Copy of the list of sub points with the dists
        for ii in range(len(tempInd)):
            subPoints.append(points[tempInd[ii]])

        # Brute Force Search them
        subUniquePts, subLink = pointReduceBruteForce(subPoints, nodeTol)
        newPoints.extend(subUniquePts)

        for ii in range(len(tempInd)):
            link[tempInd[ii]] = subLink[ii] + linkCounter

        linkCounter += max(subLink) + 1

        i = j - 1 + 1
        if i == N:
            cont = False

    return np.array(newPoints), np.array(link)


def pointReduceBruteForce(points, nodeTol=1e-4):
    """Given a list of N points in ndim space, with possible
    duplicates, return a list of the unique points AND a pointer list
    for the original points to the reduced set

    Warnings
    --------
    This is the brute force version of :func:`pointReduce`.
    """
    N = len(points)
    if N == 0:
        return points, None
    uniquePoints = [points[0]]
    link = [0]
    for i in range(1, N):
        foundIt = False
        for j in range(len(uniquePoints)):
            if eDist(points[i], uniquePoints[j]) < nodeTol:
                link.append(j)
                foundIt = True
                break

        if not foundIt:
            uniquePoints.append(points[i])
            link.append(j + 1)

    return np.array(uniquePoints), np.array(link)
