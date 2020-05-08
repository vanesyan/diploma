import cv2
import numpy as np

def diff_images(p1, p2, debug=False):
    first = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

    if debug:
        cv2.imshow("Original", first)
        cv2.imshow("Recognized", second)

        first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        mx = min(first.shape[0], second.shape[0])
        my = min(first.shape[1], second.shape[1])

        r = np.zeros((mx, my), dtype=np.uint8)
        g = np.zeros((mx, my), dtype=np.uint8)
        b = np.zeros((mx, my), dtype=np.uint8)
        for x in range(0, mx):
            for y in range(0, my):
                p1 = first[x, y]
                p2 = second[x, y]
                if p1 > p2:
                    r[x, y] = 1
                elif p1 < p2:
                    b[x, y] = 1
    
        bgr = np.dstack((b * 255, g * 255, r * 255)).astype(np.uint8)            

        cv2.imshow("Diff", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return np.sum(first - second)
