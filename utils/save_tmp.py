import numpy as np
import sys

if __name__ == '__main__':
  arg = sys.argv[1]

  retained_matches = []
  for line in sys.stdin:
    line = line.split()
    if not line or len(line) != 6 or not line[0][0].isdigit():  continue
    x0, y0, x1, y1, score, index = line
    retained_matches.append(
      (float(x0), float(y0), float(x1), float(y1), float(score), float(index)))

  retained_matches = np.asarray(retained_matches)
  np.save(arg, retained_matches)