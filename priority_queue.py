class MaxPriorityQueue:
    def __init__(self):
        self.heap = []

    def insert(self, item):
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def maximum(self):
        return self.heap[0] if self.heap else None

    def extract_max(self):
        if not self.heap:
            return None

        lastelt = self.heap.pop()
        if self.heap:
            maxitem = self.heap[0]
            self.heap[0] = lastelt
            self._sift_down(0)
            return maxitem
        return lastelt

    def _sift_up(self, pos):
        startpos = pos
        newitem = self.heap[pos]
        while pos > 0:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if newitem <= parent:
                break
            self.heap[pos] = parent
            pos = parentpos
        self.heap[pos] = newitem

    def _sift_down(self, pos):
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]
        childpos = 2 * pos + 1
        while childpos < endpos:
            rightpos = childpos + 1
            if rightpos < endpos and self.heap[childpos] <= self.heap[rightpos]:
                childpos = rightpos
            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        self.heap[pos] = newitem

    def is_empty(self):
        return len(self.heap) == 0