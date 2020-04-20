import pickle
import tempfile


class ListMemMap(object):
    def __init__(self, cache_size=-1, tmp_dir='temp/'):
        """
        Default constructor
        :param cache_size:
        :param tmp_dir:
        """
        self.item_offsets = []
        self.item_sizes = []
        self.cache = []
        self.cache_size = cache_size
        self.cache_index = 0
        self.temp_file = tempfile.TemporaryFile(dir=tmp_dir)

    def __del__(self):
        """
        Destroyer
        :return:
        """
        self.temp_file.close()

    def __add__(self, item):
        """
        Add item to list
        :param item:
        :return:
        """
        self.item_offsets.extend(item.item_offsets)
        self.item_sizes.extend(item.item_sizes)
        self.temp_file.write(item.tempfile.read())
        return self

    def __iter__(self):
        """
        ListMemMap iterator
        :return:
        """
        self.index = 0
        self.cache_index = 0
        self.cache = []
        return self

    def __next__(self):
        """
        Next ListMemMap iterator
        :return:
        """
        if self.index >= len(self.item_sizes):
            raise StopIteration
        if self.index >= self.cache_index - self.cache_size and self.index < self.cache_index:
            self.index += 1
            return self.cache[(self.index - 1) % self.cache_size]
        else:
            if self.cache_size > 0:
                self.__refresh_cache()
            self.index += 1
            return self[self.index - 1]

    def __refresh_cache(self):
        """
        Refresh cache
        :return:
        """
        self.cache_index = self.index + self.cache_size
        self.cache = self[self.index:self.index + self.cache_size]

    def __getitem__(self, index):
        """
        Get item at the given index or slide
        :param index:
        :return:
        """
        if isinstance(index, slice):
            ls = []
            for i in range(index.start if index.start is not None else 0,
                           index.stop if index.stop is not None and not index.stop > len(self.item_sizes)
                           else len(self.item_sizes), index.step if index.step is not None else 1):
                self.temp_file.seek(self.item_offsets[i])
                ls.append(pickle.loads(self.temp_file.read(self.item_sizes[i])))
            return ls
        else:
            self.temp_file.seek(self.item_offsets[index])
            return pickle.loads(self.temp_file.read(self.item_sizes[index]))

    def __setitem__(self, index, item):
        """
        Set item at the given position
        :param index:
        :param item:
        :return:
        """
        self.temp_file.seek(0, 2)
        data = pickle.dumps(item)
        self.item_offsets[index] = self.temp_file.tell()
        self.item_sizes[index] = len(data)
        self.temp_file.write(data)

    def __delitem__(self, index):
        """
        Delete item at given index
        :param index:
        :return:
        """
        del self.item_offsets[index]
        del self.item_sizes[index]

    def __len__(self):
        """
        Return size of ListMemMap object
        :return:
        """
        return len(self.item_offsets)

    def insert(self, index, item):
        """
        Insert item at given index
        :param index:
        :param item:
        :return:
        """
        self.temp_file.seek(0, 2)
        data = pickle.dumps(item)
        self.item_offsets.insert(index, self.temp_file.tell())
        self.item_sizes.insert(index, len(data))
        self.temp_file.write(data)

    def append(self, item):
        """
        Append an item at the end of list
        :param item:
        :return:
        """
        self.temp_file.seek(0, 2)
        data = pickle.dumps(item)
        self.item_offsets.append(self.temp_file.tell())
        self.item_sizes.append(len(data))
        self.temp_file.write(data)

    def extend(self, iterable):
        """
        Extend list
        :param iterable:
        :return:
        """
        for item in iterable:
            self.append(item)

    def at(self, item, start=None, end=None):
        """
        Return index of an item
        :param item:
        :param start:
        :param end:
        :return:
        """
        data = pickle.dumps(item)
        data_len = len(data)
        for index, item_size in enumerate(self.item_sizes[start:end]):
            if item_size == data_len and item == self[index]:
                return index
        # We didn't find the item!
        raise ValueError('{} is not in list'.format(str(item)))

    def remove(self, item):
        """
        Replace first item
        :param item:
        :return:
        """
        del self[self.at(item)]

    def pop(self, index=-1):
        """
        Remove item at the given position and return it
        :param index:
        :return:
        """
        item = self[index]
        del self[index]
        return item

    def count(self, item):
        """
        Return the number of item appear in the list
        :param item:
        :return:
        """
        data = pickle.dumps(item)
        data_len = len(data)
        count = 0
        for index, item_size in enumerate(self.item_sizes):
            if item_size == data_len and item == self[index]:
                count += 1
        return count

    def clear(self):
        """
        Clear the list
        :return:
        """
        self.item_offsets.clear()
        self.item_sizes.clear()
        self.cache.clear()
        self.cache_index = 0
        self.temp_file.close()
        self.temp_file = tempfile.TemporaryFile()
