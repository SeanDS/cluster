"""Implements a simple observer pattern"""

import abc
import weakref
import logging

LOGGER = logging.getLogger(__name__)

class Observable(metaclass=abc.ABCMeta):
    """Keeps a set of observers that are to be informed of events"""

    def __init__(self):
        super().__init__()

        # set of observers, using weak references to prevent references
        # to deleted objects being kept
        self.__observers = weakref.WeakSet()

    def add_observer(self, observer):
        """Adds observer"""
        # add listener to dict
        self.__observers.add(observer)

    def remove_observer(self, observer):
        """Removes observer"""
        # delete listener reference from dict
        self.__observers.remove(observer)

    def fire(self, event):
        """Sends the specified event to listeners"""
        # set event source
        event.source = self

        LOGGER.debug(f"firing '{event}' from '{self}'")
        for observer in self.__observers:
            observer._receive_event(event)


class Observer(metaclass=abc.ABCMeta):
    """Listens for notifications from :class:`~.Notifier` objects"""

    def _receive_event(self, event):
        LOGGER.debug(f"received '{event}' in '{self}'")
        self._handle_event(event)

    @abc.abstractmethod
    def _handle_event(self, event):
        raise NotImplementedError


class Event:
    def __init__(self, message, **data):
        self.message = message
        self.data = data
        self._source = None

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        self._source = source

    def __str__(self):
        return f"Event('{self.message}' from '{self.source}')"


class UnknownEventException(Exception):
    def __init__(self, event):
        super().__init__(f"unrecognised event '{event}'")
